"""Model-free stop, preservation, replay and resource-bound counterexamples."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_isolated_reliability_20261026 import runner as r, common as c


class Envelope:
    def __init__(self):self.stage=None;self.events=[];self.start=0.;self.wall_start=0.
    def begin(self,stage):self.stage=stage;self.events.append(('begin',stage))
    def finish(self):self.events.append(('finish',self.stage))
    def guard(self):pass
    def admit(self):pass
    def remaining(self):return 10000.
    def timeout(self,maximum):return maximum
    def effective_deadline(self):return 99999.


def specs():return [{'stage':'operational','game_index':i} for i in range(6)]+[{'stage':'main','game_index':i} for i in range(194)]


def test_operational_fault_finalizes_prefix_without_main_dispatch(monkeypatch):
    envelope=Envelope();games=[{'stage':'operational','game_index':0,'reason':'forfeit'}];stop={'clock_status':'timeout'};final=[]
    monkeypatch.setattr(r,'qualify',lambda *args:{'qualified':True})
    monkeypatch.setattr(r,'operational',lambda *args:(games,stop))
    monkeypatch.setattr(r,'play_game',lambda *args:pytest.fail('Main game dispatched after fault'))
    monkeypatch.setattr(r,'final_checks',lambda env,freeze,schedule,played,fault:final.append((played,fault)) or {'status':'captured_fault_verified_prefix'})
    result=r.run_stages(envelope,{},specs())
    assert result['status']=='captured_fault_verified_prefix' and final==[(games,stop)]
    assert [v for k,v in envelope.events if k=='begin']==['qualification','operational','final']


@pytest.mark.parametrize('fault_index',[0,3,193,None])
def test_main_stops_exactly_at_first_fault_or_completes_frozen_count(monkeypatch,fault_index):
    games=[dict(stage='operational',game_index=i,reason='natural') for i in range(6)];calls=[];captured=[]
    monkeypatch.setattr(r,'qualify',lambda *args:{})
    monkeypatch.setattr(r,'operational',lambda *args:(games,None))
    def play(spec,*args):
        calls.append(spec['game_index']);failure=spec['game_index']==fault_index
        return dict(spec,reason='forfeit' if failure else 'natural'),[],0.,({'game_index':fault_index} if failure else None)
    monkeypatch.setattr(r,'play_game',play)
    monkeypatch.setattr(r,'final_checks',lambda env,freeze,schedule,played,stop:captured.append((list(played),stop)) or {})
    r.run_stages(Envelope(),{},specs())
    assert calls==list(range(194 if fault_index is None else fault_index+1))
    assert len(captured[0][0])==6+len(calls)
    assert (captured[0][1] is None)==(fault_index is None)


def test_infrastructure_failure_is_incomplete_not_a_verified_fault_prefix(monkeypatch):
    monkeypatch.setattr(r,'qualify',lambda *args:{})
    monkeypatch.setattr(r,'operational',lambda *args:(_ for _ in ()).throw(ValueError('invalid sidecar')))
    monkeypatch.setattr(r,'final_checks',lambda *args:pytest.fail('Infrastructure failure accepted'))
    with pytest.raises(ValueError,match='sidecar'):r.run_stages(Envelope(),{},specs())


def setup_game(monkeypatch,tmp_path,*,classification='player_timeout',profile_error=False):
    monkeypatch.setattr(c,'OUT',tmp_path);events=[]
    monkeypatch.setattr(c,'save',lambda path,value:events.append(('save',r.Path(path).name,deepcopy(value))))
    state=SimpleNamespace(terminal=False,current_player=0,winner=None)
    monkeypatch.setattr(r.rules,'CorrectedEnv',lambda **kwargs:SimpleNamespace(initial_state=lambda:state))
    monkeypatch.setattr(r.rules,'serialize_snapshot',lambda value:{'same':'position'})
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:state)
    monkeypatch.setattr(r.schedules,'validate_spec',lambda spec:None)
    class Worker:
        def __init__(self,name):self.output_dir=tmp_path/name;self.name=name
        def __enter__(self):events.append(('open',self.name));return self
        def __exit__(self,*args):events.append(('close',self.name))
    monkeypatch.setattr(r,'make_worker',lambda name,*args,**kwargs:Worker(name))
    row=dict(completed=False,player_role='challenger',quota=64,mode='candidate',seat=0,action=None,
        clock={'status':'timeout','request_id':'request-00000002'},worker_clock_file='owned/request-00000002.json')
    if profile_error:row['profile_capture_error']='invalid observed hash'
    def decide(*args):events.append(('decision',));return deepcopy(row)
    monkeypatch.setattr(r,'decision',decide)
    monkeypatch.setattr(r,'verify_saved_decision',lambda *args:None)
    def classify(clock):
        events.append(('classify',))
        if isinstance(classification,Exception):raise classification
        return classification
    monkeypatch.setattr(r.audit,'classify_failure',classify)
    monkeypatch.setattr(r,'verify_worker_logs',lambda *args,**kwargs:{'status':'passed'})
    monkeypatch.setattr(r,'verify_raw_transport',lambda *args,**kwargs:{'status':'passed'})
    monkeypatch.setattr(r,'verify_graph_evidence',lambda *args,**kwargs:{'status':'passed'})
    monkeypatch.setattr(r,'verify_closed_worker',lambda *args:dict(logs={'status':'passed'},raw={'status':'passed'},graph={'status':'passed'}))
    monkeypatch.setattr(r.audit,'audit_game',lambda *args,**kwargs:{'status':'passed'})
    spec=dict(stage='main',pair_index=0,game_index=0,seed_decimal='1',challenger_seat=0,
        challenger_mode='candidate',challenger_quota=64,opponent_mode='baseline',opponent_quota=32)
    return events,spec


def test_timed_failure_saved_before_classification_and_loss_retained_after_both_workers_close(monkeypatch,tmp_path):
    events,spec=setup_game(monkeypatch,tmp_path)
    ending,rows,cost,stop=r.play_game(spec,Envelope(),{})
    assert ending['reason']=='forfeit' and ending['winner']==1 and ending['loser']==0
    assert len(rows)==1 and stop['clock']==rows[0]['clock']
    saved_move=next(i for i,e in enumerate(events) if e[:2]==('save','0000.json'))
    classified=events.index(('classify',));saved_result=next(i for i,e in enumerate(events) if e[:2]==('save','result.json'))
    assert saved_move<classified<saved_result
    assert sum(e[0]=='decision' for e in events)==1
    assert all(i<saved_result for i,e in enumerate(events) if e[0]=='close')
    assert sum(e[0]=='close' for e in events)==2


@pytest.mark.parametrize('capture_error',[False,True])
def test_infrastructure_or_invalid_profile_never_becomes_player_loss(monkeypatch,tmp_path,capture_error):
    events,spec=setup_game(monkeypatch,tmp_path,
        classification=ValueError('infrastructure cannot forfeit'),profile_error=capture_error)
    with pytest.raises(ValueError):r.play_game(spec,Envelope(),{})
    assert any(e[:2]==('save','0000.json') for e in events)
    assert not any(e[:2] in [('save','collection_stop.json'),('save','result.json')] for e in events)
    assert sum(e[0]=='close' for e in events)==2


def test_operational_descriptive_margin_does_not_reject_completed_slow_moves(monkeypatch):
    games=[];mock_specs=specs();envelope=Envelope()
    monkeypatch.setattr(r,'bytes_used',lambda:100)
    def play(spec,*args):games.append(spec);return dict(spec,reason='natural'),[{'clock':{'receipt_elapsed_seconds':.95}}],1.,None
    monkeypatch.setattr(r,'play_game',play)
    monkeypatch.setattr(r,'final_evidence_forecast',lambda *args:{'projected_final_evidence_bytes':100})
    monkeypatch.setattr(r,'operational_gate',lambda **kwargs:{'passed':True})
    monkeypatch.setattr(c,'save',lambda *args:None)
    result,stop=r.operational(envelope,{},mock_specs,{'forecast_seconds':1.})
    assert len(result)==len(games)==6 and stop is None


def test_operational_fault_skips_expansion_forecast_and_remaining_games(monkeypatch):
    called=[];monkeypatch.setattr(r,'bytes_used',lambda:100)
    def play(spec,*args):
        called.append(spec['game_index']);fault=spec['game_index']==2
        return dict(spec,reason='forfeit' if fault else 'natural'),[],0.,({'fault':True} if fault else None)
    monkeypatch.setattr(r,'play_game',play)
    monkeypatch.setattr(r,'final_evidence_forecast',lambda *args:pytest.fail('Forecast after stop'))
    games,stop=r.operational(Envelope(),{},specs(),{})
    assert called==[0,1,2] and len(games)==3 and stop=={'fault':True}


@pytest.mark.parametrize('field,value', [('after',{'different':True}),('result',{'different':True}),('completed',False)])
def test_reproduction_requires_whole_result_transition_and_completion(monkeypatch,tmp_path,field,value):
    original={'completed':True,'mode':'candidate','quota':64,'before':{},'player_role':'challenger',
        'action':{'a':1},'trace':{'t':1},'after':{'s':1},'clock':{'result':{'r':1}}}
    repeated=deepcopy(original)
    if field=='result':repeated['clock']['result']=value
    else:repeated[field]=value
    monkeypatch.setattr(r,'make_worker',lambda *args,**kwargs:nullcontext(None))
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:value)
    monkeypatch.setattr(r,'decision',lambda *args:repeated)
    saved=[];monkeypatch.setattr(c,'save',lambda path,value:saved.append(deepcopy(value)))
    monkeypatch.setattr(r,'verify_saved_decision',lambda *args:pytest.fail('Invalid reproduction accepted'))
    with pytest.raises(ValueError,match='reproduction'):r.reproduce_decision(original,0,Envelope(),{})
    assert saved==[repeated]


def valid_forecast():
    return dict(observed_seconds=100.,observed_variable_bytes=100.,current_bytes=1000.,independent_seconds=1.,
        fixed_final_seconds=100.,final_evidence_bytes=1024.,remaining_collection_seconds=16000.)


def test_operational_resource_forecast_is_finite_and_respects_each_cap():
    assert r.operational_gate(**valid_forecast())['passed']
    for field,value in [('observed_seconds',1000.),('current_bytes',c.PREFINAL_EVIDENCE_BYTES),
        ('independent_seconds',100.),('final_evidence_bytes',c.FINAL_RESERVE_BYTES),
        ('remaining_collection_seconds',1.)]:
        args=valid_forecast();args[field]=value
        assert not r.operational_gate(**args)['passed']
    args=valid_forecast();args['observed_seconds']=float('nan')
    with pytest.raises(ValueError):r.operational_gate(**args)


def test_envelope_collection_cutoff_and_final_reserve_cannot_extend_clocks(monkeypatch):
    now=SimpleNamespace(mono=19790.,wall=20790.)
    monkeypatch.setattr(r.time,'monotonic',lambda:now.mono);monkeypatch.setattr(r.time,'time',lambda:now.wall)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda root:SimpleNamespace(free=10*c.GiB));monkeypatch.setattr(c,'save',lambda *args:None)
    envelope=r.Envelope({'monotonic_origin':0.,'unix_origin':1000.})
    monkeypatch.setattr(envelope,'accounted_bytes',lambda:c.PREFINAL_EVIDENCE_BYTES-1)
    envelope.begin('main');assert envelope.remaining()==10 and envelope.deadline==19800
    monkeypatch.setattr(envelope,'accounted_bytes',lambda:c.PREFINAL_EVIDENCE_BYTES)
    with pytest.raises(ValueError,match='evidence'):envelope.guard()
    envelope.begin('final');assert envelope.deadline==21590
    now.wall=22600.
    with pytest.raises(ValueError,match='real-clock'):envelope.guard()


def test_adapter_default_is_forced_isolated_and_explicit_original_is_rejected(monkeypatch):
    calls=[]
    monkeypatch.setattr(r.adapters,'make_worker',lambda *args,**kwargs:calls.append(kwargs) or 'worker')
    assert r.make_worker('test',None,{},mode='candidate',quota=64)=='worker'
    assert calls==[{'mode':'candidate','quota':64,'placement_mode':'isolated'}]
    with pytest.raises(ValueError,match='isolated'):r.make_worker('test',None,{},mode='baseline',quota=32,placement_mode='original')
    assert len(calls)==1
    with pytest.raises(ValueError,match='only isolated'):r.switch_placement(None,{},'original')


def test_watchdog_starts_isolated_and_rejects_original_before_configuring(monkeypatch):
    import ast
    import inspect
    from tools.splendorust_isolated_reliability_20261026 import launch
    launch.require_isolated_request({'placement_mode':'isolated'})
    with pytest.raises(ValueError,match='isolated'):launch.require_isolated_request({'placement_mode':'original'})
    tree=ast.parse(inspect.getsource(launch.main))
    # First placement configuration precedes subprocess launch and fixes mode.
    configurations=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='configure_execution']
    assert isinstance(configurations[0].args[1],ast.Constant) and configurations[0].args[1].value=='isolated'
    launches=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
    assert configurations[0].lineno<launches[0].lineno


def test_resource_projection_counts_initial_six_once_and_expands_by194():
    gate=r.operational_gate(**valid_forecast())
    assert gate['operational_games']==6 and gate['remaining_games']==194 and gate['total_games']==200
    assert gate['arithmetic_factor']==pytest.approx(194/6*1.35)
    assert gate['final_arithmetic_factor']==pytest.approx(200/6*1.35)
    assert gate['projected_collection_seconds']==pytest.approx(100.*194/6*1.35)


def test_final_byte_projection_includes_placement_sidecar_evidence(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    schedule=[dict(stage='operational',pair_index=i//2,game_index=i) for i in range(6)]
    def put(path,size=32):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x'*size)
    for spec in schedule:
        folder=r.game_folder(spec)
        for name in ('moves/0000.json','online_replay.json','graph_integrity_challenger.json',
            'log_integrity_challenger.json','log_integrity_opponent.json','raw_transport_challenger.json','raw_transport_opponent.json'):
            put(folder/name)
        for role in ('challenger','opponent'):
            worker=tmp_path/'workers'/f"operational_g{spec['game_index']}_{role}"
            for name in ('inputs/request-00000001.json','request-00000001.json','stdout.log.gz',
                'placement_admission.json','placement_requests/request-00000001.json'):
                put(worker/name)
    baseline=r.final_evidence_forecast(schedule)
    put(tmp_path/'workers'/'operational_g0_challenger'/'placement_requests'/'request-00000001.json',8192)
    bigger=r.final_evidence_forecast(schedule)
    assert bigger['single_reproduction_worker_bound_bytes']-baseline['single_reproduction_worker_bound_bytes']==8192-32
    assert bigger['projected_final_evidence_bytes']>baseline['projected_final_evidence_bytes']


def test_postclose_cache_is_refreshed_before_closed_worker_guard(monkeypatch,tmp_path):
    events=[]
    env=SimpleNamespace(close_worker=lambda folder:events.append('refresh'),guard=lambda:events.append('guard'),effective_deadline=lambda:9999.)
    def leave(*a,**kw):events.append('verify');raise ValueError('test exit')
    monkeypatch.setattr(r,'verify_worker_logs',leave)
    with pytest.raises(ValueError,match='test exit'):r.verify_closed_worker(tmp_path,{},env)
    assert events==['refresh','guard','verify']


def test_preparation_and_live_runtime_both_require_frozen_isolation():
    freeze={'placement_mode':'isolated'}
    assert r.runtime_placement_mode(freeze)=='isolated'
    assert r.runtime_placement_mode(freeze,SimpleNamespace())=='isolated'
    assert r.runtime_placement_mode(freeze,SimpleNamespace(placement_mode='isolated'))=='isolated'
    with pytest.raises(ValueError,match='isolated'):r.runtime_placement_mode(freeze,SimpleNamespace(placement_mode='original'))
    with pytest.raises(ValueError,match='isolated'):r.runtime_placement_mode({'placement_mode':'original'})
