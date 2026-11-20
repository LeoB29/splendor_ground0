"""Model-free protocol, prefix preservation, exact science and resource checks."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import pytest
from tools.splendorust_supervisor_separation_20261025 import runner as r, common as c, preparation as p


class Envelope:
    def __init__(self):self.stage=None;self.events=[];self.resident={}
    def begin(self,stage):self.stage=stage;self.events.append(('begin',stage))
    def finish(self):self.events.append(('finish',self.stage))
    def guard(self):pass
    def admit(self):pass
    def remaining(self):return 10000.
    def timeout(self,maximum):return maximum
    def effective_deadline(self):return 99999.


def stage_setup(monkeypatch,*,qualified=True,timing_failure=False,fault_index=None):
    calls=[];final=[];envelope=Envelope()
    monkeypatch.setattr(r,'qualify',lambda *a:{})
    monkeypatch.setattr(r,'switch_placement',lambda *a:None)
    monkeypatch.setattr(r,'collect_timing',lambda *a:([],{'qualified':qualified},{'failed':True} if timing_failure else None))
    def game(spec,*args):
        calls.append(spec['game_index']);fault=spec['game_index']==fault_index
        return dict(spec,reason='forfeit' if fault else 'natural'),[{'clock':{'receipt_elapsed_seconds':.95}}],0,({'fault':True} if fault else None)
    monkeypatch.setattr(r,'play_game',game)
    monkeypatch.setattr(r,'final_checks',lambda *args:final.append(args) or {'status':'passed'})
    return envelope,calls,final


@pytest.mark.parametrize('timing_failure',[False,True])
def test_timing_negative_finalizes_without_operational_games(monkeypatch,timing_failure):
    env,calls,final=stage_setup(monkeypatch,qualified=False,timing_failure=timing_failure)
    r.run_stages(env,{}, {},p.operational_schedule())
    assert calls==[] and final[0][-2:]==([],None)
    assert [stage for kind,stage in env.events if kind=='begin']==['qualification','timing','final']


@pytest.mark.parametrize('fault',[0,3,5,None])
def test_live_stops_only_actual_fault_not_700ms_margin(monkeypatch,fault):
    env,calls,final=stage_setup(monkeypatch,fault_index=fault)
    r.run_stages(env,{}, {},p.operational_schedule())
    assert calls==list(range(6 if fault is None else fault+1))
    assert len(final[0][-2])==len(calls)


def test_infrastructure_error_cannot_become_verified_prefix(monkeypatch):
    env,_,_=stage_setup(monkeypatch)
    monkeypatch.setattr(r,'collect_timing',lambda *args:(_ for _ in ()).throw(ValueError('bad placement')))
    monkeypatch.setattr(r,'final_checks',lambda *args:pytest.fail('Bad evidence accepted'))
    with pytest.raises(ValueError,match='placement'):r.run_stages(env,{}, {},p.operational_schedule())


def fake_panel(monkeypatch):
    fixtures=[]
    for i in range(24):
        quota=32 if i%2 else 64
        science=dict(result={'quota':quota,'action':{'q':quota},'diagnostics':{'q':quota}},action={'q':quota},trace={'q':quota},after={'q':quota})
        fixtures.append(dict(fixture_id=str(i),snapshot={'i':i},observer={'i':i},
            historical=None if i==10 else dict(quota=quota,**science)))
    schedule=r.panel.make_schedule(fixtures);rows=[]
    for spec in schedule:
        q=spec['quota'];fixture=fixtures[spec['fixture_index']]
        rows.append(dict(spec,before=fixture['snapshot'],observer=fixture['observer'],completed=True,
            clock={'status':'ok','deadline_seconds':1.,'result':{'quota':q,'action':{'q':q},'diagnostics':{'q':q}}},
            action={'q':q},trace={'q':q},after={'q':q}))
    monkeypatch.setattr(r.panel,'validate_plan',lambda value:None)
    monkeypatch.setattr(r.audit,'validate_boundary_row',lambda value:None)
    monkeypatch.setattr(r.scientific,'check_clock',lambda value:None)
    monkeypatch.setattr(r.scientific,'validate_result',lambda *args,**kwargs:None)
    monkeypatch.setattr(r.scientific,'audit_decision',lambda *args,**kwargs:None)
    monkeypatch.setattr(r.audit,'classify_failure',lambda value:'player_timeout')
    return {'fixtures':fixtures,'schedule':schedule},rows


def test_panel_requires48_four_way_cells_and92_original_quota_historical_matches(monkeypatch):
    plan,rows=fake_panel(monkeypatch)
    result=r.validate_timing_rows(rows,plan)
    assert result['complete'] and result['fully_replicated_groups']==48 and result['historical_comparisons']==92
    #32 and64 science intentionally differ, and must never be compared across quotas.
    assert rows[0]['clock']['result']!=rows[96]['clock']['result']


@pytest.mark.parametrize('kind',['duplicate','historical','repeat','deadline'])
def test_panel_rejects_schedule_or_scientific_drift(monkeypatch,kind):
    plan,rows=fake_panel(monkeypatch)
    if kind=='duplicate':rows[1]['index']=0
    if kind=='historical':plan['fixtures'][0]['historical']['after']={'wrong':True}
    if kind=='repeat':rows[24]['after']={'wrong':True}
    if kind=='deadline':rows[0]['clock']['deadline_seconds']=.8
    with pytest.raises(ValueError):r.validate_timing_rows(rows,plan)


def test_timing_failure_only_valid_as_final_row_of_explicit_prefix(monkeypatch):
    plan,rows=fake_panel(monkeypatch);rows=rows[:5]
    rows[-1].update(completed=False,action=None,trace=None,after=None)
    rows[-1]['clock'].update(status='timeout',result=None)
    checked=r.validate_timing_rows(rows,plan,allow_prefix=True)
    assert checked['requests']==5 and checked['completed']==4 and not checked['complete']
    with pytest.raises(ValueError):r.validate_timing_rows(rows,plan)
    with pytest.raises(ValueError):r.validate_timing_rows(rows[:-1],plan,allow_prefix=True)


def game_setup(monkeypatch,tmp_path,*,bad_profile=False,bad_placement=False,infra=False):
    monkeypatch.setattr(c,'OUT',tmp_path);events=[]
    monkeypatch.setattr(c,'save',lambda path,value:events.append(('save',r.Path(path).name,deepcopy(value))))
    state=SimpleNamespace(terminal=False,current_player=0,winner=None)
    monkeypatch.setattr(r.rules,'CorrectedEnv',lambda **kwargs:SimpleNamespace(initial_state=lambda:state))
    monkeypatch.setattr(r.rules,'serialize_snapshot',lambda value:{'same':'position'})
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:state)
    class Worker:
        def __init__(self,name):self.output_dir=tmp_path/name;self.name=name
        def __enter__(self):events.append(('open',self.name));return self
        def __exit__(self,*args):events.append(('close',self.name))
    monkeypatch.setattr(r,'make_worker',lambda name,*args,**kwargs:Worker(name))
    row=dict(completed=False,player_role='challenger',quota=64,mode='candidate',seat=0,action=None,
        clock={'status':'timeout','request_id':'request-00000002'},worker_clock_file='owned/request-00000002.json')
    if bad_profile:row['profile_capture_error']='bad sidecar'
    if bad_placement:row['placement_error']='bad placement'
    monkeypatch.setattr(r,'decision',lambda *args:deepcopy(row))
    monkeypatch.setattr(r,'verify_saved_decision',lambda *args:None)
    def classify(clock):
        events.append(('classify',))
        if infra:raise ValueError('infrastructure failure')
        return 'player_timeout'
    monkeypatch.setattr(r.audit,'classify_failure',classify)
    for name in ('verify_worker_logs','verify_raw_transport','verify_graph_evidence'):
        monkeypatch.setattr(r,name,lambda *a,**kw:{'status':'passed'})
    monkeypatch.setattr(r,'measure_seal_cost',lambda *a:{'seconds':0.})
    monkeypatch.setattr(r.audit,'audit_game',lambda *a,**kw:{'status':'passed'})
    return events,p.operational_schedule()[0]


def test_original_failure_saved_before_classifier_and_both_workers_close_before_result(monkeypatch,tmp_path):
    events,spec=game_setup(monkeypatch,tmp_path)
    game,rows,cost,stop=r.play_game(spec,Envelope(),{})
    assert game['reason']=='forfeit' and game['loser']==0 and game['winner']==1 and stop['clock']==rows[0]['clock']
    save=next(i for i,e in enumerate(events) if e[:2]==('save','0000.json'))
    result=next(i for i,e in enumerate(events) if e[:2]==('save','result.json'))
    assert save<events.index(('classify',))<result
    assert sum(e[0]=='close' for e in events)==2 and all(i<result for i,e in enumerate(events) if e[0]=='close')


@pytest.mark.parametrize('flag',['bad_profile','bad_placement','infra'])
def test_integrity_failure_preserves_move_without_loss(monkeypatch,tmp_path,flag):
    events,spec=game_setup(monkeypatch,tmp_path,**{flag:True})
    with pytest.raises(ValueError):r.play_game(spec,Envelope(),{})
    assert any(e[:2]==('save','0000.json') for e in events)
    assert not any(e[:2] in [('save','collection_stop.json'),('save','result.json')] for e in events)
    assert sum(e[0]=='close' for e in events)==2


@pytest.mark.parametrize('field',['after','result','completed'])
def test_reproduction_demands_complete_exact_science_and_transition(monkeypatch,field):
    original=dict(completed=True,mode='candidate',quota=64,before={},player_role='challenger',
        action={'a':1},trace={'t':1},after={'s':1},clock={'result':{'r':1}})
    repeated=deepcopy(original)
    if field=='result':repeated['clock']['result']={'wrong':True}
    elif field=='completed':repeated[field]=False
    else:repeated[field]={'wrong':True}
    monkeypatch.setattr(r,'make_worker',lambda *a,**kw:nullcontext(None))
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:value)
    monkeypatch.setattr(r,'decision',lambda *args:repeated)
    saved=[];monkeypatch.setattr(c,'save',lambda path,value:saved.append(value))
    with pytest.raises(ValueError,match='reproduction'):r.reproduce_decision(original,0,Envelope(),{})
    assert saved==[repeated]


def test_envelope_caps_both_clocks_collection_and_final_reserve(monkeypatch):
    now=SimpleNamespace(mono=2990.,wall=3990.)
    monkeypatch.setattr(r.time,'monotonic',lambda:now.mono);monkeypatch.setattr(r.time,'time',lambda:now.wall)
    monkeypatch.setattr(r.shutil,'disk_usage',lambda root:SimpleNamespace(free=10*c.GiB))
    monkeypatch.setattr(c,'save',lambda *a:None)
    env=r.Envelope({'monotonic_origin':0.,'unix_origin':1000.})
    monkeypatch.setattr(env,'accounted_bytes',lambda:c.PREFINAL_EVIDENCE_BYTES-1)
    env.begin('timing');assert env.remaining()==10 and env.deadline==3000
    monkeypatch.setattr(env,'accounted_bytes',lambda:c.PREFINAL_EVIDENCE_BYTES)
    with pytest.raises(ValueError,match='evidence'):env.guard()
    env.begin('final');assert env.deadline==3590
    now.wall=4600.
    with pytest.raises(ValueError,match='real-clock'):env.guard()


def test_worker_live_placement_blocks_mode_switch(monkeypatch):
    env=Envelope();env.resident={1:object()}
    monkeypatch.setattr(r.placement,'configure_execution',lambda *a:pytest.fail('Switched with workers'))
    with pytest.raises(ValueError,match='resident'):r.switch_placement(env,{},'isolated')


def test_no_watchdog_ack_fails_closed_before_worker_construction(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    runner=dict(pid=30,created=40.);watchdog=dict(watchdog_pid=10,watchdog_created=20.,runner_pid=30,runner_created=40.,freeze_sha256='f')
    c.save(tmp_path/'watchdog.json',watchdog);c.save(tmp_path/'watchdog_claim.json',watchdog)
    class Process:
        def __init__(self,pid=30):self.pid=pid
        def create_time(self):return 40. if self.pid==30 else 20.
        def is_running(self):return True
    monkeypatch.setattr(r.os,'getpid',lambda:30);monkeypatch.setattr(r.psutil,'Process',Process)
    monkeypatch.setattr(r.placement,'configure_execution',lambda *a:{'monotonic_ns':1})
    now=[0.]
    def clock():now[0]+=.1;return now[0]
    monkeypatch.setattr(r.time,'monotonic',clock);monkeypatch.setattr(r.time,'sleep',lambda seconds:None)
    with pytest.raises(ValueError,match='acknowledgement missing'):r.switch_placement(Envelope(),{},'isolated')
    assert (tmp_path/'placement_requests'/'0001.json').exists()


def test_saved_exchange_body_sequence_must_match_filename(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    c.save(tmp_path/'placement_requests'/'0001.json',{'sequence':2})
    c.save(tmp_path/'placement_acks'/'0001.json',{})
    c.save(tmp_path/'watchdog.json',{})
    c.save(tmp_path/'launch.json',{'pid':1,'created':2.})
    env=Envelope();env.placement_sequence=1
    monkeypatch.setattr(c,'validate_exchange',lambda *a:pytest.fail('Misnumbered receipt accepted'))
    with pytest.raises(ValueError,match='sequence differs'):r.verify_exchanges(env,{})
