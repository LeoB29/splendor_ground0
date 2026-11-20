"""Model-free Stage A clocks, original transport, failure preservation and gates."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import pytest
from tools.dom_screen_20261027 import runner as r, verification as v, clock as clock, schedule as schedule, common as c
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker
from test_splendorust_live_boundary_adapters import profile_row

class Envelope:
    def __init__(self):self.stage='qualification';self.events=[];self.start=0.;self.wall_start=0.
    def begin(self,stage):self.stage=stage;self.events.append(stage)
    def finish(self):pass
    def guard(self):pass
    def admit(self):pass
    def remaining(self):return 9999.
    def timeout(self,x):return x
    def effective_deadline(self):return None


def test_original_readers_request_and_minimal_observer_inherited():
    cls=clock.PlacedProfiledStudyWorker
    assert cls.request is ProfiledStudyWorker.request
    for name in ('_receive','_stdout_reader','_stderr_reader','_put'):
        assert getattr(cls,name) is getattr(ProfiledStudyWorker,name)
    assert clock._PlacementConstructor.request is PersistentWorker.request


def test_fresh_paired_schedule_has_exact66_distinct_world_pairs():
    plan=schedule.generate_schedule('ab'*32)
    assert schedule.validate_schedule(plan)==plan and len(plan)==66
    assert sum(x['stage']=='main' for x in plan)==60
    assert len({x['seed_decimal'] for x in plan})==33
    for i in range(0,66,2):
        assert plan[i]['seed_decimal']==plan[i+1]['seed_decimal']
        assert plan[i]['challenger_seat']!=plan[i+1]['challenger_seat']
        assert plan[i]['challenger_quota'] is None and plan[i]['opponent_quota']==32
    changed=deepcopy(plan);changed[5]['seed_decimal']='0'
    with pytest.raises(ValueError):schedule.validate_schedule(changed)
    assert schedule.generate_schedule('cd'*32)!=plan


@pytest.mark.parametrize('fault',[0,29,59,None])
def test_first_screen_fault_stops_exact_prefix_and_still_finalizes(monkeypatch,fault):
    plan=schedule.generate_schedule('ab'*32);calls=[];final=[]
    monkeypatch.setattr(r,'qualify',lambda *x:{'qualified':True})
    monkeypatch.setattr(r,'operational',lambda *x:([dict(p,reason='natural') for p in plan[:6]],None))
    monkeypatch.setattr(c,'read',lambda p:{'passed':True})
    def play(spec,*args):
        calls.append(spec['game_index']);failed=spec['game_index']==fault
        return dict(spec,reason='forfeit' if failed else 'natural'),[],0.,({'failed':True} if failed else None)
    monkeypatch.setattr(r,'play_game',play)
    monkeypatch.setattr(r,'final_checks',lambda env,f,p,g,s,q:final.append((len(g),s)) or {})
    status,_=r.run_stages(Envelope(),{},plan)
    assert calls==list(range(60 if fault is None else fault+1))
    assert final[0][0]==6+len(calls)
    assert status==('completed_verified' if fault is None else 'captured_fault_verified')


@pytest.mark.parametrize('reason',['qualification','operational','fault'])
def test_negative_admission_never_dispatches_screen(monkeypatch,reason):
    plan=schedule.generate_schedule('ab'*32);final=[]
    monkeypatch.setattr(r,'qualify',lambda *x:{'qualified':reason!='qualification'})
    monkeypatch.setattr(r,'operational',lambda *x:([dict(p,reason='natural') for p in plan[:6]],{'fault':True} if reason=='fault' else None))
    monkeypatch.setattr(c,'read',lambda p:{'passed':False})
    monkeypatch.setattr(r,'play_game',lambda *x:pytest.fail('unadmitted screen dispatched'))
    monkeypatch.setattr(r,'final_checks',lambda *x:final.append(x) or {})
    status,_=r.run_stages(Envelope(),{},plan)
    assert len(final)==1
    assert status=={'qualification':'qualification_not_admitted_verified','operational':'operational_not_admitted_verified','fault':'captured_fault_verified'}[reason]


def test_screen_gate_requires15_natural_points_zero_fault_and_full60():
    def games(wins):return [dict(stage='operational',reason='natural',winner=0,challenger_seat=0) for _ in range(6)]+[dict(stage='main',reason='natural',winner=0 if i<wins else 1,challenger_seat=0) for i in range(60)]
    assert not v.screen_assessment(games(14))['qualified_for_future_confirmation']
    assert v.screen_assessment(games(15))['qualified_for_future_confirmation']
    forged=games(60);forged[-1].update(reason='forfeit',failed_role='opponent')
    assert not v.screen_assessment(forged)['qualified_for_future_confirmation']
    assert not v.screen_assessment(games(60)[:-1])['qualified_for_future_confirmation']


def test_resource_forecast_charges60_screen_and66_final_at1point5():
    result=r.operational_gate(observed_seconds=100.,observed_variable_bytes=1000,current_bytes=5000,
        independent_seconds=10.,fixed_final_seconds=100.,final_evidence_bytes=1000,remaining_collection_seconds=9999.)
    assert result['passed'] and result['projected_collection_seconds']==1500
    assert result['projected_final_seconds']==265 and result['projected_prefinal_bytes']==20000
    with pytest.raises(ValueError):r.operational_gate(observed_seconds=float('nan'),observed_variable_bytes=1,current_bytes=1,independent_seconds=1,fixed_final_seconds=1,final_evidence_bytes=1,remaining_collection_seconds=1)


def test_every_guard_enforces_clocks_even_when_resources_amortized(monkeypatch):
    monkeypatch.setattr(r.time,'monotonic',lambda:100.)
    monkeypatch.setattr(r.time,'time',lambda:1000.)
    env=r.Envelope(dict(monotonic_origin=99.,unix_origin=999.));env.last_resource_guard=100.
    env.deadline=100.1;env.wall_deadline=1000.1;env.guard()
    env.wall_deadline=999.9
    with pytest.raises(ValueError,match='deadline'):env.guard()


def test_late_sidecar_cache_refresh_counts_new_evidence(tmp_path):
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{},'baseline':0,'parent_bytes':0}
    folder=tmp_path/'worker';folder.mkdir();(folder/'a').write_bytes(b'123');env.register_worker(folder);env.close_worker(folder)
    assert env.evidence['workers'][folder]==3
    (folder/'late_sidecar').write_bytes(b'4567');env.close_worker(folder)
    assert env.evidence['workers'][folder]==7


def external_row():
    row=profile_row(32);row.update(player_role='challenger',quota=None)
    row['clock'].update(clock='perf_counter_ns',elapsed_seconds=.45)
    row['child_profile'].update(quota=None,backend='dom_numpy_policy',graph_identity_sha256=None,
        payload_sha256=c.digest({'player':'dom','observer':row['observer']}))
    return row


def test_external_boundaries_use_truthful_backend_and_original_clock():
    row=external_row();assert v.validate_profiles(row)['complete_decision']
    row['child_profile']['backend']='unchanged_eager32'
    with pytest.raises(ValueError):v.validate_profiles(row)


@pytest.mark.parametrize('corruption',['request_hash','response_hash','child_pid','late_accepted','missing_parent','unexpected_handle'])
def test_external_profile_corruptions_fail(corruption):
    row=external_row()
    if corruption=='request_hash':row['child_profile']['request_raw_sha256']='7'*64
    elif corruption=='response_hash':row['parent_profile']['response_raw_sha256']='7'*64
    elif corruption=='child_pid':row['child_profile']['pid']=43
    elif corruption=='late_accepted':row['clock']['receipt_elapsed_seconds']=1.1
    elif corruption=='missing_parent':row['parent_profile']=None
    else:row['child_profile']['profiles'].append(deepcopy(row['child_profile']['profiles'][0]))
    with pytest.raises(ValueError):v.validate_profiles(row)


@pytest.mark.parametrize('external',[False,True])
@pytest.mark.parametrize('failed',[False,True])
def test_exact_raw_binding_preserves_late_or_completed_packets(monkeypatch,tmp_path,external,failed):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');monkeypatch.setattr(c,'WRITE_OBSERVER',None)
    row=external_row() if external else profile_row(32);receipt=row['clock'];receipt.update(clock='perf_counter_ns',elapsed_seconds=.45,detail=None)
    parent,child=row['parent_profile'],row['child_profile'];rid=receipt['request_id']
    player='dom' if external else 'search';metadata={'player':'dom'} if external else {'search_simulations':32}
    folder=c.OUT/'workers'/'main_g0_challenger';folder.mkdir(parents=True)
    payload=dict(player=player,observer=row['observer'])
    request=(json.dumps(dict(type='request',request_id=rid,payload=payload),separators=(',',':'))+'\n').encode()
    packet=(json.dumps(dict(type='response',request_id=rid,result=deepcopy(receipt['result'])),separators=(',',':'))+'\r\n').encode()
    parent.update(request_raw_sha256=hashlib.sha256(request).hexdigest(),request_raw_bytes=len(request),response_raw_sha256=hashlib.sha256(packet).hexdigest(),response_raw_bytes=len(packet))
    child.update(request_raw_sha256=parent['request_raw_sha256'],serialized_packet_utf8_sha256=hashlib.sha256(packet[:-2]+b'\n').hexdigest(),serialized_packet_utf8_bytes=len(packet)-1)
    if failed:
        receipt.update(status='timeout',accepted=False,received_before_deadline=False,receipt_elapsed_seconds=1.1,elapsed_seconds=1.11,result=None)
        parent.update(receipt_arrival_ns=2100000000,receipt_reconstructed_arrival_ns=2100000000,capture_snapshot_ns=2200000000,raw_arrivals_ns=[2100000000])
    c.save(folder/'initialization.json',dict(result=metadata,identity={'pid':42}))
    c.save(folder/'cleanup.json',dict(log_compression={'stdout.log':{}}))
    c.save(folder/(rid+'.json'),receipt);c.save(folder/'inputs'/(rid+'.json'),dict(**payload,before={}))
    c.save(folder/'parent_profiles'/(rid+'.json'),parent);c.save(folder/(rid+'_child_profile.json'),child)
    wire=(json.dumps(dict(type='ready',metadata=metadata))+'\r\n').encode()+packet
    def stream(path,descriptor,**kwargs):
        for start in range(0,len(wire),11):kwargs['sink'](wire[start:start+11])
        return dict(passed=True)
    monkeypatch.setattr(v,'verify_log',stream)
    original=deepcopy(receipt);result=v.verify_raw_transport(folder,{},Envelope())
    assert result['failed_requests']==int(failed) and c.read(folder/(rid+'.json'))==original
    assert result['requests_checked'][0]['serialized_response_observed']
    bad=deepcopy(child);bad['serialized_packet_utf8_sha256']='f'*64
    (folder/(rid+'_child_profile.json')).write_text(c.canonical(bad))
    with pytest.raises(ValueError):v.verify_raw_transport(folder,{},Envelope())


def test_operational850ms_does_not_change_admission(monkeypatch):
    plan=schedule.generate_schedule('ab'*32);saved={}
    monkeypatch.setattr(r,'bytes_used',lambda:1000)
    monkeypatch.setattr(r,'play_game',lambda spec,*x:(dict(spec,reason='natural'),[{'completed':True,'clock':{'receipt_elapsed_seconds':.85}}],.01,None))
    monkeypatch.setattr(r,'final_evidence_forecast',lambda *x:{'projected_final_evidence_bytes':100})
    monkeypatch.setattr(r,'operational_gate',lambda **kw:{'passed':True})
    monkeypatch.setattr(c,'save',lambda path,value:saved.setdefault(path.name,value))
    games,stop=r.operational(Envelope(),{},plan,{'forecast_seconds':1})
    assert len(games)==6 and stop is None and saved['operational_gate.json']['passed']
    assert saved['operational_gate.json']['operational_700ms_is_descriptive_only']


def test_stage_cannot_dispatch_a_shortened_one_second_move(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'save',lambda *x:None)
    monkeypatch.setattr(clock.rules,'serialize_snapshot',lambda x:{})
    monkeypatch.setattr(clock.rules,'observation_payload',lambda x:{})
    monkeypatch.setattr(clock.rules,'legal_actions',lambda x:[])
    monkeypatch.setattr(clock.ledger,'check_observer',lambda *x:None)
    monkeypatch.setattr(clock,'request',lambda *x,**kw:pytest.fail('shortened move dispatched'))
    env=Envelope();env.remaining=lambda:.99
    child=SimpleNamespace(player='dom',output_dir=tmp_path/'worker',_counter=0)
    with pytest.raises(ValueError,match='full one-second'):clock.decision(child,SimpleNamespace(current_player=0),env,'challenger')


def test_attributable_failed_move_saved_before_loss_and_both_workers_close(monkeypatch,tmp_path):
    events=[];monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'save',lambda path,value:events.append(('save',path.name,deepcopy(value))))
    state=SimpleNamespace(terminal=False,current_player=0,winner=None)
    monkeypatch.setattr(r.rules,'CorrectedEnv',lambda **kw:SimpleNamespace(initial_state=lambda:state))
    monkeypatch.setattr(r.rules,'serialize_snapshot',lambda state:{'state':'unchanged'})
    monkeypatch.setattr(r,'decode_training_snapshot',lambda value:state)
    monkeypatch.setattr(r.schedules,'validate_spec',lambda spec:None)
    class Worker:
        def __init__(self,name):self.output_dir=tmp_path/name;self.name=name
        def __enter__(self):return self
        def __exit__(self,*x):events.append(('close',self.name))
    monkeypatch.setattr(r,'make_worker',lambda name,*x,**kw:Worker(name))
    row=dict(completed=False,player_role='challenger',player='dom',seat=0,action=None,after=None,trace=None,clock={'status':'timeout','request_id':'request-00000001'},worker_clock_file='worker/request-00000001.json')
    monkeypatch.setattr(r,'decision',lambda *x:deepcopy(row))
    monkeypatch.setattr(r,'verify_saved_decision',lambda *x:None)
    def failure(receipt):events.append(('classify',));return 'timeout'
    monkeypatch.setattr(v,'classify_failure',failure)
    monkeypatch.setattr(r,'verify_closed_worker',lambda *x:dict(logs={},raw={}))
    monkeypatch.setattr(v,'audit_game',lambda *x,**kw:{'status':'passed'})
    spec=dict(stage='main',pair_index=0,game_index=0,seed_decimal='1',challenger_seat=0)
    game,rows,cost,stop=r.play_game(spec,Envelope(),{})
    move=next(i for i,e in enumerate(events) if e[:2]==('save','0000.json'))
    ending=next(i for i,e in enumerate(events) if e[:2]==('save','result.json'))
    assert move<events.index(('classify',))<ending
    assert sum(e[0]=='close' for e in events)==2 and all(i<ending for i,e in enumerate(events) if e[0]=='close')
    assert game['reason']=='forfeit' and game['winner']==1 and stop['clock']==row['clock'] and len(rows)==1


def test_ram_recovery_waits_a_stable_minute_and_charges_existing_clock(monkeypatch):
    tick=[100.];readings=iter([0,c.GiB*3])
    monkeypatch.setattr(r.time,'monotonic',lambda:tick[0]);monkeypatch.setattr(r.time,'time',lambda:tick[0])
    monkeypatch.setattr(r.time,'sleep',lambda seconds:tick.__setitem__(0,tick[0]+seconds))
    def memory():
        try:value=next(readings)
        except StopIteration:value=3*c.GiB
        return SimpleNamespace(available=value)
    monkeypatch.setattr(r.psutil,'virtual_memory',memory);monkeypatch.setattr(c,'save',lambda *x:None)
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.admitted=True;env.guard=lambda:None
    env.admit();assert env.waited>=60 and tick[0]>=160


def test_dom_seed_domain_is_fresh_even_with_same_namespace():
    from tools.breck_screen_20261026 import schedule as prior
    namespace='4f'*32
    new={x['seed_decimal'] for x in schedule.generate_schedule(namespace)}
    old={x['seed_decimal'] for x in prior.generate_schedule(namespace)}
    assert not new.intersection(old)
    assert schedule.DOMAIN=='dom_screen_1s_20261027_v1'
    assert v.AUDIT_DOMAIN=='dom_screen_1s_20261027_audit_v1'


def test_owning_clock_cleanup_and_placement_semantics_unchanged():
    import inspect
    from tools.dom_screen_20261027 import watchdog,placement
    from tools.breck_screen_20261026 import watchdog as old_watchdog,placement as old_placement
    for name in ('cleanup_owned','inner_main','guardian_main'):
        old=inspect.getsource(getattr(old_watchdog,name)).replace('tools.breck_screen_20261026','tools.dom_screen_20261027')
        assert inspect.getsource(getattr(watchdog,name))==old
    for name in ('plan','configure_execution','verify_runtime','verify_saved_runtime','verify_saved_request'):
        assert inspect.getsource(getattr(placement,name))==inspect.getsource(getattr(old_placement,name))
    watchdog.require_isolated_request({'placement_mode':'isolated'})
    with pytest.raises(ValueError):watchdog.require_isolated_request({'placement_mode':'original'})


def test_saved_multiphase_reproduction_exactly_bound_without_forward(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    original=dict(completed=True,player_role='challenger',player='dom',quota=None,mode='dom',arm='dom',seat=0,
        observer={'lawful':True},before={'world':'private'},action={'complete':True},
        trace={'phases':[{'logits':[1.,2.]},{'logits':[3.,4.]}]},after={'world':'next'},clock={'result':{'scientific':'original'}})
    row=deepcopy(original);row['worker_clock_file']='out/workers/audit_02/request-00000001.json'
    checked=[];monkeypatch.setattr(r,'verify_saved_decision',lambda row,freeze:checked.append(True))
    r.verify_reproduction(row,original,2,{})
    assert checked==[True]
    row['trace']['phases'][1]['logits'][0]=9.
    with pytest.raises(ValueError,match='trace'):r.verify_reproduction(row,original,2,{})


def test_saved_reproduction_wrong_worker_or_failed_move_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    original=dict(completed=True,player_role='challenger',player='dom',quota=None,mode='dom',arm='dom',seat=0,
        observer={},before={},action={},trace={},after={},clock={'result':{}},worker_clock_file='out/workers/audit_03/request-00000001.json')
    monkeypatch.setattr(r,'verify_saved_decision',lambda *args:pytest.fail('Unbound row reached saved verifier'))
    with pytest.raises(ValueError,match='owning worker'):r.verify_reproduction(deepcopy(original),original,2,{})
    failed=deepcopy(original);failed['completed']=False
    with pytest.raises(ValueError,match='Failed moves'):r.verify_reproduction(failed,original,3,{})
