"""Model-free ownership, rejected-prefix, transition and exact-clocking regressions."""
import ast
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import pytest
from tools.splendorust_network_live_20261031 import common as c,control as r,transport as t,watchdog as wd,independent_review as ir
from tools.splendorust_search256_recovery_20261030_02 import transport as old_transport
from tools.splendorust_search256_accounting_20261029 import WorkerAccounting,LOG_NAMES
from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile
from tools import splendorust_network_practical_handoff_20261031 as held_handoff
from tools.splendorust_network_live_20261031 import design

WORKSPACE=Path(__file__).resolve().parents[1]
FAILED=WORKSPACE/'outputs/splendorust_network_practical_20261031_01'


def actual_live_child(folder,variant='control',deadline=1900.):
    variant=variant.removeprefix('live_').removeprefix('replay_')
    """Use actual saved READY bytes, deriving disk-only fields from the constructor."""
    prior=FAILED/'workers'/variant/'initialization.json'
    saved=json.loads(prior.read_text());tree=ast.parse((WORKSPACE/'tools/splendorust_network_practical_20261031/transport.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and
        n.func.id=='_new_json' and 'initialization.json' in ast.unparse(n))
    extra={key.value for key in call.args[1].keys if key is not None}
    assert extra=={'command','identity','affinity','topology','thread_environment'}
    live={key:deepcopy(value) for key,value in saved.items() if key not in extra}
    identity=deepcopy(saved['identity']);folder.mkdir(parents=True,exist_ok=True)
    (folder/'initialization.json').write_bytes(prior.read_bytes())
    assert 'identity' not in live
    return SimpleNamespace(_closed=False,_counter=3,_failure=None,_lock=threading.Lock(),
        process=SimpleNamespace(pid=identity['pid'],poll=lambda:None),identity=identity,
        _ps=SimpleNamespace(pid=identity['pid'],create_time=lambda:saved['identity']['created']),
        initialization=live,absolute_deadline=deadline,output_dir=folder,name=variant)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n',encoding='utf-8')

def root_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');c.OUT.mkdir()
    write(c.OUT/'freeze.json',{'inputs':[]});return c.read(c.OUT/'freeze.json')

def test_fixed_candidate_source_and_tolerances_cannot_be_relaxed():
    freeze=dict(r.contract(),candidate_module=c.artifact(c.CANDIDATE_SOURCE),candidate_import_module=c.CANDIDATE_MODULE,
        candidate_id=c.CANDIDATE_ID,candidate_function='build_candidate')
    assert r.validate_freeze_contract(freeze)
    for field in ['numerical_tolerances','candidate_module','candidate_id','seconds_per_move']:
        wrong=deepcopy(freeze)
        if field=='numerical_tolerances':wrong[field]['policy_atol']*=2
        elif field=='candidate_module':wrong[field]['sha256']='0'*64
        elif field=='candidate_id':wrong[field]='new_candidate'
        else:wrong[field]=20.
        with pytest.raises(ValueError):r.validate_freeze_contract(wrong)

def test_final_cannot_start_with_resident_model_tree(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch);envelope=r.Envelope(dict(monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    envelope.resident[(123,1.)]=object();monkeypatch.setattr(envelope,'guard',lambda:None)
    with pytest.raises(ValueError,match='no resident'):envelope.begin('final')
    assert not (c.OUT/'stages/final_start.json').exists()

def test_numerical_operation_real_accounting_and_atomic_barrier(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch);folder=c.OUT/'workers/live_candidate';account=WorkerAccounting(folder);folder.mkdir(parents=True)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),_logs={n:SimpleNamespace() for n in LOG_NAMES},
        max_log_bytes=128*c.MiB,_closed=False,_counter=1,challenger_quota=256,process=SimpleNamespace(pid=os.getpid()),
        _placement_freeze={'unchanged_graph_backend_identity':{'graph_identity_sha256':'a'*64}},placement_mode='isolated',
        recovery_accounting=account,recovery_envelope=SimpleNamespace(time_guard=lambda:None,remaining=lambda:60.))
    for name in LOG_NAMES:(folder/name).write_bytes(b'')
    account.bind_child(child);account.finish_initialization();calls=[]
    def request(payload,timeout,validate=None):
        calls.append((payload,timeout));child._counter+=1;return dict(status='ok',request_id='request-00000002')
    child.request=request;monkeypatch.setattr(t.placement,'verify_runtime',lambda *a,**k:{})
    receipt=t.request(child,{'operation':'numerical_checks'},60.)
    assert calls==[({'operation':'numerical_checks'},60.)]
    publication=deepcopy(account._publication)
    publish_profile(folder,receipt['request_id'],publication['expected'],batch=False)
    assert account.finish_request(require_published=True)['state']=='idle' and publication['maximum_bytes']==8193

def test_trusted_constructor_request_and_receive_remain_immutable():
    for name in ('request','_PlacementConstructor'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(t,name))))==ast.dump(ast.parse(inspect.getsource(getattr(old_transport,name))))
    assert t.PlacedProfiledStudyWorker.request is old_transport.PlacedProfiledStudyWorker.request
    assert t.PlacedProfiledStudyWorker._receive is old_transport.PlacedProfiledStudyWorker._receive
    assert 'allowance==c.SECONDS_PER_MOVE' in inspect.getsource(t.ordinary)
    assert c.SECONDS_PER_MOVE==10. and c.INTERNAL_SECONDS==9.

@pytest.mark.parametrize('rejected_at,ordinary,extra,expected',[
    ('graph',False,False,1),('numerical',False,False,2),(None,True,False,3),
    ('graph',True,False,None),('numerical',True,False,None),('graph',False,True,None)])
def test_exact_stopped_startup_prefix_not_laundered_as_full_qualification(tmp_path,monkeypatch,rejected_at,ordinary,extra,expected):
    root_fixture(tmp_path,monkeypatch);folder=c.OUT/'workers/live_candidate';freeze={'declared_worker_lifecycles':r.lifecycle_inventory()}
    write(folder/'startup_inputs/request-00000001.json',{'operation':'graph_parity'})
    if rejected_at!='graph':write(folder/'startup_inputs/request-00000002.json',{'operation':'numerical_checks'})
    if rejected_at is None:
        write(folder/'startup_inputs/request-00000003.json',{'operation':'entrypoint_checks'})
        receipt=dict(status='ok',request_id='request-00000003',result=dict(execution_route='research',counts={'complete_decisions':67}))
        write(folder/'request-00000003.json',receipt);write(folder/'entrypoint_clock.json',receipt)
    if ordinary:write(folder/'inputs/request-00000004.json',{'player':'search'})
    if extra:write(folder/'startup_inputs/request-00000002.json',{'operation':'numerical_checks'})
    monkeypatch.setattr(t,'verify_graph_evidence',lambda *a,**k:{'accepted':rejected_at!='graph'})
    monkeypatch.setattr(t,'verify_numerical_evidence',lambda *a,**k:{'accepted':rejected_at!='numerical'})
    if expected is None:
        with pytest.raises(ValueError):t.authenticated_lifecycle(folder,freeze,'research')
    else:
        result=t.authenticated_lifecycle(folder,freeze,'research')
        assert result['actual_startup_requests']==expected and result['numerical_rejection']==(rejected_at is not None)

@pytest.mark.parametrize('name,changed,passed',[('control',False,True),('control',True,False),('candidate',True,True)])
def test_historical_suite_equality_applies_to_control_not_candidate(tmp_path,monkeypatch,name,changed,passed):
    root_fixture(tmp_path,monkeypatch);monkeypatch.setattr(c,'OLD_OUT',tmp_path/'prior')
    write(c.OLD_OUT/'entrypoint_checks/research.json',{'result':{'exact_original':True}})
    folder=c.OUT/'workers'/('live_'+name)
    write(folder/'entrypoint_clock.json',{'result':{'candidate_different':True} if changed else {'exact_original':True}})
    write(folder/'entrypoint_profiles.json',{'parent_profile':{},'child_profile':{}})
    monkeypatch.setattr(t,'verify_graph_evidence',lambda *a,**k:{'accepted':True})
    monkeypatch.setattr(t,'verify_numerical_evidence',lambda *a,**k:{'accepted':True})
    monkeypatch.setattr(t,'_require_profiles',lambda *a:None);monkeypatch.setattr(t,'validate_profiles',lambda *a,**k:None)
    monkeypatch.setattr(r.qualification,'verify_entrypoint',lambda *a,**k:{'status':'passed'})
    if passed:assert r.qualification_evidence('live_'+name,SimpleNamespace(guard=lambda:None),{}, {},writing=True)['accepted']
    else:
        with pytest.raises(ValueError,match='Control full'):r.qualification_evidence('live_'+name,SimpleNamespace(guard=lambda:None),{}, {},writing=True)

def test_candidate_transition_uses_its_own_legal_action_on_real_saved_state():
    panel=c.read(c.PREVIOUS_OUT/'panel.json')
    root=panel['roots'][0];before=deepcopy(root['before'])
    state=r.decode_training_snapshot(before)
    alternatives=[r.rules.serialize_action(a) for a in r.rules.CorrectedEnv().legal_actions(state)
                  if r.rules.serialize_action(a)!=root['action']]
    assert alternatives
    after=r.actual_after(before,alternatives[0])
    assert after==r.ledger.transition(before,alternatives[0]) and before==root['before']
    assert after!=root['after']

def test_illegal_candidate_action_cannot_acquire_historical_after():
    root=c.read(c.PREVIOUS_OUT/'panel.json')['roots'][0]
    bad=deepcopy(root['action']);bad['card_id']='INVALID_CARD'
    with pytest.raises((ValueError,KeyError,AssertionError)):r.actual_after(root['before'],bad)

def test_idle_transition_keeps_original_cutoff_and_records_seam(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch)
    child=actual_live_child(c.OUT/'workers/live_candidate','candidate')
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    r.advance_idle_worker(child,envelope)
    receipt=c.read(c.OUT/'worker_phase_transitions/live_candidate.json')
    assert receipt['old_absolute_deadline']==1900. and child.absolute_deadline==receipt['new_absolute_deadline']==5000.
    assert receipt['active_request'] is False and receipt['next_request_id']=='request-00000004' and receipt['request_allowance_unchanged']==10.

@pytest.mark.parametrize('cause',['active','failed','not_qualified','renewed_cutoff'])
def test_idle_transition_cannot_move_active_or_failed_clock(tmp_path,monkeypatch,cause):
    root_fixture(tmp_path,monkeypatch)
    child=actual_live_child(c.OUT/'workers/live_candidate','candidate')
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    if cause=='active':child._lock.acquire()
    elif cause=='failed':child._failure='timeout'
    elif cause=='not_qualified':child._counter=2
    else:envelope.deadline=12000.
    with pytest.raises(ValueError):r.advance_idle_worker(child,envelope)
    assert not(c.OUT/'worker_phase_transitions/live_candidate.json').exists()

@pytest.mark.parametrize('tamper',[None,'before_suite','after_request','renewed_deadline'])
def test_saved_idle_transition_binds_actual_parent_clock_fields(tmp_path,monkeypatch,tamper):
    root_fixture(tmp_path,monkeypatch)
    write(c.OUT/'stages/qualification_start.json',{'monotonic_deadline':1900.})
    write(c.OUT/'stages/live_start.json',{'monotonic_deadline':5000.})
    write(c.OUT/'live_admission.json',{'monotonic':300.})
    folder=c.OUT/'workers/live_candidate'
    write(folder/'initialization.json',{'identity':{'pid':123,'created':1200.}})
    write(folder/'parent_profiles/request-00000003.json',{'capture_snapshot_ns':1000})
    value=dict(variant='live_candidate',worker_pid=123,worker_created=1200.,completed_startup_requests=3,
        next_request_id='request-00000004',active_request=False,request_allowance_unchanged=10.,
        original_monotonic_origin=100.,original_unix_origin=1000.,old_absolute_deadline=1900.,
        new_absolute_deadline=5000.,monotonic=301.,perf_counter_ns=1500)
    if tamper=='before_suite':value['perf_counter_ns']=999
    elif tamper=='after_request':value['perf_counter_ns']=2001
    elif tamper=='renewed_deadline':value['new_absolute_deadline']=5001.
    write(c.OUT/'worker_phase_transitions/live_candidate.json',value)
    args=('live_candidate',SimpleNamespace(start=100.,wall_start=1000.),[{'parent_profile':{'request_start_ns':2000}}])
    if tamper:
        with pytest.raises(ValueError):r.verify_phase_transition(*args)
    else:assert r.verify_phase_transition(*args)==value

def test_new_owner_import_model_free_in_actual_child():
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    code='import sys;from tools.splendorust_network_live_20261031 import control,transport,watchdog,independent_review,worker;assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd=c.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr

@pytest.mark.parametrize('variant',['control','candidate'])
def test_v2_actual_saved_failure_live_shape_passes_real_handoff_and_saved_reader(tmp_path,monkeypatch,variant):
    root_fixture(tmp_path,monkeypatch)
    child=actual_live_child(c.OUT/'workers'/('live_'+variant),variant)
    identity=deepcopy(child.identity)
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    clock=SimpleNamespace(monotonic=lambda:301.,time=lambda:1201.,perf_counter_ns=lambda:1500)
    monkeypatch.setattr(held_handoff,'time',clock)
    # Use the real new controller entry point and held helper; neither is stubbed.
    r.advance_idle_worker(child,envelope)
    assert child.absolute_deadline==5000. and 'identity' not in child.initialization
    receipt=c.read(c.OUT/'worker_phase_transitions'/f'live_{variant}.json')
    assert receipt['worker_created']==identity['created'] and receipt['worker_pid']==identity['pid']
    write(c.OUT/'stages/qualification_start.json',{'monotonic_deadline':1900.})
    write(c.OUT/'stages/live_start.json',{'monotonic_deadline':5000.})
    write(c.OUT/'live_admission.json',{'monotonic':300.})
    write(child.output_dir/'parent_profiles/request-00000003.json',{'capture_snapshot_ns':1000})
    assert r.verify_phase_transition('live_'+variant,envelope,[{'parent_profile':{'request_start_ns':2000}}])==receipt

@pytest.mark.parametrize('variant',['control','candidate'])
@pytest.mark.parametrize('fault',['missing','live_created','disk_created','ready_pid'])
def test_v2_actual_identity_fault_stops_before_phase_mutation(tmp_path,monkeypatch,variant,fault):
    root_fixture(tmp_path,monkeypatch);child=actual_live_child(c.OUT/'workers'/('live_'+variant),variant)
    if fault=='missing':del child.identity
    elif fault=='live_created':child.identity['created']+=1
    elif fault=='ready_pid':child.initialization['result']['pid']+=1
    else:
        value=c.read(child.output_dir/'initialization.json');value['identity']['created']+=1
        write(child.output_dir/'initialization.json',value)
    with pytest.raises((ValueError,AttributeError,KeyError)):
        r.advance_idle_worker(child,SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.))
    assert child.absolute_deadline==1900. and not(c.OUT/'worker_phase_transitions'/f'live_{variant}.json').exists()


@pytest.mark.parametrize('stage,now,wall,expected',[
    ('qualification',110.,1010.,(1910.,2810.)),('live',9500.,10400.,(10900.,11800.)),
    ('replay',10700.,11600.,(10900.,11800.)),('final',14000.,14900.,(14500.,15400.))])
def test_new_stage_clocks_never_renew_original_origin(tmp_path,monkeypatch,stage,now,wall,expected):
    root_fixture(tmp_path,monkeypatch);monkeypatch.setattr(r.time,'monotonic',lambda:now);monkeypatch.setattr(r.time,'time',lambda:wall)
    envelope=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.));monkeypatch.setattr(envelope,'guard',lambda:None)
    envelope.begin(stage);assert(envelope.deadline,envelope.wall_deadline)==expected
    assert not wd.stage_clock_open(c.read(c.OUT/'stages'/f'{stage}_start.json'),envelope.deadline,wall)


def test_external_cutoff_and_saved_reader_obey_new_original_clock():
    assert not wd.model_clock_open(100.,1000.,10900.,11800.,final_started=False,owned_children=[])
    with pytest.raises(ValueError):wd.model_clock_open(100.,1000.,10000.,10900.,final_started=True,owned_children=[1])
    assert ir.review_deadlines(dict(monotonic_origin=100.,unix_origin=1000.),
        dict(monotonic_deadline=14400.,unix_deadline=15300.),14000.,14900.)==(14400.,15300.)


@pytest.mark.parametrize('tamper',[None,'required_seconds','original_monotonic_origin','effective_unix_deadline','monotonic'])
def test_admission_independently_recomputes_unchanged_original_deadline(tmp_path,monkeypatch,tamper):
    root_fixture(tmp_path,monkeypatch);write(c.OUT/'stages/live_start.json',dict(monotonic=200.,unix_time=1100.,monotonic_deadline=5600.,unix_deadline=6500.))
    value=dict(freeze_sha256=c.file_hash(c.OUT/'freeze.json'),original_monotonic_origin=100.,original_unix_origin=1000.,
        required_seconds=1800.,automatic_retries=0,monotonic=201.,unix_time=1101.,effective_monotonic_deadline=5600.,effective_unix_deadline=6500.)
    if tamper:value[tamper]+=6000.
    write(c.OUT/'live_admission.json',value);envelope=SimpleNamespace(start=100.,wall_start=1000.)
    if tamper:
        with pytest.raises(ValueError):r.verify_admission(envelope,'live',1800.)
    else:assert r.verify_admission(envelope,'live',1800.)==value


@pytest.mark.parametrize('remaining',[0.,100.,1799.,1800.])
def test_no_live_admission_with_insufficient_remaining_clock(tmp_path,monkeypatch,remaining):
    root_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):r.phase_admission(SimpleNamespace(remaining=lambda:remaining),'live',1800.)
    assert not(c.OUT/'live_admission.json').exists()


def test_worker_namespaces_are_bounded_and_variant_identity_is_separate(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch);freeze={'declared_worker_lifecycles':r.lifecycle_inventory()}
    for name in c.LIFECYCLES:
        value=t.declared_lifecycle(c.OUT/'workers'/name,freeze,'research')
        assert value['execution_variant']==name.split('_')[1]
        assert value['ordinary_requests']==(6000 if name.startswith('live_') else 48)
    for name in ('candidate','control','replay2_candidate','live_128'):
        with pytest.raises(ValueError):t.declared_lifecycle(c.OUT/'workers'/name,freeze,'research')
    with pytest.raises(ValueError):t.declared_lifecycle(tmp_path/'outside/live_control',freeze,'research')


@pytest.mark.parametrize('outcome',['full','graph_rejection','numeric_rejection','game_failure','replay_failure'])
def test_full_owner_flow_six_games_fresh_replays_or_preserved_stop(tmp_path,monkeypatch,outcome):
    root_fixture(tmp_path,monkeypatch);expected=c.file_hash(c.OUT/'freeze.json')
    write(c.OUT/'watchdog_claim.json',dict(freeze_sha256=expected,monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    events=[];children={};schedule=design.make_schedule()
    class Envelope:
        def __init__(self,claim):self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin'];self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;self.deadline=self.start+c.STAGES[stage]+1;events.append(('begin',stage))
        def finish(self):events.append(('finish',self.stage))
        def admit(self):pass
        def guard(self):pass
    monkeypatch.setattr(r,'Envelope',Envelope);monkeypatch.setattr(r.placement,'configure_execution',lambda *a:None)
    monkeypatch.setattr(r,'verify_runtime_sources',lambda:[]);monkeypatch.setattr(r,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(r,'load_inputs',lambda *a:(schedule,{}));monkeypatch.setattr(r,'switch_placement',lambda *a:None)
    def qualify(name,workers,envelope,*args):
        events.append(('qualify',name));child=actual_live_child(c.OUT/'workers'/name,name,envelope.deadline)
        child.name=name;workers[name]=children[name]=child
        rejection=name=='live_candidate' and outcome in ('graph_rejection','numeric_rejection')
        at='graph' if rejection and outcome=='graph_rejection' else 'numerical' if rejection else None
        return dict(accepted=not rejection,rejected_at=at,graph_cases=11,numerical_cases=0 if at=='graph' else 38,
            scientific_decisions=0 if rejection else 67,parent_requests=1 if at=='graph' else 2 if rejection else 3)
    def close(child,*args):events.append(('close',child.name));child._closed=True
    def game(spec,workers,envelope,freeze):
        assert all((c.OUT/'worker_phase_transitions'/f'{n}.json').exists() for n in ('live_control','live_candidate'))
        assert set(workers)=={'live_control','live_candidate'};events.append(('game',spec['game_index']))
        if outcome=='game_failure':raise ValueError('saved failed live request; no replacement')
    def replays(selection,workers,*args):
        assert set(workers)=={'replay_control','replay_candidate'}
        assert all(children[n]._closed for n in ('live_control','live_candidate'))
        events.append(('replays',len(selection['decisions'])))
        if outcome=='replay_failure':raise ValueError('full output mismatch remains incomplete')
    monkeypatch.setattr(r,'qualify',qualify);monkeypatch.setattr(t,'close_and_audit',close)
    monkeypatch.setattr(r,'phase_admission',lambda env,name,*a,**k:events.append(('admit',name)))
    monkeypatch.setattr(r,'play_game',game);monkeypatch.setattr(r,'read_games',lambda *a:([],[]))
    monkeypatch.setattr(design,'select_replay_sample',lambda *a,**k:{'decisions':[{'variant':'control'},{'variant':'candidate'}]})
    monkeypatch.setattr(r,'replay_selected',replays)
    monkeypatch.setattr(r,'final_checks',lambda *a:events.append(('saved_final',True)) or {'completed_games':6 if outcome=='full' else 0})
    assert r.run(expected)==(1 if outcome in ('game_failure','replay_failure') else 0)
    value=c.read(c.OUT/'completion.json');assert value['automatic_retries']==0
    if outcome=='full':
        assert [v for k,v in events if k=='game']==list(range(6))
        assert [v for k,v in events if k=='qualify']==list(c.LIFECYCLES)
        assert events.index(('close','live_candidate'))<events.index(('qualify','replay_control'))<events.index(('replays',2))
        assert events.index(('close','replay_candidate'))<events.index(('begin','final'))
    elif 'rejection' in outcome:
        assert not any(k in ('game','replays') for k,v in events) and value['status']=='completed_verified'
    elif outcome=='game_failure':
        assert [v for k,v in events if k=='game']==[0] and not any(k=='replays' for k,v in events)
    if 'failure' in outcome:
        assert value['status']=='incomplete' and not any(k=='saved_final' for k,v in events) and (c.OUT/'failure.json').exists()
    assert all(child._closed for child in children.values())


@pytest.mark.parametrize('failure',['hard_guard','nonterminal_cap'])
def test_played_failure_saved_before_stop_and_never_invents_terminal_result(tmp_path,monkeypatch,failure):
    initial=design.initial_snapshot(design.make_schedule()[0]);root_fixture(tmp_path,monkeypatch)
    spec=design.make_schedule()[0];monkeypatch.setattr(design,'initial_snapshot',lambda s:initial)
    monkeypatch.setattr(c,'MAX_PLIES',2);calls=[]
    def decide(child,spec,before,envelope,freeze,*,ply):
        calls.append(ply)
        return dict(**spec,spec=spec,ply=ply,completed=failure!='hard_guard',variant='candidate',worker_clock_file='savedclock',
            before=before,after=before,action={'action_type':'TAKE'},clock={'receipt_elapsed_seconds':.1})
    monkeypatch.setattr(r,'decision',decide)
    def validate(row,*a):
        if not row['completed']:raise ValueError('preserved full10s failed response')
    monkeypatch.setattr(r,'validate_row',validate)
    with pytest.raises(ValueError):r.play_game(spec,{'live_control':object(),'live_candidate':object()},SimpleNamespace(),{})
    assert calls==[0] and (r.game_folder(spec)/'moves/000000.json').exists()
    assert not(r.game_folder(spec)/'result.json').exists()
    assert (c.OUT/'collection_stop.json').exists()==(failure=='hard_guard')


def test_controller_and_saved_reader_use_fresh_replay_scientific_result_and_source_binding():
    source=inspect.getsource(r.verify_replays)
    assert 'analysis.verify_replay' in source and "row['source_move']==c.artifact(path)" in source
    assert 'verify_replays(selection' in inspect.getsource(r.verify_saved)
    assert 'c.read(c.OUT/\'replay_selection.json\')==selection' in inspect.getsource(r.verify_saved)
    assert 'init[\'identity\'][\'created\']>admit[\'unix_time\']' in inspect.getsource(r.verify_saved)
