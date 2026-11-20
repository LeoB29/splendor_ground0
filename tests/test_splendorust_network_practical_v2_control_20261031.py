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
from tools.splendorust_network_practical_v2_20261031 import common as c,control as r,transport as t,watchdog as wd,independent_review as ir
from tools.splendorust_search256_recovery_20261030_02 import transport as old_transport
from tools.splendorust_search256_accounting_20261029 import WorkerAccounting,LOG_NAMES
from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile
from tools import splendorust_network_practical_handoff_20261031 as held_handoff

WORKSPACE=Path(__file__).resolve().parents[1]
FAILED=WORKSPACE/'outputs/splendorust_network_practical_20261031_01'


def actual_live_child(folder,variant='control',deadline=1900.):
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


@pytest.mark.parametrize('stage,now,wall,expected',[
    ('qualification',110.,1010.,(1910.,2810.)),('evaluation',6500.,7400.,(7300.,8200.)),
    ('final',10000.,10900.,(10900.,11800.))])
def test_stage_clocks_keep_original_origins(tmp_path,monkeypatch,stage,now,wall,expected):
    root_fixture(tmp_path,monkeypatch);monkeypatch.setattr(r.time,'monotonic',lambda:now);monkeypatch.setattr(r.time,'time',lambda:wall)
    envelope=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.));monkeypatch.setattr(envelope,'guard',lambda:None)
    envelope.begin(stage);assert (envelope.deadline,envelope.wall_deadline)==expected
    assert not wd.stage_clock_open(c.read(c.OUT/'stages'/f'{stage}_start.json'),envelope.deadline,wall)


def test_external_model_cutoff_and_reader_remain_original_clock():
    assert not wd.model_clock_open(100.,1000.,7300.,8200.,final_started=False,owned_children=[])
    with pytest.raises(ValueError):wd.model_clock_open(100.,1000.,7000.,7900.,final_started=True,owned_children=[123])
    assert ir.review_deadlines(dict(monotonic_origin=100.,unix_origin=1000.),
        dict(monotonic_deadline=10800.,unix_deadline=11700.),10500.,11400.)==(10800.,11700.)


def test_final_cannot_start_with_resident_model_tree(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch);envelope=r.Envelope(dict(monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    envelope.resident[(123,1.)]=object();monkeypatch.setattr(envelope,'guard',lambda:None)
    with pytest.raises(ValueError,match='no resident'):envelope.begin('final')
    assert not (c.OUT/'stages/final_start.json').exists()


def test_numerical_operation_real_accounting_and_atomic_barrier(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch);folder=c.OUT/'workers/candidate';account=WorkerAccounting(folder);folder.mkdir(parents=True)
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
    root_fixture(tmp_path,monkeypatch);folder=c.OUT/'workers/candidate';freeze={'declared_worker_lifecycles':r.lifecycle_inventory()}
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
    folder=c.OUT/'workers'/name
    write(folder/'entrypoint_clock.json',{'result':{'candidate_different':True} if changed else {'exact_original':True}})
    write(folder/'entrypoint_profiles.json',{'parent_profile':{},'child_profile':{}})
    monkeypatch.setattr(t,'verify_graph_evidence',lambda *a,**k:{'accepted':True})
    monkeypatch.setattr(t,'verify_numerical_evidence',lambda *a,**k:{'accepted':True})
    monkeypatch.setattr(t,'_require_profiles',lambda *a:None);monkeypatch.setattr(t,'validate_profiles',lambda *a,**k:None)
    monkeypatch.setattr(r.qualification,'verify_entrypoint',lambda *a,**k:{'status':'passed'})
    if passed:assert r.qualification_evidence(name,SimpleNamespace(guard=lambda:None),{}, {},writing=True)['accepted']
    else:
        with pytest.raises(ValueError,match='Control full'):r.qualification_evidence(name,SimpleNamespace(guard=lambda:None),{}, {},writing=True)


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
    child=actual_live_child(c.OUT/'workers/candidate','candidate')
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    r.advance_idle_worker(child,envelope)
    receipt=c.read(c.OUT/'worker_phase_transitions/candidate.json')
    assert receipt['old_absolute_deadline']==1900. and child.absolute_deadline==receipt['new_absolute_deadline']==5000.
    assert receipt['active_request'] is False and receipt['next_request_id']=='request-00000004' and receipt['request_allowance_unchanged']==10.


@pytest.mark.parametrize('cause',['active','failed','not_qualified','renewed_cutoff'])
def test_idle_transition_cannot_move_active_or_failed_clock(tmp_path,monkeypatch,cause):
    root_fixture(tmp_path,monkeypatch)
    child=actual_live_child(c.OUT/'workers/candidate','candidate')
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    if cause=='active':child._lock.acquire()
    elif cause=='failed':child._failure='timeout'
    elif cause=='not_qualified':child._counter=2
    else:envelope.deadline=8000.
    with pytest.raises(ValueError):r.advance_idle_worker(child,envelope)
    assert not(c.OUT/'worker_phase_transitions/candidate.json').exists()


@pytest.mark.parametrize('tamper',[None,'before_suite','after_request','renewed_deadline'])
def test_saved_idle_transition_binds_actual_parent_clock_fields(tmp_path,monkeypatch,tamper):
    root_fixture(tmp_path,monkeypatch)
    write(c.OUT/'stages/qualification_start.json',{'monotonic_deadline':1900.})
    write(c.OUT/'stages/evaluation_start.json',{'monotonic_deadline':5000.})
    write(c.OUT/'evaluation_admission.json',{'monotonic':300.})
    folder=c.OUT/'workers/candidate'
    write(folder/'initialization.json',{'identity':{'pid':123,'created':1200.}})
    write(folder/'parent_profiles/request-00000003.json',{'capture_snapshot_ns':1000})
    value=dict(variant='candidate',worker_pid=123,worker_created=1200.,completed_startup_requests=3,
        next_request_id='request-00000004',active_request=False,request_allowance_unchanged=10.,
        original_monotonic_origin=100.,original_unix_origin=1000.,old_absolute_deadline=1900.,
        new_absolute_deadline=5000.,monotonic=301.,perf_counter_ns=1500)
    if tamper=='before_suite':value['perf_counter_ns']=999
    elif tamper=='after_request':value['perf_counter_ns']=2001
    elif tamper=='renewed_deadline':value['new_absolute_deadline']=5001.
    write(c.OUT/'worker_phase_transitions/candidate.json',value)
    args=('candidate',SimpleNamespace(start=100.,wall_start=1000.),[{'parent_profile':{'request_start_ns':2000}}])
    if tamper:
        with pytest.raises(ValueError):r.verify_phase_transition(*args)
    else:assert r.verify_phase_transition(*args)==value


@pytest.mark.parametrize('remaining',[0.,100.,1559.,1560.])
def test_evaluation_does_not_reset_clock_to_fit(tmp_path,monkeypatch,remaining):
    root_fixture(tmp_path,monkeypatch);envelope=SimpleNamespace(remaining=lambda:remaining)
    with pytest.raises(ValueError,match='Original evaluation'):r.evaluation_admission(envelope,{})
    assert not(c.OUT/'evaluation_admission.json').exists()


@pytest.mark.parametrize('tamper',[None,'required_seconds','original_monotonic_origin','effective_unix_deadline','monotonic'])
def test_saved_evaluation_admission_recomputes_cost_and_original_cutoff(tmp_path,monkeypatch,tamper):
    root_fixture(tmp_path,monkeypatch);write(c.OUT/'qualification_summary.json',{'accepted':True})
    write(c.OUT/'stages/evaluation_start.json',dict(monotonic=200.,unix_time=1100.,monotonic_deadline=3800.,unix_deadline=4700.))
    value=dict(freeze_sha256=c.file_hash(c.OUT/'freeze.json'),qualification_summary=c.artifact(c.OUT/'qualification_summary.json'),
        original_monotonic_origin=100.,original_unix_origin=1000.,required_seconds=1560.,automatic_retries=0,
        monotonic=201.,unix_time=1101.,effective_monotonic_deadline=3800.,effective_unix_deadline=4700.)
    if tamper:value[tamper]+=3000.
    write(c.OUT/'evaluation_admission.json',value)
    envelope=SimpleNamespace(start=100.,wall_start=1000.)
    if tamper:
        with pytest.raises(ValueError):r.verify_evaluation_admission(envelope)
    else:assert r.verify_evaluation_admission(envelope)==value


@pytest.mark.parametrize('outcome',['full','graph_rejection','numeric_rejection','infrastructure_failure'])
def test_valid_negative_finishes_final_reader_path_but_infrastructure_failure_is_incomplete(tmp_path,monkeypatch,outcome):
    root_fixture(tmp_path,monkeypatch);expected=c.file_hash(c.OUT/'freeze.json')
    write(c.OUT/'watchdog_claim.json',dict(freeze_sha256=expected,monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    events=[]
    class Envelope:
        def __init__(self,claim):self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin'];self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;events.append(('begin',stage))
        def finish(self):events.append(('finish',self.stage))
        def admit(self):pass
    monkeypatch.setattr(r,'Envelope',Envelope);monkeypatch.setattr(r.placement,'configure_execution',lambda *a:None)
    monkeypatch.setattr(r,'verify_runtime_sources',lambda:[]);monkeypatch.setattr(r,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(r,'load_inputs',lambda *a:({},[],{}));monkeypatch.setattr(r,'switch_placement',lambda *a:None)
    def qualify(name,workers,*args):
        events.append(('qualify',name));workers[name]=SimpleNamespace(_closed=False,name=name)
        if name=='candidate' and outcome=='infrastructure_failure':raise ValueError('preserved actual failure')
        rejection=(name=='candidate' and outcome!='full')
        at='graph' if outcome=='graph_rejection' else 'numerical' if rejection else None
        return dict(accepted=not rejection,rejected_at=at,graph_cases=11,numerical_cases=0 if at=='graph' else 38,
            scientific_decisions=0 if rejection else 67,parent_requests=1 if at=='graph' else 2 if rejection else 3)
    def close(child,*args):events.append(('close',child.name));child._closed=True
    monkeypatch.setattr(r,'qualify',qualify);monkeypatch.setattr(t,'close_and_audit',close)
    monkeypatch.setattr(r,'evaluation_admission',lambda *a:events.append(('admit','evaluation')))
    monkeypatch.setattr(r,'advance_idle_worker',lambda child,*a:events.append(('advance',child.name)))
    monkeypatch.setattr(r,'evaluate',lambda *a:events.append(('evaluate',144)))
    monkeypatch.setattr(r,'final_checks',lambda *a:events.append(('saved_final',True)) or {'verified':True})
    assert r.run(expected)==(1 if outcome=='infrastructure_failure' else 0)
    completion=c.read(c.OUT/'completion.json');assert completion['automatic_retries']==0
    assert events.count(('qualify','candidate'))==1 and events.count(('close','candidate'))==1
    if outcome=='full':
        assert events.index(('advance','candidate'))<events.index(('evaluate',144))<events.index(('saved_final',True))
    else:assert not any(x[0]=='evaluate' for x in events)
    if outcome!='infrastructure_failure':
        assert completion['status']=='completed_verified' and completion['failure'] is None
        assert events.index(('close','candidate'))<events.index(('begin','final'))
    else:assert completion['status']=='incomplete' and completion['failure'] and not any(x[0]=='saved_final' for x in events)


def test_new_owner_import_model_free_in_actual_child():
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    code='import sys;from tools.splendorust_network_practical_v2_20261031 import control,transport,watchdog,independent_review,worker;assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd=c.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('variant',['control','candidate'])
def test_v2_actual_saved_failure_live_shape_passes_real_handoff_and_saved_reader(tmp_path,monkeypatch,variant):
    root_fixture(tmp_path,monkeypatch)
    child=actual_live_child(c.OUT/'workers'/variant,variant)
    identity=deepcopy(child.identity)
    envelope=SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.)
    clock=SimpleNamespace(monotonic=lambda:301.,time=lambda:1201.,perf_counter_ns=lambda:1500)
    monkeypatch.setattr(held_handoff,'time',clock)
    # Use the real new controller entry point and held helper; neither is stubbed.
    r.advance_idle_worker(child,envelope)
    assert child.absolute_deadline==5000. and 'identity' not in child.initialization
    receipt=c.read(c.OUT/'worker_phase_transitions'/f'{variant}.json')
    assert receipt['worker_created']==identity['created'] and receipt['worker_pid']==identity['pid']
    write(c.OUT/'stages/qualification_start.json',{'monotonic_deadline':1900.})
    write(c.OUT/'stages/evaluation_start.json',{'monotonic_deadline':5000.})
    write(c.OUT/'evaluation_admission.json',{'monotonic':300.})
    write(child.output_dir/'parent_profiles/request-00000003.json',{'capture_snapshot_ns':1000})
    assert r.verify_phase_transition(variant,envelope,[{'parent_profile':{'request_start_ns':2000}}])==receipt


@pytest.mark.parametrize('variant',['control','candidate'])
@pytest.mark.parametrize('fault',['missing','live_created','disk_created','ready_pid'])
def test_v2_actual_identity_fault_stops_before_phase_mutation(tmp_path,monkeypatch,variant,fault):
    root_fixture(tmp_path,monkeypatch);child=actual_live_child(c.OUT/'workers'/variant,variant)
    if fault=='missing':del child.identity
    elif fault=='live_created':child.identity['created']+=1
    elif fault=='ready_pid':child.initialization['result']['pid']+=1
    else:
        value=c.read(child.output_dir/'initialization.json');value['identity']['created']+=1
        write(child.output_dir/'initialization.json',value)
    with pytest.raises((ValueError,AttributeError,KeyError)):
        r.advance_idle_worker(child,SimpleNamespace(start=100.,wall_start=1000.,deadline=5000.))
    assert child.absolute_deadline==1900. and not(c.OUT/'worker_phase_transitions'/f'{variant}.json').exists()


@pytest.mark.parametrize('outcome',['full','graph_rejection','numeric_rejection'])
def test_real_v2_handoff_in_controller_flow_reaches_all144_or_valid_negative(tmp_path,monkeypatch,outcome):
    selected=json.loads((FAILED/'panel.json').read_text());matrix=json.loads((FAILED/'evaluation_matrix.json').read_text())
    root_fixture(tmp_path,monkeypatch);expected=c.file_hash(c.OUT/'freeze.json')
    write(c.OUT/'watchdog_claim.json',dict(freeze_sha256=expected,monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    events=[];children={}
    class Envelope:
        def __init__(self,claim):self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin'];self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;self.deadline=self.start+c.STAGES[stage]+1;events.append(('begin',stage))
        def finish(self):events.append(('finish',self.stage))
        def admit(self):pass
        def guard(self):pass
    monkeypatch.setattr(r,'Envelope',Envelope);monkeypatch.setattr(r.placement,'configure_execution',lambda *a:None)
    monkeypatch.setattr(r,'verify_runtime_sources',lambda:[]);monkeypatch.setattr(r,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(r,'load_inputs',lambda *a:(selected,matrix,{}));monkeypatch.setattr(r,'switch_placement',lambda *a:None)
    def qualify(name,workers,envelope,*args):
        events.append(('qualify',name));child=actual_live_child(c.OUT/'workers'/name,name,envelope.deadline)
        workers[name]=children[name]=child;rejection=name=='candidate' and outcome!='full'
        at='graph' if outcome=='graph_rejection' and rejection else 'numerical' if rejection else None
        return dict(accepted=not rejection,rejected_at=at,graph_cases=11,numerical_cases=0 if at=='graph' else 38,
            scientific_decisions=0 if rejection else 67,parent_requests=1 if at=='graph' else 2 if rejection else 3)
    def close(child,*args):events.append(('close',child.name));child._closed=True
    roots={root['root_id']:root for root in selected['roots']}
    def ordinary(child,spec,observer,*args):
        assert (c.OUT/'worker_phase_transitions'/f'{child.name}.json').exists()
        assert child.absolute_deadline==children['control'].absolute_deadline==children['candidate'].absolute_deadline
        events.append(('ordinary',spec['index']));root=roots[spec['root_id']]
        return dict(spec=spec,observer=observer,completed=True,action=root['action'],trace=root['trace'])
    def validate(row,spec,*args):
        assert row['after']==roots[spec['root_id']]['after'] and row['spec']==spec
        events.append(('validated',spec['index']))
    monkeypatch.setattr(r,'qualify',qualify);monkeypatch.setattr(t,'close_and_audit',close)
    monkeypatch.setattr(r,'evaluation_admission',lambda *a:events.append(('admit','evaluation')))
    monkeypatch.setattr(t,'ordinary',ordinary);monkeypatch.setattr(r,'validate_row',validate)
    monkeypatch.setattr(r,'final_checks',lambda *a:events.append(('saved_final',True)) or {'verified':True})
    assert r.run(expected)==0
    assert c.read(c.OUT/'completion.json')['status']=='completed_verified'
    if outcome=='full':
        assert [x[1] for x in events if x[0]=='ordinary']==list(range(144))
        assert [x[1] for x in events if x[0]=='validated']==list(range(144))
        assert len(list((c.OUT/'observations').glob('*.json')))==144
        assert len(list((c.OUT/'worker_phase_transitions').glob('*.json')))==2
    else:
        assert not any(x[0]=='ordinary' for x in events)
        assert not list((c.OUT/'observations').glob('*.json'))
        assert not list((c.OUT/'worker_phase_transitions').glob('*.json'))
    assert events.index(('close','candidate'))<events.index(('saved_final',True))


def test_new_controller_changes_only_handoff_and_preserves_saved_identity_reader():
    old=ast.parse((WORKSPACE/'tools/splendorust_network_practical_20261031/control.py').read_text())
    new=ast.parse(Path(r.__file__).read_text())
    olddefs={node.name:node for node in old.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    newdefs={node.name:node for node in new.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    assert olddefs.keys()==newdefs.keys()
    changed=[name for name in olddefs if ast.dump(olddefs[name])!=ast.dump(newdefs[name])]
    assert changed==['advance_idle_worker']
    assert ast.dump(olddefs['verify_phase_transition'])==ast.dump(newdefs['verify_phase_transition'])
    assert 'common=c' in ast.unparse(newdefs['advance_idle_worker'])
