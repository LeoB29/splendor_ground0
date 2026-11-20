"""No-model original-clock, one-candidate and publication integration tests."""
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

from tools.splendorust_network_execution_20261031 import common as c,control as r,transport as t,watchdog as wd,independent_review as ir
from tools.splendorust_search256_recovery_20261030_02 import transport as old_transport
from tools.splendorust_search256_accounting_20261029 import WorkerAccounting,LOG_NAMES
from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n',encoding='utf-8')


def root_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');c.OUT.mkdir()
    write(c.OUT/'freeze.json',{'inputs':[]})
    return c.read(c.OUT/'freeze.json')


def supplement_fixture(tmp_path,monkeypatch):
    freeze=root_fixture(tmp_path,monkeypatch)
    write(c.OUT/'localization_result.json',{'profile':True})
    write(c.OUT/'localization_verification.json',{'status':'verified_localization'})
    module=c.ROOT/'tools/splendorust_network_execution_20261031/candidate_example.py'
    module.parent.mkdir(parents=True);module.write_text('def build_candidate(**kwargs):\n    return kwargs["compiled"]\n')
    module_desc=c.artifact(module)
    decision=c.OUT/'candidate_preparation/decision.json';write(decision,{'candidate':'example'})
    log=c.OUT/'candidate_preparation/tests.log';log.write_text('model-free tests passed\n')
    tests=c.OUT/'candidate_preparation/tests.json';write(tests,dict(exit_code=0,models=0,games=0,inputs=[module_desc],log=c.artifact(log)))
    review=c.OUT/'candidate_preparation/review.json';write(review,dict(approved=True,blocking_findings=[],inputs=[module_desc]))
    value=dict(schema='splendorust_network_execution_candidate_v1',phase1_freeze_sha256=c.file_hash(c.OUT/'freeze.json'),
        localization_result=c.artifact(c.OUT/'localization_result.json'),localization_verification=c.artifact(c.OUT/'localization_verification.json'),
        candidate_id='example',candidate_module=module_desc,candidate_import_module='tools.splendorust_network_execution_20261031.candidate_example',
        candidate_function='build_candidate',decision=c.artifact(decision),model_free_tests=c.artifact(tests),independent_review=c.artifact(review))
    value['inputs']=[module_desc,c.artifact(decision),c.artifact(tests),c.artifact(review),c.artifact(log)]
    envelope=SimpleNamespace(guard=lambda:None,resident={},remaining=lambda:3000.,start=100.,wall_start=1000.,deadline=6100.,wall_deadline=7000.)
    return freeze,value,envelope


def test_candidate_supplement_exact_source_and_localization_binding(tmp_path,monkeypatch):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch)
    assert r.verify_supplement(value,freeze,envelope)['candidate_id']=='example'
    (c.ROOT/value['candidate_module']['path']).write_text('changed candidate\n')
    with pytest.raises(ValueError,match='Changed frozen'):r.verify_supplement(value,freeze,envelope)


@pytest.mark.parametrize('mutation',['wrong_freeze','wrong_localization','wrong_module','old_module','bad_test','bad_review','missing_test_pin'])
def test_supplement_rejects_unreviewed_or_rebound_candidate(tmp_path,monkeypatch,mutation):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch)
    if mutation=='wrong_freeze':value['phase1_freeze_sha256']='0'*64
    elif mutation=='wrong_localization':value['localization_result']=value['decision']
    elif mutation=='wrong_module':value['candidate_import_module']='os'
    elif mutation=='old_module':freeze['inputs']=[value['candidate_module']]
    elif mutation=='missing_test_pin':value['inputs'].remove(value['model_free_tests'])
    else:
        key='model_free_tests' if mutation=='bad_test' else 'independent_review'
        path=c.ROOT/value[key]['path'];doc=c.read(path)
        doc.update(exit_code=1) if mutation=='bad_test' else doc.update(approved=False)
        write(path,doc);old=value[key];value[key]=c.artifact(path);value['inputs']=[value[key] if x==old else x for x in value['inputs']]
    with pytest.raises(ValueError):r.verify_supplement(value,freeze,envelope)


def test_single_commitment_keeps_original_clock_and_cannot_be_reused(tmp_path,monkeypatch):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch)
    write(c.OUT/'candidate_supplement.json',value)
    r.admit_candidate(value,freeze,envelope)
    receipt=c.read(c.OUT/'candidate_admission.json')
    assert receipt['original_monotonic_origin']==100. and receipt['original_unix_origin']==1000.
    assert receipt['required_seconds']==2400. and receipt['automatic_retries']==0
    with pytest.raises(FileExistsError):r.admit_candidate(value,freeze,envelope)


@pytest.mark.parametrize('tamper',[None,'required_seconds','original_monotonic_origin','effective_unix_deadline','candidate_id','monotonic'])
def test_independent_admission_recomputes_original_budget(tmp_path,monkeypatch,tamper):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch)
    write(c.OUT/'candidate_supplement.json',value)
    write(c.OUT/'stages/candidate_start.json',dict(monotonic=200.,unix_time=1100.,monotonic_deadline=3800.,unix_deadline=4700.))
    receipt=dict(phase1_freeze_sha256=c.file_hash(c.OUT/'freeze.json'),candidate_id='example',
        supplement=c.artifact(c.OUT/'candidate_supplement.json'),localization_verification=c.artifact(c.OUT/'localization_verification.json'),
        original_monotonic_origin=100.,original_unix_origin=1000.,commitment_consumed=True,automatic_retries=0,
        required_seconds=2400.,monotonic=201.,unix_time=1101.,effective_monotonic_deadline=3800.,effective_unix_deadline=4700.)
    if tamper=='candidate_id':receipt[tamper]='other'
    elif tamper:receipt[tamper]+=3000.
    if tamper:
        with pytest.raises(ValueError):r.verify_candidate_admission(receipt,value,envelope)
    else:assert r.verify_candidate_admission(receipt,value,envelope)['status']=='passed'


@pytest.mark.parametrize('remaining',[0.,100.,2399.,2400.])
def test_candidate_does_not_reset_remaining_clock_to_fit(tmp_path,monkeypatch,remaining):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch);envelope.remaining=lambda:remaining
    write(c.OUT/'candidate_supplement.json',value)
    with pytest.raises(ValueError,match='Remaining original'):r.admit_candidate(value,freeze,envelope)
    assert not(c.OUT/'candidate_admission.json').exists()


def test_candidate_cannot_start_while_localization_tree_resident(tmp_path,monkeypatch):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch);envelope.resident[123]=object()
    with pytest.raises(ValueError,match='No model overlaps'):r.admit_candidate(value,freeze,envelope)


def test_no_candidate_stop_and_supplement_are_mutually_exclusive(tmp_path,monkeypatch):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch)
    write(c.OUT/'candidate_supplement.json',value);write(c.OUT/'candidate_stop.json',{})
    with pytest.raises(ValueError,match='mutually exclusive'):r.wait_for_selection(envelope,freeze)


def test_selection_timeout_is_a_saved_localization_only_outcome(tmp_path,monkeypatch):
    freeze,value,envelope=supplement_fixture(tmp_path,monkeypatch);envelope.remaining=lambda:1.5
    assert r.wait_for_selection(envelope,freeze) is None
    assert(c.OUT/'selection_expired.json').exists() and not(c.OUT/'candidate_admission.json').exists()


@pytest.mark.parametrize('stage,now,wall,expected',[
    ('localization',110.,1010.,(1910.,2810.)),('selection',7000.,7900.,(7300.,8200.)),
    ('candidate',6500.,7400.,(7300.,8200.)),('final',10000.,10900.,(10900.,11800.))])
def test_stage_clocks_use_same_original_origins(tmp_path,monkeypatch,stage,now,wall,expected):
    root_fixture(tmp_path,monkeypatch);monkeypatch.setattr(r.time,'monotonic',lambda:now);monkeypatch.setattr(r.time,'time',lambda:wall)
    envelope=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.));monkeypatch.setattr(envelope,'guard',lambda:None)
    envelope.begin(stage);assert(envelope.deadline,envelope.wall_deadline)==expected
    start=c.read(c.OUT/'stages'/f'{stage}_start.json')
    assert not wd.stage_clock_open(start,envelope.deadline,wall)


def test_external_model_cutoff_and_final_reader_never_renew_original_clock():
    assert not wd.model_clock_open(100.,1000.,7300.,8200.,final_started=False,owned_children=[])
    with pytest.raises(ValueError):wd.model_clock_open(100.,1000.,7000.,7900.,final_started=True,owned_children=[123])
    assert ir.review_deadlines(dict(monotonic_origin=100.,unix_origin=1000.),
        dict(monotonic_deadline=10800.,unix_deadline=11700.),10500.,11400.)==(10800.,11700.)


def test_profile_operation_uses_real_fixed_accounting_and_existing_atomic_barrier(tmp_path,monkeypatch):
    root_fixture(tmp_path,monkeypatch);folder=c.OUT/'workers/localization';account=WorkerAccounting(folder);folder.mkdir(parents=True)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),_logs={n:SimpleNamespace() for n in LOG_NAMES},
        max_log_bytes=128*c.MiB,_closed=False,_counter=2,challenger_quota=256,process=SimpleNamespace(pid=os.getpid()),
        _placement_freeze={'unchanged_graph_backend_identity':{'graph_identity_sha256':'a'*64}},placement_mode='isolated',
        recovery_accounting=account,recovery_envelope=SimpleNamespace(time_guard=lambda:None,remaining=lambda:60.))
    for name in LOG_NAMES:(folder/name).write_bytes(b'')
    account.bind_child(child);account.finish_initialization();calls=[]
    def request(payload,timeout,validate=None):
        calls.append((payload,timeout));child._counter+=1;return dict(status='ok',request_id='request-00000003')
    child.request=request;monkeypatch.setattr(t.placement,'verify_runtime',lambda *a,**k:{})
    receipt=t.request(child,{'operation':'operator_profile'},120.)
    assert calls==[({'operation':'operator_profile'},120.)]
    publication=deepcopy(account._publication)
    publish_profile(folder,receipt['request_id'],publication['expected'],batch=False)
    assert account.finish_request(require_published=True)['state']=='idle'
    assert publication['maximum_bytes']==8193


def test_trusted_request_constructor_and_ordinary_clock_are_preserved():
    for name in ('request','_PlacementConstructor'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(t,name))))==ast.dump(ast.parse(inspect.getsource(getattr(old_transport,name))))
    assert t.PlacedProfiledStudyWorker.request is old_transport.PlacedProfiledStudyWorker.request
    assert t.PlacedProfiledStudyWorker._receive is old_transport.PlacedProfiledStudyWorker._receive
    source=inspect.getsource(t.ordinary)
    assert 'allowance==c.SECONDS_PER_MOVE' in source and 'envelope.admit()' in source
    assert c.SECONDS_PER_MOVE==10. and c.INTERNAL_SECONDS==9.


def test_registered_lifecycles_have_no_room_for_extra_candidate_or_warmups():
    values=r.lifecycle_inventory();assert set(values)=={'localization','control','candidate'}
    assert values['localization']['startup_requests']==3 and values['localization']['ordinary_requests']==0
    assert all(values[x]['startup_requests']==2 and values[x]['ordinary_requests']==72 for x in c.VARIANTS)
    assert sum(x['startup_requests']+x['ordinary_requests'] for x in values.values())==151


@pytest.mark.parametrize('selected,fail_phase',[(False,None),(True,None),(True,'profile'),(True,'candidate')])
def test_orchestration_closes_and_verifies_profile_before_single_candidate(tmp_path,monkeypatch,selected,fail_phase):
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
        events.append(('qualify',name));child=SimpleNamespace(_closed=False,name=name);workers[name]=child;return child
    def profile(*args):
        events.append(('profile',60))
        if fail_phase=='profile':raise ValueError('preserved profile failure')
    def evaluate(*args):
        events.append(('evaluate',144))
        if fail_phase=='candidate':raise ValueError('preserved candidate failure')
    def close(child,*args):events.append(('close',child.name));child._closed=True
    monkeypatch.setattr(r,'qualify',qualify);monkeypatch.setattr(r,'profile_request',profile)
    monkeypatch.setattr(r,'verify_localization',lambda *a,**k:events.append(('verified','profile')))
    monkeypatch.setattr(r,'wait_for_selection',lambda *a: {'selected':True} if selected else None)
    monkeypatch.setattr(r,'admit_candidate',lambda *a:events.append(('commit','one_candidate')))
    monkeypatch.setattr(r,'evaluate',evaluate);monkeypatch.setattr(t,'close_and_audit',close)
    monkeypatch.setattr(r,'final_checks',lambda *a:{'fixture':True})
    assert r.run(expected)==(1 if fail_phase else 0)
    if fail_phase!='profile':
        assert events.index(('close','localization'))<events.index(('verified','profile'))<events.index(('begin','selection'))
    if selected and fail_phase!='profile':
        assert events.index(('commit','one_candidate'))<events.index(('qualify','control'))<events.index(('qualify','candidate'))
        assert events.count(('evaluate',144))==1
    else:assert not any(e[0]=='evaluate' for e in events)
    assert c.read(c.OUT/'completion.json')['automatic_retries']==0


def test_new_owner_import_remains_model_free_in_actual_child():
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    code='import sys;from tools.splendorust_network_execution_20261031 import control,watchdog,independent_review,profile_worker;assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd=c.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
