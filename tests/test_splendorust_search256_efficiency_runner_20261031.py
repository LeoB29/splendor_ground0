"""Model-free controller, clock, publication and exact-science integration checks."""
from copy import deepcopy
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from tools import splendorust_search256_accounting_20261029 as accounting
from tools.splendorust_search256_efficiency_20261031 import (
    common as c, runner as r, transport as t, worker as w, watchdog as wd,
    independent_review as reader, qualification as q, panel, design)
from tools.splendorust_search256_recovery_20261030_02 import transport as accepted_transport


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n',encoding='utf-8')


def account_fixture(tmp_path,monkeypatch,mode='off',rid='request-00000003'):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out')
    folder=c.OUT/'workers'/('observer_'+mode);account=accounting.WorkerAccounting(folder)
    folder.mkdir(parents=True)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),
        _logs={name:SimpleNamespace() for name in accounting.LOG_NAMES},max_log_bytes=128*c.MiB,_closed=False)
    for name in accounting.LOG_NAMES:(folder/name).write_bytes(b'')
    account.bind_child(child);account.finish_initialization()
    payload={'player':'search','observer':{'fixture':True}}
    write(folder/'inputs'/(rid+'.json'),payload)
    proof=account.begin_request(rid,payload,pid=os.getpid(),quota=256,graph_identity_sha256='a'*64)
    evidence=deepcopy(proof['publication']['expected'])
    evidence.update(result_sha256='b'*64,serialized_packet_utf8_sha256='c'*64,
        serialized_packet_utf8_bytes=60,outcome='response_flushed',error=None,
        pre_write_ns=10,post_flush_ns=20,profiles=[])
    context=dict(folder=folder,observer_mode=mode,quota=256,efficiency_profiles=[])
    init=dict(identity={'pid':os.getpid()},telemetry={'process_tree':[
        dict(role='search',pid=123,created=11.),dict(role='reference',pid=124,created=12.)]})
    write(folder/'initialization.json',init)
    child.recovery_accounting=account;child.placement_mode='isolated';child.last_placement_error=None
    child.last_placement_evidence={};child.last_parent_profile=dict(request_id=rid,
        request_raw_sha256=evidence['request_raw_sha256'])
    receipt=dict(status='ok',request_id=rid,result={})
    return child,account,context,payload,evidence,receipt


def test_real_accounting_accepts_direct_efficiency_growth_before_atomic_barrier(tmp_path,monkeypatch):
    child,account,context,payload,evidence,receipt=account_fixture(tmp_path,monkeypatch)
    observed=[];real_fsync=w.os.fsync
    def fsync(fd):
        real_fsync(fd)
        if not observed:
            assert (child.output_dir/'request-00000003_efficiency_profile.json').exists()
            assert not (child.output_dir/'request-00000003_child_profile.json').exists()
            observed.append(account.snapshot())
    monkeypatch.setattr(w.os,'fsync',fsync)
    w.publish_evidence(context,receipt['request_id'],evidence)
    assert len(observed)==1 and account.state=='request'
    envelope=SimpleNamespace(remaining=lambda:60.,guard=lambda:None,time_guard=lambda:None)
    result=t.collect_profiles(child,receipt,envelope)
    assert result['profile_capture_status']=='complete' and account.state=='idle'
    assert result['component_profile']['profiles']==[]
    assert not list(child.output_dir.glob('*efficiency*.pending'))
    assert account.snapshot()['reserved_bytes']==0


def test_fsync_failure_preserves_efficiency_file_without_publication_or_retry(tmp_path,monkeypatch):
    child,account,context,_,evidence,receipt=account_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(w.os,'fsync',lambda fd:(_ for _ in ()).throw(OSError('saved fixture fsync failure')))
    with pytest.raises(OSError,match='fsync failure'):w.publish_evidence(context,receipt['request_id'],evidence)
    assert (child.output_dir/'request-00000003_efficiency_profile.json').exists()
    assert not (child.output_dir/'request-00000003_child_profile.json').exists()
    with pytest.raises(accounting.AccountingError):account.finish_request(require_published=True)
    with pytest.raises(FileExistsError):w.publish_evidence(context,receipt['request_id'],evidence)


@pytest.mark.parametrize('change',['missing','partial','foreign_request'])
def test_completed_boundary_cannot_hide_missing_or_malformed_component_file(tmp_path,monkeypatch,change):
    child,account,context,_,evidence,receipt=account_fixture(tmp_path,monkeypatch)
    w.publish_evidence(context,receipt['request_id'],evidence)
    path=child.output_dir/'request-00000003_efficiency_profile.json'
    if change=='missing':path.unlink()
    elif change=='partial':path.write_text('{',encoding='utf-8')
    else:
        value=json.loads(path.read_text());value['request_id']='request-00000004';write(path,value)
    result=t.collect_profiles(child,receipt,SimpleNamespace(remaining=lambda:60.,guard=lambda:None,time_guard=lambda:None))
    assert result['profile_capture_status']=='evidence_error'
    with pytest.raises(ValueError):t._require_profiles(result)


@pytest.mark.parametrize('identity',[(123,11.),(124,12.),(123,12.),(999,11.)])
def test_ordinary_native_cpu_identity_must_be_the_authenticated_search_process(tmp_path,monkeypatch,identity):
    child,_,context,payload,evidence,receipt=account_fixture(tmp_path,monkeypatch,mode='on')
    write(child.output_dir/'request-00000003_efficiency_profile.json',{})
    monkeypatch.setattr(w,'verify_efficiency_profile',lambda *a,**k:{'status':'passed'})
    component={'profiles':[dict(native_pid=identity[0],native_created=identity[1])]}
    if identity==(123,11.):assert t.verify_efficiency_evidence(child.output_dir,receipt,{},evidence,component)['status']=='passed'
    else:
        with pytest.raises(ValueError,match='authenticated scientific native'):
            t.verify_efficiency_evidence(child.output_dir,receipt,{},evidence,component)


def test_trusted_request_and_constructor_bodies_remain_exact():
    for name in ('request','_PlacementConstructor'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(t,name))))==ast.dump(ast.parse(inspect.getsource(getattr(accepted_transport,name))))
    # Only evidence collection follows the trusted response clock.
    assert t.PlacedProfiledStudyWorker.request is accepted_transport.PlacedProfiledStudyWorker.request
    assert t.PlacedProfiledStudyWorker._receive is accepted_transport.PlacedProfiledStudyWorker._receive


@pytest.mark.parametrize('stage,mono,wall,expected',[
    ('qualification',110.,1010.,(1910.,2810.)),
    ('diagnostic',7000.,7900.,(7300.,8200.)),
    ('final',10000.,10900.,(10900.,11800.)),
])
def test_stage_clocks_are_clipped_to_original_dual_clock(tmp_path,monkeypatch,stage,mono,wall,expected):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    monkeypatch.setattr(r.time,'monotonic',lambda:mono);monkeypatch.setattr(r.time,'time',lambda:wall)
    envelope=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    monkeypatch.setattr(envelope,'guard',lambda:None)
    envelope.begin(stage)
    assert (envelope.deadline,envelope.wall_deadline)==expected
    saved=c.read(tmp_path/'stages'/f'{stage}_start.json')
    assert wd.stage_clock_open(saved,envelope.deadline-.1,envelope.wall_deadline-.1)
    assert not wd.stage_clock_open(saved,envelope.deadline,envelope.wall_deadline-.1)
    assert not wd.stage_clock_open(saved,envelope.deadline-.1,envelope.wall_deadline)


def test_saved_only_final_requires_no_resident_model():
    envelope=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.));envelope.resident['test']=object()
    with pytest.raises(ValueError,match='no resident'):envelope.begin('final')
    with pytest.raises(ValueError,match='cannot own'):
        wd.model_clock_open(100.,1000.,8000.,8900.,final_started=True,owned_children=[object()])
    assert not wd.model_clock_open(100.,1000.,7300.,8200.,final_started=False,owned_children=[])


def test_reader_does_not_receive_a_new_final_or_overall_allowance():
    claim=dict(monotonic_origin=100.,unix_origin=1000.)
    final=dict(monotonic_deadline=10500.,unix_deadline=11400.)
    assert reader.review_deadlines(claim,final,10000.,10900.)==(10500.,11400.)
    guard=object.__new__(reader.Guard)
    with pytest.raises(ValueError,match='never receives'):guard.close_worker('fixture',object())


def test_phase_transition_requires_idle_requests_and_preserves_counters(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    workers={m:SimpleNamespace(_closed=False,recovery_accounting=SimpleNamespace(state='idle'),
        output_dir=tmp_path/('observer_'+m),absolute_deadline=100.,_counter=26) for m in c.OBSERVER_MODES}
    envelope=SimpleNamespace(stage='diagnostic',effective_deadline=lambda:500.)
    r.transition_workers(workers,envelope)
    assert all(w.absolute_deadline==500. and w._counter==26 for w in workers.values())
    workers['on'].recovery_accounting.state='request'
    with pytest.raises(ValueError,match='active move'):r.transition_workers(workers,envelope)


def test_narrow_qualification_retains_only_actual256_suites_and_all_counterexamples():
    plan=q.build_plan()
    assert set(plan['actual_suites'])=={'256'}
    assert len(plan['actual_suites']['256']['counterexamples'])==15
    assert [(x['mode'],x['route'],x['complete_decisions']) for x in plan['entrypoint_requests']]==[
        ('off','research',67),('on','research',67)]
    assert plan['scientific_decisions']==134 and plan['ordinary_requests']==0 and plan['total_parent_requests']==4
    assert 'panel_requests' not in plan
    c.verify([plan['accepted_suite'],plan['accepted_plan']])


@pytest.mark.parametrize('passed,fail_phase',[(False,None),(True,None),(True,'overhead'),(True,'diagnostic')])
def test_controller_gates_full_panel_and_stops_failures_without_retry(tmp_path,monkeypatch,passed,fail_phase):
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',tmp_path/'out');c.OUT.mkdir()
    write(c.OUT/'freeze.json',{});expected=c.file_hash(c.OUT/'freeze.json')
    write(c.OUT/'watchdog_claim.json',dict(freeze_sha256=expected,monotonic_origin=r.time.monotonic(),unix_origin=r.time.time()))
    events=[]
    class Envelope:
        def __init__(self,claim):self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin'];self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;events.append(('begin',stage))
        def finish(self):events.append(('end',self.stage))
        def admit(self):pass
    monkeypatch.setattr(r,'Envelope',Envelope);monkeypatch.setattr(r.placement,'configure_execution',lambda *a:None)
    monkeypatch.setattr(r,'verify_runtime_sources',lambda:[]);monkeypatch.setattr(r,'verify_runtime',lambda *a:None)
    monkeypatch.setattr(r,'load_inputs',lambda *a:({},{}));monkeypatch.setattr(r,'switch_placement',lambda *a:None)
    def qualify(workers,*args):
        events.append(('qualify',134));workers.update({m:SimpleNamespace(_closed=False,mode=m) for m in c.OBSERVER_MODES})
    monkeypatch.setattr(r,'qualification_checks',qualify)
    monkeypatch.setattr(r,'transition_workers',lambda *a:events.append(('idle_transition',a[1].stage)))
    def collect(phase,*args):
        events.append(('collect',phase))
        if phase==fail_phase:raise ValueError('preserved fixture failure')
        return ['model-free']*(48 if phase=='overhead' else 144)
    monkeypatch.setattr(r,'collect_phase',collect)
    monkeypatch.setattr(r.analysis,'analyze_overhead',lambda rows,selected:dict(passed=passed))
    def close(child,*args):events.append(('close',child.mode));child._closed=True
    monkeypatch.setattr(t,'close_and_audit',close)
    monkeypatch.setattr(r,'final_checks',lambda *a:dict(fixture=True))
    assert r.run(expected)==(1 if fail_phase else 0)
    collects=[x for x in events if x[0]=='collect']
    assert collects==[('collect','overhead')]+([('collect','diagnostic')] if passed and fail_phase!='overhead' else [])
    assert len([x for x in events if x[0]=='close'])==2
    completion=c.read(c.OUT/'completion.json')
    assert completion['automatic_retries']==0 and completion['new_games']==0
    if fail_phase:assert 'preserved fixture failure' in c.read(c.OUT/'failure.json')['error']
    else:assert events.index(('close','on'))<events.index(('begin','final'))


def test_real_panel_lifecycle_counts_and_no_model_import_in_child():
    selected=c.read(c.OUT/'panel.json');all_specs=[]
    for phase,n in [('overhead',48),('diagnostic',144)]:
        specs=panel.request_matrix(selected,phase);assert len(specs)==n;all_specs.extend(specs)
    for mode in c.OBSERVER_MODES:
        assert sum(x['mode']==mode for x in all_specs)==96
        assert design.lifecycle_inventory()['observer_'+mode]['ordinary_first_request_id']=='request-00000003'
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    code='import sys; from tools.splendorust_search256_efficiency_20261031 import runner,watchdog,independent_review,worker; assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd=c.ROOT,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
