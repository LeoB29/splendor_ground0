"""Model-free fixed-core admission, request bindings and native thread masks."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools.splendorust_pool6_core_compare_20261027_02 import placement as p


def frozen():
    path = Path(__file__).resolve().parents[1]/'outputs/splendorust_pool6_hostdiag_20261027_03/freeze.json'
    topology = json.loads(path.read_text())['topology']
    return dict(topology=topology, placement=p.plan(topology))


def supervisor(freeze, stamp=100):
    plan = freeze['placement']
    return dict(schema=p.SCHEMA, role='runner', placement_mode='isolated', pid=101, created=10.,
        monotonic_ns=stamp, current_thread_id=111, process_mask=list(range(4,20)), system_mask=list(range(20)),
        priority_class=p.NORMAL_PRIORITY, physical_core_masks=plan['physical_core_masks'], player_masks=plan['player_masks'],
        host_topology_sha256=plan['host_topology_sha256'],full_thread_inventory=True,
        threads=[dict(thread_id=111,state='observed',group=0,mask=list(range(4,20)),relative_priority=0)],
        short_lived_threads_constrained_by_process_mask=True)


def player(core, start=110, end=120):
    rows=[]
    for i,role in enumerate(('python','search','reference','console:python','console:search','console:reference')):
        pid=200+i; parent=101 if role=='python' else 200 if role in ('search','reference','console:python') else 201 if role=='console:search' else 202
        rows.append(dict(role=role,pid=pid,created=100.+i,parent_pid=parent,executable=f'C:/fixture/{role}.exe',
            process_mask=[core],priority_class=p.NORMAL_PRIORITY,
            threads=[dict(thread_id=1000+i,state='observed',group=0,mask=[core],relative_priority=0)]))
    return dict(schema=p.SCHEMA,logical_cpu=core,player_pid=200,player_created=100.,start_ns=start,
        monotonic_ns=end,full_thread_inventory=True,process_mask_constrains_all_threads=True,processes=rows)


def request_fixture(core):
    freeze=frozen(); before=player(core); after=player(core,310,320)
    value=dict(schema=p.SCHEMA,placement_mode='isolated',logical_cpu=core,request_id='request-00000003',player_pid=200,
        before=supervisor(freeze),after=supervisor(freeze,330),player_before=before,player_after=after,error=None,
        original_request_status='ok',player_closed_by_original_request=False,
        checks_outside_original_request_clock=True,sample_checks_inside_original_clock=True,
        original_clock_and_failure_receipt_preserved=True)
    kw=dict(logical_cpu=core,request_id=value['request_id'],player_pid=200,
        supervisor_identity=dict(pid=101,created=10.),player_identity=dict(pid=200,created=100.),
        expected_processes=before['processes'],original_clock=dict(status='ok',accepted=True),
        parent_profile=dict(pid=101,child_pid=200,request_id=value['request_id'],request_start_ns=200,capture_snapshot_ns=300))
    return freeze,value,kw


def test_two_distinct_equal_performance_cores_and_one_shared_supervisor_mask():
    f=frozen();plan=f['placement']
    assert plan['player_masks']=={'0':[0],'2':[2]}
    assert plan['physical_core_masks']=={'0':[0,1],'2':[2,3]}
    assert plan['role_masks']['isolated']==list(range(4,20))
    assert plan['fixed_player_affinity_for_entire_lifetime'] and not plan['live_affinity_switches']
    assert plan['placement_is_not_exclusive_reservation_against_external_processes']


@pytest.mark.parametrize('change',['efficiency','siblings','same_core','missing_cpu'])
def test_wrong_physical_comparison_topology_fails(change):
    topology=copy.deepcopy(frozen()['topology']);other=next(x for x in topology['cores'] if 2 in x['logical_processors'])
    if change=='efficiency':other['efficiency_class']=0
    if change=='siblings':other['logical_processors']=[2]
    if change=='same_core':other['core_index']=0
    if change=='missing_cpu':other['logical_processors']=[3]
    with pytest.raises(ValueError):p.plan(topology)


@pytest.mark.parametrize('core',[0,2])
def test_complete_fixed_player_and_supervisor_proofs(core):
    f,v,kw=request_fixture(core)
    assert p.verify_saved_request(v,f,'isolated',**kw)==v
    assert p.verify_saved_player_runtime(v['player_before'],f,core,player_identity=kw['player_identity'])==v['player_before']


@pytest.mark.parametrize('change',['wrong_core','mask','thread_mask','priority','thread_priority','thread_group',
                                 'no_observed_thread','duplicate_thread','pid','created','parent','exe','clock'])
def test_corrupt_or_foreign_player_proof_fails(change):
    f,v,kw=request_fixture(2);proof=v['player_after'];process=proof['processes'][1]
    if change=='wrong_core':proof['logical_cpu']=0
    if change=='mask':process['process_mask']=[0]
    if change=='thread_mask':process['threads'][0]['mask']=[0]
    if change=='priority':process['priority_class']=128
    if change=='thread_priority':process['threads'][0]['relative_priority']=1
    if change=='thread_group':process['threads'][0]['group']=1
    if change=='no_observed_thread':process['threads']=[dict(thread_id=1001,state='exited_before_query')]
    if change=='duplicate_thread':process['threads'][0]['thread_id']=1000
    if change=='pid':process['pid']=9999
    if change=='created':process['created']+=1
    if change=='parent':process['parent_pid']=9999
    if change=='exe':process['executable']='C:/other.exe'
    if change=='clock':proof['start_ns']=299
    with pytest.raises(ValueError):p.verify_saved_request(v,f,'isolated',**kw)


def test_success_never_accepted_without_player_aftermath():
    f,v,kw=request_fixture(0);v['player_after']=None
    with pytest.raises(ValueError,match='Missing player aftermath'):p.verify_saved_request(v,f,'isolated',**kw)


@pytest.mark.parametrize('bad',[False,0.0])
def test_numeric_equal_but_wrong_core_type_cannot_pass(bad):
    f,v,kw=request_fixture(0);v['logical_cpu']=bad
    with pytest.raises(ValueError):p.verify_saved_request(v,f,'isolated',**kw)
    v['logical_cpu']=0;v['player_before']['logical_cpu']=bad
    with pytest.raises(ValueError):p.verify_saved_request(v,f,'isolated',**kw)


def test_original_failed_request_can_bind_already_closed_owned_tree():
    f,v,kw=request_fixture(2);v.update(player_after=None,player_closed_by_original_request=True,original_request_status='timeout')
    kw['original_clock']=dict(status='timeout',accepted=False)
    kw['cleanup_receipt']=dict(status='passed',errors=[],alive=[],confirmed=[dict(pid=x['pid'],created=x['created'],state='exit_wait_confirmed',exit_status=1) for x in v['player_before']['processes']])
    assert p.verify_saved_request(v,f,'isolated',**kw)==v
    # Placement verification preserves the failure; it cannot manufacture acceptance.
    assert kw['original_clock']==dict(status='timeout',accepted=False)
    kw['cleanup_receipt']['confirmed'].pop()
    with pytest.raises(ValueError,match='inventory incomplete'):p.verify_saved_request(v,f,'isolated',**kw)


@pytest.mark.parametrize('change',['status','accepted','unconfirmed','error','still_alive'])
def test_failed_closure_cannot_bypass_live_player_verification(change):
    f,v,kw=request_fixture(0);v.update(player_after=None,player_closed_by_original_request=True,original_request_status='timeout')
    kw['original_clock']=dict(status='timeout',accepted=False)
    kw['cleanup_receipt']=dict(status='passed',errors=[],alive=[],confirmed=[dict(pid=x['pid'],created=x['created'],state='absent') for x in v['player_before']['processes']])
    if change=='status':v['original_request_status']='ok'
    if change=='accepted':kw['original_clock']['accepted']=True
    if change=='unconfirmed':kw['cleanup_receipt']['confirmed'][0]['state']='unconfirmed'
    if change=='error':kw['cleanup_receipt']['errors']=['cannot close']
    if change=='still_alive':kw['cleanup_receipt']['alive']=[200]
    with pytest.raises(ValueError):p.verify_saved_request(v,f,'isolated',**kw)


@pytest.mark.parametrize('core',[0,2])
def test_immutable_admission_binds_initial_affinity_and_all_owned_processes(tmp_path,core):
    f=frozen();pl=player(core);initial=dict(topology=f['topology'],affinity=[core],identity=dict(pid=200,created=100.),telemetry=dict(process_tree=pl['processes']))
    proof=dict(schema=p.SCHEMA,placement_mode='isolated',logical_cpu=core,player_pid=200,player_created=100.,
        player_mask=[core],player_priority_class=p.NORMAL_PRIORITY,supervisor=supervisor(f),player=pl)
    (tmp_path/'initialization.json').write_text(json.dumps(initial));(tmp_path/'placement_admission.json').write_text(json.dumps(proof))
    assert p.verify_worker_placement(tmp_path,f,'isolated',logical_cpu=core)==proof
    with pytest.raises(ValueError):p.verify_worker_placement(tmp_path,f,'isolated',logical_cpu=2-core)


def test_live_player_checker_uses_original_sampler_then_fresh_identity_and_thread_checks(monkeypatch):
    f=frozen();pl=player(2);identities=p._process_identity_rows(pl['processes']);calls=[]
    class Proc:
        def __init__(self,pid):self.pid=pid;self.row=next(x for x in identities if x['pid']==pid)
        def create_time(self):return self.row['created']
        def ppid(self):return self.row['parent_pid']
        def exe(self):return self.row['executable']
        def cpu_affinity(self):return [2]
        def nice(self):return p.NORMAL_PRIORITY
    # Real paths are normalized exactly as the original sampler does.
    for row in pl['processes']:row['executable']=str(Path(row['executable']).resolve())
    identities=p._process_identity_rows(pl['processes'])
    def sample():calls.append('original_sample');return dict(process_tree=pl['processes'])
    child=SimpleNamespace(affinity=[2],identity=dict(pid=200,created=100.),_sample=sample,_failure=None,_closed=False)
    monkeypatch.setattr(p.psutil,'Process',Proc)
    monkeypatch.setattr(p,'_thread_rows',lambda proc,mask,require_current: [dict(thread_id=proc.pid+1000,state='observed',group=0,mask=mask,relative_priority=0)])
    value=p.verify_player_runtime(child,f,2)
    assert calls==['original_sample'] and value['logical_cpu']==2 and len(value['processes'])==6


@pytest.mark.skipif(os.name!='nt',reason='Windows native self-process thread mask fixture')
@pytest.mark.parametrize('core',[0,2])
def test_native_process_mask_constrains_owned_threads_on_each_core(core):
    process=psutil.Process();original=process.cpu_affinity()
    try:
        process.cpu_affinity([core])
        rows=p._thread_rows(process,[core],require_current=False)
        assert rows and all(row['mask']==[core] for row in rows if row['state']=='observed')
    finally:
        process.cpu_affinity(original)
    assert process.cpu_affinity()==original


def test_configure_execution_cannot_change_supervisor_with_resident_workers(monkeypatch):
    f=frozen();marker=SimpleNamespace(_closed=False)
    monkeypatch.setattr(p,'_workers',[marker])
    with pytest.raises(ValueError,match='resident workers'):p.configure_execution(f,'isolated')


def test_old_two_core_supervisor_mask_rejected_in_new_fixed_comparison():
    f=frozen();value=supervisor(f);value['process_mask']=list(range(2,20))
    with pytest.raises(ValueError,match='mask/priority/topology'):p.verify_saved_runtime(value,f,'isolated')
