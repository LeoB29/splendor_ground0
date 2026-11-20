"""Model-free fixed-arm orchestration, unchanged clocks and exact science."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time
import pytest
from tools.splendorust_pool6_core_compare_20261027_02 import runner as r,common as c,design,preparation,audit,qualification
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def row(spec,status='ok',elapsed=.4):
    success=status=='ok'
    clock=dict(status=status,clock='perf_counter_ns',request_id='request-00000003',deadline_seconds=1.,
        elapsed_seconds=elapsed if success else 1.01,receipt_elapsed_seconds=elapsed if success else None,
        received_before_deadline=success,accepted=success,result={} if success else None)
    return dict(spec=deepcopy(spec),route=spec['route'],logical_cpu=spec['logical_cpu'],arm_id=spec['arm_id'],
        observer=spec['observer'],clock=clock,completed=success,action={} if success else None,trace={} if success else None,
        host_observation=dict(status='observed' if spec['observer_enabled'] else 'disabled',errors=[]))


def test_fixed_inventory_and_original_transport():
    r.validate_freeze_contract(preparation.contract())
    assert c.MAX_WORKERS==4 and c.MAX_GRAPH_BUILDS==4 and c.JOB_SECONDS==7200
    assert r.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert r.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert r.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert r.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader
    assert len(set(r.ARM_IDS))==4


@pytest.mark.parametrize('mono,wall',[(7301.,1001.),(101.,8201.)])
def test_shared_utc_or_monotonic_deadline_cannot_reset(mono,wall):
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=mono),patch.object(r.time,'time',return_value=wall):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def test_fifth_tree_and_wrong_name_are_rejected_before_process_start():
    with pytest.raises(ValueError,match='four resident'):
        r.make_worker('qualification_frozen_cpu0',SimpleNamespace(resident={i:i for i in range(4)}),{},route='frozen',logical_cpu=0)
    with pytest.raises(ValueError,match='arm-to-worker'):
        r.make_worker('qualification_frozen_cpu0',None,{},route='frozen',logical_cpu=2)


def test_four_actual_entrypoints_have_separate_immutable_files(tmp_path):
    calls=[];workers={};env=SimpleNamespace()
    def make(name,*_,route,logical_cpu):
        calls.append((name,route,logical_cpu));return SimpleNamespace(arm_id=r.arm_id(route,logical_cpu))
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(r,'make_worker',side_effect=make),\
        patch('tools.splendorust_pool6_core_compare_20261027_02.adapters.actual_entrypoint',side_effect=lambda child,*a,**k:{'arm':child.arm_id}):
        result=r.qualify(env,{},workers)
    assert set(workers)==set(r.ARM_IDS) and len(calls)==4
    assert result['complete_scientific_decisions']==208 and result['graph_cases']==44
    assert {p.stem for p in (tmp_path/'entrypoint_checks').glob('*.json')}==set(r.ARM_IDS)


def test_four_resident_stage_counters_are_bound_per_arm(tmp_path):
    children={assigned:SimpleNamespace(route=route,logical_cpu=core,_lock=SimpleNamespace(locked=lambda:False),
        _counter=2,_closed=False,absolute_deadline=900.,output_dir=tmp_path/('qualification_'+assigned),identity={'pid':i,'created':1.})
        for i,(route,core) in enumerate(r.ARM_SPECS,1) for assigned in [r.arm_id(route,core)]}
    env=SimpleNamespace(stage='diagnostic',resident={i:i for i in range(4)},effective_deadline=lambda:1700.,deadline=1700.,end=7200.,start=0.)
    values=[]
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path):receipt=r.bind_collection_deadline(env,children,values)
    assert [x['arm_id'] for x in receipt['bindings']]==list(r.ARM_IDS)
    assert all(x['request_counter']==2 for x in receipt['bindings'])
    assert all(x.absolute_deadline==1700. for x in children.values()) and not receipt['request_clocks_reset']


@pytest.mark.parametrize('kind',['counter','active','too_many'])
def test_no_stage_rebinding_with_unaccounted_request_or_worker(tmp_path,kind):
    children={assigned:SimpleNamespace(route=route,logical_cpu=core,_lock=SimpleNamespace(locked=lambda:kind=='active'),
        _counter=3 if kind=='counter' else 2,_closed=False,absolute_deadline=900.,output_dir=tmp_path/assigned,identity={})
        for route,core in r.ARM_SPECS for assigned in [r.arm_id(route,core)]}
    env=SimpleNamespace(stage='diagnostic',resident={i:i for i in range(5 if kind=='too_many' else 4)},effective_deadline=lambda:1700.,deadline=1700.,end=7200.,start=0.)
    with pytest.raises(ValueError):r.bind_collection_deadline(env,children,[])


def test_lean_diagnostic_has_no_overhead_gate_or_extra_stage(tmp_path):
    matrix=design.build_matrix();env=SimpleNamespace(stage='diagnostic')
    with patch.object(r,'collect') as collect:
        r.collect_stages(env,{},matrix,{},[],{})
    collect.assert_called_once()
    assert not hasattr(r,'overhead_gate') and set(c.STAGES)=={'qualification','diagnostic','final'}
    assert not (tmp_path/'overhead_gate.json').exists()


@pytest.mark.parametrize('change',['enabled','old_stage','missing'])
def test_controller_rejects_enabled_or_partial_prospective_matrix_before_dispatch(change):
    matrix=design.build_matrix()
    if change=='enabled':matrix['requests'][0]['observer_enabled']=True
    elif change=='old_stage':matrix['requests'][0]['stage']='overhead'
    else:matrix['requests'].pop()
    with patch.object(r,'collect') as collect,pytest.raises(ValueError):
        r.collect_stages(SimpleNamespace(stage='diagnostic'),{},matrix,{},[],{})
    collect.assert_not_called()


def test_ordinary_rejects_on_before_resource_admission_or_clock():
    spec=deepcopy(design.build_matrix()['requests'][0]);spec['observer_enabled']=True
    with pytest.raises(ValueError,match='disabled extra sampler'):
        r.ordinary(None,spec,spec['observer'],None,{})


def test_first_actual_failure_stops_all_arms_after_preserving_clock(tmp_path):
    matrix=design.build_matrix();values=[];calls=[]
    env=SimpleNamespace(stage='diagnostic',guard=lambda:None)
    workers={assigned:SimpleNamespace(arm_id=assigned) for assigned in r.ARM_IDS}
    def ordinary(child,spec,*_):
        assert child.arm_id==spec['arm_id'];calls.append(spec['sequence'])
        return row(spec,'timeout' if spec['sequence']==3 else 'ok')
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(r,'ordinary',side_effect=ordinary),\
        patch.object(r,'host_targets',return_value=[]),patch.object(r.ledger,'transition',return_value={}),\
        patch.object(r,'verify_saved_record'),patch.object(r,'verify_host_row'),patch.object(audit,'validate_record'),\
        patch.object(qualification,'validate_repeat_row'):
        r.collect(env,{},matrix,workers,values,{})
    assert calls==[0,1,2,3] and values[-1]['after'] is None and not values[-1]['completed']
    assert c.read(tmp_path/'collection_stop.json')['clock']==values[-1]['clock']
    assert not(tmp_path/'records/0004.json').exists()


def test_host_target_inventory_distinguishes_twelve_fixed_compute_identities(tmp_path):
    workers={};pid=100
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path):
        for assigned in r.ARM_IDS:
            folder=tmp_path/assigned;workers[assigned]=SimpleNamespace(output_dir=folder)
            tree=[]
            for role in ('python','search','reference'):
                pid+=1;tree.append(dict(role=role,pid=pid,created=1.))
            c.save(folder/'initialization.json',dict(telemetry={'process_tree':tree}))
        targets=r.host_targets(workers)
    assert len(targets)==12 and {x['role'] for x in targets}=={a+':'+role for a in r.ARM_IDS for role in ('python','search','reference')}


def test_science_repeat_cache_compares_across_cores_instead_of_rebaselining():
    spec=deepcopy(design.build_matrix()['requests'][0]);spec.pop('expected_record',None);spec['expected_first_observation']=True
    first=row(spec);cache={}
    assert not qualification.validate_repeat_row(first,spec,cache)['first_repeat_comparison']
    second=deepcopy(spec);second.update(logical_cpu=2,arm_id=r.arm_id(spec['route'],2))
    value=row(second)
    assert qualification.validate_repeat_row(value,second,cache)['first_repeat_comparison']
    value['clock']['result']={'changed':'bits'}
    with pytest.raises(ValueError,match='repeated result'):qualification.validate_repeat_row(value,second,cache)
    assert len(cache)==1


def test_saved_actual_science_unchanged_with_explicit_new_core_annotation():
    spec=design.build_matrix()['requests'][0];saved=c.read(c.ROOT/spec['expected_record']['path'])
    value=deepcopy(saved);value.update(spec=deepcopy(spec),logical_cpu=spec['logical_cpu'],arm_id=spec['arm_id'])
    old=c.read(c.ROOT/'outputs/splendorust_pool6_hostdiag_20261027_03/freeze.json')
    assert audit.validate_record(value,spec,old)['transition_checked']
    value['logical_cpu']=2 if spec['logical_cpu']==0 else 0
    with pytest.raises(ValueError,match='arm'):audit.validate_record(value,spec,old)


def test_qualification_inventory_proves_four_suites_and_all_core_requests(tmp_path):
    panel=design.build_panel();matrix=design.build_matrix(panel)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path):
        c.save(tmp_path/'saved_panel.json',panel);c.save(tmp_path/'request_matrix.json',matrix)
        plan=qualification.build_plan()
    assert (plan['complete_decisions'],plan['graph_cases'],plan['ordinary_requests'],plan['maximum_parent_requests'])==(208,44,192,200)


@pytest.mark.parametrize('failed',[False,True])
def test_request_wrapper_preserves_original_receipt_and_clean_failed_closure(tmp_path,failed):
    clock=dict(status='timeout' if failed else 'ok',accepted=not failed,request_id='request-00000003')
    child=SimpleNamespace(_placement_freeze={},placement_mode='isolated',logical_cpu=2,process=SimpleNamespace(pid=22),
        output_dir=tmp_path,_closed=False)
    def request(*args,**kwargs):child._closed=failed;return clock
    child.request=request
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),\
        patch.object(r.placement,'verify_runtime',return_value={'supervisor':True}),\
        patch.object(r.placement,'verify_player_runtime',return_value={'player':True}) as proof:
        result=r.request(child,{},1.)
    assert result is clock and result['status']==clock['status']
    assert proof.call_count==(1 if failed else 2)
    evidence=c.read(tmp_path/'placement_requests/request-00000003.json')
    assert evidence['logical_cpu']==2 and evidence['player_closed_by_original_request'] is failed
    assert (evidence['player_after'] is None) is failed


def test_failed_player_aftercheck_keeps_completed_original_result(tmp_path):
    clock=dict(status='ok',accepted=True,request_id='request-00000003',result={'retained':True})
    child=SimpleNamespace(_placement_freeze={},placement_mode='isolated',logical_cpu=0,process=SimpleNamespace(pid=22),
        output_dir=tmp_path,_closed=False,request=lambda *a,**k:clock)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(r.placement,'verify_runtime',return_value={}),\
        patch.object(r.placement,'verify_player_runtime',side_effect=[{},ValueError('wrong mask')]):
        assert r.request(child,{},1.) is clock
    assert 'wrong mask' in child.last_placement_error and clock['accepted'] is True
    assert c.read(tmp_path/'placement_requests/request-00000003.json')['player_after'] is None


def test_all_192_exact_saved_outputs_and_cross_core_repeats_are_counted():
    matrix=design.build_matrix();records=[];source_cache={}
    for spec in matrix['requests']:
        item=spec['expected_record'];key=item['path']
        if key not in source_cache:source_cache[key]=c.read(c.ROOT/key)
        value=row(spec);value['clock']['result']=deepcopy(source_cache[key]['clock']['result'])
        records.append(value)
    result=qualification.validate_repeat_rows(records,{'repeat_requests':matrix['requests']})
    assert result['complete'] and result['successful']==192
    assert result['exact_historical_comparisons']==192 and result['exact_first_repeat_comparisons']==184
    assert len(result['cells'])==8 and all(x['successful']==24 for x in result['cells'])


@pytest.mark.parametrize('kind,status,exit_code',[
    ('complete','completed_verified',0),('timeout','verified_partial_diagnostic',0),
    ('illegal','incomplete',1),('measurement','incomplete_measurement_failure_verified',1),
    ('measurement_and_player','incomplete',1)])
def test_lean_completion_preserves_original_negative_and_forensic_failure_dispositions(tmp_path,kind,status,exit_code):
    class FakeEnvelope:
        def __init__(self):
            self.stage=None;self.resident={};self.start=time.monotonic();self.wall_start=time.time()
        def begin(self,stage):self.stage=stage
        def finish(self):pass
        def enable_evidence_accounting(self):pass
        def admit(self):pass
    env=FakeEnvelope();coverage=dict(measurement_failure_verified=kind.startswith('measurement'),
        failed_requests=int(kind in ('timeout','illegal','measurement_and_player')),complete=kind=='complete')
    def collect(*args):
        if kind in ('timeout','illegal'):
            c.save(tmp_path/'collection_stop.json',dict(diagnostic_deadline_failure=kind=='timeout'))
        elif kind.startswith('measurement'):raise r.MeasurementFailure('original disabled lifecycle failure')
    with ExitStack() as stack:
        stack.enter_context(patch.object(c,'ROOT',tmp_path));stack.enter_context(patch.object(c,'OUT',tmp_path))
        c.save(tmp_path/'freeze.json',{});sha=c.file_hash(tmp_path/'freeze.json')
        c.save(tmp_path/'watchdog_claim.json',dict(freeze_sha256=sha))
        stack.enter_context(patch.object(r,'Envelope',return_value=env))
        stack.enter_context(patch.object(r,'verify_runtime_sources',return_value=[]))
        for name in ('verify_runtime','switch_placement','bind_collection_deadline'):stack.enter_context(patch.object(r,name))
        stack.enter_context(patch.object(r,'load_inputs',return_value=({},{},{})))
        stack.enter_context(patch.object(r,'qualify',return_value={'status':'passed'}))
        stack.enter_context(patch.object(r,'start_host_observer',return_value=None))
        stack.enter_context(patch.object(r,'collect_stages',side_effect=collect))
        final=stack.enter_context(patch.object(r,'final_checks',return_value=coverage))
        assert r.run(sha)==exit_code
        completion=c.read(tmp_path/'completion.json')
        assert completion['status']==status and completion['coverage']==coverage and final.call_count==1
        assert completion['automatic_retries']==0 and completion['stage']=='final'
        assert not (tmp_path/'overhead_gate.json').exists()
        if kind.startswith('measurement'):assert 'original disabled lifecycle failure' in completion['failure']
