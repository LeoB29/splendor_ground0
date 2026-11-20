"""No model execution: fixed repeats, trusted clocks, failure and budget proofs."""
from copy import deepcopy
from contextlib import ExitStack
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_pool6_hostdiag_20261027_02 import runner as r, audit as a, common as c, guardian, preparation, design, qualification
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def receipt(status='ok',elapsed=.85):
    ok=status=='ok'
    return dict(status=status,clock='perf_counter_ns',request_id='request-00000003',deadline_seconds=1.,
        elapsed_seconds=elapsed if ok else 1.01,receipt_elapsed_seconds=elapsed if ok else None,
        received_before_deadline=ok,accepted=ok,result={} if ok else None)


def row(spec,status='ok'):
    return dict(spec=spec,observer=spec['observer'],route=spec['route'],clock=receipt(status),completed=status=='ok',
        action={'action_type':'PASS'} if status=='ok' else None,trace={} if status=='ok' else None)


def test_contract_is_exact_216_conditional_no_game_no_promotion():
    r.validate_freeze_contract(preparation.contract())
    assert (c.JOB_SECONDS,c.COLLECTION_CUTOFF_SECONDS,c.MAX_WORKERS,c.RAM_WAIT_SECONDS)==(7200,3600,2,600)
    assert c.STAGES==dict(qualification=900,overhead=900,diagnostic=900,final=3600)
    assert c.MAX_EVIDENCE_BYTES==2*c.GiB and c.FINAL_RESERVE_BYTES==512*c.MiB
    bad=preparation.contract();bad['total_requests']=217
    with pytest.raises(ValueError):r.validate_freeze_contract(bad)


def test_original_timed_transport_is_still_inherited():
    assert r.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert r.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert r.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert r.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader


def test_only_two_resident_players():
    with pytest.raises(ValueError,match='two resident'):
        r.make_worker('third',SimpleNamespace(resident={1:1,2:2}),{},route='research')


@pytest.mark.parametrize('mono,utc',[(7301.,1001.),(101.,8201.)])
def test_either_utc_or_monotonic_clock_expires_without_reset(mono,utc):
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=mono),patch.object(r.time,'time',return_value=utc):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def test_final_reserve_not_accessible_to_collection():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=3600.),patch.object(r.time,'time',return_value=4500.),patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('diagnostic');assert env.deadline==3700.
        env.begin('final');assert env.deadline==7200.


def test_resource_stage_binding_changes_only_job_deadline(tmp_path):
    children={route:SimpleNamespace(_lock=SimpleNamespace(locked=lambda:False),_counter=2,_closed=False,
        absolute_deadline=900.,output_dir=tmp_path/('qualification_'+route),identity={'pid':i,'created':1.})
        for i,route in enumerate(('frozen','research'),1)}
    env=SimpleNamespace(stage='diagnostic',resident={1:1,2:2},effective_deadline=lambda:1700.,
        deadline=1700.,end=3600.,start=0.)
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path):value=r.bind_collection_deadline(env,children,[])
    assert all(x.absolute_deadline==1700. and x._counter==2 for x in children.values())
    assert not value['request_clocks_reset'] and value['ordinary_deadline_seconds']==1. and value['internal_seconds']==.9
    assert [x['previous_absolute_deadline'] for x in value['bindings']]==[900.,900.]


@pytest.mark.parametrize('pending,deadline',[(True,1700.),(False,3700.)])
def test_resource_rebinding_refuses_active_request_or_collection_extension(tmp_path,pending,deadline):
    children={route:SimpleNamespace(_lock=SimpleNamespace(locked=lambda:pending),_counter=2,_closed=False,
        absolute_deadline=900.,output_dir=tmp_path/route,identity={'pid':1}) for route in ('frozen','research')}
    env=SimpleNamespace(stage='diagnostic',resident={1:1,2:2},effective_deadline=lambda:deadline,deadline=deadline,end=3600.,start=0.)
    with pytest.raises(ValueError):r.bind_collection_deadline(env,children,[])
    assert children['frozen'].absolute_deadline==900.


def test_late_sidecars_are_charged_after_failed_worker_closes(tmp_path):
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{}}
    folder=tmp_path/'worker';folder.mkdir();(folder/'cleanup.json').write_bytes(b'abc')
    env.register_worker(folder);env.close_worker(folder);assert env.evidence['workers'][folder]==3
    (folder/'parent.json').write_bytes(b'late-sidecar');env.close_worker(folder)
    assert env.evidence['workers'][folder]==15


def collect_fixture(tmp_path,*,failure=None,transition_error=False,exact_error=False):
    matrix=design.build_matrix();rows=[];env=SimpleNamespace(guard=lambda:None,stage='overhead')
    children={route:SimpleNamespace(route=route) for route in ('frozen','research')}
    calls=[]
    def ordinary(child,spec,*_):
        calls.append(spec['sequence']);return row(spec,'timeout' if spec['sequence']==failure else 'ok')
    transition=ValueError('ledger corruption') if transition_error else lambda *args: {'successor':'saved'}
    exact=ValueError('historical numeric mismatch') if exact_error else lambda *args,**kwargs: {'status':'passed'}
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'ordinary',side_effect=ordinary),\
        patch.object(r.ledger,'transition',side_effect=transition),patch.object(r,'verify_saved_record'),patch.object(r,'verify_host_row'),patch.object(r,'host_targets',return_value=[]),\
        patch.object(a,'validate_record'),patch.object(qualification,'validate_repeat_row',side_effect=exact):
        if transition_error or exact_error:
            with pytest.raises(ValueError):r.collect(env,{},matrix,children,rows,{})
        else:r.collect(env,{},matrix,children,rows,{})
    return rows,calls


def test_all_72_overhead_slow_successes_retained_without_700ms_gate(tmp_path):
    rows,calls=collect_fixture(tmp_path)
    assert calls==list(range(72)) and len(rows)==72 and all(x['clock']['receipt_elapsed_seconds']==.85 for x in rows)
    assert not(tmp_path/'collection_stop.json').exists()
    assert len(list((tmp_path/'records').glob('*.json')))==72


def test_first_real_timeout_saved_and_no_further_dispatch(tmp_path):
    rows,calls=collect_fixture(tmp_path,failure=3)
    assert calls==[0,1,2,3] and not rows[-1]['completed'] and rows[-1]['after'] is None
    saved=json.loads((tmp_path/'collection_stop.json').read_text())
    assert saved['clock']==rows[-1]['clock'] and saved['diagnostic_deadline_failure'] and saved['no_retry']
    assert not(tmp_path/'records/0004.json').exists()


@pytest.mark.parametrize('transition_error,exact_error',[(True,False),(False,True)])
def test_transition_or_exact_science_failure_preserves_first_original_row(tmp_path,transition_error,exact_error):
    rows,calls=collect_fixture(tmp_path,transition_error=transition_error,exact_error=exact_error)
    assert calls==[0] and len(rows)==1
    saved=json.loads((tmp_path/'records/0000.json').read_text())
    assert saved['clock']==rows[0]['clock'] and saved['clock']['status']=='ok'
    if transition_error:assert saved['after'] is None and 'transition_error' in saved
    assert not(tmp_path/'records/0001.json').exists()


def test_authentic_deadline_classification_does_not_change_clock():
    outer=receipt('timeout');original=deepcopy(outer)
    assert r.deadline_failure(outer,'frozen') and outer==original
    inner=receipt('worker_error');inner.update(detail='SAMPLING_INTERNAL_DEADLINE: original deadline',
        receipt_elapsed_seconds=.91,received_before_deadline=True,elapsed_seconds=.91)
    assert r.deadline_failure(inner,'research')
    wrong=deepcopy(inner);wrong['detail']='unexpected exception'
    assert not r.deadline_failure(wrong,'research')


def test_retained_rejected_response_remains_forensic_only():
    rejected=receipt('illegal');rejected.update(receipt_elapsed_seconds=.4,received_before_deadline=True,
        elapsed_seconds=.4,result={'original':'rejected'},detail='DECISION_QUALITY_AUDIT: malformed science')
    assert a.check_clock(rejected)==.4 and not rejected['accepted']
    assert not r.deadline_failure(rejected,'research')


def profile_fixture():
    from test_splendorust_boundary_timing_analyze import profile_row
    row=profile_row();parent=row['parent_profile'];child=row['child_profile'];clock=row['clock']
    payload=dict(player='search',observer=row['observer'])
    child.update(backend='research_graph',quota=64,payload_sha256=c.digest(payload))
    clock.update(clock='perf_counter_ns',elapsed_seconds=.45)
    return parent,child,clock,payload


def test_minimal_observer_exact_one_second_sidecar():
    parent,child,clock,payload=profile_fixture()
    assert r.validate_profiles(parent,child,clock,payload,'research',worker_pid=42)['status']=='passed'


@pytest.mark.parametrize('field,value',[('quota',32),('payload_sha256','f'*64),('result_sha256','f'*64),
                                      ('backend','wrong'),('graph_identity_sha256','f'*64)])
def test_corrupt_scientific_sidecar_is_never_accepted(field,value):
    parent,child,clock,payload=profile_fixture();child[field]=value
    with pytest.raises(ValueError):r.validate_profiles(parent,child,clock,payload,'research',worker_pid=42)


def test_raw_packet_binding_does_not_trust_reserialized_content():
    packets=[r._packet_evidence(b'{"type":"error"}\n')]
    parent=dict(response_raw_sha256='0'*64,response_raw_bytes=17,response_capture_overflow=False)
    with pytest.raises(ValueError):r._bind_parent_wire(parent,packets,completed=True)


def test_original_supervisor_placement_is_rejected():
    with pytest.raises(ValueError):guardian.require_isolated_request({'placement_mode':'original'})


def test_actual_same_player_sources_full_science_without_model():
    panel=design.build_panel();matrix=design.build_matrix(panel)
    freeze=c.read(c.QUAL_OUT/'freeze.json');checked=0
    for spec in matrix['requests'][:18]:
        result=qualification.historical_result(spec)['result']
        if result is None:continue
        clock=receipt();clock['result']=result
        saved=dict(spec=spec,route=spec['route'],observer=spec['observer'],before=spec['before_snapshot'],
            after=r.ledger.transition(spec['before_snapshot'],result['action']),clock=clock,completed=True,
            action=result['action'],trace=result['diagnostics'])
        assert a.validate_record(saved,spec,freeze)['transition_checked'];checked+=1
        bad=deepcopy(saved);bad['after']=deepcopy(bad['before'])
        with pytest.raises(ValueError):a.validate_record(bad,bad['spec'],freeze)
    assert checked==9


def test_prefix_cannot_skip_replace_or_continue_failed_request():
    matrix=design.build_matrix();rows=[row(s) for s in matrix['requests'][:2]]
    with patch.object(a,'validate_record',return_value={'transition_checked':True}):
        bad=deepcopy(rows);bad.reverse()
        with pytest.raises(ValueError):a.audit_prefix(bad,matrix,{})
        rows[0]['completed']=False
        with pytest.raises(ValueError):a.audit_prefix(rows,matrix,{})


def overhead_rows(off=.4,on=.409):
    rows=[]
    for spec in design.build_matrix()['requests'][:72]:
        value=row(spec);value['clock']['receipt_elapsed_seconds']=on if spec['observer_enabled'] else off
        value['host_observation']=dict(status='observed' if spec['observer_enabled'] else 'disabled',errors=[])
        rows.append(value)
    return rows


def test_overhead_gate_qualifies_exact_abba_and_no_margin_gate():
    rows=overhead_rows(.8,.809);checked=r.overhead_gate(rows)
    assert checked['passed'] and checked['complete'] and checked['completed']==72
    assert all(v['off_count']==v['on_count']==18 for v in checked['per_player'].values())
    assert not checked['trimming'] and checked['margin_700ms']=='descriptive_only'


@pytest.mark.parametrize('kind',['ratio','mean','p95','deadline','incomplete','failure'])
def test_each_prospective_overhead_limit_fails_closed(kind):
    values=overhead_rows(.1,.106) if kind=='ratio' else overhead_rows(.6,.611) if kind=='mean' else overhead_rows(.4,.4)
    if kind=='p95':next(x for x in values if x['route']=='research' and x['spec']['observer_enabled'])['clock']['receipt_elapsed_seconds']=.43
    if kind=='deadline':values[-1]['clock']['receipt_elapsed_seconds']=1.000001
    if kind=='incomplete':values=values[:-1]
    if kind=='failure':values[-1]['clock']=receipt('timeout');values[-1]['completed']=False
    assert not r.overhead_gate(values)['passed']


@pytest.mark.parametrize('passes',[True,False])
def test_diagnostic_runs_only_after_overhead_pass(tmp_path,passes):
    rows=[];called=[];prepared=overhead_rows(.4,.409 if passes else .5);env=SimpleNamespace(stage='overhead',finish=lambda:None)
    def begin(stage):env.stage=stage
    env.begin=begin
    def collect(*args):
        called.append(env.stage)
        if env.stage=='overhead':rows.extend(prepared)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(r,'collect',side_effect=collect),\
        patch.object(r,'bind_collection_deadline'):
        result=r.collect_stages(env,{},None,{},rows,{})
    assert result['passed'] is passes
    assert called==(['overhead','diagnostic'] if passes else ['overhead'])
    assert json.loads((tmp_path/'overhead_gate.json').read_text())==result


def test_missing_counterpart_target_is_explicit_and_repeat_exact():
    spec=next(x for x in design.build_matrix()['requests'] if x.get('expected_first_observation'))
    cache={};value=row(spec)
    proof=qualification.validate_repeat_row(value,spec,cache)
    assert not proof['historical_comparison'] and not proof['first_repeat_comparison']
    assert qualification.validate_repeat_row(value,spec,cache)['first_repeat_comparison']
    changed=deepcopy(value);changed['clock']['result']={'changed':'science'}
    with pytest.raises(ValueError,match='repeated result mismatch'):qualification.validate_repeat_row(changed,spec,cache)


def test_no_historical_target_is_invented_or_mixed():
    spec=next(x for x in design.build_matrix()['requests'] if x.get('expected_first_observation'))
    bad=deepcopy(spec);bad.pop('expected_first_observation')
    with pytest.raises(ValueError):qualification.historical_result(bad)
    bad=deepcopy(spec);bad['expected_record']={'path':'invented'}
    with pytest.raises(ValueError):qualification.historical_result(bad)


def test_actual_worker_science_methods_are_byte_equivalent_ast():
    import ast
    from tools.splendorust_pool6_hostdiag_20261027_02 import worker,adapters
    from tools.splendorust_pool6_repeat_timing_20261027 import worker as oldworker,adapters as oldadapters
    def functions(module):
        source=Path(module.__file__).read_text();tree=ast.parse(source)
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    actual,original=functions(worker),functions(oldworker)
    for name in actual.keys()-{'initialize'}:assert actual[name]==original[name]
    assert adapters.LiveOpponent is oldadapters.LiveOpponent
    assert adapters.initialize_adapters is oldadapters.initialize_adapters


def test_four_gib_headroom_required_before_requests_without_changing_hard_clock():
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.admitted=True
    with patch.object(env,'guard'),patch.object(r.psutil,'virtual_memory',return_value=SimpleNamespace(available=4*c.GiB)),\
        patch.object(r.time,'monotonic',return_value=10.),patch.object(r.time,'perf_counter_ns',return_value=42):
        proof=env.admit()
    assert proof['current_available_bytes']==4*c.GiB and proof['required_headroom_bytes']==4*c.GiB
    assert proof['monotonic_ns']==42 and proof['wait_seconds']==0 and not proof['stable_recovery_required']


def test_observer_start_failure_does_not_dispatch_player(tmp_path):
    spec=dict(sequence=0,stage='overhead',observer_enabled=True,route='frozen',quota=64)
    host=SimpleNamespace(start_window=lambda *a,**k:{'status':'observer_failed'},finish_window=lambda *a,**k:{'status':'observer_failed'})
    env=SimpleNamespace(admit=lambda:{},remaining=lambda:100.,timeout=lambda x:x,host_observer=host)
    child=SimpleNamespace(_counter=2,route='frozen',output_dir=tmp_path/'worker',identity={'pid':1,'created':1.})
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),\
        patch.object(r.rules,'state_from_observation',return_value=None),patch.object(r.rules,'legal_actions',return_value=[]),\
        patch.object(r,'request') as dispatch:
        with pytest.raises(ValueError,match='no player request'):r.ordinary(child,spec,{},env,{})
    dispatch.assert_not_called();assert (tmp_path/'host_observer_start_failures/0000.json').exists()


def test_finished_player_receipt_survives_observer_finish_error(tmp_path):
    spec=dict(sequence=0,stage='overhead',observer_enabled=True,route='frozen',quota=64)
    def finish(*a,**k):raise ValueError('observer evidence failed')
    host=SimpleNamespace(start_window=lambda *a,**k:{'status':'ready'},finish_window=finish)
    env=SimpleNamespace(admit=lambda:{},remaining=lambda:100.,timeout=lambda x:x,host_observer=host)
    child=SimpleNamespace(_counter=2,route='frozen',output_dir=tmp_path/'worker',identity={'pid':1,'created':1.})
    clock=receipt();clock['result']={'action':{},'diagnostics':{}}
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),\
        patch.object(r.rules,'state_from_observation',return_value=None),patch.object(r.rules,'legal_actions',return_value=[]),\
        patch.object(r,'request',return_value=clock),patch.object(r,'collect_profiles',return_value={}):
        saved=r.ordinary(child,spec,{},env,{})
    assert saved['clock']==clock and saved['completed'] and 'observer evidence failed' in saved['host_observer_error']
    with pytest.raises(ValueError,match='Host observer evidence'):r.verify_host_row(saved,[])


def test_exact_original_one_second_boundary_is_not_relabelled_timeout():
    values=overhead_rows(1.,1.);assert r.overhead_gate(values)['passed']
    assert all(x['clock']['status']=='ok' and x['clock']['accepted'] for x in values)


def host_row_fixture(tmp_path,*,failed=False):
    """Real observer receipt schema and lifecycle, synthetic native counters only."""
    from tools.splendorust_pool6_hostdiag_20261027_02 import host_observer as h
    from test_splendorust_pool6_hostdiag_20261027_02_observer import Backend,TARGETS
    backend=Backend();backend.fail_context=2 if failed else None
    observer=h.HostObserver(TARGETS,_backend=backend)
    binding=dict(stage='overhead',sequence=0,route='frozen',request_id='request-00000003',
        active_player=dict(pid=TARGETS[0]['pid'],created=TARGETS[0]['created']))
    token=observer.start_window(0,True,binding=binding)
    start=time.perf_counter_ns();end=time.perf_counter_ns()
    value=observer.finish_window(token);closed=observer.close()
    spec=dict(sequence=0,stage='overhead',route='frozen',observer_enabled=True,case_id='fixture')
    clock=receipt(elapsed=(end-start)/1e9)
    result=dict(spec=spec,route='frozen',clock=clock,completed=True,
        parent_profile=dict(request_start_ns=start,receipt_arrival_ns=end),host_observation=value,
        worker_clock_file='preserved-original-clock.json',worker_folder='workers/qualification_frozen',
        ram_admission=dict(required_headroom_bytes=4*c.GiB,current_available_bytes=5*c.GiB,
            monotonic_ns=start-1,cumulative_wait_seconds=0.,stable_recovery_required=False))
    result['host_observation_file']=c.save(tmp_path/'host_observations/0000.json',value)
    c.save(tmp_path/'host_observer_identity.json',dict(runner=dict(pid=1234,created=2000.)))
    return result,TARGETS,closed


def write_measurement_stop(value):
    return c.save(c.OUT/'measurement_stop.json',dict(sequence=value['spec']['sequence'],stage=value['spec']['stage'],
        route=value['route'],clock=value['clock'],worker_clock_file=value['worker_clock_file'],
        host_observation=value['host_observation_file'],measurement_accepted=False,overhead_qualified=False,no_retry=True))


def test_real_new_observer_result_shape_is_not_duplicate_keyword(tmp_path):
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path):
        value,targets,_=host_row_fixture(tmp_path)
        checked=r.verify_host_row(value,targets)
        assert checked['accepted'] is True and checked['measurement_accepted'] is True
        assert r.verify_host_row(value,targets,forensic=True)==checked


def test_failed_observer_is_strict_live_failure_and_only_forensic_prefix(tmp_path):
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path):
        value,targets,_=host_row_fixture(tmp_path,failed=True);original=deepcopy(value['clock'])
        with pytest.raises(r.MeasurementFailure):r.verify_host_row(value,targets)
        checked=r.verify_host_row(value,targets,forensic=True)
        assert not checked['measurement_accepted'] and value['clock']==original and value['completed']
        with pytest.raises(ValueError,match='stop'):r.verify_measurement_prefix([value],[checked])
        write_measurement_stop(value)
        assert r.verify_measurement_prefix([value],[checked])[0]['original_player_clock_status']=='ok'
        with pytest.raises(ValueError,match='terminal'):r.verify_measurement_prefix([value,value],[checked,checked])
        with pytest.raises(ValueError,match='terminal'):
            r.verify_measurement_prefix([value,value],[checked,{'measurement_accepted':True}])
        with pytest.raises(ValueError,match='Unrepresented'):
            r.verify_measurement_prefix([value],[{'measurement_accepted':True}])


def test_invalid_window_never_qualifies_full_overhead():
    values=overhead_rows(.4,.4)
    values[-1]['host_observation']=dict(status='observer_failed',errors=[dict(type='OSError',detail='native capture failed')])
    gate=r.overhead_gate(values)
    assert not gate['passed'] and not gate['complete'] and not gate['measurements_valid'] and gate['completed']==72


def test_actual_saved_window28_preserves_valid_science_clock_and_missing_placement():
    from tools.splendorust_pool6_hostdiag_20261027_02 import host_observer as h
    old=c.ROOT/'outputs/splendorust_pool6_hostdiag_20261027_01'
    source=old/'records/0028.json';before=c.file_hash(source);value=c.read(source);saved=deepcopy(value)
    identity=c.read(old/'host_observer_identity.json');freeze=c.read(old/'freeze.json')
    with patch.object(c,'OUT',old),patch.object(h,'verify_forensic_receipt',side_effect=h.verify_legacy_forensic_receipt):
        checked=r.verify_host_row(value,identity['targets'],forensic=True)
        assert a.validate_record(value,value['spec'],freeze)['transition_checked']
    assert not checked['measurement_accepted'] and 'placement_end' in checked['missing_fields']
    assert value==saved and c.file_hash(source)==before
    assert value['clock']['accepted'] is True and value['clock']['receipt_elapsed_seconds']==.3743033
    assert value['host_observation']['placement_end'] is None


@pytest.mark.parametrize('kind',['binding','placement','ram','hash'])
def test_failed_observer_available_bindings_remain_strict(tmp_path,kind):
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path):
        value,targets,_=host_row_fixture(tmp_path,failed=True)
        if kind=='binding':value['host_observation']['binding']['request_id']='wrong-request'
        if kind=='placement':value['host_observation']['placement_end']['pid']+=1
        if kind=='ram':value['ram_admission']['current_available_bytes']=1
        if kind in ('binding','placement'):
            path=tmp_path/'host_observations/0000.json';path.write_text(c.canonical(value['host_observation']),encoding='utf8')
            value['host_observation_file']=c.artifact(path)
        if kind=='hash':value['host_observation_file']['sha256']='0'*64
        with pytest.raises(ValueError):r.verify_host_row(value,targets,forensic=True)


def test_terminal_measurement_failure_stops_before_next_model_request(tmp_path):
    matrix=design.build_matrix();spec=matrix['requests'][0];value=row(spec)
    value.update(worker_clock_file='clock.json',host_observation_file={'path':'window.json'})
    env=SimpleNamespace(stage='overhead',guard=lambda:None);rows=[]
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),\
        patch.object(r,'ordinary',return_value=value) as dispatch,patch.object(r,'host_targets',return_value=[]),\
        patch.object(r.ledger,'transition',return_value={}),patch.object(r,'verify_saved_record'),\
        patch.object(r,'verify_host_row',side_effect=r.MeasurementFailure('measurement failed')):
        with pytest.raises(r.MeasurementFailure):r.collect(env,{},matrix,{spec['route']:object()},rows,{})
    assert dispatch.call_count==1 and len(rows)==1 and (tmp_path/'records/0000.json').exists()
    assert c.read(tmp_path/'measurement_stop.json')['clock']==value['clock']


def test_forensic_finalization_seals_prefix_and_finishes_source_preservation(tmp_path):
    from tools.splendorust_pool6_hostdiag_20261027_02 import analysis
    with ExitStack() as stack:
        stack.enter_context(patch.object(c,'ROOT',tmp_path));stack.enter_context(patch.object(c,'OUT',tmp_path))
        value,targets,close=host_row_fixture(tmp_path,failed=True)
        identity=dict(runner=dict(pid=1234,created=2000.),targets=targets,runtime={'pinned':True},contract={'fixed':True})
        (tmp_path/'host_observer_identity.json').write_text(c.canonical(identity),encoding='utf8')
        c.save(tmp_path/'host_observer_close.json',close);c.save(tmp_path/'launch.json',identity['runner'])
        c.save(tmp_path/'records/0000.json',value);write_measurement_stop(value)
        qualified={'status':'passed'};c.save(tmp_path/'qualification_summary.json',qualified)
        for route in ('frozen','research'):
            name='qualification_'+route;folder=tmp_path/'workers'/name
            (folder/'inputs').mkdir(parents=True)
            if route=='frozen':c.save(folder/'inputs/request-00000003.json',{'saved':'payload'})
            suite={'request_id':'request-00000002'}
            c.save(folder/'request-00000002.json',suite);c.save(tmp_path/'entrypoint_checks'/f'{route}.json',suite)
            c.save(tmp_path/'closed_workers'/f'{name}.json',{'raw':{'requests':3 if route=='frozen' else 2}})
        matrix={'requests':[value['spec']]};panel={};freeze={'inputs':[],'host_observer_runtime':identity['runtime'],
            'host_observer_contract':identity['contract']}
        coverage=dict(complete=False,failed_requests=0,attempted_requests=1,completed_requests=1)
        env=SimpleNamespace(guard=lambda:None,peak_combined_rss=123)
        stack.enter_context(patch.object(design,'build_panel',return_value=panel))
        stack.enter_context(patch.object(design,'build_matrix',return_value=matrix))
        saved_check=stack.enter_context(patch.object(r,'verify_saved_record'))
        stack.enter_context(patch.object(a,'audit_prefix',return_value=coverage))
        stack.enter_context(patch.object(qualification,'validate_repeat_rows',return_value={'status':'passed'}))
        stack.enter_context(patch.object(qualification,'verify_entrypoint'))
        closed_check=stack.enter_context(patch.object(r,'verify_closed_worker',
            side_effect=lambda folder,*_:c.read(tmp_path/'closed_workers'/f'{folder.name}.json')))
        stack.enter_context(patch.object(r,'verify_stage_transitions',return_value=[]))
        stack.enter_context(patch.object(r,'host_targets',return_value=targets))
        stack.enter_context(patch.object(r,'verify_exchanges',return_value={'status':'passed'}))
        stack.enter_context(patch.object(analysis,'analyze',return_value={'complete':False,'measurement_accepted':False}))
        runtime=stack.enter_context(patch.object(r,'verify_runtime'))
        preservation=stack.enter_context(patch.object(r,'verify_prior_preservation',
            side_effect=lambda *_:c.save(tmp_path/'prior_preservation_after.json',{'original_incomplete_preserved':True})))
        result=r.final_checks(env,freeze,panel,matrix,{},qualified)
        assert result['measurement_failure_verified'] and not result['complete'] and result['failed_requests']==0
        assert result['overhead_qualified'] is False and not result['overhead_not_qualified']
        assert c.read(tmp_path/'host_observer_forensic_audit.json')['status']=='preserved_failed_prefix'
        assert c.read(tmp_path/'verification_summary.json')['all_raw_placement_logs_graph_science_sources_verified']
        seal=c.read(tmp_path/'completed_evidence_seal.json');c.verify(seal['artifacts'])
        assert {'records/0000.json','measurement_stop.json','host_observations/0000.json'}<=set(x['path'] for x in seal['artifacts'])
        assert saved_check.call_count==1 and closed_check.call_count==2 and runtime.call_count==preservation.call_count==1


@pytest.mark.parametrize('player_failed',[False,True])
def test_verified_measurement_failure_is_nonzero_and_cannot_hide_player_failure(tmp_path,player_failed):
    class FakeEnvelope:
        def __init__(self):
            self.stage=None;self.resident={};self.start=time.monotonic();self.wall_start=time.time()
        def begin(self,stage):self.stage=stage
        def finish(self):pass
        def enable_evidence_accounting(self):pass
        def admit(self):pass
    env=FakeEnvelope();coverage=dict(measurement_failure_verified=True,failed_requests=int(player_failed),
        complete=False,overhead_not_qualified=False)
    with ExitStack() as stack:
        stack.enter_context(patch.object(c,'ROOT',tmp_path));stack.enter_context(patch.object(c,'OUT',tmp_path))
        c.save(tmp_path/'freeze.json',{});sha=c.file_hash(tmp_path/'freeze.json')
        c.save(tmp_path/'watchdog_claim.json',dict(freeze_sha256=sha))
        stack.enter_context(patch.object(r,'Envelope',return_value=env))
        stack.enter_context(patch.object(r,'verify_runtime_sources',return_value=[]))
        stack.enter_context(patch.object(r,'verify_runtime'))
        stack.enter_context(patch.object(r,'switch_placement'))
        stack.enter_context(patch.object(r,'load_inputs',return_value=({},{},{})))
        stack.enter_context(patch.object(r,'qualify',return_value={'status':'passed'}))
        stack.enter_context(patch.object(r,'bind_collection_deadline'))
        stack.enter_context(patch.object(r,'start_host_observer',return_value=None))
        stack.enter_context(patch.object(r,'collect_stages',side_effect=r.MeasurementFailure('original measurement failure')))
        final=stack.enter_context(patch.object(r,'final_checks',return_value=coverage))
        assert r.run(sha)==1
        completed=c.read(tmp_path/'completion.json')
        assert completed['status']==('incomplete' if player_failed else 'incomplete_measurement_failure_verified')
        assert 'original measurement failure' in completed['failure'] and completed['coverage']==coverage
        assert final.call_count==1 and completed['automatic_retries']==0
        assert completed['actual_os_exit_requirement_scope']=='successful_study_acceptance'
        assert completed['incomplete_verified_requires_nonzero_actual_exits'] is True
