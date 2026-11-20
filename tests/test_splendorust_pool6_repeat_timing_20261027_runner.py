"""No model execution: fixed repeats, trusted clocks, failure and budget proofs."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_pool6_repeat_timing_20261027 import runner as r, audit as a, common as c, guardian, preparation, design, qualification
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


def test_contract_is_exact_144_no_game_no_promotion():
    r.validate_freeze_contract(preparation.contract())
    assert (c.JOB_SECONDS,c.COLLECTION_CUTOFF_SECONDS,c.MAX_WORKERS,c.RAM_WAIT_SECONDS)==(3600,1800,2,600)
    assert c.STAGES==dict(qualification=900,diagnostic=900,final=1800)
    assert c.MAX_EVIDENCE_BYTES==2*c.GiB and c.FINAL_RESERVE_BYTES==512*c.MiB
    bad=preparation.contract();bad['total_requests']=145
    with pytest.raises(ValueError):r.validate_freeze_contract(bad)


def test_original_timed_transport_is_still_inherited():
    assert r.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert r.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert r.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert r.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader


def test_only_two_resident_players():
    with pytest.raises(ValueError,match='two resident'):
        r.make_worker('third',SimpleNamespace(resident={1:1,2:2}),{},route='research')


@pytest.mark.parametrize('mono,utc',[(3701.,1001.),(101.,4601.)])
def test_either_utc_or_monotonic_clock_expires_without_reset(mono,utc):
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=mono),patch.object(r.time,'time',return_value=utc):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def test_final_reserve_not_accessible_to_collection():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=1800.),patch.object(r.time,'time',return_value=2700.),patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('diagnostic');assert env.deadline==1900.
        env.begin('final');assert env.deadline==3600.


def test_resource_stage_binding_changes_only_job_deadline(tmp_path):
    children={route:SimpleNamespace(_lock=SimpleNamespace(locked=lambda:False),_counter=2,_closed=False,
        absolute_deadline=900.,output_dir=tmp_path/('qualification_'+route),identity={'pid':i,'created':1.})
        for i,route in enumerate(('frozen','research'),1)}
    env=SimpleNamespace(stage='diagnostic',resident={1:1,2:2},effective_deadline=lambda:1700.,
        deadline=1700.,end=3600.,start=0.)
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path):value=r.bind_diagnostic_deadline(env,children)
    assert all(x.absolute_deadline==1700. and x._counter==2 for x in children.values())
    assert not value['request_clocks_reset'] and value['ordinary_deadline_seconds']==1. and value['internal_seconds']==.9
    assert [x['previous_absolute_deadline'] for x in value['bindings']]==[900.,900.]


@pytest.mark.parametrize('pending,deadline',[(True,1700.),(False,1900.)])
def test_resource_rebinding_refuses_active_request_or_collection_extension(tmp_path,pending,deadline):
    children={route:SimpleNamespace(_lock=SimpleNamespace(locked=lambda:pending),_counter=2,_closed=False,
        absolute_deadline=900.,output_dir=tmp_path/route,identity={'pid':1}) for route in ('frozen','research')}
    env=SimpleNamespace(stage='diagnostic',resident={1:1,2:2},effective_deadline=lambda:deadline,deadline=deadline,end=3600.,start=0.)
    with pytest.raises(ValueError):r.bind_diagnostic_deadline(env,children)
    assert children['frozen'].absolute_deadline==900.


def test_late_sidecars_are_charged_after_failed_worker_closes(tmp_path):
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{}}
    folder=tmp_path/'worker';folder.mkdir();(folder/'cleanup.json').write_bytes(b'abc')
    env.register_worker(folder);env.close_worker(folder);assert env.evidence['workers'][folder]==3
    (folder/'parent.json').write_bytes(b'late-sidecar');env.close_worker(folder)
    assert env.evidence['workers'][folder]==15


def collect_fixture(tmp_path,*,failure=None,transition_error=False,exact_error=False):
    matrix=design.build_matrix();rows=[];env=SimpleNamespace(guard=lambda:None)
    children={route:SimpleNamespace(route=route) for route in ('frozen','research')}
    calls=[]
    def ordinary(child,spec,*_):
        calls.append(spec['sequence']);return row(spec,'timeout' if spec['sequence']==failure else 'ok')
    transition=ValueError('ledger corruption') if transition_error else lambda *args: {'successor':'saved'}
    exact=ValueError('historical numeric mismatch') if exact_error else lambda *args,**kwargs: {'status':'passed'}
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'ordinary',side_effect=ordinary),\
        patch.object(r.ledger,'transition',side_effect=transition),patch.object(r,'verify_saved_record'),\
        patch.object(a,'validate_record'),patch.object(qualification,'validate_repeat_row',side_effect=exact):
        if transition_error or exact_error:
            with pytest.raises(ValueError):r.collect(env,{},matrix,children,rows)
        else:r.collect(env,{},matrix,children,rows)
    return rows,calls


def test_all_144_slow_successes_retained_without_700ms_gate(tmp_path):
    rows,calls=collect_fixture(tmp_path)
    assert calls==list(range(144)) and len(rows)==144 and all(x['clock']['receipt_elapsed_seconds']==.85 for x in rows)
    assert not(tmp_path/'collection_stop.json').exists()
    assert len(list((tmp_path/'records').glob('*.json')))==144


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


def test_actual_saved_sources_full_science_and_successors_without_model():
    panel=design.build_panel();matrix=design.build_matrix(panel)
    freeze=c.read(c.QUAL_OUT/'freeze.json');rows=[]
    for spec in matrix['requests'][:12]:
        result=qualification.historical_result(spec)['result']
        clock=receipt();clock['result']=result
        saved=dict(spec=spec,route=spec['route'],observer=spec['observer'],before=spec['before_snapshot'],
            after=r.ledger.transition(spec['before_snapshot'],result['action']),clock=clock,completed=True,
            action=result['action'],trace=result['diagnostics'])
        assert a.validate_record(saved,spec,freeze)['transition_checked']
        rows.append(saved)
    checked=a.audit_prefix(rows,matrix,freeze)
    assert checked['attempted_requests']==12 and checked['transitions']==12 and not checked['complete']
    bad=deepcopy(rows[0]);bad['after']=deepcopy(bad['before'])
    with pytest.raises(ValueError):a.validate_record(bad,bad['spec'],freeze)


def test_prefix_cannot_skip_replace_or_continue_failed_request():
    matrix=design.build_matrix();rows=[row(s) for s in matrix['requests'][:2]]
    with patch.object(a,'validate_record',return_value={'transition_checked':True}):
        bad=deepcopy(rows);bad.reverse()
        with pytest.raises(ValueError):a.audit_prefix(bad,matrix,{})
        rows[0]['completed']=False
        with pytest.raises(ValueError):a.audit_prefix(rows,matrix,{})
