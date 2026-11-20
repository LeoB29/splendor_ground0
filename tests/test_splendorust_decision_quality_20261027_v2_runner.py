"""Model-free bounded controller counterexamples; no learned models or games."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from tools.splendorust_decision_quality_20261027_v2 import runner as r, common as c, guardian
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def gate_inputs():
    matrix=c.read(c.ROOT/'outputs/decision_quality_design_20261027_01/prepared_request_matrix.json')
    costs=[dict(sequence=s['sequence'],root_id=s['root_id'],quota=s['quota'],seconds=.5,
                bytes=10000,verification_seconds=.01,row_bytes=5000)
           for s in matrix['requests'] if s['stage']=='research_pilot']
    return dict(costs=costs,matrix=matrix,current_bytes=10*c.MiB,collection_remaining=3000.,
        global_remaining=5000.,source_runtime_seconds=3.,closed_proof_seconds=.1,
        closed_proof_bytes=c.MiB,seal_measurement=dict(seconds=.1,bytes=c.MiB))


def test_fixed_cost_gate_passes_and_uses_exact_remaining102():
    value=r.pilot_gate(**gate_inputs())
    assert value['passed'] and value['remaining_requests']==102 and value['cost_only']
    assert value['projected_remaining_close_and_verify_seconds']>0


@pytest.mark.parametrize('field,value,check',[
    ('collection_remaining',.001,'collection_time'),('global_remaining',.001,'global_time'),
    ('current_bytes',c.PREFINAL_EVIDENCE_BYTES,'prefinal_evidence'),
    ('source_runtime_seconds',2000,'final_time')])
def test_cost_gate_rejects_each_bounded_dimension(field,value,check):
    args=gate_inputs();args[field]=value
    result=r.pilot_gate(**args)
    assert not result['passed'] and not result['checks'][check]


def test_cost_gate_rejects_final_and_total_evidence_independently():
    args=gate_inputs()
    for row in args['costs']:row['row_bytes']=4*c.MiB
    value=r.pilot_gate(**args)
    assert not value['checks']['final_evidence'] and not value['checks']['total_evidence']


def test_cost_gate_cannot_drop_slow_pilot_or_reorder_roots():
    args=gate_inputs();args['costs'].pop()
    with pytest.raises(ValueError,match='predetermined'):r.pilot_gate(**args)
    args=gate_inputs();args['costs'][0]['root_id']='replacement'
    with pytest.raises(ValueError,match='identity'):r.pilot_gate(**args)


def test_request_transport_clock_and_raw_readers_are_inherited():
    assert r.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert r.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert r.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert r.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader


def test_bound_constants_match_prepared_protocol():
    assert (c.JOB_SECONDS,c.COLLECTION_CUTOFF_SECONDS,c.RAM_WAIT_SECONDS)==(7200,5400,300)
    assert c.STAGES==dict(qualification=900,diagnostic=4500,finalization=1800)
    assert c.MAX_EVIDENCE_BYTES==c.GiB and c.FINAL_RESERVE_BYTES==c.PREFINAL_EVIDENCE_BYTES==512*c.MiB
    assert c.MAX_WORKERS==4


def test_guard_uses_either_utc_or_monotonic_expiry():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=101.),patch.object(r.time,'time',return_value=8201.):
        with pytest.raises(ValueError,match='deadline'):env.guard()
    with patch.object(r.time,'monotonic',return_value=7301.),patch.object(r.time,'time',return_value=1001.):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def test_final_stage_is_clamped_to_shared_origin(tmp_path):
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=6500.),patch.object(r.time,'time',return_value=7400.),\
         patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('finalization')
    assert env.deadline==7300. and env.wall_deadline==8200.


def test_late_sidecar_refresh_counts_bytes_after_original_close(tmp_path):
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{}}
    folder=tmp_path/'worker';folder.mkdir();(folder/'cleanup.json').write_bytes(b'abc')
    env.register_worker(folder);env.close_worker(folder);assert env.evidence['workers'][folder]==3
    (folder/'parent.json').write_bytes(b'late-evidence')
    env.close_worker(folder);assert env.evidence['workers'][folder]==16


def test_original_placement_request_rejected_by_guardian():
    with pytest.raises(ValueError,match='isolated'):guardian.require_isolated_request({'placement_mode':'original'})


def test_factory_forbids_second_resident_tree_before_model_import():
    env=SimpleNamespace(resident={1:'active'})
    with pytest.raises(ValueError,match='one resident'):r.make_worker('bad',env,{},route='research')


def test_factory_rejects_unknown_route():
    with pytest.raises(ValueError,match='allowlist'):r.make_worker('bad',None,{},route='other')


def test_ordinary_never_shortens_clock_or_sends_private_state(tmp_path):
    child=SimpleNamespace(route='frozen',_counter=0,output_dir=tmp_path)
    env=SimpleNamespace(admit=lambda:None,remaining=lambda:.99,timeout=lambda n:n,guard=lambda:None)
    observer={'lawful':'only'};spec={'quota':64,'seed_family':'A','complete_response_seconds':1.}
    writes=[]
    with patch.object(c,'save',side_effect=lambda path,value:writes.append((path,value))),patch.object(r,'request') as request:
        with pytest.raises(ValueError,match='unchanged full'):r.ordinary(child,spec,observer,env,{})
        request.assert_not_called()
    assert writes[0][1]=={'player':'search','observer':observer}


def test_collect_stops_first_failure_without_dispatching_remaining_requests(tmp_path):
    from tools.splendorust_decision_quality_20261027_v2 import audit
    specs=[dict(sequence=i,stage='frozen_reference_reproduction',root_id=f'r{i}',quota=64) for i in range(2)]
    matrix={'requests':specs};saved={f'r{i}':{'observer':{}} for i in range(2)}
    row=dict(completed=False,clock={'status':'timeout','clock':'perf_counter_ns','request_id':'request-00000001',
        'deadline_seconds':1.,'elapsed_seconds':1.01,'receipt_elapsed_seconds':None,'received_before_deadline':False,
        'accepted':False,'result':None},spec=specs[0])
    env=SimpleNamespace(accounted_bytes=lambda:0,guard=lambda:None)
    child=SimpleNamespace(route='frozen')
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'make_worker',return_value=child) as factory,\
         patch.object(r,'ordinary',return_value=row) as ordinary,patch.object(r,'verify_saved_record'),\
         patch.object(audit,'audit_record'),patch.object(r,'close_and_audit') as closed:
        rows,stop=r.collect(env,{},matrix,saved)
    assert rows==[row] and stop['sequence']==0 and stop['no_retry']
    assert ordinary.call_count==factory.call_count==closed.call_count==1
    assert (tmp_path/'records/0000.json').exists() and not (tmp_path/'records/0001.json').exists()


def test_parent_captured_wire_hash_must_match_raw_closed_bytes():
    packets=[r._packet_evidence(b'{"type":"error"}\n')]
    parent=dict(response_raw_sha256='0'*64,response_raw_bytes=17,response_capture_overflow=False)
    with pytest.raises(ValueError,match='bytes differ'):r._bind_parent_wire(parent,packets,completed=True)


def test_failed_no_arrival_does_not_accept_partial_result():
    parent=dict(response_raw_sha256=None,receipt_arrival_ns=None,response_raw_bytes=None)
    assert r._bind_parent_wire(parent,[],completed=False) is False
    with pytest.raises(ValueError,match='failed request'):r._bind_parent_wire(parent,[],completed=True)


def test_child_claim_cannot_match_different_late_raw_response():
    child=dict(outcome='response_flushed',serialized_packet_utf8_sha256='0'*64,serialized_packet_utf8_bytes=12)
    receipt=dict(status='timeout',receipt_elapsed_seconds=None)
    with pytest.raises(ValueError,match='serialized packet differs'):
        r._bind_child_wire(child,[r._packet_evidence(b'{"x":1}\n')],receipt)


def test_saved_success_requires_exact_sidecars_and_placement(tmp_path):
    row=dict(profile_capture_error='corrupt')
    with pytest.raises(ValueError,match='integrity'):r.verify_saved_record(row,{})


def research_profile_fixture():
    from test_splendorust_boundary_timing_analyze import profile_row
    row=profile_row();parent=row['parent_profile'];child=row['child_profile'];clock=row['clock']
    payload=dict(player='search',observer=row['observer'],quota=1024,seed_label='B',telemetry=True)
    parent.update(receipt_arrival_ns=3_000_000_000,receipt_reconstructed_arrival_ns=3_000_000_000,
                  capture_snapshot_ns=3_100_000_000,raw_arrivals_ns=[3_000_000_000],response_raw_bytes=400000)
    child.update(backend='research_graph',quota=1024,payload_sha256=c.digest(payload),serialized_packet_utf8_bytes=400000)
    clock.update(clock='perf_counter_ns',deadline_seconds=30.,receipt_elapsed_seconds=2.,elapsed_seconds=2.1)
    return parent,child,clock,payload


def test_research_boundary_keeps_true_two_second_arrival_and_large_packet():
    parent,child,clock,payload=research_profile_fixture()
    result=r.validate_profiles(parent,child,clock,payload,'research',worker_pid=42)
    assert result['status']=='passed' and result['request_deadline_unchanged']
    assert clock['receipt_elapsed_seconds']==2. and clock['deadline_seconds']==30.


@pytest.mark.parametrize('field,value,match',[
    ('quota',64,'quota'),('backend','qualified_graph64','backend'),
    ('graph_identity_sha256','f'*64,'backend'),('payload_sha256','f'*64,'input'),
    ('result_sha256','f'*64,'result'),('serialized_packet_utf8_bytes',4*c.MiB+1,'size bound')])
def test_research_sidecar_identity_or_content_corruption_rejected(field,value,match):
    parent,child,clock,payload=research_profile_fixture();child[field]=value
    with pytest.raises(ValueError,match=match):r.validate_profiles(parent,child,clock,payload,'research',worker_pid=42)


def test_research_clock_cannot_be_projected_to_frozen_one_second():
    parent,child,clock,payload=research_profile_fixture();clock['deadline_seconds']=1.
    with pytest.raises(ValueError):r.validate_profiles(parent,child,clock,payload,'research',worker_pid=42)


def failure_clock(status='timeout',detail=None):
    return dict(clock='perf_counter_ns',request_id='request-00000001',status=status,detail=detail,
        deadline_seconds=30.,elapsed_seconds=30.01,receipt_elapsed_seconds=None,
        received_before_deadline=False,accepted=False,result=None)


def test_only_authentic_deadline_failures_can_be_verified_partial_diagnostics():
    assert r.deadline_failure(failure_clock(),'research')
    assert r.deadline_failure(failure_clock('worker_error','RESEARCH_INTERNAL_DEADLINE: limit'),'research')
    assert not r.deadline_failure(failure_clock('resource_guard','RSS exceeded'),'research')
    assert not r.deadline_failure(failure_clock('worker_error','research source changed'),'research')
    assert not r.deadline_failure(failure_clock('malformed','invalid JSON'),'research')


def test_false_outer_timeout_before_allowance_is_not_diagnostic_negative():
    receipt=failure_clock();receipt['elapsed_seconds']=.4
    assert not r.deadline_failure(receipt,'research')
