"""Model-free controller, admission, transport and live-science counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_pool6_live_20261027 import runner as r, audit as a, common as c, guardian, preparation, schedule
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def clock(status='ok',elapsed=.4):
    good=status=='ok'
    return dict(status=status,clock='perf_counter_ns',request_id='request-00000001',deadline_seconds=1.,
        elapsed_seconds=elapsed if good else 1.01,receipt_elapsed_seconds=elapsed if good else None,
        received_before_deadline=good,accepted=good,result={} if good else None)


def args():
    return dict(observed_seconds=60.,observed_variable_bytes=6*c.MiB,independent_seconds=6.,
        fixed_final_seconds=100.,final_evidence_bytes=128*c.MiB,maximum_move_seconds=.6,all_natural=True)


def test_source_transport_clock_readers_unchanged():
    assert r.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert r.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert r.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert r.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader


def test_phase_contract_matches_freezer_exactly():
    value=preparation.contract();r.validate_freeze_contract(value)
    assert value['phase']=='qualification' and value['qualification_decisions']==194
    assert c.MAX_WORKERS==62 and c.MAX_GRAPH_BUILDS==62
    assert c.JOB_SECONDS==7200 and c.COLLECTION_CUTOFF_SECONDS==3600
    assert c.STAGES==dict(qualification=1800,operational=1800,final=3600)
    assert c.MAX_EVIDENCE_BYTES==4*c.GiB and c.FINAL_RESERVE_BYTES==c.GiB
    assert c.RAM_WAIT_SECONDS==600 and c.INITIAL_FREE_DISK_BYTES==12*c.GiB
    value['phase']='assessment'
    with pytest.raises(ValueError):r.validate_freeze_contract(value)


def test_allowance_never_uses_final_reserve_for_collection():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=3600.),patch.object(r.time,'time',return_value=4500.),patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('operational');assert env.deadline==3700.
        env.begin('final');assert env.deadline==7200. and env.wall_deadline==8100.


@pytest.mark.parametrize('mono,utc',[(7301.,1001.),(101.,8201.)])
def test_either_real_clock_expiration_ends_job(mono,utc):
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=mono),patch.object(r.time,'time',return_value=utc):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def test_third_resident_worker_rejected_before_model():
    with pytest.raises(ValueError,match='two resident'):
        r.make_worker('extra',SimpleNamespace(resident={1:1,2:2}),{},route='research')


def test_late_sidecar_accounting_refresh(tmp_path):
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{}}
    folder=tmp_path/'worker';folder.mkdir();(folder/'cleanup.json').write_bytes(b'abc')
    env.register_worker(folder);env.close_worker(folder);assert env.evidence['workers'][folder]==3
    (folder/'late_sidecar.json').write_bytes(b'original-clock');env.close_worker(folder)
    assert env.evidence['workers'][folder]==17


def test_assessment_fault_continues_next_scheduled_game_once():
    specs=schedule.stage_schedule('main')[:3];seen=[]
    def play(spec,*_):
        seen.append(spec['game_index']);fault={'original':'timeout'} if spec['game_index']==0 else None
        return dict(spec,reason='forfeit' if fault else 'natural'),[],0.,fault
    with patch.object(r,'play_game',side_effect=play):games=r.collect_assessment(None,{},specs)
    assert seen==[0,1,2] and games[0]['reason']=='forfeit' and len(games)==3
    assert specs[0]['seed_decimal']==specs[1]['seed_decimal'] and specs[0]['challenger_seat']!=specs[1]['challenger_seat']


def test_assessment_infrastructure_failure_stops_without_losing_prior_prefix():
    specs=schedule.stage_schedule('main')[:3];games=[]
    def play(spec,*_):
        if spec['game_index']==1:raise ValueError('corrupted raw evidence')
        return dict(spec,reason='natural'),[],0.,None
    with patch.object(r,'play_game',side_effect=play) as run:
        with pytest.raises(ValueError,match='corrupted'):r.collect_assessment(None,{},specs,games)
    assert run.call_count==2 and len(games)==1 and games[0]['game_index']==0


def test_operational_stops_first_attributable_fault():
    specs=schedule.stage_schedule('operational');fault={'original':'timeout'}
    with patch.object(r,'play_game',return_value=({'reason':'forfeit'},[],0.,fault)) as run,patch.object(r,'bytes_used',return_value=0):
        games,stop=r.operational(None,{},specs)
    assert run.call_count==1 and len(games)==1 and stop==fault


def test_qualification_failure_is_saved_once_no_remaining_dispatch(tmp_path):
    plan={'parity_requests':[dict(index=i,route='frozen',observer={},quota=64) for i in range(3)]}
    row=dict(spec=plan['parity_requests'][0],clock=clock('timeout'),completed=False,route='frozen')
    child=SimpleNamespace(route='frozen');env=SimpleNamespace(guard=lambda:None)
    from tools.splendorust_pool6_live_20261027 import adapters
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'make_worker',return_value=child),\
        patch.object(adapters,'actual_entrypoint',return_value={'status':'ok'}),patch.object(r,'ordinary',return_value=row) as ordinary,\
        patch.object(r,'verify_saved_record'),patch.object(r,'close_and_audit'):
        result=r.qualify(env,{},plan)
    assert not result['qualified'] and ordinary.call_count==1
    assert json.loads((tmp_path/'qualification_stop.json').read_text())['clock']==row['clock']
    assert (tmp_path/'qualification_rows/0000.json').exists() and not(tmp_path/'qualification_rows/0001.json').exists()


def test_qualification_margin_is_admission_only():
    rows=[dict(completed=True,clock=clock()) for _ in range(90)]
    assert r.qualification_gate(rows)['passed']
    rows[-1]['clock']=clock(elapsed=.7)
    gate=r.qualification_gate(rows)
    assert gate['complete'] and not gate['passed'] and rows[-1]['clock']['status']=='ok'
    assert not r.qualification_gate(rows[:-1])['passed']


def test_cost_projection_uses_max_game_and_includes_fresh_closure():
    measured=[dict(collection_seconds=x,independent_seconds=x/10,bytes=x*1000) for x in (1,1,1,1,1,10)]
    components=dict(initialization=2.,graph=.4,closure=.5,runtime=3.)
    values=r.assessment_inputs(measured,components,10000,.4)
    result=r.operational_gate(**values)
    assert values['observed_seconds']==60. and values['observed_variable_bytes']==60000
    assert result['projected_collection_seconds']==10*200*1.5+606.
    assert values['fixed_final_seconds']==48*1.5*(2+.4+.5+2)+6+300
    assert result['passed'] and not result['outcomes_used']


@pytest.mark.parametrize('field,value',[('observed_seconds',10000.),('observed_variable_bytes',100*c.MiB),
    ('fixed_final_seconds',7200.),('final_evidence_bytes',c.GiB),('maximum_move_seconds',.7),('all_natural',False)])
def test_admission_resource_and_timing_limits_fail_closed(field,value):
    values=args();values[field]=value
    assert not r.operational_gate(**values)['passed']


def test_successful_850ms_operational_move_is_negative_admission_not_loss():
    receipt=clock(elapsed=.85)
    assert a.check_clock(receipt)==.85
    values=args();values['maximum_move_seconds']=.85
    result=r.operational_gate(**values)
    assert not result['passed'] and result['resource_passed'] and result['margin_is_not_timeout']
    with pytest.raises(ValueError,match='Successful'):a.classify_failure(receipt)


@pytest.mark.parametrize('status,detail',[('resource_guard','resource ceiling'),('worker_error','unclassified engine error'),
    ('illegal','DECISION_QUALITY_AUDIT: mismatched numeric output')])
def test_integrity_or_unknown_fault_is_not_a_forfeit(status,detail):
    receipt=clock(status);receipt['detail']=detail
    if status=='illegal':receipt.update(receipt_elapsed_seconds=.4,received_before_deadline=True,result={})
    with pytest.raises(ValueError):a.classify_failure(receipt)


def test_authentic_internal_and_external_deadlines_are_losses():
    outer=clock('timeout');assert a.classify_failure(outer)=='timeout'
    inner=clock('worker_error');inner.update(detail='SAMPLING_INTERNAL_DEADLINE: fixed work unfinished',
        receipt_elapsed_seconds=.91,received_before_deadline=True,elapsed_seconds=.91)
    assert a.classify_failure(inner)=='worker_error'
    assert r.deadline_failure(inner,'research')
    bad=deepcopy(outer);bad['elapsed_seconds']=.5
    with pytest.raises(ValueError):a.classify_failure(bad)


def research_profile_fixture():
    from test_splendorust_boundary_timing_analyze import profile_row
    row=profile_row();parent=row['parent_profile'];child=row['child_profile'];receipt=row['clock']
    payload=dict(player='search',observer=row['observer'])
    child.update(backend='research_graph',quota=64,payload_sha256=c.digest(payload))
    receipt.update(clock='perf_counter_ns',elapsed_seconds=.45)
    return parent,child,receipt,payload


def test_two_field_candidate_payload_has_fixed64_sidecar():
    parent,child,receipt,payload=research_profile_fixture()
    checked=r.validate_profiles(parent,child,receipt,payload,'research',worker_pid=42)
    assert checked['status']=='passed' and checked['request_deadline_unchanged']
    assert set(payload)=={'player','observer'}


@pytest.mark.parametrize('field,value',[('quota',32),('payload_sha256','f'*64),('result_sha256','f'*64),
                                      ('backend','qualified_graph64'),('graph_identity_sha256','f'*64)])
def test_raw_sidecar_identity_corruption_is_rejected(field,value):
    parent,child,receipt,payload=research_profile_fixture();child[field]=value
    with pytest.raises(ValueError):r.validate_profiles(parent,child,receipt,payload,'research',worker_pid=42)


def test_parent_wire_hash_does_not_accept_other_packet():
    packets=[r._packet_evidence(b'{"type":"error"}\n')]
    parent=dict(response_raw_sha256='0'*64,response_raw_bytes=17,response_capture_overflow=False)
    with pytest.raises(ValueError):r._bind_parent_wire(parent,packets,completed=True)


def test_failed_receipt_keeps_rejected_result_without_accepted_action():
    receipt=clock('illegal');receipt.update(result={'rejected':'evidence'},receipt_elapsed_seconds=.4,
        received_before_deadline=True,elapsed_seconds=.4,detail='DECISION_QUALITY_AUDIT: evidence mismatch')
    assert a.check_clock(receipt)==.4 and not receipt['accepted']
    with pytest.raises(ValueError):a.classify_failure(receipt)


def test_original_placement_rejected():
    with pytest.raises(ValueError,match='isolated'):guardian.require_isolated_request({'placement_mode':'original'})


def test_actual_saved_pool6_science_with_new_declared_live_identity():
    prior=c.ROOT/'outputs/splendorust_sampling_stability_20261027_01'
    matrix=c.read(prior/'prepared_request_matrix.json')
    spec=next(s for s in matrix['requests'] if s.get('world_pool')==6 and s.get('seed_label')=='A')
    row=c.read(prior/'records'/f"{spec['sequence']:04d}.json");result=deepcopy(row['clock']['result'])
    identity=dict(profile_id='live_puct64_pool6_20261027_v1',profile_sha256='1'*64,
        wrapper_id='live_puct64_pool6_20261027_v1_1s',native_binary_sha256=result['research']['native_binary_sha256'])
    result.update({k:v for k,v in identity.items() if k!='native_binary_sha256'});result['diagnostics']['profile_id']=identity['profile_id']
    result['research'].update(schema='splendorust_pool6_live_v1',telemetry_enabled=False)
    for phase in result['research']['phases']:phase['telemetry']=None
    checked=a.validate_result(result,row['observer'],'research',identity=identity)
    assert checked['simulations']==64 and checked['neural_callbacks']>0
    result['research']['world_pool']=3
    with pytest.raises(ValueError):a.validate_result(result,row['observer'],'research',identity=identity)


def test_live_turn124_heuristic_is_a_valid_complete_decision():
    from tools.splendorust_pool6_live_20261027 import qualification
    spec=next(s for s in qualification.build_plan()['parity_requests'] if s.get('exact_frozen_shortcut'))
    source=c.read(c.ROOT/spec['expected_entrypoint']['path'])
    result=deepcopy(next(f for f in source['result']['normalized_suite']['fixtures'] if f['id']==spec['expected_case'])['results']['search'])
    pid='live_puct64_pool6_20261027_v1';identity=dict(profile_id=pid,profile_sha256='1'*64,wrapper_id=pid+'_1s',native_binary_sha256='2'*64)
    result.update({k:v for k,v in identity.items() if k!='native_binary_sha256'});result['diagnostics']['profile_id']=pid
    phases=result['diagnostics']['phases']
    result['research']=dict(schema='splendorust_pool6_live_v1',quota=64,world_pool=6,seed_label='A',
        seed=result['diagnostics']['observer_seed'],telemetry_enabled=False,internal_seconds=.9,native_binary_sha256='2'*64,
        phases=[dict(phase_index=i,phase_kind='Main' if i==0 else 'TakeNoble',telemetry=None,callback_bits=[])
                for i,p in enumerate(phases)])
    for p in phases:p.setdefault('inference',[])
    checked=a.validate_result(result,spec['observer'],'research',identity=identity)
    assert checked['simulations']==checked['neural_callbacks']==0


def test_positive_qualification_handoff_exact_status_and_gate_names(tmp_path):
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'QUAL_OUT',tmp_path):
        write('freeze.json',{'inputs':[]});write('completion.json',{'status':'qualified_verified'})
        write('completed_evidence_seal.json',{'artifacts':[]});write('assessment_gate.json',{'passed':True})
        for name in ('verification_summary.json','guardian_os_exit.json','execution.json','watchdog_execution.json'):write(name,{'passed':True})
        names=('completion.json','verification_summary.json','completed_evidence_seal.json','guardian_os_exit.json',
               'execution.json','watchdog_execution.json','assessment_gate.json')
        write('postrun/independent_completion_review.json',dict(accepted=True,freeze_sha256=c.file_hash(tmp_path/'freeze.json'),
            consumed_inputs=[c.artifact(tmp_path/n) for n in names]))
        admitted=preparation.accepted_admission()
        assert admitted['status']=='accepted' and admitted['gate']['path']=='assessment_gate.json'
        write('assessment_gate.json',{'passed':False})
        with pytest.raises(ValueError):preparation.accepted_admission()
