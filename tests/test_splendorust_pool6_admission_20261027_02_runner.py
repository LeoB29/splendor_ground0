"""Model-free controller, admission, transport and live-science counterexamples."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_pool6_admission_20261027_02 import runner as r, audit as a, common as c, guardian, preparation, schedule
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def clock(status='ok',elapsed=.4):
    good=status=='ok'
    return dict(status=status,clock='perf_counter_ns',request_id='request-00000001',deadline_seconds=1.,
        elapsed_seconds=elapsed if good else 1.01,receipt_elapsed_seconds=elapsed if good else None,
        received_before_deadline=good,accepted=good,result={} if good else None)


def args():
    return dict(observed_seconds=60.,observed_variable_bytes=6*c.MiB,independent_seconds=6.,
        fixed_final_seconds=100.,final_evidence_bytes=128*c.MiB,maximum_move_seconds=.6,all_natural=True,current_evidence_bytes=20*c.MiB)


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


def test_operational_stops_first_attributable_fault():
    specs=schedule.stage_schedule('operational');fault={'original':'timeout'}
    with patch.object(r,'play_game',return_value=({'reason':'forfeit'},[],0.,fault)) as run,patch.object(r,'bytes_used',return_value=0):
        games,stop=r.operational(None,{},specs)
    assert run.call_count==1 and len(games)==1 and stop==fault


def test_qualification_failure_is_saved_once_no_remaining_dispatch(tmp_path):
    plan={'parity_requests':[dict(index=i,route='frozen',observer={},quota=64) for i in range(3)]}
    row=dict(spec=plan['parity_requests'][0],clock=clock('timeout'),completed=False,route='frozen')
    child=SimpleNamespace(route='frozen');env=SimpleNamespace(guard=lambda:None)
    from tools.splendorust_pool6_admission_20261027_02 import adapters
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
    values=r.admission_inputs(measured,components,10000,.4,20*c.MiB)
    result=r.operational_gate(**values)
    assert values['observed_seconds']==60. and values['observed_variable_bytes']==60000
    assert result['projected_games']==0 and not result['benchmark_authorized_to_launch']
    assert result['projected_final_seconds']==1.5*6+values['fixed_final_seconds']
    assert values['fixed_final_seconds']==48*1.5*(2+.4+.5+2)+6+300
    assert result['passed'] and not result['outcomes_used']


@pytest.mark.parametrize('field,value',[('current_evidence_bytes',c.PREFINAL_EVIDENCE_BYTES),
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
    from tools.splendorust_pool6_admission_20261027_02 import qualification
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




def test_scope_has_no_assessment_dispatch_or_schedule():
    assert not hasattr(r,'collect_assessment')
    with pytest.raises(ValueError):schedule.stage_schedule('main')
    assert len(schedule.stage_schedule('operational'))==6
    assert c.PHASE=='qualification'


def test_two_resident_qualification_follows_prospective_order_and_exact_checks(tmp_path):
    from tools.splendorust_pool6_admission_20261027_02 import adapters,qualification
    plan={'parity_requests':[dict(index=i,route='research' if i%2 else 'frozen',observer={},quota=64) for i in range(90)]}
    events=[];env=SimpleNamespace(guard=lambda:None)
    def make(name,*args,route):events.append(('make',route));return SimpleNamespace(route=route)
    def ordinary(child,spec,*args):
        events.append(('request',spec['index'],child.route))
        return dict(spec=spec,clock=clock(elapsed=.7 if spec['index']==0 else .4),completed=True,route=child.route)
    def exact(row,spec,cache,**kwargs):
        assert (tmp_path/'qualification_rows'/f"{spec['index']:04d}.json").exists()
        assert kwargs['frozen_entrypoint']=={'status':'ok'}
        events.append(('exact',spec['index']))
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'make_worker',side_effect=make),\
        patch.object(adapters,'actual_entrypoint',return_value={'status':'ok'}),patch.object(r,'ordinary',side_effect=ordinary),\
        patch.object(r,'verify_saved_record'),patch.object(r,'close_and_audit',side_effect=lambda child,*a:events.append(('close',child.route))),\
        patch.object(qualification,'validate_admission_row',side_effect=exact),patch.object(qualification,'validate_parity_rows',return_value={'passed':True}):
        result=r.qualify(env,{},plan)
    assert events[:2]==[('make','frozen'),('make','research')]
    assert [e[1] for e in events if e[0]=='request']==list(range(90))
    assert all(events.index(('exact',i))<events.index(('request',i+1,plan['parity_requests'][i+1]['route'])) for i in range(89))
    assert events[-2:]==[('close','frozen'),('close','research')]
    assert not result['qualified'] and result['timing_gate']['complete'] and result['failure'] is None


def test_immediate_exact_mismatch_preserves_row_and_closes_both(tmp_path):
    from tools.splendorust_pool6_admission_20261027_02 import adapters,qualification
    spec=dict(index=0,route='frozen',observer={},quota=64);plan={'parity_requests':[spec,dict(spec,index=1)]}
    row=dict(spec=spec,clock=clock(),completed=True,route='frozen');env=SimpleNamespace(guard=lambda:None)
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),\
        patch.object(r,'make_worker',side_effect=lambda name,*a,route:SimpleNamespace(route=route)),\
        patch.object(adapters,'actual_entrypoint',return_value={'status':'ok'}),patch.object(r,'ordinary',return_value=row) as dispatch,\
        patch.object(r,'verify_saved_record'),patch.object(r,'close_and_audit') as close,\
        patch.object(qualification,'validate_admission_row',side_effect=ValueError('exact historical mismatch')):
        with pytest.raises(ValueError,match='historical mismatch'):r.qualify(env,{},plan)
    assert dispatch.call_count==1 and close.call_count==2
    assert json.loads((tmp_path/'qualification_rows/0000.json').read_text())['clock']==row['clock']
    assert not (tmp_path/'qualification_rows/0001.json').exists()


def test_additional_negative_predecessor_inventory_exact(tmp_path):
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path/'new'):
        scopes=[];pins=[]
        for name,status in [('repeat','completed_verified'),('negative','not_admitted_verified')]:
            old=tmp_path/name;write(old/'freeze.json',{'id':name});write(old/'result.json',{'status':status})
            items=[c.artifact(p) for p in sorted(old.iterdir())];pins+=items
            scopes.append(dict(prior_freeze=c.artifact(old/'freeze.json'),prior_status=status,prior_attempt_artifacts=items))
        manifest=dict(scopes[0],additional_prior_attempts=[scopes[1]])
        write(tmp_path/'new/preservation.json',manifest)
        freeze={'inputs':pins,'prior_accepted_attempt':{'preservation':c.artifact(tmp_path/'new/preservation.json')}}
        # Same immutable full-input verification which precedes preservation in final_checks.
        c.verify(freeze['inputs']);proof=r.verify_prior_preservation(freeze,SimpleNamespace(guard=lambda:None))
        assert c.read(tmp_path/proof['path'])['prior_files']==4
        write(tmp_path/'negative/unexpected.json',{})
        with pytest.raises(ValueError,match='gained or lost'):r.verify_prior_preservation(freeze,SimpleNamespace(guard=lambda:None))
        (tmp_path/'negative/unexpected.json').unlink();(tmp_path/'negative/result.json').unlink()
        with pytest.raises(ValueError,match='gained or lost'):r.verify_prior_preservation(freeze,SimpleNamespace(guard=lambda:None))
        write(tmp_path/'negative/result.json',{'changed':True})
        with pytest.raises(ValueError):c.verify(freeze['inputs'])


def resource_fixture(tmp_path,*,remaining=3600,margin_gate=None):
    folder=tmp_path/'workers/operational_g0_challenger';folder.mkdir(parents=True)
    (folder/'closed.json').write_text('{}');(tmp_path/'closed_workers').mkdir()
    (tmp_path/'closed_workers/operational_g0_challenger.json').write_text('{}')
    env=SimpleNamespace(guard=lambda:None,remaining=lambda:remaining,waited=60.)
    components=dict(initialization=1.,graph=.2,closure=.1,runtime=2.)
    return env,components


def test_prefix_reproduction_gate_is_resource_only_and_has_remaining_clock(tmp_path):
    env,components=resource_fixture(tmp_path)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),patch.object(r,'reproduction_fixed_components',return_value=components):
        proof=r.ensure_reproduction_resources({'decisions':[{}]},[{'clock':clock(elapsed=.85)}],env)
    assert proof['passed'] and proof['selected_reproductions']==1
    assert proof['remaining_ram_wait_seconds']==540. and not proof['engineering_margin_used']
    assert not proof['benchmark_authorized_to_launch']


@pytest.mark.parametrize('kind',['clock','final_bytes','total_bytes','prior_gate'])
def test_reproduction_gate_rejects_before_inference(tmp_path,kind):
    env,components=resource_fixture(tmp_path,remaining=1. if kind=='clock' else 3600.)
    gate=dict(resource_passed=kind!='prior_gate',projected_final_seconds=1.,
              projected_final_evidence_bytes=c.FINAL_RESERVE_BYTES if kind=='final_bytes' else 0)
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path),\
        patch.object(r,'reproduction_fixed_components',return_value=components),\
        patch.object(r,'bytes_used',return_value=c.MAX_EVIDENCE_BYTES if kind=='total_bytes' else 100),\
        patch.object(r,'reproduce_decision') as inference:
        with pytest.raises(ValueError,match='no final inference'):
            r.ensure_reproduction_resources({'decisions':[{}]},[{'clock':clock()}],env,gate)
        inference.assert_not_called()
    assert not json.loads((tmp_path/'reproduction_resource_gate.json').read_text())['passed']


def test_empty_final_sample_requires_no_inference_forecast():
    assert r.ensure_reproduction_resources({'decisions':[]},[],None) is None


def test_live_transition_error_retains_original_clock_before_rejection(tmp_path):
    env=SimpleNamespace(guard=lambda:None);state=SimpleNamespace(terminal=False,current_player=0)
    spec=dict(stage='operational',pair_index=0,game_index=0,challenger_seat=0,seed_decimal='1',
              challenger_mode='research',opponent_mode='frozen',challenger_quota=64,opponent_quota=64)
    row=dict(clock=clock(),completed=True,route='research',player_role='challenger',transition_error='invalid successor',after=None)
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(schedule,'validate_spec'),\
        patch.object(r.rules,'CorrectedEnv',return_value=SimpleNamespace(initial_state=lambda:state)),\
        patch.object(r.rules,'serialize_snapshot',return_value={}),patch.object(r,'decode_training_snapshot'),\
        patch.object(r,'make_worker',return_value=SimpleNamespace()),patch.object(r,'decision',return_value=row),\
        patch.object(r,'close_and_audit',return_value={}):
        with pytest.raises(ValueError,match='Saved transition failure'):r.play_game(spec,env,{})
    stored=list(tmp_path.rglob('moves/0000.json'))
    assert len(stored)==1 and json.loads(stored[0].read_text())['clock']==row['clock']


def test_final_partial_game_gate_precedes_reproduction_dispatch(tmp_path):
    spec=dict(stage='operational',pair_index=0,game_index=0,challenger_seat=0,seed_decimal='1',
              challenger_mode='research',opponent_mode='frozen',challenger_quota=64,opponent_quota=64)
    game=dict(spec,reason='natural',plies=1)
    row=dict(stage='operational',pair_index=0,game_index=0,ply=0,player_role='challenger',completed=True,clock=clock())
    timing={'passed':True};qualified=dict(qualified=True,requests=0,timing_gate=timing)
    env=SimpleNamespace(guard=lambda:None)
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path):
        folder=r.game_folder(spec);c.save(folder/'initial.json',dict(spec,initial={}))
        c.save(folder/'result.json',game);c.save(folder/'moves/0000.json',row);c.save(folder/'online_replay.json',{})
        c.save(tmp_path/'qualification_summary.json',qualified)
        with patch.object(schedule,'make_schedule',return_value=[spec]),patch.object(r,'qualification_gate',return_value=timing),\
            patch.object(a,'validate_stopped_prefix',return_value={}),\
            patch.object(r.rules,'CorrectedEnv',return_value=SimpleNamespace(initial_state=lambda:{})),\
            patch.object(r.rules,'serialize_snapshot',return_value={}),patch.object(r,'verify_saved_decision'),\
            patch.object(a,'audit_game',return_value={}),patch.object(a,'performance',return_value={}),\
            patch.object(a,'summarize_boundary_rows',return_value={}),\
            patch.object(a,'select_replay_sample',return_value={'decisions':[row]}),\
            patch.object(r,'ensure_reproduction_resources',side_effect=ValueError('no final inference')) as resource,\
            patch.object(r,'reproduce_decision') as inference:
            with pytest.raises(ValueError,match='no final inference'):
                r.final_checks(env,{},[spec],None,[game],None,qualified)
        resource.assert_called_once();inference.assert_not_called()
