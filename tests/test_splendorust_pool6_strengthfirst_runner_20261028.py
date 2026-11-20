"""Model-free research-clock, continuation, exact evidence and resource checks."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from tools.splendorust_pool6_strengthfirst_20261028 import runner as r, audit as a, common as c, schedule
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def clock(status='ok', elapsed=1.2):
    good=status=='ok'
    return dict(status=status,clock='perf_counter_ns',request_id='request-00000001',deadline_seconds=10.,
        elapsed_seconds=elapsed if good else 10.01,receipt_elapsed_seconds=elapsed if good else None,
        received_before_deadline=good,accepted=good,result={} if good else None)


def gate_args():
    return dict(observed_seconds=6*60.,observed_gross_seconds=6*300.,observed_variable_bytes=6*9*c.MiB,
        independent_seconds=6*4.,fixed_final_seconds=900.,final_evidence_bytes=600*c.MiB,
        maximum_move_seconds=2.5,all_natural=True,current_evidence_bytes=100*c.MiB,
        current_final_evidence_bytes=300*c.MiB,fixed_collection_seconds=650.)


def test_original_request_reader_and_placement_are_retained():
    assert r.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert r.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert r.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert r.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader
    here=Path(r.__file__).parent/'placement.py'
    old=c.ROOT/'tools/splendorust_pool6_live_20261027/placement.py'
    assert here.read_bytes()==old.read_bytes()


@pytest.mark.parametrize('elapsed',[.7,.9,1.,1.2,5.,10.])
def test_completed_responses_are_not_rejected_at_old_thresholds(elapsed):
    receipt=clock(elapsed=elapsed)
    assert a.check_clock(receipt)==elapsed
    with pytest.raises(ValueError,match='never a forfeit'):a.classify_failure(receipt)
    rows=[dict(completed=True,clock=deepcopy(receipt)) for _ in range(90)]
    gate=r.qualification_gate(rows)
    assert gate['passed'] and not gate['timing_used_for_admission']
    values=gate_args();values['maximum_move_seconds']=elapsed
    assert r.operational_gate(**values)['passed']


def test_actual_clock_cannot_be_relabelled_or_extended():
    receipt=clock(elapsed=1.2);receipt['deadline_seconds']=1.
    with pytest.raises(ValueError):a.check_clock(receipt)
    receipt=clock(elapsed=10.000001)
    with pytest.raises(ValueError):a.check_clock(receipt)


def test_only_authentic_research_hard_guards_are_administrative_forfeits():
    outer=clock('timeout')
    assert a.classify_failure(outer)=='research_outer_hard_guard'
    inner=clock('worker_error');inner.update(detail='STRENGTH_INTERNAL_DEADLINE: 9s internal hard guard exceeded',
        receipt_elapsed_seconds=9.01,received_before_deadline=True,elapsed_seconds=9.02)
    assert a.classify_failure(inner)=='research_internal_hard_guard'
    assert r.deadline_failure(inner,'research')
    for mutate in (dict(elapsed_seconds=9.5),dict(receipt_elapsed_seconds=2.,received_before_deadline=True)):
        bad=deepcopy(outer);bad.update(mutate)
        with pytest.raises(ValueError):a.classify_failure(bad)


@pytest.mark.parametrize('status,detail',[('resource_guard','RAM'),('worker_error','model failed'),
    ('worker_error','ONE_SECOND_INTERNAL_DEADLINE: historical .9s guard'),
    ('worker_error','STRENGTH_INTERNAL_DEADLINE: resource prerequisite failed'),
    ('illegal','STRENGTHFIRST_AUDIT: incorrect science')])
def test_infrastructure_science_or_old_deadline_cannot_score_loss(status,detail):
    value=clock(status);value['detail']=detail
    if status=='illegal':value.update(result={'retained':'rejected'},received_before_deadline=True,receipt_elapsed_seconds=.4)
    with pytest.raises(ValueError):a.classify_failure(value)


def test_ops_stops_first_hard_fault_but_assessment_continues_next_distinct_game():
    specs=schedule.stage_schedule('main')[:3]
    calls=[]
    def play(spec,*unused):
        calls.append(spec['game_index'])
        return dict(reason='forfeit' if len(calls)==1 else 'natural'),[],0.,({'fault':True} if len(calls)==1 else None)
    with patch.object(r,'play_game',side_effect=play):
        games=r.collect_assessment(None,{},specs)
    assert calls==[0,1,2] and len(games)==3
    calls.clear()
    with patch.object(r,'play_game',side_effect=play),patch.object(r,'bytes_used',return_value=0):
        games,stop=r.operational(None,{},schedule.stage_schedule('operational'))
    assert calls==[0] and stop and len(games)==1


def test_assessment_does_not_continue_after_integrity_exception():
    with patch.object(r,'play_game',side_effect=ValueError('scientific mismatch')) as play:
        with pytest.raises(ValueError,match='scientific'):r.collect_assessment(None,{},schedule.stage_schedule('main')[:3])
    assert play.call_count==1


def test_forecast_uses_maximum_game_and_separate_bounded_ram_wait():
    measured=[dict(nonwait_collection_seconds=x,collection_seconds=x+(235 if i==4 else 0),
        independent_seconds=x/10,bytes=x*1000) for i,x in enumerate((1,1,1,1,10,1))]
    components=dict(initialization=2.,graph=.4,closure=.5,runtime=3.)
    values=r.assessment_inputs(measured,components,10000,2.,20*c.MiB,10000)
    result=r.operational_gate(**values)
    assert values['observed_seconds']==60. and values['observed_gross_seconds']==1470.
    assert values['observed_variable_bytes']==60000
    assert result['projected_collection_seconds']==10*400*1.5+606+600
    assert result['gross_game_cost_projection_seconds']==245*400*1.5+606
    assert result['ram_wait_charged_once_seconds']==600 and result['move_timing_subtracted_seconds']==0
    assert values['fixed_final_seconds']==48*1.5*(2+.4+.5+10+1)+6+300
    assert result['passed'] and result['projected_games']==400 and not result['outcomes_used']


@pytest.mark.parametrize('field,value',[
    ('observed_seconds',10000.),('observed_variable_bytes',1024*c.MiB),
    ('fixed_final_seconds',14400.),('final_evidence_bytes',4*c.GiB),
    ('current_evidence_bytes',3*c.GiB),('current_final_evidence_bytes',c.GiB),('all_natural',False)])
def test_prospective_caps_and_current_final_reserve_fail_closed(field,value):
    args=gate_args();args[field]=value
    if field=='observed_seconds':args['observed_gross_seconds']=value
    assert not r.operational_gate(**args)['passed']


def test_source_and_job_clocks_are_not_reset_or_extended():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=3600.),patch.object(r.time,'time',return_value=4500.),patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('operational');assert env.deadline==3700.
        env.begin('final');assert env.deadline==7200.
    with patch.object(r.time,'monotonic',return_value=7301.),patch.object(r.time,'time',return_value=1001.):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def profile_fixture(route='research',elapsed=1.2):
    from test_splendorust_boundary_timing_analyze import profile_row
    row=profile_row();parent,child,receipt=row['parent_profile'],row['child_profile'],row['clock']
    payload=dict(player='search',observer=row['observer'])
    child.update(backend='research_graph' if route=='research' else 'qualified_graph64',quota=64,payload_sha256=c.digest(payload))
    receipt.update(clock='perf_counter_ns',elapsed_seconds=elapsed+.05,deadline_seconds=10.,receipt_elapsed_seconds=elapsed)
    arrival=parent['request_start_ns']+round(elapsed*1e9)
    parent.update(receipt_arrival_ns=arrival,receipt_reconstructed_arrival_ns=arrival,
        capture_snapshot_ns=arrival+60_000_000,raw_arrivals_ns=[arrival])
    child['post_flush_ns']=arrival+10_000_000
    return parent,child,receipt,payload


@pytest.mark.parametrize('route',['frozen','research'])
def test_late_completed_sidecar_uses_actual_ten_second_clock(route):
    parent,child,receipt,payload=profile_fixture(route)
    original=deepcopy(receipt)
    checked=r.validate_profiles(parent,child,receipt,payload,route,worker_pid=42)
    assert checked['status']=='passed' and checked['research_deadline_seconds']==10.
    assert receipt==original and checked['accepted_science']


@pytest.mark.parametrize('field,value',[('quota',32),('payload_sha256','f'*64),('result_sha256','f'*64),
                                      ('backend','qualified_graph64'),('graph_identity_sha256','f'*64)])
def test_profile_identity_corruption_rejected(field,value):
    parent,child,receipt,payload=profile_fixture();child[field]=value
    with pytest.raises(ValueError):r.validate_profiles(parent,child,receipt,payload,'research',worker_pid=42)


def test_raw_packet_hash_cannot_be_replaced():
    packets=[r._packet_evidence(b'{"type":"error"}\n')]
    parent=dict(response_raw_sha256='0'*64,response_raw_bytes=17,response_capture_overflow=False)
    with pytest.raises(ValueError):r._bind_parent_wire(parent,packets,completed=True)


def test_rejected_complete_packet_remains_forensic_not_accepted():
    parent,child,receipt,payload=profile_fixture()
    receipt.update(status='illegal',accepted=False,detail='STRENGTHFIRST_AUDIT: numerical mismatch')
    checked=r.validate_profiles(parent,child,receipt,payload,'research',worker_pid=42)
    assert checked['rejected_response_retained'] and not checked['accepted_science']
    with pytest.raises(ValueError):a.classify_failure(receipt)


def test_failed_qualification_row_is_saved_before_stop_no_retry(tmp_path):
    from tools.splendorust_pool6_strengthfirst_20261028 import adapters
    plan={'parity_requests':[dict(index=i,route='frozen',observer={},quota=64) for i in range(3)]}
    row=dict(spec=plan['parity_requests'][0],clock=clock('timeout'),completed=False,route='frozen')
    env=SimpleNamespace(guard=lambda:None)
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'make_worker',return_value=SimpleNamespace(route='frozen')),\
        patch.object(adapters,'actual_entrypoint',return_value={'status':'ok'}),patch.object(r,'ordinary',return_value=row) as ordinary,\
        patch.object(r,'verify_saved_record'),patch.object(r,'close_and_audit'):
        result=r.qualify(env,{},plan)
    assert not result['qualified'] and ordinary.call_count==1
    assert json.loads((tmp_path/'qualification_stop.json').read_text())['clock']==row['clock']
    assert (tmp_path/'qualification_rows/0000.json').exists() and not (tmp_path/'qualification_rows/0001.json').exists()


def test_two_qualification_workers_and_interleaved_immediate_exact_validation(tmp_path):
    from tools.splendorust_pool6_strengthfirst_20261028 import adapters,qualification
    plan={'parity_requests':[dict(index=i,route='research' if i%2 else 'frozen',observer={},quota=64) for i in range(90)]}
    events=[];env=SimpleNamespace(guard=lambda:None)
    def make(name,*args,route):events.append(('make',route));return SimpleNamespace(route=route)
    def ordinary(child,spec,*args):
        events.append(('request',spec['index'],child.route))
        return dict(spec=spec,clock=clock(elapsed=1.2),completed=True,route=child.route)
    def exact(row,spec,cache,**kwargs):
        assert (tmp_path/'qualification_rows'/f"{spec['index']:04d}.json").exists()
        events.append(('exact',spec['index']))
    with patch.object(c,'OUT',tmp_path),patch.object(c,'ROOT',tmp_path),patch.object(r,'make_worker',side_effect=make),\
        patch.object(adapters,'actual_entrypoint',return_value={'status':'ok'}),patch.object(r,'ordinary',side_effect=ordinary),\
        patch.object(r,'verify_saved_record'),patch.object(r,'close_and_audit'),\
        patch.object(qualification,'validate_admission_row',side_effect=exact),patch.object(qualification,'validate_parity_rows',return_value={'status':'passed'}):
        result=r.qualify(env,{},plan)
    assert result['qualified'] and events[:2]==[('make','frozen'),('make','research')]
    assert events[2:]==[event for i in range(90) for event in [('request',i,'research' if i%2 else 'frozen'),('exact',i)]]


def test_late_evidence_refresh_is_not_lost(tmp_path):
    env=r.Envelope(dict(monotonic_origin=0.,unix_origin=0.));env.evidence={'workers':{}}
    folder=tmp_path/'worker';folder.mkdir();(folder/'cleanup.json').write_bytes(b'abc')
    env.register_worker(folder);env.close_worker(folder);assert env.evidence['workers'][folder]==3
    (folder/'late_sidecar.json').write_bytes(b'original-clock');env.close_worker(folder)
    assert env.evidence['workers'][folder]==17


def test_final_reproductions_cannot_precede_resource_gate():
    tree=ast.parse(Path(r.__file__).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='final_checks')
    calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    first=lambda name:min(n.lineno for n in calls if n.func.id==name)
    assert first('ensure_reproduction_resources')<first('reproduce_decision')
    assert r.ensure_reproduction_resources({'decisions':[]},[],None) is None



def test_descriptive_exceedances_use_strict_boundaries_and_completed_denominator():
    times=(.7,.700001,.9,.900001,1.,1.000001)
    rows=[]
    for role,mode in a.MODES.items():
        for elapsed in times:
            rows.append(dict(stage='main',pair_index=0,game_index=0,ply=len(rows),seat=0,player_role=role,arm='search',mode=mode,quota=64,
                placement_mode='isolated',observation_mode='profiled',completed=True,
                clock=clock(elapsed=elapsed),parent_profile={},child_profile={}))
    performance=a.performance(rows)
    with patch.object(a,'validate_boundary_row',return_value={}):
        boundaries=a.summarize_boundary_rows(rows)
    for result in list(performance['players'].values())+boundaries['groups']:
        assert result['completed_decisions']==6
        assert result['completed_over_700ms']==5
        assert result['completed_over_900ms']==3
        assert result['completed_over_one_second']==1
        assert result['completed_at_or_above_one_second']==2
    assert performance['descriptive_timing_thresholds_seconds']==boundaries['descriptive_timing_thresholds_seconds']==[.7,.9,1.]
    assert performance['descriptive_thresholds_change_completion_or_stopping'] is False
    assert boundaries['descriptive_thresholds_change_completion_or_stopping'] is False
