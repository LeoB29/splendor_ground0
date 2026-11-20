"""Model-free single-clock sequencing, exact counts and failure/resource gates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from tools.splendorust_search128_strength_20261028 import runner as r, audit as a, common as c, schedule, transport as t
from tools.local_player_qualification_128.clock import PersistentWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker


def clock(status='ok', elapsed=1.2):
    good=status=='ok'
    return dict(status=status,clock='perf_counter_ns',request_id='request-00000001',deadline_seconds=10.,
        elapsed_seconds=elapsed if good else 10.01,receipt_elapsed_seconds=elapsed if good else None,
        received_before_deadline=good,accepted=good,result={} if good else None)


def gate_args():
    return dict(observed_seconds=360.,observed_gross_seconds=1800.,observed_variable_bytes=54*c.MiB,
        independent_seconds=24.,fixed_final_seconds=900.,final_evidence_bytes=600*c.MiB,
        maximum_move_seconds=2.5,all_natural=True,current_evidence_bytes=100*c.MiB,
        current_final_evidence_bytes=300*c.MiB,fixed_collection_seconds=650.,
        available_collection_seconds=65000.,available_global_seconds=79000.,remaining_ram_wait_seconds=500.)


def test_original_request_readers_and_placement_unchanged():
    assert t.PlacedProfiledStudyWorker.request is ProfiledStudyWorker.request
    assert t.PlacedProfiledStudyWorker._receive is ProfiledStudyWorker._receive
    assert t.PlacedProfiledStudyWorker._stdout_reader is PersistentWorker._stdout_reader
    assert t.PlacedProfiledStudyWorker._stderr_reader is PersistentWorker._stderr_reader
    assert (Path(r.__file__).parent/'placement.py').read_bytes()==(c.ROOT/'tools/splendorust_pool6_strengthfirst_20261028/placement.py').read_bytes()


@pytest.mark.parametrize('elapsed',[.7,.9,1.,1.2,5.,10.])
def test_old_thresholds_never_reject_correct_completed_response(elapsed):
    receipt=clock(elapsed=elapsed)
    assert a.check_clock(receipt)==elapsed
    with pytest.raises(ValueError):a.classify_failure(receipt)
    rows=[dict(completed=True,clock=deepcopy(receipt)) for _ in range(84)]
    assert r.qualification_gate(rows,expected_requests=84)['passed']
    args=gate_args();args['maximum_move_seconds']=elapsed
    assert r.operational_gate(**args)['passed']


def test_ordinary_clock_is_exact_unextended_ten_seconds():
    value=clock();value['deadline_seconds']=1.
    with pytest.raises(ValueError):a.check_clock(value)
    with pytest.raises(ValueError):a.check_clock(clock(elapsed=10.000001))


def test_only_authentic_hard_guards_can_forfeit():
    assert a.classify_failure(clock('timeout'))=='research_outer_hard_guard'
    inner=clock('worker_error');inner.update(detail='STRENGTH_INTERNAL_DEADLINE: 9s guard',
        received_before_deadline=True,receipt_elapsed_seconds=9.01,elapsed_seconds=9.02)
    assert a.classify_failure(inner)=='research_internal_hard_guard'
    for status,detail in [('resource_guard','RAM'),('worker_error','failed'),('illegal','rejectedscience'),
            ('worker_error','STRENGTH_INTERNAL_DEADLINE: resource prerequisite failed')]:
        value=clock(status);value['detail']=detail
        with pytest.raises(ValueError):a.classify_failure(value)


def test_main_hard_fault_continues_distinct_schedule_ops_stops():
    calls=[]
    def play(spec,*unused):
        calls.append((spec['stage'],spec['game_index']))
        return dict(stage=spec['stage'],reason='forfeit' if len(calls)==1 else 'natural'),[],0.,({'fault':True} if len(calls)==1 else None)
    with patch.object(r,'play_game',side_effect=play):
        games=r.collect_assessment(None,{},schedule.stage_schedule('main')[:3])
    assert calls==[('main',0),('main',1),('main',2)] and len(games)==3
    calls.clear()
    with patch.object(r,'play_game',side_effect=play),patch.object(r,'bytes_used',return_value=0):
        games,stop=r.operational(None,{},schedule.stage_schedule('operational'))
    assert calls==[('operational',0)] and stop


def test_integrity_failure_never_continues_or_becomes_loss():
    with patch.object(r,'play_game',side_effect=ValueError('scientific mismatch')) as play:
        with pytest.raises(ValueError):r.collect_assessment(None,{},schedule.stage_schedule('main')[:3])
    assert play.call_count==1


def test_one_origin_and_collection_final_cutoffs_do_not_reset():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.))
    with patch.object(r.time,'monotonic',return_value=1000.),patch.object(r.time,'time',return_value=1900.),patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('qualification');assert env.deadline==4600.
        env.begin('operational');assert env.deadline==4600.
        env.begin('main');assert env.deadline==72100.
    with patch.object(r.time,'monotonic',return_value=75000.),patch.object(r.time,'time',return_value=75900.),patch.object(env,'guard'),patch.object(c,'save'):
        env.begin('final');assert env.deadline==86500. and env.wall_deadline==87400.
    assert env.start==100. and env.wall_start==1000.
    with patch.object(r.time,'monotonic',return_value=86501.),patch.object(r.time,'time',return_value=2000.):
        with pytest.raises(ValueError,match='deadline'):env.guard()


def test_forecast_uses_max_gross_recorded_wait_once_and_reserves_review():
    measured=[dict(nonwait_collection_seconds=x,collection_seconds=x+(235 if i==4 else 0),
        independent_seconds=x/10,bytes=x*1000) for i,x in enumerate((1,1,1,1,10,1))]
    components=dict(initialization=2.,graph=.4,closure=.5,runtime=3.)
    budget=dict(available_collection_seconds=65000.,available_global_seconds=79000.,remaining_ram_wait_seconds=500.)
    values=r.assessment_inputs(measured,components,10000,5.,20*c.MiB,10000,budget)
    result=r.operational_gate(**values)
    assert values['observed_seconds']==60. and values['observed_gross_seconds']==1470.
    assert result['projected_collection_seconds']==10*400*1.5+606+500
    assert result['gross_game_cost_projection_seconds']==245*400*1.5+606
    assert result['ram_wait_charged_once_seconds']==500 and result['move_timing_subtracted_seconds']==0
    assert result['independent_review_reserved_seconds']==3600
    assert result['assessment_projected_final_seconds']>3600
    assert result['passed'] and not result['timing_used_for_admission']


@pytest.mark.parametrize('field,value',[
    ('available_collection_seconds',100.),('available_global_seconds',1000.),
    ('observed_variable_bytes',1024*c.MiB),('fixed_final_seconds',14400.),
    ('final_evidence_bytes',4*c.GiB),('current_evidence_bytes',12*c.GiB),
    ('current_final_evidence_bytes',4*c.GiB),('all_natural',False)])
def test_independent_time_storage_and_reserve_caps_fail_closed(field,value):
    args=gate_args();args[field]=value
    assert not r.operational_gate(**args)['passed']


def test_forecast_absolute_clock_budget_is_reconstructed_not_refreshed():
    env=r.Envelope(dict(monotonic_origin=100.,unix_origin=1000.));env.waited=80.
    with patch.object(r.time,'monotonic',return_value=1100.),patch.object(r.time,'time',return_value=2000.):
        value=r.clock_budget(env)
    assert value['available']['available_collection_seconds']==71000.
    env.waited=150.
    with patch.object(r.time,'monotonic',return_value=2100.),patch.object(r.time,'time',return_value=3000.):
        assert r.verify_clock_budget(value,env)==value['available']
        bad=deepcopy(value);bad['available']['available_global_seconds']+=1
        with pytest.raises(ValueError):r.verify_clock_budget(bad,env)


def test_fixed_workload_and_roles():
    assert c.PHASE=='strength' and c.QUAL_OUT==c.OUT
    assert c.QUALIFICATION_DECISIONS==203 and c.QUALIFICATION_ORDINARY_REQUESTS==84
    assert c.MAX_WORKERS==2+12+800+48 and c.MAX_GRAPH_BUILDS==862
    specs=schedule.make_schedule();assert len(specs)==406
    assert [x['stage'] for x in specs[:6]]==['operational']*6
    assert len(specs[6:])==400 and all(x['challenger_quota']==128 and x['opponent_quota']==64 for x in specs)


def test_prefix_allows_main_forfeits_but_no_ops_continuation():
    specs=schedule.make_schedule()
    games=[dict(**s,complete=True,reason='natural',plies=1) for s in specs[:8]]
    games[6]['reason']='forfeit'
    assert a.validate_stopped_prefix(specs,games)['verified_games']==8
    games[0]['reason']='forfeit'
    with pytest.raises(ValueError):a.validate_stopped_prefix(specs,games)


def test_source_status_preservation_is_finite_and_rejects_relabelling(tmp_path):
    original=tmp_path/'prior.json';original.write_text('{"status":"incomplete"}')
    with patch.object(c,'ROOT',tmp_path),patch.object(c,'OUT',tmp_path/'new'):
        item=c.artifact(original);manifest=dict(scope='finite_referenced_dependencies',artifacts=[item],
            historical_statuses=[dict(path='prior.json',field='status',expected='incomplete')],
            required_absent=['never_accepted.json'],earlier_failures_preserved=True)
        target=tmp_path/'manifest.json';target.write_text(c.canonical(manifest))
        freeze=dict(prior_preservation=c.artifact(target),inputs=[item])
        env=SimpleNamespace(guard=lambda:None)
        assert r.verify_prior_preservation(freeze,env)['path'].endswith('prior_preservation_after.json')
        (tmp_path/'never_accepted.json').write_text('{}')
        with pytest.raises(ValueError,match='relabelled'):r.verify_prior_preservation(freeze,env)


def test_zero_reproduction_prefix_never_calls_model_or_budget_inventory():
    with patch.object(r,'reproduction_fixed_components',side_effect=AssertionError('no inference preparation')):
        assert r.ensure_reproduction_resources({'decisions':[]},[],None) is None


@pytest.mark.parametrize('qualification_ok,resource_ok,expected_stages,expected_games',[
    (True,True,['qualification','operational','main','final'],406),
    (True,False,['qualification','operational','final'],6),
    (False,True,['qualification','final'],0)])
def test_full_controller_crosses_stages_once_without_new_claim_or_fallback(tmp_path,qualification_ok,resource_ok,expected_stages,expected_games):
    out=tmp_path/'out';out.mkdir()
    freeze={};claim=dict(freeze_sha256='f'*64,monotonic_origin=100.,unix_origin=1000.)
    saved={str(out/'freeze.json'):freeze,str(out/'watchdog_claim.json'):claim}
    stages=[];envs=[];final_args=[]
    class E:
        def __init__(self,actual):
            assert actual==claim
            envs.append(self);self.resident={};self.start=100.;self.wall_start=1000.;self.deadline=86500.;self.wall_deadline=87400.;self.stage=None
        def enable_evidence_accounting(self):pass
        def begin(self,stage):self.stage=stage;stages.append(stage)
        def admit(self):pass
        def finish(self):pass
    def save(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}');saved[str(path)]=deepcopy(value)
    def operations(env,f,ss,games):
        assert env is envs[0] and len(ss)==6
        save(out/'assessment_gate.json',{'passed':resource_ok})
        return [dict(**s,reason='natural') for s in ss],None
    def main(env,f,ss,games):
        assert env is envs[0] and len(ss)==400 and all(s['stage']=='main' for s in ss)
        return games+[dict(**s,reason='natural') for s in ss]
    def final(env,f,ss,plan,games,stop,qualified):
        final_args.append((env,len(games),stop,qualified))
        return {'complete_planned_workload':len(games)==406}
    with patch.object(c,'OUT',out),patch.object(c,'read',side_effect=lambda p:saved[str(p)]),patch.object(c,'save',side_effect=save),patch.object(c,'file_hash',return_value='f'*64),\
        patch.object(r,'Envelope',E),patch.object(r,'verify_runtime_sources',return_value={}),patch.object(r,'verify_runtime'),patch.object(r,'switch_placement'),\
        patch.object(r,'load_inputs',return_value=(schedule.make_schedule(),{'parity_requests':list(range(84))})),\
        patch.object(r,'qualify',return_value=dict(qualified=qualification_ok,failure=None)),patch.object(r,'operational',side_effect=operations),\
        patch.object(r,'collect_assessment',side_effect=main),patch.object(r,'final_checks',side_effect=final):
        assert r.run('f'*64)==0
    assert stages==expected_stages and len(envs)==1 and final_args[0][1]==expected_games
    completion=saved[str(out/'completion.json')]
    assert completion['shared_job_origin']==claim and completion['independent_acceptance_pending'] is True
    assert completion['independent_acceptance_deadline_monotonic']==86500.
    assert completion['status']==('completed_verified' if expected_games==406 else 'not_admitted_verified')


def test_quota_and_transition_failure_are_preserved_after_original_clock():
    state=SimpleNamespace(current_player=1)
    before={'snapshot':True};observer={'observer':True}
    original=dict(completed=True,clock=clock(),action={'bad':'transition'},trace={'raw':'retained'})
    child=SimpleNamespace(route='research')
    with patch.object(r.rules,'serialize_snapshot',return_value=before),patch.object(r.rules,'observation_payload',return_value=observer),\
        patch.object(r,'decode_training_snapshot'),patch.object(r.ledger,'check_observer'),patch.object(r,'ordinary',return_value=deepcopy(original)) as ordinary,\
        patch.object(r.rules,'deserialize_action',side_effect=ValueError('bad transition')):
        row=r.decision(child,state,None,'challenger',{})
    assert row['quota']==128 and row['after'] is None and 'bad transition' in row['transition_error']
    assert row['clock']==original['clock'] and row['action']==original['action']
    assert ordinary.call_args.args[1]=={'route':'research','quota':128}
