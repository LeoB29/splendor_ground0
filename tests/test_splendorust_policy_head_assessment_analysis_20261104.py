"""Exact paired arithmetic, guard-loss sensitivity and independent inference."""
from copy import deepcopy
import numpy as np
import pytest
from tools.splendorust_policy_head_20261104.assessment import analysis as stats, design as schedule


def games(outcome=lambda spec: 1):
    result=[]
    for spec in schedule.make_schedule('main'):
        points=outcome(spec);seat=spec['challenger_seat']
        result.append(dict(spec,complete=True,reason='natural',
                           winner=None if points==1 else seat if points==2 else 1-seat))
    return result


def admin(row,loser,failure='research_outer_hard_guard'):
    row.update(reason='forfeit',loser=loser,winner=1-loser,failed_seat=loser,
               failed_role='challenger' if loser==row['challenger_seat'] else 'opponent',
               administrative_forfeit=True,failure_policy=stats.FAILURE_POLICY,failure=failure)


def test_full50000_paired_inference_and_independent_recomputation():
    rows=games(lambda spec:2 if spec['pair_index']<120 else 0)
    admin(rows[250],1-rows[250]['challenger_seat'])
    result=stats.analyze_main(rows,True)
    assert result['official']['mean_score']==241/400 and result['conservative']['mean_score']==.6
    assert result['useful_gain_established'] and result['candidate_simulations']==128
    assert result['reference_simulations']==128 and result['candidate_cpuct']==result['reference_cpuct']==.4
    assert not result['automatic_promotion'] and not result['other_study_games_pooled']
    assert not result['one_second_deployability_established'] and not result['operational_games_pooled']
    assert result['bootstrap']['replicates']==50000 and result['bootstrap']['seed']==20261104128
    proof=stats.verify_main(list(reversed(rows)),result)
    assert proof['independent_python_integer_sums'] and proof['independent_linear_percentiles']
    assert proof['independent_conservative_sensitivity']


@pytest.mark.parametrize('winning_pairs,passes',[(109,False),(110,True),(111,True)])
def test_integer55percent_threshold(winning_pairs,passes):
    score=stats._score(stats.validated_games(games(lambda spec:2 if spec['pair_index']<winning_pairs else 0)))
    assert stats._useful_gates(score)['score_at_least_55pct'] is passes


def test_each_seat_required_despite_large_aggregate():
    rows=games(lambda spec:2 if spec['challenger_seat']==0 or spec['pair_index']<80 else 0)
    result=stats.analyze_main(rows,True)
    assert result['official']['mean_score']==.7
    assert result['official']['by_candidate_seat']=={'0':1.,'1':.4}
    assert not result['useful_gain_established']


def test_favorable_reference_guard_losses_cannot_establish_gain():
    rows=games(lambda spec:2 if spec['pair_index']<100 else 0)
    for row in rows[300:]:admin(row,1-row['challenger_seat'])
    result=stats.analyze_main(rows,True)
    assert result['official']['useful_gain'] and result['conservative']['mean_score']==.5
    assert not result['useful_gain_established']
    assert result['classification']=='official_useful_gain_not_robust_to_guard_forfeits'
    assert result['administrative_sensitivity']['natural_only_strength_estimate'] is False


def test_candidate_guards_remain_losses_and_never_drop_denominator():
    rows=games(lambda spec:2)
    for row in rows[:2]:admin(row,row['challenger_seat'],'research_internal_hard_guard')
    result=stats.analyze_main(rows,True)
    assert result['official']['games']==400 and result['official']['losses']==2
    assert result['official']['mean_score']==result['conservative']['mean_score']==398/400


@pytest.mark.parametrize('mutation',['missing','extra','duplicate','wrong_stage','wrong_candidate',
    'wrong_seed','quota512','bool_winner','incomplete','unofficial','derived_score'])
def test_complete_fixed_sample_and_outcome_rejection(mutation):
    rows=games()
    if mutation=='missing':rows.pop()
    elif mutation=='extra':rows.append(deepcopy(rows[0]))
    elif mutation=='duplicate':rows[-1]=deepcopy(rows[0])
    elif mutation=='wrong_stage':rows[0]['stage']='operational'
    elif mutation=='wrong_candidate':rows[0]['candidate']='high'
    elif mutation=='wrong_seed':rows[0]['seed_decimal']='1'
    elif mutation=='quota512':rows[0]['challenger_quota']=512
    elif mutation=='bool_winner':rows[0]['winner']=True
    elif mutation=='incomplete':rows[0]['complete']=False
    elif mutation=='unofficial':rows[0]['reason']='cutoff'
    elif mutation=='derived_score':rows[0]['candidate_score_halfpoints']=2
    with pytest.raises(ValueError):stats.analyze_main(rows,True)


@pytest.mark.parametrize('field,value',[('loser',True),('winner',0),('failed_role','opponent'),
    ('administrative_forfeit',False),('failure','timeout'),('failure','source_integrity_failure'),
    ('failure_policy','one_second_timeout_loss'),('failed_seat',1)])
def test_only_coherent_classified_guards_enter_scores(field,value):
    rows=games();admin(rows[0],0);rows[0][field]=value
    with pytest.raises(ValueError):stats.analyze_main(rows,True)


@pytest.mark.parametrize('gate',[False,None,1])
def test_evidence_gate_is_not_truthiness(gate):
    with pytest.raises(ValueError):stats.analyze_main(games(),gate)


def test_degenerate_bootstrap_safeguard_and_draws():
    rows=games(lambda spec:2 if spec['game_index']%2==0 else 1)
    result=stats.analyze_main(rows,True)
    assert result['official']['bootstrap_degenerate'] and result['official']['paired_ci95']==[.75,.75]
    assert result['official']['draws']==200 and result['useful_gain_established']
    # The inherited safeguard still blocks a point-mass interval unsupported by
    # the bounded-pair concentration bound; this synthetic score is a unit case.
    score=dict(stats._score(stats.validated_games(games())),numerator=440,mean_score=.55)
    fake=stats._main_score(score,np.full(stats.REPLICATES,440,dtype=np.int64))
    assert fake['paired_ci95']==[.55,.55] and not fake['useful_gain_criteria']['degenerate_interval_safeguard']


@pytest.mark.parametrize('path,value',[(('official','numerator'),1),(('official','paired_ci95'),[0.,1.]),
    (('official','integer_totals_sha256'),'0'*64),(('bootstrap','indices_sha256'),'0'*64),
    (('candidate_simulations',),512),(('useful_gain_established',),False)])
def test_independent_reader_rejects_changed_report(monkeypatch,path,value):
    # Shortened synthetic bootstrap isolates corruption paths; the full50000
    # unmodified production construction is independently exercised above.
    monkeypatch.setattr(stats,'REPLICATES',1000)
    rows=games(lambda spec:2 if spec['pair_index']<140 else 0);result=stats.analyze_main(rows,True)
    target=result
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    with pytest.raises(ValueError):stats.verify_main(rows,result)


def test_no_timing_or_q_dependence_and_boundary_abort(monkeypatch):
    monkeypatch.setattr(stats,'REPLICATES',1000)
    rows=games();first=stats.analyze_main(rows,True)
    for row in rows:row.update(timing_seconds=999,model_q_estimate=999)
    assert stats.analyze_main(rows,True)==first
    class Stop(Exception):pass
    def stop():raise Stop()
    with pytest.raises(Stop):stats.analyze_main(rows,True,boundary=stop)


@pytest.fixture(autouse=True)
def pilot_scope(monkeypatch):
    monkeypatch.setattr(stats.c,'SCOPE','pilot');monkeypatch.setattr(stats.c,'MAIN_GAMES',400)
    monkeypatch.setattr(schedule,'PAIRS',{'operational':3,'main':200})
    monkeypatch.setattr(schedule,'DOMAIN','splendorust_policy_head_20261104_pilot_assessment_v1')

def test_qualification_cannot_produce_strength_even_from_invented400games(monkeypatch):
    rows=games()
    monkeypatch.setattr(stats.c,'SCOPE','qualification');monkeypatch.setattr(stats.c,'MAIN_GAMES',0)
    monkeypatch.setattr(schedule,'PAIRS',{'operational':3,'main':0})
    with pytest.raises(ValueError,match='no main strength'):stats.analyze_main(rows,True)
    with pytest.raises(ValueError,match='no main strength'):stats.analyze_main([],True)
