"""Adversarial model-free tests for paired sequential decision integrity."""
from copy import deepcopy
import math
import pytest
from . import sequential as s
from . import statistical_qualification as q


def calibrated():
    # In-memory fixture only. Production freeze binds the actual qualification
    # artifact, whose full inventory is checked by verify_qualification.
    return dict(schema='sixworld512_statistical_qualification_v2',status='passed',
        configuration=s.configuration(),configuration_sha256=s.digest(s.configuration()),
        acceptance=dict(null_family_simultaneous_upper_max=.024,all_null_directions_pass=True,
            power_simultaneous_lower_min=.79,both_power_cases_pass=True,
            calibration_validation_seeds_disjoint=True,production_scalar_vector_agreement=True))


def test_fixed_contract_and_historical_gate_removed():
    c=s.configuration()
    assert c['looks_games']==list(range(400,3601,400))
    assert c['teacher_threshold']==.48 and c['player_threshold']==.5
    assert c['observed_55_percent_gate'] is False and c['lower_tail_alpha']==.025
    assert c['distribution_free_guarantee'] is False


@pytest.mark.parametrize('scores,expected', [([1,1],1.),([1,.5],.75),([1,0],.5),([.5,.5],.5),([.5,0],.25),([0,0],0.)])
def test_draw_and_pair_credit(scores,expected):assert s.pair_score(scores)==expected


@pytest.mark.parametrize('scores', [[1],[1,0,1],[True,0],[.25,0],[math.nan,0]])
def test_incomplete_or_non_game_pair_rejected(scores):
    with pytest.raises(ValueError):s.pair_score(scores)


@pytest.mark.parametrize('score',[True,math.nan,math.inf,-.25,.1,1.25,'0.5'])
def test_illegal_pair_scores_rejected(score):
    with pytest.raises(ValueError):s.analyze_pairs([score]*200,calibrated())


@pytest.mark.parametrize('n',[0,1,199,201,399,1801,2000])
def test_no_unplanned_peek_or_extension(n):
    with pytest.raises(ValueError):s.analyze_pairs([.5]*n,calibrated())


def test_one_pair_is_unit_and_exact_integer_moments():
    r=s.analyze_pairs([0,1]*100,calibrated())
    assert r['look']['score']==.5
    assert r['look']['sample_pair_variance']==50/199
    assert r['look']['pair_numerator_sum']==400
    assert r['look']['pair_numerator_square_sum']==1600
    assert r['games']==400 and r['pairs']==200
    assert s.verify_analysis([0,1]*100,calibrated(),r)['status']=='passed'


def test_degenerate_pairs_keep_positive_uncertainty():
    r=s.analyze_pairs([.5]*200,calibrated())['look']
    assert r['sample_pair_variance']==0 and r['variance_floor_applied']
    assert r['lower']<.5<r['upper'] and not r['superiority_pass']
    assert not r['teacher_pass_this_look']


def test_superiority_stops_and_never_extends():
    r=s.analyze_pairs([1]*200,calibrated())
    assert r['stop'] and r['classification']=='superior' and r['player_eligible']
    with pytest.raises(ValueError,match='after an earlier'):s.analyze_pairs([1]*400,calibrated())


def test_material_harm_stops_and_no_teacher_permission():
    r=s.analyze_pairs([0]*200,calibrated())
    assert r['stop'] and r['classification']=='materially_worse'
    assert not r['teacher_eligible'] and not r['player_eligible']


def test_teacher_certificate_can_be_carried_without_player_promotion():
    first=[.75]*72+[.5]*128
    second=[.5]*160+[.25]*40
    r=s.analyze_pairs(first+second,calibrated())
    assert r['looks'][0]['teacher_pass_this_look'] and not r['looks'][0]['superiority_pass']
    assert not r['look']['teacher_pass_this_look'] and r['look']['teacher_certified_ever']
    assert not r['player_eligible'] and r['teacher_eligible'] and not r['stop']
    assert s.verify_analysis(first+second,calibrated(),r)['status']=='passed'


def test_harm_overrides_earlier_teacher_certificate():
    r=s._look([0]*400,True)
    assert r['teacher_certified_ever'] and r['material_harm_pass']
    assert not r['teacher_eligible'] and r['classification']=='materially_worse'


def test_futility_protects_unfinished_teacher_question(monkeypatch):
    monkeypatch.setattr(s,'_conditional_power',lambda mean,var,n,threshold,future,skew:.01 if threshold==.5 else .5)
    r=s._look([2]*400,False)
    assert not r['superiority_pass'] and not r['teacher_pass_this_look']
    assert not r['futility_pass'] and not r['stop']


def test_both_goals_futile_is_unresolved_not_noninferior(monkeypatch):
    monkeypatch.setattr(s,'_conditional_power',lambda *args:.01)
    r=s._look([2]*400,False)
    assert r['futility_pass'] and r['stop'] and r['classification']=='unresolved'
    assert not r['teacher_eligible']


def test_teacher_certificate_plus_player_futility_can_stop(monkeypatch):
    monkeypatch.setattr(s,'_conditional_power',lambda *args:.01)
    r=s._look([2]*400,True)
    assert r['futility_pass'] and r['classification']=='teacher_noninferior'
    assert r['teacher_eligible'] and not r['player_eligible']


def test_futility_not_before_800_games_and_strict_threshold(monkeypatch):
    monkeypatch.setattr(s,'_conditional_power',lambda *args:.0)
    assert not s._look([2]*200)['futility_pass']
    monkeypatch.setattr(s,'_conditional_power',lambda *args:.1)
    assert not s._look([2]*400)['futility_pass']


def test_exact_scientific_thresholds_are_strict(monkeypatch):
    monkeypatch.setattr(s,'_tail_criticals',lambda *args:(0.,0.,0.))
    r=s._look([2]*200)
    assert r['lower']==.5 and not r['superiority_pass']
    r=s._look([4]*96+[0]*104)
    assert r['lower']==r['upper']==.48
    assert not r['teacher_pass_this_look'] and not r['material_harm_pass']


def test_maximum_sample_can_end_teacher_only():
    scores=([1]*21+[0]*19)*45
    r=s.analyze_pairs(scores,calibrated())
    assert r['stop_reason']=='maximum_sample' and r['classification']=='teacher_noninferior'
    assert not r['player_eligible'] and r['games']==3600
    assert s.verify_analysis(scores,calibrated(),r)['status']=='passed'


@pytest.mark.parametrize('field,bad',[('status','failed'),('configuration_sha256','wrong')])
def test_failed_or_changed_calibration_rejected(field,bad):
    c=calibrated();c[field]=bad
    with pytest.raises(ValueError):s.analyze_pairs([.5]*200,c)


def test_null_or_power_qualification_failure_rejected():
    for key,value in [('null_family_simultaneous_upper_max',.02501),('power_simultaneous_lower_min',.779),
                      ('calibration_validation_seeds_disjoint',False),('production_scalar_vector_agreement',False)]:
        c=calibrated();c['acceptance'][key]=value
        with pytest.raises(ValueError):s.analyze_pairs([.5]*200,c)


@pytest.mark.parametrize('field', ['score','sample_pair_variance','lower','upper','conditional_power_player','conditional_power_teacher',
    'third_central_moment','standardized_third_moment','lower_critical_value','upper_critical_value'])
def test_independent_audit_rejects_changed_numerics(field):
    scores=[0,1]*100;r=s.analyze_pairs(scores,calibrated())
    r['looks'][0][field]+=.001
    with pytest.raises(ValueError):s.verify_analysis(scores,calibrated(),r)


def test_independent_audit_rejects_changed_stop_or_binding():
    scores=[0,1]*100;base=s.analyze_pairs(scores,calibrated())
    for key,value in [('stop',True),('classification','superior'),('pair_scores_sha256','bad'),('games',402)]:
        r=deepcopy(base);r[key]=value
        with pytest.raises(ValueError):s.verify_analysis(scores,calibrated(),r)


def test_independent_audit_rejects_added_history():
    scores=[0,1]*100;r=s.analyze_pairs(scores,calibrated());r['looks'].append(deepcopy(r['looks'][0]))
    with pytest.raises(ValueError):s.verify_analysis(scores,calibrated(),r)


def test_qualification_plan_and_null_grid_are_deterministic():
    grid=q.distributions();assert grid==q.distributions()
    assert {r['mean'] for r in grid}=={.02,.1,.25,.48,.5,.52,.75,.9,.98}
    assert len({r['case_id'] for r in grid})==len(grid)
    for row in grid:
        assert abs(sum(row['probabilities'])-1)<1e-12
        assert abs(sum(x*p for x,p in zip(s.SCORES,row['probabilities']))-row['mean'])<1e-12
    assert q.plan()['maximum_simultaneous_null_upper']==.025
    assert q.plan()['minimum_simultaneous_power_lower']==.78


def test_binomial_uncertainty_does_not_hide_observed_errors():
    upper=q._cp_bound(2400,100000,.0001,True)
    assert upper>.024
    assert q._cp_bound(0,100000,.0001,False)==0
    assert q._cp_bound(100000,100000,.0001,True)==1


def test_vector_and_scalar_paths_agree_without_models():
    result=q.implementation_agreement(lambda:None)
    assert result['passed'] and result['scheduled_looks_checked']==1080


def test_skewness_guard_widens_only_vulnerable_tail_plus_common_guard():
    positive=s._look([0]*180+[4]*20)
    negative=s._look([0]*20+[4]*180)
    assert positive['standardized_third_moment']>0
    assert positive['upper_critical_value']>positive['lower_critical_value']>positive['critical_value']
    assert negative['standardized_third_moment']<0
    assert negative['lower_critical_value']>negative['upper_critical_value']>negative['critical_value']
    assert math.isclose(positive['lower'],1-negative['upper'],abs_tol=1e-14)
    assert math.isclose(positive['upper'],1-negative['lower'],abs_tol=1e-14)


def test_exact_cubic_moment_matches_direct_centered_sum():
    nums=[0]*40+[1]*20+[2]*80+[3]*30+[4]*30
    row=s._look(nums)
    expected=sum((n/4-row['score'])**3 for n in nums)/200
    assert math.isclose(row['third_central_moment'],expected,abs_tol=1e-15)
    assert row['pair_numerator_cube_sum']==sum(x**3 for x in nums)


def test_skew_guard_never_narrows_original_interval():
    for nums in ([0]*180+[4]*20,[0,4]*100,[1]*160+[4]*40,[2]*200):
        row=s._look(nums)
        se=math.sqrt(row['variance_used']/200);base=row['critical_value']
        assert row['lower']<=max(0.,row['score']-base*se)
        assert row['upper']>=min(1.,row['score']+base*se)
