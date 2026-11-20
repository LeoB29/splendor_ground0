"""Fixed-sample integer inference and no favorable-forfeit selection."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools.splendorust_exploration128_20261029_02 import analysis as stats, schedule


def games(stage='development', candidate=None, outcome=lambda spec: 1):
    result=[]
    for spec in schedule.make_schedule(stage,candidate):
        points=outcome(spec);seat=spec['challenger_seat']
        result.append(dict(spec,complete=True,reason='natural',
                           winner=None if points==1 else seat if points==2 else 1-seat))
    return result


def admin(row,loser):
    row.update(reason='forfeit',loser=loser,winner=1-loser,failed_seat=loser,
               failed_role='challenger' if loser==row['challenger_seat'] else 'opponent',
               administrative_forfeit=True,failure_policy=stats.FAILURE_POLICY,
               failure='research_outer_hard_guard')


def balanced(spec,wins):
    # A whole pair gives the candidate two wins or two losses, so seats agree.
    return 2 if spec['pair_index'] < wins else 0


def test_fixed_common_development_and_disjoint_confirmation_schedule():
    assert stats.DEVELOPMENT_PAIRS==50 and stats.CONFIRMATION_PAIRS==200
    assert stats.REPLICATES==50000 and stats.SEED==20261029128
    dev=schedule.make_schedule('development');ops=schedule.make_schedule('operational')
    conf=schedule.make_schedule('confirmation','low')
    assert len(dev)==200 and len(ops)==12 and len(conf)==400
    for pair in range(50):
        block=dev[4*pair:4*pair+4]
        assert len({row['seed_decimal'] for row in block})==1
        assert {row['candidate'] for row in block[:2]}==({'low'} if pair%2==0 else {'high'})
        assert {(row['candidate'],row['challenger_seat']) for row in block}=={
            ('low',0),('low',1),('high',0),('high',1)}
    assert not {row['seed_decimal'] for row in dev+ops}&{row['seed_decimal'] for row in conf}
    assert all(row['challenger_quota']==row['opponent_quota']==128 for row in dev+conf)


@pytest.mark.parametrize('wins,selected',[(24,None),(25,None),(27,None),(28,'low'),(50,'low')])
def test_exact_screen_eligibility_and_predetermined_tie(wins,selected):
    rows=games(outcome=lambda spec:balanced(spec,wins))
    result=stats.select_candidate(rows,True)
    assert result['selected_candidate']==selected
    assert result['candidates']['low']['official']['mean_score']==wins/50
    assert result['development_is_strength_confirmation'] is False
    assert stats.verify_selection(list(reversed(rows)),result)['selected_candidate']==selected


def test_exact55percent_threshold_with_draws():
    rows=games(outcome=lambda spec:2 if spec['pair_index']<27 else 1 if spec['pair_index']==27 else 0)
    result=stats.select_candidate(rows,True)
    assert result['candidates']['low']['official']['numerator']==110
    assert result['selected_candidate']=='low'


def test_both_seats_required_even_with_good_aggregate():
    rows=games(outcome=lambda spec:2 if spec['challenger_seat']==0 else
               2 if spec['pair_index']<20 else 0)
    result=stats.select_candidate(rows,True)
    assert result['candidates']['low']['official']['mean_score']==.7
    assert result['selected_candidate'] is None


def test_conservative_ranking_not_official_score_or_timing():
    rows=games(outcome=lambda spec:balanced(spec,30))
    # high receives20 favorable forfeits but has identical remaining natural score.
    for row in rows:
        if row['candidate']=='high' and row['pair_index']>=40:
            admin(row,1-row['challenger_seat'])
        row['timing_seconds']=0 if row['candidate']=='high' else 999
        row['model_q_estimate']=999 if row['candidate']=='high' else -999
    result=stats.select_candidate(rows,True)
    assert result['candidates']['high']['official']['mean_score']==.8
    assert result['candidates']['high']['conservative']['mean_score']==.6
    assert result['selected_candidate']=='low'
    stats.verify_selection(rows,result)


def test_forfeits_cannot_admit_an_underperforming_candidate():
    rows=games(outcome=lambda spec:balanced(spec,25))
    for row in rows:
        if row['pair_index']>=40:admin(row,1-row['challenger_seat'])
    result=stats.select_candidate(rows,True)
    assert result['candidates']['low']['official']['mean_score']==.7
    assert result['candidates']['low']['conservative']['mean_score']==.5
    assert result['selected_candidate'] is None


@pytest.mark.parametrize('mutation',['missing','extra','duplicate','wrong_stage','wrong_candidate',
                                      'wrong_seed','quota64','bool_winner','incomplete','unofficial'])
def test_complete_fixed_sample_and_outcome_counterexamples(mutation):
    rows=games()
    if mutation=='missing':rows.pop()
    elif mutation=='extra':rows.append(deepcopy(rows[0]))
    elif mutation=='duplicate':rows[-1]=deepcopy(rows[0])
    elif mutation=='wrong_stage':rows[0]['stage']='confirmation'
    elif mutation=='wrong_candidate':rows[0]['candidate']='high'
    elif mutation=='wrong_seed':rows[0]['seed_decimal']='1'
    elif mutation=='quota64':rows[0]['opponent_quota']=64
    elif mutation=='bool_winner':rows[0]['winner']=True
    elif mutation=='incomplete':rows[0]['complete']=False
    elif mutation=='unofficial':rows[0]['reason']='cutoff'
    with pytest.raises(ValueError):stats.select_candidate(rows,True)


@pytest.mark.parametrize('field,value',[('loser',True),('winner',0),('failed_role','opponent'),
                                       ('administrative_forfeit',False),('failure','timeout'),
                                       ('failure_policy','one_second_timeout_loss'),('failed_seat',1)])
def test_only_coherent_classified_guards_enter_scores(field,value):
    rows=games();admin(rows[0],0);rows[0][field]=value
    with pytest.raises(ValueError):stats.select_candidate(rows,True)


@pytest.mark.parametrize('gate',[False,None,1])
def test_full_evidence_gate_is_not_truthiness(gate):
    with pytest.raises(ValueError):stats.select_candidate(games(),gate)
    with pytest.raises(ValueError):stats.analyze_confirmation(games('confirmation','low'),'low',gate)


def test_independent400_confirmation_both_bootstrap_paths():
    rows=games('confirmation','high',outcome=lambda spec:balanced(spec,120))
    admin(rows[250],1-rows[250]['challenger_seat'])
    result=stats.analyze_confirmation(rows,'high',True)
    assert result['official']['mean_score']==241/400
    assert result['conservative']['mean_score']==.6
    assert result['useful_gain_established'] is True
    assert result['development_games_pooled'] is False
    assert result['one_second_deployability_established'] is False
    receipt=stats.verify_confirmation(list(reversed(rows)),'high',result)
    assert receipt['independent_python_integer_sums']
    assert receipt['independent_linear_percentiles']
    assert result['bootstrap']['same_indices_for_official_and_conservative']


def test_confirmation_favorable_forfeits_do_not_promote():
    rows=games('confirmation','low',outcome=lambda spec:balanced(spec,100))
    for row in rows[300:]:admin(row,1-row['challenger_seat'])
    result=stats.analyze_confirmation(rows,'low',True)
    assert result['official']['useful_gain'] is True
    assert result['conservative']['mean_score']==.5
    assert result['useful_gain_established'] is False
    assert result['classification']=='official_useful_gain_not_robust_to_guard_forfeits'


def test_confirmation_requires_selected_branch_and_all400():
    rows=games('confirmation','low')
    for candidate in (None,'high'):
        with pytest.raises(ValueError):stats.analyze_confirmation(rows,candidate,True)
    with pytest.raises(ValueError):stats.analyze_confirmation(rows[:200],'low',True)


def test_degenerate_bootstrap_keeps_existing_concentration_safeguard():
    rows=games('confirmation','low',outcome=lambda spec:2 if spec['game_index']%2==0 else 1)
    result=stats.analyze_confirmation(rows,'low',True)
    assert result['official']['bootstrap_degenerate'] is True
    assert result['official']['paired_ci95']==[.75,.75]
    assert result['useful_gain_established'] is True
    assert result['official']['useful_gain_criteria']['degenerate_interval_safeguard']


def test_boundary_can_stop_without_a_partial_result():
    class Stop(Exception):pass
    def boundary():raise Stop()
    with pytest.raises(Stop):stats.select_candidate(games(),True,boundary=boundary)
    with pytest.raises(Stop):stats.analyze_confirmation(games('confirmation','low'),'low',True,boundary=boundary)


def test_independent_selection_reader_rejects_report_tamper():
    rows=games(outcome=lambda spec:balanced(spec,30));result=stats.select_candidate(rows,True)
    result['selected_candidate']='high'
    with pytest.raises(ValueError):stats.verify_selection(rows,result)

