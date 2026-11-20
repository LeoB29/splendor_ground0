"""Unchanged paired inference, complete evidence and strict matchup identity."""
from copy import deepcopy
import pytest

from tools.splendorust_graph64_strength_20261025 import schedule, statistics as stats
from tools.splendorust_search_scale_opt_20261024 import schedule as previous_schedule, statistics as previous_stats


def games(outcome, specs=None):
    rows = []
    for spec in schedule.stage_schedule('main') if specs is None else specs:
        pair, game, seat = spec['pair_index'], spec['game_index'] % 2, spec['challenger_seat']
        halfpoints = outcome(pair, game, seat)
        winner = None if halfpoints == 1 else seat if halfpoints == 2 else 1-seat
        rows.append({**spec, 'complete': True, 'winner': winner, 'reason': 'natural'})
    return rows


def test_arithmetic_rng_stream_and_classification_identical_to_frozen_prior_algorithm():
    outcome = lambda pair, game, seat: (pair+game) % 3
    rows = games(outcome)
    current = stats.analyze_games(rows, True)
    old = previous_stats.analyze_games(games(outcome, previous_schedule.stage_schedule('main')), True)
    skipped = {'schema', 'verified_score_rows_sha256', 'challenger_mode', 'opponent_mode'}
    assert {k: v for k, v in current.items() if k not in skipped} == {k: v for k, v in old.items() if k not in skipped}
    assert current['bootstrap']['replicates'] == 50000 and current['bootstrap']['seed'] == 2026102403
    assert current['challenger_mode'] == 'candidate' and current['opponent_mode'] == 'baseline'
    verified = stats.verify_analysis(list(reversed(rows)), current)
    assert verified['status'] == 'passed' and verified['independent_python_integer_sums']


def test_symmetric_extremes_and_degenerate_interpretation_preserved():
    win = stats.analyze_games(games(lambda *args: 2), True)
    loss = stats.analyze_games(games(lambda *args: 0), True)
    assert win['numerator'] == 400 and win['challenger_mean_score'] == 1.
    assert win['paired_ci95'] == [1., 1.] and win['bootstrap_degenerate']
    assert win['bounded_pair_hoeffding_95'][0] < 1
    assert win['classification'] == 'useful_evidence_for_more_search'
    assert loss['classification'] == 'useful_evidence_for_opponent_puct32'
    assert loss['bounded_pair_hoeffding_95'][1] > 0


def test_draws_and_one_weak_seat_do_not_claim_equivalence_or_useful_improvement():
    draw = stats.analyze_games(games(lambda *args: 1), True)
    assert draw['draws'] == 200 and draw['numerator'] == 200 and draw['pair_numerators'] == [2]*100
    assert draw['classification'] == 'inconclusive_or_small'
    weak = stats.analyze_games(games(lambda pair, game, seat: 2 if seat == 0 or pair < 40 else 0), True)
    assert weak['challenger_mean_score'] == .7 and weak['paired_ci95'][0] > .5
    assert weak['by_challenger_seat'] == {'0': 1., '1': .4}
    assert weak['classification'] == 'inconclusive_or_small'


def test_no_partial_optional_sample_or_unverified_evidence():
    rows = games(lambda *args: 2)
    for gate in (False, 1, None):
        with pytest.raises(ValueError, match='gates'): stats.analyze_games(rows, gate)
    with pytest.raises(ValueError, match='200'): stats.validated_games(rows[:-1])
    with pytest.raises(ValueError, match='duplicate'): stats.validated_games(rows[:-1]+rows[:1])


@pytest.mark.parametrize('field,value', [('challenger_mode', 'baseline'), ('opponent_mode', 'candidate'),
    ('challenger_quota', 32), ('opponent_quota', 16), ('seed_decimal', '1'), ('game_index', True),
    ('complete', False), ('reason', 'safety_cutoff'), ('winner', True), ('loser', 0)])
def test_changed_mode_quota_or_nonofficial_game_rejected(field, value):
    rows = games(lambda *args: 2); rows[0][field] = value
    with pytest.raises(ValueError): stats.validated_games(rows)


@pytest.mark.parametrize('field', ['challenger_mode', 'opponent_mode', 'winner', 'seed_decimal'])
def test_required_identity_fields_cannot_be_omitted(field):
    rows = games(lambda *args: 2); del rows[0][field]
    with pytest.raises(ValueError): stats.validated_games(rows)


@pytest.mark.parametrize('loser', [0, 1])
def test_forfeit_counts_as_loss_for_responsible_player(loser):
    rows = games(lambda *args: 2)
    role = 'challenger' if loser == rows[0]['challenger_seat'] else 'opponent'
    rows[0].update(reason='forfeit', loser=loser, winner=1-loser, failed_role=role, failed_seat=loser)
    assert stats.validated_games(rows)[0]['challenger_score_halfpoints'] == (0 if role == 'challenger' else 2)
    for changes in ({'winner': loser}, {'failed_role': 'opponent' if role == 'challenger' else 'challenger'}, {'failed_seat': 1-loser}):
        altered = deepcopy(rows); altered[0].update(changes)
        with pytest.raises(ValueError, match='forfeit'): stats.validated_games(altered)


def test_resource_deadline_boundary_interrupts_without_partial_analysis():
    class Stop(Exception): pass
    def stop(): raise Stop('bounded verification')
    with pytest.raises(Stop): stats.analyze_games(games(lambda *args: 2), True, boundary=stop)
