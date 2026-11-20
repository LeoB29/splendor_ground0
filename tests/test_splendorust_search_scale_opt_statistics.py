"""Prospective fixed-sample schedule, arithmetic, uncertainty and interpretation."""
from copy import deepcopy
import hashlib
import json
import pytest

from tools.splendorust_search_scale_opt_20261024 import schedule, statistics as stats


def games(outcome):
    rows = []
    for spec in schedule.stage_schedule('main'):
        pair, game, seat = spec['pair_index'], spec['game_index'] % 2, spec['challenger_seat']
        halfpoints = outcome(pair, game, seat)
        winner = None if halfpoints == 1 else seat if halfpoints == 2 else 1-seat
        rows.append({**spec, 'winner': winner, 'reason': 'natural'})
    return rows


def test_fresh_domain_full_hash_and_exact_swapped_schedule():
    rows = schedule.make_schedule()
    assert len(rows) == 206 and len({row['seed_decimal'] for row in rows}) == 103
    assert rows[:6] == schedule.stage_schedule('operational')
    assert rows[6:] == schedule.stage_schedule('main')
    prior_domain = 'splendorust_puct_budget_1s_20261024_v1'
    for stage, pairs in (('operational', 3), ('main', 100)):
        selected = schedule.stage_schedule(stage)
        assert len(selected) == pairs*2
        for pair in range(pairs):
            a, b = selected[2*pair:2*pair+2]
            encoded = json.dumps([schedule.DOMAIN, stage, pair], separators=(',', ':')).encode()
            expected = str(int.from_bytes(hashlib.sha256(encoded).digest(), 'big'))
            old_domain = prior_domain + ('_operational' if stage == 'operational' else '')
            old_encoded = json.dumps([old_domain, stage, pair], separators=(',', ':')).encode()
            assert expected != str(int.from_bytes(hashlib.sha256(old_encoded).digest(), 'big'))
            assert a['seed_decimal'] == b['seed_decimal'] == expected
            assert (a['challenger_seat'], b['challenger_seat']) == (pair % 2, 1-pair % 2)
            assert (a['game_index'], b['game_index']) == (2*pair, 2*pair+1)
            assert schedule.validate_spec(a) == a and schedule.validate_spec(b) == b


@pytest.mark.parametrize('stage,pair', [('bad', 0), ('main', True), ('main', -1),
                                       ('main', 100), ('operational', 3), ('main', 1.)])
def test_seed_rejects_nonfrozen_identities(stage, pair):
    with pytest.raises(ValueError):
        schedule.expected_seed(stage, pair)


def test_symmetric_extremes_and_degenerate_bound():
    challenger = stats.analyze_games(games(lambda *args: 2), True)
    assert challenger['challenger_mean_score'] == 1 and challenger['numerator'] == 400
    assert challenger['paired_ci95'] == [1., 1.] and challenger['bootstrap_degenerate']
    assert challenger['bounded_pair_hoeffding_95'][0] < 1
    assert challenger['classification'] == 'useful_evidence_for_more_search'
    opponent = stats.analyze_games(games(lambda *args: 0), True)
    assert opponent['classification'] == 'useful_evidence_for_opponent_puct32'
    assert opponent['bounded_pair_hoeffding_95'][1] > 0


def test_draws_half_credit_inconclusive_not_equivalence():
    result = stats.analyze_games(games(lambda *args: 1), True)
    assert result['draws'] == 200 and result['pair_numerators'] == [2]*100
    assert result['numerator'] == 200 and result['challenger_mean_score'] == .5
    assert result['classification'] == 'inconclusive_or_small'
    assert result['challenger_quota'] == 64 and result['opponent_quota'] == 32


def test_one_seat_weakness_blocks_positive_classification():
    result = stats.analyze_games(games(lambda pair, game, seat: 2 if seat == 0 or pair < 40 else 0), True)
    assert result['challenger_mean_score'] == .7 and result['paired_ci95'][0] > .5
    assert result['by_challenger_seat'] == {'0': 1., '1': .4}
    assert result['classification'] == 'inconclusive_or_small'


def test_independent_full_rng_stream_integer_arithmetic_and_quantiles():
    rows = games(lambda pair, game, seat: (pair+game) % 3)
    boundaries = []
    result = stats.analyze_games(rows, True, boundary=lambda: boundaries.append(1))
    assert result['bootstrap']['seed'] == 2026102403
    assert result['bootstrap']['replicates'] == 50000 and len(boundaries) == 101
    checked = stats.verify_analysis(list(reversed(rows)), result)
    assert checked['status'] == 'passed' and checked['independent_python_integer_sums']
    result['bootstrap']['indices_sha256'] = '0'*64
    with pytest.raises(ValueError, match='stream'):
        stats.verify_analysis(rows, result)


def test_complete_fixed_sample_and_evidence_gates_required():
    rows = games(lambda *args: 2)
    for gate in (False, 1, None):
        with pytest.raises(ValueError, match='gates'):
            stats.analyze_games(rows, gate)
    with pytest.raises(ValueError, match='200'):
        stats.validated_games(rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        stats.validated_games(rows[:-1]+rows[:1])


@pytest.mark.parametrize('field,value', [('game_index', True), ('challenger_seat', 1),
    ('stage', 'operational'), ('seed_decimal', '1'), ('winner', True), ('reason', 'max_turns'),
    ('challenger_quota', 32), ('opponent_quota', 16), ('challenger_quota', 64.),
    ('complete', False), ('challenger_score_halfpoints', 1), ('loser', 0)])
def test_malformed_or_changed_game_rejected(field, value):
    rows = games(lambda *args: 2)
    rows[0][field] = value
    with pytest.raises(ValueError):
        stats.validated_games(rows)


@pytest.mark.parametrize('field', ['stage', 'pair_index', 'game_index', 'challenger_seat',
                                 'seed_decimal', 'challenger_quota', 'opponent_quota', 'winner'])
def test_missing_schedule_or_winner_field_rejected(field):
    rows = games(lambda *args: 2)
    del rows[0][field]
    with pytest.raises(ValueError):
        stats.validated_games(rows)


def test_forfeit_is_loss_for_responsible_seat_and_role():
    rows = games(lambda *args: 2)
    rows[0].update(reason='forfeit', loser=0, winner=1, failed_role='challenger', failed_seat=0)
    assert stats.validated_games(rows)[0]['challenger_score_halfpoints'] == 0
    for changes in ({'winner': 0}, {'failed_role': 'opponent'}, {'failed_seat': 1}, {'loser': False}):
        broken = deepcopy(rows)
        broken[0].update(changes)
        with pytest.raises(ValueError, match='forfeit'):
            stats.validated_games(broken)


def test_resource_boundary_can_abort_without_partial_analysis():
    class Stop(Exception):
        pass
    def stop():
        raise Stop('bounded analysis')
    with pytest.raises(Stop):
        stats.analyze_games(games(lambda *args: 2), True, boundary=stop)
