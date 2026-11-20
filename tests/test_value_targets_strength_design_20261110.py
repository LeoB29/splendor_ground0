"""Model-free counterexamples for fixed two-checkpoint design and inference."""
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')
import pytest
from tools.splendorust_value_targets_20261110.strength import common as c, design, analysis

from tools.splendorust_value_targets_20261110 import schedule
ORIGINAL_RESERVED = design._reserved

def reserved_rows(): return schedule.strength_schedule()

def reserved_initials():
    # Minimal strict reservation-view fixture; full corrected decoder is tested
    # independently in reservation tests. No models/games occur here.
    state = schedule._initial(schedule.seed('unit_fixture'), 0)
    return [{'spec': row, 'initial': copy.deepcopy(state)} for row in reserved_rows()]

@pytest.fixture(autouse=True)
def reserved_fixture(monkeypatch):
    rows, initials = reserved_rows(), reserved_initials()
    design.validate_reserved(rows, initials)
    fixed = {(r['stage'], r['repetition'], r['local_pair_index'], r['order_in_pair']): (r, i['initial'])
             for r, i in zip(rows, initials)}
    monkeypatch.setattr(design, '_reserved', lambda: fixed)


def games(rep, wins=65, *, draw_every=None):
    rows = []
    for spec in design.make_schedule('main', rep):
        winner = spec['challenger_seat'] if spec['pair_index'] < wins else 1 - spec['challenger_seat']
        if draw_every and spec['pair_index'] % draw_every == 0:
            winner = None
        rows.append(dict(spec, complete=True, winner=winner, reason='natural', loser=None,
                         administrative_forfeit=False))
    return rows


def both(wins0=65, wins1=65):
    return {0: games(0, wins0), 1: games(1, wins1)}


def test_reserved_views_bind_all_fixed_games_and_states(monkeypatch):
    seen = []
    for rep in (0, 1):
        monkeypatch.setattr(c, 'REPETITION', rep)
        rows = design.make_schedule()
        assert len(rows) == 206
        assert design.validate_schedule(rows) == rows
        for spec in rows:
            row = reserved_rows()[spec['global_game_index']]
            assert spec['seed_decimal'] == row['seed_decimal']
            assert spec['checkpoint_name'] == row['candidate']
            seen.append(spec['global_game_index'])
        pair = design.make_schedule('main')[:2]
        assert design.initial_snapshot(pair[0]) == design.initial_snapshot(pair[1])
        state = design.initial_snapshot(pair[0]); state['state']['terminal'] = True
        assert design.initial_snapshot(pair[0])['state']['terminal'] is False
    assert sorted(seen) == list(range(412))


@pytest.mark.parametrize('field,value', [
    ('pair_index', False), ('game_index', 0.0), ('challenger_quota', 128.0),
    ('challenger_seat', False), ('repetition', False), ('global_game_index', 12.0),
    ('global_pair_index', 6.0), ('reserved_stage_pair_index', False),
    ('opponent_cpuct', '0.4'), ('checkpoint_name', 'r1_broad_joint'),
    ('seed_decimal', '1'), ('stage', 'operational'),
])
def test_runtime_spec_rejects_type_aliases_and_substitution(field, value):
    row = design.make_schedule('main', 0)[0]
    row[field] = value
    with pytest.raises(ValueError):
        design.validate_spec(row, 0)


@pytest.mark.parametrize('field,value', [
    ('repetition', False), ('game_index', 0.0), ('candidate_simulations', 128.0),
    ('candidate_seat', False), ('cpuct_each', '0.4'), ('depth_each', 64.0),
])
def test_reserved_reader_rejects_saved_preparation_type_aliases(field, value):
    rows = reserved_rows()
    initials = reserved_initials()
    rows[0][field] = value
    with pytest.raises(ValueError):
        design.validate_reserved(rows, initials)


def test_schedule_rejects_enrichment_missing_order_duplicate_and_wrong_rep():
    rows = design.make_schedule('main', 0)
    corruptions = [rows[:-1], rows[::-1], rows[:-1] + [rows[0]], design.make_schedule('main', 1)]
    enriched = copy.deepcopy(rows); enriched[0]['unexpected'] = True; corruptions.append(enriched)
    for bad in corruptions:
        with pytest.raises(ValueError):
            design.validate_schedule(bad, 'main', 0)
    with pytest.raises(ValueError):
        design.make_schedule('main', False)


def test_reservation_hash_failure_is_not_silent_regeneration(monkeypatch):
    from unittest.mock import Mock
    ORIGINAL_RESERVED.cache_clear()
    monkeypatch.setattr(c, 'read', lambda path: {'reservation': {'strength_schedule': {}, 'strength_initials': {}, 'reservation_receipt': {}}})
    bad = Mock(side_effect=ValueError('artifact changed'))
    monkeypatch.setattr(c, 'verify', bad)
    with pytest.raises(ValueError, match='artifact changed'): ORIGINAL_RESERVED()
    bad.assert_called_once()
    ORIGINAL_RESERVED.cache_clear()


def test_full_50000_pooled_and_independent_reproduction():
    assert analysis.REPLICATES == 50000
    rows = both(68, 62)
    result = analysis.analyze_pooled(rows, True)
    assert result['official']['mean_score'] == .65
    assert result['official']['numerator'] == 520
    assert result['official']['denominator'] == 800
    assert result['games'] == 400 and result['pairs'] == 200
    assert result['useful_gain_established']
    assert result['bootstrap']['stratum_weights'] == [.5, .5]
    assert result['descriptive_individuals']['0']['descriptive_only']
    assert result['descriptive_individuals']['1']['useful_gain_established'] is False
    receipt = analysis.verify_pooled(rows, result)
    assert receipt['status'] == 'passed'
    assert receipt['independent_stratified_draws'] and receipt['primary_replicates'] == 50000


@pytest.mark.parametrize('mutation', ['missing_rep', 'bool_rep', 'partial', 'duplicate', 'swap_rep', 'wrong_global', 'bool_winner', 'float_score'])
def test_primary_rejects_incomplete_or_corrupt_fixed_sample(mutation):
    data = both()
    if mutation == 'missing_rep': data.pop(1)
    elif mutation == 'bool_rep': data = {False: data[0], 1: data[1]}
    elif mutation == 'partial': data[1].pop()
    elif mutation == 'duplicate': data[0][-1] = data[0][0]
    elif mutation == 'swap_rep': data[1] = data[0]
    elif mutation == 'wrong_global': data[0][0]['global_game_index'] = 22
    elif mutation == 'bool_winner': data[0][0]['winner'] = False
    elif mutation == 'float_score': data[0][0]['candidate_score_halfpoints'] = 2.0
    with pytest.raises(ValueError):
        analysis.analyze_pooled(data, True)


def test_primary_requires_literal_passed_gate():
    for gate in (False, None, 1, 'passed'):
        with pytest.raises(ValueError):
            analysis.analyze_pooled(both(), gate)


@pytest.mark.parametrize('failure', ['worker_exit', 'illegal', 'malformed', 'clock_error', None])
def test_infrastructure_failures_cannot_be_administrative_player_losses(failure):
    data = games(0)
    row = data[0]
    row.update(reason='forfeit', loser=1 - row['challenger_seat'], administrative_forfeit=True,
               failed_seat=1 - row['challenger_seat'], failed_role='opponent',
               failure=failure, failure_policy=analysis.FAILURE_POLICY)
    with pytest.raises(ValueError, match='classified research hard-guard'):
        analysis.validated_games(data, 0)


def test_independent_ledger_clock_and_timing_checks_preserved_from_joint_route():
    import ast
    old = ast.parse(Path('tools/splendorust_selfplay_pilot_resume2_20261103/assessment/analysis.py').read_text())
    new = ast.parse(Path(analysis.__file__).read_text())
    old_functions = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
    for name in ('validate_completed', 'audit_game', 'distribution', 'summarize_timing'):
        assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(new_functions[name], include_attributes=False)


def test_reference_forfeits_do_not_manufacture_conservative_gain(monkeypatch):
    monkeypatch.setattr(analysis, 'REPLICATES', 700)
    data = both(100, 100)
    for rows in data.values():
        for row in rows:
            loser = 1 - row['challenger_seat']
            row.update(reason='forfeit', loser=loser, administrative_forfeit=True,
                       failed_seat=loser, failed_role='opponent', failure='research_outer_hard_guard',
                       failure_policy=analysis.FAILURE_POLICY)
    result = analysis.analyze_pooled(data, True)
    assert result['official']['mean_score'] == 1
    assert result['official']['useful_gain']
    assert result['conservative']['mean_score'] == 0
    assert result['useful_gain_established'] is False
    assert result['administrative_sensitivity']['administrative_games'] == 400
    assert result['classification'] == 'official_pooled_gain_not_robust_to_guard_forfeits'


def test_mixed_fixed_repeats_are_not_treated_as_training_seed_replications(monkeypatch):
    monkeypatch.setattr(analysis, 'REPLICATES', 700)
    result = analysis.analyze_pooled(both(100, 0), True)
    assert result['official']['mean_score'] == .5
    assert result['official']['paired_ci95'] == [.5, .5]
    assert result['official']['bootstrap_degenerate']
    assert result['official']['useful_gain_criteria']['degenerate_interval_safeguard'] is False
    assert not result['generalization_over_training_seeds_established']
    assert not result['both_individual_checkpoints_improved_claim']
    assert not result['individual_checkpoint_selection']


def test_pooled_seat_requirement_remains_required(monkeypatch):
    monkeypatch.setattr(analysis, 'REPLICATES', 700)
    data = both(100, 100)
    for rows in data.values():
        for row in rows:
            if row['challenger_seat'] == 1 and row['pair_index'] >= 40:
                row['winner'] = 0
    result = analysis.analyze_pooled(data, True)
    assert result['official']['mean_score'] == .7
    assert result['official']['by_candidate_seat'] == {'0': 1., '1': .4}
    assert result['official']['useful_gain_criteria']['each_seat_score_at_least_50pct'] is False
    assert result['useful_gain_established'] is False


def test_independent_reader_rejects_type_alias_and_changed_report(monkeypatch):
    monkeypatch.setattr(analysis, 'REPLICATES', 700)
    data = both(65, 65)
    result = analysis.analyze_pooled(data, True)
    changed = copy.deepcopy(result); changed['automatic_promotion'] = 0
    with pytest.raises(ValueError, match='report changed'):
        analysis.verify_pooled(data, changed)
    changed = copy.deepcopy(result); changed['official']['paired_ci95'][0] += .001
    with pytest.raises(ValueError, match='percentiles differ'):
        analysis.verify_pooled(data, changed)


def test_score_draws_and_integer_thresholds():
    data = games(0, 60, draw_every=5)
    ordered = analysis.validated_games(data, 0)
    score = analysis._score(ordered)
    assert score['draws'] == 40
    assert score['wins'] == 96 and score['losses'] == 64
    assert score['mean_score'] == .58
    assert analysis._useful_gates(dict(numerator=439, denominator=800, seat_numerators={'0': 220, '1': 219}, seat_denominator=400))['score_at_least_55pct'] is False
    assert analysis._useful_gates(dict(numerator=440, denominator=800, seat_numerators={'0': 220, '1': 220}, seat_denominator=400))['score_at_least_55pct'] is True


def replay_rows(rep):
    rows = []
    for stage, pair in [('operational', 0), ('main', 0), ('main', 32), ('main', 33), ('main', 66), ('main', 67), ('main', 99)]:
        for order in (0, 1):
            spec = design._spec(stage, pair, order, rep)
            for ply in (0, 20, 40):
                for offset, variant in enumerate(('candidate', 'control')):
                    seat = spec['challenger_seat'] if variant == 'candidate' else 1 - spec['challenger_seat']
                    row = dict(spec, ply=ply + offset, variant=variant, seat=seat,
                               player_role='challenger' if variant == 'candidate' else 'opponent',
                               quota=128, arm='search', route=variant, mode=variant, completed=True,
                               observer={'observation': {'current_player': seat, 'players': [{'reserves': []}, {'reserves': []}]}})
                    rows.append(row)
    return rows


def test_replay_cells_are_outcome_timing_blind_and_repetition_bound(monkeypatch):
    monkeypatch.setattr(c, 'REPETITION', 0)
    rows = replay_rows(0)
    result = design.select_replay_sample(rows)
    altered = copy.deepcopy(rows)
    for row in altered: row.update(winner=1, clock={'seconds': 999}, timing=999)
    assert design.select_replay_sample(altered) == result
    assert result['rule']['pair_index_thirds'] == [[0, 32], [33, 66], [67, 99]]
    assert result['coverage']['main_temporal']['available_cells'] == 36
    assert all(r['repetition'] == 0 and r['checkpoint_name'] == 'r0_broad_joint' for r in result['decisions'])
    assert len(result['decisions']) <= 96
    with pytest.raises(ValueError): design.select_replay_sample(rows + [rows[0]])
    with pytest.raises(ValueError): design.select_replay_sample(replay_rows(1))
    failed = dict(design._spec('main', 1, 0, 0), ply=1, variant='candidate', completed=False)
    with_failed = design.select_replay_sample(rows + [failed])
    assert with_failed['decisions'] == result['decisions'] and with_failed['excluded_failed_decisions'] == 1


def test_global_replay_binding_rejects_other_matchup_before_scientific_checks(monkeypatch):
    monkeypatch.setattr(c, 'REPETITION', 0)
    original = replay_rows(0)[0]
    replayed = copy.deepcopy(original); replayed['repetition'] = 1
    with pytest.raises(ValueError):
        analysis.verify_replay(original, replayed, {})


def test_late_completed_times_remain_descriptive():
    result = analysis.distribution([.6, .8, 1.2, 9.9])
    assert result['count'] == 4 and result['completed_over_700ms'] == 3
    assert result['completed_over_one_second'] == 2 and result['maximum'] == 9.9
    with pytest.raises(ValueError): analysis.distribution([True])
