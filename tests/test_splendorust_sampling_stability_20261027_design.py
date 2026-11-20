"""Independent model-free checks of the fixed sampling comparison and summaries."""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from tools.splendorust_sampling_stability_20261027 import design, analysis


@pytest.fixture(scope='module')
def prepared():
    return (design.c.read(design.PANEL), design.c.read(design.OLD_MATRIX),
            design.build_matrix())


def test_exact_fixed_matrix_and_immutable_prepared_artifact(prepared):
    panel, _, matrix = prepared
    assert matrix == design.c.read(design.c.OUT/'prepared_request_matrix.json')
    rows = matrix['requests']
    assert len(rows) == 324 and [r['sequence'] for r in rows] == list(range(324))
    assert Counter(r['stage'] for r in rows) == {
        'frozen_reference_reproduction': 24, 'research_pilot': 72,
        'research_main': 216, 'research_reproduction': 12}
    assert rows[:24] == [r for r in rows if r['route'] == 'frozen']
    assert [r['root_id'] for r in rows[:24]] == [r['root_id'] for r in panel['roots']]
    assert all(r['world_pool'] == 3 and r['seed_label'] == 'A' and not r['telemetry']
               for r in rows[:24])
    assert all(r['quota'] == 64 and r['complete_response_seconds'] == 1.
               and r['seconds_per_move'] == 1. and r['internal_seconds'] == .9 for r in rows)


def test_every_root_gets_all_twelve_conditions_once_before_repeats(prepared):
    panel, _, matrix = prepared
    main = [r for r in matrix['requests'] if r['stage'] in ('research_pilot', 'research_main')]
    expected = {(pool, label) for pool in (3, 6) for label in 'ABCDEF'}
    for root in panel['roots']:
        rows = [r for r in main if r['root_id'] == root['root_id']]
        assert len(rows) == 12 and {(r['world_pool'], r['seed_label']) for r in rows} == expected
        assert all(r['route'] == 'research' and r['telemetry'] for r in rows)
        for label in 'ABCDEF':
            paired = [r for r in rows if r['seed_label'] == label]
            assert paired[0]['seed'] == paired[1]['seed']
            assert paired[0]['seed_decimal'] == paired[1]['seed_decimal']
    assert [r['root_id'] for r in main[::12]] == matrix['root_order']
    # A predetermined rotation changes condition order without choosing seeds.
    base = [(pool, label) for label in 'ABCDEF' for pool in (3, 6)]
    for index in range(24):
        rows = main[index*12:(index+1)*12]
        offset = index % 12
        assert [(r['world_pool'], r['seed_label']) for r in rows] == base[offset:]+base[:offset]


def test_pilot_and_repeats_are_six_fixed_cells_not_selected_from_outputs(prepared):
    panel, old, matrix = prepared
    roots = {r['root_id']: r for r in panel['roots']}
    assert matrix['pilot_roots'] == old['pilot_roots']
    assert len({(roots[r]['phase'], roots[r]['seat']) for r in matrix['pilot_roots']}) == 6
    assert {r['root_id'] for r in matrix['requests'][24:96]} == set(old['pilot_roots'])
    assert all(r['stage'] == 'research_pilot' for r in matrix['requests'][24:96])
    repeats = matrix['requests'][312:]
    assert len(repeats) == 12
    assert {(r['root_id'], r['world_pool']) for r in repeats} == {
        (rid, pool) for rid in old['pilot_roots'] for pool in (3, 6)}
    for row in repeats:
        original = matrix['requests'][row['repeat_of']]
        assert original['stage'] == 'research_pilot' and row['seed_label'] == 'B'
        for key in ('root_id', 'quota', 'world_pool', 'seed', 'seed_label', 'observer_sha256'):
            assert row[key] == original[key]


def test_existing_seed_families_unchanged_and_new_domain_fixed(prepared):
    panel, old, matrix = prepared
    old_seeds = {(r['root_id'], r['seed_family']): r['seed_decimal'] for r in old['requests']}
    roots = {r['root_id']: r for r in panel['roots']}
    for row in matrix['requests']:
        observer = roots[row['root_id']]['observer']
        assert row['seed_label'] == row['seed_family']
        assert row['seed_decimal'] == str(row['seed'])
        if row['seed_label'] in 'ABC':
            assert row['seed_decimal'] == old_seeds[(row['root_id'], row['seed_label'])]
        else:
            canonical = json.dumps(observer, sort_keys=True, separators=(',', ':'), allow_nan=False)
            raw = ('sampling_stability_20261027_v1|'+row['seed_label']+'|'+canonical).encode()
            assert row['seed'] == int(hashlib.sha256(raw).hexdigest()[:16], 16)
    for root in panel['roots']:
        assert len({design.seed(root['observer'], label) for label in 'ABCDEF'}) == 6


@pytest.mark.parametrize('label', ['', 'G', 'a', 'AB', None, 1])
def test_unprepared_seed_family_is_rejected(label):
    with pytest.raises(ValueError):
        design.seed({'public': 'fixture'}, label)


def test_matrix_ignores_outcomes_and_timing_metadata(prepared, monkeypatch):
    panel, old, matrix = prepared
    changed = deepcopy(panel)
    for root in changed['roots']:
        root.update(winner='different', prior_receipt_seconds=9999., prior_value=-9999.)
    calls = []
    def read(path):
        calls.append(path)
        if path == design.PANEL:
            return changed
        assert path == design.OLD_MATRIX
        return old
    monkeypatch.setattr(design.c, 'read', read)
    assert design.build_matrix() == matrix
    assert calls == [design.PANEL, design.OLD_MATRIX]


def test_plan_records_development_role_coupling_and_complete_budget():
    plan = design.build_plan()
    assert plan['diagnostic_requests'] == 324 and plan['scientific_decisions_total'] == 453
    assert plan['parent_requests_total'] == 355
    assert plan['qualification']['total_decisions'] == 129
    assert plan['qualification']['graph_cases_total'] == 44
    assert plan['controls']['reference16_world_pool'] == 3
    assert plan['controls']['native_core_unchanged'] and plan['controls']['frozen_optimized64_unchanged']
    assert 'share RNG' in plan['random_coupling'] and 'Canonical transitions ignore RNG' in plan['random_coupling']
    assert plan['resources']['job_seconds'] == 3600
    assert plan['resources']['stage_seconds'] == {'qualification': 900, 'diagnostic': 1200, 'finalization': 900}
    assert plan['resources']['collection_cutoff_seconds'] == 2700
    assert plan['resources']['ordinary_external_seconds'] == 1.
    assert plan['resources']['ordinary_internal_seconds'] == .9
    assert plan['new_games'] == plan['automatic_retries'] == 0 and plan['training'] is False
    assert plan['larger_data_deferred'] and plan['external_candidate_testing'] == 'paused'
    assert 'automatic promotion' in plan['measurement']['prohibited']


@pytest.mark.parametrize('labels,unequal,unique', [
    ([0]*6, 0, 1), ([0]*5+[1], 5, 2), ([0]*4+[1]*2, 8, 2),
    ([0]*3+[1]*3, 9, 2), (list(range(6)), 15, 6),
])
def test_six_seeds_have_fifteen_correlated_pairs(labels, unequal, unique):
    value = analysis.stability([{'choice': label} for label in labels])
    assert value == dict(n=6, pair_count=15, disagreeing_pairs=unequal,
                        disagreement_rate=unequal/15, unique_actions=unique,
                        unanimous=unique == 1)


def test_action_dictionary_key_order_does_not_create_false_instability():
    left = {'kind': 'Take', 'tokens': [1, 1, 1, 0, 0]}
    right = {'tokens': [1, 1, 1, 0, 0], 'kind': 'Take'}
    assert analysis.stability([left, right]*3)['disagreeing_pairs'] == 0


def test_partial_root_never_produces_a_complete_paired_effect():
    value = analysis.paired_stability([{'choice': 0}]*6, [{'choice': 1}]*5)
    assert not value['complete_paired_root']
    assert value['pool6_minus_pool3'] is None and value['direction'] is None
    assert value['pool3']['pair_count'] == 15 and value['pool6']['pair_count'] == 10
    assert analysis.stability([])['disagreement_rate'] is None
    assert analysis.stability([])['unanimous'] is None
    assert analysis.stability([{}])['disagreement_rate'] is None


def synthetic_rows(panel, matrix):
    """Three deliberately different roots; every other root is unanimous."""
    order = {root['root_id']: i for i, root in enumerate(panel['roots'])}
    rows = []
    for spec in matrix['requests']:
        index, pool, label = order[spec['root_id']], spec['world_pool'], spec['seed_label']
        seed_index = 'ABCDEF'.index(label)
        choice = 1 if index == 0 and pool == 3 and seed_index >= 3 else 0
        if index == 1 and pool == 6:
            choice = seed_index
        main = {'choice': choice}
        complete = {'main': main, 'return_choice': seed_index if index == 2 and pool == 6 else 0}
        rows.append(dict(spec=deepcopy(spec), completed=True, clock={'status': 'ok'},
            measurement=dict(main_action=main, complete_action=complete,
                complete_response_seconds=.2 if pool == 3 else .3, callback_count=65)))
    return rows


def test_analysis_uses_paired_roots_separates_auxiliaries_and_excludes_repeats(prepared, monkeypatch):
    panel, _, matrix = prepared
    rows = synthetic_rows(panel, matrix)
    monkeypatch.setattr(analysis, '_measure', lambda row: row['measurement'])
    value = analysis.analyze(rows, panel, matrix, complete=True)
    main, full = value['primary_main_stability'], value['secondary_complete_action_stability']
    assert main['complete_paired_roots'] == full['complete_paired_roots'] == 24
    assert main['mean_pool6_minus_pool3'] == pytest.approx((1.-.6)/24)
    assert full['mean_pool6_minus_pool3'] == pytest.approx((1.-.6+1.)/24)
    assert (main['roots_lower'], main['roots_equal'], main['roots_higher']) == (1, 22, 1)
    assert (full['roots_lower'], full['roots_equal'], full['roots_higher']) == (1, 21, 2)
    assert main['pool3_all_six_agree'] == main['pool6_all_six_agree'] == 23
    assert full['pool6_all_six_agree'] == 22
    assert value['matched_seed_changes'] == {'available': 144, 'main': 8, 'complete_action': 13}
    assert value['timing']['pool3']['n'] == value['timing']['pool6']['n'] == 144
    assert value['callbacks']['pool3']['n'] == value['callbacks']['pool6']['n'] == 144
    assert value['complete_research_roots'] == 24
    assert all(cell['main']['planned_roots'] == 4 for cell in value['cell_results'].values())
    assert not value['automatic_promotion'] and not value['strength_claim']
    assert 'not independent observations' in main['unit']
    assert 'fresh paired one-second games' in value['next_study_recommendation']


def test_partial_analysis_preserves_planned_denominators_and_first_failure(prepared, monkeypatch):
    panel, _, matrix = prepared
    rows = synthetic_rows(panel, matrix)[:24+12+5]
    rows[-1].update(completed=False, clock={'status': 'timeout'}, measurement=None)
    monkeypatch.setattr(analysis, '_measure', lambda row: row['measurement'])
    value = analysis.analyze(rows, panel, matrix, complete=False)
    assert not value['complete'] and value['planned_roots'] == 24
    assert value['complete_research_roots'] == 1
    assert value['primary_main_stability']['complete_paired_roots'] == 1
    assert value['primary_main_stability']['planned_roots'] == 24
    assert sum(len(r['missing_conditions']) for r in value['roots']) == 288-16
    assert value['first_failure'] == dict(sequence=rows[-1]['spec']['sequence'],
        root_id=rows[-1]['spec']['root_id'], status='timeout')
    assert 'incomplete coverage cannot settle' in value['next_study_recommendation']
    assert not value['automatic_promotion'] and not value['strength_claim']


def test_duplicate_condition_cannot_silently_replace_an_observation(prepared, monkeypatch):
    panel, _, matrix = prepared
    row = synthetic_rows(panel, matrix)[24]
    monkeypatch.setattr(analysis, '_measure', lambda row: row['measurement'])
    with pytest.raises(ValueError, match='Duplicate sampling condition'):
        analysis.analyze([row, deepcopy(row)], panel, matrix, complete=False)
