"""Finite model-free qualification tests; no model load, inference or games."""
from copy import deepcopy

import pytest

from tools.splendorust_one_second_20261024 import qualification as q
from tools.splendorust_one_second_20261024 import worker
from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018_02 import bridge


@pytest.fixture(scope='module')
def plan():
    return q.build_plan()


def test_fixed_schedule_contains_each_pair_four_times_and_reverses_order(plan):
    schedule = plan['timed_requests']
    assert len(schedule) == 184
    assert [(r['arm'], r['variant']) for r in schedule[:4]] == [
        ('search', 'original'), ('search', 'hidden_variant'),
        ('greedy', 'original'), ('greedy', 'hidden_variant')]
    assert [(r['arm'], r['variant']) for r in schedule[92:96]] == [
        ('greedy', 'hidden_variant'), ('greedy', 'original'),
        ('search', 'hidden_variant'), ('search', 'original')]
    assert [r['fixture_index'] for r in schedule[::4]] == list(range(23)) + list(range(22, -1, -1))
    identities = {(r['fixture_index'], r['arm'], r['variant'], r['pass']) for r in schedule}
    assert len(identities) == 184


def test_panel_variants_are_strict_corrected_and_lawfully_indistinguishable(plan):
    for fixture in plan['fixtures']:
        one = q.guard.decode_training_snapshot(fixture['original'])
        two = q.guard.decode_training_snapshot(fixture['hidden_variant'])
        assert fixture['original'] != fixture['hidden_variant']
        assert rules.observation_payload(one) == rules.observation_payload(two)
        assert q.digest(rules.observation_payload(one)) == fixture['observer_sha256']
    state = q.state_for_request(plan, plan['timed_requests'][0])
    assert rules.serialize_snapshot(state) == plan['fixtures'][0]['original']
    wrong = deepcopy(plan['timed_requests'][0]); wrong['fixture_id'] = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        q.state_for_request(plan, wrong)


def test_counterexample_manifest_covers_required_edges_and_has_fixed_counts(plan):
    assert len(plan['counterexamples']) == 15
    tags = {tag for item in plan['counterexamples'] for tag in item['tags']}
    assert set(rules.CORRECTED_COSTS) <= tags
    assert {'nonpurchase_noble', 'multiple_nobles', 'noble_final_round', 'private_reserves',
            'forced_pass', 'single_choice', 'turn123', 'turn124', 'gold_returns'} <= tags
    assert plan['actual_complete_decisions']['total'] == 52
    assert plan['native_operations']['total'] == 101
    assert plan['worker_rejection_attempts'] == 26
    state = q.guard.decode_training_snapshot(plan['alternate_starting_seat'])
    assert state.start_player == state.current_player == 1 and state.turn_index == 0


def test_actual_worker_rejects_unlawful_or_historical_inputs_before_any_adapter(plan):
    state = q.guard.decode_training_snapshot(next(f['snapshot'] for f in plan['counterexamples']
                                                 if f['id'] == 'private_reserves'))
    context = {'diagnostic_capture': False, 'request_started': None, 'adapters': {}}
    rejected = q.worker_rejections(lambda request: worker.handle(context, request), state)
    assert len(rejected) == 26
    assert {r['case'] for r in rejected} == set(q.rejection_case_names())


def test_historical_search_work_and_action_pass_new_independent_checks(plan):
    for fixture in plan['fixtures']:
        old = q.read(q.ROOT / fixture['prior_search_reference']['path'])
        result = old['clock']['result']
        checked = q.validate_scientific_decision('search', old['observer'], result['action'], result['diagnostics'])
        assert checked['simulations'] == 16


def test_search_quota_seed_and_branch_tampering_are_rejected(plan):
    old = q.read(q.ROOT / plan['fixtures'][0]['prior_search_reference']['path'])
    result = old['clock']['result']
    for key, value in [('simulations', 15), ('heuristic_phase', True), ('leaf_calls', -1)]:
        trace = deepcopy(result['diagnostics']); trace['phases'][0][key] = value
        with pytest.raises(ValueError):
            q.validate_scientific_decision('search', old['observer'], result['action'], trace)
    trace = deepcopy(result['diagnostics']); trace['observer_seed'] += 1
    with pytest.raises(ValueError, match='seed'):
        q.validate_scientific_decision('search', old['observer'], result['action'], trace)


def _greedy_fixture():
    state = q.reference._initial()
    observer = rules.observation_payload(state)
    mains = bridge.main_actions(observer)
    chosen = bridge.resolve_compound(observer, mains[0], lambda _o, actions: actions[0])
    action = rules.serialize_action(chosen)
    logits = [0.0] * 67
    trace = {'profile_id': 'entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1',
        'observer_seed': int(q.digest(observer)[:16], 16), 'observer_sha256': q.digest(observer),
        'action_sha256': q.digest(action), 'forced_pass': False, 'simulations': 0, 'inference_calls': 1,
        'phases': [{'action': mains[0], 'branch': 'neural_greedy', 'native_legal_count': len(mains),
            'retained_indices': list(range(len(mains))),
            'canonical_indices': [q.reference.canonical_index(a) for a in mains],
            'selected_index': 0, 'canonical_logits': logits, 'policy_logits_sha256': q.digest(logits),
            'inference': [{'input_sha256': q.digest('input'), 'output_sha256': q.digest('output')}]}]}
    return observer, action, trace


def test_greedy_first_maximum_is_checked_against_saved_logits():
    observer, action, trace = _greedy_fixture()
    assert q.validate_scientific_decision('greedy', observer, action, trace)['inference_calls'] == 1
    bad = deepcopy(trace)
    bad['phases'][0]['selected_index'] = 1
    bad['phases'][0]['action'] = bridge.main_actions(observer)[1]
    alternative = bridge.resolve_compound(observer, bad['phases'][0]['action'], lambda _o, actions: actions[0])
    bad_action = rules.serialize_action(alternative)
    bad['action_sha256'] = q.digest(bad_action)
    with pytest.raises(ValueError, match='first maximum'):
        q.validate_scientific_decision('greedy', observer, bad_action, bad)
    bad = deepcopy(trace); bad['phases'][0]['canonical_logits'][1] = float('nan')
    with pytest.raises(ValueError, match='logits'):
        q.validate_scientific_decision('greedy', observer, action, bad)


def test_scientific_comparison_excludes_capture_but_preserves_work_and_logits():
    observer, action, trace = _greedy_fixture()
    captured = deepcopy(trace)
    captured['phases'][0]['inference'][0].update(tokens=[1], output83=[2])
    assert q.scientific_result('greedy', action, trace) == q.scientific_result('greedy', action, captured)
    captured['phases'][0]['policy_logits_sha256'] = 'changed'
    assert q.scientific_result('greedy', action, trace) != q.scientific_result('greedy', action, captured)


def test_panel_stops_on_incomplete_count_and_margin_before_admitting_any_games(plan):
    with pytest.raises(ValueError, match='incomplete'):
        q.validate_timed_panel([], plan)
    rows = [dict(spec, elapsed_seconds=.701) for spec in plan['timed_requests']]
    with pytest.raises(ValueError, match='margin'):
        q.validate_timed_panel(rows, plan)


def test_semantic_counterexamples_independently_replay_and_reject_false_terminal(plan):
    checks = q.reference._semantic_edges(plan['counterexamples'])
    assert {'six_faces', 'gold_payment', 'gold_return', 'noble_final_round'} <= set(checks)
    assert len(q.reference.identity_rejections(plan['counterexamples'][0])) == 15
