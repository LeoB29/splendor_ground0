"""Finite model-free qualification tests; no model load, inference or games."""
from copy import deepcopy
from types import SimpleNamespace
from tools.splendorust_search_scale_20261024 import common as c, adapters

import pytest

from tools.splendorust_search_scale_20261024 import qualification as q
from tools.splendorust_search_scale_20261024 import worker
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
        ('reference', 'original'), ('reference', 'hidden_variant')]
    assert [(r['arm'], r['variant']) for r in schedule[92:96]] == [
        ('reference', 'hidden_variant'), ('reference', 'original'),
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
    assert plan['native_operations']['total'] == 82
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


def _synthetic_callback_receipts(trace):
    """Model-free mock receipts; never relabel old records as new evidence."""
    trace = deepcopy(trace)
    for phase in trace['phases']:
        count = phase['leaf_calls']
        phase['neural_callback_count'] = count
        phase['neural_callback_rows'] = count
        phase['neural_callbacks'] = [q.digest(['a' * 64, 'b' * 64, 1])] * count
    return trace


def test_historical_search_actions_with_explicit_mock_callback_receipts_pass(plan):
    for fixture in plan['fixtures']:
        old = q.read(q.ROOT / fixture['prior_search_reference']['path'])
        result = old['clock']['result']
        trace = _synthetic_callback_receipts(result['diagnostics'])
        checked = q.validate_scientific_decision('reference', old['observer'], result['action'], trace)
        assert checked['simulations'] == 16


def test_search_quota_seed_and_branch_tampering_are_rejected(plan):
    old = q.read(q.ROOT / plan['fixtures'][0]['prior_search_reference']['path'])
    result = old['clock']['result']
    for key, value in [('simulations', 15), ('heuristic_phase', True), ('leaf_calls', -1)]:
        trace = _synthetic_callback_receipts(result['diagnostics']); trace['phases'][0][key] = value
        with pytest.raises(ValueError):
            q.validate_scientific_decision('reference', old['observer'], result['action'], trace)
    trace = deepcopy(result['diagnostics']); trace['observer_seed'] += 1
    with pytest.raises(ValueError, match='seed'):
        q.validate_scientific_decision('reference', old['observer'], result['action'], trace)


def _late_search_fixture(turn):
    observer = rules.observation_payload(q.reference._initial(turn))
    mains = bridge.main_actions(observer)
    action = rules.serialize_action(bridge.resolve_compound(observer, mains[0], lambda _o, actions: actions[0]))
    neural = int(turn == 123)
    trace = {'opponent_id': 'splendorust', 'profile_id': 'entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1',
             'observer_seed': int(q.digest(observer)[:16], 16), 'forced_pass': False,
             'inference_calls': neural, 'inference_rows': neural,
             'phases': [{'action': mains[0], 'simulations': 16 * neural, 'leaf_calls': 17 * neural,
                         'native_legal_count': len(mains), 'heuristic_phase': not bool(neural),
                         'neural_callback_count': neural, 'neural_callback_rows': neural,
                         'neural_callbacks': [q.digest(['a' * 64, 'b' * 64, 1])] * neural}]}
    return observer, action, trace


def test_turn123_heuristic_search_leaves_are_not_neural_forwards_and_turn124_is_direct_heuristic():
    for turn in (123, 124):
        observer, action, trace = _late_search_fixture(turn)
        checked = q.validate_scientific_decision('reference', observer, action, trace)
        assert checked['inference_calls'] == int(turn == 123)
        assert checked['search_work']['leaf_evaluations'] == 17 * int(turn == 123)
    observer, action, trace = _late_search_fixture(123)
    trace['inference_calls'] = trace['inference_rows'] = 17
    with pytest.raises(ValueError):
        q.validate_scientific_decision('reference', observer, action, trace)


def test_counterexample_full_result_is_persisted_before_assertion(monkeypatch):
    events = []
    monkeypatch.setattr(q, '_native_catalogues', lambda adapters: [])
    result = {'player': 'wrong', 'observer_sha256': 'invalid', 'raw_partial_trace': {'leaf_calls': 17}}
    with pytest.raises(ValueError, match='worker result binding'):
        q.run_entrypoint_checks({'search': SimpleNamespace(profile={'iterations': 32}),
                                 'reference': SimpleNamespace(profile={'iterations': 16})},
            request_callback=lambda request: result, evidence_callback=lambda row: events.append(deepcopy(row)))
    assert [row['event'] for row in events] == ['started', 'result']
    assert events[1]['result'] == result
    assert events[1]['case_id'].startswith('face_') and events[1]['arm'] == 'search'
    assert events[1]['before_snapshot'] and events[1]['request']['observer']


def test_native_probe_full_result_and_failure_are_persisted_before_consumption():
    class Native:
        def exchange(self, request):
            if request.get('fail'):
                raise RuntimeError('partial native failure')
            return {'action': 'invalid', 'leaf_calls': 17}
    events = []
    wrapped = q._RecordedAdapter(Native(), 'search', [], lambda row: events.append(deepcopy(row)))
    request = {'op': 'inspect', 'observation': {'fixture': 'full input'}}
    result = wrapped.exchange(request)
    assert events[-1]['event'] == 'result' and events[-1]['result'] == result
    assert events[-1]['request']['native_request'] == request
    with pytest.raises(RuntimeError, match='partial native failure'):
        wrapped.exchange({'op': 'inspect', 'fail': True})
    assert events[-1]['event'] == 'failed' and 'partial native failure' in events[-1]['error']


def test_semantic_counterexamples_independently_replay_and_reject_false_terminal(plan):
    checks = q.reference._semantic_edges(plan['counterexamples'])
    assert {'six_faces', 'gold_payment', 'gold_return', 'noble_final_round'} <= set(checks)
    assert len(q.reference.identity_rejections(plan['counterexamples'][0])) == 15


@pytest.mark.parametrize('quota', [32, 64])
def test_late_boundary_regression_uses_declared_challenger_quota(quota):
    for turn in (123, 124):
        observer, action, trace = _late_search_fixture(turn)
        trace['profile_id'] = c.profile_id(quota)
        phase = trace['phases'][0]
        phase['simulations'] = quota * int(turn == 123)
        phase['leaf_calls'] = (quota + 1) * int(turn == 123)
        checked = q.validate_scientific_decision('search', observer, action, trace, quota=quota)
        assert checked['simulations'] == quota * int(turn == 123)
        with pytest.raises(ValueError):
            q.validate_scientific_decision('search', observer, action, trace, quota=32 if quota == 64 else 64)
        with pytest.raises(ValueError, match='external challenger quota'):
            q.validate_scientific_decision('search', observer, action, trace)


def _mock_panel(plan):
    """Synthetic quota bookkeeping built from oldactions, never execution evidence."""
    rows = []
    for spec in plan['timed_requests']:
        fixture = plan['fixtures'][spec['fixture_index']]
        old = q.read(q.ROOT / fixture['prior_search_reference']['path'])['clock']['result']
        trace = _synthetic_callback_receipts(old['diagnostics'])
        if spec['arm'] == 'search':
            quota = plan['challenger_quota']
            trace['profile_id'] = c.profile_id(quota)
            phase = trace['phases'][0]
            phase['simulations'], phase['leaf_calls'] = quota, quota + 1
            phase['neural_callback_count'] = phase['neural_callback_rows'] = quota + 1
            phase['neural_callbacks'] = [q.digest(['a' * 64, 'b' * 64, 1])] * (quota + 1)
            trace['inference_calls'] = trace['inference_rows'] = quota + 1
        rows.append({**spec, 'action': old['action'], 'trace': trace, 'elapsed_seconds': .2})
    return rows


def test_complete_mock_panel_preserves_all_checks_and_reports_timing_failure(plan):
    rows = _mock_panel(plan)
    summary = q.validate_timed_panel(rows, plan)
    assert summary['timing_qualified'] and summary['historical_search_comparisons'] == 92
    rows[0]['elapsed_seconds'] = .701
    summary = q.validate_timed_panel(rows, plan)
    assert summary['status'] == 'timing_not_qualified' and not summary['timing_qualified']
    assert len(summary['timing_failures']) == 1 and summary['requests'] == 184
    rows[-1]['trace']['observer_seed'] += 1
    with pytest.raises(ValueError, match='seed'):
        q.validate_timed_panel(rows, plan)


def test_panel_rejects_incomplete_count_and_nonfinite_or_late_receipts(plan):
    with pytest.raises(ValueError, match='incomplete'):
        q.validate_timed_panel([], plan)
    for elapsed in (float('nan'), float('inf'), -1, 1.001):
        rows = _mock_panel(plan); rows[0]['elapsed_seconds'] = elapsed
        with pytest.raises(ValueError, match='receipt timing'):
            q.validate_timed_panel(rows, plan)


def test_scientific_comparison_excludes_capture_but_preserves_work():
    observer, action, trace = _late_search_fixture(123)
    captured = deepcopy(trace)
    captured['phases'][0]['inference'] = [{'tokens': [1.0], 'output83': [2.0]}]
    assert q.scientific_result('reference', action, trace) == q.scientific_result('reference', action, captured)
    captured['phases'][0]['leaf_calls'] += 1
    assert q.scientific_result('reference', action, trace) != q.scientific_result('reference', action, captured)


@pytest.mark.parametrize('quota', [32, 64])
def test_worker_binds_exact_parent_quota_and_profile(quota):
    payload = rules.observation_payload(q.reference._initial())
    class Stub:
        request_deadline = None
        def __init__(self, count): self.count = count
        def choose(self, observer, diagnostic_capture=False):
            return rules.legal_actions(rules.state_from_observation(observer))[0], {'profile_id': c.profile_id(self.count)}
    context = {'adapters': {'search': Stub(quota), 'reference': Stub(16)},
               'quota': quota, 'diagnostic_capture': False, 'request_started': None}
    for arm, count in [('search', quota), ('reference', 16)]:
        result = worker.handle(context, {'player': arm, 'observer': payload})
        assert result['simulations_quota'] == count
        assert result['profile_id'] == c.profile_id(count)
        assert result['profile_sha256'] == c.profile_sha256(count)
        assert result['wrapper_id'] == c.wrapper_id(count)
        assert context['adapters'][arm].request_deadline is None


def test_parent_readiness_requires_distinct_roles_and_exact_quotas():
    import os
    metadata = {**adapters.metadata_identity(32), 'pid': 10, 'threads': 1,
                'sources': {}, 'ruleset_id': q.guard.RULESET_ID,
                'catalogue_sha256': q.guard.CATALOGUE_SHA256, 'turn_contract': q.guard.TURN_CONTRACT,
                'native_pids': {'search': 11, 'reference': 12}}
    tree = [{'role': 'python', 'pid': 10}, {'role': 'search', 'pid': 11}, {'role': 'reference', 'pid': 12}]
    assert adapters.validate_ready_metadata(metadata, expected_sources={}, worker_pid=10,
        process_tree=tree, expected_quota=32)['status'] == 'passed'
    for bad in (64, None, True):
        with pytest.raises(ValueError):
            adapters.validate_ready_metadata(metadata, expected_sources={}, worker_pid=10,
                process_tree=tree, expected_quota=bad)
    with pytest.raises(ValueError, match='three exact'):
        adapters.validate_ready_metadata(metadata, expected_sources={}, worker_pid=10,
            process_tree=tree[:-1], expected_quota=32)


def test_challenger_native_constructor_shares_model_but_starts_own_process(monkeypatch):
    import io
    calls = []
    class Proc:
        stdout = io.StringIO('{"kind":"ready","bridge":"observer_only_fixed_work_v1",'
                             '"source_pin":"1828507b8d2f0e02bc03d102b9c6d29a54fa8d57"}\n')
    monkeypatch.setattr(adapters.subprocess, 'Popen', lambda *a, **kw: calls.append((a, kw)) or Proc())
    monkeypatch.setattr(adapters.atexit, 'register', lambda method: None)
    shared = SimpleNamespace(model=object(), reference=object())
    profile = {'opponent_id': 'splendorust', 'deterministic_fixed_work': True,
               'artifacts': [], 'native_binary': 'fake/challenger.exe', 'iterations': 32}
    result = adapters.SearchOpponent(profile, shared_from=shared)
    assert result.model is shared.model and result.reference is shared.reference
    assert result.proc is not shared and len(calls) == 1
    assert result.inference_calls == result.inference_rows == 0 and result.callbacks == []


def test_both_wrappers_keep_parent_internal_deadline_and_callback_guards(monkeypatch):
    calls = []
    monkeypatch.setattr(adapters.Opponent, 'choose', lambda *args: calls.append(args))
    policy = object.__new__(adapters.SearchOpponent)
    policy.request_deadline = 0.0
    with pytest.raises(TimeoutError, match='ONE_SECOND_INTERNAL_DEADLINE'):
        policy.choose({})
    assert not calls
    policy.profile = {'iterations': 16}
    with pytest.raises(TimeoutError, match='ONE_SECOND_INTERNAL_DEADLINE'):
        policy.reference_choose({})
    assert not calls


def test_actual_root_encoder_cost_check_rejects_old_card_face(plan):
    import struct
    for fixture in plan['counterexamples']:
        if 'corrected_face' not in fixture['tags']:
            continue
        state = q.guard.decode_training_snapshot(fixture['snapshot'])
        def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
        tokens = [0.] * (31 * 48)
        for tier in (1, 2, 3):
            for slot, card in enumerate(state.visible_tier_cards[tier]):
                index = (3 + 4 * (tier - 1) + slot) * 48
                tokens[index:index + 5] = [f32(card.cost.get(color, 0) * f32(.1)) for color in bridge.COLORS]
        trace = {'forced_pass': False, 'phases': [{'neural_callback_count': 1,
                                                 'inference': [{'tokens': tokens}]}]}
        assert q._actual_encoded_costs(state, trace, corrected_face=True)['visible_cards_checked'] == 12
        tokens[3 * 48] += .1
        with pytest.raises(ValueError, match='input card costs'):
            q._actual_encoded_costs(state, trace, corrected_face=True)
    with pytest.raises(ValueError, match='did not reach'):
        q._actual_encoded_costs(state, {'forced_pass': True, 'phases': []}, corrected_face=True)
