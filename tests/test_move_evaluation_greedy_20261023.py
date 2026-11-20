"""Model-free greedy boundary, mapping and native-reference qualification."""
import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.move_evaluation_20261023 import greedy_adapter as g


def observer():
    return rules.observation_payload(rules.CorrectedEnv(seed=712).initial_state())


@pytest.mark.parametrize('action,index', [
    ({'kind': 'Take', 'tokens': [1, 0, 1, 0, 1]}, 21),
    ({'kind': 'Take', 'tokens': [0, 0, 0, 2, 0]}, 35),
    ({'kind': 'ReserveVisible', 'index': 11}, 48),
    ({'kind': 'ReserveDeck', 'tier': 2}, 51),
    ({'kind': 'BuyVisible', 'index': 11}, 63),
    ({'kind': 'BuyReserved', 'index': 2}, 66),
])
def test_independent_canonical_vocabulary(action, index):
    assert g.canonical_index(action) == index


@pytest.mark.parametrize('action', [
    {'kind': 'Take', 'tokens': [2, 1, 0, 0, 0]},
    {'kind': 'Take', 'tokens': [True, 0, 0, 0, 0]},
    {'kind': 'BuyReserved', 'index': True},
    {'kind': 'ReserveDeck', 'tier': 3}, {'kind': 'Noble', 'index': 0},
])
def test_bad_main_vocabulary_rejected(action):
    with pytest.raises(ValueError):
        g.canonical_index(action)


def test_first_max_respects_retained_order_on_tie():
    actions = [{'kind': 'BuyReserved', 'index': 2}, {'kind': 'BuyVisible', 'index': 0}]
    logits = [-3.0] * 67
    logits[66] = logits[52] = 100.0
    assert g.first_max(logits, actions) == 0
    logits[52] = 101.0
    assert g.first_max(logits, actions) == 1


@pytest.mark.parametrize('logits', [[0.0] * 66, [float('nan')] * 67, [float('inf')] * 67])
def test_nonfinite_or_malformed_policy_fails(logits):
    with pytest.raises(ValueError):
        g.first_max(logits, [{'kind': 'BuyReserved', 'index': 0}])


def fake_greedy(mutate=lambda x: x):
    agent = object.__new__(g.GreedyOpponent)
    agent.boundary = lambda: None
    calls = []
    def exchange(request, *, capture=False):
        calls.append(request)
        legal = request['legal']
        main = request['observation']['phase']['kind'] == 'Main'
        branch = ('auxiliary_heuristic' if not main else 'single_choice' if len(legal) == 1
                  else 'late_heuristic' if request['observation']['turns'] >= 124 else 'neural_greedy')
        neural = branch == 'neural_greedy'
        agent.callbacks = [{'input_sha256': '1' * 64, 'output_sha256': '2' * 64}] if neural else []
        return mutate({'kind': 'result', 'action': legal[0], 'branch': branch,
                       'native_legal_count': len(legal), 'filtered': legal,
                       'canonical_indices': [g.canonical_index(a) for a in legal] if main else [],
                       'selected_index': 0, 'canonical_logits': [0.0] * 67 if neural else None,
                       'simulations': 0, 'leaf_calls': int(neural)})
    agent.exchange = exchange
    return agent, calls


def test_choose_lawful_payload_seed_and_deterministic_trace():
    agent, calls = fake_greedy()
    payload = observer()
    action, trace = agent.choose(payload)
    assert action in rules.legal_actions(rules.state_from_observation(payload))
    assert trace['observer_seed'] == int(digest(payload)[:16], 16)
    assert trace['profile_id'] == g.PROFILE_ID and trace['simulations'] == 0
    assert all(set(c) == {'op', 'observation', 'legal', 'seed'} for c in calls)
    assert agent.choose(deepcopy(payload)) == (action, trace)
    assert 'canonical_logits' not in trace['phases'][0]


def test_assessment_seed_or_hidden_payload_field_cannot_enter():
    agent, calls = fake_greedy()
    payload = observer()
    payload['assessment_seed'] = 123
    with pytest.raises(ValueError, match='allowlist'):
        agent.choose(payload)
    assert not calls


@pytest.mark.parametrize('key,value', [
    ('simulations', 1), ('leaf_calls', 0), ('branch', 'late_heuristic'),
    ('selected_index', True), ('canonical_indices', []), ('filtered', []),
    ('canonical_logits', [float('nan')] * 67), ('action', {'kind': 'BuyVisible', 'index': 99}),
])
def test_malformed_native_result_fails_closed(key, value):
    def mutate(result):
        result[key] = value
        return result
    agent, _ = fake_greedy(mutate)
    with pytest.raises(ValueError):
        agent.choose(observer())


def test_no_import_time_model_and_exact_qualified_forward():
    assert g.GreedyOpponent._infer is g.Opponent._infer
    tree = ast.parse(Path(g.__file__).read_text())
    # All model loading is deferred into construction; no torch import at module top.
    assert all(not (isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names)) for n in tree.body)
    rust = (Path(g.__file__).parent / 'native_bridge/src/main.rs').read_text()
    assert '<NeuralAgent as PolicyValue<Canonical>>::evaluate(&mut agent,&o)' in rust
    assert 'rollout_action' not in rust


def test_native_model_free_contract_reference_and_corrected_costs(tmp_path):
    native = g.NativeOnly(tmp_path)
    payload = observer()
    try:
        contract = native.inspect(payload)
        assert contract['legal'] == bridge.main_actions(payload)
        assert contract['canonical_indices'] == [g.canonical_index(a) for a in contract['filtered']]
        assert len(contract['transfer_features']) == 392
        encoded = bridge.encode_observation(payload)
        cards = rules.corrected_cards_by_id()
        for slot, card_id in enumerate(encoded['market']):
            if card_id == 255:
                continue
            card = cards[bridge.RUST_CARD_IDS[card_id]]
            assert contract['transfer_features'][(1+2*slot)*7:(1+2*slot)*7+5] == [card.cost.get(c, 0) for c in bridge.COLORS]
        reference = native.native_reference(payload, [0.0] * 81)
        assert reference['selected_index'] == 0
        assert reference['action'] == contract['filtered'][0]
        assert not native.callbacks
    finally:
        native.close()


def test_native_reference_81_to_67_mapping(tmp_path):
    # Independently describe the public81 vocabulary, including take-mask order.
    native = g.NativeOnly(tmp_path)
    try:
        result = native.native_reference(observer(), list(range(81)))
        expected = [-100000000.0] * 67
        masks = [1, 2, 4, 8, 16, 3, 5, 9, 17, 6, 10, 18, 12, 20, 24,
                 7, 11, 19, 13, 21, 25, 14, 22, 26, 28]
        for i, mask in enumerate(masks):
            expected[mask] = 30 + i
        expected[32:37] = range(55, 60)
        expected[37:49] = range(12, 24)
        expected[49:52] = range(24, 27)
        expected[52:64] = range(12)
        expected[64:67] = range(27, 30)
        assert result['canonical_logits'] == expected
        assert result['selected_index'] == g.first_max(expected, result['filtered'])
        assert not native.callbacks
    finally:
        native.close()
