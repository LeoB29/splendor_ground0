"""Model-free codec, public-information and complete-action counterexamples."""
from copy import deepcopy
from collections import Counter
import ast

import numpy as np
import pytest

from tools.breck_screen_20261026 import adapter as a, reference as r
from tools.external_strength_screen_20261018 import rules
from tools import future_training_rules_20261021 as training


def initial(seed=41, seat=0):
    state = rules.CorrectedEnv(seed=seed).initial_state()
    state.current_player = seat
    state.start_player = seat
    return state


def tokens(state, seat, counts):
    player = state.players[seat]
    for color in a.TOKENS:
        amount = counts.get(color, 0)
        state.bank_tokens[color] += player.tokens[color] - amount
        player.tokens[color] = amount


def reserve(state, seat, tier, source='deck'):
    if source == 'deck':
        card = state.hidden_tier_decks[tier].pop()
        state.deck_counts[tier] -= 1
    else:
        card = state.visible_tier_cards[tier].pop()
    state.players[seat].reserve_card(card, source)
    return card


def grant_bonus(state, seat, color, count):
    player = state.players[seat]
    for _ in range(count):
        located = next((tier, i, card) for tier, deck in state.hidden_tier_decks.items()
                       for i, card in enumerate(deck) if card.bonus_color == color and card.points == 0)
        tier, index, card = located
        state.hidden_tier_decks[tier].pop(index)
        state.deck_counts[tier] -= 1
        player.purchased_cards.append(card)
        player.bonuses[color] += 1


def observer(state):
    payload = rules.serialize_snapshot(state)
    training.decode_training_snapshot(payload)
    return rules.observation_payload(state)


def assert_parity(obs):
    p = a.prepare(obs)
    assert np.array_equal(p['features'], r.encode(obs))
    assert np.array_equal(p['source_mask'], r.source_mask(obs))
    state = rules.state_from_observation(obs)
    legal = rules.legal_actions(state)
    assert all(rules.deserialize_action(action) in legal for action in p['actions'] if action is not None)
    assert np.array_equal(p['legal_mask'], np.array([x is not None for x in p['actions']]))
    return p


def test_all_source_index_contract_and_catalogue():
    assert r.validate_mapping_contract()['static_indices_checked'] == 141
    cards, nobles = r.catalogue_mapping()
    assert len(cards) == 90 and len(nobles) == 10
    assert set(rules.CORRECTED_COSTS) <= set(cards)


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('seed', [1, 41, 99])
def test_initial_exact_source_features_masks_and_corrected_incomplete_take_mask(seat, seed):
    p = assert_parity(observer(initial(seed, seat)))
    assert p['features'].dtype == np.float32 and p['features'].shape == (251,)
    assert p['legal_mask'].dtype == bool and p['legal_mask'].shape == (141,)
    assert not p['legal_mask'][55:96].any()  # Source allows short takes/standalone discard.
    assert p['source_mask'][55:95].any()
    assert p['legal_mask'].sum() == 30


@pytest.mark.parametrize('count', [7, 8, 9, 10])
def test_discard_aliases_exact_return_counts_and_no_global_rng_changes(count):
    state = initial()
    tokens(state, 0, {'white': 3, 'blue': 3, 'green': count-6})
    obs = observer(state)
    before = np.random.get_state()
    p = assert_parity(obs)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    for i, action in enumerate(p['actions'][:95]):
        if action is not None:
            assert len(action['return_tokens']) == a.ACTION_TABLE[i]['return_count']
            assert count + len(action['take_tokens']) - len(action['return_tokens']) <= 10
    assert np.array_equal(p['features'], a.prepare(obs)['features'])
    assert p['actions'] == a.prepare(obs)['actions']


def test_blind_private_ids_and_deck_order_cannot_change_completion_or_decision():
    state = initial()
    reserve(state, 1, 1)
    reserve(state, 1, 2)
    reserve(state, 0, 3)
    first = observer(state)
    alternative = state.copy_shallow()
    for tier in (1, 2):
        slot = tier-1
        alternative.players[1].reserved_cards[slot], alternative.hidden_tier_decks[tier][0] = (
            alternative.hidden_tier_decks[tier][0], alternative.players[1].reserved_cards[slot])
    for deck in alternative.hidden_tier_decks.values():
        deck.reverse()
    second = observer(alternative)
    assert first == second
    p = assert_parity(first)
    assert len(p['completion_assignments']) == 2
    assert p['completion'] == a.prepare(second)['completion']
    logits = lambda features: np.arange(141, dtype=np.float32)
    assert a.decide(first, logits) == a.decide(second, logits)
    full = rules.deserialize_snapshot(p['completion'])
    assert rules.observation_payload(full) == first


def test_public_and_own_reserves_preserved_and_not_zeroed():
    state = initial()
    own = reserve(state, 0, 1)
    public = reserve(state, 1, 2, 'visible')
    reserve(state, 1, 3)
    obs = observer(state)
    p = assert_parity(obs)
    completed = rules.deserialize_snapshot(p['completion'])
    assert completed.players[0].reserved_cards[0] == own
    assert completed.players[1].reserved_cards[0] == public
    assert p['features'][157+13:157+24].any()
    assert p['features'][204+24:204+35].any()


@pytest.mark.parametrize('mutation', ['true_private', 'seed', 'legacy_rules', 'extra_history'])
def test_observation_rejects_hidden_or_unapproved_inputs(mutation):
    state = initial()
    reserve(state, 1, 1)
    obs = observer(state)
    if mutation == 'true_private':
        obs['observation']['players'][1]['reserves'][0]['card_id'] = state.players[1].reserved_cards[0].card_id
    elif mutation == 'legacy_rules':
        obs['ruleset_id'] = 'historical'
    elif mutation == 'seed':
        obs['observation']['seed'] = 41
    else:
        obs['history'] = []
    with pytest.raises(ValueError):
        a.prepare(obs)


def test_reserved_gold_return_and_colored_first_payment():
    state = initial()
    tokens(state, 0, {'white': 2, 'blue': 2, 'green': 2, 'red': 2, 'black': 1, 'gold': 1})
    reserve(state, 0, 1)
    p = assert_parity(observer(state))
    reserve_actions = [x for x in p['actions'][126:] if x is not None]
    assert reserve_actions and all(x['take_tokens'] == ['gold'] and len(x['return_tokens']) == 1 for x in reserve_actions)
    assert all('gold' not in x['return_tokens'] for x in reserve_actions)
    for index, action in enumerate(p['actions'][96:126], start=96):
        if action is not None:
            d = a.ACTION_TABLE[index]
            card = (state.visible_tier_cards[d['tier']][d['market_index']] if d['family'] == 'buy_visible'
                    else state.players[0].reserved_cards[d['reserved_index']])
            assert Counter(action['spend_tokens']) == Counter(a._spend(state.players[0], card))


def test_multiple_nobles_choose_single_lexical_id_including_nonpurchase():
    state = initial()
    available = rules.corrected_nobles_by_id()
    # Noble faces with requirements3 in threecolors need only zero-pointcards.
    selected = [n for n in available.values() if max(n.requirement.values()) == 3][:3]
    state.nobles = selected
    requirements = {c: max(n.requirement.get(c, 0) for n in selected) for c in a.COLORS}
    for color, count in requirements.items():
        grant_bonus(state, 0, color, count)
    p = assert_parity(observer(state))
    expected = min(n.noble_id for n in selected)
    assert all(action['noble_id'] == expected for action in p['actions'] if action is not None)
    assert any(action is not None and action['action_type'] == 'TAKE_TOKENS' for action in p['actions'])


def test_forced_pass_is_only_corrected_pass_and_never_standalone_discard():
    state = initial()
    tokens(state, 0, {'white': 4, 'blue': 4, 'green': 2})
    tokens(state, 1, {'green': 2, 'red': 4, 'black': 4})
    for tier in (1, 2, 3):
        state.hidden_tier_decks[tier].extend(state.visible_tier_cards[tier])
        state.visible_tier_cards[tier] = []
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    for _ in range(3):
        index = next(i for i, card in enumerate(state.hidden_tier_decks[3])
                     if 'gold' in a._spend(state.players[0], card))
        card = state.hidden_tier_decks[3].pop(index)
        state.deck_counts[3] -= 1
        state.players[0].reserve_card(card, 'deck')
    p = assert_parity(observer(state))
    assert np.flatnonzero(p['legal_mask']).tolist() == [95]
    assert p['actions'][95]['action_type'] == 'PASS'
    assert p['actions'][95]['return_tokens'] == []


@pytest.mark.parametrize('field', ['features', 'mask', 'logits', 'selected_index', 'completion_sha256', 'action', 'observer_sha256'])
def test_saved_trace_tamper_rejected_without_neural_call(field):
    obs = observer(initial())
    result = a.decide(obs, lambda _: np.arange(141, dtype=np.float32))
    assert a.validate_result(result, obs)['status'] == 'passed'
    bad = deepcopy(result)
    if field in ('features', 'logits'):
        bad['diagnostics'][field][0] += 1.0
    elif field == 'mask':
        bad['diagnostics']['mask'][0] = not bad['diagnostics']['mask'][0]
    elif field == 'selected_index':
        bad['diagnostics'][field] = 0
    elif field == 'completion_sha256':
        bad['diagnostics'][field] = '0'*64
    elif field == 'action':
        bad['action']['tier'] = 2
    else:
        bad[field] = '0'*64
    with pytest.raises(ValueError):
        a.validate_result(bad, obs)


@pytest.mark.parametrize('bad', [np.zeros(140, dtype=np.float32), np.zeros(141, dtype=np.float64),
                               np.full(141, np.nan, dtype=np.float32), np.full(141, np.inf, dtype=np.float32)])
def test_policy_contract_rejects_invalid_output(bad):
    with pytest.raises(ValueError):
        a.decide(observer(initial()), lambda _: bad)


def test_tie_uses_lowest_legal_output_and_trace_canonical_types():
    obs = observer(initial())
    result = a.decide(obs, lambda _: np.zeros(141, dtype=np.float32))
    assert result['diagnostics']['selected_index'] == 0
    result['diagnostics']['selected_index'] = False
    with pytest.raises(ValueError):
        a.validate_result(result, obs)


def test_reference_does_not_construct_decks_or_import_gui_modules(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('Model loading is not allowed in model-free codec checks')
    monkeypatch.setattr(r, 'load_policy', fail)
    obs = observer(initial())
    assert_parity(obs)
    tree = ast.parse(r.__file__ and open(r.__file__, encoding='utf-8').read())
    assert not any(isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('Splendor')
                   for node in ast.walk(tree))


def test_actual_worker_handle_fake_policy_lawful_contract_and_failure_boundary():
    from tools.breck_screen_20261026 import worker
    class FakePolicy:
        calls = 0
        def predict(self, features):
            self.calls += 1
            assert features.shape == (251,) and features.dtype == np.float32
            return np.zeros(141, dtype=np.float32)
    policy = FakePolicy()
    context = {'policy': policy, 'diagnostic_capture': False, 'component_profiles': [],
               'request_read_complete_ns': 123}
    obs = observer(initial())
    result = worker.handle(context, {'player': 'breck', 'observer': obs})
    assert policy.calls == 1 and a.validate_result(result, obs)['status'] == 'passed'
    assert context['component_profiles'][-1]['handle_failed'] is False
    invalid = deepcopy(obs)
    invalid['catalogue_sha256'] = '0'*64
    with pytest.raises(ValueError):
        worker.handle(context, {'player': 'breck', 'observer': invalid})
    assert policy.calls == 1
    assert context['component_profiles'][-1]['handle_failed'] is True
    with pytest.raises(ValueError, match='allowlist'):
        worker.handle(context, {'player': 'breck', 'observer': obs, 'snapshot': {}})
    assert policy.calls == 1


def test_actual_worker_initialize_calls_source_guard_before_fake_weight_load(monkeypatch, tmp_path):
    from tools.breck_screen_20261026 import worker, network, numerical_runtime
    events, saved = [], []
    class FakePolicy:
        def __init__(self):
            events.append('weights')
    monkeypatch.setattr(worker, 'verify_runtime_sources', lambda: events.append('guard') or {'status': 'fixture'})
    monkeypatch.setattr(network, 'FrozenPolicy', FakePolicy)
    monkeypatch.setattr(worker.c, 'OUT', tmp_path)
    monkeypatch.setattr(worker.c, 'read', lambda path: {'profile_id': a.PROFILE_ID,'numpy_blas_runtime':{'fixture':True}})
    monkeypatch.setattr(numerical_runtime,'verify_runtime',lambda expected,require_one_thread:
        events.append('numerical') or {'contract':expected})
    monkeypatch.setattr(worker.c, 'file_hash', lambda path: 'fixture-profile-hash')
    monkeypatch.setattr(worker, 'save_worker', lambda path, value: saved.append((path.name, value)))
    monkeypatch.setenv('STUDY_WORKER_DIR', str(tmp_path/'worker'))
    monkeypatch.setenv('STUDY_BOUNDARY_PROFILE', '1')
    context = worker.initialize()
    assert events == ['guard', 'numerical', 'weights']
    assert len([name for name, _ in saved if name == 'runtime_sources.json']) == 1
    assert context['metadata']['sources'] == {'status': 'fixture'}
    assert context['metadata']['ruleset_id'] == rules.RULESET_ID
