"""No model load or rollouts: fixed state fixtures exercise the feature contract."""
import copy
import numpy as np
import pytest

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE
from splendor_ai.encoding.action_features import encode_legal_action_features
from . import encoder


def rich_state(seat):
    state = SplendorEnv(seed=17).initial_state()
    state.current_player = seat
    state.turn_index = 43
    state.players[seat].tokens.update(white=2, blue=2, green=1, red=1, black=1, gold=3)
    state.players[seat].bonuses.update(white=3, blue=3, green=3, red=3, black=3)
    state.players[seat].score = 13
    state.players[1-seat].tokens.update(white=1, blue=2, green=1, red=0, black=2, gold=2)
    state.players[1-seat].bonuses.update(white=2, blue=1, green=2, red=3, black=0)
    state.players[1-seat].score = 12
    state.players[seat].reserve_card(state.hidden_tier_decks[2].pop(), RESERVE_SOURCE_DECK)
    state.players[1-seat].reserve_card(state.hidden_tier_decks[1].pop(), RESERVE_SOURCE_DECK)
    state.players[1-seat].reserve_card(state.hidden_tier_decks[3].pop(), RESERVE_SOURCE_VISIBLE)
    return state


@pytest.mark.parametrize('seat', [0, 1])
@pytest.mark.parametrize('rich', [False, True])
def test_exact_features_and_supplied_order(seat, rich):
    state = rich_state(seat) if rich else SplendorEnv(seed=seat).initial_state()
    state.current_player = seat
    actions = SplendorEnv().legal_actions(state)
    actions = list(reversed(actions)) + [Action(action_type=ActionType.PASS)]
    expected = np.asarray([r.vector for r in encode_legal_action_features(state, actions)], dtype=np.float32)
    np.testing.assert_array_equal(encoder.action_array(state, actions), expected)
    assert encoder.action_array(state, []).shape == (0, 79)


def test_opponent_private_reserve_and_deck_identity_invariance():
    state = rich_state(0)
    changed = copy.deepcopy(state)
    changed.players[1].reserved_cards[0] = changed.hidden_tier_decks[2][-1]
    for cards in changed.hidden_tier_decks.values():
        cards.reverse()
    actions = SplendorEnv().legal_actions(state)
    for before, after in zip(encoder.arrays(state, actions), encoder.arrays(changed, actions), strict=True):
        np.testing.assert_array_equal(before, after)


def test_all_action_types_gold_spend_returns_and_noble_choices_present():
    state = rich_state(0)
    actions = SplendorEnv().legal_actions(state) + [Action(action_type=ActionType.PASS)]
    assert {a.action_type for a in actions} == set(encoder.TYPE_INDEX)
    assert any(a.return_tokens for a in actions)
    assert any('gold' in a.spend_tokens for a in actions)
    # Explicit noble choice checks encoder semantics even for a fixture whose
    # selected noble is not available for that exact legal purchase.
    buy = next(a for a in actions if a.action_type == ActionType.BUY_VISIBLE)
    from dataclasses import replace
    actions.append(replace(buy, noble_id=state.nobles[0].noble_id))
    expected = np.asarray([r.vector for r in encode_legal_action_features(state, actions)], dtype=np.float32)
    np.testing.assert_array_equal(encoder.action_array(state, actions), expected)


def test_no_visible_cards_and_no_cross_state_cache():
    state = rich_state(1)
    state.visible_tier_cards = {1: [], 2: [], 3: []}
    actions = SplendorEnv().legal_actions(state)
    for gold in (0, 3, 1):
        state.players[1].tokens['gold'] = gold
        expected = np.asarray([r.vector for r in encode_legal_action_features(state, actions)], dtype=np.float32)
        np.testing.assert_array_equal(encoder.action_array(state, actions), expected)
