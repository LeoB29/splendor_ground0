"""Lawful arithmetic, masking and canonical-alias invariants."""
import numpy as np
import pytest
from . import features as f


def example():
    obs = np.zeros(295, dtype=np.float32)
    obs[189:195] = np.array([1, 0, 0, 0, 0, 1]) / 10
    obs[195] = 1 / 7
    obs[33] = 1
    obs[41:46] = np.array([3, 2, 0, 0, 0]) / 7
    obs[12] = 1
    obs[14:19] = np.array([3, 3, 0, 0, 0]) / 4
    actions = np.zeros((3, 79), dtype=np.float32)
    actions[0, 0] = 1
    actions[1, 1] = 1
    actions[1, 37] = actions[1, 55] = 2 / 4
    actions[2, 2] = actions[2, 6] = actions[2, 9] = 1
    actions[2, 41] = actions[2, 59] = 1 / 5
    return obs, actions


def test_exact_layout_shared_gold_and_reservation():
    obs, actions = example()
    x = f.concrete_features(obs, actions)
    assert x.shape == (3, 465) and x.dtype == np.float32
    np.testing.assert_array_equal(x[:, :295], np.broadcast_to(obs, (3, 295)))
    np.testing.assert_array_equal(x[:, 295:374], actions)
    # Missing white1 + blue2 - ONE gold =2; taking blue2 leaves0.
    np.testing.assert_allclose(x[:, 387:391], [[.2, .2, 1, 0], [.2, 0, 1, 0], [.2, .1, 1, 1]])
    assert x[2, 386] == np.float32(1 / 3)
    np.testing.assert_allclose(x[:, 459:461], np.full((3, 2), .25))
    assert not x[:, 391:459].any()


def test_purchase_retains_identity_and_tracks_discount_score_and_noble():
    obs = np.zeros(295, dtype=np.float32)
    obs[33] = 1
    obs[200] = 12 / 20
    obs[12] = 1
    obs[14] = 1 / 4
    action = np.zeros((1, 79), dtype=np.float32)
    action[0, [4, 6, 9, 19]] = 1
    action[0, 61] = 4 / 8
    x = f.concrete_features(obs, action)[0]
    assert x[380] == np.float32(1 / 7)
    assert x[385] == np.float32(16 / 20)
    np.testing.assert_array_equal(x[387:391], [0, 0, 0, 0])
    np.testing.assert_allclose(x[459:461], [.05, 0])
    assert x[33] == 1  # Original known identity remains, rather than invented refill.


def test_canonical_mean_no_alias_mass_and_member_order_invariance():
    obs, actions = example()
    expected = f.concrete_features(obs, actions)
    got = f.group_features(obs, actions, [[1, 2], [0]])
    np.testing.assert_allclose(got[0], expected[[1, 2]].mean(axis=0))
    np.testing.assert_array_equal(got[1], expected[0])
    np.testing.assert_array_equal(got, f.group_features(obs, actions, [[2, 1], [0]]))
    with pytest.raises(ValueError, match="Repeated"):
        f.group_features(obs, actions, [[0, 0]])
    with pytest.raises(ValueError, match="Invalid"):
        f.group_features(obs, actions, [[]])


def test_net_flow_consistency_and_nonfinite_inputs_rejected():
    obs, actions = example()
    actions[1, 55] = 0
    with pytest.raises(ValueError, match="net flow"):
        f.concrete_features(obs, actions)
    obs[1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        f.concrete_features(obs, actions)


def test_ties_and_empty_support_are_deterministic():
    assert f.first_argmax([1., 1., 0.]) == 0
    assert f.first_argmax([]) is None
    with pytest.raises(ValueError):
        f.first_argmax([float("nan")])


def game_state():
    from splendor_ai.engine.env import SplendorEnv
    from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
    from splendor_ai.engine.state import RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE
    state = SplendorEnv(seed=414).initial_state()
    actor = state.players[0]
    for color, number in zip(ALL_TOKEN_TYPES, (2, 2, 2, 1, 1, 2), strict=True):
        actor.tokens[color] = number
        state.bank_tokens[color] -= number
    actor.bonuses.update({color: 1 for color in TOKEN_COLORS})
    actor.reserve_card(state.hidden_tier_decks[1].pop(0), RESERVE_SOURCE_DECK)
    state.players[1].reserve_card(state.hidden_tier_decks[2].pop(0), RESERVE_SOURCE_VISIBLE)
    state.players[1].reserve_card(state.hidden_tier_decks[2].pop(0), RESERVE_SOURCE_DECK)
    for tier in (1, 2, 3):
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    return state


def test_engine_afterstates_and_independent_card_shortfalls():
    from splendor_ai.encoding import encode_action_features, encode_public_observation_tensor
    from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
    from splendor_ai.engine.env import SplendorEnv
    from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE
    state = game_state()
    legal = SplendorEnv().legal_actions(state)
    # One representative per action type, plus concrete gold-payment/return cases.
    chosen = {a.action_type: a for a in legal}
    actions = list(chosen.values())
    actions += [next(a for a in legal if "gold" in a.spend_tokens),
                next(a for a in legal if a.return_tokens)]
    assert len(chosen) == 5
    obs = encode_public_observation_tensor(state, 0).vector
    raw = np.asarray([encode_action_features(state, a).vector for a in actions], np.float32)
    x = f.concrete_features(obs, raw)
    known = [card for tier in (1, 2, 3) for card in state.visible_tier_cards[tier]]
    known += state.players[0].reserved_cards + [None, None]
    known += [state.players[1].reserved_cards[0], None, None]
    before = state.players[0]
    for index, action in enumerate(actions):
        after = SplendorEnv().step(state, action)
        player = after.players[0]
        expected = ([player.tokens[c]/10 for c in ALL_TOKEN_TYPES]
                    + [player.bonuses[c]/7 for c in TOKEN_COLORS]
                    + [player.score/20, len(player.reserved_cards)/3])
        np.testing.assert_allclose(x[index, 374:387], expected, atol=1e-7)
        market = {c.card_id for cards in after.visible_tier_cards.values() for c in cards}
        own = {c.card_id for c in player.reserved_cards}
        opponent = {c.card_id for i, c in enumerate(after.players[1].reserved_cards)
                    if after.players[1].reserved_card_source(i) == RESERVE_SOURCE_VISIBLE}
        for slot, card in enumerate(known):
            if card is None:
                np.testing.assert_array_equal(x[index, 387+4*slot:391+4*slot], np.zeros(4))
                continue
            def remaining(p):
                deficits = sum(max(0, card.cost.get(c, 0)-p.tokens[c]-p.bonuses[c]) for c in TOKEN_COLORS)
                return max(0, deficits-p.tokens["gold"])/10
            expected = [remaining(before), remaining(player),
                        int(card.card_id in market | own | opponent), int(card.card_id in own)]
            np.testing.assert_allclose(x[index, 387+4*slot:391+4*slot], expected, atol=1e-7)
        for slot, noble in enumerate(state.nobles):
            expected = [sum(max(0, noble.requirement.get(c, 0)-p.bonuses[c]) for c in TOKEN_COLORS)/20
                        for p in (before, player)]
            np.testing.assert_allclose(x[index, 459+2*slot:461+2*slot], expected, atol=1e-7)


def test_hidden_decks_and_opponent_private_card_identity_are_invisible():
    import copy
    from splendor_ai.encoding import encode_action_features, encode_public_observation_tensor
    from splendor_ai.engine.env import SplendorEnv
    state = game_state()
    changed = copy.deepcopy(state)
    for tier in (1, 2, 3):
        changed.hidden_tier_decks[tier].reverse()
    hidden = changed.players[1].reserved_cards[1]
    changed.players[1].reserved_cards[1] = changed.hidden_tier_decks[2][0]
    changed.hidden_tier_decks[2][0] = hidden
    encoded = []
    for value in (state, changed):
        actions = SplendorEnv().legal_actions(value)
        observation = encode_public_observation_tensor(value, 0).vector
        raw = np.asarray([encode_action_features(value, a).vector for a in actions], np.float32)
        encoded.append(f.concrete_features(observation, raw))
    np.testing.assert_array_equal(*encoded)
