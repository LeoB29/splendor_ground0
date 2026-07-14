from __future__ import annotations

from splendor_ai.encoding import (
    canonical_legal_action_descriptor,
    canonical_legal_action_descriptors,
    canonical_public_belief_key,
    canonical_successor_action_key,
)
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, RESERVE_SOURCE_DECK, RESERVE_SOURCE_VISIBLE


def test_public_belief_hides_blind_identity_but_preserves_public_tier() -> None:
    state_a = SplendorEnv(seed=0).initial_state()
    state_b = SplendorEnv(seed=0).initial_state()
    state_c = SplendorEnv(seed=0).initial_state()
    state_a.players[1].reserve_card(
        Card("private-a", 1, "white", 0, {"blue": 1}),
        RESERVE_SOURCE_DECK,
    )
    state_b.players[1].reserve_card(
        Card("private-b", 1, "black", 5, {"white": 7}),
        RESERVE_SOURCE_DECK,
    )
    state_c.players[1].reserve_card(
        Card("private-c", 3, "black", 5, {"white": 7}),
        RESERVE_SOURCE_DECK,
    )

    assert canonical_public_belief_key(state_a, 0) == canonical_public_belief_key(state_b, 0)
    assert canonical_public_belief_key(state_a, 0) != canonical_public_belief_key(state_c, 0)


def test_public_belief_includes_public_purchased_card_identities() -> None:
    state_a = SplendorEnv(seed=0).initial_state()
    state_b = SplendorEnv(seed=0).initial_state()
    state_a.players[1].purchased_cards.append(
        Card("bought-a", 1, "white", 0, {"blue": 1})
    )
    state_b.players[1].purchased_cards.append(
        Card("bought-b", 1, "white", 0, {"blue": 1})
    )

    assert canonical_public_belief_key(state_a, 0) != canonical_public_belief_key(state_b, 0)


def test_public_belief_is_invariant_to_incidental_collection_order() -> None:
    state_a = SplendorEnv(seed=11).initial_state()
    state_b = state_a.copy_shallow()
    state_b.hidden_tier_decks[1].reverse()
    state_b.visible_tier_cards[1].reverse()
    state_b.nobles.reverse()
    state_b.players[0].reserve_card(
        Card("reserve-a", 1, "white", 0, {"blue": 1}),
        RESERVE_SOURCE_DECK,
    )
    state_b.players[0].reserve_card(
        Card("reserve-b", 1, "blue", 0, {"white": 1}),
        RESERVE_SOURCE_DECK,
    )
    state_a.players[0].reserve_card(
        state_b.players[0].reserved_cards[1],
        RESERVE_SOURCE_DECK,
    )
    state_a.players[0].reserve_card(
        state_b.players[0].reserved_cards[0],
        RESERVE_SOURCE_DECK,
    )

    assert canonical_public_belief_key(state_a, 0) == canonical_public_belief_key(state_b, 0)


def test_action_descriptors_are_invariant_to_market_slot_permutations() -> None:
    env = SplendorEnv(seed=7)
    state_a = env.initial_state()
    state_b = state_a.copy_shallow()
    state_b.visible_tier_cards[1].reverse()
    target = state_a.visible_tier_cards[1][0]
    action_a = Action(
        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        market_index=0,
        take_tokens=("gold",),
    )
    action_b = Action(
        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        market_index=next(
            index
            for index, card in enumerate(state_b.visible_tier_cards[1])
            if card.card_id == target.card_id
        ),
        take_tokens=("gold",),
    )

    assert canonical_legal_action_descriptor(state_a, action_a) == canonical_legal_action_descriptor(state_b, action_b)
    assert canonical_legal_action_descriptors(state_a, env.legal_actions(state_a)) == canonical_legal_action_descriptors(state_b, env.legal_actions(state_b))


def test_action_descriptor_uses_reserved_card_identity_not_slot() -> None:
    state_a = SplendorEnv(seed=0).initial_state()
    state_a.players[0].reserve_card(
        Card("reserve-a", 1, "white", 0, {"blue": 1}),
        RESERVE_SOURCE_DECK,
    )
    state_a.players[0].reserve_card(
        Card("reserve-b", 1, "blue", 0, {"white": 1}),
        RESERVE_SOURCE_DECK,
    )
    state_b = state_a.copy_shallow()
    state_b.players[0].reserved_cards.reverse()
    state_b.players[0].reserved_card_sources.reverse()

    action_a = Action(action_type=ActionType.BUY_RESERVED, reserved_index=0)
    action_b = Action(action_type=ActionType.BUY_RESERVED, reserved_index=1)

    assert canonical_legal_action_descriptor(state_a, action_a) == canonical_legal_action_descriptor(state_b, action_b)


def test_successor_action_key_deduplicates_equivalent_token_transitions() -> None:
    env = SplendorEnv(seed=0)
    state = env.initial_state()
    player = state.players[0]
    player.tokens.update({"white": 0, "blue": 1, "green": 0, "red": 1, "black": 3, "gold": 3})
    state.bank_tokens.update({"white": 3, "blue": 2, "green": 2, "red": 2, "black": 1, "gold": 0})
    actions = [
        action
        for action in env.legal_actions(state)
        if (
            action.action_type == ActionType.TAKE_TOKENS
            and (
                (
                    set(action.take_tokens) == {"white", "blue", "green"}
                    and action.return_tokens == ("green",)
                )
                or (
                    set(action.take_tokens) == {"white", "blue", "red"}
                    and action.return_tokens == ("red",)
                )
            )
        )
    ]

    assert len(actions) == 2
    assert canonical_legal_action_descriptor(state, actions[0]) != canonical_legal_action_descriptor(state, actions[1])
    assert canonical_successor_action_key(state, actions[0]) == canonical_successor_action_key(state, actions[1])
    assert canonical_public_belief_key(env.step(state, actions[0]), 1) == canonical_public_belief_key(env.step(state, actions[1]), 1)
