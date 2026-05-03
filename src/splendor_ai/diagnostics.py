"""Shared loop/progress diagnostics for gameplay and replay collection."""

from __future__ import annotations

from splendor_ai.engine.actions import Action
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.state import SplendorState


def state_signature(state: SplendorState) -> tuple[object, ...]:
    return (
        state.current_player,
        state.turn_index % 2,
        state.pending_round_end,
        tuple((color, state.bank_tokens[color]) for color in ALL_TOKEN_TYPES),
        tuple((tier, tuple(card.card_id for card in state.visible_tier_cards[tier])) for tier in (1, 2, 3)),
        tuple((tier, tuple(card.card_id for card in state.hidden_tier_decks[tier])) for tier in (1, 2, 3)),
        tuple((tier, state.deck_counts[tier]) for tier in (1, 2, 3)),
        tuple(noble.noble_id for noble in state.nobles),
        tuple(_player_signature(player) for player in state.players),
    )


def is_progress_transition(
    before: SplendorState,
    after: SplendorState,
    action: Action,
) -> bool:
    current_before = before.players[before.current_player]
    current_after = after.players[before.current_player]
    opponent_before = before.players[1 - before.current_player]
    opponent_after = after.players[1 - before.current_player]

    if action.action_type.name.startswith("BUY") or action.action_type.name.startswith("RESERVE"):
        return True
    if len(current_after.purchased_cards) != len(current_before.purchased_cards):
        return True
    if len(current_after.reserved_cards) != len(current_before.reserved_cards):
        return True
    if len(current_after.nobles) != len(current_before.nobles):
        return True
    if current_after.score != current_before.score:
        return True
    if current_after.bonuses != current_before.bonuses:
        return True
    if opponent_after.score != opponent_before.score:
        return True
    if opponent_after.bonuses != opponent_before.bonuses:
        return True
    return False


def _player_signature(player) -> tuple[object, ...]:
    return (
        player.player_id,
        player.score,
        tuple((color, player.tokens[color]) for color in ALL_TOKEN_TYPES),
        tuple((color, player.bonuses[color]) for color in TOKEN_COLORS),
        tuple(card.card_id for card in player.reserved_cards),
        tuple(card.card_id for card in player.purchased_cards),
        tuple(noble.noble_id for noble in player.nobles),
    )
