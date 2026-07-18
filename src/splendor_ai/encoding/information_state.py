"""Perfect-recall, information-legal history keys for regret-based learning."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, TypeAlias

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.state import SplendorState

from .public_belief import canonical_public_belief_key


PublicActionEvent: TypeAlias = tuple[object, ...]
InformationStateHistoryKey: TypeAlias = tuple[object, ...]


def canonical_information_state_history_key(
    state: SplendorState,
    player_id: int,
    public_history: Iterable[PublicActionEvent],
) -> InformationStateHistoryKey:
    """Return a perfect-recall key built only from legally available facts."""

    return (
        "splendor-information-state-history-v1",
        canonical_public_belief_key(state, player_id),
        tuple(public_history),
    )


def public_action_event(
    before: SplendorState,
    action: Action,
) -> PublicActionEvent:
    """Describe what becomes public when ``action`` is taken.

    A blind reserve exposes its tier but never its card identity. Buying a
    reserved card reveals the card, so that identity is public in the event.
    """

    actor = before.current_player
    target_id: str | None = None
    public_tier: int | None = None
    if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
        if action.tier is not None and action.market_index is not None:
            cards = before.visible_tier_cards.get(action.tier, [])
            if 0 <= action.market_index < len(cards):
                target_id = cards[action.market_index].card_id
    elif action.action_type == ActionType.BUY_RESERVED:
        if action.reserved_index is not None:
            cards = before.players[actor].reserved_cards
            if 0 <= action.reserved_index < len(cards):
                target_id = cards[action.reserved_index].card_id
    elif action.action_type == ActionType.RESERVE_DECK:
        public_tier = action.tier

    return (
        "splendor-public-action-v1",
        int(before.turn_index),
        int(actor),
        action.action_type.name,
        target_id,
        public_tier,
        _token_counts(action.take_tokens),
        _token_counts(action.return_tokens),
        _token_counts(action.spend_tokens),
        action.noble_id,
    )


def _token_counts(tokens: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    counts = Counter(tokens)
    return tuple((color, int(counts[color])) for color in ALL_TOKEN_TYPES)
