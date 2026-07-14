"""Canonical, observer-relative public-belief representations.

The legacy flat observation tensor is kept stable for historical checkpoints.
New search and learning code must instead use this module.  Its keys contain
all information available to the observing player, no information from the
hidden deck order, and no incidental ordering from market/reserve lists.
"""

from __future__ import annotations

from collections import Counter
from typing import TypeAlias

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.state import (
    Card,
    PlayerState,
    RESERVE_SOURCE_DECK,
    RESERVE_SOURCE_VISIBLE,
    SplendorState,
)

CanonicalKey: TypeAlias = tuple[object, ...]

_BOARD_TIERS = (1, 2, 3)


def canonical_public_belief_key(
    state: SplendorState,
    player_id: int,
) -> CanonicalKey:
    """Return the complete legal information set for ``player_id``.

    Card ids are included only when the observer can know them.  A blind
    reserve's tier is public (the reserve-deck action names its tier), while
    its identity remains hidden.  Collections with no game-semantic order are
    sorted so a cosmetic permutation cannot create a new information state.
    """

    if player_id not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1, got {player_id}")

    player = state.players[player_id]
    opponent = state.players[1 - player_id]
    winner = (
        "draw"
        if state.terminal and state.winner is None
        else "self"
        if state.winner == player_id
        else "opponent"
        if state.terminal
        else None
    )

    return (
        "splendor-public-belief-v1",
        state.current_player == player_id,
        state.start_player == player_id,
        state.turn_index,
        state.pending_round_end,
        state.terminal,
        winner,
        _token_counts(state.bank_tokens),
        tuple((tier, int(state.deck_counts[tier])) for tier in _BOARD_TIERS),
        tuple(sorted(noble.noble_id for noble in state.nobles)),
        tuple(
            (
                tier,
                tuple(sorted(card.card_id for card in state.visible_tier_cards[tier])),
            )
            for tier in _BOARD_TIERS
        ),
        _player_key(player, observer_is_owner=True),
        _player_key(opponent, observer_is_owner=False),
    )


def canonical_legal_action_descriptor(
    state: SplendorState,
    action: Action,
    player_id: int | None = None,
) -> CanonicalKey:
    """Describe a legal action independently of transient list positions.

    Visible and reserved-card actions refer to the card id, rather than a
    market or reserve index.  Token tuples are represented as color counts.
    This descriptor is suitable for binding a tree edge back to a freshly
    determinized state and for deduplicating procedural action encodings.
    """

    actor = state.current_player if player_id is None else player_id
    if actor not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1, got {actor}")

    target = _action_target(state, action, actor)
    return (
        action.action_type.name,
        target.card_id if target is not None else None,
        action.tier if action.action_type == ActionType.RESERVE_DECK else None,
        _token_counts_from_tuple(action.take_tokens),
        _token_counts_from_tuple(action.return_tokens),
        _token_counts_from_tuple(action.spend_tokens),
        action.noble_id,
    )


def canonical_legal_action_descriptors(
    state: SplendorState,
    legal_actions: list[Action],
    player_id: int | None = None,
) -> tuple[CanonicalKey, ...]:
    """Return sorted action descriptors for an information-state key.

    A duplicate descriptor represents the same game-semantic action.  The
    caller may retain one concrete action when applying it to a state.
    """

    return tuple(
        sorted(
            {
                canonical_legal_action_descriptor(state, action, player_id)
                for action in legal_actions
            }
        )
    )


def canonical_successor_action_key(
    state: SplendorState,
    action: Action,
    player_id: int | None = None,
) -> CanonicalKey:
    """Return the deterministic public part of an action's successor.

    The engine represents some identical token transitions procedurally (for
    example, take three colors and return one of them).  This key uses the net
    token movement, target card identity, and claimed noble, so each distinct
    deterministic successor gets one search edge.  A visible-card replacement
    remains a chance event and is intentionally not represented here.
    """

    actor = state.current_player if player_id is None else player_id
    target = _action_target(state, action, actor)
    return (
        action.action_type.name,
        target.card_id if target is not None else None,
        action.tier if action.action_type == ActionType.RESERVE_DECK else None,
        _net_token_counts(action),
        action.noble_id,
    )


def canonical_successor_action_keys(
    state: SplendorState,
    legal_actions: list[Action],
    player_id: int | None = None,
) -> tuple[CanonicalKey, ...]:
    """Return the sorted, deduplicated successor-action keys."""

    return tuple(
        sorted(
            {
                canonical_successor_action_key(state, action, player_id)
                for action in legal_actions
            }
        )
    )


def actions_by_canonical_descriptor(
    state: SplendorState,
    legal_actions: list[Action],
    player_id: int | None = None,
) -> dict[CanonicalKey, Action]:
    """Bind each semantic action descriptor to one concrete legal action."""

    actions: dict[CanonicalKey, Action] = {}
    for action in legal_actions:
        descriptor = canonical_legal_action_descriptor(state, action, player_id)
        actions.setdefault(descriptor, action)
    return actions


def actions_by_canonical_successor_key(
    state: SplendorState,
    legal_actions: list[Action],
    player_id: int | None = None,
) -> dict[CanonicalKey, Action]:
    """Bind every successor key to one deterministic representative action."""

    actions: dict[CanonicalKey, Action] = {}
    for action in legal_actions:
        key = canonical_successor_action_key(state, action, player_id)
        actions.setdefault(key, action)
    return actions


def _player_key(player: PlayerState, observer_is_owner: bool) -> CanonicalKey:
    reserved = tuple(
        sorted(
            _reserved_card_key(
                card,
                player.reserved_card_source(index),
                observer_is_owner,
            )
            for index, card in enumerate(player.reserved_cards)
        )
    )
    return (
        _token_counts(player.tokens),
        tuple((color, int(player.bonuses[color])) for color in TOKEN_COLORS),
        int(player.score),
        tuple(sorted(card.card_id for card in player.purchased_cards)),
        tuple(sorted(noble.noble_id for noble in player.nobles)),
        reserved,
    )


def _reserved_card_key(
    card: Card,
    source: str,
    observer_is_owner: bool,
) -> CanonicalKey:
    if observer_is_owner or source == RESERVE_SOURCE_VISIBLE:
        return (source, card.card_id)
    if source == RESERVE_SOURCE_DECK:
        return (source, card.tier)
    # Legacy saves did not record provenance.  Their private card identity and
    # even its tier must therefore remain unavailable to the opponent.
    return ("unknown",)


def _action_target(
    state: SplendorState,
    action: Action,
    player_id: int,
) -> Card | None:
    if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
        if action.tier is None or action.market_index is None:
            return None
        cards = state.visible_tier_cards.get(action.tier, [])
        if 0 <= action.market_index < len(cards):
            return cards[action.market_index]
        return None
    if action.action_type == ActionType.BUY_RESERVED:
        if action.reserved_index is None:
            return None
        cards = state.players[player_id].reserved_cards
        if 0 <= action.reserved_index < len(cards):
            return cards[action.reserved_index]
    return None


def _token_counts(tokens: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple((color, int(tokens[color])) for color in ALL_TOKEN_TYPES)


def _token_counts_from_tuple(tokens: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    counts = Counter(tokens)
    return tuple((color, int(counts[color])) for color in ALL_TOKEN_TYPES)


def _net_token_counts(action: Action) -> tuple[tuple[str, int], ...]:
    counts = Counter(action.take_tokens)
    counts.subtract(action.return_tokens)
    counts.subtract(action.spend_tokens)
    return tuple((color, int(counts[color])) for color in ALL_TOKEN_TYPES)
