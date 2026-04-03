"""Legal-observation encoding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from splendor_ai.engine.state import SplendorState

_TOKEN_ORDER = ("white", "blue", "green", "red", "black", "gold")
_BONUS_ORDER = ("white", "blue", "green", "red", "black")
_BOARD_TIERS = (1, 2, 3)
_CARDS_PER_TIER = 4
_RESERVED_SLOTS = 3
_NOBLE_SLOTS = 3
_CARD_SLOT_SIZE = 13
_NOBLE_SLOT_SIZE = 7
_PLAYER_SUMMARY_SIZE = 14
_GLOBAL_FEATURES_SIZE = 3
_BANK_FEATURES_SIZE = 6
_DECK_FEATURES_SIZE = 3

_DECK_NORMALIZERS = {1: 36.0, 2: 26.0, 3: 16.0}
_TOKEN_NORMALIZERS = {
    "white": 4.0,
    "blue": 4.0,
    "green": 4.0,
    "red": 4.0,
    "black": 4.0,
    "gold": 5.0,
}


@dataclass(frozen=True, slots=True)
class ObservationTensor:
    """Flat legal-observation vector plus section metadata."""

    vector: tuple[float, ...]
    sections: dict[str, tuple[int, int]]


def _encode_card_slot(card: Any | None) -> list[float]:
    if card is None:
        return [0.0] * _CARD_SLOT_SIZE

    return [
        1.0,
        float(card.tier) / 3.0,
        float(card.points) / 5.0,
        *[1.0 if card.bonus_color == color else 0.0 for color in _BONUS_ORDER],
        *[float(card.cost.get(color, 0)) / 7.0 for color in _BONUS_ORDER],
    ]


def _encode_noble_slot(noble: Any | None) -> list[float]:
    if noble is None:
        return [0.0] * _NOBLE_SLOT_SIZE

    return [
        1.0,
        float(noble.points) / 3.0,
        *[float(noble.requirement.get(color, 0)) / 4.0 for color in _BONUS_ORDER],
    ]


def _encode_player_summary(player: Any) -> list[float]:
    return [
        *[float(player.tokens[color]) / 10.0 for color in _TOKEN_ORDER],
        *[float(player.bonuses[color]) / 7.0 for color in _BONUS_ORDER],
        float(player.score) / 20.0,
        float(len(player.reserved_cards)) / 3.0,
        float(len(player.purchased_cards)) / 20.0,
    ]


def encode_public_observation(state: SplendorState, player_id: int) -> dict[str, Any]:
    """Encode a legal public observation for `player_id`.

    This first version returns a structured Python dictionary rather than a
    tensor. Tensorization comes later once the canonical feature layout is
    finalized.
    """

    player = state.players[player_id]
    opponent = state.players[1 - player_id]

    return {
        "current_player": state.current_player,
        "turn_index": state.turn_index,
        "bank_tokens": dict(state.bank_tokens),
        "deck_counts": dict(state.deck_counts),
        "visible_tier_cards": {
            tier: [card.card_id for card in cards]
            for tier, cards in state.visible_tier_cards.items()
        },
        "nobles": [noble.noble_id for noble in state.nobles],
        "self": {
            "tokens": dict(player.tokens),
            "bonuses": dict(player.bonuses),
            "score": player.score,
            "reserved_cards": [card.card_id for card in player.reserved_cards],
            "purchased_cards": [card.card_id for card in player.purchased_cards],
        },
        "opponent": {
            "tokens": dict(opponent.tokens),
            "bonuses": dict(opponent.bonuses),
            "score": opponent.score,
            "reserved_count": len(opponent.reserved_cards),
            "purchased_cards": [card.card_id for card in opponent.purchased_cards],
        },
    }


def encode_public_observation_tensor(
    state: SplendorState,
    player_id: int,
) -> ObservationTensor:
    """Encode a flat legal-observation vector for model consumption.

    Current layout length: 256 floats.
    """

    player = state.players[player_id]
    opponent = state.players[1 - player_id]

    vector: list[float] = []
    sections: dict[str, tuple[int, int]] = {}

    def push(section_name: str, values: list[float]) -> None:
        start = len(vector)
        vector.extend(values)
        sections[section_name] = (start, len(vector))

    push(
        "global",
        [
            1.0 if state.current_player == player_id else 0.0,
            float(state.turn_index) / 100.0,
            1.0 if state.pending_round_end else 0.0,
        ],
    )
    push(
        "bank",
        [
            float(state.bank_tokens[color]) / _TOKEN_NORMALIZERS[color]
            for color in _TOKEN_ORDER
        ],
    )
    push(
        "decks",
        [
            float(state.deck_counts[tier]) / _DECK_NORMALIZERS[tier]
            for tier in _BOARD_TIERS
        ],
    )

    noble_values: list[float] = []
    for noble_index in range(_NOBLE_SLOTS):
        noble = state.nobles[noble_index] if noble_index < len(state.nobles) else None
        noble_values.extend(_encode_noble_slot(noble))
    push("nobles", noble_values)

    board_values: list[float] = []
    for tier in _BOARD_TIERS:
        visible_cards = state.visible_tier_cards[tier]
        for market_index in range(_CARDS_PER_TIER):
            card = visible_cards[market_index] if market_index < len(visible_cards) else None
            board_values.extend(_encode_card_slot(card))
    push("board", board_values)

    push("self_summary", _encode_player_summary(player))

    reserved_values: list[float] = []
    for reserved_index in range(_RESERVED_SLOTS):
        card = player.reserved_cards[reserved_index] if reserved_index < len(player.reserved_cards) else None
        reserved_values.extend(_encode_card_slot(card))
    push("self_reserved", reserved_values)

    push("opponent_summary", _encode_player_summary(opponent))

    return ObservationTensor(vector=tuple(vector), sections=sections)
