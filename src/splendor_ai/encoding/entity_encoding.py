"""Variable-length entity encodings for the clean Splendor policy contract."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
from splendor_ai.engine.state import (
    Card,
    Noble,
    RESERVE_SOURCE_DECK,
    RESERVE_SOURCE_VISIBLE,
    SplendorState,
)

ENTITY_NUMERIC_SIZE = 34
# Action rows reuse the generic entity-width row: the card-face fields occupy
# the prefix and the remaining slots carry token flow plus reserve-deck tier.
ACTION_NUMERIC_SIZE = 34

ENTITY_PAD = 0
ENTITY_GLOBAL = 1
ENTITY_BANK = 2
ENTITY_DECK = 3
ENTITY_SELF = 4
ENTITY_OPPONENT = 5
ENTITY_BOARD_CARD = 6
ENTITY_NOBLE = 7
ENTITY_SELF_RESERVED = 8
ENTITY_OPPONENT_VISIBLE_RESERVED = 9
ENTITY_OPPONENT_BLIND_RESERVED = 10
ENTITY_SELF_PURCHASED = 11
ENTITY_OPPONENT_PURCHASED = 12
ENTITY_TYPE_COUNT = 13

ACTION_PAD = 0
ACTION_TYPE_TO_ID = {action_type: index for index, action_type in enumerate(ActionType, start=1)}
ACTION_TYPE_COUNT = len(ACTION_TYPE_TO_ID) + 1


@dataclass(frozen=True, slots=True)
class EntityStateEncoding:
    entity_type_ids: tuple[int, ...]
    card_ids: tuple[int, ...]
    noble_ids: tuple[int, ...]
    numeric_features: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class EntityActionEncoding:
    action_type_ids: tuple[int, ...]
    card_ids: tuple[int, ...]
    noble_ids: tuple[int, ...]
    numeric_features: tuple[tuple[float, ...], ...]


def encode_canonical_state_entities(
    state: SplendorState,
    player_id: int,
) -> EntityStateEncoding:
    """Encode only observer-legal state entities in canonical collection order."""

    if player_id not in (0, 1):
        raise ValueError(f"player_id must be 0 or 1, got {player_id}")
    player = state.players[player_id]
    opponent = state.players[1 - player_id]
    rows: list[tuple[int, int, int, tuple[float, ...]]] = []
    zero = _zeros()
    global_features = list(zero)
    global_features[31] = 1.0 if state.current_player == player_id else 0.0
    global_features[32] = min(float(state.turn_index) / 100.0, 2.0)
    global_features[33] = 1.0 if state.pending_round_end else 0.0
    rows.append((ENTITY_GLOBAL, 0, 0, tuple(global_features)))
    rows.append((ENTITY_BANK, 0, 0, _token_row(state.bank_tokens)))
    for tier in (1, 2, 3):
        features = list(zero)
        features[29] = float(tier) / 3.0
        features[30] = float(state.deck_counts[tier]) / _deck_normalizer(tier)
        rows.append((ENTITY_DECK, 0, 0, tuple(features)))
    rows.append((ENTITY_SELF, 0, 0, _player_row(player)))
    rows.append((ENTITY_OPPONENT, 0, 0, _player_row(opponent)))
    for tier in (1, 2, 3):
        for card in sorted(state.visible_tier_cards[tier], key=lambda item: item.card_id):
            rows.append((ENTITY_BOARD_CARD, _card_id(card), 0, _card_row(card)))
    for noble in sorted(state.nobles, key=lambda item: item.noble_id):
        rows.append((ENTITY_NOBLE, 0, _noble_id(noble), _noble_row(noble)))
    for index, card in sorted(
        enumerate(player.reserved_cards), key=lambda item: item[1].card_id
    ):
        del index
        rows.append((ENTITY_SELF_RESERVED, _card_id(card), 0, _card_row(card)))
    for index, card in enumerate(opponent.reserved_cards):
        source = opponent.reserved_card_source(index)
        if source == RESERVE_SOURCE_VISIBLE:
            rows.append(
                (ENTITY_OPPONENT_VISIBLE_RESERVED, _card_id(card), 0, _card_row(card))
            )
        elif source == RESERVE_SOURCE_DECK:
            features = list(zero)
            features[29] = float(card.tier) / 3.0
            rows.append((ENTITY_OPPONENT_BLIND_RESERVED, 0, 0, tuple(features)))
        else:
            rows.append((ENTITY_OPPONENT_BLIND_RESERVED, 0, 0, zero))
    for card in sorted(player.purchased_cards, key=lambda item: item.card_id):
        rows.append((ENTITY_SELF_PURCHASED, _card_id(card), 0, _card_row(card)))
    for card in sorted(opponent.purchased_cards, key=lambda item: item.card_id):
        rows.append((ENTITY_OPPONENT_PURCHASED, _card_id(card), 0, _card_row(card)))
    return EntityStateEncoding(
        entity_type_ids=tuple(row[0] for row in rows),
        card_ids=tuple(row[1] for row in rows),
        noble_ids=tuple(row[2] for row in rows),
        numeric_features=tuple(row[3] for row in rows),
    )


def encode_canonical_action_entities(
    state: SplendorState,
    legal_actions: list[Action],
    player_id: int | None = None,
) -> EntityActionEncoding:
    """Encode legal actions without referring to market/reserve slot positions."""

    actor = state.current_player if player_id is None else player_id
    rows: list[tuple[int, int, int, tuple[float, ...]]] = []
    for action in legal_actions:
        target = _action_target(state, action, actor)
        features = list(_card_row(target) if target is not None else _zeros())
        offset = 13
        for tokens in (action.take_tokens, action.return_tokens, action.spend_tokens):
            counts = {color: tokens.count(color) for color in ALL_TOKEN_TYPES}
            for color_index, color in enumerate(ALL_TOKEN_TYPES):
                features[offset + color_index] = float(counts[color]) / _token_normalizer(color)
            offset += len(ALL_TOKEN_TYPES)
        features[31] = float(action.tier or 0) / 3.0
        rows.append(
            (
                ACTION_TYPE_TO_ID[action.action_type],
                _card_id(target) if target is not None else 0,
                _noble_id_from_id(action.noble_id),
                tuple(features),
            )
        )
    return EntityActionEncoding(
        action_type_ids=tuple(row[0] for row in rows),
        card_ids=tuple(row[1] for row in rows),
        noble_ids=tuple(row[2] for row in rows),
        numeric_features=tuple(row[3] for row in rows),
    )


def collate_entity_action_batch(
    state_action_batches: list[tuple[SplendorState, list[Action]]],
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Pad a variable-length state/action batch for the entity scorer."""

    if not state_action_batches:
        raise ValueError("state_action_batches must not be empty.")
    state_encodings = [
        encode_canonical_state_entities(state, state.current_player)
        for state, _actions in state_action_batches
    ]
    action_encodings = [
        encode_canonical_action_entities(state, actions, state.current_player)
        for state, actions in state_action_batches
    ]
    if any(not encoding.action_type_ids for encoding in action_encodings):
        raise ValueError("Every state must contain at least one legal action.")
    max_entities = max(len(encoding.entity_type_ids) for encoding in state_encodings)
    max_actions = max(len(encoding.action_type_ids) for encoding in action_encodings)
    target_device = None if device is None else torch.device(device)
    return {
        "entity_type_ids": _pad_int_rows(
            [encoding.entity_type_ids for encoding in state_encodings], max_entities, target_device
        ),
        "entity_card_ids": _pad_int_rows(
            [encoding.card_ids for encoding in state_encodings], max_entities, target_device
        ),
        "entity_noble_ids": _pad_int_rows(
            [encoding.noble_ids for encoding in state_encodings], max_entities, target_device
        ),
        "entity_numeric_features": _pad_float_rows(
            [encoding.numeric_features for encoding in state_encodings], max_entities, ENTITY_NUMERIC_SIZE, target_device
        ),
        "entity_mask": _mask([len(encoding.entity_type_ids) for encoding in state_encodings], max_entities, target_device),
        "action_type_ids": _pad_int_rows(
            [encoding.action_type_ids for encoding in action_encodings], max_actions, target_device
        ),
        "action_card_ids": _pad_int_rows(
            [encoding.card_ids for encoding in action_encodings], max_actions, target_device
        ),
        "action_noble_ids": _pad_int_rows(
            [encoding.noble_ids for encoding in action_encodings], max_actions, target_device
        ),
        "action_numeric_features": _pad_float_rows(
            [encoding.numeric_features for encoding in action_encodings], max_actions, ACTION_NUMERIC_SIZE, target_device
        ),
        "action_mask": _mask([len(encoding.action_type_ids) for encoding in action_encodings], max_actions, target_device),
    }


def card_vocabulary_size() -> int:
    return len(_card_id_map()) + 1


def noble_vocabulary_size() -> int:
    return len(_noble_id_map()) + 1


def _card_row(card: Card | None) -> tuple[float, ...]:
    values = list(_zeros())
    if card is None:
        return tuple(values)
    values[0] = 1.0
    values[1] = float(card.tier) / 3.0
    values[2] = float(card.points) / 5.0
    for index, color in enumerate(TOKEN_COLORS):
        values[3 + index] = 1.0 if card.bonus_color == color else 0.0
        values[8 + index] = float(card.cost.get(color, 0)) / 7.0
    return tuple(values)


def _noble_row(noble: Noble) -> tuple[float, ...]:
    values = list(_zeros())
    values[13] = 1.0
    values[14] = float(noble.points) / 3.0
    for index, color in enumerate(TOKEN_COLORS):
        values[15 + index] = float(noble.requirement.get(color, 0)) / 4.0
    return tuple(values)


def _token_row(tokens: dict[str, int]) -> tuple[float, ...]:
    values = list(_zeros())
    for index, color in enumerate(ALL_TOKEN_TYPES):
        values[20 + index] = float(tokens[color]) / _token_normalizer(color)
    return tuple(values)


def _player_row(player: object) -> tuple[float, ...]:
    values = list(_token_row(player.tokens))
    for index, color in enumerate(TOKEN_COLORS):
        values[26 + index] = float(player.bonuses[color]) / 7.0
    values[31] = float(player.score) / 20.0
    values[32] = float(len(player.reserved_cards)) / 3.0
    values[33] = float(len(player.purchased_cards)) / 20.0
    return tuple(values)


def _action_target(state: SplendorState, action: Action, player_id: int) -> Card | None:
    if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
        cards = state.visible_tier_cards.get(action.tier or 0, [])
        if action.market_index is not None and 0 <= action.market_index < len(cards):
            return cards[action.market_index]
    if action.action_type == ActionType.BUY_RESERVED:
        cards = state.players[player_id].reserved_cards
        if action.reserved_index is not None and 0 <= action.reserved_index < len(cards):
            return cards[action.reserved_index]
    return None


@lru_cache(maxsize=1)
def _card_id_map() -> dict[str, int]:
    return {
        card.card_id: index
        for index, card in enumerate(
            sorted(
                (card for cards in build_base_deck_by_tier().values() for card in cards),
                key=lambda item: item.card_id,
            ),
            start=1,
        )
    }


@lru_cache(maxsize=1)
def _noble_id_map() -> dict[str, int]:
    return {
        noble.noble_id: index
        for index, noble in enumerate(sorted(build_base_nobles(), key=lambda item: item.noble_id), start=1)
    }


def _card_id(card: Card | None) -> int:
    return 0 if card is None else _card_id_map().get(card.card_id, 0)


def _noble_id(noble: Noble) -> int:
    return _noble_id_map().get(noble.noble_id, 0)


def _noble_id_from_id(noble_id: str | None) -> int:
    return 0 if noble_id is None else _noble_id_map().get(noble_id, 0)


def _zeros() -> tuple[float, ...]:
    return (0.0,) * ENTITY_NUMERIC_SIZE


def _deck_normalizer(tier: int) -> float:
    return {1: 36.0, 2: 26.0, 3: 16.0}[tier]


def _token_normalizer(color: str) -> float:
    return 5.0 if color == "gold" else 4.0


def _pad_int_rows(rows: list[tuple[int, ...]], width: int, device: torch.device | None) -> torch.Tensor:
    return torch.tensor([list(row) + [0] * (width - len(row)) for row in rows], dtype=torch.long, device=device)


def _pad_float_rows(
    rows: list[tuple[tuple[float, ...], ...]],
    width: int,
    feature_size: int,
    device: torch.device | None,
) -> torch.Tensor:
    zero_row = [0.0] * feature_size
    return torch.tensor(
        [list(row) + [zero_row] * (width - len(row)) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _mask(lengths: list[int], width: int, device: torch.device | None) -> torch.Tensor:
    return torch.tensor(
        [[index < length for index in range(width)] for length in lengths],
        dtype=torch.bool,
        device=device,
    )
