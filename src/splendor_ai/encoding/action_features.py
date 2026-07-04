"""Structured legal-action feature encoding for action-scoring models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, TOKEN_COLORS
from splendor_ai.engine.state import Card, Noble, SplendorState

ACTION_TYPE_ORDER: tuple[ActionType, ...] = (
    ActionType.PASS,
    ActionType.TAKE_TOKENS,
    ActionType.RESERVE_VISIBLE,
    ActionType.RESERVE_DECK,
    ActionType.BUY_VISIBLE,
    ActionType.BUY_RESERVED,
)

TOKEN_ORDER: tuple[str, ...] = ALL_TOKEN_TYPES
BONUS_ORDER: tuple[str, ...] = TOKEN_COLORS
BOARD_TIERS: tuple[int, ...] = (1, 2, 3)
MARKET_SLOTS: tuple[int, ...] = (0, 1, 2, 3)
RESERVED_SLOTS: tuple[int, ...] = (0, 1, 2)

ACTION_TYPE_SIZE = len(ACTION_TYPE_ORDER)
LOCATION_SIZE = len(BOARD_TIERS) + len(MARKET_SLOTS) + len(RESERVED_SLOTS)
CARD_FEATURE_SIZE = 13
NOBLE_FEATURE_SIZE = 7
TOKEN_FLOW_SIZE = len(TOKEN_ORDER) * 4
DERIVED_FEATURE_SIZE = 4
OPPONENT_CONTEXT_FEATURE_SIZE = 15
ACTION_FEATURE_SIZE = (
    ACTION_TYPE_SIZE
    + LOCATION_SIZE
    + CARD_FEATURE_SIZE
    + NOBLE_FEATURE_SIZE
    + TOKEN_FLOW_SIZE
    + DERIVED_FEATURE_SIZE
    + OPPONENT_CONTEXT_FEATURE_SIZE
)

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
class ActionFeatureTensor:
    """Flat action feature vector plus section metadata."""

    vector: tuple[float, ...]
    sections: dict[str, tuple[int, int]]


def encode_action_features(
    state: SplendorState,
    action: Action,
    player_id: int | None = None,
) -> ActionFeatureTensor:
    """Encode one legal action as state-aware numeric features.

    The encoder is designed for future legal-action scorer models: it describes
    the candidate action being scored, including its target card/noble and token
    flow, without relying on the fixed global action index.
    """

    acting_player_id = state.current_player if player_id is None else player_id
    player = state.players[acting_player_id]
    target_card = _target_card_for_action(state, action, acting_player_id)
    chosen_noble = _chosen_noble_for_action(state, action)

    vector: list[float] = []
    sections: dict[str, tuple[int, int]] = {}

    def push(section_name: str, values: list[float]) -> None:
        start = len(vector)
        vector.extend(values)
        sections[section_name] = (start, len(vector))

    push("action_type", _encode_action_type(action.action_type))
    push("location", _encode_location(action))
    push("target_card", _encode_card(target_card))
    push("chosen_noble", _encode_noble(chosen_noble))
    push("token_flow", _encode_token_flow(action))
    push(
        "derived",
        _encode_derived_features(
            state=state,
            action=action,
            target_card=target_card,
            chosen_noble=chosen_noble,
            player_bonuses=player.bonuses,
        ),
    )
    push(
        "opponent_context",
        _encode_opponent_context_features(
            state=state,
            action=action,
            target_card=target_card,
            chosen_noble=chosen_noble,
            player_id=acting_player_id,
        ),
    )

    return ActionFeatureTensor(vector=tuple(vector), sections=sections)


def encode_legal_action_features(
    state: SplendorState,
    legal_actions: list[Action],
    player_id: int | None = None,
) -> list[ActionFeatureTensor]:
    """Encode all legal actions for a position in their supplied order."""

    return [
        encode_action_features(state=state, action=action, player_id=player_id)
        for action in legal_actions
    ]


def _encode_action_type(action_type: ActionType) -> list[float]:
    return [1.0 if action_type == candidate else 0.0 for candidate in ACTION_TYPE_ORDER]


def _encode_location(action: Action) -> list[float]:
    return [
        *[1.0 if action.tier == tier else 0.0 for tier in BOARD_TIERS],
        *[1.0 if action.market_index == index else 0.0 for index in MARKET_SLOTS],
        *[1.0 if action.reserved_index == index else 0.0 for index in RESERVED_SLOTS],
    ]


def _encode_card(card: Card | None) -> list[float]:
    if card is None:
        return [0.0] * CARD_FEATURE_SIZE

    return [
        1.0,
        float(card.tier) / 3.0,
        float(card.points) / 5.0,
        *[1.0 if card.bonus_color == color else 0.0 for color in BONUS_ORDER],
        *[float(card.cost.get(color, 0)) / 7.0 for color in BONUS_ORDER],
    ]


def _encode_noble(noble: Noble | None) -> list[float]:
    if noble is None:
        return [0.0] * NOBLE_FEATURE_SIZE

    return [
        1.0,
        float(noble.points) / 3.0,
        *[float(noble.requirement.get(color, 0)) / 4.0 for color in BONUS_ORDER],
    ]


def _encode_token_flow(action: Action) -> list[float]:
    take_counts = _token_counts(action.take_tokens)
    return_counts = _token_counts(action.return_tokens)
    spend_counts = _token_counts(action.spend_tokens)
    net_counts = {
        color: take_counts[color] - return_counts[color] - spend_counts[color]
        for color in TOKEN_ORDER
    }

    return [
        *[_normalize_token_count(color, take_counts[color]) for color in TOKEN_ORDER],
        *[_normalize_token_count(color, return_counts[color]) for color in TOKEN_ORDER],
        *[_normalize_token_count(color, spend_counts[color]) for color in TOKEN_ORDER],
        *[_normalize_token_count(color, net_counts[color]) for color in TOKEN_ORDER],
    ]


def _encode_derived_features(
    state: SplendorState,
    action: Action,
    target_card: Card | None,
    chosen_noble: Noble | None,
    player_bonuses: dict[str, int],
) -> list[float]:
    discounted_cost_total = 0
    if target_card is not None and action.action_type in (
        ActionType.BUY_VISIBLE,
        ActionType.BUY_RESERVED,
    ):
        discounted_cost_total = sum(
            max(target_card.cost.get(color, 0) - player_bonuses.get(color, 0), 0)
            for color in BONUS_ORDER
        )

    gained_prestige = 0
    if target_card is not None and action.action_type in (
        ActionType.BUY_VISIBLE,
        ActionType.BUY_RESERVED,
    ):
        gained_prestige += target_card.points
    if chosen_noble is not None:
        gained_prestige += chosen_noble.points

    deck_availability = 0.0
    if action.action_type == ActionType.RESERVE_DECK and action.tier is not None:
        deck_availability = float(state.deck_counts.get(action.tier, 0)) / _DECK_NORMALIZERS[action.tier]

    token_delta = len(action.take_tokens) - len(action.return_tokens) - len(action.spend_tokens)

    return [
        float(discounted_cost_total) / 7.0,
        float(gained_prestige) / 8.0,
        deck_availability,
        float(token_delta) / 10.0,
    ]


def _encode_opponent_context_features(
    state: SplendorState,
    action: Action,
    target_card: Card | None,
    chosen_noble: Noble | None,
    player_id: int,
) -> list[float]:
    player = state.players[player_id]
    opponent = state.players[1 - player_id]
    after_player_tokens = Counter(player.tokens)
    for color in action.take_tokens:
        after_player_tokens[color] += 1
    for color in action.return_tokens:
        after_player_tokens[color] -= 1
    for color in action.spend_tokens:
        after_player_tokens[color] -= 1

    player_affordable_before = _visible_affordable_count(state, player_id)
    player_affordable_after = _visible_affordable_count(
        state,
        player_id,
        token_override=after_player_tokens,
    )
    opponent_affordable_before = _visible_affordable_count(state, 1 - player_id)
    target_missing_player = _missing_to_card(target_card, player.bonuses, player.tokens)
    target_missing_opponent = _missing_to_card(target_card, opponent.bonuses, opponent.tokens)
    opponent_can_buy_target = target_card is not None and target_missing_opponent == 0
    opponent_close_to_target = target_card is not None and target_missing_opponent <= 2
    reserve_denies_opponent = (
        action.action_type == ActionType.RESERVE_VISIBLE
        and target_card is not None
        and opponent_close_to_target
    )
    gained_points = 0
    if target_card is not None and action.action_type in (
        ActionType.BUY_VISIBLE,
        ActionType.BUY_RESERVED,
    ):
        gained_points += target_card.points
    if chosen_noble is not None:
        gained_points += chosen_noble.points
    reaches_15 = player.score + gained_points >= 15
    opponent_at_threat_score = opponent.score >= 12
    target_points = 0 if target_card is None else target_card.points

    return [
        float(player.score - opponent.score) / 20.0,
        float(gained_points) / 8.0,
        1.0 if reaches_15 else 0.0,
        1.0 if opponent_at_threat_score else 0.0,
        float(target_points) / 5.0,
        float(target_missing_player) / 10.0,
        float(target_missing_opponent) / 10.0,
        1.0 if opponent_can_buy_target else 0.0,
        1.0 if opponent_close_to_target else 0.0,
        1.0 if reserve_denies_opponent else 0.0,
        float(len(player.reserved_cards)) / 3.0,
        float(len(opponent.reserved_cards)) / 3.0,
        float(sum(after_player_tokens.values())) / 10.0,
        float(max(sum(after_player_tokens.values()) - 10, 0)) / 5.0,
        float(player_affordable_after - player_affordable_before - opponent_affordable_before) / 12.0,
    ]


def _visible_affordable_count(
    state: SplendorState,
    player_id: int,
    token_override: Counter[str] | None = None,
) -> int:
    player = state.players[player_id]
    tokens = player.tokens if token_override is None else token_override
    return sum(
        1
        for cards in state.visible_tier_cards.values()
        for card in cards
        if _missing_to_card(card, player.bonuses, tokens) == 0
    )


def _missing_to_card(
    card: Card | None,
    bonuses: dict[str, int],
    tokens: dict[str, int] | Counter[str],
) -> int:
    if card is None:
        return 10
    missing = 0
    gold = int(tokens.get("gold", 0))
    for color in BONUS_ORDER:
        required = max(card.cost.get(color, 0) - bonuses.get(color, 0), 0)
        available = int(tokens.get(color, 0))
        missing += max(required - available, 0)
    return max(missing - gold, 0)


def _target_card_for_action(
    state: SplendorState,
    action: Action,
    player_id: int,
) -> Card | None:
    if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
        if action.tier is None or action.market_index is None:
            return None
        visible_cards = state.visible_tier_cards.get(action.tier, [])
        if action.market_index >= len(visible_cards):
            return None
        return visible_cards[action.market_index]

    if action.action_type == ActionType.BUY_RESERVED:
        if action.reserved_index is None:
            return None
        reserved_cards = state.players[player_id].reserved_cards
        if action.reserved_index >= len(reserved_cards):
            return None
        return reserved_cards[action.reserved_index]

    return None


def _chosen_noble_for_action(state: SplendorState, action: Action) -> Noble | None:
    if action.noble_id is None:
        return None
    return next((noble for noble in state.nobles if noble.noble_id == action.noble_id), None)


def _token_counts(tokens: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(tokens)
    return {color: int(counts.get(color, 0)) for color in TOKEN_ORDER}


def _normalize_token_count(color: str, count: int) -> float:
    return float(count) / _TOKEN_NORMALIZERS[color]
