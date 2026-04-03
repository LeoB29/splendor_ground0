"""Fixed action codec and legality mask generation for Splendor."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES, MAX_TOKENS_PER_PLAYER, TOKEN_COLORS
from splendor_ai.engine.state import Card, SplendorState

_MAX_RETURN_TOKENS = 3
_MAX_NOBLE_SLOTS = 3
_NOBLE_NONE_SLOT = 0


def _generate_multiset_templates(
    colors: tuple[str, ...],
    max_count: int,
) -> tuple[tuple[str, ...], ...]:
    templates: list[tuple[str, ...]] = []

    def recurse(color_index: int, remaining: int, prefix: list[str]) -> None:
        if color_index == len(colors):
            templates.append(tuple(prefix))
            return

        color = colors[color_index]
        for amount in range(remaining + 1):
            if amount:
                prefix.extend([color] * amount)
            recurse(color_index + 1, remaining - amount, prefix)
            if amount:
                del prefix[-amount:]

    for choose_n in range(max_count + 1):
        recurse(0, choose_n, [])

    unique_templates = sorted(set(templates))
    return tuple(unique_templates)


def _generate_colored_spend_patterns() -> tuple[tuple[int, int, int, int, int], ...]:
    patterns: list[tuple[int, int, int, int, int]] = []

    def recurse(color_index: int, remaining: int, prefix: list[int]) -> None:
        if color_index == len(TOKEN_COLORS) - 1:
            prefix.append(remaining)
            patterns.append(tuple(prefix))
            prefix.pop()
            return

        for amount in range(remaining + 1):
            prefix.append(amount)
            recurse(color_index + 1, remaining - amount, prefix)
            prefix.pop()

    for total in range(MAX_TOKENS_PER_PLAYER + 1):
        recurse(0, total, [])

    return tuple(patterns)


_TAKE_DISTINCT_TEMPLATES: tuple[tuple[str, ...], ...] = tuple(
    combo
    for choose_n in (1, 2, 3)
    for combo in combinations(TOKEN_COLORS, choose_n)
)
_TAKE_DOUBLE_TEMPLATES: tuple[tuple[str, str], ...] = tuple(
    (color, color) for color in TOKEN_COLORS
)
_RETURN_TEMPLATES = _generate_multiset_templates(ALL_TOKEN_TYPES, _MAX_RETURN_TOKENS)
_RETURN_TEMPLATE_TO_INDEX = {template: idx for idx, template in enumerate(_RETURN_TEMPLATES)}
_SPEND_PATTERNS = _generate_colored_spend_patterns()
_SPEND_PATTERN_TO_INDEX = {pattern: idx for idx, pattern in enumerate(_SPEND_PATTERNS)}

_RETURN_TEMPLATE_COUNT = len(_RETURN_TEMPLATES)
_SPEND_PATTERN_COUNT = len(_SPEND_PATTERNS)
_BUY_NOBLE_CHOICES = _MAX_NOBLE_SLOTS + 1

_TAKE_DISTINCT_COUNT = len(_TAKE_DISTINCT_TEMPLATES) * _RETURN_TEMPLATE_COUNT
_TAKE_DOUBLE_COUNT = len(_TAKE_DOUBLE_TEMPLATES) * _RETURN_TEMPLATE_COUNT
_RESERVE_VISIBLE_COUNT = 12 * _RETURN_TEMPLATE_COUNT
_RESERVE_DECK_COUNT = 3 * _RETURN_TEMPLATE_COUNT
_BUY_VISIBLE_COUNT = 12 * _SPEND_PATTERN_COUNT * _BUY_NOBLE_CHOICES
_BUY_RESERVED_COUNT = 3 * _SPEND_PATTERN_COUNT * _BUY_NOBLE_CHOICES

_TAKE_DISTINCT_OFFSET = 0
_TAKE_DOUBLE_OFFSET = _TAKE_DISTINCT_OFFSET + _TAKE_DISTINCT_COUNT
_RESERVE_VISIBLE_OFFSET = _TAKE_DOUBLE_OFFSET + _TAKE_DOUBLE_COUNT
_RESERVE_DECK_OFFSET = _RESERVE_VISIBLE_OFFSET + _RESERVE_VISIBLE_COUNT
_BUY_VISIBLE_OFFSET = _RESERVE_DECK_OFFSET + _RESERVE_DECK_COUNT
_BUY_RESERVED_OFFSET = _BUY_VISIBLE_OFFSET + _BUY_VISIBLE_COUNT
_PASS_OFFSET = _BUY_RESERVED_OFFSET + _BUY_RESERVED_COUNT
_ACTION_SPACE_SIZE = _PASS_OFFSET + 1


def _colored_spend_pattern_from_action(action: Action) -> tuple[int, int, int, int, int]:
    counts = {color: 0 for color in TOKEN_COLORS}
    for color in action.spend_tokens:
        if color in counts:
            counts[color] += 1
    return tuple(counts[color] for color in TOKEN_COLORS)


def _spend_tokens_from_pattern(
    state: SplendorState,
    action_type: ActionType,
    target_index: int,
    spend_pattern: tuple[int, int, int, int, int],
) -> tuple[str, ...]:
    player = state.players[state.current_player]
    if action_type == ActionType.BUY_VISIBLE:
        tier = (target_index // 4) + 1
        market_index = target_index % 4
        card = state.visible_tier_cards[tier][market_index]
    else:
        card = player.reserved_cards[target_index]

    required_by_color = {
        color: max(card.cost.get(color, 0) - player.bonuses.get(color, 0), 0)
        for color in TOKEN_COLORS
    }
    colored_total = 0
    spend_tokens: list[str] = []
    for color, amount in zip(TOKEN_COLORS, spend_pattern):
        if amount > required_by_color[color]:
            raise ValueError("Colored spend pattern exceeds discounted card cost.")
        colored_total += amount
        spend_tokens.extend([color] * amount)

    total_required = sum(required_by_color.values())
    gold_needed = total_required - colored_total
    if gold_needed < 0:
        raise ValueError("Colored spend pattern overspends the discounted card cost.")
    spend_tokens.extend(["gold"] * gold_needed)
    return tuple(spend_tokens)


@dataclass(frozen=True, slots=True)
class ActionCodec:
    """Fixed action index mapping for model-facing action heads."""

    action_space_size: int = _ACTION_SPACE_SIZE

    def encode(self, state: SplendorState, action: Action) -> int:
        if action.action_type == ActionType.PASS:
            return _PASS_OFFSET
        if action.action_type == ActionType.TAKE_TOKENS:
            return self._encode_take_action(action)
        if action.action_type == ActionType.RESERVE_VISIBLE:
            return self._encode_reserve_visible_action(action)
        if action.action_type == ActionType.RESERVE_DECK:
            return self._encode_reserve_deck_action(action)
        if action.action_type == ActionType.BUY_VISIBLE:
            return self._encode_buy_visible_action(state, action)
        if action.action_type == ActionType.BUY_RESERVED:
            return self._encode_buy_reserved_action(state, action)
        raise ValueError(f"Unsupported action type: {action.action_type}")

    def decode(self, state: SplendorState, index: int) -> Action:
        if index < 0 or index >= self.action_space_size:
            raise ValueError(f"Action index out of range: {index}")

        if index < _TAKE_DOUBLE_OFFSET:
            relative = index - _TAKE_DISTINCT_OFFSET
            take_template_index, return_index = divmod(relative, _RETURN_TEMPLATE_COUNT)
            return Action(
                action_type=ActionType.TAKE_TOKENS,
                take_tokens=_TAKE_DISTINCT_TEMPLATES[take_template_index],
                return_tokens=_RETURN_TEMPLATES[return_index],
            )

        if index < _RESERVE_VISIBLE_OFFSET:
            relative = index - _TAKE_DOUBLE_OFFSET
            take_template_index, return_index = divmod(relative, _RETURN_TEMPLATE_COUNT)
            return Action(
                action_type=ActionType.TAKE_TOKENS,
                take_tokens=_TAKE_DOUBLE_TEMPLATES[take_template_index],
                return_tokens=_RETURN_TEMPLATES[return_index],
            )

        if index < _RESERVE_DECK_OFFSET:
            relative = index - _RESERVE_VISIBLE_OFFSET
            board_slot, return_index = divmod(relative, _RETURN_TEMPLATE_COUNT)
            tier = (board_slot // 4) + 1
            market_index = board_slot % 4
            return Action(
                action_type=ActionType.RESERVE_VISIBLE,
                tier=tier,
                market_index=market_index,
                take_tokens=("gold",) if self._reserve_grants_gold(state) else (),
                return_tokens=_RETURN_TEMPLATES[return_index],
            )

        if index < _BUY_VISIBLE_OFFSET:
            relative = index - _RESERVE_DECK_OFFSET
            tier_slot, return_index = divmod(relative, _RETURN_TEMPLATE_COUNT)
            return Action(
                action_type=ActionType.RESERVE_DECK,
                tier=tier_slot + 1,
                take_tokens=("gold",) if self._reserve_grants_gold(state) else (),
                return_tokens=_RETURN_TEMPLATES[return_index],
            )

        if index < _BUY_RESERVED_OFFSET:
            relative = index - _BUY_VISIBLE_OFFSET
            board_slot, remainder = divmod(
                relative, _SPEND_PATTERN_COUNT * _BUY_NOBLE_CHOICES
            )
            spend_index, noble_slot = divmod(remainder, _BUY_NOBLE_CHOICES)
            tier = (board_slot // 4) + 1
            market_index = board_slot % 4
            noble_id = self._noble_id_from_slot(state, noble_slot)
            spend_tokens = _spend_tokens_from_pattern(
                state=state,
                action_type=ActionType.BUY_VISIBLE,
                target_index=board_slot,
                spend_pattern=_SPEND_PATTERNS[spend_index],
            )
            return Action(
                action_type=ActionType.BUY_VISIBLE,
                tier=tier,
                market_index=market_index,
                spend_tokens=spend_tokens,
                noble_id=noble_id,
            )

        relative = index - _BUY_RESERVED_OFFSET
        if index < _PASS_OFFSET:
            reserved_index, remainder = divmod(
                relative, _SPEND_PATTERN_COUNT * _BUY_NOBLE_CHOICES
            )
            spend_index, noble_slot = divmod(remainder, _BUY_NOBLE_CHOICES)
            noble_id = self._noble_id_from_slot(state, noble_slot)
            spend_tokens = _spend_tokens_from_pattern(
                state=state,
                action_type=ActionType.BUY_RESERVED,
                target_index=reserved_index,
                spend_pattern=_SPEND_PATTERNS[spend_index],
            )
            return Action(
                action_type=ActionType.BUY_RESERVED,
                reserved_index=reserved_index,
                spend_tokens=spend_tokens,
                noble_id=noble_id,
            )

        return Action(action_type=ActionType.PASS)

    def legal_action_indices(self, state: SplendorState, legal_actions: list[Action]) -> list[int]:
        return [self.encode(state, action) for action in legal_actions]

    def legal_action_mask(self, state: SplendorState, legal_actions: list[Action]) -> list[bool]:
        mask = [False] * self.action_space_size
        for index in self.legal_action_indices(state, legal_actions):
            mask[index] = True
        return mask

    def _encode_take_action(self, action: Action) -> int:
        return_index = self._return_index(action)
        if (
            1 <= len(action.take_tokens) <= 3
            and len(set(action.take_tokens)) == len(action.take_tokens)
        ):
            take_template_index = _TAKE_DISTINCT_TEMPLATES.index(action.take_tokens)
            return _TAKE_DISTINCT_OFFSET + take_template_index * _RETURN_TEMPLATE_COUNT + return_index
        if len(action.take_tokens) == 2 and action.take_tokens[0] == action.take_tokens[1]:
            take_template_index = _TAKE_DOUBLE_TEMPLATES.index(action.take_tokens)
            return _TAKE_DOUBLE_OFFSET + take_template_index * _RETURN_TEMPLATE_COUNT + return_index
        raise ValueError(f"Unsupported take action template: {action}")

    def _encode_reserve_visible_action(self, action: Action) -> int:
        if action.tier is None or action.market_index is None:
            raise ValueError("Reserve visible action requires tier and market index.")
        board_slot = (action.tier - 1) * 4 + action.market_index
        return _RESERVE_VISIBLE_OFFSET + board_slot * _RETURN_TEMPLATE_COUNT + self._return_index(action)

    def _encode_reserve_deck_action(self, action: Action) -> int:
        if action.tier is None:
            raise ValueError("Reserve deck action requires a tier.")
        return _RESERVE_DECK_OFFSET + (action.tier - 1) * _RETURN_TEMPLATE_COUNT + self._return_index(action)

    def _encode_buy_visible_action(self, state: SplendorState, action: Action) -> int:
        if action.tier is None or action.market_index is None:
            raise ValueError("Buy visible action requires tier and market index.")
        board_slot = (action.tier - 1) * 4 + action.market_index
        spend_index = self._spend_index(action)
        noble_slot = self._noble_slot(state, action.noble_id)
        return (
            _BUY_VISIBLE_OFFSET
            + board_slot * _SPEND_PATTERN_COUNT * _BUY_NOBLE_CHOICES
            + spend_index * _BUY_NOBLE_CHOICES
            + noble_slot
        )

    def _encode_buy_reserved_action(self, state: SplendorState, action: Action) -> int:
        if action.reserved_index is None:
            raise ValueError("Buy reserved action requires a reserved index.")
        spend_index = self._spend_index(action)
        noble_slot = self._noble_slot(state, action.noble_id)
        return (
            _BUY_RESERVED_OFFSET
            + action.reserved_index * _SPEND_PATTERN_COUNT * _BUY_NOBLE_CHOICES
            + spend_index * _BUY_NOBLE_CHOICES
            + noble_slot
        )

    def _return_index(self, action: Action) -> int:
        try:
            return _RETURN_TEMPLATE_TO_INDEX[action.return_tokens]
        except KeyError as exc:
            raise ValueError(f"Unsupported return token template: {action.return_tokens}") from exc

    def _spend_index(self, action: Action) -> int:
        pattern = _colored_spend_pattern_from_action(action)
        try:
            return _SPEND_PATTERN_TO_INDEX[pattern]
        except KeyError as exc:
            raise ValueError(f"Unsupported spend pattern: {pattern}") from exc

    def _noble_slot(self, state: SplendorState, noble_id: str | None) -> int:
        if noble_id is None:
            return _NOBLE_NONE_SLOT
        for noble_slot, noble in enumerate(state.nobles, start=1):
            if noble.noble_id == noble_id:
                return noble_slot
        raise ValueError(f"Noble id is not present in the current state: {noble_id}")

    def _noble_id_from_slot(self, state: SplendorState, noble_slot: int) -> str | None:
        if noble_slot == _NOBLE_NONE_SLOT:
            return None
        index = noble_slot - 1
        return state.nobles[index].noble_id

    def _reserve_grants_gold(self, state: SplendorState) -> bool:
        return state.bank_tokens["gold"] > 0
