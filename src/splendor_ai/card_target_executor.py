"""Frozen mechanics-only card-target executor shared by experiments 081 and 082."""

from __future__ import annotations

from typing import Any

from splendor_ai.encoding import ActionCodec
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.state import Card, SplendorState


def target_executor_action(
    state: SplendorState,
    actions: list[Action],
    scores: tuple[Any, ...],
    target: Card,
) -> Action:
    score_by_action = {item.action: float(item.score) for item in scores}
    codec = ActionCodec()
    addressed = []
    for action in actions:
        card = action_card(state, action)
        if card is not None and card.card_id == target.card_id:
            addressed.append(action)
    buys = [
        action for action in addressed
        if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED)
    ]
    if buys:
        return max(
            buys,
            key=lambda action: (score_by_action[action], -codec.encode(state, action)),
        )
    reserves = [
        action for action in addressed
        if action.action_type is ActionType.RESERVE_VISIBLE
    ]
    if reserves:
        return max(
            reserves,
            key=lambda action: (score_by_action[action], -codec.encode(state, action)),
        )
    player = state.players[state.current_player]
    residual = {
        color: max(0, target.cost.get(color, 0) - player.bonuses.get(color, 0))
        for color in target.cost
    }
    helpful = []
    for action in actions:
        if action.action_type not in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            continue
        card = action_card(state, action)
        if card is not None and residual.get(card.bonus_color, 0) > 0:
            helpful.append(action)
    if helpful:
        return max(
            helpful,
            key=lambda action: (score_by_action[action], -codec.encode(state, action)),
        )
    takes = [action for action in actions if action.action_type is ActionType.TAKE_TOKENS]
    if takes:
        before = colour_shortfall(target, player.tokens, player.bonuses)

        def take_key(action: Action) -> tuple[float, float, int]:
            tokens = dict(player.tokens)
            for color in action.take_tokens or ():
                tokens[color] += 1
            for color in action.return_tokens or ():
                tokens[color] -= 1
            reduction = before - colour_shortfall(target, tokens, player.bonuses)
            return reduction, score_by_action[action], -codec.encode(state, action)

        return max(takes, key=take_key)
    return scores[0].action


def action_card(state: SplendorState, action: Action) -> Card | None:
    if action.action_type in (ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE):
        return state.visible_tier_cards[action.tier][action.market_index]
    if action.action_type is ActionType.BUY_RESERVED:
        return state.players[state.current_player].reserved_cards[action.reserved_index]
    return None


def colour_shortfall(
    card: Card,
    tokens: dict[str, int],
    bonuses: dict[str, int],
) -> int:
    return sum(
        max(
            0,
            card.cost.get(color, 0)
            - bonuses.get(color, 0)
            - tokens.get(color, 0),
        )
        for color in card.cost
    )
