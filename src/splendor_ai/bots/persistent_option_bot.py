"""Persistent target-conditioned option planner for experiment `079`."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, SplendorState

from .heuristic_bot import GreedyHeuristicBot
from .hidden_information import determinize_for_observer
from .value_reserve_bot import EFFICIENCY_FILTER, deficit, effective_cost, effective_value
from .value_reserve_search_bot import (
    FrozenFullActionSearchBot,
    canonical_action_identity,
    phase_for_turn,
    reserved_option_value,
)


OPTION_DEPTH = 3
OPTION_ROOT_BRANCHING = 8
OPTION_INTERNAL_BRANCHING = 4
OPTION_DETERMINIZATIONS = 4


OptionTuple = tuple[float, float, float, float, float]


def _target_key(card: Card, state: SplendorState, player_id: int) -> tuple[Any, ...]:
    bonuses = state.players[player_id].bonuses
    colours = sum(value > 0 for value in card.cost.values())
    return (-effective_value(card, bonuses), -card.points, colours, card.card_id)


def select_option_target(state: SplendorState, player_id: int) -> Card | None:
    player = state.players[player_id]
    reserved = [
        card
        for card in player.reserved_cards
        if effective_value(card, player.bonuses) >= EFFICIENCY_FILTER
    ]
    if reserved:
        return min(reserved, key=lambda card: _target_key(card, state, player_id))
    visible = [
        card
        for cards in state.visible_tier_cards.values()
        for card in cards
        if effective_value(card, player.bonuses) >= EFFICIENCY_FILTER
    ]
    return min(visible, key=lambda card: _target_key(card, state, player_id)) if visible else None


class PersistentCardOptionSearch:
    def __init__(self, target: Card, game_seed: int) -> None:
        self.target = target
        self.game_seed = game_seed
        self._base = GreedyHeuristicBot(seed=0)

    def choose_action(
        self, env: SplendorEnv, state: SplendorState, actions: list[Action]
    ) -> Action:
        root = state.current_player
        determinizations = [
            determinize_for_observer(
                state,
                observer_id=root,
                rng=random.Random(
                    self.game_seed * 1_000_003
                    + state.turn_index * 10_007
                    + sample * 97
                    + 79
                ),
            )
            for sample in range(OPTION_DETERMINIZATIONS)
        ]
        immediate: dict[Action, OptionTuple] = {}
        for action in actions:
            values = [self._value(env.step(sample, action), root) for sample in determinizations]
            immediate[action] = self._mean_tuple(values)
        roots = sorted(actions, key=lambda action: (immediate[action], repr(action)), reverse=True)[
            :OPTION_ROOT_BRANCHING
        ]
        final: dict[Action, OptionTuple] = {}
        for action in roots:
            values = [
                self._minimax(env, env.step(sample, action), root, OPTION_DEPTH - 1)
                for sample in determinizations
            ]
            final[action] = self._mean_tuple(values)
        return max(roots, key=lambda action: (final[action], repr(action)))

    def _minimax(
        self, env: SplendorEnv, state: SplendorState, root: int, depth: int
    ) -> OptionTuple:
        if depth <= 0 or state.terminal:
            return self._value(state, root)
        actions = env.legal_actions(state)
        if not actions:
            return self._value(state, root)
        maximizing = state.current_player == root
        scored = [(self._value(env.step(state, action), root), action) for action in actions]
        scored.sort(key=lambda item: (item[0], repr(item[1])), reverse=maximizing)
        candidates = [item[1] for item in scored[:OPTION_INTERNAL_BRANCHING]]
        values = [
            self._minimax(env, env.step(state, action), root, depth - 1)
            for action in candidates
        ]
        return max(values) if maximizing else min(values)

    def _value(self, state: SplendorState, root: int) -> OptionTuple:
        if state.terminal:
            terminal = 0.0 if state.winner is None else (1.0 if state.winner == root else -1.0)
        else:
            terminal = 0.0
        player = state.players[root]
        target_id = self.target.card_id
        purchased = any(card.card_id == target_id for card in player.purchased_cards)
        reserved = any(card.card_id == target_id for card in player.reserved_cards)
        visible = any(
            card.card_id == target_id
            for cards in state.visible_tier_cards.values()
            for card in cards
        )
        status = 3.0 if purchased else 2.0 if reserved else 1.0 if visible else 0.0
        funded = reserved_option_value(self.target, player)
        missing = float(deficit(self.target, player))
        return (terminal, status, funded, -missing, self._base.evaluate_state(state, root))

    @staticmethod
    def _mean_tuple(values: list[OptionTuple]) -> OptionTuple:
        return tuple(sum(value[index] for value in values) / len(values) for index in range(5))  # type: ignore[return-value]


@dataclass(slots=True)
class OptionPlannerTelemetry:
    decisions: int = 0
    option_decisions: int = 0
    option_selections: int = 0
    option_completions: int = 0
    option_losses: int = 0
    moved_from_raw035: int = 0
    purchases: int = 0
    deck_reserves: int = 0
    gold_spent: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    reserves_by_tier: dict[int, int] = field(default_factory=dict)
    purchases_by_tier: dict[int, int] = field(default_factory=dict)
    phase_decisions: dict[str, int] = field(default_factory=dict)
    phase_movement: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in (
            "decisions", "option_decisions", "option_selections", "option_completions",
            "option_losses", "moved_from_raw035", "purchases", "deck_reserves", "gold_spent"
        )}
        for name in ("action_counts", "reserves_by_tier", "purchases_by_tier", "phase_decisions", "phase_movement"):
            payload[name] = {str(k): v for k, v in sorted(getattr(self, name).items())}
        return payload

    def merge(self, other: dict[str, Any]) -> None:
        for name in ("decisions", "option_decisions", "option_selections", "option_completions", "option_losses", "moved_from_raw035", "purchases", "deck_reserves", "gold_spent"):
            setattr(self, name, getattr(self, name) + int(other[name]))
        for name in ("action_counts", "reserves_by_tier", "purchases_by_tier", "phase_decisions", "phase_movement"):
            target = getattr(self, name)
            for key, value in dict(other[name]).items():
                key2: Any = int(key) if name in ("reserves_by_tier", "purchases_by_tier") else str(key)
                target[key2] = target.get(key2, 0) + int(value)


class PersistentCardOptionPlannerBot:
    def __init__(self, raw035_bot: Any) -> None:
        self.raw035 = raw035_bot
        self.game_seed = 0
        self.target: Card | None = None
        self.player_id: int | None = None
        self.telemetry = OptionPlannerTelemetry()

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:
        self.game_seed = seed
        self.target = None
        self.player_id = None
        self.telemetry = OptionPlannerTelemetry()

    def choose_action(self, env: SplendorEnv, state: SplendorState, legal_actions: list[Action] | None = None) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        self.player_id = state.current_player
        if self.target is None:
            self.target = select_option_target(state, self.player_id)
            if self.target is not None:
                self.telemetry.option_selections += 1
        if self.target is None:
            chosen = FrozenFullActionSearchBot(with_bonus=False)
            chosen.start_game(self.game_seed)
            action = chosen.choose_action(env, state, actions)
        else:
            self.telemetry.option_decisions += 1
            action = PersistentCardOptionSearch(self.target, self.game_seed).choose_action(env, state, actions)
        raw = self.raw035.choose_action(env, state, actions)
        if action is None or raw is None:
            raise RuntimeError("079 bot or comparator returned no action")
        self._record(state, action, raw)
        return action

    def _record(self, state: SplendorState, action: Action, raw: Action) -> None:
        t = self.telemetry
        t.decisions += 1
        phase = phase_for_turn(state.turn_index)
        t.phase_decisions[phase] = t.phase_decisions.get(phase, 0) + 1
        name = action.action_type.name
        t.action_counts[name] = t.action_counts.get(name, 0) + 1
        if canonical_action_identity(state, action) != canonical_action_identity(state, raw):
            t.moved_from_raw035 += 1
            t.phase_movement[phase] = t.phase_movement.get(phase, 0) + 1
        t.gold_spent += sum(token == "gold" for token in action.spend_tokens)
        if action.action_type in (ActionType.RESERVE_VISIBLE, ActionType.RESERVE_DECK):
            if action.tier is not None:
                t.reserves_by_tier[action.tier] = t.reserves_by_tier.get(action.tier, 0) + 1
            if action.action_type == ActionType.RESERVE_DECK:
                t.deck_reserves += 1
        if action.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            t.purchases += 1
            card = (state.visible_tier_cards[action.tier][action.market_index]
                    if action.action_type == ActionType.BUY_VISIBLE
                    else state.players[state.current_player].reserved_cards[action.reserved_index])
            t.purchases_by_tier[card.tier] = t.purchases_by_tier.get(card.tier, 0) + 1

    def observe_action(self, before: SplendorState, action: Action, after: SplendorState) -> None:  # noqa: ARG002
        if self.target is None or self.player_id is None:
            return
        target_id = self.target.card_id
        player = after.players[self.player_id]
        if any(card.card_id == target_id for card in player.purchased_cards):
            self.telemetry.option_completions += 1
            self.target = None
            return
        secure = any(card.card_id == target_id for card in player.reserved_cards)
        visible = any(card.card_id == target_id for cards in after.visible_tier_cards.values() for card in cards)
        if not secure and not visible:
            self.telemetry.option_losses += 1
            self.target = None
