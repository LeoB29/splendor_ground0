"""Search-executed value-reserve teacher for experiment `077`.

The frozen contract is documented in
``docs/experiments/value_reserve_leaf_search_teacher_077.md``.  This module has
no learned component of its own and never reads human replays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import Card, PlayerState, SplendorState

from .heuristic_bot import GreedyHeuristicBot
from .search_bot import ShallowSearchBot
from .value_reserve_bot import EFFICIENCY_FILTER, deficit, effective_cost, effective_value


PHASES = ("early", "mid", "late")
SEARCH_DEPTH = 1
SEARCH_DETERMINIZATIONS = 4
UNBOUNDED_ACTION_CAP = 10_000


def phase_for_turn(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


def reserved_option_value(card: Card, player: PlayerState) -> float:
    """Frozen `077` funded-prestige value for one reserved card."""
    cost = effective_cost(card, player.bonuses)
    if effective_value(card, player.bonuses) < EFFICIENCY_FILTER:
        return 0.0
    progress = 1.0 if cost == 0 else 1.0 - (deficit(card, player) / cost)
    progress = min(max(progress, 0.0), 1.0)
    return 1_000.0 * card.points * progress


class ValueReserveLeafEvaluator:
    """Greedy state value plus the best funded qualifying hand target."""

    def __init__(self) -> None:
        self._base = GreedyHeuristicBot(seed=0)

    @staticmethod
    def strategic_bonus(state: SplendorState, player_id: int) -> float:
        player = state.players[player_id]
        return max(
            (reserved_option_value(card, player) for card in player.reserved_cards),
            default=0.0,
        )

    def evaluate_state(self, state: SplendorState, player_id: int) -> float:
        return self._base.evaluate_state(state, player_id) + self.strategic_bonus(
            state, player_id
        )


def canonical_action_identity(state: SplendorState, action: Action) -> tuple[Any, ...]:
    """Strategic action identity, ignoring buy-payment aliases."""
    action_type = action.action_type
    card_id: str | None = None
    if action_type in (ActionType.RESERVE_VISIBLE, ActionType.BUY_VISIBLE):
        if action.tier is not None and action.market_index is not None:
            card_id = state.visible_tier_cards[action.tier][action.market_index].card_id
    elif action_type == ActionType.BUY_RESERVED and action.reserved_index is not None:
        card_id = state.players[state.current_player].reserved_cards[
            action.reserved_index
        ].card_id
    return (
        action_type.name,
        card_id,
        action.tier if action_type == ActionType.RESERVE_DECK else None,
        tuple(sorted(action.take_tokens)),
        tuple(sorted(action.return_tokens)),
        action.noble_id,
    )


@dataclass(slots=True)
class ValueReserveSearchTelemetry:
    decisions: int = 0
    moved_from_raw035: int = 0
    moved_from_search_ablation: int = 0
    decisions_with_live_option: int = 0
    qualifying_visible_reserves: int = 0
    converted_qualifying_visible_reserves: int = 0
    reserves: int = 0
    deck_reserves: int = 0
    reserved_purchases: int = 0
    purchases: int = 0
    gold_spent: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    reserves_by_tier: dict[int, int] = field(default_factory=dict)
    purchases_by_tier: dict[int, int] = field(default_factory=dict)
    phase_decisions: dict[str, int] = field(default_factory=dict)
    phase_moved_from_raw035: dict[str, int] = field(default_factory=dict)
    phase_moved_from_search_ablation: dict[str, int] = field(default_factory=dict)
    movement_by_action_type: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "moved_from_raw035": self.moved_from_raw035,
            "moved_from_search_ablation": self.moved_from_search_ablation,
            "decisions_with_live_option": self.decisions_with_live_option,
            "qualifying_visible_reserves": self.qualifying_visible_reserves,
            "converted_qualifying_visible_reserves": (
                self.converted_qualifying_visible_reserves
            ),
            "reserves": self.reserves,
            "deck_reserves": self.deck_reserves,
            "reserved_purchases": self.reserved_purchases,
            "purchases": self.purchases,
            "gold_spent": self.gold_spent,
            "action_counts": dict(sorted(self.action_counts.items())),
            "reserves_by_tier": {
                str(key): value for key, value in sorted(self.reserves_by_tier.items())
            },
            "purchases_by_tier": {
                str(key): value for key, value in sorted(self.purchases_by_tier.items())
            },
            "phase_decisions": dict(sorted(self.phase_decisions.items())),
            "phase_moved_from_raw035": dict(
                sorted(self.phase_moved_from_raw035.items())
            ),
            "phase_moved_from_search_ablation": dict(
                sorted(self.phase_moved_from_search_ablation.items())
            ),
            "movement_by_action_type": dict(sorted(self.movement_by_action_type.items())),
        }

    def merge(self, other: dict[str, Any]) -> None:
        for name in (
            "decisions",
            "moved_from_raw035",
            "moved_from_search_ablation",
            "decisions_with_live_option",
            "qualifying_visible_reserves",
            "converted_qualifying_visible_reserves",
            "reserves",
            "deck_reserves",
            "reserved_purchases",
            "purchases",
            "gold_spent",
        ):
            setattr(self, name, getattr(self, name) + int(other[name]))
        for name in (
            "action_counts",
            "phase_decisions",
            "phase_moved_from_raw035",
            "phase_moved_from_search_ablation",
            "movement_by_action_type",
        ):
            target = getattr(self, name)
            for key, value in dict(other[name]).items():
                target[str(key)] = target.get(str(key), 0) + int(value)
        for name in ("reserves_by_tier", "purchases_by_tier"):
            target = getattr(self, name)
            for key, value in dict(other[name]).items():
                tier = int(key)
                target[tier] = target.get(tier, 0) + int(value)


class ValueReserveLeafSearchBot:
    """Stateless-per-decision `077` search plus outcome-blind shadow choices."""

    def __init__(self, raw035_bot: Any | None) -> None:
        self._raw035_bot = raw035_bot
        self._game_seed = 0
        self.telemetry = ValueReserveSearchTelemetry()
        self._qualifying_reserved_ids: set[str] = set()

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:
        self._game_seed = seed
        self.telemetry = ValueReserveSearchTelemetry()
        self._qualifying_reserved_ids = set()

    def _decision_seed(self, state: SplendorState) -> int:
        return (
            self._game_seed * 1_000_003
            + state.turn_index * 1_009
            + state.current_player * 97
            + 77
        )

    @staticmethod
    def _search(seed: int, with_bonus: bool) -> ShallowSearchBot:
        return ShallowSearchBot(
            depth=SEARCH_DEPTH,
            max_branching=UNBOUNDED_ACTION_CAP,
            max_buy_actions=UNBOUNDED_ACTION_CAP,
            max_reserve_actions=UNBOUNDED_ACTION_CAP,
            max_take_actions=UNBOUNDED_ACTION_CAP,
            max_pass_actions=UNBOUNDED_ACTION_CAP,
            seed=seed,
            value_evaluator=ValueReserveLeafEvaluator() if with_bonus else None,
            value_weight=1.0 if with_bonus else 0.0,
            information_legal=True,
            determinizations=SEARCH_DETERMINIZATIONS,
            determinization_seed=seed + 7_700_077,
        )

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        seed = self._decision_seed(state)
        chosen = self._search(seed, with_bonus=True).choose_action(env, state, actions)
        ablated = self._search(seed, with_bonus=False).choose_action(env, state, actions)
        raw035 = (
            self._raw035_bot.choose_action(env, state, actions)
            if self._raw035_bot is not None
            else None
        )
        if chosen is None or ablated is None:
            raise RuntimeError("A `077` comparator returned no legal action.")
        self._record(state, chosen, ablated, raw035)
        return chosen

    def _record(
        self,
        state: SplendorState,
        chosen: Action,
        ablated: Action,
        raw035: Action | None,
    ) -> None:
        telemetry = self.telemetry
        telemetry.decisions += 1
        phase = phase_for_turn(state.turn_index)
        telemetry.phase_decisions[phase] = telemetry.phase_decisions.get(phase, 0) + 1
        action_name = chosen.action_type.name
        telemetry.action_counts[action_name] = telemetry.action_counts.get(action_name, 0) + 1
        if ValueReserveLeafEvaluator.strategic_bonus(state, state.current_player) > 0.0:
            telemetry.decisions_with_live_option += 1

        chosen_identity = canonical_action_identity(state, chosen)
        if raw035 is not None and chosen_identity != canonical_action_identity(state, raw035):
            telemetry.moved_from_raw035 += 1
            telemetry.phase_moved_from_raw035[phase] = (
                telemetry.phase_moved_from_raw035.get(phase, 0) + 1
            )
            telemetry.movement_by_action_type[action_name] = (
                telemetry.movement_by_action_type.get(action_name, 0) + 1
            )
        if chosen_identity != canonical_action_identity(state, ablated):
            telemetry.moved_from_search_ablation += 1
            telemetry.phase_moved_from_search_ablation[phase] = (
                telemetry.phase_moved_from_search_ablation.get(phase, 0) + 1
            )

        actor = state.players[state.current_player]
        telemetry.gold_spent += sum(token == "gold" for token in chosen.spend_tokens)
        if chosen.action_type == ActionType.RESERVE_VISIBLE:
            telemetry.reserves += 1
            assert chosen.tier is not None and chosen.market_index is not None
            telemetry.reserves_by_tier[chosen.tier] = (
                telemetry.reserves_by_tier.get(chosen.tier, 0) + 1
            )
            card = state.visible_tier_cards[chosen.tier][chosen.market_index]
            if effective_value(card, actor.bonuses) >= EFFICIENCY_FILTER:
                telemetry.qualifying_visible_reserves += 1
                self._qualifying_reserved_ids.add(card.card_id)
        elif chosen.action_type == ActionType.RESERVE_DECK:
            telemetry.reserves += 1
            telemetry.deck_reserves += 1
            if chosen.tier is not None:
                telemetry.reserves_by_tier[chosen.tier] = (
                    telemetry.reserves_by_tier.get(chosen.tier, 0) + 1
                )
        elif chosen.action_type in (ActionType.BUY_VISIBLE, ActionType.BUY_RESERVED):
            telemetry.purchases += 1
            card: Card
            if chosen.action_type == ActionType.BUY_VISIBLE:
                assert chosen.tier is not None and chosen.market_index is not None
                card = state.visible_tier_cards[chosen.tier][chosen.market_index]
            else:
                assert chosen.reserved_index is not None
                card = actor.reserved_cards[chosen.reserved_index]
                telemetry.reserved_purchases += 1
                if card.card_id in self._qualifying_reserved_ids:
                    telemetry.converted_qualifying_visible_reserves += 1
                    self._qualifying_reserved_ids.remove(card.card_id)
            telemetry.purchases_by_tier[card.tier] = (
                telemetry.purchases_by_tier.get(card.tier, 0) + 1
            )


class FrozenFullActionSearchBot:
    """Pure frozen `077` search player without outcome-blind shadow calls."""

    def __init__(self, with_bonus: bool) -> None:
        self.with_bonus = with_bonus
        self._game_seed = 0

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def start_game(self, seed: int) -> None:
        self._game_seed = seed

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        seed = (
            self._game_seed * 1_000_003
            + state.turn_index * 1_009
            + state.current_player * 97
            + 77
        )
        return ValueReserveLeafSearchBot._search(seed, self.with_bonus).choose_action(
            env, state, actions
        )
