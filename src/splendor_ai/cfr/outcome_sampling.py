"""Outcome-sampling MCCFR estimators for reduced and full-game diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable

from splendor_ai.encoding.entity_encoding import (
    ENTITY_HIDDEN_DECK_CARD,
    ENTITY_PRIVILEGED_OPPONENT_BLIND_RESERVED,
    EntityActionEncoding,
    EntityStateEncoding,
)
from .full_game import PublicHistoryEncoding
from .reduced_splendor import ActionKey, InformationKey, ReducedSplendorCFRGame, ReducedSplendorCFRState
from .tabular import RegretNode, _BaseCFRSolver, _sample_index


@dataclass(frozen=True, slots=True)
class OutcomeSamplingUpdate:
    information_key: InformationKey
    player: int
    actions: tuple[ActionKey, ...]
    target_strategy: tuple[float, ...]
    behavior_strategy: tuple[float, ...]
    selected_index: int
    regret_estimate: tuple[float, ...]
    log_importance_weight: float


@dataclass(frozen=True, slots=True)
class OutcomeSamplingAdvantageRow:
    state: EntityStateEncoding
    history: PublicHistoryEncoding
    actions: EntityActionEncoding
    target_strategy: tuple[float, ...]
    behavior_strategy: tuple[float, ...]
    selected_index: int
    regret_target: tuple[float, ...]
    player: int
    turn: int
    terminal_reason: str
    log_importance_weight: float
    provenance: str = "outcome_sampling_048"

    def __post_init__(self) -> None:
        count = len(self.actions.action_type_ids)
        if not (
            count
            == len(self.target_strategy)
            == len(self.behavior_strategy)
            == len(self.regret_target)
        ):
            raise ValueError("Outcome-sampling vectors must align with legal actions.")
        if not 0 <= self.selected_index < count:
            raise ValueError("selected_index is outside the legal action set.")
        if self.player not in (0, 1):
            raise ValueError("player must be 0 or 1.")
        if self.provenance != "outcome_sampling_048":
            raise ValueError("Unexpected outcome-sampling provenance.")
        if any(
            value in (ENTITY_HIDDEN_DECK_CARD, ENTITY_PRIVILEGED_OPPONENT_BLIND_RESERVED)
            for value in self.state.entity_type_ids
        ):
            raise ValueError("Privileged entities are forbidden in advantage rows.")
        if not all(math.isfinite(value) for value in self.regret_target):
            raise ValueError("Regret targets must be finite and unclipped.")
        if not math.isfinite(self.log_importance_weight):
            raise ValueError("Log importance weight must be finite.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": _encoding_payload(self.state),
            "history": _encoding_payload(self.history),
            "actions": _encoding_payload(self.actions),
            "target_strategy": list(self.target_strategy),
            "behavior_strategy": list(self.behavior_strategy),
            "selected_index": self.selected_index,
            "regret_target": list(self.regret_target),
            "player": self.player,
            "turn": self.turn,
            "terminal_reason": self.terminal_reason,
            "log_importance_weight": self.log_importance_weight,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class _SampledDecision:
    key: InformationKey
    node: RegretNode
    strategy: tuple[float, ...]
    behavior: tuple[float, ...]
    selected: int
    step_index: int
    log_opponent_chance_prefix: float


StrategyProvider = Callable[
    [ReducedSplendorCFRState, RegretNode], tuple[float, ...] | list[float]
]


def exact_fixed_policy_regret_increments(
    game: ReducedSplendorCFRGame,
    traverser: int,
) -> dict[InformationKey, tuple[float, ...]]:
    """Enumerate exact counterfactual increments under a fixed uniform policy."""

    increments: dict[InformationKey, list[float]] = {}

    def recurse(
        state: ReducedSplendorCFRState,
        reach: tuple[float, float],
        chance_reach: float,
    ) -> float:
        if game.is_terminal(state):
            return game.utility(state, traverser)
        if game.is_chance_node(state):
            return sum(
                probability
                * recurse(game.apply_chance(state, outcome), reach, chance_reach * probability)
                for outcome, probability in game.chance_outcomes(state)
            )
        player = game.current_player(state)
        actions = game.legal_action_keys(state)
        strategy = tuple([1.0 / len(actions)] * len(actions))
        action_utilities: list[float] = []
        for action, probability in zip(actions, strategy, strict=True):
            next_reach = list(reach)
            next_reach[player] *= probability
            action_utilities.append(
                recurse(
                    game.apply_action(state, action),
                    (next_reach[0], next_reach[1]),
                    chance_reach,
                )
            )
        node_utility = sum(
            probability * utility
            for probability, utility in zip(strategy, action_utilities, strict=True)
        )
        if player == traverser:
            key = game.information_state_key(state)
            values = increments.setdefault(key, [0.0] * len(actions))
            counterfactual_reach = chance_reach * reach[1 - player]
            for index, utility in enumerate(action_utilities):
                values[index] += counterfactual_reach * (utility - node_utility)
        return node_utility

    recurse(game.root(), (1.0, 1.0), 1.0)
    return {key: tuple(values) for key, values in increments.items()}


def behavior_mixture(
    target_strategy: tuple[float, ...] | list[float],
    exploration: float,
) -> tuple[float, ...]:
    if not 0.0 < exploration < 1.0:
        raise ValueError("exploration must lie strictly between zero and one.")
    if not target_strategy:
        raise ValueError("target_strategy must not be empty.")
    values = [float(value) for value in target_strategy]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("target strategy must be finite and nonnegative.")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("target strategy must have positive mass.")
    normalized = [value / total for value in values]
    uniform = 1.0 / len(normalized)
    mixed = tuple((1.0 - exploration) * value + exploration * uniform for value in normalized)
    if min(mixed) <= 0.0 or not math.isclose(sum(mixed), 1.0, abs_tol=1e-12):
        raise ValueError("behavior mixture did not produce full-support probabilities.")
    return mixed


def sample_reduced_outcome_updates(
    game: ReducedSplendorCFRGame,
    traverser: int,
    rng: random.Random,
    node_lookup: Callable[[ReducedSplendorCFRState], RegretNode],
    strategy_provider: StrategyProvider,
    *,
    exploration: float = 0.1,
) -> tuple[list[OutcomeSamplingUpdate], float, list[float]]:
    """Sample one terminal and return unbiased regret increments along its path."""

    state = game.root()
    sampled_log_target_steps: list[float] = []
    sampled: list[_SampledDecision] = []
    log_behavior_reach = 0.0
    log_opponent_chance_prefix = 0.0

    while not game.is_terminal(state):
        if game.is_chance_node(state):
            outcomes = game.chance_outcomes(state)
            probabilities = [float(probability) for _outcome, probability in outcomes]
            selected = _sample_index(probabilities, rng)
            probability = probabilities[selected]
            log_probability = math.log(probability)
            sampled_log_target_steps.append(log_probability)
            log_behavior_reach += log_probability
            log_opponent_chance_prefix += log_probability
            state = game.apply_chance(state, outcomes[selected][0])
            continue

        player = game.current_player(state)
        node = node_lookup(state)
        strategy = tuple(float(value) for value in strategy_provider(state, node))
        if len(strategy) != len(node.actions):
            raise ValueError("Strategy action count does not match the information set.")
        behavior = behavior_mixture(strategy, exploration)
        selected = _sample_index(list(behavior), rng)
        step_index = len(sampled_log_target_steps)
        if player == traverser:
            sampled.append(
                _SampledDecision(
                    key=game.information_state_key(state),
                    node=node,
                    strategy=strategy,
                    behavior=behavior,
                    selected=selected,
                    step_index=step_index,
                    log_opponent_chance_prefix=log_opponent_chance_prefix,
                )
            )
        selected_target = strategy[selected]
        sampled_log_target_steps.append(
            math.log(selected_target) if selected_target > 0.0 else -math.inf
        )
        log_behavior_reach += math.log(behavior[selected])
        if player != traverser:
            log_opponent_chance_prefix += sampled_log_target_steps[-1]
        state = game.apply_action(state, node.actions[selected])

    utility = game.utility(state, traverser)
    suffix_logs = _exclusive_suffix_sums(sampled_log_target_steps)
    updates: list[OutcomeSamplingUpdate] = []
    importance_logs: list[float] = []
    for record in sampled:
        log_weight = (
            record.log_opponent_chance_prefix
            + suffix_logs[record.step_index]
            - log_behavior_reach
        )
        importance_logs.append(log_weight)
        magnitude = _signed_exp(utility, log_weight)
        selected_probability = record.strategy[record.selected]
        regrets = tuple(
            ((1.0 - selected_probability) if index == record.selected else -selected_probability)
            * magnitude
            for index in range(len(record.strategy))
        )
        updates.append(
            OutcomeSamplingUpdate(
                information_key=record.key,
                player=traverser,
                actions=record.node.actions,
                target_strategy=record.strategy,
                behavior_strategy=record.behavior,
                selected_index=record.selected,
                regret_estimate=regrets,
                log_importance_weight=log_weight,
            )
        )
    return updates, utility, importance_logs


class OutcomeSamplingMCCFRSolver(_BaseCFRSolver):
    """Single-trajectory outcome-sampling MCCFR diagnostic solver."""

    def __init__(self, game: ReducedSplendorCFRGame, seed: int, exploration: float = 0.1) -> None:
        super().__init__(game)
        self.seed = seed
        self.exploration = exploration
        self.rng = random.Random(seed)
        self.sampled_trajectories = 0
        self.sampled_updates = 0
        self.max_log_importance_weight = -math.inf

    def run(self, iterations: int, snapshot_iterations: tuple[int, ...] = (100,)) -> dict[str, Any]:
        snapshots: dict[str, dict[str, float]] = {}
        for iteration in range(1, iterations + 1):
            for traverser in (0, 1):
                updates, _utility, logs = sample_reduced_outcome_updates(
                    self.game,
                    traverser,
                    self.rng,
                    self._node,
                    lambda _state, node: node.strategy(),
                    exploration=self.exploration,
                )
                self.sampled_trajectories += 1
                self.sampled_updates += len(updates)
                if logs:
                    self.max_log_importance_weight = max(self.max_log_importance_weight, max(logs))
                for update in updates:
                    node = self.nodes[update.information_key]
                    for index, value in enumerate(update.regret_estimate):
                        node.regrets[index] += value
            if iteration in snapshot_iterations:
                snapshots[str(iteration)] = self.normalized_positive_regret(iteration)
        snapshots[str(iterations)] = self.normalized_positive_regret(iterations)
        payload = self.payload(iterations, snapshots)
        payload["estimator"] = "outcome_sampling_unclipped_logspace_v1"
        payload["seed"] = self.seed
        payload["exploration"] = self.exploration
        payload["sampled_trajectories"] = self.sampled_trajectories
        payload["sampled_updates"] = self.sampled_updates
        payload["max_log_importance_weight"] = self.max_log_importance_weight
        payload["current_policy"] = [
            {
                "key": repr(key),
                "strategy": node.strategy(),
            }
            for key, node in sorted(self.nodes.items(), key=lambda item: repr(item[0]))
        ]
        return payload

    def current_policy(self) -> dict[InformationKey, tuple[float, ...]]:
        return {key: tuple(node.strategy()) for key, node in self.nodes.items()}


def _exclusive_suffix_sums(log_steps: list[float]) -> list[float]:
    suffix = [0.0] * len(log_steps)
    running = 0.0
    for index in range(len(log_steps) - 1, -1, -1):
        suffix[index] = running
        value = log_steps[index]
        if running == -math.inf or value == -math.inf:
            running = -math.inf
        else:
            running += value
    return suffix


def _signed_exp(sign: float, log_magnitude: float) -> float:
    if sign == 0.0 or log_magnitude == -math.inf:
        return 0.0
    magnitude = math.exp(log_magnitude)
    return math.copysign(magnitude, sign)


def _encoding_payload(encoding: Any) -> dict[str, Any]:
    type_ids = encoding.entity_type_ids if hasattr(encoding, "entity_type_ids") else encoding.action_type_ids
    type_key = "entity_type_ids" if hasattr(encoding, "entity_type_ids") else "action_type_ids"
    return {
        type_key: list(type_ids),
        "card_ids": list(encoding.card_ids),
        "noble_ids": list(encoding.noble_ids),
        "numeric_features": [list(row) for row in encoding.numeric_features],
    }
