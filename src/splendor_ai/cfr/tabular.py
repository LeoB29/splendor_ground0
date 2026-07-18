"""Reference exact CFR and external-sampling MCCFR for finite game adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

from .reduced_splendor import ActionKey, InformationKey, ReducedSplendorCFRGame, ReducedSplendorCFRState


@dataclass(slots=True)
class RegretNode:
    player: int
    actions: tuple[ActionKey, ...]
    regrets: list[float] = field(default_factory=list)
    strategy_sum: list[float] = field(default_factory=list)
    average_observations: int = 0

    def __post_init__(self) -> None:
        if not self.regrets:
            self.regrets = [0.0] * len(self.actions)
        if not self.strategy_sum:
            self.strategy_sum = [0.0] * len(self.actions)

    def strategy(self) -> list[float]:
        positive = [max(value, 0.0) for value in self.regrets]
        total = sum(positive)
        if total <= 0.0:
            return [1.0 / len(self.actions)] * len(self.actions)
        return [value / total for value in positive]

    def average_strategy(self) -> list[float]:
        total = sum(self.strategy_sum)
        if total <= 0.0:
            return self.strategy()
        return [value / total for value in self.strategy_sum]


class _BaseCFRSolver:
    def __init__(self, game: ReducedSplendorCFRGame) -> None:
        self.game = game
        self.nodes: dict[InformationKey, RegretNode] = {}
        self.illegal_actions = 0
        self.unstable_action_sets = 0

    def _node(self, state: ReducedSplendorCFRState) -> RegretNode:
        key = self.game.information_state_key(state)
        actions = self.game.legal_action_keys(state)
        player = self.game.current_player(state)
        node = self.nodes.get(key)
        if node is None:
            node = RegretNode(player=player, actions=actions)
            self.nodes[key] = node
        elif node.player != player or node.actions != actions:
            self.unstable_action_sets += 1
            raise ValueError("Unstable action set for one information state.")
        return node

    def normalized_positive_regret(self, iterations: int) -> dict[str, float]:
        values: dict[str, float] = {}
        for player in (0, 1):
            player_nodes = [node for node in self.nodes.values() if node.player == player]
            positive = sum(sum(max(value, 0.0) for value in node.regrets) for node in player_nodes)
            values[str(player)] = positive / (max(iterations, 1) * max(len(player_nodes), 1))
        return values

    def average_policy(self) -> dict[InformationKey, tuple[float, ...]]:
        return {key: tuple(node.average_strategy()) for key, node in self.nodes.items()}

    def root_key(self) -> InformationKey:
        children = [self.game.apply_chance(self.game.root(), outcome) for outcome in (0, 1)]
        keys = [self.game.information_state_key(child) for child in children]
        if keys[0] != keys[1]:
            raise ValueError("Root hidden order leaked into the acting-player information state.")
        return keys[0]

    def payload(self, iterations: int, snapshots: dict[str, dict[str, float]]) -> dict[str, Any]:
        root = self.nodes[self.root_key()]
        ordered = sorted(self.nodes.items(), key=lambda item: repr(item[0]))
        return {
            "iterations": iterations,
            "information_sets": len(self.nodes),
            "information_sets_by_player": {
                str(player): sum(node.player == player for node in self.nodes.values())
                for player in (0, 1)
            },
            "normalized_positive_regret": self.normalized_positive_regret(iterations),
            "regret_snapshots": snapshots,
            "root_actions": [repr(action) for action in root.actions],
            "root_average_strategy": root.average_strategy(),
            "expected_average_self_play_utility_p0": expected_utility(
                self.game,
                self.average_policy(),
                self.average_policy(),
            ),
            "illegal_actions": self.illegal_actions,
            "unstable_action_sets": self.unstable_action_sets,
            "nodes": [
                {
                    "key": repr(key),
                    "player": node.player,
                    "actions": [repr(action) for action in node.actions],
                    "regrets": node.regrets,
                    "average_strategy": node.average_strategy(),
                    "average_observations": node.average_observations,
                }
                for key, node in ordered
            ],
        }


class ExactCFRSolver(_BaseCFRSolver):
    def run(self, iterations: int, snapshot_iterations: tuple[int, ...] = (100,)) -> dict[str, Any]:
        snapshots: dict[str, dict[str, float]] = {}
        for iteration in range(1, iterations + 1):
            for traverser in (0, 1):
                self._traverse(self.game.root(), traverser, (1.0, 1.0), 1.0)
            if iteration in snapshot_iterations:
                snapshots[str(iteration)] = self.normalized_positive_regret(iteration)
        snapshots[str(iterations)] = self.normalized_positive_regret(iterations)
        return self.payload(iterations, snapshots)

    def _traverse(
        self,
        state: ReducedSplendorCFRState,
        traverser: int,
        reach: tuple[float, float],
        chance_reach: float,
    ) -> float:
        if self.game.is_terminal(state):
            return self.game.utility(state, traverser)
        if self.game.is_chance_node(state):
            return sum(
                probability
                * self._traverse(
                    self.game.apply_chance(state, outcome),
                    traverser,
                    reach,
                    chance_reach * probability,
                )
                for outcome, probability in self.game.chance_outcomes(state)
            )

        player = self.game.current_player(state)
        node = self._node(state)
        strategy = node.strategy()
        if player == traverser:
            own_reach = reach[player]
            for index, probability in enumerate(strategy):
                node.strategy_sum[index] += chance_reach * own_reach * probability
            node.average_observations += 1

        action_utilities: list[float] = []
        for action, probability in zip(node.actions, strategy, strict=True):
            next_reach = list(reach)
            next_reach[player] *= probability
            action_utilities.append(
                self._traverse(
                    self.game.apply_action(state, action),
                    traverser,
                    (next_reach[0], next_reach[1]),
                    chance_reach,
                )
            )
        node_utility = sum(
            probability * utility
            for probability, utility in zip(strategy, action_utilities, strict=True)
        )
        if player == traverser:
            counterfactual_reach = chance_reach * reach[1 - player]
            for index, utility in enumerate(action_utilities):
                node.regrets[index] += counterfactual_reach * (utility - node_utility)
        return node_utility


class ExternalSamplingMCCFRSolver(_BaseCFRSolver):
    def __init__(self, game: ReducedSplendorCFRGame, seed: int) -> None:
        super().__init__(game)
        self.rng = random.Random(seed)

    def run(self, iterations: int, snapshot_iterations: tuple[int, ...] = (100,)) -> dict[str, Any]:
        snapshots: dict[str, dict[str, float]] = {}
        for iteration in range(1, iterations + 1):
            for traverser in (0, 1):
                self._traverse(self.game.root(), traverser, 1.0)
            if iteration in snapshot_iterations:
                snapshots[str(iteration)] = self.normalized_positive_regret(iteration)
        snapshots[str(iterations)] = self.normalized_positive_regret(iterations)
        return self.payload(iterations, snapshots)

    def _traverse(
        self,
        state: ReducedSplendorCFRState,
        traverser: int,
        own_reach: float,
    ) -> float:
        if self.game.is_terminal(state):
            return self.game.utility(state, traverser)
        if self.game.is_chance_node(state):
            outcomes = self.game.chance_outcomes(state)
            selected = _sample_index([probability for _outcome, probability in outcomes], self.rng)
            return self._traverse(
                self.game.apply_chance(state, outcomes[selected][0]),
                traverser,
                own_reach,
            )

        player = self.game.current_player(state)
        node = self._node(state)
        strategy = node.strategy()
        if player != traverser:
            selected = _sample_index(strategy, self.rng)
            return self._traverse(
                self.game.apply_action(state, node.actions[selected]),
                traverser,
                own_reach,
            )

        for index, probability in enumerate(strategy):
            node.strategy_sum[index] += own_reach * probability
        node.average_observations += 1
        action_utilities = [
            self._traverse(
                self.game.apply_action(state, action),
                traverser,
                own_reach * probability,
            )
            for action, probability in zip(node.actions, strategy, strict=True)
        ]
        node_utility = sum(
            probability * utility
            for probability, utility in zip(strategy, action_utilities, strict=True)
        )
        for index, utility in enumerate(action_utilities):
            node.regrets[index] += utility - node_utility
        return node_utility


def expected_utility(
    game: ReducedSplendorCFRGame,
    policy0: dict[InformationKey, tuple[float, ...]],
    policy1: dict[InformationKey, tuple[float, ...]],
) -> float:
    def recurse(state: ReducedSplendorCFRState) -> float:
        if game.is_terminal(state):
            return game.utility(state, 0)
        if game.is_chance_node(state):
            return sum(
                probability * recurse(game.apply_chance(state, outcome))
                for outcome, probability in game.chance_outcomes(state)
            )
        player = game.current_player(state)
        key = game.information_state_key(state)
        actions = game.legal_action_keys(state)
        strategy = (policy0 if player == 0 else policy1).get(key)
        if strategy is None or len(strategy) != len(actions):
            strategy = tuple([1.0 / len(actions)] * len(actions))
        return sum(
            probability * recurse(game.apply_action(state, action))
            for action, probability in zip(actions, strategy, strict=True)
        )

    return recurse(game.root())


def strategy_total_variation(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("Strategy action counts differ.")
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(int(math.ceil(quantile * len(ordered))) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def _sample_index(probabilities: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1
