"""Information-legal PUCT search with observer-relative determinizations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import time
from typing import Any

from splendor_ai.encoding import ActionCodec, encode_public_observation_tensor
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import (
    SplendorState,
)

from .hidden_information import determinize_for_observer
from .heuristic_bot import GreedyHeuristicBot
from .search_bot import CheckpointLegalActionPriorScorer, CheckpointValueEvaluator


@dataclass(frozen=True, slots=True)
class PuctConfig:
    simulations: int = 64
    c_puct: float = 1.5
    prior_temperature: float = 1.0
    root_prior_mix: float = 1.0
    fpu_reduction: float = 0.1
    max_depth: int = 80
    heuristic_scale: float = 4_000.0
    seed: int = 0
    time_limit_seconds: float | None = None
    inference_batch_size: int = 8
    reuse_tree: bool = False
    max_tree_nodes: int = 50_000
    value_min_turn_index: int | None = None
    value_max_turn_index: int | None = None
    n_step_leaf_depth: int = 0
    n_step_leaf_width: int = 1
    n_step_leaf_candidate_limit: int = 8
    n_step_leaf_weight: float = 1.0
    n_step_leaf_delta_clip: float | None = None
    n_step_leaf_min_turn_index: int | None = None
    n_step_leaf_max_turn_index: int | None = None
    n_step_leaf_min_max_score: int | None = None
    rollout_leaf_count: int = 0
    rollout_leaf_max_depth: int = 12
    rollout_leaf_top_k: int = 3
    rollout_leaf_temperature: float = 0.5
    rollout_leaf_policy: str = "random"


@dataclass(slots=True)
class _Edge:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class _Node:
    player_id: int
    edges: dict[int, _Edge]
    visits: int = 0


@dataclass(slots=True)
class PuctSearchStats:
    simulations: int = 0
    expansions: int = 0
    neural_calls: int = 0
    value_calls: int = 0
    nodes_reused: int = 0
    max_depth_reached: int = 0
    prior_batch_calls: int = 0
    value_batch_calls: int = 0
    n_step_leaf_evaluations: int = 0
    n_step_leaf_cache_hits: int = 0
    n_step_leaf_clipped_evaluations: int = 0
    n_step_leaf_absolute_delta_sum: float = 0.0
    n_step_leaf_applied_absolute_delta_sum: float = 0.0
    rollout_leaf_evaluations: int = 0
    retained_tree_nodes_start: int = 0
    retained_tree_nodes_end: int = 0
    root_reused: bool = False
    determinization_seconds: float = 0.0
    traversal_seconds: float = 0.0
    inference_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    selected_action_index: int | None = None
    root_prior_entropy: float = 0.0
    root_prior_normalized_entropy: float = 0.0
    root_prior_top1_probability: float = 0.0
    root_prior_effective_actions: float = 0.0
    root_visit_entropy: float = 0.0
    root_visit_normalized_entropy: float = 0.0
    root_visit_effective_actions: float = 0.0
    root_visited_actions: int = 0
    root_edge_stats: dict[int, dict[str, float | int]] = field(
        default_factory=dict
    )
    root_action_values_by_determinization: dict[int, list[float]] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class _PathStep:
    node: _Node
    edge: _Edge
    action_index: int
    parent_player: int
    child_player: int
    virtual_value: float


@dataclass(slots=True)
class _PendingSimulation:
    state: SplendorState
    path: list[_PathStep]
    depth: int
    expansion_key: tuple[Any, ...] | None = None
    legal_actions: list[Action] | None = None
    terminal_value: float | None = None


class InformationSetPuctBot:
    """PUCT over acting-player observations with root determinizations."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        config: PuctConfig | None = None,
        value_checkpoint_path: str | Path | None = None,
        value_prediction_mode: str = "value",
        value_outcome_blend_weight: float = 0.5,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._config = config or PuctConfig()
        _validate_config(self._config)
        self._scorer = CheckpointLegalActionPriorScorer(
            self._checkpoint_path,
            device,
        )
        self._value_evaluator = (
            CheckpointValueEvaluator(
                value_checkpoint_path,
                device=device,
                value_scale=1.0,
                prediction_mode=value_prediction_mode,
                outcome_blend_weight=value_outcome_blend_weight,
            )
            if value_checkpoint_path is not None
            else None
        )
        self._codec = ActionCodec()
        self._leaf_heuristic = GreedyHeuristicBot(seed=self._config.seed)
        self._last_search_stats = PuctSearchStats()
        self._last_action_metadata: dict[str, Any] = {}
        self._search_history: list[PuctSearchStats] = []
        self._tree: dict[tuple[Any, ...], _Node] = {}
        self._n_step_leaf_cache: dict[tuple[Any, ...], float] = {}
        self._last_turn_index: int | None = None
        self._root_player: int | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def puct_config(self) -> PuctConfig:
        return self._config

    @property
    def last_search_stats(self) -> PuctSearchStats:
        return self._last_search_stats

    @property
    def last_action_metadata(self) -> dict[str, Any]:
        return dict(self._last_action_metadata)

    @property
    def search_history(self) -> tuple[PuctSearchStats, ...]:
        return tuple(self._search_history)

    @property
    def loop_fallback_triggers(self) -> int:
        return 0

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        if len(actions) == 1:
            action_index = self._codec.encode(state, actions[0])
            self._last_action_metadata = {
                "root_visit_distribution": [
                    {
                        "action_index": action_index,
                        "visits": 1,
                        "probability": 1.0,
                    }
                ],
                "root_prior_entropy": 0.0,
                "root_prior_normalized_entropy": 0.0,
                "root_prior_top1_probability": 1.0,
                "root_prior_effective_actions": 1.0,
                "root_visit_entropy": 0.0,
                "root_visit_normalized_entropy": 0.0,
                "root_visit_effective_actions": 1.0,
                "root_visited_actions": 1,
            }
            return actions[0]

        root_player = state.current_player
        stats = PuctSearchStats()
        self._n_step_leaf_cache.clear()
        started = time.perf_counter()
        root_key = information_state_key(state, root_player, actions, self._codec)
        if (
            not self._config.reuse_tree
            or (
                self._last_turn_index is not None
                and (
                    state.turn_index <= self._last_turn_index
                    or root_player != self._root_player
                )
            )
        ):
            self._tree.clear()
        self._last_turn_index = state.turn_index
        self._root_player = root_player
        stats.retained_tree_nodes_start = len(self._tree)
        stats.root_reused = root_key in self._tree
        deadline = (
            started + self._config.time_limit_seconds
            if self._config.time_limit_seconds is not None
            else None
        )

        while stats.simulations < self._config.simulations:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            pending: list[_PendingSimulation] = []
            pending_expansion_keys: set[tuple[Any, ...]] = set()
            batch_limit = min(
                self._config.inference_batch_size,
                self._config.simulations - stats.simulations,
            )
            for batch_index in range(batch_limit):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                simulation_index = stats.simulations + batch_index
                determinization_started = time.perf_counter()
                determinized = determinize_for_observer(
                    state,
                    observer_id=root_player,
                    rng=random.Random(
                        self._config.seed
                        + state.turn_index * 1_000_003
                        + simulation_index
                    ),
                )
                stats.determinization_seconds += (
                    time.perf_counter() - determinization_started
                )
                traversal_started = time.perf_counter()
                selected = self._select_leaf(
                    env=env,
                    state=determinized,
                    stats=stats,
                )
                stats.traversal_seconds += time.perf_counter() - traversal_started
                if (
                    selected.expansion_key is not None
                    and selected.expansion_key in pending_expansion_keys
                ):
                    self._release_virtual_visits(selected.path)
                    break
                pending.append(selected)
                if selected.expansion_key is not None:
                    pending_expansion_keys.add(selected.expansion_key)
            if not pending:
                break
            inference_started = time.perf_counter()
            values = self._evaluate_pending(env, pending, stats)
            stats.inference_seconds += time.perf_counter() - inference_started
            for selected, leaf_value in zip(pending, values, strict=True):
                root_value = self._backup(selected.path, leaf_value)
                if selected.path:
                    root_action_index = selected.path[0].action_index
                    stats.root_action_values_by_determinization.setdefault(
                        root_action_index,
                        [],
                    ).append(root_value)
            stats.simulations += len(pending)

        root_node = self._tree.get(root_key)
        if root_node is None:
            return actions[0]
        action_by_index = {
            self._codec.encode(state, action): action
            for action in actions
        }
        available_edges = {
            action_index: edge
            for action_index, edge in root_node.edges.items()
            if action_index in action_by_index
        }
        best_index, _best_edge = max(
            available_edges.items(),
            key=lambda item: (item[1].visits, item[1].q, item[1].prior),
        )
        stats.selected_action_index = best_index
        stats.root_edge_stats = {
            action_index: {
                "visits": edge.visits,
                "q": edge.q,
                "prior": edge.prior,
            }
            for action_index, edge in available_edges.items()
        }
        root_prior_probabilities = [edge.prior for edge in available_edges.values()]
        total_root_visits = sum(edge.visits for edge in available_edges.values())
        root_visit_probabilities = [
            edge.visits / total_root_visits if total_root_visits else 0.0
            for edge in available_edges.values()
        ]
        stats.root_prior_entropy = _distribution_entropy(root_prior_probabilities)
        stats.root_prior_normalized_entropy = _normalized_entropy(
            stats.root_prior_entropy,
            len(root_prior_probabilities),
        )
        stats.root_prior_top1_probability = max(root_prior_probabilities, default=0.0)
        stats.root_prior_effective_actions = math.exp(stats.root_prior_entropy)
        stats.root_visit_entropy = _distribution_entropy(root_visit_probabilities)
        stats.root_visit_normalized_entropy = _normalized_entropy(
            stats.root_visit_entropy,
            len(root_visit_probabilities),
        )
        stats.root_visit_effective_actions = math.exp(stats.root_visit_entropy)
        stats.root_visited_actions = sum(
            edge.visits > 0 for edge in available_edges.values()
        )
        self._last_action_metadata = {
            "root_visit_distribution": [
                {
                    "action_index": action_index,
                    "visits": edge.visits,
                    "probability": (
                        edge.visits / total_root_visits
                        if total_root_visits
                        else 0.0
                    ),
                }
                for action_index, edge in sorted(available_edges.items())
            ],
            "root_prior_entropy": stats.root_prior_entropy,
            "root_prior_normalized_entropy": stats.root_prior_normalized_entropy,
            "root_prior_top1_probability": stats.root_prior_top1_probability,
            "root_prior_effective_actions": stats.root_prior_effective_actions,
            "root_visit_entropy": stats.root_visit_entropy,
            "root_visit_normalized_entropy": stats.root_visit_normalized_entropy,
            "root_visit_effective_actions": stats.root_visit_effective_actions,
            "root_visited_actions": stats.root_visited_actions,
        }
        stats.elapsed_seconds = time.perf_counter() - started
        self._trim_tree(root_key)
        stats.retained_tree_nodes_end = len(self._tree)
        self._last_search_stats = stats
        self._search_history.append(stats)
        return action_by_index[best_index]

    def _select_leaf(
        self,
        env: SplendorEnv,
        state: SplendorState,
        stats: PuctSearchStats,
    ) -> _PendingSimulation:
        path: list[_PathStep] = []
        depth = 0
        while True:
            stats.max_depth_reached = max(stats.max_depth_reached, depth)
            if state.terminal:
                terminal_value = (
                    0.0
                    if state.winner is None
                    else (1.0 if state.winner == state.current_player else -1.0)
                )
                return _PendingSimulation(
                    state=state,
                    path=path,
                    depth=depth,
                    terminal_value=terminal_value,
                )
            if depth >= self._config.max_depth:
                return _PendingSimulation(state=state, path=path, depth=depth)

            legal_actions = env.legal_actions(state)
            key = information_state_key(
                state,
                state.current_player,
                legal_actions,
                self._codec,
            )
            node = self._tree.get(key)
            if node is None:
                return _PendingSimulation(
                    state=state,
                    path=path,
                    depth=depth,
                    expansion_key=key,
                    legal_actions=legal_actions,
                )
            stats.nodes_reused += 1
            action_by_index = {
                self._codec.encode(state, action): action
                for action in legal_actions
            }
            parent_q = (
                sum(edge.value_sum for edge in node.edges.values())
                / sum(edge.visits for edge in node.edges.values())
                if any(edge.visits for edge in node.edges.values())
                else 0.0
            )
            sqrt_visits = math.sqrt(max(node.visits, 1))

            def selection_score(item: tuple[int, _Edge]) -> float:
                action_index, edge = item
                if action_index not in action_by_index:
                    return -math.inf
                q = (
                    edge.q
                    if edge.visits
                    else parent_q - self._config.fpu_reduction
                )
                exploration = (
                    self._config.c_puct
                    * edge.prior
                    * sqrt_visits
                    / (1 + edge.visits)
                )
                return q + exploration

            action_index, edge = max(node.edges.items(), key=selection_score)
            next_state = env.step(state, action_by_index[action_index])
            virtual_value = (
                edge.q
                if edge.visits
                else parent_q - self._config.fpu_reduction
            )
            edge.visits += 1
            edge.value_sum += virtual_value
            node.visits += 1
            path.append(
                _PathStep(
                    node=node,
                    edge=edge,
                    action_index=action_index,
                    parent_player=state.current_player,
                    child_player=next_state.current_player,
                    virtual_value=virtual_value,
                )
            )
            state = next_state
            depth += 1

    def _evaluate_pending(
        self,
        env: SplendorEnv,
        pending: list[_PendingSimulation],
        stats: PuctSearchStats,
    ) -> list[float]:
        expansions = [
            selected
            for selected in pending
            if selected.expansion_key is not None
        ]
        if expansions:
            scored_batches = self._scorer.score_action_batches(
                [
                    (selected.state, selected.legal_actions or [])
                    for selected in expansions
                ]
            )
            stats.prior_batch_calls += 1
            stats.neural_calls += len(expansions)
            for selected, scores in zip(
                expansions,
                scored_batches,
                strict=True,
            ):
                legal_actions = selected.legal_actions or []
                maximum = max(scores.values())
                weights = [
                    math.exp(
                        (scores[action] - maximum)
                        / self._config.prior_temperature
                    )
                    for action in legal_actions
                ]
                total = sum(weights)
                priors = [weight / total for weight in weights]
                if not selected.path:
                    priors = _mix_root_priors(
                        priors,
                        self._config.root_prior_mix,
                    )
                self._tree[selected.expansion_key] = _Node(
                    player_id=selected.state.current_player,
                    edges={
                        self._codec.encode(selected.state, action): _Edge(
                            prior=prior
                        )
                        for action, prior in zip(
                            legal_actions,
                            priors,
                            strict=True,
                        )
                    },
                )
                stats.expansions += 1

        values: list[float | None] = [
            selected.terminal_value for selected in pending
        ]
        nonterminal_indices = [
            index for index, value in enumerate(values) if value is None
        ]
        value_indices = [
            index
            for index in nonterminal_indices
            if self._value_evaluator is not None
            and (
                self._config.value_min_turn_index is None
                or pending[index].state.turn_index
                >= self._config.value_min_turn_index
            )
            and (
                self._config.value_max_turn_index is None
                or pending[index].state.turn_index
                <= self._config.value_max_turn_index
            )
        ]
        rollout_indices = [
            index
            for index in nonterminal_indices
            if index not in value_indices
            and self._config.rollout_leaf_count > 0
            and self._config.n_step_leaf_depth <= 0
        ]
        n_step_indices = [
            index
            for index in nonterminal_indices
            if index not in value_indices
            and self._config.n_step_leaf_depth > 0
            and (
                self._config.n_step_leaf_min_turn_index is None
                or pending[index].state.turn_index
                >= self._config.n_step_leaf_min_turn_index
            )
            and (
                self._config.n_step_leaf_max_turn_index is None
                or pending[index].state.turn_index
                <= self._config.n_step_leaf_max_turn_index
            )
            and (
                self._config.n_step_leaf_min_max_score is None
                or max(player.score for player in pending[index].state.players)
                >= self._config.n_step_leaf_min_max_score
            )
        ]
        heuristic_indices = [
            index for index in nonterminal_indices if index not in value_indices
            and index not in rollout_indices
            and index not in n_step_indices
        ]
        for index in heuristic_indices:
            selected = pending[index]
            values[index] = legal_heuristic_value(
                selected.state,
                player_id=selected.state.current_player,
                scale=self._config.heuristic_scale,
            )
        for index in n_step_indices:
            selected = pending[index]
            leaf_actions = env.legal_actions(selected.state)
            leaf_key = information_state_key(
                selected.state,
                selected.state.current_player,
                leaf_actions,
                self._codec,
            )
            cached_value = self._n_step_leaf_cache.get(leaf_key)
            if cached_value is not None:
                values[index] = cached_value
                stats.n_step_leaf_cache_hits += 1
                continue
            n_step_value = self._n_step_leaf_value(
                env=env,
                state=selected.state,
                player_id=selected.state.current_player,
                legal_actions=leaf_actions,
            )
            if (
                self._config.n_step_leaf_weight < 1.0
                or self._config.n_step_leaf_delta_clip is not None
            ):
                heuristic_value = legal_heuristic_value(
                    selected.state,
                    player_id=selected.state.current_player,
                    scale=self._config.heuristic_scale,
                )
                value, raw_delta, applied_delta = _combine_n_step_leaf_value(
                    heuristic_value=heuristic_value,
                    n_step_value=n_step_value,
                    weight=self._config.n_step_leaf_weight,
                    delta_clip=self._config.n_step_leaf_delta_clip,
                )
                stats.n_step_leaf_absolute_delta_sum += abs(raw_delta)
                stats.n_step_leaf_applied_absolute_delta_sum += abs(
                    applied_delta
                )
                if abs(applied_delta) < abs(raw_delta):
                    stats.n_step_leaf_clipped_evaluations += 1
            else:
                value = n_step_value
            self._n_step_leaf_cache[leaf_key] = value
            values[index] = value
            stats.n_step_leaf_evaluations += 1
        for index in rollout_indices:
            selected = pending[index]
            values[index] = self._rollout_leaf_value(
                env=env,
                state=selected.state,
                player_id=selected.state.current_player,
                seed_offset=stats.simulations + index,
            )
            stats.rollout_leaf_evaluations += self._config.rollout_leaf_count
        if value_indices:
            assert self._value_evaluator is not None
            evaluated = self._value_evaluator.evaluate_states(
                [
                    (
                        pending[index].state,
                        pending[index].state.current_player,
                    )
                    for index in value_indices
                ]
            )
            stats.value_batch_calls += 1
            stats.value_calls += len(value_indices)
            for index, value in zip(
                value_indices,
                evaluated,
                strict=True,
            ):
                values[index] = max(-1.0, min(1.0, value))
        return [float(value) for value in values]

    def _rollout_leaf_value(
        self,
        env: SplendorEnv,
        state: SplendorState,
        player_id: int,
        seed_offset: int,
    ) -> float:
        total = 0.0
        for rollout_index in range(self._config.rollout_leaf_count):
            rng = random.Random(
                self._config.seed
                + state.turn_index * 1_000_003
                + seed_offset * 97_409
                + rollout_index
            )
            rollout_state = state
            rollout_end = state.turn_index + self._config.rollout_leaf_max_depth
            while not rollout_state.terminal and rollout_state.turn_index < rollout_end:
                legal_actions = env.legal_actions(rollout_state)
                if not legal_actions:
                    break
                action = _sample_heuristic_rollout_action(
                    env=env,
                    state=rollout_state,
                    legal_actions=legal_actions,
                    rng=rng,
                    policy=self._config.rollout_leaf_policy,
                    top_k=self._config.rollout_leaf_top_k,
                    temperature=self._config.rollout_leaf_temperature,
                )
                rollout_state = env.step(rollout_state, action)
            total += legal_heuristic_value(
                rollout_state,
                player_id=player_id,
                scale=self._config.heuristic_scale,
            )
        return total / self._config.rollout_leaf_count

    def _n_step_leaf_value(
        self,
        env: SplendorEnv,
        state: SplendorState,
        player_id: int,
        legal_actions: list[Action] | None = None,
    ) -> float:
        frontier: list[tuple[SplendorState, list[Action] | None]] = [
            (state, legal_actions)
        ]
        for _depth in range(self._config.n_step_leaf_depth):
            next_frontier: list[tuple[float, SplendorState]] = []
            for frontier_state, frontier_actions in frontier:
                if frontier_state.terminal:
                    next_frontier.append(
                        (
                            legal_heuristic_value(
                                frontier_state,
                                player_id=player_id,
                                scale=self._config.heuristic_scale,
                            ),
                            frontier_state,
                        )
                    )
                    continue
                actions = (
                    frontier_actions
                    if frontier_actions is not None
                    else env.legal_actions(frontier_state)
                )
                if not actions:
                    continue
                for score, next_state in self._rank_n_step_leaf_candidates(
                    env,
                    frontier_state,
                    actions,
                ):
                    value = (
                        score
                        if frontier_state.current_player == player_id
                        else -score
                    )
                    next_frontier.append((value, next_state))
            if not next_frontier:
                break
            next_frontier.sort(key=lambda item: item[0], reverse=True)
            frontier = [
                (next_state, None)
                for _value, next_state in next_frontier[
                    : self._config.n_step_leaf_width
                ]
            ]
            if all(item.terminal for item, _actions in frontier):
                break
        if not frontier:
            return legal_heuristic_value(
                state,
                player_id=player_id,
                scale=self._config.heuristic_scale,
            )
        return sum(
            legal_heuristic_value(
                frontier_state,
                player_id=player_id,
                scale=self._config.heuristic_scale,
            )
            for frontier_state, _actions in frontier
        ) / len(frontier)

    def _rank_n_step_leaf_candidates(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action],
    ) -> list[tuple[float, SplendorState]]:
        candidates = _prune_n_step_leaf_actions(
            legal_actions,
            limit=self._config.n_step_leaf_candidate_limit,
        )
        actor = state.current_player
        before_affordable = self._leaf_heuristic.affordable_card_count(
            state,
            actor,
        )
        before_best_missing = self._leaf_heuristic.best_purchase_missing(
            state,
            actor,
        )
        ranked: list[tuple[float, SplendorState]] = []
        for action in candidates:
            next_state = env.step(state, action)
            score = self._leaf_heuristic.score_transition(
                before=state,
                after=next_state,
                action=action,
                player_id=actor,
                before_affordable=before_affordable,
                before_best_missing=before_best_missing,
            )
            ranked.append((score, next_state))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[: self._config.n_step_leaf_width]

    @staticmethod
    def _backup(path: list[_PathStep], leaf_value: float) -> float:
        value = leaf_value
        for step in reversed(path):
            if step.child_player != step.parent_player:
                value = -value
            step.edge.value_sum += value - step.virtual_value
        return value

    @staticmethod
    def _release_virtual_visits(path: list[_PathStep]) -> None:
        for step in path:
            step.edge.visits -= 1
            step.edge.value_sum -= step.virtual_value
            step.node.visits -= 1

    def _trim_tree(self, root_key: tuple[Any, ...]) -> None:
        excess = len(self._tree) - self._config.max_tree_nodes
        if excess <= 0:
            return
        for key in list(self._tree):
            if excess <= 0:
                break
            if key == root_key:
                continue
            del self._tree[key]
            excess -= 1


def information_state_key(
    state: SplendorState,
    player_id: int,
    legal_actions: list[Action],
    codec: ActionCodec,
) -> tuple[Any, ...]:
    observation = encode_public_observation_tensor(state, player_id)
    legal_indices = tuple(codec.legal_action_indices(state, legal_actions))
    return (player_id, observation.vector, legal_indices)


def legal_heuristic_value(
    state: SplendorState,
    player_id: int,
    scale: float,
) -> float:
    if state.terminal:
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == player_id else -1.0
    player = state.players[player_id]
    opponent = state.players[1 - player_id]
    score = 0.0
    score += 1_000.0 * (player.score - opponent.score)
    score += 120.0 * (
        sum(player.bonuses.values()) - sum(opponent.bonuses.values())
    )
    score += 42.0 * (
        len(player.purchased_cards) - len(opponent.purchased_cards)
    )
    score += 60.0 * (len(player.nobles) - len(opponent.nobles))
    score += 2.5 * (player.token_count - opponent.token_count)
    score -= 25.0 * max(len(player.reserved_cards) - 2, 0)
    score += 20.0 * max(len(opponent.reserved_cards) - 2, 0)
    return math.tanh(score / scale)


def _combine_n_step_leaf_value(
    heuristic_value: float,
    n_step_value: float,
    weight: float,
    delta_clip: float | None,
) -> tuple[float, float, float]:
    raw_delta = n_step_value - heuristic_value
    applied_delta = raw_delta
    if delta_clip is not None:
        applied_delta = max(-delta_clip, min(delta_clip, raw_delta))
    return (
        heuristic_value + weight * applied_delta,
        raw_delta,
        applied_delta,
    )


def _distribution_entropy(probabilities: list[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def _normalized_entropy(entropy: float, action_count: int) -> float:
    if action_count <= 1:
        return 0.0
    return entropy / math.log(action_count)


def _sample_heuristic_rollout_action(
    env: SplendorEnv,
    state: SplendorState,
    legal_actions: list[Action],
    rng: random.Random,
    policy: str,
    top_k: int,
    temperature: float,
) -> Action:
    if policy == "random":
        return legal_actions[rng.randrange(len(legal_actions))]
    scorer = GreedyHeuristicBot(seed=0)
    scored = [
        (
            scorer.score_action(
                env=env,
                state=state,
                action=action,
                player_id=state.current_player,
            ),
            action,
        )
        for action in legal_actions
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    candidates = scored[: min(top_k, len(scored))]
    if len(candidates) == 1:
        return candidates[0][1]
    maximum = max(score for score, _action in candidates)
    weights = [
        math.exp((score - maximum) / temperature)
        for score, _action in candidates
    ]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for weight, (_score, action) in zip(weights, candidates, strict=True):
        cumulative += weight
        if cumulative >= threshold:
            return action
    return candidates[-1][1]


def _prune_n_step_leaf_actions(
    legal_actions: list[Action],
    limit: int,
) -> list[Action]:
    if len(legal_actions) <= limit:
        return list(legal_actions)
    buckets = {
        ActionType.BUY_VISIBLE: [],
        ActionType.BUY_RESERVED: [],
        ActionType.RESERVE_VISIBLE: [],
        ActionType.TAKE_TOKENS: [],
        ActionType.RESERVE_DECK: [],
        ActionType.PASS: [],
    }
    for action in legal_actions:
        buckets[action.action_type].append(action)

    def action_key(action: Action) -> tuple[int, int, int, str]:
        return (
            len(action.return_tokens),
            -len(action.take_tokens),
            action.market_index if action.market_index is not None else 99,
            action.tier if action.tier is not None else 99,
        )

    selected: list[Action] = []
    for action_type, per_type_limit in (
        (ActionType.BUY_VISIBLE, limit),
        (ActionType.BUY_RESERVED, limit),
        (ActionType.RESERVE_VISIBLE, max(1, limit // 3)),
        (ActionType.TAKE_TOKENS, max(1, limit // 2)),
        (ActionType.RESERVE_DECK, 1),
        (ActionType.PASS, 1),
    ):
        for action in sorted(buckets[action_type], key=action_key)[
            :per_type_limit
        ]:
            if len(selected) >= limit:
                return selected
            selected.append(action)
    return selected


def _mix_root_priors(priors: list[float], mix: float) -> list[float]:
    if not priors:
        return []
    uniform = 1.0 / len(priors)
    return [mix * prior + (1.0 - mix) * uniform for prior in priors]


def _validate_config(config: PuctConfig) -> None:
    if (
        config.simulations <= 0
        or config.max_depth <= 0
        or config.inference_batch_size <= 0
        or config.max_tree_nodes <= 0
        or config.n_step_leaf_width <= 0
        or config.n_step_leaf_candidate_limit <= 0
        or config.rollout_leaf_max_depth <= 0
        or config.rollout_leaf_top_k <= 0
    ):
        raise ValueError(
            "simulations, max_depth, inference_batch_size, max_tree_nodes, "
            "n_step_leaf_width, n_step_leaf_candidate_limit, "
            "rollout_leaf_max_depth, and rollout_leaf_top_k "
            "must be positive."
        )
    if config.n_step_leaf_depth < 0:
        raise ValueError("n_step_leaf_depth must be non-negative.")
    if not 0.0 <= config.n_step_leaf_weight <= 1.0:
        raise ValueError("n_step_leaf_weight must be in [0, 1].")
    if (
        config.n_step_leaf_delta_clip is not None
        and config.n_step_leaf_delta_clip <= 0.0
    ):
        raise ValueError("n_step_leaf_delta_clip must be positive when set.")
    if (
        config.n_step_leaf_min_turn_index is not None
        and config.n_step_leaf_max_turn_index is not None
        and config.n_step_leaf_min_turn_index > config.n_step_leaf_max_turn_index
    ):
        raise ValueError(
            "n_step_leaf_min_turn_index cannot exceed "
            "n_step_leaf_max_turn_index."
        )
    if (
        config.n_step_leaf_min_max_score is not None
        and config.n_step_leaf_min_max_score < 0
    ):
        raise ValueError("n_step_leaf_min_max_score must be non-negative.")
    if config.rollout_leaf_count < 0:
        raise ValueError("rollout_leaf_count must be non-negative.")
    if config.rollout_leaf_policy not in {"random", "heuristic"}:
        raise ValueError("rollout_leaf_policy must be 'random' or 'heuristic'.")
    if config.c_puct <= 0.0 or config.prior_temperature <= 0.0:
        raise ValueError("c_puct and prior_temperature must be positive.")
    if not 0.0 <= config.root_prior_mix <= 1.0:
        raise ValueError("root_prior_mix must be in [0, 1].")
    if config.rollout_leaf_temperature <= 0.0:
        raise ValueError("rollout_leaf_temperature must be positive.")
    if config.fpu_reduction < 0.0:
        raise ValueError("fpu_reduction must be non-negative.")
    if config.heuristic_scale <= 0.0:
        raise ValueError("heuristic_scale must be positive.")
    if (
        config.time_limit_seconds is not None
        and config.time_limit_seconds <= 0.0
    ):
        raise ValueError("time_limit_seconds must be positive when set.")
    if (
        config.value_min_turn_index is not None
        and config.value_max_turn_index is not None
        and config.value_min_turn_index > config.value_max_turn_index
    ):
        raise ValueError(
            "value_min_turn_index cannot exceed value_max_turn_index."
        )
