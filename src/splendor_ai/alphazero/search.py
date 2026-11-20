"""Clean shared-tree policy/value search for programme 090."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
import time
from typing import Any, Literal

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.diagnostics import state_signature
from splendor_ai.encoding import (
    ActionCodec,
    actions_by_canonical_successor_key,
    canonical_public_belief_key,
    canonical_successor_action_key,
    canonical_successor_action_keys,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.termination import (
    adjudicate_scores,
    forced_pass_deadlock_successor,
    is_forced_pass_turn,
    outcome_value,
)

from .network import AlphaZeroInference


SearchMode = Literal["debug", "real"]


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 64
    c_puct: float = 1.5
    max_depth: int = 96
    mode: SearchMode = "real"
    root_noise_fraction: float = 0.25
    dirichlet_total_concentration: float = 10.0
    temperature_turns: int = 20
    temperature: float = 1.0
    seed: int = 0

    def validate(self) -> None:
        if self.simulations <= 0:
            raise ValueError("simulations must be positive.")
        if self.c_puct < 0.0 or self.max_depth <= 0:
            raise ValueError("Invalid PUCT or depth setting.")
        if self.mode not in ("debug", "real"):
            raise ValueError(f"Unknown search mode: {self.mode}")
        if not 0.0 <= self.root_noise_fraction <= 1.0:
            raise ValueError("root_noise_fraction must be in [0,1].")
        if self.dirichlet_total_concentration <= 0.0:
            raise ValueError("dirichlet_total_concentration must be positive.")
        if self.temperature < 0.0:
            raise ValueError("temperature must be nonnegative.")


@dataclass(slots=True)
class _Edge:
    prior: float
    network_prior: float | None = None
    visits: int = 0
    value_sum: float = 0.0

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(slots=True)
class _Node:
    player_id: int
    edges: dict[tuple[object, ...], _Edge]
    visits: int = 0
    network_value: float = 0.0


@dataclass(frozen=True, slots=True)
class _PathStep:
    node: _Node
    edge: _Edge
    parent_player: int
    child_player: int


@dataclass(frozen=True, slots=True)
class RootEdgeResult:
    action_key: tuple[object, ...]
    action: Action
    visits: int
    probability: float
    network_prior: float
    search_prior: float
    q: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    action: Action
    root_edges: tuple[RootEdgeResult, ...]
    raw_network_value: float
    search_root_value: float
    simulations: int
    expansions: int
    inference_calls: int
    max_depth_reached: int
    elapsed_seconds: float
    search_mode: SearchMode
    selected_action_row: int

    @property
    def visit_probabilities(self) -> tuple[float, ...]:
        return tuple(edge.probability for edge in self.root_edges)

    @property
    def raw_policy_top_row(self) -> int:
        return max(
            range(len(self.root_edges)),
            key=lambda row: (
                self.root_edges[row].network_prior,
                repr(self.root_edges[row].action_key),
            ),
        )

    @property
    def visit_argmax_row(self) -> int:
        return max(
            range(len(self.root_edges)),
            key=lambda row: (
                self.root_edges[row].visits,
                self.root_edges[row].q,
                self.root_edges[row].search_prior,
                repr(self.root_edges[row].action_key),
            ),
        )


class AlphaZeroSearch:
    """A fresh tree per move with joint-network priors and leaf values."""

    def __init__(
        self,
        inference: AlphaZeroInference,
        config: SearchConfig | None = None,
    ) -> None:
        self.inference = inference
        self.config = config or SearchConfig()
        self.config.validate()
        self._tree: dict[tuple[Any, ...], _Node] = {}
        self._expansions = 0
        self._inference_calls = 0
        self._max_depth_reached = 0

    def run(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
        *,
        self_play: bool,
        seed: int | None = None,
    ) -> SearchResult:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            raise ValueError("Cannot search a state without legal actions.")
        self._tree = {}
        self._expansions = 0
        self._inference_calls = 0
        self._max_depth_reached = 0
        started = time.perf_counter()
        root_player = state.current_player
        search_seed = self.config.seed if seed is None else int(seed)
        rng = random.Random(search_seed)

        root_sample = self._sample_state(state, root_player, rng, simulation=-1)
        root_actions = list(
            actions_by_canonical_successor_key(root_sample, env.legal_actions(root_sample)).values()
        )
        root_key = self._node_key(root_sample, root_actions)
        root_node = self._expand(root_sample, root_actions, root_key)
        raw_network_value = root_node.network_value
        if self_play and self.config.root_noise_fraction > 0.0:
            self._apply_root_noise(root_node, rng)

        for simulation in range(self.config.simulations):
            sampled = self._sample_state(state, root_player, rng, simulation=simulation)
            leaf_value, path = self._select_and_evaluate(
                env,
                sampled,
                root_player=root_player,
                search_seed=search_seed,
                simulation=simulation,
            )
            self._backup(path, leaf_value)

        original_by_key = actions_by_canonical_successor_key(state, actions)
        available = [
            (key, edge, original_by_key[key])
            for key, edge in root_node.edges.items()
            if key in original_by_key
        ]
        if not available:
            raise RuntimeError("Root tree has no edge that maps to the legal action set.")
        total_visits = sum(edge.visits for _key, edge, _action in available)
        if total_visits != self.config.simulations:
            raise RuntimeError(
                f"Root visit accounting mismatch: {total_visits} != {self.config.simulations}"
            )
        probabilities = [edge.visits / total_visits for _key, edge, _action in available]
        selected_row = self._select_root_row(
            probabilities,
            available,
            state.turn_index,
            self_play,
            rng,
        )
        root_results = tuple(
            RootEdgeResult(
                action_key=key,
                action=action,
                visits=edge.visits,
                probability=probability,
                network_prior=(
                    edge.prior if edge.network_prior is None else edge.network_prior
                ),
                search_prior=edge.prior,
                q=edge.q,
            )
            for (key, edge, action), probability in zip(
                available,
                probabilities,
                strict=True,
            )
        )
        root_value = sum(edge.q * edge.visits for _key, edge, _action in available) / total_visits
        return SearchResult(
            action=available[selected_row][2],
            root_edges=root_results,
            raw_network_value=raw_network_value,
            search_root_value=root_value,
            simulations=self.config.simulations,
            expansions=self._expansions,
            inference_calls=self._inference_calls,
            max_depth_reached=self._max_depth_reached,
            elapsed_seconds=time.perf_counter() - started,
            search_mode=self.config.mode,
            selected_action_row=selected_row,
        )

    def _select_and_evaluate(
        self,
        env: SplendorEnv,
        state: SplendorState,
        *,
        root_player: int,
        search_seed: int,
        simulation: int,
    ) -> tuple[float, list[_PathStep]]:
        path: list[_PathStep] = []
        depth = 0
        while True:
            self._max_depth_reached = max(self._max_depth_reached, depth)
            if state.terminal:
                return _terminal_value(state, state.current_player), path
            legal_actions = env.legal_actions(state)
            if not legal_actions:
                return 0.0, path
            representatives = list(
                actions_by_canonical_successor_key(state, legal_actions).values()
            )
            key = self._node_key(state, representatives)
            node = self._tree.get(key)
            if node is None:
                node = self._expand(state, representatives, key)
                return node.network_value, path
            if depth >= self.config.max_depth:
                return node.network_value, path
            action_by_key = actions_by_canonical_successor_key(state, legal_actions)
            sqrt_visits = math.sqrt(max(node.visits, 1))

            def score(item: tuple[tuple[object, ...], _Edge]) -> tuple[float, float, str]:
                action_key, edge = item
                if action_key not in action_by_key:
                    return (-math.inf, -math.inf, "")
                exploration = (
                    self.config.c_puct
                    * edge.prior
                    * sqrt_visits
                    / (1 + edge.visits)
                )
                return (edge.q + exploration, edge.prior, repr(action_key))

            action_key, edge = max(node.edges.items(), key=score)
            action = action_by_key.get(action_key)
            if action is None:
                raise RuntimeError("Tree edge cannot be rebound in sampled state.")
            transition_state = self._prepare_transition(
                state,
                action,
                action_key,
                depth=depth,
                root_player=root_player,
                search_seed=search_seed,
                simulation=simulation,
            )
            child = env.step(transition_state, action)
            path.append(
                _PathStep(
                    node=node,
                    edge=edge,
                    parent_player=state.current_player,
                    child_player=child.current_player,
                )
            )
            if is_forced_pass_turn(legal_actions):
                adjudication_state = forced_pass_deadlock_successor(env, child)
                if adjudication_state is not None:
                    return (
                        outcome_value(
                            adjudicate_scores(adjudication_state),
                            child.current_player,
                        ),
                        path,
                    )
            state = child
            depth += 1

    def _prepare_transition(
        self,
        state: SplendorState,
        action: Action,
        action_key: tuple[object, ...],
        *,
        depth: int,
        root_player: int,
        search_seed: int,
        simulation: int,
    ) -> SplendorState:
        """Hook for information-legal transition sampling experiments.

        The production search returns the sampled state unchanged. Experimental
        subclasses may reorder still-hidden cards before a transition, but must
        preserve the observer's information set and all public state fields.
        """

        del action, action_key, depth, root_player, search_seed, simulation
        return state

    def _expand(
        self,
        state: SplendorState,
        actions: list[Action],
        key: tuple[Any, ...],
    ) -> _Node:
        prediction = self.inference.predict_one(state, actions)
        priors = _softmax(prediction.logits)
        edges = {
            canonical_successor_action_key(state, action): _Edge(
                prior=prior,
                network_prior=prior,
            )
            for action, prior in zip(actions, priors, strict=True)
        }
        if len(edges) != len(actions):
            raise RuntimeError("Expansion actions were not canonical-successor unique.")
        node = _Node(
            player_id=state.current_player,
            edges=edges,
            network_value=max(-1.0, min(1.0, prediction.value)),
        )
        self._tree[key] = node
        self._expansions += 1
        self._inference_calls += 1
        return node

    def _node_key(
        self,
        state: SplendorState,
        actions: list[Action],
    ) -> tuple[Any, ...]:
        action_keys = canonical_successor_action_keys(
            state,
            actions,
            state.current_player,
        )
        if self.config.mode == "debug":
            return ("alphazero-full-state-v1", state_signature(state), action_keys)
        return (
            "alphazero-information-state-v1",
            canonical_public_belief_key(state, state.current_player),
            action_keys,
        )

    def _sample_state(
        self,
        state: SplendorState,
        root_player: int,
        rng: random.Random,
        *,
        simulation: int,
    ) -> SplendorState:
        del simulation
        if self.config.mode == "debug":
            return state.copy_shallow()
        return determinize_for_observer(state, observer_id=root_player, rng=rng)

    def _apply_root_noise(self, node: _Node, rng: random.Random) -> None:
        count = len(node.edges)
        alpha = self.config.dirichlet_total_concentration / count
        samples = [rng.gammavariate(alpha, 1.0) for _ in range(count)]
        total = sum(samples)
        noise = [sample / total for sample in samples]
        fraction = self.config.root_noise_fraction
        for edge, value in zip(node.edges.values(), noise, strict=True):
            edge.prior = (1.0 - fraction) * edge.prior + fraction * value

    def _select_root_row(
        self,
        probabilities: list[float],
        available: list[tuple[tuple[object, ...], _Edge, Action]],
        turn_index: int,
        self_play: bool,
        rng: random.Random,
    ) -> int:
        if (
            not self_play
            or turn_index >= self.config.temperature_turns
            or self.config.temperature == 0.0
        ):
            return max(
                range(len(available)),
                key=lambda row: (
                    available[row][1].visits,
                    available[row][1].q,
                    available[row][1].prior,
                    repr(available[row][0]),
                ),
            )
        exponent = 1.0 / self.config.temperature
        weights = [probability**exponent for probability in probabilities]
        return _weighted_index(weights, rng)

    @staticmethod
    def _backup(path: list[_PathStep], leaf_value: float) -> None:
        value = leaf_value
        for step in reversed(path):
            if step.child_player != step.parent_player:
                value = -value
            step.edge.visits += 1
            step.edge.value_sum += value
            step.node.visits += 1


class AlphaZeroSearchBot:
    """Bot adapter around a configured AlphaZero search."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        device: str = "cpu",
        config: SearchConfig | None = None,
        self_play: bool = False,
        inference: AlphaZeroInference | None = None,
    ) -> None:
        if inference is None and checkpoint_path is None:
            raise ValueError("checkpoint_path is required when inference is not supplied.")
        self.inference = (
            inference
            if inference is not None
            else AlphaZeroInference(str(checkpoint_path), device=device)
        )
        self.checkpoint_path = str(self.inference.checkpoint.path)
        self.search = AlphaZeroSearch(self.inference, config=config)
        self.self_play = self_play
        self.last_result: SearchResult | None = None
        self.last_action_metadata: dict[str, Any] = {}

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        result = self.search.run(
            env,
            state,
            actions,
            self_play=self.self_play,
            seed=self.search.config.seed + state.turn_index * 1_000_003,
        )
        self.last_result = result
        codec = ActionCodec()
        self.last_action_metadata = {
            "root_visit_distribution": [
                {
                    "action_index": codec.encode(state, edge.action),
                    "visits": edge.visits,
                    "probability": edge.probability,
                }
                for edge in result.root_edges
            ],
            "raw_network_value": result.raw_network_value,
            "search_root_value": result.search_root_value,
            "search_config": asdict(self.search.config),
        }
        return result.action


class AlphaZeroRawBot:
    """Greedy raw policy from the same joint checkpoint, without search."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        device: str = "cpu",
        inference: AlphaZeroInference | None = None,
    ) -> None:
        if inference is None and checkpoint_path is None:
            raise ValueError("checkpoint_path is required when inference is not supplied.")
        self.inference = (
            inference
            if inference is not None
            else AlphaZeroInference(str(checkpoint_path), device=device)
        )
        self.checkpoint_path = str(self.inference.checkpoint.path)

    def choose_action(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
    ) -> Action | None:
        actions = legal_actions if legal_actions is not None else env.legal_actions(state)
        if not actions:
            return None
        representatives = list(actions_by_canonical_successor_key(state, actions).values())
        prediction = self.inference.predict_one(state, representatives)
        row = max(range(len(representatives)), key=lambda index: prediction.logits[index])
        return representatives[row]


def _softmax(values: tuple[float, ...]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def _weighted_index(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return len(weights) - 1


def _terminal_value(state: SplendorState, player_id: int) -> float:
    if not state.terminal or state.winner is None:
        return 0.0
    return 1.0 if state.winner == player_id else -1.0
