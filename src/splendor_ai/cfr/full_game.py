"""Full-game Deep CFR representation and terminal-traversal feasibility tools."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import random
import time
from typing import Any, Generic, Iterable, TypeVar

import torch

from splendor_ai.encoding.entity_encoding import (
    ACTION_NUMERIC_SIZE,
    ACTION_TYPE_TO_ID,
    ENTITY_HIDDEN_DECK_CARD,
    ENTITY_PRIVILEGED_OPPONENT_BLIND_RESERVED,
    EntityActionEncoding,
    EntityStateEncoding,
    encode_canonical_action_entities,
    encode_canonical_state_entities,
)
from splendor_ai.encoding.information_state import PublicActionEvent
from splendor_ai.encoding.public_belief import (
    actions_by_canonical_descriptor,
    canonical_public_belief_key,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.data import build_base_deck_by_tier, build_base_nobles
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState


@dataclass(frozen=True, slots=True)
class PublicHistoryEncoding:
    action_type_ids: tuple[int, ...]
    card_ids: tuple[int, ...]
    noble_ids: tuple[int, ...]
    numeric_features: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class DeepCFRSchemaRow:
    state: EntityStateEncoding
    history: PublicHistoryEncoding
    actions: EntityActionEncoding
    target: tuple[float, ...]
    player: int
    iteration: int
    reach_weight: float
    provenance: str = "schema_only"

    def __post_init__(self) -> None:
        action_count = len(self.actions.action_type_ids)
        if action_count < 1 or len(self.target) != action_count:
            raise ValueError("Deep CFR target must align exactly with legal actions.")
        if self.player not in (0, 1):
            raise ValueError("Deep CFR row player must be 0 or 1.")
        if self.provenance != "schema_only":
            raise ValueError("The 047 memory accepts schema_only rows only.")
        if any(
            entity_type in (
                ENTITY_HIDDEN_DECK_CARD,
                ENTITY_PRIVILEGED_OPPONENT_BLIND_RESERVED,
            )
            for entity_type in self.state.entity_type_ids
        ):
            raise ValueError("Privileged entities are forbidden in deployable CFR rows.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": _encoding_payload(self.state),
            "history": _encoding_payload(self.history),
            "actions": _encoding_payload(self.actions),
            "target": list(self.target),
            "player": self.player,
            "iteration": self.iteration,
            "reach_weight": self.reach_weight,
            "provenance": self.provenance,
        }


T = TypeVar("T")


class ReservoirMemory(Generic[T]):
    """Deterministic Algorithm-R reservoir with bounded resident rows."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("Reservoir capacity must be positive.")
        self.capacity = capacity
        self.seed = seed
        self.seen = 0
        self.rows: list[T] = []
        self._rng = random.Random(seed)

    def add(self, row: T) -> None:
        self.seen += 1
        if len(self.rows) < self.capacity:
            self.rows.append(row)
            return
        index = self._rng.randrange(self.seen)
        if index < self.capacity:
            self.rows[index] = row

    def payload(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "seed": self.seed,
            "seen": self.seen,
            "rows": [
                row.to_payload() if hasattr(row, "to_payload") else row
                for row in self.rows
            ],
        }


def encode_public_history(
    public_history: Iterable[PublicActionEvent],
    observer: int,
) -> PublicHistoryEncoding:
    if observer not in (0, 1):
        raise ValueError("observer must be 0 or 1")
    action_types: list[int] = []
    card_ids: list[int] = []
    noble_ids: list[int] = []
    numeric_rows: list[tuple[float, ...]] = []
    for event in public_history:
        if len(event) != 10 or event[0] != "splendor-public-action-v1":
            raise ValueError(f"Unsupported public action event: {event!r}")
        turn = int(event[1])
        actor = int(event[2])
        action_type = ActionType[str(event[3])]
        target_id = event[4]
        public_tier = event[5]
        noble_id = event[9]
        features = list(_card_features(None if target_id is None else str(target_id)))
        offset = 13
        for counts in (event[6], event[7], event[8]):
            count_map = dict(counts)
            for color_index, color in enumerate(ALL_TOKEN_TYPES):
                normalizer = 5.0 if color == "gold" else 4.0
                features[offset + color_index] = float(count_map[color]) / normalizer
            offset += len(ALL_TOKEN_TYPES)
        features[31] = float(public_tier or 0) / 3.0
        features[32] = 1.0 if actor == observer else -1.0
        features[33] = min(float(turn) / 400.0, 1.0)
        action_types.append(ACTION_TYPE_TO_ID[action_type])
        card_ids.append(_card_ids().get(str(target_id), 0) if target_id is not None else 0)
        noble_ids.append(_noble_ids().get(str(noble_id), 0) if noble_id is not None else 0)
        numeric_rows.append(tuple(features))
    return PublicHistoryEncoding(
        action_type_ids=tuple(action_types),
        card_ids=tuple(card_ids),
        noble_ids=tuple(noble_ids),
        numeric_features=tuple(numeric_rows),
    )


def build_schema_row(
    state: SplendorState,
    public_history: tuple[PublicActionEvent, ...],
    legal_actions,
    *,
    iteration: int = 0,
    reach_weight: float = 1.0,
) -> DeepCFRSchemaRow:
    observer = state.current_player
    canonical_actions = actions_by_canonical_descriptor(state, legal_actions)
    ordered_actions = [canonical_actions[key] for key in sorted(canonical_actions, key=repr)]
    count = len(ordered_actions)
    return DeepCFRSchemaRow(
        state=encode_canonical_state_entities(state, observer),
        history=encode_public_history(public_history, observer),
        actions=encode_canonical_action_entities(state, ordered_actions, observer),
        target=tuple([1.0 / count] * count),
        player=observer,
        iteration=iteration,
        reach_weight=reach_weight,
    )


def collate_deep_cfr_rows(
    rows: list[DeepCFRSchemaRow],
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("rows must not be empty")
    target_device = torch.device(device)
    state_width = max(len(row.state.entity_type_ids) for row in rows)
    history_width = max(max(len(row.history.action_type_ids), 1) for row in rows)
    action_width = max(len(row.actions.action_type_ids) for row in rows)
    return {
        "entity_type_ids": _pad_ids([row.state.entity_type_ids for row in rows], state_width, target_device),
        "entity_card_ids": _pad_ids([row.state.card_ids for row in rows], state_width, target_device),
        "entity_noble_ids": _pad_ids([row.state.noble_ids for row in rows], state_width, target_device),
        "entity_numeric_features": _pad_features(
            [row.state.numeric_features for row in rows], state_width, target_device
        ),
        "entity_mask": _mask(
            [len(row.state.entity_type_ids) for row in rows], state_width, target_device
        ),
        "history_action_type_ids": _pad_ids(
            [row.history.action_type_ids for row in rows], history_width, target_device
        ),
        "history_card_ids": _pad_ids(
            [row.history.card_ids for row in rows], history_width, target_device
        ),
        "history_noble_ids": _pad_ids(
            [row.history.noble_ids for row in rows], history_width, target_device
        ),
        "history_numeric_features": _pad_features(
            [row.history.numeric_features for row in rows], history_width, target_device
        ),
        "history_mask": _mask(
            [len(row.history.action_type_ids) for row in rows], history_width, target_device
        ),
        "action_type_ids": _pad_ids(
            [row.actions.action_type_ids for row in rows], action_width, target_device
        ),
        "action_card_ids": _pad_ids(
            [row.actions.card_ids for row in rows], action_width, target_device
        ),
        "action_noble_ids": _pad_ids(
            [row.actions.noble_ids for row in rows], action_width, target_device
        ),
        "action_numeric_features": _pad_features(
            [row.actions.numeric_features for row in rows], action_width, target_device
        ),
        "action_mask": _mask(
            [len(row.actions.action_type_ids) for row in rows], action_width, target_device
        ),
        "targets": _pad_float_vectors([row.target for row in rows], action_width, target_device),
        "players": torch.tensor([row.player for row in rows], dtype=torch.long, device=target_device),
        "reach_weights": torch.tensor(
            [row.reach_weight for row in rows], dtype=torch.float32, device=target_device
        ),
    }


class TraversalCutoff(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class FullGameTraversalProbe:
    root_seed: int
    traverser: int
    node_budget: int = 25_000

    def run(self) -> dict[str, Any]:
        if self.traverser not in (0, 1):
            raise ValueError("traverser must be 0 or 1")
        self.env = SplendorEnv(seed=self.root_seed)
        self.rng = random.Random((self.root_seed * 10) + self.traverser)
        self.nodes = 0
        self.nodes_by_depth: Counter[int] = Counter()
        self.action_counts: list[int] = []
        self.max_depth = 0
        self.terminal_leaves = 0
        self.forced_deadlock_leaves = 0
        self.repeated_path_states = 0
        self.candidate_advantage_rows = 0
        self.candidate_strategy_rows = 0
        self.illegal_actions = 0
        started = time.perf_counter()
        complete = True
        value: float | None = None
        cutoff_reason: str | None = None
        try:
            value = self._traverse(
                self.env.initial_state(),
                forced_pass_streak=0,
                path_signatures=set(),
            )
        except TraversalCutoff as exc:
            complete = False
            cutoff_reason = exc.reason
        elapsed = time.perf_counter() - started
        return {
            "root_seed": self.root_seed,
            "traverser": self.traverser,
            "node_budget": self.node_budget,
            "complete": complete,
            "cutoff_reason": cutoff_reason,
            "value_for_traverser": value,
            "nodes": self.nodes,
            "nodes_by_depth": {str(key): value for key, value in sorted(self.nodes_by_depth.items())},
            "max_depth": self.max_depth,
            "terminal_leaves": self.terminal_leaves,
            "forced_pass_deadlock_leaves": self.forced_deadlock_leaves,
            "repeated_path_states": self.repeated_path_states,
            "decision_nodes": len(self.action_counts),
            "mean_legal_actions": sum(self.action_counts) / max(len(self.action_counts), 1),
            "max_legal_actions": max(self.action_counts, default=0),
            "candidate_advantage_rows": self.candidate_advantage_rows,
            "candidate_strategy_rows": self.candidate_strategy_rows,
            "eligible_advantage_rows": self.candidate_advantage_rows if complete else 0,
            "eligible_strategy_rows": self.candidate_strategy_rows if complete else 0,
            "illegal_actions": self.illegal_actions,
            "elapsed_seconds": elapsed,
        }

    def _traverse(
        self,
        state: SplendorState,
        *,
        forced_pass_streak: int,
        path_signatures: set[tuple[object, ...]],
    ) -> float:
        if self.nodes >= self.node_budget:
            raise TraversalCutoff("node_budget")
        self.nodes += 1
        depth = state.turn_index
        self.nodes_by_depth[depth] += 1
        self.max_depth = max(self.max_depth, depth)
        if state.terminal:
            self.terminal_leaves += 1
            return _utility(state.winner, self.traverser)
        if state.turn_index >= 400:
            raise TraversalCutoff("turn_limit")

        signature = canonical_public_belief_key(state, state.current_player)
        if signature in path_signatures:
            self.repeated_path_states += 1
        next_path = set(path_signatures)
        next_path.add(signature)
        concrete = actions_by_canonical_descriptor(state, self.env.legal_actions(state))
        action_keys = sorted(concrete, key=repr)
        if not action_keys:
            raise RuntimeError("Non-terminal state has no legal actions.")
        self.action_counts.append(len(action_keys))
        actor = state.current_player

        if actor != self.traverser:
            self.candidate_strategy_rows += 1
            selected = action_keys[self.rng.randrange(len(action_keys))]
            return self._apply_and_continue(
                state,
                concrete[selected],
                forced_pass_streak,
                next_path,
                sole_legal_action=len(action_keys) == 1,
            )

        self.candidate_advantage_rows += 1
        utilities = [
            self._apply_and_continue(
                state,
                concrete[key],
                forced_pass_streak,
                next_path,
                sole_legal_action=len(action_keys) == 1,
            )
            for key in action_keys
        ]
        return sum(utilities) / len(utilities)

    def _apply_and_continue(
        self,
        state: SplendorState,
        action,
        forced_pass_streak: int,
        path_signatures: set[tuple[object, ...]],
        *,
        sole_legal_action: bool,
    ) -> float:
        try:
            next_state = self.env.step(state, action)
        except ValueError:
            self.illegal_actions += 1
            raise
        forced = sole_legal_action and action.action_type == ActionType.PASS
        next_streak = forced_pass_streak + 1 if forced else 0
        if next_streak >= 2:
            self.forced_deadlock_leaves += 1
            return _utility(_adjudicate_scores(next_state), self.traverser)
        return self._traverse(
            next_state,
            forced_pass_streak=next_streak,
            path_signatures=path_signatures,
        )


def traversal_determinism_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "elapsed_seconds"}


def state_encoding_multiset(encoding: EntityStateEncoding) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        sorted(
            zip(
                encoding.entity_type_ids,
                encoding.card_ids,
                encoding.noble_ids,
                encoding.numeric_features,
                strict=True,
            ),
            key=repr,
        )
    )


def _encoding_payload(encoding: Any) -> dict[str, Any]:
    type_ids = (
        encoding.entity_type_ids
        if hasattr(encoding, "entity_type_ids")
        else encoding.action_type_ids
    )
    type_key = "entity_type_ids" if hasattr(encoding, "entity_type_ids") else "action_type_ids"
    return {
        type_key: list(type_ids),
        "card_ids": list(encoding.card_ids),
        "noble_ids": list(encoding.noble_ids),
        "numeric_features": [list(row) for row in encoding.numeric_features],
    }


@lru_cache(maxsize=1)
def _card_ids() -> dict[str, int]:
    cards = sorted(
        (card for tier in build_base_deck_by_tier().values() for card in tier),
        key=lambda card: card.card_id,
    )
    return {card.card_id: index for index, card in enumerate(cards, start=1)}


@lru_cache(maxsize=1)
def _noble_ids() -> dict[str, int]:
    return {
        noble.noble_id: index
        for index, noble in enumerate(sorted(build_base_nobles(), key=lambda item: item.noble_id), start=1)
    }


@lru_cache(maxsize=None)
def _card_features(card_id: str | None) -> tuple[float, ...]:
    values = [0.0] * ACTION_NUMERIC_SIZE
    if card_id is None:
        return tuple(values)
    card = next(
        (
            item
            for tier in build_base_deck_by_tier().values()
            for item in tier
            if item.card_id == card_id
        ),
        None,
    )
    if card is None:
        return tuple(values)
    values[0] = 1.0
    values[1] = float(card.tier) / 3.0
    values[2] = float(card.points) / 5.0
    colors = ("white", "blue", "green", "red", "black")
    for index, color in enumerate(colors):
        values[3 + index] = 1.0 if card.bonus_color == color else 0.0
        values[8 + index] = float(card.cost.get(color, 0)) / 7.0
    return tuple(values)


def _pad_ids(rows, width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [list(row) + [0] * (width - len(row)) for row in rows],
        dtype=torch.long,
        device=device,
    )


def _pad_features(rows, width: int, device: torch.device) -> torch.Tensor:
    zero = [0.0] * ACTION_NUMERIC_SIZE
    return torch.tensor(
        [list(row) + [zero] * (width - len(row)) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _pad_float_vectors(rows, width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [list(row) + [0.0] * (width - len(row)) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _mask(lengths, width: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[index < length for index in range(width)] for length in lengths],
        dtype=torch.bool,
        device=device,
    )


def _adjudicate_scores(state: SplendorState) -> int | None:
    scores = [player.score for player in state.players]
    best = max(scores)
    contenders = [index for index, score in enumerate(scores) if score == best]
    if len(contenders) == 1:
        return contenders[0]
    fewest = min(len(state.players[index].purchased_cards) for index in contenders)
    finalists = [
        index for index in contenders if len(state.players[index].purchased_cards) == fewest
    ]
    return finalists[0] if len(finalists) == 1 else None


def _utility(winner: int | None, player: int) -> float:
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0
