"""Preparation and guarded execution for experiment 137."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable

from splendor_ai.encoding import encode_canonical_state_entities
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE, SplendorState
from splendor_ai.training.replay import (
    deserialize_state_snapshot,
    serialize_action,
    serialize_state_snapshot,
)

from .network import AlphaZeroInference, load_checkpoint
from .search import AlphaZeroSearch, SearchConfig, SearchResult


PROTOCOL = Path("docs/engineering/alphazero_stratified_root_refill_search_137.md")
PROTOCOL_SHA256 = "a81c68e6b9921671ad72c77f6587295214f3f0229fb7f31a8cacc16125e9a54a"
CHECKPOINT_SHA256 = "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0"
STATE_COUNT = 36
STATES_PER_CELL = 6
REFERENCE_STABLE_MINIMUM = 24
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 137_300_000
PHASE_ORDER = ("early", "middle", "late")
PHASE_WINDOWS = {
    "early": (8, 19, 14),
    "middle": (20, 49, 34),
    "late": (50, 10_000, 58),
}
REFILL_ACTION_TYPES = {ActionType.BUY_VISIBLE, ActionType.RESERVE_VISIBLE}


@dataclass(frozen=True, slots=True)
class ReplaySource:
    shard: int
    path: Path
    rows: int
    size: int
    sha256: str


SOURCES = (
    ReplaySource(
        0,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_0/sanitized.jsonl"),
        3_956, 74_919_121,
        "f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b",
    ),
    ReplaySource(
        1,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_1/sanitized.jsonl"),
        4_004, 76_570_863,
        "c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60",
    ),
    ReplaySource(
        2,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_2/sanitized.jsonl"),
        3_956, 76_478_700,
        "00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8",
    ),
    ReplaySource(
        3,
        Path("outputs/alphazero_high_budget_distillation_114/selfplay/shard_3/sanitized.jsonl"),
        4_000, 76_509_765,
        "7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e",
    ),
)


@dataclass(frozen=True, slots=True)
class StratifiedRefillConfig:
    output_dir: Path
    checkpoint: Path
    device: str = "cpu"
    torch_threads: int = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def search_config(simulations: int) -> SearchConfig:
    return SearchConfig(
        simulations=simulations,
        c_puct=2.25,
        max_depth=96,
        mode="real",
        root_noise_fraction=0.0,
        temperature_turns=0,
        temperature=0.0,
        seed=0,
    )


def _stable_cycle_seed(search_seed: int, action_key: tuple[object, ...], tier: int, cycle: int) -> int:
    payload = f"137|{search_seed}|{repr(action_key)}|{tier}|{cycle}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stratified_card_schedule(
    card_ids: Iterable[str],
    *,
    visits: int,
    search_seed: int,
    action_key: tuple[object, ...],
    tier: int,
) -> list[str]:
    pool = sorted(str(card_id) for card_id in card_ids)
    if not pool or visits < 0:
        raise ValueError("A stratified schedule requires cards and nonnegative visits")
    schedule: list[str] = []
    cycle = 0
    while len(schedule) < visits:
        permutation = list(pool)
        random.Random(_stable_cycle_seed(search_seed, action_key, tier, cycle)).shuffle(permutation)
        schedule.extend(permutation)
        cycle += 1
    return schedule[:visits]


class RootRefillTrackingSearch(AlphaZeroSearch):
    """Ordinary search plus non-invasive realized root-refill telemetry."""

    treatment = "ordinary"

    def run(
        self,
        env: SplendorEnv,
        state: SplendorState,
        legal_actions: list[Action] | None = None,
        *,
        self_play: bool,
        seed: int | None = None,
    ) -> SearchResult:
        self._root_refill_pools = {
            tier: tuple(sorted(card.card_id for card in state.hidden_tier_decks[tier]))
            for tier in (1, 2, 3)
        }
        self._root_refill_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._root_refill_action_meta: dict[str, dict[str, Any]] = {}
        return super().run(env, state, legal_actions, self_play=self_play, seed=seed)

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
        del root_player, simulation
        if (
            depth != 0
            or action.action_type not in REFILL_ACTION_TYPES
            or action.tier not in (1, 2, 3)
            or not state.hidden_tier_decks[int(action.tier)]
        ):
            return state
        tier = int(action.tier)
        action_id = repr(action_key)
        self._choose_root_refill(state, action_key, action_id, tier, search_seed)
        drawn = state.hidden_tier_decks[tier][0].card_id
        if drawn not in self._root_refill_pools[tier]:
            raise RuntimeError("Root refill telemetry observed an information-illegal card")
        self._root_refill_counts[action_id][drawn] += 1
        self._root_refill_action_meta[action_id] = {
            "action_key": action_id,
            "tier": tier,
            "possible_card_ids": list(self._root_refill_pools[tier]),
        }
        return state

    def _choose_root_refill(
        self,
        state: SplendorState,
        action_key: tuple[object, ...],
        action_id: str,
        tier: int,
        search_seed: int,
    ) -> None:
        del state, action_key, action_id, tier, search_seed

    def refill_telemetry(self) -> dict[str, Any]:
        actions: dict[str, Any] = {}
        for action_id, meta in sorted(self._root_refill_action_meta.items()):
            possible = list(meta["possible_card_ids"])
            counter = self._root_refill_counts[action_id]
            counts = {card_id: int(counter[card_id]) for card_id in possible}
            positive = sum(value > 0 for value in counts.values())
            actions[action_id] = {
                **meta,
                "counts": counts,
                "visits": sum(counts.values()),
                "realized_coverage": positive / len(possible),
                "max_minus_min_count": max(counts.values()) - min(counts.values()),
            }
        return {"treatment": self.treatment, "actions": actions}


class StratifiedRootRefillSearch(RootRefillTrackingSearch):
    """Balance root refill identities separately for every exact root action."""

    treatment = "action_conditioned_stratified_root_refill"

    def _choose_root_refill(
        self,
        state: SplendorState,
        action_key: tuple[object, ...],
        action_id: str,
        tier: int,
        search_seed: int,
    ) -> None:
        pool = self._root_refill_pools[tier]
        visit_index = sum(self._root_refill_counts[action_id].values())
        desired = stratified_card_schedule(
            pool,
            visits=visit_index + 1,
            search_seed=search_seed,
            action_key=action_key,
            tier=tier,
        )[-1]
        deck = state.hidden_tier_decks[tier]
        matches = [index for index, card in enumerate(deck) if card.card_id == desired]
        if len(matches) != 1 or {card.card_id for card in deck} != set(pool):
            raise RuntimeError("Stratified refill cannot bind its information-legal card")
        index = matches[0]
        deck[0], deck[index] = deck[index], deck[0]


def _has_any_blind_reserve(state: SplendorState) -> bool:
    return any(
        player.reserved_card_source(index) != RESERVE_SOURCE_VISIBLE
        for player in state.players
        for index in range(len(player.reserved_cards))
    )


def _phase(turn: int) -> str | None:
    for name, (low, high, _target) in PHASE_WINDOWS.items():
        if low <= turn <= high:
            return name
    return None


def _qualifying_refill_actions(env: SplendorEnv, state: SplendorState) -> int:
    return sum(
        action.action_type in REFILL_ACTION_TYPES
        and action.tier in (1, 2, 3)
        and state.deck_counts[int(action.tier)] >= 8
        for action in env.legal_actions(state)
    )


def _iter_source_games(source: ReplaySource) -> Iterable[list[dict[str, Any]]]:
    current_seed: int | None = None
    rows: list[dict[str, Any]] = []
    count = 0
    with source.path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            count += 1
            seed = int(payload["game_seed"])
            if current_seed is None:
                current_seed = seed
            if seed != current_seed:
                yield rows
                rows = []
                current_seed = seed
            rows.append(payload)
    if rows:
        yield rows
    if count != source.rows:
        raise ValueError(f"137 source row mismatch for shard {source.shard}")


def _candidate_states(source: ReplaySource) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_game_ordinal, rows in enumerate(_iter_source_games(source)):
        seed = int(rows[0]["game_seed"])
        env = SplendorEnv(seed=seed)
        state = env.initial_state()
        best: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            if (
                int(row["generation"]) != 11
                or int(row["checkpoint_generation"]) != 11
                or int(row["game_seed"]) != seed
                or int(row["turn_index"]) != state.turn_index
                or int(row["player_id"]) != state.current_player
            ):
                raise ValueError("137 replay generation/turn/player reconstruction mismatch")
            encoded = encode_canonical_state_entities(state, state.current_player)
            if (
                tuple(int(value) for value in row["entity_type_ids"]) != encoded.entity_type_ids
                or tuple(int(value) for value in row["entity_card_ids"]) != encoded.card_ids
                or tuple(int(value) for value in row["entity_noble_ids"]) != encoded.noble_ids
                or tuple(tuple(float(item) for item in values) for values in row["entity_numeric_features"])
                != encoded.numeric_features
            ):
                raise ValueError("137 stored public encoding mismatch")
            phase = _phase(state.turn_index)
            qualifying = _qualifying_refill_actions(env, state)
            if (
                phase is not None
                and not state.pending_round_end
                and not _has_any_blind_reserve(state)
                and qualifying > 0
            ):
                target = PHASE_WINDOWS[phase][2]
                item = {
                    "identity": {
                        "source_shard": source.shard,
                        "source_game_ordinal": source_game_ordinal,
                        "source_game_seed": seed,
                        "source_turn_index": state.turn_index,
                        "root_player": state.current_player,
                        "phase": phase,
                        "target_turn": target,
                        "distance_from_target": abs(state.turn_index - target),
                        "qualifying_refill_actions": qualifying,
                    },
                    "state_snapshot": serialize_state_snapshot(state),
                }
                key = (phase, state.current_player)
                previous = best.get(key)
                rank = (abs(state.turn_index - target), state.turn_index)
                if previous is None:
                    best[key] = item
                else:
                    old = previous["identity"]
                    old_rank = (old["distance_from_target"], old["source_turn_index"])
                    if rank < old_rank:
                        best[key] = item
            selected_payload = row["action_payloads"][int(row["selected_action_row"])]
            matches = [
                action for action in env.legal_actions(state)
                if serialize_action(action) == selected_payload
            ]
            if len(matches) != 1:
                raise ValueError("137 selected replay action is not uniquely legal")
            state = env.step(state, matches[0])
        if not state.terminal:
            raise ValueError("137 source game did not reconstruct to a terminal state")
        candidates.extend(best.values())
    return candidates


def select_states() -> list[dict[str, Any]]:
    candidates = [item for source in SOURCES for item in _candidate_states(source)]
    selected = _minimum_cost_state_assignment(candidates)
    game_usage = Counter(
        int(item["identity"]["source_game_seed"]) for item in selected
    )
    selected.sort(
        key=lambda item: (
            PHASE_ORDER.index(item["identity"]["phase"]),
            item["identity"]["root_player"],
            item["identity"]["distance_from_target"],
            item["identity"]["source_game_seed"],
        )
    )
    for index, item in enumerate(selected):
        item["identity"]["state_index"] = index
    if (
        len(selected) != STATE_COUNT
        or len(game_usage) < 20
        or max(game_usage.values()) > 3
    ):
        raise ValueError("137 protected state count or source-game diversity failed")
    return selected


def _minimum_cost_state_assignment(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose six states per cell while maximizing source-game diversity."""

    cells = [(phase, player) for phase in PHASE_ORDER for player in (0, 1)]
    by_pair: dict[tuple[int, tuple[str, int]], dict[str, Any]] = {}
    for item in candidates:
        identity = item["identity"]
        pair = (
            int(identity["source_game_seed"]),
            (str(identity["phase"]), int(identity["root_player"])),
        )
        if pair in by_pair:
            raise ValueError("137 retained more than one candidate per game/cell")
        by_pair[pair] = item
    games = sorted({seed for seed, _cell in by_pair})

    # Residual edges are mutable [to, reverse_index, capacity, cost].
    graph: dict[str, list[list[Any]]] = defaultdict(list)

    def add_edge(source: str, target: str, capacity: int, cost: int) -> list[Any]:
        forward: list[Any] = [target, len(graph[target]), capacity, cost]
        reverse: list[Any] = [source, len(graph[source]), 0, -cost]
        graph[source].append(forward)
        graph[target].append(reverse)
        return forward

    source_node = "source"
    sink_node = "sink"
    diversity_penalty = 1_000_000
    for seed in games:
        game_node = f"game:{seed}"
        add_edge(source_node, game_node, 1, 0)
        add_edge(source_node, game_node, 1, diversity_penalty)
        add_edge(source_node, game_node, 1, 2 * diversity_penalty)
    for cell in cells:
        cell_node = f"cell:{cell[0]}:{cell[1]}"
        add_edge(cell_node, sink_node, STATES_PER_CELL, 0)

    pair_edges: dict[tuple[int, tuple[str, int]], list[Any]] = {}
    for cell in cells:
        ranked = sorted(
            (
                item for (seed, item_cell), item in by_pair.items()
                if item_cell == cell
            ),
            key=lambda item: (
                item["identity"]["distance_from_target"],
                item["identity"]["source_game_seed"],
                item["identity"]["source_turn_index"],
                item["identity"]["source_shard"],
            ),
        )
        for rank, item in enumerate(ranked):
            seed = int(item["identity"]["source_game_seed"])
            pair_edges[(seed, cell)] = add_edge(
                f"game:{seed}", f"cell:{cell[0]}:{cell[1]}", 1, rank
            )

    flow = 0
    while flow < STATE_COUNT:
        nodes = list(graph)
        distance = {node: math.inf for node in nodes}
        previous: dict[str, tuple[str, int]] = {}
        distance[source_node] = 0
        queue = deque([source_node])
        queued = {source_node}
        while queue:
            node = queue.popleft()
            queued.discard(node)
            for edge_index, edge in enumerate(graph[node]):
                target, _reverse, capacity, cost = edge
                candidate_distance = distance[node] + int(cost)
                if capacity > 0 and candidate_distance < distance[target]:
                    distance[target] = candidate_distance
                    previous[target] = (node, edge_index)
                    if target not in queued:
                        queue.append(target)
                        queued.add(target)
        if sink_node not in previous:
            raise ValueError("137 protected state assignment has insufficient capacity")
        node = sink_node
        while node != source_node:
            parent, edge_index = previous[node]
            edge = graph[parent][edge_index]
            reverse_index = int(edge[1])
            edge[2] -= 1
            graph[node][reverse_index][2] += 1
            node = parent
        flow += 1

    selected = [
        by_pair[pair]
        for pair, edge in pair_edges.items()
        if int(edge[2]) == 0
    ]
    if len(selected) != STATE_COUNT:
        raise ValueError("137 min-cost assignment accounting failed")
    return selected


def _validate_state_manifest(states: list[dict[str, Any]]) -> dict[str, Any]:
    cells: Counter[str] = Counter()
    games: set[int] = set()
    for expected_index, item in enumerate(states):
        identity = item["identity"]
        if int(identity["state_index"]) != expected_index:
            raise ValueError("137 state indices are not canonical")
        state = deserialize_state_snapshot(item["state_snapshot"])
        env = SplendorEnv(seed=int(identity["source_game_seed"]))
        encoded = encode_canonical_state_entities(state, state.current_player)
        roundtrip = deserialize_state_snapshot(serialize_state_snapshot(state))
        if (
            state.current_player != int(identity["root_player"])
            or state.turn_index != int(identity["source_turn_index"])
            or _has_any_blind_reserve(state)
            or state.pending_round_end
            or state.terminal
            or _qualifying_refill_actions(env, state) <= 0
            or encode_canonical_state_entities(roundtrip, roundtrip.current_player) != encoded
        ):
            raise ValueError("137 protected state invariant failed")
        cells[f"{identity['phase']}/player_{identity['root_player']}"] += 1
        games.add(int(identity["source_game_seed"]))
    expected_cells = {
        f"{phase}/player_{player}": STATES_PER_CELL
        for phase in PHASE_ORDER for player in (0, 1)
    }
    usage = Counter(int(item["identity"]["source_game_seed"]) for item in states)
    if (
        dict(cells) != expected_cells
        or len(games) < 20
        or max(usage.values()) > 3
    ):
        raise ValueError("137 phase/player balance or source-game diversity failed")
    return {
        "states": len(states),
        "distinct_games": len(games),
        "maximum_states_per_game": max(usage.values()),
        "cells": dict(sorted(cells.items())),
    }


def prepare_experiment(config: StratifiedRefillConfig) -> dict[str, Any]:
    observed = {
        "protocol": sha256_file(PROTOCOL),
        "checkpoint": sha256_file(config.checkpoint),
        **{f"source_shard_{source.shard}": sha256_file(source.path) for source in SOURCES},
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "checkpoint": CHECKPOINT_SHA256,
        **{f"source_shard_{source.shard}": source.sha256 for source in SOURCES},
    }
    if observed != expected:
        raise ValueError(f"137 frozen input hash mismatch: {observed}")
    for source in SOURCES:
        if source.path.stat().st_size != source.size:
            raise ValueError(f"137 source size mismatch for shard {source.shard}")
    states = select_states()
    schedule = _validate_state_manifest(states)
    state_manifest = {
        "kind": "alphazero_stratified_root_refill_search_137_state_manifest",
        "protocol_sha256": PROTOCOL_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "selection_uses_outcomes": False,
        "states": states,
    }
    # Normalize tuple-valued snapshots to their durable JSON representation so
    # an existing manifest compares exactly on subsequent preflights.
    state_manifest = json.loads(json.dumps(state_manifest, sort_keys=True))
    manifest_path = config.output_dir / "state_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != state_manifest:
            raise ValueError("137 protected state manifest changed")
    else:
        _write_json(manifest_path, state_manifest)
    state_manifest_sha256 = sha256_file(manifest_path)
    _model, identity = load_checkpoint(config.checkpoint, device=config.device)
    if identity.generation != 11:
        raise ValueError("137 checkpoint generation mismatch")
    preflight = {
        "kind": "alphazero_stratified_root_refill_search_137_preflight",
        "status": "prepared_not_launched",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": observed,
        "state_manifest": str(manifest_path),
        "state_manifest_sha256": state_manifest_sha256,
        "state_schedule": schedule,
        "searches_if_launched": STATE_COUNT * 3 * 2,
        "simulations_if_launched": STATE_COUNT * 2 * (256 + 256 + 1_024),
        "search_configs": {
            "ordinary_256": asdict(search_config(256)),
            "stratified_256": asdict(search_config(256)),
            "ordinary_1024_reference": asdict(search_config(1_024)),
        },
        "torch_threads": config.torch_threads,
        "no_protected_search_run": True,
        "no_optimizer": True,
        "no_training": True,
        "no_gameplay": True,
        "reads_experiment_134": False,
        "launch_authorized": False,
    }
    _write_json(config.output_dir / "preflight.json", preflight)
    return preflight


def _core_key(action_key: tuple[object, ...]) -> str:
    family, target, reserve_tier, net_tokens, _noble = action_key
    if family == "TAKE_TOKENS":
        core = (family, net_tokens)
    elif family in ("BUY_VISIBLE", "BUY_RESERVED", "RESERVE_VISIBLE"):
        core = (family, target)
    elif family == "RESERVE_DECK":
        core = (family, reserve_tier)
    else:
        core = (family,)
    return repr(core)


def _search_record(
    search: RootRefillTrackingSearch,
    result: SearchResult,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "seed": seed,
        "simulations": result.simulations,
        "expansions": result.expansions,
        "inference_calls": result.inference_calls,
        "max_depth": result.max_depth_reached,
        "elapsed_seconds": result.elapsed_seconds,
        "visit_top_key": repr(result.root_edges[result.visit_argmax_row].action_key),
        "visit_top_core": _core_key(result.root_edges[result.visit_argmax_row].action_key),
        "edges": [
            {
                "key": repr(edge.action_key),
                "core": _core_key(edge.action_key),
                "visits": edge.visits,
                "probability": edge.probability,
                "q": edge.q,
                "network_prior": edge.network_prior,
            }
            for edge in result.root_edges
        ],
        "refill_telemetry": search.refill_telemetry(),
    }


def _run_state(
    inference: AlphaZeroInference,
    item: dict[str, Any],
) -> dict[str, Any]:
    index = int(item["identity"]["state_index"])
    state = deserialize_state_snapshot(item["state_snapshot"])
    env = SplendorEnv(seed=int(item["identity"]["source_game_seed"]))
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in (
        "ordinary_256", "stratified_256", "ordinary_1024_reference"
    )}
    for replica in (0, 1):
        low_seed = 137_100_000 + 100 * index + replica
        reference_seed = 137_200_000 + 100 * index + replica
        for arm, search_type, simulations, seed in (
            ("ordinary_256", RootRefillTrackingSearch, 256, low_seed),
            ("stratified_256", StratifiedRootRefillSearch, 256, low_seed),
            ("ordinary_1024_reference", RootRefillTrackingSearch, 1_024, reference_seed),
        ):
            search = search_type(inference, search_config(simulations))
            result = search.run(env, state, env.legal_actions(state), self_play=False, seed=seed)
            if sum(edge.visits for edge in result.root_edges) != simulations:
                raise RuntimeError("137 root visit accounting failed")
            records[arm].append(_search_record(search, result, arm, seed))
    return {"identity": item["identity"], "searches": records}


def _edge_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(edge["key"]): edge for edge in record["edges"]}


def _tv(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _state_metrics(row: dict[str, Any]) -> dict[str, Any]:
    references = row["searches"]["ordinary_1024_reference"]
    ref_maps = [_edge_map(record) for record in references]
    if set(ref_maps[0]) != set(ref_maps[1]):
        raise ValueError("137 reference action alignment failed")
    keys = sorted(ref_maps[0])
    pooled_probability = {
        key: statistics.fmean(float(mapping[key]["probability"]) for mapping in ref_maps)
        for key in keys
    }
    pooled_q = {}
    for key in keys:
        visits = sum(int(mapping[key]["visits"]) for mapping in ref_maps)
        pooled_q[key] = (
            sum(float(mapping[key]["q"]) * int(mapping[key]["visits"]) for mapping in ref_maps) / visits
            if visits else 0.0
        )
    reference_top = max(keys, key=lambda key: (pooled_probability[key], pooled_q[key], key))
    reference_core = ref_maps[0][reference_top]["core"]
    stable = references[0]["visit_top_key"] == references[1]["visit_top_key"]
    output: dict[str, Any] = {
        "identity": row["identity"],
        "reference_top": reference_top,
        "reference_core": reference_core,
        "reference_stable": stable,
        "arms": {},
    }
    best_reference_q = max(pooled_q.values())
    for arm in ("ordinary_256", "stratified_256"):
        records = row["searches"][arm]
        arm_metrics = []
        for record in records:
            mapping = _edge_map(record)
            if set(mapping) != set(keys):
                raise ValueError("137 low-budget action alignment failed")
            chosen = record["visit_top_key"]
            probabilities = {key: float(mapping[key]["probability"]) for key in keys}
            arm_metrics.append({
                "exact_reference_agreement": chosen == reference_top,
                "core_reference_agreement": record["visit_top_core"] == reference_core,
                "reference_tv": _tv(probabilities, pooled_probability),
                "reference_q_regret": best_reference_q - pooled_q[chosen],
            })
        first_p = {key: float(value["probability"]) for key, value in _edge_map(records[0]).items()}
        second_p = {key: float(value["probability"]) for key, value in _edge_map(records[1]).items()}
        output["arms"][arm] = {
            "replicas": arm_metrics,
            "replica_exact_agreement": records[0]["visit_top_key"] == records[1]["visit_top_key"],
            "replica_core_agreement": records[0]["visit_top_core"] == records[1]["visit_top_core"],
            "replica_visit_tv": _tv(first_p, second_p),
        }
    return output


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return statistics.fmean(materialized) if materialized else 0.0


def _summarize_arm(metrics: list[dict[str, Any]], arm: str, stable_only: bool = False) -> dict[str, Any]:
    rows = [row for row in metrics if row["reference_stable"] or not stable_only]
    replicas = [replica for row in rows for replica in row["arms"][arm]["replicas"]]
    return {
        "states": len(rows),
        "exact_reference_agreement": _mean(item["exact_reference_agreement"] for item in replicas),
        "core_reference_agreement": _mean(item["core_reference_agreement"] for item in replicas),
        "mean_reference_tv": _mean(item["reference_tv"] for item in replicas),
        "mean_reference_q_regret": _mean(item["reference_q_regret"] for item in replicas),
        "replica_exact_agreement": _mean(row["arms"][arm]["replica_exact_agreement"] for row in rows),
        "replica_core_agreement": _mean(row["arms"][arm]["replica_core_agreement"] for row in rows),
        "mean_replica_visit_tv": _mean(row["arms"][arm]["replica_visit_tv"] for row in rows),
    }


def _bootstrap(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    per_game: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in metrics:
        if not row["reference_stable"]:
            continue
        control = row["arms"]["ordinary_256"]
        treatment = row["arms"]["stratified_256"]
        item = {}
        for name in (
            "exact_reference_agreement", "core_reference_agreement",
            "reference_tv", "reference_q_regret",
        ):
            item[name] = _mean(replica[name] for replica in treatment["replicas"]) - _mean(
                replica[name] for replica in control["replicas"]
            )
        item["replica_exact_agreement"] = float(treatment["replica_exact_agreement"]) - float(
            control["replica_exact_agreement"]
        )
        seed = int(row["identity"]["source_game_seed"])
        per_game[seed].append(item)
    if not per_game:
        raise ValueError("137 has no stable reference states for bootstrap")
    game_items = [
        {name: _mean(item[name] for item in items) for name in items[0]}
        for _seed, items in sorted(per_game.items())
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    names = tuple(game_items[0])
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(BOOTSTRAP_SAMPLES):
        picked = [game_items[rng.randrange(len(game_items))] for _row in game_items]
        for name in names:
            samples[name].append(_mean(item[name] for item in picked))
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "stable_states": sum(len(items) for items in per_game.values()),
        "stable_source_game_clusters": len(game_items),
        "treatment_minus_control": {
            name: {
                "point": _mean(item[name] for item in game_items),
                "ci95": [
                    sorted(values)[int(0.025 * len(values))],
                    sorted(values)[min(int(0.975 * len(values)), len(values) - 1)],
                ],
            }
            for name, values in samples.items()
        },
    }


def _coverage_summary(panel: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    coverages: list[float] = []
    priority: list[float] = []
    balanced = True
    elapsed: list[float] = []
    for row in panel:
        for record in row["searches"][arm]:
            elapsed.append(float(record["elapsed_seconds"]))
            visits_by_key = {edge["key"]: int(edge["visits"]) for edge in record["edges"]}
            for action in record["refill_telemetry"]["actions"].values():
                coverage = float(action["realized_coverage"])
                coverages.append(coverage)
                if visits_by_key[action["action_key"]] >= 0.05 * int(record["simulations"]):
                    priority.append(coverage)
                if arm == "stratified_256" and int(action["max_minus_min_count"]) > 1:
                    balanced = False
    return {
        "refill_edges": len(coverages),
        "priority_refill_edges": len(priority),
        "mean_realized_coverage": _mean(coverages),
        "priority_mean_realized_coverage": _mean(priority),
        "all_stratified_counts_balanced": balanced,
        "median_wall_seconds": statistics.median(elapsed),
        "total_wall_seconds": sum(elapsed),
    }


def summarize_panel(panel: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [_state_metrics(row) for row in panel]
    stable = sum(row["reference_stable"] for row in metrics)
    summaries = {
        arm: {
            "all_states": _summarize_arm(metrics, arm),
            "reference_stable_states": _summarize_arm(metrics, arm, stable_only=True),
            "by_phase": {
                phase: _summarize_arm(
                    [row for row in metrics if row["identity"]["phase"] == phase], arm
                )
                for phase in PHASE_ORDER
            },
        }
        for arm in ("ordinary_256", "stratified_256")
    }
    bootstrap = _bootstrap(metrics)
    coverage = {
        arm: _coverage_summary(panel, arm)
        for arm in ("ordinary_256", "stratified_256", "ordinary_1024_reference")
    }
    control_coverage = coverage["ordinary_256"]["priority_mean_realized_coverage"]
    treatment_coverage = coverage["stratified_256"]["priority_mean_realized_coverage"]
    wall_ratio = coverage["stratified_256"]["median_wall_seconds"] / max(
        coverage["ordinary_256"]["median_wall_seconds"], 1e-12
    )
    sanity = (
        coverage["stratified_256"]["all_stratified_counts_balanced"]
        and treatment_coverage - control_coverage >= 0.10
        and wall_ratio <= 1.25
    )
    deltas = bootstrap["treatment_minus_control"]
    stable_control = summaries["ordinary_256"]["reference_stable_states"]
    stable_treatment = summaries["stratified_256"]["reference_stable_states"]
    agreement_pass = any(
        stable_treatment[name] - stable_control[name] >= 0.10
        and deltas[name]["ci95"][0] > 0.0
        for name in ("exact_reference_agreement", "core_reference_agreement")
    )
    tv_relative = (
        stable_treatment["mean_reference_tv"]
        <= 0.80 * stable_control["mean_reference_tv"]
        and deltas["reference_tv"]["ci95"][1] < 0.0
    )
    regret_relative = (
        stable_treatment["mean_reference_q_regret"]
        <= 0.80 * stable_control["mean_reference_q_regret"]
        and deltas["reference_q_regret"]["ci95"][1] < 0.0
    )
    stability_pass = (
        stable_treatment["replica_exact_agreement"]
        - stable_control["replica_exact_agreement"] >= -0.02
    )
    ranking_gain = (
        stable >= REFERENCE_STABLE_MINIMUM
        and agreement_pass
        and (tv_relative or regret_relative)
        and stability_pass
    )
    if not sanity:
        decision = "stratified_refill_mechanism_invalid_no_search_conclusion"
    elif ranking_gain:
        decision = "stratified_refill_improves_search_efficiency_design_training_pilot"
    else:
        decision = "stratified_refill_coverage_not_ranking_bottleneck_keep_ordinary_search"
    return {
        "reference_stable_states": stable,
        "arms": summaries,
        "coverage_and_runtime": coverage,
        "stratified_to_ordinary_median_wall_ratio": wall_ratio,
        "bootstrap": bootstrap,
        "gates": {
            "mechanism_sanity": sanity,
            "agreement_gain": agreement_pass,
            "tv_relative_gain": tv_relative,
            "q_regret_relative_gain": regret_relative,
            "replica_stability": stability_pass,
            "material_ranking_gain": ranking_gain,
        },
        "decision": decision,
    }


def run_experiment(config: StratifiedRefillConfig, *, launch: bool = False) -> dict[str, Any]:
    preflight = prepare_experiment(config)
    if not launch:
        return preflight
    manifest_path = Path(preflight["state_manifest"])
    if sha256_file(manifest_path) != preflight["state_manifest_sha256"]:
        raise ValueError("137 protected state manifest identity changed before launch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inference = AlphaZeroInference(config.checkpoint, device=config.device)
    state_dir = config.output_dir / "states"
    panel: list[dict[str, Any]] = []
    started = time.perf_counter()
    for item in manifest["states"]:
        index = int(item["identity"]["state_index"])
        path = state_dir / f"state_{index:02d}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("protocol_sha256") != PROTOCOL_SHA256 or record.get("identity") != item["identity"]:
                raise ValueError("137 resumed state identity mismatch")
        else:
            record = {
                "protocol_sha256": PROTOCOL_SHA256,
                **_run_state(inference, item),
            }
            _write_json(path, record)
        panel.append(record)
        _write_json(config.output_dir / "progress.json", {
            "kind": "alphazero_stratified_root_refill_search_137",
            "status": "running",
            "protocol_sha256": PROTOCOL_SHA256,
            "state_manifest_sha256": preflight["state_manifest_sha256"],
            "completed_states": len(panel),
            "total_states": STATE_COUNT,
            "training": False,
            "gameplay": False,
        })
    summary = summarize_panel(panel)
    result = {
        "kind": "alphazero_stratified_root_refill_search_137",
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "state_manifest_sha256": preflight["state_manifest_sha256"],
        "preflight": preflight,
        "summary": summary,
        "decision": summary["decision"],
        "elapsed_seconds": time.perf_counter() - started,
        "training": False,
        "gameplay": False,
        "checkpoint_changed": False,
        "experiment_134_changed": False,
        "automatic_continuation": False,
    }
    _write_json(config.output_dir / "result.json", result)
    _write_json(config.output_dir / "progress.json", result)
    return result
