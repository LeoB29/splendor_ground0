"""Repeated-return value-target A/B for AlphaZero programme 131."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    canonical_public_belief_key,
    canonical_successor_action_key,
    collate_entity_action_batch,
    encode_canonical_state_entities,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.constants import ALL_TOKEN_TYPES
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK, SplendorState
from splendor_ai.termination import adjudicate_scores, outcome_value
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot

from .arena import ArenaResult, alpha_zero_search_factory, run_paired_arena
from .historical_state_reanalysis import (
    _arena_dict,
    _iter_source_games,
    _reconstruct_start,
    phase_for_slot,
)
from .network import AlphaZeroInference, load_checkpoint, save_checkpoint
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig
from .search_primary import paired_score_share_bootstrap


@dataclass(frozen=True, slots=True)
class RepeatedReturnValueABConfig:
    output_dir: Path
    result_130: Path
    generation_11_checkpoint: Path
    installed_checkpoint: Path
    source_replays: tuple[Path, Path, Path, Path]
    device: str = "cpu"
    collection_workers: int = 4
    arena_workers: int = 4
    torch_threads: int = 1
    max_turns: int = 200
    experiment_number: int = 131
    protocol_sha256: str = "b8fbb2a0cafe1c9be5a75b33de4e6ab2325b7ec7986be138be85d38bcfbe289f"
    valid_termination_reasons: tuple[str, ...] = ("completed",)
    predecessor_result: Path | None = None
    predecessor_result_sha256: str | None = None
    predecessor_required_decision: str | None = None


PROTOCOL_SHA256 = "b8fbb2a0cafe1c9be5a75b33de4e6ab2325b7ec7986be138be85d38bcfbe289f"
SOURCE_GENERATIONS = (2, 5, 8, 11)
EXPECTED_HASHES = {
    "result_130": "cc884522a98a6b45105d28a8d878bd31267ad6504f444b14cb9d1dfd0df4d417",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
    "source_generation_2": "855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56",
    "source_generation_5": "0f9393fb020cf01c4d06f829c03e7426913dcc29d6afde49091cb9def57772e1",
    "source_generation_8": "96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10",
    "source_generation_11": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
}
TRAIN_START = 112
TRAIN_STATES_PER_SOURCE = 60
PROTECTED_START = 172
PROTECTED_STATES_PER_SOURCE = 15
FIT_PER_CELL = 16
DEV_PER_CELL = 4
WORLD_COUNT = 4
TRAIN_POLICY_COUNT = 2
PROTECTED_POLICY_COUNT = 4
WORLD_SEED_BASE = 113_300_000
POLICY_SEED_BASE = 113_400_000
TRAINING_SEED = 1_131_000
BOOTSTRAP_SAMPLES = 50_000
PAIRED_SEEDS = 200
ARENA_SEED_CONTROL = 2_730_000
ARENA_SEED_PARENT = 2_731_000


class RepeatedReturnIntegrityError(RuntimeError):
    """A scientifically invalid continuation, rather than a software failure."""


def search_config(*, simulations: int = 256) -> SearchConfig:
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


def repeated_return_decision(
    *, offline_passed: bool, control_gameplay_passed: bool | None, parent_gameplay_passed: bool | None
) -> str:
    if not offline_passed:
        return "repeated_return_value_target_not_causally_better_no_gameplay"
    if control_gameplay_passed is not True:
        return "repeated_return_value_offline_gain_not_search_causal"
    if parent_gameplay_passed is not True:
        return "repeated_return_value_gain_not_parent_robust"
    return "repeated_return_value_causal_authorize_joint_reanalysis_design_no_install"


def repeated_gameplay_assessment(
    arena: dict[str, Any],
    bootstrap: dict[str, float | int],
    valid_termination_reasons: tuple[str, ...],
) -> dict[str, Any]:
    seat_shares = {}
    for seat in (0, 1):
        rows = [row for row in arena["rows"] if int(row["candidate_seat"]) == seat]
        seat_shares[str(seat)] = statistics.fmean(float(row["candidate_points"]) for row in rows)
    terminations = dict(arena["termination_reasons"])
    checks = {
        "score_share_at_least_55_percent": float(arena["score_share"]) >= 0.55,
        "positive_margin": float(arena["mean_margin"]) > 0.0,
        "seat_zero_nonlosing": seat_shares["0"] >= 0.5,
        "seat_one_nonlosing": seat_shares["1"] >= 0.5,
        "paired_ci95_low_strictly_above_half": float(bootstrap["ci95_low"]) > 0.5,
        "all_terminations_valid": (
            sum(int(count) for count in terminations.values()) == int(arena["games"])
            and set(terminations).issubset(set(valid_termination_reasons))
        ),
    }
    return {
        "passed": all(checks.values()),
        "category": "repeated_return_strength_pass" if all(checks.values()) else "strength_not_established",
        "checks": checks,
        "seat_score_shares": seat_shares,
    }


def value_only_scope(model: nn.Module) -> dict[str, int]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("value_head."))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total_parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable}


def split_for_block_index(block_index: int) -> str:
    if not 0 <= block_index < TRAIN_STATES_PER_SOURCE:
        raise ValueError("Training block index is out of range.")
    within_cell = block_index // 3
    return "fitting" if within_cell < FIT_PER_CELL else "development"


def protected_qualification(
    *, training: dict[str, Any], metrics: dict[str, Any], bootstrap: dict[str, Any], integrity: dict[str, Any]
) -> dict[str, Any]:
    parent = metrics["generation_11"]
    control = metrics["single_outcome_control"]
    treatment = metrics["repeated_mean_treatment"]
    checks = {
        "treatment_nonzero_epoch": int(training["repeated_mean_treatment"]["selected_epoch"]) > 0,
        "integrity_passed": integrity["passed"],
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for row in metrics.values()
            for key, value in row.items()
            if key in {"mse", "mae", "pearson"}
        ),
        "treatment_mse_at_most_90_percent_control": treatment["mse"] <= 0.90 * control["mse"],
        "treatment_mse_below_parent": treatment["mse"] < parent["mse"],
        "treatment_mae_no_worse_than_control": treatment["mae"] <= control["mae"],
        "treatment_control_bootstrap_upper_below_zero": bootstrap["treatment_minus_control_squared_error_ci95"][1] < 0.0,
        "treatment_parent_bootstrap_upper_below_zero": bootstrap["treatment_minus_parent_squared_error_ci95"][1] < 0.0,
        "all_phase_mse_within_control_plus_0_02": all(
            treatment["phase_breakdown"][phase]["mse"] <= control["phase_breakdown"][phase]["mse"] + 0.02
            for phase in ("early", "middle", "late")
        ),
    }
    return {"qualified": all(checks.values()), "checks": checks}


def run_repeated_return_value_ab_131(
    config: RepeatedReturnValueABConfig, *, launch: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    states, preflight = _preflight(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "preflight.json", preflight)
    if not launch:
        return preflight
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == config.protocol_sha256:
            return payload
        raise ValueError("Existing 131 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": f"alphazero_repeated_return_value_ab_{config.experiment_number}",
        "status": "running",
        "active_stage": "repeated_return_collection",
        "protocol_sha256": config.protocol_sha256,
        "preflight": preflight,
        "collection": None,
        "training": None,
        "protected_evaluation": None,
        "panels": {},
        "external_035_arena": None,
        "installed_checkpoint_unchanged": True,
        "promotion_authorized": False,
        "automatic_continuation": False,
        "decision": "running",
    }
    _write_progress(config, result, started)
    try:
        collection = _run_or_load_collection(config, states)
    except RepeatedReturnIntegrityError as exc:
        result["integrity_error"] = str(exc)
        return _seal(
            config,
            result,
            started,
            "repeated_return_value_integrity_failed_no_strength_conclusion",
        )
    result["collection"] = collection
    result["active_stage"] = "matched_value_only_training"
    _write_progress(config, result, started)
    training = _run_or_load_training(config, collection)
    result["training"] = training
    result["active_stage"] = "protected_repeated_return_evaluation"
    _write_progress(config, result, started)
    protected = _run_or_load_protected(config, collection, training)
    result["protected_evaluation"] = protected
    _write_progress(config, result, started)
    if not protected["qualification"]["qualified"]:
        return _seal(config, result, started, repeated_return_decision(
            offline_passed=False, control_gameplay_passed=None, parent_gameplay_passed=None
        ))
    treatment_checkpoint = Path(training["repeated_mean_treatment"]["checkpoint"])
    control_checkpoint = Path(training["single_outcome_control"]["checkpoint"])
    result["active_stage"] = "treatment_vs_single_outcome_gameplay"
    _write_progress(config, result, started)
    control_panel = _run_or_load_arena(
        config,
        name="treatment_vs_single_outcome_control",
        candidate=treatment_checkpoint,
        baseline=control_checkpoint,
        seed_start=ARENA_SEED_CONTROL,
    )
    result["panels"]["single_outcome_control"] = control_panel
    _write_progress(config, result, started)
    if not control_panel["assessment"]["passed"]:
        return _seal(config, result, started, repeated_return_decision(
            offline_passed=True, control_gameplay_passed=False, parent_gameplay_passed=None
        ))
    result["active_stage"] = "treatment_vs_generation_11_gameplay"
    _write_progress(config, result, started)
    parent_panel = _run_or_load_arena(
        config,
        name="treatment_vs_generation_11",
        candidate=treatment_checkpoint,
        baseline=config.generation_11_checkpoint,
        seed_start=ARENA_SEED_PARENT,
    )
    result["panels"]["generation_11"] = parent_panel
    return _seal(config, result, started, repeated_return_decision(
        offline_passed=True,
        control_gameplay_passed=True,
        parent_gameplay_passed=bool(parent_panel["assessment"]["passed"]),
    ))


def _preflight(config: RepeatedReturnValueABConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(config.source_replays) != 4:
        raise ValueError("131 requires four source replays.")
    observed = {
        "result_130": _sha256(config.result_130),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
        **{
            f"source_generation_{generation}": _sha256(path)
            for generation, path in zip(SOURCE_GENERATIONS, config.source_replays, strict=True)
        },
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"131 frozen input mismatch: {observed}")
    if any(
        value is not None
        for value in (
            config.predecessor_result,
            config.predecessor_result_sha256,
            config.predecessor_required_decision,
        )
    ):
        if not all(
            value is not None
            for value in (
                config.predecessor_result,
                config.predecessor_result_sha256,
                config.predecessor_required_decision,
            )
        ):
            raise ValueError("Repeated-return predecessor identity is incomplete.")
        predecessor_hash = _sha256(config.predecessor_result)
        if predecessor_hash != config.predecessor_result_sha256:
            raise ValueError("Repeated-return predecessor result hash mismatch.")
        predecessor = json.loads(config.predecessor_result.read_text(encoding="utf-8"))
        if (
            predecessor.get("status") != "complete"
            or predecessor.get("decision") != config.predecessor_required_decision
        ):
            raise ValueError("Repeated-return predecessor decision mismatch.")
        observed["predecessor_result"] = predecessor_hash
    prior = json.loads(config.result_130.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "complete"
        or prior.get("decision") != "trajectory_policy_value_uncertainty_material_review_joint_target"
        or prior.get("installed_checkpoint_unchanged") is not True
    ):
        raise ValueError("131 requires the exact completed 130 decision.")
    states, source_summaries = select_repeated_return_states(config.source_replays)
    model, checkpoint = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    if checkpoint.generation != 11:
        raise ValueError("131 parent must be exact generation 11.")
    scope = value_only_scope(model)
    if scope != {"total_parameters": 356_450, "trainable_parameters": 4_705, "frozen_parameters": 351_745}:
        raise ValueError(f"131 value-only trainable scope mismatch: {scope}")
    inference = AlphaZeroInference(config.generation_11_checkpoint, device=config.device)
    invariant_rows = [_state_world_invariants(row, inference) for row in states]
    deadlock_search_contract = _deadlock_search_contract(inference)
    first_state = deserialize_state_snapshot(states[0]["state_snapshot"])
    temporary = AlphaZeroSearch(inference, search_config(simulations=8)).run(
        SplendorEnv(), first_state, self_play=False, seed=POLICY_SEED_BASE
    )
    return states, {
        "kind": f"alphazero_repeated_return_value_ab_{config.experiment_number}_preflight",
        "status": "preflight_passed",
        "protocol_sha256": config.protocol_sha256,
        "input_hashes": observed,
        "source_summaries": source_summaries,
        "states": len(states),
        "split_counts": dict(sorted(Counter(row["identity"]["split"] for row in states).items())),
        "phase_counts": dict(sorted(Counter(row["identity"]["phase"] for row in states).items())),
        "source_counts": dict(sorted(Counter(str(row["identity"]["source_generation"]) for row in states).items())),
        "continuations_if_launched": 2_880,
        "valid_termination_reasons": list(config.valid_termination_reasons),
        "value_only_scope": scope,
        "world_invariants": invariant_rows,
        "known_deadlock_regression_states": {
            str(index): next(row["identity"] for row in states if row["identity"]["state_index"] == index)
            for index in (48, 294)
        },
        "deadlock_search_contract": deadlock_search_contract,
        "temporary_root_search": {
            "simulations": temporary.simulations,
            "root_edges": len(temporary.root_edges),
            "persisted_as_evidence": False,
        },
        "no_optimizer_instantiated": True,
        "no_training": True,
        "no_durable_continuation": True,
        "no_strength_gameplay": True,
    }


def select_repeated_return_states(
    paths: tuple[Path, Path, Path, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for source_index, (generation, path) in enumerate(zip(SOURCE_GENERATIONS, paths, strict=True)):
        game_count = 0
        row_count = 0
        selected_source = 0
        for ordinal, game_rows in enumerate(_iter_source_games(path)):
            game_count += 1
            row_count += len(game_rows)
            if any(
                row.generation != generation
                or row.checkpoint_generation != generation
                or row.termination_reason != "completed"
                for row in game_rows
            ):
                raise ValueError(f"131 generation-{generation} source identity mismatch.")
            if TRAIN_START <= ordinal < TRAIN_START + TRAIN_STATES_PER_SOURCE:
                block_index = ordinal - TRAIN_START
                phase = phase_for_slot(block_index)
                split = split_for_block_index(block_index)
            elif PROTECTED_START <= ordinal < PROTECTED_START + PROTECTED_STATES_PER_SOURCE:
                block_index = ordinal - PROTECTED_START
                phase = phase_for_slot(block_index)
                split = "protected"
            else:
                continue
            _env, state, source = _reconstruct_start(
                game_rows, phase, generation, source_index, ordinal
            )
            global_index = source_index * (TRAIN_STATES_PER_SOURCE + PROTECTED_STATES_PER_SOURCE) + selected_source
            identity = {
                "state_index": global_index,
                "source_index": source_index,
                "source_generation": generation,
                "source_game_ordinal": ordinal,
                "source_game_seed": source["source_game_seed"],
                "source_turn_index": source["source_turn_index"],
                "root_player": source["source_player"],
                "phase": phase,
                "split": split,
                "source_progress": source["source_progress"],
                "pending_round_end": source["pending_round_end"],
            }
            selected.append({"identity": identity, "state_snapshot": serialize_state_snapshot(state)})
            selected_source += 1
        expected_games = 224 if generation == 11 else 256
        if game_count != expected_games or selected_source != 75:
            raise ValueError(f"131 generation-{generation} source schedule mismatch.")
        summaries.append({
            "source_index": source_index,
            "generation": generation,
            "path": str(path),
            "sha256": _sha256(path),
            "games": game_count,
            "rows": row_count,
            "selected_states": selected_source,
        })
    expected_splits = Counter({"fitting": 192, "development": 48, "protected": 60})
    if len(selected) != 300 or Counter(row["identity"]["split"] for row in selected) != expected_splits:
        raise ValueError("131 exact state split mismatch.")
    return selected, summaries


def _state_world_invariants(row: dict[str, Any], inference: AlphaZeroInference) -> dict[str, Any]:
    state = deserialize_state_snapshot(row["state_snapshot"])
    index = int(row["identity"]["state_index"])
    worlds = _possible_worlds(state, index)
    env = SplendorEnv()
    keys = [canonical_public_belief_key(world, state.current_player) for world in worlds]
    mappings = [actions_by_canonical_successor_key(world, env.legal_actions(world)) for world in worlds]
    action_keys = [tuple(sorted(mapping, key=repr)) for mapping in mappings]
    encodings = [encode_canonical_state_entities(world, state.current_player) for world in worlds]
    predictions = [
        inference.predict_one(world, [mappings[slot][key] for key in action_keys[0]])
        for slot, world in enumerate(worlds)
    ]
    checks = {
        "four_unique_worlds": len({state_signature(world) for world in worlds}) == WORLD_COUNT,
        "public_key_invariant": all(key == keys[0] for key in keys),
        "semantic_actions_invariant": all(keys_for_world == action_keys[0] for keys_for_world in action_keys),
        "canonical_tensor_invariant": all(encoding == encodings[0] for encoding in encodings),
        "parent_prediction_invariant": all(prediction == predictions[0] for prediction in predictions),
    }
    if not all(checks.values()):
        raise ValueError(f"131 world invariant failed for state {index}: {checks}")
    return {"state_index": index, "root_actions": len(action_keys[0]), "checks": checks}


def _deadlock_search_contract(inference: AlphaZeroInference) -> dict[str, Any]:
    env = SplendorEnv(seed=132)
    state = env.initial_state()
    state.bank_tokens.update({color: 0 for color in ALL_TOKEN_TYPES})
    state.players[0].score = 2
    cards = list(state.hidden_tier_decks[1][:6])
    for player_id, player in enumerate(state.players):
        for card in cards[player_id * 3 : player_id * 3 + 3]:
            player.reserve_card(card, RESERVE_SOURCE_DECK)
    legal = env.legal_actions(state)
    if len(legal) != 1 or legal[0].action_type is not ActionType.PASS:
        raise ValueError("Repeated-return synthetic deadlock is not exact.")
    result = AlphaZeroSearch(inference, search_config(simulations=8)).run(
        env,
        state,
        legal,
        self_play=False,
        seed=POLICY_SEED_BASE + 132,
    )
    checks = {
        "sole_forced_pass": len(result.root_edges) == 1,
        "all_visits_accounted": result.root_edges[0].visits == 8,
        "adjudicated_root_q_positive_one": result.root_edges[0].q == 1.0,
        "adjudicated_search_value_positive_one": result.search_root_value == 1.0,
        "deadlock_resolved_without_depth_loop": result.max_depth_reached == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Repeated-return deadlock search contract failed: {checks}")
    return {"simulations": 8, "checks": checks}


def _possible_worlds(state: SplendorState, state_index: int) -> list[SplendorState]:
    return [
        determinize_for_observer(
            state,
            state.current_player,
            random.Random(WORLD_SEED_BASE + state_index * 10_003 + world_slot),
        )
        for world_slot in range(WORLD_COUNT)
    ]


def _run_or_load_collection(
    config: RepeatedReturnValueABConfig, states: list[dict[str, Any]]
) -> dict[str, Any]:
    output = config.output_dir / "collection"
    manifest = output / "collection.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_collection(payload, config.protocol_sha256)
        return payload
    tasks = []
    for source_index, generation in enumerate(SOURCE_GENERATIONS):
        tasks.append({
            "protocol_sha256": config.protocol_sha256,
            "source_index": source_index,
            "source_generation": generation,
            "states": [row for row in states if int(row["identity"]["source_index"]) == source_index],
            "checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
            "torch_threads": config.torch_threads,
            "max_turns": config.max_turns,
            "valid_termination_reasons": list(config.valid_termination_reasons),
            "output_dir": str(output / f"source_generation_{generation}"),
        })
    shards = {}
    integrity_errors = []
    with ProcessPoolExecutor(max_workers=config.collection_workers) as executor:
        futures = {executor.submit(_collection_worker, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                record = future.result()
            except RepeatedReturnIntegrityError as exc:
                integrity_errors.append(str(exc))
            else:
                shards[int(record["source_index"])] = record
    if integrity_errors:
        raise RepeatedReturnIntegrityError("; ".join(sorted(integrity_errors)))
    if set(shards) != set(range(4)):
        raise RuntimeError("131 collection requires four exact source shards.")
    payload = {
        "protocol_sha256": config.protocol_sha256,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "shards": {str(index): shards[index] for index in range(4)},
        "states": sum(int(shards[index]["states"]) for index in range(4)),
        "continuations": sum(int(shards[index]["continuations"]) for index in range(4)),
        "training_development_paths": [
            path for index in range(4) for path in shards[index]["training_development_paths"]
        ],
        "protected_paths": [path for index in range(4) for path in shards[index]["protected_paths"]],
    }
    _validate_collection(payload, config.protocol_sha256)
    _write_json(manifest, payload)
    return payload


def _collection_worker(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(task["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    output = Path(task["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    progress_path = output / "progress.json"
    identity = {
        "protocol_sha256": str(task["protocol_sha256"]),
        "source_index": int(task["source_index"]),
        "source_generation": int(task["source_generation"]),
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "state_count": len(task["states"]),
        "search": asdict(search_config()),
        "world_count": WORLD_COUNT,
        "training_policy_count": TRAIN_POLICY_COUNT,
        "protected_policy_count": PROTECTED_POLICY_COUNT,
        "max_turns": int(task["max_turns"]),
        "valid_termination_reasons": list(task["valid_termination_reasons"]),
    }
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("131 collection shard identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError("131 progress exists without shard identity.")
        _write_json(identity_path, identity)
    inference = AlphaZeroInference(task["checkpoint"], device=task["device"])
    records: list[dict[str, Any]] = []
    for row in task["states"]:
        index = int(row["identity"]["state_index"])
        split = str(row["identity"]["split"])
        state_path = output / split / f"state_{index:03d}.json"
        if state_path.exists():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            _validate_collected_state(payload, row, str(task["protocol_sha256"]))
        else:
            payload = _collect_state(
                row,
                inference,
                int(task["max_turns"]),
                protocol_sha256=str(task["protocol_sha256"]),
                valid_termination_reasons=tuple(task["valid_termination_reasons"]),
            )
            _write_json(state_path, payload)
        records.append({
            "state_index": index,
            "split": split,
            "path": str(state_path),
            "sha256": _sha256(state_path),
            "continuations": len(payload["continuations"]),
        })
        _write_json(progress_path, {
            "identity": identity,
            "complete": False,
            "completed_states": len(records),
            "completed_continuations": sum(record["continuations"] for record in records),
            "records": records,
        })
        print(
            f"131 generation {task['source_generation']} states {len(records)}/{len(task['states'])}",
            flush=True,
        )
    payload = {
        "protocol_sha256": str(task["protocol_sha256"]),
        "source_index": int(task["source_index"]),
        "source_generation": int(task["source_generation"]),
        "states": len(records),
        "continuations": sum(record["continuations"] for record in records),
        "training_development_paths": [record["path"] for record in records if record["split"] != "protected"],
        "protected_paths": [record["path"] for record in records if record["split"] == "protected"],
        "records": records,
    }
    _write_json(progress_path, {"identity": identity, "complete": True, "record": payload, "records": records})
    return payload


def _collect_state(
    row: dict[str, Any],
    inference: AlphaZeroInference,
    max_turns: int,
    *,
    protocol_sha256: str = PROTOCOL_SHA256,
    valid_termination_reasons: tuple[str, ...] = ("completed",),
) -> dict[str, Any]:
    state = deserialize_state_snapshot(row["state_snapshot"])
    identity = row["identity"]
    state_index = int(identity["state_index"])
    invariants = _state_world_invariants(row, inference)
    policy_count = PROTECTED_POLICY_COUNT if identity["split"] == "protected" else TRAIN_POLICY_COUNT
    env = SplendorEnv(seed=int(identity["source_game_seed"]))
    root_search = AlphaZeroSearch(inference, search_config())
    actions: list[tuple[object, ...]] = []
    root_results = []
    for policy_slot in range(policy_count):
        seed = POLICY_SEED_BASE + state_index * 10_003 + policy_slot
        result = root_search.run(env, state, self_play=False, seed=seed)
        action_key = canonical_successor_action_key(state, result.action)
        actions.append(action_key)
        root_results.append({
            "policy_slot": policy_slot,
            "seed": seed,
            "selected_action_key": repr(action_key),
            "raw_network_value": result.raw_network_value,
            "search_root_value": result.search_root_value,
        })
    continuations = []
    for world_slot, world in enumerate(_possible_worlds(state, state_index)):
        for policy_slot, action_key in enumerate(actions):
            continuations.append(_play_continuation(
                initial_state=world,
                root_player=state.current_player,
                first_action_key=action_key,
                inference=inference,
                state_index=state_index,
                world_slot=world_slot,
                policy_slot=policy_slot,
                max_turns=max_turns,
            ))
    if any(item["termination_reason"] not in valid_termination_reasons for item in continuations):
        reasons = Counter(item["termination_reason"] for item in continuations)
        raise RepeatedReturnIntegrityError(
            f"131 state {state_index} produced abnormal continuations: {dict(reasons)}"
        )
    targets = [float(item["wdl"]) for item in continuations]
    return {
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "identity": identity,
        "state_snapshot": row["state_snapshot"],
        "invariants": invariants,
        "root_results": root_results,
        "continuations": continuations,
        "single_outcome_target": next(
            float(item["wdl"])
            for item in continuations
            if item["world_slot"] == 0 and item["policy_slot"] == 0
        ),
        "repeated_mean_target": statistics.fmean(targets),
        "target_variance": statistics.pvariance(targets),
    }


def _play_continuation(
    *, initial_state: SplendorState, root_player: int, first_action_key: tuple[object, ...],
    inference: AlphaZeroInference, state_index: int, world_slot: int, policy_slot: int,
    max_turns: int
) -> dict[str, Any]:
    env = SplendorEnv()
    state = initial_state.copy_shallow()
    legal_map = actions_by_canonical_successor_key(state, env.legal_actions(state))
    if first_action_key not in legal_map:
        raise ValueError("131 public root action cannot be rebound in a possible world.")
    action = legal_map[first_action_key]
    visits: dict[tuple[object, ...], int] = {state_signature(state): 1}
    forced_passes = int(len(legal_map) == 1 and action.action_type is ActionType.PASS)
    next_state = env.step(state, action)
    no_progress = 0 if is_progress_transition(state, next_state, action) else 1
    state = next_state
    reason = "completed"
    search = AlphaZeroSearch(inference, search_config())
    while not state.terminal:
        signature = state_signature(state)
        visits[signature] = visits.get(signature, 0) + 1
        if visits[signature] >= 4:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= max_turns:
            reason = "max_turns"
            break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"
            break
        seed = (
            POLICY_SEED_BASE
            + state_index * 10_000_019
            + policy_slot * 100_003
            + state.turn_index * 97_409
        )
        result = search.run(env, state, legal, self_play=False, seed=seed)
        action = result.action
        next_state = env.step(state, action)
        forced = len(legal) == 1 and action.action_type is ActionType.PASS
        forced_passes = forced_passes + 1 if forced else 0
        if forced_passes >= 2:
            state = next_state
            reason = "forced_pass_deadlock"
            break
        no_progress = 0 if is_progress_transition(state, next_state, action) else no_progress + 1
        state = next_state
        if no_progress >= 60:
            reason = "no_progress_cutoff"
            break
    if state.terminal and reason == "completed":
        winner = state.winner
    elif reason == "forced_pass_deadlock":
        winner = adjudicate_scores(state)
    else:
        return {
            "world_slot": world_slot,
            "policy_slot": policy_slot,
            "termination_reason": reason,
            "turns": state.turn_index,
            "wdl": None,
            "score_margin": None,
        }
    return {
        "world_slot": world_slot,
        "policy_slot": policy_slot,
        "termination_reason": reason,
        "turns": state.turn_index,
        "winner": winner,
        "final_scores": [state.players[0].score, state.players[1].score],
        "wdl": outcome_value(winner, root_player),
        "score_margin": state.players[root_player].score - state.players[1 - root_player].score,
    }


def _run_or_load_training(
    config: RepeatedReturnValueABConfig, collection: dict[str, Any]
) -> dict[str, Any]:
    output = config.output_dir / "training"
    manifest = output / "training.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        _validate_training(payload, config.protocol_sha256)
        return payload
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in collection["training_development_paths"]]
    if len(records) != 240 or any(record["identity"]["split"] == "protected" for record in records):
        raise ValueError("131 training attempted to access an invalid state set.")
    fitting = [record for record in records if record["identity"]["split"] == "fitting"]
    development = [record for record in records if record["identity"]["split"] == "development"]
    if len(fitting) != 192 or len(development) != 48:
        raise ValueError("131 fitting/development split mismatch.")
    arms = {}
    for arm, target_key in (
        ("single_outcome_control", "single_outcome_target"),
        ("repeated_mean_treatment", "repeated_mean_target"),
    ):
        checkpoint = output / f"{arm}.pt"
        arms[arm] = _train_value_arm(
            parent_checkpoint=config.generation_11_checkpoint,
            fitting=fitting,
            development=development,
            target_key=target_key,
            arm=arm,
            output_checkpoint=checkpoint,
            device=config.device,
            experiment_number=config.experiment_number,
        )
    payload = {
        "protocol_sha256": config.protocol_sha256,
        "protected_paths_accessed": False,
        "single_outcome_control": arms["single_outcome_control"],
        "repeated_mean_treatment": arms["repeated_mean_treatment"],
    }
    _validate_training(payload, config.protocol_sha256)
    _write_json(manifest, payload)
    return payload


def _train_value_arm(
    *, parent_checkpoint: Path, fitting: list[dict[str, Any]], development: list[dict[str, Any]],
    target_key: str, arm: str, output_checkpoint: Path, device: str,
    experiment_number: int = 131,
) -> dict[str, Any]:
    torch.manual_seed(TRAINING_SEED)
    model, parent = load_checkpoint(parent_checkpoint, device=device)
    parent_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    scope = value_only_scope(model)
    development_targets = [float(record["repeated_mean_target"]) for record in development]
    epoch_zero = _value_metrics(model, development, development_targets, device)
    best_loss = epoch_zero["mse"]
    best_epoch = 0
    best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    history = []
    stale = 0
    for epoch in range(1, 51):
        model.train()
        order = list(range(len(fitting)))
        random.Random(TRAINING_SEED + epoch).shuffle(order)
        losses = []
        for start in range(0, len(order), 64):
            batch_records = [fitting[index] for index in order[start : start + 64]]
            targets = torch.tensor(
                [float(record[target_key]) for record in batch_records],
                dtype=torch.float32,
                device=device,
            )
            encoded = _collate_records(batch_records, device)
            optimizer.zero_grad(set_to_none=True)
            _logits, values = model(**encoded)
            loss = torch.mean((values - targets) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        development_metrics = _value_metrics(model, development, development_targets, device)
        history.append({
            "epoch": epoch,
            "train_mse": statistics.fmean(losses),
            "development_repeated_mean": development_metrics,
        })
        if development_metrics["mse"] < best_loss - 1e-4:
            best_loss = development_metrics["mse"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best_state)
    frozen_exact = all(
        torch.equal(tensor.detach().cpu(), parent_state[name])
        for name, tensor in model.state_dict().items()
        if not name.startswith("value_head.")
    )
    save_checkpoint(
        output_checkpoint,
        model=model,
        generation=12,
        metadata={
            "experiment": experiment_number,
            "arm": arm,
            "parent_checkpoint": str(parent_checkpoint),
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "selected_epoch": best_epoch,
            "target": target_key,
            "value_only": True,
        },
    )
    reloaded, identity = load_checkpoint(output_checkpoint, device=device)
    roundtrip_exact = all(
        torch.equal(tensor.detach().cpu(), reloaded.state_dict()[name].detach().cpu())
        for name, tensor in model.state_dict().items()
    )
    if not frozen_exact or not roundtrip_exact or identity.generation != 12:
        raise RuntimeError("131 value-arm identity or checkpoint round-trip failed.")
    return {
        "arm": arm,
        "target": target_key,
        "scope": scope,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "epoch_zero_development": epoch_zero,
        "selected_development": _value_metrics(model, development, development_targets, device),
        "history": history,
        "frozen_parameters_exact": frozen_exact,
        "checkpoint_roundtrip_exact": roundtrip_exact,
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": _sha256(output_checkpoint),
    }


def _collate_records(records: list[dict[str, Any]], device: str) -> dict[str, torch.Tensor]:
    items = []
    env = SplendorEnv()
    for record in records:
        state = deserialize_state_snapshot(record["state_snapshot"])
        items.append((state, env.legal_actions(state)))
    return collate_entity_action_batch(items, device=torch.device(device))


def _value_metrics(
    model: nn.Module, records: list[dict[str, Any]], targets: list[float], device: str
) -> dict[str, float]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), 64):
            encoded = _collate_records(records[start : start + 64], device)
            _logits, values = model(**encoded)
            predictions.extend(float(value) for value in values.detach().cpu().tolist())
    return _prediction_metrics(predictions, targets)


def _prediction_metrics(predictions: list[float], targets: list[float]) -> dict[str, float]:
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Prediction metric lengths do not match.")
    errors = [prediction - target for prediction, target in zip(predictions, targets, strict=True)]
    mean_prediction = statistics.fmean(predictions)
    mean_target = statistics.fmean(targets)
    numerator = sum(
        (prediction - mean_prediction) * (target - mean_target)
        for prediction, target in zip(predictions, targets, strict=True)
    )
    denominator = math.sqrt(
        sum((prediction - mean_prediction) ** 2 for prediction in predictions)
        * sum((target - mean_target) ** 2 for target in targets)
    )
    return {
        "mse": statistics.fmean(error * error for error in errors),
        "mae": statistics.fmean(abs(error) for error in errors),
        "pearson": numerator / denominator if denominator > 0.0 else 0.0,
    }


def _run_or_load_protected(
    config: RepeatedReturnValueABConfig, collection: dict[str, Any], training: dict[str, Any]
) -> dict[str, Any]:
    output = config.output_dir / "protected"
    manifest = output / "protected.stage.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != config.protocol_sha256:
            raise ValueError("131 protected result protocol mismatch.")
        return payload
    if training.get("protected_paths_accessed") is not False:
        raise ValueError("131 protected data was accessed before arm selection.")
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in collection["protected_paths"]]
    if len(records) != 60 or any(record["identity"]["split"] != "protected" for record in records):
        raise ValueError("131 protected state identity mismatch.")
    targets = [float(record["repeated_mean_target"]) for record in records]
    checkpoint_paths = {
        "generation_11": config.generation_11_checkpoint,
        "single_outcome_control": Path(training["single_outcome_control"]["checkpoint"]),
        "repeated_mean_treatment": Path(training["repeated_mean_treatment"]["checkpoint"]),
    }
    predictions: dict[str, list[float]] = {}
    policy_checks = {}
    parent_inference = AlphaZeroInference(config.generation_11_checkpoint, device=config.device)
    for name, path in checkpoint_paths.items():
        inference = AlphaZeroInference(path, device=config.device)
        values = []
        exact_policy = True
        env = SplendorEnv()
        for record in records:
            state = deserialize_state_snapshot(record["state_snapshot"])
            actions = list(actions_by_canonical_successor_key(state, env.legal_actions(state)).values())
            prediction = inference.predict_one(state, actions)
            parent_prediction = parent_inference.predict_one(state, actions)
            values.append(prediction.value)
            exact_policy = exact_policy and prediction.logits == parent_prediction.logits
        predictions[name] = values
        policy_checks[name] = exact_policy
    metrics = {
        name: {
            **_prediction_metrics(values, targets),
            "phase_breakdown": _metric_breakdown(values, targets, records, "phase"),
            "source_breakdown": _metric_breakdown(values, targets, records, "source_generation"),
        }
        for name, values in predictions.items()
    }
    bootstrap = _paired_error_bootstrap(predictions, targets)
    integrity_checks = {
        "exact_protected_states": len(records) == 60,
        "exact_protected_continuations": sum(len(record["continuations"]) for record in records) == 960,
        "all_protected_continuations_valid": all(
            item["termination_reason"] in config.valid_termination_reasons
            for record in records
            for item in record["continuations"]
        ),
        "all_policy_logits_exact": all(policy_checks.values()),
        "single_control_frozen_exact": training["single_outcome_control"]["frozen_parameters_exact"],
        "treatment_frozen_exact": training["repeated_mean_treatment"]["frozen_parameters_exact"],
        "single_control_roundtrip_exact": training["single_outcome_control"]["checkpoint_roundtrip_exact"],
        "treatment_roundtrip_exact": training["repeated_mean_treatment"]["checkpoint_roundtrip_exact"],
    }
    integrity = {"passed": all(integrity_checks.values()), "checks": integrity_checks}
    training_summary = {
        "single_outcome_control": training["single_outcome_control"],
        "repeated_mean_treatment": training["repeated_mean_treatment"],
    }
    qualification = protected_qualification(
        training=training_summary, metrics=metrics, bootstrap=bootstrap, integrity=integrity
    )
    payload = {
        "protocol_sha256": config.protocol_sha256,
        "states": len(records),
        "continuations": sum(len(record["continuations"]) for record in records),
        "target_mean": statistics.fmean(targets),
        "target_variance": statistics.pvariance(targets),
        "metrics": metrics,
        "bootstrap": bootstrap,
        "policy_logit_exact": policy_checks,
        "integrity": integrity,
        "qualification": qualification,
    }
    _write_json(manifest, payload)
    return payload


def _metric_breakdown(
    predictions: list[float], targets: list[float], records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float]]:
    values = sorted({str(record["identity"][key]) for record in records})
    result = {}
    for value in values:
        indices = [index for index, record in enumerate(records) if str(record["identity"][key]) == value]
        result[value] = _prediction_metrics(
            [predictions[index] for index in indices], [targets[index] for index in indices]
        )
    return result


def _paired_error_bootstrap(
    predictions: dict[str, list[float]], targets: list[float]
) -> dict[str, Any]:
    treatment = predictions["repeated_mean_treatment"]
    control = predictions["single_outcome_control"]
    parent = predictions["generation_11"]
    treatment_control = [
        (treatment[index] - targets[index]) ** 2 - (control[index] - targets[index]) ** 2
        for index in range(len(targets))
    ]
    treatment_parent = [
        (treatment[index] - targets[index]) ** 2 - (parent[index] - targets[index]) ** 2
        for index in range(len(targets))
    ]
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "seed": 1_131_200,
        "treatment_minus_control_mean_squared_error": statistics.fmean(treatment_control),
        "treatment_minus_control_squared_error_ci95": _bootstrap_mean_ci(
            treatment_control, BOOTSTRAP_SAMPLES, 1_131_200
        ),
        "treatment_minus_parent_mean_squared_error": statistics.fmean(treatment_parent),
        "treatment_minus_parent_squared_error_ci95": _bootstrap_mean_ci(
            treatment_parent, BOOTSTRAP_SAMPLES, 1_131_201
        ),
    }


def _bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[min(int(0.975 * samples), samples - 1)]]


def _run_or_load_arena(
    config: RepeatedReturnValueABConfig, *, name: str, candidate: Path, baseline: Path,
    seed_start: int
) -> dict[str, Any]:
    output = config.output_dir / "arenas" / name
    manifest = output / "panel.stage.json"
    candidate_hash = _sha256(candidate)
    baseline_hash = _sha256(baseline)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != config.protocol_sha256
            or payload.get("candidate_checkpoint_sha256") != candidate_hash
            or payload.get("baseline_checkpoint_sha256") != baseline_hash
        ):
            raise ValueError("131 arena identity mismatch.")
        shards = {int(key): value for key, value in payload["shards"].items()}
    else:
        tasks = [{
            "name": name,
            "shard": shard,
            "output_dir": str(output),
            "candidate": str(candidate),
            "candidate_hash": candidate_hash,
            "baseline": str(baseline),
            "baseline_hash": baseline_hash,
            "seed_start": seed_start + shard * (PAIRED_SEEDS // 4),
            "device": config.device,
            "max_turns": config.max_turns,
            "protocol_sha256": config.protocol_sha256,
        } for shard in range(4)]
        shards = {}
        with ProcessPoolExecutor(max_workers=config.arena_workers) as executor:
            futures = {executor.submit(_arena_worker, task): task for task in tasks}
            for future in as_completed(futures):
                shard, arena = future.result()
                shards[shard] = arena
        _write_json(manifest, {
            "protocol_sha256": config.protocol_sha256,
            "candidate_checkpoint_sha256": candidate_hash,
            "baseline_checkpoint_sha256": baseline_hash,
            "shards": {str(index): shards[index] for index in range(4)},
        })
    arena = _combine_arena_shards(name, seed_start, shards)
    bootstrap = paired_score_share_bootstrap(arena, samples=BOOTSTRAP_SAMPLES, seed=seed_start + 9_999)
    return {
        "arena": arena,
        "paired_bootstrap": bootstrap,
        "assessment": repeated_gameplay_assessment(
            arena,
            bootstrap,
            config.valid_termination_reasons,
        ),
    }


def _arena_worker(task: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    torch.set_num_threads(1)
    shard = int(task["shard"])
    output = Path(task["output_dir"])
    identity = {
        "protocol_sha256": str(task["protocol_sha256"]),
        "name": task["name"],
        "shard": shard,
        "candidate_checkpoint_sha256": task["candidate_hash"],
        "baseline_checkpoint_sha256": task["baseline_hash"],
        "seed_start": int(task["seed_start"]),
        "paired_seeds": PAIRED_SEEDS // 4,
        "search": asdict(search_config()),
        "max_turns": int(task["max_turns"]),
    }
    identity_path = output / f"shard_{shard}.identity.json"
    progress_path = output / f"shard_{shard}.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("131 arena shard identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError("131 arena progress lacks identity.")
        _write_json(identity_path, identity)
    arena = _arena_dict(run_paired_arena(
        name=f"{task['name']}_shard_{shard}",
        candidate_factory=alpha_zero_search_factory(task["candidate"], task["device"], search_config()),
        baseline_factory=alpha_zero_search_factory(task["baseline"], task["device"], search_config()),
        seed_start=int(task["seed_start"]),
        paired_seeds=PAIRED_SEEDS // 4,
        max_turns=int(task["max_turns"]),
        progress_path=progress_path,
        resume=True,
    ))
    return shard, arena


def _combine_arena_shards(name: str, seed_start: int, shards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if set(shards) != set(range(4)):
        raise ValueError("131 arena requires four shards.")
    rows = []
    for shard in range(4):
        shard_start = seed_start + shard * (PAIRED_SEEDS // 4)
        expected = [(shard_start + offset, seat) for offset in range(PAIRED_SEEDS // 4) for seat in (0, 1)]
        shard_rows = list(shards[shard]["rows"])
        if [(int(row["seed"]), int(row["candidate_seat"])) for row in shard_rows] != expected:
            raise ValueError("131 arena shard schedule mismatch.")
        rows.extend(shard_rows)
    wins = sum(float(row["candidate_points"]) == 1.0 for row in rows)
    losses = sum(float(row["candidate_points"]) == 0.0 for row in rows)
    draws = sum(float(row["candidate_points"]) == 0.5 for row in rows)
    arena = ArenaResult(
        name=name,
        paired_seeds=PAIRED_SEEDS,
        games=len(rows),
        candidate_wins=wins,
        candidate_losses=losses,
        draws=draws,
        score_share=(wins + 0.5 * draws) / len(rows),
        mean_margin=statistics.fmean(float(row["candidate_margin"]) for row in rows),
        termination_reasons=dict(sorted(Counter(str(row["termination_reason"]) for row in rows).items())),
        rows=tuple(rows),
    )
    return _arena_dict(arena)


def _validate_collection(payload: dict[str, Any], protocol_sha256: str = PROTOCOL_SHA256) -> None:
    if (
        payload.get("protocol_sha256") != protocol_sha256
        or payload.get("checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("states") != 300
        or payload.get("continuations") != 2_880
        or len(payload.get("training_development_paths", [])) != 240
        or len(payload.get("protected_paths", [])) != 60
    ):
        raise ValueError("131 collection manifest mismatch.")


def _validate_collected_state(
    payload: dict[str, Any],
    row: dict[str, Any],
    protocol_sha256: str = PROTOCOL_SHA256,
) -> None:
    expected = PROTECTED_POLICY_COUNT if row["identity"]["split"] == "protected" else TRAIN_POLICY_COUNT
    if (
        payload.get("protocol_sha256") != protocol_sha256
        or payload.get("checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("identity") != row["identity"]
        or len(payload.get("continuations", [])) != WORLD_COUNT * expected
    ):
        raise ValueError("131 collected state identity mismatch.")


def _validate_training(payload: dict[str, Any], protocol_sha256: str = PROTOCOL_SHA256) -> None:
    if payload.get("protocol_sha256") != protocol_sha256 or payload.get("protected_paths_accessed") is not False:
        raise ValueError("131 training manifest mismatch.")
    for arm in ("single_outcome_control", "repeated_mean_treatment"):
        record = payload.get(arm, {})
        if (
            record.get("scope", {}).get("trainable_parameters") != 4_705
            or record.get("frozen_parameters_exact") is not True
            or record.get("checkpoint_roundtrip_exact") is not True
            or record.get("checkpoint_sha256") != _sha256(record.get("checkpoint", ""))
        ):
            raise ValueError(f"131 {arm} training identity mismatch.")


def _write_progress(
    config: RepeatedReturnValueABConfig, result: dict[str, Any], started: float
) -> None:
    payload = {**result, "elapsed_seconds": time.perf_counter() - started}
    _write_json(config.output_dir / "progress.json", payload)


def _seal(
    config: RepeatedReturnValueABConfig, result: dict[str, Any], started: float, decision: str
) -> dict[str, Any]:
    result.update({
        "status": "complete",
        "active_stage": "complete",
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "installed_checkpoint_unchanged": _sha256(config.installed_checkpoint) == EXPECTED_HASHES["installed_checkpoint"],
    })
    _write_json(config.output_dir / "result.json", result)
    _write_json(config.output_dir / "progress.json", result)
    return result
