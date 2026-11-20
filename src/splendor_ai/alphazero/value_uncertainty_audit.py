"""Repeated-continuation value-uncertainty audit for programme 130."""

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
from typing import Any

import torch

from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import (
    actions_by_canonical_successor_key,
    canonical_public_belief_key,
    canonical_successor_action_key,
    encode_canonical_state_entities,
)
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE, SplendorState
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot

from .historical_state_reanalysis import (
    _iter_source_games,
    _reconstruct_start,
    phase_for_slot,
)
from .network import AlphaZeroInference, load_checkpoint
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig


@dataclass(frozen=True, slots=True)
class ValueUncertaintyAuditConfig:
    output_dir: Path
    result_129: Path
    generation_11_checkpoint: Path
    terminal_control_checkpoint: Path
    ordinary_holdout: Path
    installed_checkpoint: Path
    device: str = "cpu"
    workers: int = 4
    torch_threads: int = 1
    max_turns: int = 200


PROTOCOL_SHA256 = "74f580d8f47a5ade2eab74676cafa04aa94b5ce448048f30d967b2c0912ee7d8"
EXPECTED_HASHES = {
    "result_129": "75cf5857ae833ba1c5609002b6d93f76660fadd3be20d2ecc98c30146c20f4fa",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "terminal_control_checkpoint": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "ordinary_holdout": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}
STATE_COUNT = 24
WORLD_COUNT = 4
POLICY_COUNT = 4
OPERATIVE_SIMULATIONS = 256
DEEP_SIMULATIONS = 1_024
WORLD_SEED_BASE = 113_000_000
POLICY_SEED_BASE = 113_100_000
BOOTSTRAP_SAMPLES = 50_000


def operative_search_config(*, simulations: int = OPERATIVE_SIMULATIONS) -> SearchConfig:
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


def value_uncertainty_decision(
    *, integrity: bool, material_noise: bool, possible_world_primary: bool
) -> str:
    if not integrity:
        return "value_uncertainty_integrity_failed_no_model_conclusion"
    if not material_noise:
        return "scalar_terminal_value_uncertainty_not_primary_close_compact_alphazero"
    if possible_world_primary:
        return "possible_world_value_distribution_material_freeze_distributional_pilot"
    return "trajectory_policy_value_uncertainty_material_review_joint_target"


def two_way_sum_squares(matrix: list[list[float]]) -> dict[str, float]:
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Two-way decomposition requires a non-empty rectangular matrix.")
    worlds = len(matrix)
    policies = len(matrix[0])
    flat = [float(value) for row in matrix for value in row]
    grand = statistics.fmean(flat)
    row_means = [statistics.fmean(row) for row in matrix]
    column_means = [statistics.fmean(matrix[row][column] for row in range(worlds)) for column in range(policies)]
    total = sum((value - grand) ** 2 for value in flat)
    world = policies * sum((value - grand) ** 2 for value in row_means)
    policy = worlds * sum((value - grand) ** 2 for value in column_means)
    interaction = max(0.0, total - world - policy)
    return {
        "total_ss": total,
        "possible_world_ss": world,
        "policy_seed_ss": policy,
        "interaction_ss": interaction,
        "possible_world_fraction": world / total if total > 0.0 else 0.0,
        "policy_seed_fraction": policy / total if total > 0.0 else 0.0,
        "interaction_fraction": interaction / total if total > 0.0 else 0.0,
    }


def target_noise_decomposition(outcome_groups: list[list[float]]) -> dict[str, float]:
    if not outcome_groups or any(not group for group in outcome_groups):
        raise ValueError("Target-noise decomposition requires non-empty state groups.")
    flat = [float(value) for group in outcome_groups for value in group]
    grand = statistics.fmean(flat)
    within = sum(
        sum((float(value) - statistics.fmean(group)) ** 2 for value in group)
        for group in outcome_groups
    )
    between = sum(
        len(group) * (statistics.fmean(group) - grand) ** 2 for group in outcome_groups
    )
    total = within + between
    return {
        "within_state_ss": within,
        "between_state_ss": between,
        "total_ss": total,
        "within_state_target_noise_fraction": within / total if total > 0.0 else 0.0,
    }


def bootstrap_uncertainty(
    outcome_groups: list[list[float]], *, samples: int = BOOTSTRAP_SAMPLES, seed: int = 113_200_000
) -> dict[str, list[float]]:
    if samples <= 0 or not outcome_groups:
        raise ValueError("Bootstrap requires states and positive samples.")
    rng = random.Random(seed)
    variance_estimates: list[float] = []
    noise_estimates: list[float] = []
    for _ in range(samples):
        selected = [outcome_groups[rng.randrange(len(outcome_groups))] for _ in outcome_groups]
        variance_estimates.append(statistics.fmean(statistics.pvariance(group) for group in selected))
        noise_estimates.append(
            target_noise_decomposition(selected)["within_state_target_noise_fraction"]
        )
    variance_estimates.sort()
    noise_estimates.sort()
    low = int(0.025 * samples)
    high = min(int(0.975 * samples), samples - 1)
    return {
        "mean_within_state_wdl_variance_ci95": [variance_estimates[low], variance_estimates[high]],
        "within_state_target_noise_fraction_ci95": [noise_estimates[low], noise_estimates[high]],
    }


def run_value_uncertainty_audit_130(
    config: ValueUncertaintyAuditConfig, *, launch: bool = False
) -> dict[str, Any]:
    started = time.perf_counter()
    states, preflight = _preflight(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config.output_dir / "preflight.json", preflight)
    if not launch:
        return preflight
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") == "complete" and result.get("protocol_sha256") == PROTOCOL_SHA256:
            return result
        raise ValueError("Existing 130 result is not a valid completed result.")
    progress = {
        "kind": "alphazero_value_uncertainty_audit_130",
        "status": "running",
        "active_stage": "crossed_repeated_continuations",
        "protocol_sha256": PROTOCOL_SHA256,
        "completed_states": 0,
        "total_states": STATE_COUNT,
        "training": False,
        "optimizer": False,
        "gameplay_strength_arena": False,
        "promotion_authorized": False,
        "automatic_continuation": False,
    }
    _write_json(config.output_dir / "progress.json", progress)
    rows = _run_or_load_panel(config, states, progress)
    integrity = _integrity(rows)
    if not integrity["passed"]:
        result = {
            "kind": "alphazero_value_uncertainty_audit_130",
            "status": "complete",
            "active_stage": "complete",
            "protocol_sha256": PROTOCOL_SHA256,
            "input_hashes": preflight["input_hashes"],
            "states": rows,
            "integrity": integrity,
            "summary": None,
            "assessment": None,
            "training": False,
            "optimizer": False,
            "gameplay_strength_arena": False,
            "installed_checkpoint_unchanged": _sha256(config.installed_checkpoint) == EXPECTED_HASHES["installed_checkpoint"],
            "promotion_authorized": False,
            "automatic_continuation": False,
            "decision": value_uncertainty_decision(
                integrity=False, material_noise=False, possible_world_primary=False
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(result_path, result)
        _write_json(config.output_dir / "progress.json", result)
        return result
    summary = _summarize_panel(rows)
    material_checks = {
        "mean_within_state_wdl_variance_at_least_0_50": summary["mean_within_state_wdl_variance"] >= 0.50,
        "mixed_win_loss_fraction_at_least_0_50": summary["mixed_win_loss_fraction"] >= 0.50,
        "within_state_target_noise_fraction_at_least_0_65": summary["target_noise"]["within_state_target_noise_fraction"] >= 0.65,
        "variance_ci95_low_at_least_0_40": summary["bootstrap"]["mean_within_state_wdl_variance_ci95"][0] >= 0.40,
        "noise_fraction_ci95_low_at_least_0_55": summary["bootstrap"]["within_state_target_noise_fraction_ci95"][0] >= 0.55,
    }
    material_noise = all(material_checks.values())
    world_checks = {
        "possible_world_fraction_at_least_0_40": summary["two_way"]["possible_world_fraction"] >= 0.40,
        "possible_world_exceeds_policy_by_0_10": summary["two_way"]["possible_world_fraction"] >= summary["two_way"]["policy_seed_fraction"] + 0.10,
    }
    possible_world_primary = all(world_checks.values())
    decision = value_uncertainty_decision(
        integrity=integrity["passed"],
        material_noise=material_noise,
        possible_world_primary=possible_world_primary,
    )
    result = {
        "kind": "alphazero_value_uncertainty_audit_130",
        "status": "complete",
        "active_stage": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": preflight["input_hashes"],
        "states": rows,
        "integrity": integrity,
        "summary": summary,
        "assessment": {
            "material_terminal_target_noise": material_noise,
            "material_checks": material_checks,
            "possible_world_primary": possible_world_primary,
            "possible_world_checks": world_checks,
        },
        "training": False,
        "optimizer": False,
        "gameplay_strength_arena": False,
        "installed_checkpoint_unchanged": _sha256(config.installed_checkpoint) == EXPECTED_HASHES["installed_checkpoint"],
        "promotion_authorized": False,
        "automatic_continuation": False,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    _write_json(config.output_dir / "progress.json", result)
    return result


def _preflight(config: ValueUncertaintyAuditConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = {
        "result_129": _sha256(config.result_129),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "terminal_control_checkpoint": _sha256(config.terminal_control_checkpoint),
        "ordinary_holdout": _sha256(config.ordinary_holdout),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"130 frozen input mismatch: {observed}")
    prior = json.loads(config.result_129.read_text(encoding="utf-8"))
    if (
        prior.get("status") != "complete"
        or prior.get("decision") != "historical_reanalysis_failed_diverse_state_gameplay_no_install"
        or prior.get("installed_checkpoint_unchanged") is not True
    ):
        raise ValueError("130 requires the exact completed 129 failure.")
    _model, generation_11 = load_checkpoint(config.generation_11_checkpoint, device=config.device)
    _control_model, terminal_control = load_checkpoint(config.terminal_control_checkpoint, device=config.device)
    if generation_11.generation != 11 or terminal_control.generation != 12:
        raise ValueError("130 checkpoint generation mismatch.")
    states = select_audit_states(config.ordinary_holdout)
    generation_inference = AlphaZeroInference(config.generation_11_checkpoint, device=config.device)
    control_inference = AlphaZeroInference(config.terminal_control_checkpoint, device=config.device)
    invariant_summaries = [
        public_world_invariants(
            deserialize_state_snapshot(row["state_snapshot"]),
            state_index=int(row["identity"]["state_index"]),
            generation_inference=generation_inference,
            control_inference=control_inference,
        )
        for row in states
    ]
    temporary = AlphaZeroSearch(generation_inference, operative_search_config(simulations=8)).run(
        SplendorEnv(),
        deserialize_state_snapshot(states[0]["state_snapshot"]),
        self_play=False,
        seed=POLICY_SEED_BASE,
    )
    return states, {
        "kind": "alphazero_value_uncertainty_audit_130_preflight",
        "status": "preflight_passed",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": observed,
        "state_count": len(states),
        "phase_counts": dict(sorted(Counter(row["identity"]["phase"] for row in states).items())),
        "possible_worlds_per_state": WORLD_COUNT,
        "policy_seeds_per_state": POLICY_COUNT,
        "continuations_if_launched": STATE_COUNT * WORLD_COUNT * POLICY_COUNT,
        "world_invariants": invariant_summaries,
        "temporary_root_search": {
            "simulations": temporary.simulations,
            "root_edges": len(temporary.root_edges),
            "persisted_as_evidence": False,
        },
        "no_optimizer_instantiated": True,
        "no_training": True,
        "no_continuation_played": True,
        "no_strength_gameplay": True,
    }


def select_audit_states(path: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    total_games = 0
    for ordinal, game_rows in enumerate(_iter_source_games(path)):
        total_games += 1
        if any(
            row.generation != 11
            or row.checkpoint_generation != 11
            or row.termination_reason != "completed"
            for row in game_rows
        ):
            raise ValueError("130 ordinary holdout identity mismatch.")
        if ordinal >= STATE_COUNT:
            continue
        phase = phase_for_slot(ordinal)
        _env, state, source = _reconstruct_start(game_rows, phase, 11, 0, ordinal)
        identity = {
            "state_index": ordinal,
            "source_game_ordinal": ordinal,
            "source_game_seed": source["source_game_seed"],
            "source_turn_index": source["source_turn_index"],
            "root_player": source["source_player"],
            "phase": phase,
            "source_progress": source["source_progress"],
            "pending_round_end": source["pending_round_end"],
            "opponent_blind_reserves": _opponent_blind_reserves(state),
        }
        selected.append({"identity": identity, "state_snapshot": serialize_state_snapshot(state)})
    if total_games != 32 or len(selected) != STATE_COUNT:
        raise ValueError("130 requires the exact 32-game holdout and first 24 states.")
    if Counter(row["identity"]["phase"] for row in selected) != Counter({"early": 8, "middle": 8, "late": 8}):
        raise ValueError("130 phase schedule mismatch.")
    return selected


def public_world_invariants(
    state: SplendorState,
    *,
    state_index: int,
    generation_inference: AlphaZeroInference,
    control_inference: AlphaZeroInference,
) -> dict[str, Any]:
    worlds = _possible_worlds(state, state_index)
    keys = [canonical_public_belief_key(world, state.current_player) for world in worlds]
    action_maps = [actions_by_canonical_successor_key(world, SplendorEnv().legal_actions(world)) for world in worlds]
    semantic_keys = [tuple(sorted(mapping, key=repr)) for mapping in action_maps]
    encodings = [encode_canonical_state_entities(world, state.current_player) for world in worlds]
    generation_predictions = [
        generation_inference.predict_one(world, [action_maps[index][key] for key in semantic_keys[0]])
        for index, world in enumerate(worlds)
    ]
    control_predictions = [
        control_inference.predict_one(world, [action_maps[index][key] for key in semantic_keys[0]])
        for index, world in enumerate(worlds)
    ]
    checks = {
        "four_unique_full_worlds": len({state_signature(world) for world in worlds}) == WORLD_COUNT,
        "public_belief_key_invariant": all(key == keys[0] for key in keys),
        "semantic_actions_invariant": all(key == semantic_keys[0] for key in semantic_keys),
        "canonical_entity_tensor_invariant": all(encoding == encodings[0] for encoding in encodings),
        "generation_11_prediction_invariant": all(prediction == generation_predictions[0] for prediction in generation_predictions),
        "terminal_control_prediction_invariant": all(prediction == control_predictions[0] for prediction in control_predictions),
    }
    if not all(checks.values()):
        raise ValueError(f"130 possible-world invariant failed for state {state_index}: {checks}")
    return {
        "state_index": state_index,
        "root_actions": len(semantic_keys[0]),
        "generation_11_raw_value": generation_predictions[0].value,
        "terminal_control_raw_value": control_predictions[0].value,
        "checks": checks,
    }


def _possible_worlds(state: SplendorState, state_index: int) -> list[SplendorState]:
    return [
        determinize_for_observer(
            state,
            state.current_player,
            random.Random(WORLD_SEED_BASE + state_index * 10_003 + world_slot),
        )
        for world_slot in range(WORLD_COUNT)
    ]


def _run_or_load_panel(
    config: ValueUncertaintyAuditConfig,
    states: list[dict[str, Any]],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    state_dir = config.output_dir / "states"
    state_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[int, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    for row in states:
        index = int(row["identity"]["state_index"])
        path = state_dir / f"state_{index:03d}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_state_result(payload, row)
            completed[index] = payload
        else:
            tasks.append({
                "protocol_sha256": PROTOCOL_SHA256,
                "identity": row["identity"],
                "state_snapshot": row["state_snapshot"],
                "generation_11_checkpoint": str(config.generation_11_checkpoint),
                "terminal_control_checkpoint": str(config.terminal_control_checkpoint),
                "device": config.device,
                "torch_threads": config.torch_threads,
                "max_turns": config.max_turns,
                "output_path": str(path),
            })
    progress["completed_states"] = len(completed)
    _write_json(config.output_dir / "progress.json", progress)
    if tasks:
        with ProcessPoolExecutor(max_workers=min(config.workers, len(tasks))) as executor:
            futures = {executor.submit(_audit_state_worker, task): task for task in tasks}
            for future in as_completed(futures):
                payload = future.result()
                index = int(payload["identity"]["state_index"])
                completed[index] = payload
                progress["completed_states"] = len(completed)
                progress["last_completed_state"] = index
                _write_json(config.output_dir / "progress.json", progress)
                print(f"130 completed states {len(completed)}/{STATE_COUNT}", flush=True)
    if set(completed) != set(range(STATE_COUNT)):
        raise RuntimeError("130 did not complete the exact 24-state panel.")
    return [completed[index] for index in range(STATE_COUNT)]


def _audit_state_worker(task: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(task["torch_threads"]))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    state = deserialize_state_snapshot(task["state_snapshot"])
    identity = task["identity"]
    state_index = int(identity["state_index"])
    generation_inference = AlphaZeroInference(task["generation_11_checkpoint"], device=task["device"])
    control_inference = AlphaZeroInference(task["terminal_control_checkpoint"], device=task["device"])
    invariants = public_world_invariants(
        state,
        state_index=state_index,
        generation_inference=generation_inference,
        control_inference=control_inference,
    )
    env = SplendorEnv(seed=int(identity["source_game_seed"]))
    root_results: dict[str, list[dict[str, Any]]] = {"256": [], "1024": []}
    operative_actions: list[tuple[object, ...]] = []
    for simulations in (OPERATIVE_SIMULATIONS, DEEP_SIMULATIONS):
        search = AlphaZeroSearch(generation_inference, operative_search_config(simulations=simulations))
        for policy_slot in range(POLICY_COUNT):
            seed = POLICY_SEED_BASE + state_index * 10_003 + policy_slot
            result = search.run(env, state, self_play=False, seed=seed)
            action_key = canonical_successor_action_key(state, result.action)
            if simulations == OPERATIVE_SIMULATIONS:
                operative_actions.append(action_key)
            root_results[str(simulations)].append({
                "policy_slot": policy_slot,
                "seed": seed,
                "raw_network_value": result.raw_network_value,
                "search_root_value": result.search_root_value,
                "selected_action_key": repr(action_key),
                "visit_distribution": [
                    {
                        "action_key": repr(edge.action_key),
                        "visits": edge.visits,
                        "probability": edge.probability,
                        "q": edge.q,
                    }
                    for edge in result.root_edges
                ],
            })
    worlds = _possible_worlds(state, state_index)
    continuations: list[dict[str, Any]] = []
    for world_slot, world in enumerate(worlds):
        for policy_slot, first_action_key in enumerate(operative_actions):
            continuations.append(
                _play_crossed_continuation(
                    initial_state=world,
                    root_player=state.current_player,
                    first_action_key=first_action_key,
                    inference=generation_inference,
                    state_index=state_index,
                    world_slot=world_slot,
                    policy_slot=policy_slot,
                    max_turns=int(task["max_turns"]),
                )
            )
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "generation_11_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "terminal_control_checkpoint_sha256": EXPECTED_HASHES["terminal_control_checkpoint"],
        "identity": identity,
        "invariants": invariants,
        "root_results": root_results,
        "continuations": continuations,
    }
    payload["summary"] = _summarize_state(payload)
    _write_json(Path(task["output_path"]), payload)
    return payload


def _play_crossed_continuation(
    *,
    initial_state: SplendorState,
    root_player: int,
    first_action_key: tuple[object, ...],
    inference: AlphaZeroInference,
    state_index: int,
    world_slot: int,
    policy_slot: int,
    max_turns: int,
) -> dict[str, Any]:
    env = SplendorEnv()
    state = initial_state.copy_shallow()
    legal_map = actions_by_canonical_successor_key(state, env.legal_actions(state))
    if first_action_key not in legal_map:
        raise ValueError("130 public root action cannot be rebound in a possible world.")
    action = legal_map[first_action_key]
    visits: dict[tuple[object, ...], int] = {state_signature(state): 1}
    forced_passes = int(action.action_type is ActionType.PASS and len(legal_map) == 1)
    next_state = env.step(state, action)
    no_progress = 0 if is_progress_transition(state, next_state, action) else 1
    state = next_state
    reason = "completed"
    search = AlphaZeroSearch(inference, operative_search_config())
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
    if not state.terminal or reason != "completed":
        return {
            "world_slot": world_slot,
            "policy_slot": policy_slot,
            "termination_reason": reason,
            "turns": state.turn_index,
            "wdl": None,
            "score_margin": None,
        }
    winner = state.winner
    wdl = 0.0 if winner is None else 1.0 if winner == root_player else -1.0
    margin = state.players[root_player].score - state.players[1 - root_player].score
    return {
        "world_slot": world_slot,
        "policy_slot": policy_slot,
        "termination_reason": "completed",
        "turns": state.turn_index,
        "winner": winner,
        "final_scores": [state.players[0].score, state.players[1].score],
        "wdl": wdl,
        "score_margin": margin,
    }


def _summarize_state(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["continuations"]
    if len(rows) != WORLD_COUNT * POLICY_COUNT or any(row["wdl"] is None for row in rows):
        return {"integrity_complete": False}
    outcomes = [float(row["wdl"]) for row in rows]
    margins = [float(row["score_margin"]) for row in rows]
    matrix = [
        [
            float(next(row["wdl"] for row in rows if row["world_slot"] == world and row["policy_slot"] == policy))
            for policy in range(POLICY_COUNT)
        ]
        for world in range(WORLD_COUNT)
    ]
    counts = Counter(outcomes)
    probabilities = [count / len(outcomes) for count in counts.values()]
    entropy = -sum(probability * math.log2(probability) for probability in probabilities if probability > 0.0)
    empirical = statistics.fmean(outcomes)
    root_256 = payload["root_results"]["256"]
    root_1024 = payload["root_results"]["1024"]
    raw = float(root_256[0]["raw_network_value"])
    control = float(payload["invariants"]["terminal_control_raw_value"])
    search_256 = [float(row["search_root_value"]) for row in root_256]
    search_1024 = [float(row["search_root_value"]) for row in root_1024]
    estimates = {
        "generation_11_raw": raw,
        "terminal_control_raw": control,
        "search_256_mean": statistics.fmean(search_256),
        "search_1024_mean": statistics.fmean(search_1024),
    }
    return {
        "integrity_complete": True,
        "wins": counts.get(1.0, 0),
        "draws": counts.get(0.0, 0),
        "losses": counts.get(-1.0, 0),
        "wdl_mean": empirical,
        "wdl_variance": statistics.pvariance(outcomes),
        "wdl_entropy_bits": entropy,
        "both_win_and_loss": counts.get(1.0, 0) > 0 and counts.get(-1.0, 0) > 0,
        "score_margin_mean": statistics.fmean(margins),
        "score_margin_std": math.sqrt(statistics.pvariance(margins)),
        "estimates": {
            name: {
                "value": value,
                "absolute_error_to_empirical_mean": abs(value - empirical),
                "squared_error_to_empirical_mean": (value - empirical) ** 2,
            }
            for name, value in estimates.items()
        },
        "root_repeat": {
            "search_256_std": math.sqrt(statistics.pvariance(search_256)),
            "search_1024_std": math.sqrt(statistics.pvariance(search_1024)),
            "action_256_agreement": _modal_fraction([row["selected_action_key"] for row in root_256]),
            "action_1024_agreement": _modal_fraction([row["selected_action_key"] for row in root_1024]),
        },
        "two_way": two_way_sum_squares(matrix),
    }


def _summarize_panel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [row["summary"] for row in rows]
    outcome_groups = [[float(item["wdl"]) for item in row["continuations"]] for row in rows]
    decomposition = target_noise_decomposition(outcome_groups)
    bootstrap = bootstrap_uncertainty(outcome_groups)
    total_two_way = {
        key: sum(float(summary["two_way"][key]) for summary in summaries)
        for key in ("total_ss", "possible_world_ss", "policy_seed_ss", "interaction_ss")
    }
    total_ss = total_two_way["total_ss"]
    total_two_way.update({
        "possible_world_fraction": total_two_way["possible_world_ss"] / total_ss if total_ss > 0.0 else 0.0,
        "policy_seed_fraction": total_two_way["policy_seed_ss"] / total_ss if total_ss > 0.0 else 0.0,
        "interaction_fraction": total_two_way["interaction_ss"] / total_ss if total_ss > 0.0 else 0.0,
    })
    estimate_names = list(summaries[0]["estimates"])
    estimate_metrics = {
        name: {
            "mae_to_empirical_state_mean": statistics.fmean(summary["estimates"][name]["absolute_error_to_empirical_mean"] for summary in summaries),
            "mse_to_empirical_state_mean": statistics.fmean(summary["estimates"][name]["squared_error_to_empirical_mean"] for summary in summaries),
        }
        for name in estimate_names
    }
    return {
        "states": len(rows),
        "continuations": sum(len(row["continuations"]) for row in rows),
        "mean_within_state_wdl_variance": statistics.fmean(summary["wdl_variance"] for summary in summaries),
        "mean_wdl_entropy_bits": statistics.fmean(summary["wdl_entropy_bits"] for summary in summaries),
        "mixed_win_loss_states": sum(summary["both_win_and_loss"] for summary in summaries),
        "mixed_win_loss_fraction": statistics.fmean(float(summary["both_win_and_loss"]) for summary in summaries),
        "mean_score_margin_std": statistics.fmean(summary["score_margin_std"] for summary in summaries),
        "target_noise": decomposition,
        "two_way": total_two_way,
        "bootstrap": bootstrap,
        "value_estimates": estimate_metrics,
        "root_repeat": {
            "mean_search_256_std": statistics.fmean(summary["root_repeat"]["search_256_std"] for summary in summaries),
            "mean_search_1024_std": statistics.fmean(summary["root_repeat"]["search_1024_std"] for summary in summaries),
            "mean_action_256_agreement": statistics.fmean(summary["root_repeat"]["action_256_agreement"] for summary in summaries),
            "mean_action_1024_agreement": statistics.fmean(summary["root_repeat"]["action_1024_agreement"] for summary in summaries),
        },
        "phase_breakdown": _breakdown(rows, "phase"),
        "opponent_blind_reserve_breakdown": _breakdown(rows, "blind"),
    }


def _breakdown(rows: list[dict[str, Any]], kind: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if kind == "phase":
            key = str(row["identity"]["phase"])
        else:
            key = "present" if int(row["identity"]["opponent_blind_reserves"]) > 0 else "absent"
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "states": len(group),
            "mean_wdl_variance": statistics.fmean(item["summary"]["wdl_variance"] for item in group),
            "mixed_win_loss_fraction": statistics.fmean(float(item["summary"]["both_win_and_loss"]) for item in group),
            "mean_score_margin_std": statistics.fmean(item["summary"]["score_margin_std"] for item in group),
        }
        for key, group in sorted(groups.items())
    }


def _integrity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    continuations = [item for row in rows for item in row["continuations"]]
    checks = {
        "exact_state_count": len(rows) == STATE_COUNT,
        "exact_continuation_count": len(continuations) == STATE_COUNT * WORLD_COUNT * POLICY_COUNT,
        "all_normal_terminations": all(item["termination_reason"] == "completed" for item in continuations),
        "all_state_summaries_complete": all(row["summary"].get("integrity_complete") is True for row in rows),
        "all_world_invariants_passed": all(all(row["invariants"]["checks"].values()) for row in rows),
        "all_values_finite": all(
            math.isfinite(float(item[key]))
            for item in continuations
            for key in ("wdl", "score_margin")
            if item[key] is not None
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _validate_state_result(payload: dict[str, Any], source: dict[str, Any]) -> None:
    if (
        payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("identity") != source["identity"]
        or payload.get("generation_11_checkpoint_sha256") != EXPECTED_HASHES["generation_11_checkpoint"]
        or payload.get("terminal_control_checkpoint_sha256") != EXPECTED_HASHES["terminal_control_checkpoint"]
    ):
        raise ValueError("130 durable state result identity mismatch.")


def _modal_fraction(values: list[str]) -> float:
    return max(Counter(values).values()) / len(values)


def _opponent_blind_reserves(state: SplendorState) -> int:
    opponent = state.players[1 - state.current_player]
    return sum(
        opponent.reserved_card_source(index) != RESERVE_SOURCE_VISIBLE
        for index in range(len(opponent.reserved_cards))
    )
