"""Reachability-weighted start-role operational importance audit 126."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.encoding import (
    canonical_public_belief_key,
    encode_canonical_state_entities,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import serialize_action

from .network import AlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig
from .start_role_audit import _action_key, _limit_worker_threads, _percentile


PROTOCOL_SHA256 = "27d53e38c8dee0665da5eac95bf358a10c94900bf827d2b9fa4fe1ff0e5520c2"
EXPECTED_HASHES = {
    "result_117": "a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283",
    "result_125": "7e11e49ed022bdb81a87a1f9be2e8c1c04a2c8e91be33297d56f7b0e7068c3f5",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "untouched_holdout": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}
GAME_COUNT = 32
OCTILES = 8
STATE_COUNT = GAME_COUNT * OCTILES
BUDGETS = (64, 256)
SHARDS = 4
BOOTSTRAP_SAMPLES = 50_000


@dataclass(frozen=True, slots=True)
class StartRoleOperationalAuditConfig:
    output_dir: Path
    result_117: Path
    result_125: Path
    generation_11_checkpoint: Path
    untouched_holdout: Path
    installed_checkpoint: Path
    device: str = "cpu"
    torch_threads: int = 1
    workers: int = 4


def run_start_role_operational_audit_126(
    config: StartRoleOperationalAuditConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = _verify_inputs(config)
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "complete"
            and payload.get("protocol_sha256") == PROTOCOL_SHA256
        ):
            return payload
        raise ValueError("Existing 126 result is not a valid completed result.")

    rows = read_replay_jsonl(config.untouched_holdout)
    states, reconstruction = reconstruct_operational_panel(rows)
    _write_json(
        config.output_dir / "progress.json",
        {
            "kind": "alphazero_start_role_operational_importance_audit_126",
            "status": "running",
            "active_stage": "coupled_search",
            "protocol_sha256": PROTOCOL_SHA256,
            "input_hashes": input_hashes,
            "reconstruction": reconstruction,
            "training_performed": False,
            "parameter_mutation_performed": False,
            "candidate_checkpoint_written": False,
            "gameplay_arena": None,
            "promotion_authorized": False,
        },
    )
    comparisons = _run_or_load_search(config, states)
    summary = summarize_operational_comparisons(comparisons)
    gate = operational_gate(summary)
    if not gate["integrity_passed"]:
        decision = "start_role_operational_audit_integrity_failed_no_conclusion"
    elif gate["overall_material"]:
        decision = (
            "start_role_operationally_material_prioritize_minimal_representation_repair"
        )
    elif gate["final_two_octiles_material"]:
        decision = (
            "start_role_material_only_near_game_end_deprioritize_general_role_repair"
        )
    else:
        decision = "start_role_not_operationally_material_close_representation_branch"
    result = {
        "kind": "alphazero_start_role_operational_importance_audit_126",
        "status": "complete",
        "active_stage": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": input_hashes,
        "reconstruction": reconstruction,
        "search_comparisons": len(comparisons),
        "summary": summary,
        "operative_gate": gate,
        "decision": decision,
        "training_performed": False,
        "optimizer_instantiated": False,
        "parameter_mutation_performed": False,
        "search_targets_generated_for_learning": False,
        "candidate_checkpoint_written": False,
        "self_play_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "installed_checkpoint_unchanged": (
            _sha256(config.installed_checkpoint)
            == EXPECTED_HASHES["installed_checkpoint"]
        ),
        "promotion_authorized": False,
        "automatic_continuation": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    _write_json(config.output_dir / "progress.json", result)
    return result


def _octile_indices(row_count: int) -> tuple[int, ...]:
    indices = tuple(((2 * index + 1) * row_count) // 16 for index in range(OCTILES))
    if len(set(indices)) != OCTILES or indices[-1] >= row_count:
        raise ValueError("126 game is too short for eight unique octile states.")
    return indices


def reconstruct_operational_panel(
    rows: list[AlphaZeroReplayRow],
) -> tuple[list[tuple[int, SplendorState, dict[str, Any]]], dict[str, Any]]:
    grouped: dict[int, list[AlphaZeroReplayRow]] = {}
    for row in rows:
        grouped.setdefault(row.game_seed, []).append(row)
    if len(grouped) != GAME_COUNT:
        raise ValueError("126 requires exactly 32 untouched games.")
    selected: list[tuple[int, SplendorState, dict[str, Any]]] = []
    game_lengths = {}
    for game_seed in sorted(grouped):
        game_rows = sorted(grouped[game_seed], key=lambda row: row.turn_index)
        env = SplendorEnv(seed=game_seed)
        state = env.initial_state()
        trajectory: list[SplendorState] = []
        for expected_turn, row in enumerate(game_rows):
            if (
                row.turn_index != expected_turn
                or row.player_id != state.current_player
                or state.turn_index != row.turn_index
            ):
                raise ValueError("126 reconstructed turn/player sequence mismatch.")
            encoded = encode_canonical_state_entities(state, state.current_player)
            if (
                encoded.entity_type_ids != row.entity_type_ids
                or encoded.card_ids != row.entity_card_ids
                or encoded.noble_ids != row.entity_noble_ids
                or encoded.numeric_features != row.entity_numeric_features
            ):
                raise ValueError("126 reconstructed canonical tensor mismatch.")
            trajectory.append(state.copy_shallow())
            selected_payload = row.action_payloads[row.selected_action_row]
            matches = [
                action
                for action in env.legal_actions(state)
                if serialize_action(action) == selected_payload
            ]
            if len(matches) != 1:
                raise ValueError("126 selected replay action is not uniquely legal.")
            state = env.step(state, matches[0])
        last = game_rows[-1]
        if (
            not state.terminal
            or last.termination_reason != "completed"
            or state.winner != last.winner
            or tuple(player.score for player in state.players) != last.final_scores
        ):
            raise ValueError("126 reconstructed terminal outcome mismatch.")
        row_count = len(trajectory)
        game_lengths[str(game_seed)] = row_count
        for octile, source_index in enumerate(_octile_indices(row_count), start=1):
            state_item = trajectory[source_index]
            metadata = {
                "octile": octile,
                "source_row_index": source_index,
                "normalized_progress": source_index / max(row_count - 1, 1),
                "turns_remaining_in_recorded_game": row_count - source_index,
                "source_reachable": True,
            }
            selected.append((game_seed, state_item, metadata))
    if len(selected) != STATE_COUNT:
        raise ValueError("126 reconstruction did not yield exactly 256 states.")
    schedule = [
        _state_identity(game_seed, state, metadata)
        for game_seed, state, metadata in selected
    ]
    return selected, {
        "games": len(grouped),
        "source_rows": len(rows),
        "states": len(selected),
        "states_per_game": OCTILES,
        "game_lengths": game_lengths,
        "schedule": schedule,
        "role_counts": {
            str(role).lower(): sum(
                item["observer_is_start_player"] is role for item in schedule
            )
            for role in (False, True)
        },
        "pending_round_end_states": sum(
            bool(item["pending_round_end"]) for item in schedule
        ),
    }


def _run_or_load_search(config, states) -> list[dict[str, Any]]:
    output_dir = config.output_dir / "search"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = STATE_COUNT // SHARDS
    tasks = [
        {
            "shard": shard,
            "panel_start": shard * chunk_size,
            "states": states[shard * chunk_size : (shard + 1) * chunk_size],
            "output_dir": str(output_dir),
            "checkpoint": str(config.generation_11_checkpoint),
            "device": config.device,
        }
        for shard in range(SHARDS)
    ]
    shards: dict[int, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_search_worker, task): task for task in tasks}
        for future in as_completed(futures):
            shard, shard_rows = future.result()
            shards[shard] = shard_rows
    if set(shards) != set(range(SHARDS)):
        raise ValueError("126 requires all four exact search shards.")
    rows = [row for shard in range(SHARDS) for row in shards[shard]]
    expected = [
        (_state_identity(game_seed, state, metadata), budget)
        for game_seed, state, metadata in states
        for budget in BUDGETS
    ]
    observed = [(row["identity"], row["budget"]) for row in rows]
    if observed != expected:
        raise ValueError("126 combined search schedule mismatch.")
    return rows


def _search_worker(task: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    _limit_worker_threads()
    shard = int(task["shard"])
    panel_start = int(task["panel_start"])
    states = task["states"]
    output_dir = Path(task["output_dir"])
    progress_path = output_dir / f"shard_{shard}.json"
    sidecar_path = output_dir / f"shard_{shard}.identity.json"
    schedule = [
        {
            "identity": _state_identity(game_seed, state, metadata),
            "budget": budget,
            "panel_index": panel_start + local,
            "seed": _search_seed(panel_start + local, budget),
        }
        for local, (game_seed, state, metadata) in enumerate(states)
        for budget in BUDGETS
    ]
    sidecar = {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard": shard,
        "panel_start": panel_start,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "budgets": list(BUDGETS),
        "search_configs": {
            str(budget): asdict(_search_config(budget, 0)) for budget in BUDGETS
        },
        "schedule": schedule,
    }
    if sidecar_path.exists():
        if json.loads(sidecar_path.read_text(encoding="utf-8")) != sidecar:
            raise ValueError(f"126 shard {shard} identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError(f"126 shard {shard} progress exists without identity.")
        _write_json(sidecar_path, sidecar)
    completed: list[dict[str, Any]] = []
    if progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload.get("identity") != sidecar:
            raise ValueError(f"126 shard {shard} progress identity mismatch.")
        completed = list(payload.get("rows", []))
        observed_prefix = [
            {
                "identity": row["identity"],
                "budget": row["budget"],
                "panel_index": row["panel_index"],
                "seed": row["seed"],
            }
            for row in completed
        ]
        if observed_prefix != schedule[: len(completed)]:
            raise ValueError(f"126 shard {shard} is not an exact completed prefix.")
    inference = AlphaZeroInference(task["checkpoint"], device=task["device"])
    flat_tasks = [
        (local, game_seed, state, metadata, budget)
        for local, (game_seed, state, metadata) in enumerate(states)
        for budget in BUDGETS
    ]
    for flat_index, (local, game_seed, state, metadata, budget) in enumerate(flat_tasks):
        if flat_index < len(completed):
            continue
        panel_index = panel_start + local
        completed.append(
            _run_search_pair(
                inference,
                game_seed,
                state,
                metadata,
                panel_index,
                budget,
            )
        )
        _write_json(
            progress_path,
            {
                "identity": sidecar,
                "completed_comparisons": len(completed),
                "target_comparisons": len(flat_tasks),
                "completed": False,
                "rows": completed,
            },
        )
    _write_json(
        progress_path,
        {
            "identity": sidecar,
            "completed_comparisons": len(completed),
            "target_comparisons": len(flat_tasks),
            "completed": True,
            "rows": completed,
        },
    )
    return shard, completed


def _run_search_pair(
    inference,
    game_seed: int,
    state: SplendorState,
    metadata: dict[str, Any],
    panel_index: int,
    budget: int,
) -> dict[str, Any]:
    identity = _state_identity(game_seed, state, metadata)
    flipped = state.copy_shallow()
    flipped.start_player = 1 - state.start_player
    if canonical_public_belief_key(
        state, state.current_player
    ) == canonical_public_belief_key(flipped, flipped.current_player):
        raise ValueError("126 role flip did not change public-belief key.")
    if encode_canonical_state_entities(
        state, state.current_player
    ) != encode_canonical_state_entities(flipped, flipped.current_player):
        raise ValueError("126 v1 tensor unexpectedly exposes start role.")
    original_env = SplendorEnv(seed=game_seed)
    flipped_env = SplendorEnv(seed=game_seed)
    original_actions = original_env.legal_actions(state)
    flipped_actions = flipped_env.legal_actions(flipped)
    original_by_key = {_action_key(action): action for action in original_actions}
    flipped_by_key = {_action_key(action): action for action in flipped_actions}
    if set(original_by_key) != set(flipped_by_key):
        raise ValueError("126 role flip changed semantic legal actions.")
    ordered_keys = sorted(original_by_key)
    original_prediction = inference.predict_one(
        state, [original_by_key[key] for key in ordered_keys]
    )
    flipped_prediction = inference.predict_one(
        flipped, [flipped_by_key[key] for key in ordered_keys]
    )
    max_logit_diff = max(
        abs(first - second)
        for first, second in zip(
            original_prediction.logits, flipped_prediction.logits, strict=True
        )
    )
    raw_value_diff = abs(original_prediction.value - flipped_prediction.value)
    seed = _search_seed(panel_index, budget)
    config = _search_config(budget, seed)
    original_result = AlphaZeroSearch(inference, config).run(
        original_env, state, original_actions, self_play=False, seed=seed
    )
    flipped_result = AlphaZeroSearch(inference, config).run(
        flipped_env, flipped, flipped_actions, self_play=False, seed=seed
    )
    original_visits = {
        _action_key(edge.action): edge.probability
        for edge in original_result.root_edges
    }
    flipped_visits = {
        _action_key(edge.action): edge.probability for edge in flipped_result.root_edges
    }
    if set(original_visits) != set(flipped_visits):
        raise ValueError("126 search pair returned different root actions.")
    visit_tv = 0.5 * sum(
        abs(original_visits[key] - flipped_visits[key]) for key in original_visits
    )
    original_top = _action_key(
        original_result.root_edges[original_result.visit_argmax_row].action
    )
    flipped_top = _action_key(
        flipped_result.root_edges[flipped_result.visit_argmax_row].action
    )
    signed_value = (
        original_result.search_root_value - flipped_result.search_root_value
    )
    return {
        "identity": identity,
        "panel_index": panel_index,
        "budget": budget,
        "seed": seed,
        "observer_is_start_player": state.current_player == state.start_player,
        "alias_checks": {
            "public_key_changed": True,
            "entity_tensor_identical": True,
            "action_keys_identical": True,
            "max_raw_logit_abs_diff": max_logit_diff,
            "raw_value_abs_diff": raw_value_diff,
        },
        "visit_tv": visit_tv,
        "top_action_changed": original_top != flipped_top,
        "abs_root_value_delta": abs(signed_value),
        "signed_root_value_delta_original_minus_flipped": signed_value,
        "original": _search_summary(original_result),
        "flipped": _search_summary(flipped_result),
    }


def summarize_operational_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_budget = {}
    for budget_index, budget in enumerate(BUDGETS):
        budget_rows = [row for row in rows if row["budget"] == budget]
        seed_base = 1_126_100 + 100 * budget_index
        by_budget[str(budget)] = {
            "overall": _metric_summary(budget_rows, bootstrap_seed=seed_base),
            "by_octile": {
                str(octile): _metric_summary(
                    [row for row in budget_rows if row["identity"]["octile"] == octile]
                )
                for octile in range(1, OCTILES + 1)
            },
            "by_original_role": {
                str(role).lower(): _metric_summary(
                    [
                        row
                        for row in budget_rows
                        if bool(row["observer_is_start_player"]) is role
                    ]
                )
                for role in (False, True)
            },
            "first_four_octiles": _metric_summary(
                [row for row in budget_rows if row["identity"]["octile"] <= 4],
                bootstrap_seed=seed_base + 10,
            ),
            "last_four_octiles": _metric_summary(
                [row for row in budget_rows if row["identity"]["octile"] >= 5],
                bootstrap_seed=seed_base + 20,
            ),
            "final_two_octiles": _metric_summary(
                [row for row in budget_rows if row["identity"]["octile"] >= 7],
                bootstrap_seed=seed_base + 30,
            ),
        }
    integrity_checks = {
        "exact_512_comparisons": len(rows) == STATE_COUNT * len(BUDGETS),
        "exact_32_games": len({row["identity"]["game_seed"] for row in rows})
        == GAME_COUNT,
        "exact_256_per_budget": all(
            sum(row["budget"] == budget for row in rows) == STATE_COUNT
            for budget in BUDGETS
        ),
        "eight_octiles_per_game_per_budget": all(
            {
                row["identity"]["octile"]
                for row in rows
                if row["budget"] == budget
                and row["identity"]["game_seed"] == game_seed
            }
            == set(range(1, OCTILES + 1))
            for budget in BUDGETS
            for game_seed in {row["identity"]["game_seed"] for row in rows}
        ),
        "all_originals_reachable": all(
            row["identity"]["source_reachable"] is True for row in rows
        ),
        "all_public_keys_changed": all(
            row["alias_checks"]["public_key_changed"] for row in rows
        ),
        "all_entity_tensors_aliased": all(
            row["alias_checks"]["entity_tensor_identical"] for row in rows
        ),
        "all_action_sets_identical": all(
            row["alias_checks"]["action_keys_identical"] for row in rows
        ),
        "raw_logits_equal_within_1e_7": max(
            row["alias_checks"]["max_raw_logit_abs_diff"] for row in rows
        )
        <= 1e-7,
        "raw_values_equal_within_1e_7": max(
            row["alias_checks"]["raw_value_abs_diff"] for row in rows
        )
        <= 1e-7,
        "all_search_counts_exact": all(
            row[side]["simulations"] == row["budget"]
            for row in rows
            for side in ("original", "flipped")
        ),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "visit_tv",
                "abs_root_value_delta",
                "signed_root_value_delta_original_minus_flipped",
            )
        ),
    }
    return {
        "by_budget": by_budget,
        "integrity_checks": integrity_checks,
        "rows": rows,
    }


def _metric_summary(
    rows: list[dict[str, Any]], *, bootstrap_seed: int | None = None
) -> dict[str, Any]:
    if not rows:
        raise ValueError("126 cannot summarize an empty stratum.")
    result = {
        "comparisons": len(rows),
        "games": len({row["identity"]["game_seed"] for row in rows}),
        "mean_visit_tv": sum(float(row["visit_tv"]) for row in rows) / len(rows),
        "top_action_change_rate": sum(bool(row["top_action_changed"]) for row in rows)
        / len(rows),
        "mean_abs_root_value_delta": sum(
            float(row["abs_root_value_delta"]) for row in rows
        )
        / len(rows),
        "mean_signed_root_value_delta_original_minus_flipped": sum(
            float(row["signed_root_value_delta_original_minus_flipped"])
            for row in rows
        )
        / len(rows),
        "fraction_visit_tv_at_least_0_05": sum(
            float(row["visit_tv"]) >= 0.05 for row in rows
        )
        / len(rows),
        "fraction_abs_value_delta_at_least_0_05": sum(
            float(row["abs_root_value_delta"]) >= 0.05 for row in rows
        )
        / len(rows),
        "fraction_any_material_event": sum(
            bool(row["top_action_changed"])
            or float(row["visit_tv"]) >= 0.05
            or float(row["abs_root_value_delta"]) >= 0.05
            for row in rows
        )
        / len(rows),
    }
    if bootstrap_seed is not None:
        result["bootstrap"] = {
            "visit_tv": _game_cluster_bootstrap(rows, "visit_tv", bootstrap_seed),
            "top_action_change": _game_cluster_bootstrap(
                rows, "top_action_changed", bootstrap_seed + 1
            ),
            "abs_root_value_delta": _game_cluster_bootstrap(
                rows, "abs_root_value_delta", bootstrap_seed + 2
            ),
        }
    return result


def _game_cluster_bootstrap(rows, key: str, seed: int) -> dict[str, Any]:
    clusters: dict[int, list[float]] = {}
    for row in rows:
        clusters.setdefault(int(row["identity"]["game_seed"]), []).append(
            float(row[key])
        )
    if len(clusters) != GAME_COUNT or len({len(values) for values in clusters.values()}) != 1:
        raise ValueError("126 cluster bootstrap requires balanced 32-game strata.")
    means = [sum(clusters[game]) / len(clusters[game]) for game in sorted(clusters)]
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(means) for _ in means) / len(means)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "games": len(means),
        "states_per_game": len(next(iter(clusters.values()))),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "mean": sum(means) / len(means),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _materiality(summary: dict[str, Any]) -> dict[str, Any]:
    policy_checks = {
        "mean_visit_tv_at_least_0_05": summary["mean_visit_tv"] >= 0.05,
        "top_action_change_at_least_10pct": summary["top_action_change_rate"]
        >= 0.10,
        "visit_tv_ci_low_above_0_01": summary["bootstrap"]["visit_tv"][
            "ci95_low"
        ]
        > 0.01,
    }
    value_checks = {
        "mean_abs_value_delta_at_least_0_05": summary[
            "mean_abs_root_value_delta"
        ]
        >= 0.05,
        "abs_value_delta_ci_low_above_0_01": summary["bootstrap"][
            "abs_root_value_delta"
        ]["ci95_low"]
        > 0.01,
    }
    return {
        "policy_material": all(policy_checks.values()),
        "value_material": all(value_checks.values()),
        "material": all(policy_checks.values()) or all(value_checks.values()),
        "policy_checks": policy_checks,
        "value_checks": value_checks,
    }


def operational_gate(summary: dict[str, Any]) -> dict[str, Any]:
    operative = summary["by_budget"]["256"]
    overall = _materiality(operative["overall"])
    final_two = _materiality(operative["final_two_octiles"])
    integrity = all(summary["integrity_checks"].values())
    return {
        "operative_budget": 256,
        "integrity_passed": integrity,
        "overall_material": integrity and overall["material"],
        "final_two_octiles_material": integrity and final_two["material"],
        "overall": overall,
        "final_two_octiles": final_two,
        "integrity_checks": summary["integrity_checks"],
    }


def _verify_inputs(config) -> dict[str, str]:
    observed = {
        "result_117": _sha256(config.result_117),
        "result_125": _sha256(config.result_125),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "untouched_holdout": _sha256(config.untouched_holdout),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"126 frozen input mismatch: {observed}")
    result_117 = json.loads(config.result_117.read_text(encoding="utf-8"))
    result_125 = json.loads(config.result_125.read_text(encoding="utf-8"))
    if (
        result_117.get("status") != "complete"
        or result_117.get("decision")
        != "teacher_role_sensitivity_confirmed_review_identity_role_aware_pilot_no_training"
        or result_125.get("status") != "complete"
        or result_125.get("decision")
        != "paired_objective_role_pilot_failed_development_no_confirmation"
        or result_125.get("confirmation_target_manifest_accessed") is not False
        or result_125.get("promotion_authorized") is not False
    ):
        raise ValueError("126 does not reproduce finalized 117/125 decisions.")
    identity = load_checkpoint(config.generation_11_checkpoint, device=config.device)[1]
    rows = read_replay_jsonl(config.untouched_holdout)
    if (
        identity.generation != 11
        or len(rows) != 1_956
        or len({row.game_seed for row in rows}) != GAME_COUNT
        or any(row.termination_reason != "completed" for row in rows)
        or config.torch_threads != 1
        or config.workers != 4
    ):
        raise ValueError("126 frozen lineage, holdout, or worker settings mismatch.")
    return observed


def _search_seed(panel_index: int, budget: int) -> int:
    return 1_126_000 + 10_000 * BUDGETS.index(budget) + panel_index


def _search_config(budget: int, seed: int) -> SearchConfig:
    return SearchConfig(
        simulations=budget,
        c_puct=2.25,
        max_depth=96,
        mode="real",
        root_noise_fraction=0.0,
        seed=seed,
    )


def _search_summary(result) -> dict[str, Any]:
    return {
        "simulations": result.simulations,
        "search_root_value": result.search_root_value,
        "raw_network_value": result.raw_network_value,
        "max_depth_reached": result.max_depth_reached,
        "inference_calls": result.inference_calls,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _state_identity(
    game_seed: int, state: SplendorState, metadata: dict[str, Any]
) -> dict[str, Any]:
    return {
        "game_seed": int(game_seed),
        "turn_index": int(state.turn_index),
        "current_player": int(state.current_player),
        "start_player": int(state.start_player),
        "observer_is_start_player": state.current_player == state.start_player,
        "scores": [int(player.score) for player in state.players],
        "pending_round_end": bool(state.pending_round_end),
        **metadata,
    }


def _jsonable_config(config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
