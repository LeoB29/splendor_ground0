"""Start-role representation and teacher-sensitivity audit 117."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch

from splendor_ai.encoding import canonical_public_belief_key, encode_canonical_state_entities
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.replay import serialize_action

from .capacity_network import load_capacity_checkpoint
from .capacity_pilot import capacity_training_config
from .network import AlphaZeroInference, load_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .scaled_continuation import _sha256, _write_json
from .search import AlphaZeroSearch, SearchConfig
from .training import _run_epoch, collate_replay_rows


@dataclass(frozen=True, slots=True)
class StartRoleAuditConfig:
    output_dir: Path
    result_114: Path
    candidate_114: Path
    result_116: Path
    candidate_116: Path
    generation_11_checkpoint: Path
    untouched_holdout: Path
    installed_checkpoint: Path
    device: str = "cpu"
    torch_threads: int | None = None
    workers: int = 4


PROTOCOL_SHA256 = "30089d05ad83a6e2757056bfeb31d79d7d9f89e777ecd424b8b18c76251da579"
EXPECTED_HASHES = {
    "result_114": "dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5",
    "candidate_114": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "result_116": "5c50ca7989381ad3c55b32f42e97a9ab67b4992085c53e1578227d932bde5718",
    "candidate_116": "bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "untouched_holdout": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}
PAIR_COUNT = 64
GAME_COUNT = 32
SHARD_COUNT = 4
PAIRS_PER_SHARD = 16
SIMULATIONS = 1_024
BOOTSTRAP_SAMPLES = 50_000


def phase_for_turn(turn: int) -> str:
    if turn < 20:
        return "early"
    if turn < 40:
        return "mid"
    return "late"


def interpretation_gate(summary: dict[str, Any]) -> dict[str, Any]:
    integrity_checks = dict(summary["integrity_checks"])
    policy_checks = {
        "top_change_at_least_10pct": summary["top_action_change_rate"] >= 0.10,
        "mean_visit_tv_at_least_0_05": summary["mean_visit_tv"] >= 0.05,
        "visit_tv_ci_low_above_0_01": summary["bootstrap"]["visit_tv"]["ci95_low"] > 0.01,
    }
    value_checks = {
        "mean_abs_root_value_delta_at_least_0_05": summary["mean_abs_root_value_delta"] >= 0.05,
        "root_value_delta_ci_low_above_0_01": summary["bootstrap"]["abs_root_value_delta"]["ci95_low"] > 0.01,
    }
    integrity = all(integrity_checks.values())
    policy = all(policy_checks.values())
    value = all(value_checks.values())
    return {
        "integrity_passed": integrity,
        "policy_role_sensitivity": policy,
        "value_role_sensitivity": value,
        "start_role_repair_authorized_for_review": integrity and (policy or value),
        "integrity_checks": integrity_checks,
        "policy_checks": policy_checks,
        "value_checks": value_checks,
    }


def audit_decision(gate: dict[str, Any]) -> str:
    if not gate["integrity_passed"]:
        return "start_role_audit_integrity_failed_no_architecture_authorization"
    if gate["start_role_repair_authorized_for_review"]:
        return "teacher_role_sensitivity_confirmed_review_identity_role_aware_pilot_no_training"
    return "teacher_role_sensitivity_not_material_review_training_objective_multigeneration_no_training"


def run_start_role_audit_117(config: StartRoleAuditConfig) -> dict[str, Any]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    sealed = config.output_dir / "result.json"
    if sealed.exists():
        payload = json.loads(sealed.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 117 result is not a valid completed result.")
    result: dict[str, Any] = {
        "kind": "alphazero_start_role_representation_audit_117",
        "status": "running",
        "active_stage": "stratified_untouched_holdout",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": dict(EXPECTED_HASHES),
        "stratified_holdout": None,
        "reconstruction": None,
        "counterfactual_teacher": None,
        "interpretation_gate": None,
        "training_performed": False,
        "selfplay_generated": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
        "installed_checkpoint_sha256": EXPECTED_HASHES["installed_checkpoint"],
        "decision": "running",
    }
    _write_progress(config.output_dir, result, started)
    rows = read_replay_jsonl(config.untouched_holdout)
    result["stratified_holdout"] = _run_or_load_stratified(config, rows)
    result["active_stage"] = "exact_state_reconstruction"
    _write_progress(config.output_dir, result, started)
    states, reconstruction = reconstruct_panel(rows)
    result["reconstruction"] = reconstruction
    result["active_stage"] = "four_shard_coupled_1024_teacher"
    _write_progress(config.output_dir, result, started)
    pairs = _run_or_load_counterfactual(config, states)
    summary = summarize_pairs(pairs)
    gate = interpretation_gate(summary)
    result["counterfactual_teacher"] = summary
    result["interpretation_gate"] = gate
    result["decision"] = audit_decision(gate)
    result["status"] = "complete"
    result["active_stage"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - started
    _write_json(sealed, result)
    return result


def _run_or_load_stratified(
    config: StartRoleAuditConfig, rows: list[AlphaZeroReplayRow]
) -> dict[str, Any]:
    stage = config.output_dir / "stratified_holdout.stage.json"
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if payload.get("protocol_sha256") != PROTOCOL_SHA256 or payload.get("input_hashes") != EXPECTED_HASHES:
            raise ValueError("117 stratified stage identity mismatch.")
        return payload
    for row in rows:
        if row.player_id != row.turn_index % 2:
            raise ValueError("117 holdout player id does not match fixed-start turn parity.")
    groups: dict[str, list[AlphaZeroReplayRow]] = {"all": list(rows)}
    for role_name, role in (("start", 0), ("nonstart", 1)):
        role_rows = [row for row in rows if row.player_id == role]
        groups[f"role:{role_name}"] = role_rows
        for phase in ("early", "mid", "late"):
            groups[f"role:{role_name}|phase:{phase}"] = [
                row for row in role_rows if phase_for_turn(row.turn_index) == phase
            ]
    for phase in ("early", "mid", "late"):
        groups[f"phase:{phase}"] = [
            row for row in rows if phase_for_turn(row.turn_index) == phase
        ]
    models = {
        "generation_11": load_checkpoint(config.generation_11_checkpoint, device=config.device)[0],
        "candidate_114": load_checkpoint(config.candidate_114, device=config.device)[0],
        "candidate_116": load_capacity_checkpoint(config.candidate_116, device=config.device)[0],
    }
    metrics: dict[str, dict[str, Any]] = {}
    for group_name, group_rows in groups.items():
        if not group_rows:
            raise ValueError(f"117 stratified group is empty: {group_name}")
        batches = _batches(group_rows)
        metrics[group_name] = {
            name: _run_epoch(model, batches, config.device, None, capacity_training_config())
            for name, model in models.items()
        }
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    keys = (
        "policy_loss",
        "value_loss",
        "objective_loss",
        "policy_top_target_agreement",
        "policy_target_top_probability",
        "policy_entropy",
        "value_mae",
    )
    for group_name, group_metrics in metrics.items():
        baseline = group_metrics["generation_11"]
        deltas[group_name] = {
            candidate: {
                key: float(group_metrics[candidate][key]) - float(baseline[key])
                for key in keys
            }
            for candidate in ("candidate_114", "candidate_116")
        }
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": dict(EXPECTED_HASHES),
        "rows": len(rows),
        "games": len({row.game_seed for row in rows}),
        "group_rows": {name: len(group) for name, group in groups.items()},
        "metrics": metrics,
        "deltas_vs_generation_11": deltas,
    }
    _write_json(stage, payload)
    return payload


def reconstruct_panel(
    rows: list[AlphaZeroReplayRow],
) -> tuple[list[tuple[int, SplendorState]], dict[str, Any]]:
    grouped: dict[int, list[AlphaZeroReplayRow]] = {}
    for row in rows:
        grouped.setdefault(row.game_seed, []).append(row)
    if len(grouped) != GAME_COUNT:
        raise ValueError("117 requires exactly 32 untouched games.")
    selected: list[tuple[int, SplendorState]] = []
    schedule: list[dict[str, Any]] = []
    for game_seed in sorted(grouped):
        game_rows = sorted(grouped[game_seed], key=lambda row: row.turn_index)
        env = SplendorEnv(seed=game_seed)
        state = env.initial_state()
        eligible: list[SplendorState] = []
        for expected_turn, row in enumerate(game_rows):
            if (
                row.turn_index != expected_turn
                or row.player_id != state.current_player
                or state.turn_index != row.turn_index
            ):
                raise ValueError("117 reconstructed turn/player sequence mismatch.")
            encoded = encode_canonical_state_entities(state, state.current_player)
            if (
                encoded.entity_type_ids != row.entity_type_ids
                or encoded.card_ids != row.entity_card_ids
                or encoded.noble_ids != row.entity_noble_ids
                or encoded.numeric_features != row.entity_numeric_features
            ):
                raise ValueError("117 reconstructed canonical entity tensor mismatch.")
            if not state.pending_round_end:
                eligible.append(state.copy_shallow())
            selected_payload = row.action_payloads[row.selected_action_row]
            matches = [
                action
                for action in env.legal_actions(state)
                if serialize_action(action) == selected_payload
            ]
            if len(matches) != 1:
                raise ValueError("117 selected replay action is not uniquely legal.")
            state = env.step(state, matches[0])
        last = game_rows[-1]
        if (
            not state.terminal
            or state.winner != last.winner
            or tuple(player.score for player in state.players) != last.final_scores
        ):
            raise ValueError("117 reconstructed terminal outcome mismatch.")
        if len(eligible) < 2:
            raise ValueError("117 game lacks two pre-round-end states.")
        for state_item in eligible[-2:]:
            selected.append((game_seed, state_item))
            schedule.append(_state_identity(game_seed, state_item))
    if len(selected) != PAIR_COUNT:
        raise ValueError("117 reconstruction did not yield exactly 64 states.")
    roles = {identity["observer_is_start_player"] for identity in schedule}
    if roles != {False, True}:
        raise ValueError("117 reconstructed panel lacks both start-role classes.")
    return selected, {
        "games": len(grouped),
        "pairs": len(selected),
        "states_per_game": 2,
        "schedule": schedule,
        "role_counts": {
            str(role).lower(): sum(item["observer_is_start_player"] is role for item in schedule)
            for role in (False, True)
        },
    }


def _run_or_load_counterfactual(
    config: StartRoleAuditConfig, states: list[tuple[int, SplendorState]]
) -> list[dict[str, Any]]:
    counterfactual_dir = config.output_dir / "counterfactual"
    counterfactual_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for shard in range(SHARD_COUNT):
        start = shard * PAIRS_PER_SHARD
        tasks.append(
            {
                "shard": shard,
                "states": states[start : start + PAIRS_PER_SHARD],
                "panel_start": start,
                "output_dir": str(counterfactual_dir),
                "checkpoint": str(config.generation_11_checkpoint),
                "device": config.device,
            }
        )
    shards: dict[int, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(_counterfactual_worker, task): task for task in tasks}
        for future in as_completed(futures):
            shard, shard_rows = future.result()
            shards[shard] = shard_rows
    if set(shards) != set(range(SHARD_COUNT)):
        raise ValueError("117 counterfactual audit requires all four shards.")
    rows = [row for shard in range(SHARD_COUNT) for row in shards[shard]]
    observed = [row["identity"] for row in rows]
    expected = [_state_identity(game_seed, state) for game_seed, state in states]
    if observed != expected:
        raise ValueError("117 combined counterfactual schedule mismatch.")
    return rows


def _counterfactual_worker(task: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    _limit_worker_threads()
    shard = int(task["shard"])
    states: list[tuple[int, SplendorState]] = task["states"]
    panel_start = int(task["panel_start"])
    output_dir = Path(task["output_dir"])
    progress_path = output_dir / f"shard_{shard}.json"
    sidecar_path = output_dir / f"shard_{shard}.identity.json"
    identities = [_state_identity(game_seed, state) for game_seed, state in states]
    sidecar = {
        "protocol_sha256": PROTOCOL_SHA256,
        "shard": shard,
        "panel_start": panel_start,
        "checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
        "search_config": asdict(_search_config(0)),
        "operative_search_seed": "1117000 + 100 * panel_index",
        "states": identities,
    }
    if sidecar_path.exists():
        if json.loads(sidecar_path.read_text(encoding="utf-8")) != sidecar:
            raise ValueError(f"117 shard {shard} identity mismatch.")
    else:
        if progress_path.exists():
            raise ValueError(f"117 shard {shard} progress exists without identity.")
        _write_json(sidecar_path, sidecar)
    completed: list[dict[str, Any]] = []
    if progress_path.exists():
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        if payload.get("identity") != sidecar:
            raise ValueError(f"117 shard {shard} progress identity mismatch.")
        completed = list(payload.get("rows", []))
        if [row["identity"] for row in completed] != identities[: len(completed)]:
            raise ValueError(f"117 shard {shard} is not an exact completed prefix.")
    inference = AlphaZeroInference(task["checkpoint"], device=task["device"])
    for local, (game_seed, state) in enumerate(states):
        if local < len(completed):
            continue
        panel_index = panel_start + local
        completed.append(_run_pair(inference, game_seed, state, panel_index))
        _write_json(
            progress_path,
            {
                "identity": sidecar,
                "completed_pairs": len(completed),
                "target_pairs": len(states),
                "completed": False,
                "rows": completed,
            },
        )
    _write_json(
        progress_path,
        {
            "identity": sidecar,
            "completed_pairs": len(completed),
            "target_pairs": len(states),
            "completed": True,
            "rows": completed,
        },
    )
    return shard, completed


def _run_pair(
    inference: AlphaZeroInference,
    game_seed: int,
    state: SplendorState,
    panel_index: int,
) -> dict[str, Any]:
    identity = _state_identity(game_seed, state)
    flipped = state.copy_shallow()
    flipped.start_player = 1 - state.start_player
    if canonical_public_belief_key(state, state.current_player) == canonical_public_belief_key(flipped, flipped.current_player):
        raise ValueError("117 role flip did not change canonical public-belief key.")
    original_encoding = encode_canonical_state_entities(state, state.current_player)
    flipped_encoding = encode_canonical_state_entities(flipped, flipped.current_player)
    if original_encoding != flipped_encoding:
        raise ValueError("117 v1 entity tensor unexpectedly exposes start role.")
    original_env = SplendorEnv(seed=game_seed)
    flipped_env = SplendorEnv(seed=game_seed)
    original_actions = original_env.legal_actions(state)
    flipped_actions = flipped_env.legal_actions(flipped)
    original_by_action = {_action_key(action): action for action in original_actions}
    flipped_by_action = {_action_key(action): action for action in flipped_actions}
    if set(original_by_action) != set(flipped_by_action):
        raise ValueError("117 role flip changed the legal semantic action set.")
    ordered_keys = sorted(original_by_action)
    original_prediction = inference.predict_one(
        state, [original_by_action[key] for key in ordered_keys]
    )
    flipped_prediction = inference.predict_one(
        flipped, [flipped_by_action[key] for key in ordered_keys]
    )
    max_logit_diff = max(
        abs(original_prediction.logits[index] - flipped_prediction.logits[index])
        for index in range(len(original_prediction.logits))
    )
    value_diff = abs(original_prediction.value - flipped_prediction.value)
    seed = 1_117_000 + 100 * panel_index
    original_result = AlphaZeroSearch(inference, _search_config(seed)).run(
        original_env, state, original_actions, self_play=False, seed=seed
    )
    flipped_result = AlphaZeroSearch(inference, _search_config(seed)).run(
        flipped_env, flipped, flipped_actions, self_play=False, seed=seed
    )
    original_visits = {_action_key(edge.action): edge.probability for edge in original_result.root_edges}
    flipped_visits = {_action_key(edge.action): edge.probability for edge in flipped_result.root_edges}
    if set(original_visits) != set(flipped_visits):
        raise ValueError("117 search pair returned different root action keys.")
    visit_tv = 0.5 * sum(
        abs(original_visits[key] - flipped_visits[key]) for key in original_visits
    )
    original_top = _action_key(original_result.root_edges[original_result.visit_argmax_row].action)
    flipped_top = _action_key(flipped_result.root_edges[flipped_result.visit_argmax_row].action)
    return {
        "identity": identity,
        "panel_index": panel_index,
        "observer_is_start_player": state.current_player == state.start_player,
        "alias_checks": {
            "public_key_changed": True,
            "entity_tensor_identical": True,
            "action_keys_identical": True,
            "max_raw_logit_abs_diff": max_logit_diff,
            "raw_value_abs_diff": value_diff,
        },
        "visit_tv": visit_tv,
        "top_action_changed": original_top != flipped_top,
        "signed_root_value_delta_original_minus_flipped": original_result.search_root_value - flipped_result.search_root_value,
        "abs_root_value_delta": abs(original_result.search_root_value - flipped_result.search_root_value),
        "original": _search_summary(original_result),
        "flipped": _search_summary(flipped_result),
    }


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PAIR_COUNT or len({(row["identity"]["game_seed"], row["identity"]["turn_index"]) for row in rows}) != PAIR_COUNT:
        raise ValueError("117 summary requires 64 unique scheduled pairs.")
    summary = _metric_summary(rows)
    summary["by_original_role"] = {
        str(role).lower(): _metric_summary(
            [row for row in rows if bool(row["observer_is_start_player"]) is role],
            include_bootstrap=False,
        )
        for role in (False, True)
    }
    summary["integrity_checks"] = {
        "exact_64_pairs": len(rows) == 64,
        "exact_32_games": len({row["identity"]["game_seed"] for row in rows}) == 32,
        "both_role_classes": {bool(row["observer_is_start_player"]) for row in rows} == {False, True},
        "all_public_keys_changed": all(row["alias_checks"]["public_key_changed"] for row in rows),
        "all_entity_tensors_aliased": all(row["alias_checks"]["entity_tensor_identical"] for row in rows),
        "all_action_sets_identical": all(row["alias_checks"]["action_keys_identical"] for row in rows),
        "raw_logits_equal_within_1e_7": max(row["alias_checks"]["max_raw_logit_abs_diff"] for row in rows) <= 1e-7,
        "raw_values_equal_within_1e_7": max(row["alias_checks"]["raw_value_abs_diff"] for row in rows) <= 1e-7,
        "all_searches_exact_1024": all(row[side]["simulations"] == 1024 for row in rows for side in ("original", "flipped")),
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in ("visit_tv", "abs_root_value_delta", "signed_root_value_delta_original_minus_flipped")
        ),
    }
    summary["rows"] = rows
    return summary


def _metric_summary(
    rows: list[dict[str, Any]], *, include_bootstrap: bool = True
) -> dict[str, Any]:
    result = {
        "pairs": len(rows),
        "games": len({row["identity"]["game_seed"] for row in rows}),
        "mean_visit_tv": sum(float(row["visit_tv"]) for row in rows) / len(rows),
        "top_action_change_rate": sum(bool(row["top_action_changed"]) for row in rows) / len(rows),
        "mean_abs_root_value_delta": sum(float(row["abs_root_value_delta"]) for row in rows) / len(rows),
        "mean_signed_root_value_delta_original_minus_flipped": sum(float(row["signed_root_value_delta_original_minus_flipped"]) for row in rows) / len(rows),
    }
    if include_bootstrap:
        result["bootstrap"] = {
            "visit_tv": _game_cluster_bootstrap(rows, "visit_tv", seed=1_117_100),
            "top_action_change": _game_cluster_bootstrap(rows, "top_action_changed", seed=1_117_101),
            "abs_root_value_delta": _game_cluster_bootstrap(rows, "abs_root_value_delta", seed=1_117_102),
        }
    return result


def _game_cluster_bootstrap(
    rows: list[dict[str, Any]], key: str, *, seed: int
) -> dict[str, float | int]:
    clusters: dict[int, list[float]] = {}
    for row in rows:
        clusters.setdefault(int(row["identity"]["game_seed"]), []).append(float(row[key]))
    if len(clusters) != GAME_COUNT or any(len(values) != 2 for values in clusters.values()):
        raise ValueError("117 bootstrap requires two scheduled states per game.")
    means = [sum(clusters[game]) / 2.0 for game in sorted(clusters)]
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choice(means) for _ in means) / len(means)
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    return {
        "games": len(means),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "mean": sum(means) / len(means),
        "ci95_low": _percentile(draws, 0.025),
        "ci95_high": _percentile(draws, 0.975),
    }


def _verify_inputs(config: StartRoleAuditConfig) -> None:
    paths = {
        "result_114": config.result_114,
        "candidate_114": config.candidate_114,
        "result_116": config.result_116,
        "candidate_116": config.candidate_116,
        "generation_11_checkpoint": config.generation_11_checkpoint,
        "untouched_holdout": config.untouched_holdout,
        "installed_checkpoint": config.installed_checkpoint,
    }
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_HASHES[name]:
            raise ValueError(f"Frozen 117 input hash mismatch for {name}: {actual}")
    result_116 = json.loads(config.result_116.read_text(encoding="utf-8"))
    if (
        result_116.get("status") != "complete"
        or result_116.get("decision") != "identity_capacity_offline_gain_not_converted_review_representation_design_no_install"
        or result_116.get("training", {}).get("checkpoint_sha256") != EXPECTED_HASHES["candidate_116"]
        or result_116.get("promotion_authorized") is not False
    ):
        raise ValueError("Frozen 116 result does not authorize representation audit 117.")
    result_114 = json.loads(config.result_114.read_text(encoding="utf-8"))
    if result_114.get("training", {}).get("checkpoint_sha256") != EXPECTED_HASHES["candidate_114"]:
        raise ValueError("Frozen 114 candidate lineage mismatch.")
    parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)[1]
    candidate_114 = load_checkpoint(config.candidate_114, device=config.device)[1]
    candidate_116 = load_capacity_checkpoint(config.candidate_116, device=config.device)[1]
    if (
        parent.generation != 11
        or candidate_114.metadata.get("parent_generation") != 11
        or candidate_116.metadata.get("parent_generation") != 11
    ):
        raise ValueError("Frozen 117 checkpoint lineage mismatch.")
    rows = read_replay_jsonl(config.untouched_holdout)
    if (
        len(rows) != 1_956
        or len({row.game_seed for row in rows}) != 32
        or any(row.termination_reason != "completed" for row in rows)
    ):
        raise ValueError("Frozen 117 holdout is not the exact 32-game normal panel.")
    if config.torch_threads != 1 or config.workers != 4:
        raise ValueError("117 requires one thread and four workers.")


def _batches(rows: list[AlphaZeroReplayRow]) -> list[dict[str, torch.Tensor]]:
    return [
        collate_replay_rows(
            rows[start : start + 256],
            include_corrective_fields=True,
        )
        for start in range(0, len(rows), 256)
    ]


def _search_config(seed: int) -> SearchConfig:
    return SearchConfig(
        simulations=SIMULATIONS,
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


def _state_identity(game_seed: int, state: SplendorState) -> dict[str, Any]:
    return {
        "game_seed": int(game_seed),
        "turn_index": int(state.turn_index),
        "current_player": int(state.current_player),
        "start_player": int(state.start_player),
        "observer_is_start_player": state.current_player == state.start_player,
        "scores": [int(player.score) for player in state.players],
    }


def _action_key(action) -> str:
    return json.dumps(serialize_action(action), sort_keys=True, separators=(",", ":"))


def _percentile(values: list[float], fraction: float) -> float:
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _limit_worker_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _jsonable_config(config: StartRoleAuditConfig) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _write_progress(output: Path, result: dict[str, Any], started: float) -> None:
    _write_json(output / "progress.json", {**result, "elapsed_seconds": time.perf_counter() - started})
