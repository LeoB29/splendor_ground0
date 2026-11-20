"""Guarded split-readout start-role pilot for experiment 120."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .network import load_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl
from .role_readout_network import (
    create_identity_readout_role_model,
    load_readout_role_checkpoint,
    readout_role_parameter_counts,
    save_readout_role_checkpoint,
)
from .scaled_continuation import _sha256, _write_json
from .start_role_pilot import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    SHUFFLE_SEED,
    VALUE_WEIGHT,
    _RoleTargetDataset,
    _all_finite,
    _candidate_contrasts,
    _cluster_bootstrap_mean,
    _collate_role_target_rows,
    _confirmation_prediction_rows,
    _evaluate_replay_rows,
    _evaluate_target_loader,
    _loss_means,
    _seal,
    _state_identity,
    _target_losses,
    _target_manifest_hash,
    _target_model_inputs,
    _training_rows_from_records,
    _verify_inputs as _verify_118_base_inputs,
    _write_progress,
    identity_audit,
    reconstruct_role_panel,
    run_or_load_arena,
    run_or_load_target_panel,
    split_pair_games,
    summarize_target_records,
)


PROTOCOL_SHA256 = "d444269d44f8ef144afb071fc9cd71f037f1cbc935021478ed8618e9d775761a"
TARGET_PROTOCOL_SHA256 = "b7b92bf53ab1801bef96fbdb4f9544a81943889c82a243c20b4349cb4b2bf186"
EXPECTED_HASHES = {
    "result_119": "6256cc23f24e7c6b601884bbdbbe0fb9f9c4126e64700818db03f692f6fd1828",
    "result_118": "4846b79e1fbcd6344e759155befc469463a9100423f8785ff0e352fa9127fbc1",
    "result_117": "a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "candidate_114": "93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd",
    "candidate_116": "bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565",
    "partition_manifest": "61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "holdout_replay": "33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88",
    "training_target_manifest": "278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f",
    "installed_checkpoint": "60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc",
}


@dataclass(frozen=True, slots=True)
class SplitReadoutRolePilotConfig:
    output_dir: Path
    result_119: Path
    result_118: Path
    result_117: Path
    generation_11_checkpoint: Path
    candidate_114: Path
    candidate_116: Path
    partition_manifest: Path
    train_replay: Path
    holdout_replay: Path
    training_target_dir: Path
    installed_checkpoint: Path
    device: str = "cpu"
    torch_threads: int = 1
    workers: int = 4
    arena_workers: int = 4
    max_turns: int = 200


def preflight_split_readout_role_pilot_120(
    config: SplitReadoutRolePilotConfig,
) -> dict[str, Any]:
    """Validate the complete frozen pilot without optimizing or searching."""

    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    train_replay = read_replay_jsonl(config.train_replay)
    holdout_replay = read_replay_jsonl(config.holdout_replay)
    train_states, train_reconstruction = reconstruct_role_panel(
        train_replay, expected_games=224, label="120 fitting"
    )
    holdout_states, holdout_reconstruction = reconstruct_role_panel(
        holdout_replay, expected_games=32, label="120 protected confirmation"
    )
    targets = _load_training_targets(config.training_target_dir, train_states)
    oriented_rows = _training_rows_from_records(train_states, targets)
    split = split_pair_games(train_states)

    model, parent_identity = create_identity_readout_role_model(
        config.generation_11_checkpoint,
        device=config.device,
    )
    counts = readout_role_parameter_counts(model)
    expected_counts = {
        "parent": 356_450,
        "policy_role": 96,
        "value_role": 96,
        "role_total": 192,
        "total": 356_642,
        "trainable": 192,
    }
    if counts != expected_counts:
        raise ValueError(f"120 split-readout model has wrong parameter counts: {counts}")
    identity = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=model,
        states=train_states + holdout_states,
        device=config.device,
    )
    if identity["max_logit_abs_diff"] > 1e-7 or identity["max_value_abs_diff"] > 1e-7:
        raise ValueError("120 epoch-zero split-readout bridge is not exact gen11.")
    separation = _endpoint_separation_audit(model, oriented_rows[:16], config.device)
    if not all(separation["checks"].values()):
        raise ValueError("120 endpoint separation contract failed.")

    epoch_zero = config.output_dir / "checkpoints" / "epoch_zero_identity.pt"
    save_readout_role_checkpoint(
        epoch_zero,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "parent_generation": parent_identity.generation,
            "selected_epoch": 0,
            "preflight_only": True,
        },
    )
    restored, restored_identity = load_readout_role_checkpoint(
        epoch_zero, device=config.device
    )
    roundtrip = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=restored,
        states=train_states[:4] + holdout_states[:4],
        device=config.device,
    )
    if (
        restored_identity.generation != 12
        or restored_identity.metadata.get("selected_epoch") != 0
        or roundtrip["max_logit_abs_diff"] > 1e-7
        or roundtrip["max_value_abs_diff"] > 1e-7
    ):
        raise ValueError("120 epoch-zero checkpoint round-trip failed.")

    target_summary = summarize_target_records(targets)
    sealed_summary = json.loads(config.result_118.read_text(encoding="utf-8"))["target_summary"]
    if any(abs(float(target_summary[key]) - float(sealed_summary[key])) > 1e-12 for key in (
        "mean_visit_tv",
        "top_action_change_rate",
        "mean_abs_root_value_delta",
    )):
        raise ValueError("120 does not reproduce the sealed 118 target summary.")
    result = {
        "kind": "alphazero_split_readout_role_pilot_120_preflight",
        "status": "preflight_complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": _observed_hashes(config),
        "train_reconstruction": train_reconstruction,
        "protected_reconstruction": holdout_reconstruction,
        "split": split,
        "target_summary": target_summary,
        "target_rows_validated": len(oriented_rows),
        "parameter_counts": counts,
        "identity_audit": identity,
        "endpoint_separation": separation,
        "epoch_zero_checkpoint": str(epoch_zero),
        "epoch_zero_checkpoint_sha256": _sha256(epoch_zero),
        "roundtrip_identity_audit": roundtrip,
        "training_performed": False,
        "protected_confirmation_targets_accessed": False,
        "search_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
        "launch_required": True,
    }
    _write_json(config.output_dir / "preflight.json", result)
    return result


def run_split_readout_role_pilot_120(
    config: SplitReadoutRolePilotConfig,
    *,
    launch: bool = False,
) -> dict[str, Any]:
    preflight = preflight_split_readout_role_pilot_120(config)
    if not launch:
        return preflight
    started = time.perf_counter()
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 120 result is not a valid completed result.")

    train_replay = read_replay_jsonl(config.train_replay)
    holdout_replay = read_replay_jsonl(config.holdout_replay)
    train_states, _ = reconstruct_role_panel(train_replay, expected_games=224, label="120 fitting")
    holdout_states, _ = reconstruct_role_panel(
        holdout_replay, expected_games=32, label="120 protected confirmation"
    )
    targets = _load_training_targets(config.training_target_dir, train_states)
    result: dict[str, Any] = {
        "kind": "alphazero_split_readout_role_pilot_120",
        "status": "running",
        "active_stage": "development_fitting",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": _observed_hashes(config),
        "preflight": preflight,
        "existing_targets_reused": True,
        "new_fitting_targets_generated": False,
        "training_performed": False,
        "protected_confirmation_accessed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_progress(config.output_dir, result, started)

    training = _run_or_load_training(config, train_states, targets)
    result["training_performed"] = True
    result["training"] = training
    if not training["selection_passed"]:
        result["decision"] = "split_readout_role_failed_development_uptake_no_confirmation"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "protected_role_confirmation"
    result["protected_confirmation_accessed"] = True
    _write_progress(config.output_dir, result, started)
    confirmation_targets = run_or_load_target_panel(
        config,
        holdout_states,
        confirmation=True,
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="120",
    )
    confirmation = _evaluate_confirmation(
        config,
        holdout_states,
        confirmation_targets,
        candidate_checkpoint=Path(training["checkpoint"]),
    )
    result["role_confirmation"] = confirmation
    if not confirmation["passed"]:
        result["decision"] = "split_readout_role_failed_protected_confirmation_no_gameplay"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "reachable_safety"
    _write_progress(config.output_dir, result, started)
    safety = _evaluate_safety(
        config,
        holdout_replay,
        candidate_checkpoint=Path(training["checkpoint"]),
    )
    result["reachable_safety"] = safety
    if not safety["passed"]:
        result["decision"] = "split_readout_role_confirmed_but_reachable_regression_no_gameplay"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "ordinary_budget_arena"
    _write_progress(config.output_dir, result, started)
    arena = run_or_load_arena(
        config,
        Path(training["checkpoint"]),
        training["checkpoint_sha256"],
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="120",
        seed_base=2_200_000,
        bootstrap_seed=1_120_200,
        arena_name="split_readout_role_primary_256",
    )
    result["gameplay_arena"] = arena
    result["decision"] = (
        "split_readout_role_confirmed_review_short_continuation_no_install"
        if arena["assessment"]["passed"]
        else "split_readout_role_offline_gain_not_converted_close_role_only_repair_no_install"
    )
    return _seal(config.output_dir, result, started)


def _run_or_load_training(config, states, records) -> dict[str, Any]:
    stage = config.output_dir / "training" / "split_readout.stage.json"
    target_hash = _target_manifest_hash(config.training_target_dir)
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("target_manifest_sha256") != target_hash
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        ):
            raise ValueError("120 training stage identity mismatch.")
        return payload

    split = split_pair_games(states)
    validation_games = set(split["validation_games"])
    rows = _training_rows_from_records(states, records)
    train_rows = [row for row in rows if row.game_seed not in validation_games]
    validation_rows = [row for row in rows if row.game_seed in validation_games]
    if len(train_rows) != 716 or len(validation_rows) != 180:
        raise ValueError("120 oriented fitting/development rows have wrong counts.")
    model, parent_identity = create_identity_readout_role_model(
        config.generation_11_checkpoint, device=config.device
    )
    optimizer = AdamW(
        [model.policy_role, model.value_role],
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    train_loader = DataLoader(
        _RoleTargetDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SHUFFLE_SEED),
        collate_fn=_collate_role_target_rows,
    )
    validation_loader = DataLoader(
        _RoleTargetDataset(validation_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate_role_target_rows,
    )
    epoch_zero = _evaluate_target_loader(model, validation_loader, config.device)
    best_epoch = 0
    best_objective = float(epoch_zero["objective_loss"])
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = [{"epoch": 0, "train": None, "validation": epoch_zero}]
    for epoch in range(1, EPOCHS + 1):
        train_metrics = _train_epoch(model, train_loader, optimizer, config.device)
        validation_metrics = _evaluate_target_loader(model, validation_loader, config.device)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        objective = float(validation_metrics["objective_loss"])
        if objective < best_objective:
            best_epoch = epoch
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_train = _evaluate_target_loader(
        model,
        DataLoader(
            _RoleTargetDataset(train_rows),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=_collate_role_target_rows,
        ),
        config.device,
    )
    selected_validation = _evaluate_target_loader(model, validation_loader, config.device)
    relative_improvement = (
        float(epoch_zero["objective_loss"]) - float(selected_validation["objective_loss"])
    ) / float(epoch_zero["objective_loss"])
    policy_norm = float(torch.linalg.vector_norm(model.policy_role.detach()).item())
    value_norm = float(torch.linalg.vector_norm(model.value_role.detach()).item())
    selection_checks = {
        "nonzero_selected_epoch": best_epoch > 0,
        "relative_objective_improvement_at_least_0_5pct": relative_improvement >= 0.005,
        "policy_role_nonzero": policy_norm > 0.0,
        "value_role_nonzero": value_norm > 0.0,
        "parent_frozen": all(not parameter.requires_grad for parameter in model.parent.parameters()),
        "metrics_finite": _all_finite(
            {"history": history, "selected_train": selected_train, "selected_validation": selected_validation}
        ),
    }
    selection_passed = all(selection_checks.values())
    checkpoint = config.output_dir / "checkpoints" / "split_readout_generation_012.pt"
    save_readout_role_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "parent_checkpoint_sha256": EXPECTED_HASHES["generation_11_checkpoint"],
            "parent_generation": parent_identity.generation,
            "selected_epoch": best_epoch,
            "train_games": split["train_games"],
            "validation_games": split["validation_games"],
            "split_readout_role_only_training": True,
        },
    )
    roundtrip, identity = load_readout_role_checkpoint(checkpoint, device=config.device)
    if (
        identity.generation != 12
        or identity.metadata.get("selected_epoch") != best_epoch
        or readout_role_parameter_counts(roundtrip)["trainable"] != 192
    ):
        raise RuntimeError("120 trained checkpoint failed round-trip validation.")
    metrics = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "gradient_clip": 5.0,
            "shuffle_seed": SHUFFLE_SEED,
            "policy_weight": 1.0,
            "value_weight": VALUE_WEIGHT,
        },
        "split": split,
        "selected_epoch": best_epoch,
        "epoch_zero_validation": epoch_zero,
        "selected_train": selected_train,
        "selected_validation": selected_validation,
        "relative_validation_objective_improvement": relative_improvement,
        "selection_checks": selection_checks,
        "selection_passed": selection_passed,
        "policy_role_norm": policy_norm,
        "value_role_norm": value_norm,
        "history": history,
        "parameter_counts": readout_role_parameter_counts(roundtrip),
    }
    metrics_path = config.output_dir / "training" / "split_readout.json"
    _write_json(metrics_path, metrics)
    payload = {
        "protocol_sha256": PROTOCOL_SHA256,
        "target_manifest_sha256": target_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "selection_passed": selection_passed,
        "training": metrics,
    }
    _write_json(stage, payload)
    return payload


def _train_epoch(model, loader, optimizer, device: str) -> dict[str, float]:
    model.train()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "objective_loss": 0.0}
    count = 0
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        logits, values = model(**_target_model_inputs(moved))
        policy, value, objective = _target_losses(
            logits,
            values,
            moved["action_mask"],
            moved["policy_target"],
            moved["value_target"],
        )
        optimizer.zero_grad(set_to_none=True)
        objective.mean().backward()
        torch.nn.utils.clip_grad_norm_([model.policy_role, model.value_role], 5.0)
        optimizer.step()
        batch_count = int(values.shape[0])
        count += batch_count
        totals["policy_loss"] += float(policy.sum().item())
        totals["value_loss"] += float(value.sum().item())
        totals["objective_loss"] += float(objective.sum().item())
    model.eval()
    return {**{key: value / count for key, value in totals.items()}, "rows": count}


def _evaluate_confirmation(
    config,
    states,
    records,
    *,
    candidate_checkpoint: Path,
    candidate_name: str = "candidate_120",
    candidate_loader=load_readout_role_checkpoint,
    candidate_parameter_checks=None,
    bootstrap_seed_base: int = 1_120_120,
) -> dict[str, Any]:
    summary = summarize_target_records(records)
    sealed = json.loads(config.result_117.read_text(encoding="utf-8"))["counterfactual_teacher"]
    reproduction = {
        key: abs(float(summary[key]) - float(sealed[key]))
        for key in ("mean_visit_tv", "top_action_change_rate", "mean_abs_root_value_delta")
    }
    if any(delta > 1e-9 for delta in reproduction.values()):
        raise ValueError("120 confirmation failed to reproduce sealed 117 teacher metrics.")
    checkpoints = {
        "generation_11": config.generation_11_checkpoint,
        "candidate_114": config.candidate_114,
        "candidate_116": config.candidate_116,
        candidate_name: candidate_checkpoint,
    }
    predictions = {
        name: _confirmation_prediction_rows(path, states, records, config.device)
        for name, path in checkpoints.items()
    }
    means = {name: _loss_means(rows) for name, rows in predictions.items()}
    blind_names = ("generation_11", "candidate_114", "candidate_116")
    selected = {
        endpoint: min(blind_names, key=lambda name: means[name][endpoint])
        for endpoint in ("policy_loss", "value_loss", "objective_loss")
    }
    candidate = predictions[candidate_name]
    endpoint_results = {}
    seeds = {
        "policy_loss": bootstrap_seed_base,
        "value_loss": bootstrap_seed_base + 1,
        "objective_loss": bootstrap_seed_base + 2,
    }
    for endpoint, baseline_name in selected.items():
        baseline = predictions[baseline_name]
        deltas = [
            {"game_seed": row["game_seed"], "value": float(row[endpoint]) - float(base[endpoint])}
            for row, base in zip(candidate, baseline, strict=True)
        ]
        by_role = {}
        for role in (False, True):
            candidate_role = [row for row in candidate if row["observer_is_start_player"] is role]
            baseline_role = [row for row in baseline if row["observer_is_start_player"] is role]
            candidate_mean = sum(float(row[endpoint]) for row in candidate_role) / len(candidate_role)
            baseline_mean = sum(float(row[endpoint]) for row in baseline_role) / len(baseline_role)
            by_role[str(role).lower()] = {
                "candidate": candidate_mean,
                "baseline": baseline_mean,
                "delta": candidate_mean - baseline_mean,
            }
        endpoint_results[endpoint] = {
            "baseline": baseline_name,
            "candidate": means[candidate_name][endpoint],
            "baseline_loss": means[baseline_name][endpoint],
            "delta": means[candidate_name][endpoint] - means[baseline_name][endpoint],
            "by_role": by_role,
            "bootstrap": _cluster_bootstrap_mean(deltas, samples=50_000, seed=seeds[endpoint]),
        }
    contrasts = _candidate_contrasts(candidate)
    policy_bootstrap = _cluster_bootstrap_mean(
        [{"game_seed": row["game_seed"], "value": row["policy_tv"]} for row in contrasts],
        samples=50_000,
        seed=bootstrap_seed_base + 3,
    )
    value_bootstrap = _cluster_bootstrap_mean(
        [{"game_seed": row["game_seed"], "value": row["abs_value_delta"]} for row in contrasts],
        samples=50_000,
        seed=bootstrap_seed_base + 4,
    )
    role_model, _identity = candidate_loader(candidate_checkpoint, device=config.device)
    if candidate_parameter_checks is None:
        parameter_checks = {
            "policy_role_nonzero": float(torch.linalg.vector_norm(role_model.policy_role).item()) > 0.0,
            "value_role_nonzero": float(torch.linalg.vector_norm(role_model.value_role).item()) > 0.0,
        }
    else:
        parameter_checks = candidate_parameter_checks(role_model)
    checks = {
        "teacher_reproduction_within_1e_9": all(delta <= 1e-9 for delta in reproduction.values()),
        "policy_overall_improves": endpoint_results["policy_loss"]["delta"] < 0.0,
        "policy_both_roles_improve": all(item["delta"] < 0.0 for item in endpoint_results["policy_loss"]["by_role"].values()),
        "policy_ci_high_below_zero": endpoint_results["policy_loss"]["bootstrap"]["ci95_high"] < 0.0,
        "value_overall_improves": endpoint_results["value_loss"]["delta"] < 0.0,
        "value_both_roles_improve": all(item["delta"] < 0.0 for item in endpoint_results["value_loss"]["by_role"].values()),
        "value_ci_high_below_zero": endpoint_results["value_loss"]["bootstrap"]["ci95_high"] < 0.0,
        "objective_ci_high_below_zero": endpoint_results["objective_loss"]["bootstrap"]["ci95_high"] < 0.0,
        "raw_policy_tv_at_least_0_025": policy_bootstrap["mean"] >= 0.025,
        "raw_policy_tv_ci_low_above_0_01": policy_bootstrap["ci95_low"] > 0.01,
        "raw_value_delta_at_least_0_05": value_bootstrap["mean"] >= 0.05,
        "raw_value_delta_ci_low_above_0_01": value_bootstrap["ci95_low"] > 0.01,
        "metrics_finite": _all_finite({"means": means, "endpoints": endpoint_results, "policy": policy_bootstrap, "value": value_bootstrap}),
    }
    checks.update(parameter_checks)
    return {
        "teacher_summary": summary,
        "teacher_reproduction_abs_error": reproduction,
        "loss_means": means,
        "selected_role_blind_baselines": selected,
        "endpoint_results": endpoint_results,
        "candidate_contrasts": {
            "mean_policy_tv": policy_bootstrap["mean"],
            "policy_tv_bootstrap": policy_bootstrap,
            "mean_abs_value_delta": value_bootstrap["mean"],
            "abs_value_delta_bootstrap": value_bootstrap,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _evaluate_safety(
    config,
    rows: list[AlphaZeroReplayRow],
    *,
    candidate_checkpoint: Path,
    candidate_name: str = "candidate_120",
    candidate_loader=load_readout_role_checkpoint,
) -> dict[str, Any]:
    from .start_role_audit import phase_for_turn

    groups: dict[str, list[AlphaZeroReplayRow]] = {"all": list(rows)}
    for role_name, role_id in (("start", 0), ("nonstart", 1)):
        role_rows = [row for row in rows if row.player_id == role_id]
        groups[f"role:{role_name}"] = role_rows
        for phase in ("early", "mid", "late"):
            groups[f"role:{role_name}|phase:{phase}"] = [
                row for row in role_rows if phase_for_turn(row.turn_index) == phase
            ]
    for phase in ("early", "mid", "late"):
        groups[f"phase:{phase}"] = [row for row in rows if phase_for_turn(row.turn_index) == phase]
    parent = load_checkpoint(config.generation_11_checkpoint, device=config.device)[0]
    candidate = candidate_loader(candidate_checkpoint, device=config.device)[0]
    metrics = {
        name: {
            "generation_11": _evaluate_replay_rows(parent, group, config.device, role_aware=False),
            candidate_name: _evaluate_replay_rows(candidate, group, config.device, role_aware=True),
        }
        for name, group in groups.items()
    }
    deltas = {
        name: {
            key: float(values[candidate_name][key]) - float(values["generation_11"][key])
            for key in ("policy_loss", "value_loss", "objective_loss", "policy_entropy", "value_mae")
        }
        for name, values in metrics.items()
    }
    checks = {
        "all_objective_nonregression": deltas["all"]["objective_loss"] <= 0.005,
        "start_objective_nonregression": deltas["role:start"]["objective_loss"] <= 0.005,
        "nonstart_objective_nonregression": deltas["role:nonstart"]["objective_loss"] <= 0.005,
        "all_metrics_finite": _all_finite({"metrics": metrics, "deltas": deltas}),
    }
    return {
        "rows": len(rows),
        "games": len({row.game_seed for row in rows}),
        "metrics": metrics,
        "deltas_candidate_minus_gen11": deltas,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _endpoint_separation_audit(model, rows, device: str) -> dict[str, Any]:
    batch = _collate_role_target_rows(rows)
    moved = {key: value.to(device) for key, value in batch.items()}
    logits, values = model(**_target_model_inputs(moved))
    policy, value, _objective = _target_losses(
        logits, values, moved["action_mask"], moved["policy_target"], moved["value_target"]
    )
    model.zero_grad(set_to_none=True)
    policy.mean().backward()
    policy_own = float(model.policy_role.grad.abs().sum().item()) if model.policy_role.grad is not None else 0.0
    policy_leak = float(model.value_role.grad.abs().max().item()) if model.value_role.grad is not None else 0.0
    model.zero_grad(set_to_none=True)
    logits, values = model(**_target_model_inputs(moved))
    _policy, value, _objective = _target_losses(
        logits, values, moved["action_mask"], moved["policy_target"], moved["value_target"]
    )
    value.mean().backward()
    value_own = float(model.value_role.grad.abs().sum().item()) if model.value_role.grad is not None else 0.0
    value_leak = float(model.policy_role.grad.abs().max().item()) if model.policy_role.grad is not None else 0.0
    checks = {
        "policy_gradient_nonzero": policy_own > 0.0,
        "value_gradient_nonzero": value_own > 0.0,
        "policy_to_value_leakage_zero": policy_leak == 0.0,
        "value_to_policy_leakage_zero": value_leak == 0.0,
        "parameters_remain_zero": float(model.policy_role.abs().max().item()) == 0.0 and float(model.value_role.abs().max().item()) == 0.0,
    }
    model.zero_grad(set_to_none=True)
    return {
        "policy_gradient_l1": policy_own,
        "value_gradient_l1": value_own,
        "policy_to_value_gradient_max_abs": policy_leak,
        "value_to_policy_gradient_max_abs": value_leak,
        "checks": checks,
    }


def _load_training_targets(directory: Path, states) -> list[dict[str, Any]]:
    records = []
    for shard in range(4):
        payload = json.loads((directory / f"shard_{shard}.json").read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        if (
            payload.get("completed") is not True
            or payload.get("completed_pairs") != 112
            or identity.get("protocol_sha256") != TARGET_PROTOCOL_SHA256
            or identity.get("panel") != "training"
        ):
            raise ValueError(f"120 sealed fitting target shard {shard} is invalid.")
        records.extend(payload["rows"])
    if len(records) != 448:
        raise ValueError("120 requires 448 sealed fitting target pairs.")
    if [record["identity"] for record in records] != [
        _state_identity(game_seed, state) for game_seed, state in states
    ]:
        raise ValueError("120 fitting target schedule mismatch.")
    return records


def _verify_inputs(config: SplitReadoutRolePilotConfig) -> None:
    _verify_118_base_inputs(config)
    observed = _observed_hashes(config)
    if observed != EXPECTED_HASHES:
        raise ValueError(f"120 frozen input hash mismatch: {observed}")
    result_119 = json.loads(config.result_119.read_text(encoding="utf-8"))
    result_118 = json.loads(config.result_118.read_text(encoding="utf-8"))
    if (
        result_119.get("status") != "complete"
        or result_119.get("decision")
        != "direct_split_additive_role_gradient_qualifies_freeze_minimal_pilot_no_training"
        or result_119.get("recommended_architecture") != "split_readout_additive"
        or result_119.get("training_performed") is not False
        or result_118.get("status") != "complete"
        or result_118.get("decision")
        != "minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay"
        or result_118.get("training", {}).get("target_manifest_sha256")
        != EXPECTED_HASHES["training_target_manifest"]
    ):
        raise ValueError("120 does not reproduce the finalized 118/119 lineage.")


def _observed_hashes(config) -> dict[str, str]:
    return {
        "result_119": _sha256(config.result_119),
        "result_118": _sha256(config.result_118),
        "result_117": _sha256(config.result_117),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "candidate_114": _sha256(config.candidate_114),
        "candidate_116": _sha256(config.candidate_116),
        "partition_manifest": _sha256(config.partition_manifest),
        "train_replay": _sha256(config.train_replay),
        "holdout_replay": _sha256(config.holdout_replay),
        "training_target_manifest": _target_manifest_hash(config.training_target_dir),
        "installed_checkpoint": _sha256(config.installed_checkpoint),
    }


def _jsonable_config(config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
