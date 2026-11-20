"""Guarded state-dependent split-FiLM start-role pilot 121."""

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

from .replay import read_replay_jsonl
from .role_film_network import (
    create_identity_role_film_model,
    load_role_film_checkpoint,
    role_film_parameter_counts,
    save_role_film_checkpoint,
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
    _collate_role_target_rows,
    _evaluate_target_loader,
    _seal,
    _target_manifest_hash,
    _training_rows_from_records,
    _write_progress,
    identity_audit,
    reconstruct_role_panel,
    run_or_load_arena,
    run_or_load_target_panel,
    split_pair_games,
    summarize_target_records,
)
from .start_role_readout_pilot import (
    _endpoint_separation_audit,
    _evaluate_confirmation,
    _evaluate_safety,
    _load_training_targets,
    _observed_hashes as _observed_120_hashes,
    _train_epoch,
    _verify_inputs as _verify_120_inputs,
)


PROTOCOL_SHA256 = "8fd2b022926b1e2e1d65200e53690381de570c2a2a4e3e9974056b2857beb0f4"
EXPECTED_RESULT_120_HASH = "3e1ef85c2b4a8231694a60e089e0b366ceba299c5a1925de116657d420ada3cd"


@dataclass(frozen=True, slots=True)
class StateDependentRoleFiLMPilotConfig:
    output_dir: Path
    result_120: Path
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


def preflight_state_dependent_role_film_pilot_121(config) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    train_replay = read_replay_jsonl(config.train_replay)
    holdout_replay = read_replay_jsonl(config.holdout_replay)
    train_states, train_reconstruction = reconstruct_role_panel(
        train_replay, expected_games=224, label="121 fitting"
    )
    holdout_states, holdout_reconstruction = reconstruct_role_panel(
        holdout_replay, expected_games=32, label="121 protected confirmation"
    )
    targets = _load_training_targets(config.training_target_dir, train_states)
    oriented_rows = _training_rows_from_records(train_states, targets)
    split = split_pair_games(train_states)
    model, parent_identity = create_identity_role_film_model(
        config.generation_11_checkpoint, device=config.device
    )
    counts = role_film_parameter_counts(model)
    expected_counts = {
        "parent": 356_450,
        "policy_gain": 96,
        "policy_bias": 96,
        "value_gain": 96,
        "value_bias": 96,
        "role_total": 384,
        "total": 356_834,
        "trainable": 384,
    }
    if counts != expected_counts:
        raise ValueError(f"121 FiLM model has wrong parameter counts: {counts}")
    identity = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=model,
        states=train_states + holdout_states,
        device=config.device,
    )
    if identity["max_logit_abs_diff"] > 1e-7 or identity["max_value_abs_diff"] > 1e-7:
        raise ValueError("121 epoch-zero FiLM bridge is not exact gen11.")
    separation = _endpoint_separation_audit(model, oriented_rows[:16], config.device)
    if not all(separation["checks"].values()):
        raise ValueError("121 endpoint separation contract failed.")
    epoch_zero = config.output_dir / "checkpoints" / "epoch_zero_identity.pt"
    save_role_film_checkpoint(
        epoch_zero,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint_sha256": (
                "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0"
            ),
            "parent_generation": parent_identity.generation,
            "selected_epoch": 0,
            "preflight_only": True,
        },
    )
    restored, restored_identity = load_role_film_checkpoint(epoch_zero, device=config.device)
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
        raise ValueError("121 epoch-zero checkpoint round-trip failed.")
    result = {
        "kind": "alphazero_state_dependent_role_film_pilot_121_preflight",
        "status": "preflight_complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": _observed_hashes(config),
        "train_reconstruction": train_reconstruction,
        "protected_reconstruction": holdout_reconstruction,
        "split": split,
        "target_summary": summarize_target_records(targets),
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


def run_state_dependent_role_film_pilot_121(config, *, launch: bool = False) -> dict[str, Any]:
    preflight = preflight_state_dependent_role_film_pilot_121(config)
    if not launch:
        return preflight
    started = time.perf_counter()
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 121 result is not a valid completed result.")
    train_replay = read_replay_jsonl(config.train_replay)
    holdout_replay = read_replay_jsonl(config.holdout_replay)
    train_states, _ = reconstruct_role_panel(train_replay, expected_games=224, label="121 fitting")
    holdout_states, _ = reconstruct_role_panel(
        holdout_replay, expected_games=32, label="121 protected confirmation"
    )
    targets = _load_training_targets(config.training_target_dir, train_states)
    result: dict[str, Any] = {
        "kind": "alphazero_state_dependent_role_film_pilot_121",
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
        result["decision"] = "state_dependent_role_film_failed_development_close_small_role_adapters"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "protected_role_confirmation"
    result["protected_confirmation_accessed"] = True
    _write_progress(config.output_dir, result, started)
    confirmation_targets = run_or_load_target_panel(
        config,
        holdout_states,
        confirmation=True,
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="121",
    )
    confirmation = _evaluate_confirmation(
        config,
        holdout_states,
        confirmation_targets,
        candidate_checkpoint=Path(training["checkpoint"]),
        candidate_name="candidate_121",
        candidate_loader=load_role_film_checkpoint,
        candidate_parameter_checks=_film_parameter_checks,
        bootstrap_seed_base=1_121_120,
    )
    result["role_confirmation"] = confirmation
    if not confirmation["passed"]:
        result["decision"] = "state_dependent_role_film_failed_confirmation_close_small_role_adapters"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "reachable_safety"
    _write_progress(config.output_dir, result, started)
    safety = _evaluate_safety(
        config,
        holdout_replay,
        candidate_checkpoint=Path(training["checkpoint"]),
        candidate_name="candidate_121",
        candidate_loader=load_role_film_checkpoint,
    )
    result["reachable_safety"] = safety
    if not safety["passed"]:
        result["decision"] = "state_dependent_role_film_confirmed_but_reachable_regression_no_gameplay"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "ordinary_budget_arena"
    _write_progress(config.output_dir, result, started)
    arena = run_or_load_arena(
        config,
        Path(training["checkpoint"]),
        training["checkpoint_sha256"],
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="121",
        seed_base=2_210_000,
        bootstrap_seed=1_121_200,
        arena_name="state_dependent_role_film_primary_256",
    )
    result["gameplay_arena"] = arena
    result["decision"] = (
        "state_dependent_role_film_confirmed_review_short_continuation_no_install"
        if arena["assessment"]["passed"]
        else "state_dependent_role_film_offline_gain_not_converted_no_install"
    )
    return _seal(config.output_dir, result, started)


def _run_or_load_training(config, states, records) -> dict[str, Any]:
    stage = config.output_dir / "training" / "role_film.stage.json"
    target_hash = _target_manifest_hash(config.training_target_dir)
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("target_manifest_sha256") != target_hash
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        ):
            raise ValueError("121 training stage identity mismatch.")
        return payload
    split = split_pair_games(states)
    validation_games = set(split["validation_games"])
    rows = _training_rows_from_records(states, records)
    train_rows = [row for row in rows if row.game_seed not in validation_games]
    validation_rows = [row for row in rows if row.game_seed in validation_games]
    if len(train_rows) != 716 or len(validation_rows) != 180:
        raise ValueError("121 fitting/development rows have wrong counts.")
    model, parent_identity = create_identity_role_film_model(
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
    history = [{"epoch": 0, "train": None, "validation": epoch_zero}]
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
    norms = _film_norms(model)
    selection_checks = {
        "nonzero_selected_epoch": best_epoch > 0,
        "relative_objective_improvement_at_least_0_5pct": relative_improvement >= 0.005,
        **{f"{name}_nonzero": value > 0.0 for name, value in norms.items()},
        "parent_frozen": all(not parameter.requires_grad for parameter in model.parent.parameters()),
        "metrics_finite": _all_finite(
            {"history": history, "selected_train": selected_train, "selected_validation": selected_validation}
        ),
    }
    selection_passed = all(selection_checks.values())
    checkpoint = config.output_dir / "checkpoints" / "role_film_generation_012.pt"
    save_role_film_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "parent_checkpoint_sha256": (
                "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0"
            ),
            "parent_generation": parent_identity.generation,
            "selected_epoch": best_epoch,
            "train_games": split["train_games"],
            "validation_games": split["validation_games"],
            "state_dependent_role_film_training": True,
        },
    )
    roundtrip, identity = load_role_film_checkpoint(checkpoint, device=config.device)
    if (
        identity.generation != 12
        or identity.metadata.get("selected_epoch") != best_epoch
        or role_film_parameter_counts(roundtrip)["trainable"] != 384
    ):
        raise RuntimeError("121 trained checkpoint failed round-trip validation.")
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
        "film_group_norms": norms,
        "history": history,
        "parameter_counts": role_film_parameter_counts(roundtrip),
    }
    metrics_path = config.output_dir / "training" / "role_film.json"
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


def _film_norms(model) -> dict[str, float]:
    return {
        "policy_gain": float(torch.linalg.vector_norm(model.policy_role[0]).item()),
        "policy_bias": float(torch.linalg.vector_norm(model.policy_role[1]).item()),
        "value_gain": float(torch.linalg.vector_norm(model.value_role[0]).item()),
        "value_bias": float(torch.linalg.vector_norm(model.value_role[1]).item()),
    }


def _film_parameter_checks(model) -> dict[str, bool]:
    return {f"{name}_nonzero": value > 0.0 for name, value in _film_norms(model).items()}


def _verify_inputs(config) -> None:
    _verify_120_inputs(config)
    actual = _sha256(config.result_120)
    if actual != EXPECTED_RESULT_120_HASH:
        raise ValueError(f"121 frozen 120 result hash mismatch: {actual}")
    result = json.loads(config.result_120.read_text(encoding="utf-8"))
    if (
        result.get("status") != "complete"
        or result.get("decision") != "split_readout_role_failed_development_uptake_no_confirmation"
        or result.get("protected_confirmation_accessed") is not False
        or result.get("promotion_authorized") is not False
    ):
        raise ValueError("121 does not reproduce the finalized 120 decision.")


def _observed_hashes(config) -> dict[str, str]:
    return {"result_120": _sha256(config.result_120), **_observed_120_hashes(config)}


def _jsonable_config(config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
