"""Guarded partial state-role representation pilot 122."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .network import load_checkpoint
from .replay import read_replay_jsonl
from .role_state_network import (
    create_identity_role_state_model,
    load_role_state_checkpoint,
    role_state_parameter_counts,
    role_state_trainable_names,
    save_role_state_checkpoint,
)
from .scaled_continuation import _sha256, _write_json
from .start_role_film_pilot import (
    _observed_hashes as _observed_121_hashes,
    _verify_inputs as _verify_121_inputs,
)
from .start_role_pilot import (
    BATCH_SIZE,
    SHUFFLE_SEED,
    VALUE_WEIGHT,
    _RoleTargetDataset,
    _all_finite,
    _collate_role_target_rows,
    _evaluate_target_loader,
    _seal,
    _target_losses,
    _target_manifest_hash,
    _target_model_inputs,
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
    _evaluate_confirmation,
    _evaluate_safety,
    _load_training_targets,
)


PROTOCOL_SHA256 = "4e4c179eaae1f2887e66b95b0f3eba101d6d299f331385a197da3c7dfd1b3550"
EXPECTED_RESULT_121_HASH = "f871427980697f4036c84b7617ba671db801563af6d6a0434f698a7d473d4e66"
EPOCHS = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0


@dataclass(frozen=True, slots=True)
class PartialStateRolePilotConfig:
    output_dir: Path
    result_121: Path
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


def preflight_partial_state_role_pilot_122(config) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    train_replay = read_replay_jsonl(config.train_replay)
    holdout_replay = read_replay_jsonl(config.holdout_replay)
    train_states, train_reconstruction = reconstruct_role_panel(
        train_replay, expected_games=224, label="122 fitting"
    )
    holdout_states, holdout_reconstruction = reconstruct_role_panel(
        holdout_replay, expected_games=32, label="122 protected confirmation"
    )
    targets = _load_training_targets(config.training_target_dir, train_states)
    oriented_rows = _training_rows_from_records(train_states, targets)
    split = split_pair_games(train_states)
    model, parent_identity = create_identity_role_state_model(
        config.generation_11_checkpoint, device=config.device
    )
    counts = role_state_parameter_counts(model)
    expected_counts = {
        "parent": 356_450,
        "frozen_parent": 281_666,
        "final_entity_block": 74_784,
        "role_input": 96,
        "total": 356_546,
        "trainable": 74_880,
    }
    if counts != expected_counts:
        raise ValueError(f"122 role-state model has wrong parameter counts: {counts}")
    trainable_names = role_state_trainable_names(model)
    expected_names = ["start_role_direction"] + [
        f"parent.entity_encoder.layers.1.{name}"
        for name, _parameter in model.parent.entity_encoder.layers[-1].named_parameters()
    ]
    if trainable_names != expected_names:
        raise ValueError(f"122 trainable scope mismatch: {trainable_names}")
    identity = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=model,
        states=train_states + holdout_states,
        device=config.device,
    )
    if identity["max_logit_abs_diff"] > 1e-7 or identity["max_value_abs_diff"] > 1e-7:
        raise ValueError("122 epoch-zero representation is not exact generation 11.")
    gradients = _joint_representation_gradient_audit(
        model, oriented_rows[:16], config.device
    )
    if not all(gradients["checks"].values()):
        raise ValueError(f"122 joint-gradient contract failed: {gradients['checks']}")

    epoch_zero = config.output_dir / "checkpoints" / "epoch_zero_identity.pt"
    save_role_state_checkpoint(
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
    restored, restored_identity = load_role_state_checkpoint(
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
        raise ValueError("122 epoch-zero checkpoint round-trip failed.")
    result = {
        "kind": "alphazero_partial_state_role_representation_pilot_122_preflight",
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
        "trainable_parameter_names": trainable_names,
        "identity_audit": identity,
        "joint_gradient_audit": gradients,
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


def run_partial_state_role_pilot_122(
    config, *, launch: bool = False
) -> dict[str, Any]:
    preflight = preflight_partial_state_role_pilot_122(config)
    if not launch:
        return preflight
    started = time.perf_counter()
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 122 result is not a valid completed result.")

    train_replay = read_replay_jsonl(config.train_replay)
    holdout_replay = read_replay_jsonl(config.holdout_replay)
    train_states, _ = reconstruct_role_panel(
        train_replay, expected_games=224, label="122 fitting"
    )
    holdout_states, _ = reconstruct_role_panel(
        holdout_replay, expected_games=32, label="122 protected confirmation"
    )
    targets = _load_training_targets(config.training_target_dir, train_states)
    result: dict[str, Any] = {
        "kind": "alphazero_partial_state_role_representation_pilot_122",
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
        result["decision"] = (
            "partial_state_representation_failed_development_no_protected_evaluation"
        )
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "protected_role_confirmation"
    result["protected_confirmation_accessed"] = True
    _write_progress(config.output_dir, result, started)
    confirmation_targets = run_or_load_target_panel(
        config,
        holdout_states,
        confirmation=True,
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="122",
    )
    confirmation = _evaluate_confirmation(
        config,
        holdout_states,
        confirmation_targets,
        candidate_checkpoint=Path(training["checkpoint"]),
        candidate_name="candidate_122",
        candidate_loader=load_role_state_checkpoint,
        candidate_parameter_checks=_representation_parameter_checks,
        bootstrap_seed_base=1_122_120,
    )
    result["role_confirmation"] = confirmation
    if not confirmation["passed"]:
        result["decision"] = "partial_state_representation_failed_protected_confirmation"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "reachable_safety"
    _write_progress(config.output_dir, result, started)
    safety = _evaluate_safety(
        config,
        holdout_replay,
        candidate_checkpoint=Path(training["checkpoint"]),
        candidate_name="candidate_122",
        candidate_loader=load_role_state_checkpoint,
    )
    result["reachable_safety"] = safety
    if not safety["passed"]:
        result["decision"] = (
            "partial_state_representation_confirmed_but_reachable_regression_no_gameplay"
        )
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "ordinary_budget_arena"
    _write_progress(config.output_dir, result, started)
    arena = run_or_load_arena(
        config,
        Path(training["checkpoint"]),
        training["checkpoint_sha256"],
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="122",
        seed_base=2_220_000,
        bootstrap_seed=1_122_200,
        arena_name="partial_state_role_representation_primary_256",
    )
    result["gameplay_arena"] = arena
    result["decision"] = (
        "partial_state_representation_confirmed_review_role_aware_continuation_no_install"
        if arena["assessment"]["passed"]
        else "partial_state_representation_offline_gain_not_converted_no_install"
    )
    return _seal(config.output_dir, result, started)


def _run_or_load_training(config, states, records) -> dict[str, Any]:
    stage = config.output_dir / "training" / "partial_state_role.stage.json"
    target_hash = _target_manifest_hash(config.training_target_dir)
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("target_manifest_sha256") != target_hash
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        ):
            raise ValueError("122 training stage identity mismatch.")
        return payload

    split = split_pair_games(states)
    validation_games = set(split["validation_games"])
    rows = _training_rows_from_records(states, records)
    train_rows = [row for row in rows if row.game_seed not in validation_games]
    validation_rows = [row for row in rows if row.game_seed in validation_games]
    if len(train_rows) != 716 or len(validation_rows) != 180:
        raise ValueError("122 fitting/development rows have wrong counts.")
    model, parent_identity = create_identity_role_state_model(
        config.generation_11_checkpoint, device=config.device
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
        train_metrics = _train_epoch(
            model, train_loader, optimizer, config.device, trainable
        )
        validation_metrics = _evaluate_target_loader(
            model, validation_loader, config.device
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        )
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
    selected_validation = _evaluate_target_loader(
        model, validation_loader, config.device
    )
    relative_improvement = (
        float(epoch_zero["objective_loss"])
        - float(selected_validation["objective_loss"])
    ) / float(epoch_zero["objective_loss"])
    movement = _parameter_movement(model, config.generation_11_checkpoint)
    trainable_names = role_state_trainable_names(model)
    expected_names = ["start_role_direction"] + [
        f"parent.entity_encoder.layers.1.{name}"
        for name, _parameter in model.parent.entity_encoder.layers[-1].named_parameters()
    ]
    role_norm = float(torch.linalg.vector_norm(model.start_role_direction).item())
    selection_checks = {
        "nonzero_selected_epoch": best_epoch > 0,
        "relative_objective_improvement_at_least_0_5pct": relative_improvement >= 0.005,
        "role_input_nonzero": role_norm > 0.0,
        "final_entity_block_changed": movement["final_entity_block_l2"] > 0.0,
        "frozen_parent_exact": movement["frozen_parent_max_abs"] == 0.0,
        "exact_trainable_scope": trainable_names == expected_names,
        "metrics_finite": _all_finite(
            {
                "history": history,
                "selected_train": selected_train,
                "selected_validation": selected_validation,
                "movement": movement,
                "role_norm": role_norm,
            }
        ),
    }
    selection_passed = all(selection_checks.values())
    checkpoint = config.output_dir / "checkpoints" / "partial_state_role_generation_012.pt"
    save_role_state_checkpoint(
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
            "partial_state_role_training": True,
        },
    )
    roundtrip, identity = load_role_state_checkpoint(checkpoint, device=config.device)
    if (
        identity.generation != 12
        or identity.metadata.get("selected_epoch") != best_epoch
        or role_state_parameter_counts(roundtrip)["trainable"] != 74_880
    ):
        raise RuntimeError("122 trained checkpoint failed round-trip validation.")
    metrics = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
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
        "role_input_norm": role_norm,
        "parameter_movement": movement,
        "history": history,
        "parameter_counts": role_state_parameter_counts(roundtrip),
        "trainable_parameter_names": trainable_names,
    }
    metrics_path = config.output_dir / "training" / "partial_state_role.json"
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


def _train_epoch(model, loader, optimizer, device: str, trainable) -> dict[str, float]:
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
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        optimizer.step()
        batch_count = int(values.shape[0])
        count += batch_count
        totals["policy_loss"] += float(policy.sum().item())
        totals["value_loss"] += float(value.sum().item())
        totals["objective_loss"] += float(objective.sum().item())
    model.eval()
    return {**{key: value / count for key, value in totals.items()}, "rows": count}


def _joint_representation_gradient_audit(model, rows, device: str) -> dict[str, Any]:
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = _collate_role_target_rows(rows)
    moved = {key: value.to(device) for key, value in batch.items()}
    vectors = {}
    details = {}
    for endpoint in ("policy", "value"):
        model.zero_grad(set_to_none=True)
        logits, values = model(**_target_model_inputs(moved))
        policy, value, _objective = _target_losses(
            logits,
            values,
            moved["action_mask"],
            moved["policy_target"],
            moved["value_target"],
        )
        (policy.mean() if endpoint == "policy" else value.mean()).backward()
        role_grad = model.start_role_direction.grad
        block_grads = [
            parameter.grad.reshape(-1)
            for parameter in model.parent.entity_encoder.layers[-1].parameters()
            if parameter.grad is not None
        ]
        trainable_grads = [
            parameter.grad.reshape(-1)
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        frozen_with_grad = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        vectors[endpoint] = torch.cat(trainable_grads).detach().cpu()
        details[endpoint] = {
            "role_gradient_l1": (
                float(role_grad.abs().sum().item()) if role_grad is not None else 0.0
            ),
            "final_block_gradient_l1": float(
                torch.cat(block_grads).abs().sum().item()
            ),
            "frozen_parameters_with_gradient": frozen_with_grad,
        }
    denominator = float(
        torch.linalg.vector_norm(vectors["policy"])
        * torch.linalg.vector_norm(vectors["value"])
    )
    cosine = (
        float(torch.dot(vectors["policy"], vectors["value"]).item()) / denominator
        if denominator > 0.0
        else float("nan")
    )
    unchanged = all(
        torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    checks = {
        "policy_role_gradient_nonzero": details["policy"]["role_gradient_l1"] > 0.0,
        "value_role_gradient_nonzero": details["value"]["role_gradient_l1"] > 0.0,
        "policy_final_block_gradient_nonzero": details["policy"]["final_block_gradient_l1"] > 0.0,
        "value_final_block_gradient_nonzero": details["value"]["final_block_gradient_l1"] > 0.0,
        "frozen_parameters_receive_no_gradient": not details["policy"]["frozen_parameters_with_gradient"]
        and not details["value"]["frozen_parameters_with_gradient"],
        "gradient_cosine_finite": math.isfinite(cosine),
        "parameters_unchanged": unchanged,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"endpoints": details, "joint_gradient_cosine": cosine, "checks": checks}


def _parameter_movement(model, parent_checkpoint: Path) -> dict[str, float]:
    parent, _identity = load_checkpoint(parent_checkpoint, device="cpu")
    reference = parent.state_dict()
    final_sq = 0.0
    frozen_max = 0.0
    for name, parameter in model.parent.state_dict().items():
        delta = parameter.detach().cpu() - reference[name].detach().cpu()
        if name.startswith("entity_encoder.layers.1."):
            final_sq += float(torch.sum(delta.double().square()).item())
        elif delta.numel():
            frozen_max = max(frozen_max, float(delta.abs().max().item()))
    return {
        "final_entity_block_l2": math.sqrt(final_sq),
        "frozen_parent_max_abs": frozen_max,
    }


def _representation_parameter_checks(model) -> dict[str, bool]:
    return {
        "role_input_nonzero": float(
            torch.linalg.vector_norm(model.start_role_direction).item()
        )
        > 0.0,
        "trainable_state_finite": all(
            bool(torch.isfinite(parameter).all())
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _verify_inputs(config) -> None:
    _verify_121_inputs(config)
    actual = _sha256(config.result_121)
    if actual != EXPECTED_RESULT_121_HASH:
        raise ValueError(f"122 frozen 121 result hash mismatch: {actual}")
    result = json.loads(config.result_121.read_text(encoding="utf-8"))
    if (
        result.get("status") != "complete"
        or result.get("decision")
        != "state_dependent_role_film_failed_development_close_small_role_adapters"
        or result.get("protected_confirmation_accessed") is not False
        or result.get("promotion_authorized") is not False
    ):
        raise ValueError("122 does not reproduce the finalized 121 decision.")


def _observed_hashes(config) -> dict[str, str]:
    return {"result_121": _sha256(config.result_121), **_observed_121_hashes(config)}


def _jsonable_config(config) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
