"""Guarded paired-objective start-role training pilot 125."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .replay import read_replay_jsonl
from .role_state_network import (
    create_identity_role_state_model,
    load_role_state_checkpoint,
    role_state_parameter_counts,
    role_state_trainable_names,
    save_role_state_checkpoint,
)
from .scaled_continuation import _sha256, _write_json
from .start_role_interaction_audit import _collate_pairs
from .start_role_pilot import (
    SHUFFLE_SEED,
    _all_finite,
    _seal,
    _state_identity,
    _target_manifest_hash,
    _training_rows_from_records,
    _write_progress,
    identity_audit,
    reconstruct_role_panel,
    run_or_load_arena,
    split_pair_games,
    summarize_target_records,
)
from .start_role_readout_pilot import (
    _evaluate_confirmation,
    _evaluate_safety,
    _load_training_targets,
)
from .start_role_state_pilot import (
    _jsonable_config,
    _observed_hashes as _observed_122_hashes,
    _parameter_movement,
    _representation_parameter_checks,
    _verify_inputs as _verify_122_inputs,
)


PROTOCOL_SHA256 = "d5f5cced1e74255f981f6e1c8471ccae49244c7ee9bc5c212753e54a4988ed4d"
EXPECTED_RESULT_124_HASH = "29437a929887c9e34a51c52de218df82fb3c2b0f06298c92e240c001364b18d9"
EXPECTED_CONFIRMATION_TARGET_MANIFEST = (
    "e9289abc0d8068558534596aca30e3c480a413aab822bff96b50aca44e8aa0fe"
)
EXPECTED_GEN11_HASH = "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0"
EXPECTED_122_PROTOCOL = "4e4c179eaae1f2887e66b95b0f3eba101d6d299f331385a197da3c7dfd1b3550"
LAMBDA_POLICY = 702.476696552303
LAMBDA_VALUE = 39.4241527478651
EPOCHS = 12
PAIR_BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0


@dataclass(frozen=True, slots=True)
class PairedObjectiveRolePilotConfig:
    output_dir: Path
    result_124: Path
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
    confirmation_target_dir: Path
    installed_checkpoint: Path
    device: str = "cpu"
    torch_threads: int = 1
    workers: int = 4
    arena_workers: int = 4
    max_turns: int = 200


class _PairDataset(Dataset):
    def __init__(self, pairs) -> None:
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        return self.pairs[index]


def _collate_pair_rows(pairs):
    return _collate_pairs(list(pairs), "cpu")


def preflight_paired_objective_role_pilot_125(config) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _verify_inputs(config)
    replay = read_replay_jsonl(config.train_replay)
    states, reconstruction = reconstruct_role_panel(
        replay, expected_games=224, label="125 fitting-only preflight"
    )
    records = _load_training_targets(config.training_target_dir, states)
    rows = _training_rows_from_records(states, records)
    pairs = _paired_rows(rows)
    split = split_pair_games(states)
    pair_contract = _pair_contract(pairs)
    if not all(pair_contract["checks"].values()):
        raise ValueError(f"125 fitting pair contract failed: {pair_contract}")

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
        raise ValueError(f"125 model parameter counts mismatch: {counts}")
    identity = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=model,
        states=states,
        device=config.device,
    )
    if identity["max_logit_abs_diff"] > 1e-7 or identity["max_value_abs_diff"] > 1e-7:
        raise ValueError("125 epoch-zero model is not exact generation 11.")
    gradient_audit = _joint_gradient_audit(model, pairs[:16], config.device)
    if not all(gradient_audit["checks"].values()):
        raise ValueError(f"125 joint-objective gradient audit failed: {gradient_audit}")

    epoch_zero = config.output_dir / "checkpoints" / "epoch_zero_identity.pt"
    save_role_state_checkpoint(
        epoch_zero,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint_sha256": EXPECTED_GEN11_HASH,
            "parent_generation": parent_identity.generation,
            "selected_epoch": 0,
            "preflight_only": True,
            "paired_objective": True,
        },
    )
    restored, restored_identity = load_role_state_checkpoint(
        epoch_zero, device=config.device
    )
    roundtrip = identity_audit(
        parent_checkpoint=config.generation_11_checkpoint,
        role_model=restored,
        states=states[:8],
        device=config.device,
    )
    if (
        restored_identity.generation != 12
        or restored_identity.metadata.get("selected_epoch") != 0
        or roundtrip["max_logit_abs_diff"] > 1e-7
        or roundtrip["max_value_abs_diff"] > 1e-7
    ):
        raise ValueError("125 epoch-zero checkpoint round-trip failed.")
    result = {
        "kind": "alphazero_paired_objective_role_pilot_125_preflight",
        "status": "preflight_complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": _jsonable_config(config),
        "input_hashes": _observed_hashes(config),
        "reconstruction": reconstruction,
        "split": split,
        "target_summary": summarize_target_records(records),
        "target_rows_validated": len(rows),
        "pair_contract": pair_contract,
        "parameter_counts": counts,
        "trainable_parameter_names": role_state_trainable_names(model),
        "identity_audit": identity,
        "joint_gradient_audit": gradient_audit,
        "epoch_zero_checkpoint": str(epoch_zero),
        "epoch_zero_checkpoint_sha256": _sha256(epoch_zero),
        "roundtrip_identity_audit": roundtrip,
        "training_performed": False,
        "protected_states_reconstructed": False,
        "confirmation_target_manifest_accessed": False,
        "search_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
        "launch_required": True,
    }
    _write_json(config.output_dir / "preflight.json", result)
    return result


def run_paired_objective_role_pilot_125(
    config, *, launch: bool = False
) -> dict[str, Any]:
    preflight = preflight_paired_objective_role_pilot_125(config)
    if not launch:
        return preflight
    started = time.perf_counter()
    result_path = config.output_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("protocol_sha256") == PROTOCOL_SHA256:
            return payload
        raise ValueError("Existing 125 result is not a valid completed result.")

    replay = read_replay_jsonl(config.train_replay)
    states, _reconstruction = reconstruct_role_panel(
        replay, expected_games=224, label="125 fitting"
    )
    records = _load_training_targets(config.training_target_dir, states)
    pairs = _paired_rows(_training_rows_from_records(states, records))
    result: dict[str, Any] = {
        "kind": "alphazero_paired_objective_role_pilot_125",
        "status": "running",
        "active_stage": "development_fitting",
        "protocol_sha256": PROTOCOL_SHA256,
        "input_hashes": _observed_hashes(config),
        "preflight": preflight,
        "existing_fitting_targets_reused": True,
        "new_fitting_targets_generated": False,
        "training_performed": False,
        "protected_states_reconstructed": False,
        "confirmation_target_manifest_accessed": False,
        "confirmation_targets_reused": False,
        "new_confirmation_search_generated": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_progress(config.output_dir, result, started)
    training = _run_or_load_training(config, states, pairs)
    result["training_performed"] = True
    result["training"] = training
    if not training["selection_passed"]:
        result["decision"] = "paired_objective_role_pilot_failed_development_no_confirmation"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "protected_role_confirmation"
    result["protected_states_reconstructed"] = True
    result["confirmation_target_manifest_accessed"] = True
    result["confirmation_targets_reused"] = True
    _write_progress(config.output_dir, result, started)
    holdout = read_replay_jsonl(config.holdout_replay)
    holdout_states, protected_reconstruction = reconstruct_role_panel(
        holdout, expected_games=32, label="125 protected confirmation"
    )
    confirmation_targets = _load_confirmation_targets(config, holdout_states)
    result["protected_reconstruction"] = protected_reconstruction
    result["confirmation_target_manifest_sha256"] = _target_manifest_hash(
        config.confirmation_target_dir
    )
    confirmation = _evaluate_confirmation(
        config,
        holdout_states,
        confirmation_targets,
        candidate_checkpoint=Path(training["checkpoint"]),
        candidate_name="candidate_125",
        candidate_loader=load_role_state_checkpoint,
        candidate_parameter_checks=_representation_parameter_checks,
        bootstrap_seed_base=1_125_120,
    )
    result["role_confirmation"] = confirmation
    if not confirmation["passed"]:
        result["decision"] = "paired_objective_role_pilot_failed_protected_confirmation"
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "reachable_safety"
    _write_progress(config.output_dir, result, started)
    safety = _evaluate_safety(
        config,
        holdout,
        candidate_checkpoint=Path(training["checkpoint"]),
        candidate_name="candidate_125",
        candidate_loader=load_role_state_checkpoint,
    )
    result["reachable_safety"] = safety
    if not safety["passed"]:
        result["decision"] = (
            "paired_objective_role_pilot_confirmed_but_reachable_regression_no_gameplay"
        )
        return _seal(config.output_dir, result, started)

    result["active_stage"] = "ordinary_budget_arena"
    _write_progress(config.output_dir, result, started)
    arena = run_or_load_arena(
        config,
        Path(training["checkpoint"]),
        training["checkpoint_sha256"],
        protocol_sha256=PROTOCOL_SHA256,
        experiment_label="125",
        seed_base=2_250_000,
        bootstrap_seed=1_125_200,
        arena_name="paired_objective_role_primary_256",
    )
    result["gameplay_arena"] = arena
    result["decision"] = (
        "paired_objective_role_pilot_confirmed_review_role_aware_continuation_no_install"
        if arena["assessment"]["passed"]
        else "paired_objective_role_pilot_offline_gain_not_converted_no_install"
    )
    return _seal(config.output_dir, result, started)


def _run_or_load_training(config, states, pairs) -> dict[str, Any]:
    stage = config.output_dir / "training" / "paired_objective.stage.json"
    target_hash = _target_manifest_hash(config.training_target_dir)
    if stage.exists():
        payload = json.loads(stage.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_sha256") != PROTOCOL_SHA256
            or payload.get("target_manifest_sha256") != target_hash
            or payload.get("checkpoint_sha256") != _sha256(payload["checkpoint"])
        ):
            raise ValueError("125 training stage identity mismatch.")
        return payload

    split = split_pair_games(states)
    development_games = set(split["validation_games"])
    train_pairs = [pair for pair in pairs if pair[0].game_seed not in development_games]
    development_pairs = [pair for pair in pairs if pair[0].game_seed in development_games]
    if (len(train_pairs), len(development_pairs)) != (358, 90):
        raise ValueError("125 fitting/development pair counts are wrong.")
    model, parent_identity = create_identity_role_state_model(
        config.generation_11_checkpoint, device=config.device
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = DataLoader(
        _PairDataset(train_pairs),
        batch_size=PAIR_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SHUFFLE_SEED),
        collate_fn=_collate_pair_rows,
    )
    development_loader = DataLoader(
        _PairDataset(development_pairs),
        batch_size=PAIR_BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate_pair_rows,
    )
    epoch_zero = _evaluate_pair_loader(model, development_loader, config.device)
    best_epoch = 0
    best_joint = float(epoch_zero["joint_objective"])
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "train": None, "development": epoch_zero}]
    for epoch in range(1, EPOCHS + 1):
        train_metrics = _train_epoch(
            model, train_loader, optimizer, trainable, config.device
        )
        development = _evaluate_pair_loader(
            model, development_loader, config.device
        )
        history.append(
            {"epoch": epoch, "train": train_metrics, "development": development}
        )
        if float(development["joint_objective"]) < best_joint:
            best_epoch = epoch
            best_joint = float(development["joint_objective"])
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    selected_train = _evaluate_pair_loader(
        model,
        DataLoader(
            _PairDataset(train_pairs),
            batch_size=PAIR_BATCH_SIZE,
            shuffle=False,
            collate_fn=_collate_pair_rows,
        ),
        config.device,
    )
    selected_development = _evaluate_pair_loader(
        model, development_loader, config.device
    )
    ordinary_improvement = _relative_reduction(
        epoch_zero["ordinary_objective"], selected_development["ordinary_objective"]
    )
    policy_contrast_reduction = _relative_reduction(
        epoch_zero["canonical_policy_contrast"],
        selected_development["canonical_policy_contrast"],
    )
    value_contrast_reduction = _relative_reduction(
        epoch_zero["value_contrast"], selected_development["value_contrast"]
    )
    summed_contrast_reduction = _relative_reduction(
        epoch_zero["summed_policy_contrast"],
        selected_development["summed_policy_contrast"],
    )
    movement = _parameter_movement(model, config.generation_11_checkpoint)
    role_norm = float(torch.linalg.vector_norm(model.start_role_direction).item())
    expected_names = ["start_role_direction"] + [
        f"parent.entity_encoder.layers.1.{name}"
        for name, _parameter in model.parent.entity_encoder.layers[-1].named_parameters()
    ]
    checks = {
        "nonzero_selected_epoch": best_epoch > 0,
        "ordinary_objective_relative_improvement_at_least_0_5pct": ordinary_improvement
        >= 0.005,
        "policy_contrast_relative_reduction_at_least_10pct": policy_contrast_reduction
        >= 0.10,
        "value_contrast_relative_reduction_at_least_10pct": value_contrast_reduction
        >= 0.10,
        "raw_policy_tv_at_least_0_025": selected_development["mean_raw_policy_tv"]
        >= 0.025,
        "raw_abs_value_delta_at_least_0_05": selected_development[
            "mean_raw_abs_value_delta"
        ]
        >= 0.05,
        "summed_policy_contrast_improves": summed_contrast_reduction > 0.0,
        "role_input_nonzero": role_norm > 0.0,
        "final_entity_block_changed": movement["final_entity_block_l2"] > 0.0,
        "frozen_parent_exact": movement["frozen_parent_max_abs"] == 0.0,
        "exact_trainable_scope": role_state_trainable_names(model) == expected_names,
        "metrics_finite": _all_finite(
            {
                "history": history,
                "selected_train": selected_train,
                "selected_development": selected_development,
                "movement": movement,
                "role_norm": role_norm,
            }
        ),
    }
    selection_passed = all(checks.values())
    checkpoint = config.output_dir / "checkpoints" / "paired_objective_generation_012.pt"
    save_role_state_checkpoint(
        checkpoint,
        model=model,
        generation=12,
        metadata={
            "protocol_sha256": PROTOCOL_SHA256,
            "parent_checkpoint": str(config.generation_11_checkpoint),
            "parent_checkpoint_sha256": EXPECTED_GEN11_HASH,
            "parent_generation": parent_identity.generation,
            "selected_epoch": best_epoch,
            "train_games": split["train_games"],
            "validation_games": split["validation_games"],
            "paired_objective_training": True,
            "lambda_policy": LAMBDA_POLICY,
            "lambda_value": LAMBDA_VALUE,
        },
    )
    roundtrip, identity = load_role_state_checkpoint(checkpoint, device=config.device)
    if (
        identity.generation != 12
        or identity.metadata.get("selected_epoch") != best_epoch
        or role_state_parameter_counts(roundtrip)["trainable"] != 74_880
    ):
        raise RuntimeError("125 trained checkpoint round-trip failed.")
    metrics = {
        "config": {
            "epochs": EPOCHS,
            "pair_batch_size": PAIR_BATCH_SIZE,
            "oriented_rows_per_full_batch": 2 * PAIR_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "shuffle_seed": SHUFFLE_SEED,
            "lambda_policy": LAMBDA_POLICY,
            "lambda_value": LAMBDA_VALUE,
        },
        "split": split,
        "selected_epoch": best_epoch,
        "epoch_zero_development": epoch_zero,
        "selected_train": selected_train,
        "selected_development": selected_development,
        "ordinary_objective_relative_improvement": ordinary_improvement,
        "canonical_policy_contrast_relative_reduction": policy_contrast_reduction,
        "value_contrast_relative_reduction": value_contrast_reduction,
        "summed_policy_contrast_relative_reduction": summed_contrast_reduction,
        "selection_checks": checks,
        "selection_passed": selection_passed,
        "role_input_norm": role_norm,
        "parameter_movement": movement,
        "history": history,
        "parameter_counts": role_state_parameter_counts(roundtrip),
        "trainable_parameter_names": role_state_trainable_names(model),
    }
    metrics_path = config.output_dir / "training" / "paired_objective.json"
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


def _joint_losses(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    logits, values = model(**batch["inputs"])
    mask = batch["inputs"]["action_mask"]
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    log_probabilities = torch.log_softmax(masked, dim=-1)
    ordinary_policy = -(
        batch["target_policy"] * log_probabilities.masked_fill(~mask, 0.0)
    ).sum(dim=-1).mean()
    ordinary_value = (values - batch["target_value"]).square().mean()
    probabilities = torch.softmax(masked, dim=-1)
    pair_count = probabilities.shape[0] // 2
    width = probabilities.shape[1]
    roles = batch["inputs"]["observer_is_start_player"].to(probabilities.dtype)
    signed = roles.mul(2.0).sub(1.0).reshape(pair_count, 2, 1)
    predicted_policy_delta = (
        probabilities.reshape(pair_count, 2, width) * signed
    ).sum(dim=1)
    target_policy_delta = (
        batch["target_policy"].reshape(pair_count, 2, width) * signed
    ).sum(dim=1)
    pair_mask = mask.reshape(pair_count, 2, width).all(dim=1)
    policy_error = (predicted_policy_delta - target_policy_delta).square() * pair_mask
    summed_policy_contrast = policy_error.sum(dim=1).mean()
    canonical_policy_contrast = (
        policy_error.sum(dim=1)
        / pair_mask.sum(dim=1).clamp_min(1).to(policy_error.dtype)
    ).mean()
    value_signed = signed.squeeze(-1)
    predicted_value_delta = (
        values.reshape(pair_count, 2) * value_signed
    ).sum(dim=1)
    target_value_delta = (
        batch["target_value"].reshape(pair_count, 2) * value_signed
    ).sum(dim=1)
    value_contrast = (predicted_value_delta - target_value_delta).square().mean()
    raw_policy_tv = 0.5 * (
        probabilities.reshape(pair_count, 2, width)[:, 0]
        - probabilities.reshape(pair_count, 2, width)[:, 1]
    ).abs().mul(pair_mask).sum(dim=1).mean()
    raw_value_delta = (
        values.reshape(pair_count, 2)[:, 0] - values.reshape(pair_count, 2)[:, 1]
    ).abs().mean()
    ordinary_objective = ordinary_policy + 0.5 * ordinary_value
    joint = (
        ordinary_objective
        + LAMBDA_POLICY * summed_policy_contrast
        + 0.5 * LAMBDA_VALUE * value_contrast
    )
    return {
        "ordinary_policy": ordinary_policy,
        "ordinary_value": ordinary_value,
        "ordinary_objective": ordinary_objective,
        "summed_policy_contrast": summed_policy_contrast,
        "canonical_policy_contrast": canonical_policy_contrast,
        "value_contrast": value_contrast,
        "mean_raw_policy_tv": raw_policy_tv,
        "mean_raw_abs_value_delta": raw_value_delta,
        "joint_objective": joint,
    }


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "inputs": {key: value.to(device) for key, value in batch["inputs"].items()},
        "target_policy": batch["target_policy"].to(device),
        "target_value": batch["target_value"].to(device),
    }


def _train_epoch(model, loader, optimizer, trainable, device: str) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        moved = _move_batch(batch, device)
        losses = _joint_losses(model, moved)
        optimizer.zero_grad(set_to_none=True)
        losses["joint_objective"].backward()
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        optimizer.step()
        pairs = int(moved["target_value"].shape[0] // 2)
        count += pairs
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().item()) * pairs
    model.eval()
    return {**{name: value / count for name, value in totals.items()}, "pairs": count}


def _evaluate_pair_loader(model, loader, device: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch(batch, device)
            losses = _joint_losses(model, moved)
            pairs = int(moved["target_value"].shape[0] // 2)
            count += pairs
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.item()) * pairs
    return {**{name: value / count for name, value in totals.items()}, "pairs": count}


def _joint_gradient_audit(model, pairs, device: str) -> dict[str, Any]:
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    batch = _move_batch(_collate_pair_rows(pairs), device)
    model.zero_grad(set_to_none=True)
    losses = _joint_losses(model, batch)
    losses["joint_objective"].backward()
    role_l1 = float(model.start_role_direction.grad.abs().sum().item())
    block_l1 = float(
        torch.cat(
            [
                parameter.grad.reshape(-1)
                for parameter in model.parent.entity_encoder.layers[-1].parameters()
            ]
        )
        .abs()
        .sum()
        .item()
    )
    frozen_with_grad = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    unchanged = all(
        torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    checks = {
        "joint_loss_finite": bool(torch.isfinite(losses["joint_objective"])),
        "role_gradient_nonzero": role_l1 > 0.0,
        "final_block_gradient_nonzero": block_l1 > 0.0,
        "frozen_parameters_receive_no_gradient": not frozen_with_grad,
        "parameters_unchanged": unchanged,
        "lambda_policy_exact": LAMBDA_POLICY == 702.476696552303,
        "lambda_value_exact": LAMBDA_VALUE == 39.4241527478651,
    }
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "losses": {name: float(value.detach().item()) for name, value in losses.items()},
        "role_gradient_l1": role_l1,
        "final_block_gradient_l1": block_l1,
        "frozen_parameters_with_gradient": frozen_with_grad,
        "checks": checks,
    }


def _paired_rows(rows):
    if len(rows) % 2:
        raise ValueError("125 oriented fitting rows are not paired.")
    return [(rows[index], rows[index + 1]) for index in range(0, len(rows), 2)]


def _pair_contract(pairs) -> dict[str, Any]:
    same_game = all(first.game_seed == second.game_seed for first, second in pairs)
    opposite_roles = all(
        first.observer_is_start_player != second.observer_is_start_player
        for first, second in pairs
    )
    same_actions = all(first.actions == second.actions for first, second in pairs)
    widths = [len(first.actions) for first, _second in pairs]
    return {
        "pairs": len(pairs),
        "oriented_rows": 2 * len(pairs),
        "minimum_actions": min(widths),
        "maximum_actions": max(widths),
        "checks": {
            "exact_pair_count": len(pairs) == 448,
            "same_game_within_pair": same_game,
            "opposite_roles_within_pair": opposite_roles,
            "same_semantic_actions_within_pair": same_actions,
        },
    }


def _relative_reduction(before: float, after: float) -> float:
    return (float(before) - float(after)) / float(before) if float(before) > 0.0 else 0.0


def _load_confirmation_targets(config, states) -> list[dict[str, Any]]:
    actual = _target_manifest_hash(config.confirmation_target_dir)
    if actual != EXPECTED_CONFIRMATION_TARGET_MANIFEST:
        raise ValueError(f"125 confirmation target manifest mismatch: {actual}")
    records: list[dict[str, Any]] = []
    for shard in range(4):
        payload = json.loads(
            (config.confirmation_target_dir / f"shard_{shard}.json").read_text(
                encoding="utf-8"
            )
        )
        identity = payload.get("identity", {})
        if (
            payload.get("completed") is not True
            or payload.get("completed_pairs") != 16
            or payload.get("target_pairs") != 16
            or identity.get("protocol_sha256") != EXPECTED_122_PROTOCOL
            or identity.get("panel") != "confirmation"
            or identity.get("checkpoint_sha256") != EXPECTED_GEN11_HASH
            or identity.get("simulations") != 1024
        ):
            raise ValueError(f"125 confirmation shard {shard} is invalid.")
        records.extend(payload["rows"])
    if len(records) != 64:
        raise ValueError("125 requires exactly 64 confirmation target pairs.")
    expected = [_state_identity(game_seed, state) for game_seed, state in states]
    if [record["identity"] for record in records] != expected:
        raise ValueError("125 confirmation target schedule mismatch.")
    return records


def _verify_inputs(config) -> None:
    _verify_122_inputs(config)
    actual = _sha256(config.result_124)
    if actual != EXPECTED_RESULT_124_HASH:
        raise ValueError(f"125 frozen 124 result hash mismatch: {actual}")
    result = json.loads(config.result_124.read_text(encoding="utf-8"))
    recommendation = result.get("recommended_future_objective") or {}
    if (
        result.get("status") != "complete"
        or result.get("decision")
        != "summed_policy_contrast_qualifies_freeze_joint_objective_pilot_no_training"
        or result.get("selected_policy_contrast")
        != "summed_probability_difference_mse"
        or result.get("training_performed") is not False
        or result.get("protected_117_panel_read") is not False
        or abs(float(recommendation.get("lambda_policy", 0.0)) - LAMBDA_POLICY) > 1e-12
        or abs(float(recommendation.get("lambda_value", 0.0)) - LAMBDA_VALUE) > 1e-12
    ):
        raise ValueError("125 does not reproduce finalized 124.")


def _observed_hashes(config) -> dict[str, str]:
    return {"result_124": _sha256(config.result_124), **_observed_122_hashes(config)}
