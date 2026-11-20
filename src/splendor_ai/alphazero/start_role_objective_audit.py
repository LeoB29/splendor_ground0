"""No-training paired start-role objective audit for experiment 123."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .role_state_network import (
    create_identity_role_state_model,
    load_role_state_checkpoint,
    role_state_parameter_counts,
    role_state_trainable_names,
)
from .replay import read_replay_jsonl
from .scaled_continuation import _sha256, _write_json
from .start_role_interaction_audit import (
    _all_finite,
    _chunks,
    _collate_pairs,
    _cosine,
    _identity_audit,
    _load_target_records,
)
from .start_role_pilot import (
    _target_manifest_hash,
    _training_rows_from_records,
    reconstruct_role_panel,
    split_pair_games,
    summarize_target_records,
)


PROTOCOL_SHA256 = "94f26d42a6b13009cd43277e7a817881e0607c390764f1a61d3eeec248f3c43f"
EXPECTED_HASHES = {
    "candidate_122": "843e2f0d69cea9f799491552259bb867357a4a25b3faa08b78f9ef13e88961d6",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "target_manifest": "278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f",
}
LOSS_NAMES = (
    "ordinary_policy",
    "ordinary_value",
    "policy_contrast",
    "value_contrast",
)
GROUP_NAMES = ("role_input", "final_entity_block", "combined")
SPLIT_NAMES = ("train", "development", "all")


@dataclass(frozen=True, slots=True)
class PairedRoleObjectiveAuditConfig:
    output_dir: Path
    candidate_122: Path
    generation_11_checkpoint: Path
    train_replay: Path
    target_dir: Path
    device: str = "cpu"
    torch_threads: int = 1
    batch_pairs: int = 32


def run_paired_role_objective_audit_123(
    config: PairedRoleObjectiveAuditConfig,
) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = _verify_inputs(config)
    replay = read_replay_jsonl(config.train_replay)
    states, reconstruction = reconstruct_role_panel(
        replay, expected_games=224, label="123 fitting-only reconstruction"
    )
    records = _load_target_records(config.target_dir)
    oriented = _training_rows_from_records(states, records)
    pairs = [(oriented[index], oriented[index + 1]) for index in range(0, len(oriented), 2)]
    split = split_pair_games(states)
    development_games = set(split["validation_games"])
    pair_splits = {
        "train": [pair for pair in pairs if pair[0].game_seed not in development_games],
        "development": [pair for pair in pairs if pair[0].game_seed in development_games],
        "all": pairs,
    }
    if tuple(len(pair_splits[name]) for name in SPLIT_NAMES) != (358, 90, 448):
        raise ValueError("123 pair split does not reproduce the sealed 118 split.")

    identity_model, parent_identity = create_identity_role_state_model(
        config.generation_11_checkpoint, device=config.device
    )
    if parent_identity.generation != 11:
        raise ValueError("123 requires exact generation 11.")
    before = {
        name: parameter.detach().clone()
        for name, parameter in identity_model.named_parameters()
    }
    identity = _identity_audit(
        identity_model,
        pairs,
        device=config.device,
        batch_pairs=config.batch_pairs,
    )
    gradients = {
        loss_name: {
            split_name: _gradient_metrics(
                identity_model,
                pair_splits[split_name],
                loss_name,
                config.device,
                config.batch_pairs,
            )
            for split_name in SPLIT_NAMES
        }
        for loss_name in LOSS_NAMES
    }
    stability = {
        loss_name: {
            group: _cosine(
                gradients[loss_name]["train"]["vectors"][group],
                gradients[loss_name]["development"]["vectors"][group],
            )
            for group in GROUP_NAMES
        }
        for loss_name in LOSS_NAMES
    }
    ordinary_contrast_cosine = {
        "policy": {
            group: _cosine(
                gradients["ordinary_policy"]["train"]["vectors"][group],
                gradients["policy_contrast"]["train"]["vectors"][group],
            )
            for group in GROUP_NAMES
        },
        "value": {
            group: _cosine(
                gradients["ordinary_value"]["train"]["vectors"][group],
                gradients["value_contrast"]["train"]["vectors"][group],
            )
            for group in GROUP_NAMES
        },
    }
    normalized = _normalized_objective(gradients)

    identity_evaluation = {
        name: _evaluate_model(
            identity_model, pair_splits[name], config.device, config.batch_pairs
        )
        for name in SPLIT_NAMES
    }
    candidate, candidate_identity = load_role_state_checkpoint(
        config.candidate_122, device=config.device
    )
    if (
        candidate_identity.generation != 12
        or candidate_identity.metadata.get("selected_epoch") != 12
        or candidate_identity.metadata.get("parent_checkpoint_sha256")
        != EXPECTED_HASHES["generation_11_checkpoint"]
    ):
        raise ValueError("123 finalized candidate 122 lineage mismatch.")
    candidate_before = {
        name: parameter.detach().clone()
        for name, parameter in candidate.named_parameters()
    }
    candidate_contrast_gradients = {
        loss_name: {
            split_name: _gradient_metrics(
                candidate,
                pair_splits[split_name],
                loss_name,
                config.device,
                config.batch_pairs,
            )
            for split_name in SPLIT_NAMES
        }
        for loss_name in ("policy_contrast", "value_contrast")
    }
    candidate_contrast_stability = {
        loss_name: {
            group: _cosine(
                candidate_contrast_gradients[loss_name]["train"]["vectors"][group],
                candidate_contrast_gradients[loss_name]["development"]["vectors"][group],
            )
            for group in GROUP_NAMES
        }
        for loss_name in ("policy_contrast", "value_contrast")
    }
    candidate_evaluation = {
        name: _evaluate_model(
            candidate, pair_splits[name], config.device, config.batch_pairs
        )
        for name in SPLIT_NAMES
    }
    teacher_signal = {
        name: _teacher_signal(pair_splits[name]) for name in SPLIT_NAMES
    }
    proposed_gate = _future_development_gate(
        identity_evaluation["development"],
        candidate_evaluation["development"],
    )

    unchanged = all(
        torch.equal(before[name], parameter.detach())
        for name, parameter in identity_model.named_parameters()
    )
    candidate_unchanged = all(
        torch.equal(candidate_before[name], parameter.detach())
        for name, parameter in candidate.named_parameters()
    )
    expected_names = ["start_role_direction"] + [
        f"parent.entity_encoder.layers.1.{name}"
        for name, _parameter in identity_model.parent.entity_encoder.layers[-1].named_parameters()
    ]
    integrity_checks = {
        "exact_identity_logits": identity["max_logit_abs_diff"] <= 1e-7,
        "exact_identity_values": identity["max_value_abs_diff"] <= 1e-7,
        "deterministic_repeat": identity["max_repeat_abs_diff"] == 0.0,
        "exact_parameter_counts": role_state_parameter_counts(identity_model)
        == {
            "parent": 356_450,
            "frozen_parent": 281_666,
            "final_entity_block": 74_784,
            "role_input": 96,
            "total": 356_546,
            "trainable": 74_880,
        },
        "exact_trainable_scope": role_state_trainable_names(identity_model)
        == expected_names,
        "role_input_zero": float(identity_model.start_role_direction.abs().max().item())
        == 0.0,
        "parameters_unchanged": unchanged,
        "candidate_122_parameters_unchanged": candidate_unchanged,
        "metrics_finite": _all_finite(
            {
                "identity": identity,
                "stability": stability,
                "ordinary_contrast_cosine": ordinary_contrast_cosine,
                "normalized": _strip_vectors(normalized),
                "identity_evaluation": identity_evaluation,
                "candidate_evaluation": candidate_evaluation,
                "candidate_contrast_gradients": _strip_vectors(
                    candidate_contrast_gradients
                ),
                "candidate_contrast_stability": candidate_contrast_stability,
                "teacher_signal": teacher_signal,
            }
        ),
    }
    qualification_checks = _qualification_checks(
        gradients,
        stability,
        candidate_contrast_gradients,
        candidate_contrast_stability,
        normalized,
        teacher_signal,
        proposed_gate,
        integrity_checks,
    )
    qualified = all(qualification_checks.values())
    result = {
        "kind": "alphazero_paired_start_role_objective_audit_123",
        "status": "complete",
        "protocol_sha256": PROTOCOL_SHA256,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "input_hashes": input_hashes,
        "reconstruction": reconstruction,
        "split": split,
        "target_summary": summarize_target_records(records),
        "identity_audit": identity,
        "parameter_counts": role_state_parameter_counts(identity_model),
        "trainable_parameter_names": role_state_trainable_names(identity_model),
        "gradient_measurements": _strip_vectors(gradients),
        "train_development_gradient_cosines": stability,
        "ordinary_contrast_training_cosines": ordinary_contrast_cosine,
        "candidate_122_contrast_gradient_measurements": _strip_vectors(
            candidate_contrast_gradients
        ),
        "candidate_122_contrast_train_development_cosines": (
            candidate_contrast_stability
        ),
        "normalization": _strip_vectors(normalized),
        "teacher_role_signal": teacher_signal,
        "identity_fitting_evaluation": identity_evaluation,
        "candidate_122_fitting_evaluation": candidate_evaluation,
        "proposed_future_development_gate_on_122": proposed_gate,
        "integrity_checks": integrity_checks,
        "qualification_checks": qualification_checks,
        "objective_qualified": qualified,
        "recommended_future_objective": (
            {
                "formula": (
                    "ordinary_policy + 0.5*ordinary_value + "
                    "lambda_policy*policy_contrast + 0.5*lambda_value*value_contrast"
                ),
                "lambda_policy": normalized["policy"]["coefficient"],
                "lambda_value": normalized["value"]["coefficient"],
                "future_development_gate": {
                    "aggregate_relative_improvement_minimum": 0.005,
                    "policy_contrast_relative_reduction_minimum": 0.10,
                    "value_contrast_relative_reduction_minimum": 0.10,
                    "raw_policy_tv_minimum": 0.025,
                    "raw_abs_value_delta_minimum": 0.05,
                },
            }
            if qualified
            else None
        ),
        "decision": (
            "paired_role_contrast_objective_qualifies_freeze_pilot_no_training"
            if qualified
            else "paired_role_contrast_objective_not_stable_no_training"
        ),
        "training_performed": False,
        "optimizer_instantiated": False,
        "parameters_mutated": not (unchanged and candidate_unchanged),
        "checkpoint_written": False,
        "protected_117_path_accepted": False,
        "protected_117_panel_read": False,
        "protected_evidence_used_for_coefficients_or_gates": False,
        "search_targets_generated": False,
        "self_play_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_json(config.output_dir / "result.json", result)
    return result


def _verify_inputs(config: PairedRoleObjectiveAuditConfig) -> dict[str, str]:
    observed = {
        "candidate_122": _sha256(config.candidate_122),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "train_replay": _sha256(config.train_replay),
        "target_manifest": _target_manifest_hash(config.target_dir),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"123 frozen input mismatch: {observed}")
    return observed


def _objective_losses(model, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
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
    policy_error = (predicted_policy_delta - target_policy_delta).square()
    policy_contrast = (
        (policy_error * pair_mask).sum(dim=1)
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
    return {
        "ordinary_policy": ordinary_policy,
        "ordinary_value": ordinary_value,
        "policy_contrast": policy_contrast,
        "value_contrast": value_contrast,
    }


def _parameter_groups(model) -> dict[str, list[nn.Parameter]]:
    role = [model.start_role_direction]
    block = list(model.parent.entity_encoder.layers[-1].parameters())
    return {
        "role_input": role,
        "final_entity_block": block,
        "combined": role + block,
    }


def _gradient_vector(parameters: Iterable[nn.Parameter]) -> torch.Tensor:
    pieces = [
        (
            torch.zeros_like(parameter).reshape(-1)
            if parameter.grad is None
            else parameter.grad.reshape(-1)
        )
        for parameter in parameters
    ]
    return torch.cat(pieces).detach().cpu() if pieces else torch.empty(0)


def _gradient_metrics(
    model,
    pairs,
    loss_name: str,
    device: str,
    batch_pairs: int,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    total_pairs = len(pairs)
    mean_loss = 0.0
    for chunk in _chunks(pairs, batch_pairs):
        batch = _collate_pairs(chunk, device)
        loss = _objective_losses(model, batch)[loss_name]
        weight = len(chunk) / total_pairs
        (loss * weight).backward()
        mean_loss += float(loss.detach().item()) * weight
    vectors = {
        name: _gradient_vector(parameters)
        for name, parameters in _parameter_groups(model).items()
    }
    groups = {}
    for name, vector in vectors.items():
        norm = float(torch.linalg.vector_norm(vector).item())
        groups[name] = {
            "parameter_count": int(vector.numel()),
            "gradient_l2": norm,
            "gradient_rms": norm / math.sqrt(max(int(vector.numel()), 1)),
            "gradient_nonzero": norm > 0.0,
            "gradient_finite": bool(torch.isfinite(vector).all()),
        }
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "pairs": total_pairs,
        "loss": mean_loss,
        "groups": groups,
        "vectors": vectors,
    }


def _normalized_objective(gradients: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for endpoint, ordinary_name, contrast_name in (
        ("policy", "ordinary_policy", "policy_contrast"),
        ("value", "ordinary_value", "value_contrast"),
    ):
        ordinary_norm = gradients[ordinary_name]["train"]["groups"]["combined"][
            "gradient_l2"
        ]
        contrast_norm = gradients[contrast_name]["train"]["groups"]["combined"][
            "gradient_l2"
        ]
        coefficient = ordinary_norm / contrast_norm if contrast_norm > 0.0 else float("nan")
        groups = {}
        for group in GROUP_NAMES:
            ordinary_train = gradients[ordinary_name]["train"]["vectors"][group]
            contrast_train = gradients[contrast_name]["train"]["vectors"][group]
            ordinary_development = gradients[ordinary_name]["development"]["vectors"][group]
            contrast_development = gradients[contrast_name]["development"]["vectors"][group]
            normalized_contrast_train = coefficient * contrast_train
            summed_train = ordinary_train + normalized_contrast_train
            summed_development = ordinary_development + coefficient * contrast_development
            ordinary_group_norm = float(torch.linalg.vector_norm(ordinary_train).item())
            contrast_group_norm = float(
                torch.linalg.vector_norm(normalized_contrast_train).item()
            )
            summed_norm = float(torch.linalg.vector_norm(summed_train).item())
            groups[group] = {
                "ordinary_training_gradient_l2": ordinary_group_norm,
                "normalized_contrast_training_gradient_l2": contrast_group_norm,
                "normalized_contrast_to_ordinary_ratio": (
                    contrast_group_norm / ordinary_group_norm
                    if ordinary_group_norm > 0.0
                    else float("inf")
                ),
                "summed_training_gradient_l2": summed_norm,
                "summed_to_ordinary_training_ratio": (
                    summed_norm / ordinary_group_norm
                    if ordinary_group_norm > 0.0
                    else float("inf")
                ),
                "summed_train_development_cosine": _cosine(
                    summed_train, summed_development
                ),
                "vectors": {
                    "summed_train": summed_train,
                    "summed_development": summed_development,
                },
            }
        result[endpoint] = {"coefficient": coefficient, "groups": groups}
    return result


def _evaluate_model(model, pairs, device: str, batch_pairs: int) -> dict[str, float | int]:
    totals = {name: 0.0 for name in LOSS_NAMES}
    policy_tv = 0.0
    value_delta = 0.0
    total_pairs = len(pairs)
    model.eval()
    with torch.no_grad():
        for chunk in _chunks(pairs, batch_pairs):
            batch = _collate_pairs(chunk, device)
            losses = _objective_losses(model, batch)
            weight = len(chunk) / total_pairs
            for name in LOSS_NAMES:
                totals[name] += float(losses[name].item()) * weight
            logits, values = model(**batch["inputs"])
            mask = batch["inputs"]["action_mask"]
            probabilities = torch.softmax(
                logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1
            )
            reshaped = probabilities.reshape(len(chunk), 2, probabilities.shape[1])
            pair_mask = mask.reshape(len(chunk), 2, mask.shape[1]).all(dim=1)
            policy_tv += float(
                (0.5 * ((reshaped[:, 0] - reshaped[:, 1]).abs() * pair_mask).sum(dim=1)).sum().item()
            )
            value_delta += float(
                (values.reshape(len(chunk), 2)[:, 0] - values.reshape(len(chunk), 2)[:, 1])
                .abs()
                .sum()
                .item()
            )
    return {
        "pairs": total_pairs,
        **totals,
        "ordinary_objective": totals["ordinary_policy"] + 0.5 * totals["ordinary_value"],
        "mean_raw_policy_tv": policy_tv / total_pairs,
        "mean_raw_abs_value_delta": value_delta / total_pairs,
    }


def _teacher_signal(pairs) -> dict[str, float | int]:
    policy_tv = 0.0
    value_delta = 0.0
    for first, second in pairs:
        if len(first.policy_target) != len(second.policy_target):
            raise ValueError("123 paired semantic action widths differ.")
        policy_tv += 0.5 * sum(
            abs(float(a) - float(b))
            for a, b in zip(first.policy_target, second.policy_target, strict=True)
        )
        value_delta += abs(float(first.value_target) - float(second.value_target))
    return {
        "pairs": len(pairs),
        "mean_policy_tv": policy_tv / len(pairs),
        "mean_abs_value_delta": value_delta / len(pairs),
    }


def _future_development_gate(identity: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    policy_reduction = (
        (identity["policy_contrast"] - candidate["policy_contrast"])
        / identity["policy_contrast"]
        if identity["policy_contrast"] > 0.0
        else float("nan")
    )
    value_reduction = (
        (identity["value_contrast"] - candidate["value_contrast"])
        / identity["value_contrast"]
        if identity["value_contrast"] > 0.0
        else float("nan")
    )
    aggregate_improvement = (
        (identity["ordinary_objective"] - candidate["ordinary_objective"])
        / identity["ordinary_objective"]
    )
    checks = {
        "aggregate_relative_improvement_at_least_0_5pct": aggregate_improvement >= 0.005,
        "policy_contrast_relative_reduction_at_least_10pct": policy_reduction >= 0.10,
        "value_contrast_relative_reduction_at_least_10pct": value_reduction >= 0.10,
        "raw_policy_tv_at_least_0_025": candidate["mean_raw_policy_tv"] >= 0.025,
        "raw_abs_value_delta_at_least_0_05": candidate["mean_raw_abs_value_delta"] >= 0.05,
    }
    return {
        "aggregate_relative_improvement": aggregate_improvement,
        "policy_contrast_relative_reduction": policy_reduction,
        "value_contrast_relative_reduction": value_reduction,
        "candidate_mean_raw_policy_tv": candidate["mean_raw_policy_tv"],
        "candidate_mean_raw_abs_value_delta": candidate["mean_raw_abs_value_delta"],
        "checks": checks,
        "passed": all(checks.values()),
        "rejected": not all(checks.values()),
    }


def _qualification_checks(
    gradients,
    stability,
    candidate_gradients,
    candidate_stability,
    normalized,
    teacher_signal,
    proposed_gate,
    integrity_checks,
) -> dict[str, bool]:
    checks = {"integrity": all(integrity_checks.values())}
    for loss_name in ("ordinary_policy", "ordinary_value"):
        for split_name in SPLIT_NAMES:
            for group in ("role_input", "final_entity_block"):
                metrics = gradients[loss_name][split_name]["groups"][group]
                checks[f"{loss_name}_{split_name}_{group}_finite_nonzero"] = bool(
                    metrics["gradient_finite"] and metrics["gradient_nonzero"]
                )
    for loss_name in ("policy_contrast", "value_contrast"):
        for split_name in SPLIT_NAMES:
            role_metrics = gradients[loss_name][split_name]["groups"]["role_input"]
            block_metrics = gradients[loss_name][split_name]["groups"][
                "final_entity_block"
            ]
            combined_metrics = gradients[loss_name][split_name]["groups"]["combined"]
            checks[f"identity_{loss_name}_{split_name}_role_finite_nonzero"] = bool(
                role_metrics["gradient_finite"] and role_metrics["gradient_nonzero"]
            )
            checks[
                f"identity_{loss_name}_{split_name}_block_structurally_negligible"
            ] = (
                block_metrics["gradient_finite"]
                and block_metrics["gradient_rms"]
                <= 0.001 * role_metrics["gradient_rms"]
            )
            checks[f"identity_{loss_name}_{split_name}_combined_finite_nonzero"] = bool(
                combined_metrics["gradient_finite"]
                and combined_metrics["gradient_nonzero"]
            )
            for group in ("role_input", "final_entity_block"):
                candidate_metrics = candidate_gradients[loss_name][split_name][
                    "groups"
                ][group]
                checks[
                    f"candidate122_{loss_name}_{split_name}_{group}_finite_nonzero"
                ] = bool(
                    candidate_metrics["gradient_finite"]
                    and candidate_metrics["gradient_nonzero"]
                )
        for group in ("role_input", "combined"):
            checks[
                f"identity_{loss_name}_{group}_train_development_cosine_positive"
            ] = stability[loss_name][group] > 0.0
        for group in GROUP_NAMES:
            checks[
                f"candidate122_{loss_name}_{group}_train_development_cosine_positive"
            ] = candidate_stability[loss_name][group] > 0.0
    for endpoint in ("policy", "value"):
        coefficient = normalized[endpoint]["coefficient"]
        checks[f"{endpoint}_coefficient_finite_positive_bounded"] = (
            math.isfinite(coefficient) and 0.0 < coefficient <= 10_000.0
        )
        checks[f"{endpoint}_normalized_role_contribution_at_least_ordinary"] = (
            normalized[endpoint]["groups"]["role_input"][
                "normalized_contrast_to_ordinary_ratio"
            ]
            >= 1.0
        )
        checks[f"{endpoint}_summed_gradient_train_development_cosine_positive"] = (
            normalized[endpoint]["groups"]["combined"][
                "summed_train_development_cosine"
            ]
            > 0.0
        )
        checks[f"{endpoint}_summed_gradient_at_most_2_5x_ordinary"] = (
            normalized[endpoint]["groups"]["combined"][
                "summed_to_ordinary_training_ratio"
            ]
            <= 2.5
        )
    checks["development_teacher_policy_tv_at_least_0_15"] = (
        teacher_signal["development"]["mean_policy_tv"] >= 0.15
    )
    checks["development_teacher_value_delta_at_least_0_15"] = (
        teacher_signal["development"]["mean_abs_value_delta"] >= 0.15
    )
    checks["proposed_future_gate_rejects_finalized_122"] = bool(
        proposed_gate["rejected"]
    )
    return checks


def _strip_vectors(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_vectors(item)
            for key, item in value.items()
            if key not in {"vectors", "vector"}
        }
    if isinstance(value, list):
        return [_strip_vectors(item) for item in value]
    if isinstance(value, torch.Tensor):
        raise TypeError("Tensor leaked into serialized 123 result.")
    return value
