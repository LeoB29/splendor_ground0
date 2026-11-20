"""No-training scale-native policy contrast audit for experiment 124."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import torch
from torch import nn

from .replay import read_replay_jsonl
from .role_state_network import (
    create_identity_role_state_model,
    load_role_state_checkpoint,
    role_state_parameter_counts,
    role_state_trainable_names,
)
from .scaled_continuation import _sha256, _write_json
from .start_role_interaction_audit import (
    _all_finite,
    _chunks,
    _collate_pairs,
    _cosine,
    _identity_audit,
    _load_target_records,
)
from .start_role_objective_audit import (
    _evaluate_model,
    _future_development_gate,
    _gradient_metrics as _ordinary_gradient_metrics,
    _parameter_groups,
    _strip_vectors,
    _teacher_signal,
)
from .start_role_pilot import (
    _target_manifest_hash,
    _training_rows_from_records,
    reconstruct_role_panel,
    split_pair_games,
    summarize_target_records,
)


PROTOCOL_SHA256 = "52e1db9043b9c9b9a0e0f2a023f736e8dd2c7362c8c9ed8ec8dc009972152a6b"
EXPECTED_HASHES = {
    "result_123": "12a7332ec08de864f7a55fb5d57d189924078a6513149cf4e020f410f39bce36",
    "candidate_122": "843e2f0d69cea9f799491552259bb867357a4a25b3faa08b78f9ef13e88961d6",
    "generation_11_checkpoint": "c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0",
    "train_replay": "59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a",
    "target_manifest": "278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f",
}
ARMS = (
    "summed_probability_difference_mse",
    "teacher_energy_normalized_mse",
)
SPLITS = ("train", "development", "all")
GROUPS = ("role_input", "final_entity_block", "combined")
RETAINED_LAMBDA_VALUE = 39.4241527478651


@dataclass(frozen=True, slots=True)
class ScaleNativePolicyContrastAuditConfig:
    output_dir: Path
    result_123: Path
    candidate_122: Path
    generation_11_checkpoint: Path
    train_replay: Path
    target_dir: Path
    device: str = "cpu"
    torch_threads: int = 1
    batch_pairs: int = 32


def run_scale_native_policy_contrast_audit_124(
    config: ScaleNativePolicyContrastAuditConfig,
) -> dict[str, Any]:
    torch.set_num_threads(config.torch_threads)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes, result_123 = _verify_inputs(config)
    replay = read_replay_jsonl(config.train_replay)
    states, reconstruction = reconstruct_role_panel(
        replay, expected_games=224, label="124 fitting-only reconstruction"
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
    if tuple(len(pair_splits[name]) for name in SPLITS) != (358, 90, 448):
        raise ValueError("124 pair split does not reproduce sealed 118.")
    energy_summary = _teacher_energy_summary(pair_splits)
    energy_floor = float(energy_summary["train"]["median"])
    if not math.isfinite(energy_floor) or energy_floor <= 0.0:
        raise ValueError("124 training teacher-energy floor is not positive finite.")

    identity_model, parent_identity = create_identity_role_state_model(
        config.generation_11_checkpoint, device=config.device
    )
    if parent_identity.generation != 11:
        raise ValueError("124 requires exact generation 11.")
    identity_before = _snapshot(identity_model)
    identity = _identity_audit(
        identity_model, pairs, device=config.device, batch_pairs=config.batch_pairs
    )
    ordinary_gradients = {
        split_name: _ordinary_gradient_metrics(
            identity_model,
            pair_splits[split_name],
            "ordinary_policy",
            config.device,
            config.batch_pairs,
        )
        for split_name in SPLITS
    }
    identity_arm_gradients = {
        arm: {
            split_name: _variant_gradient_metrics(
                identity_model,
                pair_splits[split_name],
                arm,
                energy_floor,
                config.device,
                config.batch_pairs,
            )
            for split_name in SPLITS
        }
        for arm in ARMS
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
        raise ValueError("124 finalized candidate 122 lineage mismatch.")
    candidate_before = _snapshot(candidate)
    candidate_arm_gradients = {
        arm: {
            split_name: _variant_gradient_metrics(
                candidate,
                pair_splits[split_name],
                arm,
                energy_floor,
                config.device,
                config.batch_pairs,
            )
            for split_name in SPLITS
        }
        for arm in ARMS
    }

    identity_stability = {
        arm: _gradient_stability(identity_arm_gradients[arm]) for arm in ARMS
    }
    candidate_stability = {
        arm: _gradient_stability(candidate_arm_gradients[arm]) for arm in ARMS
    }
    ordinary_contrast_cosines = {
        arm: {
            group: _cosine(
                ordinary_gradients["train"]["vectors"][group],
                identity_arm_gradients[arm]["train"]["vectors"][group],
            )
            for group in GROUPS
        }
        for arm in ARMS
    }
    normalization = {
        arm: _normalize_arm(
            ordinary_gradients,
            identity_arm_gradients[arm],
        )
        for arm in ARMS
    }

    identity_evaluation = {
        name: _evaluate_model(
            identity_model, pair_splits[name], config.device, config.batch_pairs
        )
        for name in SPLITS
    }
    candidate_evaluation = {
        name: _evaluate_model(
            candidate, pair_splits[name], config.device, config.batch_pairs
        )
        for name in SPLITS
    }
    teacher_signal = {name: _teacher_signal(pair_splits[name]) for name in SPLITS}
    future_gate = _future_development_gate(
        identity_evaluation["development"], candidate_evaluation["development"]
    )

    identity_unchanged = _unchanged(identity_model, identity_before)
    candidate_unchanged = _unchanged(candidate, candidate_before)
    expected_names = ["start_role_direction"] + [
        f"parent.entity_encoder.layers.1.{name}"
        for name, _parameter in identity_model.parent.entity_encoder.layers[-1].named_parameters()
    ]
    integrity_checks = {
        "123_failed_only_policy_coefficient_bound": _result_123_lineage_ok(result_123),
        "identity_logits_exact": identity["max_logit_abs_diff"] <= 1e-7,
        "identity_values_exact": identity["max_value_abs_diff"] <= 1e-7,
        "identity_deterministic": identity["max_repeat_abs_diff"] == 0.0,
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
        "identity_parameters_unchanged": identity_unchanged,
        "candidate_parameters_unchanged": candidate_unchanged,
        "retained_value_coefficient_matches_frozen_precision": (
            _retained_value_coefficient_matches(
                float(result_123["normalization"]["value"]["coefficient"])
            )
        ),
        "energy_floor_positive_finite": math.isfinite(energy_floor)
        and energy_floor > 0.0,
        "metrics_finite": _all_finite(
            {
                "identity": identity,
                "energy": energy_summary,
                "ordinary": _strip_vectors(ordinary_gradients),
                "identity_arms": _strip_vectors(identity_arm_gradients),
                "candidate_arms": _strip_vectors(candidate_arm_gradients),
                "identity_stability": identity_stability,
                "candidate_stability": candidate_stability,
                "cosines": ordinary_contrast_cosines,
                "normalization": _strip_vectors(normalization),
                "teacher": teacher_signal,
                "future_gate": future_gate,
            }
        ),
    }
    arm_results = {}
    for arm in ARMS:
        checks = _arm_qualification_checks(
            identity_arm_gradients[arm],
            candidate_arm_gradients[arm],
            identity_stability[arm],
            candidate_stability[arm],
            normalization[arm],
            teacher_signal,
            future_gate,
            integrity_checks,
        )
        arm_results[arm] = {
            "qualified": all(checks.values()),
            "qualification_checks": checks,
        }
    if arm_results[ARMS[0]]["qualified"]:
        selected = ARMS[0]
        decision = "summed_policy_contrast_qualifies_freeze_joint_objective_pilot_no_training"
    elif arm_results[ARMS[1]]["qualified"]:
        selected = ARMS[1]
        decision = (
            "teacher_normalized_policy_contrast_qualifies_freeze_joint_objective_pilot_no_training"
        )
    else:
        selected = None
        decision = "scale_native_policy_contrast_not_qualified_no_training"

    result = {
        "kind": "alphazero_scale_native_policy_contrast_audit_124",
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
        "teacher_policy_energy": energy_summary,
        "teacher_energy_floor": energy_floor,
        "teacher_role_signal": teacher_signal,
        "identity_audit": identity,
        "parameter_counts": role_state_parameter_counts(identity_model),
        "trainable_parameter_names": role_state_trainable_names(identity_model),
        "ordinary_policy_gradient_measurements": _strip_vectors(ordinary_gradients),
        "identity_arm_gradient_measurements": _strip_vectors(identity_arm_gradients),
        "candidate_122_arm_gradient_measurements": _strip_vectors(
            candidate_arm_gradients
        ),
        "identity_arm_train_development_cosines": identity_stability,
        "candidate_122_arm_train_development_cosines": candidate_stability,
        "ordinary_arm_training_cosines": ordinary_contrast_cosines,
        "normalization": _strip_vectors(normalization),
        "retained_value_contrast": {
            "formula": "mean squared error of paired root-value differences",
            "lambda_value": RETAINED_LAMBDA_VALUE,
        },
        "identity_fitting_evaluation": identity_evaluation,
        "candidate_122_fitting_evaluation": candidate_evaluation,
        "retained_future_development_gate_on_122": future_gate,
        "integrity_checks": integrity_checks,
        "arms": arm_results,
        "selected_policy_contrast": selected,
        "recommended_future_objective": (
            {
                "formula": (
                    "ordinary_policy + 0.5*ordinary_value + "
                    "lambda_policy*selected_policy_contrast + "
                    "0.5*lambda_value*value_contrast"
                ),
                "policy_contrast": selected,
                "lambda_policy": normalization[selected]["coefficient"],
                "lambda_value": RETAINED_LAMBDA_VALUE,
                "future_development_gate": {
                    "aggregate_relative_improvement_minimum": 0.005,
                    "policy_contrast_relative_reduction_minimum": 0.10,
                    "value_contrast_relative_reduction_minimum": 0.10,
                    "raw_policy_tv_minimum": 0.025,
                    "raw_abs_value_delta_minimum": 0.05,
                },
            }
            if selected is not None
            else None
        ),
        "decision": decision,
        "training_performed": False,
        "optimizer_instantiated": False,
        "parameters_mutated": not (identity_unchanged and candidate_unchanged),
        "checkpoint_written": False,
        "protected_117_path_accepted": False,
        "protected_117_panel_read": False,
        "search_targets_generated": False,
        "self_play_performed": False,
        "gameplay_arena": None,
        "external_035_arena": None,
        "promotion_authorized": False,
    }
    _write_json(config.output_dir / "result.json", result)
    return result


def _verify_inputs(
    config: ScaleNativePolicyContrastAuditConfig,
) -> tuple[dict[str, str], dict[str, Any]]:
    observed = {
        "result_123": _sha256(config.result_123),
        "candidate_122": _sha256(config.candidate_122),
        "generation_11_checkpoint": _sha256(config.generation_11_checkpoint),
        "train_replay": _sha256(config.train_replay),
        "target_manifest": _target_manifest_hash(config.target_dir),
    }
    if observed != EXPECTED_HASHES:
        raise ValueError(f"124 frozen input mismatch: {observed}")
    result = json.loads(config.result_123.read_text(encoding="utf-8"))
    if not _result_123_lineage_ok(result):
        raise ValueError("124 does not reproduce finalized 123.")
    return observed, result


def _result_123_lineage_ok(result: dict[str, Any]) -> bool:
    failed = [
        name for name, passed in result.get("qualification_checks", {}).items() if not passed
    ]
    return bool(
        result.get("status") == "complete"
        and result.get("decision")
        == "paired_role_contrast_objective_not_stable_no_training"
        and result.get("training_performed") is False
        and result.get("protected_117_panel_read") is False
        and failed == ["policy_coefficient_finite_positive_bounded"]
    )


def _variant_loss(
    model,
    batch: dict[str, Any],
    arm: str,
    energy_floor: float,
) -> torch.Tensor:
    logits, _values = model(**batch["inputs"])
    mask = batch["inputs"]["action_mask"]
    probabilities = torch.softmax(
        logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1
    )
    pair_count = probabilities.shape[0] // 2
    width = probabilities.shape[1]
    roles = batch["inputs"]["observer_is_start_player"].to(probabilities.dtype)
    signed = roles.mul(2.0).sub(1.0).reshape(pair_count, 2, 1)
    predicted_delta = (probabilities.reshape(pair_count, 2, width) * signed).sum(dim=1)
    teacher_delta = (
        batch["target_policy"].reshape(pair_count, 2, width) * signed
    ).sum(dim=1)
    pair_mask = mask.reshape(pair_count, 2, width).all(dim=1)
    squared_error = ((predicted_delta - teacher_delta).square() * pair_mask).sum(dim=1)
    if arm == ARMS[0]:
        return squared_error.mean()
    if arm == ARMS[1]:
        teacher_energy = (teacher_delta.square() * pair_mask).sum(dim=1)
        return (squared_error / teacher_energy.clamp_min(energy_floor)).mean()
    raise ValueError(f"Unknown 124 arm: {arm}")


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


def _variant_gradient_metrics(
    model,
    pairs,
    arm: str,
    energy_floor: float,
    device: str,
    batch_pairs: int,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    mean_loss = 0.0
    total_pairs = len(pairs)
    for chunk in _chunks(pairs, batch_pairs):
        batch = _collate_pairs(chunk, device)
        loss = _variant_loss(model, batch, arm, energy_floor)
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
        count = int(vector.numel())
        groups[name] = {
            "parameter_count": count,
            "gradient_l2": norm,
            "gradient_rms": norm / math.sqrt(max(count, 1)),
            "gradient_nonzero": norm > 0.0,
            "gradient_finite": bool(torch.isfinite(vector).all()),
        }
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"pairs": total_pairs, "loss": mean_loss, "groups": groups, "vectors": vectors}


def _gradient_stability(measurements: dict[str, Any]) -> dict[str, float]:
    return {
        group: _cosine(
            measurements["train"]["vectors"][group],
            measurements["development"]["vectors"][group],
        )
        for group in GROUPS
    }


def _normalize_arm(ordinary, contrast) -> dict[str, Any]:
    ordinary_norm = ordinary["train"]["groups"]["combined"]["gradient_l2"]
    contrast_norm = contrast["train"]["groups"]["combined"]["gradient_l2"]
    coefficient = ordinary_norm / contrast_norm if contrast_norm > 0.0 else float("nan")
    groups = {}
    for group in GROUPS:
        ordinary_train = ordinary["train"]["vectors"][group]
        ordinary_development = ordinary["development"]["vectors"][group]
        normalized_train = coefficient * contrast["train"]["vectors"][group]
        normalized_development = coefficient * contrast["development"]["vectors"][group]
        summed_train = ordinary_train + normalized_train
        summed_development = ordinary_development + normalized_development
        ordinary_l2 = float(torch.linalg.vector_norm(ordinary_train).item())
        contrast_l2 = float(torch.linalg.vector_norm(normalized_train).item())
        summed_l2 = float(torch.linalg.vector_norm(summed_train).item())
        groups[group] = {
            "ordinary_training_gradient_l2": ordinary_l2,
            "normalized_contrast_training_gradient_l2": contrast_l2,
            "normalized_contrast_to_ordinary_ratio": (
                contrast_l2 / ordinary_l2 if ordinary_l2 > 0.0 else float("inf")
            ),
            "summed_training_gradient_l2": summed_l2,
            "summed_to_ordinary_training_ratio": (
                summed_l2 / ordinary_l2 if ordinary_l2 > 0.0 else float("inf")
            ),
            "summed_train_development_cosine": _cosine(
                summed_train, summed_development
            ),
            "vectors": {
                "summed_train": summed_train,
                "summed_development": summed_development,
            },
        }
    return {"coefficient": coefficient, "groups": groups}


def _arm_qualification_checks(
    identity_gradients,
    candidate_gradients,
    identity_stability,
    candidate_stability,
    normalization,
    teacher_signal,
    future_gate,
    integrity_checks,
) -> dict[str, bool]:
    checks = {"integrity": all(integrity_checks.values())}
    for split_name in SPLITS:
        role = identity_gradients[split_name]["groups"]["role_input"]
        block = identity_gradients[split_name]["groups"]["final_entity_block"]
        combined = identity_gradients[split_name]["groups"]["combined"]
        checks[f"identity_{split_name}_role_finite_nonzero"] = bool(
            role["gradient_finite"] and role["gradient_nonzero"]
        )
        checks[f"identity_{split_name}_combined_finite_nonzero"] = bool(
            combined["gradient_finite"] and combined["gradient_nonzero"]
        )
        checks[f"identity_{split_name}_block_structurally_negligible"] = bool(
            block["gradient_finite"]
            and block["gradient_rms"] <= 0.001 * role["gradient_rms"]
        )
        for group in ("role_input", "final_entity_block"):
            candidate = candidate_gradients[split_name]["groups"][group]
            checks[f"candidate122_{split_name}_{group}_finite_nonzero"] = bool(
                candidate["gradient_finite"] and candidate["gradient_nonzero"]
            )
    for group in ("role_input", "combined"):
        checks[f"identity_{group}_train_development_cosine_positive"] = (
            identity_stability[group] > 0.0
        )
    for group in GROUPS:
        checks[f"candidate122_{group}_train_development_cosine_positive"] = (
            candidate_stability[group] > 0.0
        )
    coefficient = normalization["coefficient"]
    checks["coefficient_finite_positive_bounded"] = (
        math.isfinite(coefficient) and 0.0 < coefficient <= 10_000.0
    )
    checks["normalized_role_contribution_at_least_ordinary"] = (
        normalization["groups"]["role_input"][
            "normalized_contrast_to_ordinary_ratio"
        ]
        >= 1.0
    )
    checks["summed_gradient_train_development_cosine_positive"] = (
        normalization["groups"]["combined"]["summed_train_development_cosine"]
        > 0.0
    )
    checks["summed_gradient_at_most_2_5x_ordinary"] = (
        normalization["groups"]["combined"]["summed_to_ordinary_training_ratio"]
        <= 2.5
    )
    checks["development_teacher_policy_tv_at_least_0_15"] = (
        teacher_signal["development"]["mean_policy_tv"] >= 0.15
    )
    checks["retained_future_gate_rejects_122"] = bool(future_gate["rejected"])
    return checks


def _teacher_energy_summary(pair_splits) -> dict[str, Any]:
    result = {}
    for split_name, pairs in pair_splits.items():
        energies = sorted(_pair_teacher_energy(pair) for pair in pairs)
        result[split_name] = {
            "pairs": len(energies),
            "zero_pairs": sum(value == 0.0 for value in energies),
            "minimum": energies[0],
            "p25": _quantile(energies, 0.25),
            "median": float(statistics.median(energies)),
            "p75": _quantile(energies, 0.75),
            "maximum": energies[-1],
            "mean": sum(energies) / len(energies),
        }
    return result


def _pair_teacher_energy(pair) -> float:
    first, second = pair
    return sum(
        (float(a) - float(b)) ** 2
        for a, b in zip(first.policy_target, second.policy_target, strict=True)
    )


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }


def _retained_value_coefficient_matches(value: float) -> bool:
    return abs(value - RETAINED_LAMBDA_VALUE) <= 1e-12


def _unchanged(model, before: dict[str, torch.Tensor]) -> bool:
    return all(
        torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
