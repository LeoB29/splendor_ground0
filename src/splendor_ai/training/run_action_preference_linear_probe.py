"""Probe whether frozen 035 representations predict top-two action preferences."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.run_league_generation import _sha256
from splendor_ai.training.run_trajectory_policy_gradient import _load_attention_model


@dataclass(frozen=True, slots=True)
class ActionPreferenceLinearProbeConfig:
    parent_checkpoint: Path
    train_replay: Path
    validation_replay: Path
    confirmation_replay: Path
    output_dir: Path
    l2_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    folds: int = 5
    batch_size: int = 128
    max_lbfgs_iterations: int = 200
    bootstrap_samples: int = 10_000
    minimum_validation_rows: int = 150
    minimum_validation_balanced_accuracy: float = 0.55
    minimum_validation_auc_lower95: float = 0.50
    minimum_relative_log_loss_improvement: float = 0.02
    maximum_train_validation_balanced_accuracy_gap: float = 0.10
    minimum_breakdown_balanced_accuracy: float = 0.45
    seed: int = 35
    device: str = "cpu"


def run_action_preference_linear_probe(
    config: ActionPreferenceLinearProbeConfig,
) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(config)
    protocol_path = config.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing linear-probe protocol does not match.")
    else:
        _atomic_write(protocol_path, protocol)
    result_path = config.output_dir / "probe_final.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    parent, _ = _load_attention_model(config.parent_checkpoint, config.device)
    parent.eval()
    train = extract_frozen_pair_features(config.train_replay, parent, config)
    folds = stratified_fold_indices(train["metadata"], config.folds, config.seed)
    selection_rows = []
    for l2 in config.l2_grid:
        fold_metrics = []
        for fold_index in range(config.folds):
            heldout = set(folds[fold_index])
            train_indices = [index for index in range(len(train["labels"])) if index not in heldout]
            validation_indices = sorted(heldout)
            model, normalizer = fit_linear_probe(
                train["features"][train_indices],
                train["labels"][train_indices],
                l2,
                config,
            )
            probabilities = predict_linear_probe(
                model, normalizer, train["features"][validation_indices]
            )
            fold_metrics.append(
                binary_metrics(train["labels"][validation_indices], probabilities)
            )
        selection_rows.append(
            {
                "l2": l2,
                "mean_log_loss": statistics.fmean(row["log_loss"] for row in fold_metrics),
                "mean_balanced_accuracy": statistics.fmean(
                    row["balanced_accuracy"] for row in fold_metrics
                ),
                "mean_auc": statistics.fmean(row["auc"] for row in fold_metrics),
                "folds": fold_metrics,
            }
        )
    selected = min(selection_rows, key=lambda row: (row["mean_log_loss"], -row["l2"]))
    selected_l2 = float(selected["l2"])
    model, normalizer = fit_linear_probe(
        train["features"], train["labels"], selected_l2, config
    )
    train_probabilities = predict_linear_probe(model, normalizer, train["features"])
    train_metrics = binary_metrics(train["labels"], train_probabilities)

    # Validation is first read only after feature contract, L2 selection, final
    # fit, and decision thresholds are fixed. Confirmation is never opened.
    validation = extract_frozen_pair_features(config.validation_replay, parent, config)
    validation_probabilities = predict_linear_probe(
        model, normalizer, validation["features"]
    )
    validation_metrics = binary_metrics(validation["labels"], validation_probabilities)
    baseline_probability = float(train["labels"].mean().item())
    baseline_probabilities = torch.full_like(validation["labels"], baseline_probability)
    baseline_metrics = binary_metrics(validation["labels"], baseline_probabilities)
    intervals = bootstrap_metric_intervals(
        validation["labels"],
        validation_probabilities,
        config.bootstrap_samples,
        config.seed + 1,
    )
    breakdowns = metric_breakdowns(
        validation["labels"], validation_probabilities, validation["metadata"]
    )
    log_loss_improvement = (
        baseline_metrics["log_loss"] - validation_metrics["log_loss"]
    ) / baseline_metrics["log_loss"]
    breakdown_values = [
        float(bucket["balanced_accuracy"])
        for category in breakdowns.values()
        for bucket in category.values()
        if int(bucket["rows"]) >= 10
    ]
    gates = {
        "enough_validation_rows": len(validation["labels"]) >= config.minimum_validation_rows,
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"]
        >= config.minimum_validation_balanced_accuracy,
        "validation_auc_ci_lower_above_chance": intervals["auc"][0]
        > config.minimum_validation_auc_lower95,
        "validation_log_loss_improves": log_loss_improvement
        >= config.minimum_relative_log_loss_improvement,
        "bounded_train_validation_gap": train_metrics["balanced_accuracy"]
        - validation_metrics["balanced_accuracy"]
        <= config.maximum_train_validation_balanced_accuracy_gap,
        "breakdown_floor": bool(breakdown_values)
        and min(breakdown_values) >= config.minimum_breakdown_balanced_accuracy,
        "confirmation_untouched": True,
    }
    checkpoint_path = config.output_dir / "linear_probe.pt"
    torch.save(
        {
            "format": "action_preference_linear_probe_v1",
            "weight": model.weight.detach().cpu(),
            "bias": model.bias.detach().cpu(),
            "feature_mean": normalizer[0].cpu(),
            "feature_scale": normalizer[1].cpu(),
            "selected_l2": selected_l2,
            "feature_contract": frozen_feature_contract(),
            "non_promotional": True,
        },
        checkpoint_path,
    )
    payload = {
        "format": "action_preference_learnability_probe_v1",
        "decision": "learnable_generate_fresh_corpus" if all(gates.values()) else "not_learnable_close_direction",
        "passed": all(gates.values()),
        "non_promotional": True,
        "confirmation_path_recorded_but_not_read": str(config.confirmation_replay),
        "feature_size": int(train["features"].shape[1]),
        "train_rows": len(train["labels"]),
        "validation_rows": len(validation["labels"]),
        "selection": selection_rows,
        "selected_l2": selected_l2,
        "train": train_metrics,
        "validation": validation_metrics,
        "validation_intervals": intervals,
        "baseline_validation": baseline_metrics,
        "relative_validation_log_loss_improvement": log_loss_improvement,
        "validation_breakdowns": breakdowns,
        "gates": gates,
        "checkpoint_path": str(checkpoint_path),
    }
    _atomic_write(result_path, payload)
    print(json.dumps({"decision": payload["decision"], "passed": payload["passed"]}, indent=2))
    return payload


def extract_frozen_pair_features(path: Path, parent: nn.Module, config):
    payloads = _read_metadata(path)
    dataset = LegalActionReplayDataset(
        path,
        observation_size=parent.config.observation_size,
        action_feature_size=parent.config.action_feature_size,
    )
    if len(dataset) != len(payloads):
        raise ValueError("Preference metadata and reconstructed dataset differ in size.")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_legal_action_replay_samples,
    )
    features = []
    labels = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            count = len(batch["target_action_row"])
            batch_metadata = payloads[offset : offset + count]
            offset += count
            observation = batch["observation"].to(config.device)
            action_features = batch["legal_action_features"].to(config.device)
            mask = batch["legal_action_mask"].to(config.device)
            preferred = batch["target_action_row"].to(config.device)
            rejected = batch["preference_competitor_row"].to(config.device)
            state_hidden, contextual, scores = frozen_attention_representations(
                parent, observation, action_features, mask
            )
            flags = torch.tensor(
                [bool(row["preference_challenger_better"]) for row in batch_metadata],
                dtype=torch.bool,
                device=config.device,
            )
            challenger = torch.where(flags, preferred, rejected)
            parent_rows = torch.where(flags, rejected, preferred)
            rows = torch.arange(count, device=config.device)
            challenger_hidden = contextual[rows, challenger]
            parent_hidden = contextual[rows, parent_rows]
            difference = challenger_hidden - parent_hidden
            pair_mean = 0.5 * (challenger_hidden + parent_hidden)
            logit_gap = (scores[rows, challenger] - scores[rows, parent_rows]).unsqueeze(1)
            phase = torch.tensor(
                [
                    _phase_one_hot(str(row["phase"]))
                    for row in batch_metadata
                ],
                dtype=state_hidden.dtype,
                device=config.device,
            )
            features.append(
                torch.cat([state_hidden, difference, pair_mean, logit_gap, phase], dim=1).cpu()
            )
            labels.append(flags.to(dtype=torch.float32).cpu())
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "metadata": payloads,
    }


def frozen_attention_representations(model, observation, action_features, mask):
    batch_size, legal_count, _ = action_features.shape
    state_hidden = model.state_encoder(observation)
    action_hidden = model.action_encoder(
        action_features.reshape(batch_size * legal_count, model.config.action_feature_size)
    ).reshape(batch_size, legal_count, model.config.action_hidden_size)
    contextual = model.action_set_encoder(
        action_hidden + model.state_to_action_context(state_hidden).unsqueeze(1),
        src_key_padding_mask=~mask,
    )
    contextual = model.action_set_layer_norm(contextual).masked_fill(~mask.unsqueeze(-1), 0.0)
    expanded_state = state_hidden.unsqueeze(1).expand(
        batch_size, legal_count, model.config.state_hidden_size
    )
    scores = model.score_head(torch.cat([expanded_state, contextual], dim=-1)).squeeze(-1)
    return state_hidden, contextual, scores


def fit_linear_probe(features, labels, l2, config):
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / scale
    torch.manual_seed(config.seed)
    model = nn.Linear(features.shape[1], 1).to(config.device)
    nn.init.zeros_(model.weight)
    prevalence = float(labels.mean().clamp(1e-6, 1 - 1e-6))
    nn.init.constant_(model.bias, math.log(prevalence / (1.0 - prevalence)))
    x = normalized.to(config.device)
    y = labels.to(config.device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=config.max_lbfgs_iterations,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.5 * l2 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    return model, (mean, scale)


def predict_linear_probe(model, normalizer, features):
    mean, scale = normalizer
    with torch.inference_mode():
        logits = model(((features - mean) / scale).to(model.weight.device)).squeeze(1)
        return torch.sigmoid(logits).cpu()


def stratified_fold_indices(metadata, folds, seed):
    buckets: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(metadata):
        key = (
            bool(row["preference_challenger_better"]),
            str(row["phase"]),
            str(row["opponent_label"]),
            int(row["player_id"]),
        )
        buckets.setdefault(key, []).append(index)
    result = [[] for _ in range(folds)]
    rng = random.Random(seed)
    for key in sorted(buckets, key=str):
        rows = buckets[key]
        rng.shuffle(rows)
        for offset, index in enumerate(rows):
            result[offset % folds].append(index)
    return [sorted(fold) for fold in result]


def binary_metrics(labels, probabilities):
    labels = labels.to(dtype=torch.bool)
    predictions = probabilities >= 0.5
    positives = labels
    negatives = ~labels
    tpr = float((predictions[positives]).float().mean()) if positives.any() else 0.0
    tnr = float((~predictions[negatives]).float().mean()) if negatives.any() else 0.0
    clipped = probabilities.clamp(1e-7, 1 - 1e-7)
    losses = -(labels.float() * clipped.log() + (~labels).float() * (1 - clipped).log())
    return {
        "rows": len(labels),
        "positive_fraction": float(labels.float().mean()),
        "accuracy": float((predictions == labels).float().mean()),
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "auc": binary_auc(labels, probabilities),
        "log_loss": float(losses.mean()),
    }


def binary_auc(labels, scores):
    pairs = sorted(
        (float(score), bool(label))
        for score, label in zip(scores.tolist(), labels.tolist(), strict=True)
    )
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[start][0]:
            end += 1
        average_rank = 0.5 * (start + end) + 1.0
        rank_sum += average_rank * sum(pairs[index][1] for index in range(start, end + 1))
        start = end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def bootstrap_metric_intervals(labels, probabilities, samples, seed):
    rng = random.Random(seed)
    count = len(labels)
    values = {"balanced_accuracy": [], "auc": [], "log_loss": []}
    for _ in range(samples):
        indices = torch.tensor([rng.randrange(count) for _ in range(count)])
        metrics = binary_metrics(labels[indices], probabilities[indices])
        for key in values:
            values[key].append(float(metrics[key]))
    return {key: _percentile_interval(rows) for key, rows in values.items()}


def metric_breakdowns(labels, probabilities, metadata):
    definitions = {
        "phase": lambda row: str(row["phase"]),
        "opponent": lambda row: str(row["opponent_label"]),
        "seat": lambda row: f"seat{int(row['player_id'])}",
    }
    result = {}
    for name, key_fn in definitions.items():
        buckets: dict[str, list[int]] = {}
        for index, row in enumerate(metadata):
            buckets.setdefault(key_fn(row), []).append(index)
        result[name] = {
            key: binary_metrics(
                labels[torch.tensor(indices)], probabilities[torch.tensor(indices)]
            )
            for key, indices in sorted(buckets.items())
        }
    return result


def frozen_feature_contract():
    return [
        "frozen_035_state_hidden",
        "challenger_minus_parent_contextual_action_hidden",
        "challenger_parent_contextual_mean",
        "frozen_035_challenger_minus_parent_logit",
        "legal_turn_phase_one_hot",
    ]


def _read_metadata(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _phase_one_hot(phase):
    phases = ("early", "mid", "late")
    if phase not in phases:
        raise ValueError(f"Unknown phase: {phase}")
    return [float(phase == value) for value in phases]


def _percentile_interval(values):
    ordered = sorted(values)
    return [
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    ]


def _protocol(config):
    return {
        "format": "action_preference_linear_probe_protocol_v1",
        "hypothesis": (
            "A regularized linear head on frozen 035 information-legal representations "
            "can predict matched-return top-two preferences out of sample."
        ),
        "feature_contract": frozen_feature_contract(),
        "selection": "five-fold train-only stratified CV minimizes mean log loss",
        "validation_labels_parsed_once_after_selection": True,
        "confirmation_read_allowed": False,
        "gameplay_allowed": False,
        "promotion_allowed": False,
        "checkpoint_hashes": {
            "035": _sha256(config.parent_checkpoint),
            "train": _sha256(config.train_replay),
            "validation": _sha256(config.validation_replay),
        },
        "confirmation_path_record_only": str(config.confirmation_replay),
        "config": _config_payload(config),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    root = "outputs/action_preference_generation_002_loop_repair"
    parser.add_argument(
        "--parent-checkpoint",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument("--train-replay", default=f"{root}/data/preferences_train.jsonl")
    parser.add_argument(
        "--validation-replay", default=f"{root}/data/preferences_validation.jsonl"
    )
    parser.add_argument(
        "--confirmation-replay", default=f"{root}/data/preferences_confirmation.jsonl"
    )
    parser.add_argument("--output-dir", default="outputs/action_preference_learnability_probe_003")
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    args = build_parser().parse_args()
    run_action_preference_linear_probe(
        ActionPreferenceLinearProbeConfig(
            parent_checkpoint=Path(args.parent_checkpoint),
            train_replay=Path(args.train_replay),
            validation_replay=Path(args.validation_replay),
            confirmation_replay=Path(args.confirmation_replay),
            output_dir=Path(args.output_dir),
            device=args.device,
        )
    )


def _config_payload(config):
    payload = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


def _validate(config):
    for path in (
        config.parent_checkpoint,
        config.train_replay,
        config.validation_replay,
        config.confirmation_replay,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if config.folds < 2 or len(config.l2_grid) < 2:
        raise ValueError("Probe requires at least two folds and two fixed L2 values.")
    if any(value <= 0.0 for value in config.l2_grid):
        raise ValueError("Every L2 value must be positive.")


def _atomic_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
