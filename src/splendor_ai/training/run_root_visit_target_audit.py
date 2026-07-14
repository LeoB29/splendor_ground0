"""Measure policy uptake of stored PUCT root-visit targets on held-out games."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from splendor_ai.encoding.action_features import ACTION_TYPE_ORDER

from .dataset import LegalActionReplayDataset, collate_legal_action_replay_samples
from .legal_action_supervised import split_legal_action_replay_dataset
from .model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    masked_legal_action_scores,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit checkpoint alignment with held-out PUCT root-visit targets. "
            "Replay paths must match the training mix so the original validation split is reproduced."
        )
    )
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Snapshot replay path from the original mixed training data. Repeat for every source.",
    )
    parser.add_argument(
        "--target-source-path",
        required=True,
        help="The PUCT replay source to isolate inside the held-out split.",
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=36)
    parser.add_argument(
        "--split",
        choices=("validation", "train"),
        default="validation",
        help="Which side of the reproduced training split to audit.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    return parser


def _phase_bucket(turn_index: int) -> str:
    if turn_index < 20:
        return "early_0_19"
    if turn_index < 40:
        return "middle_20_39"
    if turn_index < 60:
        return "late_40_59"
    return "endgame_60_plus"


def _load_attention_model(path: str | Path, device: torch.device) -> LegalActionAttentionScorerModel:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError(
            f"Expected legal_action_attention_scorer checkpoint, got {checkpoint.get('model_kind')!r}: {checkpoint_path}"
        )
    config_keys = {field.name for field in fields(LegalActionAttentionScorerConfig)}
    config = LegalActionAttentionScorerConfig(
        **{
            key: value
            for key, value in checkpoint.get("model_config", {}).items()
            if key in config_keys
        }
    )
    model = LegalActionAttentionScorerModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _row_metrics(
    target: torch.Tensor,
    baseline_log_probs: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    behavior_action_row: int,
) -> tuple[dict[str, float], int]:
    target = target.masked_fill(~legal_mask, 0.0)
    target = target / target.sum().clamp_min(torch.finfo(target.dtype).eps)
    target_log = torch.where(target > 0.0, target.log(), torch.zeros_like(target))
    target_entropy = float((-(target * target_log).sum()).item())
    baseline_ce = float((-(target * baseline_log_probs).sum()).item())
    candidate_ce = float((-(target * candidate_log_probs).sum()).item())
    baseline_probs = baseline_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    baseline_entropy = float((-(baseline_probs * baseline_log_probs).sum()).item())
    candidate_entropy = float((-(candidate_probs * candidate_log_probs).sum()).item())
    target_top = int(target.argmax().item())
    baseline_top = int(baseline_probs.argmax().item())
    candidate_top = int(candidate_probs.argmax().item())
    candidate_to_baseline_kl = float(
        (candidate_probs * (candidate_log_probs - baseline_log_probs)).sum().item()
    )
    return (
        {
            "target_entropy": target_entropy,
            "target_effective_actions": math.exp(target_entropy),
            "target_top_probability": float(target[target_top].item()),
            "target_support_actions": float((target > 0.0).sum().item()),
            "legal_actions": float(legal_mask.sum().item()),
            "target_support_fraction": float(
                (target > 0.0).sum().item() / legal_mask.sum().item()
            ),
            "behavior_action_matches_target_top": float(behavior_action_row == target_top),
            "target_probability_at_behavior_action": float(target[behavior_action_row].item()),
            "baseline_cross_entropy": baseline_ce,
            "candidate_cross_entropy": candidate_ce,
            "cross_entropy_delta_candidate_minus_baseline": candidate_ce - baseline_ce,
            "baseline_kl_to_target": baseline_ce - target_entropy,
            "candidate_kl_to_target": candidate_ce - target_entropy,
            "baseline_target_top_probability": float(baseline_probs[target_top].item()),
            "candidate_target_top_probability": float(candidate_probs[target_top].item()),
            "baseline_target_top1_agreement": float(baseline_top == target_top),
            "candidate_target_top1_agreement": float(candidate_top == target_top),
            "model_top1_disagreement": float(baseline_top != candidate_top),
            "candidate_to_baseline_policy_kl": candidate_to_baseline_kl,
            "baseline_policy_entropy": baseline_entropy,
            "candidate_policy_entropy": candidate_entropy,
        },
        target_top,
    )


def _summarize_grouped_metrics(
    rows: dict[str, dict[int, list[dict[str, float]]]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for group_name, by_game in sorted(rows.items()):
        metric_names = sorted({name for game_rows in by_game.values() for row in game_rows for name in row})
        metric_summary: dict[str, dict[str, float]] = {}
        for metric_name in metric_names:
            per_game = [
                sum(row[metric_name] for row in game_rows) / len(game_rows)
                for game_rows in by_game.values()
            ]
            metric_summary[metric_name] = _mean_ci(
                per_game,
                bootstrap_samples=bootstrap_samples,
                seed=_stable_seed(bootstrap_seed, group_name, metric_name),
            )
        summaries[group_name] = {
            "rows": sum(len(game_rows) for game_rows in by_game.values()),
            "games": len(by_game),
            "metrics": metric_summary,
        }
    return summaries


def _mean_ci(values: list[float], bootstrap_samples: int, seed: int) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric slice.")
    mean = sum(values) / len(values)
    if len(values) == 1 or bootstrap_samples <= 0:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    samples = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(bootstrap_samples)
    )
    return {
        "mean": mean,
        "ci95_low": samples[int(0.025 * (bootstrap_samples - 1))],
        "ci95_high": samples[int(0.975 * (bootstrap_samples - 1))],
    }


def _stable_seed(base_seed: int, *parts: str) -> int:
    value = base_seed
    for part in parts:
        for character in part:
            value = ((value * 33) + ord(character)) & 0xFFFFFFFF
    return value


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in the range (0, 1).")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap-samples must be non-negative.")

    device = torch.device(args.device)
    replay_paths = tuple(Path(path) for path in args.replay_path)
    target_source = Path(args.target_source_path).resolve()
    dataset = LegalActionReplayDataset(replay_paths)
    train_dataset, validation_dataset = split_legal_action_replay_dataset(
        dataset,
        validation_fraction=args.validation_fraction,
        seed=args.validation_seed,
        validation_split_unit="game",
    )
    selected_dataset = validation_dataset if args.split == "validation" else train_dataset
    if not isinstance(selected_dataset, Subset):
        raise ValueError("Expected a non-empty game split.")
    target_indices = [
        index
        for index in selected_dataset.indices
        if dataset.source_path_for_index(index).resolve() == target_source
    ]
    if not target_indices:
        raise ValueError("Selected split contains no rows from --target-source-path.")

    target_dataset = Subset(dataset, target_indices)
    dataloader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_legal_action_replay_samples,
    )
    baseline = _load_attention_model(args.baseline_checkpoint, device)
    candidate = _load_attention_model(args.candidate_checkpoint, device)

    grouped_rows: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    offset = 0
    with torch.inference_mode():
        for batch in dataloader:
            batch_size = batch["observation"].shape[0]
            batch_indices = target_indices[offset : offset + batch_size]
            offset += batch_size
            observation = batch["observation"].to(device)
            action_features = batch["legal_action_features"].to(device)
            legal_mask = batch["legal_action_mask"].to(device)
            root_target = batch["root_visit_target"].to(device)
            root_mask = batch["root_visit_mask"].to(device)
            baseline_scores, _ = baseline(observation, action_features, legal_mask)
            candidate_scores, _ = candidate(observation, action_features, legal_mask)
            baseline_log_probs = torch.log_softmax(
                masked_legal_action_scores(baseline_scores, legal_mask), dim=1
            )
            candidate_log_probs = torch.log_softmax(
                masked_legal_action_scores(candidate_scores, legal_mask), dim=1
            )

            for row_index, dataset_index in enumerate(batch_indices):
                if not bool(root_mask[row_index].item()):
                    continue
                metadata = dataset.replay_step_metadata_for_index(dataset_index)
                metrics, target_top = _row_metrics(
                    root_target[row_index],
                    baseline_log_probs[row_index],
                    candidate_log_probs[row_index],
                    legal_mask[row_index],
                    int(batch["target_action_row"][row_index].item()),
                )
                action_type_index = int(
                    action_features[row_index, target_top, : len(ACTION_TYPE_ORDER)].argmax().item()
                )
                action_family = ACTION_TYPE_ORDER[action_type_index].name
                phase = _phase_bucket(metadata.turn_index)
                for group_name in (
                    "overall",
                    f"phase:{phase}",
                    f"target_top_action_family:{action_family}",
                ):
                    grouped_rows[group_name][metadata.game_seed].append(metrics)

    summaries = _summarize_grouped_metrics(
        grouped_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.validation_seed,
    )
    overall = summaries["overall"]
    output = {
        "audit": "root_visit_target_uptake",
        "dataset": {
            "replay_paths": [str(path) for path in replay_paths],
            "target_source_path": str(Path(args.target_source_path)),
            "total_mixed_rows": len(dataset),
            "split": args.split,
            "target_rows": overall["rows"],
            "target_games": overall["games"],
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "validation_split_unit": "game",
        },
        "checkpoints": {
            "baseline": str(args.baseline_checkpoint),
            "candidate": str(args.candidate_checkpoint),
        },
        "slices": summaries,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    metrics = overall["metrics"]
    print(
        "[root-visit-audit]"
        f" split={args.split} games={overall['games']} rows={overall['rows']}"
        f" target_effective_actions={metrics['target_effective_actions']['mean']:.3f}"
        f" baseline_ce={metrics['baseline_cross_entropy']['mean']:.4f}"
        f" candidate_ce={metrics['candidate_cross_entropy']['mean']:.4f}"
        f" delta_ce={metrics['cross_entropy_delta_candidate_minus_baseline']['mean']:.4f}"
    )
    print(f"[root-visit-audit] wrote: {output_path}")


if __name__ == "__main__":
    main()
