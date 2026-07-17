"""Gate an actor-only update on measurable, bounded policy movement."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from splendor_ai.training.dataset import (
    LegalActionReplayDataset,
    collate_legal_action_replay_samples,
)
from splendor_ai.training.run_trajectory_policy_gradient import (
    _exploration_probabilities,
    _load_attention_model,
)


@dataclass(frozen=True, slots=True)
class ActorMovementAuditConfig:
    parent_checkpoint: Path
    candidate_checkpoint: Path
    training_metrics_path: Path
    on_policy_replay_paths: tuple[Path, ...]
    broad_replay_paths: tuple[Path, ...]
    output_path: Path
    max_rows_per_stream: int = 5_000
    batch_size: int = 128
    sampling_epsilon: float = 0.3
    sampling_temperature: float = 0.5
    sampling_top_k: int = 3
    min_on_policy_change_rate: float = 0.01
    min_broad_change_rate: float = 0.005
    max_change_rate: float = 0.10
    max_mean_exploration_kl: float = 0.01
    max_behavior_log_prob_mae: float = 1e-5
    max_clip_fraction: float = 0.10
    seed: int = 40
    device: str = "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether an actor update moved enough without excessive drift."
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--training-metrics-path", required=True)
    parser.add_argument("--on-policy-replay-path", action="append", required=True)
    parser.add_argument("--broad-replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-rows-per-stream", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampling-epsilon", type=float, default=0.3)
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument("--sampling-top-k", type=int, default=3)
    parser.add_argument("--min-on-policy-change-rate", type=float, default=0.01)
    parser.add_argument("--min-broad-change-rate", type=float, default=0.005)
    parser.add_argument("--max-change-rate", type=float, default=0.10)
    parser.add_argument("--max-mean-exploration-kl", type=float, default=0.01)
    parser.add_argument("--max-behavior-log-prob-mae", type=float, default=1e-5)
    parser.add_argument("--max-clip-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    return parser


def audit_actor_movement(config: ActorMovementAuditConfig) -> dict[str, Any]:
    _validate(config)
    parent, _ = _load_attention_model(config.parent_checkpoint, config.device)
    candidate, _ = _load_attention_model(config.candidate_checkpoint, config.device)
    parent.eval()
    candidate.eval()
    on_policy = _measure_stream(
        config.on_policy_replay_paths,
        parent,
        candidate,
        config,
        seed=config.seed,
    )
    broad = _measure_stream(
        config.broad_replay_paths,
        parent,
        candidate,
        config,
        seed=config.seed + 1,
    )
    training_metrics = json.loads(config.training_metrics_path.read_text(encoding="utf-8"))
    final_epoch = training_metrics["epochs"][-1]
    gates = _evaluate_gates(config, on_policy, broad, final_epoch)
    payload = {
        "format": "actor_movement_audit_v1",
        "config": _config_payload(config),
        "on_policy": on_policy,
        "broad": broad,
        "training": {
            key: final_epoch[key]
            for key in (
                "policy_loss",
                "kl_loss",
                "entropy",
                "clip_fraction",
                "approx_ratio",
                "behavior_log_prob_mae",
            )
        },
        "gates": gates,
        "pass": all(gates.values()),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _measure_stream(
    replay_paths: tuple[Path, ...],
    parent: torch.nn.Module,
    candidate: torch.nn.Module,
    config: ActorMovementAuditConfig,
    seed: int,
) -> dict[str, Any]:
    dataset = LegalActionReplayDataset(
        replay_paths,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=parent.config.observation_size,
        action_feature_size=parent.config.action_feature_size,
    )
    count = min(config.max_rows_per_stream, len(dataset))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(dataset)), count)) if count < len(dataset) else list(range(len(dataset)))
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_legal_action_replay_samples,
    )
    changes = 0
    raw_kls: list[float] = []
    exploration_kls: list[float] = []
    parent_entropies: list[float] = []
    candidate_entropies: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            observation = batch["observation"].to(config.device)
            action_features = batch["legal_action_features"].to(config.device)
            legal_mask = batch["legal_action_mask"].to(config.device)
            parent_scores, _ = parent(observation, action_features, legal_mask)
            candidate_scores, _ = candidate(observation, action_features, legal_mask)
            masked_parent_scores = parent_scores.masked_fill(~legal_mask, -torch.inf)
            masked_candidate_scores = candidate_scores.masked_fill(~legal_mask, -torch.inf)
            changes += int(
                (
                    torch.argmax(masked_parent_scores, dim=1)
                    != torch.argmax(masked_candidate_scores, dim=1)
                )
                .sum()
                .item()
            )

            parent_logits = masked_parent_scores
            candidate_logits = masked_candidate_scores
            parent_probs = torch.softmax(parent_logits, dim=1)
            candidate_probs = torch.softmax(candidate_logits, dim=1)
            raw_kls.extend(_row_kl(parent_probs, candidate_probs).cpu().tolist())
            parent_entropies.extend(_row_entropy(parent_probs).cpu().tolist())
            candidate_entropies.extend(_row_entropy(candidate_probs).cpu().tolist())

            current_exploration, reference_exploration = _exploration_probabilities(
                candidate_scores,
                parent_scores,
                legal_mask,
                config.sampling_epsilon,
                config.sampling_temperature,
                config.sampling_top_k,
            )
            exploration_kls.extend(
                _row_kl(reference_exploration, current_exploration).cpu().tolist()
            )

    return {
        "source_paths": [str(path) for path in replay_paths],
        "dataset_rows": len(dataset),
        "rows_sampled": count,
        "top_action_changes": changes,
        "top_action_change_rate": changes / count if count else 0.0,
        "raw_policy_kl": _distribution_summary(raw_kls),
        "exploration_policy_kl": _distribution_summary(exploration_kls),
        "parent_entropy_mean": statistics.fmean(parent_entropies) if parent_entropies else 0.0,
        "candidate_entropy_mean": statistics.fmean(candidate_entropies) if candidate_entropies else 0.0,
    }


def _row_kl(reference: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    reference_log = reference.clamp_min(1e-12).log()
    current_log = current.clamp_min(1e-12).log()
    return (reference * (reference_log - current_log)).sum(dim=1)


def _row_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


def _distribution_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered) if ordered else 0.0,
        "p95": _percentile(ordered, 0.95),
        "max": max(ordered, default=0.0),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def _evaluate_gates(
    config: ActorMovementAuditConfig,
    on_policy: dict[str, Any],
    broad: dict[str, Any],
    final_epoch: dict[str, Any],
) -> dict[str, bool]:
    on_policy_change = float(on_policy["top_action_change_rate"])
    broad_change = float(broad["top_action_change_rate"])
    return {
        "on_policy_movement_floor": on_policy_change >= config.min_on_policy_change_rate,
        "broad_movement_floor": broad_change >= config.min_broad_change_rate,
        "on_policy_movement_ceiling": on_policy_change <= config.max_change_rate,
        "broad_movement_ceiling": broad_change <= config.max_change_rate,
        "on_policy_kl_ceiling": float(on_policy["exploration_policy_kl"]["mean"])
        <= config.max_mean_exploration_kl,
        "broad_kl_ceiling": float(broad["exploration_policy_kl"]["mean"])
        <= config.max_mean_exploration_kl,
        "behavior_reconstruction": float(final_epoch["behavior_log_prob_mae"])
        <= config.max_behavior_log_prob_mae,
        "clip_fraction": float(final_epoch["clip_fraction"]) <= config.max_clip_fraction,
        "mean_importance_ratio": 0.95 <= float(final_epoch["approx_ratio"]) <= 1.05,
    }


def _config_payload(config: ActorMovementAuditConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "parent_checkpoint",
        "candidate_checkpoint",
        "training_metrics_path",
        "output_path",
    ):
        payload[key] = str(payload[key])
    payload["on_policy_replay_paths"] = [str(path) for path in config.on_policy_replay_paths]
    payload["broad_replay_paths"] = [str(path) for path in config.broad_replay_paths]
    return payload


def _validate(config: ActorMovementAuditConfig) -> None:
    if not config.on_policy_replay_paths or not config.broad_replay_paths:
        raise ValueError("Both on-policy and broad replay streams are required.")
    if config.max_rows_per_stream <= 0 or config.batch_size <= 0:
        raise ValueError("Row and batch limits must be positive.")
    if not 0.0 < config.sampling_epsilon <= 1.0:
        raise ValueError("sampling_epsilon must be in (0, 1].")
    if config.sampling_temperature <= 0.0 or config.sampling_top_k <= 0:
        raise ValueError("Sampling temperature and top-k must be positive.")


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_actor_movement(
        ActorMovementAuditConfig(
            parent_checkpoint=Path(args.parent_checkpoint),
            candidate_checkpoint=Path(args.candidate_checkpoint),
            training_metrics_path=Path(args.training_metrics_path),
            on_policy_replay_paths=tuple(Path(path) for path in args.on_policy_replay_path),
            broad_replay_paths=tuple(Path(path) for path in args.broad_replay_path),
            output_path=Path(args.output_path),
            max_rows_per_stream=args.max_rows_per_stream,
            batch_size=args.batch_size,
            sampling_epsilon=args.sampling_epsilon,
            sampling_temperature=args.sampling_temperature,
            sampling_top_k=args.sampling_top_k,
            min_on_policy_change_rate=args.min_on_policy_change_rate,
            min_broad_change_rate=args.min_broad_change_rate,
            max_change_rate=args.max_change_rate,
            max_mean_exploration_kl=args.max_mean_exploration_kl,
            max_behavior_log_prob_mae=args.max_behavior_log_prob_mae,
            max_clip_fraction=args.max_clip_fraction,
            seed=args.seed,
            device=args.device,
        )
    )
    print(
        "Actor movement audit:"
        f" pass={payload['pass']}"
        f" on_policy_change={payload['on_policy']['top_action_change_rate']:.4f}"
        f" broad_change={payload['broad']['top_action_change_rate']:.4f}"
        f" broad_kl={payload['broad']['exploration_policy_kl']['mean']:.6g}"
    )


if __name__ == "__main__":
    main()
