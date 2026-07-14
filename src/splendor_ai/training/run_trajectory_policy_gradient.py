"""Run a constrained PPO-style update from stochastic trajectory replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.optim import AdamW

from splendor_ai.training.dataset import LegalActionReplayDataset
from splendor_ai.training.legal_action_supervised import (
    create_legal_action_replay_dataloader,
)
from splendor_ai.training.model import (
    LegalActionAttentionScorerConfig,
    LegalActionAttentionScorerModel,
    masked_legal_action_scores,
)


@dataclass(frozen=True, slots=True)
class TrajectoryPolicyGradientConfig:
    replay_paths: tuple[Path, ...]
    init_checkpoint: Path
    output_dir: Path
    epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 3e-6
    weight_decay: float = 1e-4
    clip_ratio: float = 0.1
    kl_weight: float = 1.0
    entropy_weight: float = 0.001
    sampling_epsilon: float = 0.2
    sampling_temperature: float = 0.5
    sampling_top_k: int = 3
    max_grad_norm: float = 1.0
    device: str = "cpu"
    log_every_batches: int = 20
    seed: int = 31


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune an attention scorer with signed trajectory advantages."
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.001)
    parser.add_argument("--sampling-epsilon", type=float, default=0.2)
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument("--sampling-top-k", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrajectoryPolicyGradientConfig(
        replay_paths=tuple(Path(path) for path in args.replay_path),
        init_checkpoint=Path(args.init_checkpoint),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_ratio=args.clip_ratio,
        kl_weight=args.kl_weight,
        entropy_weight=args.entropy_weight,
        sampling_epsilon=args.sampling_epsilon,
        sampling_temperature=args.sampling_temperature,
        sampling_top_k=args.sampling_top_k,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        log_every_batches=args.log_every_batches,
        seed=args.seed,
    )
    summary = train_trajectory_policy(config)
    final = summary["epochs"][-1]
    print(
        "Trajectory policy update complete:"
        f" samples={final['samples']}"
        f" policy_loss={final['policy_loss']:.6f}"
        f" kl={final['kl_loss']:.6f}"
        f" entropy={final['entropy']:.6f}"
    )


def train_trajectory_policy(
    config: TrajectoryPolicyGradientConfig,
) -> dict[str, Any]:
    _validate_config(config)
    torch.manual_seed(config.seed)
    model, checkpoint = _load_attention_model(config.init_checkpoint, config.device)
    reference, _ = _load_attention_model(config.init_checkpoint, config.device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    dataset = LegalActionReplayDataset(
        config.replay_paths,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    if not all(dataset[index].trajectory_mask for index in range(len(dataset))):
        raise ValueError("Every trajectory training row must contain advantage and behavior probability.")
    dataloader = create_legal_action_replay_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_summaries = []
    start = time.perf_counter()
    for epoch_index in range(1, config.epochs + 1):
        model.train()
        totals = {
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "clip_fraction": 0.0,
            "approx_ratio": 0.0,
            "behavior_log_prob_mae": 0.0,
        }
        samples = 0
        batches = 0
        epoch_start = time.perf_counter()
        for batch_index, batch in enumerate(dataloader, start=1):
            observation = batch["observation"].to(config.device)
            action_features = batch["legal_action_features"].to(config.device)
            legal_mask = batch["legal_action_mask"].to(config.device)
            target_rows = batch["target_action_row"].to(config.device)
            advantages = batch["trajectory_advantage"].to(config.device)
            behavior_log_probs = batch["behavior_log_probability"].to(config.device)
            sample_weights = batch["sample_weight"].to(config.device)

            optimizer.zero_grad(set_to_none=True)
            scores, _value = model(observation, action_features, legal_mask)
            with torch.no_grad():
                reference_scores, _reference_value = reference(
                    observation,
                    action_features,
                    legal_mask,
                )
            losses = compute_clipped_trajectory_loss(
                action_scores=scores,
                reference_scores=reference_scores,
                legal_action_mask=legal_mask,
                target_action_row=target_rows,
                advantages=advantages,
                behavior_log_probability=behavior_log_probs,
                sample_weight=sample_weights,
                clip_ratio=config.clip_ratio,
                kl_weight=config.kl_weight,
                entropy_weight=config.entropy_weight,
                sampling_epsilon=config.sampling_epsilon,
                sampling_temperature=config.sampling_temperature,
                sampling_top_k=config.sampling_top_k,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            batch_samples = int(target_rows.numel())
            samples += batch_samples
            batches += 1
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_samples
            if (
                batch_index == 1
                or batch_index == len(dataloader)
                or (
                    config.log_every_batches > 0
                    and batch_index % config.log_every_batches == 0
                )
            ):
                print(
                    f"[trajectory-train] epoch={epoch_index}/{config.epochs}"
                    f" batch={batch_index}/{len(dataloader)}"
                    f" samples={samples}"
                    f" policy={totals['policy_loss'] / samples:.6f}"
                    f" kl={totals['kl_loss'] / samples:.6f}"
                    f" entropy={totals['entropy'] / samples:.6f}"
                    f" elapsed={time.perf_counter() - epoch_start:.1f}s"
                )
        epoch_summaries.append(
            {
                key: value / samples
                for key, value in totals.items()
            }
            | {
                "epoch": epoch_index,
                "samples": samples,
                "batches": batches,
                "seconds": time.perf_counter() - epoch_start,
            }
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "legal_action_attention_scorer.pt"
    metrics_path = config.output_dir / "trajectory_policy_metrics.json"
    train_config = {
        **asdict(config),
        "replay_paths": [str(path) for path in config.replay_paths],
        "init_checkpoint": str(config.init_checkpoint),
        "output_dir": str(config.output_dir),
    }
    torch.save(
        {
            "model_kind": "legal_action_attention_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": checkpoint["model_config"],
            "train_config": train_config,
        },
        checkpoint_path,
    )
    summary = {
        "config": train_config,
        "dataset_samples": len(dataset),
        "checkpoint_path": str(checkpoint_path),
        "epochs": epoch_summaries,
        "training_seconds": time.perf_counter() - start,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[trajectory-train] wrote checkpoint: {checkpoint_path}")
    print(f"[trajectory-train] wrote metrics: {metrics_path}")
    return summary


def compute_clipped_trajectory_loss(
    action_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    target_action_row: torch.Tensor,
    advantages: torch.Tensor,
    behavior_log_probability: torch.Tensor,
    sample_weight: torch.Tensor,
    clip_ratio: float,
    kl_weight: float,
    entropy_weight: float,
    sampling_epsilon: float,
    sampling_temperature: float,
    sampling_top_k: int,
) -> dict[str, torch.Tensor]:
    current_probs, reference_probs = _exploration_probabilities(
        action_scores=action_scores,
        reference_scores=reference_scores,
        legal_action_mask=legal_action_mask,
        epsilon=sampling_epsilon,
        temperature=sampling_temperature,
        top_k=sampling_top_k,
    )
    rows = torch.arange(action_scores.shape[0], device=action_scores.device)
    chosen_probs = current_probs[rows, target_action_row].clamp_min(1e-12)
    chosen_log_probs = chosen_probs.log()
    ratio = torch.exp(chosen_log_probs - behavior_log_probability)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = _weighted_mean(-surrogate, sample_weight)

    current_log_probs = current_probs.clamp_min(1e-12).log()
    reference_log_probs = reference_probs.clamp_min(1e-12).log()
    per_row_kl = (
        reference_probs * (reference_log_probs - current_log_probs)
    ).sum(dim=1)
    kl_loss = _weighted_mean(per_row_kl, sample_weight)
    per_row_entropy = -(current_probs * current_log_probs).sum(dim=1)
    entropy = _weighted_mean(per_row_entropy, sample_weight)
    total_loss = policy_loss + kl_weight * kl_loss - entropy_weight * entropy
    clip_fraction = _weighted_mean(
        (torch.abs(ratio - 1.0) > clip_ratio).to(action_scores.dtype),
        sample_weight,
    )
    approx_ratio = _weighted_mean(ratio, sample_weight)
    reference_chosen_log_probs = reference_probs[
        rows, target_action_row
    ].clamp_min(1e-12).log()
    behavior_log_prob_mae = _weighted_mean(
        torch.abs(reference_chosen_log_probs - behavior_log_probability),
        sample_weight,
    )
    return {
        "policy_loss": policy_loss,
        "kl_loss": kl_loss,
        "entropy": entropy,
        "total_loss": total_loss,
        "clip_fraction": clip_fraction,
        "approx_ratio": approx_ratio,
        "behavior_log_prob_mae": behavior_log_prob_mae,
    }


def _exploration_probabilities(
    action_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    epsilon: float,
    temperature: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_reference = masked_legal_action_scores(
        reference_scores,
        legal_action_mask,
    )
    legal_counts = legal_action_mask.sum(dim=1)
    max_support = min(top_k, action_scores.shape[1])
    top_rows = torch.topk(masked_reference, k=max_support, dim=1).indices
    support_mask = torch.zeros_like(legal_action_mask)
    support_mask.scatter_(1, top_rows, True)
    support_mask &= legal_action_mask

    current_support_logits = (action_scores / temperature).masked_fill(
        ~support_mask,
        -torch.inf,
    )
    reference_support_logits = (reference_scores / temperature).masked_fill(
        ~support_mask,
        -torch.inf,
    )
    current_probs = epsilon * torch.softmax(current_support_logits, dim=1)
    reference_probs = epsilon * torch.softmax(reference_support_logits, dim=1)
    greedy_rows = torch.argmax(masked_reference, dim=1)
    greedy_bonus = torch.full(
        (action_scores.shape[0], 1),
        1.0 - epsilon,
        dtype=action_scores.dtype,
        device=action_scores.device,
    )
    current_probs.scatter_add_(1, greedy_rows.unsqueeze(1), greedy_bonus)
    reference_probs.scatter_add_(1, greedy_rows.unsqueeze(1), greedy_bonus)
    if bool((legal_counts <= 0).any().item()):
        raise ValueError("Every row must have at least one legal action.")
    return current_probs, reference_probs


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    normalized_weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * normalized_weights).sum() / normalized_weights.sum()


def _load_attention_model(
    path: Path,
    device: str,
) -> tuple[LegalActionAttentionScorerModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = LegalActionAttentionScorerModel(
        LegalActionAttentionScorerConfig(**checkpoint.get("model_config", {}))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def _validate_config(config: TrajectoryPolicyGradientConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 <= config.clip_ratio < 1.0:
        raise ValueError("clip_ratio must be in [0, 1).")
    if config.kl_weight < 0.0 or config.entropy_weight < 0.0:
        raise ValueError("kl_weight and entropy_weight must be non-negative.")
    if not 0.0 < config.sampling_epsilon <= 1.0:
        raise ValueError("sampling_epsilon must be in (0, 1].")
    if config.sampling_temperature <= 0.0 or config.sampling_top_k <= 0:
        raise ValueError("sampling temperature and top_k must be positive.")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")


if __name__ == "__main__":
    main()
