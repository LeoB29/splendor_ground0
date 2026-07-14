"""Train a policy from complete-rollout action returns with parent KL."""

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
class RolloutCreditTrainingConfig:
    replay_paths: tuple[Path, ...]
    init_checkpoint: Path
    output_dir: Path
    anchor_replay_paths: tuple[Path, ...] = ()
    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 3e-6
    weight_decay: float = 1e-4
    target_temperature: float = 0.25
    parent_kl_weight: float = 2.0
    anchor_kl_weight: float = 2.0
    anchor_batch_size: int = 128
    entropy_weight: float = 0.0
    max_grad_norm: float = 1.0
    device: str = "cpu"
    log_every_batches: int = 10
    seed: int = 34


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train from complete-rollout action-return distributions."
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--anchor-replay-path", action="append", default=[])
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-temperature", type=float, default=0.25)
    parser.add_argument("--parent-kl-weight", type=float, default=2.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=2.0)
    parser.add_argument("--anchor-batch-size", type=int, default=128)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=34)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = train_rollout_credit_policy(
        RolloutCreditTrainingConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            anchor_replay_paths=tuple(
                Path(path) for path in args.anchor_replay_path
            ),
            init_checkpoint=Path(args.init_checkpoint),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            target_temperature=args.target_temperature,
            parent_kl_weight=args.parent_kl_weight,
            anchor_kl_weight=args.anchor_kl_weight,
            anchor_batch_size=args.anchor_batch_size,
            entropy_weight=args.entropy_weight,
            max_grad_norm=args.max_grad_norm,
            device=args.device,
            log_every_batches=args.log_every_batches,
            seed=args.seed,
        )
    )
    final = summary["epochs"][-1]
    print(
        "Rollout-credit training complete:"
        f" samples={final['samples']}"
        f" target_loss={final['target_loss']:.6f}"
        f" parent_kl={final['parent_kl']:.6f}"
        f" target_accuracy={final['target_accuracy']:.4f}"
    )


def train_rollout_credit_policy(
    config: RolloutCreditTrainingConfig,
) -> dict[str, Any]:
    _validate_config(config)
    torch.manual_seed(config.seed)
    model, checkpoint = _load_model(config.init_checkpoint, config.device)
    parent, _ = _load_model(config.init_checkpoint, config.device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    dataset = LegalActionReplayDataset(
        config.replay_paths,
        include_stalled_games=False,
        include_timed_out_games=False,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    dataloader = create_legal_action_replay_dataloader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    anchor_dataloader = None
    if config.anchor_replay_paths:
        anchor_dataset = LegalActionReplayDataset(
            config.anchor_replay_paths,
            include_stalled_games=False,
            include_timed_out_games=False,
            observation_size=model.config.observation_size,
            action_feature_size=model.config.action_feature_size,
        )
        anchor_dataloader = create_legal_action_replay_dataloader(
            anchor_dataset,
            batch_size=config.anchor_batch_size,
            shuffle=True,
        )
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_summaries = []
    training_start = time.perf_counter()
    for epoch_index in range(1, config.epochs + 1):
        model.train()
        totals = {
            "target_loss": 0.0,
            "parent_kl": 0.0,
            "anchor_kl": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "target_accuracy": 0.0,
        }
        samples = 0
        batches = 0
        epoch_start = time.perf_counter()
        anchor_iterator = iter(anchor_dataloader) if anchor_dataloader else None
        for batch_index, batch in enumerate(dataloader, start=1):
            observation = batch["observation"].to(config.device)
            action_features = batch["legal_action_features"].to(config.device)
            legal_mask = batch["legal_action_mask"].to(config.device)
            rollout_targets = batch["search_value_target"].to(config.device)
            rollout_mask = batch["search_value_mask"].to(config.device)
            sample_weight = batch["sample_weight"].to(config.device)

            optimizer.zero_grad(set_to_none=True)
            action_scores, _value = model(
                observation,
                action_features,
                legal_mask,
            )
            with torch.no_grad():
                parent_scores, _parent_value = parent(
                    observation,
                    action_features,
                    legal_mask,
                )
            losses = compute_rollout_credit_loss(
                action_scores=action_scores,
                parent_scores=parent_scores,
                legal_action_mask=legal_mask,
                rollout_return_target=rollout_targets,
                rollout_action_mask=rollout_mask,
                sample_weight=sample_weight,
                target_temperature=config.target_temperature,
                parent_kl_weight=config.parent_kl_weight,
                entropy_weight=config.entropy_weight,
            )
            anchor_kl = action_scores.sum() * 0.0
            if anchor_dataloader is not None and anchor_iterator is not None:
                try:
                    anchor_batch = next(anchor_iterator)
                except StopIteration:
                    anchor_iterator = iter(anchor_dataloader)
                    anchor_batch = next(anchor_iterator)
                anchor_observation = anchor_batch["observation"].to(config.device)
                anchor_features = anchor_batch["legal_action_features"].to(
                    config.device
                )
                anchor_mask = anchor_batch["legal_action_mask"].to(config.device)
                anchor_weights = anchor_batch["sample_weight"].to(config.device)
                anchor_scores, _anchor_value = model(
                    anchor_observation,
                    anchor_features,
                    anchor_mask,
                )
                with torch.no_grad():
                    parent_anchor_scores, _parent_anchor_value = parent(
                        anchor_observation,
                        anchor_features,
                        anchor_mask,
                    )
                anchor_kl = compute_parent_policy_kl(
                    action_scores=anchor_scores,
                    parent_scores=parent_anchor_scores,
                    legal_action_mask=anchor_mask,
                    sample_weight=anchor_weights,
                )
                losses["total_loss"] = (
                    losses["total_loss"]
                    + config.anchor_kl_weight * anchor_kl
                )
            losses["anchor_kl"] = anchor_kl
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            batch_samples = int(observation.shape[0])
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
                    f"[rollout-train] epoch={epoch_index}/{config.epochs}"
                    f" batch={batch_index}/{len(dataloader)}"
                    f" target={totals['target_loss'] / samples:.6f}"
                    f" parent_kl={totals['parent_kl'] / samples:.6f}"
                    f" anchor_kl={totals['anchor_kl'] / samples:.6f}"
                    f" accuracy={totals['target_accuracy'] / samples:.4f}"
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
    metrics_path = config.output_dir / "rollout_credit_metrics.json"
    train_config = {
        **asdict(config),
        "replay_paths": [str(path) for path in config.replay_paths],
        "anchor_replay_paths": [
            str(path) for path in config.anchor_replay_paths
        ],
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
        "training_seconds": time.perf_counter() - training_start,
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[rollout-train] wrote checkpoint: {checkpoint_path}")
    print(f"[rollout-train] wrote metrics: {metrics_path}")
    return summary


def compute_rollout_credit_loss(
    action_scores: torch.Tensor,
    parent_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    rollout_return_target: torch.Tensor,
    rollout_action_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    target_temperature: float,
    parent_kl_weight: float,
    entropy_weight: float,
) -> dict[str, torch.Tensor]:
    effective_mask = legal_action_mask & rollout_action_mask
    if bool((effective_mask.sum(dim=1) < 2).any().item()):
        raise ValueError("Every rollout-credit row must score at least two legal actions.")

    model_target_logits = action_scores.masked_fill(~effective_mask, -torch.inf)
    rollout_logits = (rollout_return_target / target_temperature).masked_fill(
        ~effective_mask,
        -torch.inf,
    )
    model_target_log_probs = torch.log_softmax(model_target_logits, dim=1)
    target_probs = torch.softmax(rollout_logits, dim=1).detach()
    model_target_log_probs = model_target_log_probs.masked_fill(
        ~effective_mask,
        0.0,
    )
    target_probs = target_probs.masked_fill(~effective_mask, 0.0)
    per_row_target_loss = -(target_probs * model_target_log_probs).sum(dim=1)
    target_loss = _weighted_mean(per_row_target_loss, sample_weight)

    masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)
    model_log_probs = torch.log_softmax(masked_scores, dim=1)
    model_probs = torch.softmax(masked_scores, dim=1)
    model_log_probs = model_log_probs.masked_fill(~legal_action_mask, 0.0)
    model_probs = model_probs.masked_fill(~legal_action_mask, 0.0)
    parent_kl = compute_parent_policy_kl(
        action_scores=action_scores,
        parent_scores=parent_scores,
        legal_action_mask=legal_action_mask,
        sample_weight=sample_weight,
    )
    per_row_entropy = -(model_probs * model_log_probs).sum(dim=1)
    entropy = _weighted_mean(per_row_entropy, sample_weight)
    total_loss = (
        target_loss
        + parent_kl_weight * parent_kl
        - entropy_weight * entropy
    )

    target_rows = torch.argmax(rollout_logits, dim=1)
    predicted_rows = torch.argmax(model_target_logits, dim=1)
    target_accuracy = _weighted_mean(
        (target_rows == predicted_rows).to(action_scores.dtype),
        sample_weight,
    )
    return {
        "target_loss": target_loss,
        "parent_kl": parent_kl,
        "entropy": entropy,
        "total_loss": total_loss,
        "target_accuracy": target_accuracy,
    }


def compute_parent_policy_kl(
    action_scores: torch.Tensor,
    parent_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)
    masked_parent = masked_legal_action_scores(parent_scores, legal_action_mask)
    model_log_probs = torch.log_softmax(masked_scores, dim=1).masked_fill(
        ~legal_action_mask,
        0.0,
    )
    parent_log_probs = torch.log_softmax(masked_parent, dim=1).masked_fill(
        ~legal_action_mask,
        0.0,
    )
    parent_probs = torch.softmax(masked_parent, dim=1).masked_fill(
        ~legal_action_mask,
        0.0,
    ).detach()
    per_row_kl = (
        parent_probs * (parent_log_probs - model_log_probs)
    ).sum(dim=1)
    return _weighted_mean(per_row_kl, sample_weight)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    typed_weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * typed_weights).sum() / typed_weights.sum()


def _load_model(
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


def _validate_config(config: RolloutCreditTrainingConfig) -> None:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if (
        config.epochs <= 0
        or config.batch_size <= 0
        or config.anchor_batch_size <= 0
    ):
        raise ValueError("epochs and batch sizes must be positive.")
    if config.learning_rate <= 0.0 or config.target_temperature <= 0.0:
        raise ValueError("learning_rate and target_temperature must be positive.")
    if (
        config.parent_kl_weight < 0.0
        or config.anchor_kl_weight < 0.0
        or config.entropy_weight < 0.0
    ):
        raise ValueError("KL and entropy weights must be non-negative.")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")


if __name__ == "__main__":
    main()
