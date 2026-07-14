"""Train a conflict-free PUCT root-visit correction with broad parent-KL anchors."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from .dataset import LegalActionReplayDataset, collate_legal_action_replay_samples
from .legal_action_supervised import compute_root_visit_distillation_loss, split_legal_action_replay_dataset
from .model import LegalActionAttentionScorerConfig, LegalActionAttentionScorerModel, masked_legal_action_scores


@dataclass(frozen=True, slots=True)
class RootVisitCorrectionConfig:
    split_replay_paths: tuple[Path, ...]
    target_source_path: Path
    anchor_replay_paths: tuple[Path, ...]
    init_checkpoint: Path
    output_dir: Path
    epochs: int = 2
    batch_size: int = 128
    anchor_batch_size: int = 128
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    anchor_kl_weight: float = 2.0
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.1
    validation_seed: int = 36
    device: str = "cpu"
    log_every_batches: int = 25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train soft PUCT root visits without one-hot replay CE, anchored by "
            "frozen-parent KL on a separate broad legal stream."
        )
    )
    parser.add_argument(
        "--split-replay-path",
        action="append",
        required=True,
        help="Every replay source used by policy036, in the original order, to reproduce its split.",
    )
    parser.add_argument("--target-source-path", required=True)
    parser.add_argument("--anchor-replay-path", action="append", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--anchor-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--anchor-kl-weight", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=36)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every-batches", type=int, default=25)
    return parser


def compute_parent_policy_kl(
    action_scores: torch.Tensor,
    parent_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
) -> torch.Tensor:
    model_log_probs = torch.log_softmax(
        masked_legal_action_scores(action_scores, legal_action_mask), dim=1
    ).masked_fill(~legal_action_mask, 0.0)
    parent_masked_scores = masked_legal_action_scores(parent_scores, legal_action_mask)
    parent_log_probs = torch.log_softmax(parent_masked_scores, dim=1).masked_fill(
        ~legal_action_mask, 0.0
    )
    parent_probs = torch.softmax(parent_masked_scores, dim=1).masked_fill(
        ~legal_action_mask, 0.0
    ).detach()
    return (parent_probs * (parent_log_probs - model_log_probs)).sum(dim=1).mean()


def compute_root_visit_correction_loss(
    target_scores: torch.Tensor,
    target_mask: torch.Tensor,
    root_visit_target: torch.Tensor,
    root_visit_mask: torch.Tensor,
    anchor_scores: torch.Tensor,
    parent_anchor_scores: torch.Tensor,
    anchor_mask: torch.Tensor,
    anchor_kl_weight: float,
) -> dict[str, torch.Tensor]:
    root_visit_loss = compute_root_visit_distillation_loss(
        action_scores=target_scores,
        legal_action_mask=target_mask,
        root_visit_target=root_visit_target,
        root_visit_mask=root_visit_mask,
    )
    anchor_kl = compute_parent_policy_kl(
        action_scores=anchor_scores,
        parent_scores=parent_anchor_scores,
        legal_action_mask=anchor_mask,
    )
    return {
        "root_visit_loss": root_visit_loss,
        "anchor_kl": anchor_kl,
        "total_loss": root_visit_loss + anchor_kl_weight * anchor_kl,
    }


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[LegalActionAttentionScorerModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "legal_action_attention_scorer":
        raise ValueError("Root-visit correction requires an attention legal-action checkpoint.")
    config_keys = {field.name for field in fields(LegalActionAttentionScorerConfig)}
    model_config = LegalActionAttentionScorerConfig(
        **{
            key: value
            for key, value in checkpoint.get("model_config", {}).items()
            if key in config_keys
        }
    )
    model = LegalActionAttentionScorerModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def _target_indices(
    dataset: LegalActionReplayDataset,
    split_dataset: Subset,
    target_source: Path,
) -> list[int]:
    target_source = target_source.resolve()
    indices = [
        index
        for index in split_dataset.indices
        if dataset.source_path_for_index(index).resolve() == target_source
    ]
    if not indices:
        raise ValueError("Selected split has no rows from --target-source-path.")
    return indices


def _dataloader(dataset: Subset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_legal_action_replay_samples,
    )


def _evaluate_target_uptake(
    model: LegalActionAttentionScorerModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"root_visit_loss": 0.0, "policy_entropy": 0.0, "top1_agreement": 0.0}
    rows = 0
    with torch.inference_mode():
        for batch in dataloader:
            observation = batch["observation"].to(device)
            action_features = batch["legal_action_features"].to(device)
            legal_mask = batch["legal_action_mask"].to(device)
            root_target = batch["root_visit_target"].to(device)
            root_mask = batch["root_visit_mask"].to(device)
            scores, _ = model(observation, action_features, legal_mask)
            root_loss = compute_root_visit_distillation_loss(
                action_scores=scores,
                legal_action_mask=legal_mask,
                root_visit_target=root_target,
                root_visit_mask=root_mask,
            )
            masked_scores = masked_legal_action_scores(scores, legal_mask)
            log_probs = torch.log_softmax(masked_scores, dim=1).masked_fill(~legal_mask, 0.0)
            probs = log_probs.exp().masked_fill(~legal_mask, 0.0)
            valid_rows = root_mask & (root_target.sum(dim=1) > 0.0)
            target_top = root_target.argmax(dim=1)
            model_top = probs.argmax(dim=1)
            row_entropy = -(probs * log_probs).sum(dim=1)
            count = int(valid_rows.sum().item())
            if count == 0:
                continue
            totals["root_visit_loss"] += float(root_loss.item()) * count
            totals["policy_entropy"] += float(row_entropy[valid_rows].sum().item())
            totals["top1_agreement"] += float(
                (target_top[valid_rows] == model_top[valid_rows]).sum().item()
            )
            rows += count
    if rows == 0:
        raise ValueError("Target validation loader has no valid root-visit rows.")
    return {key: value / rows for key, value in totals.items()} | {"rows": rows}


def train_root_visit_correction(config: RootVisitCorrectionConfig) -> dict[str, Any]:
    _validate_config(config)
    device = torch.device(config.device)
    torch.manual_seed(config.validation_seed)
    model, checkpoint = _load_model(config.init_checkpoint, device)
    parent, _ = _load_model(config.init_checkpoint, device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    split_dataset = LegalActionReplayDataset(
        config.split_replay_paths,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    train_split, validation_split = split_legal_action_replay_dataset(
        split_dataset,
        validation_fraction=config.validation_fraction,
        seed=config.validation_seed,
        validation_split_unit="game",
    )
    if not isinstance(train_split, Subset) or not isinstance(validation_split, Subset):
        raise ValueError("Root-visit correction requires a non-empty game-level validation split.")
    target_train_indices = _target_indices(split_dataset, train_split, config.target_source_path)
    target_validation_indices = _target_indices(split_dataset, validation_split, config.target_source_path)
    target_train_loader = _dataloader(Subset(split_dataset, target_train_indices), config.batch_size, True)
    target_validation_loader = _dataloader(
        Subset(split_dataset, target_validation_indices), config.batch_size, False
    )

    anchor_dataset = LegalActionReplayDataset(
        config.anchor_replay_paths,
        observation_size=model.config.observation_size,
        action_feature_size=model.config.action_feature_size,
    )
    anchor_loader = _dataloader(Subset(anchor_dataset, list(range(len(anchor_dataset)))), config.anchor_batch_size, True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    started = time.perf_counter()
    epochs: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch = None
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"root_visit_loss": 0.0, "anchor_kl": 0.0, "total_loss": 0.0}
        rows = 0
        anchor_iterator = iter(anchor_loader)
        epoch_started = time.perf_counter()
        for batch_index, target_batch in enumerate(target_train_loader, start=1):
            try:
                anchor_batch = next(anchor_iterator)
            except StopIteration:
                anchor_iterator = iter(anchor_loader)
                anchor_batch = next(anchor_iterator)
            target_observation = target_batch["observation"].to(device)
            target_features = target_batch["legal_action_features"].to(device)
            target_mask = target_batch["legal_action_mask"].to(device)
            root_target = target_batch["root_visit_target"].to(device)
            root_mask = target_batch["root_visit_mask"].to(device)
            anchor_observation = anchor_batch["observation"].to(device)
            anchor_features = anchor_batch["legal_action_features"].to(device)
            anchor_mask = anchor_batch["legal_action_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            target_scores, _ = model(target_observation, target_features, target_mask)
            anchor_scores, _ = model(anchor_observation, anchor_features, anchor_mask)
            with torch.no_grad():
                parent_anchor_scores, _ = parent(anchor_observation, anchor_features, anchor_mask)
            losses = compute_root_visit_correction_loss(
                target_scores=target_scores,
                target_mask=target_mask,
                root_visit_target=root_target,
                root_visit_mask=root_mask,
                anchor_scores=anchor_scores,
                parent_anchor_scores=parent_anchor_scores,
                anchor_mask=anchor_mask,
                anchor_kl_weight=config.anchor_kl_weight,
            )
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            batch_rows = int(target_observation.shape[0])
            rows += batch_rows
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_rows
            if (
                batch_index == 1
                or batch_index == len(target_train_loader)
                or (config.log_every_batches > 0 and batch_index % config.log_every_batches == 0)
            ):
                print(
                    f"[root-visit-correction] epoch={epoch}/{config.epochs}"
                    f" batch={batch_index}/{len(target_train_loader)}"
                    f" root_ce={totals['root_visit_loss'] / rows:.5f}"
                    f" anchor_kl={totals['anchor_kl'] / rows:.5f}"
                    f" elapsed={time.perf_counter() - epoch_started:.1f}s"
                )

        validation = _evaluate_target_uptake(model, target_validation_loader, device)
        train_metrics = {key: value / rows for key, value in totals.items()} | {"rows": rows}
        epoch_summary = {"epoch": epoch, "train": train_metrics, "validation": validation}
        epochs.append(epoch_summary)
        if validation["root_visit_loss"] < best_validation_loss:
            best_validation_loss = validation["root_visit_loss"]
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.output_dir / "legal_action_attention_scorer.pt"
    best_path = config.output_dir / "legal_action_attention_scorer_best.pt"
    metrics_path = config.output_dir / "root_visit_correction_metrics.json"
    config_payload = asdict(config) | {
        "split_replay_paths": [str(path) for path in config.split_replay_paths],
        "target_source_path": str(config.target_source_path),
        "anchor_replay_paths": [str(path) for path in config.anchor_replay_paths],
        "init_checkpoint": str(config.init_checkpoint),
        "output_dir": str(config.output_dir),
    }
    checkpoint_payload = {
        "model_kind": "legal_action_attention_scorer",
        "model_config": checkpoint["model_config"],
        "train_config": config_payload,
    }
    torch.save(checkpoint_payload | {"model_state_dict": model.state_dict()}, final_path)
    if best_state is None or best_epoch is None:
        raise RuntimeError("Root-visit correction did not produce a validation checkpoint.")
    torch.save(checkpoint_payload | {"model_state_dict": best_state}, best_path)
    summary = {
        "config": config_payload,
        "split_dataset_rows": len(split_dataset),
        "target_train_rows": len(target_train_indices),
        "target_validation_rows": len(target_validation_indices),
        "anchor_rows": len(anchor_dataset),
        "best_validation_epoch": best_epoch,
        "epochs": epochs,
        "training_seconds": time.perf_counter() - started,
        "final_checkpoint": str(final_path),
        "best_checkpoint": str(best_path),
    }
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[root-visit-correction] wrote final checkpoint: {final_path}")
    print(f"[root-visit-correction] wrote best checkpoint: {best_path} (epoch {best_epoch})")
    print(f"[root-visit-correction] wrote metrics: {metrics_path}")
    return summary


def _validate_config(config: RootVisitCorrectionConfig) -> None:
    if not config.split_replay_paths or not config.anchor_replay_paths:
        raise ValueError("Both split replay paths and anchor replay paths are required.")
    if config.epochs <= 0 or config.batch_size <= 0 or config.anchor_batch_size <= 0:
        raise ValueError("Epoch and batch sizes must be positive.")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    if config.anchor_kl_weight < 0.0 or config.max_grad_norm <= 0.0:
        raise ValueError("Anchor KL weight must be non-negative and max grad norm positive.")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("Validation fraction must be in the range (0, 1).")


def main() -> None:
    args = build_parser().parse_args()
    summary = train_root_visit_correction(
        RootVisitCorrectionConfig(
            split_replay_paths=tuple(Path(path) for path in args.split_replay_path),
            target_source_path=Path(args.target_source_path),
            anchor_replay_paths=tuple(Path(path) for path in args.anchor_replay_path),
            init_checkpoint=Path(args.init_checkpoint),
            output_dir=Path(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            anchor_batch_size=args.anchor_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            anchor_kl_weight=args.anchor_kl_weight,
            max_grad_norm=args.max_grad_norm,
            validation_fraction=args.validation_fraction,
            validation_seed=args.validation_seed,
            device=args.device,
            log_every_batches=args.log_every_batches,
        )
    )
    validation = summary["epochs"][summary["best_validation_epoch"] - 1]["validation"]
    print(
        "Root-visit correction complete:"
        f" validation_ce={validation['root_visit_loss']:.5f}"
        f" validation_entropy={validation['policy_entropy']:.5f}"
        f" validation_top1={validation['top1_agreement']:.4f}"
    )


if __name__ == "__main__":
    main()
