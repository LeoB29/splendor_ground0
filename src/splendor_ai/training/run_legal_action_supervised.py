"""CLI entry point for supervised legal-action scorer training."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .dataset import LegalActionReplayDataset
from .legal_action_supervised import (
    create_legal_action_replay_dataloader,
    fit_legal_action_scorer_dataloaders_with_artifacts,
)
from .model import LegalActionScorerConfig, LegalActionScorerModel
from .supervised import BatchProgress, SupervisedTrainConfig, split_replay_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Splendor legal-action scorer from snapshot-enabled replay JSONL data."
    )
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Snapshot-enabled replay JSONL path. Repeat to mix multiple corpora.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--state-hidden-size", type=int, default=256)
    parser.add_argument("--action-hidden-size", type=int, default=128)
    parser.add_argument("--scorer-hidden-size", type=int, default=256)
    parser.add_argument("--state-depth", type=int, default=3)
    parser.add_argument("--action-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--exclude-stalled-games", action="store_true")
    parser.add_argument("--exclude-timeout-games", action="store_true")
    parser.add_argument("--max-game-model-loop-fallback-triggers", type=int, default=None)
    return parser


def _build_model(args: argparse.Namespace) -> LegalActionScorerModel:
    if args.init_checkpoint is None:
        return LegalActionScorerModel(
            LegalActionScorerConfig(
                state_hidden_size=args.state_hidden_size,
                action_hidden_size=args.action_hidden_size,
                scorer_hidden_size=args.scorer_hidden_size,
                state_depth=args.state_depth,
                action_depth=args.action_depth,
                dropout=args.dropout,
            )
        )

    checkpoint_path = Path(args.init_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = LegalActionScorerConfig(**checkpoint.get("model_config", {}))
    model = LegalActionScorerModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[legal-train] initialized model from checkpoint: {checkpoint_path}")
    return model


def _save_checkpoint(
    path: Path,
    model: LegalActionScorerModel,
    train_config_payload: dict[str, object],
) -> None:
    torch.save(
        {
            "model_kind": "legal_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "train_config": train_config_payload,
        },
        path,
    )


def _print_batch_progress(progress: BatchProgress) -> None:
    rate = progress.samples / progress.elapsed_seconds if progress.elapsed_seconds > 0 else 0.0
    eta = (
        ((progress.total_batches - progress.batch_index) * (progress.elapsed_seconds / progress.batch_index))
        if progress.batch_index > 0
        else math.inf
    )
    eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
    print(
        f"[{progress.phase}] epoch={progress.epoch_index}/{progress.total_epochs}"
        f" batch={progress.batch_index}/{progress.total_batches}"
        f" elapsed={progress.elapsed_seconds:.1f}s"
        f" rate={rate:.1f} samples/s"
        f" eta={eta_text}"
        f" loss={progress.total_loss:.4f}"
        f" policy_acc={progress.policy_accuracy:.4f}"
    )


def _serialize_metric(metric: object | None) -> dict[str, object] | None:
    return asdict(metric) if metric is not None else None


def _train_config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "replay_paths": [str(Path(path)) for path in args.replay_path],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "value_loss_weight": args.value_loss_weight,
        "epochs": args.epochs,
        "device": args.device,
        "init_checkpoint": args.init_checkpoint,
        "include_stalled_games": not args.exclude_stalled_games,
        "include_timed_out_games": not args.exclude_timeout_games,
        "max_model_loop_fallback_triggers": args.max_game_model_loop_fallback_triggers,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "log_every_batches": args.log_every_batches,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = tuple(Path(path) for path in args.replay_path)

    print(
        f"[legal-train] replay_paths={', '.join(str(path) for path in replay_paths)}"
        f" output_dir={output_dir}"
        f" device={args.device}"
        f" epochs={args.epochs}"
        f" batch_size={args.batch_size}"
    )
    model = _build_model(args)
    train_config = SupervisedTrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        epochs=args.epochs,
        device=args.device,
        include_stalled_games=not args.exclude_stalled_games,
        include_timed_out_games=not args.exclude_timeout_games,
        max_model_loop_fallback_triggers=args.max_game_model_loop_fallback_triggers,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )

    print("[legal-train] loading snapshot replay dataset...")
    dataset = LegalActionReplayDataset(
        replay_paths,
        include_stalled_games=train_config.include_stalled_games,
        include_timed_out_games=train_config.include_timed_out_games,
        max_model_loop_fallback_triggers=train_config.max_model_loop_fallback_triggers,
    )
    train_dataset, validation_dataset = split_replay_dataset(
        dataset,
        validation_fraction=train_config.validation_fraction,
        seed=train_config.validation_seed,
    )
    print(
        f"[legal-train] dataset_samples={len(dataset)}"
        f" train_samples={len(train_dataset)}"
        f" validation_samples={len(validation_dataset) if validation_dataset is not None else 0}"
    )
    train_dataloader = create_legal_action_replay_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=train_config.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_legal_action_replay_dataloader(
            validation_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
        )

    print("[legal-train] starting optimization...")
    start_time = time.perf_counter()
    artifacts = fit_legal_action_scorer_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=train_config,
        model=model,
        log_every_batches=args.log_every_batches,
        batch_progress_callback=_print_batch_progress,
    )
    elapsed = time.perf_counter() - start_time

    checkpoint_path = output_dir / "legal_action_scorer.pt"
    best_checkpoint_path = output_dir / "legal_action_scorer_best.pt"
    metrics_path = output_dir / "legal_action_scorer_metrics.json"
    train_payload = _train_config_payload(args)
    _save_checkpoint(checkpoint_path, artifacts.model, train_payload)
    print(f"[legal-train] wrote final checkpoint: {checkpoint_path}")

    best_checkpoint_written = False
    if artifacts.best_model_state_dict is not None and artifacts.best_epoch_index is not None:
        best_model = LegalActionScorerModel(artifacts.model.config)
        best_model.load_state_dict(artifacts.best_model_state_dict)
        _save_checkpoint(best_checkpoint_path, best_model, train_payload)
        best_checkpoint_written = True
        print(
            f"[legal-train] wrote best validation checkpoint: {best_checkpoint_path}"
            f" (epoch {artifacts.best_epoch_index})"
        )

    metrics_payload = {
        "dataset": {
            "replay_path": str(replay_paths[0]) if len(replay_paths) == 1 else None,
            "replay_paths": [str(path) for path in replay_paths],
            "total_samples": len(dataset),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            "validation_fraction": args.validation_fraction,
            "validation_seed": args.validation_seed,
            "excluded_stalled_games": args.exclude_stalled_games,
            "excluded_timeout_games": args.exclude_timeout_games,
            "max_game_model_loop_fallback_triggers": args.max_game_model_loop_fallback_triggers,
        },
        "checkpoints": {
            "final_checkpoint_path": checkpoint_path.name,
            "best_validation_checkpoint_path": best_checkpoint_path.name if best_checkpoint_written else None,
            "best_validation_epoch": artifacts.best_epoch_index,
        },
        "epochs": [
            {
                "train": _serialize_metric(metric.train),
                "validation": _serialize_metric(metric.validation),
                "is_best_validation": epoch_index == artifacts.best_epoch_index,
            }
            for epoch_index, metric in enumerate(artifacts.metrics, start=1)
        ],
        "timing": {
            "training_seconds": elapsed,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[legal-train] wrote metrics: {metrics_path}")

    final_epoch = artifacts.metrics[-1]
    print(
        "Legal-action scorer training complete:"
        f" train_loss={final_epoch.train.total_loss:.4f}"
        f" train_acc={final_epoch.train.policy_accuracy:.4f}"
        f" train_samples={final_epoch.train.samples}"
        f" training_seconds={elapsed:.1f}"
    )
    if final_epoch.validation is not None:
        print(
            "Validation:"
            f" loss={final_epoch.validation.total_loss:.4f}"
            f" acc={final_epoch.validation.policy_accuracy:.4f}"
            f" samples={final_epoch.validation.samples}"
            f" best_epoch={artifacts.best_epoch_index}"
        )


if __name__ == "__main__":
    main()
