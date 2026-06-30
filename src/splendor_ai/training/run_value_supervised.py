"""CLI entry point for supervised value-model training."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .dataset import ValueReplayDataset
from .model import ValueMLP, ValueModelConfig
from .supervised import BatchProgress, SupervisedTrainConfig
from .value_supervised import (
    create_value_replay_dataloader,
    fit_value_dataloaders_with_artifacts,
    split_value_replay_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Splendor state-value model from replay JSONL data."
    )
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Replay JSONL path. Repeat to mix multiple corpora.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--margin-scale",
        type=float,
        default=15.0,
        help="Final score margin divisor used to map value targets into [-1, 1].",
    )
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument(
        "--validation-split-unit",
        choices=("row", "game", "source"),
        default="row",
        help="Validation split granularity. Use 'game' to avoid rows from the same game leaking across splits.",
    )
    parser.add_argument("--exclude-stalled-games", action="store_true")
    parser.add_argument("--exclude-timeout-games", action="store_true")
    parser.add_argument("--max-game-model-loop-fallback-triggers", type=int, default=None)
    return parser


def _build_model(args: argparse.Namespace) -> ValueMLP:
    if args.init_checkpoint is None:
        return ValueMLP(
            ValueModelConfig(
                hidden_size=args.hidden_size,
                depth=args.depth,
                dropout=args.dropout,
            )
        )

    checkpoint_path = Path(args.init_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = ValueModelConfig(**checkpoint.get("model_config", {}))
    model = ValueMLP(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[value-train] initialized model from checkpoint: {checkpoint_path}")
    return model


def _save_checkpoint(
    path: Path,
    model: ValueMLP,
    train_config_payload: dict[str, object],
) -> None:
    torch.save(
        {
            "model_kind": "value_mlp",
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
        f" mse={progress.value_loss:.4f}"
        f" sign_acc={progress.policy_accuracy:.4f}"
    )


def _serialize_metric(metric: object | None) -> dict[str, object] | None:
    return asdict(metric) if metric is not None else None


def _train_config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "replay_paths": [str(Path(path)) for path in args.replay_path],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "device": args.device,
        "init_checkpoint": args.init_checkpoint,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "dropout": args.dropout,
        "margin_scale": args.margin_scale,
        "include_stalled_games": not args.exclude_stalled_games,
        "include_timed_out_games": not args.exclude_timeout_games,
        "max_model_loop_fallback_triggers": args.max_game_model_loop_fallback_triggers,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "validation_split_unit": args.validation_split_unit,
        "log_every_batches": args.log_every_batches,
    }


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = tuple(Path(path) for path in args.replay_path)
    model = _build_model(args)

    print(
        f"[value-train] replay_paths={', '.join(str(path) for path in replay_paths)}"
        f" output_dir={output_dir}"
        f" device={args.device}"
        f" epochs={args.epochs}"
        f" batch_size={args.batch_size}"
        f" margin_scale={args.margin_scale}"
        f" validation_split_unit={args.validation_split_unit}"
    )
    config = SupervisedTrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        include_stalled_games=not args.exclude_stalled_games,
        include_timed_out_games=not args.exclude_timeout_games,
        max_model_loop_fallback_triggers=args.max_game_model_loop_fallback_triggers,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )

    print("[value-train] loading replay dataset...")
    dataset = ValueReplayDataset(
        replay_paths,
        include_stalled_games=config.include_stalled_games,
        include_timed_out_games=config.include_timed_out_games,
        max_model_loop_fallback_triggers=config.max_model_loop_fallback_triggers,
        margin_scale=args.margin_scale,
    )
    train_dataset, validation_dataset = split_value_replay_dataset(
        dataset,
        validation_fraction=config.validation_fraction,
        seed=config.validation_seed,
        validation_split_unit=args.validation_split_unit,
    )
    print(
        f"[value-train] dataset_samples={len(dataset)}"
        f" train_samples={len(train_dataset)}"
        f" validation_samples={len(validation_dataset) if validation_dataset is not None else 0}"
    )
    train_dataloader = create_value_replay_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_value_replay_dataloader(
            validation_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )

    print("[value-train] starting optimization...")
    start_time = time.perf_counter()
    artifacts = fit_value_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
        model=model,
        log_every_batches=args.log_every_batches,
        batch_progress_callback=_print_batch_progress,
    )
    elapsed = time.perf_counter() - start_time

    checkpoint_path = output_dir / "value_model.pt"
    best_checkpoint_path = output_dir / "value_model_best.pt"
    metrics_path = output_dir / "value_model_metrics.json"
    train_payload = _train_config_payload(args)
    _save_checkpoint(checkpoint_path, artifacts.model, train_payload)
    print(f"[value-train] wrote final checkpoint: {checkpoint_path}")

    best_checkpoint_written = False
    if artifacts.best_model_state_dict is not None and artifacts.best_epoch_index is not None:
        best_model = ValueMLP(artifacts.model.config)
        best_model.load_state_dict(artifacts.best_model_state_dict)
        _save_checkpoint(best_checkpoint_path, best_model, train_payload)
        best_checkpoint_written = True
        print(
            f"[value-train] wrote best validation checkpoint: {best_checkpoint_path}"
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
            "validation_split_unit": args.validation_split_unit,
            "margin_scale": args.margin_scale,
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
    print(f"[value-train] wrote metrics: {metrics_path}")

    final_epoch = artifacts.metrics[-1]
    print(
        "Value training complete:"
        f" train_mse={final_epoch.train.mse_loss:.4f}"
        f" train_mae={final_epoch.train.mae:.4f}"
        f" train_sign_acc={final_epoch.train.sign_accuracy:.4f}"
        f" train_pearson={final_epoch.train.pearson_correlation:.4f}"
        f" train_samples={final_epoch.train.samples}"
        f" training_seconds={elapsed:.1f}"
    )
    if final_epoch.validation is not None:
        print(
            "Validation:"
            f" mse={final_epoch.validation.mse_loss:.4f}"
            f" mae={final_epoch.validation.mae:.4f}"
            f" sign_acc={final_epoch.validation.sign_accuracy:.4f}"
            f" outcome_sign_acc={final_epoch.validation.outcome_sign_accuracy:.4f}"
            f" pearson={final_epoch.validation.pearson_correlation:.4f}"
            f" samples={final_epoch.validation.samples}"
            f" best_epoch={artifacts.best_epoch_index}"
        )


if __name__ == "__main__":
    main()
