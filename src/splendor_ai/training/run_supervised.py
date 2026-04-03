"""CLI entry point for supervised warm-start training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .model import PolicyValueMLP, PolicyValueModelConfig
from .supervised import SupervisedTrainConfig, fit_supervised


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline Splendor policy/value model from replay JSONL data.")
    parser.add_argument("--replay-path", required=True, help="Path to replay JSONL exported by the replay collector.")
    parser.add_argument("--output-dir", required=True, help="Directory where checkpoints and metrics should be written.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trunk-hidden-size", type=int, default=256)
    parser.add_argument("--trunk-depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--exclude-stalled-games",
        action="store_true",
        help="Ignore replay rows from games that ended in the stalled-state fallback.",
    )
    parser.add_argument(
        "--exclude-timeout-games",
        action="store_true",
        help="Ignore replay rows from games that hit the max-turn cap fallback.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PolicyValueMLP(
        PolicyValueModelConfig(
            trunk_hidden_size=args.trunk_hidden_size,
            trunk_depth=args.trunk_depth,
            dropout=args.dropout,
        )
    )
    model, metrics = fit_supervised(
        replay_path=args.replay_path,
        config=SupervisedTrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            value_loss_weight=args.value_loss_weight,
            epochs=args.epochs,
            device=args.device,
            include_stalled_games=not args.exclude_stalled_games,
            include_timed_out_games=not args.exclude_timeout_games,
        ),
        model=model,
    )

    checkpoint_path = output_dir / "supervised_policy_value.pt"
    metrics_path = output_dir / "supervised_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "train_config": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "value_loss_weight": args.value_loss_weight,
                "epochs": args.epochs,
                "device": args.device,
                "include_stalled_games": not args.exclude_stalled_games,
                "include_timed_out_games": not args.exclude_timeout_games,
            },
        },
        checkpoint_path,
    )
    metrics_payload = [
        {
            "policy_loss": metric.policy_loss,
            "value_loss": metric.value_loss,
            "total_loss": metric.total_loss,
            "batches": metric.batches,
        }
        for metric in metrics
    ]
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
