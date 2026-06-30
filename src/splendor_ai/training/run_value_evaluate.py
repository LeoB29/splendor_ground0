"""CLI entry point for evaluating a trained value model on replay JSONL data."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from .dataset import ValueReplayDataset
from .model import ValueMLP, ValueModelConfig
from .value_supervised import create_value_replay_dataloader, evaluate_value_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Splendor value model checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Replay JSONL path. Repeat to concatenate multiple corpora.",
    )
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margin-scale", type=float, default=15.0)
    parser.add_argument("--exclude-stalled-games", action="store_true")
    parser.add_argument("--exclude-timeout-games", action="store_true")
    parser.add_argument("--max-game-model-loop-fallback-triggers", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ValueMLP(ValueModelConfig(**checkpoint.get("model_config", {})))
    model.load_state_dict(checkpoint["model_state_dict"])
    replay_paths = tuple(Path(path) for path in args.replay_path)
    dataset = ValueReplayDataset(
        replay_paths,
        include_stalled_games=not args.exclude_stalled_games,
        include_timed_out_games=not args.exclude_timeout_games,
        max_model_loop_fallback_triggers=args.max_game_model_loop_fallback_triggers,
        margin_scale=args.margin_scale,
    )
    dataloader = create_value_replay_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    metric = evaluate_value_epoch(model, dataloader, device=args.device)
    payload = {
        "checkpoint": str(checkpoint_path),
        "replay_paths": [str(path) for path in replay_paths],
        "samples": len(dataset),
        "margin_scale": args.margin_scale,
        "metric": asdict(metric),
    }
    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[value-eval] wrote metrics: {output_path}")
    print(
        "Value evaluation:"
        f" samples={metric.samples}"
        f" mse={metric.mse_loss:.4f}"
        f" mae={metric.mae:.4f}"
        f" sign_acc={metric.sign_accuracy:.4f}"
        f" outcome_sign_acc={metric.outcome_sign_accuracy:.4f}"
        f" pearson={metric.pearson_correlation:.4f}"
    )


if __name__ == "__main__":
    main()
