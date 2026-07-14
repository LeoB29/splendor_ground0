"""Bucketed diagnostics for trained Splendor value checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from .dataset import ValueReplayDataset
from .model import ValueMLP, ValueModelConfig
from .value_supervised import split_value_replay_dataset


@dataclass(slots=True)
class _BucketAccumulator:
    predictions: list[float]
    targets: list[float]
    outcomes: list[float]
    outcome_logits: list[tuple[float, float, float]]

    @classmethod
    def empty(cls) -> "_BucketAccumulator":
        return cls(predictions=[], targets=[], outcomes=[], outcome_logits=[])

    def add(
        self,
        prediction: float,
        target: float,
        outcome: float,
        outcome_logits: tuple[float, float, float],
    ) -> None:
        self.predictions.append(prediction)
        self.targets.append(target)
        self.outcomes.append(outcome)
        self.outcome_logits.append(outcome_logits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bucketed diagnostics for a value checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Replay JSONL path. Repeat to concatenate multiple corpora.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--margin-scale", type=float, default=15.0)
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="validation",
        help="Subset to diagnose. Defaults to the validation side of the configured split.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument(
        "--validation-split-unit",
        choices=("row", "game", "source"),
        default="game",
    )
    parser.add_argument("--exclude-stalled-games", action="store_true")
    parser.add_argument("--exclude-timeout-games", action="store_true")
    parser.add_argument("--max-game-model-loop-fallback-triggers", type=int, default=None)
    parser.add_argument("--min-turn-index", type=int, default=None)
    parser.add_argument("--max-turn-index", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_value_diagnostics(
        checkpoint_path=Path(args.checkpoint),
        replay_paths=tuple(Path(path) for path in args.replay_path),
        output_path=Path(args.output_path),
        device=args.device,
        batch_size=args.batch_size,
        margin_scale=args.margin_scale,
        split=args.split,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
        validation_split_unit=args.validation_split_unit,
        include_stalled_games=not args.exclude_stalled_games,
        include_timed_out_games=not args.exclude_timeout_games,
        max_model_loop_fallback_triggers=args.max_game_model_loop_fallback_triggers,
        min_turn_index=args.min_turn_index,
        max_turn_index=args.max_turn_index,
    )
    print(
        "Value diagnostics:"
        f" split={payload['split']['selected']}"
        f" samples={payload['samples']}"
        f" mse={payload['overall']['mse']:.4f}"
        f" mae={payload['overall']['mae']:.4f}"
        f" sign_acc={payload['overall']['sign_accuracy']:.4f}"
        f" pearson={payload['overall']['pearson']:.4f}"
    )
    print(f"[value-diagnostics] wrote report: {args.output_path}")


def run_value_diagnostics(
    checkpoint_path: Path,
    replay_paths: tuple[Path, ...],
    output_path: Path,
    device: str = "cpu",
    batch_size: int = 512,
    margin_scale: float = 15.0,
    split: str = "validation",
    validation_fraction: float = 0.1,
    validation_seed: int = 0,
    validation_split_unit: str = "game",
    include_stalled_games: bool = True,
    include_timed_out_games: bool = True,
    max_model_loop_fallback_triggers: int | None = None,
    min_turn_index: int | None = None,
    max_turn_index: int | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = ValueMLP(ValueModelConfig(**checkpoint.get("model_config", {})))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    dataset = ValueReplayDataset(
        replay_paths,
        include_stalled_games=include_stalled_games,
        include_timed_out_games=include_timed_out_games,
        max_model_loop_fallback_triggers=max_model_loop_fallback_triggers,
        margin_scale=margin_scale,
        min_turn_index=min_turn_index,
        max_turn_index=max_turn_index,
    )
    selected_indices = _selected_indices(
        dataset=dataset,
        split=split,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        validation_split_unit=validation_split_unit,
    )
    if not selected_indices:
        raise ValueError("Selected diagnostic split is empty.")

    buckets: dict[str, dict[str, _BucketAccumulator]] = defaultdict(dict)
    overall = _BucketAccumulator.empty()
    examples: list[dict[str, Any]] = []
    for start in range(0, len(selected_indices), batch_size):
        batch_indices = selected_indices[start : start + batch_size]
        samples = [dataset[index] for index in batch_indices]
        observations = torch.stack([sample.observation for sample in samples], dim=0).to(device)
        with torch.no_grad():
            predictions = model(observations).detach().cpu().tolist()
            outcome_logits_batch = model.outcome_logits(observations).detach().cpu().tolist()

        for index, sample, prediction, outcome_logits in zip(
            batch_indices,
            samples,
            predictions,
            outcome_logits_batch,
            strict=True,
        ):
            target = float(sample.value_target)
            outcome = float(sample.outcome_target)
            row = dataset._samples[index]  # Same-package diagnostic access.
            prediction_value = float(prediction)
            outcome_logits_tuple = (
                float(outcome_logits[0]),
                float(outcome_logits[1]),
                float(outcome_logits[2]),
            )
            overall.add(prediction_value, target, outcome, outcome_logits_tuple)
            for group_name, bucket_name in _bucket_memberships(row, target):
                bucket = buckets[group_name].setdefault(bucket_name, _BucketAccumulator.empty())
                bucket.add(prediction_value, target, outcome, outcome_logits_tuple)
            examples.append(
                {
                    "absolute_error": abs(prediction_value - target),
                    "prediction": prediction_value,
                    "target": target,
                    "outcome": outcome,
                    "outcome_probabilities": _softmax3(outcome_logits_tuple),
                    "turn_index": int(row.get("turn_index", 0)),
                    "player_id": int(row.get("player_id", 0)),
                    "game_seed": row.get("game_seed"),
                    "source": Path(str(row["_source_path"])).parent.name,
                    "actor": _actor_label(row),
                }
            )

    examples.sort(key=lambda item: float(item["absolute_error"]), reverse=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "replay_paths": [str(path) for path in replay_paths],
        "samples": len(selected_indices),
        "margin_scale": margin_scale,
        "filters": {
            "include_stalled_games": include_stalled_games,
            "include_timed_out_games": include_timed_out_games,
            "max_model_loop_fallback_triggers": max_model_loop_fallback_triggers,
            "min_turn_index": min_turn_index,
            "max_turn_index": max_turn_index,
        },
        "split": {
            "selected": split,
            "validation_fraction": validation_fraction,
            "validation_seed": validation_seed,
            "validation_split_unit": validation_split_unit,
        },
        "overall": _metrics(overall),
        "buckets": {
            group_name: {
                bucket_name: _metrics(accumulator)
                for bucket_name, accumulator in sorted(group.items())
            }
            for group_name, group in sorted(buckets.items())
        },
        "largest_errors": examples[:25],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _selected_indices(
    dataset: ValueReplayDataset,
    split: str,
    validation_fraction: float,
    validation_seed: int,
    validation_split_unit: str,
) -> list[int]:
    if split == "all":
        return list(range(len(dataset)))
    train_dataset, validation_dataset = split_value_replay_dataset(
        dataset,
        validation_fraction=validation_fraction,
        seed=validation_seed,
        validation_split_unit=validation_split_unit,
    )
    selected = train_dataset if split == "train" else validation_dataset
    if selected is None:
        return list(range(len(dataset))) if split == "train" else []
    if isinstance(selected, Subset):
        return [int(index) for index in selected.indices]
    return list(range(len(dataset)))


def _bucket_memberships(row: dict[str, Any], target: float) -> tuple[tuple[str, str], ...]:
    turn_index = int(row.get("turn_index", 0))
    player_id = int(row.get("player_id", 0))
    player_score, opponent_score = _current_scores(row, player_id)
    score_leader = player_score - opponent_score
    return (
        ("phase", _phase_for_turn(turn_index)),
        ("turn_band", _turn_band(turn_index)),
        ("seat", f"seat{player_id}"),
        ("source", Path(str(row["_source_path"])).parent.name),
        ("actor", _actor_label(row)),
        ("current_score", _score_bucket(player_score)),
        ("opponent_score", _score_bucket(opponent_score)),
        ("race_state", _race_bucket(max(player_score, opponent_score))),
        ("current_score_margin", _score_margin_bucket(score_leader)),
        ("target_margin", _target_bucket(target)),
    )


def _metrics(accumulator: _BucketAccumulator) -> dict[str, float | int]:
    predictions = torch.tensor(accumulator.predictions, dtype=torch.float32)
    targets = torch.tensor(accumulator.targets, dtype=torch.float32)
    outcomes = torch.tensor(accumulator.outcomes, dtype=torch.float32)
    outcome_logits = torch.tensor(accumulator.outcome_logits, dtype=torch.float32)
    errors = predictions - targets
    probabilities = torch.softmax(outcome_logits, dim=1)
    return {
        "samples": int(targets.numel()),
        "mse": float(torch.mean(errors.square()).item()),
        "mae": float(torch.mean(errors.abs()).item()),
        "bias": float(torch.mean(errors).item()),
        "mean_prediction": float(torch.mean(predictions).item()),
        "mean_target": float(torch.mean(targets).item()),
        "sign_accuracy": _sign_accuracy(predictions, targets),
        "outcome_sign_accuracy": _sign_accuracy(predictions, outcomes),
        "outcome_class_accuracy": _outcome_class_accuracy(outcome_logits, outcomes),
        "mean_loss_probability": float(probabilities[:, 0].mean().item()),
        "mean_draw_probability": float(probabilities[:, 1].mean().item()),
        "mean_win_probability": float(probabilities[:, 2].mean().item()),
        "pearson": _pearson(predictions, targets),
    }


def _sign_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    mask = targets != 0
    if not bool(mask.any().item()):
        return math.nan
    return float((torch.sign(predictions[mask]) == torch.sign(targets[mask])).float().mean().item())


def _pearson(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    denominator = torch.linalg.vector_norm(pred_centered) * torch.linalg.vector_norm(target_centered)
    if float(denominator.item()) == 0.0:
        return math.nan
    return float((pred_centered * target_centered).sum().div(denominator).item())


def _outcome_classes(outcomes: torch.Tensor) -> torch.Tensor:
    return torch.where(
        outcomes < 0,
        torch.zeros_like(outcomes, dtype=torch.long),
        torch.where(
            outcomes > 0,
            torch.full_like(outcomes, 2, dtype=torch.long),
            torch.ones_like(outcomes, dtype=torch.long),
        ),
    )


def _outcome_class_accuracy(outcome_logits: torch.Tensor, outcomes: torch.Tensor) -> float:
    classes = _outcome_classes(outcomes)
    predictions = torch.argmax(outcome_logits, dim=1)
    return float((predictions == classes).float().mean().item())


def _softmax3(values: tuple[float, float, float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    probabilities = torch.softmax(tensor, dim=0).tolist()
    return {
        "loss": float(probabilities[0]),
        "draw": float(probabilities[1]),
        "win": float(probabilities[2]),
    }


def _phase_for_turn(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


def _turn_band(turn_index: int) -> str:
    lower = (turn_index // 10) * 10
    upper = lower + 9
    return f"{lower:03d}-{upper:03d}"


def _score_bucket(score: int) -> str:
    if score <= 0:
        return "00"
    if score <= 4:
        return "01-04"
    if score <= 9:
        return "05-09"
    if score <= 14:
        return "10-14"
    return "15+"


def _race_bucket(best_score: int) -> str:
    if best_score < 5:
        return "pre_race"
    if best_score < 10:
        return "mid_race"
    if best_score < 15:
        return "late_race"
    return "terminal_race"


def _score_margin_bucket(score_margin: int) -> str:
    if score_margin <= -5:
        return "behind_5plus"
    if score_margin < 0:
        return "behind_1_4"
    if score_margin == 0:
        return "tied"
    if score_margin < 5:
        return "ahead_1_4"
    return "ahead_5plus"


def _target_bucket(target: float) -> str:
    if target <= -0.75:
        return "large_loss"
    if target < 0:
        return "small_loss"
    if target == 0:
        return "draw"
    if target < 0.75:
        return "small_win"
    return "large_win"


def _current_scores(row: dict[str, Any], player_id: int) -> tuple[int, int]:
    snapshot = row.get("state_snapshot")
    if isinstance(snapshot, dict):
        players = snapshot.get("players")
        if isinstance(players, list) and len(players) == 2:
            return (
                int(players[player_id].get("score", 0)),
                int(players[1 - player_id].get("score", 0)),
            )
    return 0, 0


def _actor_label(row: dict[str, Any]) -> str:
    player_id = int(row.get("player_id", 0))
    metadata = row.get("game_bot_metadata")
    seats = row.get("game_bot_seats")
    if isinstance(metadata, list) and player_id < len(metadata) and isinstance(metadata[player_id], dict):
        item = metadata[player_id]
        checkpoint_path = item.get("checkpoint_path")
        if checkpoint_path:
            return Path(str(checkpoint_path)).parent.name
        return str(item.get("bot_class", "unknown"))
    if isinstance(seats, list) and player_id < len(seats):
        return str(seats[player_id])
    return "unknown"


if __name__ == "__main__":
    main()
