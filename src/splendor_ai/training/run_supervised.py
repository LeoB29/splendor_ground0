"""CLI entry point for supervised warm-start training."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from splendor_ai.bots import CheckpointPolicyBot, GreedyHeuristicBot, RandomLegalBot
from splendor_ai.eval import MatchConfig, play_game
from splendor_ai.eval.run_benchmark import summarize_games

from .dataset import SupervisedReplayDataset, summarize_replay_dataset_sources
from .model import PolicyValueMLP, PolicyValueModelConfig
from .supervised import (
    BatchProgress,
    SupervisedTrainConfig,
    create_replay_dataloader,
    fit_supervised_dataloaders_with_artifacts,
    split_replay_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the baseline Splendor policy/value model from replay JSONL data.")
    parser.add_argument(
        "--replay-path",
        action="append",
        required=True,
        help="Path to replay JSONL exported by the replay collector. Repeat to mix multiple corpora.",
    )
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
        "--log-every-batches",
        type=int,
        default=100,
        help="Print training and validation progress every N batches. Set to 0 to only log first/last batch.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of replay rows reserved for held-out validation. Set to 0 to disable.",
    )
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=0,
        help="Random seed for the train/validation replay split.",
    )
    parser.add_argument(
        "--benchmark-games",
        type=int,
        default=0,
        help="Number of post-train games to run against each selected baseline bot. Set to 0 to skip.",
    )
    parser.add_argument(
        "--benchmark-opponents",
        nargs="+",
        choices=("random", "greedy"),
        default=("random", "greedy"),
        help="Baseline opponents to evaluate against after training.",
    )
    parser.add_argument(
        "--benchmark-seed-start",
        type=int,
        default=0,
        help="First game seed used for post-train benchmark matches.",
    )
    parser.add_argument(
        "--benchmark-max-turns",
        type=int,
        default=400,
        help="Maximum turns per post-train benchmark game.",
    )
    parser.add_argument(
        "--benchmark-repetition-limit",
        type=int,
        default=4,
        help="Repeated-state cutoff used during post-train benchmarks. Set to 0 to disable.",
    )
    parser.add_argument(
        "--benchmark-no-progress-limit",
        type=int,
        default=60,
        help="No-progress cutoff used during post-train benchmarks. Set to 0 to disable.",
    )
    parser.add_argument(
        "--benchmark-log-every",
        type=int,
        default=5,
        help="Print benchmark progress every N completed games. Set to 0 to only log first/last game.",
    )
    parser.add_argument(
        "--benchmark-device",
        default=None,
        help="Device for the checkpoint policy bot during benchmarks. Defaults to the training device.",
    )
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


def _serialize_train_metrics(metric: object) -> dict[str, object]:
    return asdict(metric)


def _print_batch_progress(progress: BatchProgress) -> None:
    rate = (progress.samples / progress.elapsed_seconds) if progress.elapsed_seconds > 0 else 0.0
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


def _save_checkpoint(path: Path, model: PolicyValueMLP, train_config_payload: dict[str, object]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
            "train_config": train_config_payload,
        },
        path,
    )


def _run_benchmarks(
    checkpoint_path: Path,
    checkpoint_label: str,
    benchmark_games: int,
    opponents: tuple[str, ...],
    benchmark_device: str,
    benchmark_seed_start: int,
    benchmark_max_turns: int,
    benchmark_repetition_limit: int,
    benchmark_no_progress_limit: int,
    benchmark_log_every: int,
) -> list[dict[str, object]]:
    if benchmark_games <= 0 or not opponents:
        return []

    opponent_factories = {
        "random": lambda: RandomLegalBot(seed=benchmark_seed_start),
        "greedy": lambda: GreedyHeuristicBot(seed=benchmark_seed_start),
    }
    benchmark_payload: list[dict[str, object]] = []

    for opponent_name in opponents:
        print(
            f"[benchmark] checkpoint={checkpoint_label} opponent={opponent_name} games={benchmark_games}"
            f" device={benchmark_device} max_turns={benchmark_max_turns}"
        )
        games = []
        model_wins = 0
        opponent_wins = 0
        draws = 0
        timed_out_games = 0
        start_time = time.perf_counter()
        cfg = MatchConfig(
            games=benchmark_games,
            max_turns_per_game=benchmark_max_turns,
            repetition_limit=benchmark_repetition_limit,
            no_progress_limit=benchmark_no_progress_limit,
        )
        model_bot = CheckpointPolicyBot(checkpoint_path, device=benchmark_device)

        for game_index in range(cfg.games):
            swap = cfg.swap_seats and (game_index % 2 == 1)
            opponent_bot = opponent_factories[opponent_name]()
            seat0 = opponent_bot if swap else model_bot
            seat1 = model_bot if swap else opponent_bot
            game = play_game(
                bot_seat_0=seat0,
                bot_seat_1=seat1,
                seed=benchmark_seed_start + game_index,
                max_turns=cfg.max_turns_per_game,
                repetition_limit=cfg.repetition_limit,
                no_progress_limit=cfg.no_progress_limit,
            )
            games.append(game)
            if game.winner is None:
                draws += 1
            else:
                winner_name = game.bot_seats[game.winner]
                if winner_name == "CheckpointPolicyBot":
                    model_wins += 1
                else:
                    opponent_wins += 1
            if game.timed_out:
                timed_out_games += 1

            done = game_index + 1
            if done == cfg.games or done == 1 or (benchmark_log_every > 0 and done % benchmark_log_every == 0):
                elapsed = time.perf_counter() - start_time
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = ((cfg.games - done) / rate) if rate > 0 else math.inf
                eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
                print(
                    f"[benchmark:{checkpoint_label}:{opponent_name}] {done}/{cfg.games} games"
                    f" elapsed={elapsed:.1f}s"
                    f" rate={rate:.2f} games/s"
                    f" eta={eta_text}"
                    f" model_wins={model_wins}"
                    f" opponent_wins={opponent_wins}"
                    f" draws={draws}"
                    f" timed_out={timed_out_games}"
                )

        opponent_summary = summarize_games(
            tuple(games),
            opponent_name=opponent_name,
            seed_start=benchmark_seed_start,
            max_turns=benchmark_max_turns,
            device=benchmark_device,
            repetition_limit=benchmark_repetition_limit,
            no_progress_limit=benchmark_no_progress_limit,
        )
        opponent_summary.update(
            {
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": checkpoint_path.name,
                "max_turns_per_game": benchmark_max_turns,
            }
        )
        benchmark_payload.append(opponent_summary)
    return benchmark_payload


def _summarize_candidate_score(benchmarks: list[dict[str, object]]) -> int:
    return sum(int(benchmark["model_wins"]) - int(benchmark["opponent_wins"]) for benchmark in benchmarks)


def _select_benchmark_champion(
    candidate_benchmarks: list[dict[str, object]],
) -> dict[str, object] | None:
    if not candidate_benchmarks:
        return None
    return max(
        candidate_benchmarks,
        key=lambda payload: (
            _summarize_candidate_score(payload["benchmarks"]),
            sum(int(benchmark["model_wins"]) for benchmark in payload["benchmarks"]),
        ),
    )


def _build_train_config_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "replay_paths": [str(Path(path)) for path in args.replay_path],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "value_loss_weight": args.value_loss_weight,
        "epochs": args.epochs,
        "device": args.device,
        "include_stalled_games": not args.exclude_stalled_games,
        "include_timed_out_games": not args.exclude_timeout_games,
        "validation_fraction": args.validation_fraction,
        "validation_seed": args.validation_seed,
        "log_every_batches": args.log_every_batches,
    }


def _merge_source_counts(
    total_counts: tuple[object, ...],
    train_counts: tuple[object, ...],
    validation_counts: tuple[object, ...],
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for counts, field_name in (
        (total_counts, "total_samples"),
        (train_counts, "train_samples"),
        (validation_counts, "validation_samples"),
    ):
        for summary in counts:
            entry = merged.setdefault(
                summary.path,
                {
                    "path": summary.path,
                    "total_samples": 0,
                    "train_samples": 0,
                    "validation_samples": 0,
                },
            )
            entry[field_name] = summary.samples
    return [merged[path] for path in sorted(merged)]


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = tuple(Path(path) for path in args.replay_path)

    print(
        f"[train] replay_paths={', '.join(str(path) for path in replay_paths)}"
        f" output_dir={output_dir}"
        f" device={args.device}"
        f" epochs={args.epochs}"
        f" batch_size={args.batch_size}"
    )

    model = PolicyValueMLP(
        PolicyValueModelConfig(
            trunk_hidden_size=args.trunk_hidden_size,
            trunk_depth=args.trunk_depth,
            dropout=args.dropout,
        )
    )
    train_config = SupervisedTrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        epochs=args.epochs,
        device=args.device,
        include_stalled_games=not args.exclude_stalled_games,
        include_timed_out_games=not args.exclude_timeout_games,
        validation_fraction=args.validation_fraction,
        validation_seed=args.validation_seed,
    )
    print("[train] loading replay dataset...")
    dataset = SupervisedReplayDataset(
        replay_paths,
        include_stalled_games=train_config.include_stalled_games,
        include_timed_out_games=train_config.include_timed_out_games,
    )
    train_dataset, validation_dataset = split_replay_dataset(
        dataset,
        validation_fraction=train_config.validation_fraction,
        seed=train_config.validation_seed,
    )
    print(
        f"[train] dataset_samples={len(dataset)}"
        f" train_samples={len(train_dataset)}"
        f" validation_samples={len(validation_dataset) if validation_dataset is not None else 0}"
    )
    dataset_source_counts = summarize_replay_dataset_sources(dataset)
    train_source_counts = summarize_replay_dataset_sources(train_dataset)
    validation_source_counts = (
        summarize_replay_dataset_sources(validation_dataset)
        if validation_dataset is not None
        else tuple()
    )
    if len(dataset_source_counts) > 1:
        print(
            "[train] replay_mix="
            + ", ".join(f"{summary.path}:{summary.samples}" for summary in dataset_source_counts)
        )
    train_dataloader = create_replay_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        action_space_size=model.config.action_space_size,
        shuffle=train_config.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_replay_dataloader(
            validation_dataset,
            batch_size=train_config.batch_size,
            action_space_size=model.config.action_space_size,
            shuffle=False,
        )
    print(
        f"[train] train_batches={len(train_dataloader)}"
        f" validation_batches={len(validation_dataloader) if validation_dataloader is not None else 0}"
    )
    print("[train] starting supervised optimization...")
    fit_start = time.perf_counter()
    fit_artifacts = fit_supervised_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=train_config,
        model=model,
        log_every_batches=args.log_every_batches,
        batch_progress_callback=_print_batch_progress,
    )
    fit_elapsed = time.perf_counter() - fit_start
    model = fit_artifacts.model
    metrics = fit_artifacts.metrics

    checkpoint_path = output_dir / "supervised_policy_value.pt"
    best_checkpoint_path = output_dir / "supervised_policy_value_best.pt"
    metrics_path = output_dir / "supervised_metrics.json"
    train_config_payload = _build_train_config_payload(args)

    _save_checkpoint(
        checkpoint_path,
        model=model,
        train_config_payload=train_config_payload,
    )
    print(f"[train] wrote final checkpoint: {checkpoint_path}")
    best_epoch_index = fit_artifacts.best_epoch_index
    best_checkpoint_written = False
    if fit_artifacts.best_model_state_dict is not None and best_epoch_index is not None:
        best_model = PolicyValueMLP(model.config)
        best_model.load_state_dict(fit_artifacts.best_model_state_dict)
        _save_checkpoint(
            best_checkpoint_path,
            model=best_model,
            train_config_payload=train_config_payload,
        )
        best_checkpoint_written = True
        print(f"[train] wrote best validation checkpoint: {best_checkpoint_path} (epoch {best_epoch_index})")

    benchmark_device = args.benchmark_device or args.device
    benchmark_candidates = [
        {
            "checkpoint_label": "final",
            "checkpoint_path": checkpoint_path,
        }
    ]
    if best_checkpoint_written:
        benchmark_candidates.append(
            {
                "checkpoint_label": "best_validation",
                "checkpoint_path": best_checkpoint_path,
            }
        )
    candidate_benchmark_payload: list[dict[str, object]] = []
    if args.benchmark_games > 0:
        print(
            "[benchmark] candidate_checkpoints="
            + ", ".join(
                f"{candidate['checkpoint_label']}:{candidate['checkpoint_path'].name}"
                for candidate in benchmark_candidates
            )
        )
    for candidate in benchmark_candidates if args.benchmark_games > 0 else []:
        candidate_benchmark_payload.append(
            {
                "checkpoint_label": candidate["checkpoint_label"],
                "checkpoint_path": candidate["checkpoint_path"].name,
                "benchmarks": _run_benchmarks(
                    checkpoint_path=candidate["checkpoint_path"],
                    checkpoint_label=candidate["checkpoint_label"],
                    benchmark_games=args.benchmark_games,
                    opponents=tuple(args.benchmark_opponents),
                    benchmark_device=benchmark_device,
                    benchmark_seed_start=args.benchmark_seed_start,
                    benchmark_max_turns=args.benchmark_max_turns,
                    benchmark_repetition_limit=args.benchmark_repetition_limit,
                    benchmark_no_progress_limit=args.benchmark_no_progress_limit,
                    benchmark_log_every=args.benchmark_log_every,
                ),
            }
        )
    benchmark_champion = _select_benchmark_champion(candidate_benchmark_payload)
    merged_source_counts = _merge_source_counts(
        dataset_source_counts,
        train_source_counts,
        validation_source_counts,
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
            "sources": merged_source_counts,
        },
        "checkpoints": {
            "final_checkpoint_path": checkpoint_path.name,
            "best_validation_checkpoint_path": best_checkpoint_path.name if best_checkpoint_written else None,
            "best_validation_epoch": best_epoch_index,
            "benchmark_champion_checkpoint_path": (
                benchmark_champion["checkpoint_path"] if benchmark_champion is not None else None
            ),
            "benchmark_champion_label": (
                benchmark_champion["checkpoint_label"] if benchmark_champion is not None else None
            ),
        },
        "epochs": [
            {
                "train": _serialize_train_metrics(metric.train),
                "validation": _serialize_train_metrics(metric.validation) if metric.validation is not None else None,
                "is_best_validation": epoch_index == best_epoch_index,
            }
            for epoch_index, metric in enumerate(metrics, start=1)
        ],
        "benchmarks": {
            "candidates": candidate_benchmark_payload,
        },
        "timing": {
            "training_seconds": fit_elapsed,
        },
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[train] wrote metrics: {metrics_path}")

    final_epoch = metrics[-1]
    print(
        "Training complete:"
        f" train_loss={final_epoch.train.total_loss:.4f}"
        f" train_acc={final_epoch.train.policy_accuracy:.4f}"
        f" train_samples={final_epoch.train.samples}"
        f" training_seconds={fit_elapsed:.1f}"
    )
    if final_epoch.validation is not None:
        print(
            "Validation:"
            f" loss={final_epoch.validation.total_loss:.4f}"
            f" acc={final_epoch.validation.policy_accuracy:.4f}"
            f" samples={final_epoch.validation.samples}"
            f" best_epoch={best_epoch_index}"
        )
    if candidate_benchmark_payload:
        if benchmark_champion is not None:
            print(
                "Benchmark Champion:"
                f" checkpoint={benchmark_champion['checkpoint_label']}"
                f" path={benchmark_champion['checkpoint_path']}"
            )
        for candidate in candidate_benchmark_payload:
            for benchmark in candidate["benchmarks"]:
                print(
                    "Benchmark:"
                    f" checkpoint={benchmark['checkpoint_label']}"
                    f" opponent={benchmark['opponent']}"
                    f" model_wins={benchmark['model_wins']}"
                    f" opponent_wins={benchmark['opponent_wins']}"
                    f" draws={benchmark['draws']}"
                    f" timed_out={benchmark['timed_out_games']}"
                    f" termination_reasons={benchmark['termination_reasons']}"
                )


if __name__ == "__main__":
    main()
