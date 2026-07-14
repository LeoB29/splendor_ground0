"""Resumable, parallel paired-seed checkpoint evaluation.

This is intentionally separate from :mod:`run_benchmark`, whose simple
sequential implementation is useful for small interactive checks.  Long
candidate gates and ES fitness evaluations need a durable unit of work: one
environment seed played once from each seat.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.eval.match import GameResult, play_game


_WORKER_BOTS: tuple[LegalActionAttentionScorerBot, LegalActionAttentionScorerBot] | None = None
_WORKER_LIMITS: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class PairedCheckpointConfig:
    candidate_checkpoint: Path
    baseline_checkpoint: Path
    output_path: Path
    checkpoint_path: Path
    pairs: int
    seed_start: int = 0
    workers: int = 4
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60
    bootstrap_samples: int = 20_000
    outcome_weight: float = 0.75
    score_margin_weight: float = 0.25
    score_margin_scale: float = 5.0
    device: str = "cpu"


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a resumable parallel paired-seed legal checkpoint benchmark."
    )
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Progress JSON path; defaults to <output-path>.progress.json.",
    )
    parser.add_argument("--pairs", type=int, required=True, help="Number of seed/seat pairs.")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-weight", type=float, default=0.25)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    return parser


def config_from_args(args: Any) -> PairedCheckpointConfig:
    output_path = Path(args.output_path)
    return PairedCheckpointConfig(
        candidate_checkpoint=Path(args.candidate_checkpoint),
        baseline_checkpoint=Path(args.baseline_checkpoint),
        output_path=output_path,
        checkpoint_path=(
            Path(args.checkpoint_path)
            if args.checkpoint_path is not None
            else output_path.with_suffix(output_path.suffix + ".progress.json")
        ),
        pairs=args.pairs,
        seed_start=args.seed_start,
        workers=args.workers,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
        bootstrap_samples=args.bootstrap_samples,
        outcome_weight=args.outcome_weight,
        score_margin_weight=args.score_margin_weight,
        score_margin_scale=args.score_margin_scale,
        device=args.device,
    )


def run_paired_checkpoint_benchmark(config: PairedCheckpointConfig) -> dict[str, Any]:
    _validate_config(config)
    existing = _load_progress(config)
    pair_rows: dict[int, dict[str, Any]] = {
        int(row["seed"]): row for row in existing.get("completed_pairs", [])
    }
    target_seeds = list(range(config.seed_start, config.seed_start + config.pairs))
    pending = [seed for seed in target_seeds if seed not in pair_rows]
    started = time.perf_counter()
    print(
        "[paired]"
        f" pairs={config.pairs} complete={len(pair_rows)} pending={len(pending)}"
        f" workers={config.workers} device={config.device}"
    )

    if pending:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_worker,
            initargs=(
                str(config.candidate_checkpoint),
                str(config.baseline_checkpoint),
                config.device,
                config.max_turns,
                config.repetition_limit,
                config.no_progress_limit,
            ),
        ) as executor:
            futures = {executor.submit(_play_pair_worker, seed): seed for seed in pending}
            for completed, future in enumerate(as_completed(futures), start=1):
                seed = futures[future]
                pair_rows[seed] = future.result()
                _write_progress(config, pair_rows)
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed > 0.0 else 0.0
                remaining = len(pending) - completed
                eta = remaining / rate if rate > 0.0 else math.inf
                eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
                print(
                    f"[paired] completed={len(pair_rows)}/{config.pairs}"
                    f" latest_seed={seed} rate={rate:.3f} pairs/s eta={eta_text}"
                )

    ordered_rows = [pair_rows[seed] for seed in target_seeds]
    payload = _summarize(config, ordered_rows)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_progress(config, pair_rows, completed=True)
    print(f"[paired] wrote results: {config.output_path}")
    return payload


def _initialize_worker(
    candidate_checkpoint: str,
    baseline_checkpoint: str,
    device: str,
    max_turns: int,
    repetition_limit: int,
    no_progress_limit: int,
) -> None:
    global _WORKER_BOTS, _WORKER_LIMITS
    # Each process owns its models.  This keeps torch/model state isolated and
    # avoids sharing a mutable bot across concurrent games.
    _WORKER_BOTS = (
        LegalActionAttentionScorerBot(candidate_checkpoint, device=device),
        LegalActionAttentionScorerBot(baseline_checkpoint, device=device),
    )
    _WORKER_LIMITS = (max_turns, repetition_limit, no_progress_limit)


def _play_pair_worker(seed: int) -> dict[str, Any]:
    if _WORKER_BOTS is None or _WORKER_LIMITS is None:
        raise RuntimeError("Paired worker was not initialized.")
    candidate, baseline = _WORKER_BOTS
    max_turns, repetition_limit, no_progress_limit = _WORKER_LIMITS
    first = play_game(
        candidate,
        baseline,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    second = play_game(
        baseline,
        candidate,
        seed=seed,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [_serialize_game(first), _serialize_game(second)],
    }


def _serialize_game(game: GameResult) -> dict[str, Any]:
    return asdict(game)


def _candidate_outcome(game: dict[str, Any], candidate_seat: int) -> float:
    winner = game["winner"]
    if winner is None:
        return 0.0
    return 1.0 if int(winner) == candidate_seat else -1.0


def _candidate_margin(game: dict[str, Any], candidate_seat: int) -> float:
    scores = game["final_scores"]
    return float(scores[candidate_seat]) - float(scores[1 - candidate_seat])


def _pair_metrics(row: dict[str, Any], config: PairedCheckpointConfig) -> dict[str, float]:
    outcomes = [
        _candidate_outcome(game, seat)
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True)
    ]
    margins = [
        _candidate_margin(game, seat)
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True)
    ]
    score_share = statistics.fmean((outcome + 1.0) / 2.0 for outcome in outcomes)
    mean_outcome = statistics.fmean(outcomes)
    mean_margin = statistics.fmean(margins)
    utility = (
        config.outcome_weight * mean_outcome
        + config.score_margin_weight * math.tanh(mean_margin / config.score_margin_scale)
    )
    return {
        "score_share": score_share,
        "outcome": mean_outcome,
        "final_score_margin": mean_margin,
        "bounded_utility": utility,
    }


def _summarize(config: PairedCheckpointConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["metrics"] = _pair_metrics(row, config)
        enriched_rows.append(enriched)
    metrics = {
        key: [float(row["metrics"][key]) for row in enriched_rows]
        for key in ("score_share", "outcome", "final_score_margin", "bounded_utility")
    }
    all_games = [game for row in enriched_rows for game in row["games"]]
    return {
        "format": "parallel_paired_checkpoint_benchmark_v1",
        "config": _config_payload(config),
        "completed_pairs": len(enriched_rows),
        "completed_games": len(all_games),
        "timed_out_games": sum(bool(game["timed_out"]) for game in all_games),
        "stalled_games": sum(bool(game["stalled"]) for game in all_games),
        "termination_reasons": _termination_counts(all_games),
        "paired_statistics": {
            key: _mean_ci(values, config.bootstrap_samples, config.seed_start + index)
            for index, (key, values) in enumerate(metrics.items())
        },
        "pairs": enriched_rows,
    }


def _mean_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = statistics.fmean(values)
    if samples <= 0 or len(values) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    bootstrap = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": bootstrap[int(0.025 * (samples - 1))],
        "ci95_high": bootstrap[int(0.975 * (samples - 1))],
    }


def _termination_counts(games: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game in games:
        reason = str(game["termination_reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _config_payload(config: PairedCheckpointConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("candidate_checkpoint", "baseline_checkpoint", "output_path", "checkpoint_path"):
        payload[key] = str(payload[key])
    return payload


def _progress_identity(config: PairedCheckpointConfig) -> dict[str, Any]:
    payload = _config_payload(config)
    payload.pop("output_path")
    payload.pop("checkpoint_path")
    return payload


def _load_progress(config: PairedCheckpointConfig) -> dict[str, Any]:
    if not config.checkpoint_path.exists():
        return {"config": _progress_identity(config), "completed_pairs": []}
    payload = json.loads(config.checkpoint_path.read_text(encoding="utf-8"))
    if payload.get("config") != _progress_identity(config):
        raise ValueError(
            "Progress checkpoint configuration does not match this paired benchmark. "
            "Use a new --checkpoint-path for a different experiment."
        )
    return payload


def _write_progress(
    config: PairedCheckpointConfig,
    rows: dict[int, dict[str, Any]],
    completed: bool = False,
) -> None:
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "parallel_paired_checkpoint_progress_v1",
        "config": _progress_identity(config),
        "completed": completed,
        "completed_pairs": [rows[seed] for seed in sorted(rows)],
    }
    temp_path = config.checkpoint_path.with_suffix(config.checkpoint_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Windows can briefly hold a just-read JSON file open (for example through
    # indexing or antivirus).  Retrying preserves atomic replacement without
    # turning an otherwise resumable long gate into a failed experiment.
    for attempt in range(10):
        try:
            os.replace(temp_path, config.checkpoint_path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _validate_config(config: PairedCheckpointConfig) -> None:
    if config.pairs <= 0 or config.workers <= 0:
        raise ValueError("pairs and workers must be positive.")
    if config.max_turns <= 0 or config.bootstrap_samples < 0:
        raise ValueError("max_turns must be positive and bootstrap_samples non-negative.")
    if config.score_margin_scale <= 0.0:
        raise ValueError("score_margin_scale must be positive.")
    if config.outcome_weight < 0.0 or config.score_margin_weight < 0.0:
        raise ValueError("utility weights must be non-negative.")
    if not math.isclose(config.outcome_weight + config.score_margin_weight, 1.0):
        raise ValueError("outcome_weight and score_margin_weight must sum to 1.0.")
    if not config.candidate_checkpoint.exists() or not config.baseline_checkpoint.exists():
        raise FileNotFoundError("Candidate and baseline checkpoints must exist.")


def main() -> None:
    args = build_parser().parse_args()
    run_paired_checkpoint_benchmark(config_from_args(args))


if __name__ == "__main__":
    main()
