"""Convert stochastic checkpoint replay into signed trajectory-advantage rows."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrajectoryAdvantageCorpusConfig:
    replay_paths: tuple[Path, ...]
    output_path: Path
    summary_path: Path
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    training_weight: float = 1.0
    include_timed_out_games: bool = False
    include_stalled_games: bool = False
    sampling_bot_class: str = "LegalActionAttentionSamplingBot"
    log_every: int = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build signed trajectory-policy rows from stochastic replay."
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--training-weight", type=float, default=1.0)
    parser.add_argument("--include-timed-out-games", action="store_true")
    parser.add_argument("--include-stalled-games", action="store_true")
    parser.add_argument(
        "--sampling-bot-class",
        default="LegalActionAttentionSamplingBot",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_trajectory_advantage_corpus(
        TrajectoryAdvantageCorpusConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            summary_path=Path(args.summary_path),
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            training_weight=args.training_weight,
            include_timed_out_games=args.include_timed_out_games,
            include_stalled_games=args.include_stalled_games,
            sampling_bot_class=args.sampling_bot_class,
            log_every=args.log_every,
        )
    )
    print(
        "Trajectory advantage corpus:"
        f" read={summary['rows_read']}"
        f" written={summary['rows_written']}"
        f" positive={summary['positive_rows']}"
        f" negative={summary['negative_rows']}"
    )


def build_trajectory_advantage_corpus(
    config: TrajectoryAdvantageCorpusConfig,
) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if not 0.0 <= config.outcome_weight <= 1.0:
        raise ValueError("outcome_weight must be in [0, 1].")
    if config.score_margin_scale <= 0.0:
        raise ValueError("score_margin_scale must be positive.")
    if config.training_weight <= 0.0:
        raise ValueError("training_weight must be positive.")

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    advantages: list[float] = []
    games: set[tuple[str, int]] = set()

    with config.output_path.open("w", encoding="utf-8") as output:
        for replay_path in config.replay_paths:
            source_counter = source_counts.setdefault(str(replay_path), Counter())
            with replay_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    counts["rows_read"] += 1
                    source_counter["rows_read"] += 1
                    payload = json.loads(line)
                    reason = _skip_reason(payload, config)
                    if reason is not None:
                        counts[f"skipped_{reason}"] += 1
                        source_counter[f"skipped_{reason}"] += 1
                        continue
                    advantage = _trajectory_advantage(payload, config)
                    row = dict(payload)
                    for key in (
                        "model_top_action_index",
                        "model_top_action_payload",
                        "model_top_score",
                        "model_top_search_value",
                    ):
                        row.pop(key, None)
                    row.update(
                        {
                            "trajectory_advantage": advantage,
                            "training_weight": config.training_weight,
                            "target_source": "trajectory_policy_gradient",
                            "trajectory_metadata": {
                                "outcome_weight": config.outcome_weight,
                                "score_margin_scale": config.score_margin_scale,
                                "source_replay_path": str(replay_path),
                            },
                        }
                    )
                    output.write(json.dumps(row) + "\n")
                    counts["rows_written"] += 1
                    source_counter["rows_written"] += 1
                    sign_key = (
                        "positive_rows"
                        if advantage > 0.0
                        else "negative_rows"
                        if advantage < 0.0
                        else "zero_rows"
                    )
                    counts[sign_key] += 1
                    source_counter[sign_key] += 1
                    advantages.append(advantage)
                    games.add(
                        (
                            str(replay_path),
                            int(payload.get("game_seed", payload.get("seed", -1))),
                        )
                    )
                    if (
                        config.log_every > 0
                        and counts["rows_read"] % config.log_every == 0
                    ):
                        print(
                            f"[trajectory-corpus] read={counts['rows_read']}"
                            f" written={counts['rows_written']}"
                        )

    summary = {
        **dict(counts),
        "games_written": len(games),
        "by_source": {
            source: dict(counter)
            for source, counter in source_counts.items()
        },
        "advantage_mean": (
            sum(advantages) / len(advantages) if advantages else None
        ),
        "advantage_abs_mean": (
            sum(abs(value) for value in advantages) / len(advantages)
            if advantages
            else None
        ),
        "advantage_min": min(advantages) if advantages else None,
        "advantage_max": max(advantages) if advantages else None,
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "summary_path": str(config.summary_path),
        },
    }
    config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _skip_reason(
    payload: dict[str, Any],
    config: TrajectoryAdvantageCorpusConfig,
) -> str | None:
    if not config.include_timed_out_games and payload.get("game_timed_out", False):
        return "timed_out_game"
    if not config.include_stalled_games and payload.get("game_stalled", False):
        return "stalled_game"
    if payload.get("behavior_log_probability") is None:
        return "missing_behavior_probability"
    player_id = int(payload.get("player_id", -1))
    bot_metadata = payload.get("game_bot_metadata")
    if (
        not isinstance(bot_metadata, list)
        or player_id < 0
        or player_id >= len(bot_metadata)
        or not isinstance(bot_metadata[player_id], dict)
        or bot_metadata[player_id].get("bot_class") != config.sampling_bot_class
    ):
        return "non_sampling_turn"
    scores = payload.get("game_final_scores")
    if not isinstance(scores, list | tuple) or len(scores) != 2:
        return "missing_final_scores"
    return None


def _trajectory_advantage(
    payload: dict[str, Any],
    config: TrajectoryAdvantageCorpusConfig,
) -> float:
    player_id = int(payload["player_id"])
    scores = payload["game_final_scores"]
    score_margin = float(scores[player_id]) - float(scores[1 - player_id])
    score_signal = math.tanh(score_margin / config.score_margin_scale)
    outcome_signal = float(payload.get("final_value", 0.0))
    return (
        config.outcome_weight * outcome_signal
        + (1.0 - config.outcome_weight) * score_signal
    )


if __name__ == "__main__":
    main()
