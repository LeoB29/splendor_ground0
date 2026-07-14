"""Build group-relative advantages from repeated stochastic rollout seeds."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from splendor_ai.training.run_trajectory_advantage_corpus import (
    TrajectoryAdvantageCorpusConfig,
    _skip_reason,
    _trajectory_advantage,
)


@dataclass(frozen=True, slots=True)
class GroupRelativeTrajectoryCorpusConfig:
    replay_paths: tuple[Path, ...]
    output_path: Path
    summary_path: Path
    outcome_weight: float = 0.75
    score_margin_scale: float = 5.0
    training_weight: float = 1.0
    min_group_trajectories: int = 3
    min_abs_advantage: float = 1e-6
    sampling_bot_class: str = "LegalActionAttentionSamplingBot"
    baseline_mode: str = "leave_one_out"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Center stochastic trajectory returns within repeated seed/role groups."
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--outcome-weight", type=float, default=0.75)
    parser.add_argument("--score-margin-scale", type=float, default=5.0)
    parser.add_argument("--training-weight", type=float, default=1.0)
    parser.add_argument("--min-group-trajectories", type=int, default=3)
    parser.add_argument("--min-abs-advantage", type=float, default=1e-6)
    parser.add_argument(
        "--sampling-bot-class",
        default="LegalActionAttentionSamplingBot",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("leave_one_out", "group_mean"),
        default="leave_one_out",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_group_relative_trajectory_corpus(
        GroupRelativeTrajectoryCorpusConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            summary_path=Path(args.summary_path),
            outcome_weight=args.outcome_weight,
            score_margin_scale=args.score_margin_scale,
            training_weight=args.training_weight,
            min_group_trajectories=args.min_group_trajectories,
            min_abs_advantage=args.min_abs_advantage,
            sampling_bot_class=args.sampling_bot_class,
            baseline_mode=args.baseline_mode,
        )
    )
    print(
        "Group-relative trajectory corpus:"
        f" groups={summary['groups_total']}"
        f" informative_groups={summary['groups_informative']}"
        f" rows={summary['rows_written']}"
    )


def build_group_relative_trajectory_corpus(
    config: GroupRelativeTrajectoryCorpusConfig,
) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.min_group_trajectories < 2:
        raise ValueError("min_group_trajectories must be at least 2.")
    if config.min_abs_advantage < 0.0:
        raise ValueError("min_abs_advantage must be non-negative.")
    if config.baseline_mode not in {"leave_one_out", "group_mean"}:
        raise ValueError("baseline_mode must be 'leave_one_out' or 'group_mean'.")

    base_config = TrajectoryAdvantageCorpusConfig(
        replay_paths=config.replay_paths,
        output_path=config.output_path,
        summary_path=config.summary_path,
        outcome_weight=config.outcome_weight,
        score_margin_scale=config.score_margin_scale,
        training_weight=config.training_weight,
        sampling_bot_class=config.sampling_bot_class,
        log_every=0,
    )
    trajectory_rows: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    trajectory_returns: dict[tuple[str, int, int], float] = {}
    group_members: dict[tuple[Any, ...], list[tuple[str, int, int]]] = defaultdict(list)
    counts: Counter[str] = Counter()

    for replay_path in config.replay_paths:
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counts["rows_read"] += 1
                payload = json.loads(line)
                reason = _skip_reason(payload, base_config)
                if reason is not None:
                    counts[f"skipped_{reason}"] += 1
                    continue
                seed = int(payload.get("game_seed", payload.get("seed", -1)))
                player_id = int(payload["player_id"])
                trajectory_key = (str(replay_path), seed, player_id)
                trajectory_rows[trajectory_key].append(payload)
                trajectory_returns[trajectory_key] = _trajectory_advantage(
                    payload,
                    base_config,
                )

    for trajectory_key, rows in trajectory_rows.items():
        sample = rows[0]
        group_key = _group_key(sample)
        group_members[group_key].append(trajectory_key)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    group_sizes: Counter[str] = Counter()
    relative_advantages: list[float] = []
    informative_groups = 0
    with config.output_path.open("w", encoding="utf-8") as output:
        for group_key, member_keys in group_members.items():
            group_sizes[str(len(member_keys))] += 1
            if len(member_keys) < config.min_group_trajectories:
                counts["trajectories_skipped_small_group"] += len(member_keys)
                continue
            group_return_sum = sum(trajectory_returns[key] for key in member_keys)
            group_mean = group_return_sum / len(member_keys)
            if config.baseline_mode == "leave_one_out":
                member_baselines = {
                    key: (group_return_sum - trajectory_returns[key]) / (len(member_keys) - 1)
                    for key in member_keys
                }
            else:
                member_baselines = {key: group_mean for key in member_keys}
            member_advantages = {
                key: trajectory_returns[key] - member_baselines[key]
                for key in member_keys
            }
            informative = [
                value
                for value in member_advantages.values()
                if abs(value) >= config.min_abs_advantage
            ]
            if not informative:
                counts["groups_without_return_variation"] += 1
                continue
            informative_groups += 1
            for trajectory_key, relative_advantage in member_advantages.items():
                if abs(relative_advantage) < config.min_abs_advantage:
                    counts["trajectories_skipped_zero_advantage"] += 1
                    continue
                relative_advantages.append(relative_advantage)
                counts["trajectories_written"] += 1
                for payload in trajectory_rows[trajectory_key]:
                    row = dict(payload)
                    row.update(
                        {
                            "trajectory_advantage": relative_advantage,
                            "training_weight": config.training_weight,
                            "target_source": "group_relative_trajectory_policy_gradient",
                            "trajectory_metadata": {
                                "raw_return": trajectory_returns[trajectory_key],
                                "group_baseline": member_baselines[trajectory_key],
                                "group_mean_return": group_mean,
                                "baseline_mode": config.baseline_mode,
                                "group_size": len(member_keys),
                                "group_key": list(group_key),
                                "source_replay_path": trajectory_key[0],
                            },
                        }
                    )
                    output.write(json.dumps(row) + "\n")
                    counts["rows_written"] += 1
                    counts[
                        "positive_rows"
                        if relative_advantage > 0.0
                        else "negative_rows"
                    ] += 1

    summary = {
        **dict(counts),
        "groups_total": len(group_members),
        "groups_informative": informative_groups,
        "group_size_histogram": dict(group_sizes),
        "relative_advantage_mean": (
            sum(relative_advantages) / len(relative_advantages)
            if relative_advantages
            else None
        ),
        "relative_advantage_abs_mean": (
            sum(abs(value) for value in relative_advantages)
            / len(relative_advantages)
            if relative_advantages
            else None
        ),
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "summary_path": str(config.summary_path),
        },
    }
    config.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _group_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    player_id = int(payload["player_id"])
    metadata = payload["game_bot_metadata"]
    actor = metadata[player_id]
    opponent = metadata[1 - player_id]
    return (
        int(payload.get("game_seed", payload.get("seed", -1))),
        player_id,
        str(actor.get("bot_class")),
        str(opponent.get("bot_class")),
        str(opponent.get("checkpoint_path", "")),
    )


if __name__ == "__main__":
    main()
