"""Compare plain policy-guided search with value-guided search on replay states."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from splendor_ai.bots import (
    CheckpointLegalActionPriorScorer,
    CheckpointValueEvaluator,
    ShallowSearchBot,
)
from splendor_ai.encoding import ActionCodec
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_action


@dataclass(frozen=True, slots=True)
class SearchValueDiffConfig:
    replay_paths: tuple[Path, ...]
    output_path: Path
    policy_prior_checkpoint_path: Path
    policy_prior_device: str = "cpu"
    policy_prior_weight: float = 3_000.0
    value_checkpoint_path: Path | None = None
    value_device: str = "cpu"
    value_weight: float = 0.05
    value_scale: float = 15_000.0
    value_prediction_mode: str = "value"
    value_outcome_blend_weight: float = 0.5
    value_min_turn_index: int | None = None
    value_max_turn_index: int | None = None
    search_depth: int = 2
    search_max_branching: int = 10
    search_buy_branching: int = 6
    search_reserve_branching: int = 3
    search_take_branching: int = 3
    seed: int = 0
    max_rows: int | None = None
    log_every: int = 500
    player_id_filter: tuple[int, ...] | None = None
    phase_filter: tuple[str, ...] | None = None
    min_turn_index: int | None = None
    examples_limit: int = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare plain PGS and value-guided PGS decisions on replay snapshots."
    )
    parser.add_argument("--replay-path", action="append", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--policy-prior-checkpoint", required=True)
    parser.add_argument("--policy-prior-device", default="cpu")
    parser.add_argument("--policy-prior-weight", type=float, default=3_000.0)
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--value-device", default="cpu")
    parser.add_argument("--value-weight", type=float, default=0.05)
    parser.add_argument("--value-scale", type=float, default=15_000.0)
    parser.add_argument(
        "--value-prediction-mode",
        choices=("value", "outcome", "blend"),
        default="value",
    )
    parser.add_argument("--value-outcome-blend-weight", type=float, default=0.5)
    parser.add_argument("--value-min-turn-index", type=int, default=None)
    parser.add_argument("--value-max-turn-index", type=int, default=None)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--min-turn-index", type=int, default=None)
    parser.add_argument(
        "--player-id-filter",
        nargs="*",
        type=int,
        choices=(0, 1),
        default=None,
    )
    parser.add_argument(
        "--phase-filter",
        nargs="*",
        choices=("early", "mid", "late"),
        default=None,
    )
    parser.add_argument("--examples-limit", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = compare_search_value_diff(
        SearchValueDiffConfig(
            replay_paths=tuple(Path(path) for path in args.replay_path),
            output_path=Path(args.output_path),
            policy_prior_checkpoint_path=Path(args.policy_prior_checkpoint),
            policy_prior_device=args.policy_prior_device,
            policy_prior_weight=args.policy_prior_weight,
            value_checkpoint_path=Path(args.value_checkpoint),
            value_device=args.value_device,
            value_weight=args.value_weight,
            value_scale=args.value_scale,
            value_prediction_mode=args.value_prediction_mode,
            value_outcome_blend_weight=args.value_outcome_blend_weight,
            value_min_turn_index=args.value_min_turn_index,
            value_max_turn_index=args.value_max_turn_index,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            seed=args.seed,
            max_rows=args.max_rows,
            log_every=args.log_every,
            player_id_filter=(
                tuple(args.player_id_filter)
                if args.player_id_filter is not None and len(args.player_id_filter) > 0
                else None
            ),
            phase_filter=(
                tuple(args.phase_filter)
                if args.phase_filter is not None and len(args.phase_filter) > 0
                else None
            ),
            min_turn_index=args.min_turn_index,
            examples_limit=args.examples_limit,
        )
    )
    print(
        "[search-value-diff]"
        f" analyzed={summary['rows_analyzed']}"
        f" changed={summary['overall']['changed']}"
        f" changed_rate={summary['overall']['changed_rate']:.4f}"
        f" baseline_regret_mean={summary['overall']['baseline_regret_mean']}"
        f" output={args.output_path}"
    )


def compare_search_value_diff(config: SearchValueDiffConfig) -> dict[str, Any]:
    if not config.replay_paths:
        raise ValueError("At least one replay path is required.")
    if config.value_checkpoint_path is None:
        raise ValueError("value_checkpoint_path is required.")
    unknown_phases = set(config.phase_filter or ()) - {"early", "mid", "late"}
    if unknown_phases:
        raise ValueError(f"Unknown phases: {sorted(unknown_phases)}")

    codec = ActionCodec()
    action_prior = CheckpointLegalActionPriorScorer(
        checkpoint_path=config.policy_prior_checkpoint_path,
        device=config.policy_prior_device,
    )
    value_evaluator = CheckpointValueEvaluator(
        checkpoint_path=config.value_checkpoint_path,
        device=config.value_device,
        value_scale=config.value_scale,
        prediction_mode=config.value_prediction_mode,
        outcome_blend_weight=config.value_outcome_blend_weight,
    )
    counters = _empty_counters()
    examples: list[dict[str, Any]] = []
    rows_seen = 0
    rows_skipped_by_player_id = 0
    rows_skipped_by_phase = 0
    rows_skipped_by_turn = 0
    rows_missing_snapshot = 0
    rows_without_actions = 0

    from splendor_ai.engine.env import SplendorEnv

    env = SplendorEnv()
    for replay_path in config.replay_paths:
        with replay_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if config.max_rows is not None and rows_seen >= config.max_rows:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                rows_seen += 1
                payload = json.loads(stripped)
                player_id = int(payload.get("player_id", -1))
                if config.player_id_filter is not None and player_id not in config.player_id_filter:
                    rows_skipped_by_player_id += 1
                    continue
                turn_index = int(payload.get("turn_index", 0))
                if config.min_turn_index is not None and turn_index < config.min_turn_index:
                    rows_skipped_by_turn += 1
                    continue
                phase = _phase_for_turn(turn_index)
                if config.phase_filter is not None and phase not in config.phase_filter:
                    rows_skipped_by_phase += 1
                    continue
                snapshot = payload.get("state_snapshot")
                if not isinstance(snapshot, dict):
                    rows_missing_snapshot += 1
                    continue
                state = deserialize_state_snapshot(snapshot)
                legal_actions = env.legal_actions(state)
                if not legal_actions:
                    rows_without_actions += 1
                    continue
                baseline_bot = _build_search_bot(
                    config=config,
                    row_offset=rows_seen,
                    action_prior=action_prior,
                )
                value_bot = _build_search_bot(
                    config=config,
                    row_offset=rows_seen,
                    action_prior=action_prior,
                    value_evaluator=value_evaluator,
                )
                baseline_scores = baseline_bot.rank_actions(env, state, legal_actions)
                value_scores = value_bot.rank_actions(env, state, legal_actions)
                if not baseline_scores or not value_scores:
                    rows_without_actions += 1
                    continue
                _record_row(
                    counters=counters,
                    examples=examples,
                    examples_limit=config.examples_limit,
                    codec=codec,
                    state=state,
                    payload=payload,
                    replay_path=replay_path,
                    baseline_scores=baseline_scores,
                    value_scores=value_scores,
                )
                if config.log_every > 0 and rows_seen % config.log_every == 0:
                    print(
                        f"[search-value-diff] rows_seen={rows_seen}"
                        f" analyzed={counters['overall']['count']}"
                        f" changed={counters['overall']['changed']}"
                    )
        if config.max_rows is not None and rows_seen >= config.max_rows:
            break

    summary = {
        "rows_seen": rows_seen,
        "rows_analyzed": counters["overall"]["count"],
        "rows_skipped_by_player_id": rows_skipped_by_player_id,
        "rows_skipped_by_phase": rows_skipped_by_phase,
        "rows_skipped_by_turn": rows_skipped_by_turn,
        "rows_missing_snapshot": rows_missing_snapshot,
        "rows_without_actions": rows_without_actions,
        "overall": _finalize_counter(counters["overall"]),
        "by_seat": _finalize_buckets(counters["by_seat"]),
        "by_phase": _finalize_buckets(counters["by_phase"]),
        "by_seat_phase": _finalize_buckets(counters["by_seat_phase"]),
        "by_score_margin": _finalize_buckets(counters["by_score_margin"]),
        "by_baseline_action_type": _finalize_buckets(counters["by_baseline_action_type"]),
        "by_value_action_type": _finalize_buckets(counters["by_value_action_type"]),
        "action_type_matrix": counters["action_type_matrix"],
        "examples": examples,
        "config": {
            **asdict(config),
            "replay_paths": [str(path) for path in config.replay_paths],
            "output_path": str(config.output_path),
            "policy_prior_checkpoint_path": str(config.policy_prior_checkpoint_path),
            "value_checkpoint_path": str(config.value_checkpoint_path),
        },
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_search_bot(
    config: SearchValueDiffConfig,
    row_offset: int,
    action_prior: CheckpointLegalActionPriorScorer,
    value_evaluator: CheckpointValueEvaluator | None = None,
) -> ShallowSearchBot:
    return ShallowSearchBot(
        depth=config.search_depth,
        max_branching=config.search_max_branching,
        max_buy_actions=config.search_buy_branching,
        max_reserve_actions=config.search_reserve_branching,
        max_take_actions=config.search_take_branching,
        seed=config.seed + row_offset,
        action_prior=action_prior,
        action_prior_weight=config.policy_prior_weight,
        value_evaluator=value_evaluator,
        value_weight=config.value_weight if value_evaluator is not None else 0.0,
        value_min_turn_index=config.value_min_turn_index,
        value_max_turn_index=config.value_max_turn_index,
    )


def _empty_counters() -> dict[str, Any]:
    return {
        "overall": _new_counter(),
        "by_seat": {},
        "by_phase": {},
        "by_seat_phase": {},
        "by_score_margin": {},
        "by_baseline_action_type": {},
        "by_value_action_type": {},
        "action_type_matrix": {},
    }


def _new_counter() -> dict[str, Any]:
    return {
        "count": 0,
        "changed": 0,
        "unchanged": 0,
        "baseline_regrets": [],
        "value_regrets": [],
    }


def _record_row(
    counters: dict[str, Any],
    examples: list[dict[str, Any]],
    examples_limit: int,
    codec: ActionCodec,
    state: Any,
    payload: dict[str, Any],
    replay_path: Path,
    baseline_scores: tuple[Any, ...],
    value_scores: tuple[Any, ...],
) -> None:
    baseline_top = baseline_scores[0]
    value_top = value_scores[0]
    baseline_top_index = codec.encode(state, baseline_top.action)
    value_top_index = codec.encode(state, value_top.action)
    changed = baseline_top_index != value_top_index
    baseline_values = {
        codec.encode(state, score.action): float(score.value)
        for score in baseline_scores
    }
    value_values = {
        codec.encode(state, score.action): float(score.value)
        for score in value_scores
    }
    baseline_regret = baseline_values.get(baseline_top_index, 0.0) - baseline_values.get(
        value_top_index,
        float("nan"),
    )
    value_regret = value_values.get(value_top_index, 0.0) - value_values.get(
        baseline_top_index,
        float("nan"),
    )
    turn_index = int(payload.get("turn_index", 0))
    player_id = int(payload.get("player_id", state.current_player))
    score_margin = state.players[player_id].score - state.players[1 - player_id].score
    baseline_type = baseline_top.action.action_type.name
    value_type = value_top.action.action_type.name
    phase = _phase_for_turn(turn_index)
    seat = f"seat{player_id}"

    for counter in (
        counters["overall"],
        _bucket(counters["by_seat"], seat),
        _bucket(counters["by_phase"], phase),
        _bucket(counters["by_seat_phase"], f"{seat}:{phase}"),
        _bucket(counters["by_score_margin"], _score_margin_bucket(score_margin)),
        _bucket(counters["by_baseline_action_type"], baseline_type),
        _bucket(counters["by_value_action_type"], value_type),
    ):
        _record_counter(counter, changed, baseline_regret, value_regret)
    _increment_matrix(counters["action_type_matrix"], baseline_type, value_type)
    if changed:
        _store_example(
            examples=examples,
            examples_limit=examples_limit,
            replay_path=replay_path,
            payload=payload,
            state=state,
            baseline_top=baseline_top,
            value_top=value_top,
            baseline_top_index=baseline_top_index,
            value_top_index=value_top_index,
            baseline_regret=baseline_regret,
            value_regret=value_regret,
        )


def _bucket(buckets: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    return buckets.setdefault(key, _new_counter())


def _record_counter(
    counter: dict[str, Any],
    changed: bool,
    baseline_regret: float,
    value_regret: float,
) -> None:
    counter["count"] += 1
    if changed:
        counter["changed"] += 1
        if math.isfinite(baseline_regret):
            counter["baseline_regrets"].append(float(baseline_regret))
        if math.isfinite(value_regret):
            counter["value_regrets"].append(float(value_regret))
    else:
        counter["unchanged"] += 1


def _increment_matrix(matrix: dict[str, dict[str, int]], row: str, column: str) -> None:
    matrix.setdefault(row, {})
    matrix[row][column] = matrix[row].get(column, 0) + 1


def _store_example(
    examples: list[dict[str, Any]],
    examples_limit: int,
    replay_path: Path,
    payload: dict[str, Any],
    state: Any,
    baseline_top: Any,
    value_top: Any,
    baseline_top_index: int,
    value_top_index: int,
    baseline_regret: float,
    value_regret: float,
) -> None:
    if examples_limit <= 0:
        return
    player_id = int(payload.get("player_id", state.current_player))
    examples.append(
        {
            "replay_path": str(replay_path),
            "game_seed": payload.get("game_seed"),
            "turn_index": payload.get("turn_index"),
            "phase": _phase_for_turn(int(payload.get("turn_index", 0))),
            "player_id": player_id,
            "scores": [state.players[0].score, state.players[1].score],
            "score_margin": state.players[player_id].score - state.players[1 - player_id].score,
            "baseline_action_index": baseline_top_index,
            "baseline_action_type": baseline_top.action.action_type.name,
            "baseline_value": float(baseline_top.value),
            "baseline_action_payload": serialize_action(baseline_top.action),
            "value_action_index": value_top_index,
            "value_action_type": value_top.action.action_type.name,
            "value_search_value": float(value_top.value),
            "value_action_payload": serialize_action(value_top.action),
            "baseline_regret_of_value_action": baseline_regret,
            "value_regret_of_baseline_action": value_regret,
        }
    )
    examples.sort(
        key=lambda item: (
            0.0
            if not math.isfinite(float(item["baseline_regret_of_value_action"]))
            else float(item["baseline_regret_of_value_action"])
        ),
        reverse=True,
    )
    del examples[examples_limit:]


def _finalize_buckets(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {key: _finalize_counter(buckets[key]) for key in sorted(buckets)}


def _finalize_counter(counter: dict[str, Any]) -> dict[str, Any]:
    count = counter["count"]
    return {
        "count": count,
        "changed": counter["changed"],
        "unchanged": counter["unchanged"],
        "changed_rate": _rate(counter["changed"], count),
        "baseline_regret_mean": _mean(counter["baseline_regrets"]),
        "baseline_regret_max": _max_or_none(counter["baseline_regrets"]),
        "value_regret_mean": _mean(counter["value_regrets"]),
        "value_regret_max": _max_or_none(counter["value_regrets"]),
    }


def _phase_for_turn(turn_index: int) -> str:
    if turn_index < 20:
        return "early"
    if turn_index < 50:
        return "mid"
    return "late"


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


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


if __name__ == "__main__":
    main()
