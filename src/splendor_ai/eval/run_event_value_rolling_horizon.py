"""Preregistered CPU gate for event-value branching-mutation planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Literal

from splendor_ai.bots import LegalActionAttentionScorerBot
from splendor_ai.bots.rolling_horizon_bot import (
    BranchingMutationRollingHorizonBot,
    RollingHorizonConfig,
)
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.eval.match import GameResult, play_game


Stage = Literal["smoke", "control", "parent", "all"]


@dataclass(frozen=True, slots=True)
class EventValueRollingHorizonGateConfig:
    checkpoint_035: Path
    output_dir: Path
    stage: Stage = "all"
    device: str = "cpu"
    bootstrap_samples: int = 20_000
    max_turns: int = 400
    repetition_limit: int = 4
    no_progress_limit: int = 60


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-035",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument("--output-dir", default="outputs/event_value_rolling_horizon_040")
    parser.add_argument("--stage", choices=("smoke", "control", "parent", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    return parser


def config_from_args(args: argparse.Namespace) -> EventValueRollingHorizonGateConfig:
    return EventValueRollingHorizonGateConfig(
        checkpoint_035=Path(args.checkpoint_035),
        output_dir=Path(args.output_dir),
        stage=args.stage,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
    )


def run_gate(config: EventValueRollingHorizonGateConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.output_dir / "gate_final.json"
    smoke = _load_or_run_stage(config, "smoke", 402000, 2, "score_delta")
    smoke_gate = _smoke_gate(smoke)
    payload: dict[str, Any] = {
        "format": "event_value_rolling_horizon_gate_v1",
        "protocol": "docs/experiments/event_value_rolling_horizon_040.md",
        "config": _config_payload(config),
        "smoke": smoke,
        "smoke_gate": smoke_gate,
    }
    if config.stage == "smoke" or not all(smoke_gate.values()):
        payload["decision"] = "stop_after_operational_smoke" if not all(smoke_gate.values()) else "smoke_only_complete"
        _write_json(final_path, payload)
        return payload

    control = _load_or_run_stage(config, "control", 402100, 30, "score_delta")
    control_gate = _strength_gate(control)
    payload.update({"control": control, "control_gate": control_gate})
    if config.stage == "control" or not all(control_gate.values()):
        payload["decision"] = "reject_fixed_event_evaluator" if not all(control_gate.values()) else "control_only_complete"
        _write_json(final_path, payload)
        return payload

    parent = _load_or_run_stage(config, "parent", 402200, 30, "raw_035")
    parent_gate = _strength_gate(parent)
    payload.update({"parent": parent, "parent_gate": parent_gate})
    payload["decision"] = (
        "authorize_broad_confirmation"
        if all(parent_gate.values())
        else "reject_as_stronger_teacher"
    )
    _write_json(final_path, payload)
    return payload


def _load_or_run_stage(
    config: EventValueRollingHorizonGateConfig,
    name: str,
    seed_start: int,
    pairs: int,
    opponent: Literal["score_delta", "raw_035"],
) -> dict[str, Any]:
    result_path = config.output_dir / f"{name}.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        identity = _stage_identity(config, name, seed_start, pairs, opponent)
        if payload.get("identity") != identity:
            raise ValueError(f"Existing {name} result does not match the frozen configuration.")
        return payload

    progress_path = config.output_dir / f"{name}.progress.json"
    identity = _stage_identity(config, name, seed_start, pairs, opponent)
    rows: dict[int, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError(f"Existing {name} progress does not match the frozen configuration.")
        rows = {int(row["seed"]): row for row in progress.get("pairs", [])}

    started = time.perf_counter()
    for seed in range(seed_start, seed_start + pairs):
        if seed in rows:
            continue
        rows[seed] = _play_pair(config, seed, opponent)
        _write_json(
            progress_path,
            {"format": "event_value_rolling_horizon_progress_v1", "identity": identity, "pairs": [rows[key] for key in sorted(rows)]},
        )
        print(f"[rhea040] stage={name} pairs={len(rows)}/{pairs} seed={seed}", flush=True)

    elapsed = time.perf_counter() - started
    ordered = [rows[seed] for seed in range(seed_start, seed_start + pairs)]
    payload = _summarize_stage(config, identity, ordered, elapsed)
    _write_json(result_path, payload)
    return payload


def _play_pair(
    config: EventValueRollingHorizonGateConfig,
    seed: int,
    opponent: Literal["score_delta", "raw_035"],
) -> dict[str, Any]:
    first_event = _planner(config, seed, "event_value_hc")
    first_opponent = _opponent(config, seed, opponent)
    first = play_game(
        first_event,
        first_opponent,
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    second_opponent = _opponent(config, seed, opponent)
    second_event = _planner(config, seed, "event_value_hc")
    second = play_game(
        second_opponent,
        second_event,
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
        "candidate_telemetry": [
            _planner_telemetry(first_event),
            _planner_telemetry(second_event),
        ],
        "opponent_telemetry": [
            _planner_telemetry(first_opponent),
            _planner_telemetry(second_opponent),
        ],
    }


def _planner(
    config: EventValueRollingHorizonGateConfig,
    seed: int,
    evaluator_mode: Literal["event_value_hc", "score_delta"],
) -> BranchingMutationRollingHorizonBot:
    prior = CheckpointLegalActionPriorScorer(config.checkpoint_035, device=config.device)
    return BranchingMutationRollingHorizonBot(
        prior,
        RollingHorizonConfig(
            actor_horizon=2,
            sequence_evaluations=32,
            determinizations=2,
            max_actor_candidates=4,
            seed=40_000_000 + seed,
            determinization_seed=41_000_000 + seed,
            evaluator_mode=evaluator_mode,
        ),
    )


def _opponent(
    config: EventValueRollingHorizonGateConfig,
    seed: int,
    opponent: Literal["score_delta", "raw_035"],
) -> Any:
    if opponent == "score_delta":
        return _planner(config, seed, "score_delta")
    return LegalActionAttentionScorerBot(config.checkpoint_035, device=config.device)


def _planner_telemetry(bot: Any) -> dict[str, int]:
    return {
        "planning_decisions": int(getattr(bot, "total_planning_decisions", 0)),
        "sequence_evaluations": int(getattr(bot, "total_sequence_evaluations", 0)),
        "simulated_actions": int(getattr(bot, "total_simulated_actions", 0)),
    }


def _summarize_stage(
    config: EventValueRollingHorizonGateConfig,
    identity: dict[str, Any],
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    shares = []
    margins = []
    for row in rows:
        outcomes = []
        row_margins = []
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
            winner = game["winner"]
            outcomes.append(0.5 if winner is None else float(int(winner) == int(seat)))
            scores = game["final_scores"]
            row_margins.append(float(scores[seat]) - float(scores[1 - seat]))
        shares.append(statistics.fmean(outcomes))
        margins.append(statistics.fmean(row_margins))
    games = [game for row in rows for game in row["games"]]
    return {
        "format": "event_value_rolling_horizon_stage_v1",
        "identity": identity,
        "elapsed_seconds": elapsed_seconds,
        "pairs": rows,
        "summary": {
            "completed_pairs": len(rows),
            "completed_games": len(games),
            "score_share": _mean_ci(shares, config.bootstrap_samples, int(identity["seed_start"])),
            "final_score_margin": _mean_ci(margins, config.bootstrap_samples, int(identity["seed_start"]) + 1),
            "timed_out_games": sum(bool(game["timed_out"]) for game in games),
            "stalled_games": sum(bool(game["stalled"]) for game in games),
            "noncompleted_games": sum(game["termination_reason"] != "completed" for game in games),
            "candidate_sequence_evaluations": sum(item["sequence_evaluations"] for row in rows for item in row["candidate_telemetry"]),
            "candidate_planning_decisions": sum(item["planning_decisions"] for row in rows for item in row["candidate_telemetry"]),
            "candidate_simulated_actions": sum(item["simulated_actions"] for row in rows for item in row["candidate_telemetry"]),
            "opponent_sequence_evaluations": sum(item["sequence_evaluations"] for row in rows for item in row["opponent_telemetry"]),
            "opponent_planning_decisions": sum(item["planning_decisions"] for row in rows for item in row["opponent_telemetry"]),
            "opponent_simulated_actions": sum(item["simulated_actions"] for row in rows for item in row["opponent_telemetry"]),
        },
    }


def _smoke_gate(payload: dict[str, Any]) -> dict[str, bool]:
    summary = payload["summary"]
    return {
        "four_games_completed": int(summary["completed_games"]) == 4,
        "zero_noncompleted_games": int(summary["noncompleted_games"]) == 0,
        "candidate_exact_budget": int(summary["candidate_sequence_evaluations"])
        == 32 * int(summary["candidate_planning_decisions"]),
        "control_exact_budget": int(summary["opponent_sequence_evaluations"])
        == 32 * int(summary["opponent_planning_decisions"]),
        "elapsed_below_60_minutes": float(payload["elapsed_seconds"]) <= 3600.0,
    }


def _strength_gate(payload: dict[str, Any]) -> dict[str, bool]:
    summary = payload["summary"]
    share = summary["score_share"]
    margin = summary["final_score_margin"]
    return {
        "zero_noncompleted_games": int(summary["noncompleted_games"]) == 0,
        "score_share_at_least_55pct": float(share["mean"]) >= 0.55,
        "score_share_ci95_lower_above_50pct": float(share["ci95_low"]) > 0.50,
        "positive_mean_paired_margin": float(margin["mean"]) > 0.0,
    }


def _mean_ci(values: list[float], samples: int, seed: int) -> dict[str, float]:
    mean = statistics.fmean(values) if values else 0.0
    if not values or samples <= 0 or len(values) == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    count = len(values)
    boot = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": boot[int(0.025 * (samples - 1))],
        "ci95_high": boot[int(0.975 * (samples - 1))],
    }


def _stage_identity(
    config: EventValueRollingHorizonGateConfig,
    name: str,
    seed_start: int,
    pairs: int,
    opponent: str,
) -> dict[str, Any]:
    return {
        "stage": name,
        "seed_start": seed_start,
        "pairs": pairs,
        "opponent": opponent,
        "checkpoint_035": str(config.checkpoint_035),
        "device": config.device,
        "max_turns": config.max_turns,
        "repetition_limit": config.repetition_limit,
        "no_progress_limit": config.no_progress_limit,
        "actor_horizon": 2,
        "sequence_evaluations": 32,
        "determinizations": 2,
        "max_actor_candidates": 4,
    }


def _config_payload(config: EventValueRollingHorizonGateConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["checkpoint_035"] = str(config.checkpoint_035)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _validate(config: EventValueRollingHorizonGateConfig) -> None:
    if not config.checkpoint_035.exists():
        raise FileNotFoundError(config.checkpoint_035)
    if config.device != "cpu":
        raise ValueError("Protocol 040 authorizes CPU only.")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    payload = run_gate(config_from_args(build_parser().parse_args()))
    print(json.dumps({"decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
