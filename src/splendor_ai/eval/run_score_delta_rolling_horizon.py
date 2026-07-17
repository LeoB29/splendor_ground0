"""Frozen direct and broad gate for the score-delta BMRH control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any, Literal

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.bots.rolling_horizon_bot import (
    BranchingMutationRollingHorizonBot,
    RollingHorizonConfig,
)
from splendor_ai.bots.search_bot import CheckpointLegalActionPriorScorer
from splendor_ai.eval.match import play_game
from splendor_ai.eval.run_shallow_operational_gate import SHALLOW_CONTRACT


Stage = Literal["direct", "all"]
Opponent = Literal["035", "shallow", "016", "023"]


@dataclass(frozen=True, slots=True)
class ScoreDeltaRollingHorizonGateConfig:
    checkpoint_035: Path
    checkpoint_016: Path
    checkpoint_023: Path
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
    parser.add_argument(
        "--checkpoint-016",
        default="outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt",
    )
    parser.add_argument(
        "--checkpoint-023",
        default="outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt",
    )
    parser.add_argument("--output-dir", default="outputs/score_delta_rolling_horizon_041")
    parser.add_argument("--stage", choices=("direct", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    return parser


def config_from_args(args: argparse.Namespace) -> ScoreDeltaRollingHorizonGateConfig:
    return ScoreDeltaRollingHorizonGateConfig(
        checkpoint_035=Path(args.checkpoint_035),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_023=Path(args.checkpoint_023),
        output_dir=Path(args.output_dir),
        stage=args.stage,
        device=args.device,
        bootstrap_samples=args.bootstrap_samples,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
    )


def run_gate(config: ScoreDeltaRollingHorizonGateConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.output_dir / "gate_final.json"
    direct = _load_or_run_block(config, "direct_vs_035", "035", 403000, 100)
    direct_gate = _strength_gate(direct, expected_games=200)
    payload: dict[str, Any] = {
        "format": "score_delta_rolling_horizon_gate_v1",
        "protocol": "docs/experiments/score_delta_rolling_horizon_041.md",
        "config": _config_payload(config),
        "provenance": _provenance(config),
        "direct": direct,
        "direct_gate": direct_gate,
    }
    if not all(direct_gate.values()):
        payload["decision"] = "reject_score_delta_close_rolling_horizon"
        _write_json(final_path, payload)
        return payload
    if config.stage == "direct":
        payload["decision"] = "direct_pass_only"
        _write_json(final_path, payload)
        return payload

    broad_specs: tuple[tuple[str, Opponent, int], ...] = (
        ("broad_vs_shallow", "shallow", 403200),
        ("broad_vs_016", "016", 403300),
        ("broad_vs_023", "023", 403400),
    )
    broad: dict[str, Any] = {}
    broad_gates: dict[str, Any] = {}
    for name, opponent, seed_start in broad_specs:
        block = _load_or_run_block(config, name, opponent, seed_start, 50)
        broad[opponent] = block
        broad_gates[opponent] = _strength_gate(block, expected_games=100)
    payload["broad"] = broad
    payload["broad_gates"] = broad_gates
    payload["decision"] = (
        "authorize_search_teacher"
        if all(all(gate.values()) for gate in broad_gates.values())
        else "reject_broad_keep_035"
    )
    _write_json(final_path, payload)
    return payload


def _load_or_run_block(
    config: ScoreDeltaRollingHorizonGateConfig,
    name: str,
    opponent: Opponent,
    seed_start: int,
    pairs: int,
) -> dict[str, Any]:
    identity = _block_identity(config, name, opponent, seed_start, pairs)
    result_path = config.output_dir / f"{name}.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError(f"Existing {name} result does not match frozen `041`.")
        return payload

    progress_path = config.output_dir / f"{name}.progress.json"
    rows: dict[int, dict[str, Any]] = {}
    accumulated_elapsed = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError(f"Existing {name} progress does not match frozen `041`.")
        rows = {int(row["seed"]): row for row in progress.get("pairs", [])}
        accumulated_elapsed = float(progress.get("elapsed_seconds", 0.0))

    started = time.perf_counter()
    for seed in range(seed_start, seed_start + pairs):
        if seed in rows:
            continue
        rows[seed] = _play_pair(config, seed, opponent)
        elapsed = accumulated_elapsed + time.perf_counter() - started
        _write_json(
            progress_path,
            {
                "format": "score_delta_rolling_horizon_progress_v1",
                "identity": identity,
                "elapsed_seconds": elapsed,
                "pairs": [rows[key] for key in sorted(rows)],
            },
        )
        print(f"[rhea041] block={name} pairs={len(rows)}/{pairs} seed={seed}", flush=True)

    elapsed = accumulated_elapsed + time.perf_counter() - started
    ordered = [rows[seed] for seed in range(seed_start, seed_start + pairs)]
    payload = _summarize_block(config, identity, ordered, elapsed)
    _write_json(result_path, payload)
    return payload


def _play_pair(
    config: ScoreDeltaRollingHorizonGateConfig,
    seed: int,
    opponent: Opponent,
) -> dict[str, Any]:
    candidate = _candidate(config, seed)
    baseline = _opponent(config, opponent)
    first = play_game(
        candidate,
        baseline,
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    second = play_game(
        baseline,
        candidate,
        seed=seed,
        max_turns=config.max_turns,
        repetition_limit=config.repetition_limit,
        no_progress_limit=config.no_progress_limit,
    )
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [asdict(first), asdict(second)],
        "candidate_telemetry": _planner_telemetry(candidate),
    }


def _candidate(
    config: ScoreDeltaRollingHorizonGateConfig,
    seed: int,
) -> BranchingMutationRollingHorizonBot:
    return BranchingMutationRollingHorizonBot(
        CheckpointLegalActionPriorScorer(config.checkpoint_035, device=config.device),
        RollingHorizonConfig(
            actor_horizon=2,
            sequence_evaluations=32,
            determinizations=2,
            max_actor_candidates=4,
            seed=40_000_000 + seed,
            determinization_seed=41_000_000 + seed,
            evaluator_mode="score_delta",
        ),
    )


def _opponent(config: ScoreDeltaRollingHorizonGateConfig, opponent: Opponent) -> Any:
    if opponent == "035":
        return LegalActionAttentionScorerBot(config.checkpoint_035, device=config.device)
    if opponent == "016":
        return LegalActionScorerBot(config.checkpoint_016, device=config.device)
    if opponent == "023":
        return CanonicalEntityActionScorerBot(config.checkpoint_023, device=config.device)
    if opponent == "shallow":
        return ShallowSearchBot(**SHALLOW_CONTRACT)
    raise ValueError(opponent)


def _planner_telemetry(bot: BranchingMutationRollingHorizonBot) -> dict[str, int]:
    return {
        "planning_decisions": bot.total_planning_decisions,
        "sequence_evaluations": bot.total_sequence_evaluations,
        "simulated_actions": bot.total_simulated_actions,
    }


def _summarize_block(
    config: ScoreDeltaRollingHorizonGateConfig,
    identity: dict[str, Any],
    rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    shares: list[float] = []
    margins: list[float] = []
    wins = losses = draws = 0
    for row in rows:
        pair_points: list[float] = []
        pair_margins: list[float] = []
        for game, seat in zip(row["games"], row["candidate_seats"], strict=True):
            winner = game["winner"]
            if winner is None:
                draws += 1
                pair_points.append(0.5)
            elif int(winner) == int(seat):
                wins += 1
                pair_points.append(1.0)
            else:
                losses += 1
                pair_points.append(0.0)
            scores = game["final_scores"]
            pair_margins.append(float(scores[seat]) - float(scores[1 - seat]))
        shares.append(statistics.fmean(pair_points))
        margins.append(statistics.fmean(pair_margins))
    games = [game for row in rows for game in row["games"]]
    decisions = sum(int(row["candidate_telemetry"]["planning_decisions"]) for row in rows)
    sequences = sum(int(row["candidate_telemetry"]["sequence_evaluations"]) for row in rows)
    return {
        "format": "score_delta_rolling_horizon_block_v1",
        "identity": identity,
        "elapsed_seconds": elapsed_seconds,
        "pairs": rows,
        "summary": {
            "completed_pairs": len(rows),
            "completed_games": len(games),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "score_share": _mean_ci(shares, config.bootstrap_samples, int(identity["seed_start"])),
            "final_score_margin": _mean_ci(margins, config.bootstrap_samples, int(identity["seed_start"]) + 1),
            "timed_out_games": sum(bool(game["timed_out"]) for game in games),
            "stalled_games": sum(bool(game["stalled"]) for game in games),
            "noncompleted_games": sum(game["termination_reason"] != "completed" for game in games),
            "candidate_planning_decisions": decisions,
            "candidate_sequence_evaluations": sequences,
            "candidate_simulated_actions": sum(int(row["candidate_telemetry"]["simulated_actions"]) for row in rows),
            "candidate_exact_budget": sequences == 32 * decisions,
        },
    }


def _strength_gate(payload: dict[str, Any], expected_games: int) -> dict[str, bool]:
    summary = payload["summary"]
    share = summary["score_share"]
    margin = summary["final_score_margin"]
    return {
        "expected_games_completed": int(summary["completed_games"]) == expected_games,
        "zero_noncompleted_games": int(summary["noncompleted_games"]) == 0,
        "exact_candidate_budget": bool(summary["candidate_exact_budget"]),
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
    bootstrap = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci95_low": bootstrap[int(0.025 * (samples - 1))],
        "ci95_high": bootstrap[int(0.975 * (samples - 1))],
    }


def _block_identity(
    config: ScoreDeltaRollingHorizonGateConfig,
    name: str,
    opponent: Opponent,
    seed_start: int,
    pairs: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "opponent": opponent,
        "seed_start": seed_start,
        "pairs": pairs,
        "device": config.device,
        "max_turns": config.max_turns,
        "repetition_limit": config.repetition_limit,
        "no_progress_limit": config.no_progress_limit,
        "actor_horizon": 2,
        "sequence_evaluations": 32,
        "determinizations": 2,
        "max_actor_candidates": 4,
        "evaluator_mode": "score_delta",
        "checkpoint_035": str(config.checkpoint_035),
        "checkpoint_016": str(config.checkpoint_016),
        "checkpoint_023": str(config.checkpoint_023),
        "shallow_contract": SHALLOW_CONTRACT,
    }


def _provenance(config: ScoreDeltaRollingHorizonGateConfig) -> dict[str, Any]:
    return {
        "checkpoint_035_sha256": _sha256(config.checkpoint_035),
        "checkpoint_016_sha256": _sha256(config.checkpoint_016),
        "checkpoint_023_sha256": _sha256(config.checkpoint_023),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_payload(config: ScoreDeltaRollingHorizonGateConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("checkpoint_035", "checkpoint_016", "checkpoint_023", "output_dir"):
        payload[key] = str(payload[key])
    return payload


def _validate(config: ScoreDeltaRollingHorizonGateConfig) -> None:
    if config.device != "cpu":
        raise ValueError("Protocol `041` authorizes CPU only.")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    for path in (config.checkpoint_035, config.checkpoint_016, config.checkpoint_023):
        if not path.exists():
            raise FileNotFoundError(path)


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
