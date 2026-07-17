"""Fresh deadlock-valid replication of the frozen score-delta BMRH gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import time
from typing import Any, Literal

from splendor_ai.eval.run_score_delta_rolling_horizon import (
    _play_pair as _play_frozen_pair,
    _provenance,
    _summarize_block as _summarize_frozen_block,
    _write_json,
)
from splendor_ai.eval.run_shallow_operational_gate import SHALLOW_CONTRACT


Stage = Literal["direct", "all"]
Opponent = Literal["035", "shallow", "016", "023"]
VALID_TERMINATIONS = ("completed", "forced_pass_deadlock")


@dataclass(frozen=True, slots=True)
class DeadlockValidatedRollingHorizonConfig:
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
    parser.add_argument("--output-dir", default="outputs/deadlock_validated_rolling_horizon_042")
    parser.add_argument("--stage", choices=("direct", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    return parser


def config_from_args(args: argparse.Namespace) -> DeadlockValidatedRollingHorizonConfig:
    return DeadlockValidatedRollingHorizonConfig(
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


def run_gate(config: DeadlockValidatedRollingHorizonConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.output_dir / "gate_final.json"
    direct = _load_or_run_block(config, "direct_vs_035", "035", 404000, 100)
    direct_gate = _strength_gate(direct, expected_games=200)
    payload: dict[str, Any] = {
        "format": "deadlock_validated_rolling_horizon_gate_v1",
        "protocol": "docs/experiments/deadlock_validated_rolling_horizon_042.md",
        "config": _config_payload(config),
        "provenance": _provenance(config),
        "direct": direct,
        "direct_gate": direct_gate,
    }
    if not all(direct_gate.values()):
        payload["decision"] = "reject_fresh_replication_close_rolling_horizon"
        _write_json(final_path, payload)
        return payload
    if config.stage == "direct":
        payload["decision"] = "direct_pass_only"
        _write_json(final_path, payload)
        return payload

    broad_specs: tuple[tuple[str, Opponent, int], ...] = (
        ("broad_vs_shallow", "shallow", 404200),
        ("broad_vs_016", "016", 404300),
        ("broad_vs_023", "023", 404400),
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
        "authorize_score_delta_search_teacher"
        if all(all(gate.values()) for gate in broad_gates.values())
        else "reject_broad_keep_035"
    )
    _write_json(final_path, payload)
    return payload


def _load_or_run_block(
    config: DeadlockValidatedRollingHorizonConfig,
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
            raise ValueError(f"Existing {name} result does not match frozen `042`.")
        return payload

    progress_path = config.output_dir / f"{name}.progress.json"
    rows: dict[int, dict[str, Any]] = {}
    accumulated_elapsed = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError(f"Existing {name} progress does not match frozen `042`.")
        rows = {int(row["seed"]): row for row in progress.get("pairs", [])}
        accumulated_elapsed = float(progress.get("elapsed_seconds", 0.0))

    started = time.perf_counter()
    for seed in range(seed_start, seed_start + pairs):
        if seed in rows:
            continue
        rows[seed] = _play_frozen_pair(config, seed, opponent)
        elapsed = accumulated_elapsed + time.perf_counter() - started
        _write_json(
            progress_path,
            {
                "format": "deadlock_validated_rolling_horizon_progress_v1",
                "identity": identity,
                "elapsed_seconds": elapsed,
                "pairs": [rows[key] for key in sorted(rows)],
            },
        )
        print(f"[rhea042] block={name} pairs={len(rows)}/{pairs} seed={seed}", flush=True)

    elapsed = accumulated_elapsed + time.perf_counter() - started
    ordered = [rows[seed] for seed in range(seed_start, seed_start + pairs)]
    payload = _summarize_frozen_block(config, identity, ordered, elapsed)
    _add_corrected_integrity(payload)
    payload["format"] = "deadlock_validated_rolling_horizon_block_v1"
    _write_json(result_path, payload)
    return payload


def _add_corrected_integrity(payload: dict[str, Any]) -> None:
    reasons: dict[str, int] = {}
    invalid = 0
    valid_deadlocks = 0
    for pair in payload["pairs"]:
        for game in pair["games"]:
            reason = str(game["termination_reason"])
            reasons[reason] = reasons.get(reason, 0) + 1
            clean_deadlock = (
                reason == "forced_pass_deadlock"
                and not bool(game["timed_out"])
                and not bool(game["stalled"])
            )
            if clean_deadlock:
                valid_deadlocks += 1
            if reason not in VALID_TERMINATIONS or bool(game["timed_out"]) or bool(game["stalled"]):
                invalid += 1
    payload["summary"]["termination_reasons"] = reasons
    payload["summary"]["valid_forced_pass_deadlocks"] = valid_deadlocks
    payload["summary"]["invalid_termination_games"] = invalid


def _strength_gate(payload: dict[str, Any], expected_games: int) -> dict[str, bool]:
    summary = payload["summary"]
    share = summary["score_share"]
    margin = summary["final_score_margin"]
    return {
        "expected_games_completed": int(summary["completed_games"]) == expected_games,
        "zero_invalid_terminations": int(summary["invalid_termination_games"]) == 0,
        "zero_timeouts": int(summary["timed_out_games"]) == 0,
        "zero_stalls": int(summary["stalled_games"]) == 0,
        "exact_candidate_budget": bool(summary["candidate_exact_budget"]),
        "score_share_at_least_55pct": float(share["mean"]) >= 0.55,
        "score_share_ci95_lower_above_50pct": float(share["ci95_low"]) > 0.50,
        "positive_mean_paired_margin": float(margin["mean"]) > 0.0,
    }


def _block_identity(
    config: DeadlockValidatedRollingHorizonConfig,
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
        "valid_termination_reasons": list(VALID_TERMINATIONS),
        "forced_pass_outcome": "existing_score_and_card_count_adjudication",
    }


def _config_payload(config: DeadlockValidatedRollingHorizonConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("checkpoint_035", "checkpoint_016", "checkpoint_023", "output_dir"):
        payload[key] = str(payload[key])
    payload["valid_termination_reasons"] = list(VALID_TERMINATIONS)
    return payload


def _validate(config: DeadlockValidatedRollingHorizonConfig) -> None:
    if config.device != "cpu":
        raise ValueError("Protocol `042` authorizes CPU only.")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")
    for path in (config.checkpoint_035, config.checkpoint_016, config.checkpoint_023):
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    payload = run_gate(config_from_args(build_parser().parse_args()))
    print(json.dumps({"decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
