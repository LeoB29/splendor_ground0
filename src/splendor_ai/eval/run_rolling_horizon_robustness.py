"""Run the frozen five-block score-delta BMRH robustness gate `043`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import time
from typing import Any, Literal

from splendor_ai.eval.run_deadlock_validated_rolling_horizon import (
    VALID_TERMINATIONS,
    _add_corrected_integrity,
    _play_frozen_pair,
    _provenance,
    _strength_gate,
    _summarize_frozen_block,
    _write_json,
)
from splendor_ai.eval.run_shallow_operational_gate import SHALLOW_CONTRACT


Stage = Literal["direct", "all"]
Opponent = Literal["035", "shallow", "016", "023"]
DIRECT_BLOCKS: tuple[tuple[str, int], ...] = (
    ("A", 431000),
    ("B", 443000),
    ("C", 457000),
    ("D", 469000),
    ("E", 487000),
)
BROAD_BLOCKS: tuple[tuple[str, Opponent, int], ...] = (
    ("broad_vs_shallow", "shallow", 510000),
    ("broad_vs_016", "016", 520000),
    ("broad_vs_023", "023", 530000),
)


@dataclass(frozen=True, slots=True)
class RollingHorizonRobustnessConfig:
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
    parser.add_argument("--output-dir", default="outputs/rolling_horizon_robustness_043")
    parser.add_argument("--stage", choices=("direct", "all"), default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    return parser


def config_from_args(args: argparse.Namespace) -> RollingHorizonRobustnessConfig:
    return RollingHorizonRobustnessConfig(
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


def run_gate(config: RollingHorizonRobustnessConfig) -> dict[str, Any]:
    _validate(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = config.output_dir / "gate_final.json"

    direct_blocks: dict[str, dict[str, Any]] = {}
    for label, seed_start in DIRECT_BLOCKS:
        direct_blocks[label] = _load_or_run_block(
            config,
            f"direct_block_{label.lower()}",
            "035",
            seed_start,
            100,
        )
    direct = _aggregate_direct(config, direct_blocks)
    direct_block_gates = {
        label: _integrity_gate(block, expected_games=200)
        for label, block in direct_blocks.items()
    }
    direct_gate = _direct_gate(direct, direct_blocks, direct_block_gates)
    payload: dict[str, Any] = {
        "format": "rolling_horizon_robustness_gate_v1",
        "protocol": "docs/experiments/rolling_horizon_robustness_043.md",
        "config": _config_payload(config),
        "provenance": _provenance(config),
        "direct_blocks": direct_blocks,
        "direct": direct,
        "direct_block_gates": direct_block_gates,
        "direct_gate": direct_gate,
    }
    if not all(direct_gate.values()):
        payload["decision"] = "reject_final_robustness_close_rolling_horizon"
        _write_json(final_path, payload)
        return payload
    if config.stage == "direct":
        payload["decision"] = "direct_pass_only"
        _write_json(final_path, payload)
        return payload

    broad: dict[str, Any] = {}
    broad_gates: dict[str, Any] = {}
    for name, opponent, seed_start in BROAD_BLOCKS:
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
    config: RollingHorizonRobustnessConfig,
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
            raise ValueError(f"Existing {name} result does not match frozen `043`.")
        return payload

    progress_path = config.output_dir / f"{name}.progress.json"
    rows: dict[int, dict[str, Any]] = {}
    accumulated_elapsed = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError(f"Existing {name} progress does not match frozen `043`.")
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
                "format": "rolling_horizon_robustness_progress_v1",
                "identity": identity,
                "elapsed_seconds": elapsed,
                "pairs": [rows[key] for key in sorted(rows)],
            },
        )
        print(f"[rhea043] block={name} pairs={len(rows)}/{pairs} seed={seed}", flush=True)

    elapsed = accumulated_elapsed + time.perf_counter() - started
    ordered = [rows[seed] for seed in range(seed_start, seed_start + pairs)]
    payload = _summarize_frozen_block(config, identity, ordered, elapsed)
    _add_corrected_integrity(payload)
    payload["format"] = "rolling_horizon_robustness_block_v1"
    _write_json(result_path, payload)
    return payload


def _aggregate_direct(
    config: RollingHorizonRobustnessConfig,
    blocks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    elapsed = 0.0
    for label, _seed_start in DIRECT_BLOCKS:
        block = blocks[label]
        rows.extend(block["pairs"])
        elapsed += float(block["elapsed_seconds"])
    seeds = [int(row["seed"]) for row in rows]
    if len(seeds) != 500 or len(set(seeds)) != 500:
        raise ValueError("Direct aggregate must contain 500 unique paired seeds.")
    identity = {
        "name": "direct_aggregate",
        "opponent": "035",
        "seed_start": 430043,
        "pairs": 500,
        "seed_blocks": [
            {"label": label, "seed_start": seed_start, "pairs": 100}
            for label, seed_start in DIRECT_BLOCKS
        ],
        "bootstrap_unit": "paired_seed",
        "bootstrap_seed": 430043,
    }
    payload = _summarize_frozen_block(config, identity, rows, elapsed)
    _add_corrected_integrity(payload)
    payload["format"] = "rolling_horizon_robustness_aggregate_v1"
    payload["summary"]["block_score_shares"] = {
        label: float(blocks[label]["summary"]["score_share"]["mean"])
        for label, _seed_start in DIRECT_BLOCKS
    }
    _write_json(config.output_dir / "direct_aggregate.json", payload)
    return payload


def _integrity_gate(payload: dict[str, Any], expected_games: int) -> dict[str, bool]:
    summary = payload["summary"]
    return {
        "expected_games_completed": int(summary["completed_games"]) == expected_games,
        "zero_invalid_terminations": int(summary["invalid_termination_games"]) == 0,
        "zero_timeouts": int(summary["timed_out_games"]) == 0,
        "zero_stalls": int(summary["stalled_games"]) == 0,
        "exact_candidate_budget": bool(summary["candidate_exact_budget"]),
    }


def _direct_gate(
    aggregate: dict[str, Any],
    blocks: dict[str, dict[str, Any]],
    block_gates: dict[str, dict[str, bool]],
) -> dict[str, bool]:
    strength = _strength_gate(aggregate, expected_games=1000)
    blocks_above = sum(
        float(blocks[label]["summary"]["score_share"]["mean"]) > 0.50
        for label, _seed_start in DIRECT_BLOCKS
    )
    return {
        "all_five_blocks_integrity": all(
            all(block_gates[label].values()) for label, _seed_start in DIRECT_BLOCKS
        ),
        "expected_aggregate_games_completed": strength["expected_games_completed"],
        "zero_aggregate_invalid_terminations": strength["zero_invalid_terminations"],
        "zero_aggregate_timeouts": strength["zero_timeouts"],
        "zero_aggregate_stalls": strength["zero_stalls"],
        "exact_aggregate_candidate_budget": strength["exact_candidate_budget"],
        "aggregate_score_share_at_least_55pct": strength["score_share_at_least_55pct"],
        "aggregate_score_share_ci95_lower_above_50pct": strength[
            "score_share_ci95_lower_above_50pct"
        ],
        "positive_aggregate_mean_paired_margin": strength["positive_mean_paired_margin"],
        "at_least_four_of_five_blocks_above_50pct": blocks_above >= 4,
    }


def _block_identity(
    config: RollingHorizonRobustnessConfig,
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


def _config_payload(config: RollingHorizonRobustnessConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("checkpoint_035", "checkpoint_016", "checkpoint_023", "output_dir"):
        payload[key] = str(payload[key])
    payload["direct_blocks"] = [
        {"label": label, "seed_start": seed_start, "pairs": 100}
        for label, seed_start in DIRECT_BLOCKS
    ]
    payload["valid_termination_reasons"] = list(VALID_TERMINATIONS)
    return payload


def _validate(config: RollingHorizonRobustnessConfig) -> None:
    if config.device != "cpu":
        raise ValueError("Protocol `043` authorizes CPU only.")
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
