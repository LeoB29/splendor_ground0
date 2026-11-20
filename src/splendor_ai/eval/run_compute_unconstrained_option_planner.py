"""Run compute-unconstrained persistent option planner experiment `080`."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from splendor_ai.bots.persistent_option_bot import (
    OPTION_DEPTH,
    OPTION_DETERMINIZATIONS,
    OPTION_INTERNAL_BRANCHING,
    OPTION_ROOT_BRANCHING,
)
from splendor_ai.eval.run_persistent_card_option_planner import (
    CHECKPOINT_SHA,
    OptionPlannerConfig,
    _init,
    _integrity,
    _merge,
    _pair,
    _panel,
    _rates,
    _sha,
    _strength,
    _write,
)


PROTOCOL = "docs/experiments/compute_unconstrained_option_planner_080.md"
MECHANICS_TIME_LIMIT = 12 * 60 * 60
STRENGTH_PROJECTION_LIMIT = 48 * 60 * 60


def strength_decision(integrity: dict[str, bool], score: dict[str, float]) -> str:
    if not all(integrity.values()):
        return "compute_unconstrained_option_planner_080_invalid_no_strength_decision"
    if score["mean"] >= 0.55 and score["ci95_low"] > 0.50:
        return "compute_unconstrained_option_planner_080_stronger_promote_research_champion"
    if score["ci95_high"] < 0.50:
        return "compute_unconstrained_option_planner_080_weaker_close_architecture"
    return "compute_unconstrained_option_planner_080_not_decisively_stronger_close_architecture"


def run_compute_unconstrained_option_planner(
    config: OptionPlannerConfig,
) -> dict[str, Any]:
    began = time.perf_counter()
    digest = _sha(config.checkpoint)
    stage_a_rows, stage_a_elapsed = _panel(
        config, config.mechanics_seed_start, config.mechanics_pairs, "mechanics"
    )
    _init(
        str(config.checkpoint),
        (config.max_turns, config.repetition_limit, config.no_progress_limit),
    )
    deterministic = _pair(config.mechanics_seed_start) == stage_a_rows[0]
    stage_a_telemetry = _merge(stage_a_rows)
    stage_a_rates = _rates(stage_a_telemetry, 2 * config.mechanics_pairs)
    stage_a_integrity = _integrity(
        stage_a_rows, config.mechanics_pairs, config.mechanics_seed_start
    )
    stage_a_integrity.update(
        {
            "deterministic": deterministic,
            "checkpoint": digest == CHECKPOINT_SHA,
            "frozen_contract": (
                OPTION_DEPTH,
                OPTION_ROOT_BRANCHING,
                OPTION_INTERNAL_BRANCHING,
                OPTION_DETERMINIZATIONS,
            )
            == (3, 8, 4, 4),
        }
    )
    projected = stage_a_elapsed * config.strength_pairs / config.mechanics_pairs
    operational = {
        "mechanics_under_12_hours": stage_a_elapsed <= MECHANICS_TIME_LIMIT,
        "projected_strength_under_48_hours": projected <= STRENGTH_PROJECTION_LIMIT,
    }
    mechanics = {
        "movement": stage_a_rates["movement"] >= 0.15,
        "option_active": stage_a_rates["option_decision_rate"] >= 0.50,
        "completion": stage_a_rates["option_completion_rate"] >= 0.50,
        "purchases": stage_a_rates["mean_purchases"] >= 5.0,
    }
    stage_b = None
    if not all(stage_a_integrity.values()) or not all(operational.values()):
        decision = "compute_unconstrained_option_planner_080_invalid_no_strength_decision"
    elif not all(mechanics.values()):
        decision = "compute_unconstrained_option_planner_080_mechanics_failed_close_architecture"
    else:
        stage_b_rows, stage_b_elapsed = _panel(
            config, config.strength_seed_start, config.strength_pairs, "strength"
        )
        stage_b_telemetry = _merge(stage_b_rows)
        stage_b_rates = _rates(stage_b_telemetry, 2 * config.strength_pairs)
        stage_b_integrity = _integrity(
            stage_b_rows, config.strength_pairs, config.strength_seed_start
        )
        stage_b_integrity.update(
            {
                "checkpoint": digest == CHECKPOINT_SHA,
                "movement": stage_b_rates["movement"] >= 0.15,
            }
        )
        strength = _strength(stage_b_rows, config)
        decision = strength_decision(stage_b_integrity, strength["score_share"])
        stage_b = {
            "integrity": stage_b_integrity,
            "rates": stage_b_rates,
            "telemetry": stage_b_telemetry.as_dict(),
            "strength": strength,
            "elapsed_seconds": stage_b_elapsed,
            "clusters": stage_b_rows,
        }
    passed = decision.endswith("_stronger_promote_research_champion")
    payload = {
        "experiment": "080",
        "protocol": PROTOCOL,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "checkpoint_sha256": digest,
        "stage_a": {
            "integrity": stage_a_integrity,
            "operational": operational,
            "mechanics": mechanics,
            "rates": stage_a_rates,
            "telemetry": stage_a_telemetry.as_dict(),
            "elapsed_seconds": stage_a_elapsed,
            "projected_stage_b_seconds": projected,
            "strength_not_used": True,
            "clusters": stage_a_rows,
        },
        "stage_b": stage_b,
        "decision": decision,
        "passed": passed,
        "research_champion_promoted": passed,
        "fast_champion": "035",
        "human_data_used": False,
        "elapsed_seconds": time.perf_counter() - began,
    }
    _write(payload, config.output_dir)
    print(f"[080] decision={decision}")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute-unconstrained persistent option planner (080)"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--output-dir", default="outputs/compute_unconstrained_option_planner_080"
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_compute_unconstrained_option_planner(
        OptionPlannerConfig(
            checkpoint=Path(args.checkpoint),
            output_dir=Path(args.output_dir),
            workers=args.workers,
            mechanics_seed_start=775_000,
            strength_seed_start=776_000,
            bootstrap_seed=776_080,
        )
    )


if __name__ == "__main__":
    main()
