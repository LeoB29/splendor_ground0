"""Run the preregistered recurrent population-self-play Stage B pilot (062)."""

from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    collect_recurrent_league,
    policy_diagnostics,
    recurrent_snapshot_factory,
    run_parent_gate,
    run_static_population_gate,
    save_generation_checkpoint,
    sha256,
    static_opponent_factories,
    train_recurrent_generation,
    verify_checkpoint_reload,
)
from splendor_ai.training.recurrent_policy import load_recurrent_policy_checkpoint


RESULT_061_SHA256 = (
    "1b1469fca9b7abcc22525e6f57a2fcdf9091863efe80bbb5e862f6deb1665f08"
)
EPOCH0_SHA256 = (
    "2d807231666420bb221049aa7e2a4a28358ef62618cf1c652ede67dd7249acef"
)
CHECKPOINT_035_SHA256 = (
    "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
)
CHECKPOINT_016_SHA256 = (
    "876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056"
)
CHECKPOINT_023_SHA256 = (
    "44f4fe6213ec2377eccfbdb439180aa37b9e70c2e29e7177d7982340934fb49f"
)
COLLECTION_SEEDS = {1: 620_000, 2: 621_000, 3: 622_000}
PARENT_GATE_SEEDS = {1: 623_100, 2: 623_200, 3: 623_300}
STATIC_GATE_SEEDS = {
    "035": 624_000,
    "016": 624_100,
    "023": 624_200,
    "shallow": 624_300,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_prerequisites(
    result_061: Path,
    epoch0: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    paths = {
        "result_061": (result_061, RESULT_061_SHA256),
        "epoch0": (epoch0, EPOCH0_SHA256),
        "035": (checkpoint_035, CHECKPOINT_035_SHA256),
        "016": (checkpoint_016, CHECKPOINT_016_SHA256),
        "023": (checkpoint_023, CHECKPOINT_023_SHA256),
    }
    payload = {}
    for label, (path, expected) in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        payload[label] = {
            "path": str(path).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": expected,
            "passed": actual == expected,
        }
    result = json.loads(result_061.read_text(encoding="utf-8"))
    payload["result_061"]["decision"] = result.get("decision")
    payload["result_061"]["decision_passed"] = (
        result.get("decision")
        == "recurrent_stage_a_feasible_authorize_stage_b_preregistration"
    )
    payload["all_passed"] = all(
        row["passed"] for key, row in payload.items() if key != "all_passed"
    ) and payload["result_061"]["decision_passed"]
    return payload


def _balance_passed(summary, opponent_count: int, config: RecurrentLeagueConfig) -> bool:
    expected = config.games // (2 * opponent_count)
    validation = max(
        config.minimum_validation_games_per_cell,
        expected // config.validation_divisor,
    )
    train = expected - validation
    for label, seats in summary.opponent_seat_games.items():
        if any(count != expected for count in seats.values()):
            return False
        for seat in ("seat0", "seat1"):
            if summary.split_opponent_seat_games["train"][label][seat] != train:
                return False
            if (
                summary.split_opponent_seat_games["validation"][label][seat]
                != validation
            ):
                return False
    return True


def _generation_gates(
    summary,
    update,
    parent_gate,
    *,
    opponent_count: int,
    config: RecurrentLeagueConfig,
    model,
) -> dict[str, bool]:
    finite_values = [
        update.behavior_reconstruction_max_error,
        update.first_gradient_norm,
        update.mean_gradient_norm,
        update.validation_raw_kl,
        update.validation_argmax_movement,
        update.validation_surrogate,
        update.validation_normalized_entropy,
        update.validation_mean_total_variation,
        *update.epoch_validation_kl,
        *update.epoch_policy_loss,
    ]
    return {
        "exact_collection_size": (
            summary.eligible_games == config.games
            and summary.attempted_games <= config.maximum_attempts
        ),
        "opponent_seat_split_balance": _balance_passed(
            summary,
            opponent_count,
            config,
        ),
        "eligible_only": (
            sum(summary.eligible_termination_counts.values())
            == summary.eligible_games
            and set(summary.eligible_termination_counts).issubset(
                {"completed", "forced_pass_deadlock"}
            )
        ),
        "zero_illegal_actions": summary.termination_counts.get(
            "illegal_action",
            0,
        )
        == 0,
        "behavior_reconstruction": (
            update.behavior_reconstruction_max_error <= 1e-5
        ),
        "finite_metrics": all(math.isfinite(value) for value in finite_values),
        "gradient_present": update.first_gradient_norm > 1e-8,
        "parent_frozen": all(
            not parameter.requires_grad for parameter in model.parent.parameters()
        ),
        "residual_trainable": any(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not name.startswith("parent.")
        ),
        "validation_kl_band": 0.0001
        <= update.validation_raw_kl
        <= 0.0100,
        "validation_argmax_movement_band": 0.005
        <= update.validation_argmax_movement
        <= 0.12,
        "validation_surrogate_positive": update.validation_surrogate > 0.0,
        "validation_entropy": update.validation_normalized_entropy >= 0.30,
        "checkpoint_reload_exact": update.checkpoint_reload_exact,
        "parent_gate_valid": bool(parent_gate["valid"]),
        "parent_score_share": float(parent_gate["candidate_score_share"]) >= 0.45,
        "parent_score_margin": float(parent_gate["mean_paired_score_margin"])
        >= -2.0,
    }


def run(
    *,
    output_dir: Path,
    result_061: Path,
    epoch0_path: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
    config: RecurrentLeagueConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RecurrentLeagueConfig()
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    prerequisites = _validate_prerequisites(
        result_061,
        epoch0_path,
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    if not prerequisites["all_passed"]:
        payload = {
            "experiment": "062",
            "prerequisites": prerequisites,
            "decision": "recurrent_stage_b_invalid_no_learning_decision",
        }
        _atomic_json(output_dir / "result.json", payload)
        return payload

    torch.manual_seed(620_062)
    epoch0 = load_recurrent_policy_checkpoint(
        epoch0_path,
        device=cfg.device,
    ).eval()
    current_parent = epoch0
    current_parent_path = epoch0_path
    static = static_opponent_factories(
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
        device=cfg.device,
    )
    accepted_models = []
    accepted_paths: list[Path] = []
    generation_payloads = []
    final_validation_steps = None
    stopped_generation = None

    for generation in (1, 2, 3):
        league = list(static)
        for historical_generation, historical_model in enumerate(
            accepted_models,
            start=1,
        ):
            league.append(
                (
                    f"snapshot_g{historical_generation}",
                    recurrent_snapshot_factory(historical_model),
                )
            )
        games, steps, collection = collect_recurrent_league(
            current_parent,
            league,
            generation=generation,
            seed_start=COLLECTION_SEEDS[generation],
            config=cfg,
        )
        candidate, update = train_recurrent_generation(
            current_parent,
            steps,
            generation=generation,
            config=cfg,
        )
        candidate_path = output_dir / f"generation_{generation}.pt"
        save_generation_checkpoint(
            candidate_path,
            candidate,
            generation=generation,
            parent_checkpoint=current_parent_path,
            provenance={
                "collection_seed_start": COLLECTION_SEEDS[generation],
                "eligible_games": collection.eligible_games,
                "train_games": collection.train_games,
                "validation_games": collection.validation_games,
                "league": [label for label, _factory in league],
                "train_config": asdict(cfg),
            },
        )
        validation_steps = [step for step in steps if step.split == "validation"]
        reload_exact = verify_checkpoint_reload(
            candidate_path,
            candidate,
            validation_steps,
            cfg,
        )
        update = replace(update, checkpoint_reload_exact=reload_exact)
        parent_gate = run_parent_gate(
            candidate,
            current_parent,
            seed_start=PARENT_GATE_SEEDS[generation],
            pairs=20,
            config=cfg,
        )
        gates = _generation_gates(
            collection,
            update,
            parent_gate,
            opponent_count=len(league),
            config=cfg,
            model=candidate,
        )
        accepted = all(gates.values())
        generation_payload = {
            "generation": generation,
            "parent_checkpoint": str(current_parent_path).replace("\\", "/"),
            "candidate_checkpoint": str(candidate_path).replace("\\", "/"),
            "candidate_sha256": sha256(candidate_path),
            "league": [label for label, _factory in league],
            "collection": asdict(collection),
            "games": [game.summary() for game in games],
            "update": asdict(update),
            "parent_gate": parent_gate,
            "gates": gates,
            "accepted": accepted,
        }
        generation_payloads.append(generation_payload)
        _atomic_json(
            output_dir / f"generation_{generation}.json",
            generation_payload,
        )
        if not accepted:
            stopped_generation = generation
            break
        accepted_models.append(candidate)
        accepted_paths.append(candidate_path)
        current_parent = candidate
        current_parent_path = candidate_path
        final_validation_steps = validation_steps

    final_population = None
    final_cumulative = None
    final_gates: dict[str, bool] = {}
    if len(accepted_models) == 3 and final_validation_steps is not None:
        final_population = run_static_population_gate(
            accepted_models[-1],
            epoch0,
            static,
            seed_starts=STATIC_GATE_SEEDS,
            pairs=10,
            config=cfg,
        )
        final_cumulative = policy_diagnostics(
            epoch0,
            accepted_models[-1],
            final_validation_steps,
            cfg,
        )
        deltas = [
            float(payload["delta"])
            for payload in final_population["opponents"].values()
        ]
        final_gates = {
            "population_all_valid": bool(final_population["valid"])
            and int(final_population["games"]) == 160,
            "aggregate_delta_positive": float(
                final_population["aggregate_delta"]
            )
            > 0.0,
            "two_opponents_nonnegative": sum(delta >= 0.0 for delta in deltas)
            >= 2,
            "no_opponent_collapse": min(deltas) >= -0.10,
            "cumulative_kl_band": 0.0005
            <= final_cumulative["raw_kl"]
            <= 0.025,
            "cumulative_argmax_band": 0.01
            <= final_cumulative["argmax_movement"]
            <= 0.25,
        }

    passed = len(accepted_models) == 3 and all(final_gates.values())
    decision = (
        "recurrent_stage_b_iterates_authorize_stage_c_scale_preregistration"
        if passed
        else "recurrent_stage_b_generation_failed_diagnose_before_scale"
    )
    payload = {
        "experiment": "062",
        "protocol": "docs/experiments/recurrent_league_stage_b_062.md",
        "prerequisites": prerequisites,
        "config": asdict(cfg),
        "generation_count_attempted": len(generation_payloads),
        "generation_count_accepted": len(accepted_models),
        "stopped_generation": stopped_generation,
        "generations": generation_payloads,
        "final_population_gate": final_population,
        "final_cumulative_movement": final_cumulative,
        "final_gates": final_gates,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": decision,
        "passed": passed,
    }
    _atomic_json(output_dir / "result.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/recurrent_league_stage_b_062"),
    )
    parser.add_argument(
        "--result-061",
        type=Path,
        default=Path("outputs/recurrent_league_stage_a_061/result.json"),
    )
    parser.add_argument(
        "--epoch0",
        type=Path,
        default=Path(
            "outputs/recurrent_league_stage_a_061/recurrent_policy_epoch0.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-035",
        type=Path,
        default=Path(
            "outputs/legal_action_scorer_035_legal_observation/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-016",
        type=Path,
        default=Path(
            "outputs/legal_action_scorer_016_policy_guided_mix/"
            "legal_action_scorer_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint-023",
        type=Path,
        default=Path(
            "outputs/legal_action_scorer_023_attention_pgs_distill/"
            "legal_action_attention_scorer_best.pt"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        output_dir=args.output_dir,
        result_061=args.result_061,
        epoch0_path=args.epoch0,
        checkpoint_035=args.checkpoint_035,
        checkpoint_016=args.checkpoint_016,
        checkpoint_023=args.checkpoint_023,
    )
    print(
        json.dumps(
            {
                "accepted_generations": result.get(
                    "generation_count_accepted",
                    0,
                ),
                "decision": result["decision"],
                "elapsed_seconds": result.get("elapsed_seconds"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
