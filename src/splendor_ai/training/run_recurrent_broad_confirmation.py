"""Run the fresh 800-game recurrent broad confirmation (066)."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from splendor_ai.training.run_recurrent_broad_failure_diagnosis import (
    _paired_breakdown,
    _paired_seed_records,
)
from splendor_ai.training.recurrent_league import (
    RecurrentLeagueConfig,
    VALID_TERMINATIONS,
    run_static_population_gate,
    sha256,
    static_opponent_factories,
)
from splendor_ai.training.recurrent_policy import load_recurrent_policy_checkpoint


RESULT_065_SHA256 = (
    "616e0f6eb0ff7b968cfc0f9d60c51c2c186557c179c07a52586b978d784e0b4d"
)
EPOCH0_SHA256 = (
    "3dd945ba0c525e936603042040b2c33ab70ad8552e007b6c8f2dc4619813290a"
)
GENERATION2_SHA256 = (
    "3ed0bd7e2dcda21646c7ac4f68e2d0ffcfe0bdacf0c151c453c52ea6a08fe09e"
)
CHECKPOINT_035_SHA256 = (
    "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
)
CHECKPOINT_016_SHA256 = (
    "876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056"
)
CHECKPOINT_023_SHA256 = (
    "29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070"
)
SEED_STARTS = {
    "035": 660_000,
    "016": 660_100,
    "023": 660_200,
    "shallow": 660_300,
}
OPPONENTS = ("035", "016", "023", "shallow")
PAIRS_PER_OPPONENT = 50


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return False


def _population_recomputed_exact(
    population: dict[str, Any],
    uncertainty: dict[str, Any],
    paired_records: list[dict[str, Any]],
) -> bool:
    aggregate = uncertainty["aggregate"]["score_share_delta"]["mean"]
    if not math.isclose(
        float(aggregate),
        float(population["aggregate_delta"]),
        abs_tol=1e-12,
    ):
        return False
    aggregate_candidate = sum(
        float(row["candidate_score_share"]) for row in paired_records
    ) / len(paired_records)
    aggregate_control = sum(
        float(row["control_score_share"]) for row in paired_records
    ) / len(paired_records)
    if not math.isclose(
        aggregate_candidate,
        float(population["aggregate_candidate_score_share"]),
        abs_tol=1e-12,
    ) or not math.isclose(
        aggregate_control,
        float(population["aggregate_control_score_share"]),
        abs_tol=1e-12,
    ):
        return False
    for opponent in OPPONENTS:
        opponent_rows = [
            row for row in paired_records if row["opponent"] == opponent
        ]
        candidate = sum(
            float(row["candidate_score_share"]) for row in opponent_rows
        ) / len(opponent_rows)
        control = sum(
            float(row["control_score_share"]) for row in opponent_rows
        ) / len(opponent_rows)
        observed = uncertainty["by_opponent"][opponent]["score_share_delta"][
            "mean"
        ]
        expected = population["opponents"][opponent]["delta"]
        if (
            not math.isclose(float(observed), float(expected), abs_tol=1e-12)
            or not math.isclose(
                candidate,
                float(population["opponents"][opponent]["candidate_score_share"]),
                abs_tol=1e-12,
            )
            or not math.isclose(
                control,
                float(population["opponents"][opponent]["control_score_share"]),
                abs_tol=1e-12,
            )
        ):
            return False
    return True


def _integrity_gates(
    population: dict[str, Any],
    uncertainty: dict[str, Any],
    paired_records: list[dict[str, Any]],
    *,
    hashes_unchanged: bool,
) -> dict[str, bool]:
    exact_panel = set(population["opponents"]) == set(OPPONENTS)
    exact_rows = exact_panel and all(
        len(population["opponents"][opponent]["rows"]) == PAIRS_PER_OPPONENT
        for opponent in OPPONENTS
    )
    exact_seeds = exact_rows and all(
        [
            int(row["seed"])
            for row in population["opponents"][opponent]["rows"]
        ]
        == list(
            range(
                SEED_STARTS[opponent],
                SEED_STARTS[opponent] + PAIRS_PER_OPPONENT,
            )
        )
        for opponent in OPPONENTS
    )
    endings = [
        result
        for opponent in OPPONENTS
        for row in population["opponents"][opponent]["rows"]
        for label in ("candidate", "control")
        for result in row[label]
    ] if exact_rows else []
    return {
        "exact_opponents": exact_panel,
        "exact_rows_per_opponent": exact_rows,
        "exact_fresh_seed_bands": exact_seeds,
        "exact_800_games": population["games"] == 800 and len(endings) == 800,
        "all_endings_valid": bool(endings)
        and all(
            result["termination_reason"] in VALID_TERMINATIONS
            for result in endings
        )
        and bool(population["valid"]),
        "zero_illegal_actions": all(
            result["termination_reason"] != "illegal_action"
            for result in endings
        ),
        "finite_metrics": _finite(population) and _finite(uncertainty),
        "score_metrics_recomputed_exact": _population_recomputed_exact(
            population,
            uncertainty,
            paired_records,
        ),
        "checkpoint_hashes_unchanged": hashes_unchanged,
        "no_training_or_selection": True,
    }


def _strength_gates(
    population: dict[str, Any],
    uncertainty: dict[str, Any],
) -> dict[str, bool]:
    score_ci = uncertainty["aggregate"]["score_share_delta"]
    margin_ci = uncertainty["aggregate"]["margin_delta"]
    opponent_deltas = [
        float(population["opponents"][opponent]["delta"])
        for opponent in OPPONENTS
    ]
    seat_deltas = [
        float(uncertainty["by_seat"][f"seat{seat}"]["score_share_delta"])
        for seat in (0, 1)
    ]
    half_deltas = [
        float(uncertainty["split_halves"][half]["score_share_delta"])
        for half in ("even", "odd")
    ]
    return {
        "aggregate_score_positive": float(population["aggregate_delta"]) > 0.0,
        "aggregate_score_ci_noninferior": float(score_ci["lower"]) >= -0.05,
        "aggregate_margin_positive": float(margin_ci["mean"]) > 0.0,
        "aggregate_margin_ci_noninferior": float(margin_ci["lower"]) >= -1.0,
        "three_opponents_nonnegative": sum(
            delta >= -1e-12 for delta in opponent_deltas
        ) >= 3,
        "no_opponent_below_minus_005": min(opponent_deltas) >= -0.05 - 1e-12,
        "both_seats_noninferior": min(seat_deltas) >= -0.05 - 1e-12,
        "both_split_halves_noninferior": min(half_deltas) >= -0.05 - 1e-12,
    }


def _decision(
    integrity: dict[str, bool],
    strength: dict[str, bool],
    uncertainty: dict[str, Any],
) -> str:
    if not all(integrity.values()):
        return "recurrent_broad_confirmation_invalid_no_strength_decision"
    if all(strength.values()):
        return "recurrent_broad_confirmation_pass_authorize_stage_c_preregistration"
    score_upper = float(
        uncertainty["aggregate"]["score_share_delta"]["upper"]
    )
    margin_upper = float(uncertainty["aggregate"]["margin_delta"]["upper"])
    if score_upper < 0.0 or margin_upper < 0.0:
        return "recurrent_broad_confirmation_systematic_failure_change_direction"
    return "recurrent_broad_confirmation_failed_no_scale"


def run(
    *,
    output_path: Path,
    result_065_path: Path,
    epoch0_path: Path,
    generation2_path: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    expected = {
        "result_065": (result_065_path, RESULT_065_SHA256),
        "epoch0": (epoch0_path, EPOCH0_SHA256),
        "generation2": (generation2_path, GENERATION2_SHA256),
        "035": (checkpoint_035, CHECKPOINT_035_SHA256),
        "016": (checkpoint_016, CHECKPOINT_016_SHA256),
        "023": (checkpoint_023, CHECKPOINT_023_SHA256),
    }
    prerequisites = {}
    for label, (path, digest) in expected.items():
        actual = sha256(path)
        prerequisites[label] = {
            "path": str(path).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": digest,
            "passed": actual == digest,
        }
    if not all(row["passed"] for row in prerequisites.values()):
        payload = {
            "experiment": "066",
            "prerequisites": prerequisites,
            "decision": "recurrent_broad_confirmation_invalid_no_strength_decision",
        }
        _atomic_json(output_path, payload)
        return payload

    result_065 = json.loads(result_065_path.read_text(encoding="utf-8"))
    authorization = {
        "decision_exact": result_065.get("decision")
        == (
            "recurrent_broad_result_inconclusive_"
            "authorize_confirmation_preregistration"
        ),
        "replay_integrity_passed": bool(
            result_065.get("replay_integrity", {}).get("passed")
        ),
        "replay_games_exact": result_065.get("replay_integrity", {}).get("games")
        == 160,
    }
    if not all(authorization.values()):
        payload = {
            "experiment": "066",
            "prerequisites": prerequisites,
            "authorization": authorization,
            "decision": "recurrent_broad_confirmation_invalid_no_strength_decision",
        }
        _atomic_json(output_path, payload)
        return payload

    candidate = load_recurrent_policy_checkpoint(generation2_path).eval()
    control = load_recurrent_policy_checkpoint(epoch0_path).eval()
    opponents = static_opponent_factories(
        checkpoint_035,
        checkpoint_016,
        checkpoint_023,
    )
    population = run_static_population_gate(
        candidate,
        control,
        opponents,
        seed_starts=SEED_STARTS,
        pairs=PAIRS_PER_OPPONENT,
        config=RecurrentLeagueConfig(),
    )
    paired_records = _paired_seed_records({"final_population_gate": population})
    uncertainty = _paired_breakdown(
        paired_records,
        aggregate_bootstrap_seed=660_066,
        opponent_bootstrap_seed=660_100,
        score_sign_flip_seed=660_166,
        margin_sign_flip_seed=660_167,
    )
    hashes_unchanged = all(
        sha256(path) == digest
        for label, (path, digest) in expected.items()
        if label != "result_065"
    )
    integrity = _integrity_gates(
        population,
        uncertainty,
        paired_records,
        hashes_unchanged=hashes_unchanged,
    )
    strength = _strength_gates(population, uncertainty)
    decision = _decision(integrity, strength, uncertainty)
    payload = {
        "experiment": "066",
        "protocol": "docs/experiments/recurrent_broad_confirmation_066.md",
        "prerequisites": prerequisites,
        "authorization": authorization,
        "configuration": {
            "seed_starts": SEED_STARTS,
            "pairs_per_opponent": PAIRS_PER_OPPONENT,
            "games": 800,
            "training": False,
        },
        "population_gate": population,
        "paired_seed_records": paired_records,
        "paired_uncertainty": uncertainty,
        "integrity_gates": integrity,
        "strength_gates": strength,
        "strong_positive_evidence": float(
            uncertainty["aggregate"]["score_share_delta"]["lower"]
        ) > 0.0,
        "passed": all(integrity.values()) and all(strength.values()),
        "decision": decision,
    }
    _atomic_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/recurrent_broad_confirmation_066/result.json"),
    )
    parser.add_argument(
        "--result-065",
        type=Path,
        default=Path("outputs/recurrent_broad_failure_diagnosis_065/result.json"),
    )
    parser.add_argument(
        "--epoch0",
        type=Path,
        default=Path(
            "outputs/history_required_recurrent_repair_064/"
            "history_required_epoch0.pt"
        ),
    )
    parser.add_argument(
        "--generation2",
        type=Path,
        default=Path("outputs/history_required_recurrent_repair_064/generation_2.pt"),
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
            "outputs/canonical_entity_coherent_023/"
            "canonical_entity_action_scorer_coherent_width64_best.pt"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        output_path=args.output_path,
        result_065_path=args.result_065,
        epoch0_path=args.epoch0,
        generation2_path=args.generation2,
        checkpoint_035=args.checkpoint_035,
        checkpoint_016=args.checkpoint_016,
        checkpoint_023=args.checkpoint_023,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "passed": result.get("passed", False),
                "aggregate_delta": result.get("population_gate", {}).get(
                    "aggregate_delta"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
