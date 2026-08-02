"""Run the fresh seat-0-only recurrent confirmation (068)."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from splendor_ai.eval.match import play_game
from splendor_ai.training.recurrent_league import (
    InMemoryRecurrentBot,
    RecurrentLeagueConfig,
    VALID_TERMINATIONS,
    sha256,
    static_opponent_factories,
)
from splendor_ai.training.recurrent_policy import load_recurrent_policy_checkpoint
from splendor_ai.training.run_recurrent_broad_failure_diagnosis import (
    _bootstrap_mean,
    _sign_flip_test,
    _stratified_bootstrap,
)


RESULT_067_SHA256 = (
    "03bb2d7386ab71d93302d02f4da266859c7e472b7facefeaa94f81c02c264662"
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
OPPONENTS = ("035", "016", "023", "shallow")
SEED_STARTS = {
    "035": 680_000,
    "016": 680_200,
    "023": 680_400,
    "shallow": 680_600,
}
SEEDS_PER_OPPONENT = 100
SCORE_FLOOR = -0.05


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _point(result: dict[str, Any]) -> float:
    winner = result["winner"]
    return 0.5 if winner is None else 1.0 if int(winner) == 0 else 0.0


def _margin(result: dict[str, Any]) -> float:
    return float(int(result["final_scores"][0]) - int(result["final_scores"][1]))


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


def _run_panel(
    candidate,
    control,
    opponents: list[tuple[str, Callable[[], Any]]],
) -> dict[str, Any]:
    config = RecurrentLeagueConfig()
    factories = dict(opponents)
    rows = []
    valid = True
    for opponent in OPPONENTS:
        for ordinal in range(SEEDS_PER_OPPONENT):
            seed = SEED_STARTS[opponent] + ordinal
            candidate_result = play_game(
                InMemoryRecurrentBot(candidate),
                factories[opponent](),
                seed=seed,
                max_turns=config.max_turns,
                repetition_limit=config.repetition_limit,
                no_progress_limit=config.no_progress_limit,
            )
            control_result = play_game(
                InMemoryRecurrentBot(control),
                factories[opponent](),
                seed=seed,
                max_turns=config.max_turns,
                repetition_limit=config.repetition_limit,
                no_progress_limit=config.no_progress_limit,
            )
            candidate_payload = asdict(candidate_result)
            control_payload = asdict(control_result)
            valid &= (
                candidate_result.termination_reason in VALID_TERMINATIONS
                and control_result.termination_reason in VALID_TERMINATIONS
            )
            candidate_point = _point(candidate_payload)
            control_point = _point(control_payload)
            candidate_margin = _margin(candidate_payload)
            control_margin = _margin(control_payload)
            rows.append(
                {
                    "opponent": opponent,
                    "seed": seed,
                    "seed_ordinal": ordinal,
                    "candidate": candidate_payload,
                    "control": control_payload,
                    "candidate_point": candidate_point,
                    "control_point": control_point,
                    "score_share_delta": candidate_point - control_point,
                    "candidate_margin": candidate_margin,
                    "control_margin": control_margin,
                    "margin_delta": candidate_margin - control_margin,
                }
            )
    return {"valid": valid, "games": 2 * len(rows), "rows": rows}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _summarize(
    panel: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = panel["rows"]
    by_opponent = {
        opponent: [row for row in rows if row["opponent"] == opponent]
        for opponent in OPPONENTS
    }
    opponent_summary = {}
    for index, opponent in enumerate(OPPONENTS):
        chosen = by_opponent[opponent]
        opponent_summary[opponent] = {
            "clusters": len(chosen),
            "candidate_score_share": _mean(chosen, "candidate_point"),
            "control_score_share": _mean(chosen, "control_point"),
            "score_share_delta": _bootstrap_mean(
                [float(row["score_share_delta"]) for row in chosen],
                samples=50_000,
                seed=680_100 + index,
            ),
            "margin_delta": _bootstrap_mean(
                [float(row["margin_delta"]) for row in chosen],
                samples=50_000,
                seed=680_100 + index,
            ),
        }
    score_strata = {
        opponent: [float(row["score_share_delta"]) for row in chosen]
        for opponent, chosen in by_opponent.items()
    }
    margin_strata = {
        opponent: [float(row["margin_delta"]) for row in chosen]
        for opponent, chosen in by_opponent.items()
    }
    aggregate = {
        "clusters": len(rows),
        "candidate_score_share": _mean(rows, "candidate_point"),
        "control_score_share": _mean(rows, "control_point"),
        "score_share_delta": _stratified_bootstrap(
            score_strata,
            samples=100_000,
            seed=680_068,
        ),
        "margin_delta": _stratified_bootstrap(
            margin_strata,
            samples=100_000,
            seed=680_068,
        ),
        "score_sign_flip": _sign_flip_test(
            [float(row["score_share_delta"]) for row in rows],
            samples=100_000,
            seed=680_168,
        ),
        "margin_sign_flip": _sign_flip_test(
            [float(row["margin_delta"]) for row in rows],
            samples=100_000,
            seed=680_169,
        ),
    }
    halves = {}
    for label, parity in (("even", 0), ("odd", 1)):
        chosen = [row for row in rows if int(row["seed_ordinal"]) % 2 == parity]
        halves[label] = {
            "clusters": len(chosen),
            "score_share_delta": _mean(chosen, "score_share_delta"),
            "margin_delta": _mean(chosen, "margin_delta"),
        }
    leave_one_opponent = {}
    for opponent in OPPONENTS:
        chosen = [row for row in rows if row["opponent"] != opponent]
        leave_one_opponent[opponent] = {
            "score_share_delta": _mean(chosen, "score_share_delta"),
            "margin_delta": _mean(chosen, "margin_delta"),
        }
    leave_one_score = []
    leave_one_margin = []
    for removed in rows:
        chosen = [row for row in rows if row is not removed]
        leave_one_score.append(_mean(chosen, "score_share_delta"))
        leave_one_margin.append(_mean(chosen, "margin_delta"))
    summary = {
        "aggregate": aggregate,
        "by_opponent": opponent_summary,
        "split_halves": halves,
        "leave_one_opponent_out": leave_one_opponent,
        "leave_one_seed_out": {
            "score_share_delta_min": min(leave_one_score),
            "score_share_delta_max": max(leave_one_score),
            "margin_delta_min": min(leave_one_margin),
            "margin_delta_max": max(leave_one_margin),
        },
    }
    population = {
        "candidate_score_share": aggregate["candidate_score_share"],
        "control_score_share": aggregate["control_score_share"],
        "score_share_delta": aggregate["score_share_delta"]["mean"],
        "opponents": {
            opponent: {
                "candidate_score_share": values["candidate_score_share"],
                "control_score_share": values["control_score_share"],
                "score_share_delta": values["score_share_delta"]["mean"],
            }
            for opponent, values in opponent_summary.items()
        },
    }
    return summary, population


def _integrity_gates(
    panel: dict[str, Any],
    summary: dict[str, Any],
    population: dict[str, Any],
    *,
    hashes_unchanged: bool,
) -> dict[str, bool]:
    rows = panel["rows"]
    exact_counts = len(rows) == 400 and all(
        sum(row["opponent"] == opponent for row in rows) == SEEDS_PER_OPPONENT
        for opponent in OPPONENTS
    )
    exact_seeds = exact_counts and all(
        [
            int(row["seed"])
            for row in rows
            if row["opponent"] == opponent
        ]
        == list(range(SEED_STARTS[opponent], SEED_STARTS[opponent] + 100))
        for opponent in OPPONENTS
    )
    endings = [row[label] for row in rows for label in ("candidate", "control")]
    recomputed = (
        math.isclose(
            _mean(rows, "candidate_point"),
            float(population["candidate_score_share"]),
            abs_tol=1e-12,
        )
        and math.isclose(
            _mean(rows, "control_point"),
            float(population["control_score_share"]),
            abs_tol=1e-12,
        )
        and math.isclose(
            _mean(rows, "score_share_delta"),
            float(summary["aggregate"]["score_share_delta"]["mean"]),
            abs_tol=1e-12,
        )
    )
    return {
        "exact_four_opponents": set(row["opponent"] for row in rows)
        == set(OPPONENTS),
        "exact_100_clusters_per_opponent": exact_counts,
        "exact_fresh_seed_bands": exact_seeds,
        "exact_800_games": panel["games"] == 800 and len(endings) == 800,
        "recurrent_always_seat0": all(
            result["bot_seats"][0] == "InMemoryRecurrentBot"
            for result in endings
        ),
        "all_endings_valid": panel["valid"]
        and all(
            result["termination_reason"] in VALID_TERMINATIONS
            for result in endings
        ),
        "zero_illegal_actions": all(
            result["termination_reason"] != "illegal_action" for result in endings
        ),
        "finite_metrics": _finite(summary) and _finite(population),
        "metrics_recomputed_exact": recomputed,
        "checkpoint_hashes_unchanged": hashes_unchanged,
        "no_training_selection_or_adaptation": True,
    }


def _strength_gates(
    summary: dict[str, Any],
) -> dict[str, bool]:
    aggregate_score = summary["aggregate"]["score_share_delta"]
    aggregate_margin = summary["aggregate"]["margin_delta"]
    opponents = summary["by_opponent"]
    opponent_means = [
        float(opponents[opponent]["score_share_delta"]["mean"])
        for opponent in OPPONENTS
    ]
    halves = [
        float(summary["split_halves"][half]["score_share_delta"])
        for half in ("even", "odd")
    ]
    return {
        "aggregate_score_mean_noninferior": float(aggregate_score["mean"])
        >= SCORE_FLOOR - 1e-12,
        "aggregate_score_ci_proves_noninferiority": float(aggregate_score["lower"])
        >= SCORE_FLOOR - 1e-12,
        "aggregate_margin_nonnegative": float(aggregate_margin["mean"]) >= 0.0,
        "aggregate_margin_ci_noninferior": float(aggregate_margin["lower"])
        >= -1.0,
        "three_opponents_noninferior": sum(
            value >= SCORE_FLOOR - 1e-12 for value in opponent_means
        ) >= 3,
        "no_opponent_below_minus_010": min(opponent_means) >= -0.10 - 1e-12,
        "no_systematic_opponent_collapse": all(
            float(opponents[opponent]["score_share_delta"]["upper"])
            >= SCORE_FLOOR - 1e-12
            for opponent in OPPONENTS
        ),
        "both_split_halves_noninferior": min(halves) >= SCORE_FLOOR - 1e-12,
    }


def _decision(
    integrity: dict[str, bool],
    strength: dict[str, bool],
    summary: dict[str, Any],
) -> str:
    if not all(integrity.values()):
        return "recurrent_seat0_confirmation_invalid_no_strength_decision"
    if all(strength.values()):
        return "recurrent_seat0_confirmation_pass_authorize_stage_c_preregistration"
    score_upper = float(summary["aggregate"]["score_share_delta"]["upper"])
    margin_upper = float(summary["aggregate"]["margin_delta"]["upper"])
    if score_upper < SCORE_FLOOR or margin_upper < -1.0:
        return "recurrent_seat0_confirmation_systematic_failure_change_direction"
    return "recurrent_seat0_confirmation_failed_no_scale"


def run(
    *,
    output_path: Path,
    result_067_path: Path,
    epoch0_path: Path,
    generation2_path: Path,
    checkpoint_035: Path,
    checkpoint_016: Path,
    checkpoint_023: Path,
) -> dict[str, Any]:
    expected = {
        "result_067": (result_067_path, RESULT_067_SHA256),
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
            "experiment": "068",
            "prerequisites": prerequisites,
            "decision": "recurrent_seat0_confirmation_invalid_no_strength_decision",
        }
        _atomic_json(output_path, payload)
        return payload

    result_067 = json.loads(result_067_path.read_text(encoding="utf-8"))
    authorization = {
        "decision_exact": result_067.get("decision")
        == (
            "recurrent_seat_failure_not_replicated_"
            "authorize_targeted_confirmation_preregistration"
        ),
        "all_integrity_gates": all(result_067["integrity"].values()),
        "zero_games_in_audit": result_067["configuration"]["games"] == 0,
    }
    if not all(authorization.values()):
        payload = {
            "experiment": "068",
            "prerequisites": prerequisites,
            "authorization": authorization,
            "decision": "recurrent_seat0_confirmation_invalid_no_strength_decision",
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
    panel = _run_panel(candidate, control, opponents)
    summary, population = _summarize(panel)
    hashes_unchanged = all(
        sha256(path) == digest
        for label, (path, digest) in expected.items()
        if label != "result_067"
    )
    integrity = _integrity_gates(
        panel,
        summary,
        population,
        hashes_unchanged=hashes_unchanged,
    )
    strength = _strength_gates(summary)
    decision = _decision(integrity, strength, summary)
    payload = {
        "experiment": "068",
        "protocol": "docs/experiments/recurrent_seat0_confirmation_068.md",
        "prerequisites": prerequisites,
        "authorization": authorization,
        "configuration": {
            "seed_starts": SEED_STARTS,
            "seeds_per_opponent": SEEDS_PER_OPPONENT,
            "clusters": 400,
            "games": 800,
            "recurrent_seat": 0,
            "training": False,
        },
        "panel": panel,
        "population": population,
        "paired_inference": summary,
        "integrity_gates": integrity,
        "strength_gates": strength,
        "strong_positive_evidence": float(
            summary["aggregate"]["score_share_delta"]["lower"]
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
        default=Path("outputs/recurrent_seat0_confirmation_068/result.json"),
    )
    parser.add_argument(
        "--result-067",
        type=Path,
        default=Path("outputs/recurrent_seat_heterogeneity_audit_067/result.json"),
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
        result_067_path=args.result_067,
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
                "score_share_delta": result.get("population", {}).get(
                    "score_share_delta"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
