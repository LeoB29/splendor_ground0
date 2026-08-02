"""Audit the 066 seat failure using disjoint 065/066 paired records."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from splendor_ai.training.recurrent_league import sha256


RESULT_065_SHA256 = (
    "616e0f6eb0ff7b968cfc0f9d60c51c2c186557c179c07a52586b978d784e0b4d"
)
RESULT_066_SHA256 = (
    "dbbdedbb74b59ea5d0d6562eced9c08afdd43795107e2e3567e9dddb60ff162c"
)
OPPONENTS = ("035", "016", "023", "shallow")
PANEL_COUNTS = {"065": 40, "066": 200}
PANEL_CELL_COUNTS = {"065": 10, "066": 50}
FLOOR = -0.05
METRICS = (
    "seat0_score_delta",
    "seat1_score_delta",
    "seat0_margin_delta",
    "seat1_margin_delta",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _percentile(sorted_values: list[float], probability: float) -> float:
    return sorted_values[
        min(int(probability * len(sorted_values)), len(sorted_values) - 1)
    ]


def _interval(
    observed: float,
    draws: list[float],
) -> dict[str, float | int]:
    draws.sort()
    return {
        "mean": observed,
        "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975),
        "samples": len(draws),
    }


def _load_rows(
    result_065: dict[str, Any],
    result_066: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for panel, result in (("065", result_065), ("066", result_066)):
        for source in result["paired_seed_records"]:
            row = dict(source)
            row["panel"] = panel
            rows.append(row)
    return rows


def _bootstrap(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(str(row["panel"]), str(row["opponent"]))].append(row)
    expected = {
        (panel, opponent)
        for panel in PANEL_COUNTS
        for opponent in OPPONENTS
    }
    if set(strata) != expected or any(not group for group in strata.values()):
        raise ValueError("Every panel/opponent bootstrap stratum is required")

    rng = random.Random(seed)
    draw_values: dict[str, list[float]] = defaultdict(list)
    ordered = sorted(strata)
    for _ in range(samples):
        sampled = []
        sampled_by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key in ordered:
            group = strata[key]
            chosen = [group[rng.randrange(len(group))] for _ in group]
            sampled.extend(chosen)
            sampled_by_panel[key[0]].extend(chosen)
        for metric in METRICS:
            draw_values[f"pooled:{metric}"].append(_mean(sampled, metric))
            for panel in PANEL_COUNTS:
                draw_values[f"{panel}:{metric}"].append(
                    _mean(sampled_by_panel[panel], metric)
                )
        draw_values["pooled:score_interaction"].append(
            _mean(sampled, "seat1_score_delta")
            - _mean(sampled, "seat0_score_delta")
        )
        draw_values["pooled:margin_interaction"].append(
            _mean(sampled, "seat1_margin_delta")
            - _mean(sampled, "seat0_margin_delta")
        )
        for metric in METRICS:
            draw_values[f"difference:{metric}"].append(
                _mean(sampled_by_panel["066"], metric)
                - _mean(sampled_by_panel["065"], metric)
            )

    observed_by_panel = {
        panel: [row for row in rows if row["panel"] == panel]
        for panel in PANEL_COUNTS
    }
    pooled = {
        metric: _interval(
            _mean(rows, metric),
            draw_values[f"pooled:{metric}"],
        )
        for metric in METRICS
    }
    pooled["score_interaction_seat1_minus_seat0"] = _interval(
        _mean(rows, "seat1_score_delta")
        - _mean(rows, "seat0_score_delta"),
        draw_values["pooled:score_interaction"],
    )
    pooled["margin_interaction_seat1_minus_seat0"] = _interval(
        _mean(rows, "seat1_margin_delta")
        - _mean(rows, "seat0_margin_delta"),
        draw_values["pooled:margin_interaction"],
    )
    panels = {
        panel: {
            metric: _interval(
                _mean(observed_by_panel[panel], metric),
                draw_values[f"{panel}:{metric}"],
            )
            for metric in METRICS
        }
        for panel in PANEL_COUNTS
    }
    difference = {
        metric: _interval(
            _mean(observed_by_panel["066"], metric)
            - _mean(observed_by_panel["065"], metric),
            draw_values[f"difference:{metric}"],
        )
        for metric in METRICS
    }
    return {
        "pooled": pooled,
        "panels": panels,
        "panel_difference_066_minus_065": difference,
        "strata": len(strata),
        "clusters": len(rows),
    }


def _opponent_bootstrap(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {}
    for index, opponent in enumerate(OPPONENTS):
        chosen = [row for row in rows if row["opponent"] == opponent]
        rng = random.Random(670_100 + index)
        draws0 = []
        draws1 = []
        interactions = []
        for _ in range(50_000):
            sample = [chosen[rng.randrange(len(chosen))] for _ in chosen]
            seat0 = _mean(sample, "seat0_score_delta")
            seat1 = _mean(sample, "seat1_score_delta")
            draws0.append(seat0)
            draws1.append(seat1)
            interactions.append(seat1 - seat0)
        observed0 = _mean(chosen, "seat0_score_delta")
        observed1 = _mean(chosen, "seat1_score_delta")
        payload[opponent] = {
            "clusters": len(chosen),
            "seat0_score_delta": _interval(observed0, draws0),
            "seat1_score_delta": _interval(observed1, draws1),
            "score_interaction_seat1_minus_seat0": _interval(
                observed1 - observed0,
                interactions,
            ),
        }
    return payload


def _panel_recomputed_exact(
    panel: str,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
) -> bool:
    population_key = (
        "recomputed_population" if panel == "065" else "population_gate"
    )
    population = result[population_key]
    score_delta = sum(
        float(row["seat0_score_delta"]) + float(row["seat1_score_delta"])
        for row in rows
    ) / (2 * len(rows))
    if not math.isclose(
        score_delta,
        float(population["aggregate_delta"]),
        abs_tol=1e-12,
    ):
        return False
    for opponent in OPPONENTS:
        opponent_rows = [row for row in rows if row["opponent"] == opponent]
        delta = sum(
            float(row["seat0_score_delta"])
            + float(row["seat1_score_delta"])
            for row in opponent_rows
        ) / (2 * len(opponent_rows))
        if not math.isclose(
            delta,
            float(population["opponents"][opponent]["delta"]),
            abs_tol=1e-12,
        ):
            return False
    stored = result["paired_uncertainty"]
    for seat in (0, 1):
        if not math.isclose(
            _mean(rows, f"seat{seat}_score_delta"),
            float(stored["by_seat"][f"seat{seat}"]["score_share_delta"]),
            abs_tol=1e-12,
        ) or not math.isclose(
            _mean(rows, f"seat{seat}_margin_delta"),
            float(stored["by_seat"][f"seat{seat}"]["margin_delta"]),
            abs_tol=1e-12,
        ):
            return False
    for half, parity in (("even", 0), ("odd", 1)):
        half_rows = [
            row for row in rows if int(row["seed_ordinal"]) % 2 == parity
        ]
        half_delta = sum(
            float(row["seat0_score_delta"])
            + float(row["seat1_score_delta"])
            for row in half_rows
        ) / (2 * len(half_rows))
        if not math.isclose(
            half_delta,
            float(stored["split_halves"][half]["score_share_delta"]),
            abs_tol=1e-12,
        ):
            return False
    return True


def _integrity(
    result_065: dict[str, Any],
    result_066: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, bool]:
    false_strength = sorted(
        name
        for name, passed in result_066["strength_gates"].items()
        if not passed
    )
    keys = [
        (str(row["panel"]), str(row["opponent"]), int(row["seed"]))
        for row in rows
    ]
    exact_counts = all(
        sum(row["panel"] == panel for row in rows) == count
        for panel, count in PANEL_COUNTS.items()
    ) and all(
        sum(
            row["panel"] == panel and row["opponent"] == opponent
            for row in rows
        )
        == PANEL_CELL_COUNTS[panel]
        for panel in PANEL_COUNTS
        for opponent in OPPONENTS
    )
    return {
        "065_decision_exact": result_065.get("decision")
        == (
            "recurrent_broad_result_inconclusive_"
            "authorize_confirmation_preregistration"
        ),
        "066_decision_exact": result_066.get("decision")
        == "recurrent_broad_confirmation_failed_no_scale",
        "065_replay_integrity": bool(
            result_065.get("replay_integrity", {}).get("passed")
        ),
        "066_all_integrity_gates": all(result_066["integrity_gates"].values()),
        "066_only_seat_gate_failed": false_strength
        == ["both_seats_noninferior"],
        "exact_panel_opponent_counts": exact_counts,
        "unique_panel_opponent_seeds": len(keys) == len(set(keys)),
        "065_metrics_recomputed_exact": _panel_recomputed_exact(
            "065",
            [row for row in rows if row["panel"] == "065"],
            result_065,
        ),
        "066_metrics_recomputed_exact": _panel_recomputed_exact(
            "066",
            [row for row in rows if row["panel"] == "066"],
            result_066,
        ),
        "no_games_training_or_model_loading": True,
    }


def _flags(inference: dict[str, Any]) -> dict[str, bool]:
    pooled = inference["pooled"]
    panels = inference["panels"]
    difference = inference["panel_difference_066_minus_065"]
    seat0 = pooled["seat0_score_delta"]
    seat1 = pooled["seat1_score_delta"]
    interaction = pooled["score_interaction_seat1_minus_seat0"]

    def excludes_zero(interval: dict[str, Any]) -> bool:
        return float(interval["lower"]) > 0.0 or float(interval["upper"]) < 0.0

    return {
        "seat0_mean_below_floor": float(seat0["mean"]) < FLOOR,
        "seat1_mean_below_floor": float(seat1["mean"]) < FLOOR,
        "seat0_systematically_below_floor": float(seat0["upper"]) < FLOOR,
        "seat1_systematically_below_floor": float(seat1["upper"]) < FLOOR,
        "seat0_proven_noninferior": float(seat0["lower"]) >= FLOOR,
        "seat1_proven_noninferior": float(seat1["lower"]) >= FLOOR,
        "score_seat_interaction_systematic": excludes_zero(interaction),
        "seat0_panel_difference_systematic": excludes_zero(
            difference["seat0_score_delta"]
        ),
        "seat1_panel_difference_systematic": excludes_zero(
            difference["seat1_score_delta"]
        ),
        "seat_signs_reverse_across_panels": (
            float(panels["065"]["seat0_score_delta"]["mean"])
            * float(panels["066"]["seat0_score_delta"]["mean"])
            < 0.0
            and float(panels["065"]["seat1_score_delta"]["mean"])
            * float(panels["066"]["seat1_score_delta"]["mean"])
            < 0.0
        ),
        "pooled_both_seats_within_floor": (
            float(seat0["mean"]) >= FLOOR and float(seat1["mean"]) >= FLOOR
        ),
    }


def _decision(integrity: dict[str, bool], flags: dict[str, bool]) -> str:
    if not all(integrity.values()):
        return "recurrent_seat_audit_invalid_no_inference"
    if (
        flags["seat0_systematically_below_floor"]
        or flags["seat1_systematically_below_floor"]
    ):
        return "recurrent_seat_failure_systematic_change_direction"
    if (
        flags["seat_signs_reverse_across_panels"]
        and flags["pooled_both_seats_within_floor"]
    ):
        return (
            "recurrent_seat_failure_not_replicated_"
            "authorize_targeted_confirmation_preregistration"
        )
    if (
        flags["seat0_mean_below_floor"]
        or flags["seat1_mean_below_floor"]
        or flags["seat0_panel_difference_systematic"]
        or flags["seat1_panel_difference_systematic"]
        or flags["score_seat_interaction_systematic"]
    ):
        return (
            "recurrent_seat_effect_heterogeneous_"
            "authorize_targeted_confirmation_preregistration"
        )
    return "recurrent_seat_audit_complete_no_scale_authorized"


def run(
    *,
    output_path: Path,
    result_065_path: Path,
    result_066_path: Path,
) -> dict[str, Any]:
    paths = {
        "result_065": (result_065_path, RESULT_065_SHA256),
        "result_066": (result_066_path, RESULT_066_SHA256),
    }
    prerequisites = {}
    for label, (path, expected) in paths.items():
        actual = sha256(path)
        prerequisites[label] = {
            "path": str(path).replace("\\", "/"),
            "sha256": actual,
            "expected_sha256": expected,
            "passed": actual == expected,
        }
    if not all(row["passed"] for row in prerequisites.values()):
        payload = {
            "experiment": "067",
            "prerequisites": prerequisites,
            "decision": "recurrent_seat_audit_invalid_no_inference",
        }
        _atomic_json(output_path, payload)
        return payload

    result_065 = json.loads(result_065_path.read_text(encoding="utf-8"))
    result_066 = json.loads(result_066_path.read_text(encoding="utf-8"))
    rows = _load_rows(result_065, result_066)
    integrity = _integrity(result_065, result_066, rows)
    if not all(integrity.values()):
        payload = {
            "experiment": "067",
            "protocol": "docs/experiments/recurrent_seat_heterogeneity_audit_067.md",
            "prerequisites": prerequisites,
            "integrity": integrity,
            "decision": "recurrent_seat_audit_invalid_no_inference",
        }
        _atomic_json(output_path, payload)
        return payload

    inference = _bootstrap(rows, samples=100_000, seed=670_067)
    opponents = _opponent_bootstrap(rows)
    flags = _flags(inference)
    decision = _decision(integrity, flags)
    payload = {
        "experiment": "067",
        "protocol": "docs/experiments/recurrent_seat_heterogeneity_audit_067.md",
        "prerequisites": prerequisites,
        "integrity": integrity,
        "configuration": {
            "panels": PANEL_COUNTS,
            "opponents": OPPONENTS,
            "clusters": len(rows),
            "bootstrap_samples": 100_000,
            "bootstrap_seed": 670_067,
            "practical_floor": FLOOR,
            "games": 0,
            "training": False,
        },
        "clustered_inference": inference,
        "opponent_by_seat": opponents,
        "interpretation_flags": flags,
        "decision": decision,
    }
    _atomic_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/recurrent_seat_heterogeneity_audit_067/result.json"),
    )
    parser.add_argument(
        "--result-065",
        type=Path,
        default=Path("outputs/recurrent_broad_failure_diagnosis_065/result.json"),
    )
    parser.add_argument(
        "--result-066",
        type=Path,
        default=Path("outputs/recurrent_broad_confirmation_066/result.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(
        output_path=args.output_path,
        result_065_path=args.result_065,
        result_066_path=args.result_066,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "clusters": result.get("configuration", {}).get("clusters", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
