from __future__ import annotations

from splendor_ai.training.run_recurrent_seat_heterogeneity_audit import (
    OPPONENTS,
    _bootstrap,
    _decision,
    _flags,
)


def _rows():
    rows = []
    for panel in ("065", "066"):
        for index, opponent in enumerate(OPPONENTS):
            rows.append(
                {
                    "panel": panel,
                    "opponent": opponent,
                    "seat0_score_delta": -0.1 + 0.01 * index,
                    "seat1_score_delta": 0.1 - 0.01 * index,
                    "seat0_margin_delta": -1.0,
                    "seat1_margin_delta": 1.0,
                }
            )
    return rows


def test_cluster_bootstrap_is_deterministic_and_keeps_all_strata() -> None:
    first = _bootstrap(_rows(), samples=500, seed=67)
    second = _bootstrap(_rows(), samples=500, seed=67)

    assert first == second
    assert first["strata"] == 8
    assert first["clusters"] == 8
    assert first["pooled"]["seat0_score_delta"]["mean"] < 0.0
    assert first["pooled"]["seat1_score_delta"]["mean"] > 0.0


def _inference():
    interval = lambda mean, lower, upper: {
        "mean": mean,
        "lower": lower,
        "upper": upper,
    }
    return {
        "pooled": {
            "seat0_score_delta": interval(-0.03, -0.10, 0.04),
            "seat1_score_delta": interval(0.06, -0.01, 0.13),
            "score_interaction_seat1_minus_seat0": interval(
                0.09,
                -0.01,
                0.19,
            ),
        },
        "panels": {
            "065": {
                "seat0_score_delta": interval(0.10, -0.10, 0.30),
                "seat1_score_delta": interval(-0.12, -0.32, 0.08),
            },
            "066": {
                "seat0_score_delta": interval(-0.06, -0.14, 0.02),
                "seat1_score_delta": interval(0.10, 0.02, 0.18),
            },
        },
        "panel_difference_066_minus_065": {
            "seat0_score_delta": interval(-0.16, -0.40, 0.08),
            "seat1_score_delta": interval(0.22, -0.02, 0.46),
        },
    }


def test_flags_recognize_nonreplicated_seat_signs() -> None:
    flags = _flags(_inference())

    assert flags["seat_signs_reverse_across_panels"]
    assert flags["pooled_both_seats_within_floor"]
    assert not flags["seat0_systematically_below_floor"]


def test_decision_authorizes_only_targeted_confirmation_for_reversal() -> None:
    flags = _flags(_inference())

    assert _decision({"all": True}, flags) == (
        "recurrent_seat_failure_not_replicated_"
        "authorize_targeted_confirmation_preregistration"
    )


def test_decision_prioritizes_systematic_seat_failure() -> None:
    inference = _inference()
    inference["pooled"]["seat0_score_delta"] = {
        "mean": -0.10,
        "lower": -0.15,
        "upper": -0.06,
    }
    flags = _flags(inference)

    assert _decision({"all": True}, flags) == (
        "recurrent_seat_failure_systematic_change_direction"
    )


def test_decision_rejects_invalid_inputs_before_interpretation() -> None:
    assert _decision({"hash": False}, _flags(_inference())) == (
        "recurrent_seat_audit_invalid_no_inference"
    )
