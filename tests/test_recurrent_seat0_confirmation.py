from __future__ import annotations

from splendor_ai.training.run_recurrent_seat0_confirmation import (
    OPPONENTS,
    _decision,
    _margin,
    _point,
    _strength_gates,
)


def _interval(mean: float, lower: float, upper: float) -> dict[str, float]:
    return {"mean": mean, "lower": lower, "upper": upper}


def _passing_summary() -> dict:
    return {
        "aggregate": {
            "score_share_delta": _interval(0.01, -0.05, 0.07),
            "margin_delta": _interval(0.2, -1.0, 1.4),
        },
        "by_opponent": {
            opponent: {
                "score_share_delta": _interval(-0.05, -0.15, 0.05),
            }
            for opponent in OPPONENTS
        },
        "split_halves": {
            "even": {"score_share_delta": -0.05},
            "odd": {"score_share_delta": 0.07},
        },
    }


def test_outcome_helpers_use_seat_zero_perspective() -> None:
    assert _point({"winner": 0}) == 1.0
    assert _point({"winner": 1}) == 0.0
    assert _point({"winner": None}) == 0.5
    assert _margin({"final_scores": [14, 11]}) == 3.0


def test_strength_gates_accept_preregistered_boundaries() -> None:
    gates = _strength_gates(_passing_summary())

    assert all(gates.values())


def test_strength_gates_reject_unproven_aggregate_noninferiority() -> None:
    summary = _passing_summary()
    summary["aggregate"]["score_share_delta"]["lower"] = -0.05001

    gates = _strength_gates(summary)

    assert gates["aggregate_score_mean_noninferior"]
    assert not gates["aggregate_score_ci_proves_noninferiority"]


def test_strength_gates_reject_systematic_opponent_collapse() -> None:
    summary = _passing_summary()
    summary["by_opponent"]["035"]["score_share_delta"] = _interval(
        -0.06,
        -0.09,
        -0.051,
    )

    gates = _strength_gates(summary)

    assert not gates["no_systematic_opponent_collapse"]


def test_decision_prioritizes_integrity_then_pass_then_systematic_failure() -> None:
    summary = _passing_summary()
    strength = _strength_gates(summary)

    assert _decision({"hashes": False}, strength, summary) == (
        "recurrent_seat0_confirmation_invalid_no_strength_decision"
    )
    assert _decision({"all": True}, strength, summary) == (
        "recurrent_seat0_confirmation_pass_authorize_stage_c_preregistration"
    )

    strength["aggregate_score_ci_proves_noninferiority"] = False
    summary["aggregate"]["score_share_delta"]["upper"] = -0.051
    assert _decision({"all": True}, strength, summary) == (
        "recurrent_seat0_confirmation_systematic_failure_change_direction"
    )


def test_decision_returns_bounded_failure_when_evidence_is_inconclusive() -> None:
    summary = _passing_summary()
    strength = _strength_gates(summary)
    strength["both_split_halves_noninferior"] = False

    assert _decision({"all": True}, strength, summary) == (
        "recurrent_seat0_confirmation_failed_no_scale"
    )
