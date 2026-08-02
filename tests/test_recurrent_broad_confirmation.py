from __future__ import annotations

from splendor_ai.training.run_recurrent_broad_confirmation import (
    OPPONENTS,
    _decision,
    _finite,
    _population_recomputed_exact,
    _strength_gates,
)


def _population(delta: float = 0.03):
    return {
        "aggregate_delta": delta,
        "aggregate_candidate_score_share": 0.53,
        "aggregate_control_score_share": 0.50,
        "opponents": {
            opponent: {
                "candidate_score_share": 0.53,
                "control_score_share": 0.50,
                "delta": delta,
            }
            for opponent in OPPONENTS
        },
    }


def _uncertainty(
    *,
    score_mean: float = 0.03,
    score_lower: float = -0.02,
    score_upper: float = 0.08,
    margin_mean: float = 0.5,
    margin_lower: float = -0.5,
    margin_upper: float = 1.5,
):
    return {
        "aggregate": {
            "score_share_delta": {
                "mean": score_mean,
                "lower": score_lower,
                "upper": score_upper,
            },
            "margin_delta": {
                "mean": margin_mean,
                "lower": margin_lower,
                "upper": margin_upper,
            },
        },
        "by_opponent": {
            opponent: {"score_share_delta": {"mean": score_mean}}
            for opponent in OPPONENTS
        },
        "by_seat": {
            "seat0": {"score_share_delta": 0.02},
            "seat1": {"score_share_delta": 0.04},
        },
        "split_halves": {
            "even": {"score_share_delta": 0.01},
            "odd": {"score_share_delta": 0.05},
        },
    }


def test_strength_gates_accept_broad_bounded_improvement() -> None:
    gates = _strength_gates(_population(), _uncertainty())

    assert all(gates.values())


def test_strength_gates_reject_hidden_opponent_collapse() -> None:
    population = _population()
    population["opponents"]["016"]["delta"] = -0.06

    gates = _strength_gates(population, _uncertainty())

    assert not gates["no_opponent_below_minus_005"]


def test_decision_prioritizes_integrity_then_systematic_failure() -> None:
    integrity = {"hashes": False}
    strength = {"aggregate": False}
    uncertainty = _uncertainty(score_upper=-0.01)

    assert _decision(integrity, strength, uncertainty) == (
        "recurrent_broad_confirmation_invalid_no_strength_decision"
    )
    integrity["hashes"] = True
    assert _decision(integrity, strength, uncertainty) == (
        "recurrent_broad_confirmation_systematic_failure_change_direction"
    )


def test_decision_pass_requires_every_strength_gate() -> None:
    integrity = {"all": True}
    strength = {"a": True, "b": True}

    assert _decision(integrity, strength, _uncertainty()) == (
        "recurrent_broad_confirmation_pass_authorize_stage_c_preregistration"
    )
    strength["b"] = False
    assert _decision(integrity, strength, _uncertainty()) == (
        "recurrent_broad_confirmation_failed_no_scale"
    )


def test_population_recomputation_checks_candidate_control_and_delta() -> None:
    records = [
        {
            "opponent": opponent,
            "candidate_score_share": 0.53,
            "control_score_share": 0.50,
        }
        for opponent in OPPONENTS
    ]

    assert _population_recomputed_exact(
        _population(),
        _uncertainty(),
        records,
    )
    records[0]["candidate_score_share"] = 0.52
    assert not _population_recomputed_exact(
        _population(),
        _uncertainty(),
        records,
    )


def test_finite_rejects_nested_nonfinite_metric() -> None:
    assert _finite({"a": [1.0, 2.0], "label": "ok"})
    assert not _finite({"a": [1.0, float("nan")]})
