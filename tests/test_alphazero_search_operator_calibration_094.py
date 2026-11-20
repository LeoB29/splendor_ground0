from __future__ import annotations

import pytest

from splendor_ai.alphazero.search_operator_calibration import (
    SEARCH_GRID,
    confirmation_acceptance,
    discovery_eligibility,
    select_search_configuration,
    summarize_value_predictions,
)


def _arena(
    *, seat_zero_points: float, seat_one_points: float, margin: float = 1.0
) -> dict[str, object]:
    rows = []
    for seat, points in ((0, seat_zero_points), (1, seat_one_points)):
        wins = int(points)
        draw = points - wins >= 0.5
        for index in range(10):
            point = 1.0 if index < wins else (0.5 if index == wins and draw else 0.0)
            rows.append({"seed": 100 + index, "candidate_seat": seat, "candidate_points": point})
    return {
        "games": 20,
        "rows": rows,
        "score_share": (seat_zero_points + seat_one_points) / 20.0,
        "mean_margin": margin,
        "termination_reasons": {"completed": 20},
    }


def test_grid_is_exact_bounded_three_by_three() -> None:
    assert [(row.simulations, row.c_puct) for row in SEARCH_GRID] == [
        (simulations, c_puct)
        for simulations in (64, 128, 256)
        for c_puct in (0.75, 1.5, 2.25)
    ]


def test_value_summary_reports_error_and_probability_calibration() -> None:
    summary = summarize_value_predictions([(1.0, 1.0), (-0.5, -1.0)])
    assert summary["count"] == 2
    assert summary["mae"] == pytest.approx(0.25)
    assert summary["mse"] == pytest.approx(0.125)
    assert summary["sign_accuracy"] == 1.0
    assert summary["brier_score"] == pytest.approx(0.03125)
    assert sum(row["count"] for row in summary["calibration_bins"]) == 2


def test_discovery_requires_share_margin_seats_and_normal_completion() -> None:
    eligible = discovery_eligibility(_arena(seat_zero_points=6, seat_one_points=5))
    assert eligible["eligible"] is True
    collapsed = discovery_eligibility(_arena(seat_zero_points=10, seat_one_points=1))
    assert collapsed["eligible"] is False
    assert collapsed["checks"]["seat_one_at_least_45pct"] is False


def test_selection_uses_frozen_tiebreaks() -> None:
    rows = [
        {
            "order": 0,
            "spec": {"name": "large", "simulations": 256, "c_puct": 1.5},
            "arena": {"score_share": 0.6, "mean_margin": 1.0},
            "eligibility": {"eligible": True},
        },
        {
            "order": 1,
            "spec": {"name": "small", "simulations": 64, "c_puct": 2.25},
            "arena": {"score_share": 0.6, "mean_margin": 1.0},
            "eligibility": {"eligible": True},
        },
    ]
    assert select_search_configuration(rows)["spec"]["name"] == "small"


def test_confirmation_requires_strict_ci_and_both_seats() -> None:
    arena = _arena(seat_zero_points=6, seat_one_points=5)
    assert confirmation_acceptance(arena, {"ci95_low": 0.500001})["accepted"] is True
    boundary = confirmation_acceptance(arena, {"ci95_low": 0.5})
    assert boundary["accepted"] is False
    assert boundary["checks"]["bootstrap_ci95_low_above_half"] is False
