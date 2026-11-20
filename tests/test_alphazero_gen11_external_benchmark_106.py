from __future__ import annotations

import pytest

from splendor_ai.alphazero.external_benchmark import (
    external_decision,
    external_parity_assessment,
    independent_paired_difference_bootstrap,
)


def _arena(seed_start: int, seat_wins: tuple[int, int], margin: float = 1.0) -> dict[str, object]:
    rows = []
    for offset in range(100):
        for seat in (0, 1):
            win = offset < seat_wins[seat]
            rows.append(
                {
                    "seed": seed_start + offset,
                    "candidate_seat": seat,
                    "candidate_points": 1.0 if win else 0.0,
                    "candidate_margin": margin if win else -margin,
                    "termination_reason": "completed",
                }
            )
    return {
        "games": 200,
        "rows": rows,
        "score_share": sum(seat_wins) / 200,
        "mean_margin": margin * (2 * sum(seat_wins) - 200) / 200,
        "termination_reasons": {"completed": 200},
    }


def test_external_parity_requires_both_seats_and_positive_margin() -> None:
    assert external_parity_assessment(_arena(10_000, (55, 55)))["passed"]
    failed = external_parity_assessment(_arena(10_000, (70, 40)))
    assert not failed["passed"]
    assert not failed["checks"]["seat_one_nonlosing"]


def test_independent_difference_bootstrap_reports_exact_point_difference() -> None:
    result = independent_paired_difference_bootstrap(
        _arena(20_000, (60, 60)),
        _arena(10_000, (50, 50)),
        samples=1_000,
        seed=106,
    )
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["current_paired_seeds"] == result["anchor_paired_seeds"] == 100


def test_independent_difference_rejects_seed_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        independent_paired_difference_bootstrap(
            _arena(10_000, (60, 60)),
            _arena(10_000, (50, 50)),
            samples=10,
            seed=106,
        )


def test_external_decision_separates_parity_confirmed_and_directional() -> None:
    assert "parity" in external_decision({"passed": True}, {"ci95_low": -0.1, "mean_difference": 0.0})
    assert "confirmed" in external_decision({"passed": False}, {"ci95_low": 0.01, "mean_difference": 0.1})
    assert "directional" in external_decision({"passed": False}, {"ci95_low": -0.01, "mean_difference": 0.1})

