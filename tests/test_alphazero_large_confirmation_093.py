from __future__ import annotations

from splendor_ai.alphazero.large_confirmation import large_confirmation_acceptance


def _arena(
    *,
    seat_zero_wins: int,
    seat_one_wins: int,
    margin: float = 1.0,
    completed: int = 100,
) -> dict[str, object]:
    rows = []
    for seat, wins in ((0, seat_zero_wins), (1, seat_one_wins)):
        for index in range(50):
            rows.append(
                {
                    "seed": 1_630_000 + index,
                    "candidate_seat": seat,
                    "candidate_points": 1.0 if index < wins else 0.0,
                }
            )
    total_wins = seat_zero_wins + seat_one_wins
    return {
        "games": 100,
        "rows": rows,
        "score_share": total_wins / 100,
        "mean_margin": margin,
        "termination_reasons": {"completed": completed},
    }


def test_large_confirmation_accepts_exact_frozen_boundaries() -> None:
    arena = _arena(seat_zero_wins=28, seat_one_wins=27)
    result = large_confirmation_acceptance(arena, {"ci95_low": 0.500001})
    assert result["accepted"] is True
    assert result["seat_score_shares"] == {"0": 0.56, "1": 0.54}


def test_large_confirmation_requires_strict_ci_and_both_seats() -> None:
    arena = _arena(seat_zero_wins=45, seat_one_wins=10)
    at_boundary = large_confirmation_acceptance(arena, {"ci95_low": 0.5})
    assert at_boundary["accepted"] is False
    assert at_boundary["checks"]["bootstrap_ci95_low_above_half"] is False
    assert at_boundary["checks"]["seat_one_nonlosing"] is False


def test_large_confirmation_rejects_abnormal_termination() -> None:
    arena = _arena(seat_zero_wins=30, seat_one_wins=30, completed=99)
    result = large_confirmation_acceptance(arena, {"ci95_low": 0.51})
    assert result["accepted"] is False
    assert result["checks"]["normal_terminations"] is False
