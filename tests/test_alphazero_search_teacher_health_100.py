from __future__ import annotations

from splendor_ai.alphazero.search_teacher_health import (
    search_teacher_acceptance,
    value_calibration,
)


def _arena(
    *, seat_zero_wins: int, seat_one_wins: int, completed: int = 100,
    margin: float = 1.0,
) -> dict[str, object]:
    rows = []
    for seat, wins in ((0, seat_zero_wins), (1, seat_one_wins)):
        for index in range(50):
            rows.append({
                "seed": 1_790_000 + index,
                "candidate_seat": seat,
                "candidate_points": 1.0 if index < wins else 0.0,
            })
    return {
        "games": 100,
        "rows": rows,
        "score_share": (seat_zero_wins + seat_one_wins) / 100,
        "mean_margin": margin,
        "termination_reasons": {"completed": completed},
    }


def test_search_teacher_accepts_exact_frozen_boundaries() -> None:
    result = search_teacher_acceptance(_arena(seat_zero_wins=30, seat_one_wins=25))
    assert result["accepted"] is True
    assert result["seat_score_shares"] == {"0": 0.6, "1": 0.5}


def test_search_teacher_requires_both_seats_margin_and_normal_games() -> None:
    seat_failure = search_teacher_acceptance(
        _arena(seat_zero_wins=40, seat_one_wins=15)
    )
    assert seat_failure["accepted"] is False
    assert seat_failure["checks"]["seat_one_nonlosing"] is False
    abnormal = search_teacher_acceptance(
        _arena(seat_zero_wins=30, seat_one_wins=30, completed=99, margin=0.0)
    )
    assert abnormal["accepted"] is False
    assert abnormal["checks"]["positive_margin"] is False
    assert abnormal["checks"]["normal_terminations"] is False


def test_value_calibration_reports_raw_and_search_errors() -> None:
    result = value_calibration(
        [(0.5, 1.0, 1.0), (-0.5, -1.0, -1.0), (0.2, 0.1, 0.0)]
    )
    assert result["rows"] == 3
    assert result["raw_network"]["decisive_sign_accuracy"] == 1.0
    assert result["search_root"]["mae"] < result["raw_network"]["mae"]
    assert value_calibration([]) == {"rows": 0}
