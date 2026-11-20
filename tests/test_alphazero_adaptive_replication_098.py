from __future__ import annotations

from splendor_ai.alphazero.adaptive_replication import adaptive_replication_acceptance


def _arena(
    *, seat_zero_wins: int, seat_one_wins: int, margin: float = 1.0,
    completed: int = 200,
) -> dict[str, object]:
    rows = []
    for seat, wins in ((0, seat_zero_wins), (1, seat_one_wins)):
        for index in range(100):
            rows.append({
                "seed": 1_690_000 + index,
                "candidate_seat": seat,
                "candidate_points": 1.0 if index < wins else 0.0,
            })
    return {
        "games": 200,
        "rows": rows,
        "score_share": (seat_zero_wins + seat_one_wins) / 200,
        "mean_margin": margin,
        "termination_reasons": {"completed": completed},
    }


def test_adaptive_replication_accepts_exact_boundaries() -> None:
    result = adaptive_replication_acceptance(
        _arena(seat_zero_wins=60, seat_one_wins=50), {"ci95_low": 0.500001}
    )
    assert result["accepted"] is True
    assert result["seat_score_shares"] == {"0": 0.6, "1": 0.5}


def test_adaptive_replication_requires_strict_ci_and_both_seats() -> None:
    result = adaptive_replication_acceptance(
        _arena(seat_zero_wins=70, seat_one_wins=40), {"ci95_low": 0.5}
    )
    assert result["accepted"] is False
    assert result["checks"]["bootstrap_ci95_low_above_half"] is False
    assert result["checks"]["seat_one_nonlosing"] is False


def test_adaptive_replication_rejects_margin_and_abnormal_termination() -> None:
    result = adaptive_replication_acceptance(
        _arena(seat_zero_wins=60, seat_one_wins=60, margin=0.0, completed=199),
        {"ci95_low": 0.51},
    )
    assert result["accepted"] is False
    assert result["checks"]["positive_margin"] is False
    assert result["checks"]["normal_terminations"] is False
