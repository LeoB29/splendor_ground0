from __future__ import annotations

from splendor_ai.alphazero.endpoint_evaluation import (
    endpoint_decision,
    practical_assessment,
)


def _arena(
    *, seat_zero_wins: int, seat_one_wins: int, margin: float = 1.0,
    completed: int = 200,
) -> dict[str, object]:
    rows = []
    for seat, wins in ((0, seat_zero_wins), (1, seat_one_wins)):
        for index in range(100):
            rows.append({
                "seed": 1_800_000 + index,
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


def test_practical_assessment_accepts_exact_boundaries() -> None:
    result = practical_assessment(_arena(seat_zero_wins=60, seat_one_wins=50))
    assert result["passed"] is True
    assert result["category"] == "practical_strength_pass"
    assert result["seat_score_shares"] == {"0": 0.6, "1": 0.5}


def test_practical_assessment_separates_flat_and_mixed_results() -> None:
    flat = practical_assessment(_arena(seat_zero_wins=52, seat_one_wins=50))
    assert flat["passed"] is False
    assert flat["category"] == "practically_flat"
    seat_failure = practical_assessment(_arena(seat_zero_wins=70, seat_one_wins=40))
    assert seat_failure["category"] == "failed_or_mixed"
    abnormal = practical_assessment(
        _arena(seat_zero_wins=60, seat_one_wins=50, completed=199)
    )
    assert abnormal["passed"] is False


def test_endpoint_decision_classifies_mechanism() -> None:
    passed = {"passed": True, "category": "practical_strength_pass"}
    failed = {"passed": False, "category": "failed_or_mixed"}
    flat = {"passed": False, "category": "practically_flat"}
    assert endpoint_decision(passed, passed, passed).endswith("healthy_search")
    assert "compatibility" in endpoint_decision(failed, passed, passed)
    assert "search_health" in endpoint_decision(failed, failed, failed)
    assert "practically_flat" in endpoint_decision(flat, passed, passed)

