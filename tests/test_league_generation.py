from __future__ import annotations

from splendor_ai.training.run_league_generation import (
    OPPONENTS,
    _group_assignment,
    evaluate_direct_gate,
)


def test_group_assignment_is_exactly_opponent_and_seat_balanced() -> None:
    cells = [_group_assignment(index) for index in range(64)]

    assert set(cells) == {(opponent, seat) for opponent in OPPONENTS for seat in (0, 1)}
    assert set({cell: cells.count(cell) for cell in set(cells)}.values()) == {8}


def test_direct_gate_requires_integrity_mean_and_confidence() -> None:
    payload = {
        "timed_out_games": 0,
        "stalled_games": 0,
        "paired_statistics": {
            "score_share": {"mean": 0.57, "ci95_low": 0.51, "ci95_high": 0.63}
        },
    }
    assert all(evaluate_direct_gate(payload).values())

    payload["paired_statistics"]["score_share"]["ci95_low"] = 0.50
    assert not all(evaluate_direct_gate(payload).values())
