from __future__ import annotations

from splendor_ai.alphazero.clean_continuation_recovery import arena_progress_valid


def _payload(*, name: str, rows: int, completed: bool) -> dict:
    seed_start = 1_780_000 if "raw_child_vs_raw_parent" in name else 1_780_100
    schedule = [
        (seed_start + offset, seat) for offset in range(8) for seat in (0, 1)
    ]
    return {
        "identity": {
            "name": name,
            "seed_start": seed_start,
            "paired_seeds": 8,
            "max_turns": 200,
        },
        "target_games": 16,
        "completed_games": rows,
        "completed": completed,
        "rows": [
            {"seed": seed, "candidate_seat": seat}
            for seed, seat in schedule[:rows]
        ],
    }


def test_recovery_accepts_complete_raw_and_exact_searched_prefix() -> None:
    assert arena_progress_valid(
        _payload(
            name="generation_005_raw_child_vs_raw_parent",
            rows=16,
            completed=True,
        ),
        expected_name="generation_005_raw_child_vs_raw_parent",
        expected_rows=16,
        expected_completed=True,
    )
    assert arena_progress_valid(
        _payload(
            name="generation_005_searched_child_vs_searched_parent",
            rows=3,
            completed=False,
        ),
        expected_name="generation_005_searched_child_vs_searched_parent",
        expected_rows=3,
        expected_completed=False,
    )


def test_recovery_rejects_nonprefix_or_wrong_completion_state() -> None:
    payload = _payload(
        name="generation_005_searched_child_vs_searched_parent",
        rows=3,
        completed=False,
    )
    payload["rows"][2]["candidate_seat"] = 1
    assert not arena_progress_valid(
        payload,
        expected_name="generation_005_searched_child_vs_searched_parent",
        expected_rows=3,
        expected_completed=False,
    )
    payload = _payload(
        name="generation_005_raw_child_vs_raw_parent", rows=16, completed=False
    )
    assert not arena_progress_valid(
        payload,
        expected_name="generation_005_raw_child_vs_raw_parent",
        expected_rows=16,
        expected_completed=True,
    )
