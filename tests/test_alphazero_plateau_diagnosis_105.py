from __future__ import annotations

import pytest

from splendor_ai.alphazero.plateau_diagnosis import (
    SAMPLED_SEED_OFFSETS,
    TARGET_TURNS,
    pool_arena_rows,
    pooled_decision,
)


def _arena(seed_start: int, wins_by_seat: tuple[int, int]) -> dict[str, object]:
    rows = []
    for offset in range(100):
        for seat in (0, 1):
            won = offset < wins_by_seat[seat]
            rows.append(
                {
                    "seed": seed_start + offset,
                    "candidate_seat": seat,
                    "candidate_points": 1.0 if won else 0.0,
                    "candidate_margin": 1 if won else -1,
                    "termination_reason": "completed",
                }
            )
    return {"rows": rows}


def test_105_sample_schedule_is_fixed_and_one_state_per_game() -> None:
    assert len(SAMPLED_SEED_OFFSETS) == len(TARGET_TURNS) == 16
    assert list(SAMPLED_SEED_OFFSETS) == sorted(set(SAMPLED_SEED_OFFSETS))
    assert min(SAMPLED_SEED_OFFSETS) >= 0 and max(SAMPLED_SEED_OFFSETS) < 256


def test_pool_arena_rows_uses_both_complete_disjoint_panels() -> None:
    pooled = pool_arena_rows(_arena(10_000, (57, 52)), _arena(20_000, (60, 51)))
    assert pooled["games"] == 400
    assert pooled["paired_seeds"] == 200
    assert pooled["candidate_wins"] == 220
    assert pooled["score_share"] == 0.55
    assert pooled["termination_reasons"] == {"completed": 400}


def test_pool_arena_rows_rejects_seed_reuse() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        pool_arena_rows(_arena(10_000, (57, 52)), _arena(10_000, (60, 51)))


def test_pooled_decision_distinguishes_pass_flat_and_unhealthy() -> None:
    assert "confirmed" in pooled_decision(
        {"passed": True, "category": "practical_strength_pass"}
    )
    assert "practically_flat" in pooled_decision(
        {"passed": False, "category": "practically_flat"}
    )
    assert "unhealthy" in pooled_decision(
        {"passed": False, "category": "failed_or_mixed"}
    )

