from __future__ import annotations

import pytest

from splendor_ai.training.run_entity_coherent_supervised import (
    EXPECTED_TEACHER,
    validate_coherent_source_summary,
)


def _summary(teacher: str = EXPECTED_TEACHER) -> dict[str, object]:
    return {
        "stalled_games": 0,
        "timed_out_games": 0,
        "loop_fallback_triggers": 0,
        "metadata": {
            "contract": "canonical_public_belief_v1",
            "teacher": teacher,
        },
    }


def test_coherent_source_contract_accepts_clean_shallow_selfplay() -> None:
    validate_coherent_source_summary(_summary())


def test_coherent_source_contract_rejects_mixed_teacher() -> None:
    with pytest.raises(ValueError, match="shallow self-play"):
        validate_coherent_source_summary(_summary("Greedy_vs_Shallow"))


def test_coherent_source_contract_rejects_failed_games() -> None:
    summary = _summary()
    summary["timed_out_games"] = 1

    with pytest.raises(ValueError, match="failed or fallback"):
        validate_coherent_source_summary(summary)
