from __future__ import annotations

from pathlib import Path

from splendor_ai.alphazero.scaled_continuation import final_decision, rolling_window


def test_rolling_window_uses_previous_and_current_clean_corpora() -> None:
    previous = Path("generation_004.jsonl")
    current = Path("generation_005.jsonl")
    assert rolling_window(previous, current) == [previous, current]


def test_rolling_window_rejects_duplicate_corpus() -> None:
    replay = Path("same.jsonl")
    try:
        rolling_window(replay, replay)
    except ValueError as error:
        assert "distinct" in str(error)
    else:
        raise AssertionError("Duplicate replay paths were accepted.")


def test_final_decision_keeps_035_conditional_and_never_implies_install() -> None:
    assert final_decision(
        internal_primary_passed=False, benchmark_passed=None
    ) == "generation_8_endpoint_not_compounded_035_skipped"
    assert final_decision(
        internal_primary_passed=True, benchmark_passed=False
    ) == "generation_8_compounded_035_benchmark_complete_no_install"
    assert final_decision(
        internal_primary_passed=True, benchmark_passed=True
    ).endswith("review_no_install")

