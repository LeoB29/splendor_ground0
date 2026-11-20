from __future__ import annotations

import pytest

from splendor_ai.alphazero.continued_scaling import (
    final_decision_104,
    selfplay_seed_start,
)


def test_104_selfplay_seed_bands_are_fresh_and_fixed() -> None:
    assert [selfplay_seed_start(generation) for generation in (8, 9, 10)] == [
        1_880_000,
        1_890_000,
        1_900_000,
    ]
    with pytest.raises(ValueError, match="must be 8, 9, or 10"):
        selfplay_seed_start(11)


def test_104_final_decision_keeps_benchmark_conditional_and_no_install() -> None:
    assert final_decision_104(
        internal_primary_passed=False, benchmark_passed=None
    ) == "generation_11_endpoint_not_compounded_035_skipped"
    assert final_decision_104(
        internal_primary_passed=True, benchmark_passed=False
    ) == "generation_11_compounded_035_benchmark_complete_no_install"
    assert final_decision_104(
        internal_primary_passed=True, benchmark_passed=True
    ).endswith("review_no_install")

