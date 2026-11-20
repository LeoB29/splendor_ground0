from __future__ import annotations

from types import SimpleNamespace

import pytest

from splendor_ai.alphazero.clean_continuation import partition_game_terminations


def _row(seed: int, turn: int, reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        game_seed=seed,
        turn_index=turn,
        termination_reason=reason,
    )


def test_partition_retains_only_normally_completed_games() -> None:
    clean, abnormal = partition_game_terminations(
        [
            _row(10, 0, "completed"),
            _row(10, 1, "completed"),
            _row(11, 0, "forced_pass_deadlock"),
            _row(11, 1, "forced_pass_deadlock"),
            _row(12, 0, "completed"),
        ]
    )
    assert clean == [10, 12]
    assert abnormal == {11: "forced_pass_deadlock"}


def test_partition_rejects_inconsistent_game_termination() -> None:
    with pytest.raises(ValueError, match="inconsistent termination"):
        partition_game_terminations(
            [_row(10, 0, "completed"), _row(10, 1, "max_turns")]
        )
