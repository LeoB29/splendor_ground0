from __future__ import annotations

import pytest

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_selective_rollout_gate import (
    _fresh_breakdowns,
    _read_development_rows_without_confirmation,
    distinct_canonical_scores,
    threshold_metrics,
)
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore


def test_threshold_metrics_uses_discovery_half_only() -> None:
    rows = [
        {
            "candidate_minus_parent_first": 0.10,
            "candidate_minus_parent_second": 0.20,
        },
        {
            "candidate_minus_parent_first": 0.01,
            "candidate_minus_parent_second": -0.30,
        },
    ]
    metrics = threshold_metrics(rows, 0.05, 100, 7, symmetric=False)
    assert metrics["switches"] == 1
    assert metrics["switch_fraction"] == 0.5
    assert metrics["evaluation_gain"]["mean"] == 0.1
    assert metrics["evaluation_positive_fraction"] == 1.0


def test_threshold_metrics_can_symmetrize_development_halves() -> None:
    rows = [
        {
            "candidate_minus_parent_first": 0.10,
            "candidate_minus_parent_second": 0.20,
        },
        {
            "candidate_minus_parent_first": -0.10,
            "candidate_minus_parent_second": -0.20,
        },
    ]
    metrics = threshold_metrics(rows, 0.05, 100, 11, symmetric=True)
    assert metrics["orientations_per_row"] == 2
    assert metrics["switches"] == 2
    assert metrics["switch_fraction"] == 0.5
    assert metrics["evaluation_gain"]["mean"] == pytest.approx(0.075)


def test_distinct_canonical_scores_keeps_best_representative() -> None:
    env = SplendorEnv(seed=3)
    state = env.initial_state()
    action = env.legal_actions(state)[0]
    rows = (
        ModelActionScore(action=action, action_index=10, score=2.0),
        ModelActionScore(action=action, action_index=10, score=1.0),
    )
    assert distinct_canonical_scores(state, rows) == (rows[0],)


def test_fresh_breakdowns_reports_only_triggered_evaluation_gain() -> None:
    rows = [
        {
            "phase": "early",
            "opponent": "035",
            "actor_seat": 0,
            "candidate_minus_parent_first": 0.10,
            "candidate_minus_parent_second": 0.25,
        },
        {
            "phase": "early",
            "opponent": "035",
            "actor_seat": 0,
            "candidate_minus_parent_first": 0.01,
            "candidate_minus_parent_second": -0.50,
        },
    ]
    breakdowns = _fresh_breakdowns(rows, 0.05)
    assert breakdowns["phase"]["early"]["switches"] == 1
    assert breakdowns["phase"]["early"]["evaluation_gain_mean"] == 0.25


def test_development_reader_skips_confirmation_before_json_parsing(tmp_path) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_text(
        '{"row": {"split": "train", "value": 1}}\n'
        '{"row": {"split": "confirmation", BROKEN}}\n'
        '{"row": {"split": "validation", "value": 2}}\n',
        encoding="utf-8",
    )
    rows = _read_development_rows_without_confirmation(path)
    assert [row["value"] for row in rows] == [1, 2]
