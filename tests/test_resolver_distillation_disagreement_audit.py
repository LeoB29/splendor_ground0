from __future__ import annotations

import pytest

from splendor_ai.training.run_resolver_distillation_disagreement_audit import (
    _cluster_bootstrap,
    _fallback_report,
    _pair_summary,
    _parent_gap_bin,
    _resolver_value_maps,
)


def _pair(
    *,
    seed: int,
    parent_gap: float,
    student_gap: float,
    captured: bool,
    phase: str = "mid",
) -> dict[str, object]:
    return {
        "game_seed": seed,
        "turn_index": 20,
        "phase": phase,
        "seat": 0,
        "outcome": "winner",
        "preferred_action_type": "BUY_VISIBLE",
        "parent_gap": parent_gap,
        "student_gap": student_gap,
        "gap_delta": student_gap - parent_gap,
        "directional_movement": student_gap > parent_gap,
        "crossed": parent_gap <= 0.0 and student_gap > 0.0,
        "captured": captured,
        "retained_parent": not captured,
        "moved_third": False,
        "parent_rank": 2,
        "student_rank": 1 if captured else 2,
        "target_probability_advantage": 0.2,
        "parent_gap_bin": _parent_gap_bin(parent_gap),
        "resolver_value_advantage": 0.1,
    }


def test_pair_summary_reports_direction_crossing_and_capture() -> None:
    rows = [
        _pair(seed=1, parent_gap=-0.2, student_gap=0.1, captured=True),
        _pair(seed=2, parent_gap=-0.4, student_gap=-0.3, captured=False),
        _pair(seed=3, parent_gap=-0.1, student_gap=-0.2, captured=False),
    ]

    summary = _pair_summary(rows)

    assert summary["rows"] == 3
    assert summary["directional_movement"] == pytest.approx(2 / 3)
    assert summary["crossed"] == pytest.approx(1 / 3)
    assert summary["preferred_action_capture"] == pytest.approx(1 / 3)
    assert summary["gap_delta"]["mean"] == pytest.approx(0.1)


def test_cluster_bootstrap_is_game_clustered_and_deterministic() -> None:
    rows = [
        _pair(seed=1, parent_gap=-0.2, student_gap=0.2, captured=True),
        _pair(seed=1, parent_gap=-0.3, student_gap=0.1, captured=True),
        _pair(seed=2, parent_gap=-0.2, student_gap=-0.3, captured=False),
    ]

    first = _cluster_bootstrap(rows, samples=200, seed=52)
    second = _cluster_bootstrap(rows, samples=200, seed=52)

    assert first == second
    assert set(first) == {
        "mean_gap_delta",
        "directional_movement",
        "preferred_action_capture",
    }
    assert first["directional_movement"]["lower"] <= first["directional_movement"]["upper"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-2.0, "(-inf,-1]"),
        (-1.0, "(-inf,-1]"),
        (-0.75, "(-1,-0.5]"),
        (-0.3, "(-0.5,-0.25]"),
        (-0.2, "(-0.25,-0.1]"),
        (-0.08, "(-0.1,-0.05]"),
        (-0.01, "(-0.05,0]"),
    ],
)
def test_parent_gap_bins_are_frozen(value: float, expected: str) -> None:
    assert _parent_gap_bin(value) == expected


def test_resolver_value_maps_identify_aliases() -> None:
    groups, values, aligned = _resolver_value_maps(
        [
            {"action_indices": [10, 11], "value": 0.25},
            {"action_indices": [12], "value": -0.5},
        ]
    )

    assert aligned
    assert groups[10] == groups[11]
    assert groups[10] != groups[12]
    assert values == {10: 0.25, 11: 0.25, 12: -0.5}


def test_fallback_report_preserves_phase_and_seat_breakdowns() -> None:
    rows = [
        {
            "fallback_kl": 0.1,
            "fallback_moved": False,
            "fallback_entropy_delta": 0.2,
            "phase": "early",
            "seat": 0,
        },
        {
            "fallback_kl": 0.3,
            "fallback_moved": True,
            "fallback_entropy_delta": -0.2,
            "phase": "mid",
            "seat": 1,
        },
    ]

    report = _fallback_report(rows)

    assert report["mean_kl"] == pytest.approx(0.2)
    assert report["action_movement"] == pytest.approx(0.5)
    assert report["phase"]["late"]["rows"] == 0
    assert report["seat"]["0"]["rows"] == 1
