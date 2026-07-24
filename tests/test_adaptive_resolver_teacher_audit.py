from __future__ import annotations

from collections import Counter

import pytest

from splendor_ai.eval.run_adaptive_resolver_teacher_audit import (
    LABELS,
    NEW_LABELS,
    NEW_NAMESPACE_INDICES,
    AdaptiveResolverTeacherConfig,
    _equivalence_gates,
    _jensen_shannon,
    _point_gates,
    _resolver_config,
    _validate_config,
    adaptive_replay,
    select_adaptive_states,
)


def _estimates(rows: list[list[float]]) -> dict[str, dict[str, object]]:
    return {
        label: {
            "completed": True,
            "action_values": values,
            "best_index": values.index(max(values)),
        }
        for label, values in zip(LABELS[:8], rows, strict=True)
    }


def test_adaptive_rule_stops_early_for_clear_stable_best() -> None:
    cfg = AdaptiveResolverTeacherConfig()
    estimates = _estimates(
        [
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
            [0.50, 0.00, -0.10, -0.20],
        ]
    )

    replay = adaptive_replay(estimates, LABELS[:8], cfg)

    assert replay["best_index"] == 0
    assert replay["active_set"] == [0]
    assert replay["stop_reason"] == "stable_confidence_set"
    assert replay["sample_counts"] == [6, 2, 2, 2]
    assert replay["root_units"] == 12
    assert replay["continuations"] == 576


def test_adaptive_rule_returns_equivalence_set_and_prefix_counts() -> None:
    cfg = AdaptiveResolverTeacherConfig()
    estimates = _estimates([[0.1, 0.1, 0.1, 0.1] for _ in range(8)])

    replay = adaptive_replay(estimates, LABELS[:8], cfg)

    assert replay["active_set"] == [0, 1, 2, 3]
    assert replay["sample_counts"] == [3, 3, 3, 3]
    assert replay["root_units"] == 12
    assert replay["diagnostic_distribution"] == pytest.approx([0.25] * 4)
    assert len(replay["allocation_order"]) == replay["root_units"] - 8


def test_jsd_is_zero_for_identity_and_one_for_disjoint_points() -> None:
    assert _jensen_shannon([0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]) == 0.0
    assert _jensen_shannon([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_fresh_namespaces_are_disjoint_from_056_indices() -> None:
    cfg = AdaptiveResolverTeacherConfig()
    assert tuple(NEW_NAMESPACE_INDICES) == NEW_LABELS
    assert set(NEW_NAMESPACE_INDICES.values()) == set(range(17, 31))
    for label, index in NEW_NAMESPACE_INDICES.items():
        resolver = _resolver_config(cfg, label)
        assert resolver.belief_root_index_start == 10_000_000 * index + 1_000_000
        assert resolver.continuation_seed_start == 1_000_000_000_000_000 * index


def test_057_selects_balanced_nonoverlapping_remainder() -> None:
    cfg = AdaptiveResolverTeacherConfig()
    _validate_config(cfg)
    selected, frozen = select_adaptive_states(cfg)

    assert len(selected) == 36
    assert len({row["selection_cell"] for row in selected}) == 18
    assert set(Counter(row["selection_cell"] for row in selected).values()) == {2}
    assert len({row["game_seed"] for row in selected}) >= 15
    assert all(
        frozen[int(row["sample_index"])]["replicates"]["A"]["completed"]
        and frozen[int(row["sample_index"])]["replicates"]["B"]["completed"]
        for row in selected
    )


def _summary(
    *,
    adaptive_exact: float = 0.75,
    adaptive_equivalence: float = 0.82,
    compute_ratio: float = 0.80,
) -> dict[str, object]:
    breakdowns = {
        "phase": {
            phase: {
                "exact_best_agreement": adaptive_exact,
                "value_equivalent_agreement": adaptive_equivalence,
            }
            for phase in ("early", "mid", "late")
        },
        "seat": {
            str(seat): {
                "exact_best_agreement": adaptive_exact,
                "value_equivalent_agreement": adaptive_equivalence,
            }
            for seat in (0, 1)
        },
    }
    adaptive = {
        "exact_best_agreement": adaptive_exact,
        "flattened_value_pearson": 0.90,
        "mean_within_state_spearman": 0.70,
        "mean_symmetric_cross_regret": 0.02,
        "p90_symmetric_cross_regret": 0.05,
        "value_equivalent_agreement": adaptive_equivalence,
        "breakdowns": breakdowns,
        "bootstrap": {
            "exact_best_agreement": {"lower": 0.62, "upper": 0.85},
            "value_equivalent_agreement": {"lower": 0.68, "upper": 0.92},
        },
        "mean_set_jaccard": 0.75,
        "mean_distribution_jsd": 0.10,
        "mean_compute_ratio_vs_uniform_8x": compute_ratio,
    }
    uniform = {
        "exact_best_agreement": 0.76,
        "value_equivalent_agreement": 0.83,
        "mean_symmetric_cross_regret": 0.018,
    }
    return {"adaptive": adaptive, "uniform_8x": uniform}


def test_gates_separate_point_and_equivalence_targets() -> None:
    cfg = AdaptiveResolverTeacherConfig()
    summary = _summary()
    assert all(_point_gates(cfg, summary).values())
    assert all(_equivalence_gates(cfg, summary).values())

    summary["adaptive"]["exact_best_agreement"] = 0.60
    assert not all(_point_gates(cfg, summary).values())
    assert all(_equivalence_gates(cfg, summary).values())

    summary["adaptive"]["mean_compute_ratio_vs_uniform_8x"] = 0.95
    assert not all(_equivalence_gates(cfg, summary).values())
