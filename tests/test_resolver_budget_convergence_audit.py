from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    ResolverGameplayConfig,
)
from splendor_ai.eval.run_resolver_budget_convergence_audit import (
    BUDGET_BLOCKS,
    LABELS,
    ResolverBudgetConvergenceConfig,
    _ensemble,
    _equivalence_gates,
    _metric_helper_config,
    _point_gates,
    _positive_convergence_gates,
    _resolver_config,
    _validate_config,
    select_budget_states,
)


def _completed(values: list[float]) -> dict[str, object]:
    return {
        "completed": True,
        "action_values": values,
        "best_index": values.index(max(values)),
    }


def test_budget_blocks_are_disjoint_and_use_every_estimate() -> None:
    assert tuple(BUDGET_BLOCKS) == (1, 2, 4, 8)
    for budget, comparisons in BUDGET_BLOCKS.items():
        assert len(comparisons) == 8 // budget
        used = []
        for left, right in comparisons:
            assert len(left) == budget
            assert len(right) == budget
            assert set(left).isdisjoint(right)
            used.extend(left)
            used.extend(right)
        assert sorted(used) == sorted(LABELS)


def test_ensemble_averages_aligned_values_and_requires_completion() -> None:
    estimates = {
        "A": _completed([0.2, 0.0, -0.1, -0.2]),
        "B": _completed([0.0, 0.2, -0.1, -0.2]),
    }
    ensemble = _ensemble(estimates, ("A", "B"))

    assert ensemble is not None
    assert ensemble["action_values"] == pytest.approx([0.1, 0.1, -0.1, -0.2])
    assert ensemble["best_index"] == 0
    estimates["B"]["completed"] = False
    assert _ensemble(estimates, ("A", "B")) is None


def test_sixteen_namespaces_change_only_random_starts() -> None:
    cfg = ResolverBudgetConvergenceConfig()
    base = ResolverGameplayConfig()
    seen = set()
    for index, label in enumerate(LABELS, start=1):
        resolver = _resolver_config(cfg, label)
        assert resolver.belief_root_index_start == (10_000_000 * index) + 1_000_000
        assert resolver.continuation_seed_start == 1_000_000_000_000_000 * index
        seen.add(
            (
                resolver.belief_root_index_start,
                resolver.continuation_seed_start,
            )
        )
        for field in base.__dataclass_fields__:
            if field in {"belief_root_index_start", "continuation_seed_start"}:
                continue
            assert getattr(resolver, field) == getattr(base, field)
    assert len(seen) == 16


def _budget_summary(
    *,
    exact: float,
    equivalence: float,
    spearman: float,
    regret: float,
    regret_ratio: float,
    exact_gain: float,
    equivalence_gain: float,
) -> dict[str, object]:
    breakdowns = {
        "phase": {
            phase: {
                "exact_best_agreement": exact,
                "value_equivalent_agreement": equivalence,
            }
            for phase in ("early", "mid", "late")
        },
        "seat": {
            str(seat): {
                "exact_best_agreement": exact,
                "value_equivalent_agreement": equivalence,
            }
            for seat in (0, 1)
        },
    }
    return {
        "exact_best_agreement": exact,
        "flattened_value_pearson": 0.90,
        "mean_within_state_spearman": spearman,
        "mean_symmetric_cross_regret": regret,
        "p90_symmetric_cross_regret": 0.05,
        "value_equivalent_agreement": equivalence,
        "directional_regret_at_most_005": 0.85,
        "eligible_state_comparisons": 36,
        "breakdowns": breakdowns,
        "bootstrap": {
            "exact_best_agreement": {"lower": 0.62, "upper": 0.85},
            "value_equivalent_agreement": {"lower": 0.68, "upper": 0.92},
            "mean_regret": {"lower": 0.01, "upper": 0.03},
        },
        "change_from_1x": {
            "exact_best_agreement": exact_gain,
            "mean_spearman": 0.18,
            "equivalence": equivalence_gain,
            "mean_regret_ratio": regret_ratio,
        },
    }


def test_fixed_decisions_distinguish_point_equivalence_and_positive_only() -> None:
    cfg = ResolverBudgetConvergenceConfig()
    one = {"eligible_state_comparisons": 288}
    eight = _budget_summary(
        exact=0.75,
        equivalence=0.82,
        spearman=0.70,
        regret=0.02,
        regret_ratio=0.50,
        exact_gain=0.20,
        equivalence_gain=0.20,
    )
    summary = {"budgets": {"1": one, "8": eight}}

    assert all(_point_gates(cfg, summary).values())
    assert all(_equivalence_gates(cfg, summary).values())
    assert all(_positive_convergence_gates(cfg, summary).values())

    eight["exact_best_agreement"] = 0.62
    assert not all(_point_gates(cfg, summary).values())
    assert all(_equivalence_gates(cfg, summary).values())
    eight["value_equivalent_agreement"] = 0.60
    assert not all(_equivalence_gates(cfg, summary).values())
    assert all(_positive_convergence_gates(cfg, summary).values())


def test_056_frozen_subset_reuses_balanced_completed_a_b() -> None:
    cfg = ResolverBudgetConvergenceConfig()
    _validate_config(cfg)
    selected, frozen = select_budget_states(cfg)

    assert len(selected) == 36
    assert len({row["selection_cell"] for row in selected}) == 18
    assert set(Counter(row["selection_cell"] for row in selected).values()) == {2}
    assert len({row["game_seed"] for row in selected}) >= 15
    assert all(
        frozen[int(row["sample_index"])]["replicates"]["A"]["completed"]
        and frozen[int(row["sample_index"])]["replicates"]["B"]["completed"]
        for row in selected
    )
    helper = _metric_helper_config(cfg, 8)
    assert isinstance(helper, SimpleNamespace)
    assert helper.bootstrap_seed == 560_064
