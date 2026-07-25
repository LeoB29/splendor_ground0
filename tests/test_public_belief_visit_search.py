from __future__ import annotations

from collections import Counter

import pytest

from splendor_ai.eval.run_public_belief_visit_search import (
    Edge,
    VisitSearchConfig,
    _feasibility_gates,
    _validate_config,
    collapse_equivalence_group,
    jensen_shannon,
    score_priors,
    select_puct_edge,
    select_visit_states,
    total_variation,
    visit_belief_root_index,
    visit_continuation_seed,
)


def test_060_reuses_exact_059_balanced_selection() -> None:
    cfg = VisitSearchConfig()
    _validate_config(cfg)
    selected = select_visit_states(cfg)

    assert len(selected) == 48
    assert len({row["game_seed"] for row in selected}) == 17
    counts = Counter(
        f"{row['phase']}|seat{row['actor_seat']}|{row['gap_stratum']}"
        for row in selected
    )
    assert len(counts) == 18
    assert all(
        count == (4 if cell.startswith("early|") else 2)
        for cell, count in counts.items()
    )


def test_prior_softmax_is_mixed_normalized_and_ordered() -> None:
    priors = score_priors([3.0, 2.0, 1.0, 0.0], 1.0, 0.05)

    assert sum(priors) == pytest.approx(1.0)
    assert priors == sorted(priors, reverse=True)
    assert all(value >= 0.0125 for value in priors)


def test_puct_forces_unvisited_then_uses_root_and_opponent_values() -> None:
    edges = [Edge(prior=0.1), Edge(prior=0.6), Edge(prior=0.2), Edge(prior=0.1)]
    assert select_puct_edge(edges, 1.5, minimizing=False) == 1
    for edge in edges:
        edge.visits = 10
    edges[0].value_sum = 8.0
    edges[1].value_sum = 2.0
    edges[2].value_sum = 1.0
    edges[3].value_sum = 0.0

    assert select_puct_edge(edges, 0.0, minimizing=False) == 0
    assert select_puct_edge(edges, 0.0, minimizing=True) == 3


def test_distribution_metrics_and_equivalence_collapse() -> None:
    left = [0.5, 0.3, 0.1, 0.1]
    right = [0.3, 0.5, 0.1, 0.1]

    assert jensen_shannon(left, left) == 0.0
    assert total_variation(left, right) == pytest.approx(0.2)
    raw = jensen_shannon(left, right)
    collapsed_left, collapsed_right = collapse_equivalence_group(
        left, right, [0, 1]
    )
    assert collapsed_left == pytest.approx([0.8, 0.1, 0.1])
    assert collapsed_right == pytest.approx([0.8, 0.1, 0.1])
    assert jensen_shannon(collapsed_left, collapsed_right) == 0.0
    assert raw > 0.0


def test_visit_namespaces_are_disjoint_and_exact() -> None:
    cfg = VisitSearchConfig()
    left_root = visit_belief_root_index(cfg, 7, "left")
    right_root = visit_belief_root_index(cfg, 7, "right")
    left_seeds = {
        visit_continuation_seed(cfg, 7, "left", simulation)
        for simulation in range(192)
    }
    right_seeds = {
        visit_continuation_seed(cfg, 7, "right", simulation)
        for simulation in range(192)
    }

    assert right_root == left_root + 1
    assert len(left_seeds) == len(right_seeds) == 192
    assert left_seeds.isdisjoint(right_seeds)


def _gate_summary() -> dict[str, object]:
    phase = {
        name: {
            "exact_best_agreement": 0.70,
            "value_equivalent_agreement": 0.75,
            "mean_spearman": 0.65,
            "mean_symmetric_cross_regret": 0.02,
        }
        for name in ("early", "mid", "late")
    }
    distribution_phase = {
        name: {
            "mean_raw_jsd": 0.08,
            "mean_collapsed_jsd": 0.05,
        }
        for name in ("early", "mid", "late")
    }
    comparison_rows = [
        {"phase": "early", "symmetric_regret": 0.03}
        for _ in range(24)
    ]
    return {
        "structural": {
            "both_completed_fraction": 1.0,
            "phase": {
                name: {"both_completed_fraction": 1.0}
                for name in ("early", "mid", "late")
            },
        },
        "distribution": {
            "mean_raw_jsd": 0.08,
            "mean_collapsed_jsd": 0.05,
            "breakdowns": {"phase": distribution_phase},
        },
        "halfway": {"mean_halfway_to_final_jsd": 0.05},
        "breakdowns": {"phase": phase},
        "prior_movement": {
            "mean_prior_to_search_total_variation": 0.20,
            "state_any_top_change_fraction": 0.25,
            "mean_prior_to_search_jsd": 0.12,
        },
        "comparison_rows": comparison_rows,
    }


def test_feasibility_gates_require_value_stability_and_prior_movement() -> None:
    cfg = VisitSearchConfig()
    summary = _gate_summary()
    assert all(_feasibility_gates(cfg, summary).values())

    summary["prior_movement"]["mean_prior_to_search_total_variation"] = 0.05
    assert not all(_feasibility_gates(cfg, summary).values())
