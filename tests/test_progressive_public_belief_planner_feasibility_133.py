from __future__ import annotations

from collections import Counter

import pytest

from splendor_ai.eval.run_progressive_public_belief_planner_feasibility import (
    CELLS,
    ProgressivePlannerConfig,
    _average_ranks,
    _validate_config,
    belief_root_index,
    continuation_seed,
    feasibility_gates,
    select_states,
)


def test_133_frozen_inputs_and_balanced_selection_reproduce() -> None:
    cfg = ProgressivePlannerConfig()
    _validate_config(cfg)
    selected = select_states(cfg)

    assert len(selected) == 36
    assert Counter(row["selection_cell"] for row in selected) == Counter(
        {cell: 6 for cell in CELLS}
    )
    assert max(int(row["expanded_root_count"]) for row in selected) == 32
    assert all(
        int(row["expanded_root_count"])
        * cfg.determinizations
        * cfg.opponent_actions
        <= cfg.simulations
        for row in selected
    )


def test_133_seed_namespaces_are_disjoint() -> None:
    cfg = ProgressivePlannerConfig()
    assert belief_root_index(cfg, 7, "right") == belief_root_index(cfg, 7, "left") + 1
    left = {continuation_seed(cfg, 7, "left", index) for index in range(cfg.simulations)}
    right = {continuation_seed(cfg, 7, "right", index) for index in range(cfg.simulations)}
    other = {continuation_seed(cfg, 8, "left", index) for index in range(cfg.simulations)}
    assert len(left) == len(right) == len(other) == cfg.simulations
    assert left.isdisjoint(right)
    assert left.isdisjoint(other)


def test_average_ranks_handles_ties() -> None:
    assert _average_ranks([2.0, 1.0, 2.0, 4.0]) == [1.5, 0.0, 1.5, 3.0]


def _passing_summary() -> dict[str, object]:
    phase = {
        name: {
            "mean_raw_jsd": 0.08,
            "mean_collapsed_jsd": 0.05,
            "mean_gain": 0.04,
            "expanded_selection_fraction": 0.25,
        }
        for name in ("early", "mid", "late")
    }
    return {
        "exact_agreement": 0.70,
        "mean_raw_jsd": 0.08,
        "mean_collapsed_jsd": 0.05,
        "mean_cross_regret": 0.02,
        "p90_cross_regret": 0.06,
        "left_equivalent_fraction": 0.75,
        "right_equivalent_fraction": 0.75,
        "mean_oracle_regret": 0.02,
        "mean_q_spearman": 0.60,
        "mean_prior_tv": 0.20,
        "prior_top_change_fraction": 0.25,
        "mean_prior_control_margin": 0.03,
        "mean_gain": 0.04,
        "gain_bootstrap_ci95": [0.01, 0.07],
        "expanded_selection_fraction": 0.25,
        "phase": phase,
    }


def test_133_feasibility_requires_prior_control_and_value_gain() -> None:
    cfg = ProgressivePlannerConfig()
    summary = _passing_summary()
    assert all(feasibility_gates(cfg, summary).values())

    summary["mean_prior_control_margin"] = 0.0
    assert not feasibility_gates(cfg, summary)["prior_control_margin_at_least_0p02"]
    summary["mean_prior_control_margin"] = 0.03
    summary["gain_bootstrap_ci95"] = [-0.01, 0.07]
    assert not feasibility_gates(cfg, summary)["gain_ci95_lower_above_zero"]


def test_133_configuration_is_frozen() -> None:
    cfg = ProgressivePlannerConfig(simulations=768)
    with pytest.raises(ValueError, match="frozen default"):
        _validate_config(cfg)

