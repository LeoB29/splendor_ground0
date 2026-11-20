from __future__ import annotations

from dataclasses import replace

import pytest

from splendor_ai.eval.run_expanded_root_regret_stability_replication import (
    ExpandedRootReplicationConfig,
    _argmax_gates,
    _primary_gates,
    _validate_config,
    _wilson_interval,
    continuation_seed,
)


def test_086_configuration_freezes_fresh_triple_evaluation_replication() -> None:
    cfg = ExpandedRootReplicationConfig()
    _validate_config(cfg)
    assert cfg.source_seed_start == 860_000
    assert cfg.source_games == 600
    assert cfg.states_per_cell == 50
    assert cfg.discovery_continuations == 4
    assert cfg.evaluation_halves == 3
    assert cfg.continuations_per_evaluation_half == 4
    assert cfg.continuations_per_leaf == 16
    assert cfg.maximum_equivalent_regret == 0.025
    assert cfg.minimum_equivalent_fraction == 0.60
    assert cfg.maximum_p90_regret == 0.075


def test_086_configuration_refuses_post_registration_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ExpandedRootReplicationConfig(), evaluation_halves=2))


def test_086_continuation_namespace_is_unique_and_bounded() -> None:
    cfg = ExpandedRootReplicationConfig()
    seeds = {
        continuation_seed(cfg, sample, det, continuation)
        for sample in (0, 299)
        for det in range(3)
        for continuation in range(16)
    }
    assert len(seeds) == 96
    assert min(seeds) >= 8_600_000_000_000_000_000
    assert max(seeds) < 2**63


def test_086_primary_and_argmax_gates_are_separate_conjunctions() -> None:
    cfg = ExpandedRootReplicationConfig()
    summary = {
        "mean_pooled_evaluation_gain": 0.06,
        "pooled_evaluation_gain_ci95_low": 0.04,
        "change_fraction": 0.70,
        "evaluation_half_mean_gains": [0.05, 0.06, 0.07],
        "phase_breakdown": {phase: {"changed": 1} for phase in ("early", "mid", "late")},
        "equivalent_regret_fraction": 0.70,
        "mean_pooled_evaluation_regret": 0.02,
        "p90_pooled_evaluation_regret": 0.07,
        "pooled_expanded_root_value_pearson": 0.90,
        "evaluation_half_reliability": {
            f"eval{index}": {"expanded_root_pearson": 0.80} for index in range(3)
        },
    }
    assert all(_primary_gates(cfg, summary).values())
    assert all(_argmax_gates(cfg, summary).values())
    summary["p90_pooled_evaluation_regret"] = 0.076
    assert not _argmax_gates(cfg, summary)["p90_regret_at_most_0p075"]
    assert all(_primary_gates(cfg, summary).values())


def test_exact_best_agreement_is_report_only_wilson_diagnostic() -> None:
    low, high = _wilson_interval(179, 300)
    assert low < 179 / 300 < high
    assert low == pytest.approx(0.540271, abs=1e-6)
    assert high == pytest.approx(0.650618, abs=1e-6)
