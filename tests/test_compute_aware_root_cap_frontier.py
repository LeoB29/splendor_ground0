from __future__ import annotations

from dataclasses import replace

import pytest

from splendor_ai.eval.run_compute_aware_root_cap_frontier import (
    ComputeAwareRootCapConfig,
    _best_index,
    _confirmation_gates,
    _discovery_gates,
    _provenance_matches,
    _replay_cap,
    _validate_config,
)


def test_088_configuration_freezes_cross_experiment_cap_frontier() -> None:
    cfg = ComputeAwareRootCapConfig()
    _validate_config(cfg)
    assert _provenance_matches(cfg)
    assert cfg.caps == (6, 8, 12, 16, 24, 32)
    assert cfg.minimum_discovery_gain == 0.045
    assert cfg.minimum_confirmation_gain == 0.040
    assert cfg.maximum_root_ratio == 0.70
    assert cfg.bootstrap_samples == 20_000


def test_088_configuration_refuses_post_registration_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ComputeAwareRootCapConfig(), caps=(8, 12)))


def test_best_index_uses_earliest_canonical_tie() -> None:
    assert _best_index([0.1, 0.3, 0.3, 0.2]) == 1


def test_cap_replay_uses_discovery_only_and_full_frontier_regret() -> None:
    row = {
        "sample_index": 0,
        "phase": "early",
        "champion_discovery_best_index": 1,
        "expanded_discovery_values": [0.1, 0.2, 0.0, -0.1, 0.3, 0.4],
        "expanded_evaluation_values": [0.0, 0.1, 0.0, 0.0, 0.2, 0.5],
    }
    capped = _replay_cap([row], 5, "085")[0]
    assert capped["selected_index"] == 4
    assert capped["champion_selected_index"] == 1
    assert capped["evaluation_gain"] == pytest.approx(0.1)
    assert capped["full_frontier_evaluation_regret"] == pytest.approx(0.3)
    assert not capped["full_discovery_action_agreement"]

    uncapped = _replay_cap([row], 99, "085")[0]
    assert uncapped["selected_index"] == 5
    assert uncapped["evaluation_gain"] == pytest.approx(0.4)
    assert uncapped["full_frontier_evaluation_regret"] == pytest.approx(0.0)


def _passing_summary() -> dict:
    return {
        "mean_evaluation_gain": 0.05,
        "evaluation_gain_ci95_low": 0.02,
        "change_fraction": 0.30,
        "changed_every_phase": True,
        "equivalent_regret_fraction": 0.70,
        "mean_full_frontier_regret": 0.02,
        "p90_full_frontier_regret": 0.07,
        "mean_root_ratio": 0.60,
        "evaluation_half_mean_gains": [0.04, 0.05, 0.06],
        "phase_breakdown": {
            phase: {"mean_gain": 0.04, "gain_ci95_high": 0.07}
            for phase in ("early", "mid", "late")
        },
    }


def test_088_discovery_and_confirmation_gates_are_conjunctive() -> None:
    cfg = ComputeAwareRootCapConfig()
    summary = _passing_summary()
    assert all(_discovery_gates(cfg, summary, 0.06).values())
    assert all(_confirmation_gates(cfg, summary).values())

    summary["mean_root_ratio"] = 0.701
    assert not _discovery_gates(cfg, summary, 0.06)["mean_root_ratio_at_most_0p70"]
    assert not _confirmation_gates(cfg, summary)["mean_root_ratio_at_most_0p70"]

    summary = _passing_summary()
    summary["evaluation_half_mean_gains"][2] = 0.0
    assert not _confirmation_gates(cfg, summary)["all_three_evaluation_halves_positive"]
