from __future__ import annotations

from dataclasses import replace

import pytest

from splendor_ai.eval.run_expanded_root_resolver_fresh_champion_comparison import (
    ExpandedRootFreshComparisonConfig,
    _operational_gate_names,
    _provenance_matches,
    _strength_gates,
    _validate_config,
    preflight,
)


def test_134_configuration_freezes_fresh_ten_worker_comparison() -> None:
    cfg = ExpandedRootFreshComparisonConfig()
    _validate_config(cfg)
    assert _provenance_matches(cfg)
    assert cfg.smoke_seed == 1_340_000
    assert cfg.strength_seed_start == 1_340_100
    assert cfg.pairs == 200
    assert cfg.workers == 10
    assert cfg.minimum_score_share == 0.55
    assert cfg.maximum_strength_wall_seconds == 72 * 60 * 60


def test_134_configuration_refuses_worker_or_seed_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ExpandedRootFreshComparisonConfig(), workers=8))
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ExpandedRootFreshComparisonConfig(), pairs=100))


def test_134_preflight_passes_without_starting_gameplay() -> None:
    result = preflight()
    assert result["passed"]
    assert all(result["checks"].values())
    assert result["launch_required"]
    assert not result["training_authorized"]
    assert not result["gameplay_started"]
    assert not result["identity"]["reads_087_strength_progress"]


def _passing_strength_with_valid_loop_fallbacks() -> dict:
    phase = {
        name: {"optional_roots_retained": 5, "expansion_switches": 1}
        for name in ("early", "mid", "late")
    }
    return {
        "active_wall_seconds": 1000.0,
        "summary": {
            "score_share": {"mean": 0.56, "ci95_low": 0.51},
            "final_score_margin": {"mean": 0.5},
            "seat_breakdown": {
                "seat0": {"score_share": 0.56},
                "seat1": {"score_share": 0.56},
            },
        },
        "integrity": {
            "games": 400,
            "invalid_games": 0,
            "timed_out_games": 0,
            "stalled_games": 0,
        },
        "candidate_telemetry": {
            "exact_branch_accounting": True,
            "all_continuations_resolved": True,
            "hard_caps": 0,
            "rollout_fallback_triggers": 0,
            "continuations": 200_000,
            "loop_adjudications": 0,
            "loop_adjudication_fraction": 0.0,
            "expansion_switch_rate": 0.20,
            "by_phase": phase,
        },
        "champion_telemetry": {
            "exact_branch_accounting": True,
            "all_continuations_resolved": True,
            "hard_caps": 0,
            "rollout_fallback_triggers": 128,
            "continuations": 200_000,
            "loop_adjudications": 2,
            "loop_adjudication_fraction": 0.00001,
        },
    }


def test_134_accepts_fallback_triggers_when_loop_rate_is_bounded() -> None:
    gates = _strength_gates(
        ExpandedRootFreshComparisonConfig(),
        _passing_strength_with_valid_loop_fallbacks(),
    )
    assert all(gates.values())
    assert "zero_rollout_fallback_triggers" not in gates
    assert "zero_rollout_fallback_triggers" not in _operational_gate_names()


def test_134_strength_effect_gates_remain_conjunctive() -> None:
    strength = _passing_strength_with_valid_loop_fallbacks()
    strength["summary"]["score_share"]["ci95_low"] = 0.50
    gates = _strength_gates(ExpandedRootFreshComparisonConfig(), strength)
    assert not gates["score_share_ci95_lower_above_50pct"]

    strength = _passing_strength_with_valid_loop_fallbacks()
    strength["candidate_telemetry"]["by_phase"]["late"]["expansion_switches"] = 0
    gates = _strength_gates(ExpandedRootFreshComparisonConfig(), strength)
    assert not gates["expanded_roots_and_switches_every_phase"]
