from __future__ import annotations

from dataclasses import replace

import pytest

from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
)
from splendor_ai.eval.run_resolver_research_champion_confirmation import (
    ResolverResearchChampionConfig,
    _provenance_matches,
    _resolver_config,
    _strength_gates,
    _validate_config,
)


def test_084_configuration_freezes_fresh_large_paired_panel() -> None:
    cfg = ResolverResearchChampionConfig()
    _validate_config(cfg)
    assert cfg.smoke_seed == 840_000
    assert cfg.strength_seed_start == 840_100
    assert cfg.pairs == 200
    assert cfg.minimum_switch_rate == 0.15
    assert _provenance_matches(cfg)

    resolver_cfg = _resolver_config(cfg)
    assert resolver_cfg.determinizations == 3
    assert resolver_cfg.root_actions == 4
    assert resolver_cfg.opponent_actions == 4
    assert resolver_cfg.discovery_continuations == 4
    assert EXPECTED_BRANCHES_PER_SEARCH == 192
    assert resolver_cfg.belief_root_index_start == 840_000_000
    assert resolver_cfg.continuation_seed_start == 8_400_000_000_000_000_000


def test_084_configuration_refuses_post_registration_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ResolverResearchChampionConfig(), pairs=201))


def _passing_strength() -> dict:
    return {
        "summary": {
            "score_share": {"mean": 0.60, "ci95_low": 0.52},
            "final_score_margin": {"mean": 1.0},
            "seat_breakdown": {
                "seat0": {"score_share": 0.61},
                "seat1": {"score_share": 0.59},
            },
        },
        "integrity": {
            "games": 400,
            "invalid_games": 0,
            "timed_out_games": 0,
            "stalled_games": 0,
        },
        "telemetry": {
            "decisions": 100,
            "switches": 20,
            "exact_branch_accounting": True,
            "all_continuations_resolved": True,
            "hard_caps": 0,
            "loop_adjudication_fraction": 0.0,
            "by_phase": {
                phase: {"completed_searches": 1, "switches": 1}
                for phase in ("early", "mid", "late")
            },
        },
    }


def test_084_strength_gates_require_effect_interval_dose_and_integrity() -> None:
    cfg = ResolverResearchChampionConfig()
    strength = _passing_strength()
    assert all(_strength_gates(cfg, strength).values())

    strength["telemetry"]["switches"] = 14
    gates = _strength_gates(cfg, strength)
    assert not gates["switch_rate_at_least_15pct"]

    strength = _passing_strength()
    strength["summary"]["score_share"]["ci95_low"] = 0.50
    gates = _strength_gates(cfg, strength)
    assert not gates["score_share_ci95_lower_above_50pct"]


def test_084_strength_gate_requires_both_seats() -> None:
    cfg = ResolverResearchChampionConfig()
    strength = _passing_strength()
    strength["summary"]["seat_breakdown"]["seat1"]["score_share"] = 0.49
    gates = _strength_gates(cfg, strength)
    assert not gates["no_seat_collapse"]
