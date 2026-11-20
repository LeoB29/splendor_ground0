from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_expanded_root_resolver_headroom import (
    CELLS,
    ExpandedRootHeadroomConfig,
    _bootstrap_mean_ci,
    _prepare_structure,
    _signal_gates,
    _validate_config,
    assigned_cell,
    continuation_seed,
)


class _OrderedFakeScorer:
    def score_actions(self, state, legal_actions):
        return tuple(
            SimpleNamespace(action=action, score=float(len(legal_actions) - index))
            for index, action in enumerate(legal_actions)
        )


def test_085_configuration_freezes_balanced_powered_audit() -> None:
    cfg = ExpandedRootHeadroomConfig()
    _validate_config(cfg)
    assert cfg.source_seed_start == 850_000
    assert cfg.source_games == 600
    assert cfg.states_per_cell == 50
    assert len(CELLS) == 6
    assert cfg.champion_roots == 4
    assert cfg.determinizations == 3
    assert cfg.opponent_actions == 4
    assert cfg.continuations_per_leaf == 8
    assert cfg.minimum_mean_gain == 0.05
    assert cfg.minimum_change_fraction == 0.15


def test_085_configuration_refuses_post_registration_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ExpandedRootHeadroomConfig(), states_per_cell=51))


def test_assignment_and_continuation_namespaces_are_exact() -> None:
    cfg = ExpandedRootHeadroomConfig()
    assert [assigned_cell(850_000 + offset, cfg) for offset in range(6)] == [
        (0, "early"),
        (0, "mid"),
        (0, "late"),
        (1, "early"),
        (1, "mid"),
        (1, "late"),
    ]
    seeds = {
        continuation_seed(cfg, sample, det, continuation)
        for sample in (0, 299)
        for det in range(3)
        for continuation in range(8)
    }
    assert len(seeds) == 48
    assert max(seeds) < 2**63


def test_expansion_retains_top_four_and_excludes_added_deck_reserves() -> None:
    cfg = ExpandedRootHeadroomConfig()
    state = SplendorEnv(seed=850_000).initial_state()
    structure = _prepare_structure(state, cfg, _OrderedFakeScorer(), 850_000, 0)
    assert structure["champion_eligible"]
    assert structure["expanded_root_count"] >= 4
    assert structure["expanded_root_keys"][:4] == structure["champion_root_keys"]
    added = structure["expanded_root_keys"][4:]
    assert all(not key.startswith("('RESERVE_DECK'") for key in added)
    assert structure["excluded_root_reasons"].get("omitted_reserve_deck", 0) >= 1


def test_bootstrap_and_signal_gates_are_deterministic_and_conjunctive() -> None:
    values = [0.10] * 300
    assert _bootstrap_mean_ci(values, 100, 850_085) == (0.1, 0.1)
    summary = {
        "mean_evaluation_gain": 0.06,
        "evaluation_gain_ci95_low": 0.01,
        "change_fraction": 0.20,
        "phase_breakdown": {phase: {"changed": 1} for phase in ("early", "mid", "late")},
    }
    cfg = ExpandedRootHeadroomConfig()
    assert all(_signal_gates(cfg, summary).values())
    summary["mean_evaluation_gain"] = 0.049
    assert not _signal_gates(cfg, summary)["mean_gain_at_least_0p05"]
    summary["mean_evaluation_gain"] = 0.06
    summary["change_fraction"] = 0.149
    assert not _signal_gates(cfg, summary)["change_fraction_at_least_15pct"]
