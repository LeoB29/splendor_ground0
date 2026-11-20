from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import splendor_ai.eval.run_expanded_root_resolver_champion_challenge as challenge087

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_expanded_root_resolver_champion_challenge import (
    ExpandedRootChampionChallengeConfig,
    ExpandedRootPublicBeliefResolverBot,
    _candidate_contract_matches,
    _prepare_expanded_structure,
    _provenance_matches,
    _resolver_config,
    _strength_gates,
    _validate_config,
)
from splendor_ai.eval.run_expanded_root_resolver_headroom import (
    ExpandedRootHeadroomConfig,
    _prepare_structure as _prepare_085_structure,
)
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    _distinct_canonical_scores,
    gameplay_belief_root_index,
    gameplay_continuation_seed,
)


class _OrderedFakeScorer:
    def score_actions(self, state, legal_actions):
        return tuple(
            SimpleNamespace(action=action, score=float(len(legal_actions) - index))
            for index, action in enumerate(legal_actions)
        )


def test_087_configuration_freezes_authorized_paired_challenge() -> None:
    cfg = ExpandedRootChampionChallengeConfig()
    _validate_config(cfg)
    assert _provenance_matches(cfg)
    assert _candidate_contract_matches(cfg)
    assert cfg.smoke_seed == 870_000
    assert cfg.strength_seed_start == 870_100
    assert cfg.pairs == 200
    assert cfg.expected_score_share == 0.565
    assert cfg.minimum_score_share == 0.55
    assert cfg.minimum_expansion_switch_rate == 0.15
    assert cfg.maximum_strength_wall_seconds == 72 * 60 * 60

    resolver_cfg = _resolver_config(cfg)
    assert resolver_cfg.root_actions == 4
    assert resolver_cfg.determinizations == 3
    assert resolver_cfg.opponent_actions == 4
    assert resolver_cfg.discovery_continuations == 4
    assert EXPECTED_BRANCHES_PER_SEARCH == 192


def test_087_configuration_refuses_post_registration_changes() -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(replace(ExpandedRootChampionChallengeConfig(), pairs=201))


def test_087_matched_namespaces_are_unique_fresh_and_bounded() -> None:
    cfg = _resolver_config(ExpandedRootChampionChallengeConfig())
    assert gameplay_belief_root_index(cfg, 870_000, 0, 0) == 870_000_000
    seeds = {
        gameplay_continuation_seed(cfg, engine, seat, decision, det, continuation)
        for engine in (870_000, 870_299)
        for seat in (0, 1)
        for decision in (0, 399)
        for det in range(3)
        for continuation in range(4)
    }
    assert len(seeds) == 96
    assert min(seeds) >= 8_700_000_000_000_000_000
    assert max(seeds) < 2**63


def test_087_structure_preserves_champion_prefix_and_omits_added_deck_reserves() -> None:
    cfg = _resolver_config(ExpandedRootChampionChallengeConfig())
    state = SplendorEnv(seed=870_000).initial_state()
    scorer = _OrderedFakeScorer()
    ranked = _distinct_canonical_scores(
        state,
        scorer.score_actions(state, SplendorEnv().legal_actions(state)),
    )
    structure = _prepare_expanded_structure(state, ranked, cfg, scorer, 870_000_000)
    assert len(structure["root_keys"]) >= 4
    assert structure["root_keys"][:4] == structure["champion_root_keys"]
    assert all(not repr(key).startswith("('RESERVE_DECK'") for key in structure["root_keys"][4:])
    assert structure["optional_root_exclusions"].get("omitted_reserve_deck", 0) >= 1


def test_087_root_structure_exactly_reproduces_085_086_candidate() -> None:
    state = SplendorEnv(seed=870_000).initial_state()
    scorer = _OrderedFakeScorer()
    gameplay_cfg = _resolver_config(ExpandedRootChampionChallengeConfig())
    ranked = _distinct_canonical_scores(
        state,
        scorer.score_actions(state, SplendorEnv().legal_actions(state)),
    )
    gameplay = _prepare_expanded_structure(
        state,
        ranked,
        gameplay_cfg,
        scorer,
        870_000_000,
    )
    audit_cfg = replace(
        ExpandedRootHeadroomConfig(),
        source_seed_start=870_000,
        belief_root_base=870_000_000,
    )
    audit = _prepare_085_structure(state, audit_cfg, scorer, 870_000, 0)
    assert [repr(key) for key in gameplay["root_keys"]] == audit["expanded_root_keys"]
    assert [repr(key) for key in gameplay["champion_root_keys"]] == audit[
        "champion_root_keys"
    ]
    assert gameplay["optional_root_exclusions"] == audit["excluded_root_reasons"]


def test_087_candidate_rejects_within_decision_parallelism() -> None:
    cfg = _resolver_config(ExpandedRootChampionChallengeConfig())
    with pytest.raises(ValueError, match="serial continuation"):
        ExpandedRootPublicBeliefResolverBot(cfg, 870_000, 0, continuation_workers=2)


def test_087_candidate_executes_variable_table_with_exact_accounting(monkeypatch) -> None:
    def fake_rollout(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            value=0.0,
            resolved=True,
            hard_cap_reached=False,
            loop_adjudicated=False,
            fallback_triggers=0,
        )

    monkeypatch.setattr(challenge087, "rollout_score_margin_loop_safe", fake_rollout)
    cfg = _resolver_config(ExpandedRootChampionChallengeConfig())
    env = SplendorEnv(seed=870_000)
    state = env.initial_state()
    bot = ExpandedRootPublicBeliefResolverBot(
        cfg,
        870_000,
        0,
        scorer=_OrderedFakeScorer(),
    )
    action = bot.choose_action(env, state, env.legal_actions(state))
    telemetry = bot.telemetry
    assert action in env.legal_actions(state)
    assert telemetry["completed_searches"] == 1
    assert telemetry["minimum_retained_roots"] >= 4
    assert telemetry["continuations"] == telemetry["expected_continuations"]
    assert telemetry["exact_branch_accounting"]
    assert telemetry["all_continuations_resolved"]


def _passing_strength() -> dict:
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
            "continuations": 1000,
            "loop_adjudications": 0,
            "loop_adjudication_fraction": 0.0,
            "expansion_switch_rate": 0.20,
            "by_phase": phase,
        },
        "champion_telemetry": {
            "exact_branch_accounting": True,
            "all_continuations_resolved": True,
            "hard_caps": 0,
            "rollout_fallback_triggers": 0,
            "continuations": 1000,
            "loop_adjudications": 0,
            "loop_adjudication_fraction": 0.0,
        },
    }


def test_087_strength_gates_are_conjunctive_and_use_expansion_dose() -> None:
    cfg = ExpandedRootChampionChallengeConfig()
    strength = _passing_strength()
    assert all(_strength_gates(cfg, strength).values())

    strength["candidate_telemetry"]["expansion_switch_rate"] = 0.149
    assert not _strength_gates(cfg, strength)["expansion_switch_rate_at_least_15pct"]

    strength = _passing_strength()
    strength["summary"]["score_share"]["ci95_low"] = 0.50
    assert not _strength_gates(cfg, strength)["score_share_ci95_lower_above_50pct"]

    strength = _passing_strength()
    strength["candidate_telemetry"]["by_phase"]["late"]["expansion_switches"] = 0
    assert not _strength_gates(cfg, strength)["expanded_roots_and_switches_every_phase"]
