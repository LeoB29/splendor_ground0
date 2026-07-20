from __future__ import annotations

from dataclasses import dataclass

from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    EXPECTED_BRANCHES_PER_SEARCH,
    FrozenPublicBeliefResolverBot,
    ResolverGameplayConfig,
    _direct_gates,
    _frozen_contract_matches_049,
    gameplay_belief_root_index,
    gameplay_continuation_seed,
)
from splendor_ai.training.run_action_preference_repair import LoopSafeContinuationResult
from splendor_ai.training.run_policy_search_disagreement import ModelActionScore


class _CanonicalFakeScorer:
    def score_actions(self, state, legal_actions):
        ordered = sorted(
            enumerate(legal_actions),
            key=lambda item: repr(canonical_successor_action_key(state, item[1])),
        )
        return tuple(
            ModelActionScore(action=action, action_index=index, score=float(-rank))
            for rank, (index, action) in enumerate(ordered)
        )


def _resolved_rollout(*args, **kwargs):
    action: Action = args[3]
    value = (sum(ord(char) for char in repr(action)) % 21 - 10) / 20.0
    return LoopSafeContinuationResult(
        value=value,
        terminal=True,
        resolved=True,
        turns=1,
        fallback_triggers=0,
        loop_adjudicated=False,
        hard_cap_reached=False,
    )


def test_gameplay_contract_is_frozen_from_049() -> None:
    cfg = ResolverGameplayConfig()

    assert _frozen_contract_matches_049(cfg)
    assert EXPECTED_BRANCHES_PER_SEARCH == 192
    assert cfg.discovery_continuations == 4
    assert cfg.direct_pairs == 100


def test_gameplay_seed_namespaces_are_deterministic_and_disjoint() -> None:
    cfg = ResolverGameplayConfig()
    root_indices = {
        gameplay_belief_root_index(cfg, seed, seat, decision)
        for seed in (500_000, 500_100, 500_199)
        for seat in (0, 1)
        for decision in (0, 399)
    }
    continuation_seeds = {
        gameplay_continuation_seed(cfg, seed, seat, decision, det, continuation)
        for seed in (500_000, 500_100, 500_199)
        for seat in (0, 1)
        for decision in (0, 399)
        for det in (0, 2)
        for continuation in (0, 3)
    }

    assert len(root_indices) == 12
    assert len(continuation_seeds) == 48
    assert gameplay_belief_root_index(cfg, 500_000, 0, 0) == 1_000_000
    assert gameplay_continuation_seed(cfg, 500_000, 0, 0, 0, 0) == 56_000_000


def test_resolver_bot_uses_exact_discovery_branch_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        "splendor_ai.eval.run_paired_public_belief_resolver_gameplay.rollout_score_margin_loop_safe",
        _resolved_rollout,
    )
    cfg = ResolverGameplayConfig()
    env = SplendorEnv(seed=500_100)
    state = env.initial_state()
    legal = env.legal_actions(state)
    bot = FrozenPublicBeliefResolverBot(
        cfg,
        engine_seed=500_100,
        actor_seat=0,
        scorer=_CanonicalFakeScorer(),
    )

    selected = bot.choose_action(env, state, legal)

    assert selected in legal
    assert bot.telemetry["completed_searches"] == 1
    assert bot.telemetry["continuations"] == EXPECTED_BRANCHES_PER_SEARCH
    assert bot.telemetry["resolved_continuations"] == EXPECTED_BRANCHES_PER_SEARCH
    assert bot.telemetry["hard_caps"] == 0


def test_resolver_bot_falls_back_when_root_width_is_unavailable() -> None:
    cfg = ResolverGameplayConfig()
    env = SplendorEnv(seed=500_100)
    state = env.initial_state()
    legal = env.legal_actions(state)[:1]
    bot = FrozenPublicBeliefResolverBot(
        cfg,
        engine_seed=500_100,
        actor_seat=0,
        scorer=_CanonicalFakeScorer(),
    )

    assert bot.choose_action(env, state, legal) == legal[0]
    assert bot.telemetry["completed_searches"] == 0
    assert bot.telemetry["fallback_reasons"] == {"fewer_than_four_root_actions": 1}


def test_direct_gate_requires_strength_integrity_and_both_seats() -> None:
    cfg = ResolverGameplayConfig()
    direct = {
        "integrity": {"games": 200, "invalid_games": 0, "timed_out_games": 0, "stalled_games": 0},
        "telemetry": {
            "exact_branch_accounting": True,
            "all_continuations_resolved": True,
            "hard_caps": 0,
            "loop_adjudication_fraction": 0.0,
            "switches": 8,
            "by_phase": {
                phase: {"completed_searches": 1, "switches": 1}
                for phase in ("early", "mid", "late")
            },
        },
        "summary": {
            "score_share": {"mean": 0.56, "ci95_low": 0.51},
            "final_score_margin": {"mean": 0.1},
            "seat_breakdown": {
                "seat0": {"score_share": 0.55},
                "seat1": {"score_share": 0.57},
            },
        },
    }

    assert all(_direct_gates(cfg, direct).values())
    direct["summary"]["seat_breakdown"]["seat1"]["score_share"] = 0.49
    assert not _direct_gates(cfg, direct)["no_seat_collapse"]
