from __future__ import annotations

import pytest

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    LegalActionScorerBot,
    ShallowSearchBot,
)
from splendor_ai.eval.run_broad_public_belief_resolver_confirmation import (
    BLOCKS,
    BroadResolverConfirmationConfig,
    _block_gates,
    _opponent,
    _summarize_block,
    _validate_config,
)


def test_frozen_configuration_and_opponent_panel_are_available() -> None:
    cfg = BroadResolverConfirmationConfig()
    _validate_config(cfg)
    assert BLOCKS == (("shallow", 551_000), ("016", 561_000), ("023", 571_000))
    assert isinstance(_opponent(cfg, "shallow"), ShallowSearchBot)
    assert isinstance(_opponent(cfg, "016"), LegalActionScorerBot)
    assert isinstance(_opponent(cfg, "023"), CanonicalEntityActionScorerBot)


def test_configuration_refuses_post_registration_changes(tmp_path) -> None:
    with pytest.raises(ValueError, match="preregistered default"):
        _validate_config(BroadResolverConfirmationConfig(output_dir=tmp_path))


def test_block_summary_requires_exact_frozen_seed_set() -> None:
    cfg = BroadResolverConfirmationConfig()
    with pytest.raises(ValueError, match="exact frozen seed block"):
        _summarize_block(cfg, "shallow", 551_000, [])


def test_block_gate_requires_every_strength_and_integrity_condition() -> None:
    cfg = BroadResolverConfirmationConfig()
    block = {
        "summary": {
            "score_share": {"mean": 0.60, "ci95_low": 0.52},
            "final_score_margin": {"mean": 1.0},
            "seat_breakdown": {
                "seat0": {"score_share": 0.62},
                "seat1": {"score_share": 0.58},
            },
        },
        "integrity": {
            "games": 100,
            "invalid_games": 0,
            "timed_out_games": 0,
            "stalled_games": 0,
        },
        "telemetry": {
            "exact_branch_accounting": True,
            "all_continuations_resolved": True,
            "hard_caps": 0,
            "loop_adjudication_fraction": 0.0,
            "switches": 20,
            "by_phase": {
                phase: {"completed_searches": 1, "switches": 1}
                for phase in ("early", "mid", "late")
            },
        },
    }
    gates = _block_gates(cfg, block)
    assert all(gates.values())
    block["summary"]["score_share"]["ci95_low"] = 0.50
    gates = _block_gates(cfg, block)
    assert not gates["score_share_ci95_lower_above_50pct"]
