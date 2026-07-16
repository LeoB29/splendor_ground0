from __future__ import annotations

from pathlib import Path

import pytest

from splendor_ai.eval.run_shallow_operational_gate import (
    ShallowOperationalConfig,
    block_specs,
    evaluate_broad_gate,
    evaluate_parent_gate,
    summarize_pair_rows,
)


def _game(winner, scores=(15, 10), *, timed_out=False, stalled=False):
    return {
        "winner": winner,
        "final_scores": list(scores),
        "timed_out": timed_out,
        "stalled": stalled,
        "termination_reason": "completed",
        "loop_fallback_triggers_by_seat": [0, 0],
    }


def _pair(seed, first_winner, second_winner):
    return {
        "seed": seed,
        "candidate_seats": [0, 1],
        "games": [_game(first_winner), _game(second_winner, scores=(10, 15))],
    }


def _payload(share, *, ci_low=None, failures=0):
    return {
        "summary": {
            "score_share": {
                "mean": share,
                "ci95_low": share if ci_low is None else ci_low,
                "ci95_high": share,
            },
            "timed_out_games": failures,
            "stalled_games": 0,
        }
    }


def test_pair_summary_is_candidate_role_and_seat_aware() -> None:
    summary = summarize_pair_rows(
        [_pair(10, 0, 1), _pair(11, 1, 1)],
        bootstrap_samples=1_000,
        bootstrap_seed=3,
    )

    assert summary["candidate_wins"] == 3
    assert summary["opponent_wins"] == 1
    assert summary["score_share"]["mean"] == pytest.approx(0.75)
    assert summary["timed_out_games"] == 0


def test_parent_gate_requires_strength_confidence_and_integrity() -> None:
    assert all(evaluate_parent_gate(_payload(0.65, ci_low=0.55)).values())
    assert not all(evaluate_parent_gate(_payload(0.65, ci_low=0.49)).values())
    assert not all(evaluate_parent_gate(_payload(0.65, ci_low=0.55, failures=1)).values())


def test_broad_gate_requires_absolute_floors_and_matched_deltas() -> None:
    blocks = {
        "candidate_greedy": _payload(0.85),
        "control_greedy": _payload(0.87),
        "candidate_016": _payload(0.45),
        "control_016": _payload(0.35),
        "candidate_035": _payload(0.40),
        "control_035": _payload(0.30),
    }

    result = evaluate_broad_gate(blocks)

    assert result["passed"] is True
    blocks["candidate_035"] = _payload(0.34)
    result = evaluate_broad_gate(blocks)
    assert result["gates"]["035_matched_control_delta"] is False
    assert result["passed"] is False


def test_block_specs_use_matched_broad_seeds_and_disjoint_opponents(tmp_path) -> None:
    config = ShallowOperationalConfig(
        output_dir=tmp_path,
        parent_checkpoint=Path("parent.pt"),
        checkpoint_016=Path("016.pt"),
        checkpoint_035=Path("035.pt"),
    )
    specs = block_specs(config)

    assert specs[1].seed_start == specs[2].seed_start
    assert specs[3].seed_start == specs[4].seed_start
    assert specs[5].seed_start == specs[6].seed_start
    assert len({specs[0].seed_start, specs[1].seed_start, specs[3].seed_start, specs[5].seed_start}) == 4
