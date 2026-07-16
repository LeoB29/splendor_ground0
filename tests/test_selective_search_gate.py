from __future__ import annotations

from pathlib import Path

from splendor_ai.eval.run_selective_search_gate import (
    SelectiveSearchGateConfig,
    block_specs,
    evaluate_broad_gate,
    evaluate_fast_parent_gate,
    evaluate_retention_gate,
)


def _payload(
    share: float,
    *,
    ci_low: float | None = None,
    search_fraction: float = 0.5,
    failures: int = 0,
):
    return {
        "summary": {
            "score_share": {
                "mean": share,
                "ci95_low": share if ci_low is None else ci_low,
                "ci95_high": share,
            },
            "timed_out_games": failures,
            "stalled_games": 0,
            "candidate_routing": {
                "turns": 100,
                "search_turns": int(100 * search_fraction),
                "neural_turns": 100 - int(100 * search_fraction),
                "search_fraction": search_fraction,
            },
        }
    }


def test_retention_gate_requires_strength_integrity_and_search_reduction() -> None:
    assert all(evaluate_retention_gate(_payload(0.50, ci_low=0.40)).values())
    assert not all(evaluate_retention_gate(_payload(0.44, ci_low=0.40)).values())
    assert not all(evaluate_retention_gate(_payload(0.50, ci_low=0.40, search_fraction=0.61)).values())


def test_fast_parent_gate_requires_improvement_floor_and_confidence() -> None:
    assert all(evaluate_fast_parent_gate(_payload(0.60, ci_low=0.50)).values())
    assert not all(evaluate_fast_parent_gate(_payload(0.54, ci_low=0.50)).values())
    assert not all(evaluate_fast_parent_gate(_payload(0.60, ci_low=0.45)).values())


def test_broad_gate_is_noninferior_to_matched_shallow_controls() -> None:
    blocks = {
        "candidate_016": _payload(0.40),
        "control_016": _payload(0.43),
        "candidate_035": _payload(0.35),
        "control_035": _payload(0.40),
    }

    result = evaluate_broad_gate(blocks)

    assert result["passed"] is True
    blocks["candidate_035"] = _payload(0.34)
    assert evaluate_broad_gate(blocks)["passed"] is False


def test_block_specs_match_broad_seeds_and_keep_stages_disjoint(tmp_path) -> None:
    config = SelectiveSearchGateConfig(
        output_dir=tmp_path,
        parent_checkpoint=Path("parent.pt"),
        checkpoint_016=Path("016.pt"),
        checkpoint_035=Path("035.pt"),
    )

    specs = block_specs(config)

    assert specs[2].seed_start == specs[3].seed_start
    assert specs[4].seed_start == specs[5].seed_start
    assert len({specs[0].seed_start, specs[1].seed_start, specs[2].seed_start, specs[4].seed_start}) == 4
