from __future__ import annotations

import json

import pytest

from splendor_ai.training.run_entity_broad_promotion_gate import summarize_broad_gate


def _checkpoint_payload(wins: int, draws: int = 0) -> dict[str, object]:
    return {
        "matchup": {
            "games": 40,
            "candidate_wins": wins,
            "draws": draws,
            "timed_out_games": 0,
            "stalled_games": 0,
        }
    }


def _baseline_payload(wins: int, draws: int = 0) -> dict[str, object]:
    return {
        "opponents": [{
            "games": 40,
            "model_wins": wins,
            "draws": draws,
            "timed_out_games": 0,
            "stalled_games": 0,
        }]
    }


def test_broad_gate_requires_absolute_and_matched_control_gains(tmp_path) -> None:
    payloads = {
        "greedy": (_baseline_payload(34), _baseline_payload(35)),
        "shallow": (_baseline_payload(16), _baseline_payload(10)),
        "016": (_checkpoint_payload(12), _checkpoint_payload(8)),
        "035": (_checkpoint_payload(10), _checkpoint_payload(6)),
    }
    for opponent, (candidate, parent) in payloads.items():
        (tmp_path / f"broad_candidate_{opponent}.json").write_text(json.dumps(candidate))
        (tmp_path / f"broad_parent_{opponent}.json").write_text(json.dumps(parent))

    result = summarize_broad_gate(tmp_path)

    assert result["all_gates_passed"] is True
    assert result["results"]["shallow"]["candidate_minus_parent"] == pytest.approx(0.15)


def test_broad_gate_rejects_candidate_that_only_beats_absolute_floor(tmp_path) -> None:
    payloads = {
        "greedy": (_baseline_payload(34), _baseline_payload(35)),
        "shallow": (_baseline_payload(12), _baseline_payload(12)),
        "016": (_checkpoint_payload(12), _checkpoint_payload(8)),
        "035": (_checkpoint_payload(10), _checkpoint_payload(6)),
    }
    for opponent, (candidate, parent) in payloads.items():
        (tmp_path / f"broad_candidate_{opponent}.json").write_text(json.dumps(candidate))
        (tmp_path / f"broad_parent_{opponent}.json").write_text(json.dumps(parent))

    result = summarize_broad_gate(tmp_path)

    assert result["gates"]["shallow_absolute_floor"] is True
    assert result["gates"]["shallow_matched_control_delta"] is False
    assert result["all_gates_passed"] is False
