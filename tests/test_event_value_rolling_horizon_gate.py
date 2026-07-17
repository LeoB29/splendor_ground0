from __future__ import annotations

from pathlib import Path

from splendor_ai.eval.run_event_value_rolling_horizon import (
    EventValueRollingHorizonGateConfig,
    _smoke_gate,
    _strength_gate,
    build_parser,
    config_from_args,
    run_gate,
)


def _stage(games: int, share: float = 0.6, low: float = 0.51, margin: float = 1.0):
    return {
        "elapsed_seconds": 1.0,
        "summary": {
            "completed_games": games,
            "noncompleted_games": 0,
            "candidate_sequence_evaluations": 32,
            "candidate_planning_decisions": 1,
            "opponent_sequence_evaluations": 32,
            "opponent_planning_decisions": 1,
            "score_share": {"mean": share, "ci95_low": low, "ci95_high": 0.7},
            "final_score_margin": {"mean": margin, "ci95_low": 0.1, "ci95_high": 2.0},
        },
    }


def test_parser_uses_frozen_defaults() -> None:
    args = build_parser().parse_args([])
    config = config_from_args(args)
    assert config.stage == "all"
    assert config.device == "cpu"
    assert config.output_dir == Path("outputs/event_value_rolling_horizon_040")


def test_gate_helpers_enforce_preregistered_thresholds() -> None:
    assert all(_smoke_gate(_stage(4)).values())
    assert all(_strength_gate(_stage(60)).values())
    failed = _strength_gate(_stage(60, share=0.55, low=0.50, margin=0.1))
    assert failed["score_share_at_least_55pct"]
    assert not failed["score_share_ci95_lower_above_50pct"]


def test_run_gate_stops_before_parent_when_control_fails(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "035.pt"
    checkpoint.write_bytes(b"placeholder")
    calls = []

    def fake_stage(config, name, seed_start, pairs, opponent):
        calls.append(name)
        if name == "smoke":
            return _stage(4)
        return _stage(60, share=0.50, low=0.40, margin=-0.5)

    monkeypatch.setattr(
        "splendor_ai.eval.run_event_value_rolling_horizon._load_or_run_stage",
        fake_stage,
    )
    payload = run_gate(
        EventValueRollingHorizonGateConfig(
            checkpoint_035=checkpoint,
            output_dir=tmp_path / "out",
        )
    )

    assert calls == ["smoke", "control"]
    assert payload["decision"] == "reject_fixed_event_evaluator"
    assert (tmp_path / "out" / "gate_final.json").exists()
