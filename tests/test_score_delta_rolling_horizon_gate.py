from __future__ import annotations

from pathlib import Path

from splendor_ai.eval.run_score_delta_rolling_horizon import (
    ScoreDeltaRollingHorizonGateConfig,
    _strength_gate,
    build_parser,
    config_from_args,
    run_gate,
)


def _block(games: int, share: float = 0.60, low: float = 0.52, margin: float = 1.0):
    return {
        "summary": {
            "completed_games": games,
            "noncompleted_games": 0,
            "candidate_exact_budget": True,
            "score_share": {"mean": share, "ci95_low": low, "ci95_high": 0.68},
            "final_score_margin": {"mean": margin, "ci95_low": 0.1, "ci95_high": 2.0},
        }
    }


def test_parser_freezes_expected_defaults() -> None:
    config = config_from_args(build_parser().parse_args([]))
    assert config.stage == "all"
    assert config.device == "cpu"
    assert config.output_dir == Path("outputs/score_delta_rolling_horizon_041")


def test_strength_gate_uses_strict_ci_and_exact_budget() -> None:
    assert all(_strength_gate(_block(200), 200).values())
    failed = _strength_gate(_block(200, share=0.55, low=0.50, margin=0.1), 200)
    assert failed["score_share_at_least_55pct"]
    assert not failed["score_share_ci95_lower_above_50pct"]
    payload = _block(200)
    payload["summary"]["candidate_exact_budget"] = False
    assert not _strength_gate(payload, 200)["exact_candidate_budget"]


def test_direct_failure_prevents_every_broad_block(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / name for name in ("035.pt", "016.pt", "023.pt")]
    for path in paths:
        path.write_bytes(b"checkpoint")
    calls = []

    def fake_block(config, name, opponent, seed_start, pairs):
        calls.append((name, opponent, seed_start, pairs))
        return _block(200, share=0.50, low=0.43, margin=-0.2)

    monkeypatch.setattr(
        "splendor_ai.eval.run_score_delta_rolling_horizon._load_or_run_block",
        fake_block,
    )
    payload = run_gate(
        ScoreDeltaRollingHorizonGateConfig(
            checkpoint_035=paths[0],
            checkpoint_016=paths[1],
            checkpoint_023=paths[2],
            output_dir=tmp_path / "out",
        )
    )
    assert calls == [("direct_vs_035", "035", 403000, 100)]
    assert payload["decision"] == "reject_score_delta_close_rolling_horizon"


def test_direct_pass_unlocks_all_frozen_broad_blocks(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / name for name in ("035.pt", "016.pt", "023.pt")]
    for path in paths:
        path.write_bytes(b"checkpoint")
    calls = []

    def fake_block(config, name, opponent, seed_start, pairs):
        calls.append((name, opponent, seed_start, pairs))
        return _block(200 if opponent == "035" else 100)

    monkeypatch.setattr(
        "splendor_ai.eval.run_score_delta_rolling_horizon._load_or_run_block",
        fake_block,
    )
    payload = run_gate(
        ScoreDeltaRollingHorizonGateConfig(
            checkpoint_035=paths[0],
            checkpoint_016=paths[1],
            checkpoint_023=paths[2],
            output_dir=tmp_path / "out",
        )
    )
    assert calls == [
        ("direct_vs_035", "035", 403000, 100),
        ("broad_vs_shallow", "shallow", 403200, 50),
        ("broad_vs_016", "016", 403300, 50),
        ("broad_vs_023", "023", 403400, 50),
    ]
    assert payload["decision"] == "authorize_search_teacher"
