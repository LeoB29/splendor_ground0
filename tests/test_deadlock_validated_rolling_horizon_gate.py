from __future__ import annotations

from pathlib import Path

from splendor_ai.eval.run_deadlock_validated_rolling_horizon import (
    DeadlockValidatedRollingHorizonConfig,
    _add_corrected_integrity,
    _strength_gate,
    build_parser,
    config_from_args,
    run_gate,
)


def _block(games: int, share: float = 0.60, low: float = 0.52, margin: float = 1.0):
    pairs = []
    for seed in range(games // 2):
        pairs.append(
            {
                "seed": seed,
                "games": [
                    {"termination_reason": "completed", "timed_out": False, "stalled": False},
                    {"termination_reason": "completed", "timed_out": False, "stalled": False},
                ],
            }
        )
    return {
        "pairs": pairs,
        "summary": {
            "completed_games": games,
            "timed_out_games": 0,
            "stalled_games": 0,
            "candidate_exact_budget": True,
            "score_share": {"mean": share, "ci95_low": low, "ci95_high": 0.68},
            "final_score_margin": {"mean": margin, "ci95_low": 0.1, "ci95_high": 2.0},
        },
    }


def test_parser_freezes_fresh_output_and_cpu() -> None:
    config = config_from_args(build_parser().parse_args([]))
    assert config.output_dir == Path("outputs/deadlock_validated_rolling_horizon_042")
    assert config.stage == "all"
    assert config.device == "cpu"


def test_clean_forced_pass_deadlock_is_valid_but_cutoffs_are_not() -> None:
    payload = _block(4)
    payload["pairs"][0]["games"][0]["termination_reason"] = "forced_pass_deadlock"
    _add_corrected_integrity(payload)
    assert payload["summary"]["valid_forced_pass_deadlocks"] == 1
    assert payload["summary"]["invalid_termination_games"] == 0
    assert all(_strength_gate(payload, 4).values())

    payload = _block(4)
    payload["pairs"][0]["games"][0].update(
        {"termination_reason": "repetition_cutoff", "timed_out": True}
    )
    payload["summary"]["timed_out_games"] = 1
    _add_corrected_integrity(payload)
    gate = _strength_gate(payload, 4)
    assert not gate["zero_invalid_terminations"]
    assert not gate["zero_timeouts"]


def test_fresh_direct_failure_stops_before_broad(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / name for name in ("035.pt", "016.pt", "023.pt")]
    for path in paths:
        path.write_bytes(b"checkpoint")
    calls = []

    def fake_block(config, name, opponent, seed_start, pairs):
        calls.append((name, opponent, seed_start, pairs))
        payload = _block(200, share=0.50, low=0.43, margin=-0.2)
        _add_corrected_integrity(payload)
        return payload

    monkeypatch.setattr(
        "splendor_ai.eval.run_deadlock_validated_rolling_horizon._load_or_run_block",
        fake_block,
    )
    payload = run_gate(
        DeadlockValidatedRollingHorizonConfig(
            checkpoint_035=paths[0],
            checkpoint_016=paths[1],
            checkpoint_023=paths[2],
            output_dir=tmp_path / "out",
        )
    )
    assert calls == [("direct_vs_035", "035", 404000, 100)]
    assert payload["decision"] == "reject_fresh_replication_close_rolling_horizon"


def test_fresh_direct_pass_unlocks_exact_broad_blocks(monkeypatch, tmp_path) -> None:
    paths = [tmp_path / name for name in ("035.pt", "016.pt", "023.pt")]
    for path in paths:
        path.write_bytes(b"checkpoint")
    calls = []

    def fake_block(config, name, opponent, seed_start, pairs):
        calls.append((name, opponent, seed_start, pairs))
        payload = _block(200 if opponent == "035" else 100)
        _add_corrected_integrity(payload)
        return payload

    monkeypatch.setattr(
        "splendor_ai.eval.run_deadlock_validated_rolling_horizon._load_or_run_block",
        fake_block,
    )
    payload = run_gate(
        DeadlockValidatedRollingHorizonConfig(
            checkpoint_035=paths[0],
            checkpoint_016=paths[1],
            checkpoint_023=paths[2],
            output_dir=tmp_path / "out",
        )
    )
    assert calls == [
        ("direct_vs_035", "035", 404000, 100),
        ("broad_vs_shallow", "shallow", 404200, 50),
        ("broad_vs_016", "016", 404300, 50),
        ("broad_vs_023", "023", 404400, 50),
    ]
    assert payload["decision"] == "authorize_score_delta_search_teacher"
