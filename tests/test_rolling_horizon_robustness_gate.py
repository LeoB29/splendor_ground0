from __future__ import annotations

from pathlib import Path

from splendor_ai.eval.run_rolling_horizon_robustness import (
    BROAD_BLOCKS,
    DIRECT_BLOCKS,
    RollingHorizonRobustnessConfig,
    _direct_gate,
    _integrity_gate,
    build_parser,
    config_from_args,
    run_gate,
)


def _block(pairs: int, share: float = 0.60, seed_start: int = 0) -> dict:
    candidate_sweeps = round(pairs * share)
    rows = []
    wins = losses = 0
    for offset, seed in enumerate(range(seed_start, seed_start + pairs)):
        candidate_wins = offset < candidate_sweeps
        if candidate_wins:
            games = [
                _game(0, (16, 10)),
                _game(1, (10, 16)),
            ]
            wins += 2
        else:
            games = [
                _game(1, (10, 16)),
                _game(0, (16, 10)),
            ]
            losses += 2
        rows.append(
            {
                "seed": seed,
                "candidate_seats": [0, 1],
                "games": games,
                "candidate_telemetry": {
                    "planning_decisions": 10,
                    "sequence_evaluations": 320,
                    "simulated_actions": 900,
                },
            }
        )
    margin = 6.0 if share > 0.50 else -6.0 if share < 0.50 else 0.0
    return {
        "elapsed_seconds": 1.0,
        "pairs": rows,
        "summary": {
            "completed_games": pairs * 2,
            "wins": wins,
            "losses": losses,
            "draws": 0,
            "score_share": {"mean": share, "ci95_low": share - 0.04, "ci95_high": share + 0.04},
            "final_score_margin": {"mean": margin, "ci95_low": margin - 1, "ci95_high": margin + 1},
            "timed_out_games": 0,
            "stalled_games": 0,
            "invalid_termination_games": 0,
            "candidate_exact_budget": True,
        },
    }


def _game(winner: int, scores: tuple[int, int]) -> dict:
    return {
        "winner": winner,
        "final_scores": scores,
        "turns": 60,
        "timed_out": False,
        "stalled": False,
        "termination_reason": "completed",
    }


def _config(tmp_path: Path, stage: str = "all") -> RollingHorizonRobustnessConfig:
    paths = [tmp_path / name for name in ("035.pt", "016.pt", "023.pt")]
    for path in paths:
        path.write_bytes(b"checkpoint")
    return RollingHorizonRobustnessConfig(
        checkpoint_035=paths[0],
        checkpoint_016=paths[1],
        checkpoint_023=paths[2],
        output_dir=tmp_path / "out",
        stage=stage,
        bootstrap_samples=20,
    )


def test_parser_and_seed_blocks_are_frozen() -> None:
    config = config_from_args(build_parser().parse_args([]))
    assert config.output_dir == Path("outputs/rolling_horizon_robustness_043")
    assert config.stage == "all"
    assert DIRECT_BLOCKS == (
        ("A", 431000),
        ("B", 443000),
        ("C", 457000),
        ("D", 469000),
        ("E", 487000),
    )
    assert BROAD_BLOCKS == (
        ("broad_vs_shallow", "shallow", 510000),
        ("broad_vs_016", "016", 520000),
        ("broad_vs_023", "023", 530000),
    )


def test_consistency_requires_four_blocks_above_half() -> None:
    blocks = {
        label: _block(100, share)
        for (label, _seed), share in zip(DIRECT_BLOCKS, (0.8, 0.8, 0.8, 0.2, 0.2), strict=True)
    }
    aggregate = _block(500, 0.56)
    block_gates = {label: _integrity_gate(block, 200) for label, block in blocks.items()}
    gate = _direct_gate(aggregate, blocks, block_gates)
    assert gate["aggregate_score_share_at_least_55pct"]
    assert not gate["at_least_four_of_five_blocks_above_50pct"]


def test_direct_failure_still_runs_all_five_blocks_and_stops_broad(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_block(config, name, opponent, seed_start, pairs):
        calls.append((name, opponent, seed_start, pairs))
        return _block(pairs, 0.48, seed_start)

    monkeypatch.setattr(
        "splendor_ai.eval.run_rolling_horizon_robustness._load_or_run_block",
        fake_block,
    )
    payload = run_gate(_config(tmp_path))
    assert calls == [
        (f"direct_block_{label.lower()}", "035", seed_start, 100)
        for label, seed_start in DIRECT_BLOCKS
    ]
    assert payload["decision"] == "reject_final_robustness_close_rolling_horizon"


def test_direct_pass_unlocks_exact_broad_blocks(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_block(config, name, opponent, seed_start, pairs):
        calls.append((name, opponent, seed_start, pairs))
        return _block(pairs, 0.60, seed_start)

    monkeypatch.setattr(
        "splendor_ai.eval.run_rolling_horizon_robustness._load_or_run_block",
        fake_block,
    )
    payload = run_gate(_config(tmp_path))
    assert calls == [
        *[
            (f"direct_block_{label.lower()}", "035", seed_start, 100)
            for label, seed_start in DIRECT_BLOCKS
        ],
        *[(name, opponent, seed_start, 50) for name, opponent, seed_start in BROAD_BLOCKS],
    ]
    assert payload["decision"] == "authorize_score_delta_search_teacher"
