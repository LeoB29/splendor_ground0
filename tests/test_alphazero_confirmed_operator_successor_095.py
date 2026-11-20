from __future__ import annotations

import json
from pathlib import Path

import pytest

import splendor_ai.alphazero.selfplay as selfplay_module
from splendor_ai.alphazero.confirmed_operator_successor import (
    offline_successor_eligibility,
    searched_confirmation_acceptance,
    searched_development_acceptance,
    selfplay_corpus_eligibility,
)
from splendor_ai.alphazero.network import create_initial_checkpoint
from splendor_ai.alphazero.replay import read_replay_jsonl
from splendor_ai.alphazero.search import SearchConfig
from splendor_ai.alphazero.selfplay import SelfPlayConfig, generate_self_play


def _arena(*, seat_zero_wins: int, seat_one_wins: int, games_per_seat: int) -> dict[str, object]:
    rows = []
    for seat, wins in ((0, seat_zero_wins), (1, seat_one_wins)):
        for index in range(games_per_seat):
            rows.append(
                {
                    "seed": 1000 + index,
                    "candidate_seat": seat,
                    "candidate_points": 1.0 if index < wins else 0.0,
                }
            )
    games = 2 * games_per_seat
    return {
        "games": games,
        "rows": rows,
        "score_share": (seat_zero_wins + seat_one_wins) / games,
        "mean_margin": 1.0,
        "termination_reasons": {"completed": games},
    }


def _training(*, value_delta: float = -0.01, phase_delta: float = 0.01) -> dict[str, object]:
    return {
        "validation_loss_delta_child_minus_parent": {
            "objective_loss": -0.1,
            "policy_loss": -0.02,
            "value_loss": value_delta,
        },
        "parent_validation_metrics": {"policy_top_target_agreement": 0.4},
        "child_validation_metrics": {"policy_top_target_agreement": 0.395},
        "validation_slices": {
            "phase_early_lt20": {
                "delta_child_minus_parent": {"value_loss": phase_delta}
            },
            "phase_middle_20_49": {
                "delta_child_minus_parent": {"value_loss": 0.0}
            },
            "phase_late_ge50": {
                "delta_child_minus_parent": {"value_loss": -0.01}
            },
        },
    }


def test_selfplay_gate_requires_exact_clean_dose() -> None:
    summary = {
        "games": 64,
        "positions": 100,
        "simulations": 25_600,
        "termination_reasons": {"completed": 64},
    }
    assert selfplay_corpus_eligibility(summary)["eligible"] is True
    summary["termination_reasons"] = {"completed": 63, "max_turns": 1}
    assert selfplay_corpus_eligibility(summary)["eligible"] is False


def test_offline_gate_enforces_value_and_phase_stability() -> None:
    assert offline_successor_eligibility(_training())["eligible"] is True
    assert offline_successor_eligibility(_training(value_delta=0.0001))["eligible"] is False
    result = offline_successor_eligibility(_training(phase_delta=0.025001))
    assert result["eligible"] is False
    assert result["checks"]["phase_value_degradation_bounded"] is False


def test_searched_gates_require_both_seats_and_strict_confirmation_ci() -> None:
    development = _arena(seat_zero_wins=9, seat_one_wins=9, games_per_seat=16)
    assert searched_development_acceptance(development)["accepted"] is True
    confirmation = _arena(seat_zero_wins=28, seat_one_wins=27, games_per_seat=50)
    assert searched_confirmation_acceptance(
        confirmation, {"ci95_low": 0.500001}
    )["accepted"] is True
    assert searched_confirmation_acceptance(
        confirmation, {"ci95_low": 0.5}
    )["accepted"] is False


def test_selfplay_resume_reconstructs_exact_completed_seed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "generation_000.pt"
    create_initial_checkpoint(checkpoint, seed=95)
    replay = tmp_path / "replay.jsonl"
    progress = tmp_path / "progress.json"
    config = SelfPlayConfig(
        generation=0,
        game_count=2,
        seed_start=9500,
        search=SearchConfig(
            simulations=1,
            mode="debug",
            root_noise_fraction=0.0,
            seed=95,
        ),
        max_turns=1,
    )
    original = selfplay_module._play_one_game
    calls = 0

    def interrupted(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(**kwargs)

    monkeypatch.setattr(selfplay_module, "_play_one_game", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_self_play(
            checkpoint_path=checkpoint,
            device="cpu",
            config=config,
            replay_path=replay,
            progress_path=progress,
            resume=True,
        )
    assert {row.game_seed for row in read_replay_jsonl(replay)} == {9500}

    monkeypatch.setattr(selfplay_module, "_play_one_game", original)
    resumed = generate_self_play(
        checkpoint_path=checkpoint,
        device="cpu",
        config=config,
        replay_path=replay,
        progress_path=progress,
        resume=True,
    )
    assert [game.seed for game in resumed.games] == [9500, 9501]
    assert [row.game_seed for row in read_replay_jsonl(replay)] == [9500, 9501]
    assert json.loads(progress.read_text(encoding="utf-8"))["completed_games"] == 2
