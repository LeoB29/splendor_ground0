from splendor_ai.eval.match import GameResult
from splendor_ai.training.run_supervised import _run_benchmarks, _select_benchmark_champion


def test_run_benchmarks_summarizes_match_results(monkeypatch, tmp_path) -> None:
    captured = []

    class FakeCheckpointPolicyBot:
        def __init__(self, checkpoint_path, device):
            self.checkpoint_path = checkpoint_path
            self.device = device

    def fake_play_game(bot_seat_0, bot_seat_1, seed, max_turns, repetition_limit, no_progress_limit):
        captured.append(
            (
                seed,
                max_turns,
                repetition_limit,
                no_progress_limit,
                type(bot_seat_0).__name__,
                type(bot_seat_1).__name__,
            )
        )
        if seed % 2 == 1:
            return GameResult(
                seed=seed,
                turns=30,
                winner=0,
                final_scores=(15, 12),
                bot_seats=("CheckpointPolicyBot", "RandomLegalBot"),
            )
        return GameResult(
            seed=seed,
            turns=32,
            winner=1,
            final_scores=(11, 15),
            bot_seats=("RandomLegalBot", "CheckpointPolicyBot"),
            timed_out=True,
            termination_reason="max_turns",
        )

    monkeypatch.setattr("splendor_ai.training.run_supervised.CheckpointPolicyBot", FakeCheckpointPolicyBot)
    monkeypatch.setattr("splendor_ai.training.run_supervised.play_game", fake_play_game)

    payload = _run_benchmarks(
        checkpoint_path=tmp_path / "checkpoint.pt",
        checkpoint_label="final",
        benchmark_games=2,
        opponents=("random",),
        benchmark_device="cpu",
        benchmark_seed_start=11,
        benchmark_max_turns=200,
        benchmark_repetition_limit=4,
        benchmark_no_progress_limit=60,
        benchmark_log_every=0,
    )

    assert captured == [
        (11, 200, 4, 60, "FakeCheckpointPolicyBot", "RandomLegalBot"),
        (12, 200, 4, 60, "RandomLegalBot", "FakeCheckpointPolicyBot"),
    ]
    assert payload == [
        {
            "checkpoint_label": "final",
            "checkpoint_path": "checkpoint.pt",
            "opponent": "random",
            "games": 2,
            "model_wins": 2,
            "opponent_wins": 0,
            "draws": 0,
            "timed_out_games": 1,
            "stalled_games": 0,
            "termination_reasons": {"completed": 1, "max_turns": 1},
            "wins_by_seat": [1, 1],
            "seed_start": 11,
            "max_turns": 200,
            "repetition_limit": 4,
            "no_progress_limit": 60,
            "max_turns_per_game": 200,
            "device": "cpu",
            "games_detail": [
                {
                    "seed": 11,
                    "turns": 30,
                    "winner": 0,
                    "winner_name": "CheckpointPolicyBot",
                    "final_scores": [15, 12],
                    "bot_seats": ["CheckpointPolicyBot", "RandomLegalBot"],
                    "stalled": False,
                    "timed_out": False,
                    "termination_reason": "completed",
                    "repetition_count": 0,
                    "no_progress_streak": 0,
                },
                {
                    "seed": 12,
                    "turns": 32,
                    "winner": 1,
                    "winner_name": "CheckpointPolicyBot",
                    "final_scores": [11, 15],
                    "bot_seats": ["RandomLegalBot", "CheckpointPolicyBot"],
                    "stalled": False,
                    "timed_out": True,
                    "termination_reason": "max_turns",
                    "repetition_count": 0,
                    "no_progress_streak": 0,
                },
            ],
        }
    ]


def test_select_benchmark_champion_prefers_better_match_score() -> None:
    champion = _select_benchmark_champion(
        [
            {
                "checkpoint_label": "final",
                "checkpoint_path": "supervised_policy_value.pt",
                "benchmarks": [
                    {"model_wins": 4, "opponent_wins": 6},
                    {"model_wins": 0, "opponent_wins": 10},
                ],
            },
            {
                "checkpoint_label": "best_validation",
                "checkpoint_path": "supervised_policy_value_best.pt",
                "benchmarks": [
                    {"model_wins": 6, "opponent_wins": 4},
                    {"model_wins": 0, "opponent_wins": 10},
                ],
            },
        ]
    )

    assert champion is not None
    assert champion["checkpoint_label"] == "best_validation"
