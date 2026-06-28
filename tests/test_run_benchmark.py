import json

from splendor_ai.eval.match import GameResult
from splendor_ai.eval.run_benchmark import (
    _model_bot_for_checkpoint,
    default_head_to_head_output_path,
    default_output_path,
    summarize_head_to_head_games,
    summarize_games,
    write_benchmark_payload,
)


def test_summarize_games_counts_model_wins_with_swapped_seats() -> None:
    payload = summarize_games(
        (
            GameResult(
                seed=10,
                turns=60,
                winner=0,
                final_scores=(15, 12),
                bot_seats=("CheckpointPolicyBot", "GreedyHeuristicBot"),
            ),
            GameResult(
                seed=11,
                turns=62,
                winner=1,
                final_scores=(12, 15),
                bot_seats=("GreedyHeuristicBot", "CheckpointPolicyBot"),
            ),
            GameResult(
                seed=12,
                turns=400,
                winner=None,
                final_scores=(10, 10),
                bot_seats=("CheckpointPolicyBot", "GreedyHeuristicBot"),
                timed_out=True,
                termination_reason="repetition_cutoff",
                repetition_count=4,
                no_progress_streak=18,
            ),
        ),
        opponent_name="greedy",
        seed_start=10,
        max_turns=400,
        device="cuda",
        repetition_limit=4,
        no_progress_limit=60,
    )

    assert payload["opponent"] == "greedy"
    assert payload["games"] == 3
    assert payload["model_wins"] == 2
    assert payload["opponent_wins"] == 0
    assert payload["draws"] == 1
    assert payload["timed_out_games"] == 1
    assert payload["termination_reasons"] == {"completed": 2, "repetition_cutoff": 1}
    assert payload["wins_by_seat"] == [1, 1]
    assert payload["games_detail"][1]["winner_name"] == "CheckpointPolicyBot"
    assert payload["games_detail"][2]["termination_reason"] == "repetition_cutoff"
    assert payload["games_detail"][2]["repetition_count"] == 4


def test_summarize_games_counts_legal_action_scorer_wins_as_model_wins() -> None:
    payload = summarize_games(
        (
            GameResult(
                seed=20,
                turns=54,
                winner=0,
                final_scores=(15, 8),
                bot_seats=("LegalActionScorerBot", "RandomLegalBot"),
            ),
            GameResult(
                seed=21,
                turns=62,
                winner=1,
                final_scores=(6, 15),
                bot_seats=("RandomLegalBot", "LegalActionScorerBot"),
            ),
        ),
        opponent_name="random",
        seed_start=20,
        max_turns=400,
        device="cpu",
    )

    assert payload["model_wins"] == 2
    assert payload["opponent_wins"] == 0
    assert payload["games_detail"][0]["winner_name"] == "LegalActionScorerBot"


def test_summarize_head_to_head_counts_candidate_by_seat_with_swapped_seats() -> None:
    payload = summarize_head_to_head_games(
        (
            GameResult(
                seed=30,
                turns=58,
                winner=0,
                final_scores=(15, 9),
                bot_seats=("LegalActionScorerBot", "CheckpointPolicyBot"),
                loop_fallback_triggers_by_seat=(0, 1),
            ),
            GameResult(
                seed=31,
                turns=66,
                winner=0,
                final_scores=(15, 13),
                bot_seats=("CheckpointPolicyBot", "LegalActionScorerBot"),
                loop_fallback_triggers_by_seat=(2, 0),
            ),
            GameResult(
                seed=32,
                turns=400,
                winner=None,
                final_scores=(12, 12),
                bot_seats=("LegalActionScorerBot", "CheckpointPolicyBot"),
                timed_out=True,
                termination_reason="max_turns",
            ),
        ),
        candidate_label="candidate:legal_action_scorer",
        baseline_label="baseline:flat_policy",
        seed_start=30,
        max_turns=400,
        device="cpu",
        repetition_limit=4,
        no_progress_limit=60,
        swap_seats=True,
    )

    assert payload["candidate_wins"] == 1
    assert payload["baseline_wins"] == 1
    assert payload["draws"] == 1
    assert payload["timed_out_games"] == 1
    assert payload["termination_reasons"] == {"completed": 2, "max_turns": 1}
    assert payload["candidate_loop_fallback_triggers"] == 0
    assert payload["baseline_loop_fallback_triggers"] == 3
    assert payload["games_detail"][0]["winner_role"] == "candidate"
    assert payload["games_detail"][1]["winner_role"] == "baseline"
    assert payload["games_detail"][1]["candidate_seat"] == 1
    assert payload["games_detail"][1]["baseline_seat"] == 0
    assert payload["games_detail"][2]["winner_role"] is None


def test_model_bot_for_checkpoint_selects_requested_model_kind(monkeypatch, tmp_path) -> None:
    captured = []

    class FakeFlatBot:
        def __init__(self, checkpoint_path, device, loop_fallback):
            captured.append(("flat", checkpoint_path, device, loop_fallback))

    class FakeScorerBot:
        def __init__(self, checkpoint_path, device):
            captured.append(("scorer", checkpoint_path, device))

    monkeypatch.setattr("splendor_ai.eval.run_benchmark.CheckpointPolicyBot", FakeFlatBot)
    monkeypatch.setattr("splendor_ai.eval.run_benchmark.LegalActionScorerBot", FakeScorerBot)

    checkpoint_path = tmp_path / "checkpoint.pt"
    flat_bot = _model_bot_for_checkpoint(
        checkpoint_path=checkpoint_path,
        model_kind="flat_policy",
        device="cpu",
        loop_fallback=object(),
    )
    scorer_bot = _model_bot_for_checkpoint(
        checkpoint_path=checkpoint_path,
        model_kind="legal_action_scorer",
        device="cpu",
        loop_fallback=object(),
    )

    assert isinstance(flat_bot, FakeFlatBot)
    assert isinstance(scorer_bot, FakeScorerBot)
    assert captured[0][0] == "flat"
    assert captured[1] == ("scorer", checkpoint_path, "cpu")


def test_default_output_path_includes_parent_and_checkpoint_stem() -> None:
    path = default_output_path("outputs/warmstart_search_001/supervised_policy_value.pt")

    assert path.as_posix() == "outputs/benchmarks/warmstart_search_001_supervised_policy_value_benchmark.json"


def test_default_head_to_head_output_path_includes_both_checkpoints() -> None:
    path = default_head_to_head_output_path(
        "outputs/legal_action_scorer_001/legal_action_scorer_best.pt",
        "outputs/policy_improve_margin500_002/supervised_policy_value.pt",
    )

    assert (
        path.as_posix()
        == "outputs/benchmarks/legal_action_scorer_001_legal_action_scorer_best_vs_"
        "policy_improve_margin500_002_supervised_policy_value_benchmark.json"
    )


def test_write_benchmark_payload_writes_json(tmp_path) -> None:
    output_path = tmp_path / "benchmarks" / "result.json"
    payload = {"checkpoint": "checkpoint.pt", "opponents": []}

    written_path = write_benchmark_payload(output_path, payload)

    assert written_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
