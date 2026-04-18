import json

from splendor_ai.eval.match import GameResult
from splendor_ai.eval.run_benchmark import default_output_path, summarize_games, write_benchmark_payload


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
            ),
        ),
        opponent_name="greedy",
        seed_start=10,
        max_turns=400,
        device="cuda",
    )

    assert payload["opponent"] == "greedy"
    assert payload["games"] == 3
    assert payload["model_wins"] == 2
    assert payload["opponent_wins"] == 0
    assert payload["draws"] == 1
    assert payload["timed_out_games"] == 1
    assert payload["wins_by_seat"] == [1, 1]
    assert payload["games_detail"][1]["winner_name"] == "CheckpointPolicyBot"


def test_default_output_path_includes_parent_and_checkpoint_stem() -> None:
    path = default_output_path("outputs/warmstart_search_001/supervised_policy_value.pt")

    assert path.as_posix() == "outputs/benchmarks/warmstart_search_001_supervised_policy_value_benchmark.json"


def test_write_benchmark_payload_writes_json(tmp_path) -> None:
    output_path = tmp_path / "benchmarks" / "result.json"
    payload = {"checkpoint": "checkpoint.pt", "opponents": []}

    written_path = write_benchmark_payload(output_path, payload)

    assert written_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
