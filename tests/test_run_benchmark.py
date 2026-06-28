import json

from splendor_ai.eval.match import GameResult
from splendor_ai.eval.run_benchmark import (
    _opponent_factory,
    _model_bot_for_checkpoint,
    benchmark_checkpoint,
    build_parser,
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


def test_parser_accepts_search_opponent() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint.pt",
            "--opponents",
            "search",
            "--search-depth",
            "3",
            "--search-max-branching",
            "12",
            "--search-buy-branching",
            "7",
            "--search-reserve-branching",
            "4",
            "--search-take-branching",
            "2",
        ]
    )

    assert args.opponents == ["search"]
    assert args.search_depth == 3
    assert args.search_max_branching == 12
    assert args.search_buy_branching == 7
    assert args.search_reserve_branching == 4
    assert args.search_take_branching == 2


def test_opponent_factory_builds_search_bot_with_requested_settings(monkeypatch) -> None:
    captured = []

    class FakeSearchBot:
        def __init__(
            self,
            depth,
            max_branching,
            max_buy_actions,
            max_reserve_actions,
            max_take_actions,
            seed,
        ):
            captured.append(
                {
                    "depth": depth,
                    "max_branching": max_branching,
                    "max_buy_actions": max_buy_actions,
                    "max_reserve_actions": max_reserve_actions,
                    "max_take_actions": max_take_actions,
                    "seed": seed,
                }
            )

    monkeypatch.setattr("splendor_ai.eval.run_benchmark.ShallowSearchBot", FakeSearchBot)

    factory = _opponent_factory(
        "search",
        seed=123,
        search_depth=3,
        search_max_branching=12,
        search_buy_branching=7,
        search_reserve_branching=4,
        search_take_branching=2,
    )
    bot = factory()

    assert isinstance(bot, FakeSearchBot)
    assert captured == [
        {
            "depth": 3,
            "max_branching": 12,
            "max_buy_actions": 7,
            "max_reserve_actions": 4,
            "max_take_actions": 2,
            "seed": 123,
        }
    ]


def test_benchmark_checkpoint_records_search_settings(monkeypatch, tmp_path) -> None:
    def fake_benchmark_opponent(**kwargs):
        return {
            "opponent": kwargs["opponent_name"],
            "games": kwargs["games"],
            "model_wins": 0,
            "opponent_wins": 0,
            "draws": 0,
            "timed_out_games": 0,
        }

    monkeypatch.setattr("splendor_ai.eval.run_benchmark._benchmark_opponent", fake_benchmark_opponent)

    payload = benchmark_checkpoint(
        checkpoint_path=tmp_path / "checkpoint.pt",
        opponents=("search",),
        search_depth=3,
        search_max_branching=12,
        search_buy_branching=7,
        search_reserve_branching=4,
        search_take_branching=2,
    )

    assert payload["search"] == {
        "depth": 3,
        "max_branching": 12,
        "buy_branching": 7,
        "reserve_branching": 4,
        "take_branching": 2,
    }
    assert payload["opponents"][0]["opponent"] == "search"


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
