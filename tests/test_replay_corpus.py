import json

from splendor_ai.bots import GreedyHeuristicBot, RandomLegalBot
from splendor_ai.training import (
    CorpusSummary,
    ReplayGame,
    ReplayStep,
    generate_replay_corpus,
    write_replay_corpus,
)


def test_generate_replay_corpus_returns_games_and_summary() -> None:
    games, summary = generate_replay_corpus(
        bot_seat_0_factory=lambda: GreedyHeuristicBot(seed=1),
        bot_seat_1_factory=lambda: RandomLegalBot(seed=2),
        games=2,
        seed_start=7,
        max_turns=400,
    )

    assert len(games) == 2
    assert summary.games == 2
    assert summary.total_steps == sum(len(game.steps) for game in games)
    assert summary.stalled_games == sum(1 for game in games if game.stalled)
    assert summary.timed_out_games == sum(1 for game in games if game.timed_out)


def test_write_replay_corpus_writes_jsonl_and_summary(tmp_path) -> None:
    games, summary = generate_replay_corpus(
        bot_seat_0_factory=lambda: GreedyHeuristicBot(seed=3),
        bot_seat_1_factory=lambda: RandomLegalBot(seed=4),
        games=1,
        seed_start=9,
        max_turns=400,
    )

    replay_path, summary_path = write_replay_corpus(tmp_path, games, summary)

    assert replay_path.exists()
    assert summary_path.exists()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["games"] == 1
    assert "stalled_rate" in summary_payload
    assert "timed_out_rate" in summary_payload

    lines = replay_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(games[0].steps)


def test_generate_replay_corpus_can_swap_seats_across_pairings() -> None:
    games, summary = generate_replay_corpus(
        games=2,
        seed_start=12,
        max_turns=400,
        pairings=((lambda: GreedyHeuristicBot(seed=5), lambda: RandomLegalBot(seed=6)),),
        swap_seats=True,
    )

    assert summary.games == 2
    assert games[0].bot_seats != games[1].bot_seats


def test_write_replay_corpus_writes_stalled_trace_file(tmp_path) -> None:
    stalled_game = ReplayGame(
        seed=21,
        winner=0,
        final_scores=(15, 13),
        turns=42,
        stalled=True,
        timed_out=False,
        termination_reason="stalled",
        bot_seats=("GreedyHeuristicBot", "RandomLegalBot"),
        final_state_snapshot={"current_player": 1, "bank_tokens": {"red": 0}},
        steps=(
            ReplayStep(
                seed=21,
                turn_index=0,
                player_id=0,
                action_index=7,
                action_payload={"action_type": "TAKE_TOKENS"},
                legal_action_indices=(1, 7),
                observation_vector=(0.0,) * 256,
                final_value=1.0,
                winner=0,
            ),
        ),
    )

    replay_path, summary_path = write_replay_corpus(
        tmp_path,
        (stalled_game,),
        summary=CorpusSummary(
            games=1,
            total_steps=1,
            seat0_wins=1,
            seat1_wins=0,
            draws=0,
            stalled_games=1,
            timed_out_games=0,
            average_turns=42.0,
            average_final_score_seat0=15.0,
            average_final_score_seat1=13.0,
        ),
    )

    stalled_trace_path = replay_path.parent / "stalled_traces.jsonl"
    assert summary_path.exists()
    assert stalled_trace_path.exists()

    trace_payload = json.loads(stalled_trace_path.read_text(encoding="utf-8").strip())
    assert trace_payload["game_seed"] == 21
    assert trace_payload["final_state_snapshot"]["current_player"] == 1
    assert trace_payload["game_termination_reason"] == "stalled"
    assert trace_payload["steps"][0]["action_payload"]["action_type"] == "TAKE_TOKENS"


def test_write_replay_corpus_writes_timed_out_trace_file(tmp_path) -> None:
    timed_out_game = ReplayGame(
        seed=22,
        winner=1,
        final_scores=(11, 12),
        turns=400,
        stalled=False,
        timed_out=True,
        termination_reason="max_turns",
        bot_seats=("RandomLegalBot", "GreedyHeuristicBot"),
        final_state_snapshot={"current_player": 0, "turn_index": 400},
        steps=(
            ReplayStep(
                seed=22,
                turn_index=399,
                player_id=1,
                action_index=8,
                action_payload={"action_type": "BUY_VISIBLE"},
                legal_action_indices=(8, 9),
                observation_vector=(0.0,) * 256,
                final_value=1.0,
                winner=1,
            ),
        ),
    )

    replay_path, summary_path = write_replay_corpus(
        tmp_path,
        (timed_out_game,),
        summary=CorpusSummary(
            games=1,
            total_steps=1,
            seat0_wins=0,
            seat1_wins=1,
            draws=0,
            stalled_games=0,
            timed_out_games=1,
            average_turns=400.0,
            average_final_score_seat0=11.0,
            average_final_score_seat1=12.0,
        ),
    )

    timed_out_trace_path = replay_path.parent / "timed_out_traces.jsonl"
    assert summary_path.exists()
    assert timed_out_trace_path.exists()

    trace_payload = json.loads(timed_out_trace_path.read_text(encoding="utf-8").strip())
    assert trace_payload["game_seed"] == 22
    assert trace_payload["game_termination_reason"] == "max_turns"
    assert trace_payload["final_state_snapshot"]["turn_index"] == 400
    assert trace_payload["steps"][0]["action_payload"]["action_type"] == "BUY_VISIBLE"
