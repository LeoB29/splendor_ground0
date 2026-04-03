"""Replay corpus generation utilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from splendor_ai.bots.base import Bot
from splendor_ai.training.replay import (
    ReplayGame,
    collect_game_replay,
    export_replay_games_jsonl,
    export_stalled_traces_jsonl,
    export_timed_out_traces_jsonl,
)


BotFactory = Callable[[], Bot]
BotPairing = tuple[BotFactory, BotFactory]


@dataclass(frozen=True, slots=True)
class CorpusSummary:
    games: int
    total_steps: int
    seat0_wins: int
    seat1_wins: int
    draws: int
    stalled_games: int
    timed_out_games: int
    average_turns: float
    average_final_score_seat0: float
    average_final_score_seat1: float


def generate_replay_corpus(
    games: int,
    bot_seat_0_factory: BotFactory | None = None,
    bot_seat_1_factory: BotFactory | None = None,
    seed_start: int = 0,
    max_turns: int = 400,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
    progress_callback: Callable[[int, int, ReplayGame, float], None] | None = None,
    pairings: tuple[BotPairing, ...] = (),
    swap_seats: bool = False,
) -> tuple[tuple[ReplayGame, ...], CorpusSummary]:
    if pairings:
        pairing_schedule = pairings
    elif bot_seat_0_factory is not None and bot_seat_1_factory is not None:
        pairing_schedule = ((bot_seat_0_factory, bot_seat_1_factory),)
    else:
        raise ValueError("Either seat factories or at least one bot pairing must be provided.")

    replay_games: list[ReplayGame] = []
    start_time = time.perf_counter()

    for game_index in range(games):
        pairing_index = game_index % len(pairing_schedule)
        cycle_index = game_index // len(pairing_schedule)
        seat0_factory, seat1_factory = pairing_schedule[pairing_index]
        if swap_seats and (cycle_index % 2 == 1):
            seat0_factory, seat1_factory = seat1_factory, seat0_factory

        replay_game = collect_game_replay(
            bot_seat_0=seat0_factory(),
            bot_seat_1=seat1_factory(),
            seed=seed_start + game_index,
            max_turns=max_turns,
            repetition_limit=repetition_limit,
            no_progress_limit=no_progress_limit,
        )
        replay_games.append(replay_game)
        if progress_callback is not None:
            progress_callback(
                game_index + 1,
                games,
                replay_game,
                time.perf_counter() - start_time,
            )

    summary = summarize_replay_games(tuple(replay_games))
    return tuple(replay_games), summary


def summarize_replay_games(games: tuple[ReplayGame, ...]) -> CorpusSummary:
    if not games:
        return CorpusSummary(
            games=0,
            total_steps=0,
            seat0_wins=0,
            seat1_wins=0,
            draws=0,
            stalled_games=0,
            timed_out_games=0,
            average_turns=0.0,
            average_final_score_seat0=0.0,
            average_final_score_seat1=0.0,
        )

    seat0_wins = sum(1 for game in games if game.winner == 0)
    seat1_wins = sum(1 for game in games if game.winner == 1)
    draws = sum(1 for game in games if game.winner is None)
    stalled_games = sum(1 for game in games if game.stalled)
    timed_out_games = sum(1 for game in games if game.timed_out)
    total_steps = sum(len(game.steps) for game in games)
    average_turns = sum(game.turns for game in games) / len(games)
    average_final_score_seat0 = sum(game.final_scores[0] for game in games) / len(games)
    average_final_score_seat1 = sum(game.final_scores[1] for game in games) / len(games)

    return CorpusSummary(
        games=len(games),
        total_steps=total_steps,
        seat0_wins=seat0_wins,
        seat1_wins=seat1_wins,
        draws=draws,
        stalled_games=stalled_games,
        timed_out_games=timed_out_games,
        average_turns=average_turns,
        average_final_score_seat0=average_final_score_seat0,
        average_final_score_seat1=average_final_score_seat1,
    )


def write_replay_corpus(
    output_dir: str | Path,
    games: tuple[ReplayGame, ...],
    summary: CorpusSummary,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    replay_path = output_path / "replays.jsonl"
    summary_path = output_path / "summary.json"
    stalled_trace_path = output_path / "stalled_traces.jsonl"
    timed_out_trace_path = output_path / "timed_out_traces.jsonl"

    export_replay_games_jsonl(replay_path, games)
    stalled_trace_count = export_stalled_traces_jsonl(stalled_trace_path, games)
    timed_out_trace_count = export_timed_out_traces_jsonl(timed_out_trace_path, games)
    summary_payload = {
        "games": summary.games,
        "total_steps": summary.total_steps,
        "seat0_wins": summary.seat0_wins,
        "seat1_wins": summary.seat1_wins,
        "draws": summary.draws,
        "stalled_games": summary.stalled_games,
        "stalled_rate": (summary.stalled_games / summary.games) if summary.games else 0.0,
        "timed_out_games": summary.timed_out_games,
        "timed_out_rate": (summary.timed_out_games / summary.games) if summary.games else 0.0,
        "average_turns": summary.average_turns,
        "average_final_score_seat0": summary.average_final_score_seat0,
        "average_final_score_seat1": summary.average_final_score_seat1,
        "stalled_trace_path": stalled_trace_path.name if stalled_trace_count else None,
        "timed_out_trace_path": timed_out_trace_path.name if timed_out_trace_count else None,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return replay_path, summary_path
