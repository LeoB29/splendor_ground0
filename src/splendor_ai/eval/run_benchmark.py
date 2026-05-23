"""CLI entry point for checkpoint-vs-baseline benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

from splendor_ai.bots import CheckpointPolicyBot, GreedyHeuristicBot, LoopFallbackConfig, RandomLegalBot
from splendor_ai.bots.base import Bot

from .match import GameResult, MatchConfig, play_game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a checkpoint policy bot against baseline bots.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved policy/value checkpoint.")
    parser.add_argument("--device", default="cpu", help="Device used by the checkpoint policy bot.")
    parser.add_argument("--games", type=int, default=50, help="Games per opponent.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("random", "greedy"),
        default=("random", "greedy"),
        help="Baseline opponents to benchmark against.",
    )
    parser.add_argument("--seed-start", type=int, default=0, help="First game seed.")
    parser.add_argument("--max-turns", type=int, default=400, help="Maximum turns per game.")
    parser.add_argument(
        "--repetition-limit",
        type=int,
        default=4,
        help="Optional repeated-state cutoff used for loop diagnostics. Set to 0 to disable.",
    )
    parser.add_argument(
        "--no-progress-limit",
        type=int,
        default=60,
        help="Optional no-progress cutoff used for loop diagnostics. Set to 0 to disable.",
    )
    parser.add_argument(
        "--no-swap-seats",
        action="store_true",
        help="Disable alternating seats across games.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional JSON output path. Defaults to outputs/benchmarks/<checkpoint_stem>_benchmark.json.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print progress every N games. Set to 0 to only log first/last game.",
    )
    parser.add_argument(
        "--disable-loop-fallback",
        action="store_true",
        help="Disable checkpoint inference fallback for repeated/no-progress loops.",
    )
    parser.add_argument(
        "--loop-fallback-min-state-visits",
        type=int,
        default=2,
        help="Repeated state visits needed before the loop fallback can choose a buy.",
    )
    parser.add_argument(
        "--loop-fallback-min-own-no-progress-actions",
        type=int,
        default=6,
        help="Consecutive non-progress checkpoint actions needed before fallback can choose a buy.",
    )
    parser.add_argument(
        "--loop-fallback-max-buy-logit-gap",
        type=float,
        default=8.0,
        help="Maximum legal-top-minus-buy logit gap allowed for fallback buy selection.",
    )
    return parser


def benchmark_checkpoint(
    checkpoint_path: str | Path,
    opponents: tuple[str, ...],
    device: str = "cpu",
    games: int = 50,
    seed_start: int = 0,
    max_turns: int = 400,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
    swap_seats: bool = True,
    log_every: int = 5,
    loop_fallback: LoopFallbackConfig | None = None,
) -> dict[str, object]:
    checkpoint = Path(checkpoint_path)
    fallback_config = loop_fallback or LoopFallbackConfig()
    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "device": device,
        "games_per_opponent": games,
        "seed_start": seed_start,
        "max_turns": max_turns,
        "repetition_limit": repetition_limit,
        "no_progress_limit": no_progress_limit,
        "swap_seats": swap_seats,
        "loop_fallback": {
            "enabled": fallback_config.enabled,
            "min_state_visits": fallback_config.min_state_visits,
            "min_own_non_progress_actions": fallback_config.min_own_non_progress_actions,
            "max_buy_logit_gap": fallback_config.max_buy_logit_gap,
        },
        "opponents": [],
    }

    for opponent_name in opponents:
        opponent_payload = _benchmark_opponent(
            checkpoint_path=checkpoint,
            opponent_name=opponent_name,
            device=device,
            games=games,
            seed_start=seed_start,
            max_turns=max_turns,
            repetition_limit=repetition_limit,
            no_progress_limit=no_progress_limit,
            swap_seats=swap_seats,
            log_every=log_every,
            loop_fallback=fallback_config,
        )
        payload["opponents"].append(opponent_payload)

    return payload


def summarize_games(
    games: tuple[GameResult, ...],
    opponent_name: str,
    seed_start: int,
    max_turns: int,
    device: str,
    repetition_limit: int = 0,
    no_progress_limit: int = 0,
) -> dict[str, object]:
    model_wins = 0
    opponent_wins = 0
    draws = 0
    timed_out = 0
    stalled = 0
    termination_reasons: dict[str, int] = {}
    per_game: list[dict[str, object]] = []
    model_loop_fallback_triggers = 0

    for game in games:
        winner_name = None
        game_model_loop_fallback_triggers = sum(
            game.loop_fallback_triggers_by_seat[seat]
            for seat, bot_name in enumerate(game.bot_seats)
            if bot_name == "CheckpointPolicyBot"
        )
        model_loop_fallback_triggers += game_model_loop_fallback_triggers
        if game.winner is None:
            draws += 1
        else:
            winner_name = game.bot_seats[game.winner]
            if winner_name == "CheckpointPolicyBot":
                model_wins += 1
            else:
                opponent_wins += 1
        if game.timed_out:
            timed_out += 1
        if game.stalled:
            stalled += 1
        termination_reasons[game.termination_reason] = termination_reasons.get(game.termination_reason, 0) + 1
        per_game.append(
            {
                "seed": game.seed,
                "turns": game.turns,
                "winner": game.winner,
                "winner_name": winner_name,
                "final_scores": list(game.final_scores),
                "bot_seats": list(game.bot_seats),
                "stalled": game.stalled,
                "timed_out": game.timed_out,
                "termination_reason": game.termination_reason,
                "repetition_count": game.repetition_count,
                "no_progress_streak": game.no_progress_streak,
                "loop_fallback_triggers_by_seat": list(game.loop_fallback_triggers_by_seat),
                "model_loop_fallback_triggers": game_model_loop_fallback_triggers,
            }
        )

    return {
        "opponent": opponent_name,
        "games": len(games),
        "model_wins": model_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "timed_out_games": timed_out,
        "stalled_games": stalled,
        "termination_reasons": termination_reasons,
        "wins_by_seat": [
            sum(1 for game in games if game.winner == 0),
            sum(1 for game in games if game.winner == 1),
        ],
        "seed_start": seed_start,
        "max_turns": max_turns,
        "repetition_limit": repetition_limit,
        "no_progress_limit": no_progress_limit,
        "device": device,
        "model_loop_fallback_triggers": model_loop_fallback_triggers,
        "games_detail": per_game,
    }


def default_output_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return Path("outputs") / "benchmarks" / f"{checkpoint.parent.name}_{checkpoint.stem}_benchmark.json"


def write_benchmark_payload(path: str | Path, payload: dict[str, object]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _benchmark_opponent(
    checkpoint_path: Path,
    opponent_name: str,
    device: str,
    games: int,
    seed_start: int,
    max_turns: int,
    repetition_limit: int,
    no_progress_limit: int,
    swap_seats: bool,
    log_every: int,
    loop_fallback: LoopFallbackConfig,
) -> dict[str, object]:
    opponent_factory = _opponent_factory(opponent_name, seed_start)
    played_games: list[GameResult] = []
    model_wins = 0
    opponent_wins = 0
    draws = 0
    timed_out = 0
    start_time = time.perf_counter()

    print(
        f"[benchmark] checkpoint={checkpoint_path} opponent={opponent_name}"
        f" games={games} device={device} swap_seats={swap_seats}"
    )
    model_bot = CheckpointPolicyBot(checkpoint_path, device=device, loop_fallback=loop_fallback)
    for game_index in range(games):
        swap = swap_seats and game_index % 2 == 1
        opponent_bot = opponent_factory()
        seat0 = opponent_bot if swap else model_bot
        seat1 = model_bot if swap else opponent_bot
        game = play_game(
            bot_seat_0=seat0,
            bot_seat_1=seat1,
            seed=seed_start + game_index,
            max_turns=max_turns,
            repetition_limit=repetition_limit,
            no_progress_limit=no_progress_limit,
        )
        played_games.append(game)

        if game.winner is None:
            draws += 1
        elif game.bot_seats[game.winner] == "CheckpointPolicyBot":
            model_wins += 1
        else:
            opponent_wins += 1
        if game.timed_out:
            timed_out += 1

        done = game_index + 1
        if done == 1 or done == games or (log_every > 0 and done % log_every == 0):
            elapsed = time.perf_counter() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = ((games - done) / rate) if rate > 0 else math.inf
            eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
            print(
                f"[benchmark:{opponent_name}] {done}/{games}"
                f" elapsed={elapsed:.1f}s"
                f" rate={rate:.2f} games/s"
                f" eta={eta_text}"
                f" model_wins={model_wins}"
                f" opponent_wins={opponent_wins}"
                f" draws={draws}"
                f" timed_out={timed_out}"
            )

    return summarize_games(
        tuple(played_games),
        opponent_name=opponent_name,
        seed_start=seed_start,
        max_turns=max_turns,
        device=device,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
    )


def _opponent_factory(opponent_name: str, seed: int) -> Callable[[], Bot]:
    if opponent_name == "random":
        return lambda: RandomLegalBot(seed=seed)
    if opponent_name == "greedy":
        return lambda: GreedyHeuristicBot(seed=seed)
    raise ValueError(f"Unsupported opponent: {opponent_name}")


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_path) if args.output_path is not None else default_output_path(args.checkpoint)
    payload = benchmark_checkpoint(
        checkpoint_path=args.checkpoint,
        opponents=tuple(args.opponents),
        device=args.device,
        games=args.games,
        seed_start=args.seed_start,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
        swap_seats=not args.no_swap_seats,
        log_every=args.log_every,
        loop_fallback=LoopFallbackConfig(
            enabled=not args.disable_loop_fallback,
            min_state_visits=args.loop_fallback_min_state_visits,
            min_own_non_progress_actions=args.loop_fallback_min_own_no_progress_actions,
            max_buy_logit_gap=args.loop_fallback_max_buy_logit_gap,
        ),
    )
    written_path = write_benchmark_payload(output_path, payload)
    print(f"[benchmark] wrote results: {written_path}")
    for opponent_payload in payload["opponents"]:
        print(
            "Benchmark:"
            f" opponent={opponent_payload['opponent']}"
            f" model_wins={opponent_payload['model_wins']}"
            f" opponent_wins={opponent_payload['opponent_wins']}"
            f" draws={opponent_payload['draws']}"
            f" timed_out={opponent_payload['timed_out_games']}"
        )


if __name__ == "__main__":
    main()
