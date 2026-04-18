"""CLI entry point for replay corpus generation."""

from __future__ import annotations

import argparse
import math

from splendor_ai.bots import GreedyHeuristicBot, RandomLegalBot, ShallowSearchBot

from .corpus import generate_replay_corpus, write_replay_corpus


def _build_bot_factory(bot_name: str, seed: int, args: argparse.Namespace):
    normalized = bot_name.lower()
    if normalized == "random":
        return lambda: RandomLegalBot(seed=seed)
    if normalized == "greedy":
        return lambda: GreedyHeuristicBot(seed=seed)
    if normalized == "search":
        return lambda: ShallowSearchBot(
            depth=args.search_depth,
            max_branching=args.search_max_branching,
            max_buy_actions=args.search_buy_branching,
            max_reserve_actions=args.search_reserve_branching,
            max_take_actions=args.search_take_branching,
            seed=seed,
        )
    raise ValueError(f"Unsupported bot name: {bot_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate replay JSONL corpora for supervised warm-start training.")
    parser.add_argument("--output-dir", required=True, help="Directory where replay files should be written.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to generate.")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed for generated games.")
    parser.add_argument("--max-turns", type=int, default=400, help="Per-game turn cap.")
    parser.add_argument(
        "--repetition-limit",
        type=int,
        default=4,
        help="Cut off a game when the same state is seen this many times.",
    )
    parser.add_argument(
        "--no-progress-limit",
        type=int,
        default=60,
        help="Cut off a game after this many consecutive plies without buy/reserve/score progress.",
    )
    parser.add_argument("--seat0-bot", default="greedy", choices=("greedy", "random", "search"))
    parser.add_argument("--seat1-bot", default="random", choices=("greedy", "random", "search"))
    parser.add_argument("--seat0-seed", type=int, default=0)
    parser.add_argument("--seat1-seed", type=int, default=1)
    parser.add_argument("--search-depth", type=int, default=2, help="Search depth for the search bot.")
    parser.add_argument(
        "--search-max-branching",
        type=int,
        default=10,
        help="Absolute per-node action cap for the search bot.",
    )
    parser.add_argument(
        "--search-buy-branching",
        type=int,
        default=6,
        help="Per-node combined cap for buy actions in the search bot.",
    )
    parser.add_argument(
        "--search-reserve-branching",
        type=int,
        default=3,
        help="Per-node combined cap for reserve actions in the search bot.",
    )
    parser.add_argument(
        "--search-take-branching",
        type=int,
        default=3,
        help="Per-node cap for token-take actions in the search bot.",
    )
    parser.add_argument(
        "--pairing",
        action="append",
        default=[],
        help="Optional bot pairing in BOT_A:BOT_B format. Repeat to mix multiple pairings.",
    )
    parser.add_argument(
        "--swap-seats",
        action="store_true",
        help="Alternate seat ownership for each full pairing cycle.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N completed games.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pairings = _resolve_pairings(args)
    pairing_labels = []
    if args.pairing:
        pairing_labels.extend(args.pairing)
    else:
        pairing_labels.append(f"{args.seat0_bot}:{args.seat1_bot}")

    print(
        f"[corpus] output_dir={args.output_dir}"
        f" games={args.games}"
        f" seed_start={args.seed_start}"
        f" max_turns={args.max_turns}"
        f" repetition_limit={args.repetition_limit}"
        f" no_progress_limit={args.no_progress_limit}"
        f" swap_seats={args.swap_seats}"
    )
    print(f"[corpus] pairings={', '.join(pairing_labels)}")
    if any("search" in pairing for pairing in pairing_labels):
        print(
            f"[corpus] search_settings depth={args.search_depth}"
            f" max_branching={args.search_max_branching}"
            f" buy_branching={args.search_buy_branching}"
            f" reserve_branching={args.search_reserve_branching}"
            f" take_branching={args.search_take_branching}"
        )

    stalled_games = 0
    timed_out_games = 0

    def progress_callback(done: int, total: int, replay_game, elapsed: float) -> None:
        nonlocal stalled_games, timed_out_games
        if replay_game.stalled:
            stalled_games += 1
        if replay_game.timed_out:
            timed_out_games += 1
        if done == total or done == 1 or (args.log_every > 0 and done % args.log_every == 0):
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = ((total - done) / rate) if rate > 0 else math.inf
            eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
            print(
                f"[corpus] {done}/{total} games | elapsed={elapsed:.1f}s | "
                f"rate={rate:.2f} games/s | eta={eta_text} | "
                f"last_turns={replay_game.turns} | last_reason={replay_game.termination_reason} | "
                f"stalled={stalled_games} | timed_out={timed_out_games}"
            )

    games, summary = generate_replay_corpus(
        games=args.games,
        seed_start=args.seed_start,
        max_turns=args.max_turns,
        repetition_limit=args.repetition_limit,
        no_progress_limit=args.no_progress_limit,
        progress_callback=progress_callback,
        pairings=pairings,
        swap_seats=args.swap_seats,
    )
    print("[corpus] generation complete, writing replay files...")
    replay_path, summary_path = write_replay_corpus(
        output_dir=args.output_dir,
        games=games,
        summary=summary,
    )
    stalled_trace_path = replay_path.parent / "stalled_traces.jsonl"
    timed_out_trace_path = replay_path.parent / "timed_out_traces.jsonl"
    print(f"wrote replay corpus: {replay_path}")
    print(f"wrote summary: {summary_path}")
    if stalled_trace_path.exists():
        print(f"wrote stalled traces: {stalled_trace_path}")
    if timed_out_trace_path.exists():
        print(f"wrote timed-out traces: {timed_out_trace_path}")
    print(
        f"games={summary.games} steps={summary.total_steps} "
        f"stalled={summary.stalled_games} stalled_rate="
        f"{(summary.stalled_games / summary.games) if summary.games else 0.0:.4f} "
        f"timed_out={summary.timed_out_games} timed_out_rate="
        f"{(summary.timed_out_games / summary.games) if summary.games else 0.0:.4f}"
    )


def _resolve_pairings(args: argparse.Namespace):
    if not args.pairing:
        return (
            (
                _build_bot_factory(args.seat0_bot, args.seat0_seed, args),
                _build_bot_factory(args.seat1_bot, args.seat1_seed, args),
            ),
        )

    pairings = []
    for raw_pairing in args.pairing:
        try:
            left_name, right_name = raw_pairing.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid pairing '{raw_pairing}'. Expected BOT_A:BOT_B."
            ) from exc
        pairings.append(
            (
                _build_bot_factory(left_name, args.seat0_seed, args),
                _build_bot_factory(right_name, args.seat1_seed, args),
            )
        )
    return tuple(pairings)


if __name__ == "__main__":
    main()
