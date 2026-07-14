"""CLI entry point for checkpoint-vs-baseline and checkpoint-vs-checkpoint benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

from splendor_ai.bots import (
    CheckpointPolicyBot,
    CanonicalEntityActionScorerBot,
    CheckpointValueEvaluator,
    GreedyHeuristicBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    LegalActionSetScorerBot,
    LoopFallbackConfig,
    PolicyGuidedSearchBot,
    RandomLegalBot,
    ShallowSearchBot,
)
from splendor_ai.bots.base import Bot

from .match import GameResult, MatchConfig, play_game

MODEL_BOT_NAMES = (
    "CheckpointPolicyBot",
    "LegalActionAttentionScorerBot",
    "LegalActionScorerBot",
    "LegalActionSetScorerBot",
    "CanonicalEntityActionScorerBot",
    "PolicyGuidedSearchBot",
)
MODEL_KIND_CHOICES = (
    "flat_policy",
    "legal_action_attention_scorer",
    "legal_action_scorer",
    "legal_action_set_scorer",
    "canonical_entity_action_scorer",
    "policy_guided_search",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark checkpoint policy bots.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved policy/value checkpoint.")
    parser.add_argument(
        "--model-kind",
        choices=MODEL_KIND_CHOICES,
        default="flat_policy",
        help="Checkpoint architecture to load. Defaults to the original flat policy/value model.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Optional checkpoint to play directly against --checkpoint instead of random/greedy baselines.",
    )
    parser.add_argument(
        "--baseline-model-kind",
        choices=MODEL_KIND_CHOICES,
        default="flat_policy",
        help="Architecture for --baseline-checkpoint.",
    )
    parser.add_argument("--device", default="cpu", help="Device used by the checkpoint policy bot.")
    parser.add_argument("--games", type=int, default=50, help="Games per opponent.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=("random", "greedy", "search"),
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
        "--paired-seeds",
        action="store_true",
        help=(
            "For direct checkpoint-vs-checkpoint matches, play each seed twice "
            "with seats reversed. Requires an even game count and seat swapping."
        ),
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
    parser.add_argument("--search-depth", type=int, default=2, help="Search depth for the search opponent.")
    parser.add_argument(
        "--search-max-branching",
        type=int,
        default=10,
        help="Absolute per-node action cap for the search opponent.",
    )
    parser.add_argument(
        "--search-buy-branching",
        type=int,
        default=6,
        help="Per-node combined cap for buy actions in the search opponent.",
    )
    parser.add_argument(
        "--search-reserve-branching",
        type=int,
        default=3,
        help="Per-node combined cap for reserve actions in the search opponent.",
    )
    parser.add_argument(
        "--search-take-branching",
        type=int,
        default=3,
        help="Per-node cap for token-take actions in the search opponent.",
    )
    parser.add_argument(
        "--policy-prior-weight",
        type=float,
        default=2_000.0,
        help="Heuristic-scale rank-prior weight for --model-kind policy_guided_search.",
    )
    parser.add_argument(
        "--search-value-checkpoint",
        default=None,
        help="Optional value checkpoint blended into search/PGS leaf evaluation.",
    )
    parser.add_argument(
        "--search-value-device",
        default="cpu",
        help="Device used by --search-value-checkpoint.",
    )
    parser.add_argument(
        "--search-value-weight",
        type=float,
        default=0.0,
        help="Leaf blend weight for the learned value model in search/PGS.",
    )
    parser.add_argument(
        "--search-value-scale",
        type=float,
        default=15_000.0,
        help="Multiplier mapping normalized value predictions to heuristic scale.",
    )
    parser.add_argument(
        "--search-value-prediction-mode",
        choices=("value", "outcome", "blend"),
        default="value",
        help="Use scalar value head, outcome P(win)-P(loss), or a blend for search leaves.",
    )
    parser.add_argument(
        "--search-value-outcome-blend-weight",
        type=float,
        default=0.5,
        help="Outcome contribution when --search-value-prediction-mode=blend.",
    )
    parser.add_argument(
        "--search-value-min-turn-index",
        type=int,
        default=None,
        help="Only blend learned value into search leaves at or after this turn index.",
    )
    parser.add_argument(
        "--search-value-max-turn-index",
        type=int,
        default=None,
        help="Only blend learned value into search leaves at or before this turn index.",
    )
    return parser


def benchmark_checkpoint(
    checkpoint_path: str | Path,
    opponents: tuple[str, ...],
    model_kind: str = "flat_policy",
    device: str = "cpu",
    games: int = 50,
    seed_start: int = 0,
    max_turns: int = 400,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
    swap_seats: bool = True,
    log_every: int = 5,
    loop_fallback: LoopFallbackConfig | None = None,
    search_depth: int = 2,
    search_max_branching: int = 10,
    search_buy_branching: int = 6,
    search_reserve_branching: int = 3,
    search_take_branching: int = 3,
    policy_prior_weight: float = 2_000.0,
    search_value_checkpoint: str | Path | None = None,
    search_value_device: str = "cpu",
    search_value_weight: float = 0.0,
    search_value_scale: float = 15_000.0,
    search_value_prediction_mode: str = "value",
    search_value_outcome_blend_weight: float = 0.5,
    search_value_min_turn_index: int | None = None,
    search_value_max_turn_index: int | None = None,
) -> dict[str, object]:
    checkpoint = Path(checkpoint_path)
    fallback_config = loop_fallback or LoopFallbackConfig()
    payload: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "model_kind": model_kind,
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
        "search": {
            "depth": search_depth,
            "max_branching": search_max_branching,
            "buy_branching": search_buy_branching,
            "reserve_branching": search_reserve_branching,
            "take_branching": search_take_branching,
        },
        "policy_guided_search": {
            "prior_weight": policy_prior_weight,
        },
        "value_search": {
            "checkpoint": str(search_value_checkpoint) if search_value_checkpoint is not None else None,
            "device": search_value_device,
            "weight": search_value_weight,
            "scale": search_value_scale,
            "prediction_mode": search_value_prediction_mode,
            "outcome_blend_weight": search_value_outcome_blend_weight,
            "min_turn_index": search_value_min_turn_index,
            "max_turn_index": search_value_max_turn_index,
        },
        "opponents": [],
    }

    for opponent_name in opponents:
        opponent_payload = _benchmark_opponent(
            checkpoint_path=checkpoint,
            model_kind=model_kind,
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
            search_depth=search_depth,
            search_max_branching=search_max_branching,
            search_buy_branching=search_buy_branching,
            search_reserve_branching=search_reserve_branching,
            search_take_branching=search_take_branching,
            policy_prior_weight=policy_prior_weight,
            search_value_checkpoint=search_value_checkpoint,
            search_value_device=search_value_device,
            search_value_weight=search_value_weight,
            search_value_scale=search_value_scale,
            search_value_prediction_mode=search_value_prediction_mode,
            search_value_outcome_blend_weight=search_value_outcome_blend_weight,
            search_value_min_turn_index=search_value_min_turn_index,
            search_value_max_turn_index=search_value_max_turn_index,
        )
        payload["opponents"].append(opponent_payload)

    return payload


def benchmark_head_to_head(
    candidate_checkpoint_path: str | Path,
    baseline_checkpoint_path: str | Path,
    candidate_model_kind: str = "flat_policy",
    baseline_model_kind: str = "flat_policy",
    device: str = "cpu",
    games: int = 50,
    seed_start: int = 0,
    max_turns: int = 400,
    repetition_limit: int = 4,
    no_progress_limit: int = 60,
    swap_seats: bool = True,
    paired_seeds: bool = False,
    log_every: int = 5,
    loop_fallback: LoopFallbackConfig | None = None,
    search_depth: int = 2,
    search_max_branching: int = 10,
    search_buy_branching: int = 6,
    search_reserve_branching: int = 3,
    search_take_branching: int = 3,
    policy_prior_weight: float = 2_000.0,
    search_value_checkpoint: str | Path | None = None,
    search_value_device: str = "cpu",
    search_value_weight: float = 0.0,
    search_value_scale: float = 15_000.0,
    search_value_prediction_mode: str = "value",
    search_value_outcome_blend_weight: float = 0.5,
    search_value_min_turn_index: int | None = None,
    search_value_max_turn_index: int | None = None,
) -> dict[str, object]:
    if paired_seeds and (games <= 0 or games % 2):
        raise ValueError("paired_seeds requires a positive even game count")
    if paired_seeds and not swap_seats:
        raise ValueError("paired_seeds requires seat swapping")
    candidate_checkpoint = Path(candidate_checkpoint_path)
    baseline_checkpoint = Path(baseline_checkpoint_path)
    fallback_config = loop_fallback or LoopFallbackConfig()
    payload: dict[str, object] = {
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_model_kind": candidate_model_kind,
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_model_kind": baseline_model_kind,
        "device": device,
        "games": games,
        "seed_start": seed_start,
        "max_turns": max_turns,
        "repetition_limit": repetition_limit,
        "no_progress_limit": no_progress_limit,
        "swap_seats": swap_seats,
        "paired_seeds": paired_seeds,
        "loop_fallback": {
            "enabled": fallback_config.enabled,
            "min_state_visits": fallback_config.min_state_visits,
            "min_own_non_progress_actions": fallback_config.min_own_non_progress_actions,
            "max_buy_logit_gap": fallback_config.max_buy_logit_gap,
        },
        "search": {
            "depth": search_depth,
            "max_branching": search_max_branching,
            "buy_branching": search_buy_branching,
            "reserve_branching": search_reserve_branching,
            "take_branching": search_take_branching,
        },
        "policy_guided_search": {
            "prior_weight": policy_prior_weight,
        },
        "value_search": {
            "checkpoint": str(search_value_checkpoint) if search_value_checkpoint is not None else None,
            "device": search_value_device,
            "weight": search_value_weight,
            "scale": search_value_scale,
            "prediction_mode": search_value_prediction_mode,
            "outcome_blend_weight": search_value_outcome_blend_weight,
            "min_turn_index": search_value_min_turn_index,
            "max_turn_index": search_value_max_turn_index,
        },
    }
    payload["matchup"] = _benchmark_head_to_head(
        candidate_checkpoint_path=candidate_checkpoint,
        baseline_checkpoint_path=baseline_checkpoint,
        candidate_model_kind=candidate_model_kind,
        baseline_model_kind=baseline_model_kind,
        device=device,
        games=games,
        seed_start=seed_start,
        max_turns=max_turns,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
        swap_seats=swap_seats,
        paired_seeds=paired_seeds,
        log_every=log_every,
        loop_fallback=fallback_config,
        search_depth=search_depth,
        search_max_branching=search_max_branching,
        search_buy_branching=search_buy_branching,
        search_reserve_branching=search_reserve_branching,
        search_take_branching=search_take_branching,
        policy_prior_weight=policy_prior_weight,
        search_value_checkpoint=search_value_checkpoint,
        search_value_device=search_value_device,
        search_value_weight=search_value_weight,
        search_value_scale=search_value_scale,
        search_value_prediction_mode=search_value_prediction_mode,
        search_value_outcome_blend_weight=search_value_outcome_blend_weight,
        search_value_min_turn_index=search_value_min_turn_index,
        search_value_max_turn_index=search_value_max_turn_index,
    )
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
            if bot_name in MODEL_BOT_NAMES
        )
        model_loop_fallback_triggers += game_model_loop_fallback_triggers
        if game.winner is None:
            draws += 1
        else:
            winner_name = game.bot_seats[game.winner]
            if winner_name in MODEL_BOT_NAMES:
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


def summarize_head_to_head_games(
    games: tuple[GameResult, ...],
    candidate_label: str,
    baseline_label: str,
    seed_start: int,
    max_turns: int,
    device: str,
    repetition_limit: int = 0,
    no_progress_limit: int = 0,
    swap_seats: bool = True,
) -> dict[str, object]:
    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    timed_out = 0
    stalled = 0
    candidate_loop_fallback_triggers = 0
    baseline_loop_fallback_triggers = 0
    termination_reasons: dict[str, int] = {}
    per_game: list[dict[str, object]] = []

    for game_index, game in enumerate(games):
        swapped = swap_seats and game_index % 2 == 1
        candidate_seat = 1 if swapped else 0
        baseline_seat = 0 if swapped else 1
        winner_role = None
        winner_name = None
        if game.winner is None:
            draws += 1
        elif game.winner == candidate_seat:
            candidate_wins += 1
            winner_role = "candidate"
            winner_name = candidate_label
        elif game.winner == baseline_seat:
            baseline_wins += 1
            winner_role = "baseline"
            winner_name = baseline_label
        else:
            raise RuntimeError(f"Unexpected winner seat: {game.winner}")

        if game.timed_out:
            timed_out += 1
        if game.stalled:
            stalled += 1
        candidate_fallbacks = game.loop_fallback_triggers_by_seat[candidate_seat]
        baseline_fallbacks = game.loop_fallback_triggers_by_seat[baseline_seat]
        candidate_loop_fallback_triggers += candidate_fallbacks
        baseline_loop_fallback_triggers += baseline_fallbacks
        termination_reasons[game.termination_reason] = termination_reasons.get(game.termination_reason, 0) + 1
        per_game.append(
            {
                "seed": game.seed,
                "turns": game.turns,
                "winner": game.winner,
                "winner_role": winner_role,
                "winner_name": winner_name,
                "final_scores": list(game.final_scores),
                "bot_seats": list(game.bot_seats),
                "candidate_seat": candidate_seat,
                "baseline_seat": baseline_seat,
                "candidate_score": game.final_scores[candidate_seat],
                "baseline_score": game.final_scores[baseline_seat],
                "stalled": game.stalled,
                "timed_out": game.timed_out,
                "termination_reason": game.termination_reason,
                "repetition_count": game.repetition_count,
                "no_progress_streak": game.no_progress_streak,
                "loop_fallback_triggers_by_seat": list(game.loop_fallback_triggers_by_seat),
                "candidate_loop_fallback_triggers": candidate_fallbacks,
                "baseline_loop_fallback_triggers": baseline_fallbacks,
            }
        )

    return {
        "candidate": candidate_label,
        "baseline": baseline_label,
        "games": len(games),
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
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
        "candidate_loop_fallback_triggers": candidate_loop_fallback_triggers,
        "baseline_loop_fallback_triggers": baseline_loop_fallback_triggers,
        "games_detail": per_game,
    }


def default_output_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return Path("outputs") / "benchmarks" / f"{checkpoint.parent.name}_{checkpoint.stem}_benchmark.json"


def default_head_to_head_output_path(
    candidate_checkpoint_path: str | Path,
    baseline_checkpoint_path: str | Path,
) -> Path:
    candidate_checkpoint = Path(candidate_checkpoint_path)
    baseline_checkpoint = Path(baseline_checkpoint_path)
    return (
        Path("outputs")
        / "benchmarks"
        / (
            f"{candidate_checkpoint.parent.name}_{candidate_checkpoint.stem}"
            f"_vs_{baseline_checkpoint.parent.name}_{baseline_checkpoint.stem}_benchmark.json"
        )
    )


def write_benchmark_payload(path: str | Path, payload: dict[str, object]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _benchmark_opponent(
    checkpoint_path: Path,
    model_kind: str,
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
    search_depth: int,
    search_max_branching: int,
    search_buy_branching: int,
    search_reserve_branching: int,
    search_take_branching: int,
    policy_prior_weight: float,
    search_value_checkpoint: str | Path | None,
    search_value_device: str,
    search_value_weight: float,
    search_value_scale: float,
    search_value_prediction_mode: str,
    search_value_outcome_blend_weight: float,
    search_value_min_turn_index: int | None,
    search_value_max_turn_index: int | None,
) -> dict[str, object]:
    opponent_factory = _opponent_factory(
        opponent_name,
        seed_start,
        search_depth=search_depth,
        search_max_branching=search_max_branching,
        search_buy_branching=search_buy_branching,
        search_reserve_branching=search_reserve_branching,
        search_take_branching=search_take_branching,
    )
    played_games: list[GameResult] = []
    model_wins = 0
    opponent_wins = 0
    draws = 0
    timed_out = 0
    start_time = time.perf_counter()

    print(
        f"[benchmark] checkpoint={checkpoint_path} model_kind={model_kind} opponent={opponent_name}"
        f" games={games} device={device} swap_seats={swap_seats}"
    )
    model_bot = _model_bot_for_checkpoint(
        checkpoint_path=checkpoint_path,
        model_kind=model_kind,
        device=device,
        loop_fallback=loop_fallback,
        search_depth=search_depth,
        search_max_branching=search_max_branching,
        search_buy_branching=search_buy_branching,
        search_reserve_branching=search_reserve_branching,
        search_take_branching=search_take_branching,
        policy_prior_weight=policy_prior_weight,
        search_value_checkpoint=search_value_checkpoint,
        search_value_device=search_value_device,
        search_value_weight=search_value_weight,
        search_value_scale=search_value_scale,
        search_value_prediction_mode=search_value_prediction_mode,
        search_value_outcome_blend_weight=search_value_outcome_blend_weight,
        search_value_min_turn_index=search_value_min_turn_index,
        search_value_max_turn_index=search_value_max_turn_index,
    )
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
        elif game.bot_seats[game.winner] in MODEL_BOT_NAMES:
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


def _benchmark_head_to_head(
    candidate_checkpoint_path: Path,
    baseline_checkpoint_path: Path,
    candidate_model_kind: str,
    baseline_model_kind: str,
    device: str,
    games: int,
    seed_start: int,
    max_turns: int,
    repetition_limit: int,
    no_progress_limit: int,
    swap_seats: bool,
    paired_seeds: bool,
    log_every: int,
    loop_fallback: LoopFallbackConfig,
    search_depth: int,
    search_max_branching: int,
    search_buy_branching: int,
    search_reserve_branching: int,
    search_take_branching: int,
    policy_prior_weight: float,
    search_value_checkpoint: str | Path | None,
    search_value_device: str,
    search_value_weight: float,
    search_value_scale: float,
    search_value_prediction_mode: str,
    search_value_outcome_blend_weight: float,
    search_value_min_turn_index: int | None,
    search_value_max_turn_index: int | None,
) -> dict[str, object]:
    played_games: list[GameResult] = []
    candidate_wins = 0
    baseline_wins = 0
    draws = 0
    timed_out = 0
    start_time = time.perf_counter()
    candidate_label = f"candidate:{candidate_model_kind}"
    baseline_label = f"baseline:{baseline_model_kind}"

    print(
        f"[head-to-head] candidate={candidate_checkpoint_path} kind={candidate_model_kind}"
        f" baseline={baseline_checkpoint_path} baseline_kind={baseline_model_kind}"
        f" games={games} device={device} swap_seats={swap_seats}"
        f" paired_seeds={paired_seeds}"
    )
    candidate_bot = _model_bot_for_checkpoint(
        checkpoint_path=candidate_checkpoint_path,
        model_kind=candidate_model_kind,
        device=device,
        loop_fallback=loop_fallback,
        search_depth=search_depth,
        search_max_branching=search_max_branching,
        search_buy_branching=search_buy_branching,
        search_reserve_branching=search_reserve_branching,
        search_take_branching=search_take_branching,
        policy_prior_weight=policy_prior_weight,
        search_value_checkpoint=search_value_checkpoint,
        search_value_device=search_value_device,
        search_value_weight=search_value_weight,
        search_value_scale=search_value_scale,
        search_value_prediction_mode=search_value_prediction_mode,
        search_value_outcome_blend_weight=search_value_outcome_blend_weight,
        search_value_min_turn_index=search_value_min_turn_index,
        search_value_max_turn_index=search_value_max_turn_index,
    )
    baseline_bot = _model_bot_for_checkpoint(
        checkpoint_path=baseline_checkpoint_path,
        model_kind=baseline_model_kind,
        device=device,
        loop_fallback=loop_fallback,
        search_depth=search_depth,
        search_max_branching=search_max_branching,
        search_buy_branching=search_buy_branching,
        search_reserve_branching=search_reserve_branching,
        search_take_branching=search_take_branching,
        policy_prior_weight=policy_prior_weight,
        search_value_checkpoint=search_value_checkpoint,
        search_value_device=search_value_device,
        search_value_weight=search_value_weight,
        search_value_scale=search_value_scale,
        search_value_prediction_mode=search_value_prediction_mode,
        search_value_outcome_blend_weight=search_value_outcome_blend_weight,
        search_value_min_turn_index=search_value_min_turn_index,
        search_value_max_turn_index=search_value_max_turn_index,
    )

    for game_index in range(games):
        swap = swap_seats and game_index % 2 == 1
        candidate_seat = 1 if swap else 0
        baseline_seat = 0 if swap else 1
        seat0 = baseline_bot if swap else candidate_bot
        seat1 = candidate_bot if swap else baseline_bot
        game = play_game(
            bot_seat_0=seat0,
            bot_seat_1=seat1,
            seed=seed_start + (game_index // 2 if paired_seeds else game_index),
            max_turns=max_turns,
            repetition_limit=repetition_limit,
            no_progress_limit=no_progress_limit,
        )
        played_games.append(game)

        if game.winner is None:
            draws += 1
        elif game.winner == candidate_seat:
            candidate_wins += 1
        elif game.winner == baseline_seat:
            baseline_wins += 1
        else:
            raise RuntimeError(f"Unexpected winner seat: {game.winner}")
        if game.timed_out:
            timed_out += 1

        done = game_index + 1
        if done == 1 or done == games or (log_every > 0 and done % log_every == 0):
            elapsed = time.perf_counter() - start_time
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = ((games - done) / rate) if rate > 0 else math.inf
            eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
            print(
                f"[head-to-head] {done}/{games}"
                f" elapsed={elapsed:.1f}s"
                f" rate={rate:.2f} games/s"
                f" eta={eta_text}"
                f" candidate_wins={candidate_wins}"
                f" baseline_wins={baseline_wins}"
                f" draws={draws}"
                f" timed_out={timed_out}"
            )

    return summarize_head_to_head_games(
        tuple(played_games),
        candidate_label=candidate_label,
        baseline_label=baseline_label,
        seed_start=seed_start,
        max_turns=max_turns,
        device=device,
        repetition_limit=repetition_limit,
        no_progress_limit=no_progress_limit,
        swap_seats=swap_seats,
    )


def _opponent_factory(
    opponent_name: str,
    seed: int,
    search_depth: int = 2,
    search_max_branching: int = 10,
    search_buy_branching: int = 6,
    search_reserve_branching: int = 3,
    search_take_branching: int = 3,
) -> Callable[[], Bot]:
    if opponent_name == "random":
        return lambda: RandomLegalBot(seed=seed)
    if opponent_name == "greedy":
        return lambda: GreedyHeuristicBot(seed=seed)
    if opponent_name == "search":
        return lambda: ShallowSearchBot(
            depth=search_depth,
            max_branching=search_max_branching,
            max_buy_actions=search_buy_branching,
            max_reserve_actions=search_reserve_branching,
            max_take_actions=search_take_branching,
            seed=seed,
            information_legal=True,
            determinizations=1,
            determinization_seed=seed,
        )
    raise ValueError(f"Unsupported opponent: {opponent_name}")


def _model_bot_for_checkpoint(
    checkpoint_path: Path,
    model_kind: str,
    device: str,
    loop_fallback: LoopFallbackConfig,
    search_depth: int = 2,
    search_max_branching: int = 10,
    search_buy_branching: int = 6,
    search_reserve_branching: int = 3,
    search_take_branching: int = 3,
    policy_prior_weight: float = 2_000.0,
    search_value_checkpoint: str | Path | None = None,
    search_value_device: str = "cpu",
    search_value_weight: float = 0.0,
    search_value_scale: float = 15_000.0,
    search_value_prediction_mode: str = "value",
    search_value_outcome_blend_weight: float = 0.5,
    search_value_min_turn_index: int | None = None,
    search_value_max_turn_index: int | None = None,
) -> Bot:
    if model_kind == "flat_policy":
        return CheckpointPolicyBot(checkpoint_path, device=device, loop_fallback=loop_fallback)
    if model_kind == "legal_action_scorer":
        return LegalActionScorerBot(checkpoint_path, device=device)
    if model_kind == "legal_action_attention_scorer":
        return LegalActionAttentionScorerBot(checkpoint_path, device=device)
    if model_kind == "legal_action_set_scorer":
        return LegalActionSetScorerBot(checkpoint_path, device=device)
    if model_kind == "canonical_entity_action_scorer":
        return CanonicalEntityActionScorerBot(checkpoint_path, device=device)
    if model_kind == "policy_guided_search":
        value_evaluator = _build_value_evaluator(
            checkpoint_path=search_value_checkpoint,
            device=search_value_device,
            value_scale=search_value_scale,
            prediction_mode=search_value_prediction_mode,
            outcome_blend_weight=search_value_outcome_blend_weight,
        )
        return PolicyGuidedSearchBot(
            checkpoint_path=checkpoint_path,
            device=device,
            policy_prior_weight=policy_prior_weight,
            depth=search_depth,
            max_branching=search_max_branching,
            max_buy_actions=search_buy_branching,
            max_reserve_actions=search_reserve_branching,
            max_take_actions=search_take_branching,
            seed=0,
            value_evaluator=value_evaluator,
            value_weight=search_value_weight,
            value_min_turn_index=search_value_min_turn_index,
            value_max_turn_index=search_value_max_turn_index,
            information_legal=True,
            determinizations=1,
            determinization_seed=0,
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _build_value_evaluator(
    checkpoint_path: str | Path | None,
    device: str,
    value_scale: float,
    prediction_mode: str = "value",
    outcome_blend_weight: float = 0.5,
) -> CheckpointValueEvaluator | None:
    if checkpoint_path is None:
        return None
    return CheckpointValueEvaluator(
        checkpoint_path=checkpoint_path,
        device=device,
        value_scale=value_scale,
        prediction_mode=prediction_mode,
        outcome_blend_weight=outcome_blend_weight,
    )


def main() -> None:
    args = build_parser().parse_args()
    loop_fallback = LoopFallbackConfig(
        enabled=not args.disable_loop_fallback,
        min_state_visits=args.loop_fallback_min_state_visits,
        min_own_non_progress_actions=args.loop_fallback_min_own_no_progress_actions,
        max_buy_logit_gap=args.loop_fallback_max_buy_logit_gap,
    )
    if args.baseline_checkpoint is None:
        output_path = Path(args.output_path) if args.output_path is not None else default_output_path(args.checkpoint)
        payload = benchmark_checkpoint(
            checkpoint_path=args.checkpoint,
            opponents=tuple(args.opponents),
            model_kind=args.model_kind,
            device=args.device,
            games=args.games,
            seed_start=args.seed_start,
            max_turns=args.max_turns,
            repetition_limit=args.repetition_limit,
            no_progress_limit=args.no_progress_limit,
            swap_seats=not args.no_swap_seats,
            log_every=args.log_every,
            loop_fallback=loop_fallback,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            policy_prior_weight=args.policy_prior_weight,
            search_value_checkpoint=args.search_value_checkpoint,
            search_value_device=args.search_value_device,
            search_value_weight=args.search_value_weight,
            search_value_scale=args.search_value_scale,
            search_value_prediction_mode=args.search_value_prediction_mode,
            search_value_outcome_blend_weight=args.search_value_outcome_blend_weight,
            search_value_min_turn_index=args.search_value_min_turn_index,
            search_value_max_turn_index=args.search_value_max_turn_index,
        )
    else:
        output_path = (
            Path(args.output_path)
            if args.output_path is not None
            else default_head_to_head_output_path(args.checkpoint, args.baseline_checkpoint)
        )
        payload = benchmark_head_to_head(
            candidate_checkpoint_path=args.checkpoint,
            baseline_checkpoint_path=args.baseline_checkpoint,
            candidate_model_kind=args.model_kind,
            baseline_model_kind=args.baseline_model_kind,
            device=args.device,
            games=args.games,
            seed_start=args.seed_start,
            max_turns=args.max_turns,
            repetition_limit=args.repetition_limit,
            no_progress_limit=args.no_progress_limit,
            swap_seats=not args.no_swap_seats,
            paired_seeds=args.paired_seeds,
            log_every=args.log_every,
            loop_fallback=loop_fallback,
            search_depth=args.search_depth,
            search_max_branching=args.search_max_branching,
            search_buy_branching=args.search_buy_branching,
            search_reserve_branching=args.search_reserve_branching,
            search_take_branching=args.search_take_branching,
            policy_prior_weight=args.policy_prior_weight,
            search_value_checkpoint=args.search_value_checkpoint,
            search_value_device=args.search_value_device,
            search_value_weight=args.search_value_weight,
            search_value_scale=args.search_value_scale,
            search_value_prediction_mode=args.search_value_prediction_mode,
            search_value_outcome_blend_weight=args.search_value_outcome_blend_weight,
            search_value_min_turn_index=args.search_value_min_turn_index,
            search_value_max_turn_index=args.search_value_max_turn_index,
        )
    written_path = write_benchmark_payload(output_path, payload)
    print(f"[benchmark] wrote results: {written_path}")
    if "opponents" in payload:
        for opponent_payload in payload["opponents"]:
            print(
                "Benchmark:"
                f" opponent={opponent_payload['opponent']}"
                f" model_wins={opponent_payload['model_wins']}"
                f" opponent_wins={opponent_payload['opponent_wins']}"
                f" draws={opponent_payload['draws']}"
                f" timed_out={opponent_payload['timed_out_games']}"
            )
    else:
        matchup = payload["matchup"]
        print(
            "Head-to-head:"
            f" candidate_wins={matchup['candidate_wins']}"
            f" baseline_wins={matchup['baseline_wins']}"
            f" draws={matchup['draws']}"
            f" timed_out={matchup['timed_out_games']}"
        )


if __name__ == "__main__":
    main()
