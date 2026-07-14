"""Benchmark information-set PUCT against policy and search baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import time
from typing import Any

from splendor_ai.bots import (
    InformationSetPuctBot,
    LegalActionAttentionScorerBot,
    PolicyGuidedSearchBot,
    PuctConfig,
    ShallowSearchBot,
)
from splendor_ai.eval.match import play_game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark legal-observation PUCT.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-checkpoint", default=None)
    parser.add_argument(
        "--value-prediction-mode",
        choices=("value", "outcome", "blend"),
        default="value",
    )
    parser.add_argument("--value-outcome-blend-weight", type=float, default=0.5)
    parser.add_argument("--value-min-turn-index", type=int, default=None)
    parser.add_argument("--value-max-turn-index", type=int, default=None)
    parser.add_argument(
        "--opponent",
        choices=("policy", "search", "pgs", "puct"),
        required=True,
    )
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument(
        "--opponent-plain-puct",
        action="store_true",
        help=(
            "When --opponent puct is selected, disable its n-step and rollout "
            "leaf rules while retaining the shared search budget and priors."
        ),
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--prior-temperature", type=float, default=1.0)
    parser.add_argument("--fpu-reduction", type=float, default=0.0)
    parser.add_argument("--max-depth", type=int, default=80)
    parser.add_argument("--heuristic-scale", type=float, default=4000.0)
    parser.add_argument("--time-limit-seconds", type=float, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--tree-reuse", action="store_true")
    parser.add_argument("--max-tree-nodes", type=int, default=50_000)
    parser.add_argument("--n-step-leaf-depth", type=int, default=0)
    parser.add_argument("--n-step-leaf-width", type=int, default=1)
    parser.add_argument("--n-step-leaf-candidate-limit", type=int, default=8)
    parser.add_argument("--n-step-leaf-weight", type=float, default=1.0)
    parser.add_argument("--n-step-leaf-delta-clip", type=float, default=None)
    parser.add_argument("--n-step-leaf-min-turn-index", type=int, default=None)
    parser.add_argument("--n-step-leaf-max-turn-index", type=int, default=None)
    parser.add_argument("--n-step-leaf-min-max-score", type=int, default=None)
    parser.add_argument("--rollout-leaf-count", type=int, default=0)
    parser.add_argument("--rollout-leaf-max-depth", type=int, default=12)
    parser.add_argument("--rollout-leaf-top-k", type=int, default=3)
    parser.add_argument("--rollout-leaf-temperature", type=float, default=0.5)
    parser.add_argument(
        "--rollout-leaf-policy",
        choices=("random", "heuristic"),
        default="random",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--repetition-limit", type=int, default=4)
    parser.add_argument("--no-progress-limit", type=int, default=60)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--search-determinizations", type=int, default=4)
    parser.add_argument("--allow-oracle-search", action="store_true")
    parser.add_argument("--search-max-branching", type=int, default=10)
    parser.add_argument("--search-buy-branching", type=int, default=6)
    parser.add_argument("--search-reserve-branching", type=int, default=3)
    parser.add_argument("--search-take-branching", type=int, default=3)
    parser.add_argument("--pgs-prior-weight", type=float, default=3000.0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--log-every", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.games <= 0 or args.games % 2:
        raise ValueError("games must be a positive even number for paired seeds.")
    checkpoint = Path(args.checkpoint)
    opponent_checkpoint = (
        Path(args.opponent_checkpoint)
        if args.opponent_checkpoint is not None
        else None
    )
    puct = InformationSetPuctBot(
        checkpoint,
        device=args.device,
        config=PuctConfig(
            simulations=args.simulations,
            c_puct=args.c_puct,
            prior_temperature=args.prior_temperature,
            fpu_reduction=args.fpu_reduction,
            max_depth=args.max_depth,
            heuristic_scale=args.heuristic_scale,
            seed=args.seed_start,
            time_limit_seconds=args.time_limit_seconds,
            inference_batch_size=args.inference_batch_size,
            reuse_tree=args.tree_reuse,
            max_tree_nodes=args.max_tree_nodes,
            value_min_turn_index=args.value_min_turn_index,
            value_max_turn_index=args.value_max_turn_index,
            n_step_leaf_depth=args.n_step_leaf_depth,
            n_step_leaf_width=args.n_step_leaf_width,
            n_step_leaf_candidate_limit=args.n_step_leaf_candidate_limit,
            n_step_leaf_weight=args.n_step_leaf_weight,
            n_step_leaf_delta_clip=args.n_step_leaf_delta_clip,
            n_step_leaf_min_turn_index=args.n_step_leaf_min_turn_index,
            n_step_leaf_max_turn_index=args.n_step_leaf_max_turn_index,
            n_step_leaf_min_max_score=args.n_step_leaf_min_max_score,
            rollout_leaf_count=args.rollout_leaf_count,
            rollout_leaf_max_depth=args.rollout_leaf_max_depth,
            rollout_leaf_top_k=args.rollout_leaf_top_k,
            rollout_leaf_temperature=args.rollout_leaf_temperature,
            rollout_leaf_policy=args.rollout_leaf_policy,
        ),
        value_checkpoint_path=args.value_checkpoint,
        value_prediction_mode=args.value_prediction_mode,
        value_outcome_blend_weight=args.value_outcome_blend_weight,
    )
    opponent = _build_opponent(args, opponent_checkpoint)
    puct_wins = 0
    opponent_wins = 0
    draws = 0
    details = []
    started = time.perf_counter()
    history_offset = 0
    for game_index in range(args.games):
        puct_seat = game_index % 2
        pair_index = game_index // 2
        result = play_game(
            bot_seat_0=puct if puct_seat == 0 else opponent,
            bot_seat_1=opponent if puct_seat == 0 else puct,
            seed=args.seed_start + pair_index,
            max_turns=args.max_turns,
            repetition_limit=args.repetition_limit,
            no_progress_limit=args.no_progress_limit,
        )
        if result.winner is None:
            draws += 1
        elif result.winner == puct_seat:
            puct_wins += 1
        else:
            opponent_wins += 1
        new_stats = puct.search_history[history_offset:]
        history_offset += len(new_stats)
        telemetry = _summarize_search_stats(new_stats)
        details.append(
            {
                "seed": result.seed,
                "pair_index": pair_index,
                "puct_seat": puct_seat,
                "winner": result.winner,
                "final_scores": list(result.final_scores),
                "turns": result.turns,
                "timed_out": result.timed_out,
                "termination_reason": result.termination_reason,
                "puct_telemetry": telemetry,
            }
        )
        done = game_index + 1
        if (
            done == 1
            or done == args.games
            or (args.log_every > 0 and done % args.log_every == 0)
        ):
            print(
                f"[puct-benchmark] {done}/{args.games}"
                f" puct={puct_wins} opponent={opponent_wins} draws={draws}"
                f" elapsed={time.perf_counter() - started:.1f}s"
            )

    all_stats = puct.search_history
    payload = {
        "checkpoint": str(checkpoint),
        "value_checkpoint": args.value_checkpoint,
        "value_prediction_mode": args.value_prediction_mode,
        "value_outcome_blend_weight": args.value_outcome_blend_weight,
        "opponent": args.opponent,
        "opponent_checkpoint": (
            str(opponent_checkpoint)
            if opponent_checkpoint is not None
            else None
        ),
        "games": args.games,
        "paired_seeds": args.games // 2,
        "seed_start": args.seed_start,
        "puct_wins": puct_wins,
        "opponent_wins": opponent_wins,
        "draws": draws,
        "puct_config": {
            "simulations": args.simulations,
            "c_puct": args.c_puct,
            "prior_temperature": args.prior_temperature,
            "fpu_reduction": args.fpu_reduction,
            "max_depth": args.max_depth,
            "heuristic_scale": args.heuristic_scale,
            "time_limit_seconds": args.time_limit_seconds,
            "inference_batch_size": args.inference_batch_size,
            "reuse_tree": args.tree_reuse,
            "max_tree_nodes": args.max_tree_nodes,
            "value_min_turn_index": args.value_min_turn_index,
            "value_max_turn_index": args.value_max_turn_index,
            "n_step_leaf_depth": args.n_step_leaf_depth,
            "n_step_leaf_width": args.n_step_leaf_width,
            "n_step_leaf_candidate_limit": args.n_step_leaf_candidate_limit,
            "n_step_leaf_weight": args.n_step_leaf_weight,
            "n_step_leaf_delta_clip": args.n_step_leaf_delta_clip,
            "n_step_leaf_min_turn_index": args.n_step_leaf_min_turn_index,
            "n_step_leaf_max_turn_index": args.n_step_leaf_max_turn_index,
            "n_step_leaf_min_max_score": args.n_step_leaf_min_max_score,
            "rollout_leaf_count": args.rollout_leaf_count,
            "rollout_leaf_max_depth": args.rollout_leaf_max_depth,
            "rollout_leaf_top_k": args.rollout_leaf_top_k,
            "rollout_leaf_temperature": args.rollout_leaf_temperature,
            "rollout_leaf_policy": args.rollout_leaf_policy,
        },
        "search_opponent_config": {
            "information_legal": not args.allow_oracle_search,
            "determinizations": args.search_determinizations,
            "depth": args.search_depth,
        },
        "opponent_plain_puct": args.opponent_plain_puct,
        "telemetry": _summarize_search_stats(all_stats),
        "paired_statistics": _paired_statistics(
            details,
            bootstrap_seed=args.seed_start,
        ),
        "games_detail": details,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[puct-benchmark] wrote: {output_path}")


def _build_opponent(args: argparse.Namespace, checkpoint: Path | None):
    if args.opponent == "policy":
        if checkpoint is None:
            raise ValueError("policy opponent requires --opponent-checkpoint.")
        return LegalActionAttentionScorerBot(checkpoint, device=args.device)
    if args.opponent == "search":
        return ShallowSearchBot(
            depth=args.search_depth,
            max_branching=args.search_max_branching,
            max_buy_actions=args.search_buy_branching,
            max_reserve_actions=args.search_reserve_branching,
            max_take_actions=args.search_take_branching,
            seed=args.seed_start + 100_000,
            information_legal=not args.allow_oracle_search,
            determinizations=args.search_determinizations,
            determinization_seed=args.seed_start + 200_000,
        )
    if args.opponent == "puct":
        if checkpoint is None:
            raise ValueError("puct opponent requires --opponent-checkpoint.")
        return InformationSetPuctBot(
            checkpoint,
            device=args.device,
            config=PuctConfig(
                simulations=args.simulations,
                c_puct=args.c_puct,
                prior_temperature=args.prior_temperature,
                fpu_reduction=args.fpu_reduction,
                max_depth=args.max_depth,
                heuristic_scale=args.heuristic_scale,
                seed=args.seed_start + 100_000,
                time_limit_seconds=args.time_limit_seconds,
                inference_batch_size=args.inference_batch_size,
                reuse_tree=args.tree_reuse,
                max_tree_nodes=args.max_tree_nodes,
                value_min_turn_index=args.value_min_turn_index,
                value_max_turn_index=args.value_max_turn_index,
                n_step_leaf_depth=(0 if args.opponent_plain_puct else args.n_step_leaf_depth),
                n_step_leaf_width=args.n_step_leaf_width,
                n_step_leaf_candidate_limit=args.n_step_leaf_candidate_limit,
                n_step_leaf_weight=(1.0 if args.opponent_plain_puct else args.n_step_leaf_weight),
                n_step_leaf_delta_clip=(None if args.opponent_plain_puct else args.n_step_leaf_delta_clip),
                n_step_leaf_min_turn_index=(None if args.opponent_plain_puct else args.n_step_leaf_min_turn_index),
                n_step_leaf_max_turn_index=(None if args.opponent_plain_puct else args.n_step_leaf_max_turn_index),
                n_step_leaf_min_max_score=(None if args.opponent_plain_puct else args.n_step_leaf_min_max_score),
                rollout_leaf_count=(0 if args.opponent_plain_puct else args.rollout_leaf_count),
                rollout_leaf_max_depth=args.rollout_leaf_max_depth,
                rollout_leaf_top_k=args.rollout_leaf_top_k,
                rollout_leaf_temperature=args.rollout_leaf_temperature,
                rollout_leaf_policy=args.rollout_leaf_policy,
            ),
        )
    if checkpoint is None:
        raise ValueError("pgs opponent requires --opponent-checkpoint.")
    return PolicyGuidedSearchBot(
        checkpoint,
        device=args.device,
        policy_prior_weight=args.pgs_prior_weight,
        depth=args.search_depth,
        max_branching=args.search_max_branching,
        max_buy_actions=args.search_buy_branching,
        max_reserve_actions=args.search_reserve_branching,
        max_take_actions=args.search_take_branching,
        seed=args.seed_start + 100_000,
        information_legal=not args.allow_oracle_search,
        determinizations=args.search_determinizations,
        determinization_seed=args.seed_start + 200_000,
    )


def _summarize_search_stats(stats: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    if not stats:
        return {
            "moves": 0,
            "simulations": 0,
            "elapsed_seconds": 0.0,
            "simulations_per_second": 0.0,
        }
    elapsed = sum(item.elapsed_seconds for item in stats)
    simulations = sum(item.simulations for item in stats)
    n_step_evaluations = sum(item.n_step_leaf_evaluations for item in stats)
    n_step_clipped = sum(
        item.n_step_leaf_clipped_evaluations for item in stats
    )
    n_step_absolute_delta = sum(
        item.n_step_leaf_absolute_delta_sum for item in stats
    )
    n_step_applied_absolute_delta = sum(
        item.n_step_leaf_applied_absolute_delta_sum for item in stats
    )
    action_stds = [
        statistics.pstdev(values)
        for item in stats
        for values in item.root_action_values_by_determinization.values()
        if len(values) > 1
    ]
    return {
        "moves": len(stats),
        "simulations": simulations,
        "expansions": sum(item.expansions for item in stats),
        "neural_calls": sum(item.neural_calls for item in stats),
        "value_calls": sum(item.value_calls for item in stats),
        "prior_batch_calls": sum(item.prior_batch_calls for item in stats),
        "value_batch_calls": sum(item.value_batch_calls for item in stats),
        "n_step_leaf_evaluations": n_step_evaluations,
        "n_step_leaf_cache_hits": sum(
            item.n_step_leaf_cache_hits for item in stats
        ),
        "n_step_leaf_clipped_evaluations": n_step_clipped,
        "n_step_leaf_clipped_rate": (
            n_step_clipped / n_step_evaluations
            if n_step_evaluations
            else 0.0
        ),
        "n_step_leaf_absolute_delta_sum": n_step_absolute_delta,
        "n_step_leaf_applied_absolute_delta_sum": (
            n_step_applied_absolute_delta
        ),
        "mean_n_step_leaf_absolute_delta": (
            n_step_absolute_delta / n_step_evaluations
            if n_step_evaluations
            else 0.0
        ),
        "mean_n_step_leaf_applied_absolute_delta": (
            n_step_applied_absolute_delta / n_step_evaluations
            if n_step_evaluations
            else 0.0
        ),
        "rollout_leaf_evaluations": sum(
            item.rollout_leaf_evaluations for item in stats
        ),
        "nodes_reused": sum(item.nodes_reused for item in stats),
        "root_reused_moves": sum(item.root_reused for item in stats),
        "peak_retained_tree_nodes": max(
            item.retained_tree_nodes_end for item in stats
        ),
        "max_depth_reached": max(item.max_depth_reached for item in stats),
        "determinization_seconds": sum(
            item.determinization_seconds for item in stats
        ),
        "traversal_seconds": sum(item.traversal_seconds for item in stats),
        "inference_seconds": sum(item.inference_seconds for item in stats),
        "elapsed_seconds": elapsed,
        "simulations_per_second": simulations / elapsed if elapsed else 0.0,
        "mean_seconds_per_move": elapsed / len(stats),
        "mean_root_action_determinization_std": (
            statistics.fmean(action_stds) if action_stds else 0.0
        ),
        "root_actions_with_variance_samples": len(action_stds),
        "mean_root_prior_entropy": statistics.fmean(
            item.root_prior_entropy for item in stats
        ),
        "mean_root_prior_normalized_entropy": statistics.fmean(
            item.root_prior_normalized_entropy for item in stats
        ),
        "mean_root_prior_top1_probability": statistics.fmean(
            item.root_prior_top1_probability for item in stats
        ),
        "mean_root_prior_effective_actions": statistics.fmean(
            item.root_prior_effective_actions for item in stats
        ),
        "mean_root_visit_entropy": statistics.fmean(
            item.root_visit_entropy for item in stats
        ),
        "mean_root_visit_normalized_entropy": statistics.fmean(
            item.root_visit_normalized_entropy for item in stats
        ),
        "mean_root_visit_effective_actions": statistics.fmean(
            item.root_visit_effective_actions for item in stats
        ),
        "mean_root_visited_actions": statistics.fmean(
            item.root_visited_actions for item in stats
        ),
    }


def _paired_statistics(
    details: list[dict[str, Any]],
    bootstrap_seed: int,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    pairs: dict[int, list[dict[str, Any]]] = {}
    for game in details:
        pairs.setdefault(int(game["pair_index"]), []).append(game)
    complete_pairs = [games for games in pairs.values() if len(games) == 2]
    score_shares: list[float] = []
    score_margins: list[float] = []
    for games in complete_pairs:
        points = 0.0
        margins = []
        for game in games:
            winner = game["winner"]
            puct_seat = int(game["puct_seat"])
            if winner is None:
                points += 0.5
            elif int(winner) == puct_seat:
                points += 1.0
            final_scores = game["final_scores"]
            margins.append(
                float(final_scores[puct_seat])
                - float(final_scores[1 - puct_seat])
            )
        score_shares.append(points / 2.0)
        score_margins.append(statistics.fmean(margins))
    return {
        "complete_pairs": len(complete_pairs),
        "mean_score_share": (
            statistics.fmean(score_shares) if score_shares else 0.0
        ),
        "mean_score_share_ci95": _bootstrap_mean_ci(
            score_shares,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "mean_final_score_margin": (
            statistics.fmean(score_margins) if score_margins else 0.0
        ),
        "mean_final_score_margin_ci95": _bootstrap_mean_ci(
            score_margins,
            samples=bootstrap_samples,
            seed=bootstrap_seed + 1,
        ),
    }


def _bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1 or len(set(values)) == 1:
        mean = statistics.fmean(values)
        return [mean, mean]
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    lower = means[int(0.025 * (samples - 1))]
    upper = means[int(0.975 * (samples - 1))]
    return [lower, upper]


if __name__ == "__main__":
    main()
