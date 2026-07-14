"""CLI entry point for replay corpus generation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from splendor_ai.bots import (
    CheckpointPolicyBot,
    CheckpointValueEvaluator,
    GreedyHeuristicBot,
    InformationSetPuctBot,
    LegalActionAttentionScorerBot,
    LegalActionAttentionSamplingBot,
    LegalActionScorerBot,
    LegalActionSetScorerBot,
    LoopFallbackConfig,
    PolicyGuidedSearchBot,
    PuctConfig,
    RandomLegalBot,
    ShallowSearchBot,
)

from .corpus import generate_replay_corpus, write_replay_corpus, write_replay_corpus_manifest


def _build_bot_factory(
    bot_name: str,
    seed: int,
    args: argparse.Namespace,
    seat_index: int | None = None,
):
    normalized = bot_name.lower()
    if normalized == "random":
        return lambda: RandomLegalBot(seed=seed)
    if normalized == "greedy":
        return lambda: GreedyHeuristicBot(seed=seed)
    if normalized == "search":
        value_evaluator = _build_value_evaluator(args)
        return lambda: ShallowSearchBot(
            depth=args.search_depth,
            max_branching=args.search_max_branching,
            max_buy_actions=args.search_buy_branching,
            max_reserve_actions=args.search_reserve_branching,
            max_take_actions=args.search_take_branching,
            seed=seed,
            value_evaluator=value_evaluator,
            value_weight=args.search_value_weight,
            value_min_turn_index=args.search_value_min_turn_index,
            value_max_turn_index=args.search_value_max_turn_index,
            information_legal=not args.allow_oracle_search,
            determinizations=args.search_determinizations,
            determinization_seed=seed,
        )
    if normalized == "puct":
        checkpoint_path = _resolve_checkpoint_path(
            args,
            "checkpoint",
            seat_index=seat_index,
        )
        cached_bot = None

        def factory():
            nonlocal cached_bot
            if cached_bot is None:
                cached_bot = InformationSetPuctBot(
                    checkpoint_path,
                    device=args.checkpoint_device,
                    config=PuctConfig(
                        simulations=args.puct_simulations,
                        c_puct=args.puct_c_puct,
                        prior_temperature=args.puct_prior_temperature,
                        fpu_reduction=args.puct_fpu_reduction,
                        max_depth=args.puct_max_depth,
                        heuristic_scale=args.puct_heuristic_scale,
                        seed=seed,
                        inference_batch_size=args.puct_inference_batch_size,
                        reuse_tree=args.puct_tree_reuse,
                        max_tree_nodes=args.puct_max_tree_nodes,
                        value_min_turn_index=args.puct_value_min_turn_index,
                        value_max_turn_index=args.puct_value_max_turn_index,
                        n_step_leaf_depth=args.puct_n_step_leaf_depth,
                        n_step_leaf_width=args.puct_n_step_leaf_width,
                        n_step_leaf_candidate_limit=(
                            args.puct_n_step_leaf_candidate_limit
                        ),
                        n_step_leaf_weight=args.puct_n_step_leaf_weight,
                        n_step_leaf_delta_clip=(
                            args.puct_n_step_leaf_delta_clip
                        ),
                        n_step_leaf_min_turn_index=(
                            args.puct_n_step_leaf_min_turn_index
                        ),
                        n_step_leaf_max_turn_index=(
                            args.puct_n_step_leaf_max_turn_index
                        ),
                        n_step_leaf_min_max_score=(
                            args.puct_n_step_leaf_min_max_score
                        ),
                        rollout_leaf_count=args.puct_rollout_leaf_count,
                        rollout_leaf_max_depth=args.puct_rollout_leaf_max_depth,
                        rollout_leaf_top_k=args.puct_rollout_leaf_top_k,
                        rollout_leaf_temperature=(
                            args.puct_rollout_leaf_temperature
                        ),
                        rollout_leaf_policy=args.puct_rollout_leaf_policy,
                    ),
                    value_checkpoint_path=args.puct_value_checkpoint,
                    value_prediction_mode=args.puct_value_prediction_mode,
                    value_outcome_blend_weight=(
                        args.puct_value_outcome_blend_weight
                    ),
                )
            return cached_bot

        return factory
    if normalized == "checkpoint_sample":
        checkpoint_path = _resolve_checkpoint_path(args, bot_name, seat_index=seat_index)
        checkpoint_model_kind = _resolve_checkpoint_model_kind(args, seat_index=seat_index)
        if checkpoint_model_kind != "legal_action_attention_scorer":
            raise ValueError(
                "checkpoint_sample currently requires "
                "--checkpoint-model-kind legal_action_attention_scorer."
            )
        cached_bot = None

        def factory():
            nonlocal cached_bot
            if cached_bot is None:
                cached_bot = LegalActionAttentionSamplingBot(
                    checkpoint_path,
                    device=args.checkpoint_device,
                    seed=seed,
                    epsilon=args.sampling_epsilon,
                    temperature=args.sampling_temperature,
                    top_k=args.sampling_top_k,
                )
            return cached_bot

        return factory
    if normalized == "checkpoint":
        checkpoint_path = _resolve_checkpoint_path(args, bot_name, seat_index=seat_index)
        checkpoint_model_kind = _resolve_checkpoint_model_kind(args, seat_index=seat_index)
        fallback_config = _checkpoint_loop_fallback_config(args)
        cached_bot = None

        def factory():
            nonlocal cached_bot
            if cached_bot is None:
                if checkpoint_model_kind == "flat_policy":
                    cached_bot = CheckpointPolicyBot(
                        checkpoint_path,
                        device=args.checkpoint_device,
                        loop_fallback=fallback_config,
                    )
                elif checkpoint_model_kind == "legal_action_scorer":
                    cached_bot = LegalActionScorerBot(
                        checkpoint_path,
                        device=args.checkpoint_device,
                    )
                elif checkpoint_model_kind == "legal_action_attention_scorer":
                    cached_bot = LegalActionAttentionScorerBot(
                        checkpoint_path,
                        device=args.checkpoint_device,
                    )
                elif checkpoint_model_kind == "legal_action_set_scorer":
                    cached_bot = LegalActionSetScorerBot(
                        checkpoint_path,
                        device=args.checkpoint_device,
                    )
                elif checkpoint_model_kind == "policy_guided_search":
                    value_evaluator = _build_value_evaluator(args)
                    cached_bot = PolicyGuidedSearchBot(
                        checkpoint_path,
                        device=args.checkpoint_device,
                        policy_prior_weight=args.policy_prior_weight,
                        depth=args.search_depth,
                        max_branching=args.search_max_branching,
                        max_buy_actions=args.search_buy_branching,
                        max_reserve_actions=args.search_reserve_branching,
                        max_take_actions=args.search_take_branching,
                        seed=seed,
                        value_evaluator=value_evaluator,
                        value_weight=args.search_value_weight,
                        value_min_turn_index=args.search_value_min_turn_index,
                        value_max_turn_index=args.search_value_max_turn_index,
                        information_legal=not args.allow_oracle_search,
                        determinizations=args.search_determinizations,
                        determinization_seed=seed,
                    )
                else:
                    raise ValueError(f"Unsupported checkpoint model kind: {checkpoint_model_kind}")
            return cached_bot

        return factory
    raise ValueError(f"Unsupported bot name: {bot_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate replay JSONL corpora for supervised warm-start training.")
    parser.add_argument("--output-dir", required=True, help="Directory where replay files should be written.")
    parser.add_argument(
        "--source-type",
        default="generated_replay",
        help="Structured corpus source label written to manifest.json, e.g. league, self_play, search_vs_greedy.",
    )
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
    parser.add_argument(
        "--seat0-bot",
        default="greedy",
        choices=(
            "greedy",
            "random",
            "search",
            "checkpoint",
            "checkpoint_sample",
            "puct",
        ),
    )
    parser.add_argument(
        "--seat1-bot",
        default="random",
        choices=(
            "greedy",
            "random",
            "search",
            "checkpoint",
            "checkpoint_sample",
            "puct",
        ),
    )
    parser.add_argument("--seat0-seed", type=int, default=0)
    parser.add_argument("--seat1-seed", type=int, default=1)
    parser.add_argument("--sampling-epsilon", type=float, default=0.2)
    parser.add_argument("--sampling-temperature", type=float, default=0.5)
    parser.add_argument("--sampling-top-k", type=int, default=3)
    parser.add_argument("--search-depth", type=int, default=2, help="Search depth for the search bot.")
    parser.add_argument("--search-determinizations", type=int, default=1)
    parser.add_argument("--allow-oracle-search", action="store_true")
    parser.add_argument("--puct-simulations", type=int, default=64)
    parser.add_argument("--puct-c-puct", type=float, default=2.0)
    parser.add_argument("--puct-prior-temperature", type=float, default=1.0)
    parser.add_argument("--puct-fpu-reduction", type=float, default=0.0)
    parser.add_argument("--puct-max-depth", type=int, default=80)
    parser.add_argument("--puct-heuristic-scale", type=float, default=4_000.0)
    parser.add_argument("--puct-inference-batch-size", type=int, default=8)
    parser.add_argument("--puct-tree-reuse", action="store_true")
    parser.add_argument("--puct-max-tree-nodes", type=int, default=50_000)
    parser.add_argument("--puct-n-step-leaf-depth", type=int, default=0)
    parser.add_argument("--puct-n-step-leaf-width", type=int, default=1)
    parser.add_argument("--puct-n-step-leaf-candidate-limit", type=int, default=8)
    parser.add_argument("--puct-n-step-leaf-weight", type=float, default=1.0)
    parser.add_argument(
        "--puct-n-step-leaf-delta-clip", type=float, default=None
    )
    parser.add_argument("--puct-n-step-leaf-min-turn-index", type=int, default=None)
    parser.add_argument("--puct-n-step-leaf-max-turn-index", type=int, default=None)
    parser.add_argument("--puct-n-step-leaf-min-max-score", type=int, default=None)
    parser.add_argument("--puct-rollout-leaf-count", type=int, default=0)
    parser.add_argument("--puct-rollout-leaf-max-depth", type=int, default=12)
    parser.add_argument("--puct-rollout-leaf-top-k", type=int, default=3)
    parser.add_argument("--puct-rollout-leaf-temperature", type=float, default=0.5)
    parser.add_argument(
        "--puct-rollout-leaf-policy",
        choices=("random", "heuristic"),
        default="random",
    )
    parser.add_argument("--puct-value-checkpoint", default=None)
    parser.add_argument("--puct-value-min-turn-index", type=int, default=None)
    parser.add_argument("--puct-value-max-turn-index", type=int, default=None)
    parser.add_argument(
        "--puct-value-prediction-mode",
        choices=("value", "outcome", "blend"),
        default="value",
    )
    parser.add_argument(
        "--puct-value-outcome-blend-weight",
        type=float,
        default=0.5,
    )
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
        "--checkpoint-path",
        default=None,
        help="Checkpoint used by any BOT name set to 'checkpoint'.",
    )
    parser.add_argument(
        "--seat0-checkpoint-path",
        default=None,
        help="Optional checkpoint override when seat 0 uses BOT name 'checkpoint'.",
    )
    parser.add_argument(
        "--seat1-checkpoint-path",
        default=None,
        help="Optional checkpoint override when seat 1 uses BOT name 'checkpoint'.",
    )
    parser.add_argument(
        "--checkpoint-device",
        default="cpu",
        help="Device used by checkpoint policy bots during replay generation.",
    )
    parser.add_argument(
        "--checkpoint-model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
            "policy_guided_search",
        ),
        default="flat_policy",
        help="Architecture loaded for BOT name 'checkpoint'.",
    )
    parser.add_argument(
        "--seat0-checkpoint-model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
            "policy_guided_search",
        ),
        default=None,
        help="Optional architecture override when seat 0 uses BOT name 'checkpoint'.",
    )
    parser.add_argument(
        "--seat1-checkpoint-model-kind",
        choices=(
            "flat_policy",
            "legal_action_scorer",
            "legal_action_attention_scorer",
            "legal_action_set_scorer",
            "policy_guided_search",
        ),
        default=None,
        help="Optional architecture override when seat 1 uses BOT name 'checkpoint'.",
    )
    parser.add_argument(
        "--disable-checkpoint-loop-fallback",
        action="store_true",
        help="Disable checkpoint policy loop fallback during replay generation.",
    )
    parser.add_argument(
        "--checkpoint-loop-fallback-min-state-visits",
        type=int,
        default=2,
        help="Repeated state visits needed before checkpoint fallback can choose a buy.",
    )
    parser.add_argument(
        "--checkpoint-loop-fallback-min-own-no-progress-actions",
        type=int,
        default=6,
        help="Consecutive non-progress checkpoint actions needed before fallback can choose a buy.",
    )
    parser.add_argument(
        "--checkpoint-loop-fallback-max-buy-logit-gap",
        type=float,
        default=8.0,
        help="Maximum legal-top-minus-buy logit gap allowed for checkpoint fallback buy selection.",
    )
    parser.add_argument(
        "--policy-prior-weight",
        type=float,
        default=2_000.0,
        help="Heuristic-scale rank-prior weight for checkpoint policy-guided search bots.",
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
    parser.add_argument(
        "--include-state-snapshots",
        action="store_true",
        help="Write compact pre-action state snapshots for action improvement tools.",
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
            f" value_checkpoint={args.search_value_checkpoint}"
            f" value_weight={args.search_value_weight}"
            f" value_min_turn_index={args.search_value_min_turn_index}"
        )
    if any("checkpoint" in pairing for pairing in pairing_labels):
        print(
            f"[corpus] checkpoint path={args.checkpoint_path}"
            f" model_kind={args.checkpoint_model_kind}"
            f" seat0_path={args.seat0_checkpoint_path}"
            f" seat0_kind={args.seat0_checkpoint_model_kind}"
            f" seat1_path={args.seat1_checkpoint_path}"
            f" seat1_kind={args.seat1_checkpoint_model_kind}"
            f" device={args.checkpoint_device}"
            f" loop_fallback_enabled={not args.disable_checkpoint_loop_fallback}"
            f" policy_prior_weight={args.policy_prior_weight}"
            f" search_value_checkpoint={args.search_value_checkpoint}"
            f" search_value_weight={args.search_value_weight}"
            f" search_value_min_turn_index={args.search_value_min_turn_index}"
        )

    stalled_games = 0
    timed_out_games = 0
    loop_fallback_triggers = 0

    def progress_callback(done: int, total: int, replay_game, elapsed: float) -> None:
        nonlocal stalled_games, timed_out_games, loop_fallback_triggers
        if replay_game.stalled:
            stalled_games += 1
        if replay_game.timed_out:
            timed_out_games += 1
        loop_fallback_triggers += sum(replay_game.loop_fallback_triggers_by_seat)
        if done == total or done == 1 or (args.log_every > 0 and done % args.log_every == 0):
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = ((total - done) / rate) if rate > 0 else math.inf
            eta_text = f"{eta:.1f}s" if math.isfinite(eta) else "inf"
            print(
                f"[corpus] {done}/{total} games | elapsed={elapsed:.1f}s | "
                f"rate={rate:.2f} games/s | eta={eta_text} | "
                f"last_turns={replay_game.turns} | last_reason={replay_game.termination_reason} | "
                f"stalled={stalled_games} | timed_out={timed_out_games} | "
                f"loop_fallback_triggers={loop_fallback_triggers}"
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
        include_state_snapshots=args.include_state_snapshots,
    )
    print("[corpus] generation complete, writing replay files...")
    metadata = _summary_metadata(args, pairing_labels)
    replay_path, summary_path = write_replay_corpus(
        output_dir=args.output_dir,
        games=games,
        summary=summary,
        metadata=metadata,
    )
    manifest_path = write_replay_corpus_manifest(
        output_dir=args.output_dir,
        games=games,
        summary=summary,
        metadata=metadata,
        source_type=args.source_type,
    )
    stalled_trace_path = replay_path.parent / "stalled_traces.jsonl"
    timed_out_trace_path = replay_path.parent / "timed_out_traces.jsonl"
    print(f"wrote replay corpus: {replay_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote manifest: {manifest_path}")
    if stalled_trace_path.exists():
        print(f"wrote stalled traces: {stalled_trace_path}")
    if timed_out_trace_path.exists():
        print(f"wrote timed-out traces: {timed_out_trace_path}")
    print(
        f"games={summary.games} steps={summary.total_steps} "
        f"stalled={summary.stalled_games} stalled_rate="
        f"{(summary.stalled_games / summary.games) if summary.games else 0.0:.4f} "
        f"timed_out={summary.timed_out_games} timed_out_rate="
        f"{(summary.timed_out_games / summary.games) if summary.games else 0.0:.4f} "
        f"loop_fallback_triggers={summary.loop_fallback_triggers}"
    )


def _resolve_pairings(args: argparse.Namespace):
    if not args.pairing:
        return (
            (
                _build_bot_factory(args.seat0_bot, args.seat0_seed, args, seat_index=0),
                _build_bot_factory(args.seat1_bot, args.seat1_seed, args, seat_index=1),
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
                _build_bot_factory(left_name, args.seat0_seed, args, seat_index=0),
                _build_bot_factory(right_name, args.seat1_seed, args, seat_index=1),
            )
        )
    return tuple(pairings)


def _resolve_checkpoint_path(
    args: argparse.Namespace,
    bot_name: str,
    seat_index: int | None = None,
) -> Path:
    seat_path = None
    if seat_index == 0:
        seat_path = args.seat0_checkpoint_path
    elif seat_index == 1:
        seat_path = args.seat1_checkpoint_path
    checkpoint_path = seat_path or args.checkpoint_path
    if checkpoint_path is None:
        raise ValueError(
            f"Bot '{bot_name}' requires --checkpoint-path to be set."
        )
    return Path(checkpoint_path)


def _resolve_checkpoint_model_kind(
    args: argparse.Namespace,
    seat_index: int | None = None,
) -> str:
    if seat_index == 0 and args.seat0_checkpoint_model_kind is not None:
        return args.seat0_checkpoint_model_kind
    if seat_index == 1 and args.seat1_checkpoint_model_kind is not None:
        return args.seat1_checkpoint_model_kind
    return args.checkpoint_model_kind


def _checkpoint_loop_fallback_config(args: argparse.Namespace) -> LoopFallbackConfig:
    return LoopFallbackConfig(
        enabled=not args.disable_checkpoint_loop_fallback,
        min_state_visits=args.checkpoint_loop_fallback_min_state_visits,
        min_own_non_progress_actions=args.checkpoint_loop_fallback_min_own_no_progress_actions,
        max_buy_logit_gap=args.checkpoint_loop_fallback_max_buy_logit_gap,
    )


def _build_value_evaluator(args: argparse.Namespace) -> CheckpointValueEvaluator | None:
    if args.search_value_checkpoint is None or args.search_value_weight == 0.0:
        return None
    return CheckpointValueEvaluator(
        checkpoint_path=args.search_value_checkpoint,
        device=args.search_value_device,
        value_scale=args.search_value_scale,
        prediction_mode=args.search_value_prediction_mode,
        outcome_blend_weight=args.search_value_outcome_blend_weight,
    )


def _summary_metadata(args: argparse.Namespace, pairing_labels: list[str]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "pairings": pairing_labels,
        "source_type": args.source_type,
        "swap_seats": args.swap_seats,
        "include_state_snapshots": args.include_state_snapshots,
        "seed_start": args.seed_start,
        "max_turns": args.max_turns,
        "repetition_limit": args.repetition_limit,
        "no_progress_limit": args.no_progress_limit,
        "search": {
            "depth": args.search_depth,
            "information_legal": not args.allow_oracle_search,
            "determinizations": args.search_determinizations,
            "max_branching": args.search_max_branching,
            "buy_branching": args.search_buy_branching,
            "reserve_branching": args.search_reserve_branching,
            "take_branching": args.search_take_branching,
            "value_checkpoint": args.search_value_checkpoint,
            "value_device": args.search_value_device,
            "value_weight": args.search_value_weight,
            "value_scale": args.search_value_scale,
            "value_prediction_mode": args.search_value_prediction_mode,
            "value_outcome_blend_weight": args.search_value_outcome_blend_weight,
            "value_min_turn_index": args.search_value_min_turn_index,
            "value_max_turn_index": args.search_value_max_turn_index,
        },
        "sampling": {
            "epsilon": args.sampling_epsilon,
            "temperature": args.sampling_temperature,
            "top_k": args.sampling_top_k,
        },
        "puct": {
            "simulations": args.puct_simulations,
            "c_puct": args.puct_c_puct,
            "prior_temperature": args.puct_prior_temperature,
            "fpu_reduction": args.puct_fpu_reduction,
            "max_depth": args.puct_max_depth,
            "heuristic_scale": args.puct_heuristic_scale,
            "inference_batch_size": args.puct_inference_batch_size,
            "tree_reuse": args.puct_tree_reuse,
            "max_tree_nodes": args.puct_max_tree_nodes,
            "value_checkpoint": args.puct_value_checkpoint,
            "value_min_turn_index": args.puct_value_min_turn_index,
            "value_max_turn_index": args.puct_value_max_turn_index,
            "value_prediction_mode": args.puct_value_prediction_mode,
            "value_outcome_blend_weight": (
                args.puct_value_outcome_blend_weight
            ),
            "n_step_leaf_depth": args.puct_n_step_leaf_depth,
            "n_step_leaf_width": args.puct_n_step_leaf_width,
            "n_step_leaf_candidate_limit": args.puct_n_step_leaf_candidate_limit,
            "n_step_leaf_weight": args.puct_n_step_leaf_weight,
            "n_step_leaf_delta_clip": args.puct_n_step_leaf_delta_clip,
            "n_step_leaf_min_turn_index": args.puct_n_step_leaf_min_turn_index,
            "n_step_leaf_max_turn_index": args.puct_n_step_leaf_max_turn_index,
            "n_step_leaf_min_max_score": args.puct_n_step_leaf_min_max_score,
            "rollout_leaf_count": args.puct_rollout_leaf_count,
            "rollout_leaf_max_depth": args.puct_rollout_leaf_max_depth,
            "rollout_leaf_top_k": args.puct_rollout_leaf_top_k,
            "rollout_leaf_temperature": args.puct_rollout_leaf_temperature,
            "rollout_leaf_policy": args.puct_rollout_leaf_policy,
        },
    }
    if any(
        "checkpoint" in pairing or "puct" in pairing
        for pairing in pairing_labels
    ):
        default_model_kind = (
            "information_set_puct"
            if any("puct" in pairing for pairing in pairing_labels)
            else args.checkpoint_model_kind
        )
        metadata["checkpoint"] = {
            "path": args.checkpoint_path,
            "model_kind": default_model_kind,
            "seat0": {
                "path": args.seat0_checkpoint_path or args.checkpoint_path,
                "model_kind": (
                    args.seat0_checkpoint_model_kind or default_model_kind
                ),
            },
            "seat1": {
                "path": args.seat1_checkpoint_path or args.checkpoint_path,
                "model_kind": (
                    args.seat1_checkpoint_model_kind or default_model_kind
                ),
            },
            "device": args.checkpoint_device,
            "policy_prior_weight": args.policy_prior_weight,
            "loop_fallback": {
                "enabled": not args.disable_checkpoint_loop_fallback,
                "min_state_visits": args.checkpoint_loop_fallback_min_state_visits,
                "min_own_non_progress_actions": args.checkpoint_loop_fallback_min_own_no_progress_actions,
                "max_buy_logit_gap": args.checkpoint_loop_fallback_max_buy_logit_gap,
            },
        }
    return metadata


if __name__ == "__main__":
    main()
