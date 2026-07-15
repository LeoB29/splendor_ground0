"""Validate score margin as a denser return without updating the actor."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    GreedyHeuristicBot,
    InformationSetPuctBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    PuctConfig,
    ShallowSearchBot,
)
from splendor_ai.training.entity_actor_critic import (
    ActorCriticConfig,
    collect_league_rollouts,
    load_entity_actor,
)
from splendor_ai.training.entity_return_validation import (
    game_return_records,
    summarize_return_targets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--rollout-games", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=140_000)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--score-margin-scale", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--minimum-sign-alignment", type=float, default=0.95)
    parser.add_argument("--maximum-saturation-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-all-loss-margin-unique", type=int, default=3)
    parser.add_argument("--minimum-all-loss-normalized-std", type=float, default=0.03)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    parent = Path(args.parent_checkpoint)
    checkpoint016 = Path(args.checkpoint_016)
    checkpoint035 = Path(args.checkpoint_035)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = ActorCriticConfig(
        rollout_games=args.rollout_games,
        seed_start=args.seed_start,
        uniform_mix=args.uniform_mix,
        score_margin_scale=args.score_margin_scale,
        device=args.device,
    )
    actor = load_entity_actor(parent, args.device)
    league = [
        ("entity004", lambda: CanonicalEntityActionScorerBot(parent, device=args.device)),
        ("greedy", lambda: GreedyHeuristicBot(seed=31)),
        (
            "legal_shallow_d1",
            lambda: ShallowSearchBot(
                depth=1,
                max_branching=6,
                max_buy_actions=4,
                max_reserve_actions=2,
                max_take_actions=2,
                information_legal=True,
                determinizations=1,
                seed=37,
                determinization_seed=41,
            ),
        ),
        ("policy016", lambda: LegalActionScorerBot(checkpoint016, device=args.device)),
        ("policy035", lambda: LegalActionAttentionScorerBot(checkpoint035, device=args.device)),
        (
            "puct035_s8",
            lambda: InformationSetPuctBot(
                checkpoint035,
                device=args.device,
                config=PuctConfig(simulations=8, inference_batch_size=4, seed=43),
            ),
        ),
    ]
    steps, rollout = collect_league_rollouts(actor, league, config)
    records = game_return_records(steps)
    summary = summarize_return_targets(
        records,
        minimum_all_loss_margin_unique=args.minimum_all_loss_margin_unique,
        minimum_all_loss_normalized_std=args.minimum_all_loss_normalized_std,
    )
    gates = {
        "all_games_recorded": summary.games == rollout.games,
        "clean_rollouts": rollout.timed_out_games == 0 and rollout.stalled_games == 0,
        "opponent_seat_balance": (
            rollout.minimum_stratum_games > 0
            and rollout.minimum_stratum_games == rollout.maximum_stratum_games
        ),
        "outcome_direction_preserved": (
            summary.outcome_sign_alignment >= args.minimum_sign_alignment
        ),
        "score_margin_is_denser": (
            summary.normalized_score_margin_unique_count > summary.outcome_unique_count
        ),
        "score_margin_not_saturated": (
            summary.saturation_fraction <= args.maximum_saturation_fraction
        ),
        "all_loss_strata_present": summary.all_loss_strata > 0,
        "all_loss_strata_gain_variation": (
            summary.all_loss_strata_with_margin_variation == summary.all_loss_strata
        ),
    }
    payload = {
        "parent_checkpoint": str(parent),
        "config": asdict(config),
        "return_definition": {
            "raw": "actor final score - opponent final score",
            "normalized": f"clip(raw / {args.score_margin_scale:g}, -1, 1)",
        },
        "validation_config": {
            "minimum_sign_alignment": args.minimum_sign_alignment,
            "maximum_saturation_fraction": args.maximum_saturation_fraction,
            "minimum_all_loss_margin_unique": args.minimum_all_loss_margin_unique,
            "minimum_all_loss_normalized_std": args.minimum_all_loss_normalized_std,
        },
        "rollout": asdict(rollout),
        "return_summary": asdict(summary),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"gates": gates, "all_gates_passed": all(gates.values())}, indent=2))


if __name__ == "__main__":
    main()
