"""Run one clean entity actor-critic generation and its CPU smoke gates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from splendor_ai.bots import (
    CanonicalEntityActionScorerBot,
    GreedyHeuristicBot,
    InformationSetPuctBot,
    LegalActionAttentionScorerBot,
    LegalActionScorerBot,
    PuctConfig,
    ShallowSearchBot,
)
from splendor_ai.eval.run_benchmark import benchmark_head_to_head
from splendor_ai.training.entity_actor_critic import (
    ActorCriticConfig,
    train_actor_critic_generation,
)
from splendor_ai.training.model import PrivilegedEntityCriticConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-games", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=80_000)
    parser.add_argument("--actor-epochs", type=int, default=1)
    parser.add_argument("--critic-warmup-epochs", type=int, default=3)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--uniform-mix", type=float, default=0.05)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--minimum-effective-kl", type=float, default=0.0)
    parser.add_argument("--minimum-centered-logit-delta", type=float, default=0.0)
    parser.add_argument("--maximum-kl", type=float, default=0.02)
    parser.add_argument(
        "--skip-parent-gate-below-movement",
        action="store_true",
        help="Skip the game gate unless every offline safety/movement gate passes.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parent-gate-games", type=int, default=20)
    parser.add_argument("--minimum-parent-score-share", type=float, default=0.4)
    parser.add_argument("--maximum-parent-timeouts", type=int, default=1_000_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.minimum_parent_score_share <= 1.0:
        raise ValueError("minimum-parent-score-share must be in [0, 1].")
    if args.maximum_parent_timeouts < 0:
        raise ValueError("maximum-parent-timeouts must be non-negative.")
    parent = Path(args.parent_checkpoint)
    checkpoint016 = Path(args.checkpoint_016)
    checkpoint035 = Path(args.checkpoint_035)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ActorCriticConfig(
        rollout_games=args.rollout_games,
        seed_start=args.seed_start,
        actor_epochs=args.actor_epochs,
        critic_warmup_epochs=args.critic_warmup_epochs,
        critic_epochs=args.critic_epochs,
        batch_size=args.batch_size,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        entropy_weight=args.entropy_weight,
        uniform_mix=args.uniform_mix,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )
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
    critic_config = PrivilegedEntityCriticConfig(
        hidden_size=64,
        layers=1,
        attention_heads=4,
        attention_feedforward_size=128,
        id_embedding_size=8,
    )
    artifacts = train_actor_critic_generation(
        parent,
        league,
        config,
        critic_config=critic_config,
    )
    actor_path = output_dir / "canonical_entity_actor.pt"
    critic_path = output_dir / "privileged_entity_critic.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": artifacts.actor.state_dict(),
            "model_config": asdict(artifacts.actor.config),
            "parent_checkpoint": str(parent),
            "train_config": asdict(config),
        },
        actor_path,
    )
    torch.save(
        {
            "model_kind": "privileged_entity_critic",
            "model_state_dict": artifacts.critic.state_dict(),
            "model_config": asdict(artifacts.critic.config),
            "training_only": True,
        },
        critic_path,
    )
    gates = {
        "behavior_reconstruction": artifacts.update.behavior_log_prob_max_error <= 1e-5,
        "full_support": artifacts.rollout.minimum_behavior_probability > 0.0,
        "action_type_coverage": artifacts.rollout.action_type_count >= 4,
        "non_greedy_exploration": artifacts.rollout.non_greedy_actions >= max(1, artifacts.rollout.steps // 10),
        "entropy_not_collapsed": artifacts.update.mean_normalized_entropy >= 0.35,
        "importance_ratios_bounded": artifacts.update.max_ratio <= 1.5,
        "kl_bounded": artifacts.update.mean_kl <= args.maximum_kl,
        "gradient_signal_present": artifacts.update.first_batch_gradient_norm > 1e-8,
        "effective_policy_movement": (
            artifacts.update.mean_kl >= args.minimum_effective_kl
            and artifacts.update.mean_centered_logit_absolute_delta
            >= args.minimum_centered_logit_delta
        ),
        "opponent_seat_balance": (
            artifacts.rollout.minimum_stratum_games > 0
            and artifacts.rollout.maximum_stratum_games
            - artifacts.rollout.minimum_stratum_games
            <= 1
        ),
        "clean_rollouts": artifacts.rollout.timed_out_games == 0 and artifacts.rollout.stalled_games == 0,
    }
    offline_gates_passed = all(gates.values())
    run_parent_gate = not args.skip_parent_gate_below_movement or offline_gates_passed
    parent_gate = None
    if run_parent_gate:
        parent_gate = benchmark_head_to_head(
            candidate_checkpoint_path=actor_path,
            baseline_checkpoint_path=parent,
            candidate_model_kind="canonical_entity_action_scorer",
            baseline_model_kind="canonical_entity_action_scorer",
            device=args.device,
            games=args.parent_gate_games,
            seed_start=args.seed_start + 1_000,
            paired_seeds=True,
            log_every=0,
        )
        matchup = parent_gate["matchup"]
        candidate_wins = int(matchup["candidate_wins"])
        draws = int(matchup["draws"])
        candidate_score_share = (candidate_wins + 0.5 * draws) / args.parent_gate_games
        gates["no_parent_regression_smoke"] = (
            candidate_score_share >= args.minimum_parent_score_share
        )
        gates["parent_gate_cutoffs_bounded"] = (
            int(matchup["timed_out_games"]) <= args.maximum_parent_timeouts
        )
    else:
        gates["no_parent_regression_smoke"] = False
        gates["parent_gate_cutoffs_bounded"] = False
    payload = {
        "parent_checkpoint": str(parent),
        "actor_checkpoint": str(actor_path),
        "critic_checkpoint": str(critic_path),
        "config": asdict(config),
        "rollout": asdict(artifacts.rollout),
        "update": asdict(artifacts.update),
        "parent_gate": parent_gate,
        "gate_config": {
            "minimum_effective_kl": args.minimum_effective_kl,
            "minimum_centered_logit_delta": args.minimum_centered_logit_delta,
            "maximum_kl": args.maximum_kl,
            "minimum_parent_score_share": args.minimum_parent_score_share,
            "maximum_parent_timeouts": args.maximum_parent_timeouts,
        },
        "offline_gates_passed": offline_gates_passed,
        "parent_gate_ran": run_parent_gate,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    (output_dir / "actor_critic_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({"gates": gates, "all_gates_passed": all(gates.values())}, indent=2))


if __name__ == "__main__":
    main()
