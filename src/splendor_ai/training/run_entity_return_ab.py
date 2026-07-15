"""Run a matched binary-outcome versus score-margin actor-critic experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
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
    ActorCriticArtifacts,
    ActorCriticConfig,
    collect_league_rollouts,
    load_entity_actor,
    train_actor_critic_from_steps,
)
from splendor_ai.training.model import PrivilegedEntityCriticConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-games", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=160_000)
    parser.add_argument("--actor-epochs", type=int, default=6)
    parser.add_argument("--critic-warmup-epochs", type=int, default=5)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--critic-validation-modulus", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=3e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float, default=0.002)
    parser.add_argument("--maximum-kl", type=float, default=0.01)
    parser.add_argument("--minimum-effective-kl", type=float, default=0.0005)
    parser.add_argument("--minimum-centered-logit-delta", type=float, default=0.005)
    parser.add_argument("--minimum-critic-skill-delta", type=float, default=0.0)
    parser.add_argument("--minimum-critic-correlation-delta", type=float, default=0.0)
    parser.add_argument("--minimum-advantage-correlation-delta", type=float, default=0.0)
    parser.add_argument("--initial-gate-games", type=int, default=60)
    parser.add_argument("--minimum-treatment-control-score-share", type=float, default=0.55)
    parser.add_argument("--minimum-treatment-parent-score-share", type=float, default=0.60)
    parser.add_argument("--maximum-gate-timeouts", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.initial_gate_games <= 0 or args.initial_gate_games % 2:
        raise ValueError("initial-gate-games must be a positive even number.")
    parent = Path(args.parent_checkpoint)
    checkpoint016 = Path(args.checkpoint_016)
    checkpoint035 = Path(args.checkpoint_035)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = ActorCriticConfig(
        rollout_games=args.rollout_games,
        seed_start=args.seed_start,
        actor_epochs=args.actor_epochs,
        critic_warmup_epochs=args.critic_warmup_epochs,
        critic_epochs=args.critic_epochs,
        critic_validation_modulus=args.critic_validation_modulus,
        batch_size=args.batch_size,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        target_kl=args.target_kl,
        device=args.device,
    )
    league = _full_league(parent, checkpoint016, checkpoint035, args.device)
    behavior_actor = load_entity_actor(parent, args.device)
    source_steps, rollout = collect_league_rollouts(behavior_actor, league, base_config)
    critic_config = PrivilegedEntityCriticConfig(
        hidden_size=64,
        layers=1,
        attention_heads=4,
        attention_feedforward_size=128,
        id_embedding_size=8,
    )
    outcome_config = replace(base_config, return_mode="outcome")
    margin_config = replace(base_config, return_mode="score_margin")
    outcome = train_actor_critic_from_steps(
        parent, source_steps, rollout, outcome_config, critic_config=critic_config
    )
    margin = train_actor_critic_from_steps(
        parent, source_steps, rollout, margin_config, critic_config=critic_config
    )
    outcome_actor_path, outcome_critic_path = _save_branch(
        output_dir, "outcome", parent, outcome_config, outcome
    )
    margin_actor_path, margin_critic_path = _save_branch(
        output_dir, "score_margin", parent, margin_config, margin
    )

    gates = {
        "clean_balanced_rollout": (
            rollout.timed_out_games == 0
            and rollout.stalled_games == 0
            and rollout.minimum_stratum_games > 0
            and rollout.minimum_stratum_games == rollout.maximum_stratum_games
        ),
        "outcome_branch_safe": _branch_safe(outcome, args),
        "score_margin_branch_safe": _branch_safe(margin, args),
        "score_margin_critic_skill_improved": (
            margin.update.critic_validation_skill - outcome.update.critic_validation_skill
            >= args.minimum_critic_skill_delta
        ),
        "score_margin_critic_correlation_improved": (
            margin.update.critic_validation_correlation
            - outcome.update.critic_validation_correlation
            >= args.minimum_critic_correlation_delta
        ),
        "score_margin_advantage_correlation_improved": (
            abs(margin.update.advantage_log_probability_correlation)
            - abs(outcome.update.advantage_log_probability_correlation)
            >= args.minimum_advantage_correlation_delta
        ),
    }
    offline_gates_passed = all(gates.values())
    treatment_control_gate = None
    treatment_parent_gate = None
    if offline_gates_passed:
        treatment_control_gate = benchmark_head_to_head(
            candidate_checkpoint_path=margin_actor_path,
            baseline_checkpoint_path=outcome_actor_path,
            candidate_model_kind="canonical_entity_action_scorer",
            baseline_model_kind="canonical_entity_action_scorer",
            device=args.device,
            games=args.initial_gate_games,
            seed_start=args.seed_start + 1_000,
            paired_seeds=True,
            log_every=0,
        )
        treatment_parent_gate = benchmark_head_to_head(
            candidate_checkpoint_path=margin_actor_path,
            baseline_checkpoint_path=parent,
            candidate_model_kind="canonical_entity_action_scorer",
            baseline_model_kind="canonical_entity_action_scorer",
            device=args.device,
            games=args.initial_gate_games,
            seed_start=args.seed_start + 2_000,
            paired_seeds=True,
            log_every=0,
        )
        control_matchup = treatment_control_gate["matchup"]
        parent_matchup = treatment_parent_gate["matchup"]
        gates["treatment_beats_control"] = _matchup_passes(
            control_matchup,
            args.initial_gate_games,
            args.minimum_treatment_control_score_share,
            args.maximum_gate_timeouts,
        )
        gates["treatment_beats_parent"] = _matchup_passes(
            parent_matchup,
            args.initial_gate_games,
            args.minimum_treatment_parent_score_share,
            args.maximum_gate_timeouts,
        )
    else:
        gates["treatment_beats_control"] = False
        gates["treatment_beats_parent"] = False

    payload = {
        "parent_checkpoint": str(parent),
        "source_config": asdict(base_config),
        "league_opponents": [label for label, _ in league],
        "rollout": asdict(rollout),
        "outcome_branch": {
            "actor_checkpoint": str(outcome_actor_path),
            "critic_checkpoint": str(outcome_critic_path),
            "config": asdict(outcome_config),
            "update": asdict(outcome.update),
        },
        "score_margin_branch": {
            "actor_checkpoint": str(margin_actor_path),
            "critic_checkpoint": str(margin_critic_path),
            "config": asdict(margin_config),
            "update": asdict(margin.update),
        },
        "offline_gates_passed": offline_gates_passed,
        "treatment_control_gate": treatment_control_gate,
        "treatment_parent_gate": treatment_parent_gate,
        "gate_config": {
            "minimum_critic_skill_delta": args.minimum_critic_skill_delta,
            "minimum_critic_correlation_delta": args.minimum_critic_correlation_delta,
            "minimum_advantage_correlation_delta": args.minimum_advantage_correlation_delta,
            "minimum_treatment_control_score_share": (
                args.minimum_treatment_control_score_share
            ),
            "minimum_treatment_parent_score_share": args.minimum_treatment_parent_score_share,
            "maximum_gate_timeouts": args.maximum_gate_timeouts,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    (output_dir / "return_ab_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "offline_gates_passed": offline_gates_passed,
                "gates": gates,
                "all_gates_passed": all(gates.values()),
            },
            indent=2,
        )
    )


def _branch_safe(artifacts: ActorCriticArtifacts, args: argparse.Namespace) -> bool:
    update = artifacts.update
    return (
        update.behavior_log_prob_max_error <= 1e-5
        and update.mean_kl >= args.minimum_effective_kl
        and update.mean_kl <= args.maximum_kl
        and update.mean_centered_logit_absolute_delta >= args.minimum_centered_logit_delta
        and update.max_ratio <= 1.5
        and update.mean_normalized_entropy >= 0.35
        and update.critic_validation_steps > 0
    )


def _matchup_passes(
    matchup: dict[str, object],
    games: int,
    minimum_score_share: float,
    maximum_timeouts: int,
) -> bool:
    score_share = (int(matchup["candidate_wins"]) + 0.5 * int(matchup["draws"])) / games
    return score_share >= minimum_score_share and int(matchup["timed_out_games"]) <= maximum_timeouts


def _save_branch(
    output_dir: Path,
    label: str,
    parent: Path,
    config: ActorCriticConfig,
    artifacts: ActorCriticArtifacts,
) -> tuple[Path, Path]:
    actor_path = output_dir / f"canonical_entity_actor_{label}.pt"
    critic_path = output_dir / f"privileged_entity_critic_{label}.pt"
    torch.save(
        {
            "model_kind": "canonical_entity_action_scorer",
            "model_state_dict": artifacts.actor.state_dict(),
            "model_config": asdict(artifacts.actor.config),
            "parent_checkpoint": str(parent),
            "train_config": asdict(config),
            "matched_return_ab": True,
        },
        actor_path,
    )
    torch.save(
        {
            "model_kind": "privileged_entity_critic",
            "model_state_dict": artifacts.critic.state_dict(),
            "model_config": asdict(artifacts.critic.config),
            "training_only": True,
            "return_mode": config.return_mode,
        },
        critic_path,
    )
    return actor_path, critic_path


def _full_league(
    parent: Path,
    checkpoint016: Path,
    checkpoint035: Path,
    device: str,
) -> list[tuple[str, object]]:
    return [
        ("entity004", lambda: CanonicalEntityActionScorerBot(parent, device=device)),
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
        ("policy016", lambda: LegalActionScorerBot(checkpoint016, device=device)),
        ("policy035", lambda: LegalActionAttentionScorerBot(checkpoint035, device=device)),
        (
            "puct035_s8",
            lambda: InformationSetPuctBot(
                checkpoint035,
                device=device,
                config=PuctConfig(simulations=8, inference_batch_size=4, seed=43),
            ),
        ),
    ]


if __name__ == "__main__":
    main()
