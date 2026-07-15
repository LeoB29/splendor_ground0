"""Run matched terminal-margin versus dense score-change actor-critic training."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from splendor_ai.eval.run_benchmark import benchmark_head_to_head
from splendor_ai.training.entity_actor_critic import (
    ActorCriticArtifacts,
    ActorCriticConfig,
    collect_league_rollouts,
    load_entity_actor,
    train_actor_critic_from_steps,
)
from splendor_ai.training.model import PrivilegedEntityCriticConfig
from splendor_ai.training.run_entity_return_ab import _full_league, _matchup_passes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-games", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=170_000)
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
    parser.add_argument("--minimum-dense-reward-fraction-ratio", type=float, default=1.5)
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
    if args.minimum_dense_reward_fraction_ratio <= 1.0:
        raise ValueError("minimum-dense-reward-fraction-ratio must exceed 1.")

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
    source_steps, rollout = collect_league_rollouts(
        load_entity_actor(parent, args.device), league, base_config
    )
    critic_config = PrivilegedEntityCriticConfig(
        hidden_size=64,
        layers=1,
        attention_heads=4,
        attention_feedforward_size=128,
        id_embedding_size=8,
    )
    terminal_config = replace(base_config, return_mode="score_margin")
    dense_config = replace(base_config, return_mode="score_margin_delta")
    terminal = train_actor_critic_from_steps(
        parent, source_steps, rollout, terminal_config, critic_config=critic_config
    )
    dense = train_actor_critic_from_steps(
        parent, source_steps, rollout, dense_config, critic_config=critic_config
    )
    terminal_actor_path, terminal_critic_path = _save_branch(
        output_dir, "terminal_margin", parent, terminal_config, terminal
    )
    dense_actor_path, dense_critic_path = _save_branch(
        output_dir, "dense_score_change", parent, dense_config, dense
    )

    terminal_update = terminal.update
    dense_update = dense.update
    gates = {
        "clean_balanced_rollout": (
            rollout.timed_out_games == 0
            and rollout.stalled_games == 0
            and rollout.minimum_stratum_games > 0
            and rollout.minimum_stratum_games == rollout.maximum_stratum_games
        ),
        "terminal_branch_safe": _branch_safe(terminal, args),
        "dense_branch_safe": _branch_safe(dense, args),
        "dense_rewards_are_materially_less_sparse": (
            dense_update.nonzero_temporal_reward_fraction
            >= args.minimum_dense_reward_fraction_ratio
            * terminal_update.nonzero_temporal_reward_fraction
        ),
        "dense_critic_skill_improved": (
            dense_update.critic_validation_skill - terminal_update.critic_validation_skill
            >= args.minimum_critic_skill_delta
        ),
        "dense_critic_correlation_improved": (
            dense_update.critic_validation_correlation
            - terminal_update.critic_validation_correlation
            >= args.minimum_critic_correlation_delta
        ),
        "dense_advantage_correlation_improved": (
            abs(dense_update.advantage_log_probability_correlation)
            - abs(terminal_update.advantage_log_probability_correlation)
            >= args.minimum_advantage_correlation_delta
        ),
    }
    offline_gates_passed = all(gates.values())
    treatment_control_gate = None
    treatment_parent_gate = None
    if offline_gates_passed:
        treatment_control_gate = benchmark_head_to_head(
            candidate_checkpoint_path=dense_actor_path,
            baseline_checkpoint_path=terminal_actor_path,
            candidate_model_kind="canonical_entity_action_scorer",
            baseline_model_kind="canonical_entity_action_scorer",
            device=args.device,
            games=args.initial_gate_games,
            seed_start=args.seed_start + 1_000,
            paired_seeds=True,
            log_every=0,
        )
        treatment_parent_gate = benchmark_head_to_head(
            candidate_checkpoint_path=dense_actor_path,
            baseline_checkpoint_path=parent,
            candidate_model_kind="canonical_entity_action_scorer",
            baseline_model_kind="canonical_entity_action_scorer",
            device=args.device,
            games=args.initial_gate_games,
            seed_start=args.seed_start + 2_000,
            paired_seeds=True,
            log_every=0,
        )
        gates["treatment_beats_control"] = _matchup_passes(
            treatment_control_gate["matchup"],
            args.initial_gate_games,
            args.minimum_treatment_control_score_share,
            args.maximum_gate_timeouts,
        )
        gates["treatment_beats_parent"] = _matchup_passes(
            treatment_parent_gate["matchup"],
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
        "terminal_margin_branch": _branch_payload(
            terminal_actor_path, terminal_critic_path, terminal_config, terminal
        ),
        "dense_score_change_branch": _branch_payload(
            dense_actor_path, dense_critic_path, dense_config, dense
        ),
        "diagnostic_deltas": {
            "nonzero_temporal_reward_fraction": (
                dense_update.nonzero_temporal_reward_fraction
                - terminal_update.nonzero_temporal_reward_fraction
            ),
            "critic_validation_skill": (
                dense_update.critic_validation_skill - terminal_update.critic_validation_skill
            ),
            "critic_validation_correlation": (
                dense_update.critic_validation_correlation
                - terminal_update.critic_validation_correlation
            ),
            "absolute_advantage_log_probability_correlation": (
                abs(dense_update.advantage_log_probability_correlation)
                - abs(terminal_update.advantage_log_probability_correlation)
            ),
            "advantage_std": dense_update.advantage_std - terminal_update.advantage_std,
        },
        "offline_gates_passed": offline_gates_passed,
        "treatment_control_gate": treatment_control_gate,
        "treatment_parent_gate": treatment_parent_gate,
        "gate_config": {
            "minimum_dense_reward_fraction_ratio": args.minimum_dense_reward_fraction_ratio,
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
    (output_dir / "temporal_credit_ab_metrics.json").write_text(
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
        and args.minimum_effective_kl <= update.mean_kl <= args.maximum_kl
        and update.mean_centered_logit_absolute_delta >= args.minimum_centered_logit_delta
        and update.max_ratio <= 1.5
        and update.mean_normalized_entropy >= 0.35
        and update.critic_validation_steps > 0
    )


def _branch_payload(
    actor_path: Path,
    critic_path: Path,
    config: ActorCriticConfig,
    artifacts: ActorCriticArtifacts,
) -> dict[str, object]:
    return {
        "actor_checkpoint": str(actor_path),
        "critic_checkpoint": str(critic_path),
        "config": asdict(config),
        "update": asdict(artifacts.update),
    }


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
            "matched_temporal_credit_ab": True,
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


if __name__ == "__main__":
    main()
