"""Confirm mid/late entity counterfactual credit with fixed policy baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from splendor_ai.training.run_entity_counterfactual_credit import (
    EntityCounterfactualConfig,
    run_counterfactual_credit_diagnostic,
    summarize_counterfactual_states,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--checkpoint-016", required=True)
    parser.add_argument("--checkpoint-035", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--rollout-games", type=int, default=48)
    parser.add_argument("--seed-start", type=int, default=190_000)
    parser.add_argument("--early-control-states", type=int, default=6)
    parser.add_argument("--mid-states", type=int, default=12)
    parser.add_argument("--late-states", type=int, default=12)
    parser.add_argument("--policy-top-k", type=int, default=3)
    parser.add_argument("--reference-candidates", type=int, default=5)
    parser.add_argument("--determinizations", type=int, default=3)
    parser.add_argument("--continuations-per-determinization", type=int, default=8)
    parser.add_argument("--continuation-epsilon", type=float, default=0.1)
    parser.add_argument("--continuation-temperature", type=float, default=0.5)
    parser.add_argument("--continuation-top-k", type=int, default=3)
    parser.add_argument("--continuation-max-additional-turns", type=int, default=120)
    parser.add_argument("--minimum-action-range", type=float, default=0.05)
    parser.add_argument("--minimum-informative-fraction", type=float, default=0.50)
    parser.add_argument("--minimum-split-top1-agreement", type=float, default=0.50)
    parser.add_argument("--minimum-sign-eligible-states", type=int, default=6)
    parser.add_argument("--minimum-sign-agreement", type=float, default=0.70)
    parser.add_argument("--minimum-advantage-std", type=float, default=0.03)
    parser.add_argument("--minimum-advantage-tail-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-replicated-best-states", type=int, default=2)
    parser.add_argument("--maximum-continuation-cutoff-fraction", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    phase_targets = (
        args.early_control_states,
        args.mid_states,
        args.late_states,
    )
    config = EntityCounterfactualConfig(
        parent_checkpoint=Path(args.parent_checkpoint),
        checkpoint_016=Path(args.checkpoint_016),
        checkpoint_035=Path(args.checkpoint_035),
        output_path=Path(args.output_path),
        rollout_games=args.rollout_games,
        seed_start=args.seed_start,
        max_states=sum(phase_targets),
        phase_targets=phase_targets,
        policy_top_k=args.policy_top_k,
        max_candidates=args.reference_candidates,
        baseline_mode="policy_reference",
        determinizations=args.determinizations,
        continuations_per_determinization=args.continuations_per_determinization,
        continuation_epsilon=args.continuation_epsilon,
        continuation_temperature=args.continuation_temperature,
        continuation_top_k=args.continuation_top_k,
        continuation_max_additional_turns=args.continuation_max_additional_turns,
        minimum_action_range=args.minimum_action_range,
        minimum_informative_fraction=args.minimum_informative_fraction,
        minimum_split_top1_agreement=args.minimum_split_top1_agreement,
        minimum_chosen_advantage_sign_agreement=args.minimum_sign_agreement,
        minimum_sign_eligible_states=args.minimum_sign_eligible_states,
        minimum_chosen_advantage_std=args.minimum_advantage_std,
        minimum_advantage_tail_fraction=args.minimum_advantage_tail_fraction,
        maximum_continuation_cutoff_fraction=args.maximum_continuation_cutoff_fraction,
        bootstrap_samples=args.bootstrap_samples,
        device=args.device,
    )
    payload = run_counterfactual_credit_diagnostic(config)
    phase_summaries = {
        phase: summarize_counterfactual_states(
            [row for row in payload["states"] if row["phase"] == phase],
            minimum_action_range=args.minimum_action_range,
            minimum_advantage_magnitude=config.minimum_advantage_magnitude,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed_start + 301 + phase_index,
        )
        for phase_index, phase in enumerate(("early", "mid", "late"))
    }
    phase_gates = {
        phase: _phase_gate(summary, args)
        for phase, summary in phase_summaries.items()
        if phase in {"mid", "late"}
    }
    gates = {
        "clean_balanced_source_rollout": payload["gates"][
            "clean_balanced_source_rollout"
        ],
        "game_disjoint_preregistered_phase_counts": payload["gates"][
            "game_disjoint_phase_balanced_states"
        ],
        "continuations_complete": payload["gates"]["continuations_complete"],
        "action_independent_policy_reference_baseline": all(
            row["baseline_mode"] == "policy_reference" for row in payload["states"]
        ),
        "mid_phase_replicates": phase_gates["mid"]["passed"],
        "late_phase_replicates": phase_gates["late"]["passed"],
    }
    payload["diagnostic"] = "entity_mid_late_counterfactual_credit_confirmation"
    payload["phase_summaries"] = phase_summaries
    payload["phase_gates"] = phase_gates
    payload["confirmation_gates"] = gates
    payload["all_gates_passed"] = all(gates.values())
    payload["decision"] = (
        "Both mid and late counterfactual advantages replicate independently; "
        "a matched phase-restricted policy update is eligible."
        if payload["all_gates_passed"]
        else "Do not train: mid and late counterfactual credit did not both pass "
        "every independent confirmation gate."
    )
    Path(args.output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "phase_gates": phase_gates,
                "confirmation_gates": gates,
                "all_gates_passed": payload["all_gates_passed"],
            },
            indent=2,
        )
    )


def _phase_gate(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    criteria = {
        "informative_fraction": (
            summary["informative_fraction"] >= args.minimum_informative_fraction
        ),
        "split_top1_agreement": (
            summary["split_top1_agreement"] >= args.minimum_split_top1_agreement
        ),
        "split_spearman_lower95_positive": summary["split_spearman_ci95"][0] > 0.0,
        "chosen_advantage_correlation_lower95_positive": (
            summary["chosen_advantage_correlation_ci95"][0] > 0.0
        ),
        "chosen_advantage_sign_stable": (
            summary["chosen_advantage_sign_eligible_states"]
            >= args.minimum_sign_eligible_states
            and summary["chosen_advantage_sign_agreement"]
            >= args.minimum_sign_agreement
        ),
        "chosen_advantage_nondegenerate": (
            summary["chosen_advantage_std"] >= args.minimum_advantage_std
            and summary["chosen_advantage_positive_fraction"]
            >= args.minimum_advantage_tail_fraction
            and summary["chosen_advantage_negative_fraction"]
            >= args.minimum_advantage_tail_fraction
        ),
        "replicated_best_action_headroom": (
            summary["replicated_discovery_best_states"]
            >= args.minimum_replicated_best_states
        ),
    }
    return {"criteria": criteria, "passed": all(criteria.values())}


if __name__ == "__main__":
    main()
