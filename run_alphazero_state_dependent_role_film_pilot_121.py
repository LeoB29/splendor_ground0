from __future__ import annotations

import argparse
import json
from pathlib import Path

from splendor_ai.alphazero.start_role_film_pilot import (
    StateDependentRoleFiLMPilotConfig,
    run_state_dependent_role_film_pilot_121,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or explicitly launch state-dependent role FiLM pilot 121."
    )
    parser.add_argument("--output-dir", default="outputs/alphazero_state_dependent_role_film_pilot_121")
    parser.add_argument("--result-120", default="outputs/alphazero_split_readout_role_pilot_120/result.json")
    parser.add_argument("--result-119", default="outputs/alphazero_start_role_interaction_audit_119/result.json")
    parser.add_argument("--result-118", default="outputs/alphazero_start_role_joint_pilot_118/result.json")
    parser.add_argument("--result-117", default="outputs/alphazero_start_role_audit_117/result.json")
    parser.add_argument("--generation-11-checkpoint", default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")
    parser.add_argument("--candidate-114", default="outputs/alphazero_high_budget_distillation_114/checkpoints/high_budget_generation_012.pt")
    parser.add_argument("--candidate-116", default="outputs/alphazero_capacity_pilot_116/checkpoints/capacity_generation_012.pt")
    parser.add_argument("--partition-manifest", default="outputs/alphazero_high_budget_distillation_114/partition/train_holdout.stage.json")
    parser.add_argument("--train-replay", default="outputs/alphazero_high_budget_distillation_114/partition/train_224.jsonl")
    parser.add_argument("--holdout-replay", default="outputs/alphazero_high_budget_distillation_114/partition/holdout_32.jsonl")
    parser.add_argument("--training-target-dir", default="outputs/alphazero_start_role_joint_pilot_118/targets/training")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arena-workers", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--launch", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> StateDependentRoleFiLMPilotConfig:
    return StateDependentRoleFiLMPilotConfig(
        output_dir=Path(args.output_dir),
        result_120=Path(args.result_120),
        result_119=Path(args.result_119),
        result_118=Path(args.result_118),
        result_117=Path(args.result_117),
        generation_11_checkpoint=Path(args.generation_11_checkpoint),
        candidate_114=Path(args.candidate_114),
        candidate_116=Path(args.candidate_116),
        partition_manifest=Path(args.partition_manifest),
        train_replay=Path(args.train_replay),
        holdout_replay=Path(args.holdout_replay),
        training_target_dir=Path(args.training_target_dir),
        installed_checkpoint=Path(args.installed_checkpoint),
        device=args.device,
        torch_threads=args.torch_threads,
        workers=args.workers,
        arena_workers=args.arena_workers,
        max_turns=args.max_turns,
    )


def main() -> int:
    args = build_parser().parse_args()
    result = run_state_dependent_role_film_pilot_121(
        config_from_args(args), launch=args.launch
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
