from __future__ import annotations

import argparse
import json
from pathlib import Path

from splendor_ai.alphazero.start_role_policy_contrast_audit import (
    ScaleNativePolicyContrastAuditConfig,
    run_scale_native_policy_contrast_audit_124,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the no-training scale-native policy contrast audit 124."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/alphazero_scale_native_policy_contrast_audit_124",
    )
    parser.add_argument(
        "--result-123",
        default="outputs/alphazero_paired_role_objective_audit_123/result.json",
    )
    parser.add_argument(
        "--candidate-122",
        default=(
            "outputs/alphazero_partial_state_role_representation_pilot_122/"
            "checkpoints/partial_state_role_generation_012.pt"
        ),
    )
    parser.add_argument(
        "--generation-11-checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument(
        "--train-replay",
        default="outputs/alphazero_high_budget_distillation_114/partition/train_224.jsonl",
    )
    parser.add_argument(
        "--target-dir",
        default="outputs/alphazero_start_role_joint_pilot_118/targets/training",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--batch-pairs", type=int, default=32)
    return parser


def config_from_args(args: argparse.Namespace) -> ScaleNativePolicyContrastAuditConfig:
    return ScaleNativePolicyContrastAuditConfig(
        output_dir=Path(args.output_dir),
        result_123=Path(args.result_123),
        candidate_122=Path(args.candidate_122),
        generation_11_checkpoint=Path(args.generation_11_checkpoint),
        train_replay=Path(args.train_replay),
        target_dir=Path(args.target_dir),
        device=args.device,
        torch_threads=args.torch_threads,
        batch_pairs=args.batch_pairs,
    )


def main() -> int:
    args = build_parser().parse_args()
    result = run_scale_native_policy_contrast_audit_124(config_from_args(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
