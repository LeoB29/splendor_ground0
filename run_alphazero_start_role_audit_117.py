"""Run the frozen start-role representation audit 117."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import torch

from splendor_ai.alphazero.start_role_audit import (
    StartRoleAuditConfig,
    run_start_role_audit_117,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="outputs/alphazero_start_role_audit_117"
    )
    parser.add_argument(
        "--result-114",
        default="outputs/alphazero_high_budget_distillation_114/result.json",
    )
    parser.add_argument(
        "--candidate-114",
        default="outputs/alphazero_high_budget_distillation_114/checkpoints/high_budget_generation_012.pt",
    )
    parser.add_argument(
        "--result-116", default="outputs/alphazero_capacity_pilot_116/result.json"
    )
    parser.add_argument(
        "--candidate-116",
        default="outputs/alphazero_capacity_pilot_116/checkpoints/capacity_generation_012.pt",
    )
    parser.add_argument(
        "--generation-11-checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument(
        "--untouched-holdout",
        default="outputs/alphazero_high_budget_distillation_114/partition/holdout_32.jsonl",
    )
    parser.add_argument(
        "--installed-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = run_start_role_audit_117(
        StartRoleAuditConfig(
            output_dir=Path(args.output_dir),
            result_114=Path(args.result_114),
            candidate_114=Path(args.candidate_114),
            result_116=Path(args.result_116),
            candidate_116=Path(args.candidate_116),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            untouched_holdout=Path(args.untouched_holdout),
            installed_checkpoint=Path(args.installed_checkpoint),
            device=args.device,
            torch_threads=args.torch_threads,
            workers=args.workers,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
