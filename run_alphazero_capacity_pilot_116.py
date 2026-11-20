"""Run the frozen identity-preserving medium-capacity pilot 116."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import torch

from splendor_ai.alphazero.capacity_pilot import CapacityPilotConfig, run_capacity_pilot_116


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_capacity_pilot_116")
    parser.add_argument(
        "--result-114",
        default="outputs/alphazero_high_budget_distillation_114/result.json",
    )
    parser.add_argument(
        "--result-115",
        default="outputs/alphazero_high_budget_replication_115/result.json",
    )
    parser.add_argument(
        "--partition-manifest",
        default="outputs/alphazero_high_budget_distillation_114/partition/train_holdout.stage.json",
    )
    parser.add_argument(
        "--train-replay",
        default="outputs/alphazero_high_budget_distillation_114/partition/train_224.jsonl",
    )
    parser.add_argument(
        "--untouched-holdout",
        default="outputs/alphazero_high_budget_distillation_114/partition/holdout_32.jsonl",
    )
    parser.add_argument(
        "--installed-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
    parser.add_argument(
        "--generation-11-checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--arena-workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = run_capacity_pilot_116(
        CapacityPilotConfig(
            output_dir=Path(args.output_dir),
            result_114=Path(args.result_114),
            result_115=Path(args.result_115),
            partition_manifest=Path(args.partition_manifest),
            train_replay=Path(args.train_replay),
            untouched_holdout=Path(args.untouched_holdout),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
            arena_workers=args.arena_workers,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
