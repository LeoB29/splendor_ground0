"""Run the frozen high-budget distillation pilot 114."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import torch

from splendor_ai.alphazero.high_budget_distillation import (
    HighBudgetDistillationConfig,
    run_high_budget_distillation_114,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_high_budget_distillation_114")
    parser.add_argument("--result-113", default="outputs/alphazero_search_headroom_113/result.json")
    parser.add_argument(
        "--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt"
    )
    parser.add_argument(
        "--generation-11-checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--shard-workers", type=int, default=4)
    parser.add_argument("--arena-workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = run_high_budget_distillation_114(
        HighBudgetDistillationConfig(
            output_dir=Path(args.output_dir),
            result_113=Path(args.result_113),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
            shard_workers=args.shard_workers,
            arena_workers=args.arena_workers,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
