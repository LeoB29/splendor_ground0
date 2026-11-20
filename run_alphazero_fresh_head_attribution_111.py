"""Run the frozen fresh-dose head-attribution experiment 111."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import torch

from splendor_ai.alphazero.fresh_head_attribution import (
    FreshHeadAttributionConfig,
    run_fresh_head_attribution_111,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_fresh_head_attribution_111")
    parser.add_argument("--result-110", default="outputs/alphazero_fresh_dose_110/result.json")
    parser.add_argument(
        "--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt"
    )
    parser.add_argument(
        "--generation-11-checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument(
        "--high-dose-checkpoint",
        default="outputs/alphazero_fresh_dose_110/checkpoints/high_dose_generation_012.pt",
    )
    parser.add_argument(
        "--high-dose-replay", default="outputs/alphazero_fresh_dose_110/partition/high_896.jsonl"
    )
    parser.add_argument(
        "--common-holdout", default="outputs/alphazero_fresh_dose_110/partition/holdout_128.jsonl"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--arena-workers", type=int, default=3)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = run_fresh_head_attribution_111(
        FreshHeadAttributionConfig(
            output_dir=Path(args.output_dir),
            result_110=Path(args.result_110),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            high_dose_checkpoint=Path(args.high_dose_checkpoint),
            high_dose_replay=Path(args.high_dose_replay),
            common_holdout=Path(args.common_holdout),
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
