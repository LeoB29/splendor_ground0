from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import torch

from splendor_ai.alphazero.repeated_return_value_ab import (
    RepeatedReturnValueABConfig,
    run_repeated_return_value_ab_131,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight or launch repeated-return value A/B 131.")
    parser.add_argument("--output-dir", default="outputs/alphazero_repeated_return_value_ab_131")
    parser.add_argument("--result-130", default="outputs/alphazero_value_uncertainty_audit_130/result.json")
    parser.add_argument("--generation-11-checkpoint", default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument(
        "--source-replays",
        nargs=4,
        default=[
            "outputs/alphazero_scaled_loop_099/selfplay/generation_002.jsonl",
            "outputs/alphazero_scaled_continuation_103/selfplay/generation_005_sanitized.jsonl",
            "outputs/alphazero_continued_scaling_104/selfplay/generation_008_sanitized.jsonl",
            "outputs/alphazero_high_budget_distillation_114/partition/train_224.jsonl",
        ],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--collection-workers", type=int, default=4)
    parser.add_argument("--arena-workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--launch", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = run_repeated_return_value_ab_131(
        RepeatedReturnValueABConfig(
            output_dir=Path(args.output_dir),
            result_130=Path(args.result_130),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            installed_checkpoint=Path(args.installed_checkpoint),
            source_replays=tuple(Path(path) for path in args.source_replays),
            device=args.device,
            collection_workers=args.collection_workers,
            arena_workers=args.arena_workers,
            torch_threads=args.torch_threads,
            max_turns=args.max_turns,
        ),
        launch=args.launch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
