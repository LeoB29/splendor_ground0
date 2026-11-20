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
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS


PROTOCOL_SHA256 = "31461066a80133938bbd37c56f1e61906d2e2f28d5311aa2e9bcf22355f92e84"
RESULT_131_SHA256 = "dd0e0cd2cec25c5ba1fbf52e25bd81d75f3c548b9706c4dd056642343301d9dc"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or launch deadlock-consistent repeated-return value A/B 132."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/alphazero_deadlock_consistent_repeated_return_value_ab_132",
    )
    parser.add_argument(
        "--result-131",
        default="outputs/alphazero_repeated_return_value_ab_131/result.json",
    )
    parser.add_argument(
        "--result-130",
        default="outputs/alphazero_value_uncertainty_audit_130/result.json",
    )
    parser.add_argument(
        "--generation-11-checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument(
        "--installed-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
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
            experiment_number=132,
            protocol_sha256=PROTOCOL_SHA256,
            valid_termination_reasons=VALID_NATURAL_TERMINATIONS,
            predecessor_result=Path(args.result_131),
            predecessor_result_sha256=RESULT_131_SHA256,
            predecessor_required_decision=(
                "repeated_return_value_integrity_failed_no_strength_conclusion"
            ),
        ),
        launch=args.launch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
