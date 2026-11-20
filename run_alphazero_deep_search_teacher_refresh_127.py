from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from splendor_ai.alphazero.deep_search_teacher_refresh import (
    DeepSearchTeacherRefreshConfig,
    run_deep_search_teacher_refresh_127,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or launch deep-search teacher-refresh experiment 127."
    )
    parser.add_argument("--output-dir", default="outputs/alphazero_deep_search_teacher_refresh_127")
    parser.add_argument("--result-114", default="outputs/alphazero_high_budget_distillation_114/result.json")
    parser.add_argument("--result-115", default="outputs/alphazero_high_budget_replication_115/result.json")
    parser.add_argument("--result-126", default="outputs/alphazero_start_role_operational_audit_126/result.json")
    parser.add_argument("--generation-11-checkpoint", default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")
    parser.add_argument("--generation-12-checkpoint", default="outputs/alphazero_high_budget_distillation_114/checkpoints/high_budget_generation_012.pt")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument(
        "--initial-replays",
        nargs=4,
        default=[f"outputs/alphazero_high_budget_distillation_114/selfplay/shard_{index}/sanitized.jsonl" for index in range(4)],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--shard-workers", type=int, default=4)
    parser.add_argument("--arena-workers", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--launch", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = DeepSearchTeacherRefreshConfig(
        output_dir=Path(args.output_dir),
        result_114=Path(args.result_114),
        result_115=Path(args.result_115),
        result_126=Path(args.result_126),
        generation_11_checkpoint=Path(args.generation_11_checkpoint),
        generation_12_checkpoint=Path(args.generation_12_checkpoint),
        installed_checkpoint=Path(args.installed_checkpoint),
        initial_replays=tuple(Path(path) for path in args.initial_replays),
        device=args.device,
        torch_threads=args.torch_threads,
        shard_workers=args.shard_workers,
        arena_workers=args.arena_workers,
        max_turns=args.max_turns,
    )
    result = run_deep_search_teacher_refresh_127(config, launch=args.launch)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
