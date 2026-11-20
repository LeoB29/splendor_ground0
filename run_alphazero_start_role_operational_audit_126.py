from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

from splendor_ai.alphazero.start_role_operational_audit import (
    StartRoleOperationalAuditConfig,
    run_start_role_operational_audit_126,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen no-training start-role operational audit 126."
    )
    parser.add_argument(
        "--output-dir", default="outputs/alphazero_start_role_operational_audit_126"
    )
    parser.add_argument(
        "--result-117", default="outputs/alphazero_start_role_audit_117/result.json"
    )
    parser.add_argument(
        "--result-125",
        default="outputs/alphazero_paired_objective_role_pilot_125/result.json",
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_start_role_operational_audit_126(
        StartRoleOperationalAuditConfig(
            output_dir=Path(args.output_dir),
            result_117=Path(args.result_117),
            result_125=Path(args.result_125),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            untouched_holdout=Path(args.untouched_holdout),
            installed_checkpoint=Path(args.installed_checkpoint),
            device=args.device,
            torch_threads=args.torch_threads,
            workers=args.workers,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
