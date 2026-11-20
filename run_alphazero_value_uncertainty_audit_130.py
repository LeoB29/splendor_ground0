from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path

import torch

from splendor_ai.alphazero.value_uncertainty_audit import (
    ValueUncertaintyAuditConfig,
    run_value_uncertainty_audit_130,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight or launch AlphaZero value-uncertainty audit 130.")
    parser.add_argument("--output-dir", default="outputs/alphazero_value_uncertainty_audit_130")
    parser.add_argument("--result-129", default="outputs/alphazero_historical_state_reanalysis_129/result.json")
    parser.add_argument("--generation-11-checkpoint", default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")
    parser.add_argument("--terminal-control-checkpoint", default="outputs/alphazero_high_budget_distillation_114/checkpoints/high_budget_generation_012.pt")
    parser.add_argument("--ordinary-holdout", default="outputs/alphazero_high_budget_distillation_114/partition/holdout_32.jsonl")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
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
    result = run_value_uncertainty_audit_130(
        ValueUncertaintyAuditConfig(
            output_dir=Path(args.output_dir),
            result_129=Path(args.result_129),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            terminal_control_checkpoint=Path(args.terminal_control_checkpoint),
            ordinary_holdout=Path(args.ordinary_holdout),
            installed_checkpoint=Path(args.installed_checkpoint),
            device=args.device,
            workers=args.workers,
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
