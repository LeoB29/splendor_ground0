"""Run the frozen scaled rolling AlphaZero sandbox 099."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.scaled_loop import ScaledLoopConfig, run_scaled_loop_099


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_scaled_loop_099")
    parser.add_argument(
        "--incumbent-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
    parser.add_argument(
        "--bootstrap-replay",
        default="outputs/alphazero_confirmed_operator_successor_095/selfplay/generation_002_high_dose.jsonl",
    )
    parser.add_argument(
        "--result-098",
        default="outputs/alphazero_adaptive_replication_098/result.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_scaled_loop_099(
        ScaledLoopConfig(
            output_dir=Path(args.output_dir),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            bootstrap_replay=Path(args.bootstrap_replay),
            result_098=Path(args.result_098),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
