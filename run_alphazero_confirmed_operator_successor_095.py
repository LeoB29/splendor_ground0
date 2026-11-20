"""Run the frozen confirmed-operator AlphaZero successor experiment 095."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.confirmed_operator_successor import (
    ConfirmedOperatorSuccessorConfig,
    run_confirmed_operator_successor_095,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/alphazero_confirmed_operator_successor_095",
    )
    parser.add_argument(
        "--incumbent-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
    parser.add_argument(
        "--replay-generation-2",
        default="outputs/alphazero_pivot_090/pilot/generation_002_selfplay.jsonl",
    )
    parser.add_argument(
        "--result-094",
        default="outputs/alphazero_search_operator_calibration_094/result.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_confirmed_operator_successor_095(
        ConfirmedOperatorSuccessorConfig(
            output_dir=Path(args.output_dir),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            replay_generation_2=Path(args.replay_generation_2),
            result_094=Path(args.result_094),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
