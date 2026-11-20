"""Run the frozen AlphaZero search-operator calibration 094."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.search_operator_calibration import (
    SearchOperatorCalibrationConfig,
    run_search_operator_calibration_094,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/alphazero_search_operator_calibration_094",
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
        "--result-093",
        default="outputs/alphazero_large_confirmation_093/result.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_search_operator_calibration_094(
        SearchOperatorCalibrationConfig(
            output_dir=Path(args.output_dir),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            replay_generation_2=Path(args.replay_generation_2),
            result_093=Path(args.result_093),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
