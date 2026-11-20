"""Run the frozen rolling-lineage endpoint evaluation 102."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.endpoint_evaluation import (
    EndpointEvaluationConfig,
    run_endpoint_evaluation_102,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_endpoint_evaluation_102")
    parser.add_argument("--result-101", default="outputs/alphazero_clean_continuation_101/result.json")
    parser.add_argument("--generation-2-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--generation-5-checkpoint", default="outputs/alphazero_clean_continuation_101/checkpoints/generation_005.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_endpoint_evaluation_102(
        EndpointEvaluationConfig(
            output_dir=Path(args.output_dir),
            result_101=Path(args.result_101),
            generation_2_checkpoint=Path(args.generation_2_checkpoint),
            generation_5_checkpoint=Path(args.generation_5_checkpoint),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

