"""Run the frozen AlphaZero adaptive replication 098."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.adaptive_replication import (
    AdaptiveReplicationConfig,
    run_adaptive_replication_098,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="outputs/alphazero_adaptive_replication_098"
    )
    parser.add_argument(
        "--result-097", default="outputs/alphazero_decisive_objective_097/result.json"
    )
    parser.add_argument(
        "--incumbent-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        default="outputs/alphazero_decisive_objective_097/candidates/visit_argmax_aux/generation_003.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_adaptive_replication_098(
        AdaptiveReplicationConfig(
            output_dir=Path(args.output_dir),
            result_097=Path(args.result_097),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            candidate_checkpoint=Path(args.candidate_checkpoint),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
