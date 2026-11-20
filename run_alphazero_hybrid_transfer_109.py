"""Run the frozen hybrid transfer experiment 109."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.hybrid_transfer import (
    HybridTransferConfig,
    run_hybrid_transfer_109,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_hybrid_transfer_109")
    parser.add_argument("--result-108", default="outputs/alphazero_consolidation_repair_108/result.json")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--generation-11-checkpoint", default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")
    parser.add_argument("--consolidation-checkpoint", default="outputs/alphazero_replay_consolidation_107/checkpoints/consolidated_generation_012.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_hybrid_transfer_109(
        HybridTransferConfig(
            output_dir=Path(args.output_dir),
            result_108=Path(args.result_108),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            consolidation_checkpoint=Path(args.consolidation_checkpoint),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
