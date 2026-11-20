from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.stratified_refill_search import (
    StratifiedRefillConfig,
    run_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or explicitly launch AlphaZero stratified refill pilot 137."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/alphazero_stratified_root_refill_search_137",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Run the protected 216-search panel; omission performs preparation only.",
    )
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    result = run_experiment(
        StratifiedRefillConfig(
            output_dir=Path(args.output_dir),
            checkpoint=Path(args.checkpoint),
            device=args.device,
            torch_threads=args.torch_threads,
        ),
        launch=args.launch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
