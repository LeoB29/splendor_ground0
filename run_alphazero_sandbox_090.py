"""Run the programme-090 AlphaZero engineering smoke or pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.generation import (
    SandboxConfig,
    run_three_generation_pilot,
    run_vertical_slice,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "pilot"))
    parser.add_argument(
        "--output-dir",
        default="outputs/alphazero_pivot_090",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initialization-seed", type=int, default=900090)
    parser.add_argument("--seed-start", type=int, default=900000)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--arena-pairs", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--checkpoint-035",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument(
        "--resource-limited",
        action="store_true",
        help="Run an explicitly labelled reduced pilot instead of target dose.",
    )
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    checkpoint_035 = Path(args.checkpoint_035) if args.checkpoint_035 else None
    if checkpoint_035 is not None and not checkpoint_035.exists():
        checkpoint_035 = None
    config = SandboxConfig(
        output_dir=Path(args.output_dir),
        device=args.device,
        initialization_seed=args.initialization_seed,
        seed_start=args.seed_start,
        max_turns=args.max_turns,
        arena_pairs=args.arena_pairs,
        checkpoint_035=checkpoint_035,
        torch_threads=args.torch_threads,
    )
    result = (
        run_vertical_slice(config)
        if args.mode == "smoke"
        else run_three_generation_pilot(
            config,
            target_dose=not args.resource_limited,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
