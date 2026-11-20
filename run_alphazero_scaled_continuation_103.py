"""Run the frozen generation-5 to generation-8 continuation 103."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.scaled_continuation import (
    ScaledContinuationConfig,
    run_scaled_continuation_103,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_scaled_continuation_103")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--generation-5-checkpoint", default="outputs/alphazero_clean_continuation_101/checkpoints/generation_005.pt")
    parser.add_argument("--replay-generation-4", default="outputs/alphazero_clean_continuation_101/selfplay/generation_004_sanitized.jsonl")
    parser.add_argument("--result-101", default="outputs/alphazero_clean_continuation_101/result.json")
    parser.add_argument("--result-102", default="outputs/alphazero_endpoint_evaluation_102/result.json")
    parser.add_argument("--checkpoint-035", default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_scaled_continuation_103(
        ScaledContinuationConfig(
            output_dir=Path(args.output_dir),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_5_checkpoint=Path(args.generation_5_checkpoint),
            replay_generation_4=Path(args.replay_generation_4),
            result_101=Path(args.result_101),
            result_102=Path(args.result_102),
            checkpoint_035=Path(args.checkpoint_035),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

