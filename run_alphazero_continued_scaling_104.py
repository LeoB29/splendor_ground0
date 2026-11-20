"""Run the frozen generation-8 to generation-11 continuation 104."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.continued_scaling import (
    ContinuedScalingConfig,
    run_continued_scaling_104,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_continued_scaling_104")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--generation-8-checkpoint", default="outputs/alphazero_scaled_continuation_103/checkpoints/generation_008.pt")
    parser.add_argument("--replay-generation-7", default="outputs/alphazero_scaled_continuation_103/selfplay/generation_007_sanitized.jsonl")
    parser.add_argument("--result-103", default="outputs/alphazero_scaled_continuation_103/result.json")
    parser.add_argument("--checkpoint-035", default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_continued_scaling_104(
        ContinuedScalingConfig(
            output_dir=Path(args.output_dir),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_8_checkpoint=Path(args.generation_8_checkpoint),
            replay_generation_7=Path(args.replay_generation_7),
            result_103=Path(args.result_103),
            checkpoint_035=Path(args.checkpoint_035),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

