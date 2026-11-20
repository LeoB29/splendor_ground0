"""Run the frozen clean rolling-loop continuation 101."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.clean_continuation import (
    CleanContinuationConfig,
    run_clean_continuation_101,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_clean_continuation_101")
    parser.add_argument("--incumbent-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--generation-3-checkpoint", default="outputs/alphazero_scaled_loop_099/checkpoints/generation_003.pt")
    parser.add_argument("--replay-generation-2", default="outputs/alphazero_scaled_loop_099/selfplay/generation_002.jsonl")
    parser.add_argument("--contaminated-replay-generation-3", default="outputs/alphazero_scaled_loop_099/selfplay/generation_003.jsonl")
    parser.add_argument("--result-099", default="outputs/alphazero_scaled_loop_099/result.json")
    parser.add_argument("--result-100", default="outputs/alphazero_search_teacher_health_100/result.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_clean_continuation_101(
        CleanContinuationConfig(
            output_dir=Path(args.output_dir),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            generation_3_checkpoint=Path(args.generation_3_checkpoint),
            replay_generation_2=Path(args.replay_generation_2),
            contaminated_replay_generation_3=Path(args.contaminated_replay_generation_3),
            result_099=Path(args.result_099),
            result_100=Path(args.result_100),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
