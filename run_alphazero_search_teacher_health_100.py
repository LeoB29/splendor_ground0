"""Run the frozen AlphaZero search-teacher health audit 100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.search_teacher_health import (
    SearchTeacherHealthConfig,
    run_search_teacher_health_100,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_search_teacher_health_100")
    parser.add_argument("--incumbent-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--child-checkpoint", default="outputs/alphazero_scaled_loop_099/checkpoints/generation_003.pt")
    parser.add_argument("--replay-generation-2", default="outputs/alphazero_scaled_loop_099/selfplay/generation_002.jsonl")
    parser.add_argument("--replay-generation-3", default="outputs/alphazero_scaled_loop_099/selfplay/generation_003.jsonl")
    parser.add_argument("--result-099", default="outputs/alphazero_scaled_loop_099/result.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_search_teacher_health_100(
        SearchTeacherHealthConfig(
            output_dir=Path(args.output_dir),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            child_checkpoint=Path(args.child_checkpoint),
            replay_generation_2=Path(args.replay_generation_2),
            replay_generation_3=Path(args.replay_generation_3),
            result_099=Path(args.result_099),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
