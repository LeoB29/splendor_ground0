"""Run the frozen AlphaZero decisive-objective experiment 097."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from splendor_ai.alphazero.decisive_objective import DecisiveObjectiveConfig, run_decisive_objective_097


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_decisive_objective_097")
    parser.add_argument("--parent-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--replay-generation-2", default="outputs/alphazero_pivot_090/pilot/generation_002_selfplay.jsonl")
    parser.add_argument("--replay-095", default="outputs/alphazero_confirmed_operator_successor_095/selfplay/generation_002_high_dose.jsonl")
    parser.add_argument("--result-095", default="outputs/alphazero_confirmed_operator_successor_095/result.json")
    parser.add_argument("--result-096", default="outputs/alphazero_head_attribution_096/result.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_decisive_objective_097(
        DecisiveObjectiveConfig(
            output_dir=Path(args.output_dir), parent_checkpoint=Path(args.parent_checkpoint),
            replay_generation_2=Path(args.replay_generation_2), replay_095=Path(args.replay_095),
            result_095=Path(args.result_095), result_096=Path(args.result_096),
            device=args.device, max_turns=args.max_turns, torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
