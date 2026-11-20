"""Run the frozen consolidation repair experiment 108."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.consolidation_repair import (
    ConsolidationRepairConfig,
    run_consolidation_repair_108,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/alphazero_consolidation_repair_108")
    parser.add_argument("--result-107", default="outputs/alphazero_replay_consolidation_107/result.json")
    parser.add_argument("--installed-checkpoint", default="outputs/alphazero_pivot_090/pilot/generation_002.pt")
    parser.add_argument("--generation-11-checkpoint", default="outputs/alphazero_continued_scaling_104/checkpoints/generation_011.pt")
    parser.add_argument("--consolidation-checkpoint", default="outputs/alphazero_replay_consolidation_107/checkpoints/consolidated_generation_012.pt")
    parser.add_argument("--replay-7", default="outputs/alphazero_scaled_continuation_103/selfplay/generation_007_sanitized.jsonl")
    parser.add_argument("--replay-8", default="outputs/alphazero_continued_scaling_104/selfplay/generation_008_sanitized.jsonl")
    parser.add_argument("--replay-9", default="outputs/alphazero_continued_scaling_104/selfplay/generation_009_sanitized.jsonl")
    parser.add_argument("--replay-10", default="outputs/alphazero_continued_scaling_104/selfplay/generation_010_sanitized.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    result = run_consolidation_repair_108(
        ConsolidationRepairConfig(
            output_dir=Path(args.output_dir),
            result_107=Path(args.result_107),
            installed_checkpoint=Path(args.installed_checkpoint),
            generation_11_checkpoint=Path(args.generation_11_checkpoint),
            consolidation_checkpoint=Path(args.consolidation_checkpoint),
            replay_7=Path(args.replay_7),
            replay_8=Path(args.replay_8),
            replay_9=Path(args.replay_9),
            replay_10=Path(args.replay_10),
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
