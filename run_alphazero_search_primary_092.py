"""Run the frozen search-primary AlphaZero engineering protocol 092."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from splendor_ai.alphazero.search_primary import (
    SearchPrimaryConfig,
    run_search_primary_092,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="outputs/alphazero_search_primary_092"
    )
    parser.add_argument(
        "--result-091", default="outputs/alphazero_corrective_091/result.json"
    )
    parser.add_argument(
        "--incumbent-checkpoint",
        default="outputs/alphazero_pivot_090/pilot/generation_002.pt",
    )
    parser.add_argument(
        "--balanced-half-checkpoint",
        default="outputs/alphazero_corrective_091/candidates/balanced_half/generation_003.pt",
    )
    parser.add_argument(
        "--value-quarter-checkpoint",
        default="outputs/alphazero_corrective_091/candidates/value_quarter/generation_003.pt",
    )
    parser.add_argument(
        "--replay-generation-2",
        default="outputs/alphazero_pivot_090/pilot/generation_002_selfplay.jsonl",
    )
    parser.add_argument(
        "--checkpoint-035",
        default="outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    checkpoint_035 = Path(args.checkpoint_035) if args.checkpoint_035 else None
    if checkpoint_035 is not None and not checkpoint_035.exists():
        checkpoint_035 = None
    result = run_search_primary_092(
        SearchPrimaryConfig(
            output_dir=Path(args.output_dir),
            result_091=Path(args.result_091),
            incumbent_checkpoint=Path(args.incumbent_checkpoint),
            balanced_half_checkpoint=Path(args.balanced_half_checkpoint),
            value_quarter_checkpoint=Path(args.value_quarter_checkpoint),
            replay_generation_2=Path(args.replay_generation_2),
            checkpoint_035=checkpoint_035,
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
