"""Workspace entry point for league self-play generation 032."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from splendor_ai.training.run_league_generation import (
        LeagueGenerationConfig,
        run_league_generation,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/league_generation_032")
    parser.add_argument("--data-dir", default="data/league_generation_032")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_league_generation(
        LeagueGenerationConfig(
            output_dir=Path(args.output_dir),
            data_dir=Path(args.data_dir),
            parent_checkpoint=Path("outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt"),
            checkpoint_016=Path("outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt"),
            checkpoint_023=Path("outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt"),
            calibration_replay=Path("data/corpus_generation1_policy035_selfplay_038/replays.jsonl"),
            broad_replay=Path("data/corpus_generation1_policy035_selfplay_038/replays.jsonl"),
            workers=args.workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
