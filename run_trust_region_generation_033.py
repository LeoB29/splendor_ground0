"""Workspace entry point for trust-region league generation 033."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from splendor_ai.training.run_league_generation import LeagueGenerationConfig
    from splendor_ai.training.run_trust_region_generation import run_trust_region_generation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_trust_region_generation(
        LeagueGenerationConfig(
            output_dir=Path("outputs/trust_region_generation_033"),
            data_dir=Path("data/trust_region_generation_033"),
            parent_checkpoint=Path("outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt"),
            checkpoint_016=Path("outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt"),
            checkpoint_023=Path("outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt"),
            calibration_replay=Path("data/corpus_generation1_policy035_selfplay_038/replays.jsonl"),
            broad_replay=Path("data/corpus_generation1_policy035_selfplay_038/replays.jsonl"),
            collection_seed_start=355_000,
            sampling_seed_start=33_000_000,
            direct_seed_start=356_000,
            broad_016_seed_start=357_000,
            broad_shallow_seed_start=358_000,
            broad_023_seed_start=359_000,
            workers=args.workers,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
