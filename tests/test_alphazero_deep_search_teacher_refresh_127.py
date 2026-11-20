from __future__ import annotations

import hashlib
from pathlib import Path

from run_alphazero_deep_search_teacher_refresh_127 import build_parser
from splendor_ai.alphazero.deep_search_teacher_refresh import (
    ENDPOINT_PAIRED_SEEDS,
    EXPECTED_INITIAL_REPLAYS,
    PROTOCOL_SHA256,
    SELFPLAY_SEED_STARTS,
    endpoint_assessment,
    refresh_training_config,
)


def test_127_protocol_hash_and_explicit_launch_guard() -> None:
    protocol = Path("docs/engineering/alphazero_deep_search_teacher_refresh_127.md")
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    assert len(EXPECTED_INITIAL_REPLAYS) == 4
    assert build_parser().parse_args([]).launch is False
    assert build_parser().parse_args(["--launch"]).launch is True


def test_127_has_two_exact_teacher_refresh_schedules() -> None:
    assert SELFPLAY_SEED_STARTS == {
        12: (2_300_000, 2_310_000, 2_320_000, 2_330_000),
        13: (2_400_000, 2_410_000, 2_420_000, 2_430_000),
    }
    assert ENDPOINT_PAIRED_SEEDS == 200
    assert set(range(2_300_000, 2_300_064)).isdisjoint(
        range(2_400_000, 2_400_064)
    )


def test_127_training_recipe_is_parent_safe_and_generation_seeded() -> None:
    generation_13 = refresh_training_config(13)
    generation_14 = refresh_training_config(14)
    assert generation_13.epochs == generation_14.epochs == 12
    assert generation_13.batch_size == generation_14.batch_size == 256
    assert generation_13.learning_rate == generation_14.learning_rate == 1e-4
    assert generation_13.value_loss_weight == generation_14.value_loss_weight == 0.5
    assert generation_13.early_stopping_patience == 3
    assert (generation_13.seed, generation_14.seed) == (1_127_013, 1_127_014)


def test_127_endpoint_requires_strict_paired_confidence() -> None:
    arena = {
        "games": 400,
        "score_share": 0.56,
        "mean_margin": 0.5,
        "termination_reasons": {"completed": 400},
        "rows": [
            {"candidate_seat": seat, "candidate_points": 0.56}
            for seat in (0, 1)
            for _ in range(200)
        ],
    }
    assert endpoint_assessment(arena, {"ci95_low": 0.5})["passed"] is False
    assert endpoint_assessment(arena, {"ci95_low": 0.501})["passed"] is True
