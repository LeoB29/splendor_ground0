from __future__ import annotations

import pytest

from splendor_ai.alphazero.search_headroom import (
    PAIRED_SEEDS,
    SEED_START,
    SHARD_COUNT,
    SHARD_PAIRED_SEEDS,
    _shard_identity,
    combine_shard_arenas,
    search_headroom_assessment,
    search_headroom_decision,
)


def _shards() -> dict[int, dict[str, object]]:
    shards: dict[int, dict[str, object]] = {}
    for index in range(SHARD_COUNT):
        start = SEED_START + index * SHARD_PAIRED_SEEDS
        rows = [
            {
                "seed": start + offset,
                "candidate_seat": seat,
                "candidate_points": 1.0 if seat == 0 else 0.0,
                "candidate_margin": 1 if seat == 0 else -1,
                "termination_reason": "completed",
            }
            for offset in range(SHARD_PAIRED_SEEDS)
            for seat in (0, 1)
        ]
        shards[index] = {
            "paired_seeds": SHARD_PAIRED_SEEDS,
            "games": 2 * SHARD_PAIRED_SEEDS,
            "rows": rows,
        }
    return shards


def test_identity_is_only_a_search_budget_intervention() -> None:
    identity = _shard_identity(0, 200)
    assert identity["candidate_checkpoint_sha256"] == identity["baseline_checkpoint_sha256"]
    candidate = identity["candidate_search_config"]
    baseline = identity["baseline_search_config"]
    assert candidate["simulations"] == 1_024
    assert baseline["simulations"] == 256
    for key in candidate:
        if key != "simulations":
            assert candidate[key] == baseline[key]


def test_identities_cover_fresh_contiguous_schedule() -> None:
    identities = [_shard_identity(index, 200) for index in range(SHARD_COUNT)]
    assert [item["seed_start"] for item in identities] == [
        2_080_000,
        2_080_050,
        2_080_100,
        2_080_150,
    ]


def test_combiner_requires_all_exact_schedules() -> None:
    arena = combine_shard_arenas(_shards())
    assert arena["paired_seeds"] == PAIRED_SEEDS
    assert arena["games"] == 400
    assert arena["score_share"] == 0.5
    missing = _shards()
    del missing[3]
    with pytest.raises(ValueError, match="four exact"):
        combine_shard_arenas(missing)  # type: ignore[arg-type]


def test_headroom_gate_adds_strict_bootstrap_requirement() -> None:
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
    failed = search_headroom_assessment(arena, {"ci95_low": 0.5})
    passed = search_headroom_assessment(arena, {"ci95_low": 0.501})
    assert failed["passed"] is False
    assert passed["passed"] is True


def test_decision_never_installs() -> None:
    passed = search_headroom_decision({"passed": True})
    failed = search_headroom_decision({"passed": False})
    assert "headroom_confirmed" in passed
    assert "proceed_capacity" in failed
    assert passed.endswith("no_install") and failed.endswith("no_install")
