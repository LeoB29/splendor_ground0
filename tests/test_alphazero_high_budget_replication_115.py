from __future__ import annotations

import pytest

from splendor_ai.alphazero.high_budget_replication import (
    FRESH_SEED_START,
    PAIRED_SEEDS,
    PRIOR_SEED_START,
    SHARD_COUNT,
    SHARD_PAIRED_SEEDS,
    _shard_identity,
    _summarize_rows,
    combine_fresh_shards,
    confirmation_assessment,
    high_budget_replication_decision,
    pool_prior_and_fresh,
)


def _rows(seed_start: int, paired_seeds: int) -> list[dict[str, object]]:
    return [
        {
            "seed": seed_start + offset,
            "candidate_seat": seat,
            "candidate_points": 1.0 if seat == 0 else 0.0,
            "candidate_margin": 1 if seat == 0 else -1,
            "termination_reason": "completed",
        }
        for offset in range(paired_seeds)
        for seat in (0, 1)
    ]


def _shards() -> dict[int, dict[str, object]]:
    shards: dict[int, dict[str, object]] = {}
    for index in range(SHARD_COUNT):
        start = FRESH_SEED_START + index * SHARD_PAIRED_SEEDS
        shards[index] = {
            "paired_seeds": SHARD_PAIRED_SEEDS,
            "games": 2 * SHARD_PAIRED_SEEDS,
            "rows": _rows(start, SHARD_PAIRED_SEEDS),
        }
    return shards


def test_identity_binds_exact_high_budget_matchup_and_schedule() -> None:
    identities = [_shard_identity(index, 200) for index in range(SHARD_COUNT)]
    assert [identity["seed_start"] for identity in identities] == [
        2_150_000,
        2_150_025,
        2_150_050,
        2_150_075,
    ]
    assert all(
        identity["candidate_search_config"] == identity["baseline_search_config"]
        for identity in identities
    )
    assert all(identity["candidate_search_config"]["simulations"] == 1_024 for identity in identities)


def test_fresh_combiner_requires_all_exact_shards() -> None:
    arena = combine_fresh_shards(_shards())
    assert arena["games"] == 200
    assert arena["paired_seeds"] == PAIRED_SEEDS
    assert arena["score_share"] == 0.5
    missing = _shards()
    del missing[3]
    with pytest.raises(ValueError, match="four exact"):
        combine_fresh_shards(missing)  # type: ignore[arg-type]


def test_pool_preserves_two_disjoint_exact_seed_blocks() -> None:
    prior = _summarize_rows(
        "prior", _rows(PRIOR_SEED_START, PAIRED_SEEDS), PAIRED_SEEDS
    )
    fresh = combine_fresh_shards(_shards())
    pooled = pool_prior_and_fresh(prior, fresh)
    assert pooled["games"] == 400
    assert pooled["paired_seeds"] == 200
    assert len({(row["seed"], row["candidate_seat"]) for row in pooled["rows"]}) == 400


def test_pool_rejects_mutated_prior_schedule() -> None:
    prior = _summarize_rows(
        "prior", _rows(PRIOR_SEED_START, PAIRED_SEEDS), PAIRED_SEEDS
    )
    prior["rows"][0]["seed"] = -1
    with pytest.raises(ValueError, match="114 anchor"):
        pool_prior_and_fresh(prior, combine_fresh_shards(_shards()))


def test_confirmation_requires_strict_confidence_interval() -> None:
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
    assert not confirmation_assessment(arena, {"ci95_low": 0.5})["passed"]
    assert confirmation_assessment(arena, {"ci95_low": 0.501})["passed"]


def test_decision_never_installs_or_continues_automatically() -> None:
    passed = high_budget_replication_decision({"passed": True})
    failed = high_budget_replication_decision({"passed": False})
    assert "review_short_1024_continuation" in passed
    assert "proceed_medium_capacity" in failed
    assert passed.endswith("no_install") and failed.endswith("no_install")
