from __future__ import annotations

import pytest

from splendor_ai.alphazero.direct_value_path import (
    PAIRED_SEEDS,
    SEED_START,
    SHARD_COUNT,
    SHARD_PAIRED_SEEDS,
    _shard_identity,
    combine_shard_arenas,
    direct_value_path_decision,
)


def _shards() -> dict[int, dict[str, object]]:
    shards: dict[int, dict[str, object]] = {}
    for index in range(SHARD_COUNT):
        start = SEED_START + index * SHARD_PAIRED_SEEDS
        rows = []
        for offset in range(SHARD_PAIRED_SEEDS):
            for seat in (0, 1):
                rows.append(
                    {
                        "seed": start + offset,
                        "candidate_seat": seat,
                        "candidate_points": 1.0 if seat == 0 else 0.0,
                        "candidate_margin": 1 if seat == 0 else -1,
                        "termination_reason": "completed",
                    }
                )
        shards[index] = {
            "paired_seeds": SHARD_PAIRED_SEEDS,
            "games": 2 * SHARD_PAIRED_SEEDS,
            "rows": rows,
        }
    return shards


def test_shard_identities_cover_fresh_contiguous_schedule() -> None:
    identities = [_shard_identity(index, 200) for index in range(SHARD_COUNT)]
    assert [item["seed_start"] for item in identities] == [
        2_070_000,
        2_070_050,
        2_070_100,
        2_070_150,
    ]
    assert all(item["paired_seeds"] == 50 for item in identities)
    assert all(item["search_config"]["simulations"] == 256 for item in identities)
    assert all(item["search_config"]["root_noise_fraction"] == 0.0 for item in identities)


def test_shard_identity_holds_policy_path_fixed_and_changes_value_path() -> None:
    identity = _shard_identity(0, 200)
    assert identity["candidate_policy_checkpoint_sha256"] == identity[
        "baseline_policy_checkpoint_sha256"
    ]
    assert identity["candidate_value_checkpoint_sha256"] != identity[
        "baseline_value_checkpoint_sha256"
    ]


def test_combine_shards_validates_and_summarizes_exact_schedule() -> None:
    arena = combine_shard_arenas(_shards())
    assert arena["paired_seeds"] == PAIRED_SEEDS
    assert arena["games"] == 400
    assert arena["candidate_wins"] == 200
    assert arena["candidate_losses"] == 200
    assert arena["draws"] == 0
    assert arena["score_share"] == 0.5
    assert arena["mean_margin"] == 0.0
    assert arena["termination_reasons"] == {"completed": 400}


def test_combine_shards_rejects_nonprefix_or_missing_shard() -> None:
    missing = _shards()
    del missing[3]
    with pytest.raises(ValueError, match="four exact"):
        combine_shard_arenas(missing)  # type: ignore[arg-type]
    reordered = _shards()
    reordered[0]["rows"][0], reordered[0]["rows"][1] = (  # type: ignore[index]
        reordered[0]["rows"][1],  # type: ignore[index]
        reordered[0]["rows"][0],  # type: ignore[index]
    )
    with pytest.raises(ValueError, match="exact completed schedule"):
        combine_shard_arenas(reordered)  # type: ignore[arg-type]


def test_direct_value_path_decision_never_authorizes_install() -> None:
    passed = direct_value_path_decision({"passed": True})
    failed = direct_value_path_decision({"passed": False})
    assert "causal_gameplay_pass" in passed
    assert "not_confirmed" in failed
    assert passed.endswith("no_install")
    assert failed.endswith("no_install")
