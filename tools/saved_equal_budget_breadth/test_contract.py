"""Cross-implementation synthetic contracts; never read production outcomes."""
from copy import deepcopy

import pytest

from tools.saved_equal_budget_breadth import analysis, common, verify


def synthetic_root(support, index=0):
    m, n = min(support, 4), min(support, 8)
    extension = 32 * (n - m)
    keys = [f"action-{j}" for j in range(n)]

    def block(width, count, offset):
        seeds = list(range(offset, offset + count))
        return {
            "candidate_indices": list(range(width)), "world_seeds": seeds,
            "worlds": [{
                "seed": seed,
                "outcomes": [{"wdl": ((wi * (j + 3) + j * 5 + offset) % 7) % 3 - 1}
                             for j in range(width)],
            } for wi, seed in enumerate(seeds)],
        }

    return {
        "spec": {"index": index, "pair": index + 1000,
                 "game_index": index + 1000, "fit": 161001, "seat": 0, "phase": 1},
        "budgets": {"canonical_support": support, "anchor_candidates": m,
                    "breadth_candidates": n, "extension_worlds": extension},
        "union": {"keys": keys},
        "candidates": {arm: {"indices": list(range(size)), "keys": keys[:size]}
                       for arm, size in (("anchor", m), ("breadth", n), ("precision", m))},
        "selection": block(n, 128, 10000),
        "extension": block(m, extension, 20000),
    }


@pytest.mark.parametrize("support", [1, 2, 3, 4, 5, 6, 7, 8, 9, 31])
def test_production_partition_count_independent_agreement(support):
    record = synthetic_root(support, index=support)
    result = analysis.analyze_root(record, common.CONFIG)
    receipt = verify.verify_root(record, result, common.CONFIG)
    assert receipt["passed"] and receipt["partitions"] == 64
    for partition in result["partitions"]:
        assert sum(partition["broad_counts"]) == sum(partition["reference_counts"]) == 128 * min(support, 4)
        assert set(partition["assessment_indices"]).isdisjoint(partition["common_order"][:partition["L"]])
    if support <= 4:
        assert result["mean_difference_fraction"] == [0, 1]
        assert all(p["assessment_sums"] == {"breadth": None, "reference": None}
                   for p in result["partitions"])


def test_correlated_candidates_keep_the_same_world_vector():
    record = synthetic_root(8)
    for wi, world in enumerate(record["selection"]["worlds"]):
        shared = 1 if wi % 3 else -1
        for j, outcome in enumerate(world["outcomes"]):
            outcome["wdl"] = shared if j in (0, 4) else -1
    result = analysis.analyze_root(record, common.CONFIG)
    assert verify.verify_root(record, result, common.CONFIG)["passed"]
    for partition in result["partitions"]:
        assert partition["broad_sums"][0] == partition["broad_sums"][4]
        assert partition["choices"]["breadth"] == 0


def test_partition_orders_are_independent_of_every_outcome():
    record = synthetic_root(5, index=12)
    before = analysis.analyze_root(record, common.CONFIG)
    altered = deepcopy(record)
    for block in ("selection", "extension"):
        for world in altered[block]["worlds"]:
            for outcome in world["outcomes"]:
                outcome["wdl"] = -outcome["wdl"]
    after = analysis.analyze_root(altered, common.CONFIG)
    assert verify.verify_root(altered, after, common.CONFIG)["passed"]
    for first, second in zip(before["partitions"], after["partitions"], strict=True):
        for field in ("common_order", "extension_order", "remainder_order", "remainder_candidates", "assessment_indices"):
            assert first[field] == second[field]


def test_verifier_does_not_call_collection_decision_helpers(monkeypatch):
    record = synthetic_root(7, index=23)
    result = analysis.analyze_root(record, common.CONFIG)

    def forbidden(*args, **kwargs):
        raise AssertionError("Verifier called the analysis implementation")

    monkeypatch.setattr(analysis, "analyze_root", forbidden)
    monkeypatch.setattr(analysis, "_order", forbidden)
    monkeypatch.setattr(analysis, "_winner", forbidden)
    monkeypatch.setattr(analysis, "_matrix", forbidden)
    assert verify.verify_root(record, result, common.CONFIG)["passed"]
