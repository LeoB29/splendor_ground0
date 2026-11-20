"""Synthetic contract checks; never evaluate a saved production root."""
import copy
from fractions import Fraction
import hashlib
import json

import numpy as np
import pytest

from tools.saved_equal_budget_breadth import analysis as a


CONFIG = {"partitions": 4, "partition_seed": 2140000000,
          "bootstrap_seed": 2143000000, "bootstrap_resamples": 1000,
          "practical_gain": .005, "expected_cells": 18, "cell_size": 20,
          "fits": [161001, 161002, 161003], "bootstrap_chunk": 500}


def fixture_record(support=8, index=0, scores=None):
    n, m = min(support, 8), min(support, 4)
    extension = 32 * (n - m)
    keys = [f"action-{j}" for j in range(n)]
    scores = scores or (lambda world, action: ((world * 7 + action * 11) % 3) - 1)
    def block(count, width, seed):
        return {"candidate_indices": list(range(width)), "world_seeds": list(range(seed, seed + count)),
                "worlds": [{"seed": seed + i, "outcomes": [{"wdl": scores(i, j)} for j in range(width)]}
                           for i in range(count)]}
    fit, seat, phase = 161001 + (index % 18) // 6, ((index % 18) // 3) % 2, index % 3
    return {"spec": {"index": index, "pair": index + 100, "game_index": index + 100,
                     "fit": fit, "seat": seat, "phase": phase},
            "budgets": {"canonical_support": support, "anchor_candidates": m,
                        "breadth_candidates": n, "extension_worlds": extension},
            "union": {"keys": keys},
            "candidates": {arm: {"keys": keys[:count], "indices": list(range(count))}
                           for arm, count in (("anchor", m), ("breadth", n), ("precision", m))},
            "selection": block(128, n, 1000), "extension": block(extension, m, 2000)}


def external_order(length, index, partition, domain):
    ranks = []
    for item in range(length):
        payload = ["saved-equal-budget-breadth-v1", 2140000000, index, partition, domain, item]
        ranks.append((hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest(), item))
    return [item for _, item in sorted(ranks)]


@pytest.mark.parametrize("support", [1, 2, 3, 4])
def test_short_support_keeps_all_worlds_exact_zero_without_invented_assessment(support):
    result = a.analyze_root(fixture_record(support), CONFIG)
    assert result["mean_difference_fraction"] == [0, 1]
    assert result["choice_changes"] == result["breadth_addition_choices"] == 0
    for row in result["partitions"]:
        assert row["broad_counts"] == row["reference_counts"] == [128] * support
        assert row["broad_sums"] == row["reference_sums"]
        assert row["L"] == row["q"] == 128 and row["r"] == row["H"] == 0
        assert row["assessment_indices"] == row["extension_order"] == []
        assert row["assessment_sums"] == {"breadth": None, "reference": None}
        assert row["difference_fraction"] == [0, 1]
        assert row["selection_cell_counts"] == dict.fromkeys(("breadth", "reference"), 128 * support)


@pytest.mark.parametrize("support,q,r,L,H,E", [(5, 102, 2, 103, 25, 32), (6, 85, 2, 86, 42, 64),
                                             (7, 73, 1, 74, 54, 96), (8, 64, 0, 64, 64, 128),
                                             (71, 64, 0, 64, 64, 128)])
def test_all_wider_supports_exact_counts_cells_choices_and_heldout_scores(support, q, r, L, H, E):
    record = fixture_record(support)
    result = a.analyze_root(record, CONFIG)
    n = min(support, 8)
    for row in result["partitions"]:
        assert (row["q"], row["r"], row["L"], row["H"]) == (q, r, L, H)
        for domain, length in (("common", 128), ("extension", E), ("remainder", n)):
            assert row[f"{domain}_order"] == external_order(length, 0, row["partition"], domain)
        assert row["remainder_candidates"] == row["remainder_order"][:r]
        assert sorted(row["broad_counts"]) == [q] * (n - r) + [q + 1] * r
        assert row["reference_counts"] == [128] * 4
        assert sum(row["broad_counts"]) == sum(row["reference_counts"]) == 512
        assert row["assessment_indices"] == row["common_order"][L:]
        assert len(row["assessment_indices"]) == H <= E
        for j, count in enumerate(row["broad_counts"]):
            indices = row["common_order"][:count]
            assert not set(indices).intersection(row["assessment_indices"])
            total = sum(record["selection"]["worlds"][wi]["outcomes"][j]["wdl"] + 1 for wi in indices)
            assert total == row["broad_sums"][j]
        for j in range(4):
            total = sum(record["selection"]["worlds"][wi]["outcomes"][j]["wdl"] + 1
                        for wi in row["common_order"][:L])
            total += sum(record["extension"]["worlds"][wi]["outcomes"][j]["wdl"] + 1
                         for wi in row["extension_order"][:H])
            assert total == row["reference_sums"][j]
        for arm, name in (("breadth", "broad"), ("reference", "reference")):
            means = [Fraction(total, count) for total, count in zip(row[f"{name}_sums"], row[f"{name}_counts"])]
            assert row["choices"][arm] == means.index(max(means))
            expected = sum(record["selection"]["worlds"][wi]["outcomes"][row["choices"][arm]]["wdl"] + 1
                           for wi in row["assessment_indices"])
            assert row["assessment_sums"][arm] == expected
        exact = Fraction(row["assessment_sums"]["breadth"] - row["assessment_sums"]["reference"], 2 * H)
        assert row["difference_fraction"] == [exact.numerator, exact.denominator]
    mean = sum((Fraction(*row["difference_fraction"]) for row in result["partitions"]), Fraction(0)) / CONFIG["partitions"]
    assert result["mean_difference_fraction"] == [mean.numerator, mean.denominator]
    json.dumps(result, allow_nan=False)


def test_exact_unequal_count_comparison_preserves_first_tie():
    assert a._winner([102, 103], [102, 103]) == 0
    assert a._winner([102, 104], [102, 103]) == 1
    assert a._winner([103, 104], [102, 103]) == 0
    assert a._winner([0, 0, 0], [102, 103, 102]) == 0
    with pytest.raises(ValueError):
        a._winner([1, 2], [1, 0])


def test_assessment_never_changes_own_partition_choice_and_fraction_is_not_old_grid():
    config = CONFIG | {"partitions": 1}
    record = fixture_record(5, scores=lambda w, j: 1 if j == 4 else -1)
    original = a.analyze_root(record, config)
    row = original["partitions"][0]
    assert row["choices"] == {"breadth": 4, "reference": 0}
    changed = copy.deepcopy(record)
    for wi in row["assessment_indices"]:
        changed["selection"]["worlds"][wi]["outcomes"][4]["wdl"] = -1
    changed["selection"]["worlds"][row["assessment_indices"][0]]["outcomes"][4]["wdl"] = 0
    second = a.analyze_root(changed, config)["partitions"][0]
    assert second["choices"] == row["choices"]
    assert second["broad_sums"] == row["broad_sums"]
    assert second["reference_sums"] == row["reference_sums"]
    assert second["difference_fraction"] == [1, 50]
    assert second["difference"] * 256 != round(second["difference"] * 256)
    assert all(second[key] == row[key] for key in ("common_order", "extension_order", "remainder_order"))


@pytest.mark.parametrize("damage", ["wdl", "missing_world", "missing_action", "seed", "overlap", "budget", "prefix", "union", "failed"])
def test_corrupted_synthetic_records_rejected(damage):
    record = fixture_record()
    if damage == "wdl":
        record["selection"]["worlds"][0]["outcomes"][0]["wdl"] = .5
    elif damage == "missing_world":
        record["selection"]["worlds"].pop()
    elif damage == "missing_action":
        record["extension"]["worlds"][0]["outcomes"].pop()
    elif damage == "seed":
        record["selection"]["worlds"][0]["seed"] += 1
    elif damage == "overlap":
        record["extension"]["worlds"][0]["seed"] = record["selection"]["world_seeds"][0]
        record["extension"]["world_seeds"][0] = record["selection"]["world_seeds"][0]
    elif damage == "budget":
        record["budgets"]["extension_worlds"] -= 1
    elif damage == "prefix":
        record["candidates"]["anchor"]["indices"] = [1, 0, 2, 3]
    elif damage == "union":
        record["union"]["keys"][1] = record["union"]["keys"][0]
    else:
        record["failure"] = "fixture"
    with pytest.raises(ValueError):
        a.analyze_root(record, CONFIG)


def synthetic_results(config=CONFIG, constant=None):
    rows = []
    for index in range(360):
        fit, seat, phase = 161001 + (index % 18) // 6, ((index % 18) // 3) % 2, index % 3
        fractions = [constant if constant is not None else Fraction((index // 18) % 7 - 2 + p, 50)
                     for p in range(config["partitions"])]
        mean = sum(fractions, Fraction(0)) / config["partitions"]
        rows.append({"spec": {"index": index, "pair": index, "game_index": index,
                              "fit": fit, "seat": seat, "phase": phase},
                     "anchor_candidates": 4, "breadth_candidates": 5,
                     "partitions": [{"partition": p, "difference_fraction": [f.numerator, f.denominator],
                                     "difference": float(f), "choices": {"breadth": 4, "reference": 0}}
                                    for p, f in enumerate(fractions)],
                     "mean_difference_fraction": [mean.numerator, mean.denominator],
                     "mean_score_difference": float(mean), "choice_changes": config["partitions"],
                     "breadth_addition_choices": config["partitions"]})
    return rows


def test_bootstrap_averages_partitions_before_resampling_complete_roots():
    records = synthetic_results()
    result = a.summarize(records, CONFIG)
    cells = sorted({tuple(r["spec"][key] for key in ("fit", "seat", "phase")) for r in records})
    matrix = np.array([[r["mean_score_difference"] for r in records
                        if tuple(r["spec"][key] for key in ("fit", "seat", "phase")) == cell] for cell in cells])
    random = np.random.default_rng(CONFIG["bootstrap_seed"])
    samples = []
    for _ in range(2):
        draws = random.integers(0, 20, size=(500, 18, 20), dtype=np.int64)
        # Separate loop arithmetic checks the root/cell indexing and denominator.
        for indices in draws:
            samples.append(sum(float(matrix[cell, index]) for cell in range(18) for index in indices[cell]) / 360)
    expected = np.quantile(samples, [.025, .975], method="linear")
    assert result["ci95"] == pytest.approx(expected, abs=1e-15)
    mean = sum((Fraction(*r["mean_difference_fraction"]) for r in records), Fraction(0)) / 360
    assert result["mean_difference_fraction"] == [mean.numerator, mean.denominator]
    assert result["mean_score_difference"] == float(mean)
    assert result["root_partitions"] == 1440 and result["roots"] == 360
    assert result["descriptive"]["choice_change_fraction"] == 1.
    assert result["descriptive"]["support"]["5"]["roots"] == 360


@pytest.mark.parametrize("delta,passed", [(Fraction(1, 200), True), (Fraction(1, 201), False),
                                         (Fraction(0), False), (Fraction(-1, 200), False)])
def test_gate_uses_exact_practical_threshold_and_positive_lower_bound(delta, passed):
    result = a.summarize(synthetic_results(constant=delta), CONFIG)
    assert result["gate_pass"] is passed
    assert result["classification"] == ("positive_development_screen" if passed else "practical_progress_not_established")


@pytest.mark.parametrize("damage", ["order", "duplicate", "cell", "partitions", "rootmean", "float", "fraction", "movement"])
def test_root_aggregation_rejects_missing_or_inconsistent_units(damage):
    rows = synthetic_results()
    if damage == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "duplicate":
        rows[1]["spec"]["pair"] = rows[1]["spec"]["game_index"] = rows[0]["spec"]["game_index"]
    elif damage == "cell":
        rows[0]["spec"]["seat"] = 1
    elif damage == "partitions":
        rows[0]["partitions"].pop()
    elif damage == "rootmean":
        rows[0]["mean_score_difference"] += .001
    elif damage == "float":
        rows[0]["partitions"][0]["difference"] += .001
    elif damage == "fraction":
        rows[0]["partitions"][0]["difference_fraction"] = [-2, 50]
    else:
        rows[0]["choice_changes"] -= 1
    with pytest.raises(ValueError):
        a.summarize(rows, CONFIG)
