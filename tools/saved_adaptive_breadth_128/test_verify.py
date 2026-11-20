"""Synthetic independent reconstruction, leakage and arithmetic tamper checks."""
import copy
from fractions import Fraction

import pytest

from . import analysis as a
from . import verify as v


CONFIG = {"partitions": 64, "partition_seed": 2140000000, "bootstrap_seed": 2163000000,
          "bootstrap_resamples": 50000, "bootstrap_chunk": 500, "incremental_gain": .0025,
          "anchor_gain": .005, "expected_cells": 18, "cell_size": 20,
          "fits": [161001, 161002, 161003], "roots": 360}


def record(support=8, index=0, cell=(161001, 0, 0), flat=False):
    n, m = min(support, 8), min(support, 4)
    extension = 32*(n-m)
    def block(count, width, offset):
        seeds = list(range(offset, offset+count))
        return {"candidate_indices": list(range(width)), "world_seeds": seeds,
                "worlds": [{"seed": seed, "outcomes": [{"wdl": 0 if flat else ((seed*11+j*7+j*seed)%3)-1}
                                                       for j in range(width)]} for seed in seeds]}
    return {"spec": {"index": index, "pair": 1000+index, "game_index": 1000+index,
                     "fit": cell[0], "seat": cell[1], "phase": cell[2]},
            "budgets": {"canonical_support": support, "anchor_candidates": m,
                        "breadth_candidates": n, "extension_worlds": extension},
            "union": {"keys": [f"action_{j}" for j in range(n)]},
            "candidates": {arm: {"indices": list(range(width)), "keys": [f"action_{j}" for j in range(width)]}
                           for arm, width in (("anchor", m), ("breadth", n), ("precision", m))},
            "selection": block(128, n, 100000), "extension": block(extension, m, 200000)}


def hand_source(combined=False):
    source = record(8, flat=True)
    order = v._order(128, 0, 0, "common", CONFIG["partition_seed"])
    for block in ("selection", "extension"):
        for world in source[block]["worlds"]:
            for outcome in world["outcomes"]:
                outcome["wdl"] = -1
    for candidate, (initial, later) in enumerate(((8, 56), (16, 48), (32, 32), (32, 16 if combined else 0))):
        for wi in order[:initial]:
            source["selection"]["worlds"][wi]["outcomes"][candidate]["wdl"] = 1 if candidate == 3 else 0
        for wi in order[32:32+later]:
            source["selection"]["worlds"][wi]["outcomes"][candidate]["wdl"] = 0
    # Candidate4 would win the second-stage observations but must stay eliminated.
    for wi in order[32:96]:
        source["selection"]["worlds"][wi]["outcomes"][4]["wdl"] = 1
    for wi in order[96:]:
        source["selection"]["worlds"][wi]["outcomes"][0]["wdl"] = 1
    return source


@pytest.mark.parametrize("combined", [False, True])
def test_final_uses_initial_observations_original_ties_and_only_survivors(combined):
    config = {**CONFIG, "partitions": 1}
    source = hand_source(combined)
    saved = a.analyze_root(source, config)
    part = saved["partitions"][0]
    assert part["adaptive"]["ranked_survivors"] == [3, 2, 1, 0]
    assert part["adaptive"]["final_order"] == [0, 1, 2, 3]
    assert part["adaptive"]["final_sums"] == ([64, 64, 64, 80] if combined else [64]*4)
    assert part["choices"]["adaptive"] == (3 if combined else 0)
    assert part["adaptive"]["final_counts"] == [96]*4
    assert part["selection_cell_counts"] == dict.fromkeys(v.ARMS, 512)
    assert part["H"] == 64 and part["assessment_count"] == 32
    assert [len(x) for x in part["selection_common_indices"]["adaptive"]] == [96]*4+[32]*4
    assert v.verify_root(source, saved, config)["passed"]


def test_stage_one_ties_keep_first_four_and_support_above_eight_is_active():
    source = record(71, flat=True)
    saved = a.analyze_root(source, CONFIG)
    assert saved["adaptive_partitions"] == 64
    assert all(p["adaptive"]["ranked_survivors"] == [0, 1, 2, 3]
               and p["choices"] == dict.fromkeys(v.ARMS, 0) for p in saved["partitions"])
    assert v.verify_root(source, saved, CONFIG)["passed"]


@pytest.mark.parametrize("support", [1, 4, 5, 6, 7])
def test_short_support_is_exact_uniform_fallback(support):
    source = record(support)
    saved = a.analyze_root(source, CONFIG)
    for part in saved["partitions"]:
        assert not part["adaptive"]["active"]
        assert part["choices"]["adaptive"] == part["choices"]["uniform"]
        assert part["selection_common_indices"]["adaptive"] == part["selection_common_indices"]["uniform"]
        assert part["difference_fractions"]["adaptive_minus_uniform"] == [0, 1]
    assert v.verify_root(source, saved, CONFIG)["passed"]


def test_hand_checked_one_fiftieth_difference_is_not_quantized():
    source = record(5, flat=True)
    order = v._order(128, 0, 0, "common", CONFIG["partition_seed"])
    for block in ("selection", "extension"):
        for world in source[block]["worlds"]:
            for candidate in range(1, 4):
                world["outcomes"][candidate]["wdl"] = -1
    for wi in order[:104]:
        source["selection"]["worlds"][wi]["outcomes"][4]["wdl"] = 1
    saved = a.analyze_root(source, CONFIG)
    part = saved["partitions"][0]
    assert part["choices"] == {"anchor": 0, "uniform": 4, "adaptive": 4}
    assert part["assessment_count"] == 25
    assert part["difference_fractions"]["adaptive_minus_anchor"] == [1, 50]
    assert v.verify_root(source, saved, CONFIG)["passed"]


def test_assessment_changes_scores_but_cannot_change_locked_selection():
    config = {**CONFIG, "partitions": 1}
    source = hand_source()
    before = a.analyze_root(source, config)
    changed = copy.deepcopy(source)
    for wi in before["partitions"][0]["assessment_indices"]:
        for outcome in changed["selection"]["worlds"][wi]["outcomes"]:
            outcome["wdl"] = -outcome["wdl"]
    after = a.analyze_root(changed, config)
    for field in ("choices", "adaptive", "uniform_sums", "anchor_sums", "selection_common_indices"):
        assert before["partitions"][0][field] == after["partitions"][0][field]
    assert before["mean_difference_fractions"] != after["mean_difference_fractions"]
    assert v.verify_root(changed, after, config)["passed"]
    with pytest.raises(ValueError):
        v.verify_root(changed, before, config)


@pytest.mark.parametrize("damage", ["common_order", "extension_order", "remainder_order", "survivor", "final_order",
    "final_sum", "final_count", "stage_one", "stage_two", "access", "budget", "choice", "assessment", "tail_count",
    "fraction", "float", "rootmean", "duplicate_partition"])
def test_partition_and_aggregate_mutations_rejected(damage):
    source = hand_source()
    saved = a.analyze_root(source, CONFIG)
    part = saved["partitions"][0]
    if damage in ("common_order", "extension_order", "remainder_order"):
        part[damage][0], part[damage][1] = part[damage][1], part[damage][0]
    elif damage == "survivor": part["adaptive"]["ranked_survivors"][0] = 4
    elif damage == "final_order": part["adaptive"]["final_order"].reverse()
    elif damage == "final_sum": part["adaptive"]["final_sums"][0] += 1
    elif damage == "final_count": part["adaptive"]["final_counts"][0] = 64
    elif damage == "stage_one": part["adaptive"]["stage1_world_indices"][0] = part["assessment_indices"][0]
    elif damage == "stage_two": part["adaptive"]["stage2_world_indices"][0] = part["assessment_indices"][0]
    elif damage == "access": part["selection_common_indices"]["adaptive"][4].append(part["common_order"][32])
    elif damage == "budget": part["selection_cell_counts"]["adaptive"] += 1
    elif damage == "choice": part["choices"]["adaptive"] = 3
    elif damage == "assessment": part["assessment_indices"][0] = part["common_order"][95]
    elif damage == "tail_count": part["assessment_count"] = 64
    elif damage == "fraction": part["difference_fractions"]["adaptive_minus_uniform"] = [2, 4]
    elif damage == "float": part["differences"]["adaptive_minus_uniform"] += 1/256
    elif damage == "rootmean": saved["mean_score_differences"]["adaptive_minus_uniform"] += 1/256
    else: saved["partitions"][-1] = copy.deepcopy(part)
    with pytest.raises(ValueError):
        v.verify_root(source, saved, CONFIG)


@pytest.mark.parametrize("value", [True, False, .5, 2, -2, None, float("nan"), float("inf")])
def test_invalid_source_outcomes_rejected(value):
    source = record()
    saved = a.analyze_root(source, CONFIG)
    source["selection"]["worlds"][0]["outcomes"][0]["wdl"] = value
    with pytest.raises(ValueError, match="WDL"):
        v.verify_root(source, saved, CONFIG)


@pytest.mark.parametrize("damage", ["collision", "duplicate_seed", "mapping", "width", "candidate_order", "duplicate_group"])
def test_source_world_or_candidate_corruption_rejected(damage):
    source = record()
    saved = a.analyze_root(source, CONFIG)
    if damage == "collision":
        source["extension"]["world_seeds"][0] = source["selection"]["world_seeds"][0]
        source["extension"]["worlds"][0]["seed"] = source["selection"]["world_seeds"][0]
    elif damage == "duplicate_seed": source["selection"]["world_seeds"][1] = source["selection"]["world_seeds"][0]
    elif damage == "mapping": source["selection"]["worlds"][0]["seed"] += 1
    elif damage == "width": source["selection"]["worlds"][0]["outcomes"].pop()
    elif damage == "candidate_order": source["candidates"]["breadth"]["indices"].reverse()
    else: source["union"]["keys"][-1] = source["union"]["keys"][0]
    with pytest.raises(ValueError):
        v.verify_root(source, saved, CONFIG)


@pytest.fixture(scope="module")
def panel():
    prototypes = [a.analyze_root(record(support), CONFIG) for support in (4, 5, 6, 7, 8)]
    rows = []
    for fit in CONFIG["fits"]:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                for within in range(20):
                    index = len(rows)
                    row = copy.deepcopy(prototypes[within%5])
                    row["spec"] = {"index": index, "pair": 1000+index, "game_index": 1000+index,
                                   "fit": fit, "seat": seat, "phase": phase}
                    rows.append(row)
    return rows, a.summarize(rows, CONFIG)


def test_independent_integer_histogram_50000_bootstrap_matches_indexed_path(panel):
    rows, stats = panel
    receipt = v.verify_statistics(rows, stats, CONFIG)
    assert receipt["passed"] and receipt["resampling_unit"] == "root"
    assert stats["roots"] == 360 and stats["root_partitions"] == 23040
    assert stats["roots_per_cell"] == 20 and stats["bootstrap_resamples"] == 50000
    for name, _, _ in v.CONTRASTS:
        mean = sum((Fraction(*row["mean_difference_fractions"][name]) for row in rows), Fraction(0))/360
        assert stats["contrasts"][name]["mean_difference_fraction"] == [mean.numerator, mean.denominator]


@pytest.mark.parametrize("damage", ["interval", "denominator", "gate", "mean", "partition_units"])
def test_summary_tampering_rejected(panel, damage):
    rows, stats = panel
    broken = copy.deepcopy(stats)
    contrast = broken["contrasts"]["adaptive_minus_uniform"]
    if damage == "interval": contrast["ci95"][0] += .00001
    elif damage == "denominator": broken["bootstrap_integer_denominator"] += 1
    elif damage == "gate": broken["gate_pass"] = not broken["gate_pass"]
    elif damage == "mean": contrast["mean_difference_fraction"] = [1, 256]
    else: broken["roots_per_cell"] *= 64
    with pytest.raises(ValueError):
        v.verify_statistics(rows, broken, CONFIG)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "source_game", "cell", "fraction", "denominator", "samechoice"])
def test_root_collection_corruption_rejected_before_bootstrap(panel, damage):
    rows, stats = panel
    broken = list(rows)
    if damage == "missing": broken.pop()
    elif damage == "duplicate": broken[-1] = broken[0]
    else:
        broken[-1] = copy.deepcopy(broken[-1])
        row = broken[-1]
        if damage == "source_game": row["spec"]["pair"] = row["spec"]["game_index"] = broken[0]["spec"]["game_index"]
        elif damage == "cell": row["spec"]["seat"] = 0
        elif damage == "fraction": row["partitions"][0]["difference_fractions"]["adaptive_minus_uniform"] = [2, 4]
        elif damage == "denominator": row["partitions"][0]["assessment_count"] = 64
        else:
            part = row["partitions"][0]
            part["choices"]["adaptive"] = part["choices"]["uniform"]
            part["assessment_sums"]["adaptive"] = part["assessment_sums"]["uniform"]+1
    with pytest.raises(ValueError):
        v.verify_statistics(broken, stats, CONFIG)
