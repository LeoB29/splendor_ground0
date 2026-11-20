"""Synthetic independent reconstruction and arithmetic tamper checks only."""
import copy
from fractions import Fraction

import pytest

from . import analysis as a
from . import verify as v


CONFIG = {"partitions":64,"partition_seed":2140000000,"bootstrap_seed":2143000000,
    "bootstrap_resamples":50000,"bootstrap_chunk":500,"practical_gain":.005,
    "expected_cells":18,"cell_size":20,"fits":[161001,161002,161003],"roots":360}


def record(support=8,index=0,cell=(161001,0,0),flat=False):
    n,m = min(support,8),min(support,4)
    extension = 32*(n-m)
    def block(worlds,width,offset):
        seeds = list(range(offset,offset+worlds))
        return {"candidate_indices":list(range(width)),"world_seeds":seeds,
            "worlds":[{"seed":seed,"outcomes":[{"wdl":0 if flat else ((seed*11+j*7+j*seed)%3)-1}
                for j in range(width)]} for seed in seeds]}
    return {"spec":{"index":index,"pair":1000+index,"game_index":1000+index,
                    "fit":cell[0],"seat":cell[1],"phase":cell[2]},
        "budgets":{"canonical_support":support,"anchor_candidates":m,"breadth_candidates":n,
                   "extension_worlds":extension},"union":{"keys":[f"action_{j}" for j in range(n)]},
        "candidates":{arm:{"indices":list(range(width)),"keys":[f"action_{j}" for j in range(width)]}
                      for arm,width in (("anchor",m),("breadth",n),("precision",m))},
        "selection":block(128,n,100000),"extension":block(extension,m,200000)}


@pytest.mark.parametrize("support",range(1,13))
def test_independent_all_supports_partition_budgets_and_canonical_means(support):
    source = record(support)
    saved = a.analyze_root(source,CONFIG)
    audit = v.verify_root(source,saved,CONFIG)
    assert audit["passed"] and audit["partitions"] == 64
    for part in saved["partitions"]:
        assert part["selection_cell_counts"] == {"breadth":128*min(support,4),"reference":128*min(support,4)}
        assert not set(part["common_order"][:part["L"]]) & set(part["assessment_indices"])
        if support <= 4:
            assert part["H"] == 0 and part["difference_fraction"] == [0,1]
            assert part["choices"]["breadth"] == part["choices"]["reference"]
        else:
            assert part["H"] == {5:25,6:42,7:54,8:64}[min(support,8)]
    assert Fraction(*saved["mean_difference_fraction"]) == sum((Fraction(*p["difference_fraction"]) for p in saved["partitions"]),Fraction(0))/64


@pytest.mark.parametrize("support",[5,6,7])
def test_unequal_allocation_equal_means_preserve_first_tie_not_largest_sum(support):
    source = record(support,flat=True)
    saved = a.analyze_root(source,CONFIG)
    assert all(p["choices"] == {"breadth":0,"reference":0} for p in saved["partitions"])
    assert any(max(range(support),key=lambda i:p["broad_sums"][i]) != 0 for p in saved["partitions"])
    assert v.verify_root(source,saved,CONFIG)["passed"]
    broken = copy.deepcopy(saved)
    p = next(p for p in broken["partitions"] if max(range(support),key=lambda i:p["broad_sums"][i]) != 0)
    p["choices"]["breadth"] = max(range(support),key=lambda i:p["broad_sums"][i])
    with pytest.raises(ValueError,match="mismatch"):
        v.verify_root(source,broken,CONFIG)


def test_hand_checked_one_fiftieth_effect_not_legacy_one_256_grid():
    source = record(5,flat=True)
    ordering = v._order(128,0,0,"common",CONFIG["partition_seed"])
    for world in source["selection"]["worlds"]:
        for index in range(1,4):
            world["outcomes"][index]["wdl"] = -1
    for world in source["extension"]["worlds"]:
        for index in range(1,4):
            world["outcomes"][index]["wdl"] = -1
    for index in ordering[:103]:
        source["selection"]["worlds"][index]["outcomes"][4]["wdl"] = 1
    source["selection"]["worlds"][ordering[103]]["outcomes"][4]["wdl"] = 1
    saved = a.analyze_root(source,CONFIG)
    first = saved["partitions"][0]
    assert first["choices"] == {"breadth":4,"reference":0}
    assert first["assessment_sums"] == {"breadth":26,"reference":25}
    assert first["difference_fraction"] == [1,50]
    assert Fraction(*first["difference_fraction"])*256 != int(Fraction(*first["difference_fraction"])*256)
    assert v.verify_root(source,saved,CONFIG)["passed"]
    broken = copy.deepcopy(saved)
    broken["partitions"][0]["difference_fraction"] = [1,256]
    broken["partitions"][0]["difference"] = 1/256
    with pytest.raises(ValueError):
        v.verify_root(source,broken,CONFIG)


@pytest.mark.parametrize("damage",["common_order","extension_order","remainder_order","count","sum","choice","assessment","fraction","float","rootmean","duplicate_partition"])
def test_saved_partition_or_aggregate_tampering_is_rejected(damage):
    source = record(7)
    saved = a.analyze_root(source,CONFIG)
    p = saved["partitions"][0]
    if damage in ("common_order","extension_order","remainder_order"):
        p[damage][0],p[damage][1] = p[damage][1],p[damage][0]
    elif damage == "count": p["broad_counts"][0] += 1
    elif damage == "sum": p["reference_sums"][0] += 1
    elif damage == "choice": p["choices"]["breadth"] = (p["choices"]["breadth"]+1)%7
    elif damage == "assessment": p["assessment_indices"][0] = p["common_order"][0]
    elif damage == "fraction": p["difference_fraction"] = [2,4]
    elif damage == "float": p["difference"] += 1/256
    elif damage == "rootmean": saved["mean_score_difference"] += 1/256
    else: saved["partitions"][-1] = copy.deepcopy(p)
    with pytest.raises(ValueError):
        v.verify_root(source,saved,CONFIG)


@pytest.mark.parametrize("value",[True,False,.5,2,-2,None,float("nan"),float("inf")])
def test_source_invalid_wdl_domain_rejected(value):
    source = record(8)
    saved = a.analyze_root(source,CONFIG)
    source["selection"]["worlds"][0]["outcomes"][0]["wdl"] = value
    with pytest.raises(ValueError,match="WDL"):
        v.verify_root(source,saved,CONFIG)


@pytest.mark.parametrize("damage",["collision","duplicate_seed","seed_mapping","vector_width","support","candidate_order","duplicate_group"])
def test_source_world_vectors_and_identity_damage_rejected(damage):
    source = record(8)
    saved = a.analyze_root(source,CONFIG)
    if damage == "collision":
        source["extension"]["world_seeds"][0] = source["selection"]["world_seeds"][0]
        source["extension"]["worlds"][0]["seed"] = source["selection"]["world_seeds"][0]
    elif damage == "duplicate_seed": source["selection"]["world_seeds"][1] = source["selection"]["world_seeds"][0]
    elif damage == "seed_mapping": source["selection"]["worlds"][0]["seed"] += 1
    elif damage == "vector_width": source["selection"]["worlds"][0]["outcomes"].pop()
    elif damage == "support": source["budgets"]["breadth_candidates"] = 7
    elif damage == "candidate_order": source["selection"]["candidate_indices"].reverse()
    else: source["union"]["keys"][-1] = source["union"]["keys"][0]
    with pytest.raises(ValueError):
        v.verify_root(source,saved,CONFIG)


@pytest.fixture(scope="module")
def panel():
    # Synthetic summaries preserve exact64-partition effects but avoid360 full
    # serialized outcome blocks in the statistics-only tests.
    prototypes = [a.analyze_root(record(support),CONFIG) for support in (4,5,6,7,8)]
    values = []
    for fit in CONFIG["fits"]:
        for seat in (0,1):
            for phase in (0,1,2):
                for within in range(20):
                    index = len(values)
                    value = copy.deepcopy(prototypes[within%5])
                    value["spec"] = {"index":index,"pair":1000+index,"game_index":1000+index,"fit":fit,"seat":seat,"phase":phase}
                    values.append(value)
    summary = a.summarize(values,CONFIG)
    return values,summary


def test_independent_full_50000_bootstrap_uses360_roots_and18_cells(panel):
    values,summary = panel
    audit = v.verify_statistics(values,summary,CONFIG)
    assert audit["passed"] and audit["resampling_unit"] == "root"
    assert summary["roots"] == 360 and summary["root_partitions"] == 23040
    assert summary["roots_per_cell"] == 20 and summary["bootstrap_resamples"] == 50000
    assert summary["mean_score_difference"] == float(sum((Fraction(*v["mean_difference_fraction"]) for v in values),Fraction(0))/360)


@pytest.mark.parametrize("damage",["interval","denominator","gate","mean","rootcount","partitions_as_units"])
def test_statistics_tampering_rejected(panel,damage):
    values,summary = panel
    broken = copy.deepcopy(summary)
    if damage == "interval": broken["ci95"][0] += .00001
    elif damage == "denominator": broken["mean_difference_fraction"] = [1,256]
    elif damage == "gate": broken["gate_pass"] = not broken["gate_pass"]
    elif damage == "mean": broken["mean_score_difference"] += .001
    elif damage == "rootcount": broken["roots"] = 23040
    else: broken["roots_per_cell"] *= 64
    with pytest.raises(ValueError):
        v.verify_statistics(values,broken,CONFIG)


@pytest.mark.parametrize("damage",["missing","duplicate","source_game","cell","fraction"])
def test_malformed_root_collection_rejected_before_resampling(panel,damage):
    values,summary = panel
    broken = list(values)
    if damage == "missing": broken.pop()
    elif damage == "duplicate": broken[-1] = broken[0]
    else:
        broken[-1] = copy.deepcopy(broken[-1])
        if damage == "source_game": broken[-1]["spec"]["pair"] = broken[-1]["spec"]["game_index"] = broken[0]["spec"]["game_index"]
        elif damage == "cell": broken[-1]["spec"]["seat"] = 0
        else: broken[-1]["partitions"][0]["difference_fraction"] = [2,4]
    with pytest.raises(ValueError):
        v.verify_statistics(broken,summary,CONFIG)
