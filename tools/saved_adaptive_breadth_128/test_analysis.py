"""Synthetic tests only: exact ties, retained observations, fallback and units."""
import copy
from fractions import Fraction

import numpy as np
import pytest

from . import analysis as a


CFG = {"partitions":64,"partition_seed":2140000000,"bootstrap_seed":2163000000,
       "bootstrap_resamples":50000,"bootstrap_chunk":500,"incremental_gain":.0025,"anchor_gain":.005,
       "expected_cells":18,"cell_size":20,"fits":[161001,161002,161003],"roots":360}


def synthetic_root(support=8,index=0,flat=False):
    n,m = min(support,8),min(support,4)
    extension = 32*(n-m)
    keys = [f"candidate_{j}" for j in range(n)]
    def block(width,count,start):
        seeds = list(range(start,start+count))
        return {"candidate_indices":list(range(width)),"world_seeds":seeds,
            "worlds":[{"seed":seed,"outcomes":[{"wdl":0 if flat else ((seed*3+j*13+(seed%7)*j)%3)-1}
                      for j in range(width)]} for seed in seeds]}
    return {"spec":{"index":index,"pair":1000+index,"game_index":1000+index,"fit":161001,"seat":0,"phase":0},
        "budgets":{"canonical_support":support,"anchor_candidates":m,"breadth_candidates":n,"extension_worlds":extension},
        "union":{"keys":keys},"candidates":{arm:{"keys":keys[:count],"indices":list(range(count))}
            for arm,count in (("anchor",m),("breadth",n),("precision",m))},
        "selection":block(n,128,100000),"extension":block(m,extension,200000)}


@pytest.mark.parametrize("support",[1,2,3,4,5,6,7,8,9,50])
def test_all_supports_exact512_or_short_budget_and_fallback(support):
    value = a.analyze_root(synthetic_root(support),CFG)
    for p in value["partitions"]:
        assert set(p["choices"]) == {"anchor","uniform","adaptive"}
        assert p["selection_cell_counts"] == dict.fromkeys(a.ARMS,128*min(support,4))
        assert p["assessment_count"] == (32 if support >= 8 else {1:0,2:0,3:0,4:0,5:25,6:42,7:54}[support])
        assert all(set(indices).isdisjoint(p["assessment_indices"]) for arm in a.ARMS for indices in p["selection_common_indices"][arm])
        if support < 8:
            assert p["choices"]["adaptive"] == p["choices"]["uniform"]
            assert p["selection_common_indices"]["adaptive"] == p["selection_common_indices"]["uniform"]
            assert p["difference_fractions"]["adaptive_minus_uniform"] == [0,1]
            assert not p["adaptive"]["active"] and p["adaptive"]["final_counts"] == []
        else:
            assert sorted(map(len,p["selection_common_indices"]["adaptive"])) == [32]*4+[96]*4
            assert p["adaptive"]["stage1_world_indices"] == p["common_order"][:32]
            assert p["adaptive"]["stage2_world_indices"] == p["common_order"][32:96]


def test_original_order_ties_in_both_stages():
    value = a.analyze_root(synthetic_root(flat=True),CFG)
    for p in value["partitions"]:
        assert p["adaptive"]["ranked_survivors"] == [0,1,2,3]
        assert p["adaptive"]["final_order"] == [0,1,2,3]
        assert p["choices"] == dict.fromkeys(a.ARMS,0)


def staged_tie_root():
    root = synthetic_root(flat=True)
    order = a._order(128,0,0,"common",CFG["partition_seed"])
    for w in root["selection"]["worlds"]:
        for outcome in w["outcomes"]:
            outcome["wdl"] = -1
    for candidate in range(4):
        initial_score2 = 32+8*candidate
        for i,world_index in enumerate(order[:32]):
            root["selection"]["worlds"][world_index]["outcomes"][candidate]["wdl"] = 1 if i < initial_score2-32 else 0
        extra_score2 = 96-initial_score2
        for i,world_index in enumerate(order[32:96]):
            root["selection"]["worlds"][world_index]["outcomes"][candidate]["wdl"] = 0 if i < extra_score2 else -1
    # Candidate7 is poor initially, excellent later, and must remain eliminated.
    for world_index in order[32:96]:
        root["selection"]["worlds"][world_index]["outcomes"][7]["wdl"] = 1
    return root,order


def test_combined96_not_only_new64_and_final_tie_not_initial_ranking():
    root,order = staged_tie_root()
    p = a.analyze_root(root,CFG)["partitions"][0]
    assert p["adaptive"]["stage1_sums"] == [32,40,48,56,0,0,0,0]
    assert p["adaptive"]["ranked_survivors"] == [3,2,1,0]
    assert p["adaptive"]["final_order"] == [0,1,2,3]
    assert p["adaptive"]["final_sums"] == [96]*4
    assert p["choices"]["adaptive"] == 0
    assert len(p["selection_common_indices"]["adaptive"][7]) == 32
    assert sum(root["selection"]["worlds"][i]["outcomes"][7]["wdl"]+1 for i in order[:96]) == 128
    # Now make combined96 favor initial leader3 while only-new64 favors0.
    root["selection"]["worlds"][order[32]]["outcomes"][3]["wdl"] = 1
    p = a.analyze_root(root,CFG)["partitions"][0]
    assert p["adaptive"]["final_sums"] == [96,96,96,97]
    assert p["choices"]["adaptive"] == 3


def test_assessment_changes_scores_without_changing_choices_and_includes_half_draws():
    root,order = staged_tie_root()
    original = a.analyze_root(root,CFG)["partitions"][0]
    changed = copy.deepcopy(root)
    for i in order[96:]:
        changed["selection"]["worlds"][i]["outcomes"][0]["wdl"] = 0
    altered = a.analyze_root(changed,CFG)["partitions"][0]
    assert altered["choices"] == original["choices"]
    assert altered["adaptive"] == original["adaptive"]
    assert altered["assessment_sums"]["adaptive"] == 32
    assert original["assessment_sums"]["adaptive"] == 0


@pytest.mark.parametrize("damage",["wdl","bool","missingworld","duplicateworld","candidateprefix","blockcollision"])
def test_malformed_record_rejected(damage):
    root = synthetic_root()
    if damage == "wdl":root["selection"]["worlds"][0]["outcomes"][0]["wdl"] = .5
    elif damage == "bool":root["selection"]["worlds"][0]["outcomes"][0]["wdl"] = True
    elif damage == "missingworld":root["selection"]["worlds"].pop()
    elif damage == "duplicateworld":root["selection"]["world_seeds"][1] = root["selection"]["world_seeds"][0]
    elif damage == "candidateprefix":root["candidates"]["anchor"]["keys"].reverse()
    else:
        root["extension"]["world_seeds"][0] = root["selection"]["world_seeds"][0]
        root["extension"]["worlds"][0]["seed"] = root["selection"]["world_seeds"][0]
    with pytest.raises(ValueError):a.analyze_root(root,CFG)


def statistic_rows(increment_units=11,anchor_units=21):
    """Hand-built synthetic audited-score records for gate/units tests."""
    result = []
    for fit in CFG["fits"]:
        for seat in (0,1):
            for phase in (0,1,2):
                for within in range(20):
                    parts = []
                    for p in range(64):
                        scores = {"anchor":32,"adaptive":32+(anchor_units if p == 0 else 0),
                                  "uniform":32+(anchor_units-increment_units if p == 0 else 0)}
                        fractions = {name:Fraction(scores[left]-scores[right],64) for name,left,right in a.CONTRASTS}
                        parts.append({"partition":p,"assessment_count":32,"assessment_sums":scores,
                            "choices":{"anchor":0,"uniform":1,"adaptive":2},"adaptive":{"active":True},
                            "difference_fractions":{name:[value.numerator,value.denominator] for name,value in fractions.items()},
                            "differences":{name:float(value) for name,value in fractions.items()}})
                    means = {name:sum((Fraction(*p["difference_fractions"][name]) for p in parts),Fraction(0))/64 for name,_,_ in a.CONTRASTS}
                    index = len(result)
                    result.append({"spec":{"index":index,"pair":1000+index,"game_index":1000+index,"fit":fit,"seat":seat,"phase":phase},
                        "anchor_candidates":4,"breadth_candidates":8,"partitions":parts,
                        "mean_difference_fractions":{name:[value.numerator,value.denominator] for name,value in means.items()},
                        "mean_score_differences":{name:float(value) for name,value in means.items()},
                        "choice_changes":dict.fromkeys(means,64),"added_choices":dict.fromkeys(a.ARMS,0),"adaptive_partitions":64})
    return result


@pytest.mark.parametrize("increment,anchor,passed",[(11,21,True),(10,21,False),(11,20,False),(0,21,False),(-1,21,False),(11,0,False)])
def test_exact_incremental_and_anchor_gates(increment,anchor,passed):
    rows = statistic_rows(increment,anchor)
    value = a.summarize(rows,{**CFG,"bootstrap_resamples":1000})
    assert value["gate_pass"] is passed
    assert value["classification"] == ("positive_development_screen" if passed else "practical_progress_not_established")
    assert value["contrasts"]["adaptive_minus_uniform"]["mean_score_difference"] == increment/4096
    assert value["contrasts"]["adaptive_minus_anchor"]["mean_score_difference"] == anchor/4096
    assert value["contrasts"]["adaptive_minus_uniform"]["ci95"] == [increment/4096]*2


def test_full50000_bootstrap_root_units_and_integer_denominator():
    rows = statistic_rows()
    value = a.summarize(rows,CFG)
    assert value["roots"] == 360 and value["root_partitions"] == 23040 and value["roots_per_cell"] == 20
    assert value["bootstrap_resamples"] == 50000 and value["bootstrap_integer_denominator"] == 4096
    assert value["contrasts"]["adaptive_minus_uniform"]["ci95"] == [11/4096]*2
    assert "teacher" not in a.ARMS and "teacher_minus_anchor" not in value["contrasts"]


@pytest.mark.parametrize("damage",["missingroot","duplicategame","cell","partition","unreduced","denominator","rootmean"])
def test_malformed_summary_inputs_rejected(damage):
    rows = statistic_rows()
    if damage == "missingroot":rows.pop()
    elif damage == "duplicategame":rows[1]["spec"]["pair"] = rows[1]["spec"]["game_index"] = rows[0]["spec"]["game_index"]
    elif damage == "cell":rows[0]["spec"]["seat"] = 1
    elif damage == "partition":rows[0]["partitions"].pop()
    elif damage == "unreduced":rows[0]["partitions"][0]["difference_fractions"]["adaptive_minus_uniform"] = [22,128]
    elif damage == "denominator":rows[0]["partitions"][0]["assessment_count"] = 64
    else:rows[0]["mean_score_differences"]["adaptive_minus_uniform"] += .001
    with pytest.raises(ValueError):a.summarize(rows,{**CFG,"bootstrap_resamples":10})
