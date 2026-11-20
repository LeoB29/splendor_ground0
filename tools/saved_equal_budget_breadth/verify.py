"""Independent arithmetic audit; no production analysis or game/model imports."""
from fractions import Fraction
import hashlib
import json
import math

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(saved, expected, path="result"):
    """Strict structure/numeric comparison, including bool versus integer."""
    _require(type(saved) is type(expected), f"Type mismatch at {path}")
    if isinstance(expected,dict):
        _require(saved.keys() == expected.keys(), f"Field mismatch at {path}")
        for key in expected:
            _same(saved[key],expected[key],f"{path}.{key}")
    elif isinstance(expected,list):
        _require(len(saved) == len(expected),f"Length mismatch at {path}")
        for index,(left,right) in enumerate(zip(saved,expected,strict=True)):
            _same(left,right,f"{path}[{index}]")
    elif isinstance(expected,float):
        _require(math.isfinite(saved) and saved == expected,f"Number mismatch at {path}")
    else:
        _require(saved == expected,f"Value mismatch at {path}")


def _fraction(value):
    _require(isinstance(value,list) and len(value) == 2 and all(type(x) is int for x in value),"Malformed rational value")
    _require(value[1] > 0,"Nonpositive rational denominator")
    result = Fraction(value[0],value[1])
    _require(value == [result.numerator,result.denominator],"Noncanonical rational value")
    return result


def _wire(value):
    return [value.numerator,value.denominator]


def _order(size, root, partition, domain, seed):
    keys = []
    for index in range(size):
        identity = ["saved-equal-budget-breadth-v1",seed,root,partition,domain,index]
        encoded = json.dumps(identity,separators=(",",":"),ensure_ascii=True).encode("utf-8")
        keys.append((hashlib.sha256(encoded).digest(),index))
    return [index for _,index in sorted(keys)]


def _scores(record, block, count, width):
    value = record[block]
    _require(value["candidate_indices"] == list(range(width)),f"Changed {block} candidate order")
    _require(all(type(index) is int for index in value["candidate_indices"]),"Noninteger candidate index")
    _require(len(value["worlds"]) == len(value["world_seeds"]) == count,f"Incorrect {block} world count")
    _require(all(type(seed) is int for seed in value["world_seeds"]),"Noninteger world seed")
    _require(len(set(value["world_seeds"])) == count,"Repeated world identity")
    matrix = []
    for seed,world in zip(value["world_seeds"],value["worlds"],strict=True):
        _require(type(world["seed"]) is int and world["seed"] == seed,"World identity mismatch")
        _require(len(world["outcomes"]) == width,"Changed world candidate vector")
        row = []
        for outcome in world["outcomes"]:
            wdl = outcome["wdl"]
            _require(type(wdl) in (int,float) and wdl in (-1,0,1),"Invalid source WDL domain")
            row.append(int(wdl)+1)
        matrix.append(row)
    return matrix


def _rebuild_root(record, config):
    _require(type(config["partitions"]) is int and config["partitions"] > 0,"Invalid partition count")
    _require(type(config["partition_seed"]) is int,"Invalid partition seed")
    _require("failure" not in record,"Failed source root")
    fields = ("index","pair","game_index","fit","seat","phase")
    spec = {key:record["spec"][key] for key in fields}
    _require(all(type(value) is int for value in spec.values()),"Noninteger source identity")
    _require(spec["index"] >= 0 and spec["pair"] == spec["game_index"],"Invalid root or source-game identity")
    _require(spec["fit"] in (161001,161002,161003) and spec["seat"] in (0,1) and spec["phase"] in (0,1,2),"Invalid source cell")
    support = record["budgets"]["canonical_support"]
    _require(type(support) is int and support >= 1,"Invalid canonical support")
    m,n = min(4,support),min(8,support)
    extension_count = 32*(n-m)
    _require(record["budgets"]["anchor_candidates"] == m and record["budgets"]["breadth_candidates"] == n,"Source support mismatch")
    _require(record["budgets"]["extension_worlds"] == extension_count,"Source extension budget mismatch")
    _require(len(record["union"]["keys"]) == len(set(record["union"]["keys"])) == n,"Invalid canonical candidate union")
    for arm,width in (("anchor",m),("breadth",n),("precision",m)):
        _require(record["candidates"][arm]["indices"] == list(range(width)) and
                 record["candidates"][arm]["keys"] == record["union"]["keys"][:width],"Changed original candidate prefix")
    common = _scores(record,"selection",128,n)
    extension = _scores(record,"extension",extension_count,m)
    _require(not set(record["selection"]["world_seeds"]) & set(record["extension"]["world_seeds"]),"Common/extension world collision")
    budget = 128*m
    q,r = divmod(budget,n)
    L = q+int(r > 0)
    H = 128-L
    _require(0 <= H <= extension_count,"Insufficient independent extension")
    partitions = []
    for partition in range(config["partitions"]):
        common_order = _order(128,spec["index"],partition,"common",config["partition_seed"])
        extension_order = _order(extension_count,spec["index"],partition,"extension",config["partition_seed"])
        remainder_order = _order(n,spec["index"],partition,"remainder",config["partition_seed"])
        remainder = remainder_order[:r]
        counts = [q+int(candidate in remainder) for candidate in range(n)]
        broad_totals = [sum(common[world][candidate] for world in common_order[:counts[candidate]]) for candidate in range(n)]
        reference_totals = [sum(common[world][candidate] for world in common_order[:L])+
                            sum(extension[world][candidate] for world in extension_order[:H]) for candidate in range(m)]
        broad_values = [Fraction(total,count) for total,count in zip(broad_totals,counts,strict=True)]
        reference_values = [Fraction(total,128) for total in reference_totals]
        broad_choice = next(index for index,value in enumerate(broad_values) if value == max(broad_values))
        reference_choice = next(index for index,value in enumerate(reference_values) if value == max(reference_values))
        assessment = common_order[L:]
        _require(len(assessment) == H,"Assessment budget mismatch")
        for count in counts:
            _require(not set(common_order[:count]) & set(assessment),"Broad choice/assessment cell leakage")
        _require(not set(common_order[:L]) & set(assessment),"Reference choice/assessment cell leakage")
        _require(sum(counts) == budget == 128*m,"Unequal selection budgets")
        if H:
            broad_sum = sum(common[world][broad_choice] for world in assessment)
            reference_sum = sum(common[world][reference_choice] for world in assessment)
            delta = Fraction(broad_sum-reference_sum,2*H)
        else:
            _require(broad_choice == reference_choice and broad_totals == reference_totals,"Short support failed exact identity")
            broad_sum = reference_sum = None
            delta = Fraction(0,1)
        partitions.append({"partition":partition,"common_order":common_order,"extension_order":extension_order,
            "remainder_order":remainder_order,"q":q,"r":r,"L":L,"H":H,"remainder_candidates":remainder,
            "broad_counts":counts,"reference_counts":[128]*m,"broad_sums":broad_totals,"reference_sums":reference_totals,
            "choices":{"breadth":broad_choice,"reference":reference_choice},"assessment_indices":assessment,
            "assessment_sums":{"breadth":broad_sum,"reference":reference_sum},"difference_fraction":_wire(delta),
            "difference":float(delta),"selection_cell_counts":{"breadth":budget,"reference":budget}})
    mean = sum((_fraction(part["difference_fraction"]) for part in partitions),Fraction(0))/len(partitions)
    return {"spec":spec,"canonical_support":support,"anchor_candidates":m,"breadth_candidates":n,
        "extension_worlds":extension_count,"partitions":partitions,"mean_difference_fraction":_wire(mean),
        "mean_score_difference":float(mean),"choice_changes":sum(part["choices"]["breadth"] != part["choices"]["reference"] for part in partitions),
        "breadth_addition_choices":sum(part["choices"]["breadth"] >= m for part in partitions)}


def verify_root(record, saved, config):
    rebuilt = _rebuild_root(record,config)
    _same(saved,rebuilt)
    return {"passed":True,"index":rebuilt["spec"]["index"],"partitions":len(rebuilt["partitions"]),
        "independent_selection_and_assessment_reconstruction":True,"whole_world_vectors_preserved":True,
        "exact_rational_comparisons":True,"equal_per_arm_selection_budget":True,
        "mean_difference_fraction":rebuilt["mean_difference_fraction"],"mean_score_difference":rebuilt["mean_score_difference"]}


def _rebuild_statistics(root_results, config):
    _require(root_results and type(config["partitions"]) is int and config["partitions"] > 0,"Missing roots or partitions")
    _require(config["bootstrap_chunk"] == 500,"Bootstrap chunk changed")
    _require(type(config["bootstrap_resamples"]) is int and config["bootstrap_resamples"] > 0,"Invalid bootstrap count")
    _require(type(config["cell_size"]) is int and config["cell_size"] > 0,"Invalid source-cell size")
    ordered = sorted(root_results,key=lambda root:root["spec"]["index"])
    _require([root["spec"]["index"] for root in ordered] == list(range(len(ordered))),"Missing or repeated root index")
    _require(len({root["spec"]["game_index"] for root in ordered}) == len(ordered),"Repeated source game")
    cell_rows,means = {},[]
    for root in ordered:
        spec = root["spec"]
        _require(all(type(value) is int for value in spec.values()),"Noninteger source identity")
        _require(spec["pair"] == spec["game_index"],"Source pair differs from game")
        _require(spec["fit"] in config["fits"] and spec["seat"] in (0,1) and spec["phase"] in (0,1,2),"Invalid source cell")
        n,m = root["breadth_candidates"],root["anchor_candidates"]
        _require(n == min(root["canonical_support"],8) and m == min(root["canonical_support"],4),"Changed support summary")
        _require(len(root["partitions"]) == config["partitions"],"Missing partition in statistics")
        _require([p["partition"] for p in root["partitions"]] == list(range(config["partitions"])),"Duplicate or reordered partitions")
        deltas = []
        for p in root["partitions"]:
            delta = _fraction(p["difference_fraction"])
            _require(type(p["difference"]) is float and p["difference"] == float(delta),"Rounded or altered partition effect")
            _require(type(p["H"]) is int and p["H"] >= 0,"Invalid assessment count")
            if p["H"]:
                sums = p["assessment_sums"]
                _require(all(type(sums[key]) is int and 0 <= sums[key] <= 2*p["H"] for key in ("breadth","reference")),"Invalid assessment totals")
                _require(delta == Fraction(sums["breadth"]-sums["reference"],2*p["H"]),"Wrong assessment denominator")
            else:
                _require(p["assessment_sums"] == {"breadth":None,"reference":None} and p["choices"]["breadth"] == p["choices"]["reference"] and delta == 0,"Short-support effect is not exact identity")
            deltas.append(delta)
        mean = sum(deltas,Fraction(0))/config["partitions"]
        _require(mean == _fraction(root["mean_difference_fraction"]),"Root aggregation mismatch")
        _require(type(root["mean_score_difference"]) is float and root["mean_score_difference"] == float(mean),"Root effect rounded or changed")
        _require(root["choice_changes"] == sum(p["choices"]["breadth"] != p["choices"]["reference"] for p in root["partitions"]),"Changed root choice count")
        _require(root["breadth_addition_choices"] == sum(p["choices"]["breadth"] >= m for p in root["partitions"]),"Changed root addition count")
        cell = (spec["fit"],spec["seat"],spec["phase"])
        cell_rows.setdefault(cell,[]).append((spec["index"],float(mean)))
        means.append(mean)
    cells = sorted(cell_rows)
    expected_cells = [(fit,seat,phase) for fit in config["fits"] for seat in (0,1) for phase in (0,1,2)]
    _require(cells == sorted(expected_cells) and len(cells) == config["expected_cells"],"Incomplete or unexpected source cells")
    _require(all(len(cell_rows[cell]) == config["cell_size"] for cell in cells),"Incorrect roots per cell")
    _require(len(ordered) == config.get("roots",len(cells)*config["cell_size"]),"Incorrect root count")
    matrix = np.array([[value for _,value in sorted(cell_rows[cell])] for cell in cells],dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(config["bootstrap_seed"]))
    replicates = np.empty(config["bootstrap_resamples"],dtype=np.float64)
    for start in range(0,len(replicates),500):
        length = min(500,len(replicates)-start)
        draws = generator.integers(0,config["cell_size"],size=(length,len(cells),config["cell_size"]),dtype=np.int64)
        expanded = np.broadcast_to(matrix,(length,*matrix.shape))
        sampled = np.take_along_axis(expanded,draws,axis=2)
        replicates[start:start+length] = sampled.reshape(length,-1).mean(axis=1)
    bounds = np.quantile(replicates,[.025,.975],method="linear").tolist()
    overall = sum(means,Fraction(0))/len(means)
    practical = overall >= Fraction(str(config["practical_gain"]))
    superiority = bounds[0] > 0
    changed = sum(root["choice_changes"] for root in ordered)
    added = sum(root["breadth_addition_choices"] for root in ordered)
    total = len(ordered)*config["partitions"]
    support = {}
    for n in sorted({root["breadth_candidates"] for root in ordered}):
        subset = [root for root in ordered if root["breadth_candidates"] == n]
        value = sum((_fraction(root["mean_difference_fraction"]) for root in subset),Fraction(0))/len(subset)
        support[str(n)] = {"roots":len(subset),"root_partitions":len(subset)*config["partitions"],
            "mean_score_difference":float(value),"choice_changes":sum(root["choice_changes"] for root in subset),
            "breadth_addition_choices":sum(root["breadth_addition_choices"] for root in subset)}
    return {"roots":len(ordered),"partitions_per_root":config["partitions"],"root_partitions":total,
        "cells":[list(cell) for cell in cells],"roots_per_cell":config["cell_size"],
        "bootstrap_seed":config["bootstrap_seed"],"bootstrap_resamples":config["bootstrap_resamples"],
        "bootstrap_chunk":500,"primary":"breadth_minus_reference","mean_difference_fraction":_wire(overall),
        "mean_score_difference":float(overall),"score_percentage_points":100*float(overall),
        "ci95":bounds,"ci95_percentage_points":[100*value for value in bounds],
        "practical_gain":practical,"superiority":superiority,"gate_pass":practical and superiority,
        "classification":"positive_development_screen" if practical and superiority else "practical_progress_not_established",
        "descriptive":{"choice_changes":changed,"choice_change_fraction":changed/total,
            "breadth_addition_choices":added,"breadth_addition_fraction":added/total,"support":support},
        "scope":"Reused-data development comparison under greedy035 continuations; roots, not worlds or partitions, are resampling units. Equal continuation counts do not establish equal time or whole-game strength."}


def verify_statistics(root_results, savedstats, config):
    rebuilt = _rebuild_statistics(root_results,config)
    _same(savedstats,rebuilt,"statistics")
    return {"passed":True,"independent_bootstrap":True,"resampling_unit":"root",
            "mean_difference_fraction":rebuilt["mean_difference_fraction"],"ci95":rebuilt["ci95"],
            "gate_pass":rebuilt["gate_pass"]}
