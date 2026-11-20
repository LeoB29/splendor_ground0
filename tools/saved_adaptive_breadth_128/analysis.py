"""Pure saved-outcome arithmetic for one fixed adaptive-breadth development screen.

This module has no game, model, filesystem or production-launch dependencies.
Scores are accumulated as integers in twice-win-score units, then compared and
averaged as exact fractions. Only bootstrap arithmetic uses floating point.
"""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math

import numpy as np


TAG = "saved-equal-budget-breadth-v1"
SCOPE = ("Reused-data development comparison under greedy035 continuations; "
         "roots, not worlds or partitions, are resampling units. "
         "Equal continuation counts do not establish equal time or whole-game strength.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid {name}")
    return value


def _fraction_pair(value):
    return [value.numerator, value.denominator]


def _read_fraction(value):
    _require(type(value) is list and len(value) == 2 and
             type(value[0]) is type(value[1]) is int and value[1] > 0,
             "Invalid rational representation")
    result = Fraction(*value)
    _require(_fraction_pair(result) == value, "Unreduced rational representation")
    return result


def _order(length, root_index, partition, domain, seed):
    def priority(index):
        encoded = json.dumps([TAG, seed, root_index, partition, domain, index],
                             separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest(), index
    return sorted(range(length), key=priority)


def _winner(sums, counts):
    """Exact first maximum even when candidates receive unequal counts."""
    _require(bool(sums) and len(sums) == len(counts), "Empty or inconsistent support")
    _require(all(type(value) is int for value in sums) and
             all(type(count) is int and count > 0 for count in counts),
             "Invalid selection sums or counts")
    best = 0
    for index in range(1, len(sums)):
        if sums[index] * counts[best] > sums[best] * counts[index]:
            best = index
    return best


def _matrix(block, candidates, worlds):
    _require(block["candidate_indices"] == list(range(candidates)), "Changed candidate indices")
    _require(len(block["worlds"]) == worlds, "Wrong saved world count")
    _require(len(block["world_seeds"]) == worlds and
             len(set(block["world_seeds"])) == worlds, "Repeated or missing world seeds")
    _require(all(type(seed) is int and seed >= 0 for seed in block["world_seeds"]), "Invalid saved world seed")
    result = []
    for index, row in enumerate(block["worlds"]):
        _require(type(row["seed"]) is int and row["seed"] == block["world_seeds"][index], "Changed saved world identity")
        _require(len(row["outcomes"]) == candidates, "Missing or extra saved outcome")
        values = []
        for outcome in row["outcomes"]:
            wdl = outcome["wdl"]
            _require(type(wdl) in (int, float) and wdl in (-1, 0, 1), "Invalid terminal WDL")
            values.append(int(wdl) + 1)
        result.append(values)
    return result


ARMS = ("anchor", "uniform", "adaptive")
CONTRASTS = (("adaptive_minus_uniform", "adaptive", "uniform"),
             ("adaptive_minus_anchor", "adaptive", "anchor"),
             ("uniform_minus_anchor", "uniform", "anchor"))


def analyze_root(record,config):
    partitions = _integer(config["partitions"],"partition count",1)
    seed = _integer(config["partition_seed"],"partition seed")
    _require("failure" not in record,"Failed source root")
    spec = {key:record["spec"][key] for key in ("index","pair","game_index","fit","seat","phase")}
    for key,value in spec.items():
        _integer(value,f"source {key}")
    _require(spec["pair"] == spec["game_index"] and spec["seat"] in (0,1) and spec["phase"] in (0,1,2),"Invalid source identity")
    support = _integer(record["budgets"]["canonical_support"],"canonical support",1)
    m,n = min(4,support),min(8,support)
    extension_count = 32*(n-m)
    _require(record["budgets"]["anchor_candidates"] == m and record["budgets"]["breadth_candidates"] == n
             and record["budgets"]["extension_worlds"] == extension_count,"Changed source budgets")
    keys = record["union"]["keys"]
    _require(len(keys) == len(set(keys)) == n,"Changed candidate union")
    for arm,width in (("anchor",m),("breadth",n),("precision",m)):
        _require(record["candidates"][arm]["indices"] == list(range(width)) and
                 record["candidates"][arm]["keys"] == keys[:width],"Changed original candidate prefix")
    common = _matrix(record["selection"],n,128)
    extension = _matrix(record["extension"],m,extension_count)
    _require(set(record["selection"]["world_seeds"]).isdisjoint(record["extension"]["world_seeds"]),"World blocks overlap")
    budget = 128*m
    q,r = divmod(budget,n)
    L,H = q+int(r>0),128-q-int(r>0)
    _require(H <= extension_count,"Insufficient saved extension")
    output = []
    totals = {name:Fraction(0) for name,_,_ in CONTRASTS}
    changes = dict.fromkeys(totals,0)
    additions = dict.fromkeys(ARMS,0)
    for partition in range(partitions):
        order = _order(128,spec["index"],partition,"common",seed)
        extended = _order(extension_count,spec["index"],partition,"extension",seed)
        remainder_order = _order(n,spec["index"],partition,"remainder",seed)
        remainder = remainder_order[:r]
        uniform_counts = [q+int(j in remainder) for j in range(n)]
        anchor_counts = [128]*m
        common_access = {"anchor":[order[:L] for _ in range(m)],
                         "uniform":[order[:count] for count in uniform_counts]}
        extension_access = {"anchor":[extended[:H] for _ in range(m)],"uniform":[[] for _ in range(n)]}
        uniform_sums = [sum(common[wi][j] for wi in common_access["uniform"][j]) for j in range(n)]
        anchor_sums = [sum(common[wi][j] for wi in common_access["anchor"][j])+
                       sum(extension[wi][j] for wi in extension_access["anchor"][j]) for j in range(m)]
        choices = {"anchor":_winner(anchor_sums,anchor_counts),"uniform":_winner(uniform_sums,uniform_counts)}
        adaptive = {"active":n == 8,"stage1_world_indices":[],"stage1_sums":[],"ranked_survivors":[],
                    "final_order":[],"stage2_world_indices":[],"final_sums":[],"final_counts":[]}
        if n == 8:
            initial = [sum(common[wi][j] for wi in order[:32]) for j in range(n)]
            ranked = sorted(range(n),key=lambda j:(-initial[j],j))[:4]
            final_order = sorted(ranked)
            final_sums = [sum(common[wi][j] for wi in order[:96]) for j in final_order]
            choices["adaptive"] = final_order[_winner(final_sums,[96]*4)]
            adaptive.update(stage1_world_indices=order[:32],stage1_sums=initial,ranked_survivors=ranked,
                            final_order=final_order,stage2_world_indices=order[32:96],final_sums=final_sums,final_counts=[96]*4)
            common_access["adaptive"] = [order[:96] if j in ranked else order[:32] for j in range(n)]
            assessment = order[96:]
        else:
            choices["adaptive"] = choices["uniform"]
            common_access["adaptive"] = [list(values) for values in common_access["uniform"]]
            assessment = order[L:]
        extension_access["adaptive"] = [[] for _ in range(n)]
        spent = {arm:sum(map(len,common_access[arm]))+sum(map(len,extension_access[arm])) for arm in ARMS}
        _require(all(value == budget for value in spent.values()),"Unequal selection budgets")
        _require(all(set(assessment).isdisjoint(indices) for arm in ARMS for indices in common_access[arm]),"Selection/assessment overlap")
        count = len(assessment)
        if count:
            heldout = {arm:sum(common[wi][choice] for wi in assessment) for arm,choice in choices.items()}
            differences = {name:Fraction(heldout[left]-heldout[right],2*count) for name,left,right in CONTRASTS}
        else:
            _require(n <= 4 and len(set(choices.values())) == 1,"Unsupported no-assessment case")
            heldout = dict.fromkeys(ARMS,None)
            differences = {name:Fraction(0) for name,_,_ in CONTRASTS}
        if n < 8:
            _require(differences["adaptive_minus_uniform"] == 0,"Fallback differs from uniform")
        for name,left,right in CONTRASTS:
            totals[name] += differences[name]
            changes[name] += int(choices[left] != choices[right])
        for arm in ARMS:
            additions[arm] += int(choices[arm] >= m)
        output.append({"partition":partition,"common_order":order,"extension_order":extended,"remainder_order":remainder_order,
            "q":q,"r":r,"L":L,"H":H,"remainder_candidates":remainder,"uniform_counts":uniform_counts,"anchor_counts":anchor_counts,
            "uniform_sums":uniform_sums,"anchor_sums":anchor_sums,"adaptive":adaptive,
            "selection_common_indices":common_access,"selection_extension_indices":extension_access,"selection_cell_counts":spent,
            "choices":choices,"assessment_indices":assessment,"assessment_count":count,"assessment_sums":heldout,
            "difference_fractions":{name:_fraction_pair(value) for name,value in differences.items()},
            "differences":{name:float(value) for name,value in differences.items()}})
    means = {name:value/partitions for name,value in totals.items()}
    return {"spec":spec,"canonical_support":support,"anchor_candidates":m,"breadth_candidates":n,"extension_worlds":extension_count,
        "partitions":output,"mean_difference_fractions":{name:_fraction_pair(value) for name,value in means.items()},
        "mean_score_differences":{name:float(value) for name,value in means.items()},"choice_changes":changes,
        "added_choices":additions,"adaptive_partitions":partitions if n == 8 else 0}


def summarize(root_results,config):
    partitions = _integer(config["partitions"],"partition count",1)
    cell_size = _integer(config["cell_size"],"cell size",1)
    expected_cells = _integer(config["expected_cells"],"cell count",1)
    count = cell_size*expected_cells
    _require(config.get("roots",count) == count,"Root/cell counts disagree")
    _require(len(root_results) == count and [root["spec"]["index"] for root in root_results] == list(range(count)),"Missing, duplicated or reordered roots")
    _require(len({root["spec"]["game_index"] for root in root_results}) == count and
             all(root["spec"]["pair"] == root["spec"]["game_index"] for root in root_results),"Repeated source game")
    resamples = _integer(config["bootstrap_resamples"],"bootstrap count",1)
    seed = _integer(config["bootstrap_seed"],"bootstrap seed")
    _require(config["bootstrap_chunk"] == 500,"Changed bootstrap chunk")
    thresholds = {"incremental":Fraction(str(config["incremental_gain"])),"anchor":Fraction(str(config["anchor_gain"]))}
    _require(all(0 < threshold <= 1 for threshold in thresholds.values()),"Invalid gain thresholds")
    grouped,exact,supports = defaultdict(list),[],defaultdict(list)
    denominator = 1
    names = [name for name,_,_ in CONTRASTS]
    for root in root_results:
        spec = root["spec"]
        _require(spec["seat"] in (0,1) and spec["phase"] in (0,1,2),"Invalid source cell")
        rows = root["partitions"]
        _require(len(rows) == partitions and [p["partition"] for p in rows] == list(range(partitions)),"Incomplete partitions")
        means = []
        for name,left,right in CONTRASTS:
            values = [_read_fraction(p["difference_fractions"][name]) for p in rows]
            _require(all(-1 <= value <= 1 for value in values),"Invalid score difference")
            for p,value in zip(rows,values,strict=True):
                _require(p["differences"][name] == float(value),"Changed partition effect")
                if p["assessment_count"]:
                    _require(value == Fraction(p["assessment_sums"][left]-p["assessment_sums"][right],2*p["assessment_count"]),"Changed assessment denominator")
                else:
                    _require(value == 0 and p["choices"][left] == p["choices"][right],"Changed identity effect")
            mean = sum(values,Fraction(0))/partitions
            _require(root["mean_difference_fractions"][name] == _fraction_pair(mean) and root["mean_score_differences"][name] == float(mean),"Changed root average")
            _require(root["choice_changes"][name] == sum(p["choices"][left] != p["choices"][right] for p in rows),"Changed choice count")
            means.append(mean)
            denominator = math.lcm(denominator,mean.denominator)
        _require(root["adaptive_partitions"] == sum(p["adaptive"]["active"] for p in rows),"Changed adaptive count")
        for arm in ARMS:
            _require(root["added_choices"][arm] == sum(p["choices"][arm] >= root["anchor_candidates"] for p in rows),"Changed addition count")
        grouped[(spec["fit"],spec["seat"],spec["phase"])].append(means)
        exact.append(means)
        supports[root["breadth_candidates"]].append(root)
    cells = sorted(grouped)
    fits = sorted({cell[0] for cell in cells})
    _require(len(cells) == expected_cells and all(len(grouped[cell]) == cell_size for cell in cells),"Incomplete cells")
    _require(cells == [(fit,seat,phase) for fit in fits for seat in (0,1) for phase in range(3)],"Missing fit/seat/phase cell")
    if "fits" in config:
        _require(fits == config["fits"],"Changed source fits")
    _require(denominator*count <= np.iinfo(np.int64).max,"Rational bootstrap overflows int64")
    matrix = np.asarray([[[int(value*denominator) for value in row] for row in grouped[cell]] for cell in cells],dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples,len(CONTRASTS)),dtype=np.int64)
    cell_indices = np.arange(expected_cells,dtype=np.int64)[None,:,None]
    for start in range(0,resamples,500):
        size = min(500,resamples-start)
        draws = rng.integers(0,cell_size,size=(size,expected_cells,cell_size),dtype=np.int64)
        samples[start:start+size] = matrix[cell_indices,draws].sum(axis=(1,2))
    contrasts = {}
    for j,name in enumerate(names):
        mean = sum((row[j] for row in exact),Fraction(0))/count
        interval = (np.quantile(samples[:,j],[.025,.975],method="linear")/(count*denominator)).tolist()
        contrasts[name] = {"mean_difference_fraction":_fraction_pair(mean),"mean_score_difference":float(mean),
            "score_percentage_points":100*float(mean),"ci95":interval,"ci95_percentage_points":[100*value for value in interval]}
    gates = {"incremental_practical_gain":_read_fraction(contrasts[names[0]]["mean_difference_fraction"]) >= thresholds["incremental"],
             "incremental_superiority":contrasts[names[0]]["ci95"][0] > 0,
             "anchor_practical_gain":_read_fraction(contrasts[names[1]]["mean_difference_fraction"]) >= thresholds["anchor"],
             "anchor_superiority":contrasts[names[1]]["ci95"][0] > 0}
    passed = all(gates.values())
    changes = {name:sum(root["choice_changes"][name] for root in root_results) for name in names}
    additions = {arm:sum(root["added_choices"][arm] for root in root_results) for arm in ARMS}
    support_summary = {}
    for n in sorted(supports):
        subset = supports[n]
        support_summary[str(n)] = {"roots":len(subset),"root_partitions":len(subset)*partitions,
            "mean_score_differences":{name:float(sum((_read_fraction(root["mean_difference_fractions"][name]) for root in subset),Fraction(0))/len(subset)) for name in names},
            "choice_changes":{name:sum(root["choice_changes"][name] for root in subset) for name in names},
            "added_choices":{arm:sum(root["added_choices"][arm] for root in subset) for arm in ARMS},
            "adaptive_partitions":sum(root["adaptive_partitions"] for root in subset)}
    total = count*partitions
    return {"roots":count,"partitions_per_root":partitions,"root_partitions":total,"cells":[list(cell) for cell in cells],
        "roots_per_cell":cell_size,"bootstrap_seed":seed,"bootstrap_resamples":resamples,"bootstrap_chunk":500,
        "bootstrap_integer_denominator":denominator,"primary":names[0],"practical_comparison":names[1],"secondary":names[2],
        "contrasts":contrasts,**gates,"gate_pass":passed,
        "classification":"positive_development_screen" if passed else "practical_progress_not_established",
        "descriptive":{"choice_changes":changes,"choice_change_fractions":{name:value/total for name,value in changes.items()},
            "added_choices":additions,"added_choice_fractions":{arm:value/total for arm,value in additions.items()},
            "adaptive_partitions":sum(root["adaptive_partitions"] for root in root_results),"support":support_summary},"scope":SCOPE}


