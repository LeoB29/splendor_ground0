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


ARMS = ("anchor", "uniform", "adaptive")
CONTRASTS = (("adaptive_minus_uniform", "adaptive", "uniform"),
             ("adaptive_minus_anchor", "adaptive", "anchor"),
             ("uniform_minus_anchor", "uniform", "anchor"))


def _rebuild_root(record, config):
    _require(type(config["partitions"]) is int and config["partitions"] > 0, "Invalid partition count")
    _require(type(config["partition_seed"]) is int, "Invalid partition seed")
    _require("failure" not in record, "Failed source root")
    fields = ("index", "pair", "game_index", "fit", "seat", "phase")
    spec = {key: record["spec"][key] for key in fields}
    _require(all(type(value) is int for value in spec.values()), "Noninteger source identity")
    _require(spec["index"] >= 0 and spec["pair"] == spec["game_index"], "Invalid root or source-game identity")
    _require(spec["fit"] in (161001, 161002, 161003) and spec["seat"] in (0, 1)
             and spec["phase"] in (0, 1, 2), "Invalid source cell")
    support = record["budgets"]["canonical_support"]
    _require(type(support) is int and support >= 1, "Invalid canonical support")
    m, n = min(4, support), min(8, support)
    extension_count = 32*(n-m)
    _require(record["budgets"]["anchor_candidates"] == m
             and record["budgets"]["breadth_candidates"] == n, "Source support mismatch")
    _require(record["budgets"]["extension_worlds"] == extension_count, "Source extension budget mismatch")
    _require(len(record["union"]["keys"]) == len(set(record["union"]["keys"])) == n,
             "Invalid canonical candidate union")
    for arm, width in (("anchor", m), ("breadth", n), ("precision", m)):
        _require(record["candidates"][arm]["indices"] == list(range(width))
                 and record["candidates"][arm]["keys"] == record["union"]["keys"][:width],
                 "Changed original candidate prefix")
    common = _scores(record, "selection", 128, n)
    extension = _scores(record, "extension", extension_count, m)
    _require(not set(record["selection"]["world_seeds"]) & set(record["extension"]["world_seeds"]),
             "Common/extension world collision")
    budget = 128*m
    q, r = divmod(budget, n)
    L, H = q+int(r > 0), 128-q-int(r > 0)
    _require(0 <= H <= extension_count, "Insufficient independent extension")
    partitions = []
    for partition in range(config["partitions"]):
        order = _order(128, spec["index"], partition, "common", config["partition_seed"])
        extra = _order(extension_count, spec["index"], partition, "extension", config["partition_seed"])
        remainder_order = _order(n, spec["index"], partition, "remainder", config["partition_seed"])
        remainder = remainder_order[:r]
        counts = [q+int(candidate in remainder) for candidate in range(n)]
        uniform_sums = [sum(common[wi][j] for wi in order[:counts[j]]) for j in range(n)]
        anchor_sums = [sum(common[wi][j] for wi in order[:L])
                       + sum(extension[wi][j] for wi in extra[:H]) for j in range(m)]
        values = [Fraction(value, count) for value, count in zip(uniform_sums, counts, strict=True)]
        choices = {"anchor": next(j for j, value in enumerate(anchor_sums) if value == max(anchor_sums)),
                   "uniform": next(j for j, value in enumerate(values) if value == max(values))}
        common_access = {"anchor": [order[:L] for _ in range(m)],
                         "uniform": [order[:count] for count in counts]}
        extension_access = {"anchor": [extra[:H] for _ in range(m)],
                            "uniform": [[] for _ in range(n)], "adaptive": [[] for _ in range(n)]}
        adaptive = {"active": n == 8, "stage1_world_indices": [], "stage1_sums": [],
                    "ranked_survivors": [], "final_order": [], "stage2_world_indices": [],
                    "final_sums": [], "final_counts": []}
        if n == 8:
            stage1 = [sum(common[wi][j] for wi in order[:32]) for j in range(n)]
            ranked = sorted(range(n), key=lambda j: (-stage1[j], j))[:4]
            final_order = sorted(ranked)
            final_sums = [stage1[j]+sum(common[wi][j] for wi in order[32:96]) for j in final_order]
            choices["adaptive"] = final_order[next(k for k, value in enumerate(final_sums)
                                                     if value == max(final_sums))]
            common_access["adaptive"] = [order[:96] if j in ranked else order[:32] for j in range(n)]
            adaptive.update(stage1_world_indices=order[:32], stage1_sums=stage1,
                            ranked_survivors=ranked, final_order=final_order,
                            stage2_world_indices=order[32:96], final_sums=final_sums,
                            final_counts=[96]*4)
            assessment = order[96:]
        else:
            choices["adaptive"] = choices["uniform"]
            common_access["adaptive"] = [list(indices) for indices in common_access["uniform"]]
            assessment = order[L:]
        assessment_count = len(assessment)
        cell_counts = {arm: sum(map(len, common_access[arm]))+sum(map(len, extension_access[arm])) for arm in ARMS}
        _require(all(value == budget for value in cell_counts.values()), "Unequal selection budgets")
        for accesses in common_access.values():
            _require(all(not set(indices) & set(assessment) for indices in accesses), "Selection/assessment cell leakage")
        if assessment_count:
            sums = {arm: sum(common[wi][choices[arm]] for wi in assessment) for arm in ARMS}
            differences = {name: Fraction(sums[left]-sums[right], 2*assessment_count)
                           for name, left, right in CONTRASTS}
        else:
            _require(len(set(choices.values())) == 1 and uniform_sums == anchor_sums,
                     "Short support failed exact identity")
            sums = dict.fromkeys(ARMS)
            differences = {name: Fraction(0) for name, _, _ in CONTRASTS}
        partitions.append({"partition": partition, "common_order": order, "extension_order": extra,
            "remainder_order": remainder_order, "q": q, "r": r, "L": L, "H": H,
            "remainder_candidates": remainder, "uniform_counts": counts, "anchor_counts": [128]*m,
            "uniform_sums": uniform_sums, "anchor_sums": anchor_sums, "adaptive": adaptive,
            "selection_common_indices": common_access, "selection_extension_indices": extension_access,
            "selection_cell_counts": cell_counts, "choices": choices, "assessment_indices": assessment,
            "assessment_count": assessment_count, "assessment_sums": sums,
            "difference_fractions": {name: _wire(value) for name, value in differences.items()},
            "differences": {name: float(value) for name, value in differences.items()}})
    means = {name: sum((_fraction(part["difference_fractions"][name]) for part in partitions), Fraction(0))/len(partitions)
             for name, _, _ in CONTRASTS}
    return {"spec": spec, "canonical_support": support, "anchor_candidates": m, "breadth_candidates": n,
            "extension_worlds": extension_count, "partitions": partitions,
            "mean_difference_fractions": {name: _wire(value) for name, value in means.items()},
            "mean_score_differences": {name: float(value) for name, value in means.items()},
            "choice_changes": {name: sum(part["choices"][left] != part["choices"][right] for part in partitions)
                               for name, left, right in CONTRASTS},
            "added_choices": {arm: sum(part["choices"][arm] >= m for part in partitions) for arm in ARMS},
            "adaptive_partitions": sum(part["adaptive"]["active"] for part in partitions)}


def verify_root(record, saved, config):
    rebuilt = _rebuild_root(record, config)
    _same(saved, rebuilt)
    return {"passed": True, "index": rebuilt["spec"]["index"], "partitions": len(rebuilt["partitions"]),
            "independent_selection_and_assessment_reconstruction": True, "whole_world_vectors_preserved": True,
            "exact_rational_comparisons": True, "equal_per_arm_selection_budget": True,
            "mean_difference_fractions": rebuilt["mean_difference_fractions"],
            "mean_score_differences": rebuilt["mean_score_differences"]}


def _rebuild_statistics(root_results, config):
    _require(root_results and type(config["partitions"]) is int and config["partitions"] > 0,
             "Missing roots or partitions")
    _require(config["bootstrap_chunk"] == 500, "Bootstrap chunk changed")
    _require(type(config["bootstrap_resamples"]) is int and config["bootstrap_resamples"] > 0,
             "Invalid bootstrap count")
    _require(type(config["cell_size"]) is int and config["cell_size"] > 0, "Invalid source-cell size")
    ordered = sorted(root_results, key=lambda root: root["spec"]["index"])
    _require([root["spec"]["index"] for root in ordered] == list(range(len(ordered))),
             "Missing or repeated root index")
    _require(len({root["spec"]["game_index"] for root in ordered}) == len(ordered), "Repeated source game")
    names = [name for name, _, _ in CONTRASTS]
    grouped, all_means = {}, []
    for root in ordered:
        spec = root["spec"]
        _require(all(type(value) is int for value in spec.values()), "Noninteger source identity")
        _require(spec["pair"] == spec["game_index"], "Source pair differs from game")
        _require(spec["fit"] in config["fits"] and spec["seat"] in (0, 1) and spec["phase"] in (0, 1, 2),
                 "Invalid source cell")
        n, m = root["breadth_candidates"], root["anchor_candidates"]
        _require(type(root["canonical_support"]) is int and root["canonical_support"] > 0
                 and n == min(root["canonical_support"], 8) and m == min(root["canonical_support"], 4),
                 "Changed support summary")
        _require(root["extension_worlds"] == 32*(n-m), "Changed extension summary")
        _require(len(root["partitions"]) == config["partitions"]
                 and [p["partition"] for p in root["partitions"]] == list(range(config["partitions"])),
                 "Missing, duplicate or reordered partitions")
        effects = {name: [] for name in names}
        for part in root["partitions"]:
            count = part["assessment_count"]
            q, remainder = divmod(128*m, n)
            old_tail = 128-q-int(remainder > 0)
            _require(type(count) is int and count == (32 if n == 8 else old_tail)
                     and part["H"] == old_tail, "Invalid assessment count")
            _require(set(part["choices"]) == set(ARMS)
                     and all(type(value) is int and 0 <= value < (m if arm == "anchor" else n)
                             for arm, value in part["choices"].items()), "Choice outside candidate support")
            _require(type(part["adaptive"]["active"]) is bool and part["adaptive"]["active"] == (n == 8),
                     "Changed adaptive activation")
            if n < 8:
                _require(part["choices"]["adaptive"] == part["choices"]["uniform"], "Short adaptive fallback changed")
            sums = part["assessment_sums"]
            _require(set(sums) == set(ARMS), "Missing assessment arm")
            if count:
                _require(all(type(value) is int and 0 <= value <= 2*count for value in sums.values()),
                         "Invalid assessment totals")
                for left in ARMS:
                    for right in ARMS:
                        if part["choices"][left] == part["choices"][right]:
                            _require(sums[left] == sums[right], "Identical choices have different assessment")
            else:
                _require(sums == dict.fromkeys(ARMS) and len(set(part["choices"].values())) == 1,
                         "Short-support effect is not exact identity")
            _require(set(part["difference_fractions"]) == set(part["differences"]) == set(names),
                     "Missing partition contrast")
            for name, left, right in CONTRASTS:
                delta = _fraction(part["difference_fractions"][name])
                expected = Fraction(sums[left]-sums[right], 2*count) if count else Fraction(0)
                _require(delta == expected, "Wrong assessment denominator")
                _require(type(part["differences"][name]) is float and part["differences"][name] == float(delta),
                         "Rounded or altered partition effect")
                effects[name].append(delta)
        means = {name: sum(effects[name], Fraction(0))/config["partitions"] for name in names}
        _require(set(root["mean_difference_fractions"]) == set(root["mean_score_differences"]) == set(names),
                 "Missing root contrast")
        for name, left, right in CONTRASTS:
            _require(means[name] == _fraction(root["mean_difference_fractions"][name]), "Root aggregation mismatch")
            _require(type(root["mean_score_differences"][name]) is float
                     and root["mean_score_differences"][name] == float(means[name]), "Root effect rounded or changed")
            _require(root["choice_changes"][name] == sum(p["choices"][left] != p["choices"][right]
                                                         for p in root["partitions"]), "Changed root choice count")
        _require(root["added_choices"] == {arm: sum(p["choices"][arm] >= m for p in root["partitions"])
                                           for arm in ARMS}, "Changed root addition count")
        _require(root["adaptive_partitions"] == config["partitions"]*int(n == 8), "Changed adaptive count")
        cell = (spec["fit"], spec["seat"], spec["phase"])
        grouped.setdefault(cell, []).append((spec["index"], means))
        all_means.append(means)
    cells = sorted(grouped)
    expected_cells = [(fit, seat, phase) for fit in config["fits"] for seat in (0, 1) for phase in (0, 1, 2)]
    _require(cells == sorted(expected_cells) and len(cells) == config["expected_cells"],
             "Incomplete or unexpected source cells")
    _require(all(len(grouped[cell]) == config["cell_size"] for cell in cells), "Incorrect roots per cell")
    _require(len(ordered) == config.get("roots", len(cells)*config["cell_size"]), "Incorrect root count")
    denominator = math.lcm(*(value.denominator for means in all_means for value in means.values()))
    _require(denominator*len(ordered) <= np.iinfo(np.int64).max, "Bootstrap integer scale overflow")
    matrix = np.asarray([[[int(means[name]*denominator) for name in names]
                          for _, means in sorted(grouped[cell])] for cell in cells], dtype=np.int64)
    generator = np.random.Generator(np.random.PCG64(config["bootstrap_seed"]))
    replicates = np.empty((config["bootstrap_resamples"], len(names)), dtype=np.int64)
    for start in range(0, len(replicates), 500):
        length = min(500, len(replicates)-start)
        draws = generator.integers(0, config["cell_size"], size=(length, len(cells), config["cell_size"]), dtype=np.int64)
        totals = np.zeros((length, len(names)), dtype=np.int64)
        # Count root multiplicities independently instead of gathering and
        # summing the production analysis's indexed root-value arrays.
        for cell_index in range(len(cells)):
            multiplicities = np.zeros((length, config["cell_size"]), dtype=np.int64)
            np.add.at(multiplicities, (np.arange(length)[:, None], draws[:, cell_index, :]), 1)
            totals += multiplicities @ matrix[cell_index]
        replicates[start:start+length] = totals
    contrasts = {}
    divisor = len(ordered)*denominator
    for column, name in enumerate(names):
        values = sorted(int(value) for value in replicates[:, column])
        def percentile(probability):
            location = (len(values)-1)*probability
            low, high = math.floor(location), math.ceil(location)
            weight = location-low
            difference = values[high]-values[low]
            interpolated = (values[high]-difference*(1-weight) if weight >= .5
                            else values[low]+difference*weight)
            return interpolated/divisor
        interval = [percentile(.025), percentile(.975)]
        mean = sum((row[name] for row in all_means), Fraction(0))/len(ordered)
        contrasts[name] = {"mean_difference_fraction": _wire(mean), "mean_score_difference": float(mean),
                           "score_percentage_points": 100*float(mean), "ci95": interval,
                           "ci95_percentage_points": [100*value for value in interval]}
    incremental = contrasts["adaptive_minus_uniform"]
    anchor = contrasts["adaptive_minus_anchor"]
    gates = {"incremental_practical_gain": _fraction(incremental["mean_difference_fraction"]) >= Fraction(str(config["incremental_gain"])),
             "incremental_superiority": incremental["ci95"][0] > 0,
             "anchor_practical_gain": _fraction(anchor["mean_difference_fraction"]) >= Fraction(str(config["anchor_gain"])),
             "anchor_superiority": anchor["ci95"][0] > 0}
    gate_pass = all(gates.values())
    changes = {name: sum(root["choice_changes"][name] for root in ordered) for name in names}
    added = {arm: sum(root["added_choices"][arm] for root in ordered) for arm in ARMS}
    total = len(ordered)*config["partitions"]
    support = {}
    for n in sorted({root["breadth_candidates"] for root in ordered}):
        subset = [root for root in ordered if root["breadth_candidates"] == n]
        means = {name: sum((_fraction(root["mean_difference_fractions"][name]) for root in subset), Fraction(0))/len(subset)
                 for name in names}
        support[str(n)] = {"roots": len(subset), "root_partitions": len(subset)*config["partitions"],
            "mean_score_differences": {name: float(value) for name, value in means.items()},
            "choice_changes": {name: sum(root["choice_changes"][name] for root in subset) for name in names},
            "added_choices": {arm: sum(root["added_choices"][arm] for root in subset) for arm in ARMS},
            "adaptive_partitions": sum(root["adaptive_partitions"] for root in subset)}
    return {"roots": len(ordered), "partitions_per_root": config["partitions"], "root_partitions": total,
            "cells": [list(cell) for cell in cells], "roots_per_cell": config["cell_size"],
            "bootstrap_seed": config["bootstrap_seed"], "bootstrap_resamples": config["bootstrap_resamples"],
            "bootstrap_chunk": 500, "bootstrap_integer_denominator": denominator,
            "primary": "adaptive_minus_uniform", "practical_comparison": "adaptive_minus_anchor",
            "secondary": "uniform_minus_anchor", "contrasts": contrasts, **gates, "gate_pass": gate_pass,
            "classification": "positive_development_screen" if gate_pass else "practical_progress_not_established",
            "descriptive": {"choice_changes": changes, "choice_change_fractions": {name: value/total for name, value in changes.items()},
                "added_choices": added, "added_choice_fractions": {arm: value/total for arm, value in added.items()},
                "adaptive_partitions": sum(root["adaptive_partitions"] for root in ordered), "support": support},
            "scope": "Reused-data development comparison under greedy035 continuations; roots, not worlds or partitions, are resampling units. Equal continuation counts do not establish equal time or whole-game strength."}


def verify_statistics(root_results, savedstats, config):
    rebuilt = _rebuild_statistics(root_results, config)
    _same(savedstats, rebuilt, "statistics")
    return {"passed": True, "independent_bootstrap": True, "resampling_unit": "root",
            "bootstrap_integer_denominator": rebuilt["bootstrap_integer_denominator"],
            "contrasts": rebuilt["contrasts"],
            **{key: rebuilt[key] for key in ("incremental_practical_gain", "incremental_superiority",
                                             "anchor_practical_gain", "anchor_superiority", "gate_pass", "classification")}}


