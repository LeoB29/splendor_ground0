"""Pure saved-outcome arithmetic for one fixed equal-budget breadth screen.

This module has no game, model, filesystem or production-launch dependencies.
Scores are accumulated as integers in twice-win-score units, then compared and
averaged as exact fractions. Only bootstrap arithmetic uses floating point.
"""
from collections import defaultdict
from fractions import Fraction
import hashlib
import json

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


def analyze_root(record, config):
    """Return all prospectively defined partition accesses and paired scores."""
    partitions = _integer(config["partitions"], "partition count", 1)
    seed = _integer(config["partition_seed"], "partition seed")
    _require("failure" not in record, "Failed root")
    spec = {key: record["spec"][key] for key in ("index", "pair", "game_index", "fit", "seat", "phase")}
    for key, value in spec.items():
        _integer(value, f"source {key}")
    _require(spec["pair"] == spec["game_index"] and spec["seat"] in (0, 1)
             and spec["phase"] in (0, 1, 2), "Invalid source identity")
    support = _integer(record["budgets"]["canonical_support"], "canonical support", 1)
    m, n = min(support, 4), min(support, 8)
    extension_count = 32 * (n - m)
    _require(record["budgets"]["anchor_candidates"] == m and
             record["budgets"]["breadth_candidates"] == n and
             record["budgets"]["extension_worlds"] == extension_count,
             "Changed saved support or extension budget")
    keys = record["union"]["keys"]
    _require(len(keys) == n and len(set(keys)) == n, "Invalid saved union")
    for arm, count in (("anchor", m), ("breadth", n), ("precision", m)):
        _require(record["candidates"][arm]["indices"] == list(range(count)) and
                 record["candidates"][arm]["keys"] == keys[:count],
                 "Changed original candidate prefix")
    common = _matrix(record["selection"], n, 128)
    extension = _matrix(record["extension"], m, extension_count)
    _require(set(record["selection"]["world_seeds"]).isdisjoint(record["extension"]["world_seeds"]),
             "Common and extension worlds overlap")
    budget = 128 * m
    q, r = divmod(budget, n)
    L, H = q + int(r > 0), 128 - q - int(r > 0)
    _require(H <= extension_count and (n > 4 or (L == 128 and H == r == 0)),
             "Insufficient independent extension")
    output, total = [], Fraction(0)
    changes, additions = 0, 0
    for partition in range(partitions):
        common_order = _order(128, spec["index"], partition, "common", seed)
        extension_order = _order(extension_count, spec["index"], partition, "extension", seed)
        remainder_order = _order(n, spec["index"], partition, "remainder", seed)
        remainder = remainder_order[:r]
        broad_counts = [q + int(j in remainder) for j in range(n)]
        broad_sums = [sum(common[wi][j] for wi in common_order[:broad_counts[j]]) for j in range(n)]
        reference_counts = [128] * m
        reference_sums = [sum(common[wi][j] for wi in common_order[:L]) +
                          sum(extension[wi][j] for wi in extension_order[:H]) for j in range(m)]
        choices = {"breadth": _winner(broad_sums, broad_counts),
                   "reference": _winner(reference_sums, reference_counts)}
        assessment = common_order[L:]
        _require(len(assessment) == H and not set(assessment).intersection(common_order[:L]),
                 "Assessment overlaps selection")
        if H:
            heldout = {arm: sum(common[wi][choice] for wi in assessment) for arm, choice in choices.items()}
            difference = Fraction(heldout["breadth"] - heldout["reference"], 2 * H)
        else:
            _require(choices["breadth"] == choices["reference"] and broad_sums == reference_sums,
                     "Short-support rules differ")
            heldout, difference = {"breadth": None, "reference": None}, Fraction(0)
        _require(sum(broad_counts) == sum(reference_counts) == budget, "Unequal selection budgets")
        total += difference
        changes += int(choices["breadth"] != choices["reference"])
        additions += int(choices["breadth"] >= m)
        output.append({"partition": partition, "common_order": common_order,
                       "extension_order": extension_order, "remainder_order": remainder_order,
                       "q": q, "r": r, "L": L, "H": H, "remainder_candidates": remainder,
                       "broad_counts": broad_counts, "reference_counts": reference_counts,
                       "broad_sums": broad_sums, "reference_sums": reference_sums,
                       "choices": choices, "assessment_indices": assessment, "assessment_sums": heldout,
                       "difference_fraction": _fraction_pair(difference), "difference": float(difference),
                       "selection_cell_counts": {"breadth": budget, "reference": budget}})
    mean = total / partitions
    return {"spec": spec, "canonical_support": support, "anchor_candidates": m,
            "breadth_candidates": n, "extension_worlds": extension_count, "partitions": output,
            "mean_difference_fraction": _fraction_pair(mean), "mean_score_difference": float(mean),
            "choice_changes": changes, "breadth_addition_choices": additions}


def summarize(root_results, config):
    """Average within roots, then resample whole roots within the fixed cells."""
    partitions = _integer(config["partitions"], "partition count", 1)
    expected_cells = _integer(config["expected_cells"], "expected cells", 1)
    cell_size = _integer(config["cell_size"], "cell size", 1)
    resamples = _integer(config["bootstrap_resamples"], "bootstrap resamples", 1)
    seed = _integer(config["bootstrap_seed"], "bootstrap seed")
    chunk = 500
    _require(config.get("bootstrap_chunk", chunk) == chunk, "Changed bootstrap chunk")
    threshold = Fraction(str(config["practical_gain"]))
    _require(0 < threshold <= 1, "Invalid practical gain")
    count = expected_cells * cell_size
    _require(config.get("roots", count) == count, "Root and cell counts disagree")
    _require(len(root_results) == count and
             [row["spec"]["index"] for row in root_results] == list(range(count)), "Incomplete or reordered roots")
    _require(len({row["spec"]["game_index"] for row in root_results}) == count and
             all(row["spec"]["pair"] == row["spec"]["game_index"] for row in root_results),
             "Repeated source game")
    grouped, exact, supports = defaultdict(list), [], defaultdict(list)
    changes, additions = 0, 0
    for root in root_results:
        spec = root["spec"]
        _require(spec["seat"] in (0, 1) and spec["phase"] in (0, 1, 2), "Invalid cell")
        values = root["partitions"]
        _require(len(values) == partitions and [v["partition"] for v in values] == list(range(partitions)),
                 "Incomplete or reordered partitions")
        fractions = [_read_fraction(v["difference_fraction"]) for v in values]
        _require(all(-1 <= value <= 1 for value in fractions), "Difference outside score bounds")
        _require(all(v["difference"] == float(f) for v, f in zip(values, fractions)), "Changed partition difference")
        mean = sum(fractions, Fraction(0)) / partitions
        _require(root["mean_difference_fraction"] == _fraction_pair(mean) and
                 root["mean_score_difference"] == float(mean), "Changed root average")
        root_changes = sum(v["choices"]["breadth"] != v["choices"]["reference"] for v in values)
        root_additions = sum(v["choices"]["breadth"] >= root["anchor_candidates"] for v in values)
        _require(root["choice_changes"] == root_changes and root["breadth_addition_choices"] == root_additions,
                 "Changed root movement counts")
        grouped[(spec["fit"], spec["seat"], spec["phase"])].append(float(mean))
        exact.append(mean)
        supports[root["breadth_candidates"]].append((mean, root_changes, root_additions))
        changes += root_changes
        additions += root_additions
    cells = sorted(grouped)
    _require(len(cells) == expected_cells and all(len(grouped[cell]) == cell_size for cell in cells),
             "Incomplete fixed cells")
    fits = sorted({cell[0] for cell in cells})
    _require(cells == [(fit, seat, phase) for fit in fits for seat in (0, 1) for phase in range(3)],
             "Missing fit-seat-phase cell")
    if "fits" in config:
        _require(fits == list(config["fits"]), "Changed original fits")
    matrix = np.asarray([grouped[cell] for cell in cells], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    cell_indices = np.arange(expected_cells, dtype=np.int64)[None, :, None]
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        draws = rng.integers(0, cell_size, size=(size, expected_cells, cell_size), dtype=np.int64)
        samples[start:start + size] = matrix[cell_indices, draws].mean(axis=(1, 2))
    interval = np.quantile(samples, [.025, .975], method="linear").tolist()
    exact_mean = sum(exact, Fraction(0)) / count
    mean = float(exact_mean)
    practical, superiority = exact_mean >= threshold, interval[0] > 0
    passed = bool(practical and superiority)
    support_summary = {}
    for n in sorted(supports):
        rows = supports[n]
        support_summary[str(n)] = {
            "roots": len(rows), "root_partitions": len(rows) * partitions,
            "mean_score_difference": float(sum((row[0] for row in rows), Fraction(0)) / len(rows)),
            "choice_changes": sum(row[1] for row in rows), "breadth_addition_choices": sum(row[2] for row in rows)}
    return {"roots": count, "partitions_per_root": partitions, "root_partitions": count * partitions,
            "cells": [list(cell) for cell in cells], "roots_per_cell": cell_size,
            "bootstrap_seed": seed, "bootstrap_resamples": resamples, "bootstrap_chunk": chunk,
            "primary": "breadth_minus_reference", "mean_difference_fraction": _fraction_pair(exact_mean),
            "mean_score_difference": mean, "score_percentage_points": 100 * mean,
            "ci95": interval, "ci95_percentage_points": [100 * value for value in interval],
            "practical_gain": bool(practical), "superiority": bool(superiority), "gate_pass": passed,
            "classification": "positive_development_screen" if passed else "practical_progress_not_established",
            "descriptive": {"choice_changes": changes, "choice_change_fraction": changes / (count * partitions),
                            "breadth_addition_choices": additions, "breadth_addition_fraction": additions / (count * partitions),
                            "support": support_summary}, "scope": SCOPE}
