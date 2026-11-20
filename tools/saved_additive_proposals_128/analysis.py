"""Exact arithmetic for preserving the anchor while appending a frozen proposal.

This module never loads a model, reconstructs a game, or starts a rollout.
Selection and independent assessment retain their original disjoint roles.
"""
from collections import defaultdict
from fractions import Fraction
import math

import numpy as np

from . import common as c

ARMS = ("anchor", "broad", "narrow", "additive_broad", "additive_narrow", "teacher")
ORIGINAL = ("anchor", "broad", "narrow", "teacher")
ADDITIVE = ("additive_broad", "additive_narrow")
CONTRASTS = {
    "additive_broad_minus_anchor": ("additive_broad", "anchor"),
    "additive_broad_minus_broad": ("additive_broad", "broad"),
    "additive_broad_minus_additive_narrow": ("additive_broad", "additive_narrow"),
    "additive_narrow_minus_anchor": ("additive_narrow", "anchor"),
    "broad_minus_anchor": ("broad", "anchor"),
    "broad_minus_narrow": ("broad", "narrow"),
    "narrow_minus_anchor": ("narrow", "anchor"),
    "teacher_minus_anchor": ("teacher", "anchor"),
}
_COST_FIELDS = ("candidate_counts", "selection_continuations", "extra_candidates",
                "extra_selection_continuations", "changed_choices", "learned_extra_wall",
                "additional_learned_extra_wall")


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _block(data, width, actor):
    """Integer observed sums and unresolved counts, without midpoint imputation."""
    c.require(type(width) is int and 1 <= width <= 10, "Invalid recorded support")
    seeds = data["world_seeds"]
    c.require(len(seeds) == len(set(seeds)) == len(data["worlds"]) == 128
              and all(type(seed) is int for seed in seeds), "Wrong world count or identities")
    totals, unknown = [0]*width, [0]*width
    for seed, world in zip(seeds, data["worlds"], strict=True):
        c.require(world["seed"] == seed and len(world["outcomes"]) == width, "Changed world or outcome count")
        for index, outcome in enumerate(world["outcomes"]):
            reason, winner, value = outcome["reason"], outcome["winner"], outcome["wdl"]
            if reason == "repetition_cutoff":
                c.require(value is None and winner is None, "Invented repetition outcome")
                unknown[index] += 1
            else:
                c.require(reason in ("completed", "forced_pass_deadlock") and _finite(value)
                          and value in (-1, 0, 1) and (winner is None or type(winner) is int and winner in (0, 1)),
                          "Invalid continuation outcome")
                expected = 0 if winner is None else 1 if winner == actor else -1
                c.require(value == expected, "Outcome has wrong scoring actor")
                totals[index] += int(value)
    return totals, unknown


def _select(indices, sums, unknown):
    """Certify the first maximum for every completion of independent unknowns."""
    c.require(indices and len(indices) == len(set(indices))
              and all(type(i) is int and 0 <= i < len(sums) for i in indices), "Invalid ordered arm support")
    winners = []
    for position, candidate in enumerate(indices):
        margins = [(sums[candidate]-sums[other]-unknown[candidate]-unknown[other])
                   for other in indices]
        if all(margin > 0 if j < position else margin >= 0
               for j, margin in enumerate(margins) if j != position):
            winners.append(candidate)
    c.require(len(winners) == 1, "Unresolved outcomes can change the original-order choice")
    return winners[0]


def analyze_root(record):
    """Reconstruct exact128 selections and their already-recorded held-out scores."""
    c.require("failure" not in record, "Failed source root")
    # The authorized source has no coupled roots. Do not silently treat an
    # unfamiliar equivalence proof as a justified correlation between unknowns.
    c.require(not record.get("equivalence_proofs"), "Coupled source proofs require a separately verified adapter")
    spec = record["spec"]
    c.require(type(spec["seat"]) is int and spec["seat"] in (0, 1), "Invalid actor")
    keys, actions = record["union"]["keys"], record["union"]["actions"]
    support = len(keys)
    c.require(1 <= support <= 10 and len(set(keys)) == len(actions) == support, "Invalid source union")
    c.require(set(record["candidates"]) == set(ORIGINAL), "Changed original arms")
    offered = {}
    for arm in ORIGINAL:
        candidate = record["candidates"][arm]
        indices = candidate["indices"]
        c.require(indices and len(indices) == len(set(indices))
                  and all(type(i) is int and 0 <= i < support for i in indices), "Invalid source candidates")
        c.require(candidate["keys"] == [keys[i] for i in indices]
                  and candidate["actions"] == [actions[i] for i in indices], "Changed source representative")
        offered[arm] = list(indices)
    anchor = offered["anchor"]
    c.require(1 <= len(anchor) <= 4 and len(offered["teacher"]) <= 8
              and offered["teacher"][:len(anchor)] == anchor, "Changed original teacher prefix")
    for family in ("broad", "narrow"):
        c.require(len(offered[family]) == len(anchor) and offered[family][:3] == anchor[:3], "Changed shared three")
        offered[f"additive_{family}"] = anchor+[i for i in offered[family] if i not in anchor]
        c.require(len(offered[f"additive_{family}"])-len(anchor) in (0, 1), "More than one additional proposal")
    selection, assessment = record["selection"], record["assessment"]
    c.require(selection["candidate_indices"] == list(range(support)), "Source selection omitted candidates")
    c.require(set(selection["world_seeds"]).isdisjoint(assessment["world_seeds"]), "Selection and assessment overlap")
    sums, missing = _block(selection, support, spec["seat"])
    original_choices = {arm: _select(offered[arm], sums, missing) for arm in ORIGINAL}
    lock = record["selection_lock"]
    c.require(original_choices == lock["choices"] == lock["selection_statistics"]["choices"], "Original choices differ")
    c.require(lock["selection_sha256"] == c.digest(selection), "Selection lock changed")
    c.require(lock["proofs_sha256"] == c.digest([]), "Unexpected proof seal")
    c.require(type(lock["created_at_ns"]) is int and type(record["assessment_started_at_ns"]) is int
              and 0 < lock["created_at_ns"] <= record["assessment_started_at_ns"], "Choice not locked before assessment")
    choices = dict(original_choices)
    for family in ("broad", "narrow"):
        arm = f"additive_{family}"
        chosen = _select(offered[arm], sums, missing)
        first, second = original_choices["anchor"], original_choices[family]
        # This separate two-winner reconstruction checks the sparse-assessment
        # theorem. The anchor is first, including every exact mean tie.
        constituent = _select(list(dict.fromkeys((first, second))), sums, missing)
        c.require(chosen == constituent and chosen in (first, second), "Union winner escaped constituent winners")
        if missing[first] == missing[second] == 0:
            shortcut = second if sums[second] > sums[first] else first
            c.require(chosen == shortcut, "Exact128 shortcut or original-order tie differs")
        choices[arm] = chosen
    choices = {arm: choices[arm] for arm in ARMS}
    assessed = assessment["candidate_indices"]
    c.require(assessed == sorted(set(original_choices.values())) == lock["assessment_indices"]
              and len(assessed) <= 4, "Changed sparse assessment support")
    evaluated, unresolved = _block(assessment, len(assessed), spec["seat"])
    lookup = {candidate: position for position, candidate in enumerate(assessed)}
    c.require(set(choices.values()).issubset(lookup), "Missing held-out outcome for a union choice")
    bounds = {}
    for arm, choice in choices.items():
        position = lookup[choice]
        bounds[arm] = [(128+evaluated[position]-unresolved[position])/256,
                       (128+evaluated[position]+unresolved[position])/256]
    deltas = {name: [0., 0.] if choices[left] == choices[right] else
              [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
              for name, (left, right) in CONTRASTS.items()}
    if "summary" in record:
        c.require(all(record["summary"]["arm_score_bounds"][arm] == bounds[arm] for arm in ORIGINAL),
                  "Original assessment summary differs")
    learned = {arm: record["proposals"][arm]["timing"]["learned_extra_wall"] for arm in ("anchor", "broad", "narrow")}
    c.require(all(_finite(value) and value >= 0 for value in learned.values()), "Invalid saved proposer timing")
    candidate_counts = {arm: len(offered[arm]) for arm in ARMS}
    extra = {arm: candidate_counts[arm]-candidate_counts["anchor"] for arm in ADDITIVE}
    costs = {"candidate_counts": candidate_counts,
             "selection_continuations": {arm: 128*count for arm, count in candidate_counts.items()},
             "extra_candidates": extra, "extra_selection_continuations": {arm: 128*n for arm, n in extra.items()},
             "changed_choices": {arm: choices[arm] != choices["anchor"] for arm in ARMS},
             "learned_extra_wall": learned,
             "additional_learned_extra_wall": {"additive_broad": learned["broad"], "additive_narrow": learned["narrow"]}}
    return {**{key: spec[key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "source_arm")},
            "choices": choices, "arm_score_bounds": bounds, "delta_bounds": deltas, "costs": costs}


def _validate_row(row):
    c.require(set(row["choices"]) == set(row["arm_score_bounds"]) == set(ARMS), "Incomplete analysis arms")
    c.require(set(row["delta_bounds"]) == set(CONTRASTS), "Incomplete contrasts")
    for arm in ARMS:
        c.require(type(row["choices"][arm]) is int and 0 <= row["choices"][arm] < 10, "Invalid union choice")
        bound = row["arm_score_bounds"][arm]
        c.require(len(bound) == 2 and all(_finite(x) and x*256 == int(x*256) for x in bound)
                  and 0 <= bound[0] <= bound[1] <= 1, "Invalid score bound")
    for left in ARMS:
        for right in ARMS:
            c.require(row["choices"][left] != row["choices"][right]
                      or row["arm_score_bounds"][left] == row["arm_score_bounds"][right], "Same action has different score bounds")
    for additive, family in zip(ADDITIVE, ("broad", "narrow"), strict=True):
        c.require(row["choices"][additive] in (row["choices"]["anchor"], row["choices"][family]),
                  "Additive choice escaped constituent winners")
    for name, (left, right) in CONTRASTS.items():
        expected = [0., 0.] if row["choices"][left] == row["choices"][right] else [
            row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
            row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
        c.require(row["delta_bounds"][name] == expected, "Contrast differs from raw arm bounds")
    costs = row["costs"]
    c.require(set(costs) == set(_COST_FIELDS), "Incomplete cost accounting")
    c.require(set(costs["candidate_counts"]) == set(costs["selection_continuations"])
              == set(costs["changed_choices"]) == set(ARMS), "Incomplete arm costs")
    for arm in ARMS:
        count = costs["candidate_counts"][arm]
        c.require(type(count) is int and 1 <= count <= (8 if arm == "teacher" else 5 if arm in ADDITIVE else 4)
                  and costs["selection_continuations"][arm] == 128*count, "Invalid candidate cost")
        c.require(type(costs["changed_choices"][arm]) is bool
                  and costs["changed_choices"][arm] == (row["choices"][arm] != row["choices"]["anchor"]), "Wrong changed-choice cost")
    for field in ("extra_candidates", "extra_selection_continuations", "additional_learned_extra_wall"):
        c.require(set(costs[field]) == set(ADDITIVE), "Incomplete additive costs")
    c.require(set(costs["learned_extra_wall"]) == {"anchor", "broad", "narrow"}, "Incomplete learned timing")
    for arm, elapsed in costs["learned_extra_wall"].items():
        c.require(_finite(elapsed) and elapsed >= 0, "Invalid learned timing")
    for arm, family in zip(ADDITIVE, ("broad", "narrow"), strict=True):
        extra = costs["extra_candidates"][arm]
        c.require(type(extra) is int and extra in (0, 1)
                  and costs["candidate_counts"][arm]-costs["candidate_counts"]["anchor"] == extra
                  and costs["extra_selection_continuations"][arm] == 128*extra
                  and costs["additional_learned_extra_wall"][arm] == costs["learned_extra_wall"][family], "Incorrect extra work")


def summarize(rows, budget=lambda: None):
    """Shared-draw whole-pair bootstrap of conservative score-difference bounds."""
    budget()
    c.require(len(rows) == c.CONFIG["roots"] == 360 and {r["index"] for r in rows} == set(range(360)), "Complete fixed360 panel required")
    c.require(len({r["game_index"] for r in rows}) == 360, "Repeated source game")
    pairs = defaultdict(list)
    for row in rows:
        _validate_row(row)
        pairs[row["pair"]].append(row)
    c.require(len(pairs) == 180, "180 independent whole pairs required")
    strata = defaultdict(list)
    for pair, records in sorted(pairs.items()):
        c.require(len(records) == 2 and {r["seat"] for r in records} == {0, 1}
                  and {r["game_index"] for r in records} == {pair*2, pair*2+1}, "Broken whole source pair")
        cell = tuple(records[0][key] for key in ("fit", "phase", "source_arm"))
        c.require(all(tuple(r[key] for key in ("fit", "phase", "source_arm")) == cell for r in records), "Pair spans sampling cells")
        units = np.zeros((len(CONTRASTS), 2), dtype=np.int64)
        for row in records:
            for j, name in enumerate(CONTRASTS):
                units[j] += np.asarray([int(x*256) for x in row["delta_bounds"][name]], dtype=np.int64)
        strata[cell].append(units)
    cells = [(fit, phase, arm) for fit in c.CONFIG["fits"] for phase in range(3) for arm in ("candidate", "reference")]
    c.require(set(strata) == set(cells) and all(len(v) == 10 for v in strata.values()), "Expected18 strata of10 pairs")
    arrays = [np.stack(strata[cell]) for cell in cells]
    totals = sum(array.sum(axis=0) for array in arrays)
    denominator = len(rows)*256
    point = totals/denominator
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    chunks = []
    for start in range(0, c.CONFIG["bootstrap_resamples"], c.CONFIG["bootstrap_chunk"]):
        budget()
        count = min(c.CONFIG["bootstrap_chunk"], c.CONFIG["bootstrap_resamples"]-start)
        chunk = np.zeros((count, len(CONTRASTS), 2), dtype=np.int64)
        for array in arrays:
            chunk += array[rng.integers(0, 10, size=(count, 10))].sum(axis=1)
        chunks.append(chunk/denominator)
    samples = np.concatenate(chunks)
    low = np.quantile(samples[:, :, 0], .025, axis=0, method="linear")
    high = np.quantile(samples[:, :, 1], .975, axis=0, method="linear")
    contrasts = {name: {"mean": float(point[j, 0]) if totals[j, 0] == totals[j, 1] else None,
                        "mean_bounds": point[j].tolist(), "ci95": [float(low[j]), float(high[j])]}
                 for j, name in enumerate(CONTRASTS)}
    threshold = Fraction(str(c.CONFIG["practical_gain"]))
    gates = {"practical_mean": bool(int(totals[0, 0])*threshold.denominator >= denominator*threshold.numerator),
             "anchor_superiority": bool(low[0] > 0), "replacement_superiority": bool(low[1] > 0),
             "attribution_superiority": bool(low[2] > 0)}
    practical = all(gates[key] for key in ("practical_mean", "anchor_superiority", "replacement_superiority"))
    attribution = practical and gates["attribution_superiority"]
    aggregate = {field: {arm: sum(row["costs"][field][arm] for row in rows)
                         for arm in rows[0]["costs"][field]} for field in _COST_FIELDS}
    aggregate["selection_cost_ratio_to_anchor"] = {
        arm: aggregate["selection_continuations"][arm]/aggregate["selection_continuations"]["anchor"] for arm in ARMS}
    aggregate["elapsed_time_claim"] = "Saved per-model scoring times only; no new deployed wall-time benchmark"
    return {"roots": 360, "pairs": 180, "strata": 18, "bootstrap_samples": c.CONFIG["bootstrap_resamples"],
            "development_only": True, "unknown_outcomes_imputed": False, "contrasts": contrasts, "gates": gates,
            "practical_pass": practical, "attribution_pass": attribution,
            "classification": "additive_broad_practical_and_attribution_signal" if attribution else
                              "additive_broad_practical_signal_only" if practical else "additive_broad_not_established",
            "aggregate_costs": aggregate}
