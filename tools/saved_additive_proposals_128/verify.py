"""Independent arithmetic audit; never imports the collector, model or simulator."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from . import common as c

ORIGINAL = ("anchor", "broad", "narrow", "teacher")
ARMS = ("anchor", "broad", "narrow", "additive_broad", "additive_narrow", "teacher")
CONTRASTS = (("additive_broad_minus_anchor", "additive_broad", "anchor"),
             ("additive_broad_minus_broad", "additive_broad", "broad"),
             ("additive_broad_minus_additive_narrow", "additive_broad", "additive_narrow"),
             ("additive_narrow_minus_anchor", "additive_narrow", "anchor"),
             ("broad_minus_anchor", "broad", "anchor"),
             ("broad_minus_narrow", "broad", "narrow"),
             ("narrow_minus_anchor", "narrow", "anchor"),
             ("teacher_minus_anchor", "teacher", "anchor"))
META = ("index", "pair", "game_index", "fit", "seat", "phase", "source_arm")
FITS = (161001, 161002, 161003)


def compare(left, right, location="root"):
    """Exact structure and discrete values; only numerical roundoff is tolerated."""
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), location
        for key in left:
            compare(left[key], right[key], location+"."+str(key))
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple)) and len(left) == len(right), location
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, location+"["+str(index)+"]")
    elif type(left) in (float, int) and type(right) in (float, int):
        assert math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=0, abs_tol=1e-12), location
    else:
        assert type(left) is type(right) and left == right, location


def _column_sums(worlds, support, actor):
    assert len(worlds) == 128 and type(support) is int and 1 <= support <= 10
    totals, missing, cells = [0]*support, [0]*support, []
    for wi, world in enumerate(worlds):
        assert len(world["outcomes"]) == support
        for ci, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            if value is None:
                assert outcome["reason"] == "repetition_cutoff" and outcome["winner"] is None
                missing[ci] += 1
                cells.append({"world": wi, "candidate": ci})
            else:
                assert outcome["reason"] in ("completed", "forced_pass_deadlock")
                assert type(value) in (float, int) and value in (-1, 0, 1)
                winner = outcome["winner"]
                assert winner is None or type(winner) is int and winner in (0, 1)
                assert value == (0 if winner is None else 1 if winner == actor else -1)
                totals[ci] += int(value)
    return totals, missing, cells


def _certified_choice(indices, totals, missing):
    assert indices and len(indices) == len(set(indices))
    assert all(type(j) is int and 0 <= j < len(totals) for j in indices)
    winners = []
    for position, candidate in enumerate(indices):
        margins = [0 if other == candidate else totals[candidate]-totals[other]-missing[candidate]-missing[other]
                   for other in indices]
        if all(margin >= (1 if other_position < position else 0)
               for other_position, margin in enumerate(margins)):
            winners.append((candidate, margins))
    assert len(winners) == 1, "Unresolved outcome can change the ordered union choice; hold"
    return winners[0]


def _row(record):
    """Rebuild old and additive choices from raw outcomes, preserving source order."""
    assert record["manifest"] == c.SOURCE_BINDING and record["stage"] == "main"
    assert "failure" not in record and not record.get("equivalence_proofs"), "Unsupported source coupling proof; hold"
    spec = record["spec"]
    assert type(spec["index"]) is int and 0 <= spec["index"] < 360
    assert spec["fit"] in FITS and spec["seat"] in (0, 1) and spec["phase"] in (0, 1, 2)
    assert spec["source_arm"] in ("candidate", "reference")
    keys, actions = record["union"]["keys"], record["union"]["actions"]
    assert len(keys) == len(actions) == len(set(keys)) and 1 <= len(keys) <= 10
    support = spec["canonical_support"]
    assert type(support) is int and support > 0
    assert set(record["candidates"]) == set(ORIGINAL)
    offered = {}
    for arm in ORIGINAL:
        candidate = record["candidates"][arm]
        indices = candidate["indices"]
        assert len(indices) == len(set(indices)) == min(support, 8 if arm == "teacher" else 4)
        assert all(type(j) is int and 0 <= j < len(keys) for j in indices)
        assert candidate["keys"] == [keys[j] for j in indices]
        assert candidate["actions"] == [actions[j] for j in indices]
        offered[arm] = list(indices)
    anchor = offered["anchor"]
    assert all(offered[arm][:3] == anchor[:3] for arm in ORIGINAL)
    assert offered["teacher"][:len(anchor)] == anchor
    assert set().union(*(set(ids) for ids in offered.values())) == set(range(len(keys)))
    for label, alternate in (("additive_broad", "broad"), ("additive_narrow", "narrow")):
        offered[label] = anchor + [j for j in offered[alternate] if j not in anchor]
        assert len(offered[label]) <= 5 and len(offered[label])-len(anchor) in (0, 1)
    lock, selection, assessment = record["selection_lock"], record["selection"], record["assessment"]
    assert lock["manifest"] == c.SOURCE_BINDING and lock["stage"] == "main" and lock["index"] == spec["index"]
    assert lock["selection_sha256"] == c.digest(selection) and lock["proofs_sha256"] == c.digest([])
    assert lock["value_scale"] == "terminal_wdl"
    assert selection["candidate_indices"] == list(range(len(keys)))
    assert type(lock["created_at_ns"]) is int and type(record["assessment_started_at_ns"]) is int
    assert 0 < lock["created_at_ns"] < record["assessment_started_at_ns"]
    for block, offset in ((selection, 0), (assessment, 256)):
        seeds = list(range(2181000000+spec["index"]*512+offset, 2181000000+spec["index"]*512+offset+128))
        assert block["world_seeds"] == seeds and [w["seed"] for w in block["worlds"]] == seeds
    totals, missing, unknown_cells = _column_sums(selection["worlds"], len(keys), spec["seat"])
    choices, certificates = {}, {}
    for arm in ARMS:
        choice, margins = _certified_choice(offered[arm], totals, missing)
        choices[arm] = choice
        certificates[arm] = {"indices": offered[arm], "choice": choice, "pairwise_lower_numerators": margins}
    assert lock["choices"] == {arm: choices[arm] for arm in ORIGINAL}
    old_selection = {"means": [total/128 if not missing[j] else None for j, total in enumerate(totals)],
                     "means_bounds": [[(total-missing[j])/128, (total+missing[j])/128] for j, total in enumerate(totals)],
                     "sums_wdl": totals, "unknown_counts": missing, "unresolved_cells": unknown_cells,
                     "choices": {arm: choices[arm] for arm in ORIGINAL},
                     "certificate": {"rule": "bounded_repetition_original_order_v1",
                                     "unknown_groups": [{"context_sha256": None, "cells": [cell]} for cell in unknown_cells],
                                     "arm_certificates": {arm: certificates[arm] for arm in ORIGINAL}}}
    compare(old_selection, lock["selection_statistics"], "source.selection_statistics")
    for label, alternate in (("additive_broad", "broad"), ("additive_narrow", "narrow")):
        assert choices[label] in (choices["anchor"], choices[alternate])
        if not missing[choices["anchor"]] and not missing[choices[alternate]]:
            expected = choices[alternate] if totals[choices[alternate]] > totals[choices["anchor"]] else choices["anchor"]
            assert choices[label] == expected, "Exact union ties must preserve anchor"
    assessed = sorted(set(lock["choices"].values()))
    assert lock["assessment_indices"] == assessment["candidate_indices"] == assessed
    assert len(assessed) <= 4 and all(choice in assessed for choice in choices.values())
    sums, unknown, _ = _column_sums(assessment["worlds"], len(assessed), spec["seat"])
    bounds = {index: [(128+sums[column]-unknown[column])/256, (128+sums[column]+unknown[column])/256]
              for column, index in enumerate(assessed)}
    arm_bounds = {arm: list(bounds[choices[arm]]) for arm in ARMS}
    deltas = {name: [0., 0.] if choices[left] == choices[right] else [
        arm_bounds[left][0]-arm_bounds[right][1], arm_bounds[left][1]-arm_bounds[right][0]]
              for name, left, right in CONTRASTS}
    compare(record["summary"]["choices"], lock["choices"], "source.summary.choices")
    compare(record["summary"]["arm_score_bounds"], {arm: arm_bounds[arm] for arm in ORIGINAL}, "source.summary.bounds")
    compare(record["summary"]["delta_bounds"], {name: deltas[name] for name, _, _ in CONTRASTS[4:]}, "source.summary.deltas")
    times = {}
    for arm in ("anchor", "broad", "narrow"):
        value = record["proposals"][arm]["timing"]["learned_extra_wall"]
        assert type(value) in (float, int) and math.isfinite(value) and value >= 0
        times[arm] = value
    costs = {"candidate_counts": {arm: len(offered[arm]) for arm in ARMS},
             "selection_continuations": {arm: 128*len(offered[arm]) for arm in ARMS},
             "extra_candidates": {arm: len(offered[arm])-len(anchor) for arm in ("additive_broad", "additive_narrow")},
             "extra_selection_continuations": {arm: 128*(len(offered[arm])-len(anchor)) for arm in ("additive_broad", "additive_narrow")},
             "changed_choices": {arm: choices[arm] != choices["anchor"] for arm in ARMS},
             "learned_extra_wall": times,
             "additional_learned_extra_wall": {"additive_broad": times["broad"], "additive_narrow": times["narrow"]}}
    return {**{key: spec[key] for key in META}, "choices": choices, "arm_score_bounds": arm_bounds,
            "delta_bounds": deltas, "costs": costs}


def verify_root(record, analysis_row):
    rebuilt = _row(record)
    compare(rebuilt, analysis_row, "row")
    return {"passed": True, "row": rebuilt}


def _integer_row(row):
    """Validate the compact receipt and return exact 1/256 contrast endpoints."""
    assert set(row) == set(META) | {"choices", "arm_score_bounds", "delta_bounds", "costs"}
    assert all(type(row[key]) is int for key in META if key != "source_arm")
    assert row["fit"] in FITS and row["phase"] in (0, 1, 2) and row["seat"] in (0, 1)
    assert row["source_arm"] in ("candidate", "reference") and row["pair"] >= 0
    choices, bounds, costs = row["choices"], row["arm_score_bounds"], row["costs"]
    assert set(choices) == set(bounds) == set(ARMS)
    assert set(row["delta_bounds"]) == {name for name, _, _ in CONTRASTS}
    integers = {}
    for arm in ARMS:
        assert type(choices[arm]) is int and 0 <= choices[arm] < 10
        endpoints = bounds[arm]
        assert len(endpoints) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in endpoints)
        assert 0 <= endpoints[0] <= endpoints[1] <= 1
        assert all(x*256 == int(x*256) for x in endpoints)
        integers[arm] = [int(x*256) for x in endpoints]
    by_action = {}
    for arm in ARMS:
        if choices[arm] in by_action:
            assert integers[arm] == by_action[choices[arm]], "Identical action has different assessment bounds"
        by_action[choices[arm]] = integers[arm]
    for additive, family in (("additive_broad", "broad"), ("additive_narrow", "narrow")):
        assert choices[additive] in (choices["anchor"], choices[family])
    deltas = []
    for name, left, right in CONTRASTS:
        expected = [0, 0] if choices[left] == choices[right] else [
            integers[left][0]-integers[right][1], integers[left][1]-integers[right][0]]
        compare([x/256 for x in expected], row["delta_bounds"][name], name)
        deltas.append(expected)
    expected_fields = {"candidate_counts", "selection_continuations", "extra_candidates",
                       "extra_selection_continuations", "changed_choices", "learned_extra_wall",
                       "additional_learned_extra_wall"}
    assert set(costs) == expected_fields
    for field in ("candidate_counts", "selection_continuations", "changed_choices"):
        assert set(costs[field]) == set(ARMS)
    counts = costs["candidate_counts"]
    assert counts["anchor"] == counts["broad"] == counts["narrow"]
    for arm in ARMS:
        assert type(counts[arm]) is int and 1 <= counts[arm] <= (8 if arm == "teacher" else 5 if arm.startswith("additive_") else 4)
        assert type(costs["selection_continuations"][arm]) is int and costs["selection_continuations"][arm] == 128*counts[arm]
        assert type(costs["changed_choices"][arm]) is bool and costs["changed_choices"][arm] == (choices[arm] != choices["anchor"])
    assert counts["teacher"] >= counts["anchor"]
    for field in ("extra_candidates", "extra_selection_continuations", "additional_learned_extra_wall"):
        assert set(costs[field]) == {"additive_broad", "additive_narrow"}
    assert set(costs["learned_extra_wall"]) == {"anchor", "broad", "narrow"}
    for elapsed in costs["learned_extra_wall"].values():
        assert type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
    for additive, family in (("additive_broad", "broad"), ("additive_narrow", "narrow")):
        extra = costs["extra_candidates"][additive]
        assert type(extra) is int and extra in (0, 1) and counts[additive] == counts["anchor"]+extra
        assert type(costs["extra_selection_continuations"][additive]) is int and costs["extra_selection_continuations"][additive] == 128*extra
        assert costs["additional_learned_extra_wall"][additive] == costs["learned_extra_wall"][family]
    return np.asarray(deltas, dtype=np.int64)


def independent_statistics(rows, budget=lambda: None):
    """Reproduce the shared bootstrap with draw multiplicities, never root resampling."""
    budget()
    assert len(rows) == 360 and {row["index"] for row in rows} == set(range(360))
    assert len({row["game_index"] for row in rows}) == 360
    assert c.CONFIG["fits"] == list(FITS) and c.CONFIG["roots"] == 360
    assert c.CONFIG["bootstrap_seed"] == 2203000000 and c.CONFIG["bootstrap_resamples"] == 50000
    assert c.CONFIG["bootstrap_chunk"] == 1000 and c.CONFIG["practical_gain"] == .005
    pairs = {}
    for row in rows:
        values = _integer_row(row)
        pairs.setdefault(row["pair"], []).append((row, values))
    assert len(pairs) == 180
    cells = [(fit, phase, arm) for fit in FITS for phase in range(3) for arm in ("candidate", "reference")]
    strata = {cell: [] for cell in cells}
    for pair in sorted(pairs):
        records = pairs[pair]
        assert len(records) == 2
        assert {row["seat"] for row, _ in records} == {0, 1}
        assert {row["game_index"] for row, _ in records} == {2*pair, 2*pair+1}
        memberships = {(row["fit"], row["phase"], row["source_arm"]) for row, _ in records}
        assert len(memberships) == 1
        cell = memberships.pop()
        assert cell in strata
        strata[cell].append(records[0][1]+records[1][1])
    assert Counter(map(len, strata.values())) == {10: 18}
    # Flatten endpoints so independent matrix multiplication reconstructs each
    # draw's integer sum; analysis uses indexed sample arrays instead.
    matrices = [np.stack(strata[cell]).reshape(10, 16) for cell in cells]
    point_integer = np.sum(np.concatenate(matrices), axis=0).reshape(8, 2)
    normalizer = 360*256
    random = np.random.default_rng(2203000000)
    sampled = np.empty((50000, 8, 2), dtype=np.int64)
    for start in range(0, 50000, 1000):
        budget()
        batch = np.zeros((1000, 16), dtype=np.int64)
        for matrix in matrices:
            labels = random.integers(0, 10, size=(1000, 10))
            counts = np.zeros((1000, 10), dtype=np.int64)
            np.add.at(counts, (np.repeat(np.arange(1000), 10), labels.ravel()), 1)
            assert np.all(counts.sum(axis=1) == 10)
            batch += counts @ matrix
        sampled[start:start+1000] = batch.reshape(1000, 8, 2)
    # The linear percentile is written out to avoid sharing the collector's
    # quantile operation, including interpolation when the rank is fractional.
    def percentile(values, fraction):
        ordered = np.sort(values, axis=0)
        position = (len(ordered)-1)*fraction
        lower = math.floor(position)
        return (ordered[lower]+(ordered[math.ceil(position)]-ordered[lower])*(position-lower))/normalizer
    lower = percentile(sampled[:, :, 0], .025)
    upper = percentile(sampled[:, :, 1], .975)
    contrasts = {}
    for column, (name, _, _) in enumerate(CONTRASTS):
        points = (point_integer[column]/normalizer).tolist()
        contrasts[name] = {"mean": points[0] if point_integer[column, 0] == point_integer[column, 1] else None,
                           "mean_bounds": points, "ci95": [float(lower[column]), float(upper[column])]}
    gates = {"practical_mean": bool(int(point_integer[0, 0])*200 >= normalizer),
             "anchor_superiority": bool(lower[0] > 0),
             "replacement_superiority": bool(lower[1] > 0),
             "attribution_superiority": bool(lower[2] > 0)}
    practical = gates["practical_mean"] and gates["anchor_superiority"] and gates["replacement_superiority"]
    attribution = practical and gates["attribution_superiority"]
    aggregate = {}
    for row in rows:
        for field, values in row["costs"].items():
            acc = aggregate.setdefault(field, {})
            for arm, value in values.items():
                acc[arm] = acc.get(arm, 0)+value
    aggregate["selection_cost_ratio_to_anchor"] = {
        arm: aggregate["selection_continuations"][arm]/aggregate["selection_continuations"]["anchor"] for arm in ARMS}
    aggregate["elapsed_time_claim"] = "Saved per-model scoring times only; no new deployed wall-time benchmark"
    result = {"roots": 360, "pairs": 180, "strata": 18, "bootstrap_samples": 50000,
              "development_only": True, "unknown_outcomes_imputed": False,
              "contrasts": contrasts, "gates": gates, "practical_pass": practical, "attribution_pass": attribution,
              "classification": "additive_broad_practical_and_attribution_signal" if attribution else
                                "additive_broad_practical_signal_only" if practical else "additive_broad_not_established",
              "aggregate_costs": aggregate}
    budget()
    return result


def verify_statistics(rows, reported, budget=lambda: None):
    rebuilt = independent_statistics(rows, budget)
    compare(rebuilt, reported, "statistics")
    return {"passed": True, "statistics": rebuilt}
