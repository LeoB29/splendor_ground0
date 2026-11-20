"""Whole-source-pair inference for fixed-final focused teacher fitting.

Training and validation source games are resampled independently within fit.
Every contrast uses the same draws; the four roots of a pair never separate.
"""
from collections import defaultdict
import math
import numpy as np
from . import common as c

SPLITS = ("train", "validation")
CONTRASTS = tuple((name, *name.split("_minus_")) for name in (
    "focused_outsider_minus_anchor_outsider", "teacher_outsider_minus_focused_outsider",
    "focused_supported_minus_anchor", "focused_supported_minus_broad_supported",
    "focused_outsider_minus_broad_outsider", "teacher_minus_focused_supported",
    "teacher_minus_anchor", "teacher_outsider_minus_anchor_outsider",
    "broad_supported_minus_anchor", "broad_supported_minus_narrow_supported",
    "focused_supported_minus_specialist_supported", "focused_outsider_minus_specialist_outsider",
    "specialist_supported_minus_anchor", "specialist_outsider_minus_anchor_outsider",
    "focused_supported_minus_lowrate_supported", "focused_outsider_minus_lowrate_outsider",
    "lowrate_supported_minus_anchor", "lowrate_outsider_minus_anchor_outsider",
    "focused_supported_minus_highrate_supported", "focused_outsider_minus_highrate_outsider",
    "highrate_supported_minus_anchor", "highrate_outsider_minus_anchor_outsider"))
NAMES = tuple(x[0] for x in CONTRASTS)
PRIMARY = ("focused_supported_minus_anchor", "focused_supported_minus_lowrate_supported")
CONTEXT = ("teacher_minus_anchor", "teacher_outsider_minus_anchor_outsider",
           "broad_supported_minus_anchor", "broad_supported_minus_narrow_supported")
ARMS = ("anchor", "teacher", "teacher_outsider", "original_fourth") + tuple(
    f"{arm}_{role}" for arm in ("anchor", "broad", "narrow")
    for role in ("direct", "outsider", "supported")) + tuple(
    f"{family}_{role}" for family in ("focused", "specialist", "lowrate", "highrate") for role in ("outsider", "supported"))
NEW_ARMS = {f"{family}_{role}" for family in ("focused", "specialist", "lowrate", "highrate") for role in ("outsider", "supported")}


def development_screen(statistics):
    """An investment screen on reused development outcomes, never promotion."""
    entry = statistics["groups"]["validation"]["contrasts"][PRIMARY[0]]
    control = statistics["groups"]["validation"]["contrasts"][PRIMARY[1]]
    passes = (entry["mean_bounds"][0] >= c.CONFIG["minimum_progress"] and entry["ci95"][0] > 0
              and control["ci95"][0] > 0)
    return {"primary_endpoint": PRIMARY[0], "split": "validation",
            "minimum_mean": c.CONFIG["minimum_progress"], "positive_lower_ci_required": True,
            "required_control_endpoint": PRIMARY[1], "control_ci95": control["ci95"],
            "control_lower_strictly_positive": control["ci95"][0] > 0,
            "highrate_attribution_is_additional_gate": False,
            "passed": passes, "classification": "fresh_confirmation_justified" if passes else "progress_not_established",
            "specialist_attribution_is_additional_gate": False,
            "fresh_full_legal_confirmation_required": True, "reused_development": True,
            "automatic_successor": False, "promotion": False}



def _units(bounds, scale=256):
    c.require(isinstance(bounds, list) and len(bounds) == 2
              and all(type(v) in (int, float) and math.isfinite(v) for v in bounds)
              and -1 <= bounds[0] <= bounds[1] <= 1, "Invalid paired bounds")
    values = np.rint(np.asarray(bounds)*scale).astype(np.int64)
    c.require(np.array_equal(values/scale, bounds), "Paired bounds use wrong score scale")
    return values


def _arrays(rows):
    c.require(len(rows) == 336 and [r["index"] for r in rows] == list(range(336)), "Complete ordered main panel required")
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair"]].append(row)
    c.require(len(groups) == 84, "Expected 84 independent source pairs")
    strata = defaultdict(list)
    for pair, records in sorted(groups.items()):
        c.require(len(records) == 4, "A main source pair needs four roots")
        first = records[0]
        cell = (first["split"], first["fit"])
        c.require(cell[0] in SPLITS and cell[1] in c.FITS, "Unknown source stratum")
        c.require(all((r["split"], r["fit"]) == cell for r in records), "A pair crosses split or fit")
        c.require({r["game_index"] for r in records} == {2*pair, 2*pair+1}, "Both source legs required")
        for game in (2*pair, 2*pair+1):
            leg = [r for r in records if r["game_index"] == game]
            c.require(len(leg) == 2 and len({r["phase"] for r in leg}) == 2, "A leg needs two distinct phase roots")
        c.require({r["phase"] for r in records if r["game_index"] == 2*pair}
                  == {r["phase"] for r in records if r["game_index"] == 2*pair+1}, "Phase balance differs between legs")
        values = np.zeros((len(NAMES), 2), dtype=np.int64)
        for row in records:
            c.require(set(row["choices"]) == set(row["arm_scores"]) == set(row["arm_score_bounds"]) == set(ARMS),
                      "Incomplete arm evidence")
            n = row["support"]
            c.require(type(n) is int and 1 <= n <= 8 and row["outsider_available"] == (n > 3), "Invalid support")
            for arm in ARMS:
                action, bounds = row["choices"][arm], row["arm_score_bounds"][arm]
                outside = arm.endswith("_outsider") or arm == "original_fourth"
                c.require((action is None) == (outside and n <= 3), "Invalid absent arm")
                if action is None:
                    c.require(bounds is None and row["arm_scores"][arm] is None, "Invented absent score")
                else:
                    c.require(type(action) is int and (3 if outside else 0) <= action < n, "Invalid chosen support")
                    _units(bounds)
                    c.require(0 <= bounds[0] <= bounds[1] <= 1
                              and row["arm_scores"][arm] == (bounds[0] if bounds[0] == bounds[1] else None),
                              "Invalid arm score")
            c.require(row["choices"]["anchor_outsider"] == row["choices"]["original_fourth"] == (3 if n > 3 else None)
                      and row["choices"]["anchor_supported"] == row["choices"]["anchor"], "Changed anchor identity")
            for family in ("anchor", "broad", "narrow", "focused", "specialist", "lowrate", "highrate"):
                allowed = list(range(min(n, 3))) + ([] if n <= 3 else [row["choices"][f"{family}_outsider"]])
                c.require(row["choices"][f"{family}_supported"] in allowed, "Supported decision escaped its arm")
            c.require(set(row["delta_bounds"]) == set(NAMES), "Missing primary arithmetic")
            for j, (name, left, right) in enumerate(CONTRASTS):
                expected_bounds = [0., 0.] if row["choices"][left] == row["choices"][right] else [
                    row["arm_score_bounds"][left][0]-row["arm_score_bounds"][right][1],
                    row["arm_score_bounds"][left][1]-row["arm_score_bounds"][right][0]]
                c.require(row["delta_bounds"][name] == expected_bounds, "Paired bounds differ from arm evidence")
                values[j] += _units(row["delta_bounds"][name])
        strata[cell].append(values)
    expected = [(s, f) for s in SPLITS for f in c.FITS]
    c.require(set(strata) == set(expected) and all(len(v) == 14 for v in strata.values()), "Six balanced 14-pair strata required")
    return [np.stack(strata[cell]) for cell in expected]


def _entry(bounds, interval):
    low, high = map(float, bounds)
    ci = list(map(float, interval))
    return {"mean": low if low == high else None, "mean_bounds": [low, high], "ci95": ci,
            "material_signal": bool(low >= .005 and ci[0] > 0),
            "material_target_excluded": bool(ci[1] < .005),
            "positive_signal": bool(ci[0] > 0), "negative_signal": bool(ci[1] < 0),
            "classification": "material_signal" if low >= .005 and ci[0] > 0 else "target_not_established"}


def summarize(rows):
    arrays = _arrays(rows)
    totals = np.stack([sum((a.sum(axis=0) for a in arrays[i:i+3]),
                          np.zeros((len(NAMES), 2), dtype=np.int64)) for i in (0, 3)])
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    samples = np.empty((c.CONFIG["bootstrap_samples"], 2, len(NAMES), 2), dtype=np.int64)
    for start in range(0, len(samples), c.CONFIG["bootstrap_chunk"]):
        count = min(c.CONFIG["bootstrap_chunk"], len(samples)-start)
        batch = np.zeros((count, 2, len(NAMES), 2), dtype=np.int64)
        for k, values in enumerate(arrays):
            batch[:, k//3] += values[rng.integers(0, 14, size=(count, 14))].sum(axis=1)
        samples[start:start+count] = batch
    denominator = 168*256
    groups = {}
    for si, split in enumerate(SPLITS):
        lo = np.quantile(samples[:, si, :, 0]/denominator, .025, axis=0, method="linear")
        hi = np.quantile(samples[:, si, :, 1]/denominator, .975, axis=0, method="linear")
        groups[split] = {"states": 168, "pairs": 42, "strata": 3,
                         "contrasts": {name: _entry(totals[si, j]/denominator, [lo[j], hi[j]])
                                       for j, name in enumerate(NAMES)}}
    difference = np.stack([samples[:, 1, :, 0]-samples[:, 0, :, 1],
                           samples[:, 1, :, 1]-samples[:, 0, :, 0]], axis=-1)
    point = np.stack([totals[1, :, 0]-totals[0, :, 1], totals[1, :, 1]-totals[0, :, 0]], axis=-1)
    lo = np.quantile(difference[:, :, 0]/denominator, .025, axis=0, method="linear")
    hi = np.quantile(difference[:, :, 1]/denominator, .975, axis=0, method="linear")
    groups["validation_minus_train"] = {"contrasts": {
        name: _entry(point[j]/denominator, [lo[j], hi[j]]) for j, name in enumerate(NAMES)}}
    return {"states": 336, "pairs": 84, "strata": 6,
            "bootstrap_samples": c.CONFIG["bootstrap_samples"], "bootstrap_seed": c.CONFIG["bootstrap_seed"],
            "resampling_unit": "whole_source_pair_four_roots", "groups": groups,
            "primary_contrasts": list(PRIMARY), "unknown_outcomes_imputed": False,
            "classification": "diagnostic_only", "promotion": False,
            "train_validation_difference_is_causal": False,
            "validation_previously_used_for_checkpoint_selection": True,
            "assessment_reused_development": True, "assessment_used_for_selection": False}
