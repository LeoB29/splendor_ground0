"""Whole-source-pair inference for the frozen teacher signal diagnostic.

Training and validation source games are resampled independently within fit.
Every contrast uses the same draws; the four roots of a pair never separate.
"""
from collections import defaultdict
import math
import numpy as np
from . import common as c

SPLITS = ("train", "validation")
PROPOSERS = ("anchor", "broad", "narrow")
ARMS = ("anchor", "teacher", "teacher_outsider", "original_fourth") + tuple(
    f"{p}_{kind}" for p in PROPOSERS for kind in ("direct", "outsider", "supported"))
CONTRASTS = (
    ("teacher_minus_anchor", "teacher", "anchor"),
    ("teacher_outsider_minus_anchor_outsider", "teacher_outsider", "anchor_outsider"),
    ("broad_direct_minus_anchor_direct", "broad_direct", "anchor_direct"),
    ("narrow_direct_minus_anchor_direct", "narrow_direct", "anchor_direct"),
    ("broad_direct_minus_narrow_direct", "broad_direct", "narrow_direct"),
    ("teacher_minus_anchor_direct", "teacher", "anchor_direct"),
    ("teacher_minus_broad_direct", "teacher", "broad_direct"),
    ("teacher_minus_narrow_direct", "teacher", "narrow_direct"),
    ("broad_outsider_minus_anchor_outsider", "broad_outsider", "anchor_outsider"),
    ("narrow_outsider_minus_anchor_outsider", "narrow_outsider", "anchor_outsider"),
    ("broad_outsider_minus_narrow_outsider", "broad_outsider", "narrow_outsider"),
    ("teacher_outsider_minus_broad_outsider", "teacher_outsider", "broad_outsider"),
    ("teacher_outsider_minus_narrow_outsider", "teacher_outsider", "narrow_outsider"),
    ("broad_supported_minus_anchor", "broad_supported", "anchor"),
    ("narrow_supported_minus_anchor", "narrow_supported", "anchor"),
    ("broad_supported_minus_narrow_supported", "broad_supported", "narrow_supported"),
    ("teacher_minus_broad_supported", "teacher", "broad_supported"),
    ("teacher_minus_narrow_supported", "teacher", "narrow_supported"),
)
NAMES = tuple(x[0] for x in CONTRASTS)
PRIMARY = ("teacher_minus_anchor", "broad_supported_minus_anchor",
           "broad_supported_minus_narrow_supported", "teacher_outsider_minus_anchor_outsider")


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
            c.require(set(row["delta_bounds"]) == set(NAMES), "Missing primary arithmetic")
            for j, name in enumerate(NAMES):
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
        repeatability = {}
        selected_rows = [row for row in rows if row["split"] == split]
        for kind in ("teacher", "outsider"):
            available = [row for row in selected_rows if kind == "teacher" or row["outsider_available"]]
            values = [row["fresh_selection"][f"{kind}_agreement"] for row in available]
            c.require(all(value is None or type(value) is bool for value in values), "Invalid fresh preference certificate")
            agreed, disagreed = sum(value is True for value in values), sum(value is False for value in values)
            repeatability[kind] = {"states": 168, "available": len(available), "absent": 168-len(available),
                                  "agreed": agreed, "disagreed": disagreed,
                                  "unresolved": sum(value is None for value in values),
                                  "agreement_rate_among_certified": agreed/(agreed+disagreed) if agreed+disagreed else None,
                                  "diagnostic_only": True}
        groups[split]["repeatability"] = repeatability
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
            "validation_previously_used_for_checkpoint_selection": True}
