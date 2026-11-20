"""Paired saved-trajectory diagnostics with one prespecified simultaneous family.

Checkpoints are correlated measurements of the same 84 source pairs. Shared
draws retain that dependence; neither checkpoint nor model selection is done.
"""
from collections import defaultdict
import math
import numpy as np
from . import common as c
from tools.higher_teacher_rate_128 import statistics as inherited

CHECKPOINTS, FITS = c.CHECKPOINTS, c.FITS
SPLITS = ("train", "validation")
NAMES = inherited.NAMES + ("focused_outsider_minus_finalrate_outsider",
                           "focused_supported_minus_finalrate_supported")
CONTRASTS = tuple((name, *name.split("_minus_")) for name in NAMES)
PRIMARY = tuple(c.CONFIG["primary_contrasts"])
ANCHOR_CONTRASTS = ("focused_outsider_minus_anchor_outsider", "focused_supported_minus_anchor")
FINAL_CONTRASTS = ("focused_outsider_minus_finalrate_outsider", "focused_supported_minus_finalrate_supported")
ARMS = inherited.ARMS + ("finalrate_outsider", "finalrate_supported")
FIXED_ARMS = tuple(name for name in ARMS if not name.startswith("focused_"))
IDENTITY_FIELDS = ("index", "pair", "game_index", "fit", "seat", "phase", "split", "privacy", "support", "outsider_available", "unknown_counts")
DENOMINATOR = 168*256


def _boundary(boundary):
    if boundary is not None:
        boundary()


def _metadata():
    return {"checkpoints": list(CHECKPOINTS), "states": 336, "pairs": 84, "strata": 6,
        "bootstrap_samples": c.CONFIG["bootstrap_samples"], "bootstrap_seed": c.CONFIG["bootstrap_seed"],
        "bootstrap_chunk": c.CONFIG["bootstrap_chunk"], "resampling_unit": "whole_source_pair_four_roots",
        "shared_draws_across_checkpoints_and_endpoints": True, "train_validation_draws_independent": True,
        "primary_contrasts": list(PRIMARY), "classification": "diagnostic_only",
        "pointwise_intervals_are_diagnostic": True, "per_fit_means_are_descriptive": True,
        "train_validation_difference_is_causal": False, "inference_conditions_on_fixed_fits": True,
        "assessment_reused_development": True, "validation_previously_used_for_checkpoint_selection": True,
        "assessment_used_for_selection": False, "checkpoint_selection": False,
        "best_fit_selection": False, "promotion": False, "unknown_outcomes_imputed": False}


def _units(bounds, *, absolute=False):
    c.require(isinstance(bounds, list) and len(bounds) == 2 and
              all(type(x) in (int, float) and math.isfinite(x) for x in bounds), "Invalid score bounds")
    c.require((0 if absolute else -1) <= bounds[0] <= bounds[1] <= 1, "Invalid score envelope")
    integers = np.rint(np.array(bounds)*256).astype(np.int64)
    c.require(np.array_equal(integers/256, bounds), "Scores must use the integer/256 grid")
    return integers


def _checked_row(row):
    n = row["support"]
    c.require(type(n) is int and 1 <= n <= 8 and row["outsider_available"] == (n > 3), "Invalid candidate support")
    c.require(set(row["choices"]) == set(row["arm_score_bounds"]) == set(row["arm_scores"]) == set(ARMS), "Incomplete arms")
    missing = row["unknown_counts"]
    c.require(len(missing) == n and all(type(x) is int and 0 <= x <= 128 for x in missing), "Invalid unknown counts")
    by_choice = {}
    for arm in ARMS:
        choice, bounds, score = row["choices"][arm], row["arm_score_bounds"][arm], row["arm_scores"][arm]
        outsider = arm.endswith("_outsider") or arm == "original_fourth"
        if choice is None:
            c.require(n <= 3 and outsider and bounds is None and score is None, "Unlawful absent action")
        else:
            c.require(type(choice) is int and 0 <= choice < n and (not outsider or choice >= 3), "Choice outside support")
            values = _units(bounds, absolute=True)
            c.require(values[1]-values[0] == 2*missing[choice], "Unknown envelope differs from recorded count")
            c.require(score == (bounds[0] if bounds[0] == bounds[1] else None), "Point score disagrees with bounds")
            c.require(choice not in by_choice or by_choice[choice] == bounds, "Shared action scores differ")
            by_choice[choice] = bounds
    c.require(row["choices"]["original_fourth"] == (3 if n > 3 else None)
              and row["choices"]["anchor_outsider"] == row["choices"]["original_fourth"]
              and row["choices"]["anchor_supported"] == row["choices"]["anchor"], "Anchor identity changed")
    for family in ("anchor", "broad", "narrow", "focused", "specialist", "lowrate", "finalrate"):
        selected = row["choices"][family+"_supported"]
        c.require(selected < min(3, n) or selected == row["choices"][family+"_outsider"], "Supported choice not offered")
    c.require(set(row["delta_bounds"]) == set(NAMES), "Missing contrasts")
    output = []
    for name, left, right in CONTRASTS:
        if row["choices"][left] == row["choices"][right]:
            expected = [0., 0.]
        else:
            a, b = row["arm_score_bounds"][left], row["arm_score_bounds"][right]
            c.require(a is not None and b is not None, "Asymmetric absent contrast")
            expected = [a[0]-b[1], a[1]-b[0]]
        c.require(row["delta_bounds"][name] == expected, "Raw arm/contrast arithmetic differs")
        output.append(_units(row["delta_bounds"][name]))
    return np.stack(output)


def _arrays(curves, boundary=None):
    c.require(set(curves) == {str(update) for update in CHECKPOINTS}, "All fixed checkpoints required")
    initial = curves["0"]
    data = []
    for update in CHECKPOINTS:
        _boundary(boundary)
        rows = curves[str(update)]
        c.require(len(rows) == 336 and [r["index"] for r in rows] == list(range(336)), "Complete ordered panel required")
        matrix = []
        for row, oldrow in zip(rows, initial, strict=True):
            c.require(all(row[k] == oldrow[k] for k in IDENTITY_FIELDS), "Checkpoint population changed")
            c.require(all(row[k][name] == oldrow[k][name] for name in FIXED_ARMS
                          for k in ("choices", "arm_scores", "arm_score_bounds")), "Fixed control changed across checkpoints")
            if update in (0, CHECKPOINTS[-1]):
                reference = "anchor" if update == 0 else "finalrate"
                c.require(all(row[key]["focused_"+role] == row[key][reference+"_"+role]
                              for role in ("outsider", "supported")
                              for key in ("choices", "arm_scores", "arm_score_bounds")),
                          "Initial or final checkpoint action identity changed")
            matrix.append(_checked_row(row))
        data.append(np.stack(matrix))
    data = np.stack(data)
    c.require(np.all(data[0, :, [NAMES.index(name) for name in ANCHOR_CONTRASTS], :] == 0), "Initial focused model differs from anchor")
    c.require(np.all(data[-1, :, [NAMES.index(name) for name in FINAL_CONTRASTS], :] == 0), "Final focused model differs from fixed final control")
    pairs = defaultdict(list)
    for row in initial:
        c.require(type(row["pair"]) is int and row["pair"] >= 0, "Invalid source pair")
        c.require(row["fit"] in FITS and row["split"] in SPLITS and row["phase"] in (0, 1, 2)
                  and row["seat"] in (0, 1) and row["game_index"] == 2*row["pair"]+row["seat"], "Invalid source metadata")
        pairs[row["pair"]].append(row)
    c.require(len(pairs) == 84, "Expected 84 whole pairs")
    strata = defaultdict(list)
    for pair, rows in sorted(pairs.items()):
        cell = (rows[0]["split"], rows[0]["fit"])
        c.require(len(rows) == 4 and all((r["split"], r["fit"]) == cell for r in rows), "Source pair is incomplete or crosses stratum")
        legs = [[r for r in rows if r["seat"] == seat] for seat in (0, 1)]
        c.require(all(len(leg) == 2 and len({r["phase"] for r in leg}) == 2 for leg in legs)
                  and {r["phase"] for r in legs[0]} == {r["phase"] for r in legs[1]}, "Paired phase coverage changed")
        strata[cell].append(data[:, [r["index"] for r in rows]].sum(axis=1))
    order = [(split, fit) for split in SPLITS for fit in FITS]
    c.require(set(strata) == set(order) and all(len(v) == 14 for v in strata.values()), "Six 14-pair strata required")
    return [np.stack(strata[cell]) for cell in order]


def _bootstrap(arrays, boundary=None):
    samples = c.CONFIG["bootstrap_samples"]
    chunk = c.CONFIG["bootstrap_chunk"]
    c.require(type(samples) is int and samples > 0 and type(chunk) is int and chunk > 0, "Invalid bootstrap budget")
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    boot = np.zeros((samples, 2, len(CHECKPOINTS), len(NAMES), 2), dtype=np.int64)
    for start in range(0, samples, chunk):
        _boundary(boundary)
        count = min(chunk, samples-start)
        for k, array in enumerate(arrays):
            draws = rng.integers(0, 14, size=(count, 14))
            boot[start:start+count, k//3] += array[draws].sum(axis=1)
    return boot


def _entry(bounds, interval=None):
    low, high = map(float, bounds)
    result = {"mean": low if low == high else None, "mean_bounds": [low, high]}
    if interval is not None:
        result["pointwise_ci95"] = list(map(float, interval))
    return result


def _standard_errors(arrays):
    # Each pair mean contains four roots; three equal 14-pair strata determine
    # the validation mean. Fix these original-sample scales for every draw.
    pair_means = np.stack(arrays[3:])/(4*256)
    variances = np.var(pair_means, axis=1, ddof=1)
    return np.sqrt(variances.sum(axis=0)/(9*14))


def _simultaneous_critical_value(samples, totals, standard_errors):
    columns = [NAMES.index(name) for name in PRIMARY]
    # Endpoints stay separate: unresolved outcomes receive no midpoint imputation.
    deviations = np.abs(samples[:, 1, 1:][:, :, columns, :]-totals[1, 1:][:, columns, :])
    scales = standard_errors[1:][:, columns, :]
    c.require(np.all(np.isfinite(scales)) and np.all(scales >= 0), "Invalid fixed standard errors")
    zero = scales == 0
    c.require(np.all(deviations[:, zero] == 0), "Zero-SE endpoint has nonzero bootstrap deviation")
    standardized = np.divide(deviations/DENOMINATOR, scales,
                             out=np.zeros(deviations.shape, dtype=np.float64), where=scales > 0)
    maxima = standardized.max(axis=(1, 2, 3))
    return float(np.quantile(maxima, .95, method="linear"))


def _simultaneous(totals, standard_errors, critical_value):
    family = [{"update": update, "contrast": name} for update in CHECKPOINTS[1:] for name in PRIMARY]
    groups = {}
    for k, update in enumerate(CHECKPOINTS):
        if update == 0:
            continue
        groups[str(update)] = {}
        for name in PRIMARY:
            column = NAMES.index(name)
            point = totals[1, k, column]/DENOMINATOR
            scales = standard_errors[k, column]
            groups[str(update)][name] = {**_entry(point), "standard_error_bounds": list(map(float, scales)),
                "ci95": [float(point[0]-critical_value*scales[0]), float(point[1]+critical_value*scales[1])]}
    return {"family": family, "family_size": len(family), "initial_checkpoint_excluded": True,
        "method": "bootstrap_max_absolute_centered_deviation_fixed_se_standardized", "confidence": .95,
        "centered_bound_endpoints_both_in_maximum": True, "critical_value": critical_value,
        "standard_errors": "fixed_original_stratified_whole_pair_sample_variance_ddof1",
        "zero_standard_error": "require_exact_zero_bootstrap_deviation_then_contribute_zero",
        "groups": groups, "development_only": True, "applies_to_training_or_per_fit": False}


def summarize(curves, boundary=None):
    arrays = _arrays(curves, boundary)
    totals = np.stack([sum((a.sum(axis=0) for a in arrays[i:i+3]), np.zeros_like(arrays[0][0])) for i in (0, 3)])
    boot = _bootstrap(arrays, boundary)
    result = _metadata()
    result["curves"] = {}
    for k, update in enumerate(CHECKPOINTS):
        _boundary(boundary)
        groups = {}
        for si, split in enumerate(SPLITS):
            lo = np.quantile(boot[:, si, k, :, 0]/DENOMINATOR, .025, axis=0, method="linear")
            hi = np.quantile(boot[:, si, k, :, 1]/DENOMINATOR, .975, axis=0, method="linear")
            groups[split] = {"states": 168, "pairs": 42, "strata": 3,
                "contrasts": {name: _entry(totals[si, k, j]/DENOMINATOR, [lo[j], hi[j]]) for j, name in enumerate(NAMES)}}
        difference = np.stack([boot[:, 1, k, :, 0]-boot[:, 0, k, :, 1], boot[:, 1, k, :, 1]-boot[:, 0, k, :, 0]], axis=-1)
        point = np.stack([totals[1, k, :, 0]-totals[0, k, :, 1], totals[1, k, :, 1]-totals[0, k, :, 0]], axis=-1)
        lower = np.quantile(difference[:, :, 0]/DENOMINATOR, .025, axis=0, method="linear")
        upper = np.quantile(difference[:, :, 1]/DENOMINATOR, .975, axis=0, method="linear")
        groups["validation_minus_train"] = {"contrasts": {
            name: _entry(point[j]/DENOMINATOR, [lower[j], upper[j]]) for j, name in enumerate(NAMES)}}
        by_fit = {}
        for fi, fit in enumerate(FITS):
            by_fit[str(fit)] = {}
            for si, split in enumerate(SPLITS):
                means = arrays[si*3+fi][:, k].sum(axis=0)/(56*256)
                by_fit[str(fit)][split] = {"states": 56, "pairs": 14,
                    "contrasts": {name: _entry(means[j]) for j, name in enumerate(NAMES)}}
        result["curves"][str(update)] = {"groups": groups, "per_fit": by_fit}
    standard_errors = _standard_errors(arrays)
    result["simultaneous_validation"] = _simultaneous(
        totals, standard_errors, _simultaneous_critical_value(boot, totals, standard_errors))
    _boundary(boundary)
    return result


def independent_summarize(curves, boundary=None):
    """Explicit arm arithmetic, separate grouping and histogram-multiply bootstrap."""
    c.require(isinstance(curves, dict) and sorted(map(int, curves)) == list(CHECKPOINTS)
              and set(curves) == set(map(str, CHECKPOINTS)), "Independent checkpoint inventory differs")
    first_rows = curves["0"]
    c.require(len(first_rows) == 336, "Independent panel count differs")
    pair_ids = sorted(set(row["pair"] for row in first_rows))
    c.require(len(pair_ids) == 84 and all(type(p) is int and p >= 0 for p in pair_ids), "Independent pairs differ")
    cells, matrices = {}, {}
    width = len(CHECKPOINTS)*len(CONTRASTS)*2
    for pair in pair_ids:
        records = [row for row in first_rows if row["pair"] == pair]
        c.require(len(records) == 4 and {r["seat"] for r in records} == {0, 1}, "Independent pair incomplete")
        split, fit = records[0]["split"], records[0]["fit"]
        c.require(split in ("train", "validation") and fit in FITS and
                  all((row["split"], row["fit"]) == (split, fit) for row in records), "Independent stratum invalid")
        phases = []
        for seat in (0, 1):
            leg = [row for row in records if row["seat"] == seat]
            c.require(len(leg) == 2 and all(row["game_index"] == 2*pair+seat and row["phase"] in (0, 1, 2) for row in leg),
                      "Independent source legs invalid")
            phase_set = set(row["phase"] for row in leg)
            c.require(len(phase_set) == 2, "Independent leg phase repeated")
            phases.append(phase_set)
        c.require(phases[0] == phases[1], "Independent phase balance differs")
        cells[pair], matrices[pair] = (split, fit), np.zeros(width, dtype=np.int64)
    for ck, update in enumerate(CHECKPOINTS):
        _boundary(boundary)
        rows = curves[str(update)]
        c.require(len(rows) == 336 and [row["index"] for row in rows] == list(range(336)), "Independent ordered panel differs")
        for row, original in zip(rows, first_rows, strict=True):
            for key in ("index", "pair", "game_index", "fit", "seat", "phase", "split", "privacy", "support", "outsider_available", "unknown_counts"):
                c.require(row[key] == original[key], "Independent checkpoint identity changed")
            n = row["support"]
            c.require(type(n) is int and 1 <= n <= 8 and row["outsider_available"] == (n >= 4), "Independent support invalid")
            c.require(len(row["unknown_counts"]) == n and all(type(x) is int and x in range(129) for x in row["unknown_counts"]),
                      "Independent unknown accounting invalid")
            c.require(set(row["choices"]) == set(row["arm_score_bounds"]) == set(row["arm_scores"]) == set(ARMS), "Independent arms missing")
            same = {}
            for arm in ARMS:
                choice, bound = row["choices"][arm], row["arm_score_bounds"][arm]
                outsider = arm.endswith("_outsider") or arm == "original_fourth"
                if choice is None:
                    c.require(n < 4 and outsider and bound is None and row["arm_scores"][arm] is None, "Independent absent choice invalid")
                else:
                    c.require(type(choice) is int and choice in range(n) and (not outsider or choice > 2), "Independent choice invalid")
                    c.require(isinstance(bound, list) and len(bound) == 2 and all(type(x) in (int, float) and math.isfinite(x)
                              and 256*x == int(256*x) for x in bound) and 0 <= bound[0] <= bound[1] <= 1,
                              "Independent absolute score invalid")
                    c.require(int((bound[1]-bound[0])*256) == 2*row["unknown_counts"][choice], "Independent unknown envelope invalid")
                    c.require(row["arm_scores"][arm] == (bound[0] if bound[0] == bound[1] else None), "Independent point score invalid")
                    c.require(choice not in same or same[choice] == bound, "Independent shared choice differs")
                    same[choice] = bound
                if not arm.startswith("focused_"):
                    c.require(all(row[key][arm] == original[key][arm] for key in ("choices", "arm_score_bounds", "arm_scores")),
                              "Independent fixed control changed")
            c.require(row["choices"]["original_fourth"] == (3 if n >= 4 else None)
                      and row["choices"]["anchor_outsider"] == row["choices"]["original_fourth"]
                      and row["choices"]["anchor_supported"] == row["choices"]["anchor"], "Independent anchor identities differ")
            for arm in ("anchor", "broad", "narrow", "focused", "specialist", "lowrate", "finalrate"):
                allowed = list(range(min(3, n)))
                if n >= 4:
                    allowed.append(row["choices"][arm+"_outsider"])
                c.require(row["choices"][arm+"_supported"] in allowed, "Independent support membership differs")
            if update == 0:
                for role in ("outsider", "supported"):
                    c.require(all(row[key]["focused_"+role] == row[key]["anchor_"+role]
                                  for key in ("choices", "arm_score_bounds", "arm_scores")),
                              "Independent initial action identity differs")
            if update == CHECKPOINTS[-1]:
                for role in ("outsider", "supported"):
                    c.require(all(row[key]["focused_"+role] == row[key]["finalrate_"+role]
                                  for key in ("choices", "arm_score_bounds", "arm_scores")),
                              "Independent final action identity differs")
            c.require(set(row["delta_bounds"]) == set(NAMES), "Independent contrasts missing")
            for j, (name, left, right) in enumerate(CONTRASTS):
                if row["choices"][left] == row["choices"][right]:
                    lo = hi = 0
                else:
                    a, b = row["arm_score_bounds"][left], row["arm_score_bounds"][right]
                    c.require(a is not None and b is not None, "Independent asymmetric absence")
                    lo, hi = int(a[0]*256)-int(b[1]*256), int(a[1]*256)-int(b[0]*256)
                c.require(row["delta_bounds"][name] == [lo/256, hi/256] and -256 <= lo <= hi <= 256,
                          "Independent raw difference disagrees")
                if update == 0 and name in ANCHOR_CONTRASTS:
                    c.require(lo == hi == 0, "Independent initial model differs")
                if update == CHECKPOINTS[-1] and name in FINAL_CONTRASTS:
                    c.require(lo == hi == 0, "Independent final model differs")
                offset = 2*(ck*len(NAMES)+j)
                matrices[row["pair"]][offset:offset+2] += [lo, hi]
    strata = []
    for split in ("train", "validation"):
        for fit in FITS:
            members = [pair for pair in pair_ids if cells[pair] == (split, fit)]
            c.require(len(members) == 14, "Independent stratum not 14 whole pairs")
            strata.append(np.stack([matrices[pair] for pair in members]))
    point = np.stack([sum((matrix.sum(axis=0) for matrix in strata[si:si+3]), np.zeros(width, dtype=np.int64)) for si in (0, 3)])
    count, chunk = c.CONFIG["bootstrap_samples"], c.CONFIG["bootstrap_chunk"]
    c.require(type(count) is int and count > 0 and type(chunk) is int and chunk > 0, "Independent bootstrap size invalid")
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    samples = np.zeros((count, 2, width), dtype=np.int64)
    for start in range(0, count, chunk):
        _boundary(boundary)
        size = min(chunk, count-start)
        for cell, matrix in enumerate(strata):
            draws = rng.integers(0, 14, size=(size, 14))
            histogram = np.stack([np.bincount(draw, minlength=14) for draw in draws])
            samples[start:start+size, cell//3] += histogram @ matrix

    def quantile(column, fraction):
        ordered = sorted(int(x) for x in column)
        place = (len(ordered)-1)*fraction
        low, high = math.floor(place), math.ceil(place)
        return (ordered[low]+(ordered[high]-ordered[low])*(place-low))/(168*256)

    def entry(lo, hi, left=None, right=None, denominator=168*256):
        bounds = [int(lo)/denominator, int(hi)/denominator]
        item = {"mean": bounds[0] if lo == hi else None, "mean_bounds": bounds}
        if left is not None:
            item["pointwise_ci95"] = [quantile(left, .025), quantile(right, .975)]
        return item

    result = _metadata()
    result["curves"] = {}
    for k, update in enumerate(CHECKPOINTS):
        _boundary(boundary)
        groups, per_fit = {}, {}
        for si, split in enumerate(("train", "validation")):
            contrasts = {}
            for j, name in enumerate(NAMES):
                low = 2*(k*len(NAMES)+j)
                contrasts[name] = entry(point[si, low], point[si, low+1], samples[:, si, low], samples[:, si, low+1])
            groups[split] = {"states": 168, "pairs": 42, "strata": 3, "contrasts": contrasts}
        contrasts = {}
        for j, name in enumerate(NAMES):
            low = 2*(k*len(NAMES)+j)
            contrasts[name] = entry(point[1, low]-point[0, low+1], point[1, low+1]-point[0, low],
                samples[:, 1, low]-samples[:, 0, low+1], samples[:, 1, low+1]-samples[:, 0, low])
        groups["validation_minus_train"] = {"contrasts": contrasts}
        for fi, fit in enumerate(FITS):
            per_fit[str(fit)] = {}
            for si, split in enumerate(("train", "validation")):
                totals = strata[si*3+fi].sum(axis=0)
                per_fit[str(fit)][split] = {"states": 56, "pairs": 14, "contrasts": {
                    name: entry(totals[2*(k*len(NAMES)+j)], totals[2*(k*len(NAMES)+j)+1], denominator=56*256)
                    for j, name in enumerate(NAMES)}}
        result["curves"][str(update)] = {"groups": groups, "per_fit": per_fit}
    # Algebraically independent unbiased variances, computed with exact integer
    # pair sums before the final division/square root. No bootstrap SE estimate.
    standard_errors = np.zeros(width, dtype=np.float64)
    for column in range(width):
        numerator = 0
        for matrix in strata[3:]:
            values = [int(v) for v in matrix[:, column]]
            numerator += 14*sum(v*v for v in values)-sum(values)**2
        c.require(numerator >= 0, "Independent variance numerator negative")
        standard_errors[column] = math.sqrt(numerator/(14*13*(4*256)**2*9*14))
    scan_columns = [2*(k*len(NAMES)+NAMES.index(name))+endpoint
                    for k in range(1, len(CHECKPOINTS)) for name in PRIMARY for endpoint in (0, 1)]
    maxima = np.zeros(count, dtype=np.float64)
    for column in scan_columns:
        differences = np.abs(samples[:, 1, column]-point[1, column])
        if standard_errors[column] == 0:
            c.require(np.all(differences == 0), "Independent zero-SE endpoint varies in bootstrap")
        else:
            maxima = np.maximum(maxima, differences/((168*256)*standard_errors[column]))
    ordered = sorted(float(value) for value in maxima)
    location = (len(ordered)-1)*.95
    below, above = math.floor(location), math.ceil(location)
    critical_value = ordered[below]+(ordered[above]-ordered[below])*(location-below)
    family, bands = [], {}
    for k in range(1, len(CHECKPOINTS)):
        update = CHECKPOINTS[k]
        bands[str(update)] = {}
        for name in PRIMARY:
            family.append({"update": update, "contrast": name})
            column = 2*(k*len(NAMES)+NAMES.index(name))
            value = entry(point[1, column], point[1, column+1])
            scales = [float(standard_errors[column]), float(standard_errors[column+1])]
            value["standard_error_bounds"] = scales
            value["ci95"] = [value["mean_bounds"][0]-critical_value*scales[0], value["mean_bounds"][1]+critical_value*scales[1]]
            bands[str(update)][name] = value
    result["simultaneous_validation"] = {"family": family, "family_size": len(family), "initial_checkpoint_excluded": True,
        "method": "bootstrap_max_absolute_centered_deviation_fixed_se_standardized", "confidence": .95,
        "centered_bound_endpoints_both_in_maximum": True, "critical_value": critical_value,
        "standard_errors": "fixed_original_stratified_whole_pair_sample_variance_ddof1",
        "zero_standard_error": "require_exact_zero_bootstrap_deviation_then_contribute_zero", "groups": bands,
        "development_only": True, "applies_to_training_or_per_fit": False}
    _boundary(boundary)
    return result
