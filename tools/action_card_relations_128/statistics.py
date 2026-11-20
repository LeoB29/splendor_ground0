"""Independent paired integer bootstrap, preserving unresolved outcome bounds."""
from collections import Counter
import numpy as np

from .common import CONFIG, FITS, ARMS, CONTRASTS, require


def _numerator(value, lower, upper):
    require(type(value) in (int, float) and np.isfinite(value)
            and lower <= value <= upper, "Invalid bounded score")
    number = int(round(value * 256))
    require(abs(number / 256 - value) < 1e-12, "Score is not a 128-world score")
    return number


def summarize(rows):
    require(len(rows) == 360 and sorted(r["index"] for r in rows) == list(range(360)),
            "Complete fixed 360-position panel required")
    require(len({r["game_index"] for r in rows}) == 360,
            "Repeated source games are not independent observations")
    cells = [(fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in (0, 1, 2)]
    require(Counter((r["fit"], r["seat"], r["phase"]) for r in rows)
            == Counter({cell: 20 for cell in cells}), "Fixed evaluation strata changed")
    matrices = []
    for cell in cells:
        subset = sorted((r for r in rows if (r["fit"], r["seat"], r["phase"]) == cell),
                        key=lambda r: r["index"])
        numbers = []
        for row in subset:
            require(set(row["arm_score_bounds"]) == set(row["choices"]) == set(ARMS),
                    "Incomplete assessed arms")
            require(set(row["delta_bounds"]) == set(CONTRASTS), "Incomplete contrasts")
            bounds = {}
            for arm in ARMS:
                pair = row["arm_score_bounds"][arm]
                require(len(pair) == 2, "Arm bounds require two endpoints")
                bounds[arm] = [_numerator(x, 0, 1) for x in pair]
                require(bounds[arm][0] <= bounds[arm][1], "Reversed arm bounds")
            for left in ARMS:
                for right in ARMS:
                    if row["choices"][left] == row["choices"][right]:
                        require(bounds[left] == bounds[right], "One assessed action has inconsistent bounds")
            vector = []
            for contrast in CONTRASTS:
                left, right = contrast.split("_minus_")
                pair = row["delta_bounds"][contrast]
                require(len(pair) == 2, "Contrast bounds require two endpoints")
                encoded = [_numerator(x, -1, 1) for x in pair]
                expected = ([0, 0] if row["choices"][left] == row["choices"][right]
                            else [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]])
                require(encoded == expected, "Contrast violates shared-choice coupling or bounds")
                vector.extend(encoded)
            numbers.append(vector)
        matrices.append(np.asarray(numbers, dtype=np.int64))
    total = sum((m.sum(axis=0) for m in matrices), np.zeros(2*len(CONTRASTS), dtype=np.int64))
    means = total.reshape(len(CONTRASTS), 2) / (360*256)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    draws = np.empty((CONFIG["bootstrap_samples"], 2*len(CONTRASTS)), dtype=np.int64)
    for start in range(0, len(draws), CONFIG["bootstrap_chunk"]):
        count = min(CONFIG["bootstrap_chunk"], len(draws)-start)
        accumulator = np.zeros((count, 2*len(CONTRASTS)), dtype=np.int64)
        for matrix in matrices:
            accumulator += matrix[rng.integers(0, 20, size=(count, 20))].sum(axis=1)
        draws[start:start+count] = accumulator
    quantiles = np.quantile(draws / (360*256), [.025, .975], axis=0, method="linear")
    contrasts = {}
    for j, name in enumerate(CONTRASTS):
        limits = means[j].tolist()
        contrasts[name] = {"mean_score_difference": limits[0] if limits[0] == limits[1] else None,
                           "mean_bounds": limits,
                           "ci95": [float(quantiles[0, 2*j]), float(quantiles[1, 2*j+1])]}
    anchor = contrasts["relations_minus_anchor"]
    flags = {"anchor_practical_gain": bool(anchor["mean_bounds"][0] >= CONFIG["minimum_progress"]),
             "anchor_superiority": bool(anchor["ci95"][0] > 0),
             "raw_superiority": bool(contrasts["relations_minus_raw"]["ci95"][0] > 0),
             "relations_attribution": bool(contrasts["relations_minus_raw"]["ci95"][0] > 0)}
    flags["gate_pass"] = all(flags[k] for k in ("anchor_practical_gain", "anchor_superiority", "raw_superiority"))
    per_fit = {}
    for fit in FITS:
        selected = [row for row in rows if row["fit"] == fit]
        per_fit[str(fit)] = {}
        for name in CONTRASTS:
            bounds = [sum(_numerator(row["delta_bounds"][name][endpoint], -1, 1) for row in selected)/(len(selected)*256) for endpoint in (0,1)]
            per_fit[str(fit)][name] = {"mean_score_difference": bounds[0] if bounds[0] == bounds[1] else None,
                                       "mean_bounds": bounds, "score_percentage_points": 100*bounds[0] if bounds[0] == bounds[1] else None}
    return {"contrasts": contrasts, "per_fit": per_fit, **flags,
            "classification": "positive_conditional_relations_correction" if flags["gate_pass"] else "practical_progress_not_established",
            "unknown_outcomes_imputed": False}
