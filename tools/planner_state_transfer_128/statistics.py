"""Paired, stratified inference with conservative unknown-outcome envelopes."""
from collections import defaultdict
import numpy as np
from . import common as c

NAMES = ("broad_minus_anchor", "broad_minus_narrow", "narrow_minus_anchor", "teacher_minus_anchor")

def summarize(rows):
    c.require(len(rows) == 360 and {r["index"] for r in rows} == set(range(360)), "Complete fixed main panel required")
    c.require(len({r["game_index"] for r in rows}) == 360, "Repeated source game")
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair"]].append(row)
    c.require(len(pairs) == 180, "180 independent source pairs required")
    strata = defaultdict(list)
    for pair, records in sorted(pairs.items()):
        c.require(len(records) == 2 and {r["seat"] for r in records} == {0,1}
                  and {r["game_index"] for r in records} == {pair*2,pair*2+1}, "Broken source pair")
        cell = tuple(records[0][k] for k in ("fit","phase","source_arm"))
        c.require(all(tuple(r[k] for k in ("fit","phase","source_arm")) == cell for r in records), "Pair spans cells")
        values = np.zeros((4,2), dtype=np.int64)
        for row in records:
            c.require(set(row["delta_bounds"]) == set(NAMES), "Missing contrast")
            for j, name in enumerate(NAMES):
                bounds = row["delta_bounds"][name]
                c.require(len(bounds) == 2 and -1 <= bounds[0] <= bounds[1] <= 1, "Invalid score bounds")
                units = np.rint(np.array(bounds)*256).astype(np.int64)
                c.require(np.allclose(units/256,bounds,rtol=0,atol=1e-12), "Wrong score scale")
                values[j] += units
        strata[cell].append(values)
    expected = [(f,p,a) for f in c.FITS for p in range(3) for a in ("candidate","reference")]
    c.require(set(strata) == set(expected) and all(len(v) == 10 for v in strata.values()), "18 balanced pair strata required")
    arrays = [np.stack(strata[cell]) for cell in expected]
    totals = sum(a.sum(axis=0) for a in arrays)
    point = totals/(360*256)
    rng = np.random.default_rng(c.CONFIG["bootstrap_seed"])
    batches = []
    for start in range(0,c.CONFIG["bootstrap_samples"],c.CONFIG["bootstrap_chunk"]):
        count = min(c.CONFIG["bootstrap_chunk"],c.CONFIG["bootstrap_samples"]-start)
        batch = np.zeros((count,4,2),dtype=np.int64)
        for values in arrays:
            batch += values[rng.integers(0,10,size=(count,10))].sum(axis=1)
        batches.append(batch/(360*256))
    samples = np.concatenate(batches)
    low = np.quantile(samples[:,:,0],.025,axis=0,method="linear")
    high = np.quantile(samples[:,:,1],.975,axis=0,method="linear")
    contrasts = {name:{"mean":float(point[j,0]) if totals[j,0] == totals[j,1] else None,
                      "mean_bounds":point[j].tolist(),"ci95":[float(low[j]),float(high[j])]}
                 for j,name in enumerate(NAMES)}
    gates = {"anchor_practical_gain":bool(totals[0,0]*200 >= 360*256),
             "anchor_superiority":bool(low[0] > 0), "narrow_attribution":bool(low[1] > 0)}
    gates["gate_pass"] = all(gates.values())
    return {"states":360,"pairs":180,"strata":18,"bootstrap_samples":c.CONFIG["bootstrap_samples"],
            "unknown_outcomes_imputed":False,"contrasts":contrasts,**gates,
            "classification":"positive_planner_state_transfer" if gates["gate_pass"] else "planner_state_transfer_not_established"}
