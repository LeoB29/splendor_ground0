"""Whole-pair development statistics; each root uses its assigned source fit."""
from collections import defaultdict
import math
import numpy as np
from . import common as c

SPLITS = ("train", "validation")
ARMS = ("anchor", "anchor_outsider", "broad_supported", "broad_outsider",
        "headonly_supported", "headonly_outsider", "teacher", "teacher_outsider")
CONTRASTS = tuple((name, *name.split("_minus_")) for name in (
    "headonly_supported_minus_anchor", "headonly_supported_minus_broad_supported",
    "headonly_outsider_minus_anchor_outsider", "headonly_outsider_minus_broad_outsider",
    "broad_supported_minus_anchor", "teacher_minus_anchor", "teacher_minus_headonly_supported",
    "teacher_outsider_minus_headonly_outsider"))
NAMES = tuple(row[0] for row in CONTRASTS)


def _integer_bounds(bounds):
    c.require(isinstance(bounds, list) and len(bounds) == 2
              and all(type(x) in (int,float) and math.isfinite(x) for x in bounds)
              and -1 <= bounds[0] <= bounds[1] <= 1, "Invalid score interval")
    values = [int(round(x*256)) for x in bounds]
    c.require([x/256 for x in values] == bounds, "Incorrect 128-world score grid")
    return values


def _validate_row(row):
    n = row["support"]
    c.require(type(n) is int and 1 <= n <= 8 and row["outsider_available"] == (n > 3), "Invalid support")
    c.require(set(row["choices"]) == set(row["arm_score_bounds"]) == set(row["arm_scores"]) == set(ARMS)
              and set(row["delta_bounds"]) == set(NAMES), "Incomplete planning arms")
    c.require(len(row["unknown_counts"]) == n and all(type(v) is int and 0 <= v <= 128
              for v in row["unknown_counts"]), "Invalid unknown counts")
    for arm in ARMS:
        choice, bounds = row["choices"][arm], row["arm_score_bounds"][arm]
        absent = n <= 3 and arm.endswith("_outsider")
        c.require((choice is None) == absent and (bounds is None) == absent, "Absent outsider accounting")
        if not absent:
            c.require(type(choice) is int and 0 <= choice < n, "Choice leaves recorded support")
            _integer_bounds(bounds)
            c.require(bounds[0] >= 0 and bounds[1] <= 1, "Invalid terminal score")
            if arm.endswith("_outsider"):
                c.require(choice >= 3, "Outsider enters common first three")
        c.require(row["arm_scores"][arm] == (bounds[0] if bounds is not None and bounds[0] == bounds[1] else None),
                  "Invented point score")
    for family, supported in (("anchor","anchor"),("broad","broad_supported"),("headonly","headonly_supported")):
        candidate = row["choices"][family+"_outsider"]
        c.require(row["choices"][supported] in list(range(min(3,n)))+([] if candidate is None else [candidate]),
                  "Supported selection leaves its shortlist")
    for left in ARMS:
        for right in ARMS:
            if row["choices"][left] == row["choices"][right]:
                c.require(row["arm_score_bounds"][left] == row["arm_score_bounds"][right], "Equal action has unequal score")
    for name,left,right in CONTRASTS:
        a,b = row["arm_score_bounds"][left],row["arm_score_bounds"][right]
        expected = [0.,0.] if row["choices"][left] == row["choices"][right] else [a[0]-b[1],a[1]-b[0]]
        c.require(row["delta_bounds"][name] == expected, "Paired action bounds differ")


def _arrays(rows):
    c.require(len(rows) == 336 and [r["index"] for r in rows] == list(range(336)), "Complete ordered 336-root panel required")
    pairs = defaultdict(list)
    for row in rows:
        _validate_row(row)
        pairs[row["pair"]].append(row)
    c.require(len(pairs) == 84, "Expected 84 source pairs")
    strata = defaultdict(list)
    for pair,records in sorted(pairs.items()):
        cell = records[0]["split"],records[0]["fit"]
        c.require(len(records) == 4 and all((r["split"],r["fit"]) == cell for r in records), "Pair spans fit or split")
        c.require({r["game_index"] for r in records} == {2*pair,2*pair+1}, "Missing paired game leg")
        phase_sets=[]
        for game in (2*pair,2*pair+1):
            leg=[r for r in records if r["game_index"] == game]
            c.require(len(leg) == 2 and len({r["phase"] for r in leg}) == 2, "Two distinct phase roots per leg required")
            phase_sets.append({r["phase"] for r in leg})
        c.require(phase_sets[0] == phase_sets[1], "Paired phase schedule differs")
        values = [[sum(_integer_bounds(r["delta_bounds"][name])[j] for r in records) for j in (0,1)] for name in NAMES]
        strata[cell].append(values)
    cells=[(split,fit) for split in SPLITS for fit in c.FITS]
    c.require(set(strata) == set(cells) and all(len(strata[cell]) == 14 for cell in cells), "Six strata of14 pairs required")
    return [np.asarray(strata[cell],dtype=np.int64) for cell in cells]


def _entry(bounds,ci=None):
    bounds=list(map(float,bounds))
    result={"mean":bounds[0] if bounds[0] == bounds[1] else None,"mean_bounds":bounds}
    if ci is not None: result["ci95"]=list(map(float,ci))
    return result


def _report(arrays,samples,quantile):
    totals=np.stack([sum((a.sum(axis=0) for a in arrays[i:i+3]),np.zeros((len(NAMES),2),dtype=np.int64)) for i in (0,3)])
    denominator=168*256
    groups={}
    for si,split in enumerate(SPLITS):
        groups[split]={"states":168,"pairs":42,"strata":3,"contrasts":{
            name:_entry(totals[si,j]/denominator,[quantile(samples[:,si,j,0],.025)/denominator,
                                                quantile(samples[:,si,j,1],.975)/denominator]) for j,name in enumerate(NAMES)}}
    difference=np.stack([samples[:,1,:,0]-samples[:,0,:,1],samples[:,1,:,1]-samples[:,0,:,0]],axis=-1)
    point=np.stack([totals[1,:,0]-totals[0,:,1],totals[1,:,1]-totals[0,:,0]],axis=-1)
    groups["validation_minus_train"]={"contrasts":{name:_entry(point[j]/denominator,[
        quantile(difference[:,j,0],.025)/denominator,quantile(difference[:,j,1],.975)/denominator]) for j,name in enumerate(NAMES)}}
    per_fit={str(fit):{split:{"states":56,"pairs":14,"contrasts":{
        name:_entry(arrays[3*si+fi].sum(axis=0)[j]/(56*256)) for j,name in enumerate(NAMES)}}
        for si,split in enumerate(SPLITS)} for fi,fit in enumerate(c.FITS)}
    return {"states":336,"pairs":84,"groups":groups,"per_fit":per_fit,"fits":list(c.FITS),
            "bootstrap_samples":c.CONFIG["bootstrap_samples"],"bootstrap_seed":c.CONFIG["bootstrap_seed"],
            "resampling_unit":"whole_source_pair_four_roots","source_fit_assignment":"one_assigned_model_per_root",
            "unknown_outcomes_imputed":False,"reused_development":True,"promotion":False}


def summarize(rows,boundary=None):
    arrays=_arrays(rows)
    rng=np.random.default_rng(c.CONFIG["bootstrap_seed"])
    samples=np.zeros((c.CONFIG["bootstrap_samples"],2,len(NAMES),2),dtype=np.int64)
    for start in range(0,len(samples),c.CONFIG["bootstrap_chunk"]):
        if boundary: boundary()
        count=min(c.CONFIG["bootstrap_chunk"],len(samples)-start)
        for cell,values in enumerate(arrays):
            samples[start:start+count,cell//3] += values[rng.integers(0,14,size=(count,14))].sum(axis=1)
    return _report(arrays,samples,lambda a,q:float(np.quantile(a,q,method="linear")))


def independent_summarize(rows,boundary=None):
    """Separate pair construction and histogram bootstrap; never calls summarize."""
    _arrays(rows)  # Shared structural assertions, independently construct numeric arrays below.
    matrices=[]
    for split in SPLITS:
        for fit in c.FITS:
            selected=[r for r in rows if r["split"] == split and r["fit"] == fit]
            pair_ids=sorted({r["pair"] for r in selected})
            matrix=[]
            for pair in pair_ids:
                records=[r for r in selected if r["pair"] == pair]
                values=[]
                for _,left,right in CONTRASTS:
                    endpoints=[0,0]
                    for row in records:
                        if row["choices"][left] != row["choices"][right]:
                            a,b=row["arm_score_bounds"][left],row["arm_score_bounds"][right]
                            endpoints[0]+=round(256*a[0])-round(256*b[1])
                            endpoints[1]+=round(256*a[1])-round(256*b[0])
                    values.append(endpoints)
                matrix.append(values)
            matrices.append(np.asarray(matrix,dtype=np.int64))
    random=np.random.default_rng(c.CONFIG["bootstrap_seed"])
    values=np.zeros((c.CONFIG["bootstrap_samples"],2,len(NAMES),2),dtype=np.int64)
    for offset in range(0,len(values),c.CONFIG["bootstrap_chunk"]):
        if boundary: boundary()
        size=min(c.CONFIG["bootstrap_chunk"],len(values)-offset)
        for cell,matrix in enumerate(matrices):
            sampled=random.integers(0,14,size=(size,14))
            frequencies=np.stack([np.bincount(row,minlength=14) for row in sampled])
            values[offset:offset+size,cell//3]+=(frequencies @ matrix.reshape(14,-1)).reshape(size,len(NAMES),2)
    def percentile(array,q):
        ordered=sorted(int(value) for value in array)
        position=(len(ordered)-1)*q
        lo,hi=math.floor(position),math.ceil(position)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    result_groups={}
    all_numerators={}
    for si,split in enumerate(SPLITS):
        means={}
        all_numerators[split]=[]
        for j,name in enumerate(NAMES):
            numerator=[sum(int(matrices[si*3+fi][pair,j,end]) for fi in range(3) for pair in range(14))
                       for end in (0,1)]
            all_numerators[split].append(numerator)
            endpoints=[x/43008 for x in numerator]
            means[name]={"mean":endpoints[0] if endpoints[0] == endpoints[1] else None,"mean_bounds":endpoints,
                         "ci95":[percentile(values[:,si,j,0],.025)/43008,percentile(values[:,si,j,1],.975)/43008]}
        result_groups[split]={"states":168,"pairs":42,"strata":3,"contrasts":means}
    difference={}
    for j,name in enumerate(NAMES):
        train,val=all_numerators["train"][j],all_numerators["validation"][j]
        endpoints=[(val[0]-train[1])/43008,(val[1]-train[0])/43008]
        difference[name]={"mean":endpoints[0] if endpoints[0] == endpoints[1] else None,"mean_bounds":endpoints,
            "ci95":[percentile(values[:,1,j,0]-values[:,0,j,1],.025)/43008,
                    percentile(values[:,1,j,1]-values[:,0,j,0],.975)/43008]}
    result_groups["validation_minus_train"]={"contrasts":difference}
    by_fit={}
    for fi,fit in enumerate(c.FITS):
        by_fit[str(fit)]={}
        for si,split in enumerate(SPLITS):
            means={}
            for j,name in enumerate(NAMES):
                endpoints=[sum(int(matrices[si*3+fi][p,j,end]) for p in range(14))/14336 for end in (0,1)]
                means[name]={"mean":endpoints[0] if endpoints[0] == endpoints[1] else None,"mean_bounds":endpoints}
            by_fit[str(fit)][split]={"states":56,"pairs":14,"contrasts":means}
    return {"states":336,"pairs":84,"groups":result_groups,"per_fit":by_fit,"fits":list(c.FITS),
            "bootstrap_samples":c.CONFIG["bootstrap_samples"],"bootstrap_seed":c.CONFIG["bootstrap_seed"],
            "resampling_unit":"whole_source_pair_four_roots","source_fit_assignment":"one_assigned_model_per_root",
            "unknown_outcomes_imputed":False,"reused_development":True,"promotion":False}


def development_screen(statistics,validation,acquisition):
    primary=c.CONFIG["primary_endpoint"]
    control=c.CONFIG["required_control_endpoint"]
    endpoint=statistics["groups"]["validation"]["contrasts"][primary]
    reference=statistics["groups"]["validation"]["contrasts"][control]
    capture={str(fit):acquisition[str(fit)] for fit in c.FITS}
    acquired=all(type(v) in (int,float) and math.isfinite(v)
                 and v >= c.CONFIG["minimum_added_regret_capture_each_fit"] for v in capture.values())
    old_regret={}
    for fit in c.FITS:
        values={family:validation["families"][family]["per_fit"][str(fit)]["pair_weighted"]["mean_old_score_regret"]
                for family in c.MODEL_ARMS}
        old_regret[str(fit)]={"regrets":values,"headonly_better_than_both":
            values["headonly"] < values["anchor"] and values["headonly"] < values["broad"]}
    generalization=all(v["headonly_better_than_both"] for v in old_regret.values())
    consistent=all(statistics["per_fit"][str(fit)]["validation"]["contrasts"][name]["mean_bounds"][0] >= 0
                   for fit in c.FITS for name in (primary,control))
    practical=endpoint["mean_bounds"][0] >= c.CONFIG["minimum_progress"] and endpoint["ci95"][0] > 0
    superior=reference["ci95"][0] > 0
    passed=bool(acquired and generalization and consistent and practical and superior)
    return {"passed":passed,"classification":"fresh_confirmation_justified" if passed else
                "acquisition_failed" if not acquired else "transfer_not_established",
            "primary_endpoint":primary,"required_control_endpoint":control,"split":"validation",
            "added_regret_capture_each_fit":capture,"acquisition_pass":acquired,
            "minimum_added_regret_capture_each_fit":c.CONFIG["minimum_added_regret_capture_each_fit"],
            "old_validation_regret_each_fit":old_regret,"old_validation_pass":generalization,
            "minimum_mean":c.CONFIG["minimum_progress"],"primary_practical_pass":bool(practical),
            "control_ci95":reference["ci95"],"control_lower_strictly_positive":bool(superior),
            "per_fit_planning_nonnegative":bool(consistent),"reused_development":True,
            "fresh_full_legal_confirmation_required":True,"automatic_successor":False,"promotion":False}
