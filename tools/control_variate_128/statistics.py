"""TRAIN-only finite nested-control selection and paired development statistics.

All variances are within a root's common worlds. The terminal estimand is
unchanged; the finite auxiliary mean is never treated as known.
"""
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from . import common as c


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def _panel(rows,split,horizons):
    c.require(len(rows) == 168 and len({r["index"] for r in rows}) == 168,"Complete168-root split required")
    pairs=defaultdict(list)
    for row in rows:
        n=row["support"]
        c.require(row["split"] == split and row["fit"] in c.FITS and type(n) is int and 1 <= n <= 8
                  and type(row["reference"]) is int and 0 <= row["reference"] < n,"Invalid root identity/support")
        c.require(set(row["z"]) == set(map(str,horizons)),"Unplanned prefix horizon")
        c.require(len(row["y"]) == 128 and all(len(w) == n and all(_finite(v) and v in (0.,.5,1.) for v in w)
                  for w in row["y"]),"Unknown or invalid terminal outcome; do not impute")
        for h in horizons:
            matrix=row["z"][str(h)]
            c.require(len(matrix) == 128 and all(len(w) == n and all(_finite(v) for v in w) for w in matrix),
                      "Invalid prefix score matrix")
        pairs[row["pair"]].append(row)
    c.require(len(pairs) == 42,"Expected42 source pairs")
    for pair,group in pairs.items():
        c.require(len(group) == 4 and len({r["fit"] for r in group}) == 1,"Pair spans fit or lacks roots")
        c.require({r["game_index"] for r in group} == {2*pair,2*pair+1},"Incorrect paired game identities")
        phases=[]
        for game in (2*pair,2*pair+1):
            leg=[r for r in group if r["game_index"] == game]
            c.require(len(leg) == 2 and len({r["phase"] for r in leg}) == 2,"Pair must retain two distinct phases")
            phases.append({r["phase"] for r in leg})
        c.require(phases[0] == phases[1] and phases[0] <= {0,1,2},"Mismatched paired phases")
    c.require(all(sum(group[0]["fit"] == fit for group in pairs.values()) == 14 for fit in c.FITS),
              "Expected14 pairs per assigned source fit")


def moments(row,horizon,*,audit=False):
    n=row["support"];reference=row["reference"]
    identity={key:row[key] for key in ("index","pair","fit","phase","support")}
    if n == 1:
        return {**identity,"contrasts":0,"var_y":0.,"var_z":0.,"covariance":0.}
    if audit:
        values=[]
        for action in range(n):
            if action == reference: continue
            d=[world[action]-world[reference] for world in row["y"]]
            u=[world[action]-world[reference] for world in row["z"][str(horizon)]]
            md,mu=math.fsum(d)/128,math.fsum(u)/128
            values.append((math.fsum((x-md)**2 for x in d)/127,
                           math.fsum((x-mu)**2 for x in u)/127,
                           math.fsum((d[i]-md)*(u[i]-mu) for i in range(128))/127))
        vy,vz,co=[math.fsum(v[i] for v in values)/len(values) for i in range(3)]
    else:
        y=np.asarray(row["y"],dtype=np.float64);z=np.asarray(row["z"][str(horizon)],dtype=np.float64)
        columns=[j for j in range(n) if j != reference]
        d=y[:,columns]-y[:,reference,None];u=z[:,columns]-z[:,reference,None]
        d-=d.mean(axis=0);u-=u.mean(axis=0)
        vy=float((d*d).sum(axis=0).mean()/127)
        vz=float((u*u).sum(axis=0).mean()/127)
        co=float((d*u).sum(axis=0).mean()/127)
    c.require(vy >= 0 and vz >= 0 and co*co <= vy*vz+1e-12,"Invalid within-root covariance")
    return {**identity,"contrasts":n-1,"var_y":vy,"var_z":vz,"covariance":co}


def cost_model(timing_rows,*,audit=False):
    c.require(len(timing_rows) == 18 and len({r["index"] for r in timing_rows}) == 18,"Exactly18 timing roots required")
    c.require({(r["fit"],r["seat"],r["phase"]) for r in timing_rows}
              == {(fit,seat,phase) for fit in c.FITS for seat in (0,1) for phase in (0,1,2)},
              "Timing panel must balance fit/seat/phase")
    for row in timing_rows:
        c.require(row["split"] == "train" and 1 <= row["support"] <= 8,"Timing uses TRAIN only")
        blocks=[row["timing"]["full_only"]]
        for name in ("full_plus","prefix_only"):
            c.require(set(row["timing"][name]) == set(map(str,c.HORIZONS)),"Timing horizons differ")
            blocks.extend(row["timing"][name].values())
        c.require(all(type(b["cells"]) is int and b["cells"] == row["support"]*c.CONFIG["timing_worlds"]
                  and _finite(b["cpu"]) and b["cpu"] > 0 for b in blocks),"Invalid matched CPU timing cells")
    reducer=math.fsum if audit else sum
    phases={};dispersion={}
    for phase in (0,1,2):
        group=[r for r in timing_rows if r["phase"] == phase]
        def rate(kind,h=None):
            blocks=[r["timing"][kind] if h is None else r["timing"][kind][str(h)] for r in group]
            if audit:
                numerator=math.fsum(b["cpu"] for b in blocks);denominator=sum(b["cells"] for b in blocks)
            else: numerator,denominator=sum(b["cpu"] for b in blocks),sum(b["cells"] for b in blocks)
            return numerator/denominator
        full=rate("full_only")
        horizons={}
        for h in c.HORIZONS:
            combined=rate("full_plus",h)
            horizons[str(h)]={"full_plus":max(full,combined),"full_plus_unclamped":combined,"prefix_only":rate("prefix_only",h)}
        phases[str(phase)]={"full_only":full,"horizons":horizons,"roots":len(group)}
        rates=[r["timing"]["full_only"]["cpu"]/r["timing"]["full_only"]["cells"] for r in group]
        dispersion[str(phase)]={"full_only_min":min(rates),"full_only_max":max(rates),
            "full_only_root_mean":reducer(rates)/len(rates),"prefix_root_ranges":{
            str(h):[min(r["timing"]["prefix_only"][str(h)]["cpu"]/r["timing"]["prefix_only"][str(h)]["cells"] for r in group),
                    max(r["timing"]["prefix_only"][str(h)]["cpu"]/r["timing"]["prefix_only"][str(h)]["cells"] for r in group)]
            for h in c.HORIZONS},"full_plus_root_ranges":{
            str(h):[min(r["timing"]["full_plus"][str(h)]["cpu"]/r["timing"]["full_plus"][str(h)]["cells"] for r in group),
                    max(r["timing"]["full_plus"][str(h)]["cpu"]/r["timing"]["full_plus"][str(h)]["cells"] for r in group)]
            for h in c.HORIZONS}}
    return {"unit":"process_cpu_seconds_per_candidate_world","phases":phases,"timing_roots":18,
            "full_plus_clamped_to_full_only":True,"dispersion":dispersion,
            "timing_uncertainty_in_primary_ci":False,"timing_population":"TRAIN_only_balanced18"}


def _variance(moment,beta,n,M,*,audit=False):
    if audit:
        residual=moment["var_y"]+beta*beta*moment["var_z"]-2*beta*moment["covariance"]
        c.require(residual >= -1e-12,"Invalid residual variance")
        value=max(0.,residual)*(1/n-1/M)+moment["var_y"]/M
    else:
        value=moment["var_y"]/n+(1/n-1/M)*(beta*beta*moment["var_z"]-2*beta*moment["covariance"])
    c.require(math.isfinite(value) and value >= -1e-12,"Invalid predicted nested variance")
    return max(0.,value)


def _cost(row,config,model):
    phase=model["phases"][str(row["phase"])];rates=phase["horizons"][str(config["horizon"])];n,M=config["n"],config["M"]
    return (row["support"]*(n*rates["full_plus"]+(M-n)*rates["prefix_only"]),
            row["support"]*128*phase["full_only"])


def _feasible(config,model):
    h,n,M=config["horizon"],config["n"],config["M"]
    return all(n*p["horizons"][str(h)]["full_plus"]+(M-n)*p["horizons"][str(h)]["prefix_only"]
               <= 128*p["full_only"]*(1+1e-12) for p in model["phases"].values())


def calibrate(train_rows,timing_rows,*,audit=False,boundary=None):
    _panel(train_rows,"train",c.HORIZONS)
    model=cost_model(timing_rows,audit=audit)
    reducer=math.fsum if audit else sum
    baseline_cost=reducer(r["support"]*128*model["phases"][str(r["phase"])]["full_only"] for r in train_rows)/168
    per_horizon={};best=None;total=0
    for h in c.HORIZONS:
        if boundary: boundary()
        items=[moments(r,h,audit=audit) for r in train_rows]
        pooled={name:reducer(r[name] for r in items)/168 for name in ("var_y","var_z","covariance")}
        beta=pooled["covariance"]/pooled["var_z"] if pooled["var_z"] else 0.
        c.require(math.isfinite(beta),"Nonfinite calibrated coefficient")
        local=None;count=0
        combined=reducer(r["support"]*model["phases"][str(r["phase"])]["horizons"][str(h)]["full_plus"] for r in train_rows)/168
        prefix=reducer(r["support"]*model["phases"][str(r["phase"])]["horizons"][str(h)]["prefix_only"] for r in train_rows)/168
        if pooled["var_y"] > 0:
            for n in range(c.CONFIG["minimum_full_samples"],c.CONFIG["maximum_full_samples"]+1):
                if boundary and n % 16 == 0: boundary()
                for M in range(n+1,c.CONFIG["maximum_prefix_samples"]+1):
                    config={"horizon":h,"beta":beta,"n":n,"M":M}
                    if not _feasible(config,model): continue
                    variance=_variance(pooled,beta,n,M,audit=audit)
                    cpu=n*combined+(M-n)*prefix
                    ratio=variance*cpu/(pooled["var_y"]/128*baseline_cost)
                    entry={"config":config,"efficiency_ratio":ratio,"predicted_variance":variance,"predicted_cpu":cpu}
                    key=(round(ratio,c.CONFIG.get("selection_ratio_round_digits",10)),h,n,M)
                    if local is None or key < local[0]: local=(key,entry)
                    count+=1
        per_horizon[str(h)]={"pooled_moments":pooled,"beta":beta,"feasible_allocations":count,
                             "best":None if local is None else local[1],"root_moments":items}
        if local is not None and (best is None or local[0] < best[0]): best=local
        total+=count
    zero=all(block["pooled_moments"]["var_y"] == 0 for block in per_horizon.values())
    return {"status":"undefined_zero_training_variance" if zero else "no_feasible_allocation" if best is None else "selected",
        "config":None if best is None else best[1]["config"],"selected_training_prediction":None if best is None else best[1],
        "cost_model":model,"training":{"states":168,"pairs":42,"horizons":per_horizon},
        "search":{"feasible_allocations":total,"tie_order":"rounded_ratio_then_horizon_n_M",
                  "ratio_round_digits":c.CONFIG.get("selection_ratio_round_digits",10),
                  "phase_budget_tolerance":1e-12,"allocation_budget":"each_TRAIN_calibrated_phase_at_most128_full",
                  "prefix_mean":"finite_nested_M_not_known"},
        "selection_split":"train","validation_read":False,"selected_before_validation":True,
        "reused_development":True,"actual_estimator_run":False,"automatic_successor":False}


def _point(vectors,*,audit=False):
    reducer=math.fsum if audit else sum
    means=[reducer(row[j] for row in vectors)/len(vectors) for j in range(4)]
    vc,vb,cc,cb=means
    ratio=vc*cc/(vb*cb) if vb > 0 and cb > 0 else None
    return {"mean_predicted_variance":vc,"mean_baseline_variance":vb,
            "mean_predicted_cpu":cc,"mean_baseline_cpu":cb,"efficiency_ratio":ratio}


def _sensitivity(rows,point,selection):
    h=str(selection["config"]["horizon"]);n,M=selection["config"]["n"],selection["config"]["M"]
    costs=[]
    worst_budget=[]
    for row in rows:
        p=selection["cost_model"]["dispersion"][str(row["phase"])]
        f0,f1=p["full_only_min"],p["full_only_max"]
        b0,b1=p["full_plus_root_ranges"][h];p0,p1=p["prefix_root_ranges"][h]
        lower=n*max(b0,f0)+(M-n)*p0;upper=n*max(b1,f1)+(M-n)*p1
        costs.append([row["support"]*lower,row["support"]*upper,row["support"]*128*f0,row["support"]*128*f1])
        worst_budget.append(upper/(128*f0))
    means=[math.fsum(v[j] for v in costs)/len(costs) for j in range(4)]
    scale=point["mean_predicted_variance"]/point["mean_baseline_variance"] if point["mean_baseline_variance"] else None
    return {"definition":"observed_timing_root_rate_envelope_not_a_confidence_interval",
            "efficiency_ratio_envelope":None if scale is None else [scale*means[0]/means[3],scale*means[1]/means[2]],
            "largest_worst_envelope_phase_budget_ratio":max(worst_budget),"used_for_selection_or_gate":False}


def summarize(val_rows,selection,*,audit=False,boundary=None):
    c.require(selection["status"] == "selected" and selection["selection_split"] == "train"
              and selection["validation_read"] is False and selection["selected_before_validation"] is True,
              "A sealed TRAIN-only selection is required")
    config=selection["config"];h,beta,n,M=(config[k] for k in ("horizon","beta","n","M"))
    c.require(h in c.HORIZONS and _finite(beta) and type(n) is int and type(M) is int
              and c.CONFIG["minimum_full_samples"] <= n <= c.CONFIG["maximum_full_samples"]
              and n < M <= c.CONFIG["maximum_prefix_samples"] and _feasible(config,selection["cost_model"]),
              "Invalid sealed finite allocation")
    _panel(val_rows,"validation",(h,))
    vectors=[];records=[]
    for row in val_rows:
        if boundary: boundary()
        moment=moments(row,h,audit=audit);cpu,baseline=_cost(row,config,selection["cost_model"])
        vector=[_variance(moment,beta,n,M,audit=audit),moment["var_y"]/128,cpu,baseline]
        vectors.append(vector);records.append({**moment,"predicted_variance":vector[0],"baseline_variance":vector[1],
                                               "predicted_cpu":cpu,"baseline_cpu":baseline})
    point=_point(vectors,audit=audit)
    matrices=[]
    for fit in c.FITS:
        pairs=sorted({r["pair"] for r in val_rows if r["fit"] == fit})
        matrix=[]
        for pair in pairs:
            group=[v for r,v in zip(val_rows,vectors,strict=True) if r["fit"] == fit and r["pair"] == pair]
            matrix.append([math.fsum(v[j] for v in group) if audit else sum(v[j] for v in group) for j in range(4)])
        matrices.append(np.asarray(matrix,dtype=np.float64))
    rng=np.random.default_rng(c.CONFIG["bootstrap_seed"])
    samples=np.zeros((c.CONFIG["bootstrap_samples"],4),dtype=np.float64)
    for start in range(0,len(samples),c.CONFIG["bootstrap_chunk"]):
        if boundary: boundary()
        size=min(c.CONFIG["bootstrap_chunk"],len(samples)-start)
        for matrix in matrices:
            draws=rng.integers(0,14,size=(size,14))
            if audit:
                counts=np.stack([np.bincount(d,minlength=14) for d in draws])
                # Elementwise histogram reduction avoids loading a second BLAS
                # runtime alongside the production Torch inference runtime.
                samples[start:start+size]+=(counts[:,:,None]*matrix[None,:,:]).sum(axis=1)
            else: samples[start:start+size]+=matrix[draws].sum(axis=1)
    zero=int(np.count_nonzero(samples[:,1] <= 0))
    ci=None
    if zero == 0:
        ratios=samples[:,0]*samples[:,2]/(samples[:,1]*samples[:,3])
        if audit:
            ordered=sorted(float(v) for v in ratios)
            def quantile(q):
                p=(len(ordered)-1)*q;lo,hi=math.floor(p),math.ceil(p)
                return ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
            ci=[quantile(.025),quantile(.975)]
        else: ci=list(map(float,np.quantile(ratios,[.025,.975],method="linear")))
    threshold=1-c.CONFIG["minimum_variance_cost_reduction"]
    passed=point["efficiency_ratio"] is not None and point["efficiency_ratio"] <= threshold and ci is not None and ci[1] < 1
    per_fit={str(fit):_point([v for r,v in zip(val_rows,vectors,strict=True) if r["fit"] == fit],audit=audit) for fit in c.FITS}
    phases={str(phase):_point([v for r,v in zip(val_rows,vectors,strict=True) if r["phase"] == phase],audit=audit) for phase in (0,1,2)}
    return {**point,"ci95":ci,"states":168,"pairs":42,"assigned_fits":list(c.FITS),"config":config,
        "per_fit":per_fit,"phase_diagnostics":phases,"rows":records,
        "bootstrap_samples":c.CONFIG["bootstrap_samples"],"bootstrap_seed":c.CONFIG["bootstrap_seed"],
        "bootstrap_zero_denominator_count":zero,"resampling_unit":"whole_pair_four_roots_within_source_fit",
        "passed":bool(passed),"maximum_efficiency_ratio":threshold,
        "classification":"efficiency_feasible" if passed else "undefined_zero_validation_variance" if point["efficiency_ratio"] is None
            else "efficiency_not_established","timing_uncertainty_in_ci":False,
        "ci_conditions_on":"fixed_TRAIN_selection_and_TRAIN_timing_calibration",
        "ratio_definition":"mean_root_variance_times_mean_root_vector_CPU_relative_to128_full",
        "timing_sensitivity":_sensitivity(val_rows,point,selection),
        "reused_development":True,"actual_estimator_run":False,"promotion":False}


def _same(a,b):
    if isinstance(a,dict):
        c.require(isinstance(b,dict) and set(a) == set(b),"Independent statistic keys differ")
        for key in a: _same(a[key],b[key])
    elif isinstance(a,list):
        c.require(isinstance(b,list) and len(a) == len(b),"Independent statistic lengths differ")
        for x,y in zip(a,b,strict=True): _same(x,y)
    elif type(a) in (float,int) and type(b) in (float,int):
        c.require(math.isclose(a,b,rel_tol=0.,abs_tol=1e-12),"Independent numeric statistic differs")
    else: c.require(a == b,"Independent statistic differs")


def _read_roots(refs,binding,stage):
    rows=[]
    identity=("index","pair","fit","split","game_index","phase","seat","support")
    keys=identity+(("timing",) if stage == "timing" else ("reference","y","z"))
    for ref in refs:
        root=c.verified(ref)
        c.require(root["manifest"] == binding and root.get("complete") is True,"Incomplete root or manifest differs")
        c.require(root["split"] == ("train" if stage == "timing" else stage),"Root stage differs")
        if stage != "timing": c.require(root.get("stage") == stage,"Collection stage differs")
        c.require(root["statistics"] == {key:root[key] for key in keys},"Statistics projection differs from collected root")
        rows.append(root["statistics"])
        # The endpoint/cell tree is much larger than these compact matrices.
        # Retain only its authenticated projection before opening the next root.
        del root
    return rows


def _sealed_selection(selectionref,binding,base):
    c.require(c.artifact(base/"selection_lock.json") == selectionref,"Selection must be the local immutable seal")
    selected=c.verified(selectionref)
    receipt=c.old.read_json(base/"selection_audit.json")
    c.require(selected["manifest"] == receipt["manifest"] == binding and receipt.get("passed") is True
              and receipt.get("selection") == selectionref and receipt.get("validation_read") is False
              and all(receipt.get(key) is True for key in ("moments_independently_reconstructed",
                      "allocation_independently_reproduced","fixed_train_rule_preserved")),
              "Passed linked TRAIN selection audit required before validation")
    return selected


def select(trainrefs,timingrefs,binding,base=None):
    base=Path(c.OUTPUT if base is None else base);c.budget_boundary(base)
    c.require(not (base/"selection_lock.json").exists(),"Preserve the sealed selection")
    result=calibrate(_read_roots(trainrefs,binding,"train"),_read_roots(timingrefs,binding,"timing"),boundary=lambda:c.budget_boundary(base))
    result.update(manifest=binding,training_roots=trainrefs,timing_roots=timingrefs,time=c.old.now())
    path=base/"selection_lock.json";c.old.save_new(path,result);return c.artifact(path)


def audit_selection(trainrefs,timingrefs,selectionref,binding,base=None):
    base=Path(c.OUTPUT if base is None else base);c.budget_boundary(base)
    c.require(not (base/"selection_audit.json").exists(),"Preserve selection audit")
    result=c.verified(selectionref)
    c.require(result["manifest"] == binding and result["training_roots"] == trainrefs and result["timing_roots"] == timingrefs,
              "Selection source linkage differs")
    expected=calibrate(_read_roots(trainrefs,binding,"train"),_read_roots(timingrefs,binding,"timing"),audit=True,boundary=lambda:c.budget_boundary(base))
    _same({k:v for k,v in result.items() if k not in ("manifest","training_roots","timing_roots","time")},expected)
    receipt={"manifest":binding,"selection":selectionref,"passed":True,"TRAIN_moments_selection_cost_independently_reproduced":True,
             "moments_independently_reconstructed":True,"allocation_independently_reproduced":True,
             "fixed_train_rule_preserved":True,
             "validation_read":False,"time":c.old.now()}
    path=base/"selection_audit.json";c.old.save_new(path,receipt);return c.artifact(path)


def assess(valrefs,selectionref,timingrefs,binding,base=None):
    base=Path(c.OUTPUT if base is None else base);c.budget_boundary(base)
    c.require(not (base/"assessment_result.json").exists(),"Preserve assessment")
    selected=_sealed_selection(selectionref,binding,base)
    c.require(selected["manifest"] == binding and selected["timing_roots"] == timingrefs,"Changed selection identity")
    _same(selected["cost_model"],cost_model(_read_roots(timingrefs,binding,"timing")))
    result={"manifest":binding,"selection":selectionref,"validation_roots":valrefs,"timing_roots":timingrefs,
        "statistics":summarize(_read_roots(valrefs,binding,"validation"),selected,boundary=lambda:c.budget_boundary(base)),
        "reused_development":True,"actual_estimator_run":False,"time":c.old.now()}
    path=base/"assessment_result.json";c.old.save_new(path,result);return c.artifact(path)


def audit(valrefs,selectionref,timingrefs,resultref,binding,base=None):
    base=Path(c.OUTPUT if base is None else base);c.budget_boundary(base)
    c.require(not (base/"assessment_audit.json").exists(),"Preserve assessment audit")
    result=c.verified(resultref);selected=_sealed_selection(selectionref,binding,base)
    c.require(result["manifest"] == selected["manifest"] == binding and result["selection"] == selectionref
              and result["validation_roots"] == valrefs and result["timing_roots"] == selected["timing_roots"] == timingrefs,
              "Assessment audit identity differs")
    _same(selected["cost_model"],cost_model(_read_roots(timingrefs,binding,"timing"),audit=True))
    expected=summarize(_read_roots(valrefs,binding,"validation"),selected,audit=True,boundary=lambda:c.budget_boundary(base))
    _same(result["statistics"],expected)
    receipt={"manifest":binding,"result":resultref,"passed":True,"all_moments_and_statistics_independently_reproduced":True,
             "moments_independently_reconstructed":True,"bootstrap_independently_reproduced":True,
             "fixed_train_rule_preserved":True,
             "selection":selectionref,"timing_uncertainty_in_ci":False,"time":c.old.now()}
    path=base/"assessment_audit.json";c.old.save_new(path,receipt);return c.artifact(path)
