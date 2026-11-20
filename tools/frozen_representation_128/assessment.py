"""Fixed predictions and old-label diagnostics precede reused outcome access.

No fit, game, world, or rollout is generated here. Pure mature encoding and
outcome primitives are reused without modifying their modules or globals.
"""
from pathlib import Path
import math
import time
from . import common as c, statistics as st
from tools.soft_teacher_targets_128 import assessment as prior
from splendor_ai.alphazero.strong_policy import StrongPolicyInference

old,OUTPUT=c.old,c.OUTPUT
compare=prior.compare


def _models(binding,base):
    from . import source,learning
    registry=learning.validate_models(binding,base)
    reference=c.artifact(Path(base)/"models_lock.json")
    c.require(c.verified(reference) == registry and registry["manifest"] == binding, "Model seal differs")
    expected={f"{family}_{fit}" for family in c.MODEL_ARMS for fit in c.FITS}
    c.require(set(registry["models"]) == expected, "Incomplete nine-model seal")
    for name,ref in source.control_models(binding,base).items():
        c.require(registry["models"][name] == ref, "Frozen control identity changed")
    models={}
    for name,ref in registry["models"].items():
        c.require(ref["source_fit"] == int(name.rsplit("_",1)[1]), "Wrong model fit")
        models[name]=StrongPolicyInference(c.verify_hash(ref)).model
        models[name].eval()
    return reference,registry,models


def _panel(binding,base):
    from . import source
    rows=sorted(source.load_panel_rows(binding,base),key=lambda r:r.get("index",r.get("globalindex")))
    inputs,descriptors=source.old_inputs(binding,base),source.evaluation_sources(binding,base)
    c.require(len(rows) == len(inputs) == len(descriptors) > 0, "Incomplete old-only panel")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        c.require(len(rows) == 336, "Complete production panel required")
    for index,(row,item,descriptor) in enumerate(zip(rows,inputs,descriptors,strict=True)):
        c.require(row.get("index",row.get("globalindex")) == item["index"] == descriptor["index"] == index
                  and item["spec"] == descriptor["spec"] and row["source_fit"] == item["spec"]["fit"]
                  and row["id"] == item["spec"]["source_rowid"] and row["split"] == item["spec"]["split"],
                  "Changed panel ordering, split or assigned fit")
    return rows,inputs,descriptors


def _choices(totals,missing,proposals,*,audit=False):
    n=len(totals)
    c.require(1 <= n <= 8 and len(missing) == n and set(proposals) == set(c.MODEL_ARMS), "Invalid proposal support")
    best=prior._independent_best if audit else prior._best
    result={"teacher":best(totals,missing,list(range(n))),
            "teacher_outsider":best(totals,missing,list(range(3,n)))}
    for family,proposal in proposals.items():
        scores=proposal["offered_group_scores"]
        c.require(len(scores) == n and all(type(v) in (int,float) and math.isfinite(v) for v in scores), "Invalid offered scores")
        if audit:
            outsider=sorted(range(3,n),key=lambda j:(-scores[j],j))[0] if n > 3 else None
        else:
            outsider=max(range(3,n),key=lambda j:scores[j]) if n > 3 else None
        support=list(range(min(3,n)))+([] if outsider is None else [outsider])
        result[family+"_outsider"]=outsider
        result["anchor" if family == "anchor" else family+"_supported"]=best(totals,missing,support)
    c.require(result["anchor_outsider"] == (3 if n > 3 else None), "Original assigned anchor fourth changed")
    return result


def _score(spec,choices,worlds,*,audit=False):
    n=len(worlds[0]["outcomes"])
    totals,missing=(prior._independent_columns if audit else prior._columns)(worlds,n)
    bounds={}
    for name,index in choices.items():
        if index is None:
            bounds[name]=None
        elif audit:
            outcomes=[w["outcomes"][index]["wdl"] for w in worlds]
            bounds[name]=[sum((0 if v is None else (1+v)/2) for v in outcomes)/128,
                          sum((1 if v is None else (1+v)/2) for v in outcomes)/128]
        else:
            bounds[name]=[(128+totals[index]-missing[index])/256,(128+totals[index]+missing[index])/256]
    deltas={}
    for name,left,right in st.CONTRASTS:
        if choices[left] == choices[right]:
            deltas[name]=[0.,0.]
        else:
            a,b=bounds[left],bounds[right]
            c.require(a is not None and b is not None,"Asymmetric absent outsider")
            deltas[name]=[a[0]-b[1],a[1]-b[0]]
    return {**{key:spec[key] for key in ("index","pair","game_index","fit","seat","phase","split")},
            "privacy":spec["privacy"],"support":n,"outsider_available":n > 3,"choices":choices,
            "arm_score_bounds":bounds,"arm_scores":{name:v[0] if v is not None and v[0] == v[1] else None
                                                       for name,v in bounds.items()},
            "delta_bounds":deltas,"unknown_counts":missing}


def _pair_metrics(records,*,audit=False):
    pairs=sorted({r["pair"] for r in records})
    reducer=math.fsum if audit else sum
    fields={"mean_loss":"loss","mean_soft_loss":"soft_loss","mean_target_kl":"target_kl",
            "mean_policy_entropy":"policy_entropy","mean_target_entropy":"target_entropy"}
    result={"pairs":len(pairs)}
    for target,field in fields.items():
        result[target]=reducer(reducer(r[field] for r in records if r["pair"] == pair)
                    /sum(r["pair"] == pair for r in records) for pair in pairs)/len(pairs)
    result["mean_old_score_regret"]=reducer(reducer(r["old_regret_units"] for r in records if r["pair"] == pair)
                    /(256*sum(r["pair"] == pair for r in records)) for pair in pairs)/len(pairs)
    movement=[r.get("initial_policy_kl") for r in records]
    c.require(all(v is None for v in movement) or all(type(v) in (int,float) and math.isfinite(v) for v in movement),
              "Partial policy movement report")
    result["mean_initial_policy_kl"]=None if movement[0] is None else reducer(
        reducer(r["initial_policy_kl"] for r in records if r["pair"] == pair)
        /sum(r["pair"] == pair for r in records) for pair in pairs)/len(pairs)
    return result


def _validation(binding,registry,models,base,*,audit=False):
    from . import source,learning
    rows=source.load_rows(binding,"validation",base)
    c.require(rows and len({r["id"] for r in rows}) == len(rows)
              and all(r["split"] == "validation" for r in rows), "Invalid original validation corpus")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        c.require(len(rows) == 2489 and len({r["pair"] for r in rows}) == 42, "Complete original validation required")
    families={};initial_reports={}
    for family in c.MODEL_ARMS:
        reports={}
        for fit in c.FITS:
            c.budget_boundary(base)
            name=f"{family}_{fit}"
            report=learning.training_report(models[name],rows,base,learning.CONFIG,
                       initial=initial_reports.get(fit),audit=audit,
                       data_scope="validation",diagnostic_scope="all_original_validation_each_fit")
            if family == "anchor": initial_reports[fit]=report
            c.require(report["validation_read"] is True and report["assessment_read"] is False
                      and len(report["rows"]) == len(rows), "Diagnostic scope changed")
            reports[str(fit)]={"model":registry["models"][name],"states":len(rows),"rows":report["rows"],
                              "metrics":report["metrics"],"pair_weighted":_pair_metrics(report["rows"],audit=audit)}
        families[family]={"per_fit":reports,"mean_pair_weighted_regret":math.fsum(
            r["pair_weighted"]["mean_old_score_regret"] for r in reports.values())/len(reports)}
    return {"manifest":binding,"states":len(rows),"pairs":len({r["pair"] for r in rows}),"families":families,
            "all_models_on_all_rows":True,"old_targets_only":True,"assessment_read":False,
            "checkpoint_selection":False,"reused_development":True}


def _acquisition(registry):
    values={}
    for ref in registry["fits"]:
        fit=c.verified(ref)
        report=c.verified(fit["selected_training"])
        c.require(fit["fit"] in c.FITS and str(fit["fit"]) not in values,"Duplicate or unknown fitted model")
        values[str(fit["fit"])]=report["added"]["regret_capture_vs_initializer"]
    c.require(set(values) == set(map(str,c.FITS)),"Missing acquisition reports")
    return values


def _open_assessment(descriptor,prediction,lock_ref,base):
    c.require(Path(base,"predictions_lock.json").exists(),"Seal all predictions before opening outcomes")
    lock=c.verified(lock_ref)
    c.require(lock["complete"] is True and lock["states"] == len(lock["predictions"])
              and lock["predictions"][descriptor["index"]] == prediction,"Unsealed or incomplete prediction")
    root,receipt=c.verified(descriptor["root"]),c.verified(descriptor["audit"])
    c.require(root["manifest"] == receipt["manifest"] == c.SOURCE_BINDING and receipt["passed"] is True
              and prior._same_ref(receipt["root"],descriptor["root"]) and "failure" not in root
              and root["stage"] == "main" and root["spec"] == descriptor["spec"]
              and prior._same_ref(root["selection_artifact"],prediction["source_selection"]),"Changed inherited outcome identity")
    block=root["assessment"]
    seeds=list(range(2211000000+512*descriptor["index"],2211000000+512*descriptor["index"]+128))
    c.require(block["candidate_indices"] == list(range(len(root["candidates"]["keys"])))
              and block["world_seeds"] == [w["seed"] for w in block["worlds"]] == seeds,"Changed inherited outcome mapping")
    return root


def _context(row,root):
    for arm in ("anchor","teacher","teacher_outsider","anchor_outsider"):
        c.require(row["choices"][arm] == root["selection_lock"]["choices"][arm]
                  and row["arm_score_bounds"][arm] == root["summary"]["arm_score_bounds"][arm],"Historical control changed")
    c.require(row["delta_bounds"]["teacher_minus_anchor"] == root["summary"]["delta_bounds"]["teacher_minus_anchor"],
              "Teacher positive control changed")


def assess(binding,base=OUTPUT):
    base=Path(base);c.budget_boundary(base)
    c.require(not (base/"predictions_lock.json").exists() and not (base/"assessment_result.json").exists(),"Preserve assessment evidence")
    model_ref,registry,models=_models(binding,base)
    rows,inputs,descriptors=_panel(binding,base)
    predictions=[]
    for row,item in zip(rows,inputs,strict=True):
        c.budget_boundary(base)
        _,totals,missing=prior._old_input(item)
        fit=item["spec"]["fit"]
        proposals={family:prior._predict(row,models[f"{family}_{fit}"]) for family in c.MODEL_ARMS}
        predictions.append({"index":item["index"],"fit":fit,"row_id":row["id"],"spec":item["spec"],
            "models":{family:registry["models"][f"{family}_{fit}"] for family in c.MODEL_ARMS},
            "source_selection":item["selection_artifact"],"proposals":proposals,"choices":_choices(totals,missing,proposals)})
    validation=_validation(binding,registry,models,base)
    old.save_new(base/"validation_old_targets.json",validation)
    validation_ref=c.artifact(base/"validation_old_targets.json")
    lock={"manifest":binding,"models":model_ref,"complete":True,"states":len(predictions),"predictions":predictions,
          "validation_old_targets":validation_ref,"created_at_ns":time.time_ns(),"time":old.now(),
          "selection_used_assessment":False,"all_predictions_precede_assessment_open":True,
          "source_fit_assignment":"one_assigned_model_per_root"}
    old.save_new(base/"predictions_lock.json",lock)
    lock_ref=c.artifact(base/"predictions_lock.json")
    opened=time.time_ns();scored=[]
    for descriptor,prediction in zip(descriptors,predictions,strict=True):
        c.budget_boundary(base)
        root=_open_assessment(descriptor,prediction,lock_ref,base)
        row=_score(descriptor["spec"],prediction["choices"],root["assessment"]["worlds"])
        _context(row,root);scored.append(row)
    statistics=st.summarize(scored,boundary=lambda:c.budget_boundary(base))
    acquisition=_acquisition(registry)
    result={"manifest":binding,"models":model_ref,"predictions":lock_ref,"validation_old_targets":validation_ref,
            "rows":scored,"statistics":statistics,"acquisition":acquisition,
            "development_screen":st.development_screen(statistics,validation,acquisition),
            "assessment_opened_at_ns":opened,"inherited_assessments":descriptors,"new_simulations":0,
            "new_source_games":0,"reused_development":True,"assessment_used_for_selection":False,
            "context_controls_reproduced":True,"classification":"diagnostic_only","promotion":False,"time":old.now()}
    old.save_new(base/"assessment_result.json",result)
    return c.artifact(base/"assessment_result.json")


def audit(binding,result_ref,base=OUTPUT):
    base=Path(base);c.budget_boundary(base)
    c.require(not (base/"assessment_audit.json").exists(),"Preserve assessment audit")
    result=c.verified(result_ref)
    c.require(result["manifest"] == binding and result["new_simulations"] == result["new_source_games"] == 0
              and result["reused_development"] is True and result["assessment_used_for_selection"] is False
              and result["classification"] == "diagnostic_only" and result["promotion"] is False,"Changed assessment scope")
    lock=c.verified(result["predictions"])
    model_ref,registry,models=_models(binding,base)
    c.require(lock["manifest"] == binding and result["models"] == lock["models"] == model_ref
              and lock["complete"] is True and lock["selection_used_assessment"] is False
              and lock["all_predictions_precede_assessment_open"] is True
              and lock["source_fit_assignment"] == "one_assigned_model_per_root"
              and lock["validation_old_targets"] == result["validation_old_targets"],"Changed global seal")
    c.require(c.verify_hash(model_ref).stat().st_mtime_ns <= c.verify_hash(result["validation_old_targets"]).stat().st_mtime_ns
              <= lock["created_at_ns"] <= c.verify_hash(result["predictions"]).stat().st_mtime_ns
              <= result["assessment_opened_at_ns"],"Prediction/outcome timing failed")
    rows,inputs,descriptors=_panel(binding,base)
    c.require(lock["states"] == len(rows) == len(lock["predictions"]) == len(result["rows"])
              and result["inherited_assessments"] == descriptors,"Changed assessment panel")
    independent=[]
    for row,item,descriptor,prediction in zip(rows,inputs,descriptors,lock["predictions"],strict=True):
        c.budget_boundary(base)
        fit=item["spec"]["fit"]
        c.require(prediction["index"] == item["index"] and prediction["fit"] == row["source_fit"] == fit
                  and prediction["row_id"] == row["id"] and prediction["spec"] == item["spec"]
                  and prediction["source_selection"] == item["selection_artifact"]
                  and prediction["models"] == {family:registry["models"][f"{family}_{fit}"] for family in c.MODEL_ARMS},
                  "Changed assigned prediction identity")
        seal,_,_=prior._old_input(item)
        totals,missing=prior._independent_columns(seal["old_selection"]["worlds"],len(item["candidates"]["keys"]))
        state=old.deserialize_state_snapshot(item["snapshot"])
        alternate=prior.input_audit.hidden_alternative(state)
        proposals={}
        for family in c.MODEL_ARMS:
            model=models[f"{family}_{fit}"]
            proposed=prior._independent_proposal(row,item,model)
            compare(prediction["proposals"][family],proposed)
            compare(proposed,prior._independent_proposal(row,item,model,state=alternate))
            proposals[family]=proposed
        choices=_choices(totals,missing,proposals,audit=True)
        c.require(prediction["choices"] == choices,"Audited proposal choices differ")
        # Separate auditor access path after confirming the complete global seal.
        root,receipt=c.verified(descriptor["root"]),c.verified(descriptor["audit"])
        c.require(root["manifest"] == receipt["manifest"] == c.SOURCE_BINDING and receipt["passed"] is True
                  and prior._same_ref(receipt["root"],descriptor["root"]) and "failure" not in root
                  and root["stage"] == "main" and root["spec"] == item["spec"]
                  and root["selection_lock"] == seal["selection_lock"]
                  and prior._same_ref(root["selection_artifact"],item["selection_artifact"]),"Changed inherited audit linkage")
        block=root["assessment"];n=len(item["candidates"]["keys"])
        seeds=[2211000000+512*item["index"]+j for j in range(128)]
        c.require(block["candidate_indices"] == list(range(n)) and block["world_seeds"]
                  == [w["seed"] for w in block["worlds"]] == seeds,"Changed assessment namespace")
        scored=_score(item["spec"],choices,block["worlds"],audit=True)
        c.require(scored == result["rows"][item["index"]],"Independent outcome arithmetic differs")
        _context(scored,root);independent.append(scored)
    statistics=st.independent_summarize(independent,boundary=lambda:c.budget_boundary(base))
    compare(result["statistics"],statistics,rel_tol=0.,abs_tol=1e-12)
    validation=_validation(binding,registry,models,base,audit=True)
    compare(c.verified(result["validation_old_targets"]),validation,rel_tol=0.,abs_tol=1e-12)
    acquisition=_acquisition(registry)
    compare(result["acquisition"],acquisition,rel_tol=0.,abs_tol=1e-12)
    compare(result["development_screen"],st.development_screen(statistics,validation,acquisition),rel_tol=0.,abs_tol=1e-12)
    c.require(result["context_controls_reproduced"] is True,"Changed historical controls flag")
    receipt={"manifest":binding,"result":result_ref,"passed":True,"time":old.now(),
             "all_predictions_independently_reproduced":True,"all_private_input_checks_pass":True,
             "full_validation_old_label_metrics_independently_reproduced":True,
             "all_outcome_arithmetic_independently_reproduced":True,"statistics_independently_reproduced":True,
             "fixed_controls_reproduced":True,"source_fit_assignment":"one_assigned_model_per_root",
             "new_simulations":0,"reused_development":True,"promotion":False}
    old.save_new(base/"assessment_audit.json",receipt)
    return c.artifact(base/"assessment_audit.json")
