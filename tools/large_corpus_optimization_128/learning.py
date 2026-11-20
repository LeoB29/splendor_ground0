"""Fixed continuation of authenticated large-corpus model and AdamW states.

The exact 7136-update batch plan is repeated four times. Historical model
reproduction is inherited; AdamW moments are authenticated writer artifacts,
structurally checked here. Every new update and saved optimizer checkpoint is
independently reproduced. No historical RNG state is fabricated: the inherited
dropout-free training uses precomputed deterministic batches and no RNG draws.
"""
from pathlib import Path
from dataclasses import asdict
import os
import uuid

import torch

from splendor_ai.alphazero.fast_selector_157 import collate
from scripts.analysis.verify_fast_selector_transfer_157 import audit_batch_tensors
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, MODEL_KIND
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.independent_game_scaling_128 import learning as inherited
from . import common as c, training_source as source

FITS,OUTPUT,old = c.FITS,c.OUTPUT,c.old
ARMS = ("continued",)
CONFIG = {**inherited.CONFIG, **c.CONFIG, "fresh_optimizer":False,"restore_saved_AdamW":True,
          "continuation_repeats":4,"historical_rng_state_present":False,
          "diagnostic_checkpoints":"inherited_start_and_three_fixed_continuation_endpoints",
          "historical_optimizer_moments_independently_replayed":False}
_compare,_equal_torch = inherited._compare,inherited._equal_torch
_new_json_or_equal,_checkpoint_model = inherited._new_json_or_equal,inherited._checkpoint_model
_check_frozen,_rows_identity = inherited._check_frozen,inherited._rows_identity
objective,audit_objective = inherited.objective,inherited.audit_objective
training_report = inherited._clone(inherited.prior.training_report,{"c":c})
_model = inherited._clone(inherited._model,{"c":c,"ARMS":ARMS})
_diagnostics = inherited._clone(inherited._diagnostics,{"c":c,"training_report":training_report})


def _base(base):
    return Path(OUTPUT if base is None else base)


def _inputs(fit_id,binding,base,rows,initializer,config,partitions,continuation):
    supplied = (rows,initializer,config,partitions,continuation)
    fixture = any(v is not None for v in supplied)
    c.require(fit_id in FITS,"Unknown continuation fit")
    if fixture:
        c.require(not base.resolve().is_relative_to(OUTPUT.resolve()),"Fixture cannot target production")
        c.require(all(v is not None for v in supplied),"Complete fixture inputs required")
        pools,initial,settings,subsets,record = rows,initializer,dict(config),partitions,continuation
    else:
        pools,initial = source.load_pools(binding,base),source.initializer(fit_id,base)
        settings,subsets,record = dict(CONFIG),source.partitions(binding,base),source.continuation(fit_id,binding,base)
    parent = source.validate_continuation(record,fit_id,expected_update=settings["start_update"])
    # Reuse the predecessor's complete TRAIN/cell/target firewall with its own
    # authenticated recipe, while forbidding any production fixture override.
    c.require(set(pools) == {"old","added"} and set(subsets) == {"small","medium","large"},"Unexpected pools")
    for values in pools.values():
        inherited._validate_rows(values)
        c.require(all(not {"teacher_worlds","teacher_sums_wdl","teacher_outsider"} & row.keys() for row in values),
                  "Only original128 teacher targets may enter continuation")
    c.require(initial == record["initializer"] and parent["identity"]["partitions"] == subsets
              and parent["identity"]["old_rows_sha256"] == _rows_identity(pools["old"])
              and parent["identity"]["added_rows_sha256"] == _rows_identity(pools["added"]),
              "Continuation corpus/initializer differs from original fit")
    c.require(subsets["large"] == list(range(len(pools["added"]))),"Incomplete large pool")
    recipe = ("batch_size","target_temperature","old_fraction","objective","lr","weight_decay",
              "gradient_norm","value_head_frozen","torch_threads","dtype","device","train_seeds",
              "batch_tag","reference_batch_tag","curve_tag","curve_old_rows","curve_added_rows_per_cell")
    c.require(all(settings[key] == parent["config"][key] for key in recipe),"Inherited learning recipe changed")
    c.require(settings["start_update"] == parent["config"]["updates"]
              and settings["updates"] == 4*settings["start_update"]
              and settings["extra_updates"] == 3*settings["start_update"]
              and settings["diagnostic_updates"] == [settings["start_update"]*i for i in range(1,5)]
              and settings["continuation_repeats"] == 4 and settings["fresh_optimizer"] is False
              and settings["restore_saved_AdamW"] is True and not settings["selection_uses_intermediate_checkpoints"],
              "Fixed continuation dose/selection changed")
    if not fixture:
        c.require(settings == CONFIG and len(pools["old"]) == 9461 and len(pools["added"]) == 11520
                  and parent["config"] == inherited.CONFIG,"Production continuation recipe changed")
    return pools,initial,settings,subsets,record,parent,fixture


def _plans(pools,subsets,fit_id,parent,settings,*,audit=False):
    plans = (inherited.independent_plan if audit else inherited.batch_plan)(pools,subsets,fit_id,parent["config"])
    paired = inherited._paired_receipt(plans,pools,parent["config"],fixture=settings["start_update"] != 7136)
    _compare(paired,parent["identity"]["paired_batch_receipt"])
    prefix = plans["large"]
    if audit:
        full = [list(prefix[i % len(prefix)]) for i in range(settings["updates"])]
    else:
        full = [list(batch) for _ in range(settings["continuation_repeats"]) for batch in prefix]
    c.require(full[:settings["start_update"]] == prefix and len(full) == settings["updates"],"Changed historical batch prefix")
    summary = inherited._plan_receipt(full,pools)
    original = paired["arms"]["large"]
    for key in ("row_exposures","old_exposures","added_exposures"):
        c.require(summary[key] == 4*original[key],"Continuation dose differs")
    for key in ("per_row_exposures","per_added_game_exposures","per_old_game_exposures","per_added_cell_exposures"):
        c.require(summary[key] == {name:4*value for name,value in original[key].items()},"Continuation exposure allocation changed")
    return full,{"prefix":original,"total":summary,"exact_four_repeats":True,"historical_prefix_unchanged":True}


def _restore_start(model,optimizer,frozen,names,record,parent,settings):
    payload = torch.load(c.verify_hash(record["optimizer"]),map_location="cpu",weights_only=True)
    c.require(names == parent["trainable_parameter_names"] == payload["identity"]["trainable_parameter_names"],
              "Optimizer parameter order changed")
    source.validate_adamw(payload,names,parent["config"],settings["start_update"])
    model.load_state_dict(payload["model_state_dict"],strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    c.require(_equal_torch(optimizer.state_dict(),payload["optimizer_state_dict"]),"AdamW restore differs")
    _check_frozen(model,frozen)
    return list(payload["preclip_gradient_norms"])


def _identity(fit_id,binding,initial,settings,record,parent,pools,plan_receipt,names,parameters):
    return {"manifest":binding,"fit":fit_id,"arm":"continued","initializer":initial,"config":settings,
        "continuation":record,"parent_identity_sha256":old.digest(parent["identity"]),
        "old_rows_sha256":_rows_identity(pools["old"]),"added_rows_sha256":_rows_identity(pools["added"]),
        "prefix_batch_sequence_sha256":plan_receipt["prefix"]["batch_sequence_sha256"],
        "batch_sequence_sha256":plan_receipt["total"]["batch_sequence_sha256"],
        "trainable_parameter_names":names,"trainable_parameters":sum(p.numel() for p in parameters)}


def _metadata(fit_id,binding,update):
    return {"experiment":"large_corpus_optimization_128","manifest":binding,"fit":fit_id,"arm":"continued",
            "update":update,"checkpoint_selection":"fixed_final","assessment_used":False,"value_head_frozen":True}


def _progress_updates(settings):
    start,end = settings["start_update"],settings["updates"]
    return sorted({start,start+1,end,*settings["diagnostic_updates"],
                   *(u for u in range(start+1,end+1) if u % settings["resume_checkpoint_interval"] == 0)})


def _save_progress(folder,model,optimizer,update,norms,identity):
    path = folder/f"optimizer_{update:05d}.pt"
    payload = {"identity":identity,"update":update,"model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),"preclip_gradient_norms":list(norms),
        "torch_rng_state":torch.get_rng_state(),"historical_rng_state_fabricated":False}
    inherited.checkpoint_helpers._new_torch_or_equal(path,payload)
    pointer = {"identity_sha256":old.digest(identity),"update":update,"checkpoint":c.artifact(path)}
    temporary = folder/("resume."+uuid.uuid4().hex+".tmp")
    old.save_new(temporary,pointer)
    os.replace(temporary,folder/"resume.json")


def _resume(folder,identity,model,optimizer,frozen,names,settings):
    pointer = old.read_json(folder/"resume.json")
    payload = torch.load(c.verify_hash(pointer["checkpoint"]),map_location="cpu",weights_only=True)
    update = pointer["update"]
    c.require(pointer["identity_sha256"] == old.digest(identity) and payload["identity"] == identity
              and payload["update"] == update and update in _progress_updates(settings)
              and len(payload["preclip_gradient_norms"]) == update,"Continuation resume identity changed")
    source.validate_adamw(payload,names,settings,update)
    model.load_state_dict(payload["model_state_dict"],strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"])
    _check_frozen(model,frozen)
    return update,list(payload["preclip_gradient_norms"])


def _original_reports(record):
    value = c.verified(record["initial_training"])
    c.require(value["update"] == 0 and not value["assessment_read"] and not value["validation_read"],
              "Original acquisition denominator report differs")
    return value["full"],value["probe"]


def fit(fit_id,binding,base=None,*,rows=None,initializer=None,config=None,partitions=None,continuation=None):
    base = _base(base); c.budget_boundary(base)
    pools,initial,settings,subsets,record,parent,fixture = _inputs(
        fit_id,binding,base,rows,initializer,config,partitions,continuation)
    plan,receipt = _plans(pools,subsets,fit_id,parent,settings)
    model,parameters,optimizer,frozen,names = _model(initial,"continued",settings["train_seeds"][str(fit_id)],settings)
    norms = _restore_start(model,optimizer,frozen,names,record,parent,settings)
    identity = _identity(fit_id,binding,initial,settings,record,parent,pools,receipt,names,parameters)
    folder = base/"fits"/f"fit_{fit_id}"
    folder.mkdir(parents=True,exist_ok=True)
    result_path = folder/"result.json"
    if result_path.exists():
        result = old.read_json(result_path)
        c.require(result["identity"] == identity and result["complete"],"Completed continuation differs")
        for point in result["checkpoints"]:
            for key in ("checkpoint","training","receipt"):c.verify_hash(point[key])
        for ref in result["optimizer_checkpoints"]:c.verify_hash(ref)
        return c.artifact(result_path,fit=fit_id,index=FITS.index(fit_id))
    original_full,original_probe = _original_reports(record)
    selected = subsets["large"]

    def save(update):
        c.budget_boundary(base)
        full = _diagnostics(model,pools,selected,base,settings,original_full) if update in (settings["start_update"],settings["updates"]) else None
        probe = _diagnostics(model,pools,selected,base,settings,original_probe,probe=True)
        report = {"full":full,"probe":probe,"update":update,"assessment_read":False,"validation_read":False}
        if update == settings["start_update"]:_compare(report,c.verified(record["final_training"]))
        checkpoint,report_path = folder/f"checkpoint_{update:05d}.pt",folder/f"training_{update:05d}.json"
        _checkpoint_model(checkpoint,model,_metadata(fit_id,binding,update))
        _new_json_or_equal(report_path,report)
        point = {"update":update,"checkpoint":c.artifact(checkpoint,source_fit=fit_id,arm="continued"),
                 "training":c.artifact(report_path),"identity_sha256":old.digest(identity)}
        path = folder/f"point_{update:05d}.json"
        _new_json_or_equal(path,point)
    save(settings["start_update"])
    if not (folder/"resume.json").exists():
        _save_progress(folder,model,optimizer,settings["start_update"],norms,identity)
    start,norms = _resume(folder,identity,model,optimizer,frozen,names,settings)
    progress = set(_progress_updates(settings))
    reporter = StatusReporter(folder,{"manifest":binding,"fit":fit_id,"arm":"continued"})
    for step in range(start,len(plan)):
        c.budget_boundary(base)
        batch = [pools[kind][index] for kind,index in plan[step]]
        optimizer.zero_grad(set_to_none=True)
        logits,_ = model(*collate(batch))
        loss = objective(logits,batch,settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)),"Nonfinite continuation loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters,settings["gradient_norm"],error_if_nonfinite=True)))
        optimizer.step(); _check_frozen(model,frozen)
        c.require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()),"Nonfinite continuation model")
        update = step+1
        if update in settings["diagnostic_updates"]:save(update)
        if update in progress:
            _save_progress(folder,model,optimizer,update,norms,identity)
            reporter.event("large_corpus_continuation",update=update,total_updates=settings["updates"],resumed_at_update=start)
    c.require(len(norms) == settings["updates"],"Incomplete continuation")
    points = []
    for update in settings["diagnostic_updates"]:
        ref = c.artifact(folder/f"point_{update:05d}.json"); point = c.verified(ref)
        c.require(point["identity_sha256"] == old.digest(identity) and point["update"] == update,"Point identity changed")
        c.verify_hash(point["checkpoint"]); c.verify_hash(point["training"])
        points.append({**point,"receipt":ref})
    optimizer_refs = [c.artifact(folder/f"optimizer_{update:05d}.pt",update=update) for update in _progress_updates(settings)]
    result = {"manifest":binding,"fit":fit_id,"arm":"continued","identity":identity,"config":settings,
        "initializer":initial,"continuation":record,"checkpoints":points,"optimizer_checkpoints":optimizer_refs,
        "selected_update":settings["updates"],"selected_model":points[-1]["checkpoint"],"selected_training":points[-1]["training"],
        "batch_receipt":receipt,"trainable_parameter_names":names,"trainable_parameters":sum(p.numel() for p in parameters),
        "optimization_stability":inherited.prior._stability(norms,settings),
        "historical_optimizer_moments_independently_replayed":False,"AdamW_state_continued":True,
        "value_head_frozen":True,"model_always_eval":True,"final_always_selected":True,"complete":True,
        "assessment_read":False,"validation_read":False,"original_initializer_acquisition_reference":True}
    old.save_new(result_path,result)
    return c.artifact(result_path,fit=fit_id,index=FITS.index(fit_id))


AUDIT_FLAGS = (*inherited.AUDIT_FLAGS,"all_new_optimizer_checkpoints_independently_reproduced",
               "exact_inherited_schedule_prefix","original_initializer_acquisition_reference","AdamW_state_continued")


def audit_fit(fit_id,fit_ref,binding,base=None,*,rows=None,initializer=None,config=None,partitions=None,continuation=None):
    base = _base(base); c.budget_boundary(base)
    pools,initial,settings,subsets,record,parent,fixture = _inputs(
        fit_id,binding,base,rows,initializer,config,partitions,continuation)
    plan,receipt = _plans(pools,subsets,fit_id,parent,settings,audit=True)
    model,parameters,optimizer,frozen,names = _model(initial,"continued",settings["train_seeds"][str(fit_id)],settings,audit=True)
    norms = _restore_start(model,optimizer,frozen,names,record,parent,settings)
    identity = _identity(fit_id,binding,initial,settings,record,parent,pools,receipt,names,parameters)
    expected = c.verified(fit_ref)
    c.require(expected["identity"] == identity and expected["manifest"] == binding and expected["fit"] == fit_id
              and expected["config"] == settings and expected["initializer"] == initial and expected["continuation"] == record
              and expected["complete"] and expected["arm"] == "continued","Continuation fit identity changed")
    _compare(expected["batch_receipt"],receipt)
    c.require(expected["trainable_parameter_names"] == names and expected["trainable_parameters"] == sum(p.numel() for p in parameters),
              "Trainable membership changed")
    points = expected["checkpoints"]
    c.require([p["update"] for p in points] == settings["diagnostic_updates"],"Unplanned diagnostic points")
    original_full,original_probe = _original_reports(record)
    points_by_update = {p["update"]:p for p in points}
    optimizer_refs = expected["optimizer_checkpoints"]
    c.require([r["update"] for r in optimizer_refs] == _progress_updates(settings),"Optimizer checkpoint schedule changed")
    optimizers = {r["update"]:r for r in optimizer_refs}

    def inspect(update):
        if update in points_by_update:
            point = points_by_update[update]
            c.require(c.verified(point["receipt"]) == {k:v for k,v in point.items() if k != "receipt"}
                      and point["identity_sha256"] == old.digest(identity),"Point receipt changed")
            payload = torch.load(c.verify_hash(point["checkpoint"]),map_location="cpu",weights_only=True)
            c.require(payload["metadata"] == _metadata(fit_id,binding,update) and payload["generation"] == update
                      and payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(model.config)
                      and _equal_torch(payload["model_state_dict"],model.state_dict()),"Independent continuation trajectory differs")
            with torch.random.fork_rng():
                loaded = StrongPolicyInference(str(c.verify_hash(point["checkpoint"]))).model
                c.require(_equal_torch(loaded.state_dict(),model.state_dict()),"Inference checkpoint differs")
            full = _diagnostics(model,pools,subsets["large"],base,settings,original_full,audit=True) if update in (settings["start_update"],settings["updates"]) else None
            probe = _diagnostics(model,pools,subsets["large"],base,settings,original_probe,audit=True,probe=True)
            report = {"full":full,"probe":probe,"update":update,"assessment_read":False,"validation_read":False}
            _compare(report,c.verified(point["training"]))
            if update == settings["start_update"]:_compare(report,c.verified(record["final_training"]))
        if update in optimizers:
            saved = torch.load(c.verify_hash(optimizers[update]),map_location="cpu",weights_only=True)
            c.require(saved["identity"] == identity and saved["update"] == update
                      and saved["historical_rng_state_fabricated"] is False
                      and _equal_torch(saved["model_state_dict"],model.state_dict())
                      and _equal_torch(saved["optimizer_state_dict"],optimizer.state_dict())
                      and _equal_torch(saved["torch_rng_state"],torch.get_rng_state())
                      and saved["preclip_gradient_norms"] == norms,"Independent optimizer checkpoint differs")
        _check_frozen(model,frozen)
    inspect(settings["start_update"])
    reporter = StatusReporter(base/"fits"/f"fit_{fit_id}"/"audit_status",{"manifest":binding,"fit":fit_id})
    for step in range(settings["start_update"],settings["updates"]):
        c.budget_boundary(base)
        batch = [pools[kind][index] for kind,index in plan[step]]
        optimizer.zero_grad(set_to_none=True)
        logits,_ = model(*audit_batch_tensors(batch))
        loss = audit_objective(logits,batch,settings["target_temperature"])[0].mean()
        c.require(bool(torch.isfinite(loss)),"Nonfinite audit loss")
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(parameters,settings["gradient_norm"],error_if_nonfinite=True)))
        optimizer.step(); inspect(step+1)
        if step+1 in optimizers:reporter.event("large_corpus_continuation_reproduction",update=step+1,total_updates=settings["updates"])
    stability = inherited.prior._stability(norms,settings,audit=True)
    _compare(stability,expected["optimization_stability"])
    c.require(expected["selected_update"] == settings["updates"] and expected["selected_model"] == points[-1]["checkpoint"]
              and expected["selected_training"] == points[-1]["training"]
              and all(expected[k] is True for k in ("value_head_frozen","model_always_eval","final_always_selected",
                        "original_initializer_acquisition_reference","AdamW_state_continued"))
              and expected["historical_optimizer_moments_independently_replayed"] is False
              and not expected["assessment_read"] and not expected["validation_read"],"Continuation claims changed")
    result = {"manifest":binding,"fit":fit_id,"arm":"continued","result":fit_ref,
        **{key:True for key in AUDIT_FLAGS},"selected_update":settings["updates"],
        "selected_model":expected["selected_model"],"selected_training":expected["selected_training"],
        "historical_optimizer_moments_independently_replayed":False,"assessment_read":False,"validation_read":False}
    path = base/"fits"/f"fit_{fit_id}"/"audit.json"
    _new_json_or_equal(path,result)
    return c.artifact(path,fit=fit_id,index=FITS.index(fit_id))


def _verified_fits(fits,audits,binding,base,controls=None):
    c.require(len(fits) == len(audits) == len(FITS),"All three continuations and audits required")
    results = []
    for index,(fit,fit_ref,audit_ref) in enumerate(zip(FITS,fits,audits,strict=True)):
        result,audit = c.verified(fit_ref),c.verified(audit_ref)
        c.require(fit_ref["index"] == audit_ref["index"] == index
                  and all(v["fit"] == fit for v in (fit_ref,audit_ref,result,audit))
                  and result["manifest"] == audit["manifest"] == binding and result["complete"]
                  and audit["result"] == fit_ref and all(audit[k] is True for k in AUDIT_FLAGS)
                  and result["selected_model"] == audit["selected_model"]
                  and result["selected_training"] == audit["selected_training"]
                  and result["selected_update"] == audit["selected_update"] == result["config"]["updates"],
                  "Missing or mismatched independent continuation evidence")
        c.require(not any(v[k] for v in (result,audit) for k in ("assessment_read","validation_read")),"Held-out selection")
        if base.resolve().is_relative_to(OUTPUT.resolve()):c.require(result["config"] == CONFIG,"Production recipe changed")
        if controls is not None:
            c.require(result["initializer"] == controls[f"anchor_{fit}"]
                      and result["continuation"]["model"] == controls[f"control_{fit}"],"Wrong initializer/control identity")
        c.verify_hash(result["selected_model"]); c.verify_hash(result["selected_training"])
        payload = torch.load(c.verify_hash(result["selected_model"]),map_location="cpu",weights_only=True)
        c.require(payload["model_kind"] == MODEL_KIND
                  and payload["metadata"] == _metadata(fit,binding,result["config"]["updates"])
                  and payload["generation"] == result["config"]["updates"],"Sealed model metadata changed")
        results.append(result)
    return results


def acquisition_gate(fits,audits,binding,base=None):
    base = _base(base)
    results = _verified_fits(fits,audits,binding,base)
    thresholds,per_fit = {"added":.95,"old":.90},{}
    for result in results:
        final = c.verified(result["selected_training"])["full"]
        original = c.verified(result["continuation"]["initial_training"])["full"]
        captures = {}
        for pool in ("old","added"):
            before = sum(row["old_regret_units"] for row in original[pool]["rows"])
            after = sum(row["old_regret_units"] for row in final[pool]["rows"])
            capture = (before-after)/before if before else None
            c.require(final[pool]["initial_old_regret_units"] == before
                      and final[pool]["metrics"]["all_rows"]["old_regret_units_sum"] == after
                      and final[pool]["regret_capture_vs_initializer"] == capture,
                      "Acquisition denominator differs from original initializer")
            captures[pool] = capture
        per_fit[str(result["fit"])] = {"added_capture":captures["added"],"old_capture":captures["old"],
            "passed":all(captures[p] is not None and captures[p] >= thresholds[p] for p in thresholds),
            "training":result["selected_training"],"original_training":result["continuation"]["initial_training"]}
    value = {"manifest":binding,"passed":all(v["passed"] for v in per_fit.values()),"thresholds":thresholds,
        "fits":per_fit,"fit_results":fits,"fit_audits":audits,"original_initializer_reference":True,
        "assessment_used":False,"automatic_extension":False}
    path = base/"acquisition_gate.json"; _new_json_or_equal(path,value)
    return c.artifact(path)


def models_lock(fits,audits,binding,base=None,*,controls=None):
    base = _base(base)
    gate_ref = acquisition_gate(fits,audits,binding,base)
    c.require(c.verified(gate_ref)["passed"],"Acquisition gate failed; no fresh model seal")
    if controls is None:controls = source.control_models(binding,base)
    else:c.require(not base.resolve().is_relative_to(OUTPUT.resolve()),"Fixture controls forbidden in production")
    c.require(set(controls) == {f"{arm}_{fit}" for arm in ("anchor","control") for fit in FITS},"Incomplete controls")
    results = _verified_fits(fits,audits,binding,base,controls)
    models = {**controls,**{f"continued_{r['fit']}":r["selected_model"] for r in results}}
    for ref in models.values():c.verify_hash(ref)
    value = {"manifest":binding,"models":models,"fits":fits,"audits":audits,"acquisition_gate":gate_ref,
        "complete":True,"all_new_fits_independently_reproduced":True,"all_value_heads_frozen":True,
        "selection":"all_three_continuations_fixed_final","assessment_used_for_selection":False}
    path = base/"models_lock.json"; _new_json_or_equal(path,value)
    return c.artifact(path)


def validate_models(binding,base=None):
    base = _base(base); value = old.read_json(base/"models_lock.json")
    c.require(value["manifest"] == binding and value["complete"] and value["all_new_fits_independently_reproduced"]
              and value["all_value_heads_frozen"] and value["selection"] == "all_three_continuations_fixed_final"
              and not value["assessment_used_for_selection"],"Invalid continuation model seal")
    gate_ref = acquisition_gate(value["fits"],value["audits"],binding,base)
    c.require(value["acquisition_gate"] == gate_ref and c.verified(gate_ref)["passed"],"Unpassed acquisition gate")
    controls = source.control_models(binding,base)
    results = _verified_fits(value["fits"],value["audits"],binding,base,controls)
    expected = {**controls,**{f"continued_{r['fit']}":r["selected_model"] for r in results}}
    c.require(value["models"] == expected,"Model registry differs from audited fixed endpoints")
    for ref in expected.values():c.verify_hash(ref)
    return value
