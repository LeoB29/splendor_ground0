"""Exact fourth-slot supervision, reused controls, and independent reproduction."""
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from tools.proposal_breadth_learning_128 import learning as inherited
from scripts.analysis import verify_fast_selector_transfer_157 as independent
from splendor_ai.alphazero.strong_policy import MODEL_KIND
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from . import common as c
from . import learning_data as data

ROOT, OUTPUT, SOURCE, FITS = c.ROOT, c.OUTPUT, c.SOURCE, c.FITS
old, f, previous = c.old, inherited.f, inherited.previous
ARMS = ("specialist", "control")
TRAIN_SEEDS = dict(inherited.TRAIN_SEEDS)
CONFIG = {**inherited.CONFIG,
    "objective":"historical_offered_CE_new_outside_exact_first_three_CE",
    "validation":"common_old_CE_new_outsider_CE_equal_whole_pair_mean",
    "zero_support_semantics":"empty_singleton_differentiable_zero_keep_denominator_steps",
    "control":"reuse_closed_broad_trajectories_no_training_new_common_reselection",
    "control_fit_audit":"inherit_verified_sealed_reproduction_new_validation_independent"}
source_evidence, prepare_data = data.source_evidence, data.prepare_data
batch_plan, independent_plan = inherited.batch_plan, inherited.independent_plan


def objective(logits, rows, arm="specialist"):
    assert arm in ARMS  # Both families use this SAME selection criterion.
    losses,choices = [],[]
    for i,row in enumerate(rows):
        count = len(row["members"])
        assert 1 <= count <= 8
        if row["pair"].startswith("breadth128:"):
            start,width,target = row["outsider_start"],row["outsider_support"],row["outsider_teacher"]
            assert start == min(3,count) and width == count-start
            assert (target == -1 if width == 0 else start <= target < count)
        else:
            start,width,target = 0,count,row["teacher"]
            assert 0 <= target < width
        if width < 2:
            # Restrict to a real legal logit: padding is -inf. Multiplication
            # by zero preserves backward/AdamW steps even for an all-zero batch.
            losses.append(logits[i,row["members"][0][0]]*0.)
            choices.append(target)
        else:
            scores = torch.stack([torch.amax(logits[i,member]) for member in row["members"][start:]])
            losses.append(F.cross_entropy(scores[None],torch.tensor([target-start])))
            choices.append(start+int(scores.detach().argmax()))
    return torch.stack(losses),choices


def audit_objective(logits, rows, arm="specialist"):
    assert arm in ("specialist","control")
    losses,choices = [],[]
    for i,row in enumerate(rows):
        n = len(row["members"])
        assert 1 <= n <= 8
        is_new = row["pair"].split(":",1)[0] == "breadth128"
        indices = list(range(min(3,n),n)) if is_new else list(range(n))
        target = row["outsider_teacher"] if is_new else row["teacher"]
        if is_new:
            assert row["outsider_start"] == min(3,n) and row["outsider_support"] == len(indices)
        assert (target == -1 if not indices else target in indices)
        if len(indices) < 2:
            # Independent differentiable zero over all finite legal logits.
            losses.append((logits[i,:len(row["features"])]*0.).sum())
            choices.append(target)
            continue
        groups = [int(row["groups"][row["members"][j][0]]) for j in indices]
        scores = torch.stack([torch.amax(logits[i,torch.where(row["groups"] == group)[0]]) for group in groups])
        losses.append(F.cross_entropy(scores.unsqueeze(0),torch.tensor([indices.index(target)])))
        choices.append(indices[int(scores.detach().argmax())])
    return torch.stack(losses),choices


def pair_loss(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["pair"]].append(record["loss"])
    assert groups
    return sum(sum(groups[key])/len(groups[key]) for key in sorted(groups))/len(groups)


def validation(model, rows, base=OUTPUT, *, audit=False):
    model.eval()
    records = []
    batches,collate = (independent.audit_batches,independent.audit_batch_tensors) if audit else (f.batches,f.collate)
    loss = audit_objective if audit else objective
    with torch.no_grad():
        for indices in batches(rows,157000,-1):
            c.budget_boundary(base)
            batch = [rows[i] for i in indices]
            logits,_ = model(*collate(batch))
            losses,choices = loss(logits,batch)
            for row,value,choice in zip(batch,losses.tolist(),choices,strict=True):
                assert math.isfinite(value)
                new = row["pair"].startswith("breadth128:")
                records.append({"id":row["id"],"pair":row["pair"],"loss":value,"choice":choice,
                    "teacher":row["teacher"],"criterion_teacher":row["outsider_teacher"] if new else row["teacher"],
                    "support":len(row["members"]),"criterion_support":row["outsider_support"] if new else len(row["members"]),
                    "seat":row["seat"],"phase":row["phase"],"privacy":row["privacy"]})
    records.sort(key=lambda row:row["id"])
    if audit:
        means = []
        for pair in sorted({row["pair"] for row in records}):
            values = [row["loss"] for row in records if row["pair"] == pair]
            means.append(sum(values)/len(values))
        value = sum(means)/len(means)
    else:
        value = pair_loss(records)
    return {"pair_loss":value,"rows":records}


def _inputs(binding, base, rows):
    if rows is None:
        return data.load_data(binding,base)
    location = Path(base).resolve()
    for forbidden in (OUTPUT,SOURCE,c.BREADTH_SOURCE,c.PARENT):
        assert not location.is_relative_to(forbidden.resolve()), "Fixture cannot target production evidence"
    assert rows and len({row["id"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == {"train","validation"}
    return rows


def _model(fit_id):
    return inherited._model(fit_id)


def _metadata(fit_id, binding):
    return {"experiment":"proposal_specialist_128","arm":"specialist","fit":fit_id,
            "seed":TRAIN_SEEDS[fit_id],"manifest":binding}


def _control(fit_id, rows, control_fixture):
    if rows is None:
        assert control_fixture is None
        evidence,result = data.control_evidence(fit_id)
    else:
        assert control_fixture is not None, "Bounded fit requires independently reproduced saved control fixture"
        evidence = control_fixture
        result = c.verified(evidence["result"])
        checked = c.verified(evidence["audit"])
        assert checked["complete_fit_bitwise_identical"] and checked["value_head_frozen"]
        assert checked["result"] == c.artifact(c.verify_hash(evidence["result"]))
        assert checked["selected_updates"] == result["selected_updates"]
        assert result["config"] == inherited.CONFIG and result["fit"] == fit_id
        assert evidence["original_selected_update"] == result["selected_updates"]["broad"]
    assert result["initializer"] == previous.initializer(fit_id)
    assert result["seed"] == TRAIN_SEEDS[fit_id] and result["equal_dose_and_batches"]
    return evidence,result


def _control_model(fit_id, curve, source, initial):
    payload = torch.load(c.verify_hash(curve["checkpoint"]),map_location="cpu",weights_only=True)
    assert payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(old._RAW._model.config)
    assert payload["generation"] == curve["update"]
    assert payload["metadata"] == inherited._metadata(fit_id,"broad",source["manifest"])
    model = f.StrongPolicyInference(str(c.verify_hash(curve["checkpoint"]))).model
    assert asdict(model.config) == asdict(old._RAW._model.config)
    for name,value in model.state_dict().items():
        assert torch.isfinite(value).all()
        if name.startswith("value_head.") or curve["update"] == 0:
            assert torch.equal(value,initial[name])
    return model


def fit(fit_id, binding, base=OUTPUT, *, rows=None, control_fixture=None):
    assert fit_id in FITS
    c.budget_boundary(base); old.initialize_worker()
    values = _inputs(binding,base,rows)
    training,valid = ([row for row in values if row["split"] == kind] for kind in ("train","validation"))
    evidence,control = _control(fit_id,rows,control_fixture)
    points = f.checkpoint_updates(len(training))
    assert [curve["update"] for curve in control["trajectories"]["broad"]] == points
    assert control["row_exposures"] == 4*len(training) and control["updates"] == points[-1]
    assert control["train_rows"] == len(training) and control["validation_rows"] == len(valid)
    folder = Path(base)/"fits"/f"fit_{fit_id}"
    folder.mkdir(parents=True,exist_ok=False)
    reporter = StatusReporter(folder,{"manifest":binding,"fit":fit_id})
    start = time.perf_counter()
    model,parameters,optimizer = _model(fit_id)
    initial = {name:value.clone() for name,value in model.state_dict().items()}
    arm_folder = folder/"specialist"; arm_folder.mkdir()
    update,curves,stream,best,chosen = 0,[],[],math.inf,None
    def observe():
        nonlocal best,chosen
        c.budget_boundary(base)
        assert all(torch.equal(initial[name],value) for name,value in model.state_dict().items() if name.startswith("value_head."))
        report = validation(model,valid,base)
        checkpoint = arm_folder/f"checkpoint_{update:04d}.pt"
        f.save_model(checkpoint,model,update,_metadata(fit_id,binding))
        path = arm_folder/f"validation_{update:04d}.json"; old.save_new(path,report)
        curves.append({"update":update,"checkpoint":c.artifact(checkpoint),"validation":c.artifact(path),"pair_loss":report["pair_loss"]})
        if report["pair_loss"] < best:
            best,chosen = report["pair_loss"],update
        reporter.event("specialist_fit",update=update,total_updates=points[-1])
    observe()
    for epoch in range(4):
        for indices in batch_plan(training,fit_id,epoch):
            c.budget_boundary(base)
            batch = [training[i] for i in indices]; stream.append([row["id"] for row in batch])
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits,_ = model(*f.collate(batch))
            loss = objective(logits,batch)[0].mean()
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); optimizer.step()
            assert all(torch.isfinite(value).all() for value in model.state_dict().values())
            update += 1
            if update in points:
                observe()
    identity = old.digest(stream)
    assert update == points[-1] and identity == control["batch_sequence_sha256"]["broad"]
    selected,trajectories = {"specialist":chosen},{"specialist":curves}
    control_folder = folder/"control"; control_folder.mkdir()
    curves,best,chosen = [],math.inf,None
    for curve in control["trajectories"]["broad"]:
        c.budget_boundary(base)
        saved = _control_model(fit_id,curve,control,initial)
        report = validation(saved,valid,base)
        path = control_folder/f"validation_{curve['update']:04d}.json"; old.save_new(path,report)
        curves.append({"update":curve["update"],"checkpoint":curve["checkpoint"],"validation":c.artifact(path),
                       "pair_loss":report["pair_loss"],"inherited_validation":curve["validation"]})
        if report["pair_loss"] < best:
            best,chosen = report["pair_loss"],curve["update"]
        reporter.event("control_revalidation",update=curve["update"],total_updates=points[-1])
    selected["control"],trajectories["control"] = chosen,curves
    result = {"manifest":binding,"fit":fit_id,"seed":TRAIN_SEEDS[fit_id],"config":CONFIG,
        "initializer":previous.initializer(fit_id),"data_lock":c.artifact(Path(base)/"data_lock.json") if rows is None else None,
        "selected_updates":selected,"trajectories":trajectories,"updates":points[-1],"train_rows":len(training),
        "validation_rows":len(valid),"row_exposures":4*len(training),"batch_sequence_sha256":{arm:identity for arm in ARMS},
        "equal_dose_and_batches":True,"inherited_control":evidence,"control_retrained":False,
        "elapsed_seconds":time.perf_counter()-start}
    path = folder/"result.json"; old.save_new(path,result)
    return c.artifact(path,fit=fit_id,index=FITS.index(fit_id))


def audit_fit(fit_id, fit_item, binding, base=OUTPUT, *, rows=None, control_fixture=None):
    c.budget_boundary(base); old.initialize_worker()
    values = _inputs(binding,base,rows)
    training,valid = ([row for row in values if row["split"] == kind] for kind in ("train","validation"))
    evidence,control = _control(fit_id,rows,control_fixture)
    expected = c.verified(fit_item)
    assert expected["manifest"] == binding and expected["fit"] == fit_id and expected["config"] == CONFIG
    assert expected["initializer"] == previous.initializer(fit_id) and expected["seed"] == 212700001+FITS.index(fit_id)
    assert expected["data_lock"] == (c.artifact(Path(base)/"data_lock.json") if rows is None else None)
    assert expected["inherited_control"] == evidence and not expected["control_retrained"]
    steps = (len(training)+63)//64
    points = sorted({x for x in (0,1,2,4,8,16,32,64) if x <= 4*steps} |
                    {epoch*steps+math.ceil(q*steps/4) for epoch in range(4) for q in (1,2,3,4)})
    assert all([curve["update"] for curve in expected["trajectories"][arm]] == points for arm in ARMS)
    start = time.perf_counter()
    folder = Path(base)/"fits"/f"fit_{fit_id}"
    reporter = StatusReporter(folder/"audit_status",{"manifest":binding,"fit":fit_id})
    model,parameters,optimizer = _model(fit_id)
    initial = {name:value.clone() for name,value in model.state_dict().items()}
    update,stream,best,chosen = 0,[],math.inf,None
    curves = expected["trajectories"]["specialist"]
    def inspect():
        nonlocal best,chosen
        c.budget_boundary(base)
        curve = curves[points.index(update)]
        payload = torch.load(c.verify_hash(curve["checkpoint"]),map_location="cpu",weights_only=True)
        assert payload["model_kind"] == MODEL_KIND and payload["model_config"] == asdict(model.config)
        assert payload["generation"] == update and payload["metadata"] == _metadata(fit_id,binding)
        assert payload["model_state_dict"].keys() == model.state_dict().keys()
        for name,value in model.state_dict().items():
            assert torch.isfinite(value).all() and torch.equal(value,payload["model_state_dict"][name]), (fit_id,update,name)
            if name.startswith("value_head."):
                assert torch.equal(value,initial[name])
        report = validation(model,valid,base,audit=True)
        assert report == c.verified(curve["validation"]) and report["pair_loss"] == curve["pair_loss"]
        if report["pair_loss"] < best:
            best,chosen = report["pair_loss"],update
        reporter.event("specialist_reproduction",update=update,total_updates=4*steps)
    inspect()
    for epoch in range(4):
        for indices in independent_plan(training,fit_id,epoch):
            c.budget_boundary(base)
            batch = [training[i] for i in indices]; stream.append([row["id"] for row in batch])
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits,_ = model(*independent.audit_batch_tensors(batch))
            loss = audit_objective(logits,batch)[0].mean()
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True); optimizer.step()
            assert all(torch.isfinite(value).all() for value in model.state_dict().values())
            update += 1
            if update in points:
                inspect()
    assert update == expected["updates"] == 4*steps
    selected = {"specialist":chosen}
    identity = old.digest(stream)
    assert expected["equal_dose_and_batches"] and expected["batch_sequence_sha256"] == {arm:identity for arm in ARMS}
    assert identity == control["batch_sequence_sha256"]["broad"]
    assert expected["row_exposures"] == control["row_exposures"] == 4*len(training)
    assert expected["train_rows"] == control["train_rows"] == len(training)
    assert expected["validation_rows"] == control["validation_rows"] == len(valid)
    best,chosen = math.inf,None
    for curve, original in zip(expected["trajectories"]["control"],control["trajectories"]["broad"],strict=True):
        assert curve["update"] == original["update"] and curve["checkpoint"] == original["checkpoint"]
        assert curve["inherited_validation"] == original["validation"]
        saved = _control_model(fit_id,original,control,initial)
        report = validation(saved,valid,base,audit=True)
        assert report == c.verified(curve["validation"]) and report["pair_loss"] == curve["pair_loss"]
        if report["pair_loss"] < best:
            best,chosen = report["pair_loss"],curve["update"]
        reporter.event("control_selection_reproduction",update=curve["update"],total_updates=4*steps)
    selected["control"] = chosen
    assert selected == expected["selected_updates"]
    unchanged = {}
    for arm in ARMS:
        curve = next(value for value in expected["trajectories"][arm] if value["update"] == selected[arm])
        state = torch.load(c.verify_hash(curve["checkpoint"]),map_location="cpu",weights_only=True)["model_state_dict"]
        unchanged[arm] = all(torch.equal(value,initial[name]) for name,value in state.items())
    path = folder/"audit.json"
    old.save_new(path,{"manifest":binding,"fit":fit_id,"result":c.artifact(c.verify_hash(fit_item)),
        "selected_updates":selected,"checkpoints":points,"complete_specialist_fit_bitwise_identical":True,
        "inherited_control_fit_reproduction":evidence["audit"],"control_retrained":False,
        "value_head_frozen":True,"equal_dose_and_batches":True,"common_outsider_validation":True,
        "all_new_validation_and_selection_independently_reproduced":True,"unchanged_from_anchor":unchanged,
        "elapsed_seconds":time.perf_counter()-start})
    return c.artifact(path,fit=fit_id,index=FITS.index(fit_id))


def _models(fits, audits, binding, base):
    data.validate_data(binding,base)
    assert len(fits) == len(audits) == 3
    models = {}
    for index,(fit_id,item,checked) in enumerate(zip(FITS,fits,audits,strict=True)):
        assert item["index"] == checked["index"] == index and item["fit"] == checked["fit"] == fit_id
        result,audit = c.verified(item),c.verified(checked)
        assert result["manifest"] == audit["manifest"] == binding and result["fit"] == audit["fit"] == fit_id
        assert result["config"] == CONFIG and result["initializer"] == previous.initializer(fit_id)
        assert result["seed"] == TRAIN_SEEDS[fit_id] and result["data_lock"] == c.artifact(Path(base)/"data_lock.json")
        assert audit["result"] == c.artifact(c.verify_hash(item)) and audit["complete_specialist_fit_bitwise_identical"]
        assert audit["value_head_frozen"] and audit["equal_dose_and_batches"] and audit["common_outsider_validation"]
        assert audit["all_new_validation_and_selection_independently_reproduced"]
        assert not result["control_retrained"] and not audit["control_retrained"]
        assert audit["inherited_control_fit_reproduction"] == result["inherited_control"]["audit"]
        c.verify_hash(result["inherited_control"]["result"]); c.verify_hash(result["inherited_control"]["audit"])
        assert result["equal_dose_and_batches"] and len(set(result["batch_sequence_sha256"].values())) == 1
        assert result["selected_updates"] == audit["selected_updates"]
        for arm in ARMS:
            curves = result["trajectories"][arm]
            assert [curve["update"] for curve in curves] == audit["checkpoints"]
            for curve in curves:
                c.budget_boundary(base); c.verify_hash(curve["checkpoint"])
                assert c.verified(curve["validation"])["pair_loss"] == curve["pair_loss"]
            selected = min(curves,key=lambda curve:curve["pair_loss"])
            assert selected["update"] == result["selected_updates"][arm]
            models[f"{arm}_{fit_id}"] = {**selected["checkpoint"],"source_fit":fit_id,
                "selected_update":selected["update"],"unchanged_from_anchor":audit["unchanged_from_anchor"][arm]}
        models[f"anchor_{fit_id}"] = {**previous.initializer(fit_id),"unchanged_from_anchor":True}
    return models


def models_lock(fits, audits, binding, base=OUTPUT):
    fits,audits = sorted(fits,key=lambda item:item["fit"]),sorted(audits,key=lambda item:item["fit"])
    models = _models(fits,audits,binding,base)
    path = Path(base)/"models_lock.json"
    old.save_new(path,{"manifest":binding,"models":models,"data_lock":c.artifact(Path(base)/"data_lock.json"),
        "fits":fits,"audits":audits,"training_config":CONFIG,
        "all_specialist_unchanged":all(models[f"specialist_{fit_id}"]["unchanged_from_anchor"] for fit_id in FITS),
        "matched_dose_and_batches":True,"common_outsider_validation_selection":True,"control_retrained":False})
    return c.artifact(path)


def validate_models(binding, base=OUTPUT):
    lock = old.read_json(Path(base)/"models_lock.json")
    assert lock["manifest"] == binding and lock["training_config"] == CONFIG
    assert lock["data_lock"] == c.artifact(Path(base)/"data_lock.json")
    assert lock["models"] == _models(lock["fits"],lock["audits"],binding,base)
    assert lock["all_specialist_unchanged"] == all(lock["models"][f"specialist_{fit_id}"]["unchanged_from_anchor"] for fit_id in FITS)
    assert lock["matched_dose_and_batches"] and lock["common_outsider_validation_selection"] and not lock["control_retrained"]
    return lock
