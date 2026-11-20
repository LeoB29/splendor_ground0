"""Matched broader/narrower supervision and complete deterministic reproduction."""
from collections import defaultdict
from dataclasses import asdict
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from tools.proposal_iteration_128 import learning as previous
from scripts.analysis import verify_fast_selector_transfer_157 as independent
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from . import common as c
from . import learning_data as data

ROOT, OUTPUT, SOURCE, FITS = c.ROOT, c.OUTPUT, c.SOURCE, c.FITS
old, f = c.old, previous.f
ARMS = ("broad", "narrow")
TRAIN_SEEDS = {161001: 212700001, 161002: 212700002, 161003: 212700003}
CONFIG = {"epochs":4,"batch_size":64,"lr":1e-5,"weight_decay":1e-4,"gradient_norm":1.,
          "torch_threads":1,"device":"cpu","dtype":"float32","fresh_optimizer":True,
          "value_head_frozen":True,"training_seeds":{str(k):v for k,v in TRAIN_SEEDS.items()},
          "batch_tag":"proposal-breadth-learning-128-batches-v1",
          "batch_recipe":"corpus_order_shuffle_stable_feature_length_sort_chunk64_shuffle_chunks",
          "objective":"offered_group_winner_cross_entropy; broad_recorded_support_or_narrow_prefix",
          "validation":"common_broad_support_target_equal_whole_pair_mean_cross_entropy",
          "selection":"earliest_strict_minimum","initial_checkpoint_eligible":True,
          "unoffered_actions_are_negative":False,"oversampling":False}
source_evidence = data.source_evidence
prepare_data = data.prepare_data


def objective(logits, rows, arm):
    assert arm in ARMS
    losses,choices = [],[]
    for i,row in enumerate(rows):
        size = len(row["members"])
        width = size if arm == "broad" else row["narrow_support"]
        label = row["teacher"] if arm == "broad" else row["narrow_teacher"]
        assert 1 <= size <= 8 and row["narrow_support"] == min(size,4)
        assert 0 <= label < width
        scores = torch.stack([torch.amax(logits[i,member]) for member in row["members"][:width]])
        losses.append(F.cross_entropy(scores[None],torch.tensor([label])))
        choices.append(int(scores.detach().argmax()))
    return torch.stack(losses),choices


def audit_objective(logits, rows, arm):
    assert arm in ("broad","narrow")
    losses,choices = [],[]
    for index,row in enumerate(rows):
        count = len(row["members"])
        assert 1 <= count <= 8 and row["narrow_support"] == min(4,count)
        ids = [int(row["groups"][member[0]]) for member in row["members"]]
        if arm == "narrow":
            ids = ids[:min(4,count)]
            target = row["narrow_teacher"]
        else:
            target = row["teacher"]
        assert 0 <= target < len(ids)
        scores = torch.stack([torch.amax(logits[index,torch.where(row["groups"] == group)[0]]) for group in ids])
        losses.append(F.cross_entropy(scores.unsqueeze(0),torch.tensor([target])))
        choices.append(int(scores.detach().argmax()))
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
    batches = independent.audit_batches if audit else f.batches
    collate = independent.audit_batch_tensors if audit else f.collate
    loss = audit_objective if audit else objective
    with torch.no_grad():
        for indices in batches(rows,157000,-1):
            c.budget_boundary(base)
            batch = [rows[i] for i in indices]
            logits,_ = model(*collate(batch))
            # Always the full broader target/support, including narrow models.
            losses,choices = loss(logits,batch,"broad")
            for row,value,choice in zip(batch,losses.tolist(),choices,strict=True):
                assert math.isfinite(value)
                records.append({"id":row["id"],"pair":row["pair"],"loss":value,"choice":choice,
                     "teacher":row["teacher"],"narrow_teacher":row["narrow_teacher"],
                     "support":len(row["members"]),"narrow_support":row["narrow_support"],
                     "seat":row["seat"],"phase":row["phase"],"privacy":row["privacy"]})
    records.sort(key=lambda row:row["id"])
    if audit:
        means = []
        for pair in sorted({row["pair"] for row in records}):
            values = [row["loss"] for row in records if row["pair"] == pair]
            means.append(sum(values)/len(values))
        result = sum(means)/len(means)
    else:
        result = pair_loss(records)
    return {"pair_loss":result,"rows":records}


def batch_plan(rows, fit_id, epoch):
    assert fit_id in FITS and epoch in range(4) and all(row["split"] == "train" for row in rows)
    indices = list(range(len(rows)))
    generator = random.Random(f"{CONFIG['batch_tag']}|{TRAIN_SEEDS[fit_id]}|{epoch}")
    generator.shuffle(indices)
    indices.sort(key=lambda i:len(rows[i]["features"]))
    batches = [indices[i:i+64] for i in range(0,len(indices),64)]
    generator.shuffle(batches)
    return batches


def independent_plan(rows, fit_id, epoch):
    assert all(row["split"] == "train" for row in rows)
    ordered = list(range(len(rows)))
    seed = 212700001+(fit_id-161001)
    generator = random.Random("proposal-breadth-learning-128-batches-v1|%s|%s" % (seed,epoch))
    generator.shuffle(ordered)
    ordered = sorted(ordered,key=lambda index:rows[index]["features"].shape[0])
    blocks = [ordered[lo:min(lo+64,len(ordered))] for lo in range(0,len(ordered),64)]
    generator.shuffle(blocks)
    return blocks


def _inputs(binding, base, rows):
    if rows is None:
        return data.load_data(binding,base)
    location = Path(base).resolve()
    for forbidden in (OUTPUT,SOURCE,c.BREADTH_SOURCE):
        assert not location.is_relative_to(forbidden.resolve()), "Fixture cannot target production evidence"
    assert rows and len({row["id"] for row in rows}) == len(rows)
    assert {row["split"] for row in rows} == {"train","validation"}
    return rows


def _model(fit_id):
    torch.manual_seed(TRAIN_SEEDS[fit_id])
    model = f.StrongPolicyInference(str(c.verify_hash(previous.initializer(fit_id)))).model
    assert asdict(model.config) == asdict(old._RAW._model.config), "Source/model encoding configuration mismatch"
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("value_head."))
    parameters = [value for name,value in model.named_parameters() if not name.startswith("value_head.")]
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters,lr=1e-5,weight_decay=1e-4)
    assert not optimizer.state
    return model,parameters,optimizer


def _metadata(fit_id, arm, binding):
    return {"experiment":"proposal_breadth_learning_128","arm":arm,"fit":fit_id,
            "seed":TRAIN_SEEDS[fit_id],"manifest":binding}


def fit(fit_id, binding, base=OUTPUT, *, rows=None):
    assert fit_id in FITS
    c.budget_boundary(base)
    old.initialize_worker()
    values = _inputs(binding,base,rows)
    training,valid = ([row for row in values if row["split"] == kind] for kind in ("train","validation"))
    folder = Path(base)/"fits"/f"fit_{fit_id}"
    folder.mkdir(parents=True,exist_ok=False)
    reporter = StatusReporter(folder,{"manifest":binding,"fit":fit_id})
    points = f.checkpoint_updates(len(training))
    selected,trajectories,identities = {},{},{}
    start = time.perf_counter()
    for arm in ARMS:
        model,parameters,optimizer = _model(fit_id)
        initial = {name:value.clone() for name,value in model.state_dict().items()}
        arm_folder = folder/arm
        arm_folder.mkdir()
        update,curves,stream,best,chosen = 0,[],[],math.inf,None
        def observe():
            nonlocal best,chosen
            c.budget_boundary(base)
            assert all(torch.equal(initial[name],value) for name,value in model.state_dict().items() if name.startswith("value_head."))
            report = validation(model,valid,base)
            checkpoint = arm_folder/f"checkpoint_{update:04d}.pt"
            f.save_model(checkpoint,model,update,_metadata(fit_id,arm,binding))
            path = arm_folder/f"validation_{update:04d}.json"
            old.save_new(path,report)
            curves.append({"update":update,"checkpoint":c.artifact(checkpoint),"validation":c.artifact(path),"pair_loss":report["pair_loss"]})
            if report["pair_loss"] < best:
                best,chosen = report["pair_loss"],update
            reporter.event("learning_fit",arm=arm,update=update,total_updates=points[-1])
        observe()
        for epoch in range(4):
            for indices in batch_plan(training,fit_id,epoch):
                c.budget_boundary(base)
                batch = [training[i] for i in indices]
                stream.append([row["id"] for row in batch])
                model.train(); optimizer.zero_grad(set_to_none=True)
                logits,_ = model(*f.collate(batch))
                loss = objective(logits,batch,arm)[0].mean()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step()
                assert all(torch.isfinite(value).all() for value in model.state_dict().values())
                update += 1
                if update in points:
                    observe()
        assert update == points[-1]
        selected[arm],trajectories[arm],identities[arm] = chosen,curves,old.digest(stream)
    assert len(set(identities.values())) == 1
    result = {"manifest":binding,"fit":fit_id,"seed":TRAIN_SEEDS[fit_id],"config":CONFIG,
              "initializer":previous.initializer(fit_id),"data_lock":c.artifact(Path(base)/"data_lock.json") if rows is None else None,
              "selected_updates":selected,"trajectories":trajectories,"updates":points[-1],
              "train_rows":len(training),"validation_rows":len(valid),"row_exposures":4*len(training),
              "batch_sequence_sha256":identities,"equal_dose_and_batches":True,
              "elapsed_seconds":time.perf_counter()-start}
    path = folder/"result.json"
    old.save_new(path,result)
    return c.artifact(path,fit=fit_id,index=FITS.index(fit_id))


def audit_fit(fit_id, fit_item, binding, base=OUTPUT, *, rows=None):
    assert fit_id in FITS
    c.budget_boundary(base)
    old.initialize_worker()
    values = _inputs(binding,base,rows)
    training,valid = ([row for row in values if row["split"] == kind] for kind in ("train","validation"))
    expected = c.verified(fit_item)
    assert expected["manifest"] == binding and expected["fit"] == fit_id and expected["config"] == CONFIG
    assert expected["initializer"] == previous.initializer(fit_id) and expected["seed"] == 212700001+FITS.index(fit_id)
    assert expected["data_lock"] == (c.artifact(Path(base)/"data_lock.json") if rows is None else None)
    steps = (len(training)+63)//64
    points = sorted({x for x in (0,1,2,4,8,16,32,64) if x <= 4*steps} |
                    {epoch*steps+math.ceil(q*steps/4) for epoch in range(4) for q in (1,2,3,4)})
    selected,unchanged,sequences = {},{},{}
    start = time.perf_counter()
    folder = Path(base)/"fits"/f"fit_{fit_id}"
    reporter = StatusReporter(folder/"audit_status",{"manifest":binding,"fit":fit_id})
    for arm in ("broad","narrow"):
        curves = expected["trajectories"][arm]
        assert [curve["update"] for curve in curves] == points
        model,parameters,optimizer = _model(fit_id)
        initial = {name:value.clone() for name,value in model.state_dict().items()}
        update,stream,best,chosen = 0,[],math.inf,None
        def inspect():
            nonlocal best,chosen
            c.budget_boundary(base)
            curve = curves[points.index(update)]
            payload = torch.load(c.verify_hash(curve["checkpoint"]),map_location="cpu",weights_only=True)
            assert payload["generation"] == update and payload["metadata"] == _metadata(fit_id,arm,binding)
            assert payload["model_state_dict"].keys() == model.state_dict().keys()
            for name,value in model.state_dict().items():
                assert torch.isfinite(value).all() and torch.equal(value,payload["model_state_dict"][name]), (fit_id,arm,update,name)
                if name.startswith("value_head."):
                    assert torch.equal(value,initial[name])
            report = validation(model,valid,base,audit=True)
            assert report == c.verified(curve["validation"]) and report["pair_loss"] == curve["pair_loss"]
            if report["pair_loss"] < best:
                best,chosen = report["pair_loss"],update
            reporter.event("learning_fit_reproduction",arm=arm,update=update,total_updates=4*steps)
        inspect()
        for epoch in range(4):
            for indices in independent_plan(training,fit_id,epoch):
                c.budget_boundary(base)
                batch = [training[index] for index in indices]
                stream.append([row["id"] for row in batch])
                model.train(); optimizer.zero_grad(set_to_none=True)
                logits,_ = model(*independent.audit_batch_tensors(batch))
                loss = audit_objective(logits,batch,arm)[0].mean()
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step()
                assert all(torch.isfinite(value).all() for value in model.state_dict().values())
                update += 1
                if update in points:
                    inspect()
        assert update == expected["updates"] == 4*steps
        assert chosen == expected["selected_updates"][arm]
        sequences[arm] = old.digest(stream)
        assert sequences[arm] == expected["batch_sequence_sha256"][arm]
        selected[arm] = chosen
        curve = next(curve for curve in curves if curve["update"] == chosen)
        state = torch.load(c.verify_hash(curve["checkpoint"]),weights_only=True,map_location="cpu")["model_state_dict"]
        unchanged[arm] = all(torch.equal(value,initial[name]) for name,value in state.items())
    assert sequences["broad"] == sequences["narrow"] and expected["equal_dose_and_batches"]
    assert expected["row_exposures"] == 4*len(training)
    assert expected["train_rows"] == len(training) and expected["validation_rows"] == len(valid)
    path = folder/"audit.json"
    old.save_new(path,{"manifest":binding,"fit":fit_id,"result":c.artifact(c.verify_hash(fit_item)),
                 "selected_updates":selected,"checkpoints":points,"complete_fit_bitwise_identical":True,
                 "value_head_frozen":True,"equal_dose_and_batches":True,"common_broad_validation":True,
                 "unchanged_from_anchor":unchanged,"elapsed_seconds":time.perf_counter()-start})
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
        assert audit["result"] == c.artifact(c.verify_hash(item)) and audit["complete_fit_bitwise_identical"]
        assert audit["value_head_frozen"] and audit["equal_dose_and_batches"] and audit["common_broad_validation"]
        assert result["equal_dose_and_batches"] and len(set(result["batch_sequence_sha256"].values())) == 1
        assert result["selected_updates"] == audit["selected_updates"]
        for arm in ARMS:
            curves = result["trajectories"][arm]
            assert [curve["update"] for curve in curves] == audit["checkpoints"]
            for curve in curves:
                c.budget_boundary(base)
                c.verify_hash(curve["checkpoint"])
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
                 "all_broad_unchanged":all(models[f"broad_{fit_id}"]["unchanged_from_anchor"] for fit_id in FITS),
                 "matched_dose_and_batches":True,"common_broad_validation_selection":True})
    return c.artifact(path)


def validate_models(binding, base=OUTPUT):
    lock = old.read_json(Path(base)/"models_lock.json")
    assert lock["manifest"] == binding and lock["training_config"] == CONFIG
    assert lock["data_lock"] == c.artifact(Path(base)/"data_lock.json")
    assert lock["models"] == _models(lock["fits"],lock["audits"],binding,base)
    assert lock["all_broad_unchanged"] == all(lock["models"][f"broad_{fit_id}"]["unchanged_from_anchor"] for fit_id in FITS)
    assert lock["matched_dose_and_batches"] and lock["common_broad_validation_selection"]
    return lock
