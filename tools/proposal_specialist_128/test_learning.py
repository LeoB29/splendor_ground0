"""Bounded authentic-corpus, objective, optimizer and checkpoint reuse tests."""
import copy
from dataclasses import asdict
import random

import pytest
import torch

from tools.proposal_breadth_learning_128.test_learning import root_fixture, tiny_rows as inherited_rows
from . import learning as h
from . import learning_data as d


@pytest.fixture(scope="module",autouse=True)
def initialize():
    h.old.initialize_worker()


def tiny_rows():
    rows = inherited_rows()
    for row in rows:
        if row["pair"].startswith("breadth128:"):
            row.update(outsider_start=3,outsider_support=5,outsider_teacher=6)
    return rows


@pytest.mark.parametrize("support",range(1,9))
def test_exact_first_three_exclusion_and_first_maximum(support):
    root = root_fixture(support)
    primary,independent = d.targets(root),d.independent_targets(root)
    assert primary == independent
    assert primary["outsider_start"] == min(3,support)
    assert primary["outsider_support"] == max(0,support-3)
    expected = max(range(3,support),key=lambda index:root["means"][index]) if support > 3 else -1
    assert primary["outsider_teacher"] == expected
    if support >= 6:
        assert primary["outsider_teacher"] == 5  # Tied6 follows5 in recorded order.


@pytest.mark.parametrize("damage",["prefix","means","choice","worlds","outcome","duplicate"])
def test_corrupt_supervision_rejected_by_both_target_paths(damage):
    root = root_fixture()
    if damage == "prefix":
        root["proposal"]["anchor_candidates"]["actions"][0] = 100
    elif damage == "means":
        root["means"][0] += 1/128
    elif damage == "choice":
        root["choice"] = 6
    elif damage == "worlds":
        root["worlds"].pop()
    elif damage == "outcome":
        root["worlds"][0]["outcomes"][0]["wdl"] = .5
    else:
        root["candidates"]["keys"][-1] = root["candidates"]["keys"][0]
    for function in (d.targets,d.independent_targets):
        with pytest.raises(AssertionError):
            function(root)


def test_historical_CE_unchanged_new_first_three_and_unoffered_tail_not_negatives():
    rows = tiny_rows()
    for batch in (rows[:3],rows[3:5]):
        logits = torch.linspace(-1,1,len(batch[0]["features"]))[None].repeat(len(batch),1).requires_grad_()
        losses,choices = h.objective(logits,batch)
        other,other_choices = h.audit_objective(logits,batch)
        assert torch.equal(losses,other) and choices == other_choices
        gradients = torch.autograd.grad(losses.sum(),logits,retain_graph=True)[0]
        assert torch.equal(gradients,torch.autograd.grad(other.sum(),logits,retain_graph=True)[0])
        if batch[0]["pair"].startswith("breadth128:"):
            offered = {int(i) for member in batch[0]["members"][3:] for i in member}
            excluded = [i for i in range(logits.shape[1]) if i not in offered]
            assert torch.equal(gradients[:,excluded],torch.zeros_like(gradients[:,excluded]))
            changed = logits.detach().clone(); changed[:,excluded] = 10000
            assert torch.equal(h.objective(changed,batch)[0],losses.detach())
        else:
            assert torch.equal(losses,h.inherited.objective(logits,batch,"broad")[0])


@pytest.mark.parametrize("support",range(1,5))
def test_empty_singleton_zero_is_finite_differentiable_with_padding_and_extreme_logits(support):
    row = tiny_rows()[3]
    row["members"] = row["members"][:support]
    row.update(teacher=0,narrow_teacher=0,narrow_support=min(4,support),
               outsider_start=min(3,support),outsider_support=max(0,support-3),outsider_teacher=3 if support == 4 else -1)
    count = len(row["features"])
    logits = torch.full((1,count+7),torch.finfo(torch.float32).max)
    logits[:,count:] = -torch.inf
    logits.requires_grad_()
    for objective in (h.objective,h.audit_objective):
        values,_ = objective(logits,[row])
        assert torch.equal(values,torch.zeros_like(values)) and values.requires_grad
        gradient = torch.autograd.grad(values.sum(),logits,retain_graph=True)[0]
        assert torch.isfinite(gradient).all() and not gradient.any()


def test_zero_rows_preserve_batch_denominator_and_allzero_adamw_steps():
    rows = tiny_rows()[3:5]
    rows[1]["members"] = rows[1]["members"][:3]
    rows[1].update(teacher=0,narrow_teacher=0,narrow_support=3,outsider_start=3,outsider_support=0,outsider_teacher=-1)
    count = len(rows[0]["features"])
    parameter = torch.nn.Parameter(torch.linspace(-1,1,count)[None].repeat(2,1))
    other = torch.nn.Parameter(parameter.detach().clone())
    left,right = (torch.optim.AdamW([p],lr=1e-5,weight_decay=1e-4) for p in (parameter,other))
    losses,_ = h.objective(parameter,rows)
    assert losses.mean() == losses[0]/2 and losses[1] == 0
    losses.mean().backward(); left.step(); left.zero_grad(set_to_none=True)
    h.audit_objective(other,rows)[0].mean().backward(); right.step(); right.zero_grad(set_to_none=True)
    before = parameter.detach().clone()
    zero = [rows[1],rows[1]]
    h.objective(parameter,zero)[0].mean().backward(); left.step()
    h.audit_objective(other,zero)[0].mean().backward(); right.step()
    assert torch.equal(parameter,other) and not torch.equal(parameter,before)
    assert left.state[parameter]["step"] == right.state[other]["step"] == 2
    for key in left.state[parameter]:
        assert torch.equal(left.state[parameter][key],right.state[other][key])


def test_identical_splits_pair_weighting_and_corpus_order_batch_streams():
    assert d.splits() == d.independent_splits() == d.inherited.splits()
    assert h.pair_loss([{"pair":"a","loss":0.},{"pair":"b","loss":2.},{"pair":"b","loss":2.}]) == 1.
    template = tiny_rows()[0]
    rows = [{**template,"id":f"row_{129-i:03d}","features":torch.zeros((3+(i*11)%21,4))} for i in range(130)]
    for fit in h.FITS:
        for epoch in range(4):
            batch = h.batch_plan(rows,fit,epoch)
            assert batch == h.independent_plan(rows,fit,epoch) == h.inherited.batch_plan(rows,fit,epoch)
            generator = random.Random(f"proposal-breadth-learning-128-batches-v1|{h.TRAIN_SEEDS[fit]}|{epoch}")
            expected = list(range(130)); generator.shuffle(expected)
            expected.sort(key=lambda i:len(rows[i]["features"]))
            expected = [expected[i:i+64] for i in range(0,130,64)]; generator.shuffle(expected)
            assert batch == expected and sorted(map(len,batch)) == [2,64,64]


def test_one_authentic_saved_breadth_shard_targets_tensors_splits_and_hash_rejection(tmp_path):
    parent = h.old.read_json(h.c.PARENT/"data_lock.json")
    audit = h.c.verified(parent["audit"])
    item,checked = parent["exports"][0],audit["breadth_audits"][0]
    old_sha = item["sha256"]
    exported,receipt_item = d.augment_shard(item,checked,"fixture",tmp_path)
    receipt = h.c.verified(receipt_item)
    before = torch.load(h.c.verify_hash(item),weights_only=True,map_location="cpu")
    after = torch.load(h.c.verify_hash(exported),weights_only=True,map_location="cpu")
    assert item["sha256"] == old_sha and h.old.file_hash(h.c.verify_hash(item)) == old_sha
    assert receipt["passed"] and receipt["inherited_tensors_splits_unchanged"] and receipt["independent_outsider_targets"]
    for original,row,root_item in zip(before,after,receipt["source_roots"],strict=True):
        assert torch.equal(original["observation"],row["observation"]) and torch.equal(original["features"],row["features"])
        assert original["id"] == row["id"] and original["split"] == row["split"]
        target = d.independent_targets(h.c.verified(root_item))
        assert all(row[key] == value for key,value in target.items())
    tampered = {**item,"sha256":"0"*64}
    with pytest.raises(ValueError,match="Changed artifact"):
        h.c.verify_hash(tampered)
    with pytest.raises(AssertionError):
        d.augment_shard(item,checked,"fixture",tmp_path)


def saved_control(tmp_path,fit_id,rows):
    folder = tmp_path/"inherited"
    item = h.inherited.fit(fit_id,"fixture_parent",folder,rows=rows)
    checked = h.inherited.audit_fit(fit_id,item,"fixture_parent",folder,rows=rows)
    return {"result":item,"audit":checked,"original_selected_update":h.c.verified(item)["selected_updates"]["broad"]}


@pytest.mark.parametrize("fit_id,tied",[(161001,False),(161002,False),(161003,True)])
def test_new_fit_reproduction_and_reused_control_reselection_no_retraining(tmp_path,monkeypatch,fit_id,tied):
    rows = tiny_rows()
    if tied:
        for row in rows[5:]:
            row["members"] = row["members"][:3]
            row.update(teacher=0,narrow_teacher=0,narrow_support=3,outsider_start=3,outsider_support=0,outsider_teacher=-1)
    control = saved_control(tmp_path,fit_id,rows)
    previous = h.c.verified(control["result"])
    frozen = {curve["checkpoint"]["path"]:curve["checkpoint"]["sha256"] for curve in previous["trajectories"]["broad"]}
    def disallow(*args,**kwargs):
        raise AssertionError("The saved control must not be trained or replayed again")
    monkeypatch.setattr(h.inherited,"fit",disallow)
    monkeypatch.setattr(h.inherited,"audit_fit",disallow)
    model_calls = []
    original_model = h._model
    def count_model(fit):
        model_calls.append(fit)
        return original_model(fit)
    monkeypatch.setattr(h,"_model",count_model)
    base = tmp_path/"new"
    item = h.fit(fit_id,"fixture",base,rows=rows,control_fixture=control)
    checked = h.audit_fit(fit_id,item,"fixture",base,rows=rows,control_fixture=control)
    result,audit = h.c.verified(item),h.c.verified(checked)
    assert model_calls == [fit_id,fit_id]  # One specialist fit and its reproduction only.
    assert result["updates"] == 4 and result["row_exposures"] == 20
    assert not result["control_retrained"] and not audit["control_retrained"]
    assert audit["complete_specialist_fit_bitwise_identical"] and audit["all_new_validation_and_selection_independently_reproduced"]
    assert audit["inherited_control_fit_reproduction"] == control["audit"]
    assert result["batch_sequence_sha256"]["specialist"] == previous["batch_sequence_sha256"]["broad"]
    assert audit["selected_updates"] == result["selected_updates"]
    for arm in h.ARMS:
        assert min(result["trajectories"][arm],key=lambda curve:curve["pair_loss"])["update"] == result["selected_updates"][arm]
    for curve in result["trajectories"]["control"]:
        assert curve["checkpoint"]["sha256"] == frozen[curve["checkpoint"]["path"]]
        h.c.verify_hash(curve["checkpoint"])
    if tied:
        assert result["selected_updates"] == {"specialist":0,"control":0}
        assert audit["unchanged_from_anchor"] == {"specialist":True,"control":True}
        # Re-selection uses the new criterion, not the sealed old selection.
        assert previous["selected_updates"]["broad"] != result["selected_updates"]["control"]


@pytest.mark.parametrize("damage",["tensor","config"])
def test_rehashed_checkpoint_damage_rejected_by_independent_reproduction(tmp_path,damage):
    rows = tiny_rows(); fit_id = 161001
    control = saved_control(tmp_path,fit_id,rows)
    base = tmp_path/"new"
    item = h.fit(fit_id,"fixture",base,rows=rows,control_fixture=control)
    result = h.c.verified(item)
    checkpoint = result["trajectories"]["specialist"][1]["checkpoint"]
    path = h.c.verify_hash(checkpoint)
    payload = torch.load(path,weights_only=True,map_location="cpu")
    if damage == "tensor":
        next(iter(payload["model_state_dict"].values())).flatten()[0] += 1
    else:
        payload["model_config"]["fixture_unrecognized"] = True
    torch.save(payload,path); checkpoint["sha256"] = h.old.file_hash(path)
    result_path = h.c.verify_hash(item); h.old.atomic_write(result_path,result)
    mutated = h.c.artifact(result_path,fit=fit_id,index=0)
    with pytest.raises(AssertionError):
        h.audit_fit(fit_id,mutated,"fixture",base,rows=rows,control_fixture=control)


def test_closed_control_provenance_and_production_fixture_guard():
    evidence,result = d.control_evidence(161001)
    assert not h.c.verified(evidence["audit"])["unchanged_from_anchor"]["broad"]
    assert result["updates"] == 1784 and result["row_exposures"] == 114032
    assert result["config"] == h.inherited.CONFIG
    with pytest.raises(AssertionError,match="Fixture cannot target production"):
        h._inputs("fixture",h.OUTPUT/"fixture",tiny_rows())
    with pytest.raises(ValueError,match="no launch"):
        d.prepare_data("fixture",h.OUTPUT)
    for fit_id in h.FITS:
        payload = torch.load(h.c.verify_hash(h.previous.initializer(fit_id)),map_location="cpu",weights_only=True)
        assert payload["model_config"] == asdict(h.old._RAW._model.config)
