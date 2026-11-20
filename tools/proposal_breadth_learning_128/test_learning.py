"""Isolated target/mask/split checks and full miniature two-arm fit reproduction."""
import copy
from dataclasses import asdict
import random

import pytest
import torch

from . import learning as h
from . import learning_data as d


@pytest.fixture(scope="module",autouse=True)
def initialize():
    h.old.initialize_worker()


def root_fixture(support=8):
    scores = [-1,0,1,1,-1,1,1,-1][:support]
    if support > 4:
        scores[2:4] = [0,0]
    candidates = {"keys":[str(i) for i in range(support)],"actions":list(range(support))}
    return {"candidates":candidates,"proposal":{"anchor_candidates":{key:value[:4] for key,value in candidates.items()}},
            "worlds":[{"outcomes":[{"wdl":score,"reason":"completed"} for score in scores]} for _ in range(128)],
            "means":scores,"choice":scores.index(max(scores))}


@pytest.mark.parametrize("support",range(1,9))
def test_targets_retain_prefix_and_first_maximum_for_every_support(support):
    root = root_fixture(support)
    left,right = d.targets(root),d.independent_targets(root)
    assert left == right
    assert left["teacher"] == root["choice"] and left["narrow_support"] == min(support,4)
    assert left["narrow_teacher"] == max(range(min(support,4)),key=lambda i:root["means"][i])
    if support >= 6:
        assert left["teacher"] == 5 and left["narrow_teacher"] == 1


@pytest.mark.parametrize("damage",["prefix","means","choice","worlds","outcome","duplicate"])
def test_corrupt_recorded_supervision_is_rejected_independently(damage):
    root = root_fixture()
    if damage == "prefix":
        root["proposal"]["anchor_candidates"]["actions"][0] = 100
    elif damage == "means":
        root["means"][0] += 1/128
    elif damage == "choice":
        root["choice"] = 6  # Same maximum, but violates first-maximum tie rule.
    elif damage == "worlds":
        root["worlds"].pop()
    elif damage == "outcome":
        root["worlds"][0]["outcomes"][0]["wdl"] = .5
    else:
        root["candidates"]["keys"][-1] = root["candidates"]["keys"][0]
    for function in (d.targets,d.independent_targets):
        with pytest.raises(AssertionError):
            function(root)


def tiny_rows():
    lock = h.old.read_json(h.SOURCE/"data_lock.json")
    values = torch.load(h.c.verify_hash(lock["old_exports"][0]),weights_only=True,map_location="cpu")
    template = next(row for row in values if row["group_count"] >= 9)
    members = [torch.where(template["groups"] == i)[0] for i in range(8)]
    rows = []
    for i in range(7):
        row = {**copy.deepcopy(template),"id":f"{'160' if i < 3 else 'breadth128'}:0:{i}",
               "pair":f"{'160' if i < 3 else 'breadth128'}:{i//2}",
               "split":"train" if i < 5 else "validation","teacher":i%4 if i < 3 else 6,
               "narrow_teacher":i%4,"narrow_support":4,
               "members":copy.deepcopy(members[:4] if i < 3 else members)}
        rows.append(row)
    return rows


def test_authentic_masks_do_not_demote_unoffered_tail_and_group_max_gradients_agree():
    rows = tiny_rows()[3:5]
    logits = torch.linspace(-1,1,len(rows[0]["features"]))[None].repeat(2,1).requires_grad_()
    for arm in ("broad","narrow"):
        losses,choices = h.objective(logits,rows,arm)
        other,other_choices = h.audit_objective(logits,rows,arm)
        assert torch.equal(losses,other) and choices == other_choices
        gradients = torch.autograd.grad(losses.sum(),logits,retain_graph=True)[0]
        assert torch.equal(gradients,torch.autograd.grad(other.sum(),logits,retain_graph=True)[0])
        width = 8 if arm == "broad" else 4
        offered = {int(i) for group in rows[0]["members"][:width] for i in group}
        unused = [i for i in range(logits.shape[1]) if i not in offered]
        assert torch.equal(gradients[:,unused],torch.zeros_like(gradients[:,unused]))
        changed = logits.detach().clone()
        changed[:,unused] = 10000
        assert torch.equal(h.objective(changed,rows,arm)[0],losses.detach())
    changed = logits.detach().clone()
    changed[:,rows[0]["members"][6]] = 100
    assert torch.equal(h.objective(changed,rows,"narrow")[0],h.objective(logits,rows,"narrow")[0])
    assert not torch.equal(h.objective(changed,rows,"broad")[0],h.objective(logits,rows,"broad")[0])


def test_source_pair_splits_keep_all_fits_pooled_and_historical_assignments():
    splits = d.splits()
    assert splits == d.independent_splits() and len(splits) == 602
    assert {key:value for key,value in splits.items() if not key.startswith("breadth128:")} == h.previous.independent_splits()
    schedule = d.teacher_audit.independent_specs("main")
    for fit in h.FITS:
        pairs = {row["pair"] for row in schedule if row["fit"] == fit}
        assert len(pairs) == 67
        assert sum(splits[f"breadth128:{pair}"] == "train" for pair in pairs) == 53
        assert sum(splits[f"breadth128:{pair}"] == "validation" for pair in pairs) == 14
    for left,right in zip(schedule[::2],schedule[1::2],strict=True):
        assert left["pair"] == right["pair"] and left["fit"] == right["fit"]


@pytest.mark.parametrize("kind",["old","new"])
def test_retained_one_shard_preserves_every_tensor_original_target_and_split(tmp_path,kind):
    lock = h.old.read_json(h.SOURCE/"data_lock.json")
    audit = h.c.verified(lock["audit"])
    item,checked = lock[f"{kind}_exports"][0],audit[f"{kind}_export_audits"][0]
    original = torch.load(h.c.verify_hash(item),map_location="cpu",weights_only=True)
    actual = d.retained_shard(kind,item,checked,tmp_path)
    for before,after in zip(original,actual,strict=True):
        for key,value in before.items():
            if isinstance(value,torch.Tensor):
                assert torch.equal(value,after[key])
            elif key == "members":
                assert all(torch.equal(a,b) for a,b in zip(value,after[key],strict=True))
            else:
                assert value == after[key]
        target = before["labels"]["world128"] if kind == "old" else before["teacher"]
        assert after["teacher"] == after["narrow_teacher"] == target
        assert after["narrow_support"] == len(before["members"])


def test_one_actual_breadth_game_export_and_independent_lawful_tensor_audit(tmp_path):
    complete = h.old.read_json(h.c.BREADTH_SOURCE/"main_complete.json")
    item = h.c.verified(complete["game_lock"])["files"][0]
    checked = h.c.verified(complete["audit_lock"])["files"][0]
    source = {"game":item,"audit":checked,"spec":d.teacher_audit.independent_spec("main",0),"parent_binding":d.BREADTH_BINDING}
    exported,audit_item = d.export_game(source,"fixture",tmp_path)
    rows = torch.load(h.c.verify_hash(exported),map_location="cpu",weights_only=True)
    audit = h.c.verified(audit_item)
    assert audit["passed"] and audit["actual_game_replayed"] and audit["every_row_hidden_input_invariance"]
    assert audit["independent_targets"] and audit["continuations_replayed"] == 0
    assert len(rows) == exported["rows"] == audit["coverage"]["rows"]
    for row,root_item in zip(rows,audit["source_roots"],strict=True):
        expected = d.independent_targets(h.c.verified(root_item))
        assert {key:row[key] for key in expected} == expected
    assert any(len(row["members"]) == 8 for row in rows)
    with pytest.raises(AssertionError):
        d.export_game(source,"fixture",tmp_path)


def test_equal_whole_pair_validation_weighting_and_identical_exact_dose_batches():
    assert h.pair_loss([{"pair":"a","loss":0.},{"pair":"b","loss":2.},{"pair":"b","loss":2.}]) == 1.
    template = tiny_rows()[0]
    rows = [{**template,"id":f"row_{129-i:03d}","features":torch.zeros((3+(i*11)%21,4))} for i in range(130)]
    for fit in h.FITS:
        for epoch in range(4):
            batch = h.batch_plan(rows,fit,epoch)
            assert batch == h.independent_plan(rows,fit,epoch)
            assert sorted(map(len,batch)) == [2,64,64]
            assert sorted(index for values in batch for index in values) == list(range(130))
            generator = random.Random(f"proposal-breadth-learning-128-batches-v1|{h.TRAIN_SEEDS[fit]}|{epoch}")
            expected = list(range(130))
            generator.shuffle(expected)
            expected.sort(key=lambda index:len(rows[index]["features"]))
            expected = [expected[start:start+64] for start in range(0,130,64)]
            generator.shuffle(expected)
            assert batch == expected
            assert all([len(rows[index]["features"]) for index in block] == sorted(len(rows[index]["features"]) for index in block)
                       for block in batch)


@pytest.mark.parametrize("fit_id,validation_tied",[(161001,False),(161002,False),(161003,True)])
def test_both_fit_trajectories_reproduce_bitwise_with_common_validation_and_exact_no_update(tmp_path,fit_id,validation_tied):
    rows = tiny_rows()
    if validation_tied:
        for row in rows[5:]:
            row["members"] = row["members"][:1]
            row.update(teacher=0,narrow_teacher=0,narrow_support=1)
    result_item = h.fit(fit_id,"fixture",tmp_path,rows=rows)
    audit_item = h.audit_fit(fit_id,result_item,"fixture",tmp_path,rows=rows)
    result,audit = h.c.verified(result_item),h.c.verified(audit_item)
    assert result_item["fit"] == fit_id and result_item["index"] == h.FITS.index(fit_id)
    assert result["updates"] == 4 and result["row_exposures"] == 20
    assert result["batch_sequence_sha256"]["broad"] == result["batch_sequence_sha256"]["narrow"]
    assert audit["complete_fit_bitwise_identical"] and audit["value_head_frozen"] and audit["common_broad_validation"]
    assert audit["selected_updates"] == result["selected_updates"]
    for arm in h.ARMS:
        curves = result["trajectories"][arm]
        selected = min(curves,key=lambda curve:curve["pair_loss"])
        assert selected["update"] == result["selected_updates"][arm]
        initial = h.c.verified(curves[0]["validation"])
        assert initial == h.c.verified(result["trajectories"]["broad"][0]["validation"])
        assert all(row["teacher"] == (0 if validation_tied else 6) for row in initial["rows"])
        assert all(row["support"] == (1 if validation_tied else 8) for row in initial["rows"])
    if validation_tied:
        assert result["selected_updates"] == {"broad":0,"narrow":0}
        assert audit["unchanged_from_anchor"] == {"broad":True,"narrow":True}
    # Rehashing the result cannot hide a changed checkpoint from reproduction.
    checkpoint = result["trajectories"]["narrow"][1]["checkpoint"]
    path = h.c.verify_hash(checkpoint)
    payload = torch.load(path,map_location="cpu",weights_only=True)
    next(iter(payload["model_state_dict"].values())).flatten()[0] += 1
    torch.save(payload,path)
    checkpoint["sha256"] = h.old.file_hash(path)
    h.old.atomic_write(h.c.verify_hash(result_item),result)
    mutated = h.c.artifact(tmp_path/"fits"/f"fit_{fit_id}"/"result.json",fit=fit_id,index=h.FITS.index(fit_id))
    with pytest.raises(AssertionError):
        h.audit_fit(fit_id,mutated,"fixture",tmp_path,rows=rows)


def test_encoding_configuration_matches_all_original_anchors_and_fixture_guard(tmp_path):
    for fit_id in h.FITS:
        payload = torch.load(h.c.verify_hash(h.previous.initializer(fit_id)),map_location="cpu",weights_only=True)
        assert payload["model_config"] == asdict(h.old._RAW._model.config)
    with pytest.raises(AssertionError,match="Fixture cannot target production"):
        h._inputs("fixture",h.OUTPUT/"fixture",tiny_rows())
    with pytest.raises(ValueError,match="no launch"):
        d.prepare_data("fixture",h.OUTPUT)
