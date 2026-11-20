"""Original128-only inheritance and exact new lawful128 target payloads."""
import copy
import json
from collections import Counter

import pytest
import torch

from . import training_source as s


@pytest.fixture(scope="module")
def parent(): return s.closed_parent()


def test_real_closed_original128_source_reuse_no_replay_or_512(tmp_path,parent,monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("Historical reconstruction forbidden")
    monkeypatch.setattr(s.coverage,"closed_parent",forbidden)
    monkeypatch.setattr(s.coverage,"reconstruct_added",forbidden)
    monkeypatch.setattr(torch,"save",forbidden)
    ref = s.prepare("fixture",tmp_path,parent)
    lock = s.c.verified(ref)
    assert s.validate("fixture",tmp_path,authenticate_parent=False) == lock
    assert lock["counts"] == s.CORE_COUNTS and lock["old128_targets_only"]
    assert lock["historical_trajectory_replays"] == lock["new_simulations"] == 0
    assert not (tmp_path/"data_lock.json").exists() and not (tmp_path/"added.pt").exists()
    core = torch.load(s.c.verify_hash(lock["files"]["added"]),map_location="cpu",weights_only=True)
    assert len(core) == 720 and s.row_digest(core) == lock["row_digests"]["added"]
    assert [r["index"] for r in core] == list(range(11950,12670))
    assert Counter((r["source_fit"],r["seat"],r["phase"]) for r in core) == {cell:40 for cell in s.CELLS}
    for row in core:
        s.validate_added_row(row)
        assert not any(key in row for key in ("teacher_sums_wdl","teacher_worlds","assessment"))
    models = s.control_models("fixture",tmp_path)
    assert set(models) == {f"{arm}_{f}" for arm in ("anchor","control") for f in s.c.FITS}
    for fit in s.c.FITS:
        assert s.initializer(fit,tmp_path) == parent["models"][f"anchor_{fit}"]
        assert models[f"control_{fit}"] == parent["models"][f"broad_{fit}"]


@pytest.mark.parametrize("damage",["512","oldsplit","parent","assessment","identity"])
def test_inherited_source_tamper_rejected(tmp_path,parent,damage):
    s.prepare("fixture",tmp_path,parent)
    path = tmp_path/"inherited_source_lock.json"
    lock = s.c.old.read_json(path)
    if damage == "512": lock["old128_targets_only"] = False
    elif damage == "oldsplit": lock["files"]["old"] = lock["files"]["validation"]
    elif damage == "parent": lock["parent"]["coverage_data"]["sha256"] = "0"*64
    elif damage == "assessment": lock["assessment_in_payload"] = True
    else: lock["added_identities"][0]["id"] = "changed"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError): s.inherited_lock("fixture",tmp_path)


def test_new128_target_empty_outsider_and_firstmax_ties():
    for values in ([128],[-128,0,128],[0]*8,[-128,0,128,1,128]):
        target = s.target_from_sums(values)
        assert target["teacher"] == max(range(len(values)),key=lambda i:values[i])
        assert target["outsider_teacher"] == (max(range(3,len(values)),key=lambda i:values[i]) if len(values)>3 else -1)
    for values in ([129],[512],[1.],[],[0]*9):
        with pytest.raises(ValueError): s.target_from_sums(values)


def test_actual_new_tensor_strict128_fields_and_canonical_members(tmp_path):
    from .test_labels import fixture_spec
    from . import labels
    spec = fixture_spec()
    labels.collect(spec,"fixture",tmp_path)
    receipt = s.c.verified(labels.audit(spec,"fixture",tmp_path))
    row = torch.load(s.c.verify_hash(receipt["tensor"]),map_location="cpu",weights_only=True)[0]
    assert row["index"] == 12670 and row["old_sums_wdl"] == [128]
    for field,value in s.new_identity(spec).items(): assert row[field] == value
    for damage in ("teacher_worlds","membership","target","id"):
        changed = copy.deepcopy(row)
        if damage == "teacher_worlds": changed["teacher_worlds"] = 512
        elif damage == "membership": changed["members"][0] = torch.tensor([],dtype=torch.long)
        elif damage == "target": changed["old_sums_wdl"][0] = 512
        else: changed["id"] = "other:0"
        with pytest.raises(ValueError): s.validate_added_row(changed)
