"""Exact reused TRAIN tensors, privacy and scalar card/action feature coverage."""
import copy
import pytest
import torch
from . import training_source as s
from .test_learning import fixture
from .test_model import tensors,representative_state


def data_fixture(tmp_path,monkeypatch):
    values = fixture(tmp_path)
    monkeypatch.setattr(s,"COUNTS",{key:len(rows) for key,rows in values["rows"].items()})
    files = {}
    for key,rows in values["rows"].items():
        path = tmp_path/f"{key}.pt";torch.save(rows,path);files[key] = s.c.artifact(path)
    original = {"manifest":s.c.PARENT_BINDING,"files":files,
        "row_digests":{key:s.inherited_source.row_digest(rows) for key,rows in values["rows"].items()},
        "partition":values["partitions"]}
    path = tmp_path/"parent_data.json";s.c.old.save_new(path,original)
    parent = {"binding":s.c.PARENT_BINDING,"data":s.c.artifact(path),
        "models":{f"anchor_{fit}":values["initializer"] for fit in s.c.FITS}}
    return values,parent


def test_prepare_reuses_identical_bytes_only_train(tmp_path,monkeypatch):
    values,parent = data_fixture(tmp_path,monkeypatch)
    base = tmp_path/"experiment";ref = s.prepare("fixture",base,parent)
    lock = s.c.verified(ref)
    assert lock["new_labels"] == lock["historical_simulations_replayed"] == 0
    assert lock["files"] == s.c.verified(parent["data"])["files"]
    assert s.inherited._equal_torch(s.load_pools("fixture",base),values["rows"])
    assert s.partitions("fixture",base) == values["partitions"]
    assert s.control_models("fixture",base) == parent["models"]
    assert s.initializer(s.c.FITS[0],base) == values["initializer"]
    assert s.prepare("fixture",base,parent) == ref
    assert "validation" not in s.validate("fixture",base,authenticate_parent=False)["files"]


def test_tensor_change_rejected(tmp_path,monkeypatch):
    values,parent = data_fixture(tmp_path,monkeypatch)
    base = tmp_path/"experiment";s.prepare("fixture",base,parent)
    altered = copy.deepcopy(values["rows"]["old"]);altered[0]["old_sums_wdl"][0] += 1
    torch.save(altered,s.c.verify_hash(s.c.verified(parent["data"])["files"]["old"]))
    with pytest.raises(ValueError):s.load_pools("fixture",base)


def feature_rows():
    observation,features,_ = tensors(representative_state())
    row = {"id":"lawful","split":"train","observation":observation[0],"features":features[0]}
    return {"old":[row],"added":[{**row,"id":"second"}]}


def test_all_feature_inputs_both_arms_scalar_audit(tmp_path):
    rows = feature_rows()
    # Feature-only fixture digest shares the full tensor digest's minimal schema.
    for pool in rows.values():
        for row in pool:
            row.update(groups=torch.arange(len(row["features"])),members=[[i] for i in range(len(row["features"]))])
    ref = s.audit_features("fixture",tmp_path,rows=rows)
    result = s.validate_features("fixture",tmp_path,rows=rows)
    assert s.c.verified(ref) == result and result["passed"]
    assert result["all_training_rows_and_legal_actions_checked"]
    assert result["network_inference"] == result["new_simulations"] == 0
    assert set(result["arms"]) == {"raw","relations"}


def test_empty_adamw_allowed_only_at_zero(tmp_path):
    from . import learning as h
    values = fixture(tmp_path)
    model,_,optimizer,_,names = h._model(values["initializer"],"raw",42,values["config"])
    payload = {"optimizer_state_dict":optimizer.state_dict(),"model_state_dict":model.state_dict()}
    s.validate_adamw(payload,names,values["config"],0)
    with pytest.raises(ValueError):s.validate_adamw(payload,names,values["config"],1)
