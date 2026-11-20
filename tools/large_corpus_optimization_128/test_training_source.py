"""Saved-state provenance and immutable inherited TRAIN payloads."""
import copy

import pytest
import torch

from . import training_source as s
from .test_learning import parent_values,fixture_kwargs


def test_parent_continuation_authenticates_exact_model_and_optimizer(parent_values):
    _,records = parent_values
    for fit in s.c.FITS:
        result = s.validate_continuation(records[fit],fit,expected_update=16)
        assert result["selected_update"] == 16 and result["value_head_frozen"]


@pytest.mark.parametrize("damage",["missing_state","extra_state","shape","nan","negative_second","step","rate","epsilon","order","model","norms"])
def test_adamw_corruption_rejected_even_if_new_pointer_hash_is_consistent(parent_values,tmp_path,damage):
    record = copy.deepcopy(parent_values[1][s.c.FITS[0]])
    payload = torch.load(s.c.verify_hash(record["optimizer"]),map_location="cpu",weights_only=True)
    state = payload["optimizer_state_dict"]
    if damage == "missing_state":state["state"].pop(0)
    elif damage == "extra_state":state["state"][10000] = state["state"][0]
    elif damage == "shape":state["state"][0]["exp_avg"] = torch.zeros(1)
    elif damage == "nan":state["state"][0]["exp_avg"].flatten()[0] = float("nan")
    elif damage == "negative_second":state["state"][0]["exp_avg_sq"].flatten()[0] = -1
    elif damage == "step":state["state"][0]["step"] += 1
    elif damage == "rate":state["param_groups"][0]["lr"] *= 2
    elif damage == "epsilon":state["param_groups"][0]["eps"] *= 2
    elif damage == "order":state["param_groups"][0]["params"].reverse()
    elif damage == "model":next(iter(payload["model_state_dict"].values())).add_(1)
    else:payload["preclip_gradient_norms"][0] += 1
    path = tmp_path/"optimizer.pt";torch.save(payload,path)
    record["optimizer"] = s.c.artifact(path)
    resume = s.c.verified(record["resume"]);resume["checkpoint"] = record["optimizer"]
    pointer = tmp_path/"resume.json";s.c.old.save_new(pointer,resume)
    record["resume"] = s.c.artifact(pointer)
    with pytest.raises(ValueError):s.validate_continuation(record,s.c.FITS[0],expected_update=16)


def data_fixture(parent_values,tmp_path,monkeypatch):
    values,records = parent_values
    pools = values["rows"]
    monkeypatch.setattr(s,"COUNTS",{key:len(rows) for key,rows in pools.items()})
    files = {}
    for key,rows in pools.items():
        path = tmp_path/f"{key}.pt";torch.save(rows,path);files[key] = s.c.artifact(path)
    original = {"manifest":s.c.PARENT_BINDING,"files":files,
        "row_digests":{key:s.inherited_source.row_digest(rows) for key,rows in pools.items()},
        "partition":values["partitions"]}
    path = tmp_path/"parent_data.json";s.c.old.save_new(path,original)
    parent = {"binding":s.c.PARENT_BINDING,"data":s.c.artifact(path),"continuations":{str(k):v for k,v in records.items()},
        "models":{f"{arm}_{fit}":records[fit]["initializer" if arm == "anchor" else "model"]
                  for arm in ("anchor","control") for fit in s.c.FITS}}
    return values,parent


def test_prepare_reuses_byte_identical_train_only_and_preserves_labels(parent_values,tmp_path,monkeypatch):
    values,parent = data_fixture(parent_values,tmp_path,monkeypatch)
    base = tmp_path/"experiment"
    ref = s.prepare("fixture",base,parent)
    locked = s.c.verified(ref)
    assert locked["new_labels"] == locked["historical_simulations_replayed"] == 0
    assert locked["files"] == s.c.verified(parent["data"])["files"]
    assert s.partitions("fixture",base) == values["partitions"]
    loaded = s.load_pools("fixture",base)
    assert s.inherited._equal_torch(loaded,values["rows"])
    assert s.control_models("fixture",base) == parent["models"]
    assert s.prepare("fixture",base,parent) == ref
    assert s.validate("fixture",base,authenticate_parent=False)["manifest"] == "fixture"
    assert "validation" not in locked["files"]


def test_tensor_and_source_seal_changes_rejected(parent_values,tmp_path,monkeypatch):
    values,parent = data_fixture(parent_values,tmp_path,monkeypatch)
    base = tmp_path/"experiment";s.prepare("fixture",base,parent)
    path = s.c.verify_hash(s.c.verified(parent["data"])["files"]["old"])
    altered = copy.deepcopy(values["rows"]["old"]);altered[0]["old_sums_wdl"][0] += 1
    torch.save(altered,path)
    with pytest.raises(ValueError):s.load_pools("fixture",base)


def test_current_continuation_fit_does_not_trust_parent_optimizer_audit_claim(parent_values):
    record = parent_values[1][s.c.FITS[0]]
    audit = s.c.verified(record["audit"])
    assert "all_new_optimizer_checkpoints_independently_reproduced" not in audit
    # Structural validation is explicit; historical moments are never labelled
    # independently replayed by this new source protocol.
    assert s.validate_continuation(record,s.c.FITS[0],expected_update=16)["complete"]
