"""Immutable reuse, matched-control receipts and TRAIN/outcome firewalls."""
import copy
import json
import time
import uuid
from collections import Counter

import pytest
import torch

from . import common as c, training_source as s


@pytest.fixture(scope="module")
def parent():
    started = time.perf_counter()
    value = s.closed_parent()
    print(f"completed_parent_seconds={time.perf_counter()-started:.3f}")
    return value


@pytest.fixture(scope="module")
def prepared(parent):
    base = c.ROOT/"outputs/teacher_precision_512_checks"/f"source_{uuid.uuid4().hex}"
    started = time.perf_counter()
    ref = s.prepare("source-fixture", base, parent=parent)
    print(f"reuse_prepare_seconds={time.perf_counter()-started:.3f}; fixture={base}")
    return base, c.verified(ref)


def test_completed_controls_and_authenticated_conversion(parent):
    assert parent["result"]["sha256"] == s.RESULT_SHA
    assert parent["audit"]["sha256"] == s.AUDIT_SHA
    assert parent["coverage_data"]["sha256"] == s.DATA_SHA
    assert parent["inventory"] == {r["path"]: r["sha256"] for r in parent["evidence"]}
    assert len(parent["evaluation_sources"]) == 336
    assert set(parent["models"]) == {f"{arm}_{fit}" for arm in ("anchor", "broad") for fit in c.FITS}
    assert parent["conversion_audits_inherited"] and not parent["assessment_in_training_payload"]
    assert parent["historical_fit_replays"] == parent["historical_trajectory_replays"] == parent["new_simulations"] == 0
    for fit in c.FITS:
        for ref in (parent["models"][f"anchor_{fit}"], parent["models"][f"broad_{fit}"],
                    *parent["broad_fits"][str(fit)].values()):
            assert parent["inventory"][ref["path"]] == ref["sha256"]
    for descriptor in parent["evaluation_sources"]:
        for key in ("selection", "root", "audit"):
            assert parent["inventory"][descriptor[key]["path"]] == descriptor[key]["sha256"]


def test_exact_reused_tensors_partition_and_retirement(prepared):
    base, lock = prepared
    old = c.verified(lock["parent"]["coverage_data"])
    original = c.verified(lock["parent"]["original_data"])
    assert all(lock[k] == old[k] for k in s.DATA_FIELDS)
    assert all(lock[k] == original[k] for k in s.PANEL_FIELDS)
    assert not (base/"added.pt").exists()
    assert s.validate("source-fixture", base, authenticate_parent=False) == lock
    subsets = [set(s.narrow_indices(fit, "source-fixture", base)) for fit in c.FITS]
    assert all(len(v) == 180 for v in subsets)
    assert all(a.isdisjoint(b) for i, a in enumerate(subsets) for b in subsets[i+1:])
    assert lock["retired_evaluation_panels"]["permanent"]
    assert len(set(lock["retired_evaluation_panels"]["ids"])) == 720
    assert s.control_models("source-fixture", base) == lock["parent"]["models"]
    for fit in c.FITS:
        assert s.initializer(fit, base) == lock["parent"]["models"][f"anchor_{fit}"]
        assert s.broad_model(fit, "source-fixture", base) == lock["parent"]["models"][f"broad_{fit}"]
        assert s.broad_fit(fit, "source-fixture", base) == lock["parent"]["broad_fits"][str(fit)]["result"]
        assert s.broad_audit(fit, "source-fixture", base) == lock["parent"]["broad_fits"][str(fit)]["audit"]


def test_actual_pools_open_only_training_tensors_no_assessment(prepared, monkeypatch):
    base, lock = prepared
    allowed = {str(c.ROOT/lock["files"][key]["path"]) for key in ("old", "added")}
    real_load, real_read = torch.load, c.old.read_json
    seen = []
    def load(path, *args, **kwargs):
        assert str(path) in allowed
        seen.append(str(path))
        return real_load(path, *args, **kwargs)
    def read(path):
        assert "assessment" not in str(path).lower()
        return real_read(path)
    monkeypatch.setattr(torch, "load", load)
    monkeypatch.setattr(c.old, "read_json", read)
    pools = s.load_original_pools("source-fixture", base)
    assert len(pools["old"]) == 9461 and len(pools["added"]) == 720
    assert len(seen) == 2 and set(seen) == allowed
    assert all(r["split"] == "train" for rows in pools.values() for r in rows)
    assert len({r["pair"] for r in pools["old"]}) == 159
    assert len({r["environment_seed"] for r in pools["added"]}) == 720
    assert Counter(tuple(r["cell"]) for r in pools["added"]) == {
        (p, fit, seat, phase): 20 for p in s.coverage_source.PANELS
        for fit in c.FITS for seat in (0, 1) for phase in range(3)}
    assert all("assessment" not in r and "assessment_outcomes" not in r for rows in pools.values() for r in rows)


def test_actual_validation_and_panel_identity(prepared):
    base, lock = prepared
    validation = s.load_rows("source-fixture", "validation", base)
    assert len(validation) == 2489 and len({r["pair"] for r in validation}) == 42
    assert all(r["split"] == "validation" for r in validation)
    panel = s.load_panel_rows("source-fixture", base)
    assert [r["index"] for r in panel] == list(range(336))
    assert Counter(r["split"] for r in panel) == {"train": 168, "validation": 168}
    by_id = {r["id"]: r for r in validation}
    for row in panel:
        if row["split"] == "validation":
            assert s.row_digest([row]) == s.row_digest([{**by_id[row["id"]], "index": row["index"]}])
    assert not set(r["id"] for r in panel) & set(lock["retired_evaluation_panels"]["ids"])


def test_old_input_api_never_opens_independent_assessment(prepared, monkeypatch):
    base, lock = prepared
    forbidden = {str(c.ROOT/d["root"]["path"]) for d in lock["parent"]["evaluation_sources"]}
    real = c.old.read_json
    def read(path):
        assert str(path) not in forbidden
        return real(path)
    monkeypatch.setattr(c.old, "read_json", read)
    rows = s.old_inputs("source-fixture", base)
    assert len(rows) == 336 and all("assessment" not in r for r in rows)
    assert [r["index"] for r in rows] == list(range(336))
    assert all(len(r["old_sums_wdl"]) == len(r["candidates"]["keys"]) for r in rows)


@pytest.mark.parametrize("change", ["split", "partition", "digest", "retirement", "panel", "assessment", "parent"])
def test_lock_tamper_rejected(prepared, tmp_path, change):
    _, original = prepared
    value = copy.deepcopy(original)
    if change == "split":
        value["files"]["old"] = value["files"]["validation"]
    elif change == "partition":
        value["partition"]["narrow_indices"][str(c.FITS[0])][0] = 719
    elif change == "digest":
        value["row_digests"]["added"] = "0"*64
    elif change == "retirement":
        value["retired_evaluation_panels"]["permanent"] = False
    elif change == "panel":
        value["schedule"][0]["split"] = "validation"
    elif change == "assessment":
        value["assessment_in_payload"] = True
    else:
        value["parent"]["coverage_data"]["sha256"] = "0"*64
    (tmp_path/"data_lock.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        s.validate("source-fixture", tmp_path, authenticate_parent=False)


@pytest.mark.parametrize("change", ["audit", "partition", "overlap", "targets", "retirement"])
def test_conversion_receipt_rejects_weakened_claims(parent, change):
    data = c.verified(parent["coverage_data"])
    original = c.verified(parent["original_data"])
    audit = c.verified(parent["conversion_audit"])
    if change == "audit":
        audit["independent_public_private_invariant_encoding"] = False
    elif change == "partition":
        data["partition"]["blocks"][0][0] = 719
    elif change == "overlap":
        proof = data["parent"]["training_disjointness"]
        proof["added_environment_seeds"][0] = proof["historical_environment_seeds"][0]
    elif change == "targets":
        audit["every128_world_target_independently_summed"] = False
    else:
        data["retired_evaluation_panels"]["permanent"] = False
    with pytest.raises(ValueError):
        s._data_receipt(data, original, audit)


@pytest.mark.parametrize("change", ["lr", "model", "initializer", "assessment", "reproduction", "exposures", "batch", "partition"])
def test_matched_broad_receipt_tamper_rejected(parent, change):
    fit = c.FITS[0]
    fr, ar = parent["broad_fits"][str(fit)].values()
    trained, checked = c.verified(fr), c.verified(ar)
    model, anchor = parent["models"][f"broad_{fit}"], parent["models"][f"anchor_{fit}"]
    partition = c.verified(parent["coverage_data"])["partition"]
    assert s.fit_receipt(trained, checked, fr, fit, model, anchor, partition)
    if change == "lr":
        trained["config"]["lr"] = 1e-5
    elif change == "model":
        trained["arms"]["broad"]["selected_model"]["sha256"] = "0"*64
    elif change == "initializer":
        trained["initializer"]["sha256"] = "0"*64
    elif change == "assessment":
        trained["assessment_read"] = True
    elif change == "reproduction":
        checked["arms"]["broad"]["complete_fit_bitwise_identical"] = False
    elif change == "exposures":
        trained["arms"]["broad"]["row_exposures"] -= 1
    elif change == "batch":
        checked["arms"]["broad"]["batch_sequence_sha256"] = "0"*64
    else:
        trained["narrow_indices"][0] = 719
    with pytest.raises(ValueError):
        s.fit_receipt(trained, checked, fr, fit, model, anchor, partition)


def test_tensor_target_tamper_rejected(prepared, monkeypatch):
    base, _ = prepared
    real = torch.load
    def changed(*args, **kwargs):
        rows = real(*args, **kwargs)
        rows[0]["old_sums_wdl"][0] += 1
        return rows
    monkeypatch.setattr(torch, "load", changed)
    with pytest.raises(ValueError, match="Tensor contents"):
        s.load_rows("source-fixture", "added", base)


def test_no_new_tensor_export_or_historical_reconstruction(prepared, parent, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical replay or conversion is forbidden")
    monkeypatch.setattr(s.coverage_source, "closed_parent", forbidden)
    monkeypatch.setattr(s.coverage_source, "reconstruct_added", forbidden)
    monkeypatch.setattr(torch, "save", forbidden)
    base = prepared[0]/"no_reconversion"
    s.prepare("source-fixture", base, parent=parent)
    assert s.validate("source-fixture", base, authenticate_parent=False)["tensor_bytes_reused"]
    with pytest.raises(ValueError, match="One-shot"):
        s.prepare("source-fixture", base, parent=parent)



def certified_fixture():
    from .test_learning import row
    value = row(11950,"added",6)
    ref = {"path":"fixture-root","sha256":"1"*64,"index":0}
    root = {"manifest":"fixture","index":0,"row_id":value["id"],"spec":{"index":0},"complete":True,
            "old_sums_wdl":value["old_sums_wdl"],"worlds":512,"sums_wdl512":[4*v for v in value["old_sums_wdl"]],
            "unknown_outcomes":0,"assessment_read":False}
    audit = {"manifest":"fixture","index":0,"row_id":value["id"],"spec":root["spec"],"root":{k:ref[k] for k in ("path","sha256")},
             "passed":True,"unknown_outcomes":0,"assessment_read":False,"old128_authenticated":True,
             "all512_sums_independently_checked":True}
    return value,root,audit,ref


def test_attach512_preserves_all_original_inputs_and_targets():
    value,root,audit,ref = certified_fixture()
    added = s.attach_target(value,root,audit,ref,"fixture")
    assert s.row_digest([{k:v for k,v in added.items() if k not in s.TARGET_FIELDS}]) == s.row_digest([value])
    assert added["teacher_worlds"] == 512 and added["teacher_outsider"] == 4
    assert not set(s.TARGET_FIELDS) & value.keys()


@pytest.mark.parametrize("change",["split","id","index","original","denominator","sum","fraction","unknown","assessment","audit","link","incomplete"])
def test_bad512_target_or_audit_cannot_enter_training(change):
    value,root,audit,ref = certified_fixture()
    if change == "split": value["split"] = "validation"
    elif change == "id": root["row_id"] = "different"
    elif change == "index": root["index"] = 1
    elif change == "original": root["old_sums_wdl"] = [0]*6
    elif change == "denominator": root["worlds"] = 128
    elif change == "sum": root["sums_wdl512"][0] = 513
    elif change == "fraction": root["sums_wdl512"][0] = .5
    elif change == "unknown": root["unknown_outcomes"] = 1
    elif change == "assessment": audit["assessment_read"] = True
    elif change == "audit": audit["all512_sums_independently_checked"] = False
    elif change == "link": audit["root"] = {**ref,"sha256":"2"*64}
    else: root["complete"] = False
    with pytest.raises(ValueError): s.attach_target(value,root,audit,ref,"fixture")


def test_original_data_allowed_before_label_seal_but_precision_fit_blocked(prepared):
    base,lock = prepared
    assert s.validate("source-fixture",base,authenticate_parent=False) == lock
    with pytest.raises(FileNotFoundError): s.load_pools("source-fixture",base)



def test_complete_target_export_and_reload_preserves720_inputs(prepared, tmp_path, monkeypatch):
    from . import labels
    original_base, original_lock = prepared
    # A separate fixture namespace retains the inherited authenticated seal.
    (tmp_path/"data_lock.json").write_text(json.dumps(original_lock),encoding="utf-8")
    original = s.load_rows("source-fixture","added",tmp_path)
    roots,audits = [],[]
    for i,value in enumerate(original):
        root = {"manifest":"source-fixture","index":i,"row_id":value["id"],"spec":{"index":i},
                "complete":True,"old_sums_wdl":value["old_sums_wdl"],"worlds":512,
                "sums_wdl512":[4*v for v in value["old_sums_wdl"]],"unknown_outcomes":0,"assessment_read":False}
        rp=tmp_path/f"fixture_root_{i}.json"; c.old.save_new(rp,root)
        ref=c.artifact(rp,index=i)
        audit={"manifest":"source-fixture","index":i,"row_id":value["id"],"spec":root["spec"],
               "root":{k:ref[k] for k in ("path","sha256")},"passed":True,"unknown_outcomes":0,
               "assessment_read":False,"old128_authenticated":True,"all512_sums_independently_checked":True}
        ap=tmp_path/f"fixture_audit_{i}.json";c.old.save_new(ap,audit)
        roots.append(ref);audits.append(c.artifact(ap,index=i))
    refs=[]
    for name,files in (("collection",roots),("audits",audits)):
        path=tmp_path/f"{name}.json"
        c.old.save_new(path,{"manifest":"source-fixture","complete":True,"files":files})
        refs.append(c.artifact(path))
    calls=[]
    def validate(spec,binding,base,independent=False):
        assert independent and binding=="source-fixture" and Path(base)==tmp_path
        calls.append(spec["index"])
        return c.verified(roots[spec["index"]])
    from pathlib import Path
    monkeypatch.setattr(labels,"validate_root",validate)
    sealed=s.prepare_targets("source-fixture",*refs,tmp_path)
    assert calls==list(range(720))
    assert s.prepare_targets("source-fixture",*refs,tmp_path)==sealed
    loaded=s.load_pools("source-fixture",tmp_path)
    assert len(loaded["old"])==9461 and len(loaded["added"])==720
    stripped=[{k:v for k,v in r.items() if k not in s.TARGET_FIELDS} for r in loaded["added"]]
    assert s.row_digest(stripped)==s.row_digest(original)
    assert all(r["teacher_sums_wdl"]==[4*v for v in r["old_sums_wdl"]] for r in loaded["added"])
    # A late alteration of a certified source artifact breaks the lineage seal.
    with c.verify_hash(roots[0]).open("ab") as handle:handle.write(b" ")
    with pytest.raises(ValueError):s.load_pools("source-fixture",tmp_path)
