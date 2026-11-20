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
    base = c.ROOT/"outputs/action_card_relations_128_checks"/f"source_{uuid.uuid4().hex}"
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
    pools = s.load_pools("source-fixture", base)
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


def test_new_relation_features_audit_all_actions_without_inference(tmp_path, monkeypatch):
    from .test_learning import row
    from . import model
    pools = {"old": [row(0), row(1)], "added": [row(11950, "added")]}
    def forbidden(*args, **kwargs):
        pytest.fail("Feature audit must not instantiate or infer with networks")
    monkeypatch.setattr(model, "create_model", forbidden)
    monkeypatch.setattr(model.RelationModel, "forward", forbidden)
    ref = s.audit_features("features-fixture", tmp_path, rows=pools)
    value = c.verified(ref)
    assert s.validate_features("features-fixture", tmp_path, rows=pools) == value
    assert value["counts"] == {"old": 2, "added": 1}
    assert value["totals"]["old"]["legal_action_rows"] == sum(len(row["features"]) for row in pools["old"])
    assert value["network_inference"] == value["new_simulations"] == 0
    assert not value["validation_read"] and not value["assessment_read"]
    value["totals"]["old"]["rows"] -= 1
    (tmp_path/"training_feature_audit.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="coverage changed"):
        s.validate_features("features-fixture", tmp_path, rows=pools)


@pytest.mark.parametrize("damage", ["arithmetic", "mask", "source"])
def test_relation_feature_audit_rejects_changed_inputs(tmp_path, monkeypatch, damage):
    from .test_learning import row
    from . import model
    pools = {"old": [row(0)], "added": [row(11950, "added")]}
    if damage == "source":
        s.audit_features("fixture", tmp_path, rows=pools)
        pools["added"][0]["features"][0, 0] += 1
        with pytest.raises(ValueError, match="identity changed"):
            s.validate_features("fixture", tmp_path, rows=pools)
        return
    real = model.pair_features
    def changed(*args, **kwargs):
        result, mask = real(*args, **kwargs)
        if damage == "mask":
            mask[0, 0] = ~mask[0, 0]
        else:
            result[0, 0, 0, 43] += .01
        return result, mask
    monkeypatch.setattr(model, "pair_features", changed)
    with pytest.raises(ValueError, match="Independent relation"):
        s.audit_features("fixture", tmp_path, rows=pools)
    assert not (tmp_path/"training_feature_audit.json").exists()
