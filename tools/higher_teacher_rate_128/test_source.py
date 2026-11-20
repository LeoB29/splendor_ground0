"""Immutable data reuse, matched low-rate controls and fitting input firewalls."""
import copy
import json
from collections import Counter
import pytest
import torch
from . import common as c, source as s


@pytest.fixture(scope="module")
def parent():
    return s.closed_parent()


@pytest.fixture(scope="module")
def original(parent):
    return c.verified(parent["reused_data"])


def prepare_fixture(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(s, "closed_parent", lambda: parent)
    s.prepare("fixture-binding", tmp_path)
    return c.old.read_json(tmp_path/"data_lock.json")


def test_completed_parent_exact_data_models_and_control_receipts(parent, original):
    assert parent["lowrate_binding"] == s.LOWRATE_BINDING
    assert parent["result"]["sha256"] == s.LOWRATE_RESULT_SHA
    assert parent["audit"]["sha256"] == s.LOWRATE_AUDIT_SHA
    assert parent["reused_data"]["sha256"] == s.LOWRATE_DATA_SHA
    assert len(parent["models"]) == 15 and len(parent["evaluation_sources"]) == 336
    assert s.validate_schedule(parent["schedule"])
    assert parent["inventory"] == {ref["path"]: ref["sha256"] for ref in parent["evidence"]}
    assert parent["new_simulations"] == parent["historical_fit_replays"] == parent["new_target_reconstructions"] == 0
    assert parent["assessment_used_for_selection"] is False
    assert parent["original_full_conversion_audit_inherited"] is True
    context_status = c.old.read_json(c.SOURCE/"final_status.json")
    assert parent["inventory"][context_status["result"]["path"]] == s.RESULT_SHA
    assert parent["inventory"][context_status["audit"]["path"]] == s.AUDIT_SHA
    context_status_ref = c.artifact(c.SOURCE/"final_status.json")
    assert parent["inventory"][context_status_ref["path"]] == context_status_ref["sha256"]
    for fit in c.FITS:
        result = c.verified(parent["lowrate_fits"][str(fit)]["result"])
        audit = c.verified(parent["lowrate_fits"][str(fit)]["audit"])
        model = parent["models"][f"lowrate_{fit}"]
        assert model["sha256"] == s.LOWRATE_MODELS[fit]
        assert result["row_exposures"] == audit["row_exposures"] == 114044
        assert result["batch_sequence_sha256"] == audit["batch_sequence_sha256"]
        assert model["selected_update"] == result["selected_update"] == 1784
        assert result["initializer"] == parent["models"][f"anchor_{fit}"]
    assert (original["rows"], original["train_rows"], original["validation_rows"]) == (11950, 9461, 2489)


def test_prepare_reuses_exact_bytes_without_loading_tensors_or_roots(parent, original, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation must not load tensors, reconvert roots or play")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(s.previous, "corpus_rows", forbidden)
    monkeypatch.setattr(c.old, "trajectory", forbidden)
    lock = prepare_fixture(parent, tmp_path, monkeypatch)
    assert all(lock[key] == original[key] for key in s.DATA_FIELDS)
    assert lock["manifest"] == "fixture-binding" and lock["parent"] == parent
    assert lock["inherited_data"] == parent["reused_data"]
    assert lock["tensor_bytes_reused"] is True
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data_lock.json"]
    with pytest.raises(FileExistsError): s.prepare("fixture-binding", tmp_path)


def test_real_train_read_never_loads_validation_assessment_or_panel(parent, tmp_path, monkeypatch):
    lock = prepare_fixture(parent, tmp_path, monkeypatch)
    real_load = torch.load
    opened = []
    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert str(path).endswith("train.pt"), "Fitter read a non-TRAIN tensor"
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, "load", guarded)
    rows = s.load_rows("fixture-binding", "train", tmp_path)
    assert len(rows) == 9461 and {row["split"] for row in rows} == {"train"}
    assert len({row["pair"] for row in rows}) == 159
    assert all(set(row) == s.ROW_FIELDS for row in rows)
    ids = s.train_probe_ids("fixture-binding", tmp_path)
    assert len(ids) == len(set(ids)) == 168 and set(ids) <= {row["id"] for row in rows}
    assert len(opened) == 1
    assert lock["files"]["train"]["path"].startswith("outputs/full_teacher_fit_128/")


def test_reused_validation_and_panel_are_exact_original_population(parent, tmp_path, monkeypatch):
    prepare_fixture(parent, tmp_path, monkeypatch)
    rows = s.load_rows("fixture-binding", "validation", tmp_path)
    panel = s.load_panel_rows("fixture-binding", tmp_path)
    assert len(rows) == 2489 and {row["split"] for row in rows} == {"validation"}
    assert len({row["pair"] for row in rows}) == 42
    assert [row["index"] for row in panel] == list(range(336))
    assert [row["id"] for row in panel] == [spec["source_rowid"] for spec in parent["schedule"]]
    assert Counter(row["split"] for row in panel) == {"train": 168, "validation": 168}
    valid_ids = {row["id"] for row in rows}
    assert {row["id"] for row in panel if row["split"] == "validation"} <= valid_ids
    assert valid_ids.isdisjoint(s.train_probe_ids("fixture-binding", tmp_path))


@pytest.mark.parametrize("key,value", [
    ("lr", 2e-4), ("updates", 1785), ("batch_size", 32), ("weight_decay", 0.),
    ("objective", "another_loss"), ("fresh_optimizer", False), ("value_head_frozen", False),
    ("historical_replay", True), ("batch_tag", "different_stream"), ("train_rows", 168),
    ("checkpoint_selection", "best_validation"), ("selection_uses_intermediate_checkpoints", True),
])
def test_rate_is_only_fitting_recipe_change(key, value):
    high = copy.deepcopy(c.CONFIG)
    assert s.validate_recipe(s.previous_common.CONFIG, high)
    high[key] = value
    with pytest.raises(ValueError): s.validate_recipe(s.previous_common.CONFIG, high)


@pytest.mark.parametrize("kind", ["tensor_ref", "tensor_digest", "split_counts", "probe_ids", "panel_ref", "data_identity"])
def test_changed_reuse_metadata_rejected_before_tensor_read(parent, tmp_path, monkeypatch, kind):
    lock = prepare_fixture(parent, tmp_path, monkeypatch)
    if kind == "tensor_ref": lock["files"]["train"] = lock["files"]["validation"]
    if kind == "tensor_digest": lock["row_digests"]["train"] = "0"*64
    if kind == "split_counts": lock["counts"]["train"]["pairs"] = 158
    if kind == "probe_ids": lock["original_train_probe_ids"].reverse()
    if kind == "panel_ref": lock["panel_file"] = lock["files"]["validation"]
    if kind == "data_identity": lock["inherited_data"]["sha256"] = "0"*64
    # Only the new fixture seal is changed. Never alter original frozen evidence.
    (tmp_path/"data_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Changed seal reached tensor loading"))
    with pytest.raises(ValueError): s.load_rows("fixture-binding", "train", tmp_path)


def test_reused_tensor_bytes_are_hashed_on_each_read(parent, tmp_path, monkeypatch):
    prepare_fixture(parent, tmp_path, monkeypatch)
    original_verify = c.verify_hash
    def changed(ref):
        if ref["path"].endswith("/train.pt"):
            raise ValueError("Changed artifact fixture")
        return original_verify(ref)
    monkeypatch.setattr(c, "verify_hash", changed)
    with pytest.raises(ValueError, match="Changed artifact"):
        s.load_rows("fixture-binding", "train", tmp_path)


def test_wrong_binding_split_and_production_missing_freeze_fail(parent, tmp_path, monkeypatch):
    prepare_fixture(parent, tmp_path, monkeypatch)
    with pytest.raises(ValueError): s.load_rows("wrong-binding", "train", tmp_path)
    with pytest.raises(ValueError): s.load_rows("fixture-binding", "test", tmp_path)
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="lacks freeze"):
        s.load_rows("fixture-binding", "train", tmp_path)


def test_controls_and_lowrate_references_are_exact_completed_artifacts(parent, tmp_path, monkeypatch):
    prepare_fixture(parent, tmp_path, monkeypatch)
    controls = s.control_models("fixture-binding", tmp_path)
    assert set(controls) == {f"{family}_{fit}" for family in ("anchor", "specialist", "lowrate") for fit in c.FITS}
    for fit in c.FITS:
        assert s.low_rate_fit(fit, "fixture-binding", tmp_path) == parent["lowrate_fits"][str(fit)]["result"]
        assert s.low_rate_audit(fit, "fixture-binding", tmp_path) == parent["lowrate_fits"][str(fit)]["audit"]
    with pytest.raises(ValueError): s.low_rate_fit(999, "fixture-binding", tmp_path)
    evidence = s.lowrate_evidence("fixture-binding", tmp_path)
    audit = c.verified(c.verified(parent["result"])["assessment_audit"])
    assert evidence["predictions"] == audit["predictions"]
    assert evidence["assessment"] == audit["result"]


def test_old_inputs_only_read_original_data_metadata_and_preassessment_selection(parent, tmp_path, monkeypatch):
    prepare_fixture(parent, tmp_path, monkeypatch)
    original_read = c.verified
    opened = []
    def guarded(ref):
        opened.append(ref["path"])
        assert ref == parent["reused_data"] or "/main/selections/" in ref["path"], "Read an assessment outcome artifact"
        return original_read(ref)
    monkeypatch.setattr(c, "verified", guarded)
    inputs = s.old_inputs("fixture-binding", tmp_path)
    assert len(inputs) == 336 and len(opened) == 337
    assert all(not any(value["old_unknown_counts"]) for value in inputs)
    assert all("assessment" not in value for value in inputs)


def test_validation_can_reuse_previously_authenticated_parent(parent, tmp_path, monkeypatch):
    prepare_fixture(parent, tmp_path, monkeypatch)
    monkeypatch.setattr(s, "closed_parent", lambda: pytest.fail("Duplicate full-parent authentication"))
    lock = s.validate("fixture-binding", tmp_path, authenticate_parent=False)
    assert lock["tensor_bytes_reused"] is True


def test_prepare_rejects_a_parent_differing_from_frozen_manifest(parent, tmp_path):
    manifest = {"source_parent": parent}
    c.old.save_new(tmp_path/"manifest.json", manifest)
    binding = c.old.digest(manifest)
    c.old.save_new(tmp_path/"freeze.json", {"binding": binding, "manifest": c.artifact(tmp_path/"manifest.json")})
    bad = {**parent, "new_simulations": 1}
    with pytest.raises(ValueError, match="differs from new freeze"):
        s.prepare(binding, tmp_path, parent=bad)
