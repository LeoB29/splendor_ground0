"""Authenticate direct completed breadth receipts without historical replay."""
import copy
from pathlib import Path

import pytest

from tools.equal_budget_breadth_128 import models as p


def test_anchor_seal_has_exact_original_checkpoints_and_direct_closed_parent(tmp_path):
    evidence = p.source_evidence()
    assert len({item["path"] for item in evidence}) == len(evidence) == 10
    assert len([item for item in evidence if item["path"].endswith(".pt")]) == 4
    assert {Path(item["path"]).name for item in evidence if item["path"].endswith(".json")} == {
        "freeze.json", "manifest.json", "final_status.json", "final_audit.json", "result.json", "models_lock.json"}
    assert not any("/data/" in item["path"] or "recovery_" in item["path"] for item in evidence)
    artifact = p.prepare("fixture", tmp_path)
    value = p.validate("fixture", tmp_path)
    assert p.c.verified(artifact) == value and value["source_evidence"] == evidence
    assert value["models"] == p.anchor_items()
    assert value["continuation_model"] == p.continuation_item()
    assert value["continuation_model"]["sha256"] == p.c.old.CHECKPOINT_SHA
    assert value["source_binding"] == p.SOURCE_BINDING
    assert value["source_result"]["sha256"] == p.SOURCE_RESULT_SHA
    assert value["source_audit"]["sha256"] == p.SOURCE_AUDIT_SHA
    assert set(value["models"]) == {"anchor_161001", "anchor_161002", "anchor_161003"}
    assert not value["training"] and not value["ancestor_learning_replayed"]
    for fit, (update, digest) in p.ANCHORS.items():
        model = value["models"][f"anchor_{fit}"]
        assert model["sha256"] == digest and model["selected_update"] == update and model["source_fit"] == fit
        assert model["unchanged_from_anchor"]
    with pytest.raises(ValueError, match="One-shot"):
        p.prepare("fixture", tmp_path)
    p.c.verify_hash(artifact)


@pytest.mark.parametrize("change", ["manifest", "role", "update", "source_result", "source_audit", "source_models", "evidence", "training"])
def test_model_seal_mutations_fail_closed(tmp_path, change):
    p.prepare("fixture", tmp_path)
    path = tmp_path / "models_lock.json"
    value = p.c.old.read_json(path)
    if change == "manifest":
        value["manifest"] = "wrong"
    elif change == "role":
        value["models"]["ranking_161001"] = copy.deepcopy(value["models"]["anchor_161001"])
    elif change == "update":
        value["models"]["anchor_161001"]["selected_update"] += 1
    elif change.startswith("source_"):
        value[change]["sha256"] = "wrong"
    elif change == "evidence":
        value["source_evidence"].pop()
    else:
        value["training"] = True
    p.c.old.atomic_write(path, value)
    with pytest.raises(ValueError):
        p.validate("fixture", tmp_path)


@pytest.mark.parametrize("field", ["anchor_update", "source_complete", "audit_model", "published_result", "audit_result", "source_manifest"])
def test_semantic_parent_receipt_mismatch_rejected_without_editing_parent(monkeypatch, field):
    original = p.c.verified
    def changed(item):
        value = original(item)
        name = Path(item["path"]).name
        if field == "anchor_update" and name == "models_lock.json":
            value["models"]["anchor_161001"]["selected_update"] += 1
        if field == "source_complete" and name == "final_status.json":
            value["complete"] = False
        if field == "published_result" and name == "final_status.json":
            value["result"]["sha256"] = "unrecognized-published-result"
        if field == "audit_model" and name == "final_audit.json":
            value["models"]["sha256"] = "unlinked-model-registry"
        if field == "audit_result" and name == "final_audit.json":
            value["result"]["sha256"] = "unlinked-result"
        if field == "source_manifest" and name == "manifest.json":
            value["changed"] = True
        return value
    monkeypatch.setattr(p.c, "verified", changed)
    with pytest.raises(ValueError):
        p.source_evidence()


def test_checkpoint_hash_is_fixed_independently_of_parent_registry(monkeypatch):
    original = p.c.artifact
    def changed(path, **metadata):
        item = original(path, **metadata)
        if Path(path).suffix == ".pt":
            item["sha256"] = "different-network"
        return item
    monkeypatch.setattr(p.c, "artifact", changed)
    with pytest.raises(ValueError, match="Original161 anchor checkpoint changed"):
        p.anchor_items()


def test_continuation_binary_hash_is_checked_without_loading_network(monkeypatch):
    original = p.c.artifact
    target = (p.ROOT / p.c.old.CHECKPOINT).resolve()
    def changed(path, **metadata):
        item = original(path, **metadata)
        if Path(path).resolve() == target:
            item["sha256"] = "changed-continuation-weights"
        return item
    monkeypatch.setattr(p.c, "artifact", changed)
    with pytest.raises(ValueError, match="Frozen035 continuation checkpoint changed"):
        p.source_evidence()


def test_prepare_cannot_write_production_before_launch():
    with pytest.raises(ValueError, match="no launch"):
        p.prepare("fixture", p.OUTPUT / "fixture")
