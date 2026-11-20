"""Anchor-only identity/provenance checks without loading a network or old corpus."""
import copy
from pathlib import Path

import pytest

from tools.proposal_breadth_128 import models as p


def test_anchor_seal_has_exact_original_checkpoints_and_closed_recovery(tmp_path):
    evidence = p.source_evidence()
    assert len({item["path"] for item in evidence}) == len(evidence)
    assert len([item for item in evidence if item["path"].endswith(".pt")]) == 4
    assert any(item["path"].endswith("recovery_01/closure.json") for item in evidence)
    assert not any("/data/" in item["path"] for item in evidence)
    artifact = p.prepare("fixture", tmp_path)
    value = p.validate("fixture", tmp_path)
    assert p.c.verified(artifact) == value and value["source_evidence"] == evidence
    assert value["models"] == p.anchor_items()
    assert value["continuation_model"] == p.continuation_item()
    assert value["continuation_model"]["sha256"] == p.c.old.CHECKPOINT_SHA
    assert set(value["models"]) == {"anchor_161001", "anchor_161002", "anchor_161003"}
    assert not value["training"] and not value["ancestor_learning_replayed"]
    for fit, (update, digest) in p.ANCHORS.items():
        model = value["models"][f"anchor_{fit}"]
        assert model["sha256"] == digest and model["selected_update"] == update and model["source_fit"] == fit
        assert model["unchanged_from_anchor"]
    with pytest.raises(ValueError, match="One-shot"):
        p.prepare("fixture", tmp_path)
    p.c.verify_hash(artifact)


@pytest.mark.parametrize("change", ["manifest", "role", "update", "source_link", "evidence", "training"])
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
    elif change == "source_link":
        value["source_learning"]["sha256"] = "wrong"
    elif change == "evidence":
        value["source_evidence"].pop()
    else:
        value["training"] = True
    p.c.old.atomic_write(path, value)
    with pytest.raises(ValueError):
        p.validate("fixture", tmp_path)


@pytest.mark.parametrize("field", ["anchor_update", "recovery_complete"])
def test_semantic_parent_receipt_mismatch_rejected_without_editing_parent(monkeypatch, field):
    original = p.c.verified
    def changed(item):
        value = original(item)
        if field == "anchor_update" and Path(item["path"]).name == "models_lock.json":
            value["models"]["anchor_161001"]["selected_update"] += 1
        if field == "recovery_complete" and str(item["path"]).endswith("recovery_01/closure.json"):
            value["complete"] = False
        return value
    monkeypatch.setattr(p.c, "verified", changed)
    with pytest.raises(ValueError, match="anchor fit/update/hash|recovery completion"):
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
