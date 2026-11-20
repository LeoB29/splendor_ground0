"""Bounded completion/model provenance checks; no inference or old corpus reads."""
import copy
from pathlib import Path

import pytest

from tools.proposal_breadth_arena_128 import models as p


def test_exact_anchor_only_seal_through_positive_breadth_and_closed_recovery(tmp_path):
    evidence = p.source_evidence()
    assert len({item["path"] for item in evidence}) == len(evidence)
    assert len([item for item in evidence if item["path"].endswith(".pt")]) == 4
    assert any(item["path"].endswith("recovery_01/closure.json") for item in evidence)
    assert any(item["path"].endswith("proposal_breadth_128/report.md") for item in evidence)
    assert not any("/roots/" in item["path"] or "/data/" in item["path"] for item in evidence)
    item = p.prepare("fixture", tmp_path)
    value = p.validate("fixture", tmp_path)
    assert p.c.verified(item) == value and value["source_evidence"] == evidence
    assert value["models"] == p.anchor_items()
    assert value["continuation_model"] == p.continuation_item()
    assert set(value["models"]) == {f"anchor_{fit}" for fit in p.c.FITS}
    assert value["source_binding"] == p.SOURCE_BINDING
    assert not value["training"] and not value["ancestor_learning_replayed"]
    for fit, (update, digest) in p.ANCHORS.items():
        model = value["models"][f"anchor_{fit}"]
        assert model["sha256"] == digest and model["selected_update"] == update and model["source_fit"] == fit
        assert model["unchanged_from_anchor"]
    with pytest.raises(ValueError, match="One-shot"):
        p.prepare("fixture", tmp_path)
    p.c.verify_hash(item)


@pytest.mark.parametrize("change", ["manifest", "role", "update", "source_link", "evidence", "training", "continuation"])
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
        value["source_stages"]["main"]["sha256"] = "wrong"
    elif change == "evidence":
        value["source_evidence"].pop()
    elif change == "continuation":
        value["continuation_model"]["sha256"] = "wrong"
    else:
        value["training"] = True
    p.c.old.atomic_write(path, value)
    with pytest.raises(ValueError):
        p.validate("fixture", tmp_path)


@pytest.mark.parametrize("field", ["positive", "complete", "audit", "stage", "stage_coverage", "stage_indices", "registry", "result_identity"])
def test_semantic_source_receipt_mutations_fail_closed_without_editing_parent(monkeypatch, field):
    original = p.c.verified
    def changed(item):
        value = original(item)
        name = Path(item["path"]).name
        if field == "positive" and name == "final_status.json":
            value["classification"] = "null"
        elif field == "complete" and name == "final_status.json":
            value["complete"] = False
        elif field == "audit" and name == "final_audit.json":
            value["passed"] = False
        elif field == "stage" and name == "final_audit.json":
            value["stages"]["main"]["sha256"] = "different-stage"
        elif field == "stage_coverage" and name == "main_complete.json":
            value["coverage"]["states"] -= 1
        elif field == "stage_indices" and name == "main_audits_lock.json":
            value["files"][0]["index"] = value["files"][1]["index"]
        elif field == "registry" and name == "models_lock.json":
            value["models"]["anchor_161001"]["selected_update"] += 1
        elif field == "result_identity" and name == "final_status.json":
            value["result"]["sha256"] = "different-result"
        return value
    monkeypatch.setattr(p.c, "verified", changed)
    with pytest.raises(ValueError):
        p.source_evidence()


@pytest.mark.parametrize("role", ["anchor", "continuation"])
def test_checkpoint_hashes_fixed_independently_of_parent_registry(monkeypatch, role):
    original = p.c.artifact
    target = (p.ROOT / p.c.old.CHECKPOINT).resolve()
    def changed(path, **metadata):
        item = original(path, **metadata)
        is_continuation = Path(path).resolve() == target
        if Path(path).suffix == ".pt" and (is_continuation == (role == "continuation")):
            item["sha256"] = "different-network"
        return item
    monkeypatch.setattr(p.c, "artifact", changed)
    with pytest.raises(ValueError, match="checkpoint changed"):
        p.source_evidence()


def test_closed_ranking_provenance_is_checked_without_expired_parent_budget(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not invoke ancestor runtime validation")
    monkeypatch.setattr(p.parent.c, "budget_boundary", forbidden)
    assert p.source_evidence()
    original = p.parent._closed_parent
    def changed():
        value = original()
        value["source_status"]["sha256"] = "changed-closed-ranking-link"
        return value
    monkeypatch.setattr(p.parent, "_closed_parent", changed)
    with pytest.raises(ValueError, match="ranking model provenance"):
        p.source_evidence()


def test_prepare_cannot_write_production_before_launch():
    with pytest.raises(ValueError, match="no launch"):
        p.prepare("fixture", p.OUTPUT / "fixture")
