"""Exact original learner identity and closed audit-chain regression checks."""
import copy

import pytest

from tools.planner_state_transfer_128 import models as p


@pytest.fixture(scope="module")
def parent():
    return p.closed_parent()


def test_closed_parent_has_exact_nine_original_models_and_training_provenance(parent):
    assert {key: ref["sha256"] for key, ref in parent["models"].items()} == p.EXPECTED
    assert parent["continuation_model"]["sha256"] == p.c.old.CHECKPOINT_SHA
    assert parent["source_binding"] == p.c.LEARNING_BINDING
    assert len({ref["path"] for ref in parent["evidence"]}) == len(parent["evidence"])
    assert parent["training_data"] in parent["evidence"]


@pytest.mark.parametrize("change", ["incomplete", "wrong_generation", "missing_family", "changed_model",
                                   "checkpoint_reselection", "unmatched_control", "fit_audit_failed", "data_audit_failed"])
def test_closed_parent_rejects_changed_provenance(parent, monkeypatch, change):
    original = p.c.verified
    def changed(ref):
        value = copy.deepcopy(original(ref))
        name = ref["path"]
        if change == "incomplete" and name.endswith("proposal_breadth_learning_128/final_status.json"):
            value["complete"] = False
        if change == "wrong_generation" and name.endswith("proposal_breadth_learning_128/learning_complete.json"):
            value["manifest"] = "wrong"
        if name.endswith("proposal_breadth_learning_128/models_lock.json"):
            if change == "missing_family": del value["models"]["broad_161001"]
            if change == "changed_model": value["models"]["broad_161001"]["sha256"] = "0"*64
            if change == "checkpoint_reselection": value["models"]["broad_161001"]["selected_update"] = 1
            if change == "unmatched_control": value["matched_dose_and_batches"] = False
        if change == "fit_audit_failed" and name.endswith("fit_161001/audit.json"):
            value["complete_fit_bitwise_identical"] = False
        if change == "data_audit_failed" and name.endswith("proposal_breadth_learning_128/data_audit.json"):
            value["passed"] = False
        return value
    monkeypatch.setattr(p.c, "verified", changed)
    with pytest.raises(ValueError):
        p.closed_parent()


def test_private_registry_roundtrip_is_one_shot_and_detects_flag_changes(parent, tmp_path, monkeypatch):
    monkeypatch.setattr(p, "closed_parent", lambda: copy.deepcopy(parent))
    ref = p.prepare("fixture", tmp_path)
    value = p.validate("fixture", tmp_path)
    assert p.c.verified(ref) == value
    with pytest.raises((ValueError, FileExistsError)):
        p.prepare("fixture", tmp_path)
    value["ancestor_learning_replayed"] = True
    original = p.c.old.read_json
    monkeypatch.setattr(p.c.old, "read_json", lambda path: value if str(path).endswith("models_lock.json") else original(path))
    with pytest.raises(ValueError, match="seal changed"):
        p.validate("fixture", tmp_path)
