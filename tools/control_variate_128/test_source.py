"""Bounded source authentication tests; never execute a policy or trajectory."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from . import common as c, source


@pytest.fixture(scope="module")
def actual_parent():
    reads, original = [], c.old.read_json
    def metadata_only(path):
        name = str(path).replace("\\", "/")
        assert "/teacher_signal_capture_128/main/roots/" not in name
        assert "/teacher_signal_capture_128/main/selections/" not in name
        assert "/value_generalization_scale_143/test_predictions/" not in name
        reads.append(name)
        return original(path)
    started = time.perf_counter()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c.old, "read_json", metadata_only)
        parent = source.closed_parent()
    assert reads and parent["assessment_roots_read"] is False
    print(f"source metadata closure seconds={time.perf_counter()-started:.3f}")
    return parent


@pytest.fixture(scope="module")
def prepared(actual_parent, tmp_path_factory):
    base = tmp_path_factory.mktemp("source_closed")
    ref = source.prepare("fixture", base, parent=actual_parent)
    return base, ref


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def seal_selection(base, **changes):
    selected = {"manifest": "fixture", "status": "selected", "selected_before_validation": True,
                "validation_read": False, "config": {"horizon": 8, "beta": .3, "n": 64, "M": 200}}
    selected.update(changes)
    save(base/"selection_lock.json", selected)
    audit = {"manifest": "fixture", "passed": True, "selection": c.artifact(base/"selection_lock.json")}
    save(base/"selection_audit.json", audit)
    return selected, audit


def test_actual_completed_closure_and_scope(actual_parent, prepared):
    parent = actual_parent
    assert parent["binding"] == c.SOURCE_BINDING
    assert parent["value"]["binding"] == source.VALUE_BINDING
    assert parent["value"]["protected_test_predictions_read"] is False
    assert len(parent["descriptors"]) == 336
    assert parent["old_training_replays"] == parent["old_trajectory_replays"] == parent["new_simulations"] == 0
    assert c.verified(prepared[1])["assessment_in_payload"] is False
    lock = source.validate("fixture", prepared[0], authenticate_parent=False)
    assert lock["parent"] == parent
    assert source.evaluator(prepared[0])["sha256"] == source.VALUE_HASHES["simple.json"]


def test_all_roots_pairs_and_split_firewall(prepared):
    base, _ = prepared
    train, val = [source.schedule("fixture", base, split) for split in ("train", "validation")]
    assert len(train) == len(val) == 168
    assert len({s["pair"] for s in train}) == len({s["pair"] for s in val}) == 42
    assert {s["pair"] for s in train}.isdisjoint(s["pair"] for s in val)
    assert Counter(s["fit"] for s in train) == Counter({f: 56 for f in c.FITS})
    assert sorted(s["index"] for s in train+val) == list(range(336))
    for splitrows in (train, val):
        assert [s["index"] for s in splitrows] == sorted(s["index"] for s in splitrows)
        assert all(count == 4 for count in Counter(s["pair"] for s in splitrows).values())


def test_timing_balance_and_deterministic_full_stress(prepared, actual_parent):
    base, _ = prepared
    timing = source.timing_specs("fixture", base)
    assert len(timing) == 18 and all(s["split"] == "train" for s in timing)
    assert Counter((s["fit"], s["seat"], s["phase"]) for s in timing) == Counter({(f, s, p): 1 for f in c.FITS for s in (0, 1) for p in range(3)})
    for spec in actual_parent["schedule"]:
        replay = source.audit_indices(spec)
        assert replay == sorted(set(replay)) and len(replay) == (128 if spec["cv_stress"] else 8)
        if spec["split"] == "train":
            assert len(source.timing_world_indices(spec)) == 8
    for split, cells in actual_parent["stress_cells"].items():
        selected = {s["index"] for s in actual_parent["schedule"] if s["split"] == split and s["cv_stress"]}
        assert len(cells) == 39 and len(selected) <= 39
        assert selected == {r["selected_index"] for r in cells if r["selected_index"] is not None}


def test_audit_plan_is_outcome_blind(actual_parent):
    rows = [{**d["spec"], "support": d["support"], "reference": d["reference"]} for d in actual_parent["descriptors"]]
    first = source.plans(rows)
    for row in rows:
        row["reference"] = (row["reference"]+1) % row["support"]
    second = source.plans(rows)
    assert first["timing_indices"] == second["timing_indices"]
    assert first["stress_indices"] == second["stress_indices"]
    assert [s["cv_audit_indices"] for s in first["schedule"]] == [s["cv_audit_indices"] for s in second["schedule"]]


def test_validation_never_opens_root_without_both_seals(prepared, monkeypatch):
    base, _ = prepared
    spec = source.schedule("fixture", base, "validation")[0]
    def forbidden(*_):
        raise AssertionError("Scientific artifact opened before barrier")
    monkeypatch.setattr(c, "verified", forbidden)
    for allow in (False, True):
        with pytest.raises(ValueError, match="audited TRAIN selection"):
            source.load_root(spec, "fixture", base, stage="validation", allow_validation=allow)


def test_cross_split_read_and_schedule_substitution_rejected(prepared):
    base, _ = prepared
    spec = source.schedule("fixture", base, "validation")[0]
    with pytest.raises(ValueError, match="split/stage"):
        source.load_root(spec, "fixture", base)
    changed = deepcopy(source.schedule("fixture", base)[0]); changed["reference"] ^= 1
    with pytest.raises(ValueError, match="sealed schedule"):
        source.load_root(changed, "fixture", base)


def test_actual_train_target_is_independent_complete_matrix(prepared):
    base, _ = prepared
    spec = source.schedule("fixture", base)[0]
    root, ref = source.load_root(spec, "fixture", base)
    assert ref["sha256"] and root["spec"] == source._original(spec)
    assert root["old_teacher_choice"] == spec["reference"]
    assert len(root["assessment"]["worlds"]) == 128
    assert root["assessment"]["candidate_indices"] == list(range(spec["support"]))
    assert set(root["assessment"]["world_seeds"]).isdisjoint(root["old_selection"]["world_seeds"])
    assert all(len(w["outcomes"]) == spec["support"] for w in root["assessment"]["worlds"])


@pytest.mark.parametrize("mutation", ["reference", "seed", "candidate", "unknown", "actor", "premature", "old_target"])
def test_saved_root_semantic_tampering_fails(prepared, mutation):
    base, _ = prepared
    spec = source.schedule("fixture", base)[0]
    descriptor = c.old.read_json(base/"data_lock.json")["parent"]["descriptors"][spec["index"]]
    root = c.verified(descriptor["root"]); selected = c.verified(descriptor["selection"])
    if mutation == "reference": root["old_teacher_choice"] = (root["old_teacher_choice"]+1) % spec["support"]
    elif mutation == "seed": root["assessment"]["worlds"][0]["seed"] += 1
    elif mutation == "candidate": root["assessment"]["candidate_indices"].reverse()
    elif mutation == "unknown": root["assessment"]["worlds"][0]["outcomes"][0]["wdl"] = None
    elif mutation == "actor": root["snapshot"]["current_player"] ^= 1
    elif mutation == "premature": root["assessment_started_at_ns"] = 0
    else: root["assessment"] = root["old_selection"]
    with pytest.raises(ValueError): source._check_root(root, selected, descriptor)


@pytest.mark.parametrize("mutation", ["missing_audit", "failed_audit", "unbound_audit", "validation_read", "wrong_manifest", "bad_allocation"])
def test_selection_barrier_tampering(tmp_path, mutation):
    selected, audit = seal_selection(tmp_path)
    if mutation == "missing_audit": (tmp_path/"selection_audit.json").unlink()
    elif mutation == "failed_audit": audit["passed"] = False; save(tmp_path/"selection_audit.json", audit)
    elif mutation == "unbound_audit": selected["config"]["beta"] = .4; save(tmp_path/"selection_lock.json", selected)
    elif mutation == "validation_read": seal_selection(tmp_path, validation_read=True)
    elif mutation == "wrong_manifest": seal_selection(tmp_path, manifest="different")
    else:
        selected["config"]["M"] = selected["config"]["n"]
        save(tmp_path/"selection_lock.json", selected)
        audit["selection"] = c.artifact(tmp_path/"selection_lock.json"); save(tmp_path/"selection_audit.json", audit)
    with pytest.raises(ValueError): source.selection_barrier("fixture", tmp_path)


def test_valid_barrier_allows_real_validation_read(actual_parent, tmp_path):
    source.prepare("fixture", tmp_path, parent=actual_parent)
    seal_selection(tmp_path)
    spec = source.schedule("fixture", tmp_path, "validation")[0]
    root, _ = source.load_root(spec, "fixture", tmp_path, stage="validation")
    assert root["spec"]["split"] == "validation" and len(root["assessment"]["worlds"]) == 128


def test_changed_source_file_hash_rejected(actual_parent, tmp_path):
    altered = deepcopy(actual_parent)
    original = c.verified(altered["descriptors"][0]["root"])
    bad = tmp_path/"changed_root.json"; save(bad, original)
    altered["descriptors"][0]["root"]["path"] = str(bad)
    # Keep the authenticated original hash, rather than blessing modified bytes.
    source.prepare("fixture", tmp_path, parent=altered)
    with pytest.raises(ValueError, match="Changed artifact"):
        source.load_root(altered["schedule"][0], "fixture", tmp_path)


def test_production_guard_uses_current_output(actual_parent, tmp_path, monkeypatch):
    source.prepare("fixture", tmp_path, parent=actual_parent)
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="requires freeze"):
        source.schedule("fixture", tmp_path)


def test_data_lock_tamper_rejected(prepared, tmp_path):
    value = c.verified(prepared[1]); value["schedule"][0]["cv_stress"] = not value["schedule"][0]["cv_stress"]
    save(tmp_path/"data_lock.json", value)
    with pytest.raises(ValueError, match="differs from authenticated"):
        source.validate("fixture", tmp_path, authenticate_parent=False)


def test_recomputed_parent_plan_tamper_rejected(actual_parent):
    value = deepcopy(actual_parent)
    value["schedule"][0]["cv_audit_indices"] = [0]
    with pytest.raises(ValueError, match="plan changed"):
        source._parent_receipt(value)


def test_validation_timing_rejected(prepared):
    spec = source.schedule("fixture", prepared[0], "validation")[0]
    with pytest.raises(ValueError, match="TRAIN"):
        source.timing_world_indices(spec)


def test_preparation_is_one_shot(prepared, actual_parent):
    with pytest.raises(FileExistsError): source.prepare("fixture", prepared[0], parent=actual_parent)
