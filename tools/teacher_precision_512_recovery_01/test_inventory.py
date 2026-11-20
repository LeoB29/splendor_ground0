"""Read-only recovery ancestry, active-time and stopped-process regressions."""
from datetime import datetime

import pytest

from . import inventory as inv


def _operational():
    base = inv.ORIGINAL
    return [inv.c.old.read_json(base/name) for name in
            ("launch.json", "abort.json", "operational_hold.json", "learning_fits_launch.json", "labels_complete.json")]


def test_exact_original_active_clock_excludes_pause():
    receipts = _operational()
    expected = datetime.fromisoformat(receipts[2]["time"]).timestamp()-receipts[0]["epoch_seconds"]
    assert inv.prior_active_seconds(*receipts) == expected
    assert expected == pytest.approx(47550.61662483215)


@pytest.mark.parametrize("position,key,value", [
    (0, "epoch_seconds", float("nan")), (0, "epoch_seconds", 0),
    (0, "owner_authorized", False), (1, "error", "Different error"),
    (2, "reason", "Different hold"), (2, "time", "2026-10-11T17:00:00+00:00"),
    (3, "manifest", "wrong"), (4, "time", "2026-10-12T09:00:00+00:00"),
])
def test_changed_stop_contract_holds(position, key, value):
    receipts = _operational()
    receipts[position][key] = value
    with pytest.raises(ValueError):
        inv.prior_active_seconds(*receipts)


def test_closure_deduplicates_and_pins_bytes(tmp_path, monkeypatch):
    path = tmp_path/"artifact.json"
    path.write_text('{"value":1}', encoding="utf-8")
    ref = inv._ref(path)
    closure = inv._Closure()
    seen = []
    original = inv.c.verify_hash
    monkeypatch.setattr(inv.c, "verify_hash", lambda r: (seen.append(r["path"]), original(r))[1])
    closure.walk({"one": ref, "two": [ref]})
    assert closure.read(ref) == {"value": 1}
    assert closure.finish() == {ref["path"]: ref["sha256"]}
    assert seen == [ref["path"]]
    with pytest.raises(ValueError, match="Conflicting"):
        closure.add({**ref, "sha256": "0"*64})
    path.write_text('{"value":2}', encoding="utf-8")
    with pytest.raises(ValueError):
        inv._Closure().read(ref)


def test_missing_completion_artifact_holds(tmp_path):
    ref = {"path": str(tmp_path/"missing_complete.json"), "sha256": "0"*64}
    with pytest.raises((ValueError, FileNotFoundError)):
        inv._Closure().read(ref)


@pytest.mark.parametrize("which,key,value", [
    ("root", "unknown_outcomes", 1), ("root", "assessment_read", True),
    ("root", "old_selection", {"path": "changed", "sha256": "0"*64}),
    ("audit", "passed", False), ("audit", "old128_authenticated", False),
    ("audit", "root", {"path": "changed", "sha256": "0"*64}),
    ("audit", "replayed_worlds", 0), ("audit", "all512_sums_independently_checked", False),
])
def test_label_and_audit_link_tampering_holds(which, key, value):
    root_path = inv.ORIGINAL/"labels/roots/state_0000.json"
    root = inv.c.old.read_json(root_path)
    audit = inv.c.old.read_json(inv.ORIGINAL/"labels/audits/state_0000.json")
    root_ref, plan_ref = inv._ref(root_path), inv._ref(inv.ORIGINAL/"label_plan.json")
    inv._check_root(root, audit, root["spec"], root_ref, plan_ref, False)
    {"root": root, "audit": audit}[which][key] = value
    with pytest.raises(ValueError):
        inv._check_root(root, audit, root["spec"], root_ref, plan_ref, False)


def _processes(created="2026-10-12T05:00:00+00:00", exe="C:/pythonw.exe"):
    known = [{"pid": 42, "executable": "C:/pythonw.exe", "role": "supervisor"}]
    observed = [{"pid": 42, "created": created, "executable": exe, "parent_pid": 1}]
    return known, observed, inv._epoch("2026-10-11T16:59:44+00:00"), inv._epoch("2026-10-12T06:12:21+00:00")


def test_live_original_holds_even_with_changed_parent_pid():
    with pytest.raises(ValueError, match="remains live"):
        inv.classify_processes(*_processes())


@pytest.mark.parametrize("created,exe,status", [
    ("2026-10-12T06:13:00+00:00", "C:/pythonw.exe", "reused_pid_outside_original_run"),
    ("2026-10-10T12:00:00+00:00", "C:/pythonw.exe", "reused_pid_outside_original_run"),
    ("2026-10-12T05:00:00+00:00", "C:/unrelated.exe", "reused_pid_other_executable"),
])
def test_demonstrably_reused_pid_is_not_original(created, exe, status):
    assert inv.classify_processes(*_processes(created, exe)) == [{"pid": 42, "status": status}]


@pytest.mark.parametrize("missing", ["created", "executable"])
def test_ambiguous_process_identity_holds(missing):
    args = _processes()
    args[1][0][missing] = None
    with pytest.raises(ValueError):
        inv.classify_processes(*args)


def test_absent_pid_is_stopped():
    known, _, early, held = _processes()
    assert inv.classify_processes(known, [], early, held) == [{"pid": 42, "status": "absent"}]


def _empty_fit_dirs(base):
    for fit in inv.c.FITS:
        (base/"fits"/f"fit_{fit}"/"precision").mkdir(parents=True)


@pytest.mark.parametrize("unexpected", [
    "fits/fit_161001/precision/checkpoint.pt", "fits/fit_161002/precision/training_result.json",
    "source/main/game.json", "models_lock.json", "learning_fits_lock.json", "failure.json",
])
def test_stopped_guard_refuses_any_scientific_advance(tmp_path, unexpected):
    _empty_fit_dirs(tmp_path)
    inv._assert_no_scientific_outputs(tmp_path)
    path = tmp_path/unexpected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError):
        inv._assert_no_scientific_outputs(tmp_path)
    assert path.read_text() == "preserve"


def test_runtime_validation_permits_new_scientific_outputs(tmp_path, monkeypatch):
    _empty_fit_dirs(tmp_path)
    (tmp_path/"fits/fit_161001/precision/result.json").write_text("legitimate", encoding="utf-8")
    receipt = tmp_path/"immutable.json"
    receipt.write_text('{"saved":true}', encoding="utf-8")
    ref = inv._ref(receipt)
    inventory = {ref["path"]: ref["sha256"]}
    evidence = {"binding": inv.BINDING, "original_binding": inv.BINDING,
                "base": str(tmp_path), "stop_error": inv.ABORT_ERROR, "stop_reason": inv.HOLD_REASON,
                "paused_time_excluded": True, "mutable_progress_pinned": False,
                "new_simulations": 0, "historical_replays": 0, "inference_calls": 0,
                "inventory": inventory, "inventory_sha256": inv.c.old.digest(inventory),
                "evidence": [ref], "receipts": {"fixture": ref}, "prior_active_seconds": 47550.,
                "worker_health": [], "label_verification": dict(inv.LABEL_VERIFICATION)}
    def metadata(base, closure):
        closure.read(ref)
        return {"fixture": ref}, {}, 47550.
    monkeypatch.setattr(inv, "_metadata", metadata)
    monkeypatch.setattr(inv, "assert_stopped", lambda *_: pytest.fail("Runtime must not check stopped process state"))
    monkeypatch.setattr(inv, "_assert_no_scientific_outputs", lambda *_: pytest.fail("Runtime must allow added artifacts"))
    monkeypatch.setattr(inv, "_label_closure", lambda *_: pytest.fail("Authenticated bytes preserve initial arithmetic proof"))
    assert inv.validate_inventory(evidence) == inv.BINDING
    assert inv.validate_inventory(evidence, deep=True) == inv.BINDING
    evidence["label_verification"]["target_tensor_checked"] = False
    with pytest.raises(ValueError, match="contract changed"):
        inv.validate_inventory(evidence)
    evidence["label_verification"]["target_tensor_checked"] = True
    receipt.write_text('{"saved":false}', encoding="utf-8")
    with pytest.raises(ValueError):
        inv.validate_inventory(evidence)


def test_complete_real720_stopped_inventory_is_read_only(monkeypatch):
    """One full saved-evidence pass; no state reconstruction, simulation or fit."""
    monkeypatch.setattr(inv.c.old, "initialize_worker", lambda *_: pytest.fail("No model initialization"))
    monkeypatch.setattr(inv.c.old, "trajectory", lambda *_: pytest.fail("No trajectory replay"))
    evidence = inv.stopped_parent()
    assert evidence["original_binding"] == inv.BINDING
    assert evidence["label_verification"]["label_roots"] == 720
    assert evidence["label_verification"]["label_audits"] == 720
    assert evidence["label_verification"]["new_blocks"] == 2160
    assert evidence["label_verification"]["block_completion_receipts"] == 2160
    assert evidence["label_verification"]["target_tensor_checked"] is True
    assert evidence["label_verification"]["unknown_outcomes"] == 0
    assert evidence["label_verification"]["crossfit_score_percentage_points"] == pytest.approx(.4684787326388889)
    assert all(not name.endswith(("progress.json", "run.lock")) for name in evidence["inventory"]
               if name.startswith("outputs/teacher_precision_512/"))
    assert inv.validate_inventory(evidence) == inv.BINDING
