"""Active-time regression, immutable receipts and continuation ordering."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from . import recovery as r


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    study, recovery = tmp_path/"study", tmp_path/"recovery"
    study.mkdir(); recovery.mkdir()
    for name in ("STUDY", "OUTPUT", "BINDING", "REPORTER"):
        monkeypatch.setattr(r, name, getattr(r, name))
    for module, names in ((r.c, ("budget_boundary", "active_seconds")),
            (r.parent, ("OUTPUT", "budget_boundary", "validate_freeze", "worker_initialize", "run_pool", "abort", "event")),
            (r.learning, ("training_report",))):
        for name in names: monkeypatch.setattr(module, name, getattr(module, name))
    r.old.save_new(study/"launch.json", {"epoch_seconds": 100.})
    r.old.save_new(study/"abort.json", {"error": "original timer", "pid": 1})
    r.old.save_new(study/"learning_fits_launch.json", {"preserve": True})
    r.old.save_new(recovery/"launch.json", {"study_binding": "fixture", "prior_active_seconds": 13*3600.,
        "epoch_seconds": 1_000_000., "resolved_abort": r.c.artifact(study/"abort.json")})
    r.old.save_new(recovery/"launch_seal.json", {"study_binding": "fixture", "launch": r.c.artifact(recovery/"launch.json")})
    r.install_runtime(study, recovery, "fixture", fixture=True)
    monkeypatch.setattr(r.time, "time", lambda: 1_000_123.)
    monkeypatch.setattr(r.shutil, "disk_usage", lambda _: SimpleNamespace(free=10*1024**3))
    return study, recovery


def test_thirteen_hour_resume_uses_cumulative_active_not_old_wall_time(runtime):
    study, recovery = runtime
    assert r.active_seconds(study) == 13*3600+123
    r.budget_boundary(study)
    assert r.c.budget_boundary is r.parent.budget_boundary is r.budget_boundary
    assert r.c.active_seconds is r.active_seconds
    assert r.learning.training_report is r.diagnostics.training_report
    assert r.old.read_json(study/"launch.json") == {"epoch_seconds": 100.}


@pytest.mark.parametrize("seconds", [48*3600+1, 49*3600])
def test_actual_cumulative_ceiling_still_enforced(runtime, monkeypatch, seconds):
    monkeypatch.setattr(r.time, "time", lambda: 1_000_000.+seconds-13*3600)
    with pytest.raises(r.c.BudgetExceeded, match="Forty-eight-hour"):
        r.parent.budget_boundary(runtime[0])


def test_prior_runtime_is_never_reset(runtime, monkeypatch):
    monkeypatch.setattr(r.time, "time", lambda: 1_000_000.+35*3600+1)
    with pytest.raises(r.c.BudgetExceeded): r.budget_boundary(runtime[0])


def test_new_abort_blocks_but_original_abort_remains_unchanged(runtime):
    study, recovery = runtime
    before = (study/"abort.json").read_bytes()
    r.abort(ValueError("new failure"), study)
    assert (study/"abort.json").read_bytes() == before
    assert r.old.read_json(recovery/"abort.json")["error"] == "new failure"
    with pytest.raises(r.c.BudgetExceeded, match="cooperative abort"): r.budget_boundary(study)
    r.abort(ValueError("second"), study)
    assert r.old.read_json(recovery/"abort.json")["error"] == "new failure"


def test_changed_old_abort_is_never_silently_ignored(runtime):
    study, _ = runtime
    (study/"abort.json").write_text('{"error":"different"}')
    with pytest.raises(ValueError): r.budget_boundary(study)


def test_missing_receipt_and_wrong_base_rejected(runtime):
    study, recovery = runtime
    (recovery/"launch.json").rename(recovery/"saved_launch.json")
    with pytest.raises(ValueError, match="launch receipt"): r.budget_boundary(study)
    with pytest.raises(ValueError, match="different study base"): r.budget_boundary(study/"wrong")


def test_storage_reserve_enforced(runtime, monkeypatch):
    monkeypatch.setattr(r.shutil, "disk_usage", lambda _: SimpleNamespace(free=5*1024**3-1))
    with pytest.raises(r.c.BudgetExceeded, match="disk reserve"): r.budget_boundary(runtime[0])


@pytest.mark.parametrize("field,value", [("prior_active_seconds", -1), ("prior_active_seconds", float("nan")),
    ("prior_active_seconds", 0), ("epoch_seconds", 1_000_122.),
    ("epoch_seconds", float("inf")), ("study_binding", "wrong"), ("recovery", {"changed": True})])
def test_invalid_runtime_receipt_rejected(runtime, field, value):
    study, recovery = runtime
    row = r.old.read_json(recovery/"launch.json"); row[field] = value
    (recovery/"launch.json").write_text(json.dumps(row))
    with pytest.raises(ValueError): r.budget_boundary(study)


def test_unsealed_launch_cannot_start_work(runtime):
    study, recovery = runtime
    (recovery/"launch_seal.json").rename(recovery/"interrupted_seal.json")
    with pytest.raises(ValueError, match="completion seal missing"): r.budget_boundary(study)


def test_production_paths_cannot_be_rebound_as_fixtures(runtime):
    with pytest.raises(ValueError, match="Fixture touches production"):
        r.install_runtime(r.ORIGINAL_STUDY, runtime[1], "fixture", fixture=True)
    with pytest.raises(ValueError, match="destination changed"):
        r.install_runtime(*runtime, "fixture")


def test_only_recovery_receipts_are_new_and_stage_lock_follows_launch(runtime, monkeypatch):
    study, recovery = runtime
    r.old.save_new(recovery/"freeze.json", {"repair": "fixture"})
    r.old.save_new(study/"freeze.json", {"original": True})
    original = (study/"learning_fits_launch.json").read_bytes()
    monkeypatch.setattr(r, "validate_freeze", lambda **_: "fixture")
    monkeypatch.setattr(r.c.windows, "configure_spawn", lambda: {"fixture": True})
    class Pool:
        def __init__(self, **kwargs):
            assert kwargs["initializer"] is r.worker_initialize
            assert kwargs["initargs"][-2:] == (study, recovery)
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(r, "ProcessPoolExecutor", Pool)
    def dispatch(pool, arguments, job, workers, on_result, boundary):
        assert arguments == list(r.c.FITS) and workers == 3
        refs = []
        for index in range(3):
            boundary()
            path = study/f"fixture_fit_{index}.json"
            r.old.save_new(path, {"index": index})
            refs.append(r.c.artifact(path, index=index))
        on_result(3)
        return list(reversed(refs))
    monkeypatch.setattr(r.parent, "dispatch", dispatch)
    refs = r.run_pool("learning", "fits", list(r.c.FITS), "fixture", [study/"freeze.json"])
    assert [x["index"] for x in refs] == [0, 1, 2]
    assert (study/"learning_fits_launch.json").read_bytes() == original
    lock = r.parent.stage_lock("learning", "fits", "fixture")
    launch = r.c.verified(lock["launch"])
    assert r.c.artifact(recovery/"freeze.json") in launch["dependencies"]
    assert Path(lock["launch"]["path"]).name == "learning_fits_launch.json"
    assert r.c.verify_hash(lock["launch"]).parent == recovery/"stages"


def test_no_label_stage_can_be_dispatched(runtime, monkeypatch):
    monkeypatch.setattr(r, "validate_freeze", lambda: "fixture")
    with pytest.raises(ValueError, match="must never"):
        r.run_pool("labels_main", "collection", [], "fixture", [])


def test_full_pipeline_order_and_final_links_without_recollecting_labels(runtime, monkeypatch):
    study, recovery = runtime
    order = []
    r.old.save_new(recovery/"freeze.json", {"repair": True})
    monkeypatch.setattr(r, "build_learning", lambda binding: order.append("learning"))
    monkeypatch.setattr(r.parent, "build_sources", lambda binding: order.append("source"))
    monkeypatch.setattr(r.parent, "build_stage", lambda stage, binding: order.append(stage))
    monkeypatch.setattr(r.parent, "operational_decision", lambda binding: order.append("forecast") or True)
    def finish(binding):
        order.append("finish"); r.old.save_new(study/"final_status.json", {"complete": True})
    monkeypatch.setattr(r.parent, "finish", finish)
    r.pipeline("fixture")
    assert order == ["learning", "source", "probe", "forecast", "main", "finish"]
    receipt = r.old.read_json(recovery/"final_status.json")
    assert receipt["complete"] and receipt["new_label_simulations"] == 0
    assert receipt["active_seconds"] > 13*3600
    r.c.verify_hash(receipt["study_final_status"])


def test_learning_reuses_existing_target_seal_without_reexport(runtime, monkeypatch):
    study, recovery = runtime
    order = []
    for name in ("target_lock.json", "data_lock.json", "labels_complete.json"):
        r.old.save_new(study/name, {"fixture": name})
    r.old.save_new(recovery/"freeze.json", {"fixture": True})
    monkeypatch.setattr(r.training_source, "validate", lambda *a, **k: None)
    monkeypatch.setattr(r.training_source, "validate_targets", lambda *a: order.append("targets"))
    monkeypatch.setattr(r.training_source, "prepare_targets", lambda *a: pytest.fail("Reexported targets"))
    def pool(stage, kind, args, binding, upstream):
        order.append(kind)
        r.old.save_new(study/f"{stage}_{kind}_lock.json", {"fixture": kind})
        return [{"index": i} for i in range(3)]
    monkeypatch.setattr(r, "run_pool", pool)
    def models(*args):
        order.append("models"); r.old.save_new(study/"models_lock.json", {"fixture": True})
        return r.c.artifact(study/"models_lock.json")
    monkeypatch.setattr(r.learning, "models_lock", models)
    monkeypatch.setattr(r.learning, "validate_models", lambda *a: order.append("checked"))
    r.build_learning("fixture")
    assert order == ["targets", "fits", "fit_audits", "models", "checked"]
    receipt = r.old.read_json(study/"learning_complete.json")
    assert receipt["targets"] == r.c.artifact(study/"target_lock.json")
    assert receipt["recovery"] == r.c.artifact(recovery/"freeze.json")
