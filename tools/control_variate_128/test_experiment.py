from concurrent.futures import Future
from types import SimpleNamespace
import time
import pytest
from . import common as c, experiment as exp


def protected_markers():
    return {str(folder/name): c.old.file_hash(folder/name)
            for folder in (c.OUTPUT, c.SOURCE, c.CODE_PARENT) for name in exp.RUNTIME_MARKERS
            if (folder/name).is_file()}


def test_abort_destination_dynamic_and_first_reason_preserved(tmp_path, monkeypatch):
    protected = protected_markers()
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    exp.abort(ValueError("fixture first"))
    ref = c.artifact(tmp_path/"abort.json")
    exp.abort(ValueError("fixture second"))
    c.verify_hash(ref)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "fixture first"
    assert protected_markers() == protected


def test_budget_elapsed_disk_abort_and_missing_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="launch receipt"): c.budget_boundary()
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=10*1024**3))
    c.budget_boundary()
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(c.BudgetExceeded, match="disk reserve"): c.budget_boundary()
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(c.time, "time", lambda: c.old.read_json(tmp_path/"launch.json")["epoch_seconds"]+7201)
    with pytest.raises(c.BudgetExceeded, match="Two-hour"): c.budget_boundary()
    exp.abort(ValueError("fixture"), tmp_path)
    with pytest.raises(c.BudgetExceeded, match="Cooperative"): c.budget_boundary()


class FakePool:
    def __init__(self, fail=None): self.submissions=[]; self.fail=fail
    def submit(self, job, arg):
        self.submissions.append(arg)
        future = Future()
        if arg == self.fail: future.set_exception(ValueError("worker failure"))
        else: future.set_result(job(arg))
        return future


def test_dispatch_bounds_queue_and_failure_stops_additional_jobs():
    pool = FakePool(0)
    with pytest.raises(ValueError): exp.dispatch(pool, list(range(1000)), lambda x:x, 3, lambda n:None, lambda:None)
    assert len(pool.submissions) == 3
    def boundary(): raise c.BudgetExceeded("ceiling")
    pool = FakePool()
    with pytest.raises(c.BudgetExceeded): exp.dispatch(pool, list(range(1000)), lambda x:x, 3, lambda n:None, boundary)
    assert len(pool.submissions) == 3
    assert sorted(exp.dispatch(FakePool(), list(range(10)), lambda x:x, 3, lambda n:None, lambda:None)) == list(range(10))


@pytest.mark.parametrize("where", ["worker", "executor_entry"])
def test_pool_failure_signals_abort_before_shutdown(tmp_path, monkeypatch, where):
    protected = protected_markers()
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(c, "budget_boundary", lambda *a: None)
    monkeypatch.setattr(exp, "event", lambda *a, **kw: None)
    monkeypatch.setattr(c.windows, "configure_spawn", lambda: {"fixture": True})
    exited=[]
    class Executor:
        def __init__(self, **kwargs): pass
        def __enter__(self):
            if where == "executor_entry": raise RuntimeError("pool fixture")
            return self
        def submit(self, *args):
            future=Future(); future.set_exception(RuntimeError("pool fixture")); return future
        def __exit__(self, *args):
            assert (tmp_path/"abort.json").exists(); exited.append(True)
    monkeypatch.setattr(exp, "ProcessPoolExecutor", Executor)
    with pytest.raises(RuntimeError, match="pool fixture"):
        exp.run_pool("train", [{"index": 1}], "fixture")
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "pool fixture"
    assert exited == ([True] if where == "worker" else [])
    assert protected_markers() == protected


@pytest.mark.parametrize("marker", exp.RUNTIME_MARKERS)
def test_prelaunch_operations_reject_stale_runtime(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    c.old.save_new(tmp_path/marker, {"preserve": True})
    ref=c.artifact(tmp_path/marker)
    for operation in (lambda: exp.check(tmp_path/"checks"), lambda: exp.prepare(tmp_path/"unused"), exp.launch):
        with pytest.raises(ValueError, match="Runtime evidence"): operation()
        c.verify_hash(ref)
    assert not (tmp_path/"checks").exists()


def stage_fixture(tmp_path, kind, good=True):
    c.old.save_new(tmp_path/f"{kind}_root.json", {"manifest": "fixture"})
    root=c.artifact(tmp_path/f"{kind}_root.json", index=8)
    flags={key: True for key in ("all_endpoint_features_verified", "all_endpoint_values_verified",
        "sampled_prefixes_independently_replayed", "source_world_coupling_verified",
        "statistics_projection_verified", "all_full_outcomes_equal_saved", "matched_timing_arms_verified")}
    if not good: flags.pop("source_world_coupling_verified")
    c.old.save_new(tmp_path/f"{kind}_audit.json", {"manifest": "fixture", "index": 8,
        "result": root, "passed": True, **flags})
    audit=c.artifact(tmp_path/f"{kind}_audit.json", index=8)
    for suffix,ref in (("",root),("_audits",audit)):
        c.old.save_new(tmp_path/f"{kind}{suffix}_lock.json", {"manifest":"fixture","files":[ref]})


@pytest.mark.parametrize("kind", ["timing", "train", "validation"])
def test_stage_gate_requires_specific_audits_not_just_passed(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    stage_fixture(tmp_path,kind,good=False)
    with pytest.raises(ValueError,match="Incomplete stage audit"): exp.verify_stage(kind,"fixture")


def selection_fixture(tmp_path,status="selected",good=True):
    c.old.save_new(tmp_path/"selection_lock.json", {"manifest":"fixture","status":status,
        "config":{"horizon":8,"n":32,"M":128,"beta":.5} if status=="selected" else None})
    ref=c.artifact(tmp_path/"selection_lock.json")
    flags={k:True for k in ("moments_independently_reconstructed","allocation_independently_reproduced","fixed_train_rule_preserved")}
    if not good: flags.pop("allocation_independently_reproduced")
    c.old.save_new(tmp_path/"selection_audit.json", {"manifest":"fixture","passed":True,
        "selection":ref,"validation_read":False,**flags})
    return ref,c.artifact(tmp_path/"selection_audit.json")


def test_selection_gate_requires_independent_allocation(tmp_path,monkeypatch):
    ref,audit=selection_fixture(tmp_path,good=False)
    with pytest.raises(ValueError,match="Incomplete selection audit"): exp.verify_selection("fixture",ref,audit)


@pytest.mark.parametrize("passes",[False,True])
def test_final_pass_or_fail_stops_without_successor(tmp_path,monkeypatch,passes):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    monkeypatch.setattr(exp,"validate_freeze",lambda:"fixture")
    monkeypatch.setattr(exp,"event",lambda *a,**kw:None)
    selected,selected_audit=selection_fixture(tmp_path)
    for kind in ("timing","train","validation"): stage_fixture(tmp_path,kind)
    for name in ("data_lock.json","forecast.json"): c.old.save_new(tmp_path/name,{"fixture":True})
    c.old.save_new(tmp_path/"assessment_result.json",{"manifest":"fixture","selection":selected,"statistics":{"passed":passes}})
    result=c.artifact(tmp_path/"assessment_result.json")
    c.old.save_new(tmp_path/"assessment_audit.json",{"manifest":"fixture","passed":True,"result":result,"selection":selected,
        **{key:True for key in ("moments_independently_reconstructed","bootstrap_independently_reproduced","fixed_train_rule_preserved")}})
    exp.finish("fixture",result,c.artifact(tmp_path/"assessment_audit.json"),selected,selected_audit)
    final=c.old.read_json(tmp_path/"result.json")
    assert final["statistics"]["statistics"]["passed"] is passes
    assert final["new_neural_fits"]==final["new_source_games"]==0
    assert final["automatic_successor"] is final["actual_estimator_player_tested"] is False
    assert c.old.read_json(tmp_path/"final_status.json")["complete"] is True
    assert "not game win rates" in (tmp_path/"report.md").read_text()


@pytest.mark.parametrize("status",["no_feasible_allocation","undefined_zero_training_variance"])
def test_unidentified_cv_stops_without_validation(tmp_path,monkeypatch,status):
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    monkeypatch.setattr(exp,"validate_freeze",lambda:"fixture")
    monkeypatch.setattr(exp,"event",lambda *a,**kw:None)
    selected,audit=selection_fixture(tmp_path,status)
    for kind in ("timing","train"): stage_fixture(tmp_path,kind)
    exp.calibration_stop("fixture",selected,audit)
    final=c.old.read_json(tmp_path/"final_status.json")
    assert final["classification"]==status and final["validation_evaluated"] is False
    assert final["automatic_successor"] is False
    assert c.verified(final["result"])["feasibility_pass"] is False
    assert c.verified(final["audit"])["validation_evaluated"] is False
    c.verify_hash(final["report"])
    assert not (tmp_path/"validation").exists()


def test_forecast_hold_preserves_measured_receipt(tmp_path,monkeypatch):
    from . import engine
    monkeypatch.setattr(exp,"OUTPUT",tmp_path)
    monkeypatch.setattr(engine,"execution_forecast",lambda *args:{"manifest":"fixture","outcome_blind":True,
        "estimated_total_seconds":8000.,"within_runtime_ceiling":False,"disk_reserve_pass":True})
    with pytest.raises(c.BudgetExceeded,match="forecast"):exp.forecast([],"fixture")
    assert c.old.read_json(tmp_path/"forecast.json")["estimated_total_seconds"]==8000.


def test_audit_pairing_rejects_missing_and_duplicate_roots():
    with pytest.raises(ValueError):exp.paired_audit_args([{"index":1},{"index":2}],[{"index":1},{"index":1}])
    assert exp.paired_audit_args([{"index":2},{"index":1}],[{"index":1},{"index":2}])==[
        ({"index":2},{"index":2}),({"index":1},{"index":1})]
