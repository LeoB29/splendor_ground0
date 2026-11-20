from concurrent.futures import Future
from types import SimpleNamespace
import time
import pytest
from . import common as c, experiment as exp


def protected_markers():
    return {str(folder/name): c.old.file_hash(folder/name)
            for folder in (c.OUTPUT, c.PARENT) for name in exp.RUNTIME_MARKERS
            if (folder/name).is_file()}


def test_abort_destination_is_resolved_at_call_and_first_reason_preserved(tmp_path, monkeypatch):
    protected = protected_markers()
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    exp.abort(ValueError("fixture first"))
    reference = c.artifact(tmp_path/"abort.json")
    exp.abort(ValueError("fixture second"))
    c.verify_hash(reference)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "fixture first"
    assert protected_markers() == protected


def test_budget_covers_elapsed_disk_abort_and_missing_launch(tmp_path, monkeypatch):
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


def test_dispatch_bounds_queue_and_stops_after_failure():
    pool = FakePool(0)
    with pytest.raises(ValueError): exp.dispatch(pool, list(range(1000)), lambda x:x, 3, lambda n:None, lambda:None)
    assert len(pool.submissions) == 3
    def boundary(): raise c.BudgetExceeded("ceiling")
    pool = FakePool()
    with pytest.raises(c.BudgetExceeded): exp.dispatch(pool, list(range(1000)), lambda x:x, 3, lambda n:None, boundary)
    assert len(pool.submissions) == 3
    assert sorted(exp.dispatch(FakePool(), list(range(10)), lambda x:x, 3, lambda n:None, lambda:None)) == list(range(10))


def test_job_failure_is_confined_to_fixture(tmp_path, monkeypatch):
    from . import learning
    protected = protected_markers()
    def fail(*args, **kw): raise ValueError("isolated fitting failure")
    monkeypatch.setattr(learning, "fit", fail)
    with pytest.raises(ValueError, match="isolated fitting failure"):
        exp.run_job("fits", 161001, "fixture", tmp_path)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "isolated fitting failure"
    assert protected_markers() == protected


@pytest.mark.parametrize("where", ["worker", "executor_entry"])
def test_pool_failure_signals_abort_before_executor_shutdown(tmp_path, monkeypatch, where):
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
            assert (tmp_path/"abort.json").exists()
            exited.append(True)
    monkeypatch.setattr(exp, "ProcessPoolExecutor", Executor)
    with pytest.raises(RuntimeError, match="pool fixture"):
        exp.run_pool("fits", list(c.FITS), "fixture")
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "pool fixture"
    assert exited == ([True] if where == "worker" else [])
    assert protected_markers() == protected


@pytest.mark.parametrize("marker", exp.RUNTIME_MARKERS)
def test_checks_prepare_launch_reject_stale_runtime_evidence(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    c.old.save_new(tmp_path/marker, {"preserve": True})
    reference = c.artifact(tmp_path/marker)
    for operation in (lambda: exp.check(tmp_path/"checks"), lambda: exp.prepare(tmp_path/"unused"), exp.launch):
        with pytest.raises(ValueError, match="Runtime evidence"): operation()
        c.verify_hash(reference)
    assert not (tmp_path/"checks").exists()


@pytest.mark.parametrize("passes", [False, True])
def test_complete_failed_or_passed_screen_never_starts_successor(tmp_path, monkeypatch, passes):
    from . import learning
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(learning, "validate_models", lambda *a: {"fixture": True})
    monkeypatch.setattr(exp, "event", lambda *a, **kw: None)
    for name in ("models_lock.json", "data_lock.json", "validation_old_targets.json"):
        c.old.save_new(tmp_path/name, {"fixture": True})
    result = {"manifest": "fixture", "development_screen": {"passed": passes},
              "statistics": {"groups": {"validation": {"contrasts": {
                  c.CONFIG["primary_endpoint"]: {"mean_bounds": [.003,.003], "ci95": [-.001,.006]}}}}},
              "validation_old_targets": c.artifact(tmp_path/"validation_old_targets.json")}
    c.old.save_new(tmp_path/"assessment_result.json", result)
    result_ref = c.artifact(tmp_path/"assessment_result.json")
    audit = {"manifest": "fixture", "passed": True, "result": result_ref,
             **{key: True for key in ("all_predictions_independently_reproduced", "all_private_input_checks_pass",
                 "full_validation_old_label_metrics_independently_reproduced",
                 "all_outcome_arithmetic_independently_reproduced", "statistics_independently_reproduced",
                 "fixed_controls_reproduced")}}
    c.old.save_new(tmp_path/"assessment_audit.json", audit)
    exp.finish("fixture", result_ref, c.artifact(tmp_path/"assessment_audit.json"), [], [])
    final = c.old.read_json(tmp_path/"result.json")
    assert final["development_screen_pass"] is passes
    assert final["new_simulations"] == final["new_source_games"] == 0
    assert final["automatic_successor"] is False
    assert c.old.read_json(tmp_path/"final_status.json")["complete"] is True
    assert "not game win rates" in (tmp_path/"report.md").read_text()


def test_final_closure_rejects_incomplete_assessment_audit(tmp_path, monkeypatch):
    from . import learning
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(learning, "validate_models", lambda *a: {"fixture": True})
    c.old.save_new(tmp_path/"assessment_result.json", {"manifest": "fixture"})
    reference = c.artifact(tmp_path/"assessment_result.json")
    c.old.save_new(tmp_path/"assessment_audit.json", {"manifest": "fixture", "passed": True, "result": reference})
    with pytest.raises(ValueError, match="Incomplete assessment audit"):
        exp.finish("fixture", reference, c.artifact(tmp_path/"assessment_audit.json"), [], [])
    assert not (tmp_path/"result.json").exists()
