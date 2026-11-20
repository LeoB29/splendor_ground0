from concurrent.futures import Future
from types import SimpleNamespace
import time
import pytest
from . import common as c, experiment as exp


def test_budget_uses_new_launch_and_preserves_first_abort(tmp_path, monkeypatch):
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=10*1024**3))
    c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time, "time", lambda: c.old.read_json(tmp_path/"launch.json")["epoch_seconds"]+7201)
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)
    exp.abort(ValueError("first"), tmp_path)
    ref = c.artifact(tmp_path/"abort.json")
    exp.abort(ValueError("second"), tmp_path)
    c.verify_hash(ref)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "first"


def test_missing_launch_and_disk_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError): c.budget_boundary(tmp_path)
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)


class FakePool:
    def __init__(self, fail=None): self.submissions=[]; self.fail=fail
    def submit(self, job, arg):
        self.submissions.append(arg); value = Future()
        if arg == self.fail: value.set_exception(ValueError("worker failure"))
        else: value.set_result(job(arg))
        return value


def test_dispatch_bounded_after_failure():
    pool = FakePool(0)
    with pytest.raises(ValueError): exp.dispatch(pool, list(range(12)), lambda x:x, 3, lambda n:None, lambda:None)
    assert len(pool.submissions) == 3
    pool = FakePool()
    def fail(): raise c.BudgetExceeded("ceiling")
    with pytest.raises(c.BudgetExceeded): exp.dispatch(pool, list(range(12)), lambda x:x, 3, lambda n:None, fail)
    assert len(pool.submissions) == 3
    assert sorted(exp.dispatch(FakePool(), list(range(10)), lambda x:x, 3, lambda n:None, lambda:None)) == list(range(10))


def test_checkpoint_seal_requires_all_audits_and_fixed_final(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    refs, audits = [], []
    for fit in c.FITS:
        model_path = tmp_path/f"model_{fit}.bin"; model_path.write_bytes(b"model")
        model = c.artifact(model_path)
        result = {"manifest":"fixture", "fit":fit, "selected_update":1784, "updates":1784,
                  "final_always_selected":True, "selected_model":model, "selected_training":model,
                  "batch_sequence_sha256":"fixture", "row_exposures":12, "per_row_exposures":{"fixture":12}}
        path=tmp_path/f"result_{fit}.json"; c.old.save_new(path,result); ref=c.artifact(path,fit=fit)
        audit={"manifest":"fixture", "fit":fit, "passed":True, "complete_fit_bitwise_identical":True,
               "value_head_frozen":True, "result":{k:ref[k] for k in ("path","sha256")},"selected_model":model,
               "all_training_metrics_independently_reproduced":True,
               **{k:result[k] for k in ("final_always_selected","selected_update","selected_training",
                                        "batch_sequence_sha256","row_exposures","per_row_exposures")}}
        apath=tmp_path/f"audit_{fit}.json"; c.old.save_new(apath,audit)
        refs.append(ref);audits.append(c.artifact(apath,fit=fit))
    exp.seal_models("fixture",refs,audits)
    value=c.old.read_json(tmp_path/"models_lock.json")
    assert value["fixed_final"] and value["all_fit_audits_pass"] and len(value["models"])==3
    with pytest.raises(FileExistsError): exp.seal_models("fixture",refs,audits)
    corrupt=c.verified(refs[0]);corrupt["selected_update"]=512
    path=tmp_path/"wrong_checkpoint.json";c.old.save_new(path,corrupt)
    with pytest.raises(ValueError): exp.seal_models("fixture",[c.artifact(path,fit=c.FITS[0]),*refs[1:]],audits)


def test_job_exception_sets_abort(tmp_path, monkeypatch):
    from . import learning
    def fail(*args, **kw): raise ValueError("invalid fitting input")
    monkeypatch.setattr(learning,"fit",fail)
    with pytest.raises(ValueError): exp.run_job("fits",161001,"fixture",tmp_path)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "invalid fitting input"


@pytest.mark.parametrize("mean,lower,passes", [(.0049,.001,False),(.005,0.,False),(.005,.001,True)])
def test_closure_uses_only_fixed_primary_investment_endpoint(tmp_path, monkeypatch, mean, lower, passes):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(exp, "event", lambda *a, **k: None)
    for name in ("models_lock.json", "data_lock.json"):
        c.old.save_new(tmp_path/name, {"manifest": "fixture"})
    # A very strong secondary endpoint must not rescue a failed primary screen.
    stats = {"groups": {"validation": {"contrasts": {
        c.CONFIG["primary_endpoint"]: {"mean_bounds": [mean,mean], "ci95": [lower,.02]},
        "focused_outsider_minus_anchor_outsider": {"mean_bounds": [.25,.25], "ci95": [.2,.3]},
    }}}}
    c.old.save_new(tmp_path/"validation_old_targets.json", {"fixture": True})
    c.old.save_new(tmp_path/"assessment_result.json", {"manifest": "fixture", "statistics": stats,
        "development_screen": {"passed": passes, "primary_endpoint": c.CONFIG["primary_endpoint"]},
        "validation_old_targets": c.artifact(tmp_path/"validation_old_targets.json")})
    result_ref = c.artifact(tmp_path/"assessment_result.json")
    c.old.save_new(tmp_path/"assessment_audit.json", {"manifest": "fixture", "passed": True, "result": result_ref})
    exp.finish("fixture", result_ref, c.artifact(tmp_path/"assessment_audit.json"), [], [])
    result = c.old.read_json(tmp_path/"result.json")
    assert result["development_screen_pass"] is passes
    assert result["new_simulations"] == 0 and result["automatic_successor"] is False
    assert result["primary_endpoint"] == "focused_supported_minus_anchor"
    assert c.old.read_json(tmp_path/"final_status.json")["complete"] is True
