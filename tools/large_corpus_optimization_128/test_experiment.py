"""Bounded pipeline, namespace, preservation, and aged runtime regressions."""
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
import time
import pytest
from . import common as c, experiment as p


def sealed_launch(base, age=0., prior=0.):
    c.old.save_new(base/"launch.json",{"manifest":"fixture","epoch_seconds":time.time()-age,
                                       "prior_active_seconds":prior})
    c.old.save_new(base/"launch_seal.json",{"manifest":"fixture","receipt":c.artifact(base/"launch.json")})


def test_current_boundary_accepts_24_hours_and_enforces_48(tmp_path,monkeypatch):
    sealed_launch(tmp_path,24*3600)
    c.budget_boundary(tmp_path)
    launch=c.old.read_json(tmp_path/"launch.json")
    monkeypatch.setattr(c.time,"time",lambda:launch["epoch_seconds"]+48*3600+1)
    with pytest.raises(c.BudgetExceeded,match="48-hour"):c.budget_boundary(tmp_path)


def test_active_carry_counts_prior_runtime(tmp_path,monkeypatch):
    sealed_launch(tmp_path,2*3600,23*3600)
    assert 25*3600 <= c.active_seconds(tmp_path) < 25*3600+10
    c.budget_boundary(tmp_path)
    launch=c.old.read_json(tmp_path/"launch.json")
    monkeypatch.setattr(c.time,"time",lambda:launch["epoch_seconds"]+26*3600)
    with pytest.raises(c.BudgetExceeded):c.budget_boundary(tmp_path)


@pytest.mark.parametrize("mode",["changed","missing","binding"])
def test_timing_records_cannot_silently_reset_budget(tmp_path,mode):
    sealed_launch(tmp_path,24*3600)
    if mode == "missing":(tmp_path/"launch_seal.json").unlink()
    elif mode == "changed":
        (tmp_path/"launch.json").write_text('{"epoch_seconds":0}')
    else:c.old.save_new(tmp_path/"freeze.json",{"binding":"different"})
    with pytest.raises((ValueError,AssertionError)):c.budget_boundary(tmp_path)


def test_cooperative_abort_and_disk_reserve_remain_enforced(tmp_path,monkeypatch):
    sealed_launch(tmp_path,24*3600)
    p.abort(RuntimeError("fixture abort"),tmp_path)
    with pytest.raises(c.BudgetExceeded,match="Cooperative"):c.budget_boundary(tmp_path)
    clean=tmp_path/"clean";clean.mkdir()
    from types import SimpleNamespace
    monkeypatch.setattr(c.shutil,"disk_usage",lambda _:SimpleNamespace(free=0))
    with pytest.raises(c.BudgetExceeded,match="reserve"):c.budget_boundary(clean)


def test_all_new_source_and_world_domains_are_disjoint():
    schedules={stage:p.source.schedule(stage) for stage in p.SOURCE_STAGES}
    proof=p.validate_namespaces(schedules,{1,2255000000,2305000000,2310000000})
    assert proof["source_games"] == 2214
    assert len(proof["intervals"]) == 6
    with pytest.raises(ValueError,match="Historical source overlap"):
        p.validate_namespaces(schedules,{c.CONFIG["source_seed"]["main"]+1})


def evaluation_cost(**changes):
    args=dict(collection={"selection":1300.,"assessment":700.},
              replay={"selection":1000.,"assessment":500.},
              counts={"selection":10000,"assessment":5000},
              main_counts={"selection":1105920,"assessment":552960},
              main_replays={"selection":100000,"assessment":50000},
              structural=100.,audit_structural=200.,size=20*1024**2,
              free=100*1024**3,elapsed=24*3600.)
    return p.cost_projection(**{**args,**changes})


def test_fresh_forecast_prices_720_roots_and_all_prior_stages():
    assert evaluation_cost()["reasonable"]
    assert not evaluation_cost(elapsed=48*3600)["reasonable"]
    assert evaluation_cost(audit_structural=400)["remaining_seconds"] > evaluation_cost()["remaining_seconds"]
    with pytest.raises(ValueError):evaluation_cost(elapsed=float("nan"))


class FakePool:
    def __init__(self,fail=None):self.fail,self.submissions=fail,[]
    def submit(self,job,arg):
        self.submissions.append(arg);future=Future()
        if arg == self.fail:future.set_exception(ValueError("worker failure"))
        else:future.set_result(job(arg))
        return future


def test_dispatch_does_not_queue_the_whole_three_day_study():
    pool=FakePool(0)
    with pytest.raises(ValueError):p.dispatch(pool,list(range(10800)),lambda x:x,8,lambda n:None,lambda:None)
    assert len(pool.submissions)==8
    assert sorted(p.dispatch(FakePool(),list(range(18)),lambda x:x,8,lambda n:None,lambda:None))==list(range(18))


def fake_run(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p.windows,"assert_windowless_worker",lambda *args:{})
    monkeypatch.setattr(p,"validate_freeze",lambda **kwargs:"fixture")
    monkeypatch.setattr(p,"budget_boundary",lambda *args:None)
    monkeypatch.setattr(p,"event",lambda *args,**kwargs:None)
    @contextmanager
    def awake():yield {"fixture":True}
    monkeypatch.setattr(p.parent.parent,"inhibit_system_sleep",awake)
    calls=[]
    for name in ("build_learning","build_sources","finish"):
        monkeypatch.setattr(p,name,lambda *args,name=name:calls.append(name) or True)
    monkeypatch.setattr(p,"build_stage",lambda stage,*args:calls.append(stage))
    monkeypatch.setattr(p,"operational_decision",lambda *args:True)
    return calls


def test_fixed_pipeline_seals_launch_before_work_and_stops_after_result(tmp_path,monkeypatch):
    calls=fake_run(tmp_path,monkeypatch);p.run()
    assert calls==["build_learning","build_sources","probe","main","finish"]
    assert c.active_seconds(tmp_path)<10
    with pytest.raises(ValueError,match="One-shot"):p.run()


def test_failure_preserves_evidence_without_retry(tmp_path,monkeypatch):
    calls=fake_run(tmp_path,monkeypatch)
    def broken(*args):
        c.old.save_new(tmp_path/"partial.json",{"preserved":True})
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(p,"build_learning",broken)
    with pytest.raises(RuntimeError,match="fixture failure"):p.run()
    assert calls==[]
    assert c.old.read_json(tmp_path/"partial.json")["preserved"]
    assert c.old.read_json(tmp_path/"abort.json")["error"]=="fixture failure"
    assert (tmp_path/"failure.json").exists()


def test_worker_error_signals_other_workers_and_retains_provenance(tmp_path,monkeypatch):
    def bad(*args):raise ValueError("fixture learning input")
    monkeypatch.setattr(p.learning,"fit",bad)
    with pytest.raises(ValueError):p.run_job("learning","fits",(161001,),"fixture",tmp_path)
    assert c.old.read_json(tmp_path/"abort.json")["error"]=="fixture learning input"


def test_production_markers_block_second_launch_without_overwriting(tmp_path):
    c.old.save_new(tmp_path/"launch.json",{"retained":True})
    before=(tmp_path/"launch.json").read_bytes()
    with pytest.raises(ValueError,match="preserve"):p.assert_prelaunch_clean(tmp_path)
    assert (tmp_path/"launch.json").read_bytes()==before


def test_learning_gate_prevents_any_fresh_generation(tmp_path,monkeypatch):
    calls=fake_run(tmp_path,monkeypatch)
    monkeypatch.setattr(p,"build_learning",lambda binding:calls.append("learning_gate_failed") or False)
    p.run()
    assert calls == ["learning_gate_failed"]
    assert not (tmp_path/"source_probe_collection_launch.json").exists()


def test_acquisition_stop_finishes_without_models_or_assessment(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p,"event",lambda *args,**kwargs:None)
    for name in ("learning_complete","learning_fit_audits_lock"):
        c.old.save_new(tmp_path/(name+".json"),{"manifest":"fixture"})
    c.old.save_new(tmp_path/"acquisition_gate.json",{"manifest":"fixture","passed":False,"fits":{}})
    gate=c.artifact(tmp_path/"acquisition_gate.json")
    p.finish_acquisition_stop("fixture",gate)
    result=c.old.read_json(tmp_path/"final_status.json")
    assert result["complete"] and result["all_prescribed_audits_pass"]
    assert result["classification"] == "acquisition_not_established"
    assert result["fresh_evaluation_started"] is False
    assert result["automatic_successor"] is False
    c.verify_hash(result["result"]);c.verify_hash(result["audit"]);c.verify_hash(result["report"])


def test_successful_acquisition_cannot_be_reported_as_failed(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    c.old.save_new(tmp_path/"acquisition_gate.json",{"manifest":"fixture","passed":True})
    with pytest.raises(ValueError,match="Passing acquisition"):
        p.finish_acquisition_stop("fixture",c.artifact(tmp_path/"acquisition_gate.json"))
