"""Bounded orchestration, stage gates, immutable startup and failure preservation."""
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
import time
import pytest
from . import common as c, experiment as p


def cost(**changes):
    args = dict(collection={"selection":1300.,"assessment":700.},
                replay={"selection":1000.,"assessment":500.},
                counts={"selection":10000,"assessment":5000},
                main_counts={"selection":552960,"assessment":276480},
                main_replays={"selection":60000,"assessment":30000},
                structural=100.,audit_structural=200.,size=20*1024**2,
                free=100*1024**3,elapsed=3000.)
    args.update(changes)
    return p.cost_projection(**args)


def test_forecast_includes_both_blocks_and_prior_training_elapsed():
    ordinary = cost()
    assert ordinary["reasonable"] and ordinary["remaining_seconds"] > 300
    assert cost(replay={"selection":1000.,"assessment":1500.})["remaining_seconds"] > ordinary["remaining_seconds"]
    assert cost(audit_structural=400.)["remaining_seconds"] > ordinary["remaining_seconds"]
    assert not cost(elapsed=c.CONFIG["maximum_runtime_hours"]*3600-10)["reasonable"]
    assert not cost(free=ordinary["projected_bytes"]+c.CONFIG["disk_reserve_bytes"]-1)["reasonable"]
    assert ordinary["main_continuations_upper_bound"] == {"selection":552960,"assessment":276480}


@pytest.mark.parametrize("changes", [dict(collection={"selection":0.,"assessment":1.}),
    dict(replay={"selection":1.,"assessment":0.}), dict(counts={"selection":1,"assessment":0}),
    dict(structural=-1.),dict(elapsed=float("nan"))])
def test_incomplete_cost_evidence_never_authorizes_main(changes):
    with pytest.raises(ValueError): cost(**changes)


def fake_run(tmp_path, monkeypatch, gate=True):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p.windows,"assert_windowless_worker",lambda *a:{})
    monkeypatch.setattr(p,"validate_freeze",lambda **kw:"binding")
    monkeypatch.setattr(p,"budget_boundary",lambda *a:None)
    monkeypatch.setattr(p,"event",lambda *a,**kw:None)
    @contextmanager
    def awake(): yield {"fixture":True}
    monkeypatch.setattr(p.parent.parent,"inhibit_system_sleep",awake)
    calls=[]
    monkeypatch.setattr(p,"build_learning",lambda *a:calls.append("learning"))
    monkeypatch.setattr(p,"build_sources",lambda *a:calls.append("sources"))
    monkeypatch.setattr(p,"build_stage",lambda stage,*a:calls.append(stage))
    monkeypatch.setattr(p,"operational_decision",lambda *a:gate)
    monkeypatch.setattr(p,"finish",lambda *a:calls.append("finish"))
    return calls


@pytest.mark.parametrize("gate,expected", [(True,["learning","sources","probe","main","finish"]),
                                         (False,["learning","sources","probe"])])
def test_fixed_pipeline_has_no_arena_or_automatic_successor(tmp_path,monkeypatch,gate,expected):
    calls=fake_run(tmp_path,monkeypatch,gate)
    p.run()
    assert calls == expected
    with pytest.raises(ValueError,match="One-shot"): p.run()


def test_failure_preserves_partial_and_stops_before_main(tmp_path,monkeypatch):
    protected = production_markers()
    calls=fake_run(tmp_path,monkeypatch)
    def broken(*args):
        calls.append("failure")
        c.old.save_new(tmp_path/"partial.json",{"retained":True})
        raise RuntimeError("saved diagnostic")
    monkeypatch.setattr(p,"build_sources",broken)
    with pytest.raises(RuntimeError): p.run()
    assert calls == ["learning","failure"]
    assert (tmp_path/"partial.json").exists() and not (tmp_path/"final_status.json").exists()
    assert "saved diagnostic" in c.old.read_json(tmp_path/"failure.json")["traceback"]
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "saved diagnostic"
    assert production_markers() == protected


def test_forecast_or_source_shortage_is_a_hold_without_retry(tmp_path,monkeypatch):
    calls=fake_run(tmp_path,monkeypatch)
    def short(*args): raise c.BudgetExceeded("Fixed pool shortage")
    monkeypatch.setattr(p,"build_sources",short)
    p.run()
    assert calls == ["learning"]
    assert "shortage" in c.old.read_json(tmp_path/"operational_hold.json")["reason"]
    assert not (tmp_path/"failure.json").exists()


class FakePool:
    def __init__(self,fail=None): self.submissions=[];self.fail=fail
    def submit(self,job,arg):
        self.submissions.append(arg)
        future=Future()
        if arg == self.fail: future.set_exception(ValueError("worker failure"))
        else: future.set_result(job(arg))
        return future


def test_dispatch_never_drains_a_large_queue_after_failure():
    pool=FakePool(0)
    with pytest.raises(ValueError): p.dispatch(pool,list(range(1000)),lambda x:x,3,lambda n:None,lambda:None)
    assert len(pool.submissions) == 3
    def boundary(): raise c.BudgetExceeded("ceiling")
    pool=FakePool()
    with pytest.raises(c.BudgetExceeded): p.dispatch(pool,list(range(1000)),lambda x:x,3,lambda n:None,boundary)
    assert len(pool.submissions) == 3
    assert sorted(p.dispatch(FakePool(),list(range(10)),lambda x:x,3,lambda n:None,lambda:None)) == list(range(10))


def test_worker_exception_signals_other_workers_before_returning(tmp_path,monkeypatch):
    def fail(*args,**kwargs): raise ValueError("bad fit input")
    monkeypatch.setattr(p.learning,"fit",fail)
    with pytest.raises(ValueError): p.run_job("learning","fits",161001,"fixture",tmp_path)
    path=tmp_path/"abort.json"
    assert c.old.read_json(path)["error"] == "bad fit input"
    ref=c.artifact(path)
    p.abort(ValueError("later"),tmp_path)
    c.verify_hash(ref)


def test_budget_covers_elapsed_disk_and_manual_abort(tmp_path,monkeypatch):
    c.old.save_new(tmp_path/"launch.json",{"epoch_seconds":time.time()})
    monkeypatch.setattr(c.shutil,"disk_usage",lambda p:SimpleNamespace(free=10*1024**3))
    c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time,"time",lambda:c.old.read_json(tmp_path/"launch.json")["epoch_seconds"]+86401)
    with pytest.raises(c.BudgetExceeded,match="Twenty-four-hour"): c.budget_boundary(tmp_path)
    p.abort(ValueError("manual"),tmp_path)
    with pytest.raises(c.BudgetExceeded,match="Cooperative"): c.budget_boundary(tmp_path)


def test_production_requires_launch_receipt_and_disk_reserve(tmp_path,monkeypatch):
    monkeypatch.setattr(c,"OUTPUT",tmp_path)
    with pytest.raises(ValueError): c.budget_boundary(tmp_path)
    c.old.save_new(tmp_path/"launch.json",{"epoch_seconds":time.time()})
    monkeypatch.setattr(c.shutil,"disk_usage",lambda p:SimpleNamespace(free=1))
    with pytest.raises(c.BudgetExceeded,match="disk reserve"): c.budget_boundary(tmp_path)


def production_markers():
    """Observe both real output trees while fixture orchestration changes p.OUTPUT."""
    return {str(folder/name):c.old.file_hash(folder/name)
            for folder in (c.OUTPUT,c.PARENT) for name in p.RUNTIME_MARKERS
            if (folder/name).exists()}


def test_abort_default_resolves_active_destination_at_call_time(tmp_path,monkeypatch):
    protected=production_markers()
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    p.abort(ValueError("isolated default"))
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "isolated default"
    assert production_markers() == protected


@pytest.mark.parametrize("where", ["worker", "executor_entry"])
def test_pool_failure_aborts_active_destination_before_shutdown(tmp_path,monkeypatch,where):
    protected=production_markers()
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p,"validate_freeze",lambda **kw:"fixture")
    monkeypatch.setattr(p,"budget_boundary",lambda *a:None)
    monkeypatch.setattr(p,"event",lambda *a,**kw:None)
    monkeypatch.setattr(p.windows,"configure_spawn",lambda:{"fixture":True})
    exited=[]
    class Executor:
        def __init__(self,**kwargs): pass
        def __enter__(self):
            if where == "executor_entry": raise RuntimeError("pool fixture")
            return self
        def submit(self,*args):
            future=Future();future.set_exception(RuntimeError("pool fixture"));return future
        def __exit__(self,*args):
            assert (tmp_path/"abort.json").exists()
            exited.append(True)
    monkeypatch.setattr(p,"ProcessPoolExecutor",Executor)
    with pytest.raises(RuntimeError,match="pool fixture"):
        p.run_pool("source_main","collection",[0,1,2],"fixture",[])
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "pool fixture"
    assert exited == ([True] if where == "worker" else [])
    assert production_markers() == protected


@pytest.mark.parametrize("marker", p.RUNTIME_MARKERS)
def test_preflight_preparation_and_launch_refuse_stale_runtime_evidence(tmp_path,monkeypatch,marker):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    c.old.save_new(tmp_path/marker,{"preserve":True})
    reference=c.artifact(tmp_path/marker)
    for operation in (lambda:p.check(tmp_path/"checks"),lambda:p.prepare(tmp_path/"unused"),p.launch):
        with pytest.raises(ValueError,match="Runtime evidence"): operation()
        c.verify_hash(reference)
    assert not (tmp_path/"checks").exists()


def test_scientific_design_is_matched_and_fresh():
    assert c.ARMS == ("anchor", "relations", "raw", "teacher")
    assert c.CONFIG["updates"] == 1784 and c.CONFIG["train_rows"] == 9461 and c.CONFIG["added_rows"] == 720
    assert c.CONFIG["batch_tag"] == c.coverage.CONFIG["batch_tag"]
    assert c.CONFIG["train_seeds"] == c.coverage.CONFIG["train_seeds"]
    assert c.CONFIG["base_network_frozen"] and c.CONFIG["fresh_evaluation_independent_of_development_result"]
    assert c.CONFIG["maximum_candidates"] == 10 and c.CONFIG["maximum_assessed_choices"] == 4
    assert not c.CONFIG["automatic_retry"] and not c.CONFIG["automatic_successor"]


def test_new_namespaces_are_disjoint_from_previous_complete_experiment():
    from tools.independent_game_coverage_128_recovery_01 import source as prior_source, engine as prior_engine
    newsource = {s["environment_seed"] for stage in ("probe", "main") for s in p.source.schedule(stage)}
    oldsource = {s["environment_seed"] for stage in ("probe", "main") for s in prior_source.schedule(stage)}
    worlds = {seed for stage in ("probe", "main") for i in range(c.CONFIG["states"][stage]) for block in p.BLOCKS for seed in p.engine.world_seeds(stage,i,block)}
    oldworlds = {seed for stage in ("probe", "main") for i in range(c.CONFIG["states"][stage]) for block in p.BLOCKS for seed in prior_engine.world_seeds(stage,i,block)}
    assert len(newsource) == 1134 and len(worlds) == 378*256
    assert newsource.isdisjoint(oldsource|worlds|oldworlds) and worlds.isdisjoint(oldsource|oldworlds)
