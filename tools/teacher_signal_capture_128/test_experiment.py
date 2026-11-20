from concurrent.futures import Future
from types import SimpleNamespace
import time
import pytest
from . import common as c, experiment as exp

def test_resource_decision_is_timing_only_and_includes_storage_and_elapsed():
    kw = dict(collection={b:10 for b in exp.BLOCKS},replay={b:10 for b in exp.BLOCKS},
              counts={b:100 for b in exp.BLOCKS},main_counts={b:10000 for b in exp.BLOCKS},
              main_replays={b:1000 for b in exp.BLOCKS},structural=1,audit_structural=1,
              size=10000,free=10*1024**3,elapsed=100)
    v = exp.cost_projection(**kw)
    assert v["reasonable"] and v["remaining_seconds"] > 300
    assert not exp.cost_projection(**(kw|{"elapsed":8*3600}))["reasonable"]
    assert not exp.cost_projection(**(kw|{"free":c.CONFIG["disk_reserve_bytes"]}))["reasonable"]
    with pytest.raises(ValueError): exp.cost_projection(**(kw|{"collection":{"assessment":-1}}))

def test_budget_uses_new_launch_and_preserves_evidence(tmp_path,monkeypatch):
    c.old.save_new(tmp_path/"launch.json",{"epoch_seconds":time.time()})
    monkeypatch.setattr(c.shutil,"disk_usage",lambda p:SimpleNamespace(free=10*1024**3))
    c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time,"time",lambda: c.old.read_json(tmp_path/"launch.json")["epoch_seconds"]+8*3600+1)
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)
    exp.abort(ValueError("original failure"),tmp_path)
    ref = c.artifact(tmp_path/"abort.json")
    exp.abort(ValueError("secondary"),tmp_path)
    c.verify_hash(ref)
    assert c.old.read_json(tmp_path/"abort.json")["reason"] == "original failure"

class FakePool:
    def __init__(self,fail=None): self.submissions=[];self.fail=fail
    def submit(self,job,arg):
        self.submissions.append(arg)
        f = Future()
        if arg == self.fail: f.set_exception(ValueError("worker broke"))
        else: f.set_result(job(arg))
        return f

def test_bounded_dispatch_stops_queuing_on_worker_or_boundary_failure():
    p = FakePool(fail=0)
    with pytest.raises(ValueError): exp.dispatch(p,list(range(100)),lambda a:a,8,lambda n:None,lambda:None)
    assert len(p.submissions) == 8
    p = FakePool()
    def fail(): raise c.BudgetExceeded("ceiling")
    with pytest.raises(c.BudgetExceeded): exp.dispatch(p,list(range(100)),lambda a:a,8,lambda n:None,fail)
    assert len(p.submissions) == 8
    p = FakePool()
    assert sorted(exp.dispatch(p,list(range(11)),lambda a:a,8,lambda n:None,lambda:None)) == list(range(11))

def test_job_failure_sets_cooperative_abort(tmp_path,monkeypatch):
    def fail(*a,**k): raise ValueError("bad root")
    monkeypatch.setattr(exp.engine,"collect",fail)
    with pytest.raises(ValueError): exp.run_job("test","collection",{},"fixture",tmp_path)
    assert c.old.read_json(tmp_path/"abort.json")["reason"] == "bad root"
