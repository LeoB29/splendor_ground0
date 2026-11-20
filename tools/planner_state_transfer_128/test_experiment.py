from concurrent.futures import Future
import copy
from types import SimpleNamespace
import time
import pytest
from . import common as c, statistics as stats, experiment as exp

def rows():
    result = []
    pair = 0
    for fit in c.FITS:
        for phase in range(3):
            for arm in ("candidate","reference"):
                for _ in range(10):
                    for seat in (0,1):
                        result.append({"index":len(result),"pair":pair,"game_index":pair*2+seat,"fit":fit,
                            "phase":phase,"seat":seat,"source_arm":arm,
                            "delta_bounds":{k:[2/256,2/256] for k in stats.NAMES}})
                    pair += 1
    return result

def test_exact_constant_effect_and_conjunctive_gates():
    data = rows()
    value = stats.summarize(data)
    assert value["gate_pass"] and value["pairs"] == 180
    assert all(r == {"mean":2/256,"mean_bounds":[2/256]*2,"ci95":[2/256]*2} for r in value["contrasts"].values())
    for row in data: row["delta_bounds"]["broad_minus_anchor"] = [1/256]*2
    value = stats.summarize(data)
    assert value["anchor_superiority"] and not value["anchor_practical_gain"] and not value["gate_pass"]
    for row in data: row["delta_bounds"]["broad_minus_anchor"] = [2/256]*2; row["delta_bounds"]["broad_minus_narrow"] = [0,0]
    value = stats.summarize(data)
    assert value["anchor_practical_gain"] and not value["narrow_attribution"] and not value["gate_pass"]

def test_pair_not_root_is_bootstrap_unit_and_order_does_not_matter():
    data = rows()
    for row in data:
        row["delta_bounds"] = {k:([1/256]*2 if row["seat"] == 0 else [3/256]*2) for k in stats.NAMES}
    expected = stats.summarize(data)
    assert all(r["ci95"] == [2/256]*2 for r in expected["contrasts"].values())
    assert expected == stats.summarize(list(reversed(data)))

def test_unknowns_are_conservative_and_teacher_cannot_rescue():
    data = rows()
    for row in data:
        row["delta_bounds"]["broad_minus_anchor"] = [-1/256,4/256]
        row["delta_bounds"]["teacher_minus_anchor"] = [1,1]
    value = stats.summarize(data)
    assert not value["gate_pass"] and value["unknown_outcomes_imputed"] is False
    assert value["contrasts"]["broad_minus_anchor"] == {"mean":None,"mean_bounds":[-1/256,4/256],"ci95":[-1/256,4/256]}

@pytest.mark.parametrize("mutation",["duplicate","broken_pair","cross_cell","wrong_scale","bad_bounds"])
def test_reject_invalid_panel(mutation):
    data = rows()
    if mutation == "duplicate": data[-1]["game_index"] = 0
    if mutation == "broken_pair": data[-1]["pair"] = 0
    if mutation == "cross_cell": data[-1]["phase"] = 0
    if mutation == "wrong_scale": data[-1]["delta_bounds"]["broad_minus_anchor"] = [.001,.002]
    if mutation == "bad_bounds": data[-1]["delta_bounds"]["broad_minus_anchor"] = [1,-1]
    with pytest.raises(ValueError): stats.summarize(data)

def test_resource_decision_is_timing_only_and_includes_storage_and_elapsed():
    kw = dict(collection={b:10 for b in exp.BLOCKS},replay={b:10 for b in exp.BLOCKS},
              counts={b:100 for b in exp.BLOCKS},main_counts={b:10000 for b in exp.BLOCKS},
              main_replays={b:1000 for b in exp.BLOCKS},structural=1,audit_structural=1,
              size=10000,free=10*1024**3,elapsed=100)
    v = exp.cost_projection(**kw)
    assert v["reasonable"] and v["remaining_seconds"] > 300
    assert not exp.cost_projection(**(kw|{"elapsed":8*3600}))["reasonable"]
    assert not exp.cost_projection(**(kw|{"free":c.CONFIG["disk_reserve_bytes"]}))["reasonable"]
    with pytest.raises(ValueError): exp.cost_projection(**(kw|{"collection":{"selection":-1,"assessment":1}}))

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
