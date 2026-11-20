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


def test_current_boundary_accepts_96_hours_and_enforces_168(tmp_path,monkeypatch):
    sealed_launch(tmp_path,96*3600)
    c.budget_boundary(tmp_path)
    launch=c.old.read_json(tmp_path/"launch.json")
    monkeypatch.setattr(c.time,"time",lambda:launch["epoch_seconds"]+168*3600+1)
    with pytest.raises(c.BudgetExceeded,match="Seven-day"):c.budget_boundary(tmp_path)


def test_active_carry_counts_prior_runtime(tmp_path,monkeypatch):
    sealed_launch(tmp_path,2*3600,95*3600)
    assert 97*3600 <= c.active_seconds(tmp_path) < 97*3600+10
    c.budget_boundary(tmp_path)
    launch=c.old.read_json(tmp_path/"launch.json")
    monkeypatch.setattr(c.time,"time",lambda:launch["epoch_seconds"]+74*3600)
    with pytest.raises(c.BudgetExceeded):c.budget_boundary(tmp_path)


@pytest.mark.parametrize("mode",["changed","missing","binding"])
def test_timing_records_cannot_silently_reset_budget(tmp_path,mode):
    sealed_launch(tmp_path,96*3600)
    if mode == "missing":(tmp_path/"launch_seal.json").unlink()
    elif mode == "changed":
        (tmp_path/"launch.json").write_text('{"epoch_seconds":0}')
    else:c.old.save_new(tmp_path/"freeze.json",{"binding":"different"})
    with pytest.raises((ValueError,AssertionError)):c.budget_boundary(tmp_path)


def test_cooperative_abort_and_disk_reserve_remain_enforced(tmp_path,monkeypatch):
    sealed_launch(tmp_path,96*3600)
    p.abort(RuntimeError("fixture abort"),tmp_path)
    with pytest.raises(c.BudgetExceeded,match="Cooperative"):c.budget_boundary(tmp_path)
    clean=tmp_path/"clean";clean.mkdir()
    from types import SimpleNamespace
    monkeypatch.setattr(c.shutil,"disk_usage",lambda _:SimpleNamespace(free=0))
    with pytest.raises(c.BudgetExceeded,match="reserve"):c.budget_boundary(clean)


def test_all_new_source_and_world_domains_are_disjoint():
    schedules={stage:p.source.schedule(stage) for stage in p.SOURCE_STAGES}
    proof=p.validate_namespaces(schedules,{1,2255000000,2305000000,2310000000})
    assert proof["source_games"] == 34614
    assert len(proof["intervals"]) == 8
    with pytest.raises(ValueError,match="Historical source overlap"):
        p.validate_namespaces(schedules,{c.CONFIG["label_world_seed_base"]+1024})


def test_label_forecast_prices_full_new_corpus_and_downstream_reserve():
    args=dict(collection_seconds=2000.,replay_seconds=2000.,probe_count=18432,
              size=20*1024**2,free=100*1024**3,elapsed=3600.,audit_structural_seconds=18.)
    ordinary=p.label_cost_projection(**args)
    assert ordinary["remaining_label_continuations_upper_bound"] == 10782*8*128
    assert ordinary["remaining_label_replay_continuations_upper_bound"] == 10782*8*8+39*8*128
    assert ordinary["reasonable"] and ordinary["future_stages_reserve_seconds"] == 86400
    assert ordinary["future_storage_reserve_bytes"] == 32*1024**3
    assert not p.label_cost_projection(**{**args,"free":35*1024**3})["reasonable"]
    assert not p.label_cost_projection(**{**args,"elapsed":167*3600})["reasonable"]
    assert not p.label_cost_projection(**{**args,"free":c.CONFIG["disk_reserve_bytes"]})["reasonable"]
    with pytest.raises(ValueError):p.label_cost_projection(**{**args,"collection_seconds":0})


def test_source_probe_is_fixed_and_balanced_without_outcome_fields():
    from collections import Counter
    schedule=p.source.schedule("train")
    probe=p.source_probe_specs(schedule)
    assert len(probe)==len({s["index"] for s in probe})==54
    assert Counter((s["fit"],s["seat"],s["phase"]) for s in probe)=={
        (fit,seat,phase):3 for fit in c.FITS for seat in (0,1) for phase in range(3)}
    assert probe==p.source_probe_specs(list(reversed(schedule)))


def test_source_forecast_prices_all_games_and_leaves_downstream_reserves():
    args=dict(collection=100.,replay=150.,count=54,size=27*1024**2,
              free=100*1024**3,elapsed=300.)
    value=p.source_cost_projection(**args)
    assert value["remaining_games"]==32346 and value["reasonable"]
    assert value["future_stages_reserve_seconds"]==120*3600
    assert value["projected_bytes"]>32*1024**3
    assert not p.source_cost_projection(**{**args,"elapsed":49*3600})["reasonable"]
    assert not p.source_cost_projection(**{**args,"free":35*1024**3})["reasonable"]
    with pytest.raises(ValueError):p.source_cost_projection(**{**args,"replay":0})


def test_combined_source_locks_preserve_sparse_probe_indices(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setitem(p.CONFIG,"source_games",{**p.CONFIG["source_games"],"train":4})
    monkeypatch.setitem(p.CONFIG,"source_train_probe_games",2)
    for part,indices in (("probe",[1,3]),("remainder",[0,2])):
        refs=[]
        for i in indices:
            path=tmp_path/f"game_{i}.json";c.old.save_new(path,{"index":i})
            refs.append({**c.artifact(path),"index":i})
        launch=tmp_path/f"source_train_{part}_collection_launch.json"
        c.old.save_new(launch,{"manifest":"fixture","stage":"source_train_"+part,
                              "kind":"collection","expected_indices":indices,"dependencies":[]})
        c.old.save_new(tmp_path/f"source_train_{part}_collection_lock.json",
                       {"manifest":"fixture","launch":c.artifact(launch),"files":refs,"elapsed_seconds":2.})
    refs=p.combine_training_source_locks("fixture","collection")
    assert [r["index"] for r in refs]==list(range(4))
    lock=p.stage_lock("source_train","collection","fixture")
    assert lock["elapsed_seconds"]==4. and lock["files"]==refs
    (tmp_path/"game_3.json").write_text("changed")
    with pytest.raises(ValueError):p.stage_lock("source_train","collection","fixture")


def evaluation_cost(**changes):
    args=dict(collection={"selection":1300.,"assessment":700.},
              replay={"selection":1000.,"assessment":500.},
              counts={"selection":10000,"assessment":5000},
              main_counts={"selection":1105920,"assessment":552960},
              main_replays={"selection":100000,"assessment":50000},
              structural=100.,audit_structural=200.,size=20*1024**2,
              free=100*1024**3,elapsed=90*3600.)
    return p.cost_projection(**{**args,**changes})


def test_fresh_forecast_prices_720_roots_and_all_prior_stages():
    assert evaluation_cost()["reasonable"]
    assert not evaluation_cost(elapsed=168*3600)["reasonable"]
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
    for name in ("build_training_sources","build_labels","build_learning","build_sources","finish"):
        monkeypatch.setattr(p,name,lambda *args,name=name:calls.append(name))
    monkeypatch.setattr(p,"build_stage",lambda stage,*args:calls.append(stage))
    monkeypatch.setattr(p,"operational_decision",lambda *args:True)
    return calls


def test_fixed_pipeline_seals_launch_before_work_and_stops_after_result(tmp_path,monkeypatch):
    calls=fake_run(tmp_path,monkeypatch);p.run()
    assert calls==["build_training_sources","build_labels","build_learning","build_sources","probe","main","finish"]
    assert c.active_seconds(tmp_path)<10
    with pytest.raises(ValueError,match="One-shot"):p.run()


def test_failure_preserves_evidence_without_retry(tmp_path,monkeypatch):
    calls=fake_run(tmp_path,monkeypatch)
    def broken(*args):
        c.old.save_new(tmp_path/"partial.json",{"preserved":True})
        raise RuntimeError("fixture failure")
    monkeypatch.setattr(p,"build_learning",broken)
    with pytest.raises(RuntimeError,match="fixture failure"):p.run()
    assert calls==["build_training_sources","build_labels"]
    assert c.old.read_json(tmp_path/"partial.json")["preserved"]
    assert c.old.read_json(tmp_path/"abort.json")["error"]=="fixture failure"
    assert (tmp_path/"failure.json").exists()


def test_worker_error_signals_other_workers_and_retains_provenance(tmp_path,monkeypatch):
    def bad(*args):raise ValueError("fixture learning input")
    monkeypatch.setattr(p.learning,"fit",bad)
    with pytest.raises(ValueError):p.run_job("learning","fits",(161001,"large"),"fixture",tmp_path)
    assert c.old.read_json(tmp_path/"abort.json")["error"]=="fixture learning input"


def test_production_markers_block_second_launch_without_overwriting(tmp_path):
    c.old.save_new(tmp_path/"launch.json",{"retained":True})
    before=(tmp_path/"launch.json").read_bytes()
    with pytest.raises(ValueError,match="preserve"):p.assert_prelaunch_clean(tmp_path)
    assert (tmp_path/"launch.json").read_bytes()==before
