"""Isolated resource and evidence-preservation checks for the operational recovery."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from tools.control_variate_128 import common as c, experiment as parent
from . import recovery as r


def _disk(monkeypatch,free=10*1024**3):
    monkeypatch.setattr(r.shutil,"disk_usage",lambda path:SimpleNamespace(free=free))


def _launch(base,epoch=1000.,previous=120.):
    c.old.save_new(base/"launch.json",{"epoch_seconds":epoch,"previous_elapsed_seconds":previous})


def test_active_budget_counts_prior_work_but_excludes_pause(tmp_path,monkeypatch):
    monkeypatch.setattr(r,"OUTPUT",tmp_path)
    _disk(monkeypatch);_launch(tmp_path,epoch=1_000_000.,previous=600.)
    # A long wall-clock gap before this launch is intentionally absent: only
    # 600 previous active seconds plus this launch's active seconds are charged.
    monkeypatch.setattr(r.time,"time",lambda:1_007_001.)
    r.budget_boundary()
    assert r.MAXIMUM_RUNTIME_SECONDS == 10800
    monkeypatch.setattr(r.time,"time",lambda:1_010_200.)
    r.budget_boundary()
    monkeypatch.setattr(r.time,"time",lambda:1_010_200.001)
    with pytest.raises(c.BudgetExceeded):r.budget_boundary()


def test_budget_production_requires_launch_but_fixture_can_omit_it(tmp_path,monkeypatch):
    production=tmp_path/"production";fixture=tmp_path/"fixture"
    production.mkdir();fixture.mkdir()
    monkeypatch.setattr(r,"OUTPUT",production);_disk(monkeypatch)
    with pytest.raises(ValueError,match="launch"):r.budget_boundary()
    r.budget_boundary(fixture)


def test_cooperative_abort_precedes_remaining_budget_and_preserves_reason(tmp_path,monkeypatch):
    monkeypatch.setattr(r,"OUTPUT",tmp_path);_disk(monkeypatch)
    _launch(tmp_path);monkeypatch.setattr(r.time,"time",lambda:1001.)
    c.old.save_new(tmp_path/"abort.json",{"error":"fixture stop"})
    ref=c.artifact(tmp_path/"abort.json")
    with pytest.raises(c.BudgetExceeded):r.budget_boundary()
    c.verify_hash(ref)


def test_five_gib_disk_reserve_is_unchanged(tmp_path,monkeypatch):
    monkeypatch.setattr(r,"OUTPUT",tmp_path);_launch(tmp_path)
    monkeypatch.setattr(r.time,"time",lambda:1001.)
    _disk(monkeypatch,5*1024**3);r.budget_boundary()
    _disk(monkeypatch,5*1024**3-1)
    with pytest.raises(c.BudgetExceeded,match="disk"):r.budget_boundary()


def test_configuration_installs_recovery_hooks_without_scientific_changes(tmp_path,monkeypatch):
    before=copy.deepcopy(c.CONFIG);original_output=c.OUTPUT
    # Register restoration before configure installs process-local hooks.
    for module,key in ((parent,"OUTPUT"),(parent,"validate_freeze"),(parent,"worker_initialize"),
                       (c,"budget_boundary")):
        monkeypatch.setattr(module,key,getattr(module,key))
    r.configure(tmp_path)
    assert parent.OUTPUT == tmp_path
    assert parent.validate_freeze is r.validate_freeze
    assert parent.worker_initialize is r.worker_initialize
    assert c.budget_boundary is r.budget_boundary
    assert c.CONFIG == before and c.OUTPUT == original_output
    assert c.CONFIG["maximum_runtime_hours"] == 2.
    assert r.MAXIMUM_RUNTIME_SECONDS == 10800


def test_inherited_timing_transform_changes_only_binding_and_provenance():
    ref={"path":"outputs/old/timing.json","sha256":"a"*64,"index":7}
    record={"manifest":"old-binding","index":7,"complete":True,
        "timing":{"full_only":{"cpu":1.25,"wall":1.5,"cells":8}},
        "cells":[{"candidate":0,"world":3,"samples":[{"horizon":4,"cpu":.04,"z":.75}]}],
        "statistics":{"index":7,"timing":{"cpu":1.25}},"source":{"path":"old-root","sha256":"b"*64}}
    before=copy.deepcopy(record)
    transformed=r.transformed_record(record,"new-binding",ref)
    assert transformed == {**record,"manifest":"new-binding","reused_from":ref}
    assert record == before


def test_inherited_audit_transform_retains_all_checks_and_original_evidence():
    old_ref={"path":"old_audit","sha256":"a"*64}
    new_result={"path":"new_record","sha256":"b"*64,"index":4}
    audit={"manifest":"old-binding","index":4,"result":{"path":"old_record","sha256":"c"*64},
           "passed":True,"all_full_outcomes_equal_saved":True,
           "matched_timing_arms_verified":True,"replayed_candidates":64,"new_terminal_simulations":0}
    before=copy.deepcopy(audit)
    result=r.transformed_audit(audit,"new-binding",new_result,old_ref)
    assert result == {**audit,"manifest":"new-binding","result":new_result,"reused_from":old_ref}
    assert audit == before


def _timing_evidence(base):
    records=[];audits=[]
    for index in range(18):
        path=base/f"record_{index:04d}.json"
        c.old.save_new(path,{"manifest":"parent","index":index,"complete":True,
            "timing":{"full_only":{"cpu":index+.5,"wall":index+.75,"cells":8}},
            "cells":[{"world":3,"candidate":0,"z":.4}]})
        ref=c.artifact(path,index=index);records.append(ref)
        path=base/f"audit_{index:04d}.json"
        c.old.save_new(path,{"manifest":"parent","index":index,"result":ref,"passed":True,
                            **{key:True for key in r.TIMING_FLAGS}})
        audits.append(c.artifact(path,index=index))
    result={}
    for name,files in (("timing_lock",records),("timing_audits_lock",audits)):
        path=base/(name+".json")
        c.old.save_new(path,{"manifest":"parent","files":files,"elapsed_seconds":12.5})
        result[name]=c.artifact(path)
    return result,records+audits+list(result.values())


def _rewrite_fixture(path,payload,base):
    path=Path(path).resolve()
    assert path.is_relative_to(base.resolve())
    path.write_text(json.dumps(payload),encoding="utf-8")


@pytest.mark.parametrize("damage",[None,"timing_payload","audit_result","audit_flag"])
def test_reuse_revalidates_exact_scientific_payload_despite_rehashed_links(tmp_path,damage):
    evidence,originals=_timing_evidence(tmp_path/"prior")
    base=tmp_path/"recovery"
    r.import_timing(base,"recovery-binding",evidence)
    refs=r.validate_reuse(base,"recovery-binding",evidence)
    assert len(refs) == 18
    if damage is not None:
        lock_name="timing_lock" if damage == "timing_payload" else "timing_audits_lock"
        lock_path=base/(lock_name+".json");lock=c.old.read_json(lock_path)
        path=c.verify_hash(lock["files"][0]);record=c.old.read_json(path)
        if damage == "timing_payload":record["timing"]["full_only"]["cpu"]+=1
        elif damage == "audit_result":record["result"]=refs[1]
        else:record["matched_timing_arms_verified"]=False
        _rewrite_fixture(path,record,base)
        lock["files"][0]=c.artifact(path,index=0)
        _rewrite_fixture(lock_path,lock,base)
        receipt_path=base/"timing_reuse.json";receipt=c.old.read_json(receipt_path)
        receipt["records" if damage == "timing_payload" else "audits"]=c.artifact(lock_path)
        _rewrite_fixture(receipt_path,receipt,base)
        with pytest.raises(ValueError,match="Reused timing"):
            r.validate_reuse(base,"recovery-binding",evidence)
    for ref in originals:c.verify_hash(ref)


@pytest.mark.parametrize("condition",["pass","cap","disk"])
def test_forecast_reuses_remaining_work_excludes_pause_and_refreshes_disk(tmp_path,monkeypatch,condition):
    from tools.control_variate_128 import engine
    monkeypatch.setattr(r,"OUTPUT",tmp_path)
    monkeypatch.setattr(parent,"event",lambda *args,**kwargs:None)
    def no_simulation(*args,**kwargs):raise AssertionError("Recovery must reuse measured costs")
    monkeypatch.setattr(engine,"timing",no_simulation)
    monkeypatch.setattr(engine,"execution_forecast",no_simulation)
    previous={"manifest":"parent","estimated_total_seconds":8891.8218,
              "elapsed_seconds":163.55134,"estimated_remaining_bytes":1000,
              "outcome_blind":True,"within_runtime_ceiling":False,"disk_reserve_pass":True}
    c.old.save_new(tmp_path/"prior_forecast.json",previous)
    ref=c.artifact(tmp_path/"prior_forecast.json")
    c.old.save_new(tmp_path/"manifest.json",{"held_parent":{"forecast":ref}})
    c.old.save_new(tmp_path/"freeze.json",{"manifest":c.artifact(tmp_path/"manifest.json")})
    c.old.save_new(tmp_path/"timing_lock.json",{"files":[{"path":"inherited","sha256":"fixed"}]})
    _launch(tmp_path,epoch=1_000_000.,previous=163.568)
    additional=3000. if condition == "cap" else 12.
    monkeypatch.setattr(r.time,"time",lambda:1_000_000.+additional)
    free=5*1024**3+1000-(1 if condition == "disk" else 0)
    _disk(monkeypatch,free)
    if condition == "pass":r.forecast("recovery")
    else:
        with pytest.raises(c.BudgetExceeded):r.forecast("recovery")
    result=c.old.read_json(tmp_path/"forecast.json")
    assert result["estimated_total_seconds"] == pytest.approx(8891.8218-163.55134+163.568+additional,abs=1e-10)
    assert result["elapsed_seconds"] == pytest.approx(163.568+additional,abs=1e-10)
    assert result["maximum_active_seconds"] == 10800
    assert result["cost_estimates_unchanged"] and result["idle_pause_excluded"]
    assert result["within_runtime_ceiling"] is (condition != "cap")
    assert result["disk_reserve_pass"] is (condition != "disk")
    assert result["free_disk_bytes"] == free
    c.verify_hash(ref)
