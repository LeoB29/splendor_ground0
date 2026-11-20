"""Cross-module output contracts and outcome-blind stage-order regressions."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import common as c, engine, experiment as p, verify


def _outcome(value):
    return {"wdl":value, "reason":"completed", "winner":0 if value > 0 else 1 if value < 0 else None}


def _saved_statistical_root(index, positive):
    """Known independent score law passes through real root-to-score reduction."""
    fit,seat,phase = verify.CELLS[index//40]
    spec = {"index":index, "game_index":index, "pair":index, "fit":fit, "seat":seat, "phase":phase}
    choices = dict(anchor=3,small=4,medium=5,large=6,control=7,teacher=11)
    candidates = {arm:{"indices":[0,1,2,j]} for arm,j in choices.items() if arm != "teacher"}
    candidates["teacher"] = {"indices":[0,1,2,3,8,9,10,11]}
    record = {"stage":"main", "spec":spec, "privacy":{"own":0,"opponent":0}, "snapshot":{},
              "proposals":dict.fromkeys(c.MODEL_ARMS,{}), "candidates":candidates,
              "union":{"keys":[str(i) for i in range(12)]}, "proposed_indices":choices,
              "teacher_added_indices":{"ranked":[8,9],"random":[10,11]}}
    seeds = engine.world_seeds("main",index,"selection")
    outcomes = [_outcome(v) for v in [-1,-1,-1,0,1,1,1,1,-1,-1,-1,1]]
    record["selection"] = {"world_seeds":seeds,"candidate_indices":list(range(12)),
        "worlds":[{"seed":seed,"outcomes":outcomes} for seed in seeds]}
    selected = engine.choose(record["selection"]["worlds"],candidates,12)
    assert selected["choices"] == choices
    sums = [0,2,3,4 if positive else -2,6,8]
    worlds = []
    seeds = engine.world_seeds("main",index,"assessment")
    for i,seed in enumerate(seeds):
        values = [(1 if total > 0 else -1) if i < abs(total) else 0 for total in sums]
        worlds.append({"seed":seed,"outcomes":[_outcome(v) for v in values]})
    indices = [3,4,5,6,7,11]
    record["assessment"] = {"world_seeds":seeds,"candidate_indices":indices,"worlds":worlds}
    record["selection_lock"] = {"selection_statistics":selected,"choices":choices,
        "identity_digest":c.old.digest(engine.proposal_identity(record)),
        "selection_sha256":c.old.digest(record["selection"]),
        "proofs_sha256":c.old.digest([]),"assessment_indices":indices}
    record["summary"] = engine.summarize(worlds,indices,choices,choices,12)
    return record


@pytest.mark.parametrize("positive",[True,False])
def test_finalization_uses_actual_six_arm_schema_and_independent_statistics(tmp_path,monkeypatch,positive):
    """Catch late stale arm/gate keys without running any production simulation."""
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setitem(c.CONFIG,"bootstrap_samples",211)
    monkeypatch.setitem(c.CONFIG,"bootstrap_chunk",47)
    monkeypatch.setattr(p,"validate_freeze",lambda **kw:"fixture")
    monkeypatch.setattr(p.learning,"validate_models",lambda *a:None)
    monkeypatch.setattr(p,"event",lambda *a,**kw:None)
    refs,specs,rows = [],[],[]
    for index in range(720):
        record = _saved_statistical_root(index,positive)
        path = tmp_path/"main/roots"/f"state_{index:04d}.json"
        c.old.save_new(path,record)
        refs.append(c.artifact(path,index=index));specs.append(record["spec"])
        rows.append(engine.statistics_row(record))
    summary = {"manifest":"fixture","rows":rows,"coverage":{"states":720},"full_inner_replay":False}
    monkeypatch.setattr(p,"validate_selection",lambda *a:{"main":specs})
    monkeypatch.setattr(p,"stage_lock",lambda stage,kind,binding:{"files":refs if kind=="collection" else []})
    monkeypatch.setattr(p.verify,"summarize",lambda *a,**kw:copy.deepcopy(summary))
    for name in ("learning_complete.json","selection_seal.json","probe_complete.json","main_complete.json"):
        c.old.save_new(tmp_path/name,{"manifest":"fixture","complete":True})
    p.finish("fixture")
    status = c.old.read_json(tmp_path/"final_status.json")
    result,audit = c.verified(status["result"]),c.verified(status["audit"])
    c.verify_hash(status["report"])
    assert status["complete"] and status["all_prescribed_audits_pass"] and audit["passed"]
    assert result["statistics"]["gate_pass"] is positive
    assert result["statistics"]["small_superiority"] is positive
    assert result["statistics"]["coverage_attribution"] is positive
    expected = 4/256 if positive else -2/256
    assert result["statistics"]["contrasts"]["large_minus_anchor"]["mean_score_difference"] == expected
    assert result["statistics"]["contrasts"]["large_minus_small"]["mean_score_difference"] == expected-2/256
    assert result["statistics"]["contrasts"]["large_minus_control"]["ci95"][1] < 0
    assert "control_superiority" not in result["statistics"]
    assert "precision_attribution" not in result["statistics"]
    assert status["automatic_successor"] is False
    before = c.artifact(tmp_path/"result.json")
    with pytest.raises(FileExistsError):p.finish("fixture")
    c.verify_hash(before)


@pytest.mark.parametrize("forecast_hold",[False,True])
def test_label_pipeline_preserves_noncontiguous_ids_and_requires_probe_audit(tmp_path,monkeypatch,forecast_hold):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    specs = [{"index":i} for i in range(10800)]
    probe_ids = list(range(17,10800,600))
    plan = {"specs":specs,"sample":{"probe_indices":probe_ids}}
    for name in ("freeze.json","inherited_source_lock.json","source_models_lock.json",
                 "label_plan.json","label_plan_seal.json","training_sources_complete.json"):
        c.old.save_new(tmp_path/name,{"fixture":True})
    monkeypatch.setattr(p.label_source,"load_plan",lambda *a:plan)
    calls = []
    def run_pool(stage,kind,arguments,binding,upstream):
        ids = [(a[0] if kind=="audits" else a)["index"] for a in arguments]
        calls.append((stage,kind,ids))
        assert all(Path(path).exists() for path in upstream)
        if kind == "audits":assert all(a[0]["index"] == a[1]["index"] for a in arguments)
        path = tmp_path/"jobs"/stage/(kind+".json")
        c.old.save_new(path,{"manifest":binding,"fixture":True})
        ref = c.artifact(path)
        refs = [{**ref,"index":i} for i in ids]
        c.old.save_new(tmp_path/f"{stage}_{kind}_lock.json",{"files":refs})
        return refs
    monkeypatch.setattr(p,"run_pool",run_pool)
    def forecast(binding):
        assert [v[:2] for v in calls] == [("labels_probe","collection"),("labels_probe","audits")]
        if forecast_hold:raise c.BudgetExceeded("forecast fixture")
        c.old.save_new(tmp_path/"label_operational_decision.json",{"reasonable":True})
    monkeypatch.setattr(p,"label_operational_decision",forecast)
    def build_lock(binding,base):
        assert [v[:2] for v in calls] == [("labels_probe","collection"),("labels_probe","audits"),
                                        ("labels_main","collection"),("labels_main","audits")]
        assert calls[0][2] == calls[1][2] == probe_ids
        assert calls[2][2] == calls[3][2] == [i for i in range(10800) if i not in probe_ids]
        refs = []
        for name in ("labels_lock.json","labels_audits_lock.json"):
            c.old.save_new(tmp_path/name,{"manifest":binding,"complete":True})
            refs.append(c.artifact(tmp_path/name))
        return tuple(refs)
    monkeypatch.setattr(p.labels,"build_lock",build_lock)
    if forecast_hold:
        with pytest.raises(c.BudgetExceeded,match="forecast fixture"):p.build_labels("fixture")
        assert len(calls) == 2 and not (tmp_path/"labels_complete.json").exists()
    else:
        p.build_labels("fixture")
        complete = c.old.read_json(tmp_path/"labels_complete.json")
        assert complete["new_rows"] == 10800 and complete["inherited_rows"] == 720
        assert complete["worlds_per_root"] == 128
        c.verify_hash(complete["labels"]);c.verify_hash(complete["audits"])


@pytest.mark.parametrize("fully_audited",[True,False])
def test_label_forecast_counts_durable_blocks_and_uses_actual_audit_schema(tmp_path,monkeypatch,fully_audited):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p,"event",lambda *a,**kw:None)
    monkeypatch.setattr(p.shutil,"disk_usage",lambda *a:SimpleNamespace(free=100*1024**3))
    refs,audits = [],[]
    for i in range(18):
        block = tmp_path/"labels/blocks"/f"{i}.json"
        c.old.save_new(block,{"fixture_payload":"x"*8192})
        root = tmp_path/"labels/roots"/f"{i}.json"
        c.old.save_new(root,{"recorded_continuations":128*8,"block":c.artifact(block),"timing":{"wall":20.,"cpu":20.}})
        refs.append(c.artifact(root,index=i))
        audit = tmp_path/"labels/audits"/f"{i}.json"
        c.old.save_new(audit,{"passed":True,"full_inner_replay":fully_audited,"root":c.artifact(root),
            "replayed_continuations":128*8,"timing":{"wall":19.,"cpu":19.,"replay_wall":18.,"structural_wall":1.}})
        audits.append(c.artifact(audit,index=i))
    monkeypatch.setattr(p,"stage_lock",lambda stage,kind,binding:{"files":refs if kind=="collection" else audits})
    if not fully_audited:
        with pytest.raises(ValueError,match="Fully audited"):p.label_operational_decision("fixture")
        assert not (tmp_path/"label_operational_decision.json").exists()
        return
    p.label_operational_decision("fixture")
    value = c.old.read_json(tmp_path/"label_operational_decision.json")
    assert value["reasonable"] and value["manifest"] == "fixture"
    assert value["collection_seconds_per_continuation"] == 20/(128*8)
    assert value["replay_seconds_per_continuation"] == 18/(128*8)
    assert value["remaining_label_audit_structural_seconds"] == 10782
    root_bytes = sum(c.verify_hash(ref).stat().st_size for ref in refs)
    assert value["projected_bytes"] > c.CONFIG["pre_label_future_storage_bytes"]+100*root_bytes
    assert value["future_storage_reserve_bytes"] == 32*1024**3
    assert value["future_stages_reserve_seconds"] == 24*3600
