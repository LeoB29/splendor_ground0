"""Outcome-blind operational gates, frozen evidence and delivered audit coverage."""
from contextlib import contextmanager
import copy
from types import SimpleNamespace

import pytest

from tools.proposal_breadth_arena_128 import common as c, experiment as p


def costs():
    return {"game_worker_wall":60000.,"audit_worker_wall":12000.,
            "stress_extra_wall_by_role":{"candidate":6000.,"reference":3000.},
            "stress_roots_by_role":{"candidate":36,"reference":30},
            "size_bytes":100*1024**2,"free_bytes":100*1024**3,"elapsed_seconds":4*3600.}


def test_forecast_separates_ordinary_work_and_role_specific_stress_maxima():
    value = p.cost_projection(**costs())
    games = 60000*(402/12)/8
    ordinary = (12000-6000-3000)*(402/12)/8
    stress = (90*6000/36+81*3000/30)/8
    assert value["main_game_seconds_before_allowance"] == games
    assert value["main_ordinary_audit_seconds_before_allowance"] == ordinary
    assert value["main_stress_seconds_before_allowance"] == stress
    assert value["maximum_stress_roots_by_role"] == {"candidate":90,"reference":81}
    assert sum(value["maximum_stress_roots_by_role"].values()) == 171
    assert value["forecast_seconds"] == 1.25*(games+ordinary+stress)
    assert value["remaining_forecast_seconds"] == value["forecast_seconds"]+600
    assert value["projected_main_bytes"] == int(1.5*(402/12)*100*1024**2+512*1024**2)
    assert value["reasonable"]
    # The same total measured extra replay cost has different projected work
    # when its role allocation differs; candidate and reference are not pooled.
    changed = costs()
    changed["stress_extra_wall_by_role"] = {"candidate":3000.,"reference":6000.}
    swapped = p.cost_projection(**changed)
    assert swapped["main_ordinary_audit_seconds_before_allowance"] == ordinary
    assert swapped["main_stress_seconds_before_allowance"] == (90*3000/36+81*6000/30)/8
    assert swapped["main_stress_seconds_before_allowance"] != stress


def test_each_forecast_total_runtime_and_storage_limit_independently_holds():
    args = costs()
    initial = p.cost_projection(**args)
    # Raise main work just past120h while total execution would remain below144h.
    wanted = 120*3600+1
    scale = 1.25*(402/12)/8
    args["game_worker_wall"] += (wanted-initial["forecast_seconds"])/scale
    value = p.cost_projection(**args)
    assert value["forecast_seconds"] == pytest.approx(wanted)
    assert value["elapsed_seconds"]+value["remaining_forecast_seconds"] < 144*3600
    assert not value["reasonable"]
    # A forecast below120h is insufficient when earlier stages consumed the cap.
    args = costs()
    args["elapsed_seconds"] = 144*3600-initial["remaining_forecast_seconds"]+1
    assert p.cost_projection(**args)["forecast_hours"] < 120
    assert not p.cost_projection(**args)["reasonable"]
    args = costs()
    args["free_bytes"] = initial["projected_main_bytes"]+5*1024**3
    assert p.cost_projection(**args)["reasonable"]
    args["free_bytes"] -= 1
    assert not p.cost_projection(**args)["reasonable"]


@pytest.mark.parametrize("damage",["negative","nan","infinity","missing_role","zero_stress",
                                     "fractional_count","boolean_count","too_many_candidate","too_many_reference",
                                     "missing_stress_time","extra_exceeds_audit","zero_games"])
def test_malformed_or_unmeasured_probe_evidence_is_rejected(damage):
    args = costs()
    if damage == "negative":
        args["elapsed_seconds"] = -1.
    elif damage == "nan":
        args["audit_worker_wall"] = float("nan")
    elif damage == "infinity":
        args["game_worker_wall"] = float("inf")
    elif damage == "missing_role":
        args["stress_roots_by_role"].pop("reference")
    elif damage == "zero_stress":
        args["stress_roots_by_role"]["candidate"] = 0
    elif damage == "fractional_count":
        args["stress_roots_by_role"]["candidate"] = 36.5
    elif damage == "boolean_count":
        args["stress_roots_by_role"]["candidate"] = True
    elif damage == "too_many_candidate":
        args["stress_roots_by_role"]["candidate"] = 91
    elif damage == "too_many_reference":
        args["stress_roots_by_role"]["reference"] = 82
    elif damage == "missing_stress_time":
        args["stress_extra_wall_by_role"]["reference"] = 0.
    elif damage == "extra_exceeds_audit":
        args["audit_worker_wall"] = 8999.
    else:
        args["game_worker_wall"] = 0.
    with pytest.raises(ValueError):
        p.cost_projection(**args)


def audit_fixture():
    inventory,selections,roots = [],[],[]
    for index,turn,role,support,stress in ((0,2,"candidate",8,True),(0,3,"reference",4,False),
                                          (1,2,"reference",1,True),(1,3,"candidate",5,False)):
        rid = f"{index:04d}:{turn:03d}"
        ordinary = [1,8,24,40,60,75,100,127]
        replay = list(range(128)) if stress else list(ordinary)
        inventory.append({"index":index,"turn":turn,"root_id":rid,"arm":role,"fit":161001,
                          "support":support,"sha256":f"root-{rid}"})
        selections.append({"root_id":rid,"stress":stress,"ordinary_world_indices":ordinary,
                           "replay_world_indices":replay})
        roots.append({"index":index,"turn":turn,"planner":role,"fit":161001,"support":support,
                      "sha256":f"root-{rid}","worlds":128,"full_inner_replay":False,"full_fixture_replay":False,
                      "ordinary_world_indices":ordinary,"replay_world_indices":replay,"stress":stress,
                      "continuations":128*support,"replayed_branches":len(replay)*support})
    receipts = [{"sample_sha256":"sample","full_inner_replay":False,"roots":roots[:2]},
                {"sample_sha256":"sample","full_inner_replay":False,"roots":roots[2:]}]
    coverage = {"games":2,"roots":4,"recorded_worlds":512,"recorded_continuations":128*18,
                "replayed_worlds":2*128+2*8,"replayed_continuations":128*9+8*9,"stress_roots":2}
    sample = {"inventory":inventory,"selections":selections,"coverage":coverage}
    return receipts,sample,"sample"


def test_delivered_coverage_reconciles_eight_candidate_and_short_reference_roots():
    receipts,sample,digest = audit_fixture()
    assert p.checked_audit_coverage(receipts,sample,digest) == sample["coverage"]


@pytest.mark.parametrize("damage",["missing_root","duplicate_root","missing_selection","duplicate_selection",
                                     "sha","role","fit","oversized_reference","sample_link","ordinary",
                                     "replay","stress","continuations","replayed_work","totals","disclosure"])
def test_missing_or_corrupt_audited_roots_cannot_close_stage(damage):
    receipts,sample,digest = copy.deepcopy(audit_fixture())
    root = receipts[0]["roots"][0]
    if damage == "missing_root":
        receipts[0]["roots"].pop()
    elif damage == "duplicate_root":
        receipts[0]["roots"][1] = copy.deepcopy(root)
    elif damage == "missing_selection":
        sample["selections"].pop()
    elif damage == "duplicate_selection":
        sample["selections"][1] = copy.deepcopy(sample["selections"][0])
    elif damage == "sha":
        root["sha256"] = "changed"
    elif damage == "role":
        root["planner"] = "reference"
    elif damage == "fit":
        root["fit"] = 161002
    elif damage == "oversized_reference":
        receipts[0]["roots"][1]["support"] = 5
        sample["inventory"][1]["support"] = 5
    elif damage == "sample_link":
        receipts[0]["sample_sha256"] = "changed"
    elif damage == "ordinary":
        root["ordinary_world_indices"] = list(range(8))
    elif damage == "replay":
        root["replay_world_indices"] = root["replay_world_indices"][:-1]
    elif damage == "stress":
        root["stress"] = False
    elif damage == "continuations":
        root["continuations"] -= 1
    elif damage == "replayed_work":
        root["replayed_branches"] -= 1
    elif damage == "totals":
        sample["coverage"]["recorded_worlds"] -= 1
    else:
        root["full_inner_replay"] = True
    with pytest.raises(ValueError):
        p.checked_audit_coverage(receipts,sample,digest)


def frozen_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"OUTPUT",tmp_path/"study")
    monkeypatch.setattr(p,"identity",lambda:{"synthetic_identity":1})
    provenance = tmp_path/"parent_receipt.json"
    p.old.save_new(provenance,{"synthetic_parent":True})
    evidence = {"binding":"synthetic-parent","files":[p.artifact(provenance)]}
    monkeypatch.setattr(p,"parent_evidence",lambda:evidence)
    check_dir = tmp_path/"checks"
    p.old.save_new(check_dir/"stdout.json",{"synthetic_pass":True})
    p.old.save_new(check_dir/"stderr.json",{})
    path = check_dir/"checks.json"
    p.old.save_new(path,{"identity":p.identity(),"targets":p.CHECK_TARGETS,"exit_code":0,
                        "logs":[p.artifact(check_dir/name) for name in ("stdout.json","stderr.json")]})
    return path,provenance


def test_prepare_binds_checks_schedules_and_parent_provenance_once(tmp_path,monkeypatch):
    checks,_ = frozen_fixture(tmp_path,monkeypatch)
    p.prepare(checks)
    binding = p.validate_freeze()
    manifest = p.old.read_json(p.OUTPUT/"manifest.json")
    assert p.old.digest(manifest) == binding
    assert p.verified(manifest["schedule"]) == {stage:p.arena.specs(stage) for stage in ("probe","main")}
    before = p.old.file_hash(p.OUTPUT/"manifest.json")
    with pytest.raises(ValueError,match="Already prepared"):
        p.prepare(checks)
    assert p.old.file_hash(p.OUTPUT/"manifest.json") == before


@pytest.mark.parametrize("damage",["code_identity","schedule","check_log","parent","authorization"])
def test_freeze_rejects_changed_dependencies_without_repair(tmp_path,monkeypatch,damage):
    checks,parent = frozen_fixture(tmp_path,monkeypatch)
    p.prepare(checks)
    if damage == "code_identity":
        monkeypatch.setattr(p,"identity",lambda:{"synthetic_identity":2})
    elif damage == "schedule":
        path = p.OUTPUT/"schedule.json"
        value = p.old.read_json(path);value["main"][0]["environment_seed"] += 1
        p.old.atomic_write(path,value)
    elif damage == "check_log":
        p.old.atomic_write(checks.parent/"stdout.json",{"changed":True})
    elif damage == "parent":
        p.old.atomic_write(parent,{"changed":True})
    else:
        path = p.OUTPUT/"freeze.json"
        value = p.old.read_json(path);value["owner_authorized"] = False
        p.old.atomic_write(path,value)
    with pytest.raises(ValueError):
        p.validate_freeze()


def test_failed_preflight_does_not_create_a_manifest(tmp_path,monkeypatch):
    checks,_ = frozen_fixture(tmp_path,monkeypatch)
    value = p.old.read_json(checks);value["exit_code"] = 1
    p.old.atomic_write(checks,value)
    with pytest.raises(ValueError,match="passing checks"):
        p.prepare(checks)
    assert not (p.OUTPUT/"manifest.json").exists()


def fake_run(tmp_path,monkeypatch,gate=True):
    monkeypatch.setattr(p,"OUTPUT",tmp_path)
    monkeypatch.setattr(p,"REPORTER",None)
    monkeypatch.setattr(p.windows,"assert_windowless_worker",lambda *args:{})
    monkeypatch.setattr(p,"validate_freeze",lambda:"fixture")
    @contextmanager
    def awake():
        yield {"synthetic_fixture":True}
    monkeypatch.setattr(p.parent,"inhibit_system_sleep",awake)
    calls = []
    monkeypatch.setattr(p.models,"prepare",lambda *args:calls.append("models"))
    monkeypatch.setattr(p,"build_stage",lambda stage,*args:calls.append(stage))
    monkeypatch.setattr(p,"operational_decision",lambda *args:gate)
    monkeypatch.setattr(p,"finish",lambda *args:calls.append("finish"))
    return calls


@pytest.mark.parametrize("gate,expected",[(True,["models","probe","main","finish"]),
                                         (False,["models","probe"])])
def test_autonomous_pipeline_stops_at_failed_operational_gate(tmp_path,monkeypatch,gate,expected):
    calls = fake_run(tmp_path,monkeypatch,gate)
    p.run()
    assert calls == expected and not (tmp_path/"failure.json").exists()
    with pytest.raises(ValueError,match="One-shot"):
        p.run()


def test_resource_hold_retains_partial_work_without_retry_or_completion(tmp_path,monkeypatch):
    calls = fake_run(tmp_path,monkeypatch)
    def capped(stage,*args):
        p.old.save_new(tmp_path/"partial.json",{"retained":True})
        raise c.BudgetExceeded("fixture six-day ceiling")
    monkeypatch.setattr(p,"build_stage",capped)
    p.run()
    assert calls == ["models"] and (tmp_path/"partial.json").exists()
    assert "six-day ceiling" in p.old.read_json(tmp_path/"operational_hold.json")["reason"]
    assert not (tmp_path/"failure.json").exists() and not (tmp_path/"final_status.json").exists()


def test_actual_budget_boundary_rejects_elapsed_cap_and_disk_shortage(tmp_path,monkeypatch):
    c.old.save_new(tmp_path/"launch.json",{"epoch_seconds":1.})
    monkeypatch.setattr(c.time,"time",lambda:144*3600+2.)
    with pytest.raises(c.BudgetExceeded,match="Six-day"):
        c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time,"time",lambda:2.)
    monkeypatch.setattr(c.shutil,"disk_usage",lambda *args:SimpleNamespace(free=5*1024**3-1))
    with pytest.raises(c.BudgetExceeded,match="disk reserve"):
        c.budget_boundary(tmp_path)


def test_status_lock_never_weakens_strict_scientific_artifact_writes(tmp_path,monkeypatch):
    from tools.proposal_ranking_128_recovery_01 import reporting
    reporter = p.StatusReporter(tmp_path,{"manifest":"fixture"})
    monkeypatch.setattr(p,"REPORTER",reporter)
    def locked(*args,**kwargs):
        raise PermissionError("synthetic Windows replace lock")
    monkeypatch.setattr(reporting.os,"replace",locked)
    monkeypatch.setattr(reporting.time,"sleep",lambda *args:None)
    assert p.event("probe_progress",completed=1)["completed"] == 1
    assert any(e["operation"] == "progress_replace" for e in reporter.last_errors)
    assert list(tmp_path.glob("status_events/*.json"))
    with pytest.raises(PermissionError,match="replace lock"):
        p.old.save_new(tmp_path/"scientific.json",{"must_be_saved":True})
    assert not (tmp_path/"scientific.json").exists()
