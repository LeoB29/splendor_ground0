"""Independent model-free orchestration regressions; no model/search subprocesses."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_concurrency_20261105 import audit, common as c, control, design, worker


def registry(kind="ordinary"):
    return dict(pid=987654, active=[dict(client_id="case_g003", kind=kind,
        started_unix=100., deadline_unix=110., started_perf=200., deadline_perf=210.)])


def test_fixed_design_covers_the_identical_eight_cases_without_slot_renumbering():
    specs = design.cells()
    assert [(row["route"], row["concurrency"]) for row in specs] == [
        ("cpu",1),("gpu",1),("gpu",2),("cpu",2),("cpu",4),("gpu",4),("gpu",8),("cpu",8)]
    for row in specs:
        assert [index for group in row["groups"] for index in group] == list(range(8))
        assert all(len(group) == row["concurrency"] for group in row["groups"])
    assert specs[1]["groups"][-1] == [7]  # C1 still evaluates the last opening in permanent row7.
    assert sum(len(group) for row in specs for group in row["groups"]) == 64


@pytest.mark.parametrize("wall,perf", [(110.,209.), (109.,210.), (110.,210.)])
def test_either_original_clock_can_expire_an_ordinary_decision(wall, perf):
    value = registry()
    with pytest.raises(control.DecisionOuterDeadline) as caught:
        control.check_registry(value, dict(pid=987654), wall, perf)
    assert caught.value.row == dict(value["active"][0], observed_unix=wall, observed_perf=perf)
    assert "observed_unix" not in value["active"][0]


def test_unexpired_owned_registry_is_observation_only():
    control.check_registry(registry(), dict(pid=987654), 109.999, 209.999)


@pytest.mark.parametrize("kind", ["service_warmup", "initialization", "fixed_batch"])
def test_nondecision_expiry_is_never_isolated_as_a_failed_cell(kind):
    with pytest.raises(TimeoutError) as caught:
        control.check_registry(registry(kind), dict(pid=987654), 111., 211.)
    assert not isinstance(caught.value, control.DecisionOuterDeadline)


@pytest.mark.parametrize("mutation", ["foreign_pid", "extended_wall", "extended_perf", "nonfinite"])
def test_registry_identity_and_unextended_deadlines_are_required(mutation):
    value = registry()
    if mutation == "foreign_pid": value["pid"] += 1
    elif mutation == "extended_wall": value["active"][0]["deadline_unix"] += .1
    elif mutation == "extended_perf": value["active"][0]["deadline_perf"] += .1
    else: value["active"][0]["started_perf"] = float("nan")
    with pytest.raises(ValueError):
        control.check_registry(value, dict(pid=987654), 109., 209.)


def fake_stage(tmp_path, monkeypatch, *, mode, exit_code=None, active=None, fallback=False, cleanup_error=False):
    """Exercise real run_stage flow with a fake OS child; never launch a process."""
    from tools.local_player_qualification_128 import clock
    from tools.splendorust_gpu_live_20261105_03 import status_io
    monkeypatch.setattr(c, "OUT", tmp_path)
    monkeypatch.setattr(c, "PLAN", tmp_path/"plan.json")
    cleaned = []
    owned = SimpleNamespace(create_time=lambda:123., children=lambda recursive:[],
        memory_info=lambda:SimpleNamespace(rss=100), pid=987654)
    monkeypatch.setattr(control.psutil, "Process", lambda pid:owned)

    class Child:
        pid = 987654
        returncode = None
        def __init__(self, args, **kwargs):
            output = Path(args[args.index("--output")+1])
            if active is not None:
                (output/"active_decisions.json").write_text("{}", encoding="utf-8")
        def poll(self):
            if exit_code is not None: self.returncode = exit_code
            return self.returncode

    class Job:
        def __init__(self, pid): assert pid == 987654
        def close(self): cleaned.append("job")

    def finish(proc, identity):
        assert identity == dict(pid=987654, created=123.)
        cleaned.append("process")
        if proc.returncode is None: proc.returncode = -9
        if cleanup_error: raise RuntimeError("owned process still alive")

    monkeypatch.setattr(control.subprocess, "Popen", Child)
    monkeypatch.setattr(clock, "_WindowsJob", Job)
    monkeypatch.setattr(control, "finish_owned_process", finish)
    monkeypatch.setattr(control, "require_absent", lambda identity:cleaned.append("absent"))
    monkeypatch.setattr(control, "native_fallback_messages", lambda path:["unsupported native fallback"] if fallback else [])
    monkeypatch.setattr(status_io, "read_status_json", lambda path:deepcopy(active))
    plan = dict(worker_limits=dict(qualification_seconds=1200,cell_seconds=300))
    spec = None if mode == "qualification" else design.cells()[0]
    row = control.run_stage(plan,"frozen-sha",mode,spec,tmp_path/"stage",lambda deep:None)
    assert cleaned[:2] == ["job","process"]
    assert cleaned.count("absent") >= 1
    assert c.read(tmp_path/"stage/execution.json") == row
    return row


@pytest.mark.parametrize("mode,exit_code,status", [
    ("cell",0,"completed"),("cell",20,"deadline_failed"),("cell",1,"fatal"),
    ("qualification",0,"completed"),("qualification",20,"fatal")])
def test_worker_exit_convention_is_scoped_and_cleanup_is_required(tmp_path, monkeypatch, mode, exit_code, status):
    row = fake_stage(tmp_path,monkeypatch,mode=mode,exit_code=exit_code)
    assert row["status"] == status
    assert row["outer_deadline_observed"] is None


@pytest.mark.parametrize("mode,kind,status", [
    ("cell","ordinary","deadline_failed"), ("qualification","ordinary","fatal"),
    ("cell","service_warmup","fatal")])
def test_supervisor_expiry_continuation_is_only_for_cell_decisions(tmp_path, monkeypatch, mode, kind, status):
    row = fake_stage(tmp_path,monkeypatch,mode=mode,active=registry(kind))
    assert row["status"] == status
    assert (row["outer_deadline_observed"] is not None) == (kind == "ordinary")
    assert row["worker_exit_code"] == -9


@pytest.mark.parametrize("failure", ["fallback", "cleanup_error"])
def test_deadline_exit_cannot_override_backend_or_cleanup_integrity_failure(tmp_path, monkeypatch, failure):
    row = fake_stage(tmp_path,monkeypatch,mode="cell",exit_code=20,**{failure:True})
    assert row["status"] == "fatal"


@pytest.mark.parametrize("seconds", [0.,-1.,float("nan"),float("inf")])
def test_complete_cell_cannot_receive_a_rate_from_invalid_panel_time(seconds):
    with pytest.raises(ValueError):
        audit.qualified_rate("passed",[dict(completed=True,baseline_exact=True) for _ in range(8)],seconds)


def test_selection_uses_observed_complete_panel_rate_instead_of_largest_concurrency():
    cells = [dict(route="gpu",cell_id=f"gpu_c{n}",concurrency=n,eligible=True,
                  completed_decisions_per_second=rate) for n,rate in [(1,.5),(2,1.2),(4,.9),(8,.8)]]
    assert audit.select_eligible(cells,dict(passed=True))["gpu"]["concurrency"] == 2


def test_partial_c1_panels_are_diagnostics_only_without_model_or_comparison_calls(monkeypatch):
    monkeypatch.setattr(audit,"load_saved_library",lambda *args:pytest.fail("No comparison without all C1 cases"))
    cases = [dict(case_id=f"g{i:03d}") for i in range(8)]
    complete = {row["case_id"]:{} for row in cases}
    value = audit.audit_cross_baselines(dict(cases=cases),dict(cpu=complete,gpu={"g000":{}}))
    assert value["complete"] is value["passed"] is False
    assert value["rows"] == []


def test_saved_specification_mismatch_is_rejected_before_case_import(monkeypatch):
    previous = c.read(design.PRIOR/"plan.json")
    original = c.read
    def tampered(path):
        value = original(path)
        if Path(path).name == "spec.json":
            value = dict(value, injected_unfrozen_choice=True)
        return value
    monkeypatch.setattr(c,"read",tampered)
    with pytest.raises(ValueError,match="Exact saved opening specification"):
        design.build_cases(previous)


def test_eight_retained_results_from_killed_or_unclosed_c1_cannot_authorize_later_selection():
    completed = {f"g{i:03d}":{"result":i} for i in range(8)}
    cell = dict(concurrency=1)
    assert audit.admitted_baseline_results(cell,"deadline_failed",dict(final_weights_verified=False),completed) == {}
    assert audit.admitted_baseline_results(cell,"passed",dict(final_weights_verified=False),completed) == {}
    assert audit.admitted_baseline_results(cell,"passed",dict(final_weights_verified=True),completed) == completed
    assert len(completed) == 8  # Retained diagnostic evidence is preserved separately.
