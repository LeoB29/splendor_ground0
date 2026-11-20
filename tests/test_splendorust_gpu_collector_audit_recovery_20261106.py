"""Saved-only regression of original histogram-key failure and narrow correction."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path

import pytest

from tools.splendorust_gpu_collector_audit_20261106 import verifier as v


def original_failure_fixture():
    saved = v.read(v.ORIGINAL_OUT/"cells/b1_cpu/saved_audit.json")
    original = deepcopy(saved)
    original["backend_metrics"]["occupied_rows_histogram"] = {
        int(key): count for key, count in saved["backend_metrics"]["occupied_rows_histogram"].items()}
    return original, saved


def patch_only_metrics(proof):
    return dict(proof, backend_metrics=v.normalized_metrics(proof["backend_metrics"]))


def test_exact_saved_failure_is_repaired_without_general_json_roundtrip():
    original, saved = original_failure_fixture()
    diagnosis = v.read(v.ROOT/"outputs/splendorust_gpu_collector_report_20261105_01/diagnostic01/summary.json")
    assert diagnosis["raw_proof_equal"] is False and diagnosis["json_roundtrip_equal"] is True
    assert original != saved and json.loads(json.dumps(original)) == saved
    assert patch_only_metrics(original) == saved
    assert v.sha(v.ORIGINAL_AUDITOR) == v.ORIGINAL_AUDITOR_SHA


def test_nonhistogram_values_are_not_normalized_or_copied_recursively():
    raw = dict(occupied_rows_histogram={1: 7, 8: 3}, numerical=1.25,
               another_mapping={1: "this integer key must stay integer"}, nested=[{"count": 6}])
    fixed = v.normalized_metrics(raw)
    assert fixed["occupied_rows_histogram"] == {"1": 7, "8": 3}
    assert raw["occupied_rows_histogram"] == {1: 7, 8: 3}
    for key in set(raw)-{"occupied_rows_histogram"}:
        assert fixed[key] is raw[key]
    assert 1 in fixed["another_mapping"] and "1" not in fixed["another_mapping"]


@pytest.mark.parametrize("histogram", [{True: 2}, {0: 2}, {9: 2}, {-1: 2}, {1.: 2}, {"1": 2},
                                      {1: True}, {1: 0}, {1: -2}, {1: 2.}, {1: "2"}])
def test_invalid_occupancy_keys_or_counts_are_rejected(histogram):
    with pytest.raises(ValueError, match="integer occupancy"):
        v.normalized_metrics({"occupied_rows_histogram": histogram})


@pytest.mark.parametrize("mutation", ["positive_count", "occupancy", "other_metric", "other_proof"])
def test_exact_original_comparison_still_rejects_all_value_tampering(mutation):
    original, saved = original_failure_fixture()
    metrics = original["backend_metrics"]
    if mutation == "positive_count": metrics["occupied_rows_histogram"][1] += 1
    elif mutation == "occupancy": metrics["occupied_rows_histogram"][2] = metrics["occupied_rows_histogram"].pop(1)
    elif mutation == "other_metric": metrics["mean_occupied_rows"] += .01
    else: original["eligible_roots"] += 1
    assert patch_only_metrics(original) != saved


class FakeBudget:
    def __init__(self): self.calls = []
    def guard(self, claim=None): self.calls.append(claim)
    def evidence(self): return dict(models=0, games=0, clocks_reset=False)


def test_private_loader_changes_only_private_metrics_and_guard(monkeypatch):
    from tools.splendorust_gpu_collector_20261105 import audit as public
    original_index = public.JournalIndex; original_guard = public.guard_audit_clock
    budget = FakeBudget(); private = v.load_private_auditor(budget)
    assert public.JournalIndex is original_index and public.guard_audit_clock is original_guard
    assert Path(private.__file__).resolve() == v.ORIGINAL_AUDITOR
    assert private.JournalIndex is not public.JournalIndex
    assert set(private.JournalIndex.__dict__)-{"__module__", "__doc__"} == {"metrics"}
    base = private.JournalIndex.__bases__[0]
    for name in ("scan", "decision", "complete_attribution", "close"):
        assert getattr(private.JournalIndex, name) is getattr(base, name)
    for name in ("audit_study", "audit_cell", "audit_cell_cost", "audit_qualification", "comparison_summary"):
        assert getattr(private, name).__code__.co_code == getattr(public, name).__code__.co_code
        assert getattr(private, name).__code__.co_consts == getattr(public, name).__code__.co_consts
    raw = dict(occupied_rows_histogram={1: 4}, sentinel={2: "preserve"})
    monkeypatch.setattr(base, "metrics", lambda self, prefix: raw)
    index = private.JournalIndex.__new__(private.JournalIndex)
    assert index.metrics("game:") == dict(occupied_rows_histogram={"1": 4}, sentinel={2: "preserve"})
    assert public.JournalIndex is original_index and public.guard_audit_clock is original_guard


def test_original_auditor_identity_mismatch_fails_before_private_loading(monkeypatch):
    monkeypatch.setattr(v, "sha", lambda path: "wrong")
    with pytest.raises(ValueError, match="exact original frozen auditor"):
        v.load_private_auditor(FakeBudget())


def test_recovery_and_original_dual_clocks_cannot_be_reset(monkeypatch):
    clocks = {"mono": 101., "wall": 1001.}
    monkeypatch.setattr(v.time, "monotonic", lambda: clocks["mono"])
    monkeypatch.setattr(v.time, "time", lambda: clocks["wall"])
    budget = v.AuditBudget(100., 1000.)
    claim = dict(monotonic_origin=20., unix_origin=920., total_seconds=43200.)
    budget.guard(claim)
    for key, value in (("mono", 3700.), ("wall", 4600.)):
        old = clocks[key]; clocks[key] = value
        with pytest.raises(ValueError, match="one-hour"): budget.guard(claim)
        clocks[key] = old
    for key in ("monotonic_origin", "unix_origin"):
        bad = dict(claim); bad[key] -= 43200.
        with pytest.raises(ValueError, match="twelve-hour"): budget.guard(bad)
    clocks["mono"] = 99.
    with pytest.raises(ValueError, match="one-hour"): budget.guard(claim)
    assert budget.monotonic_origin == 100. and budget.unix_origin == 1000.


def test_guard_wrapper_retains_both_original_and_recovery_guards(monkeypatch):
    budget = FakeBudget(); private = v.load_private_auditor(budget)
    claim = dict(monotonic_origin=10., unix_origin=100.)
    monkeypatch.setattr(v.time, "monotonic", lambda: 11.)
    monkeypatch.setattr(v.time, "time", lambda: 101.)
    private.guard_audit_clock(claim)
    assert budget.calls == [claim]
    monkeypatch.setattr(v.time, "time", lambda: 100.+43200.)
    with pytest.raises(ValueError, match="original overall"):
        private.guard_audit_clock(claim)


def test_startup_guard_exactly_once_before_saved_verification_and_no_model_calls():
    tree = ast.parse(inspect.getsource(v.main))
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == "verify_runtime_sources"]
    runs = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "run_recovery"]
    assert len(guards) == len(runs) == 1 and guards[0].lineno < runs[0].lineno
    imports = [alias.name for node in ast.walk(ast.parse(inspect.getsource(v)))
               if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name in ("torch", "torch_directml") for name in imports)


def test_recovery_writes_are_exclusive_and_cannot_touch_original_evidence(tmp_path, monkeypatch):
    original = tmp_path/"original"; recovery = tmp_path/"recovery"
    monkeypatch.setattr(v, "OUT", recovery); monkeypatch.setattr(v, "ORIGINAL_OUT", original)
    with pytest.raises(ValueError, match="separate recovery"):
        v.save(original/"saved_audit.json", {"replacement": True})
    target = recovery/"new.json"; v.save(target, {"preserved": True})
    with pytest.raises(FileExistsError): v.save(target, {"replacement": True})


def test_recovered_label_preserves_valid_negative_result_and_original_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    original = tmp_path/"original"; original.mkdir(); (original/"plan.json").write_text("{}")
    recovery = tmp_path/"recovery_plan.json"; recovery.write_text("{}")
    monkeypatch.setattr(v, "ORIGINAL_OUT", original); monkeypatch.setattr(v, "PLAN", recovery)
    original_auditor = v.artifact(v.ORIGINAL_AUDITOR)
    accepted = dict(status="saved_collector_comparison_verified", auditor=original_auditor,
                    comparison=dict(useful_improvement=False, minimum_each_block_verified_rate_ratio=1.05),
                    sentinel={1: "must not generally canonicalize unrelated proof fields"})
    def audit(*args, **kwargs):
        assert kwargs == dict(expected_plan_sha=v.ORIGINAL_PLAN_SHA, controller_session=79670,
                              controller_exit_code=0, controller_chunk="72c5a5")
        return accepted
    monkeypatch.setattr(v, "load_private_auditor", lambda budget: SimpleNamespace(audit_study=audit))
    plan = dict(original_auditor=original_auditor, inputs=[], failed_audit={"saved": "failure"}, diagnosis={"sole": "key type"})
    result = v.run_recovery(plan, "frozen-recovery-sha", FakeBudget())
    assert result["status"] == "recovered_saved_collector_comparison_verified"
    assert result["original_final_audit_passed"] is False
    assert result["comparison"] is accepted["comparison"] and result["comparison"]["useful_improvement"] is False
    assert result["sentinel"] is accepted["sentinel"] and 1 in result["sentinel"]
    assert result["auditor"] == original_auditor and result["corrected_verifier"] == v.artifact(v.__file__)
    assert result["preserved_original_failure"] == plan["failed_audit"]
