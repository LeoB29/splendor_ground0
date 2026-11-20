"""Model-free fixed-output, admission and original-clock regression tests."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import copy
import threading
import time

import pytest

from tools.splendorust_gpu_compiled_20261105 import cell, common as c, design, qualification as support, worker


class FakeRuntime:
    runtime_sources = cell.q.guard.SOURCE_HASHES
    def __init__(self, implementation="compiled"):
        self.route = "gpu"
        self.identity = dict(execution_policy_id=c.EXECUTION_POLICY_ID, internal_seconds=60., outer_seconds=75.,
                             scheduler="row_fifo32", implementation=implementation)
        self.adapters = []; self.aborts = 0
    def make_adapter(self, client, slot, quota):
        assert quota == 512
        adapter = SimpleNamespace(client_id=client, slot=slot, last_service_calls=[], closed=False, uses=0)
        adapter.close = lambda: setattr(adapter, "closed", True)
        self.adapters.append(adapter)
        return adapter
    def operation(self, client, *, started, kind):
        assert kind == "physical_support"
        return nullcontext()
    def abort(self):
        self.aborts += 1


def setup_cell(tmp_path, monkeypatch, implementation="compiled", failing=None, drift=None):
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(case_id=f"p{i:03d}", index=i, slot=i % 8, snapshot={"index": i}, observer={"index": i},
        request={"player": "search", "observer": {"index": i}}, expected_root_tokens=[float(i)],
        request_sha256=c.digest({"player": "search", "observer": {"index": i}}),
        expected_results={"gpu": {"case": i}}) for i in range(32)]
    spec = design.qualification_cell() if implementation == "compiled" else design.cells()[0]
    plan = dict(cases=cases, cells=design.cells(), qualification_cell=design.qualification_cell(), group_gap_seconds=0.)
    monkeypatch.setattr(cell, "validate_cases", lambda value: value["cases"])
    monkeypatch.setattr(cell.q, "validate_fixed_result", lambda case, result: dict(output83=[float(case["index"])]*83))
    monkeypatch.setattr(cell.games, "actual_after", lambda before, action: {"before": before, "action": action})
    monkeypatch.setattr(cell.q.scientific, "audit_decision", lambda *args, **kwargs: None)
    def expected(case, route):
        result = dict(action={"index": case["index"]}, scientific=[1, 2, 3],
            diagnostics=dict(phases=[dict(inference=[dict(output83=[float(case["index"])]*83)])]))
        return dict(result=result, after={"before": case["snapshot"], "action": result["action"]})
    monkeypatch.setattr(cell, "expected_move", expected)
    runtime = FakeRuntime(implementation); entered = []; rendezvous = threading.Barrier(16)
    def ordinary(actual_runtime, adapters, request, *, started, client_ids, evidence_callback):
        index = request["observer"]["index"]
        assert len(runtime.adapters) == 16
        adapter = adapters["search"]; assert client_ids == {"search": f"worker_{index % 16}"}
        assert adapter.slot == index % 8
        adapter.uses += 1; adapter.last_service_calls = [{"case": index}]
        assert c.read(tmp_path/"cell"/f"p{index:03d}"/"attempt.json")["started_perf_counter"] == started
        entered.append(index); rendezvous.wait(timeout=3)
        if failing is not None and index == failing[0]: raise failing[1]
        result = expected(cases[index], "gpu")["result"]
        if drift == index: result["scientific"][-1] += 1
        evidence_callback(dict(event="complete_result_before_worker_validation", result=result))
        return result
    monkeypatch.setattr(cell.q, "ordinary", ordinary)
    return plan, spec, runtime, entered


@pytest.mark.parametrize("implementation", ["eager", "compiled"])
def test_fixed_clients_reused_and_all_outputs_preserved(tmp_path, monkeypatch, implementation):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, implementation)
    result = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert result["status"] == "passed" and result["full_cell_completed"]
    assert sorted(entered) == list(range(32)) and all(a.closed and a.uses == 2 for a in runtime.adapters)
    assert result["implementation"] == implementation and result["phase"] == spec["phase"]
    for i in range(32):
        folder = tmp_path/"cell"/f"p{i:03d}"
        assert c.read(folder/"service_calls.json") == [{"case": i}]
        assert c.read(folder/"raw_result.json")["result"] == c.read(folder/"result.json")


def test_compiled_exact_mismatch_preserves_full_original_clock_and_peers(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, drift=3)
    result = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert result["status"] == "exact_mismatch" and sorted(entered) == list(range(16))
    assert len(result["completed_case_ids"]) == 15 and len(result["never_started_case_ids"]) == 16
    assert runtime.aborts == 0
    mismatch = c.read(tmp_path/"cell/p003/exact_mismatch.json")
    c.verify([mismatch["actual_result"], mismatch["actual_root"], mismatch["actual_after"], mismatch["completed_original_clock"]])
    assert c.read(tmp_path/"cell/p003/clock.json")["completed"] is True
    assert c.read(tmp_path/"cell/p003/failure.json")["status"] == "exact_mismatch"


def test_eager_source_drift_remains_technical(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, "eager", drift=3)
    assert cell.run_cell(plan, spec, runtime, tmp_path/"cell")["status"] == "integrity_failed"
    assert not (tmp_path/"cell/p003/exact_mismatch.json").exists()


@pytest.mark.parametrize("message,expected", [
    ("GPU_LIVE_INTERNAL_DEADLINE: complete60s budget", "deadline_failed"),
    ("Unrelated status lock timed out", "integrity_failed")])
def test_only_original_deadline_failure_is_admissible(tmp_path, monkeypatch, message, expected):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, failing=(3, TimeoutError(message)))
    assert cell.run_cell(plan, spec, runtime, tmp_path/"cell")["status"] == expected
    assert sorted(entered) == list(range(16))


class FakePhysicalService:
    def __init__(self, drift_at=None, drift_row=1):
        self.clients = {}; self.pending = {}; self.calls = []; self.drift_at = drift_at; self.drift_row = drift_row
    def physical_forward(self, tokens, *, deadline):
        assert len(tokens) == 8*31*48 and 0 < deadline-time.perf_counter() <= 60.
        index = len(self.calls); self.calls.append(list(tokens))
        rows = [[tokens[slot*31*48]]*83 for slot in range(8)]
        if index == self.drift_at: rows[self.drift_row][0] += 1
        return rows


def setup_support(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(index=i, case_id=f"p{i:03d}", slot=i % 8,
        expected_root_tokens=[float(i+1)]*(31*48), expected_results={"gpu": {"case": i}}) for i in range(32)]
    monkeypatch.setattr(cell, "validate_cases", lambda plan: plan["cases"])
    monkeypatch.setattr(cell, "expected_move", lambda case, route: dict(result=dict(diagnostics=dict(
        phases=[dict(inference=[dict(output83=[float(case["index"]+1)]*83)])]))))
    plan = dict(cases=cases, support=support.specification())
    return plan, support.build_fixtures(plan)


def support_runtime(implementation, **kwargs):
    runtime = FakeRuntime(implementation); runtime.service = FakePhysicalService(**kwargs)
    return runtime


def test_fixed36_fixture_native_row_mapping_and_all_zero_padding(tmp_path, monkeypatch):
    plan, fixtures = setup_support(tmp_path, monkeypatch)
    assert len(fixtures) == 36 and [x["fixture_id"] for x in fixtures] == [f"b{i:03d}" for i in range(36)]
    assert all(x["occupied_slots"] == list(range(8)) for x in fixtures[:4])
    for index, fixture in enumerate(fixtures[4:]):
        assert fixture["occupied_slots"] == [index % 8]
        for slot in range(8):
            expected = float(index+1) if slot == index % 8 else 0.
            assert fixture["input_tokens"][slot*31*48:(slot+1)*31*48] == [expected]*(31*48)


def test_support_full154_calls_and_exact_comparison_receipts(tmp_path, monkeypatch):
    plan, fixtures = setup_support(tmp_path, monkeypatch)
    eager, compiled = support_runtime("eager"), support_runtime("compiled")
    left = support.run_implementation(plan, eager, tmp_path/"eager", fixtures)
    right = support.run_implementation(plan, compiled, tmp_path/"compiled", fixtures, eager_folder=tmp_path/"eager")
    assert left["complete_calls"] == right["complete_calls"] == 77
    assert len(eager.service.calls) == len(compiled.service.calls) == 77
    assert eager.service.calls == compiled.service.calls
    assert len(list((tmp_path/"compiled").rglob("against_eager.json"))) == 77
    assert len(list((tmp_path/"compiled").rglob("against_saved.json"))) == 72
    assert all(c.read(c.resolve(item))["status"] == "passed" for item in right["comparisons"])
    assert all(c.read(c.resolve(item))["clock"]["completed"] for item in right["results"])


def test_compiled_padding_difference_preserved_and_stops_immediately(tmp_path, monkeypatch):
    plan, fixtures = setup_support(tmp_path, monkeypatch)
    support.run_implementation(plan, support_runtime("eager"), tmp_path/"eager", fixtures)
    compiled = support_runtime("compiled", drift_at=13, drift_row=1)
    with pytest.raises(support.SupportMismatch) as caught:
        support.run_implementation(plan, compiled, tmp_path/"compiled", fixtures, eager_folder=tmp_path/"eager")
    assert len(compiled.service.calls) == 14
    receipt = c.read(c.resolve(caught.value.receipt)); difference = receipt["difference"]
    assert receipt["fixture_id"] == "b004" and receipt["all_rows_compared"] is True
    assert difference["row"] == 1 and difference["column"] == 0 and difference["expected"] == 0.
    assert c.read(c.resolve(receipt["actual"]))["outputs"][1][0] == 1.
    assert not (tmp_path/"compiled/fixtures/b004/001").exists()
    assert not (tmp_path/"compiled/summary.json").exists()


def test_eager_saved_root_difference_is_not_candidate_negative(tmp_path, monkeypatch):
    plan, fixtures = setup_support(tmp_path, monkeypatch)
    eager = support_runtime("eager", drift_at=5, drift_row=0)
    with pytest.raises(ValueError, match="Eager prerequisite changed") as caught:
        support.run_implementation(plan, eager, tmp_path/"eager", fixtures)
    assert not isinstance(caught.value, support.SupportMismatch)
    assert len(eager.service.calls) == 6
    assert (tmp_path/"eager/fixtures/b000/000/result.json").exists()


def test_signed_zero_difference_is_detected_in_float32_bytes():
    left = [[0.]*83 for _ in range(8)]; right = copy.deepcopy(left); right[3][2] = -0.
    difference = support.first_difference(left, right)
    assert difference["row"] == 3 and difference["column"] == 2
    assert difference["actual_float32_hex"] != difference["expected_float32_hex"]


@pytest.mark.parametrize("implementation,underlying,negative", [
    ("compiled", NotImplementedError("Operator not implemented on this device"), True),
    ("eager", NotImplementedError("Operator not implemented on this device"), False),
    ("compiled", RuntimeError("Allocation failed: out of memory, unsupported allocation"), False),
    ("compiled", OSError("Unsupported file operation"), False),
    ("compiled", TimeoutError("Physical qualification inference deadline"), False),
])
def test_only_original_compiled_runtime_unsupported_is_scientific_negative(tmp_path, monkeypatch, implementation, underlying, negative):
    plan, fixtures = setup_support(tmp_path, monkeypatch)
    eager_folder = None
    if implementation == "compiled":
        eager_folder = tmp_path/"eager"
        support.run_implementation(plan, support_runtime("eager"), eager_folder, fixtures)
    runtime = support_runtime(implementation)
    called = []
    def refused(tokens, *, deadline):
        called.append(True)
        raise RuntimeError("Physical qualification inference failed") from underlying
    runtime.service.physical_forward = refused
    destination = tmp_path/"failed"
    with pytest.raises(support.SupportUnsupported if negative else RuntimeError) as caught:
        support.run_implementation(plan, runtime, destination, fixtures, eager_folder=eager_folder)
    assert len(called) == 1 and not (destination/"warmup/001").exists()
    assert not (destination/"warmup/000/result.json").exists()
    clock = c.read(destination/"warmup/000/clock.json")
    assert clock["status"] == "error" and clock["completed"] is False
    if negative:
        failure = c.read(c.resolve(caught.value.receipt))
        assert failure["schema"] == "compiled_physical_support_refusal_v1"
        assert failure["exception_chain"][-1]["type"] == "NotImplementedError"
        assert support.runtime_unsupported_chain(failure["exception_chain"])
    else:
        assert not isinstance(caught.value, support.SupportUnsupported)
        assert (destination/"warmup/000/technical_failure.json").exists()


def test_unknown_wrapper_does_not_become_runtime_unsupported():
    assert not support.runtime_unsupported_chain([
        dict(type="RuntimeError", module="builtins", message="Unknown wrapper"),
        dict(type="NotImplementedError", module="builtins", message="Not implemented")])


def contract_plan():
    return dict(schema="splendorust_gpu_compiled_v1", runtime=dict(gpu_physical_batch=8, live_callback_batch=1,
        maximum_batch_wait_seconds=0., internal_seconds=60., outer_seconds=75., execution_policy_id=c.EXECUTION_POLICY_ID),
        execution_policy_id=c.EXECUTION_POLICY_ID, worker_limits=dict(qualification_seconds=2700, cell_seconds=900),
        worker_affinity=list(range(19)), cells=design.cells(), qualification_cell=design.qualification_cell(),
        support=support.specification(), new_games=0, training_updates=0, retries=0,
        maximum_panel_decisions=160, timing_decisions=128, timing_observer=False,
        qualification_routes=["compiled_gpu"], qualification_decisions_per_route=67, group_gap_seconds=0.,
        warmup=dict(calls_per_cell=5, case_index=0, client_id="warmup", slot=0, actual_service=True,
            inside_measured_panel=False, exact_within_route=True, internal_seconds=60., outer_seconds=75., failure_stops_study=True))


def test_only_frozen_worker_stages_admitted():
    plan = contract_plan()
    assert worker.validate_contract(plan, "qualification", None) is None
    assert worker.validate_contract(plan, "cell", "qual_compiled")["phase"] == "decision_qualification"
    assert worker.validate_contract(plan, "cell", "b1_eager")["implementation"] == "eager"
    for key, value in (("new_games", 1), ("timing_decisions", 129), ("timing_observer", True)):
        modified = copy.deepcopy(plan); modified[key] = value
        with pytest.raises(ValueError): worker.validate_contract(modified, "qualification", None)


def setup_admission(tmp_path, monkeypatch, full=True, passed=True):
    from tools.splendorust_gpu_compiled_20261105 import audit
    monkeypatch.setattr(c, "OUT", tmp_path); monkeypatch.setattr(c, "PLAN", tmp_path/"plan.json")
    plan = contract_plan(); c.save(c.PLAN, plan)
    identity = c.save(tmp_path/"qualification/worker/compiled/runtime_identity.json",
                      dict(implementation="compiled", compilation_fingerprint="a"*64))
    support_proof = dict(passed=passed, status="support_verified" if passed else "candidate_rejected",
                         compiled_identity=identity, compilation_fingerprint="a"*64)
    decision_proof = dict(status="passed", eligible=True,
                          case_rows=[dict(completed=True, baseline_exact=True) for _ in range(32)])
    bindings = [c.save(tmp_path/"qualification/saved_audit.json", support_proof)]
    if full: bindings.append(c.save(tmp_path/"cells/qual_compiled/saved_audit.json", decision_proof))
    gate = audit.qualification_gate(support_proof, decision_proof) if full else audit.support_admission(support_proof)
    path = tmp_path/("qualification_gate.json" if full else "support_admission.json")
    c.save(path, dict(gate, plan_sha256=c.sha(c.PLAN), source_audits=bindings))
    return plan, path


@pytest.mark.parametrize("full", [False, True])
def test_admission_recomputed_and_compiled_identity_bound(tmp_path, monkeypatch, full):
    plan, path = setup_admission(tmp_path, monkeypatch, full)
    spec = plan["cells"][0] if full else plan["qualification_cell"]
    receipt, identity = worker.verify_admission(plan, "cell", spec, None if full else c.sha(path), c.sha(path) if full else None)
    assert receipt["passed"] and receipt["independently_recomputed"] and receipt["verified_before_model_creation"]
    assert identity["compilation_fingerprint"] == receipt["compilation_fingerprint"] == "a"*64


def test_negative_or_tampered_admission_cannot_load_cell(tmp_path, monkeypatch):
    plan, path = setup_admission(tmp_path, monkeypatch, full=False, passed=False)
    with pytest.raises(ValueError, match="passed original-plan"):
        worker.verify_admission(plan, "cell", plan["qualification_cell"], c.sha(path), None)


def test_unknown_or_wrong_phase_admission_is_rejected(tmp_path, monkeypatch):
    plan, path = setup_admission(tmp_path, monkeypatch, full=True)
    with pytest.raises(ValueError, match="correct admission kind"):
        worker.verify_admission(plan, "cell", plan["qualification_cell"], None, c.sha(path))
    with pytest.raises(ValueError, match="no premature"):
        worker.verify_admission(plan, "qualification", None, c.sha(path), None)


@pytest.mark.parametrize("failed", [False, True])
def test_initialization_receipt_preserves_original_success_or_failure_without_retry(tmp_path, monkeypatch, failed):
    monkeypatch.setattr(c, "OUT", tmp_path)
    calls = []; token = object()
    def build():
        calls.append(True)
        if failed: raise NotImplementedError("Candidate trace not supported")
        return token
    if failed:
        with pytest.raises(NotImplementedError, match="Candidate trace not supported"):
            worker.initialize_runtime(build, tmp_path/"compiled", "compiled")
    else:
        assert worker.initialize_runtime(build, tmp_path/"compiled", "compiled") is token
    receipt = c.read(tmp_path/"initialization/compiled.json")
    assert len(calls) == 1 and receipt["status"] == ("failed" if failed else "completed")
    assert receipt["started_perf_counter"] <= receipt["ended_perf_counter"]
    assert receipt["elapsed_seconds"] == receipt["ended_perf_counter"]-receipt["started_perf_counter"]
    assert receipt["started_unix"] <= receipt["ended_unix"]
    assert receipt["inside_panel_timing"] is False and receipt["no_retry"] is True
    assert (receipt["error"] is not None) is failed
    assert "not isolated compiler duration" in receipt["scope"]
