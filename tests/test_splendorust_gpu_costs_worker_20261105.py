"""Model-free saved-workload accounting and exact-result admission tests."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import copy
import threading
import time

import pytest

from tools.splendorust_gpu_costs_20261105 import cell, common as c, design, worker


class FakeService:
    def __init__(self):
        self.clients = {}; self.calls = []; self.drift_at = None; self.lock = threading.Lock()

    def register(self, client, slot):
        assert client not in self.clients
        self.clients[client] = slot

    def unregister(self, client):
        del self.clients[client]

    def infer(self, client, tokens, batch, *, deadline):
        assert client in self.clients and batch == 1 and 0 < deadline-time.perf_counter() <= 60
        with self.lock:
            index = len(self.calls)
            self.calls.append((client, list(tokens), deadline))
        marker = tokens[0]+1 if index == self.drift_at else tokens[0]
        return [[marker]*83], dict(request_id=f"{client}:{index}", slot=self.clients[client])


class FakeObserver:
    def __init__(self, enabled):
        self.enabled = enabled; self.markers = []; self.spans = []
        self.active = threading.local()
    def mark(self, label):
        value = dict(label=label, event_index=0, perf_counter=time.perf_counter())
        self.markers.append(value)
        return value
    @contextmanager
    def span(self, kind, owner):
        assert self.enabled, "Control must not enter a disabled observer context"
        assert getattr(self.active, "owner", None) is None
        self.active.owner = owner
        row = dict(kind=kind, owner=owner, started=time.perf_counter())
        try:
            yield
        finally:
            row["ended"] = time.perf_counter()
            self.spans.append(row)
            self.active.owner = None


class FakeRuntime:
    runtime_sources = cell.q.guard.SOURCE_HASHES

    def __init__(self, route="gpu", scheduler="row_fifo32", observed=True):
        self.route = route
        self.observer = FakeObserver(observed)
        self.identity = dict(execution_policy_id=c.EXECUTION_POLICY_ID, internal_seconds=60., outer_seconds=75., scheduler=scheduler)
        self.adapters = []; self.aborts = 0; self.service = FakeService()

    def operation(self, client, *, started, kind):
        assert kind in ("service_warmup", "isolated_physical")
        return nullcontext()

    def make_adapter(self, client, slot, quota):
        assert quota == 512
        adapter = SimpleNamespace(client_id=client, slot=slot, last_service_calls=[], closed=False, uses=0)
        adapter.close = lambda: setattr(adapter, "closed", True)
        self.adapters.append(adapter)
        return adapter

    def abort(self):
        self.aborts += 1


def setup_cell(tmp_path, monkeypatch, observed=True, failing=None, drift=None):
    concurrency = 16
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(case_id=f"p{i:03d}", index=i, slot=i % 8, snapshot={"index": i}, observer={"index": i},
        request={"player": "search", "observer": {"index": i}}, expected_root_tokens=[float(i)],
        request_sha256=c.digest({"player": "search", "observer": {"index": i}}),
        expected_results={"cpu": {"case": i}, "gpu": {"case": i}}) for i in range(32)]
    route, scheduler = "gpu", "row_fifo32"
    spec = dict(cell_id=f"test_{concurrency}", route=route, scheduler=scheduler, concurrency=concurrency,
        observed=observed, phase="overhead",
        groups=[list(range(i, i+concurrency)) for i in range(0, 32, concurrency)])
    plan = dict(cases=cases, cells=[spec], group_gap_seconds=0.)
    monkeypatch.setattr(cell, "validate_cases", lambda value: value["cases"])
    monkeypatch.setattr(cell.q, "validate_fixed_result", lambda case, result: {"marker": case["index"]})
    monkeypatch.setattr(cell.games, "actual_after", lambda before, action: {"before": before, "action": action})
    monkeypatch.setattr(cell.q.scientific, "audit_decision", lambda *args, **kwargs: None)
    def expected(case, route):
        result = dict(action={"index": case["index"]}, diagnostics={"index": case["index"]}, scientific=[1, 2, 3])
        return dict(result=result, after={"before": case["snapshot"], "action": result["action"]})
    monkeypatch.setattr(cell, "expected_move", expected)
    runtime = FakeRuntime(route, scheduler, observed); entered = []; rendezvous = threading.Barrier(concurrency)
    def ordinary(actual_runtime, adapters, request, *, started, client_ids, evidence_callback):
        index = request["observer"]["index"]
        assert len(runtime.adapters) == concurrency
        adapter = adapters["search"]
        assert client_ids == {"search": f"worker_{index % concurrency}"}
        assert adapter.slot == index % 8
        adapter.uses += 1
        adapter.last_service_calls = [{"case": index}]
        attempt = c.read(tmp_path/"cell"/f"p{index:03d}"/"attempt.json")
        assert attempt["started_perf_counter"] == started
        entered.append(index); rendezvous.wait(timeout=3)
        if failing is not None and index == failing[0]: raise failing[1]
        result = expected(cases[index], route)["result"]
        if drift == index: result["scientific"][-1] += 1
        evidence_callback(dict(event="complete_result_before_worker_validation", result=result))
        return result
    monkeypatch.setattr(cell.q, "ordinary", ordinary)
    return plan, spec, runtime, entered


@pytest.mark.parametrize("observed", [False, True])
def test_fixed_clients_prepared_before_clock_and_reused_across_waves(tmp_path, monkeypatch, observed):
    concurrency = 16
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, observed)
    summary = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert summary["status"] == "passed" and summary["full_cell_completed"]
    assert sorted(entered) == list(range(32)) and len(summary["groups"]) == 32//concurrency
    assert len(runtime.adapters) == concurrency
    assert all(a.closed and a.uses == 32//concurrency for a in runtime.adapters)
    assert c.read(tmp_path/"cell/adapters_ready.json")["observed_perf_counter"] <= summary["started_perf_counter"]
    assert summary["wall_seconds"] >= sum(group["wall_seconds"] for group in summary["groups"])
    for index in range(32):
        folder = tmp_path/"cell"/f"p{index:03d}"
        assert c.read(folder/"service_calls.json") == [{"case": index}]
        assert c.read(folder/"outcome.json")["client_id"] == f"worker_{index % concurrency}"
        assert c.read(folder/"result.json") == c.read(folder/"raw_result.json")["result"]


def test_exact_full_result_drift_is_integrity_failure_and_preserved(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, drift=3)
    summary = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert summary["status"] == "integrity_failed" and not summary["full_cell_completed"]
    assert sorted(entered) == list(range(16))
    assert summary["never_started_case_ids"] == [f"p{i:03d}" for i in range(16, 32)]
    assert c.read(tmp_path/"cell/p003/result.json")["scientific"] == [1, 2, 4]
    assert c.read(tmp_path/"cell/p003/failure.json")["clock"]["status"] == "ok"


def test_decision_timeout_preserves_peers_and_never_starts_later_waves(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch,
        failing=(3, TimeoutError("GPU_LIVE_INTERNAL_DEADLINE: complete60s budget")))
    summary = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert summary["status"] == "deadline_failed" and len(summary["completed_case_ids"]) == 15
    assert len(summary["never_started_case_ids"]) == 16
    assert c.read(tmp_path/"cell/p003/failure.json")["clock"]["completed"] is False
    assert all(a.closed for a in runtime.adapters)


def test_unrelated_timeout_is_not_an_admissible_deadline_failure(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch,
        failing=(2, TimeoutError("Status mutex exceeded250ms acquisition bound")))
    assert cell.run_cell(plan, spec, runtime, tmp_path/"cell")["status"] == "integrity_failed"


def test_panel_markers_enclose_measured_interval_and_no_observer_export_occurs_inside(tmp_path, monkeypatch):
    plan, spec, runtime, _ = setup_cell(tmp_path, monkeypatch)
    summary = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert [x["label"] for x in runtime.observer.markers] == ["panel_start", "panel_end"]
    assert runtime.observer.markers[0]["perf_counter"] <= summary["started_perf_counter"]
    assert runtime.observer.markers[1]["perf_counter"] >= summary["ended_perf_counter"]
    assert not list(tmp_path.rglob("observer_timing.json"))


@pytest.mark.parametrize("observed", [False, True])
def test_inpanel_durable_writes_have_exact_observer_coverage_without_control_context(tmp_path, monkeypatch, observed):
    plan, spec, runtime, _ = setup_cell(tmp_path, monkeypatch, observed)
    original_save = c.save
    records = []
    def checked_save(path, value):
        records.append((path.relative_to(tmp_path/"cell").as_posix(),
                        getattr(runtime.observer.active, "owner", None)))
        return original_save(path, value)
    monkeypatch.setattr(c, "save", checked_save)
    summary = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert summary["status"] == "passed"
    expected = {f"p{i:03d}/{name}.json" for i in range(32) for name in
                ("attempt", "raw_result", "clock", "result", "root", "after", "service_calls", "outcome")}
    expected.update(f"group:{i}/{name}.json" for i in range(2) for name in ("start", "summary"))
    if observed:
        actual = [row["owner"] for row in runtime.observer.spans]
        assert len(actual) == 260 and set(actual) == expected
        assert all(row["kind"] == "case_recording" for row in runtime.observer.spans)
        assert all(runtime.observer.markers[0]["perf_counter"] <= row["started"] <= row["ended"]
                   <= runtime.observer.markers[1]["perf_counter"] for row in runtime.observer.spans)
        covered = {owner for _, owner in records if owner is not None}
        assert covered == expected
        assert all(path not in {"adapters_ready.json", "cell_summary.json"}
                   and not path.endswith(("case.json", "request.json"))
                   for path, owner in records if owner is not None)
    else:
        assert runtime.observer.spans == []
        assert all(owner is None for _, owner in records)


class FakePhysicalService:
    def __init__(self, drift_at=None):
        self.clients = {}; self.pending = {}; self.calls = []; self.drift_at = drift_at
    def physical_forward(self, tokens, *, deadline):
        assert len(tokens) == 8*31*48 and 0 < deadline-time.perf_counter() <= 60.
        index = len(self.calls); self.calls.append(dict(tokens=list(tokens), deadline=deadline))
        rows = [[tokens[i*31*48]]*83 for i in range(8)]
        if index == self.drift_at: rows[0][0] += 1.
        return rows


def setup_isolated(tmp_path, monkeypatch, drift_at=None):
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(index=i, case_id=f"p{i:03d}", slot=i % 8,
        expected_root_tokens=[float(i)]*(31*48), expected_results={"gpu": {"case": i}}) for i in range(32)]
    monkeypatch.setattr(cell, "validate_cases", lambda plan: plan["cases"])
    def expected(case, route):
        return dict(result=dict(diagnostics=dict(phases=[dict(inference=[dict(output83=[float(case["index"])]*83)])])))
    monkeypatch.setattr(cell, "expected_move", expected)
    plan = dict(cases=cases, isolated=design.isolated())
    runtime = FakeRuntime("gpu", "row_fifo32", True)
    runtime.service = FakePhysicalService(drift_at)
    return plan, runtime


def test_isolated_exact_counts_inputs_clocks_and_markers(tmp_path, monkeypatch):
    plan, runtime = setup_isolated(tmp_path, monkeypatch)
    result = cell.run_isolated(plan, plan["isolated"][0], runtime, tmp_path/"isolated")
    assert result["status"] == "isolated_batches_passed"
    assert result["warmup_calls"] == 20 and result["measured_calls"] == 128
    assert len(runtime.service.calls) == 148
    assert len(result["phases"]["measured"]["elapsed_complete_call_seconds"]) == 128
    for phase, count in (("warmup", 20), ("measured", 128)):
        for index in range(count):
            row = c.read(tmp_path/"isolated"/phase/f"{index:04d}"/"result.json")
            clock = c.read(tmp_path/"isolated"/phase/f"{index:04d}"/"clock.json")
            assert row["clock"] == clock and clock["completed"] is True
            assert row["observer_start"]["perf_counter"] <= clock["started_perf_counter"]
            assert row["observer_end"]["perf_counter"] >= clock["ended_perf_counter"]
            assert row["case_indexes"] == list(range((index % 4)*8, (index % 4+1)*8))
    assert not list(tmp_path.rglob("observer_timing.json"))


def test_isolated_output_drift_preserved_and_later_calls_not_started(tmp_path, monkeypatch):
    plan, runtime = setup_isolated(tmp_path, monkeypatch, drift_at=23)
    with pytest.raises(ValueError, match="exact isolated saved physical rows"):
        cell.run_isolated(plan, plan["isolated"][0], runtime, tmp_path/"isolated")
    assert len(runtime.service.calls) == 24
    assert (tmp_path/"isolated/measured/0003/result.json").exists()
    assert not (tmp_path/"isolated/measured/0004").exists()
    assert not (tmp_path/"isolated/summary.json").exists()


def test_isolated_rejects_unfrozen_cycle_count(tmp_path, monkeypatch):
    plan, runtime = setup_isolated(tmp_path, monkeypatch)
    plan["isolated"][0]["measured_cycles"] += 1
    with pytest.raises(ValueError, match="frozen isolated"):
        cell.run_isolated(plan, plan["isolated"][0], runtime, tmp_path/"isolated")
    assert runtime.service.calls == []


def contract_plan():
    return dict(schema="splendorust_gpu_costs_v1", runtime=dict(gpu_physical_batch=8, live_callback_batch=1,
        maximum_batch_wait_seconds=0., internal_seconds=60., outer_seconds=75., execution_policy_id=c.EXECUTION_POLICY_ID),
        execution_policy_id=c.EXECUTION_POLICY_ID, worker_limits=dict(qualification_seconds=2700, cell_seconds=900, isolated_seconds=900),
        worker_affinity=list(range(19)), cells=design.cells(), isolated=design.isolated(), new_games=0, training_updates=0,
        retries=0, maximum_diagnostic_decisions=192, qualification_routes=["gpu"], qualification_decisions_per_route=67,
        warmup=dict(calls_per_cell=5, case_index=0, client_id="warmup", slot=0, actual_service=True,
            inside_measured_panel=False, exact_within_route=True, internal_seconds=60., outer_seconds=75., failure_stops_study=True),
        group_gap_seconds=0.)


def test_worker_contract_has_only_declared_stages():
    plan = contract_plan()
    assert worker.validate_contract(plan, "qualification", None) is None
    assert worker.validate_contract(plan, "cell", "oh1_control")["observed"] is False
    assert worker.validate_contract(plan, "cell", "diag2")["observed"] is True
    assert worker.validate_contract(plan, "isolated", "iso2_post")["measured_cycles"] == 32
    with pytest.raises(ValueError): worker.validate_contract(plan, "games", None)
    with pytest.raises(ValueError): worker.validate_contract(plan, "cell", "unknown")


@pytest.mark.parametrize("key,value", [("new_games", 1), ("qualification_routes", ["cpu", "gpu"]),
    ("maximum_diagnostic_decisions", 193)])
def test_worker_contract_rejects_scope_extension(key, value):
    plan = contract_plan(); plan[key] = value
    with pytest.raises(ValueError): worker.validate_contract(plan, "qualification", None)


def setup_gate(tmp_path, monkeypatch, observed_seconds=10.):
    from tools.splendorust_gpu_costs_20261105 import audit
    monkeypatch.setattr(c, "OUT", tmp_path)
    monkeypatch.setattr(c, "PLAN", tmp_path/"plan.json")
    plan = dict(overhead=dict(order=["oh1_control", "oh1_observed", "oh2_observed", "oh2_control"]))
    c.save(c.PLAN, plan)
    proofs = []; bindings = []
    for spec in design.cells()[:4]:
        proof = dict(spec, status="passed", eligible=True,
            case_rows=[dict(completed=True, baseline_exact=True) for _ in range(32)],
            wall_seconds=observed_seconds if spec["observed"] else 10.)
        proofs.append(proof)
        bindings.append(c.save(c.OUT/"cells"/spec["cell_id"]/"saved_audit.json", proof))
    receipt = dict(audit.overhead_gate(proofs), plan_sha256=c.sha(c.PLAN), source_cell_audits=bindings)
    c.save(c.OUT/"overhead_gate.json", receipt)
    return plan, receipt


def test_full_worker_gate_is_recomputed_from_bound_actual_audits(tmp_path, monkeypatch):
    plan, receipt = setup_gate(tmp_path, monkeypatch, 10.2)
    binding = worker.verify_full_stage_gate(plan, "isolated", design.isolated()[0], c.sha(c.OUT/"overhead_gate.json"))
    assert binding["passed"] and binding["independently_recomputed"] and binding["verified_before_model_creation"]
    assert len(binding["source_cell_audits"]) == 4


def test_negative_overhead_gate_cannot_launch_full_worker(tmp_path, monkeypatch):
    plan, receipt = setup_gate(tmp_path, monkeypatch, 11.)
    assert receipt["passed"] is False
    with pytest.raises(ValueError, match="passed original-plan overhead gate"):
        worker.verify_full_stage_gate(plan, "cell", design.cells()[4], c.sha(c.OUT/"overhead_gate.json"))


def test_full_worker_rejects_tampered_source_audit(tmp_path, monkeypatch):
    plan, receipt = setup_gate(tmp_path, monkeypatch)
    target = c.OUT/"cells/oh1_control/saved_audit.json"
    target.write_text(target.read_text()+" ", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen input changed"):
        worker.verify_full_stage_gate(plan, "isolated", design.isolated()[0], c.sha(c.OUT/"overhead_gate.json"))


def test_full_worker_gate_required_and_overhead_cannot_pretend_to_use_it():
    with pytest.raises(ValueError, match="exact overhead gate before models"):
        worker.verify_full_stage_gate({}, "isolated", design.isolated()[0], None)
    assert worker.verify_full_stage_gate({}, "qualification", None, None) is None
    with pytest.raises(ValueError, match="premature"):
        worker.verify_full_stage_gate({}, "cell", design.cells()[0], "a"*64)

