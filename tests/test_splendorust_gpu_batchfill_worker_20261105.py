"""Model-free saved-workload accounting and exact-result admission tests."""
from contextlib import nullcontext
from types import SimpleNamespace
import copy
import threading
import time

import pytest

from tools.splendorust_gpu_batchfill_20261105 import cell, common as c, design, worker


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


class FakeRuntime:
    runtime_sources = cell.q.guard.SOURCE_HASHES

    def __init__(self, route="cpu", scheduler="original8"):
        self.route = route
        self.identity = dict(execution_policy_id=c.EXECUTION_POLICY_ID, internal_seconds=60., outer_seconds=75., scheduler=scheduler)
        self.adapters = []; self.aborts = 0; self.service = FakeService()

    def operation(self, client, *, started, kind):
        assert kind in ("service_warmup", "collision_qualification")
        return nullcontext()

    def make_adapter(self, client, slot, quota):
        assert quota == 512
        adapter = SimpleNamespace(client_id=client, slot=slot, last_service_calls=[], closed=False, uses=0)
        adapter.close = lambda: setattr(adapter, "closed", True)
        self.adapters.append(adapter)
        return adapter

    def abort(self):
        self.aborts += 1


def setup_cell(tmp_path, monkeypatch, concurrency=8, failing=None, drift=None):
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(case_id=f"p{i:03d}", index=i, slot=i % 8, snapshot={"index": i}, observer={"index": i},
        request={"player": "search", "observer": {"index": i}}, expected_root_tokens=[float(i)],
        request_sha256=c.digest({"player": "search", "observer": {"index": i}}),
        expected_results={"cpu": {"case": i}, "gpu": {"case": i}}) for i in range(32)]
    route, scheduler = ("cpu", "original8") if concurrency == 8 else ("gpu", "row_fifo32")
    spec = dict(cell_id=f"test_{concurrency}", route=route, scheduler=scheduler, concurrency=concurrency,
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
    runtime = FakeRuntime(route, scheduler); entered = []; rendezvous = threading.Barrier(concurrency)
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


@pytest.mark.parametrize("concurrency", [8, 16, 32])
def test_fixed_clients_prepared_before_clock_and_reused_across_waves(tmp_path, monkeypatch, concurrency):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch, concurrency)
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
    assert sorted(entered) == list(range(8))
    assert summary["never_started_case_ids"] == [f"p{i:03d}" for i in range(8, 32)]
    assert c.read(tmp_path/"cell/p003/result.json")["scientific"] == [1, 2, 4]
    assert c.read(tmp_path/"cell/p003/failure.json")["clock"]["status"] == "ok"


def test_decision_timeout_preserves_peers_and_never_starts_later_waves(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch,
        failing=(3, TimeoutError("GPU_LIVE_INTERNAL_DEADLINE: complete60s budget")))
    summary = cell.run_cell(plan, spec, runtime, tmp_path/"cell")
    assert summary["status"] == "deadline_failed" and len(summary["completed_case_ids"]) == 7
    assert len(summary["never_started_case_ids"]) == 24
    assert c.read(tmp_path/"cell/p003/failure.json")["clock"]["completed"] is False
    assert all(a.closed for a in runtime.adapters)


def test_unrelated_timeout_is_not_an_admissible_deadline_failure(tmp_path, monkeypatch):
    plan, spec, runtime, entered = setup_cell(tmp_path, monkeypatch,
        failing=(2, TimeoutError("Status mutex exceeded250ms acquisition bound")))
    assert cell.run_cell(plan, spec, runtime, tmp_path/"cell")["status"] == "integrity_failed"


def test_actual_collision_harness_has_two_requests_per_16_and_32_clients(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(case_id=f"p{i:03d}", expected_root_tokens=[float(i)], expected_results={"gpu": {"case": i}})
             for i in range(32)]
    def expected(case, route):
        return {"result": {"diagnostics": {"phases": [{"inference": [{"outputs": [[case["expected_root_tokens"][0]]*83]}]}]}}}
    monkeypatch.setattr(cell, "expected_move", expected)
    runtime = FakeRuntime("gpu", "row_fifo32")
    summary = cell.collision_qualification({"cases": cases}, runtime, tmp_path/"collisions")
    assert summary["callbacks"] == 96 and summary["status"] == "actual_collision_service_qualified"
    assert not runtime.service.clients and runtime.aborts == 0
    assert len(runtime.service.calls) == 96
    for count in (16, 32):
        for i in range(count):
            name = f"collision{count}_{i:02d}"
            assert sum(client == name for client, _, _ in runtime.service.calls) == 2


def test_collision_output_drift_fails_and_preserves_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUT", tmp_path)
    cases = [dict(case_id=f"p{i:03d}", expected_root_tokens=[float(i)], expected_results={"gpu": {"case": i}})
             for i in range(32)]
    monkeypatch.setattr(cell, "expected_move", lambda case, route: {"result": {"diagnostics": {
        "phases": [{"inference": [{"outputs": [[case["expected_root_tokens"][0]]*83]}]}]}}})
    runtime = FakeRuntime("gpu", "row_fifo32"); runtime.service.drift_at = 3
    with pytest.raises(ValueError, match="all collision requests"):
        cell.collision_qualification({"cases": cases}, runtime, tmp_path/"collisions")
    assert list((tmp_path/"collisions").rglob("failure.json"))
    assert not runtime.service.clients and runtime.aborts >= 1
    assert not (tmp_path/"collisions/clients_32").exists()


def test_reversed_blocks_complete_each_configuration_twice():
    specs = design.cells()
    assert [x["configuration"] for x in specs[:4]] == ["cpu8", "gpu8", "gpu16", "gpu32"]
    assert [x["configuration"] for x in specs[4:]] == ["gpu32", "gpu16", "gpu8", "cpu8"]
    for spec in specs:
        assert sum(spec["groups"], []) == list(range(32))


def test_spec_rejects_moved_physical_shape_or_overlap():
    spec = design.cells()[2]
    plan = dict(cells=[spec], group_gap_seconds=0.)
    cell.validate_spec(plan, spec)
    spec["groups"][1][0] = 0
    with pytest.raises(ValueError, match="fixed complete disjoint"):
        cell.validate_spec(plan, spec)
