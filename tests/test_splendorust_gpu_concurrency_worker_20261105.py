"""Model-free fixed-cell accounting, deadline classification and warmup contracts."""
from contextlib import nullcontext
from types import SimpleNamespace
import threading
import time

import pytest

from tools.splendorust_gpu_concurrency_20261105 import cell, common as c, worker


class FakeService:
    def __init__(self):
        self.clients = {}; self.calls = []; self.drift_at = None

    def register(self, client, slot):
        assert client not in self.clients and slot not in self.clients.values()
        self.clients[client] = slot

    def unregister(self, client):
        del self.clients[client]

    def infer(self, client, tokens, batch, *, deadline):
        assert client == "warmup" and self.clients == {"warmup":0} and batch == 1
        assert 0 < deadline-time.perf_counter() <= 9
        self.calls.append((client, list(tokens), deadline))
        marker = 2. if len(self.calls) == self.drift_at else 1.
        return [[marker]*83], dict(request_id=f"warmup:{len(self.calls)}",slot=0)


class FakeRuntime:
    route = "cpu"
    runtime_sources = cell.q.guard.SOURCE_HASHES

    def __init__(self):
        self.adapters = []; self.aborts = 0; self.service = FakeService()

    def operation(self, client, *, started, kind):
        assert client == "warmup" and kind == "service_warmup"
        return nullcontext()

    def make_adapter(self, client, slot, quota):
        assert client == f"case_g{slot:03d}" and quota == 512
        adapter = SimpleNamespace(client_id=client,slot=slot,last_service_calls=[],closed=False)
        adapter.close = lambda:setattr(adapter,"closed",True)
        self.adapters.append(adapter)
        return adapter

    def abort(self):
        self.aborts += 1


def setup_cell(tmp_path, monkeypatch, concurrency, failing=None):
    monkeypatch.setattr(c,"OUT",tmp_path)
    cases = [dict(case_id=f"g{i:03d}",index=i,snapshot={"index":i},observer={"index":i},
                  request={"player":"search","observer":{"index":i}},expected_root_tokens=[float(i)],
                  request_sha256=c.digest({"player":"search","observer":{"index":i}})) for i in range(8)]
    spec = dict(cell_id=f"cpu_c{concurrency}",route="cpu",concurrency=concurrency,
                groups=[list(range(i,i+concurrency)) for i in range(0,8,concurrency)])
    plan = dict(cases=cases,cells=[spec],group_gap_seconds=0.)
    monkeypatch.setattr(cell,"validate_cases",lambda value:value["cases"])
    monkeypatch.setattr(cell.q,"validate_fixed_result",lambda case,result:{"marker":case["index"]})
    monkeypatch.setattr(cell.games,"actual_after",lambda before,action:{"before":before,"action":action})
    monkeypatch.setattr(cell.q.scientific,"audit_decision",lambda *args,**kwargs:None)
    entered = []; rendezvous = threading.Barrier(concurrency)

    def ordinary(runtime, adapters, request, *, started, client_ids, evidence_callback):
        index = request["observer"]["index"]
        assert client_ids == {"search":f"case_g{index:03d}"}
        assert adapters["search"].slot == index
        attempt = c.read(tmp_path/"cell"/f"g{index:03d}"/"attempt.json")
        assert attempt["started_perf_counter"] == started
        entered.append(index); rendezvous.wait(timeout=2)
        if failing is not None and index == failing[0]:
            raise failing[1]
        result = dict(action={"index":index},diagnostics={"index":index},unchanged_scientific_field=[1,2,3])
        evidence_callback(dict(event="complete_result_before_worker_validation",result=result))
        return result

    monkeypatch.setattr(cell.q,"ordinary",ordinary)
    return plan,spec,FakeRuntime(),entered


def test_complete_cell_preserves_original_results_and_permanent_slots(tmp_path, monkeypatch):
    plan,spec,runtime,entered = setup_cell(tmp_path,monkeypatch,2)
    summary = cell.run_cell(plan,spec,runtime,tmp_path/"cell")
    assert summary["status"] == "passed" and summary["full_cell_completed"]
    assert summary["eligible_for_selection"] is False and summary["exact_C1_reproduction_pending"]
    assert sorted(entered) == list(range(8)) and len(summary["groups"]) == 4
    assert summary["never_started_case_ids"] == [] and len(summary["completed_case_ids"]) == 8
    assert [adapter.slot for adapter in runtime.adapters] == list(range(8))
    assert all(adapter.closed for adapter in runtime.adapters)
    for index in range(8):
        folder = tmp_path/"cell"/f"g{index:03d}"
        assert c.read(folder/"result.json") == c.read(folder/"raw_result.json")["result"]
        assert c.read(folder/"attempt.json")["started_perf_counter"] == c.read(folder/"clock.json")["started_perf_counter"]
        assert c.read(folder/"outcome.json")["decision_started"] is True
    assert summary["wall_seconds"] >= sum(row["wall_seconds"] for row in summary["groups"])


def test_first_deadline_group_preserves_successful_peer_and_stops_future_groups(tmp_path, monkeypatch):
    plan,spec,runtime,entered = setup_cell(tmp_path,monkeypatch,2,
        (1,TimeoutError("GPU_LIVE_INTERNAL_DEADLINE: original complete9s budget")))
    summary = cell.run_cell(plan,spec,runtime,tmp_path/"cell")
    assert summary["status"] == "deadline_failed" and not summary["full_cell_completed"]
    assert sorted(entered) == [0,1] and summary["completed_case_ids"] == ["g000"]
    assert summary["never_started_case_ids"] == [f"g{i:03d}" for i in range(2,8)]
    assert len(summary["groups"]) == 1 and runtime.aborts == 0
    failure = c.read(tmp_path/"cell/g001/failure.json")
    assert failure["status"] == "deadline_failed" and failure["clock"]["completed"] is False
    assert c.read(tmp_path/"cell/g000/result.json")["action"] == {"index":0}
    assert all(adapter.closed for adapter in runtime.adapters)


def test_status_mutex_timeout_is_global_integrity_failure_not_permitted_cell_timeout(tmp_path, monkeypatch):
    plan,spec,runtime,entered = setup_cell(tmp_path,monkeypatch,1,
        (0,TimeoutError("Status mutex exceeded250ms acquisition bound")))
    summary = cell.run_cell(plan,spec,runtime,tmp_path/"cell")
    assert summary["status"] == "integrity_failed" and entered == [0]
    assert summary["completed_case_ids"] == [] and len(summary["never_started_case_ids"]) == 7


@pytest.mark.parametrize("error", [TimeoutError("arbitrary timeout"),ValueError("GPU_LIVE_INTERNAL_DEADLINE: original complete9s budget"),
                                  TimeoutError("Status mutex exceeded250ms acquisition bound"),RuntimeError("Inference service failed; no retry")])
def test_unrecognized_errors_never_admit_later_cells(error):
    assert cell.classify_decision_error(error) == "integrity_failed"


def test_poisoned_peer_is_cancelled_not_an_independently_expired_decision():
    parent = TimeoutError("Completed inference exceeded original callback deadline")
    peer = RuntimeError("Inference request failed: case_g005:17"); peer.__cause__ = parent
    assert cell.classify_decision_error(peer) == "cancelled"
    assert cell.error_chain(peer) == [dict(type="RuntimeError",message=str(peer)),dict(type="TimeoutError",message=str(parent))]


@pytest.mark.parametrize("direct_trigger", [True, False])
def test_abort_poison_needs_direct_same_group_clock_bound_timeout(direct_trigger):
    clock = dict(status="error",completed=False,internal_seconds=9.,outer_seconds=10.)
    peer = RuntimeError("Inference service failed; no retry")
    peer.__cause__ = RuntimeError("Evaluator aborted; no replacement or retry")
    errors = {1:peer}
    rows = {1:dict(case_id="g001",status="integrity_failed",clock=clock)}
    if direct_trigger:
        errors[0] = TimeoutError("Complete player response exceeded fixed outer guard")
        rows[0] = dict(case_id="g000",status="deadline_failed",clock=dict(clock,status="timeout"))
    status = cell.resolve_group_outcomes(rows,errors)
    assert status == ("deadline_failed" if direct_trigger else "integrity_failed")
    assert rows[1]["status"] == ("cancelled" if direct_trigger else "integrity_failed")
    assert rows[1]["cancellation_trigger_case_ids"] == (["g000"] if direct_trigger else [])


def test_unrelated_peer_status_failure_is_not_hidden_by_real_deadline_trigger():
    clock = dict(status="error",completed=False,internal_seconds=9.,outer_seconds=10.)
    errors = {0:TimeoutError("Completed inference exceeded original callback deadline"),
              1:TimeoutError("Status mutex exceeded250ms acquisition bound")}
    rows = {i:dict(case_id=f"g{i:03d}",status=cell.classify_decision_error(error),clock=clock)
            for i,error in errors.items()}
    assert cell.resolve_group_outcomes(rows,errors) == "integrity_failed"
    assert rows[1]["status"] == "integrity_failed"


def test_actual_frozen_gpu_dispatch_poison_authenticates_only_expired_original_clock(monkeypatch):
    from tools.splendorust_gpu_live_20261105_03 import backend
    original_request = backend._Request
    now = [100.]
    entered = threading.Event()
    waiting = {"first":threading.Event(), "peer":threading.Event()}
    failures = {}; clocks = {}

    class ObservedEvent:
        def __init__(self, client):
            self.client = client; self.event = threading.Event()
        def set(self):self.event.set()
        def is_set(self):return self.event.is_set()
        def wait(self, timeout=None):
            waiting[self.client].set()
            return self.event.wait(timeout)

    def request(*args, **kwargs):
        value = original_request(*args, **kwargs)
        value.done = ObservedEvent(value.client_id)
        return value

    def forward(tokens, batch):
        entered.set()
        assert waiting["first"].wait(2) and waiting["peer"].wait(2)
        # First deadline is109; peer deadline is109.2. Both waits observe the
        # poisoned completion event, so BOTH real infer calls wrap the timeout.
        now[0] = 109.1
        return [[0.]*83 for _ in range(batch)]

    with monkeypatch.context() as patch:
        patch.setattr(backend.time,"perf_counter",lambda:now[0])
        patch.setattr(backend,"_Request",request)
        service = backend.InferenceService(forward,"gpu")
        service.register("first",0); service.register("peer",1)
        def invoke(client):
            begun = now[0]
            try:
                service.infer(client,[0.]*backend.WIDTH,deadline=begun+9.)
            except BaseException as error:
                failures[client] = error
                clocks[client] = dict(status="error",completed=False,internal_seconds=9.,outer_seconds=10.,
                    started_perf_counter=begun,ended_perf_counter=now[0],elapsed_seconds=now[0]-begun)
        first = threading.Thread(target=invoke,args=("first",)); first.start()
        peer = None
        try:
            assert entered.wait(2)
            now[0] = 100.2
            peer = threading.Thread(target=invoke,args=("peer",)); peer.start()
            first.join(3); peer.join(3)
            assert not first.is_alive() and not peer.is_alive()
            assert set(failures) == {"first","peer"}
            assert all(type(error) is RuntimeError and type(error.__cause__) is TimeoutError for error in failures.values())
            rows = {index:dict(case_id=f"g{index:03d}",clock=clocks[name],status=cell.classify_decision_error(failures[name]))
                    for index,name in enumerate(("first","peer"))}
            assert [row["status"] for row in rows.values()] == ["cancelled","cancelled"]
            assert cell.resolve_group_outcomes(rows,{0:failures["first"],1:failures["peer"]}) == "deadline_failed"
            assert rows[0]["status"] == "deadline_failed" and rows[0]["initial_status"] == "cancelled"
            proof = rows[0]["deadline_proof"]
            assert proof["kind"] == "wrapped_known_deadline_and_own_original_clock_expired"
            assert proof["required_elapsed_seconds"] == 9. and proof["observed_elapsed_seconds"] == 109.1-100.
            assert rows[1]["status"] == "cancelled" and rows[1]["cancellation_trigger_case_ids"] == ["g000"]
            assert "deadline_proof" not in rows[1]
        finally:
            waiting["first"].set(); waiting["peer"].set()
            first.join(3)
            if peer is not None:peer.join(3)
            service.close()


@pytest.mark.parametrize("elapsed,expected", [(9.5,"integrity_failed"),(10.,"deadline_failed")])
def test_wrapped_outer_timeout_requires_original_ten_second_clock(elapsed, expected):
    error = RuntimeError("Inference service failed; no retry")
    error.__cause__ = TimeoutError("Whole completed operation exceeded dual-clock10s bound")
    rows = {0:dict(case_id="g000",status="cancelled",clock=dict(status="error",completed=False,
             internal_seconds=9.,outer_seconds=10.,started_perf_counter=100.,ended_perf_counter=100.+elapsed,
             elapsed_seconds=elapsed))}
    assert cell.resolve_group_outcomes(rows,{0:error}) == expected
    assert rows[0]["status"] == expected


def test_five_fixed_warmups_use_actual_service_contract_without_search(tmp_path, monkeypatch):
    monkeypatch.setattr(c,"OUT",tmp_path)
    runtime = FakeRuntime()
    cell.warmup_runtime({"cases":[{"expected_root_tokens":[.25]}]},runtime,tmp_path)
    saved = c.read(tmp_path/"warmup.json")
    assert saved["status"] == "passed" and len(saved["calls"]) == 5 and len(runtime.service.calls) == 5
    assert runtime.service.clients == {} and runtime.adapters == []
    assert saved["search_decisions"] == 0 and saved["inside_cell_timing"] is False
    assert [row["receipt"]["request_id"] for row in saved["calls"]] == [f"warmup:{i}" for i in range(1,6)]


def test_warmup_output_drift_is_preserved_and_fails_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(c,"OUT",tmp_path)
    runtime = FakeRuntime(); runtime.service.drift_at = 3
    with pytest.raises(ValueError,match="exact fixed same-route"):
        cell.warmup_runtime({"cases":[{"expected_root_tokens":[.25]}]},runtime,tmp_path)
    saved = c.read(tmp_path/"warmup.json")
    assert saved["status"] == "failed" and len(saved["calls"]) == 3 and len(runtime.service.calls) == 3
    assert saved["calls"][-1]["output"] == [[2.]*83] and runtime.service.clients == {}


def test_contract_rejects_any_new_deadline_or_wrong_mode_cell():
    runtime = dict(gpu_physical_batch=8,live_callback_batch=1,maximum_batch_wait_seconds=0.,internal_seconds=9.,outer_seconds=10.)
    plan = dict(schema="splendorust_independent_concurrency_v1",runtime=runtime,group_gap_seconds=0.,
                worker_limits=dict(qualification_seconds=1200,cell_seconds=300),worker_affinity=list(range(19)),cells=[{"cell_id":"cpu_c1"}],
                warmup=dict(calls_per_cell=5,case_index=0,client_id="warmup",slot=0,actual_service=True,
                inside_measured_panel=False,exact_within_route=True,internal_seconds=9.,outer_seconds=10.,failure_stops_study=True))
    assert worker.validate_contract(plan,"cell","cpu_c1") == plan["cells"][0]
    with pytest.raises(ValueError):worker.validate_contract(plan,"qualification","cpu_c1")
    runtime["internal_seconds"] = 10.
    with pytest.raises(ValueError):worker.validate_contract(plan,"cell","cpu_c1")
