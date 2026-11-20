"""Model-free unchanged math/routing and enabled-only observation checks."""
from copy import deepcopy
import json
from types import SimpleNamespace
import threading
import time

import pytest

from tools.splendorust_gpu_costs_20261105 import backend as b
from tools.splendorust_gpu_costs_20261105.observer import Recorder, LABELS


def forward(tokens, batch):
    return [[tokens[row*b.WIDTH]] * 83 for row in range(batch)]


def test_source_proof_covers_original_off_and_observed_operation_order():
    proof = b.source_equivalence()
    assert proof["status"] == "source_equivalence_verified" and proof["models_loaded"] == 0
    expected = ("observed_dispatch_without_timing", "observed_journal_without_timing",
                "observed_forward_without_timing", "observed_callback_without_timing",
                "complete_model_factory_without_observer_wiring")
    assert all(name in proof["unchanged_ast_sha256"] for name in expected)
    assert proof["physical_batch"] == 8 and proof["kernel_time_claim"] is False


@pytest.mark.parametrize("enabled", [False, True])
def test_service_chooses_original_dispatch_when_disabled_and_keeps_routing(enabled):
    recorder = Recorder(enabled); journal = []
    service = b.InferenceService(forward, "gpu", scheduler="row_fifo32", observer=recorder, evidence=journal.append)
    try:
        expected_method = b.InferenceService._dispatch_observed if enabled else b.InferenceService._dispatch
        assert service._dispatch.__func__ is expected_method
        service.register("one", 3); service.register("two", 3)
        for name, value in (("one", 7.), ("two", 11.)):
            result, receipt = service.infer(name, [value] * b.WIDTH, deadline=time.perf_counter()+5)
            assert result == [[value]*83] and receipt["slot"] == 3
        assert service.completed_requests == 2 and service.failure is None
        assert [event["kind"] for event in journal] == ["scheduler_dispatch", "inference_batch"] * 2
        labels = [LABELS[event[2]] for event in recorder.events]
        assert bool(labels) == enabled
        if enabled:
            assert labels.count("service_dispatch_total") == labels.count("service_receipts") == 2
            assert {event[4] for event in recorder.events} == {"batch:1", "batch:2"}
    finally:
        service.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_keeps_mandatory_journal_and_off_has_no_per_call_spans(tmp_path, enabled):
    recorder = Recorder(enabled)
    runtime = b.Runtime({}, "gpu", tmp_path/"runtime", {}, forward, lambda: None, {},
                        scheduler="row_fifo32", observer=recorder)
    try:
        expected = b.Runtime._record_observed if enabled else b.Runtime._record
        assert runtime._record.__func__ is expected
        runtime.service.register("one", 0)
        out, _ = runtime.service.infer("one", [2.]*b.WIDTH, deadline=time.perf_counter()+5)
        assert out == [[2.]*83]
        runtime.service.unregister("one")
    finally:
        runtime.close()
    data = [json.loads(line) for line in (tmp_path/"runtime/inference.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in data] == ["scheduler_dispatch", "inference_batch"]
    labels = [LABELS[row[2]] for row in recorder.events]
    assert bool(labels) == enabled
    if enabled:
        assert labels.count("journal_serialization") == labels.count("journal_write_flush") == 2
    assert runtime.observer is recorder and runtime.closed


def fake_adapter(cls, observed):
    adapter = cls.__new__(cls)
    adapter.active_deadline = time.perf_counter()+5
    adapter.client_id = "native0"
    adapter.capture = True; adapter.callbacks = []; adapter.last_service_calls = []
    adapter.boundary = lambda: None
    def infer(client, tokens, batch, *, deadline):
        assert client == adapter.client_id and deadline == adapter.active_deadline
        return [[.25]*83], {"request_id": "native0:1"}
    adapter.runtime = SimpleNamespace(service=SimpleNamespace(infer=infer), observer=Recorder(observed))
    return adapter


def test_observed_callback_preserves_accepted_hashes_capture_and_return_values():
    original = fake_adapter(b.accepted_adapter.ProxySearchOpponent, False)
    observed = fake_adapter(b.ObservedProxySearchOpponent, True)
    tokens = [float(i % 3) for i in range(b.WIDTH)]
    assert original._infer(tokens, 1) == observed._infer(tokens, 1)
    assert original.callbacks == observed.callbacks
    assert original.last_service_calls == observed.last_service_calls
    assert tokens == [float(i % 3) for i in range(b.WIDTH)]
    labels = [LABELS[row[2]] for row in observed.runtime.observer.events]
    assert labels == ["callback_entry_checks", "callback_service_wait", "callback_bookkeeping", "callback_total"]


def test_observed_callback_preserves_service_error_and_does_not_fabricate_bookkeeping():
    adapter = fake_adapter(b.ObservedProxySearchOpponent, True)
    expected = TimeoutError("original complete decision deadline")
    def fail(*args, **kwargs): raise expected
    adapter.runtime.service.infer = fail
    with pytest.raises(TimeoutError) as caught:
        adapter._infer([0.]*b.WIDTH, 1)
    assert caught.value is expected
    assert adapter.callbacks == adapter.last_service_calls == []
    rows = adapter.runtime.observer.events
    assert [LABELS[row[2]] for row in rows] == ["callback_entry_checks", "callback_service_wait", "callback_total"]
    assert [row[-1] for row in rows] == [False, True, True]


def test_observed_same_row_16_client_service_keeps_all_outputs_and_unique_dispatches():
    entered, release = threading.Event(), threading.Event()
    first = [True]; journal = []; outputs = {}; errors = []
    def blocked_forward(tokens, batch):
        if first[0]:
            first[0] = False; entered.set(); assert release.wait(3)
        return forward(tokens, batch)
    recorder = Recorder(True)
    service = b.InferenceService(blocked_forward, "gpu", scheduler="row_fifo32", observer=recorder, evidence=journal.append)
    def run(i):
        try:
            outputs[i] = service.infer(f"c{i}", [float(i+1)]*b.WIDTH, deadline=time.perf_counter()+5)[0]
        except BaseException as error: errors.append(error)
    threads = []
    try:
        for i in range(16): service.register(f"c{i}", i % 8)
        for i in range(16):
            thread = threading.Thread(target=run, args=(i,)); thread.start(); threads.append(thread)
        assert entered.wait(3)
        until = time.perf_counter()+2
        while len(service.pending) < 16 and time.perf_counter() < until: time.sleep(.002)
        assert len(service.pending) == 16
        release.set()
        for thread in threads: thread.join(5)
        assert not errors and not any(thread.is_alive() for thread in threads)
        assert outputs == {i: [[float(i+1)]*83] for i in range(16)}
        batches = [row for row in journal if row["kind"] == "inference_batch"]
        assert sum(len(row["requests"]) for row in batches) == 16
        assert all(len({request["slot"] for request in row["requests"]}) == len(row["requests"]) for row in batches)
    finally:
        release.set(); service.close()
        for thread in threads: thread.join(5)
