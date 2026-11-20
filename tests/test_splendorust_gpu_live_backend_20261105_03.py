"""Model-free lifetime, watchdog and native-proxy boundary checks."""
from collections import deque
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_live_20261105_03 import backend as b
from tools.splendorust_gpu_live_20261105_03 import adapters as a


def fake_forward(tokens, batch):
    return [[float(tokens[index*b.WIDTH])] * 83 for index in range(batch)]


def runtime(tmp_path):
    return b.Runtime({}, "cpu", tmp_path / "runtime", {}, fake_forward, lambda: None, {})


def test_nested_watchdog_keeps_whole_operation_origin(tmp_path):
    owner = runtime(tmp_path)
    begun = time.perf_counter() - .1
    try:
        with owner.operation("game0", started=begun, kind="ordinary") as outer:
            with owner.operation("game0", kind="native_exchange") as inner:
                assert inner is outer and inner["started_perf"] == begun
                saved = json.loads((owner.folder / "active_decisions.json").read_text())
                assert saved["pid"] == saved["identity"]["pid"]
                assert len(saved["active"]) == 1
                assert saved["active"][0]["deadline_perf"] == begun + 10
                assert abs(saved["active"][0]["deadline_unix"] - saved["active"][0]["started_unix"] - 10) < 1e-6
            assert owner.active["game0"] is outer
        assert not owner.active
        assert json.loads((owner.folder / "active_decisions.json").read_text())["active"] == []
    finally:
        owner.close()


@pytest.mark.parametrize("clock", ["perf", "unix"])
def test_finish_rejects_either_expired_clock(tmp_path, monkeypatch, clock):
    owner = runtime(tmp_path)
    real_perf, real_wall = b.time.perf_counter, b.time.time
    try:
        with pytest.raises(TimeoutError, match="dual-clock"):
            with owner.operation("game0") as active:
                if clock == "perf":
                    monkeypatch.setattr(b.time, "perf_counter", lambda: active["deadline_perf"] + .01)
                else:
                    monkeypatch.setattr(b.time, "time", lambda: active["deadline_unix"] + .01)
        assert not owner.active
    finally:
        monkeypatch.setattr(b.time, "perf_counter", real_perf)
        monkeypatch.setattr(b.time, "time", real_wall)
        owner.close()


def test_one_native_client_cannot_have_two_operation_threads(tmp_path):
    owner = runtime(tmp_path); errors = []
    try:
        with owner.operation("game0"):
            def duplicate():
                try:
                    with owner.operation("game0"):
                        pass
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=duplicate); thread.start(); thread.join(1)
            assert not thread.is_alive() and len(errors) == 1
            assert "concurrent operation" in str(errors[0])
    finally:
        owner.close()


def test_fixed_physical_qualification_uses_gpu_thread_and_refuses_clients():
    thread_ids = []
    def forward(tokens, batch):
        thread_ids.append(threading.get_ident())
        return fake_forward(tokens, batch)
    service = b.InferenceService(forward, "gpu")
    try:
        tokens = [float(i) for i in range(8) for _ in range(b.WIDTH)]
        result = service.physical_forward(tokens, deadline=time.perf_counter()+2)
        assert [row[0] for row in result] == list(range(8))
        assert thread_ids == [service.thread.ident]
        service.register("game0", 0)
        with pytest.raises(ValueError, match="no clients"):
            service.physical_forward(tokens, deadline=time.perf_counter()+2)
    finally:
        service.close()


def test_fixed_physical_registration_cannot_race_qualification():
    entered, release = threading.Event(), threading.Event()
    def forward(tokens, batch):
        entered.set(); assert release.wait(2)
        return fake_forward(tokens, batch)
    service = b.InferenceService(forward, "gpu"); errors = []
    def run():
        try:
            service.physical_forward([0.] * (8*b.WIDTH), deadline=time.perf_counter()+3)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run); thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(ValueError, match="qualification cannot overlap"):
            service.register("game0", 0)
    finally:
        release.set(); thread.join(3); service.close()
    assert not errors and not thread.is_alive()


@pytest.mark.parametrize("route", ["cpu", "gpu"])
@pytest.mark.parametrize("deadline", [0., float("inf"), float("nan")])
def test_invalid_physical_deadline_rejected_before_forward(route, deadline):
    service = b.InferenceService(lambda *_: pytest.fail("forward must not run"), route)
    try:
        with pytest.raises(ValueError, match="qualification deadline"):
            service.physical_forward([0.]*(8*b.WIDTH), deadline=deadline)
    finally:
        service.close()


def make_proxy(tmp_path, monkeypatch):
    owner = runtime(tmp_path); captured = []
    def inherited_init(self, profile_document, *, shared_from):
        captured.append(shared_from)
        self.profile = profile_document; self.model = shared_from.model; self.reference = shared_from.reference
        self.capture = False; self.callbacks = []; self.deadline = None
        self.inference_calls = self.inference_rows = 0
    monkeypatch.setattr(a.SearchOpponent, "__init__", inherited_init)
    monkeypatch.setattr(a.Opponent, "close", lambda self: None)
    proxy = a.ProxySearchOpponent({"iterations": 16}, runtime=owner, client_id="g0", slot=0)
    owner.adapters["g0"] = proxy
    assert len(captured) == 1 and captured[0].model is None and captured[0].reference == {}
    return owner, proxy


def test_proxy_uses_only_shared_native_constructor_and_original_callback_shape(tmp_path, monkeypatch):
    owner, proxy = make_proxy(tmp_path, monkeypatch)
    try:
        proxy.active_deadline = time.perf_counter()+2; proxy.capture = True
        output = proxy._infer([.25]*b.WIDTH, 1)
        assert output == [[.25]*83] and len(proxy.last_service_calls) == 1
        assert set(proxy.callbacks[0]) == {"callback_sha256", "batch", "input_sha256", "output_sha256", "tokens", "outputs", "output83"}
        assert proxy.callbacks[0]["output83"] == output[0]
        assert proxy.last_service_calls[0]["request_id"] == "g0:1"
        proxy._infer([.5]*b.WIDTH, 1)
        assert set(proxy.callbacks[1]) == {"callback_sha256", "batch", "input_sha256", "output_sha256"}
        assert [x["request_id"] for x in proxy.last_service_calls] == ["g0:1", "g0:2"]
    finally:
        owner.close()


def test_proxy_does_not_reset_absolute_decision_deadline(tmp_path, monkeypatch):
    owner, proxy = make_proxy(tmp_path, monkeypatch)
    deadline = time.perf_counter()+1.; proxy.request_deadline = deadline
    def choose(self, observer):
        assert self.active_deadline == deadline
        assert self.runtime.active[self.client_id]["kind"] == "ordinary"
        self._infer([0.]*b.WIDTH, 1)
        return "action", {"trace": True}
    monkeypatch.setattr(a.Opponent, "choose", choose)
    try:
        with owner.operation("g0", kind="ordinary"):
            assert proxy.choose({}, diagnostic_capture=True) == ("action", {"trace": True})
        assert proxy.active_deadline is None and proxy.capture is False
    finally:
        owner.close()


def test_native_exchange_gets_own_watchdog_and_internal_deadline(tmp_path, monkeypatch):
    owner, proxy = make_proxy(tmp_path, monkeypatch)
    def exchange(self, request):
        assert self.active_deadline is not None
        assert self.runtime.active[self.client_id]["kind"] == "native_exchange"
        return {"kind": "result"}
    monkeypatch.setattr(a.Opponent, "exchange", exchange)
    try:
        result = proxy.exchange({"op": "inspect"})
        assert result["neural_callback_count"] == 0
        assert proxy.active_deadline is None and not owner.active
    finally:
        owner.close()


def test_proxy_closes_only_owned_transport_and_releases_slot(tmp_path, monkeypatch):
    owner, proxy = make_proxy(tmp_path, monkeypatch); closed = []
    monkeypatch.setattr(a.Opponent, "close", lambda self: closed.append(self.client_id))
    proxy.close(); proxy.close()
    assert closed == ["g0"] and not owner.service.clients and not owner.adapters
    owner.close()


class ControlledArrivalQueue:
    """Deterministic arrivals: a blocking get after the first item is an error.

    The idle receive may block; once work arrives only nonblocking draining is
    permitted until forward starts. This checks dispatch policy without a
    machine-speed assertion or an actual model.
    """
    def __init__(self, requests=()):
        self.ready = deque(requests)
        self.awaiting_dispatch = False
        self.calls = []

    def get(self, block=True, timeout=None):
        self.calls.append(("idle_get", block, timeout))
        assert not self.awaiting_dispatch, "timed companion gathering is forbidden"
        assert block and timeout == .05
        if not self.ready:
            return None
        self.awaiting_dispatch = True
        return self.ready.popleft()

    def get_nowait(self):
        self.calls.append(("drain_ready",))
        if not self.ready:
            raise b.queue.Empty
        return self.ready.popleft()

    def put_nowait(self, request):
        self.ready.append(request)


def controlled_gpu_loop():
    # Initialize the service without starting a worker, then drive its actual
    # GPU loop synchronously with a controlled arrival queue and fake forward.
    service = b.InferenceService(fake_forward, "cpu")
    service.route = "gpu"
    service.queue = ControlledArrivalQueue()
    evidence = []
    service.evidence = evidence.append
    return service, evidence


def pending_request(service, slot, marker, *, deadline=None):
    client = f"slot{slot}"
    service.register(client, slot)
    submitted = time.perf_counter()
    request = b._Request(client, f"{client}:1", slot, [marker]*b.WIDTH,
                         submitted, submitted+5 if deadline is None else deadline)
    service.pending[client] = request
    return request


def test_single_request_dispatches_without_any_timed_companion_get():
    service, evidence = controlled_gpu_loop()
    assert b.MAX_WAIT_SECONDS == service.max_wait_seconds == 0.0
    request = pending_request(service, 7, .75)
    service.queue.put_nowait(request)
    forwarded = []

    def forward(tokens, batch):
        service.queue.awaiting_dispatch = False
        forwarded.append((batch, [tokens[i*b.WIDTH] for i in range(batch)]))
        return fake_forward(tokens, batch)

    service.forward = forward
    try:
        service._run()
        assert service.failure is None
        assert forwarded == [(8, [0.]*7+[.75])]
        assert service.queue.calls == [("idle_get", True, .05), ("drain_ready",),
                                       ("idle_get", True, .05)]
        assert request.done.is_set() and request.outputs == [[.75]*83]
        assert evidence[0]["requests"][0]["request_id"] == "slot7:1"
    finally:
        service.close()


def test_already_ready_eight_client_backlog_coalesces_without_waiting():
    service, evidence = controlled_gpu_loop()
    arrival_order = [7, 1, 4, 0, 6, 3, 2, 5]
    requests = [pending_request(service, slot, (slot+1)/8) for slot in arrival_order]
    service.queue.ready.extend(requests)
    forwarded = []

    def forward(tokens, batch):
        service.queue.awaiting_dispatch = False
        forwarded.append([tokens[i*b.WIDTH] for i in range(batch)])
        return fake_forward(tokens, batch)

    service.forward = forward
    try:
        service._run()
        assert service.failure is None
        assert forwarded == [[(slot+1)/8 for slot in range(8)]]
        assert len(evidence) == 1 and evidence[0]["physical_batch"] == 8
        assert [row["slot"] for row in evidence[0]["requests"]] == arrival_order
        assert all(request.done.is_set() and request.outputs == [[(request.slot+1)/8]*83]
                   and request.receipt["occupied_slots"] == list(range(8)) for request in requests)
        assert service.completed_requests == 8 and service.batch_count == 1
    finally:
        service.close()


def test_arrivals_during_forward_form_next_batch_and_keep_each_slot_response():
    service, evidence = controlled_gpu_loop()
    first = pending_request(service, 5, .5)
    service.queue.put_nowait(first)
    later, forwarded = [], []

    def forward(tokens, batch):
        service.queue.awaiting_dispatch = False
        forwarded.append([tokens[i*b.WIDTH] for i in range(batch)])
        if len(forwarded) == 1:
            # Independent clients arrive while the first batch is in flight.
            for slot, marker in ((7, .7), (1, .1), (6, .6)):
                request = pending_request(service, slot, marker)
                later.append(request)
                service.queue.put_nowait(request)
        return fake_forward(tokens, batch)

    service.forward = forward
    try:
        service._run()
        assert service.failure is None
        assert forwarded == [[0., 0., 0., 0., 0., .5, 0., 0.],
                             [0., .1, 0., 0., 0., 0., .6, .7]]
        assert [row["batch_id"] for row in evidence] == [1, 2]
        assert [row["slot"] for row in evidence[1]["requests"]] == [7, 1, 6]
        assert first.outputs == [[.5]*83] and first.receipt["batch_id"] == 1
        for request, marker in zip(later, (.7, .1, .6)):
            assert request.done.is_set() and request.outputs == [[marker]*83]
            assert request.receipt["batch_id"] == 2
            assert request.receipt["request_id"] == request.request_id
        assert service.completed_requests == 4 and service.batch_count == 2
    finally:
        service.close()


def test_expired_queued_member_fails_entire_batch_before_forward():
    service, evidence = controlled_gpu_loop()
    valid = pending_request(service, 1, .1)
    expired = pending_request(service, 7, .7, deadline=time.perf_counter()-1)
    service.queue.ready.extend([valid, expired])
    called = []
    service.forward = lambda *args: called.append(args)
    try:
        service._run()
        assert isinstance(service.failure, TimeoutError)
        assert "expired before dispatch" in str(service.failure)
        assert not called and not evidence and service.completed_requests == 0
        for request in (valid, expired):
            assert request.done.is_set() and request.error is service.failure
            assert request.outputs is None and request.receipt is None
        with pytest.raises(RuntimeError, match="no retry"):
            service.infer("slot1", [.1]*b.WIDTH, deadline=time.perf_counter()+1)
    finally:
        service.close()


@pytest.mark.parametrize("wait", [.001, -.001, float("inf")])
def test_new_service_rejects_any_nonzero_gather_configuration(wait):
    with pytest.raises(ValueError, match="zero intentional gather wait"):
        b.InferenceService(fake_forward, "gpu", max_wait_seconds=wait)
