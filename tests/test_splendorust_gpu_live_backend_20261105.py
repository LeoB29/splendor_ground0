"""Model-free lifetime, watchdog and native-proxy boundary checks."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_live_20261105 import backend as b
from tools.splendorust_gpu_live_20261105 import adapters as a


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
