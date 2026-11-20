"""Model-free tests of target-only transport, unchanged clocks and provenance."""
from contextlib import contextmanager
import io
import json
import time
from types import SimpleNamespace

import pytest

from tools.splendorust_gpu_collector_20261105 import adapters as a, backend as b
from tools.splendorust_gpu_throughput_20261105.adapters import ProxySearchOpponent


class Service:
    def __init__(self): self.clients = {}; self.pending = {}; self.receipts = []
    def register(self, client, slot):
        assert client not in self.clients
        self.clients[client] = slot
    def unregister(self, client): self.clients.pop(client)
    def infer(self, client, tokens, batch, *, deadline):
        assert client in self.clients and deadline > time.perf_counter()
        receipt = dict(client_id=client, slot=self.clients[client], deadline_perf=deadline)
        self.receipts.append(receipt)
        return [[0.] * 83 for _ in range(batch)], receipt


class Runtime:
    def __init__(self): self.service = Service(); self.operations = []
    @contextmanager
    def operation(self, client, **kwargs):
        self.operations.append(("start", client, kwargs))
        try: yield
        finally: self.operations.append(("end", client, kwargs))
    def release_adapter(self, client): self.service.unregister(client)


def bare_adapter():
    obj = a.CaptureProxySearchOpponent.__new__(a.CaptureProxySearchOpponent)
    obj.runtime = Runtime(); obj.client_id = "client"; obj.slot = 3
    obj.runtime.service.register(obj.client_id, obj.slot)
    obj.profile = dict(iterations=512); obj.active_deadline = None; obj.request_deadline = None
    obj.admission_guard = None; obj.capture = False; obj.callbacks = []; obj.capture_targets = True
    obj.last_service_calls = []; obj.inference_calls = obj.inference_rows = 0
    obj.deadline = None; obj._closed = False
    return obj


def test_accepted_sources_and_capture_binary_are_exact():
    proof = b.source_equivalence()
    assert proof["byte_identical_compilation"] and proof["observer"] is False
    provenance = b.capture_provenance()
    assert provenance["new_builds"] == provenance["models_run"] == 0
    assert provenance["algorithm_sources_unchanged"] and provenance["only_bridge_capture_added"]
    assert len(provenance["build_inputs"]) == 75


def test_inherited_search_inference_and_clock_functions_are_original():
    proof = a.source_equivalence()
    for name in proof["inherited_methods"]:
        assert getattr(a.CaptureProxySearchOpponent, name) is getattr(ProxySearchOpponent, name)


def test_capture_is_separate_and_inside_original_deadline(monkeypatch):
    obj = bare_adapter(); observer = {"sentinel": "lawful observer"}
    action = {"kind": "Pass"}; trace = dict(forced_pass=True, phases=[])
    result = (action, trace); original_deadline = time.perf_counter() + 50
    obj.request_deadline = original_deadline
    obj._choose_with_original_deadline = lambda value: result
    evidence = {"eligible": False, "policy_target81": None}
    def capture(*args):
        assert obj.active_deadline == original_deadline
        assert args == (observer, 512, None, None, action)
        assert obj.runtime.operations[-1][0] == "start"
        return evidence
    monkeypatch.setattr(a, "capture_record", capture)
    assert obj.choose(observer, diagnostic_capture=True) is result
    assert obj.last_training_capture is evidence and "eligible" not in trace
    assert obj.active_deadline is None and obj.capture is False
    assert obj.runtime.operations[-1][0] == "end"


@pytest.mark.parametrize("phase", ["search", "capture", "deadline"])
def test_failure_preserves_exception_and_finishes_same_clock(monkeypatch, phase):
    obj = bare_adapter(); failure = ValueError("saved target failure")
    def choose(_):
        if phase == "search": raise failure
        return ({"kind": "Pass"}, dict(forced_pass=True, phases=[]))
    def capture(*args):
        if phase == "capture": raise failure
        obj.active_deadline = time.perf_counter() - 1
        return {"eligible": False}
    obj._choose_with_original_deadline = choose
    monkeypatch.setattr(a, "capture_record", capture)
    with pytest.raises(TimeoutError if phase == "deadline" else ValueError) as caught:
        obj.choose({})
    if phase != "deadline": assert caught.value is failure
    assert obj.capture is False and obj.active_deadline is None
    assert obj.runtime.operations[-1][0] == "end"


def test_capture_wire_keeps_original_scientific_projection_and_callback_hashes():
    obj = bare_adapter(); obj.capture = True
    native = dict(enabled=True, callback_count=1, visits67=[512]+[0]*66)
    scientific = dict(kind="result", action={"kind": "Buy", "index": 0}, simulations=512)
    infer = dict(kind="infer", callback_index=0, callback_role="root", batch=1, tokens=[0.] * 1488)
    obj.proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(
        json.dumps(infer)+"\n"+json.dumps(dict(scientific, training_capture=native))+"\n"))
    request = dict(op="select", observation=dict(phase=dict(kind="Main")), legal=[])
    response = obj.exchange(request)
    assert {key: response[key] for key in scientific} == scientific
    assert "training_capture" not in response and obj._main_capture == native
    assert obj._root_callback == dict(tokens=infer["tokens"], outputs=[[0.] * 83])
    assert response["neural_callback_count"] == 1 and len(response["neural_callbacks"]) == 1
    assert obj.inference_calls == obj.inference_rows == 1
    sent = [json.loads(line) for line in obj.proc.stdin.getvalue().splitlines()]
    assert sent[0]["capture_targets"] is True and "capture_targets" not in request
    assert sent[1] == dict(kind="inference_result", outputs=[[0.] * 83])
    assert obj.active_deadline is None and obj.last_service_calls[0]["slot"] == 3


def test_failed_capture_constructor_unregisters_only_owned_client(monkeypatch):
    runtime = Runtime(); runtime.service.register("other", 0)
    error = ValueError("bad native READY")
    def failed(*args, **kwargs): raise error
    monkeypatch.setattr(a.CaptureOpponent, "__init__", failed)
    with pytest.raises(ValueError) as caught:
        a.CaptureProxySearchOpponent({}, runtime=runtime, client_id="new", slot=1, native_artifact={})
    assert caught.value is error and runtime.service.clients == {"other": 0}


def test_constructor_uses_qualified_capture_without_model(monkeypatch):
    runtime = Runtime(); marker = {"path": "authenticated native"}; calls = []
    def init(self, profile, **kwargs):
        assert kwargs["shared_from"].model is None
        calls.append((profile, kwargs["native_artifact"], kwargs["capture_targets"]))
    monkeypatch.setattr(a.CaptureOpponent, "__init__", init)
    profile = dict(iterations=512)
    obj = a.CaptureProxySearchOpponent(profile, runtime=runtime, client_id="one", slot=4, native_artifact=marker)
    assert calls == [(profile, marker, True)] and obj.slot == 4
    assert runtime.service.clients == {"one": 4}
