"""One unchanged model per route, direct concurrent CPU or fixed-slot GPU service.

Imports are model-free. The evaluator calls verify_runtime_sources once before
build_runtime and passes its receipt. GPU batching never creates a second
outstanding request from a search or changes that search's callback ordering.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 31 * 48
SLOTS = 8
MAX_WAIT_SECONDS = 0.0


def require(condition, message):
    if not condition:
        raise ValueError("GPU live service: " + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_input(tokens, batch):
    require(type(batch) is int and batch == 1, "original live native callback must have batch one")
    require(type(tokens) is list and len(tokens) == WIDTH and all(
        type(x) in (int, float) and math.isfinite(x) for x in tokens), "finite original public token shape")


def validate_output(rows, batch):
    require(type(rows) is list and len(rows) == batch and all(type(row) is list and len(row) == 83 and
        all(type(x) in (int, float) and math.isfinite(x) for x in row) for row in rows), "finite complete response shape")


@dataclass
class _Request:
    client_id: str
    request_id: str
    slot: int
    tokens: list
    submitted: float
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    outputs: object = None
    receipt: object = None
    error: object = None


@dataclass
class _PhysicalRequest:
    tokens: list
    deadline: float
    done: threading.Event = field(default_factory=threading.Event)
    outputs: object = None
    error: object = None


class InferenceService:
    """A synchronous client API: CPU direct; GPU one immutable row per client."""
    def __init__(self, forward, route, *, max_wait_seconds=MAX_WAIT_SECONDS, evidence=lambda row: None):
        require(route in ("cpu", "gpu") and max_wait_seconds == MAX_WAIT_SECONDS, "fixed backend and zero intentional gather wait")
        self.forward, self.route, self.evidence = forward, route, evidence
        self.max_wait_seconds = max_wait_seconds
        self.lock = threading.RLock(); self.clients = {}; self.sequences = {}; self.pending = {}
        self.failure = None; self.closed = False; self.batch_count = 0; self.completed_requests = 0
        self.physical_pending = None
        self.queue = queue.Queue(maxsize=SLOTS)
        self.thread = None
        if route == "gpu":
            self.thread = threading.Thread(target=self._run, name="fixed-eight-row-gpu", daemon=True)
            self.thread.start()

    def register(self, client_id, slot):
        with self.lock:
            self._available()
            require(self.physical_pending is None, "physical qualification cannot overlap client registration")
            require(type(client_id) is str and client_id and type(slot) is int and 0 <= slot < SLOTS,
                    "explicit client and permanent slot")
            require(client_id not in self.clients and slot not in self.clients.values(), "unique live client and slot")
            self.clients[client_id] = slot; self.sequences[client_id] = 0

    def unregister(self, client_id):
        with self.lock:
            require(client_id not in self.pending, "cannot release an outstanding client's slot")
            self.clients.pop(client_id, None)

    def _available(self):
        require(not self.closed, "service closed")
        if self.failure is not None:
            raise RuntimeError("Inference service failed; no retry") from self.failure

    def poison(self, error):
        with self.lock:
            if self.failure is None:
                self.failure = error
            for request in self.pending.values():
                if not request.done.is_set():
                    request.error = self.failure; request.done.set()
            if self.physical_pending is not None:
                self.physical_pending.error = self.failure; self.physical_pending.done.set()

    def physical_forward(self, tokens, *, deadline):
        """Qualification only, on the same GPU dispatch thread as live traffic."""
        require(type(tokens) is list and len(tokens) == SLOTS * WIDTH and all(
            type(x) in (int, float) and math.isfinite(x) for x in tokens), "fixed physical qualification input")
        now = time.perf_counter()
        require(type(deadline) in (int, float) and math.isfinite(deadline) and now < deadline <= now+9.001,
                "finite unexpired physical qualification deadline")
        with self.lock:
            self._available()
            require(not self.clients and not self.pending and self.physical_pending is None, "no clients during physical qualification")
            request = _PhysicalRequest(list(tokens), deadline); self.physical_pending = request
        try:
            if self.route == "cpu":
                request.outputs = self.forward(tokens, 8); request.done.set()
            else:
                self.queue.put_nowait(request)
                if not request.done.wait(max(0., deadline-time.perf_counter())):
                    raise TimeoutError("Physical qualification inference deadline")
            if request.error is not None:
                raise RuntimeError("Physical qualification inference failed") from request.error
            if time.perf_counter() >= deadline:
                raise TimeoutError("Late physical qualification inference")
            validate_output(request.outputs, 8)
            return request.outputs
        except BaseException as error:
            self.poison(error); raise
        finally:
            with self.lock:
                self.physical_pending = None

    def infer(self, client_id, tokens, batch=1, *, deadline):
        validate_input(tokens, batch)
        now = time.perf_counter()
        require(type(deadline) in (int, float) and math.isfinite(deadline) and now < deadline <= now + 9.001,
                "finite absolute remaining internal deadline")
        with self.lock:
            self._available()
            require(client_id in self.clients and client_id not in self.pending, "one outstanding callback per registered client")
            self.sequences[client_id] += 1
            request = _Request(client_id, client_id + ":" + str(self.sequences[client_id]), self.clients[client_id],
                               list(tokens), now, deadline)
            self.pending[client_id] = request
        try:
            if self.route == "cpu":
                self._dispatch([request])
            else:
                self.queue.put_nowait(request)
                if not request.done.wait(max(0., deadline - time.perf_counter())):
                    raise TimeoutError("GPU inference callback exceeded original 9s decision deadline")
            if request.error is not None:
                raise RuntimeError("Inference request failed: " + request.request_id) from request.error
            if time.perf_counter() >= deadline:
                raise TimeoutError("Inference complete readback arrived after decision deadline")
            validate_output(request.outputs, 1)
            require(request.receipt["request_id"] == request.request_id and request.receipt["client_id"] == client_id and
                    request.receipt["slot"] == request.slot, "response routing identity")
            receipt = dict(request.receipt, client_received_perf=time.perf_counter())
            return request.outputs, receipt
        except BaseException as error:
            self.poison(error)
            raise
        finally:
            with self.lock:
                self.pending.pop(client_id, None)

    def _dispatch(self, requests):
        started = time.perf_counter()
        require(requests and len({x.slot for x in requests}) == len(requests), "independent unique slots per dispatch")
        if any(started >= x.deadline for x in requests):
            raise TimeoutError("Queued inference expired before dispatch")
        if self.route == "gpu":
            tokens = [0.] * (SLOTS * WIDTH)
            for request in requests:
                tokens[request.slot * WIDTH:(request.slot + 1) * WIDTH] = request.tokens
            physical_batch = SLOTS
        else:
            require(len(requests) == 1, "CPU direct dispatch preserves each caller")
            tokens, physical_batch = requests[0].tokens, 1
        outputs = self.forward(tokens, physical_batch)
        finished = time.perf_counter(); validate_output(outputs, physical_batch)
        if any(finished >= x.deadline for x in requests):
            raise TimeoutError("Completed inference exceeded original callback deadline")
        with self.lock:
            self._available(); self.batch_count += 1; batch_id = self.batch_count
        receipts = []
        for request in requests:
            row = outputs[request.slot if self.route == "gpu" else 0]
            receipt = dict(client_id=request.client_id, request_id=request.request_id, slot=request.slot,
                route=self.route, native_batch=1, physical_batch=physical_batch, batch_id=batch_id,
                occupied_slots=sorted(x.slot for x in requests), submitted_perf=request.submitted,
                dispatch_perf=started, complete_readback_perf=finished, deadline_perf=request.deadline,
                queue_seconds=started-request.submitted, forward_roundtrip_seconds=finished-started,
                input_sha256=digest(request.tokens), output_sha256=digest([row]))
            receipts.append(receipt)
            request.outputs = [list(row)]; request.receipt = receipt
        self.evidence(dict(kind="inference_batch", batch_id=batch_id, route=self.route,
                           physical_batch=physical_batch, requests=receipts))
        with self.lock:
            self.completed_requests += len(requests)
            for request in requests:
                request.done.set()

    def _run(self):
        try:
            while True:
                with self.lock:
                    if self.closed or self.failure is not None:
                        return
                try:
                    first = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                if first is None:
                    return
                if isinstance(first, _PhysicalRequest):
                    if time.perf_counter() >= first.deadline:
                        raise TimeoutError("Physical qualification expired before GPU dispatch")
                    first.outputs = self.forward(first.tokens, 8)
                    validate_output(first.outputs, 8)
                    if time.perf_counter() >= first.deadline:
                        raise TimeoutError("Physical qualification expired after GPU readback")
                    first.done.set()
                    continue
                requests = [first]
                while len(requests) < SLOTS:
                    try:
                        # Coalesce only requests already ready. Never wait for
                        # companions after receiving the first request.
                        request = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    if request is None:
                        return
                    requests.append(request)
                self._dispatch(requests)
        except BaseException as error:
            self.poison(error)

    def close(self):
        with self.lock:
            self.closed = True
            for request in self.pending.values():
                if not request.done.is_set():
                    request.error = RuntimeError("Service closed with outstanding inference"); request.done.set()
            if self.physical_pending is not None:
                self.physical_pending.error = RuntimeError("Service closed during physical qualification")
                self.physical_pending.done.set()
        if self.thread is not None:
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
            self.thread.join(timeout=5.)
            require(not self.thread.is_alive(), "GPU service did not stop; owner must terminate worker")


class Runtime:
    def __init__(self, plan, route, folder, runtime_sources, raw_forward, finish, identity):
        self.plan, self.route, self.folder = plan, route, Path(folder)
        self.runtime_sources = dict(runtime_sources)
        self.raw_forward, self.finish, self.identity = raw_forward, finish, identity
        self.lock = threading.RLock(); self.adapters = {}; self.active = {}; self.operation_counter = 0
        self.journal_lock = threading.Lock(); self.closed = False
        self.folder.mkdir(parents=True, exist_ok=False)
        self.journal = (self.folder / "inference.jsonl").open("x", encoding="utf-8", newline="\n")
        import psutil
        self.process_identity = dict(pid=os.getpid(), created=psutil.Process().create_time())
        self._publish_active()
        self.service = InferenceService(raw_forward, route, evidence=self._record)

    def _record(self, value):
        with self.journal_lock:
            self.journal.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            self.journal.flush()

    def _publish_active(self):
        value = dict(schema="gpu_live_active_decisions_v1", identity=self.process_identity, pid=self.process_identity["pid"],
                     route=self.route, updated_unix=time.time(), active=list(self.active.values()))
        destination = self.folder / "active_decisions.json"
        temporary = self.folder / "active_decisions.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        from .status_io import atomic_replace_status
        atomic_replace_status(temporary, destination)

    @contextmanager
    def operation(self, client_id, *, started=None, kind="decision"):
        begun = time.perf_counter() if started is None else started
        now, wall = time.perf_counter(), time.time()
        require(type(begun) in (int, float) and math.isfinite(begun) and begun <= now and now-begun < 10., "whole-operation clock origin")
        thread_id = threading.get_ident(); owned = False
        with self.lock:
            if client_id in self.active:
                require(self.active[client_id]["thread_id"] == thread_id, "no concurrent operation on one native client")
            else:
                self.operation_counter += 1; owned = True
                self.active[client_id] = dict(client_id=client_id, operation_id=self.operation_counter,
                    kind=kind, thread_id=thread_id, started_perf=begun, started_unix=wall-(now-begun),
                    deadline_perf=begun+10., deadline_unix=wall+(begun+10.-now))
                self._publish_active()
        try:
            yield self.active[client_id]
        finally:
            entry = self.active[client_id]
            expired = time.perf_counter() >= entry["deadline_perf"] or time.time() >= entry["deadline_unix"]
            if owned:
                with self.lock:
                    self.active.pop(client_id, None); self._publish_active()
            expired = expired or time.perf_counter() >= entry["deadline_perf"] or time.time() >= entry["deadline_unix"]
            if expired:
                raise TimeoutError("Whole completed operation exceeded dual-clock10s bound")

    def make_adapter(self, client_id, slot, quota):
        from tools.splendorust_gpu_probe_20261105.worker import verify_binding
        from .adapters import ProxySearchOpponent
        require(type(quota) is int and quota in (16, 512), "original16/512 quota")
        source = verify_binding(self.plan["profiles"][str(quota)])
        profile = json.loads(source.read_text(encoding="utf-8"))
        require(profile["iterations"] == quota and profile["world_pool"] == 3 and profile["cpuct"] == .4 and
                profile["depth"] == 64 and profile["internal_budget_seconds"] == 9. and
                profile["inference_batch_size"] == 1, "unchanged scientific native profile")
        with self.lock:
            require(client_id not in self.adapters, "unique adapter lifetime")
            with self.operation(client_id, kind="native_startup"):
                adapter = ProxySearchOpponent(profile, runtime=self, client_id=client_id, slot=slot)
            self.adapters[client_id] = adapter
            return adapter

    def infer_raw(self, tokens, batch):
        with self.service.lock:
            require(not self.service.pending, "raw qualification only outside live outstanding requests")
        return self.raw_forward(tokens, batch)

    def fixed_batch_infer(self, tokens8):
        return self.service.physical_forward(tokens8, deadline=time.perf_counter()+9.)

    def release_adapter(self, client_id):
        self.service.unregister(client_id)
        with self.lock:
            self.adapters.pop(client_id, None)

    def abort(self):
        self.service.poison(RuntimeError("Evaluator aborted; no replacement or retry"))
        for adapter in list(self.adapters.values()):
            adapter.close()

    def close(self):
        if self.closed:
            return
        try:
            for adapter in list(self.adapters.values()):
                adapter.close()
            self.service.close()
            self.finish()
        finally:
            with self.journal_lock:
                self.journal.flush(); os.fsync(self.journal.fileno()); self.journal.close()
            self.closed = True


def build_runtime(plan, route, folder, *, runtime_sources):
    """Called only after the evaluator's startup guard and prospective freeze."""
    from tools import future_training_rules_20261021 as guard
    from tools.splendorust_gpu_probe_20261105 import worker as probe
    require(runtime_sources == guard.SOURCE_HASHES and route in ("cpu", "gpu"), "actual evaluator startup receipt and route")
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    require(str(torch.__version__) == plan["torch_version_expected"] == "2.4.1+cpu" and
            importlib.metadata.version("torch-directml") == plan["directml_version_expected"] == "0.2.5.dev240914" and
            torch.get_num_threads() == torch.get_num_interop_threads() == 1 and
            torch.backends.mha.get_fastpath_enabled() and torch.backends.mkldnn.enabled, "unchanged accepted runtime")
    bundle = probe.build_cpu_model(plan["checkpoint"], plan["model_source"])
    model_lock = threading.Lock(); warning_records = []
    def checked(operation):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                result = operation()
            finally:
                messages = [dict(category=x.category.__name__, message=str(x.message)) for x in captured]
                warning_records.extend(messages)
        require(not any(probe.is_backend_fallback(x["message"]) for x in messages),
                "unsupported GPU operation/CPU fallback: " + json.dumps(messages))
        return result
    device = None; gpu_model = None
    with torch.inference_mode():
        if route == "cpu":
            compiled = torch.jit.freeze(torch.jit.trace(bundle["eager"], torch.zeros((1, 31, 48), dtype=torch.float32),
                check_trace=False, strict=True), optimize_numerics=False)
            device_identity = dict(torch_device="cpu", implementation="original_frozen_torchscript")
        else:
            import torch_directml
            device, device_identity = checked(lambda: probe.select_device(torch_directml, plan["device_name_contains"]))
            require(device.type == "privateuseone" and device.index == device_identity["selected"]["index"], "selected GPU actual device")
            gpu_model = checked(lambda: copy.deepcopy(bundle["model"]).to(device))
            require(all(t.device == device for t in gpu_model.state_dict().values()) and
                    checked(lambda: probe.state_digest(gpu_model, torch)) == probe.WEIGHTS_SHA256, "original GPU float32 weight bytes")
            compiled = probe.make_primitive_forward(torch, gpu_model)
    def raw_forward(tokens, batch):
        require(type(batch) is int and (batch in (1, 8) if route == "cpu" else batch == 8) and
                type(tokens) is list and len(tokens) == batch * WIDTH and all(type(x) in (int, float) and math.isfinite(x) for x in tokens),
                "physical inference shape and finite input")
        # Inference mode is thread-local: every CPU caller and GPU dispatch enters it.
        with torch.inference_mode():
            x = torch.tensor(tokens, dtype=torch.float32).reshape(batch, 31, 48)
            if route == "cpu":
                result = probe.output_to_host(compiled(x), batch, torch).tolist()
            else:
                with model_lock:
                    result = checked(lambda: probe.gpu_roundtrip(compiled, x, batch, device, torch)).tolist()
            validate_output(result, batch)
            return result
    def finish():
        require(probe.state_digest(bundle["model"], torch) == probe.WEIGHTS_SHA256, "CPU original weights unchanged")
        if gpu_model is not None:
            require(checked(lambda: probe.state_digest(gpu_model, torch)) == probe.WEIGHTS_SHA256, "GPU original weights unchanged")
        probe.save_json(Path(folder) / "runtime_final.json", dict(route=route,
            unchanged_weights_sha256=probe.WEIGHTS_SHA256, warnings=warning_records,
            service_batches=runtime.service.batch_count, service_requests=runtime.service.completed_requests,
            service_failure=None if runtime.service.failure is None else repr(runtime.service.failure)))
    identity = dict(route=route, checkpoint_sha256=probe.PARENT_SHA256, weights_sha256=probe.WEIGHTS_SHA256,
        gpu_implementation=probe.GPU_IMPLEMENTATION if route == "gpu" else None, device=device_identity,
        threads=1, interop_threads=1, cpu_dispatch="concurrent calling game threads, shared immutable compiled model",
        gpu_dispatch="one model thread; fixed eight permanent rows; zero unused rows; immediate dispatch after draining ready requests; no timed companion wait",
        numerical_implementation_changed=route == "gpu", source_profile_execution_policy_not_adopted=True,
        warnings=warning_records, ruleset_id=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
        turn_contract=guard.TURN_CONTRACT, runtime_sources=runtime_sources)
    runtime = Runtime(plan, route, folder, runtime_sources, raw_forward, finish, identity)
    probe.save_json(Path(folder) / "runtime_identity.json", identity)
    return runtime
