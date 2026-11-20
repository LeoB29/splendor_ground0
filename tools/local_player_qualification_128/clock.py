"""Persistent, externally timed JSONL workers. No game or model code lives here."""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time

import psutil


THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


class WorkerInitializationError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("Worker initialization failed: " + receipt["status"])


def _decode(raw):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    value = json.loads(raw.decode("utf-8"), parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Protocol message must be an object")
    return value


def _new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _parse_processor_topology(raw):
    """Parse documented Windows PROCESSOR_RELATIONSHIP variable records."""
    cores, offset = [], 0
    pointer_bytes = ctypes.sizeof(ctypes.c_size_t)
    mask_format = "<Q" if pointer_bytes == 8 else "<I"
    affinity_size = pointer_bytes + 8
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("Truncated processor topology header")
        relationship, size = struct.unpack_from("<II", raw, offset)
        if size < 8 or offset + size > len(raw):
            raise ValueError("Malformed processor topology record")
        if relationship == 0:
            if size < 32:
                raise ValueError("Truncated processor core record")
            flags, efficiency = raw[offset + 8], raw[offset + 9]
            groups = struct.unpack_from("<H", raw, offset + 30)[0]
            if groups < 1 or 32 + affinity_size * groups > size:
                raise ValueError("Malformed processor affinity groups")
            logical = []
            for index in range(groups):
                start = offset + 32 + index * affinity_size
                mask = struct.unpack_from(mask_format, raw, start)[0]
                group = struct.unpack_from("<H", raw, start + pointer_bytes)[0]
                logical.extend(group * 64 + bit for bit in range(pointer_bytes * 8)
                               if mask & (1 << bit))
            if not logical:
                raise ValueError("Processor core has empty affinity")
            cores.append({"core_index": len(cores), "flags": flags,
                          "efficiency_class": efficiency, "logical_processors": logical})
        offset += size
    if not cores:
        raise ValueError("No physical processor cores found")
    return cores


def processor_topology():
    allowed = psutil.Process().cpu_affinity()
    if os.name != "nt":
        return {"api": "psutil_affinity_fallback", "cores": [],
                "allowed_logical_processors": allowed, "selected_logical_processor": min(allowed),
                "selection_rule": "lowest_allowed; physical performance class unavailable"}
    api = ctypes.WinDLL("kernel32", use_last_error=True).GetLogicalProcessorInformationEx
    api.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    api.restype = ctypes.c_int
    size = ctypes.c_ulong(0)
    api(0, None, ctypes.byref(size))
    if not size.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(size.value)
    if not api(0, buffer, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    cores = _parse_processor_topology(buffer.raw[:size.value])
    eligible = [core for core in cores if set(core["logical_processors"]) & set(allowed)]
    if not eligible:
        raise ValueError("No allowed processor core")
    chosen = max(eligible, key=lambda core: (core["efficiency_class"],
                 len(core["logical_processors"]), -min(set(core["logical_processors"]) & set(allowed))))
    return {"api": "GetLogicalProcessorInformationEx(RelationProcessorCore)", "cores": cores,
            "allowed_logical_processors": allowed,
            "selected_logical_processor": min(set(chosen["logical_processors"]) & set(allowed)),
            "selected_core_index": chosen["core_index"],
            "selection_rule": "highest efficiency class; then SMT width; then lowest allowed logical ID",
            "distinct_efficiency_classes": sorted({core["efficiency_class"] for core in cores})}


class _WindowsJob:
    """A kill-on-close Job Object owns descendants created by the worker."""
    def __init__(self, pid):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", w.DWORD), ("minimum_working_set", ctypes.c_size_t),
                        ("maximum_working_set", ctypes.c_size_t), ("active_processes", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
                        ("peak_job_memory", ctypes.c_size_t)]

        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.api.SetInformationJobObject.restype = w.BOOL
        self.api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.api.OpenProcess.restype = w.HANDLE
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.AssignProcessToJobObject.restype = w.BOOL
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.api.CloseHandle.restype = w.BOOL
        handle = self.api.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        process_handle = None
        try:
            if not self.api.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            process_handle = self.api.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process_handle or not self.api.AssignProcessToJobObject(handle, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = handle
        finally:
            if process_handle:
                self.api.CloseHandle(process_handle)
            if self.handle is None:
                self.api.CloseHandle(handle)

    def close(self):
        if self.handle is not None:
            self.api.CloseHandle(self.handle)
            self.handle = None


class PersistentWorker:
    def __init__(self, module, output_dir, *, module_args=(), affinity=None,
                 initialization_timeout=60.0, env=None, max_message_bytes=2 * 1024**2,
                 max_request_bytes=1024**2, max_log_bytes=32 * 1024**2,
                 max_rss_bytes=2 * 1024**3, absolute_deadline=None, resource_guard=None):
        if (not isinstance(module, str) or not module or
                not math.isfinite(initialization_timeout) or initialization_timeout <= 0):
            raise ValueError("Invalid worker initialization")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.max_message_bytes, self.max_request_bytes = max_message_bytes, max_request_bytes
        self.max_log_bytes, self.max_rss_bytes = max_log_bytes, max_rss_bytes
        self.absolute_deadline = absolute_deadline  # Parent time.monotonic() clock.
        self.resource_guard = resource_guard
        self._last_guard_sample = None
        self._queue = queue.Queue(maxsize=128)
        self._failure = None
        self._failure_detail = None
        self._closed = False
        self._counter = 0
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._logs = {name: (self.output_dir / name).open("xb")
                      for name in ("stdout.log", "stderr.log", "events.jsonl")}
        self._log_sizes = dict.fromkeys(self._logs, 0)
        self._threads = []
        self._job = None
        self.process = None
        self._ps = None
        self._peak_rss = 0
        self._last_telemetry = {}
        try:
            self.topology = processor_topology()
        except BaseException:
            self.close()
            raise
        self.affinity = list(affinity if affinity is not None else
                             [self.topology["selected_logical_processor"]])
        if len(self.affinity) != 1 or self.affinity[0] not in self.topology["allowed_logical_processors"]:
            self.close()
            raise ValueError("Worker must use one allowed logical processor")
        child_env = os.environ.copy()
        child_env.update(env or {})
        child_env.update({name: "1" for name in THREAD_ENV})
        child_env["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-B", "-u", "-m", module, *map(str, module_args)]
        started = time.perf_counter_ns()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, env=child_env, bufsize=0,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self._ps = psutil.Process(self.process.pid)
            self._job = _WindowsJob(self.process.pid)
            self._ps.cpu_affinity(self.affinity)
            if os.name == "nt":
                self._ps.nice(psutil.NORMAL_PRIORITY_CLASS)
            self.identity = {"pid": self.process.pid, "created": self._ps.create_time()}
            for function in (self._stdout_reader, self._stderr_reader):
                thread = threading.Thread(target=function, daemon=True)
                thread.start()
                self._threads.append(thread)
            receipt = self._receive(started, initialization_timeout, "ready", None, None)
            self.initialization = receipt
            _new_json(self.output_dir / "initialization.json", {**receipt, "command": command,
                      "identity": self.identity, "affinity": self.affinity, "topology": self.topology,
                      "thread_environment": {name: child_env[name] for name in THREAD_ENV}})
            if receipt["status"] != "ok":
                self.close()
                raise WorkerInitializationError(receipt)
        except BaseException:
            self.close()
            raise

    def _put(self, event):
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._failure = "message_flood"

    def _log(self, name, raw):
        with self._log_lock:
            available = self.max_log_bytes - self._log_sizes[name]
            handle = self._logs[name]
            if not handle.closed and available > 0:
                part = raw[:available]
                handle.write(part)
                handle.flush()
                self._log_sizes[name] += len(part)
            if len(raw) > available:
                self._failure = "log_limit"

    def _stdout_reader(self):
        try:
            while True:
                raw = self.process.stdout.readline(self.max_message_bytes + 1)
                received = time.perf_counter_ns()  # Trusted receipt, before decoding or validation.
                if not raw:
                    self._put(("eof", received, None))
                    return
                if len(raw) > self.max_message_bytes or not raw.endswith(b"\n"):
                    self._log("stdout.log", raw[:4096])
                    self._put(("malformed", received, "oversized or unterminated message"))
                    return
                self._log("stdout.log", raw)
                self._put(("line", received, raw))
        except (OSError, ValueError) as error:
            self._put(("reader_error", time.perf_counter_ns(), str(error)))

    def _stderr_reader(self):
        try:
            while True:
                raw = self.process.stderr.read(4096)
                if not raw:
                    return
                self._log("stderr.log", raw)
        except (OSError, ValueError):
            return

    def _sample(self):
        try:
            cpu = self._ps.cpu_times()
            rss = self._ps.memory_info().rss
            self._peak_rss = max(self._peak_rss, rss)
            self._last_telemetry = {"cpu_seconds": cpu.user + cpu.system, "rss_bytes": rss,
                "peak_sampled_rss_bytes": self._peak_rss, "affinity": self._ps.cpu_affinity(),
                "observed_threads": self._ps.num_threads()}
            if rss > self.max_rss_bytes:
                self._failure = "rss_limit"
            if self._last_telemetry["affinity"] != self.affinity:
                self._failure = "affinity_changed"
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return dict(self._last_telemetry)

    def _check_resource_guard(self):
        now = time.perf_counter()
        if (self.resource_guard is None or (self._last_guard_sample is not None and
                                           now - self._last_guard_sample < 1.0)):
            return
        self._last_guard_sample = now
        try:
            self.resource_guard()
        except Exception as error:
            self._failure = "resource_guard"
            self._failure_detail = f"{type(error).__name__}: {error}"[:4096]

    def _receive(self, started, timeout, expected_type, request_id, validate, before_cpu=None):
        deadline = started + int(timeout * 1e9)
        if self.absolute_deadline is not None:
            # Parent global clock can differ from QueryPerformanceCounter on older Python.
            remaining_global = self.absolute_deadline - time.monotonic()
            # Reserve one coarse-clock tick rather than accidentally grant extra time.
            remaining_global -= time.get_clock_info("monotonic").resolution
            deadline = min(deadline, time.perf_counter_ns() + int(remaining_global * 1e9))
        if before_cpu is None:
            before_cpu = self._sample().get("cpu_seconds", 0.)
        status, result, received, detail = "timeout", None, None, None
        while True:
            self._check_resource_guard()
            self._sample()
            if self._failure:
                status, detail = self._failure, self._failure_detail or "Worker resource or message bound reached"
                break
            remaining = (deadline - time.perf_counter_ns()) / 1e9
            try:
                kind, stamp, raw = self._queue.get(timeout=max(0., min(.025, remaining)))
            except queue.Empty:
                if remaining <= 0:
                    break
                continue
            received = stamp
            if stamp < started:
                status, detail = "unsolicited", "Message arrived before this request began"
                break
            if stamp > deadline:  # Never decode or validate a late packet.
                status = "timeout"
                break
            if kind != "line":
                status, detail = ("worker_exit" if kind == "eof" else kind), raw
                break
            try:
                packet = _decode(raw)
            except (ValueError, UnicodeError) as error:
                status, detail = "malformed", str(error)
                break
            if packet.get("type") == "error":
                status, detail = "worker_error", str(packet.get("error"))[:4096]
                break
            if packet.get("type") != expected_type:
                status, detail = "malformed", "Unexpected protocol message type"
                break
            if expected_type == "response" and packet.get("request_id") != request_id:
                status, detail = "wrong_request", "Response request identity differs"
                break
            if expected_type == "response" and "result" not in packet:
                status, detail = "malformed", "Response has no result"
                break
            result = packet.get("result") if expected_type == "response" else packet.get("metadata")
            if validate is not None:
                try:
                    valid = validate(result)
                    if valid is False:
                        raise ValueError("Response is illegal")
                except Exception as error:
                    status, detail = "illegal", str(error)[:4096]
                    break
            status = "ok"
            break
        telemetry = self._sample()
        if status == "ok" and self._failure:
            status, detail = self._failure, self._failure_detail or "Worker resource or message bound reached"
        telemetry["request_cpu_seconds"] = max(0., telemetry.get("cpu_seconds", before_cpu) - before_cpu)
        receipt = {"status": status, "accepted": status == "ok", "request_id": request_id,
            "result": result, "detail": detail, "deadline_seconds": (deadline - started) / 1e9,
            "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
            "receipt_elapsed_seconds": None if received is None else (received - started) / 1e9,
            "received_before_deadline": received is not None and started <= received <= deadline,
            "clock": "perf_counter_ns", "clock_implementation": time.get_clock_info("perf_counter").implementation,
            "telemetry": telemetry}
        return receipt

    def request(self, payload, timeout_seconds, validate=None):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent requests are forbidden")
        try:
            if self._closed or self.process.poll() is not None:
                raise RuntimeError("Worker is closed; no automatic restart")
            self._counter += 1
            request_id = f"request-{self._counter:08d}"
            started = time.perf_counter_ns()
            if not self._queue.empty():
                self._failure = "unsolicited"
            before_cpu = self._sample().get("cpu_seconds", 0.)
            try:
                encoded = (json.dumps({"type": "request", "request_id": request_id, "payload": payload},
                                      separators=(",", ":"), allow_nan=False) + "\n").encode()
                if len(encoded) > self.max_request_bytes:
                    raise ValueError("Request message ceiling")
            except (ValueError, TypeError, OverflowError) as error:
                self._failure = "invalid_request"
                receipt = self._receive(started, timeout_seconds, "response", request_id, None, before_cpu)
                receipt["detail"] = str(error)[:4096]
                _new_json(self.output_dir / f"{request_id}.json", receipt)
                self.close()
                return receipt

            def send():
                try:
                    remaining = memoryview(encoded)
                    while remaining:
                        written = self.process.stdin.write(remaining)
                        if written is None or written <= 0:
                            raise OSError("Worker input pipe made no progress")
                        remaining = remaining[written:]
                    self.process.stdin.flush()
                except (OSError, ValueError) as error:
                    self._put(("write_error", time.perf_counter_ns(), str(error)))

            writer = threading.Thread(target=send, daemon=True)
            writer.start()
            self._threads.append(writer)
            receipt = self._receive(started, timeout_seconds, "response", request_id, validate, before_cpu)
            event = {key: value for key, value in receipt.items() if key != "result"}
            self._log("events.jsonl", (json.dumps(event, allow_nan=False) + "\n").encode())
            if receipt["status"] == "ok" and self._failure:
                receipt.update(status=self._failure, accepted=False,
                               detail="Worker resource or message bound reached")
            _new_json(self.output_dir / f"{request_id}.json", receipt)
            if receipt["status"] != "ok" or self._failure:
                self.close()
            return receipt
        finally:
            self._lock.release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        children = []
        if self._ps is not None:
            try:
                children = self._ps.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
        if self._job is not None:
            self._job.close()
        for process in reversed(children):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        psutil.wait_procs(children, timeout=5)
        for thread in self._threads:
            thread.join(timeout=.25)
        if self.process is not None:
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
        with self._log_lock:
            for handle in self._logs.values():
                handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def serve(initialize, handle, *, max_request_bytes=1024**2):
    """Worker entry point: initialize() -> context; handle(context,payload) -> result."""
    for name in THREAD_ENV:
        os.environ[name] = "1"
    from threadpoolctl import threadpool_limits

    def emit(value):
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()

    with threadpool_limits(limits=1):
        try:
            context = initialize()
            if "torch" in sys.modules:
                sys.modules["torch"].set_num_threads(1)
            emit({"type": "ready", "metadata": {"pid": os.getpid(), "threads": 1}})
        except BaseException as error:
            emit({"type": "error", "error": repr(error)[:4096]})
            return 1
        while True:
            raw = sys.stdin.buffer.readline(max_request_bytes + 1)
            if not raw:
                return 0
            request_id = None
            try:
                if len(raw) > max_request_bytes or not raw.endswith(b"\n"):
                    raise ValueError("Request message ceiling")
                request = _decode(raw)
                request_id = request.get("request_id")
                if request.get("type") != "request" or not isinstance(request_id, str):
                    raise ValueError("Malformed request")
                result = handle(context, request["payload"])
                emit({"type": "response", "request_id": request_id, "result": result})
            except BaseException as error:
                emit({"type": "error", "request_id": request_id, "error": repr(error)[:4096]})
                return 1
