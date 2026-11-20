"""Synthetic IPC clocks only: no Splendor positions, models, or simulations."""
import ctypes
import json
import os
from pathlib import Path
import struct
import time

import psutil
import pytest

from tools.local_player_qualification_128 import clock


WORKER = r'''
import json, os, subprocess, sys, time

def emit(packet):
    sys.stdout.write(json.dumps(packet) + "\n")
    sys.stdout.flush()

if os.environ.get("CLOCK_TEST_INIT") == "hang":
    time.sleep(30)
if os.environ.get("CLOCK_TEST_INIT") == "serve":
    from tools.local_player_qualification_128.clock import serve
    def initialize():
        return {"count": 0}
    def handle(context, payload):
        context["count"] += 1
        return {"count": context["count"], "action": payload["action"]}
    raise SystemExit(serve(initialize, handle))
emit({"type": "ready", "metadata": {"pid": os.getpid()}})
if os.environ.get("CLOCK_TEST_INIT") == "preemptive":
    emit({"type": "response", "request_id": "request-00000001", "result": "unsolicited"})
count = 0
for line in sys.stdin:
    request = json.loads(line)
    payload = request["payload"]
    mode = payload.get("mode", "echo")
    count += 1
    if mode == "malformed":
        sys.stdout.write("this is not JSON\n")
        sys.stdout.flush()
        continue
    if mode == "child_hang":
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        with open(payload["child_pid_file"], "x") as handle:
            handle.write(str(child.pid))
        time.sleep(30)
    if mode == "hang":
        time.sleep(30)
    if mode == "late":
        time.sleep(payload["sleep"])
    emit({"type": "response",
          "request_id": "wrong" if mode == "wrong" else request["request_id"],
          "result": {"count": count, "action": payload.get("action", "legal"),
                     "echo": payload.get("echo", "")}})
'''


@pytest.fixture
def worker_factory(tmp_path):
    (tmp_path / "synthetic_clock_worker.py").write_text(WORKER, encoding="utf-8")
    instances = []
    counter = 0

    def create(mode="", **kwargs):
        nonlocal counter
        counter += 1
        env = {"PYTHONPATH": os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[1])]),
               "CLOCK_TEST_INIT": mode}
        kwargs.setdefault("initialization_timeout", 5)
        instance = clock.PersistentWorker("synthetic_clock_worker", tmp_path / f"run-{counter}",
                                          env=env, **kwargs)
        instances.append(instance)
        return instance

    yield create
    for instance in instances:
        instance.close()


def test_timely_persistent_worker_records_exclusive_receipts(worker_factory):
    worker = worker_factory("serve")
    first = worker.request({"action": "buy"}, 2, lambda result: result["action"] == "buy")
    second = worker.request({"action": "reserve"}, 2)
    assert first["accepted"] and second["accepted"]
    assert first["result"] == {"count": 1, "action": "buy"}
    assert second["result"] == {"count": 2, "action": "reserve"}
    assert 0 <= first["receipt_elapsed_seconds"] <= first["deadline_seconds"]
    assert first["telemetry"]["affinity"] == worker.affinity
    assert first["telemetry"]["request_cpu_seconds"] >= 0
    saved = json.loads((worker.output_dir / "request-00000001.json").read_text())
    assert saved == first
    assert "-B" in json.loads((worker.output_dir / "initialization.json").read_text())["command"]
    if os.name == "nt":
        assert psutil.Process(worker.process.pid).nice() == psutil.NORMAL_PRIORITY_CLASS
    with pytest.raises(FileExistsError):
        clock.PersistentWorker("synthetic_clock_worker", worker.output_dir)
    assert json.loads((worker.output_dir / "request-00000001.json").read_text()) == first


def test_late_and_hung_responses_never_reach_validator(worker_factory):
    for mode in ("late", "hang"):
        worker = worker_factory()
        calls = []
        result = worker.request({"mode": mode, "sleep": .6}, .15,
                                lambda value: calls.append(value))
        assert result["status"] == "timeout"
        assert not result["accepted"] and calls == []
        assert worker.process.poll() is not None
        with pytest.raises(RuntimeError, match="no automatic restart"):
            worker.request({}, 1)


@pytest.mark.parametrize("mode,status", [("malformed", "malformed"), ("wrong", "wrong_request")])
def test_bad_protocol_is_preserved_and_stops_worker(worker_factory, mode, status):
    worker = worker_factory()
    result = worker.request({"mode": mode}, 2)
    assert result["status"] == status and not result["accepted"]
    assert worker.process.poll() is not None
    assert json.loads((worker.output_dir / "request-00000001.json").read_text())["status"] == status


def test_illegal_action_is_not_a_timeout(worker_factory):
    worker = worker_factory()
    result = worker.request({"action": "illegal"}, 2, lambda value: value["action"] == "legal")
    assert result["status"] == "illegal"
    assert result["received_before_deadline"] and not result["accepted"]


def test_receipt_time_precedes_slow_validation(worker_factory):
    worker = worker_factory()

    def validate(result):
        time.sleep(.4)
        return result["action"] == "legal"

    result = worker.request({}, .3, validate)
    assert result["accepted"]
    assert result["receipt_elapsed_seconds"] < .3 < result["elapsed_seconds"]


def test_serialization_is_inside_clock(worker_factory, monkeypatch):
    worker = worker_factory()
    original = clock.json.dumps

    def delayed(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("type") == "request":
            time.sleep(.2)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(clock.json, "dumps", delayed)
    result = worker.request({}, .1)
    assert result["status"] == "timeout" and not result["accepted"]
    assert result["elapsed_seconds"] >= .2


def test_prequeued_response_cannot_predict_request_identity(worker_factory):
    worker = worker_factory("preemptive")
    deadline = time.monotonic() + 2
    while worker._queue.empty() and time.monotonic() < deadline:
        time.sleep(.005)
    assert not worker._queue.empty()
    result = worker.request({}, 2)
    assert result["status"] == "unsolicited"
    assert not result["accepted"] and not result["received_before_deadline"]


def test_large_echo_and_partial_pipe_writes(worker_factory):
    worker = worker_factory()
    original = worker.process.stdin

    class PartialWriter:
        def write(self, value):
            return original.write(value[:3071])

        def flush(self):
            original.flush()

        def close(self):
            original.close()

    worker.process.stdin = PartialWriter()
    payload = "0123456789abcdef" * 8192
    result = worker.request({"echo": payload}, 5)
    assert result["accepted"] and result["result"]["echo"] == payload
    assert (worker.output_dir / "stdout.log").stat().st_size > 64 * 1024


def test_oversized_response_and_invalid_request_have_receipts(worker_factory):
    worker = worker_factory(max_message_bytes=1024)
    result = worker.request({"echo": "x" * 4096}, 2)
    assert result["status"] == "malformed"
    other = worker_factory(max_request_bytes=128)
    invalid = other.request({"echo": "x" * 4096}, 2)
    assert invalid["status"] == "invalid_request"
    assert (other.output_dir / "request-00000001.json").exists()


def test_worker_and_descendant_are_terminated(worker_factory, tmp_path):
    worker = worker_factory()
    pid_file = tmp_path / "child.pid"
    parent = worker.process.pid
    result = worker.request({"mode": "child_hang", "child_pid_file": str(pid_file)}, .6)
    assert result["status"] == "timeout"
    assert pid_file.exists(), "Synthetic worker must spawn child before timing out"
    child = int(pid_file.read_text())
    assert not psutil.pid_exists(parent)
    assert not psutil.pid_exists(child)


def test_initialization_deadline_preserves_receipt_and_kills(worker_factory, tmp_path):
    with pytest.raises(clock.WorkerInitializationError) as error:
        worker_factory("hang", initialization_timeout=.2)
    assert error.value.receipt["status"] == "timeout"
    receipt = json.loads((tmp_path / "run-1" / "initialization.json").read_text())
    assert not psutil.pid_exists(receipt["identity"]["pid"])


def test_rss_guard_rejects_initialization(worker_factory, tmp_path):
    with pytest.raises(clock.WorkerInitializationError) as error:
        worker_factory(max_rss_bytes=1)
    assert error.value.receipt["status"] == "rss_limit"
    receipt = json.loads((tmp_path / "run-1" / "initialization.json").read_text())
    assert not psutil.pid_exists(receipt["identity"]["pid"])


def test_live_resource_guard_is_rate_limited_and_preserves_failure(worker_factory):
    calls = []
    failing = False

    def guard():
        calls.append(time.perf_counter())
        if failing:
            raise RuntimeError("Synthetic artifact budget exhausted")

    worker = worker_factory(resource_guard=guard)
    assert calls, "Initialization must invoke the parent resource guard"
    assert worker.request({}, 2)["accepted"]
    assert all(later - earlier >= 1.0 for earlier, later in zip(calls, calls[1:]))
    failing = True
    result = worker.request({"mode": "hang"}, 3)
    assert result["status"] == "resource_guard" and not result["accepted"]
    assert result["detail"] == "RuntimeError: Synthetic artifact budget exhausted"
    assert worker.process.poll() is not None
    assert all(later - earlier >= 1.0 for earlier, later in zip(calls, calls[1:]))
    saved = json.loads((worker.output_dir / "request-00000002.json").read_text())
    assert saved == result


def test_absolute_parent_deadline_shortens_request(worker_factory):
    worker = worker_factory()
    worker.absolute_deadline = time.monotonic() + .1
    result = worker.request({"mode": "hang"}, 5)
    assert result["status"] == "timeout"
    assert 0 < result["deadline_seconds"] <= .1


def test_bounded_log_stops_without_accepting_response(worker_factory):
    worker = worker_factory(max_log_bytes=1024)
    result = worker.request({"echo": "x" * 4096}, 2)
    assert result["status"] == "log_limit" and not result["accepted"]
    assert (worker.output_dir / "stdout.log").stat().st_size <= 1024


def test_windows_topology_record_layout_and_current_affinity():
    pointer_bytes = ctypes.sizeof(ctypes.c_size_t)

    def core(mask, efficiency, flags):
        size = 32 + pointer_bytes + 8
        value = bytearray(size)
        struct.pack_into("<II", value, 0, 0, size)
        value[8], value[9] = flags, efficiency
        struct.pack_into("<H", value, 30, 1)
        struct.pack_into("<Q" if pointer_bytes == 8 else "<I", value, 32, mask)
        return value

    parsed = clock._parse_processor_topology(core(3, 1, 1) + core(4, 0, 0))
    assert parsed == [
        {"core_index": 0, "flags": 1, "efficiency_class": 1, "logical_processors": [0, 1]},
        {"core_index": 1, "flags": 0, "efficiency_class": 0, "logical_processors": [2]},
    ]
    with pytest.raises(ValueError):
        clock._parse_processor_topology(core(3, 1, 1)[:-1])
    topology = clock.processor_topology()
    assert topology["selected_logical_processor"] in psutil.Process().cpu_affinity()
    if os.name == "nt":
        selected = topology["cores"][topology["selected_core_index"]]
        assert selected["efficiency_class"] == max(core["efficiency_class"] for core in topology["cores"])
