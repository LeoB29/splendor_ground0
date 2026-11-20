"""Actual Windows sharing and concurrent-reader regression; no model calls."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
import traceback

import pytest

from tools.splendorust_gpu_live_20261105_02 import status_io as s

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Actual Windows sharing contract")


def encoded(sequence):
    return json.dumps({"sequence": sequence, "same_sequence": sequence,
                       "payload": str(sequence) * 100}, sort_keys=True).encode("utf-8")


def test_old_normal_reader_reproduces_replace_permission_failure(tmp_path):
    destination, temporary = tmp_path / "active_decisions.json", tmp_path / "active_decisions.tmp"
    destination.write_bytes(encoded(1)); temporary.write_bytes(encoded(2))
    with destination.open("rb") as old_reader:
        with pytest.raises(PermissionError) as caught:
            os.replace(temporary, destination)
        assert caught.value.winerror in (5, 32)
        assert old_reader.read() == encoded(1)
        assert destination.read_bytes() == encoded(1)
        assert temporary.read_bytes() == encoded(2)
    os.replace(temporary, destination)
    assert s.read_json(destination)["sequence"] == 2


def test_locked_reader_delays_publisher_then_releases_consistent_status(tmp_path):
    destination, temporary = tmp_path / "active_decisions.json", tmp_path / "active_decisions.tmp"
    destination.write_bytes(encoded(1)); temporary.write_bytes(encoded(2))
    entered = threading.Event()
    def publish():
        entered.set(); s.atomic_replace_status(temporary, destination)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with s.open_status_read(destination) as old_reader:
            future = executor.submit(publish); assert entered.wait(1)
            time.sleep(.03)
            assert not future.done()
            assert old_reader.read() == encoded(1)
        future.result(timeout=1)
    assert s.read_status_json(destination)["sequence"] == 2
    assert temporary.read_bytes() == encoded(2)


def test_shared_handle_closes_on_body_error(tmp_path):
    destination, temporary = tmp_path / "active.json", tmp_path / "replacement.json"
    destination.write_bytes(encoded(1))
    with pytest.raises(RuntimeError, match="body error"):
        with s.open_shared_read(destination) as reader:
            raise RuntimeError("body error")
    assert reader.closed
    temporary.write_bytes(encoded(2)); os.replace(temporary, destination)
    assert s.read_json(destination)["sequence"] == 2


def test_read_bounds_and_parse_errors_are_not_cached_or_retried(tmp_path):
    path = tmp_path / "status.json"; path.write_bytes(encoded(1))
    assert s.read_json(path)["sequence"] == 1
    path.write_bytes(b"{broken")
    with pytest.raises(json.JSONDecodeError):
        s.read_json(path)
    path.write_bytes(b"{\"x\":\"" + b"x" * s.MAX_STATUS_BYTES + b"\"}")
    with pytest.raises(ValueError, match="exceeds"):
        s.read_json(path)
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="object"):
        s.read_json(path)
    path.write_bytes(b'{"x":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        s.read_json(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        s.read_json(path)


@pytest.mark.parametrize("bound", [True, 0, -1, 65537, 2.5])
def test_invalid_bounds_rejected_before_open(tmp_path, bound):
    with pytest.raises(ValueError, match="bound"):
        s.read_json(tmp_path / "missing.json", max_bytes=bound)


def test_concurrent_atomic_publication_and_shared_reads(tmp_path):
    destination, temporary = tmp_path / "active_decisions.json", tmp_path / "active_decisions.tmp"
    destination.write_bytes(encoded(0))
    barrier = threading.Barrier(3); failures = []; lock = threading.Lock()
    observed = [[] for _ in range(2)]
    def reader(index):
        try:
            barrier.wait(timeout=5)
            for _ in range(1200):
                value = s.read_json(destination)
                assert value["sequence"] == value["same_sequence"]
                assert value["payload"] == str(value["sequence"]) * 100
                assert 0 <= value["sequence"] <= 200
                observed[index].append(value["sequence"])
        except BaseException:
            with lock:
                failures.append(traceback.format_exc())
    def publisher():
        try:
            barrier.wait(timeout=5)
            for sequence in range(1, 201):
                # Prepare complete bytes before bounded locked in-place
                # publication. There is no rename, retry or clock reset.
                with temporary.open("wb") as stream:
                    stream.write(encoded(sequence)); stream.flush(); os.fsync(stream.fileno())
                s.atomic_replace_status(temporary, destination)
        except BaseException:
            with lock:
                failures.append(traceback.format_exc())
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(reader, index) for index in range(2)]
        futures.append(executor.submit(publisher))
        for future in futures:
            future.result(timeout=30)
    receipt = dict(publishes=200, reader_count=2, reads_per_reader=[len(x) for x in observed],
                   final=s.read_json(destination), failures=failures)
    (tmp_path / "stress_receipt.json").write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    assert not failures, failures
    assert receipt["reads_per_reader"] == [1200] * 2
    assert receipt["final"]["sequence"] == 200


def test_initial_publication_and_existing_destination_share_same_api(tmp_path):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.tmp"
    temporary.write_bytes(encoded(0)); s.atomic_replace_status(temporary, destination)
    assert s.read_status_json(destination)["sequence"] == 0
    temporary.write_bytes(encoded(1)); s.atomic_replace_status(temporary, destination)
    assert s.read_status_json(destination)["sequence"] == 1
    assert temporary.read_bytes() == encoded(1)


def test_normal_external_read_handle_does_not_block_in_place_publication(tmp_path):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.tmp"
    destination.write_bytes(encoded(0)); temporary.write_bytes(encoded(1))
    with destination.open("rb"):
        s.atomic_replace_status(temporary, destination)
    assert s.read_status_json(destination)["sequence"] == 1
    assert temporary.read_bytes() == encoded(1)


def test_actual_runtime_registry_publication_during_held_shared_read(tmp_path):
    from tools.splendorust_gpu_live_20261105_02 import backend, control
    runtime = backend.Runtime({}, "cpu", tmp_path / "runtime", {},
                              lambda *_: pytest.fail("No model inference"), lambda: None, {})
    try:
        registry = runtime.folder / "active_decisions.json"
        entered = threading.Event()
        def publish_operation():
            entered.set()
            with runtime.operation("test", kind="model_free_windows_registry"):
                value = s.read_status_json(registry)
                assert value["active"][0]["client_id"] == "test"
                control.check_active(value, runtime.process_identity, time.time(), time.perf_counter())
        with ThreadPoolExecutor(max_workers=1) as executor:
            with s.open_status_read(registry) as old_reader:
                future = executor.submit(publish_operation); assert entered.wait(1)
                time.sleep(.03); assert not future.done()
                assert json.loads(old_reader.read())["active"] == []
            future.result(timeout=1)
        assert s.read_status_json(registry)["active"] == []
    finally:
        runtime.close()


def test_mutex_timeout_is_bounded_and_does_not_release_other_owner(tmp_path):
    path = tmp_path / "status.json"; errors = []
    def contender():
        started = time.perf_counter()
        try:
            with s.status_mutex(path):
                pytest.fail("Other thread owns the mutex")
        except BaseException as error:
            errors.append((error, time.perf_counter()-started))
    with s.status_mutex(path):
        thread = threading.Thread(target=contender); thread.start(); thread.join(1)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0][0], TimeoutError)
        assert .20 <= errors[0][1] < .75
    # The timed-out caller neither released the other owner's lock nor leaked
    # its handle; acquisition succeeds once the actual owner has released.
    with s.status_mutex(path):
        pass


def test_mutex_releases_after_body_error_and_normalizes_path(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    monkeypatch.chdir(tmp_path)
    assert s.mutex_name(path) == s.mutex_name("status.json") == s.mutex_name(str(path).upper())
    with pytest.raises(RuntimeError, match="body"):
        with s.status_mutex(path):
            raise RuntimeError("body")
    result = []
    def next_owner():
        with s.status_mutex(path):
            result.append("acquired")
    thread = threading.Thread(target=next_owner); thread.start(); thread.join(1)
    assert not thread.is_alive() and result == ["acquired"]


def test_abandoned_crossprocess_mutex_fails_and_releases_acquired_ownership(tmp_path):
    import ctypes
    from ctypes import wintypes
    import subprocess
    import sys
    path = tmp_path / "status.json"
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateMutexW
    create.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    create.restype = wintypes.HANDLE
    close = kernel32.CloseHandle; close.argtypes = (wintypes.HANDLE,); close.restype = wintypes.BOOL
    # Keep the named object alive after the child owner exits without releasing.
    retained = create(None, False, s.mutex_name(path)); assert retained
    script = ("import os,sys; from tools.splendorust_gpu_live_20261105_02.status_io import status_mutex\n"
              "with status_mutex(sys.argv[1]): os._exit(0)\n")
    try:
        result = subprocess.run([sys.executable, "-B", "-c", script, str(path)], timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        assert result.returncode == 0
        with pytest.raises(RuntimeError, match="abandoned"):
            with s.status_mutex(path):
                pytest.fail("An abandoned owner must not produce a healthy read")
        # WAIT_ABANDONED grants ownership: the failing helper must release it.
        acquired = []
        def next_owner():
            with s.status_mutex(path):
                acquired.append(True)
        thread = threading.Thread(target=next_owner); thread.start(); thread.join(1)
        assert not thread.is_alive() and acquired == [True]
    finally:
        close(retained)


def test_actual_crossprocess_two_readers_thousand_publications(tmp_path):
    import subprocess
    import sys
    destination = tmp_path / "active_decisions.json"; destination.write_bytes(encoded(0))
    script = r'''
import json,os,sys,time,traceback
from pathlib import Path
from tools.splendorust_gpu_live_20261105_02.status_io import read_status_json,atomic_replace_status
folder=Path(sys.argv[1]);role=sys.argv[2];path=folder/'active_decisions.json'
started=time.monotonic();count=0;last=0;error=None
try:
    while not (folder/'start').exists():
        if time.monotonic()-started>10:raise TimeoutError('test start barrier')
        time.sleep(.001)
    if role=='writer':
        temporary=folder/'active_decisions.tmp'
        for value in range(1,1001):
            payload=json.dumps(dict(sequence=value,same_sequence=value,payload=str(value)*100),sort_keys=True).encode()
            with temporary.open('wb') as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            atomic_replace_status(temporary,path);count+=1;last=value
    else:
        while not (folder/'done').exists() or count<100:
            value=read_status_json(path);sequence=value['sequence']
            assert last<=sequence<=1000 and sequence==value['same_sequence']
            assert value['payload']==str(sequence)*100
            last=sequence;count+=1
            if time.monotonic()-started>40:raise TimeoutError('bounded test stress')
except BaseException:error=traceback.format_exc()
finally:
    if role=='writer':(folder/'done').write_text('completed')
    (folder/(role+'_receipt.json')).write_text(json.dumps(dict(pid=os.getpid(),role=role,count=count,last=last,error=error)))
if error:sys.exit(1)
'''
    processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(tmp_path), role],
                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                 for role in ("reader0", "reader1", "writer")]
    try:
        (tmp_path / "start").write_text("start", encoding="utf-8")
        exits = [process.wait(timeout=45) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
    receipts = [json.loads((tmp_path / (role+"_receipt.json")).read_text()) for role in ("reader0", "reader1", "writer")]
    assert exits == [0, 0, 0], receipts
    assert len({item["pid"] for item in receipts}) == 3
    assert receipts[2]["count"] == receipts[2]["last"] == 1000
    assert all(item["count"] >= 100 and item["error"] is None for item in receipts)
    assert s.read_status_json(destination)["sequence"] == 1000


def test_writer_only_thousand_publications_without_any_rename(tmp_path, monkeypatch):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.tmp"
    def forbidden(*_args, **_kwargs):
        pytest.fail("In-place status publication cannot rename or unlink")
    monkeypatch.setattr(os, "replace", forbidden)
    monkeypatch.setattr(os, "rename", forbidden)
    monkeypatch.setattr(os, "unlink", forbidden)
    completed = 0; error = None
    try:
        for sequence in range(1, 1001):
            with temporary.open("wb") as stream:
                stream.write(encoded(sequence)); stream.flush(); os.fsync(stream.fileno())
            s.atomic_replace_status(temporary, destination)
            completed = sequence
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        (tmp_path / "writer_only_receipt.json").write_text(json.dumps(dict(completed=completed, error=error)), encoding="utf-8")
    assert s.read_status_json(destination)["sequence"] == 1000
    assert temporary.read_bytes() == encoded(1000)


@pytest.mark.parametrize("payload", [b"{broken", b'{"x":NaN}', b"[]", b"x" * (s.MAX_STATUS_BYTES+1)],
                         ids=["malformed", "nonfinite", "array", "oversized"])
def test_invalid_or_oversized_temporary_does_not_truncate_current_status(tmp_path, payload):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.tmp"
    destination.write_bytes(encoded(1)); temporary.write_bytes(payload)
    with pytest.raises(ValueError):
        s.atomic_replace_status(temporary, destination)
    assert destination.read_bytes() == encoded(1)


def test_publication_fsync_error_propagates_without_retry(tmp_path, monkeypatch):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.tmp"
    destination.write_bytes(encoded(1)); temporary.write_bytes(encoded(2)); calls = []
    def failed_sync(descriptor):
        calls.append(descriptor); raise OSError("deliberate status fsync failure")
    monkeypatch.setattr(os, "fsync", failed_sync)
    with pytest.raises(OSError, match="deliberate"):
        s.atomic_replace_status(temporary, destination)
    assert len(calls) == 1 and temporary.exists()


def test_reader_sees_complete_fresh_status_while_publisher_fsync_is_blocked(tmp_path, monkeypatch):
    destination, temporary = tmp_path / "status.json", tmp_path / "status.tmp"
    destination.write_bytes(encoded(1)); temporary.write_bytes(encoded(2))
    entered, release = threading.Event(), threading.Event()
    original_fsync = os.fsync
    def blocked_sync(descriptor):
        entered.set(); assert release.wait(2)
        original_fsync(descriptor)
    monkeypatch.setattr(os, "fsync", blocked_sync)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(s.atomic_replace_status, temporary, destination)
        try:
            assert entered.wait(1)
            assert not future.done()
            assert s.read_status_json(destination)["sequence"] == 2
            assert not future.done()  # No success return before durable sync.
        finally:
            release.set()
        future.result(timeout=2)


@pytest.mark.parametrize("clock", ["perf", "unix"])
def test_runtime_counts_final_publication_in_both_deadlines(tmp_path, monkeypatch, clock):
    from tools.splendorust_gpu_live_20261105_02 import backend
    runtime = backend.Runtime({}, "cpu", tmp_path / "runtime", {},
                              lambda *_: pytest.fail("No model inference"), lambda: None, {})
    original_publish = runtime._publish_active
    original_perf, original_unix = time.perf_counter, time.time
    frozen_perf, frozen_unix = original_perf(), original_unix()
    current = {"perf": frozen_perf, "unix": frozen_unix}
    monkeypatch.setattr(time, "perf_counter", lambda: current["perf"])
    monkeypatch.setattr(time, "time", lambda: current["unix"])
    def completing_publication():
        original_publish()
        if not runtime.active:
            current[clock] += .002
    monkeypatch.setattr(runtime, "_publish_active", completing_publication)
    try:
        with pytest.raises(TimeoutError, match="dual-clock"):
            with runtime.operation("test") as operation:
                current[clock] = operation["deadline_" + clock] - .001
        assert not runtime.active
    finally:
        monkeypatch.setattr(time, "perf_counter", original_perf)
        monkeypatch.setattr(time, "time", original_unix)
        runtime.close()
