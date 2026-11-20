"""Process-only checks: no models, inference, simulation or experiment launches."""
import ctypes
import json
import multiprocessing.spawn
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from splendor_ai.alphazero import windowless_workers_160 as w


def test_configure_spawn_uses_matching_executable_and_identity():
    previous = multiprocessing.spawn.get_executable()
    try:
        identity = w.configure_spawn()
        expected = Path(sys.executable).with_name("pythonw.exe") if w.WINDOWS else Path(sys.executable)
        assert Path(identity["executable"]) == expected.resolve()
        assert Path(os.fsdecode(multiprocessing.spawn.get_executable())) == expected.resolve()
        assert len(identity["sha256"]) == 64
    finally:
        multiprocessing.set_executable(previous)


def test_missing_pythonw_fails_without_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "WINDOWS", True)
    monkeypatch.setattr(w.sys, "executable", str(tmp_path / "python.exe"))
    with pytest.raises(FileNotFoundError, match="windowless interpreter missing"):
        w.configure_spawn()


def test_console_attachment_rejected_even_with_no_window(monkeypatch):
    monkeypatch.setattr(w, "WINDOWS", True)
    monkeypatch.setattr(w.sys, "executable", str(Path(sys.executable).with_name("pythonw.exe")))
    monkeypatch.setattr(w, "_process_executable", lambda _: "pythonw.exe")
    monkeypatch.setattr(w, "_windows_console_state", lambda: {
        "console_window": 0, "console_process_count": 1, "console_process_ids": [123]})
    with pytest.raises(RuntimeError, match="attached to a Windows console"):
        w.assert_windowless_worker()


def test_launch_rejects_another_interpreter_and_same_logs(tmp_path):
    with pytest.raises(ValueError, match="current Python environment"):
        w.detached_launch([tmp_path / "other.exe"], tmp_path / "out", tmp_path / "err", tmp_path)
    with pytest.raises(ValueError, match="distinct stdout and stderr"):
        w.detached_launch([sys.executable], tmp_path / "same", tmp_path / "same", tmp_path)


def test_launch_is_append_only_and_never_retries(monkeypatch, tmp_path):
    stdout, stderr = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    stdout.write_text("preserved stdout\n", encoding="utf-8")
    stderr.write_text("preserved stderr\n", encoding="utf-8")
    calls = []

    def rejected(*args, **kwargs):
        calls.append((args, kwargs))
        raise PermissionError("breakaway denied")

    monkeypatch.setattr(w.subprocess, "Popen", rejected)
    with pytest.raises(PermissionError, match="breakaway denied"):
        w.detached_launch([sys.executable, "-c", "pass"], stdout, stderr, tmp_path)
    assert len(calls) == 1
    assert stdout.read_text(encoding="utf-8") == "preserved stdout\n"
    assert stderr.read_text(encoding="utf-8") == "preserved stderr\n"


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows console and spawn behavior")
def test_actual_windowless_supervisor_and_two_eight_worker_stages(tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    helper = tmp_path / "windowless_fixture.py"
    helper.write_text(textwrap.dedent(r'''
        from concurrent.futures import ProcessPoolExecutor
        import ctypes
        from ctypes import wintypes
        import json
        import multiprocessing
        import os
        from pathlib import Path
        import sys
        import time
        import traceback

        sys.path.insert(0, SOURCE)
        from splendor_ai.alphazero import windowless_workers_160 as w
        ROOT = Path(__file__).parent

        def console_hosts(parent_ids):
            class ENTRY(ctypes.Structure):
                _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                            ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                            ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                            ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                            ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
            kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            for name in ("Process32FirstW", "Process32NextW"):
                getattr(kernel, name).argtypes = [wintypes.HANDLE, ctypes.POINTER(ENTRY)]
                getattr(kernel, name).restype = wintypes.BOOL
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
            if snapshot == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            hosts = []
            try:
                entry = ENTRY()
                entry.dwSize = ctypes.sizeof(ENTRY)
                exists = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
                while exists:
                    if entry.th32ParentProcessID in parent_ids and entry.szExeFile.lower() in (
                            "conhost.exe", "openconsole.exe", "windowsterminal.exe"):
                        hosts.append({"pid": entry.th32ProcessID, "parent_pid": entry.th32ParentProcessID,
                                      "executable": entry.szExeFile})
                    exists = kernel.Process32NextW(snapshot, ctypes.byref(entry))
            finally:
                kernel.CloseHandle(snapshot)
            return hosts

        def probe(stage, slot):
            check = w.assert_windowless_worker()
            (ROOT / f"started_{stage}_{slot}.json").write_text(json.dumps(check), encoding="utf-8")
            deadline = time.monotonic() + 40
            while not (ROOT / f"release_{stage}").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Parent did not release worker fixture")
                time.sleep(.02)
            print(f"worker stdout stage={stage} slot={slot}", flush=True)
            print(f"worker stderr stage={stage} slot={slot}", file=sys.stderr, flush=True)
            return check

        def main():
            parent = w.assert_windowless_worker()
            print("supervisor stdout", flush=True)
            print("supervisor stderr", file=sys.stderr, flush=True)
            stages = []
            for stage in range(2):
                identity = w.configure_spawn()
                with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=w.assert_windowless_worker,
                                         initargs=(ROOT / "worker_stdout.txt", ROOT / "worker_stderr.txt")) as pool:
                    futures = [pool.submit(probe, stage, slot) for slot in range(8)]
                    deadline = time.monotonic() + 40
                    while len(list(ROOT.glob(f"started_{stage}_*.json"))) != 8:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Not all eight workers started")
                        time.sleep(.02)
                    checks = [json.loads(p.read_text(encoding="utf-8"))
                              for p in sorted(ROOT.glob(f"started_{stage}_*.json"))]
                    pids = {check["pid"] for check in checks}
                    assert len(pids) == 8
                    hosts = console_hosts(pids | {os.getpid()})
                    (ROOT / f"release_{stage}").write_text("release", encoding="utf-8")
                    checks = [future.result(timeout=20) for future in futures]
                stages.append({"identity": identity, "workers": checks, "console_hosts": hosts})
            (ROOT / "complete.json").write_text(json.dumps({"parent": parent, "stages": stages}), encoding="utf-8")

        if __name__ == "__main__":
            try:
                main()
            except BaseException:
                (ROOT / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
    ''').replace("SOURCE", repr(str(source))), encoding="utf-8")
    stdout, stderr = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    stdout.write_text("prior stdout\n", encoding="utf-8")
    stderr.write_text("prior stderr\n", encoding="utf-8")
    launch = w.detached_launch([sys.executable, "-u", helper], stdout, stderr, tmp_path)
    assert Path(launch["command"][0]).name.lower() == "pythonw.exe"
    assert launch["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic() + 100
    while not (tmp_path / "complete.json").exists():
        failure = tmp_path / "failure.txt"
        if failure.exists():
            pytest.fail(failure.read_text(encoding="utf-8"))
        if time.monotonic() > deadline:
            pytest.fail(f"Fixture timed out; pid={launch['pid']}; stderr={stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = json.loads((tmp_path / "complete.json").read_text(encoding="utf-8"))
    assert record["parent"]["pid"] == launch["pid"]
    for check in [record["parent"], *[x for s in record["stages"] for x in s["workers"]]]:
        assert Path(check["executable"]).name.lower() == "pythonw.exe"
        assert check["console_window"] == check["console_process_count"] == 0
        assert check["console_process_ids"] == []
    for stage in record["stages"]:
        assert len({x["pid"] for x in stage["workers"]}) == 8
        assert stage["console_hosts"] == []
        assert all(x["parent_pid"] == launch["pid"] for x in stage["workers"])
        assert all(Path(x["parent_executable"]).name.lower() == "pythonw.exe" for x in stage["workers"])
    first_pids = {x["pid"] for x in record["stages"][0]["workers"]}
    second_pids = {x["pid"] for x in record["stages"][1]["workers"]}
    assert first_pids.isdisjoint(second_pids)
    assert stdout.read_text(encoding="utf-8").startswith("prior stdout\nsupervisor stdout\n")
    assert stderr.read_text(encoding="utf-8").startswith("prior stderr\nsupervisor stderr\n")
    assert (tmp_path / "worker_stdout.txt").read_text(encoding="utf-8").count("worker stdout stage=") == 16
    assert (tmp_path / "worker_stderr.txt").read_text(encoding="utf-8").count("worker stderr stage=") == 16
