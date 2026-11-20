"""Small, standard-library-only execution receipts and resource boundaries."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/local_player_qualification_20261018_01"
MAX_SECONDS = 4 * 3600
MAX_BYTES = 256 * 1024**2
MIN_FREE = 5 * 1024**3
MAX_WORKER_RSS = 2 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path),
            "bytes": path.stat().st_size}


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify(items):
    for item in items:
        if file_hash(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Changed bound input: " + item["path"])


def memory():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total_physical", "available_physical", "total_page", "available_page",
             "total_virtual", "available_virtual", "available_extended_virtual")]
    if os.name != "nt":
        import psutil
        m = psutil.virtual_memory()
        return {"available_physical": m.available, "total_physical": m.total,
                "available_page": m.available + psutil.swap_memory().free}
    status = Status()
    status.length = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot inspect memory headroom")
    return {key: getattr(status, key) for key, _ in status._fields_}


def directory_bytes(base):
    return sum(p.stat().st_size for p in Path(base).rglob("*") if p.is_file())


class ResourceHold(RuntimeError):
    pass


def boundary(base, deadline, *, memory_headroom=True):
    receipt = {"time": now(), "remaining_seconds": deadline - time.monotonic(),
               "free_disk_bytes": shutil.disk_usage(ROOT).free,
               "artifact_bytes": directory_bytes(base), "memory": memory()}
    if receipt["remaining_seconds"] <= 0:
        raise ResourceHold("Four-hour qualification ceiling reached")
    if receipt["free_disk_bytes"] < MIN_FREE:
        raise ResourceHold("Five-GiB free-disk reserve reached")
    if receipt["artifact_bytes"] >= MAX_BYTES - 8 * 1024**2:
        raise ResourceHold("256-MiB artifact ceiling approached; eight MiB reserved for in-flight/final receipts")
    # Reserve at least one GiB of available physical memory and enough commit
    # headroom for the worker's entire two-GiB hard RSS envelope plus one GiB.
    if memory_headroom and receipt["memory"]["available_physical"] < 1024**3:
        raise ResourceHold("Less than one GiB physical memory available before worker launch")
    if memory_headroom and receipt["memory"]["available_page"] < 3 * 1024**3:
        raise ResourceHold("Less than three GiB commit headroom before worker launch")
    return receipt


@contextmanager
def execution_lease():
    """OS-released shared lock prevents competing qualification supervisors."""
    path = ROOT / "outputs/local_player_qualification_128.lease"
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def frozen_source_check():
    path = ROOT / "outputs/proposal_breadth_arena_128/manifest.json"
    manifest = read(path)
    if digest(manifest) != "a8395f535998379a2062ebce8fed79cc6a26e5cd1deb699c96bd4b9dca06deb5":
        raise ValueError("Original manifest authentication failed")
    declared = manifest["identity"]["dependencies"]
    failures = []
    for name, expected in declared.items():
        actual = file_hash(ROOT / name)
        if actual != expected:
            failures.append({"path": name, "expected": expected, "actual": actual})
    return {"checked": len(declared), "failures": failures,
            "manifest": artifact(path)}
