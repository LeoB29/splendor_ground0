"""Serialize only mutable status I/O through a bounded Windows named mutex.

The reader holds the per-path mutex for open/read/close; the publisher holds
the same mutex for an in-place open/write/flush. Its retained writer handle
then fsyncs and closes outside that access mutex, with no further writes.
Existing single-publisher ownership remains held until the helper returns.
This is serialized publication for cooperating readers, not an atomic rename.
Temporary preparation remains outside the mutex and its bounded file remains
for the next publication to overwrite. No retries, cached status, clock resets
or model/search changes occur. Every status read opens its current path anew.
Only typed Win32 function bindings are initialized once per process; no mutex
handle, file handle or status value is retained between operations.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import hashlib
import os
from pathlib import Path
import time


MAX_STATUS_BYTES = 64 * 1024
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
FILE_SHARE_DELETE = 0x00000004
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x00000080
MUTEX_WAIT_MILLISECONDS = 250
WAIT_OBJECT_0 = 0
WAIT_ABANDONED = 0x80
WAIT_TIMEOUT = 0x102


def _load_win32_api():
    """Bind immutable function references once, outside every status mutex."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateMutexW
    create.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR)
    create.restype = wintypes.HANDLE
    wait = kernel32.WaitForSingleObject
    wait.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait.restype = wintypes.DWORD
    release = kernel32.ReleaseMutex
    release.argtypes = (wintypes.HANDLE,)
    release.restype = wintypes.BOOL
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    return create, wait, release, close, create_file


_WIN32_API = _load_win32_api() if os.name == "nt" else None


def mutex_name(path):
    normalized = os.path.normcase(os.path.abspath(os.fspath(path)))
    return "Local\\SplendorGPUStatus_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@contextmanager
def status_mutex(path):
    """Acquire once for at most250ms; a dead owner is a failure, never healthy."""
    if os.name != "nt":
        raise OSError("Status mutex requires Windows")
    create, wait, release, close, _ = _WIN32_API
    handle = create(None, False, mutex_name(path))
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    acquired = False
    try:
        started = time.perf_counter()
        result = wait(handle, MUTEX_WAIT_MILLISECONDS)
        acquired = result in (WAIT_OBJECT_0, WAIT_ABANDONED)
        if result == WAIT_ABANDONED:
            raise RuntimeError("Status mutex abandoned by a failed owner")
        if result == WAIT_TIMEOUT or acquired and time.perf_counter()-started >= .250:
            raise TimeoutError("Status mutex exceeded250ms acquisition bound")
        if result != WAIT_OBJECT_0:
            raise ctypes.WinError(ctypes.get_last_error())
        yield
    finally:
        try:
            if acquired and not release(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            close(handle)


@contextmanager
def open_shared_read(path):
    """Yield a binary file; its single owned Win32 handle closes on every exit."""
    if os.name != "nt":
        raise OSError("Share-delete status reader requires Windows")
    import msvcrt
    _, _, _, close_handle, create_file = _WIN32_API
    filename = os.fspath(Path(path))
    handle = create_file(filename, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, None)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.WinError(ctypes.get_last_error())
        error.filename = filename
        raise error
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close_handle(handle)
        raise
    # open_osfhandle transferred handle ownership to this CRT descriptor.
    try:
        stream = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


def read_json(path, *, max_bytes=MAX_STATUS_BYTES):
    """Read one complete current status object, or propagate the actual failure."""
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_STATUS_BYTES:
        raise ValueError("Status read bound must be an integer from1to65536bytes")
    with open_status_read(path) as stream:
        encoded = stream.read(max_bytes + 1)
    if len(encoded) > max_bytes:
        raise ValueError("Status JSON exceeds the frozen read bound")
    return _parse_status(encoded)


def _parse_status(encoded):
    def nonfinite(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    result = json.loads(encoded.decode("utf-8"), parse_constant=nonfinite)
    if type(result) is not dict:
        raise ValueError("Status JSON must be an object")
    return result


read_status_json = read_json


@contextmanager
def open_status_read(path):
    """Expose the same complete locked read lifetime for deterministic tests."""
    with status_mutex(path):
        with open_shared_read(path) as stream:
            yield stream


def atomic_replace_status(temporary, destination):
    """API-compatible name for bounded, mutex-serialized in-place publication.

    No rename, replacement API, unlink or retry occurs. The completed temporary
    is retained at its fixed name. Any read/write/flush/parse error propagates;
    readers can only accept a complete valid object while owning the mutex.
    """
    if os.name != "nt":
        raise OSError("Atomic status publication requires Windows")
    temporary, destination = Path(temporary), Path(destination)
    if temporary.parent.resolve() != destination.parent.resolve():
        raise ValueError("Status temporary and destination must share a directory")
    with temporary.open("rb") as stream:
        encoded = stream.read(MAX_STATUS_BYTES+1)
    if len(encoded) > MAX_STATUS_BYTES:
        raise ValueError("Status publication exceeds the frozen byte bound")
    _parse_status(encoded)
    stream = None
    try:
        with status_mutex(destination):
            stream = destination.open("wb")
            try:
                written = stream.write(encoded)
                if written != len(encoded):
                    raise OSError("Incomplete status publication write")
                stream.flush()
            except BaseException:
                # A failed buffered write/flush must also close while the
                #access mutex is owned, so close cannot flush partial bytes
                #after readers regain access.
                try:
                    stream.close()
                finally:
                    stream = None
                raise
        # Complete bytes are already visible; durability does not mutate them.
        # The caller's single-publisher lock remains owned through this call.
        os.fsync(stream.fileno())
    finally:
        if stream is not None:
            stream.close()
