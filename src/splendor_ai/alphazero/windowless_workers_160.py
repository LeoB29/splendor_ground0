"""Windowless process plumbing for160; no game or inference behavior lives here.

Windows spawn starts a fresh executable for every pool worker. Detaching only
the supervisor therefore does not prevent console-subsystem Python workers
from creating their own consoles. Use the matching GUI-subsystem interpreter
for both the supervisor and every worker, and check attachment in each worker.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


WINDOWS = sys.platform == "win32"


def _windowless_executable() -> Path:
    current = Path(sys.executable).resolve()
    executable = current.with_name("pythonw.exe") if WINDOWS else current
    if not executable.is_file():
        raise FileNotFoundError(f"Matching windowless interpreter missing: {executable}")
    return executable


def _identity(executable: Path) -> dict:
    return {"executable": str(executable),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "platform": sys.platform}


def configure_spawn() -> dict:
    """Select the matching interpreter before constructing each spawn pool."""
    executable = _windowless_executable()
    multiprocessing.set_executable(str(executable))
    return _identity(executable)


def _windows_console_state() -> dict:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetConsoleWindow.argtypes = []
    kernel.GetConsoleWindow.restype = wintypes.HWND
    kernel.GetConsoleProcessList.argtypes = [ctypes.POINTER(wintypes.DWORD), wintypes.DWORD]
    kernel.GetConsoleProcessList.restype = wintypes.DWORD
    window = int(kernel.GetConsoleWindow() or 0)
    processes = (wintypes.DWORD * 64)()
    ctypes.set_last_error(0)
    count = int(kernel.GetConsoleProcessList(processes, len(processes)))
    error = ctypes.get_last_error()
    # A detached GUI process normally reports ERROR_INVALID_HANDLE here.
    if count == 0 and error not in (0, 6):
        raise OSError(error, "Could not verify absence of a Windows console")
    if count > len(processes):
        processes = (wintypes.DWORD * count)()
        count = int(kernel.GetConsoleProcessList(processes, len(processes)))
        if count == 0:
            raise ctypes.WinError(ctypes.get_last_error())
    return {"console_window": window, "console_process_count": count,
            "console_process_ids": list(processes[:count]), "console_query_error": error}


def _process_executable(pid: int) -> str | None:
    if not WINDOWS:
        return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                               wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        buffer = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(buffer))
        if not kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return buffer.value
    finally:
        kernel.CloseHandle(handle)


def _attach_stream(name: str, path: str | Path | None) -> str | None:
    """Explicit append logs survive pythonw's absent inherited standard handles."""
    if path is not None:
        destination = str(Path(path).resolve())
        setattr(sys, name, _append_log(destination))
        return destination
    if getattr(sys, name) is None:
        setattr(sys, name, open(os.devnull, "w", encoding="utf-8", buffering=1))
        return os.devnull
    return None


def _append_log(path: str | Path, *, binary: bool = False):
    """Kernel append-only handles avoid CRT seek/write races between workers."""
    if not WINDOWS:
        return open(path, "ab", buffering=0) if binary else open(
            path, "a", encoding="utf-8", buffering=1)
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # FILE_APPEND_DATA only, sharing read/write/delete, OPEN_ALWAYS, normal file.
    handle = kernel.CreateFileW(str(path), 0x0004, 0x0007, None, 4, 0x0080, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(int(handle), os.O_WRONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        return os.fdopen(descriptor, "wb", buffering=0) if binary else os.fdopen(
            descriptor, "w", encoding="utf-8", buffering=1)
    except BaseException:
        os.close(descriptor)
        raise


def assert_windowless_worker(stdout_path: str | Path | None = None,
                             stderr_path: str | Path | None = None) -> dict:
    """Initializer check, including consoles without a visible console window.

    Pass explicit append-only log paths in production. With no paths, missing
    pythonw streams are replaced by devnull so third-party print/flush calls
    remain safe; exceptions still propagate through the executor's Future.
    """
    stdout = _attach_stream("stdout", stdout_path)
    stderr = _attach_stream("stderr", stderr_path)
    result = {"pid": os.getpid(), "parent_pid": os.getppid(),
              "executable": str(Path(sys.executable).resolve()),
              "parent_executable": _process_executable(os.getppid()),
              "stdout_path": stdout, "stderr_path": stderr}
    if WINDOWS:
        result.update(_windows_console_state())
        if Path(sys.executable).name.lower() != "pythonw.exe":
            raise RuntimeError(f"Worker is not using pythonw.exe: {result}")
        if result["console_window"] or result["console_process_count"]:
            raise RuntimeError(f"Worker unexpectedly attached to a Windows console: {result}")
    else:
        result.update(console_window=None, console_process_count=None, console_process_ids=[])
    return result


def detached_launch(command: Sequence[str | Path], stdout_path: str | Path,
                    stderr_path: str | Path, cwd: str | Path) -> dict:
    """Launch this environment's Python command without a shell or console.

    The caller owns process evidence and lifetime. This does not retry a
    rejected breakaway launch, truncate logs, wait for completion, or run any
    work other than the command explicitly supplied by the caller.
    """
    if not command:
        raise ValueError("A Python command is required")
    executable = _windowless_executable()
    identity = _identity(executable)
    requested = Path(command[0]).resolve()
    accepted = {Path(sys.executable).resolve(), executable}
    if requested not in accepted:
        raise ValueError(f"Command must use the current Python environment: {requested}")
    args = [str(executable), *map(str, command[1:])]
    stdout_path, stderr_path = Path(stdout_path).resolve(), Path(stderr_path).resolve()
    if stdout_path == stderr_path:
        raise ValueError("Use distinct stdout and stderr paths")
    flags = 0
    kwargs = {}
    if WINDOWS:
        # pythonw is the essential protection for later multiprocessing children.
        # NO_WINDOW additionally protects this direct launch if subsystem behavior
        # changes; DETACHED_PROCESS takes precedence over that flag on Windows.
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.CREATE_BREAKAWAY_FROM_JOB | subprocess.CREATE_NO_WINDOW)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    with _append_log(stdout_path, binary=True) as stdout, _append_log(stderr_path, binary=True) as stderr:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=stdout,
                                   stderr=stderr, cwd=str(Path(cwd).resolve()),
                                   close_fds=True, **kwargs)
    return {"pid": process.pid, "command": args, "creationflags": flags,
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
            **identity}
