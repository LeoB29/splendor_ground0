"""Fresh per-poll Windows parent/thread metadata; all other checks unchanged.

The only optimization batches ppid/num_threads queries into one Tool Help
process snapshot. No snapshot metadata survives between polls. Process
discovery, path resolution/hash authentication, CPU/RSS/affinity measurements,
ownership limits, trusted clocks, readers and cleanup remain the frozen ones.

API reference:
https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32w
https://learn.microsoft.com/en-us/windows/win32/api/tlhelp32/nf-tlhelp32-createtoolhelp32snapshot
"""
import ctypes
import os
from pathlib import Path

import psutil
from tools.splendorust_search_scale_20261024 import common as frozen

BACKEND_ID = 'windows_toolhelp_parent_threads_per_poll_20261024_v1'
TH32CS_SNAPPROCESS = 0x00000002
ERROR_NO_MORE_FILES = 18
MAX_SNAPSHOT_ENTRIES = 32768
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
require = frozen.require


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ('dwSize', ctypes.c_uint32),
        ('cntUsage', ctypes.c_uint32),
        ('th32ProcessID', ctypes.c_uint32),
        ('th32DefaultHeapID', ctypes.c_size_t),
        ('th32ModuleID', ctypes.c_uint32),
        ('cntThreads', ctypes.c_uint32),
        ('th32ParentProcessID', ctypes.c_uint32),
        ('pcPriClassBase', ctypes.c_int32),
        ('dwFlags', ctypes.c_uint32),
        ('szExeFile', ctypes.c_wchar * 260),
    ]


def _load_api():
    require(os.name == 'nt', 'Tool Help sampler requires Windows')
    require(ctypes.sizeof(ctypes.c_wchar) == 2 and ctypes.sizeof(ctypes.c_void_p) in (4, 8),
            'Unexpected Windows process-entry ABI')
    expected = 568 if ctypes.sizeof(ctypes.c_void_p) == 8 else 556
    require(ctypes.sizeof(PROCESSENTRY32W) == expected, 'PROCESSENTRY32W size mismatch')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    api.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    api.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    api.Process32FirstW.restype = ctypes.c_int
    api.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
    api.Process32NextW.restype = ctypes.c_int
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    return api


def _get_error():
    return ctypes.get_last_error()


def _clear_error():
    ctypes.set_last_error(0)


def _os_error(operation):
    code = _get_error()
    return OSError(code, 'Tool Help '+operation+' failed with Windows error '+str(code))


def snapshot_api_modules():
    """Return loaded DLL paths for startup/final hash pinning, never per poll.

    Paths come from actual module handles, not guessed SystemRoot locations.
    Include forwarding implementation modules as well as the kernel32 facade.
    This performs loader metadata queries only, not a process snapshot.
    """
    api = _load_api()
    api.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    api.GetModuleHandleW.restype = ctypes.c_void_p
    api.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    api.GetModuleFileNameW.restype = ctypes.c_uint32
    api.GetModuleHandleExW.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    api.GetModuleHandleExW.restype = ctypes.c_int

    def module_path(handle):
        buffer = ctypes.create_unicode_buffer(32768)
        _clear_error()
        length = api.GetModuleFileNameW(handle, buffer, len(buffer))
        if length == 0 or length >= len(buffer):
            raise _os_error('GetModuleFileNameW')
        path = Path(buffer.value)
        require(path.is_absolute(), 'Loaded system DLL path is not absolute')
        return str(path.resolve())

    modules = {}
    for name in ('kernel32.dll', 'kernelbase.dll', 'ntdll.dll'):
        _clear_error()
        handle = api.GetModuleHandleW(name)
        if not handle:
            raise _os_error('GetModuleHandleW '+name)
        modules[name] = module_path(handle)
    owners = {}
    for name in ('CreateToolhelp32Snapshot', 'Process32FirstW', 'Process32NextW', 'CloseHandle'):
        address = ctypes.cast(getattr(api, name), ctypes.c_void_p)
        owner = ctypes.c_void_p()
        _clear_error()
        # FROM_ADDRESS | UNCHANGED_REFCOUNT: inspect without taking ownership.
        if not api.GetModuleHandleExW(0x00000004 | 0x00000002, address, ctypes.byref(owner)):
            raise _os_error('GetModuleHandleExW '+name)
        path = module_path(owner.value)
        require(path in modules.values(), 'Tool Help function is outside pinned Windows system modules')
        owners[name] = path
    return dict(backend=BACKEND_ID, modules=modules, function_modules=owners,
                method='GetModuleFileNameW on loaded handles and actual function-address owners')


def capture_process_snapshot(*, max_entries=MAX_SNAPSHOT_ENTRIES, _api=None):
    """One fresh system snapshot, bounded enumeration and mandatory handle close.

    No retry, partial-result fallback, process-name use, private psutil API or
    cross-poll cache. Injectable API is solely for model-free fault tests.
    """
    require(type(max_entries) is int and 0 < max_entries <= MAX_SNAPSHOT_ENTRIES, 'Invalid snapshot entry bound')
    api = _load_api() if _api is None else _api
    _clear_error()
    handle = api.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if handle is None or handle == 0 or handle == INVALID_HANDLE_VALUE:
        raise _os_error('CreateToolhelp32Snapshot')
    primary = None
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        _clear_error()
        present = api.Process32FirstW(handle, ctypes.byref(entry))
        if not present:
            # Even ERROR_NO_MORE_FILES is unacceptable here: an operating
            # process snapshot cannot legitimately omit our running process.
            raise _os_error('Process32FirstW')
        result = {}
        while True:
            require(len(result) < max_entries, 'Tool Help snapshot entry ceiling')
            pid = int(entry.th32ProcessID)
            require(pid not in result, 'Duplicate PID in Tool Help snapshot')
            result[pid] = dict(parent_pid=int(entry.th32ParentProcessID), threads=int(entry.cntThreads))
            _clear_error()
            present = api.Process32NextW(handle, ctypes.byref(entry))
            if not present:
                if _get_error() != ERROR_NO_MORE_FILES:
                    raise _os_error('Process32NextW')
                break
        return result
    except BaseException as error:
        primary = error
        raise
    finally:
        _clear_error()
        if not api.CloseHandle(handle):
            closing = _os_error('CloseHandle')
            if primary is None:
                raise closing
            primary.add_note('Snapshot handle closure also failed: '+str(closing))


def fresh_identity(process):
    """Use a new Process object, because create_time is cached per object."""
    fresh = psutil.Process(process.pid)
    return fresh.pid, fresh.create_time()


class BatchedStudyWorker(frozen.StudyWorker):
    """Frozen supervised worker with one substituted process-metadata sampler."""

    def _sample(self):
        if self._ps is None:
            return dict(self._last_telemetry)
        # Keep the old complete discovery, including its PID-reuse checks.
        try:
            processes = [self._ps]+self._ps.children(recursive=True)
        except psutil.NoSuchProcess:
            return dict(self._last_telemetry)
        except psutil.AccessDenied as error:
            self._failure, self._failure_detail = 'process_tree_unobservable', str(error)
            return dict(self._last_telemetry)
        # Snapshot entries carry PID but no creation time. Bracket their use
        # with fresh identities so a replaced process cannot inherit metadata.
        identities = {}
        try:
            for process in processes:
                try:
                    expected = (process.pid, process.create_time())
                    require(fresh_identity(process) == expected, 'Process PID identity changed before snapshot')
                    identities[process.pid] = expected
                except psutil.NoSuchProcess:
                    continue
            snapshot = capture_process_snapshot()
        except (OSError, ValueError, psutil.AccessDenied) as error:
            self._failure, self._failure_detail = 'process_tree_unobservable', str(error)
            return dict(self._last_telemetry)
        live, rss, threads = [], 0, 0
        for proc in processes:
            try:
                if proc.pid not in identities:
                    continue
                identity = identities[proc.pid]
                require(fresh_identity(proc) == identity, 'Process PID identity changed after snapshot')
                require(proc.pid in snapshot, 'Live owned PID missing from fresh Tool Help snapshot')
                metadata = snapshot[proc.pid]
                require(type(metadata) is dict and set(metadata) == {'parent_pid', 'threads'} and
                    type(metadata['parent_pid']) is int and metadata['parent_pid'] >= 0 and
                    type(metadata['threads']) is int and metadata['threads'] >= 0, 'Invalid owned snapshot metadata')
                exe = Path(proc.exe()).resolve()
                ppid = metadata['parent_pid']
                if exe not in self._verified:
                    require(exe in self._expected and frozen.file_hash(exe) == self._expected[exe], 'Unexpected descendant executable')
                    self._verified[exe] = self._expected[exe]
                role = ('python' if proc.pid == self._ps.pid and exe == self._python else
                        self._natives.get(exe) if ppid == self._ps.pid else None)
                if exe == self._console:
                    owners = [role for role, owner in self._owners.items()
                              if owner[0] == ppid and role in ('python', 'search', 'reference')]
                    role = 'console:'+owners[0] if len(owners) == 1 else None
                memory = proc.memory_info().rss
                cpu = proc.cpu_times()
                count = metadata['threads']
                affinity = proc.cpu_affinity()
                require(fresh_identity(proc) == identity, 'Process PID identity changed during snapshot use')
                self._tree_cpu[identity] = max(self._tree_cpu.get(identity, 0), cpu.user+cpu.system)
                live.append(dict(pid=proc.pid, created=identity[1], parent_pid=ppid, role=role,
                    executable=str(exe), rss_bytes=memory, cpu_seconds=self._tree_cpu[identity],
                    affinity=affinity, observed_threads=count))
                rss += memory
                threads += count
                if role is not None:
                    require(role not in self._owners or self._owners[role] == identity, 'Computation/helper replacement')
                    self._owners[role] = identity
            except psutil.NoSuchProcess:
                # Preserve the old sampler's exited-during-observation rule.
                # A live omitted entry or PID replacement fails above instead.
                continue
            except (ValueError, psutil.AccessDenied, OSError) as error:
                self._failure, self._failure_detail = 'process_tree_limit', str(error)
        for row in live:
            identity = (row['pid'], row['created'])
            if row['role'] is None and Path(row['executable']) == self._console:
                owners = [role for role, owner in self._owners.items()
                          if owner[0] == row['parent_pid'] and role in ('python', 'search', 'reference')]
                if len(owners) == 1:
                    row['role'] = 'console:'+owners[0]
                    if row['role'] in self._owners and self._owners[row['role']] != identity:
                        self._failure, self._failure_detail = 'process_tree_limit', 'Console replacement'
                    self._owners[row['role']] = identity
            if row['role'] is None:
                self._failure, self._failure_detail = 'process_tree_limit', 'Unrecognized descendant'
            if row['affinity'] != self.affinity:
                if self._initializing and str(row['role']).startswith('console:'):
                    try:
                        psutil.Process(row['pid']).cpu_affinity(self.affinity)
                        row['startup_affinity_before'] = row['affinity']
                        row['affinity'] = psutil.Process(row['pid']).cpu_affinity()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if row['affinity'] != self.affinity:
                    self._failure, self._failure_detail = 'affinity_changed', 'Worker descendant changed assigned CPU'
        if len(self._owners) > 6:
            self._failure, self._failure_detail = 'process_tree_limit', 'More than three compute processes plus OS helpers'
        self._peak_rss = max(self._peak_rss, rss)
        self._last_telemetry = dict(cpu_seconds=sum(self._tree_cpu.values()), rss_bytes=rss,
            peak_sampled_rss_bytes=self._peak_rss, affinity=self.affinity, observed_threads=threads,
            process_tree=live, resource_scope='sampled_worker_and_all_descendants')
        if rss > self.max_rss_bytes:
            self._failure, self._failure_detail = 'rss_limit', 'Combined worker/helper RSS exceeded 2GiB'
        return dict(self._last_telemetry)
