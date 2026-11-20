"""OFF-only facade of the qualified parked observer; no player hooks.

Ordinary windows permit no host snapshots or context-switch reads. Placement,
identity, original-clock brackets and stop/join proofs remain unchanged. Native
capability/identity probes can occur outside windows during preflight/startup.

The 100 ms native snapshots and start/end psutil context-switch aggregates are
observations, not a scheduler trace. Windows psutil sums counters over CURRENT
live threads, so this sum can decrease when threads disappear. Its signed net
change is not a lifetime-process interval total, even when positive. PageFaultCount
does not distinguish hard faults. No ready-queue delay or causal attribution is
reported. All file writes are the caller's responsibility, after its move clock.

References: https://psutil.io/api/#psutil.Process.num_ctx_switches
https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex
https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getperformanceinfo
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getsystemtimes
"""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import psutil

SCHEMA = 'splendorust_core_compare_disabled_observer_20261027_v2'
LEGACY_SCHEMA = 'splendorust_host_observer_20261027_v1'
PREVIOUS_SCHEMA = 'splendorust_host_observer_20261027_v2'
CONTEXT_UNAVAILABLE = 'not_identifiable_from_live_thread_aggregates'
INTERVAL_SECONDS = .1
MAX_BUFFER_BYTES = 32 * 1024**2
MAX_TARGETS = 12
MAX_SAMPLES = 64
SAMPLE_SLOT_BYTES = 16 * 1024
ACK_SECONDS = 5.
SUPERVISOR_MASK = sum(1 << cpu for cpu in range(4, 20))
NORMAL_PRIORITY = 0x20
FILETIME_EPOCH = 116444736000000000
SEMANTICS = dict(
    context_switches='Sum over current live-thread counters; signed net change is not an interval total',
    interval_total_context_switches=None, interval_total_context_switches_reason=CONTEXT_UNAVAILABLE,
    page_faults='Cumulative process faults; soft/hard split unavailable',
    system_cpu='All logical processors; kernel includes idle; every adjacent interval must be consistent',
    system_cpu_unavailability='Any inconsistent interval makes whole-window and all subspan CPU unavailable; raw counters retained',
    process_io='Cumulative process I/O, not disk paging attribution',
    scheduling_latency=None, scheduling_latency_reason='No scheduler event trace',
    thread_wait_reason=None, thread_wait_reason_reason='Not sampled',
    hard_page_faults=None, hard_page_faults_reason='API does not distinguish them',
    causality='Temporal association only; observations do not establish cause')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _error(error):
    return dict(type=type(error).__name__, detail=str(error)[:2048])


def _integer(value, name):
    require(type(value) is int and value >= 0, 'Invalid '+name)
    return value


def _identities(targets):
    require(type(targets) is list and 0 < len(targets) <= MAX_TARGETS, 'Invalid observer target count')
    out = []
    for value in targets:
        require(set(value) == {'role', 'pid', 'created'}, 'Target identity fields differ')
        require(isinstance(value['role'], str) and 0 < len(value['role']) <= 128, 'Invalid target role')
        require(type(value['pid']) is int and value['pid'] > 0, 'Invalid target PID')
        require(type(value['created']) in (int, float) and math.isfinite(value['created']) and value['created'] > 0,
                'Invalid target creation time')
        out.append(dict(value))
    require(len({x['pid'] for x in out}) == len(out) and len({x['role'] for x in out}) == len(out),
            'Duplicate observer targets')
    return out


class FILETIME(ctypes.Structure):
    _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]


class MEMORY(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set', 'peak_paged', 'paged',
                                          'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]


class PERFORMANCE(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32)] + [(name, ctypes.c_size_t) for name in
        ('commit_total', 'commit_limit', 'commit_peak', 'physical_total', 'physical_available',
         'system_cache', 'kernel_total', 'kernel_paged', 'kernel_nonpaged', 'page_size')] + [
        (name, ctypes.c_uint32) for name in ('handle_count', 'process_count', 'thread_count')]


class IO(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('read_operations', 'write_operations', 'other_operations', 'read_bytes', 'write_bytes', 'other_bytes')]


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [('mask', ctypes.c_size_t), ('group', ctypes.c_uint16), ('reserved', ctypes.c_uint16 * 3)]


def _ticks(value):
    return (int(value.high) << 32) | int(value.low)


def _api():
    require(os.name == 'nt' and ctypes.sizeof(ctypes.c_void_p) == 8, 'Observer requires Windows x64')
    require(ctypes.sizeof(MEMORY) == 80 and ctypes.sizeof(PERFORMANCE) == 104,
            'Windows memory counter ABI differs')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'OpenProcess': ([ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32], ctypes.c_void_p),
        'CloseHandle': ([ctypes.c_void_p], ctypes.c_int),
        'GetProcessTimes': ([ctypes.c_void_p]+[ctypes.POINTER(FILETIME)]*4, ctypes.c_int),
        'GetExitCodeProcess': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)], ctypes.c_int),
        'K32GetProcessMemoryInfo': ([ctypes.c_void_p, ctypes.POINTER(MEMORY), ctypes.c_uint32], ctypes.c_int),
        'K32GetPerformanceInfo': ([ctypes.POINTER(PERFORMANCE), ctypes.c_uint32], ctypes.c_int),
        'GetProcessIoCounters': ([ctypes.c_void_p, ctypes.POINTER(IO)], ctypes.c_int),
        'GetSystemTimes': ([ctypes.POINTER(FILETIME)]*3, ctypes.c_int),
        'GetCurrentThread': ([], ctypes.c_void_p), 'GetCurrentProcess': ([], ctypes.c_void_p),
        'GetCurrentThreadId': ([], ctypes.c_uint32),
        'GetThreadGroupAffinity': ([ctypes.c_void_p, ctypes.POINTER(GROUP_AFFINITY)], ctypes.c_int),
        'GetProcessAffinityMask': ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)], ctypes.c_int),
        'GetPriorityClass': ([ctypes.c_void_p], ctypes.c_uint32),
    }
    for name, (args, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = args, result
    return api


def _call(api, name, *args):
    ctypes.set_last_error(0)
    result = getattr(api, name)(*args)
    if not result:
        raise OSError(ctypes.get_last_error(), 'Host observer '+name+' failed')
    return result


def _file_identity(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def runtime_identity():
    """Preflight capability and pin physical loaded DLL/API/psutil owners.

    Safe to call before freezing: only own-process OS counters, no models or
    game/environment work. Stable output excludes the changing probe values.
    """
    import psutil._pswindows as windows
    api = _api()
    api.GetModuleHandleW.argtypes, api.GetModuleHandleW.restype = [ctypes.c_wchar_p], ctypes.c_void_p
    api.GetModuleFileNameW.argtypes, api.GetModuleFileNameW.restype = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32], ctypes.c_uint32
    api.GetModuleHandleExW.argtypes, api.GetModuleHandleExW.restype = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int

    def path_for(handle):
        buffer = ctypes.create_unicode_buffer(32768)
        length = _call(api, 'GetModuleFileNameW', handle, buffer, len(buffer))
        require(length < len(buffer), 'Truncated observer module path')
        return str(Path(buffer.value).resolve())

    modules = {name: path_for(_call(api, 'GetModuleHandleW', name))
               for name in ('kernel32.dll', 'kernelbase.dll', 'ntdll.dll')}
    owners = {}
    for name in ('OpenProcess', 'CloseHandle', 'GetProcessTimes', 'GetExitCodeProcess',
                 'K32GetProcessMemoryInfo', 'K32GetPerformanceInfo', 'GetProcessIoCounters',
                 'GetSystemTimes', 'GetThreadGroupAffinity', 'GetProcessAffinityMask', 'GetPriorityClass'):
        handle = ctypes.c_void_p()
        _call(api, 'GetModuleHandleExW', 6, ctypes.cast(getattr(api, name), ctypes.c_void_p), ctypes.byref(handle))
        owner = path_for(handle.value)
        require(owner in modules.values(), 'Observer API owner outside pinned system modules')
        owners[name] = owner
    backend = WindowsCounters([dict(role='preflight_self', pid=os.getpid(), created=psutil.Process().create_time())])
    try:
        backend.snapshot()
        backend.context_switches()
    finally:
        backend.close()
    return dict(schema=SCHEMA, backend='windows_native_counters_plus_psutil_live_thread_switch_aggregates',
                architecture='parked_supervisor_thread', interval_seconds=INTERVAL_SECONDS,
                psutil_version=psutil.__version__, capability_preflight_passed=True,
                modules={name: _file_identity(path) for name, path in modules.items()},
                api_owners=owners, psutil_files=[_file_identity(module.__file__) for module in
                    (psutil, windows, windows.cext)], semantics=SEMANTICS)


class WindowsCounters:
    """Preopened process handles; no process enumeration during snapshots."""
    def __init__(self, targets):
        self.targets = _identities(targets)
        self.api, self.handles, self.closed = _api(), {}, False
        try:
            for target in self.targets:
                # QUERY_INFORMATION | VM_READ, read-only access.
                handle = _call(self.api, 'OpenProcess', 0x0400 | 0x0010, 0, target['pid'])
                self.handles[target['pid']] = handle
                self._times(target)
        except BaseException:
            self.close()
            raise

    def _times(self, target):
        handle = self.handles[target['pid']]
        created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        _call(self.api, 'GetProcessTimes', handle, ctypes.byref(created), ctypes.byref(exited),
              ctypes.byref(kernel), ctypes.byref(user))
        created_seconds = (_ticks(created) - FILETIME_EPOCH) / 10_000_000
        require(abs(created_seconds - target['created']) <= 1e-6, 'Observer process identity changed')
        exit_code = ctypes.c_uint32()
        _call(self.api, 'GetExitCodeProcess', handle, ctypes.byref(exit_code))
        require(exit_code.value == 259, 'Observed process exited')
        return dict(created_filetime_100ns=_ticks(created), kernel_100ns=_ticks(kernel), user_100ns=_ticks(user))

    def placement(self):
        api = self.api
        process_mask, system_mask, group = ctypes.c_size_t(), ctypes.c_size_t(), GROUP_AFFINITY()
        _call(api, 'GetProcessAffinityMask', api.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask))
        _call(api, 'GetThreadGroupAffinity', api.GetCurrentThread(), ctypes.byref(group))
        priority = _call(api, 'GetPriorityClass', api.GetCurrentProcess())
        require(process_mask.value == SUPERVISOR_MASK and group.mask == SUPERVISOR_MASK and
                group.group == 0 and priority == NORMAL_PRIORITY, 'Observer supervisor placement differs')
        return dict(pid=os.getpid(), created=psutil.Process().create_time(),
                    thread_id=int(api.GetCurrentThreadId()), process_mask=process_mask.value,
                    thread_mask=int(group.mask), processor_group=int(group.group), priority_class=int(priority))

    def snapshot(self):
        require(not self.closed, 'Observer handles closed')
        started = time.perf_counter_ns()
        perf = PERFORMANCE(); perf.cb = ctypes.sizeof(perf)
        _call(self.api, 'K32GetPerformanceInfo', ctypes.byref(perf), perf.cb)
        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        _call(self.api, 'GetSystemTimes', ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        rows = []
        for target in self.targets:
            timing = self._times(target)
            memory = MEMORY(); memory.cb = ctypes.sizeof(memory)
            io = IO(); handle = self.handles[target['pid']]
            _call(self.api, 'K32GetProcessMemoryInfo', handle, ctypes.byref(memory), memory.cb)
            _call(self.api, 'GetProcessIoCounters', handle, ctypes.byref(io))
            rows.append(dict(identity=target, **timing, working_set_bytes=int(memory.working_set),
                             private_bytes=int(memory.private), page_faults=int(memory.faults),
                             io={name: int(getattr(io, name)) for name, _ in IO._fields_}))
        return dict(start_ns=started, end_ns=time.perf_counter_ns(),
                    system=dict(available_physical_bytes=int(perf.physical_available * perf.page_size),
                                total_physical_bytes=int(perf.physical_total * perf.page_size),
                                commit_bytes=int(perf.commit_total * perf.page_size),
                                commit_limit_bytes=int(perf.commit_limit * perf.page_size),
                                commit_headroom_bytes=int((perf.commit_limit-perf.commit_total)*perf.page_size),
                                idle_100ns=_ticks(idle), kernel_100ns=_ticks(kernel), user_100ns=_ticks(user)),
                    processes=rows)

    def context_switches(self):
        started, rows = time.perf_counter_ns(), []
        for target in self.targets:
            self._times(target)
            before = psutil.Process(target['pid'])
            require(before.create_time() == target['created'], 'Context counter PID creation differs before read')
            counters = before.num_ctx_switches()
            require(psutil.Process(target['pid']).create_time() == target['created'],
                    'Context counter PID creation differs after read')
            self._times(target)
            require(type(counters.voluntary) is int and counters.voluntary >= 0 and counters.involuntary == 0,
                    'Unsupported Windows context-switch counter')
            rows.append(dict(identity=target, live_thread_context_switch_sum=counters.voluntary))
        return dict(start_ns=started, end_ns=time.perf_counter_ns(), processes=rows)

    def close(self):
        errors = []
        for pid, handle in list(self.handles.items()):
            try:
                _call(self.api, 'CloseHandle', handle)
            except BaseException as error:
                errors.append(dict(pid=pid, **_error(error)))
            del self.handles[pid]
        self.closed = True
        return dict(handles_closed=not errors, errors=errors)


def _check_snapshots(samples, targets):
    previous = None
    for sample in samples:
        require(type(sample) is dict and _integer(sample['start_ns'], 'sample start') <=
                _integer(sample['end_ns'], 'sample end'), 'Invalid sample times')
        require([p['identity'] for p in sample['processes']] == targets, 'Sample process identities differ')
        for key, value in sample['system'].items():
            _integer(value, 'system '+key)
        system = sample['system']
        require(system['commit_limit_bytes'] - system['commit_bytes'] == system['commit_headroom_bytes'] and
                system['available_physical_bytes'] <= system['total_physical_bytes'], 'Invalid system memory counters')
        for process in sample['processes']:
            for key in ('created_filetime_100ns', 'kernel_100ns', 'user_100ns', 'working_set_bytes', 'private_bytes', 'page_faults'):
                _integer(process[key], key)
            require(abs((process['created_filetime_100ns']-FILETIME_EPOCH)/10_000_000 -
                        process['identity']['created']) <= 1e-6, 'Saved process creation epoch differs')
            require(set(process['io']) == {name for name, _ in IO._fields_}, 'I/O counter fields differ')
            for value in process['io'].values():
                _integer(value, 'I/O counter')
        if previous is not None:
            require(sample['start_ns'] >= previous['end_ns'], 'Overlapping observer samples')
            for now, before in zip(sample['processes'], previous['processes']):
                require(now['created_filetime_100ns'] == before['created_filetime_100ns'], 'Process creation counter changed')
                for key in ('kernel_100ns', 'user_100ns', 'page_faults'):
                    require(now[key] >= before[key], 'Process counter reset: '+key)
                require(all(now['io'][k] >= before['io'][k] for k in now['io']), 'Process I/O counter reset')
        previous = sample


def host_cpu_availability(samples):
    """Classify ALL adjacent GetSystemTimes intervals, retaining bad-pair evidence.

    This helper does not validate unrelated process/memory counters. Invalid raw
    types and chronology raise; inconsistent numeric HOST CPU intervals do not.
    Consumers of a subspan must also require its whole window to be valid: a
    narrower locally valid interval cannot rehabilitate a failed whole window.
    """
    require(type(samples) is list and len(samples) <= MAX_SAMPLES, 'Invalid host CPU sample inventory')
    previous, offending = None, []
    for index, sample in enumerate(samples):
        require(type(sample) is dict and _integer(sample['start_ns'], 'sample start') <=
                _integer(sample['end_ns'], 'sample end'), 'Invalid sample times')
        system = sample['system']
        require(type(system) is dict, 'Invalid host CPU counter object')
        for name in ('idle', 'kernel', 'user'):
            _integer(system[name+'_100ns'], 'system '+name+'_100ns')
        if previous is not None:
            require(sample['start_ns'] >= previous['end_ns'], 'Overlapping observer samples')
            delta = {name: system[name+'_100ns']-previous['system'][name+'_100ns']
                     for name in ('idle', 'kernel', 'user')}
            delta['total'] = delta['kernel']+delta['user']
            reasons = [name+'_counter_decreased' for name in ('idle', 'kernel', 'user') if delta[name] < 0]
            # Kernel includes idle, so idle <= total alone is insufficient.
            if delta['idle'] > delta['kernel']:
                reasons.append('idle_delta_exceeds_kernel_delta')
            if delta['idle'] > delta['total']:
                reasons.append('idle_delta_exceeds_total_delta')
            if reasons:
                offending.append(dict(before_index=index-1, after_index=index,
                    before_start_ns=previous['start_ns'], after_start_ns=sample['start_ns'],
                    delta_100ns=delta, reasons=reasons))
        previous = sample
    result = dict(schema='host_global_cpu_availability_v1', valid=None, available=False,
                  reason='insufficient_samples', busy_fraction=None,
                  sample_count=len(samples), offending_pairs=offending)
    if len(samples) < 2:
        return result
    if offending:
        result.update(valid=False, reason='inconsistent_host_cpu_intervals')
        return result
    first, last = samples[0]['system'], samples[-1]['system']
    total = last['kernel_100ns']-first['kernel_100ns']+last['user_100ns']-first['user_100ns']
    idle = last['idle_100ns']-first['idle_100ns']
    result.update(valid=True, available=total > 0,
                  reason=None if total > 0 else 'no_elapsed_cpu_ticks',
                  busy_fraction=(total-idle)/total if total > 0 else None)
    return result


def _context_delta(before, after, targets, *, raw_field='live_thread_context_switch_sum'):
    require(before['start_ns'] <= before['end_ns'] <= after['start_ns'] <= after['end_ns'],
            'Context counter window ordering differs')
    require([x['identity'] for x in before['processes']] == targets == [x['identity'] for x in after['processes']],
            'Context counter targets differ')
    result = []
    for first, last in zip(before['processes'], after['processes']):
        require(set(first) == set(last) == {'identity', raw_field}, 'Context snapshot fields differ')
        a = _integer(first[raw_field], 'live-thread context aggregate')
        b = _integer(last[raw_field], 'live-thread context aggregate')
        result.append(dict(identity=first['identity'], net_change=b-a,
                           interval_total_context_switches=None,
                           interval_total_context_switches_reason=CONTEXT_UNAVAILABLE))
    return result


def summarize_window(value):
    """Derived counter deltas over the wider observer window, never handler CPU."""
    if not value['enabled'] or len(value['samples']) < 2:
        return None
    samples = value['samples']
    _check_snapshots(samples, value['targets'])
    first, last = samples[0], samples[-1]
    processes = []
    context_changes = value['context_switch_deltas']
    for index, (before, after) in enumerate(zip(first['processes'], last['processes'])):
        change = None if context_changes is None else context_changes[index]['net_change']
        processes.append(dict(identity=before['identity'],
            cpu_seconds=((after['kernel_100ns']-before['kernel_100ns'])+
                         (after['user_100ns']-before['user_100ns']))/10_000_000,
            page_faults=after['page_faults']-before['page_faults'],
            live_thread_context_switch_net_change=change,
            interval_total_context_switches=None,
            interval_total_context_switches_reason=CONTEXT_UNAVAILABLE,
            working_set_change_bytes=after['working_set_bytes']-before['working_set_bytes'],
            private_change_bytes=after['private_bytes']-before['private_bytes'],
            io_deltas={key: after['io'][key]-before['io'][key] for key in before['io']}))
    host_cpu = host_cpu_availability(samples)
    return dict(scope='observer_bracketing_window_not_exact_move_or_handler',
                sampled_span_ns=last['end_ns']-first['start_ns'],
                snapshot_query_total_ns=sum(sample['end_ns']-sample['start_ns'] for sample in samples),
                context_query_total_ns=sum(value[key]['end_ns']-value[key]['start_ns']
                                           for key in ('context_before', 'context_after') if value[key] is not None),
                context_query_snapshots_available=sum(value[key] is not None for key in ('context_before', 'context_after')),
                minimum_available_physical_bytes=min(sample['system']['available_physical_bytes'] for sample in samples),
                minimum_commit_headroom_bytes=min(sample['system']['commit_headroom_bytes'] for sample in samples),
                host_cpu_availability=host_cpu,
                system_cpu_busy_fraction=host_cpu['busy_fraction'],
                system_cpu_busy_reason=host_cpu['reason'],
                processes=processes)


class HostObserver:
    """One parked thread; request wrapper runs only outside the original clock.

    Backend injection is for model-free tests only. A real instance must be
    constructed after the runner is isolated and both fixed workers exist.
    """
    def __init__(self, process_identities, *, interval_seconds=INTERVAL_SECONDS,
                 max_buffer_bytes=MAX_BUFFER_BYTES, _backend=None):
        require(interval_seconds == INTERVAL_SECONDS and max_buffer_bytes == MAX_BUFFER_BYTES,
                'Observer frozen sampling bounds differ')
        self.targets = _identities(process_identities)
        self.backend = WindowsCounters(self.targets) if _backend is None else _backend
        self.condition = threading.Condition()
        self.active = None
        self.stopping = False
        self.closed = False
        self.next_token = 0
        self.total_bytes = 0
        self.completed_windows = 0
        self.fatal = None
        self.thread_placement = None
        self.thread_ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name='splendor-host-observer', daemon=True)
        self.thread.start()
        require(self.thread_ready.wait(ACK_SECONDS), 'Observer startup acknowledgement timed out')
        if self.fatal:
            self.close()
            raise RuntimeError('Observer startup failed: '+str(self.fatal))

    def _sample(self, window, kind, due_ns=None):
        if len(window['samples']) >= MAX_SAMPLES:
            raise RuntimeError('Host observer sample cap exceeded')
        row = self.backend.snapshot()
        row['kind'] = kind
        row['scheduled_ns'] = due_ns
        row['lateness_ns'] = None if due_ns is None else max(0, row['start_ns']-due_ns)
        row['schedule_advanced_ns'] = None
        row['skipped_after'] = None
        window['samples'].append(row)
        # Bounded objects, no encoding, hashing or file I/O in a timed window.
        require(len(window['samples'])*SAMPLE_SLOT_BYTES <= MAX_BUFFER_BYTES, 'Observer memory slot bound')

    def _run(self):
        try:
            self.thread_placement = self.backend.placement()
        except BaseException as error:
            self.fatal = _error(error)
            self.thread_ready.set()
            return
        self.thread_ready.set()
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.active is not None or self.stopping)
                if self.stopping:
                    return
                window = self.active
            try:
                window['placement_start'] = self.backend.placement()
                if window['enabled']:
                    window['context_before'] = self.backend.context_switches()
                    self._sample(window, 'before')
                window['ready_ns'] = time.perf_counter_ns()
                window['ready'].set()
                due = window['ready_ns'] + int(INTERVAL_SECONDS*1e9)
                while not window['finish'].is_set():
                    if not window['enabled']:
                        window['finish'].wait()
                        break
                    remaining = max(0, (due-time.perf_counter_ns())/1e9)
                    if window['finish'].wait(remaining):
                        break
                    self._sample(window, 'periodic', due)
                    now = time.perf_counter_ns()
                    skipped = max(0, (now-due)//int(INTERVAL_SECONDS*1e9))
                    window['samples'][-1]['schedule_advanced_ns'] = now
                    window['samples'][-1]['skipped_after'] = skipped
                    window['missed_intervals'] += skipped
                    due += (skipped+1)*int(INTERVAL_SECONDS*1e9)
                if window['enabled']:
                    self._sample(window, 'after')
                    window['context_after'] = self.backend.context_switches()
                    _check_snapshots(window['samples'], self.targets)
                    window['context_switch_deltas'] = _context_delta(window['context_before'], window['context_after'], self.targets)
            except BaseException as error:
                window['errors'].append(_error(error))
                self.fatal = window['errors'][-1]
            finally:
                # Preserve missing evidence honestly, but always attempt this
                # independent end check even when a counter/API read failed.
                try:
                    window['placement_end'] = self.backend.placement()
                except BaseException as error:
                    window['errors'].append(_error(error))
                    self.fatal = window['errors'][-1]
                if window['ready_ns'] is None:
                    window['ready_ns'] = time.perf_counter_ns()
                window['ready'].set()
                window['done_ns'] = time.perf_counter_ns()
                window['done'].set()
                with self.condition:
                    self.condition.wait_for(lambda: self.active is not window or self.stopping)
                    if self.stopping:
                        return

    def start_window(self, sequence, enabled, *, binding=None, timeout_seconds=ACK_SECONDS):
        require(enabled is False, 'Lean observer permits disabled windows only')
        require(not self.closed and not self.fatal and self.active is None, 'Observer unavailable or window active')
        require(type(sequence) is int and sequence >= 0 and type(enabled) is bool, 'Invalid window request')
        require(0 < timeout_seconds <= ACK_SECONDS, 'Observer acknowledgement allowance differs')
        require(binding is None or type(binding) is dict, 'Invalid request binding')
        require(len(_json(binding).encode()) <= 4096, 'Observer binding too large')
        self.next_token += 1
        window = dict(token=self.next_token, sequence=sequence, enabled=enabled, binding=binding,
                      requested_ns=time.perf_counter_ns(), ready_ns=None, finish_requested_ns=None, done_ns=None,
                      samples=[], missed_intervals=0, errors=[], context_before=None, context_after=None,
                      context_switch_deltas=None, placement_start=None, placement_end=None,
                      ready=threading.Event(), finish=threading.Event(), done=threading.Event())
        with self.condition:
            self.active = window
            self.condition.notify_all()
        if not window['ready'].wait(timeout_seconds):
            window['errors'].append(dict(type='TimeoutError', detail='Observer start acknowledgement timed out'))
            self.fatal = window['errors'][-1]
        return dict(token=window['token'], sequence=sequence, enabled=enabled,
                    ready_ns=window['ready_ns'], status='observer_failed' if window['errors'] else 'ready')

    def finish_window(self, token, *, timeout_seconds=ACK_SECONDS):
        window = self.active
        require(window is not None and token['token'] == window['token'] and token['sequence'] == window['sequence'],
                'Observer window token differs')
        require(0 <= timeout_seconds <= ACK_SECONDS, 'Observer finish allowance differs')
        window['finish_requested_ns'] = time.perf_counter_ns()
        window['finish'].set()
        if not window['done'].wait(timeout_seconds):
            window['errors'].append(dict(type='TimeoutError', detail='Observer finish acknowledgement timed out'))
            self.fatal = window['errors'][-1]
        result = {key: value for key, value in window.items() if key not in ('ready', 'finish', 'done')}
        # Detach the snapshot before returning; a delayed thread is never allowed
        # to mutate saved evidence. A failed join stops the collector.
        result = json.loads(_json(result))
        result.update(schema=SCHEMA, architecture='parked_supervisor_thread', interval_seconds=INTERVAL_SECONDS,
                      targets=self.targets, semantics=SEMANTICS,
                      status='observer_failed' if result['errors'] else 'observed' if result['enabled'] else 'disabled',
                      thread_alive_at_finish=self.thread.is_alive(), byte_bound=MAX_BUFFER_BYTES,
                      sample_count=len(result['samples']))
        result['host_cpu_availability'] = None
        try:
            result['host_cpu_availability'] = host_cpu_availability(result['samples'])
            result['summary'] = summarize_window(result)
        except BaseException as error:
            result['errors'].append(_error(error))
            result['status'] = 'observer_failed'
            self.fatal = result['errors'][-1]
            result['summary'] = None
        result['summary_absence_reason'] = (None if result['summary'] is not None else
            'disabled' if not result['enabled'] else 'incomplete_or_invalid_native_snapshots')
        encoded_bytes = len(_json(result).encode())
        if encoded_bytes > MAX_SAMPLES*SAMPLE_SLOT_BYTES or self.total_bytes+encoded_bytes > MAX_BUFFER_BYTES:
            result['errors'].append(dict(type='OverflowError', detail='Host observer byte cap exceeded'))
            result['status'] = 'observer_failed'
            self.fatal = result['errors'][-1]
            result['summary'] = None
            result['summary_absence_reason'] = 'observer_failed'
        self.total_bytes += len(_json(result).encode())
        self.completed_windows += 1
        with self.condition:
            if window['done'].is_set():
                self.active = None
            self.condition.notify_all()
        return result

    def close(self, *, timeout_seconds=ACK_SECONDS):
        require(0 <= timeout_seconds <= ACK_SECONDS, 'Observer close allowance differs')
        started = time.perf_counter_ns()
        if self.closed:
            raise ValueError('Observer already closed')
        errors = []
        if self.active is not None:
            errors.append(dict(type='RuntimeError', detail='Observer closed with active window'))
            self.active['finish'].set()
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout_seconds)
        joined = not self.thread.is_alive()
        if joined:
            closed = self.backend.close()
            errors.extend(closed['errors'])
        else:
            closed = dict(handles_closed=False)
            errors.append(dict(type='TimeoutError', detail='Observer thread failed to stop/join; handles retained'))
        self.closed = True
        return dict(schema=SCHEMA, status='closed' if joined and not errors else 'observer_failed',
                    start_ns=started, end_ns=time.perf_counter_ns(), targets=self.targets,
                    thread_joined=joined, thread_alive=not joined, handles_closed=closed['handles_closed'],
                    placement=self.thread_placement, completed_windows=self.completed_windows,
                    recorded_bytes=self.total_bytes, errors=errors, prior_error=self.fatal,
                    os_child_created=False, os_exit_code=None)


def _validate_placement(value):
    require(type(value) is dict and value['pid'] > 0 and value['created'] > 0 and value['thread_id'] > 0,
            'Observer thread identity missing')
    require(value['process_mask'] == value['thread_mask'] == SUPERVISOR_MASK and
            value['processor_group'] == 0 and value['priority_class'] == NORMAL_PRIORITY,
            'Observer thread placement differs')


def _require_disabled_receipt(value, expected_enabled):
    require(expected_enabled is False and value.get('enabled') is False,
            'Lean observer permits disabled windows only')
    require(value.get('samples') == [] and type(value.get('sample_count')) is int and
            value['sample_count'] == 0 and type(value.get('missed_intervals')) is int and
            value['missed_intervals'] == 0 and all(value.get(key) is None for key in
            ('context_before', 'context_after', 'context_switch_deltas', 'summary')),
            'Disabled observer captured counters or claimed a summary')


def validate_receipt(value, expected_sequence, expected_enabled, process_identities, *, clock_bounds=None):
    _require_disabled_receipt(value, expected_enabled)
    if value.get('status') == 'observer_failed':
        return verify_forensic_receipt(value, expected_sequence, expected_enabled, process_identities,
                                       clock_bounds=clock_bounds)
    targets = _identities(process_identities)
    require(value['schema'] == SCHEMA and value['architecture'] == 'parked_supervisor_thread' and
            value['interval_seconds'] == INTERVAL_SECONDS and value['targets'] == targets and
            value['sequence'] == expected_sequence and value['enabled'] is expected_enabled and
            value['semantics'] == SEMANTICS and value['byte_bound'] == MAX_BUFFER_BYTES,
            'Observer receipt identity differs')
    require(value['sample_count'] == len(value['samples']) <= MAX_SAMPLES, 'Observer sample inventory differs')
    require(type(value['errors']) is list and value['status'] in ('observed', 'disabled', 'observer_failed'),
            'Observer receipt status differs')
    require(value['status'] == 'observer_failed' if value['errors'] else value['status'] == ('observed' if expected_enabled else 'disabled'),
            'Observer error/status differs')
    _integer(value['missed_intervals'], 'missed intervals')
    _check_snapshots(value['samples'], targets)
    require(value['host_cpu_availability'] == host_cpu_availability(value['samples']),
            'Host CPU availability differs')
    require(value['requested_ns'] <= value['ready_ns'] <= value['finish_requested_ns'] <= value['done_ns'],
            'Observer lifecycle ordering differs')
    _validate_placement(value['placement_start']); _validate_placement(value['placement_end'])
    require(value['placement_start'] == value['placement_end'], 'Observer thread identity changed')
    require(value['thread_alive_at_finish'] is True, 'Observer sampler thread unexpectedly exited')
    if expected_enabled:
        require(len(value['samples']) >= 2 and value['samples'][0]['kind'] == 'before' and
                value['samples'][-1]['kind'] == 'after' and
                all(x['kind'] == 'periodic' for x in value['samples'][1:-1]), 'Observer sample boundaries differ')
        require(value['context_switch_deltas'] == _context_delta(value['context_before'], value['context_after'], targets),
                'Context switch delta differs')
        require(value['requested_ns'] <= value['context_before']['start_ns'] <= value['context_before']['end_ns'] <=
                value['samples'][0]['start_ns'] <= value['samples'][0]['end_ns'] <= value['ready_ns'] and
                value['finish_requested_ns'] <= value['samples'][-1]['start_ns'] <= value['samples'][-1]['end_ns'] <=
                value['context_after']['start_ns'] <= value['context_after']['end_ns'] <= value['done_ns'],
                'Counter snapshots do not bracket observed window')
        due, skipped_total = value['ready_ns'] + int(INTERVAL_SECONDS*1e9), 0
        for sample in (value['samples'][0], value['samples'][-1]):
            require(all(sample[field] is None for field in ('scheduled_ns', 'lateness_ns', 'schedule_advanced_ns', 'skipped_after')),
                    'Boundary sample claims periodic schedule')
        for sample in value['samples'][1:-1]:
            require(sample['scheduled_ns'] == due and sample['lateness_ns'] == max(0, sample['start_ns']-due),
                    'Sample scheduling lag differs')
            require(sample['end_ns'] <= sample['schedule_advanced_ns'] <= value['done_ns'], 'Sample schedule advancement differs')
            skipped = max(0, (sample['schedule_advanced_ns']-due)//int(INTERVAL_SECONDS*1e9))
            require(sample['skipped_after'] == skipped, 'Missed interval record differs')
            skipped_total += skipped
            due += (skipped+1)*int(INTERVAL_SECONDS*1e9)
        require(value['missed_intervals'] == skipped_total, 'Missed interval total differs')
    else:
        require(not value['samples'] and value['context_before'] is None and value['context_after'] is None and
                value['context_switch_deltas'] is None and value['missed_intervals'] == 0, 'Disabled observer captured counters')
    require(value['summary'] == summarize_window(value) and value['summary_absence_reason'] ==
            (None if expected_enabled else 'disabled'), 'Observer derived summary differs')
    if clock_bounds is not None:
        start = clock_bounds['request_start_ns']; end = clock_bounds['receipt_arrival_ns']
        require(value['ready_ns'] <= start <= value['finish_requested_ns'],
                'Host observation does not enclose original move clock')
        if end is not None:
            require(start <= end <= value['finish_requested_ns'],
                    'Host observation does not enclose original response arrival')
    return dict(status='passed', accepted=True, measurement_accepted=True, missing_fields=[],
                sample_count=len(value['samples']), context_aggregate_supported=expected_enabled,
                interval_total_context_switches_available=False)


def _verify_failed_receipt(value, expected_sequence, expected_enabled, targets, *, schema, semantics,
                           raw_context_field, clock_bounds):
    """Authenticate available evidence without accepting an incomplete window."""
    require(value['schema'] == schema and value['architecture'] == 'parked_supervisor_thread' and
            value['interval_seconds'] == INTERVAL_SECONDS and value['targets'] == targets and
            value['sequence'] == expected_sequence and value['enabled'] is expected_enabled and
            value['semantics'] == semantics and value['byte_bound'] == MAX_BUFFER_BYTES,
            'Forensic observer identity differs')
    require(value['status'] == 'observer_failed' and type(value['errors']) is list and value['errors'],
            'Forensic failed window must retain original failure')
    require(all(type(error) is dict and set(error) == {'type', 'detail'} and
                type(error['type']) is str and 0 < len(error['type']) <= 128 and
                type(error['detail']) is str and 0 < len(error['detail']) <= 2048 for error in value['errors']),
            'Malformed forensic error evidence')
    require(value['sample_count'] == len(value['samples']) <= MAX_SAMPLES, 'Forensic sample inventory differs')
    _integer(value['missed_intervals'], 'missed intervals')
    missing, counter_issues = [], []
    times = []
    for field in ('requested_ns', 'ready_ns', 'finish_requested_ns', 'done_ns'):
        current = value[field]
        if current is None:
            missing.append(field)
        else:
            times.append(_integer(current, field))
    require(value['requested_ns'] is not None and times == sorted(times), 'Forensic lifecycle times differ')
    for field in ('placement_start', 'placement_end'):
        if value[field] is None:
            missing.append(field)
        else:
            _validate_placement(value[field])
    if value['placement_start'] is not None and value['placement_end'] is not None:
        require(value['placement_start'] == value['placement_end'], 'Forensic thread identity changed')
    previous_end = value['requested_ns']
    for sample in value['samples']:
        _check_snapshots([sample], targets)  # shape, available counters, PID/epoch binding remain strict
        require(sample['start_ns'] >= previous_end and
                (value['done_ns'] is None or sample['end_ns'] <= value['done_ns']), 'Forensic sample chronology differs')
        require(sample['kind'] in ('before', 'periodic', 'after'), 'Forensic sample type differs')
        previous_end = sample['end_ns']
    due = None if value['ready_ns'] is None else value['ready_ns']+int(INTERVAL_SECONDS*1e9)
    cadence_complete, skipped_total = due is not None, 0
    for index, sample in enumerate(value['samples']):
        if sample['kind'] != 'periodic':
            require(all(sample[field] is None for field in ('scheduled_ns', 'lateness_ns', 'schedule_advanced_ns', 'skipped_after')),
                    'Forensic boundary sample claims periodic schedule')
            continue
        scheduled = sample['scheduled_ns']
        require(type(scheduled) is int and sample['lateness_ns'] == max(0, sample['start_ns']-scheduled),
                'Forensic sample lateness differs')
        if due is not None:
            require(scheduled == due, 'Forensic sample schedule differs')
        advanced, skipped = sample['schedule_advanced_ns'], sample['skipped_after']
        if advanced is None or skipped is None:
            for field in ('schedule_advanced_ns', 'skipped_after'):
                if sample[field] is None:
                    missing.append(f'samples[{index}].{field}')
            cadence_complete, due = False, None
        else:
            require(sample['end_ns'] <= advanced and (value['done_ns'] is None or advanced <= value['done_ns']),
                    'Forensic cadence advancement differs')
            require(type(skipped) is int and skipped == max(0, (advanced-scheduled)//int(INTERVAL_SECONDS*1e9)),
                    'Forensic skipped intervals differ')
            skipped_total += skipped
            due = scheduled+(skipped+1)*int(INTERVAL_SECONDS*1e9)
    if cadence_complete:
        require(value['missed_intervals'] == skipped_total, 'Forensic missed interval total differs')
    try:
        _check_snapshots(value['samples'], targets)
    except ValueError as error:
        # A recorded native counter epoch failure can be read back, never used.
        require('reset' in str(error) and any(str(error) == item['detail'] for item in value['errors']),
                'Unexplained invalid native counter evidence')
        counter_issues.append(str(error))
    for field in ('context_before', 'context_after'):
        context = value[field]
        if context is None:
            missing.append(field)
            continue
        require(value['requested_ns'] <= context['start_ns'] <= context['end_ns'] and
                (value['done_ns'] is None or context['end_ns'] <= value['done_ns']), 'Forensic context times differ')
        require([row['identity'] for row in context['processes']] == targets, 'Forensic context identities differ')
        for row in context['processes']:
            require(set(row) == {'identity', raw_context_field}, 'Forensic context fields differ')
            _integer(row[raw_context_field], 'live-thread aggregate')
    changes = None
    if value['context_before'] is not None and value['context_after'] is not None:
        changes = _context_delta(value['context_before'], value['context_after'], targets,
                                 raw_field=raw_context_field)
    if value['context_switch_deltas'] is None:
        missing.append('context_switch_deltas')
    elif schema == SCHEMA:
        require(changes is not None and value['context_switch_deltas'] == changes, 'Forensic signed changes differ')
    else:
        require(False, 'Legacy failed regression must retain its missing derived counter evidence')
    if value['summary'] is None:
        missing.append('summary')
        require(value['summary_absence_reason'] in
                ('observer_failed', 'incomplete_or_invalid_native_snapshots', 'disabled'),
                'Forensic missing summary needs explicit reason')
    else:
        require(schema == SCHEMA and not counter_issues and value['summary'] == summarize_window(value) and
                value['summary_absence_reason'] is None, 'Forensic independent native summary differs')
    if schema == SCHEMA:
        expected_cpu = host_cpu_availability(value['samples'])
        require(value['host_cpu_availability'] == expected_cpu, 'Forensic host CPU availability differs')
    if not expected_enabled:
        require(not value['samples'] and value['context_before'] is None and value['context_after'] is None,
                'Disabled failed observer captured counters')
    if clock_bounds is not None:
        start, end = clock_bounds['request_start_ns'], clock_bounds['receipt_arrival_ns']
        require(value['ready_ns'] is not None and value['finish_requested_ns'] is not None and
                value['ready_ns'] <= start <= value['finish_requested_ns'], 'Forensic request window differs')
        require(end is None or start <= end <= value['finish_requested_ns'], 'Forensic trusted arrival differs')
    return dict(status='failed_window_preserved_not_accepted', accepted=False, measurement_accepted=False,
                schema_verified=schema, sample_count=len(value['samples']), missing_fields=missing,
                complete_cadence_equations_verified=cadence_complete,
                native_counter_issues=counter_issues,
                derived_live_thread_net_changes_for_explanation_only=changes,
                original_status_preserved=value['status'], original_errors_preserved=value['errors'])


def verify_forensic_receipt(value, expected_sequence, expected_enabled, process_identities, *, clock_bounds=None):
    """Valid v3 windows stay valid; failed v3 windows can never be accepted."""
    _require_disabled_receipt(value, expected_enabled)
    if value.get('status') != 'observer_failed':
        return validate_receipt(value, expected_sequence, expected_enabled, process_identities,
                                clock_bounds=clock_bounds)
    return _verify_failed_receipt(value, expected_sequence, expected_enabled, _identities(process_identities),
                                 schema=SCHEMA, semantics=SEMANTICS,
                                 raw_context_field='live_thread_context_switch_sum', clock_bounds=clock_bounds)


def verify_legacy_forensic_receipt(value, expected_sequence, expected_enabled, process_identities, *, clock_bounds=None):
    """Explicit v1 failure readback only: no schema projection or rehabilitation."""
    from tools.splendorust_pool6_hostdiag_20261027_03.host_observer import verify_legacy_forensic_receipt as old_verify
    return old_verify(value, expected_sequence, expected_enabled, process_identities, clock_bounds=clock_bounds)


def verify_previous_forensic_receipt(value, expected_sequence, expected_enabled, process_identities, *, clock_bounds=None):
    """Exact v2 failure readback through its frozen checker; never a v3 acceptance."""
    from tools.splendorust_pool6_hostdiag_20261027_02.host_observer import verify_forensic_receipt as old_verify
    require(value['schema'] == PREVIOUS_SCHEMA and value['status'] == 'observer_failed',
            'Previous-version failure evidence required')
    return old_verify(value, expected_sequence, expected_enabled, process_identities, clock_bounds=clock_bounds)


def validate_close(value, process_identities, *, expected_windows=None):
    require(value['schema'] == SCHEMA and value['targets'] == _identities(process_identities), 'Observer close identity differs')
    require(value['status'] == 'closed' and not value['errors'] and value['thread_joined'] is True and
            value['thread_alive'] is False and value['handles_closed'] is True and
            value['os_child_created'] is False and value['os_exit_code'] is None, 'Observer close failed')
    require(value['start_ns'] <= value['end_ns'] and value['recorded_bytes'] <= MAX_BUFFER_BYTES,
            'Observer close time/byte bounds differ')
    _validate_placement(value['placement'])
    if expected_windows is not None:
        require(value['completed_windows'] == expected_windows, 'Observer close window count differs')
    return dict(status='passed', thread_joined=True, handles_closed=True,
                scope='cleanup_only_not_measurement_acceptance', prior_observer_error=value['prior_error'])
