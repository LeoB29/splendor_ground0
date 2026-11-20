"""Small aggregate timers around unchanged scientific callables.

Only a fixed seven-label table and the active nesting stack are retained. There
are no per-inference events, model hooks, tensor copies or per-request patches.
"""
from __future__ import annotations

import json
import os
import threading
import time


SCHEMA = 'splendorust_component_child_v1'
MAX_PROFILE_BYTES = 8192
LABELS = ('inference', 'native_write', 'native_flush', 'native_read_wait',
          'callback_hash', 'observer_encode', 'decode_observer')
TOTAL_FIELDS = ('calls', 'inclusive_wall_ns', 'inclusive_cpu_ns', 'max_wall_ns',
                'failures', 'exclusive_wall_ns', 'exclusive_cpu_ns')
_PATCH_LOCK = threading.Lock()
_MISSING = object()


class ChildProfiler:
    """Python-process CPU and shared QPC wall time, never deadline corrections."""
    def __init__(self, *, request_read_complete_ns=None, wall_clock=time.perf_counter_ns,
                 cpu_clock=time.process_time_ns, pid=None):
        self.wall_clock, self.cpu_clock = wall_clock, cpu_clock
        self.request_read_complete_ns = request_read_complete_ns
        self.pid = os.getpid() if pid is None else pid
        self.totals = {label: [0, 0, 0, 0, 0, 0, 0] for label in LABELS}
        self.stack = []
        self.nested_calls = 0
        self.started_wall, self.started_cpu = wall_clock(), cpu_clock()
        self.ended_wall = self.ended_cpu = None
        self.failed = False

    def begin(self, label):
        if label not in self.totals or self.ended_wall is not None or len(self.stack) >= 32:
            raise ValueError('Invalid bounded child component span')
        frame = [label, self.wall_clock(), self.cpu_clock(), 0, 0]
        self.nested_calls += bool(self.stack)
        self.stack.append(frame)
        return frame

    def end(self, frame, failed=False):
        cpu, wall = self.cpu_clock(), self.wall_clock()
        if not self.stack or self.stack.pop() is not frame:
            raise RuntimeError('Child component nesting changed')
        elapsed_wall, elapsed_cpu = wall-frame[1], cpu-frame[2]
        if min(elapsed_wall, elapsed_cpu) < 0 or frame[3] > elapsed_wall or frame[4] > elapsed_cpu:
            raise RuntimeError('Child component clocks or nesting inconsistent')
        row = self.totals[frame[0]]
        row[0] += 1
        row[1] += elapsed_wall
        row[2] += elapsed_cpu
        row[3] = max(row[3], elapsed_wall)
        row[4] += bool(failed)
        row[5] += elapsed_wall-frame[3]
        row[6] += elapsed_cpu-frame[4]
        if self.stack:
            self.stack[-1][3] += elapsed_wall
            self.stack[-1][4] += elapsed_cpu

    def finish(self, failed=False):
        if self.stack or self.ended_wall is not None:
            raise RuntimeError('Cannot finish active or already finished child profile')
        self.ended_cpu, self.ended_wall = self.cpu_clock(), self.wall_clock()
        self.failed = bool(failed)

    def snapshot(self):
        if self.ended_wall is None or self.stack:
            raise RuntimeError('Cannot snapshot unfinished child handle')
        wall, cpu = self.ended_wall-self.started_wall, self.ended_cpu-self.started_cpu
        exclusive_wall = sum(row[5] for row in self.totals.values())
        exclusive_cpu = sum(row[6] for row in self.totals.values())
        if min(wall-exclusive_wall, cpu-exclusive_cpu) < 0:
            raise RuntimeError('Component totals exceed enclosing handle')
        return dict(schema=SCHEMA, clock='perf_counter_ns', cpu_clock='process_time_ns',
            pid=self.pid, measurement_units='nanoseconds',
            request_read_complete_ns=self.request_read_complete_ns,
            handle_start_ns=self.started_wall, handle_end_ns=self.ended_wall,
            handle_cpu_ns=cpu, before_emit_ns=None, handle_failed=self.failed,
            totals={label:list(row) for label,row in self.totals.items()},
            total_fields=list(TOTAL_FIELDS), component_nested_calls=self.nested_calls,
            handle_unattributed_wall_ns=wall-exclusive_wall,
            handle_unattributed_cpu_ns=cpu-exclusive_cpu,
            cpu_scope='current_python_process_only_excludes_native_rust_and_parent',
            times_are_diagnostic_not_trusted_deadline_receipts=True,
            inclusive_labels_overlap=True,
            exclusive_totals_exclude_nested_components=True,
            native_read_wait_includes_native_execution_pipe_and_scheduling=True,
            post_handle_gap_includes_profile_assembly_serialization_transport_reader_and_scheduling=True)


def validate_payload_size(profile):
    """A fixed small envelope; serialization remains charged to the real clock."""
    # Reserve the longest signed64-bit QPC marker before it is populated by
    # emit, avoiding another profile-only serialization on the response path.
    bounded = dict(profile)
    if bounded.get('before_emit_ns') is None:
        bounded['before_emit_ns'] = 2**63-1
    size = len(json.dumps(bounded, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    if size > MAX_PROFILE_BYTES:
        raise ValueError('Child timing profile exceeds8192-byte bound')
    return size


class _StreamProxy:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner
        self.write = owner.wrap('native_write', original.write) if hasattr(original, 'write') else None
        self.flush = owner.wrap('native_flush', original.flush) if hasattr(original, 'flush') else None
        self.readline = owner.wrap('native_read_wait', original.readline) if hasattr(original, 'readline') else None

    def __getattr__(self, name):
        return getattr(self.original, name)


class InstrumentedBindings:
    """Install once after initialization; activate only during a single handle."""
    def __init__(self):
        self.active = None
        self.restorations = []
        self.installed = False

    def wrap(self, label, function):
        def measured(*args, **kwargs):
            profiler = self.active
            if profiler is None:
                return function(*args, **kwargs)
            frame = profiler.begin(label)
            failed = True
            try:
                result = function(*args, **kwargs)
                failed = False
                return result
            finally:
                profiler.end(frame, failed)
        return measured

    def replace(self, owner, name, replacement):
        previous = vars(owner).get(name, _MISSING)
        setattr(owner, name, replacement)
        self.restorations.append((owner, name, previous))

    def install(self, adapters, *, worker_module=None, adapter_module=None,
                original_module=None, bridge_module=None):
        if self.installed or not _PATCH_LOCK.acquire(blocking=False):
            raise RuntimeError('Concurrent or repeated child instrumentation is forbidden')
        self.installed = True
        try:
            if worker_module is None:
                from tools.splendorust_search_scale_20261024 import worker as worker_module
            if adapter_module is None:
                from tools.splendorust_search_scale_20261024 import adapters as adapter_module
            if original_module is None:
                from tools.external_strength_screen_20261018_02 import splendorust_adapter as original_module
            if bridge_module is None:
                from tools.external_strength_screen_20261018_02 import bridge as bridge_module
            for owner, name, label in (
                (original_module.Opponent, '_infer', 'inference'),
                (adapter_module, 'digest', 'callback_hash'),
                (bridge_module, 'encode_observation', 'observer_encode'),
                (worker_module, 'decode_observer', 'decode_observer')):
                self.replace(owner, name, self.wrap(label, getattr(owner, name)))
            for adapter in adapters.values():
                self.replace(adapter.proc, 'stdin', _StreamProxy(adapter.proc.stdin, self))
                self.replace(adapter.proc, 'stdout', _StreamProxy(adapter.proc.stdout, self))
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if not self.installed:
            return
        if self.active is not None:
            raise RuntimeError('Cannot remove active child instrumentation')
        try:
            for owner, name, previous in reversed(self.restorations):
                if previous is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, previous)
        finally:
            self.restorations.clear()
            self.installed = False
            _PATCH_LOCK.release()
