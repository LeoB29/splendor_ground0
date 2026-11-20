"""Opt-in, bounded instrumentation around the unchanged frozen player.

No model is loaded on import. Every wrapper delegates once to the original
callable and returns the same object. No scientific response is decorated here;
the caller stores ``snapshot()`` separately. Disabled instrumentation performs
no clock reads or object patches. Profiling is diagnostic, never a replacement
for the supervisor's trusted arrival clock.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import threading
import time


SCHEMA = 'splendorust_timing_profile_v1'
_PATCH_LOCK = threading.Lock()
_MISSING = object()


class DecisionProfiler:
    """Nested wall/process-CPU intervals, with complete bounded-name totals.

CPU is this Python process only; native Rust work is not included. Exclusive
time subtracts direct child spans, but retains instrumentation and unlabelled
work. Timings overlap across aggregate labels: do not sum inclusive totals.
"""

    def __init__(self, enabled=False, *, max_intervals=512,
                 wall_clock=time.perf_counter_ns, cpu_clock=time.process_time_ns):
        if type(enabled) is not bool or type(max_intervals) is not int or not 0 <= max_intervals <= 512:
            raise ValueError('Invalid bounded timing-profile configuration')
        self.enabled = enabled
        self.max_intervals = max_intervals
        self._wall_clock, self._cpu_clock = wall_clock, cpu_clock
        self._origin = wall_clock() if enabled else None
        self._stack, self._intervals, self._totals = [], [], {}
        self._count = 0

    @contextmanager
    def span(self, label):
        if not self.enabled:
            yield
            return
        if not isinstance(label, str) or not label or len(label) > 96:
            raise ValueError('Timing labels must be short fixed strings')
        if label not in self._totals and len(self._totals) >= 64:
            raise ValueError('Too many timing labels')
        started_wall, started_cpu = self._wall_clock(), self._cpu_clock()
        sequence = self._count
        self._count += 1
        frame = [sequence, started_wall, started_cpu, 0, 0]
        parent = self._stack[-1][0] if self._stack else None
        self._stack.append(frame)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            ended_cpu, ended_wall = self._cpu_clock(), self._wall_clock()
            wall, cpu = ended_wall - started_wall, ended_cpu - started_cpu
            if self._stack.pop() is not frame:
                raise RuntimeError('Timing span stack corrupted')
            exclusive_wall, exclusive_cpu = max(0, wall - frame[3]), max(0, cpu - frame[4])
            if self._stack:
                self._stack[-1][3] += wall
                self._stack[-1][4] += cpu
            total = self._totals.setdefault(label, {
                'count': 0, 'failures': 0, 'wall_ns': 0, 'cpu_ns': 0,
                'exclusive_wall_ns': 0, 'exclusive_cpu_ns': 0, 'max_wall_ns': 0})
            total['count'] += 1
            total['failures'] += int(failed)
            total['wall_ns'] += wall
            total['cpu_ns'] += cpu
            total['exclusive_wall_ns'] += exclusive_wall
            total['exclusive_cpu_ns'] += exclusive_cpu
            total['max_wall_ns'] = max(total['max_wall_ns'], wall)
            if len(self._intervals) < self.max_intervals:
                self._intervals.append({
                    'sequence': sequence, 'parent': parent, 'label': label,
                    'start_ns': started_wall - self._origin, 'wall_ns': wall, 'cpu_ns': cpu,
                    'exclusive_wall_ns': exclusive_wall, 'exclusive_cpu_ns': exclusive_cpu,
                    'failed': failed})

    def wrap(self, label, function):
        if not self.enabled:
            return function

        @wraps(function)
        def measured(*args, **kwargs):
            with self.span(label):
                return function(*args, **kwargs)
        return measured

    def snapshot(self):
        if self._stack:
            raise RuntimeError('Cannot snapshot unfinished timing spans')
        return {
            'schema': SCHEMA, 'enabled': self.enabled,
            'wall_clock': 'perf_counter_ns', 'cpu_clock': 'process_time_ns',
            'cpu_scope': 'current_python_process_only_excludes_native_rust_and_parent',
            'times_are_diagnostic_not_trusted_deadline_receipts': True,
            'inclusive_labels_overlap': True,
            'exclusive_includes_unlabelled_work_and_instrumentation': True,
            'interval_count': self._count, 'stored_intervals': len(self._intervals),
            'omitted_intervals': self._count - len(self._intervals),
            'totals': {key: dict(value) for key, value in sorted(self._totals.items())},
            'intervals': sorted((dict(item) for item in self._intervals), key=lambda item: item['sequence']),
        }


def calibrate_timer_overhead(iterations=200):
    """Small startup-only bookkeeping estimate; never subtract it from clocks.

This is the average cost of an empty span, including loop/context/bookkeeping.
It is not the total instrumentation effect; alternating actual uninstrumented
requests must assess wrapper/serialization/interaction overhead independently.
"""
    if type(iterations) is not int or not 1 <= iterations <= 2000:
        raise ValueError('Bounded calibration requires 1..2000 iterations')
    profiler = DecisionProfiler(True)
    started = time.perf_counter_ns()
    for _ in range(iterations):
        with profiler.span('calibration_empty'):
            pass
    elapsed = time.perf_counter_ns() - started
    return {
        'schema': SCHEMA + '_overhead', 'iterations': iterations,
        'total_wall_ns': elapsed, 'empty_span_mean_wall_ns': elapsed / iterations,
        'includes_loop_context_and_recording': True,
        'does_not_estimate_entire_request_instrumentation_effect': True,
        'never_subtract_from_qualification_or_deadline_time': True,
    }


class _StreamProxy:
    """Delegate the exact existing stream, timing only existing I/O calls."""
    def __init__(self, original, profiler, direction):
        self._original, self._profiler, self._direction = original, profiler, direction

    def __getattr__(self, name):
        return getattr(self._original, name)

    def write(self, *args, **kwargs):
        with self._profiler.span('native_write'):
            return self._original.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        with self._profiler.span('native_flush'):
            return self._original.flush(*args, **kwargs)

    def readline(self, *args, **kwargs):
        with self._profiler.span('native_read_wait'):
            return self._original.readline(*args, **kwargs)


@contextmanager
def instrument_adapters(adapters, profiler):
    """Temporarily time the frozen delegation chain for one serial request.

The lock rejects nested/concurrent instrumentation because two frozen module
functions must be patched during a request. Exact original instance/class
bindings are restored even on a failed request. The model's five existing
top-level neural calls are wrapped, with no layer hooks or tensor rewriting.
"""
    if not profiler.enabled:
        yield
        return
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError('Nested or concurrent player instrumentation is forbidden')
    restorations = []
    patched = set()

    def replace(owner, name, replacement):
        key = (id(owner), name)
        if key in patched:
            return
        previous = vars(owner).get(name, _MISSING)
        setattr(owner, name, replacement)
        restorations.append((owner, name, previous))
        patched.add(key)

    def measure(owner, name, label):
        replace(owner, name, profiler.wrap(label, getattr(owner, name)))

    try:
        from tools.splendorust_search_scale_20261024 import adapters as frozen
        from tools.external_strength_screen_20261018_02 import splendorust_adapter as original
        from tools.external_strength_screen_20261018_02 import bridge

        measure(original.Opponent, '_infer', 'host_inference')
        measure(frozen, 'digest', 'callback_hashing')
        measure(original, 'digest', 'observer_seed_hashing')
        measure(bridge, 'encode_observation', 'observer_encoding')
        measure(bridge, 'main_actions', 'main_action_support')
        measure(bridge, 'resolve_compound', 'compound_resolution')
        for adapter in adapters.values():
            measure(adapter, 'choose', 'complete_choose')
            measure(adapter, 'exchange', 'native_exchange_inclusive')
            measure(adapter, '_infer', 'callback_inclusive')
            for name in ('project', 'blocks', 'norm', 'policy', 'value'):
                # Frozen host code invokes these directly, bypassing model.forward.
                measure(getattr(adapter.model, name), 'forward', 'model_' + name)
            replace(adapter.proc, 'stdin', _StreamProxy(adapter.proc.stdin, profiler, 'stdin'))
            replace(adapter.proc, 'stdout', _StreamProxy(adapter.proc.stdout, profiler, 'stdout'))
        yield
    finally:
        for owner, name, previous in reversed(restorations):
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)
        _PATCH_LOCK.release()
