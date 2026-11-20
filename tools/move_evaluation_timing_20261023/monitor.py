"""Selective Python-region wall timing without replacing scientific functions.

No model import or inference occurs here. Only ordinary Python code objects are
supported. Unselected descendants, native work, waiting and instrumentation
overhead remain charged to their nearest selected ancestor.
"""
from __future__ import annotations

import hashlib
import inspect
import marshal
from pathlib import Path
import sys
import threading
import time
from types import CodeType


class TimingIntegrityError(RuntimeError):
    """The selected-region measurement cannot be accepted."""


def _require(condition, message):
    if not condition:
        raise TimingIntegrityError(message)


def _code(function):
    value = getattr(function, '__code__', None)
    _require(isinstance(value, CodeType), 'Target must be an ordinary Python function or bound method')
    _require(not value.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR),
             'Generator and coroutine targets are unsupported')
    return value


def _sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class SelectedTimer:
    """Run one selected root once, preserving its result or original exception.

    ``targets`` must contain ``root_label`` bound to the exact code of ``fn``.
    Labels map one-to-one to code objects; recursive non-root calls are allowed.
    A timer is single-use, including a failed installation or application call.
    ``last_failure`` preserves small administrative diagnostics on failure.
    """

    TOOL_ID = 2

    def __init__(self, targets: dict[str, object], root_label='scientific_body'):
        _require(type(targets) is dict and bool(targets), 'Nonempty target dictionary required')
        _require(type(root_label) is str and root_label in targets, 'Selected root label is missing')
        _require(threading.current_thread() is threading.main_thread(), 'Timer must be constructed on the main thread')
        _require(hasattr(sys, 'monitoring'), 'CPython sys.monitoring is required')
        self.root_label = root_label
        self._owner = threading.get_ident()
        self._codes = {}
        self._by_id = {}
        self._manifest = []
        source_hashes = {}
        for label, function in targets.items():
            _require(type(label) is str and bool(label), 'Nonempty string target label required')
            code = _code(function)
            _require(id(code) not in self._by_id, 'Multiple labels refer to the same code object')
            path = Path(code.co_filename).resolve()
            _require(path.is_file(), 'Target source file does not exist: '+str(path))
            if path not in source_hashes:
                source_hashes[path] = _sha(path)
            self._codes[label] = code
            self._by_id[id(code)] = label
            self._manifest.append({'label': label, 'source_path': str(path),
                'source_sha256': source_hashes[path], 'qualname': code.co_qualname,
                'line': code.co_firstlineno,
                'code_sha256': hashlib.sha256(marshal.dumps(code)).hexdigest()})
        self._manifest.sort(key=lambda row: row['label'])
        self._rows = {label: {'label': label, 'calls': 0, 'returns': 0, 'unwinds': 0,
                              'inclusive_ns': 0, 'exclusive_ns': 0} for label in self._codes}
        self._stack = []
        self._errors = []
        self._events = 0
        self._callbacks = 0
        self._callback_ns = 0
        self._last_ns = None
        self._root_start = None
        self._root_end = None
        self._used = False
        self._claimed = False
        self._registered = []
        self._activated = []
        self.last_failure = None

    def _error(self, message):
        # Avoid unbounded failure logging if a foreign thread keeps calling.
        if len(self._errors) < 16:
            self._errors.append(message)

    def _selected(self, code):
        label = self._by_id.get(id(code))
        if label is None:
            return None
        if threading.get_ident() != self._owner:
            self._error('Selected function executed on another thread')
            return None
        return label

    def _credit(self, stamp):
        if self._stack:
            elapsed = stamp-self._last_ns
            if elapsed < 0:
                self._error('Nonmonotonic event timestamp')
            else:
                self._rows[self._stack[-1][0]]['exclusive_ns'] += elapsed

    def _on_start(self, code, offset):
        stamp = time.perf_counter_ns()
        self._callbacks += 1
        try:
            label = self._selected(code)
            if label is None:
                return
            self._events += 1
            self._rows[label]['calls'] += 1
            if label == self.root_label:
                if self._root_start is not None or self._stack:
                    self._error('Selected root must be called exactly once without recursion')
                else:
                    self._root_start = stamp
            elif not self._stack:
                self._error('Selected function ran outside the root region')
            self._credit(stamp)
            self._stack.append((label, id(code), stamp))
            self._last_ns = stamp
        finally:
            self._callback_ns += time.perf_counter_ns()-stamp

    def _end(self, code, stamp, kind):
        label = self._selected(code)
        if label is None:
            return
        self._events += 1
        self._rows[label][kind] += 1
        if not self._stack or self._stack[-1][1] != id(code):
            self._error('Unmatched selected '+kind+' event: '+label)
            return
        self._credit(stamp)
        _, _, started = self._stack.pop()
        elapsed = stamp-started
        if elapsed < 0:
            self._error('Negative inclusive region duration')
        else:
            self._rows[label]['inclusive_ns'] += elapsed
        self._last_ns = stamp
        if label == self.root_label:
            if self._stack or self._root_end is not None:
                self._error('Invalid root completion nesting')
            self._root_end = stamp

    def _on_return(self, code, offset, result):
        stamp = time.perf_counter_ns()
        self._callbacks += 1
        try:
            self._end(code, stamp, 'returns')
        finally:
            self._callback_ns += time.perf_counter_ns()-stamp

    def _on_unwind(self, code, offset, exception):
        # PY_UNWIND is not a local event. Ignore all non-target code objects.
        stamp = time.perf_counter_ns()
        self._callbacks += 1
        try:
            self._end(code, stamp, 'unwinds')
        finally:
            self._callback_ns += time.perf_counter_ns()-stamp

    def _install(self):
        monitoring = sys.monitoring
        _require(monitoring.get_tool(self.TOOL_ID) is None, 'Selected timer tool ID is already occupied')
        monitoring.use_tool_id(self.TOOL_ID, 'splendor_selected_python_regions_v1')
        self._claimed = True
        _require(monitoring.get_events(self.TOOL_ID) == 0, 'Claimed tool has stale global events')
        for code in self._codes.values():
            _require(monitoring.get_local_events(self.TOOL_ID, code) == 0, 'Claimed tool has stale local events')
        events = monitoring.events
        for event, callback in ((events.PY_START, self._on_start), (events.PY_RETURN, self._on_return),
                                (events.PY_UNWIND, self._on_unwind)):
            previous = monitoring.register_callback(self.TOOL_ID, event, callback)
            self._registered.append(event)
            _require(previous is None, 'Claimed tool has a stale callback')
        monitoring.set_events(self.TOOL_ID, events.PY_UNWIND)
        for code in self._codes.values():
            self._activated.append(code)
            monitoring.set_local_events(self.TOOL_ID, code, events.PY_START | events.PY_RETURN)

    def _cleanup(self):
        if not self._claimed:
            return
        monitoring = sys.monitoring
        # free_tool_id alone does not disable events or unregister callbacks.
        # Attempt every cleanup step even if a previous cleanup step fails.
        errors = []
        for code in self._activated:
            try:
                monitoring.set_local_events(self.TOOL_ID, code, 0)
            except BaseException as exc:
                errors.append('local cleanup: '+type(exc).__name__)
        try:
            monitoring.set_events(self.TOOL_ID, 0)
        except BaseException as exc:
            errors.append('global cleanup: '+type(exc).__name__)
        for event in self._registered:
            try:
                monitoring.register_callback(self.TOOL_ID, event, None)
            except BaseException as exc:
                errors.append('callback cleanup: '+type(exc).__name__)
        try:
            monitoring.free_tool_id(self.TOOL_ID)
        except BaseException as exc:
            errors.append('tool cleanup: '+type(exc).__name__)
        self._claimed = False
        for error in errors:
            self._error(error)

    def _failure(self, phase, exc):
        self.last_failure = {'status': 'timing_not_accepted', 'phase': phase,
            'exception_type': type(exc).__name__, 'exception_message': str(exc)[:1000],
            'errors': list(self._errors), 'open_regions': [frame[0] for frame in self._stack],
            'events': self._events, 'rows': [dict(self._rows[k]) for k in sorted(self._rows)]}

    def _measurement(self, body_ns):
        _require(not self._errors, '; '.join(self._errors))
        _require(not self._stack, 'Selected stack did not close')
        root = self._rows[self.root_label]
        _require(root['calls'] == root['returns'] == 1 and root['unwinds'] == 0, 'Exactly one successful root call required')
        _require(self._root_start is not None and self._root_end is not None, 'Missing selected root boundaries')
        for row in self._rows.values():
            _require(row['calls'] == row['returns']+row['unwinds'], 'Unbalanced events: '+row['label'])
            _require(row['inclusive_ns'] >= row['exclusive_ns'] >= 0, 'Invalid region timing: '+row['label'])
        region = self._root_end-self._root_start
        exclusive = sum(row['exclusive_ns'] for row in self._rows.values())
        _require(type(body_ns) is int and body_ns > 0 and region > 0, 'Nonpositive measured body')
        _require(exclusive == region, 'Exclusive regions do not exactly partition the root span')
        gap = body_ns-region
        tolerance = max(5_000_000, body_ns//100)
        _require(0 <= gap <= tolerance, 'Root/body boundary discrepancy exceeds max(5ms,1%)')
        _require(self._events == sum(row['calls']+row['returns']+row['unwinds'] for row in self._rows.values()),
                 'Callback/event count mismatch')
        return {'schema': 'selected_python_regions_timing_v1', 'body_seconds': body_ns/1_000_000_000,
            'body_ns': body_ns, 'root_region_ns': region, 'unassigned_body_ns': gap,
            'exclusive_sum_ns': exclusive, 'balanced': True, 'events': self._events,
            'rows': [dict(self._rows[k]) for k in sorted(self._rows)],
            'manifest': [dict(row) for row in self._manifest],
            'boundary_tolerance_ns': tolerance, 'callback_count': self._callbacks,
            'callback_body_ns': self._callback_ns, 'thread_scope': 'selected functions on the main thread only',
            'interpretation': {
                'timer': 'perf_counter_ns wall time',
                'inclusive_times_overlap': True,
                'exclusive_times_partition_selected_root': True,
                'exclusive_scope': 'Excludes selected descendants only; includes unselected Python, native work, waiting and instrumentation overhead.',
                'instrumentation_overhead_subtracted': False,
                'callback_body_ns_scope': 'Diagnostic callback entry-to-exit bookkeeping only; not total VM instrumentation overhead and not subtracted.',
                'native_child_internals_attributed': False,
                'body_excludes': 'installation, teardown and measurement serialization',
                'unassigned_body_ns_scope': 'Outer call boundary and root callback tail; explicitly retained, not redistributed.'}}

    def run(self, fn, *args, **kwargs):
        _require(not self._used, 'SelectedTimer is single-use')
        self._used = True
        _require(threading.get_ident() == self._owner and threading.current_thread() is threading.main_thread(),
                 'Timer must run on its constructing main thread')
        _require(_code(fn) is self._codes[self.root_label], 'Run callable does not match the selected root')
        phase = 'installation'
        try:
            self._install()
            phase = 'application'
            start = time.perf_counter_ns()
            try:
                scientific = fn(*args, **kwargs)
            finally:
                body_ns = time.perf_counter_ns()-start
        except BaseException as exc:
            self._failure(phase, exc)
            raise
        finally:
            self._cleanup()
            if self.last_failure is not None:
                self.last_failure['errors'] = list(self._errors)
        try:
            return scientific, self._measurement(body_ns)
        except BaseException as exc:
            self._failure('integrity_validation', exc)
            raise
