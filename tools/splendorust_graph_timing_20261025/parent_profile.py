"""Bounded parent observations using the previously qualified raw reader.

The inherited reader retains the exact raw readline -> trusted timestamp
adjacency. This version derives explicit dispatch markers after request(),
without copying or weakening the trusted clock/receipt implementation.
"""
import hashlib
import threading
import time

from tools.splendorust_component_timing_20261025.parent_profile import (
    ProfiledStudyWorker as _OriginalProfiledStudyWorker, LABELS, MAX_SPANS,
)

SCHEMA = 'splendorust_graph_timing_parent_v1'
MAX_INPUT_CAPTURE_BYTES = 1024 * 1024


class _AcceptedWriteCapture:
    """Observe bytes actually accepted; retain the inherited writer behavior."""
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream

    def write(self, value):
        capture = self.owner._transport_capture
        written = self.stream.write(value)
        if capture is not None and type(written) is int and written > 0:
            with capture['lock']:
                if not capture['closed']:
                    available = MAX_INPUT_CAPTURE_BYTES - len(capture['raw'])
                    capture['raw'].extend(value[:min(written, available)])
                    capture['overflow'] |= written > available or written > len(value)
        return written

    def flush(self):
        return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class ProfiledStudyWorker(_OriginalProfiledStudyWorker):
    def __init__(self, *args, **kwargs):
        self._transport_request_lock = threading.Lock()
        self._transport_capture = None
        super().__init__(*args, **kwargs)
        self.process.stdin = _AcceptedWriteCapture(self, self.process.stdin)

    def request(self, *args, **kwargs):
        if not self._transport_request_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent requests are forbidden')
        previous = self.last_parent_profile
        capture = dict(lock=threading.Lock(), raw=bytearray(), overflow=False, closed=False)
        self._transport_capture = capture
        try:
            return super().request(*args, **kwargs)
        finally:
            try:
                with capture['lock']:
                    capture['closed'] = True
                    raw = bytes(capture['raw'])
                    overflow = capture['overflow']
                self._transport_capture = None
                value = self.last_parent_profile
                if value is not None and value is not previous:
                    value['schema'] = SCHEMA
                    for component in ('stdin_write', 'stdin_flush'):
                        rows = [row for row in value['spans'] if row['label'] == component]
                        value[component + '_start_ns'] = min((row['start_ns'] for row in rows), default=None)
                        value[component + '_end_ns'] = max((row['end_ns'] for row in rows), default=None)
                    value['dispatch_complete_ns'] = value['stdin_flush_end_ns']
                    value['request_raw_sha256'] = hashlib.sha256(raw).hexdigest()
                    value['request_raw_bytes'] = len(raw)
                    value['request_raw_overflow'] = overflow
                    value['request_raw_complete'] = bool(raw.endswith(b'\n') and not overflow and
                        not value['pending_spans_at_snapshot'].get('stdin_write'))
                    value['request_capture_maximum_bytes'] = MAX_INPUT_CAPTURE_BYTES
                    value['profile_snapshot_ns'] = time.perf_counter_ns()
                    value['times_are_diagnostic_not_trusted_deadline_receipts'] = True
                    value['measurement_units'] = 'nanoseconds'
                    value['raw_reader_semantics'] = 'unchanged raw FileIO.readline then immediate trusted perf_counter_ns'
                    value['child_read_may_precede_dispatch_completion'] = True
                    value['child_post_flush_may_follow_trusted_arrival'] = True
                    value['dispatch_markers_may_be_partial_if_spans_dropped_or_pending'] = True
            finally:
                self._transport_request_lock.release()
