"""Minimal boundary observations around the exact original supervised parent.

The raw reader, sampling, guards, deadline receipt and cleanup are inherited
unchanged. Capture existing raw arrival events; add no reader or guard timers.
Hashes are calculated only after the original request has returned.
"""
import hashlib
import os
import threading
import time

from tools.splendorust_timing_opt_20261024.common import CandidateStudyWorker

SCHEMA = 'splendorust_boundary_timing_parent_v1'
MAX_INPUT_CAPTURE_BYTES = 1024 * 1024
MAX_RESPONSE_CAPTURE_BYTES = 8 * 1024 * 1024
MAX_ARRIVAL_EVENTS = 16


class _BoundaryWriteCapture:
    """Keep actual accepted bytes and aggregate write/flush boundaries only."""
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream

    def _begin(self, label):
        context = self.owner._boundary_context
        if context is None:
            return None
        started = time.perf_counter_ns()
        with context['lock']:
            if context['closed']:
                return None
            if context[label + '_start_ns'] is None:
                context[label + '_start_ns'] = started
            context[label + '_calls'] += 1
            context['pending_' + label] += 1
        return context

    @staticmethod
    def _end(context, label, *, failed=False, accepted=None):
        if context is None:
            return
        ended = time.perf_counter_ns()
        with context['lock']:
            if context['closed']:
                return
            context['pending_' + label] -= 1
            context[label + '_end_ns'] = ended
            context[label + '_failures'] += int(failed)
            if accepted is not None:
                value, written = accepted
                if type(written) is int and written > 0:
                    available = MAX_INPUT_CAPTURE_BYTES - len(context['raw'])
                    context['raw'].extend(value[:min(written, available)])
                    context['raw_overflow'] |= written > available or written > len(value)

    def write(self, value):
        context = self._begin('write')
        failed, written = True, None
        try:
            written = self.stream.write(value)
            failed = False
            return written
        finally:
            self._end(context, 'write', failed=failed, accepted=(value, written))

    def flush(self):
        context = self._begin('flush')
        failed = True
        try:
            result = self.stream.flush()
            failed = False
            return result
        finally:
            self._end(context, 'flush', failed=failed)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class ProfiledStudyWorker(CandidateStudyWorker):
    def __init__(self, *args, **kwargs):
        self._boundary_context = None
        self._boundary_request_lock = threading.Lock()
        self.last_parent_profile = None
        # In particular, do not wrap aggregate/resource guards or sampling.
        super().__init__(*args, **kwargs)
        self.process.stdin = _BoundaryWriteCapture(self, self.process.stdin)

    def _receive(self, started, *args, **kwargs):
        context = self._boundary_context
        if context is not None:
            context['request_start_ns'] = started
        return super()._receive(started, *args, **kwargs)

    def _put(self, event):
        # _stdout_reader remains the exact inherited raw readline -> immediate
        # trusted timestamp implementation. This event already has that stamp;
        # retaining its bytes adds neither a clock read nor a hash to the reader.
        context = self._boundary_context
        if context is not None and event[0] == 'line':
            with context['lock']:
                if not context['closed']:
                    if len(context['arrivals']) < MAX_ARRIVAL_EVENTS:
                        raw = event[2]
                        overflow = len(raw) > MAX_RESPONSE_CAPTURE_BYTES
                        context['arrivals'].append((event[1], raw[:MAX_RESPONSE_CAPTURE_BYTES], overflow))
                    else:
                        context['dropped_arrivals'] += 1
        return super()._put(event)

    def request(self, *args, **kwargs):
        if not self._boundary_request_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent requests are forbidden')
        context = dict(lock=threading.Lock(), closed=False, request_start_ns=None,
            raw=bytearray(), raw_overflow=False, arrivals=[], dropped_arrivals=0,
            write_start_ns=None, write_end_ns=None, write_calls=0, write_failures=0,
            flush_start_ns=None, flush_end_ns=None, flush_calls=0, flush_failures=0,
            pending_write=0, pending_flush=0)
        self._boundary_context = context
        receipt = None
        try:
            receipt = super().request(*args, **kwargs)
            return receipt
        finally:
            try:
                with context['lock']:
                    context['closed'] = True
                    raw = bytes(context['raw'])
                    arrivals = tuple(context['arrivals'])
                    fields = {key: value for key, value in context.items()
                              if key not in ('lock', 'raw', 'arrivals')}
                    snapshot = time.perf_counter_ns()
                self._boundary_context = None
                start = fields['request_start_ns']
                elapsed = None if receipt is None else receipt.get('receipt_elapsed_seconds')
                reconstructed = None if start is None or elapsed is None else start + round(elapsed * 1e9)
                arrival = next((row for row in arrivals if row[0] == reconstructed), None)
                flush_complete = (fields['flush_end_ns'] if fields['flush_calls'] and
                    not fields['pending_flush'] and not fields['flush_failures'] else None)
                self.last_parent_profile = dict(schema=SCHEMA,
                    request_id=None if receipt is None else receipt['request_id'],
                    pid=os.getpid(), child_pid=self.process.pid, clock='perf_counter_ns',
                    clock_implementation=time.get_clock_info('perf_counter').implementation,
                    request_start_ns=start,
                    receipt_arrival_ns=None if arrival is None else arrival[0],
                    receipt_reconstructed_arrival_ns=reconstructed,
                    raw_arrivals_ns=[row[0] for row in arrivals],
                    maximum_arrival_events=MAX_ARRIVAL_EVENTS,
                    dropped_arrival_events=fields['dropped_arrivals'],
                    stdin_write_start_ns=fields['write_start_ns'],
                    stdin_write_end_ns=fields['write_end_ns'],
                    stdin_flush_start_ns=fields['flush_start_ns'],
                    stdin_flush_end_ns=fields['flush_end_ns'],
                    dispatch_complete_ns=flush_complete,
                    dispatch_write_calls=fields['write_calls'], dispatch_flush_calls=fields['flush_calls'],
                    dispatch_write_failures=fields['write_failures'], dispatch_flush_failures=fields['flush_failures'],
                    pending_writes=fields['pending_write'], pending_flushes=fields['pending_flush'],
                    request_raw_sha256=hashlib.sha256(raw).hexdigest(), request_raw_bytes=len(raw),
                    request_raw_complete=bool(raw.endswith(b'\n') and not fields['raw_overflow']
                        and not fields['pending_write'] and not fields['write_failures']),
                    request_raw_overflow=fields['raw_overflow'],
                    request_capture_maximum_bytes=MAX_INPUT_CAPTURE_BYTES,
                    response_raw_sha256=None if arrival is None else hashlib.sha256(arrival[1]).hexdigest(),
                    response_raw_bytes=None if arrival is None else len(arrival[1]),
                    response_capture_overflow=None if arrival is None else arrival[2],
                    response_capture_maximum_bytes=MAX_RESPONSE_CAPTURE_BYTES,
                    capture_snapshot_ns=snapshot,
                    times_are_diagnostic_not_trusted_deadline_receipts=True,
                    measurement_units='nanoseconds',
                    raw_reader_semantics='unchanged raw FileIO.readline then immediate trusted perf_counter_ns',
                    child_read_may_precede_dispatch_completion=True,
                    child_post_flush_may_follow_trusted_arrival=True,
                    hashes_calculated_after_original_request_returns=True,
                    dispatch_markers_may_be_partial_if_pending=True)
            finally:
                self._boundary_context = None
                self._boundary_request_lock.release()
