"""Bounded observations of the unchanged raw-reader supervised parent.

Trusted request/receipt files are never changed. The caller separately saves
last_parent_profile after request() returns. Spans overlap and are not charges
to subtract from the whole-decision clock.
"""
import os
import threading
import time

from tools.splendorust_timing_opt_20261024.common import CandidateStudyWorker

MAX_SPANS = 256
LABELS = ('initial_sample', 'periodic_sample', 'aggregate_guard', 'resource_guard_check',
          'resource_guard', 'stdin_write', 'stdin_flush', 'stdout_raw_read')


class _WriteProxy:
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream
    def write(self, value):
        return self.owner._timed('stdin_write', self.stream.write, value)
    def flush(self):
        return self.owner._timed('stdin_flush', self.stream.flush)
    def __getattr__(self, name):
        return getattr(self.stream, name)


class ProfiledStudyWorker(CandidateStudyWorker):
    def __init__(self, *args, **kwargs):
        self._component_context = None
        self._component_request_lock = threading.Lock()
        self.last_parent_profile = None
        aggregate = kwargs.get('aggregate_sample_guard')
        resource = kwargs.get('resource_guard')
        if aggregate is not None:
            kwargs['aggregate_sample_guard'] = lambda *a: self._timed('aggregate_guard', aggregate, *a)
        if resource is not None:
            kwargs['resource_guard'] = lambda: self._timed('resource_guard', resource)
        super().__init__(*args, **kwargs)
        self.process.stdin = _WriteProxy(self, self.process.stdin)

    def _record(self, context, label, started, ended, process_cpu, thread_cpu):
        if context is None:
            return
        row = dict(label=label, start_ns=started, end_ns=ended,
                   process_cpu_ns=process_cpu, thread_cpu_ns=thread_cpu)
        with context['lock']:
            total = context['totals'].setdefault(label, dict(calls=0, wall_ns=0, process_cpu_ns=0, thread_cpu_ns=0))
            total['calls'] += 1
            total['wall_ns'] += ended-started
            total['process_cpu_ns'] += process_cpu
            total['thread_cpu_ns'] += thread_cpu
            if len(context['spans']) < MAX_SPANS:
                context['spans'].append(row)
            else:
                context['dropped_spans'] += 1

    def _timed(self, label, function, *args, **kwargs):
        context = self._component_context
        if context is None:
            return function(*args, **kwargs)
        with context['lock']:
            active=context.setdefault('active_spans', {})
            active[label]=active.get(label,0)+1
        process_cpu, thread_cpu = time.process_time_ns(), time.thread_time_ns()
        started = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            ended = time.perf_counter_ns()
            self._record(context, label, started, ended,
                         time.process_time_ns()-process_cpu, time.thread_time_ns()-thread_cpu)
            with context['lock']:
                context['active_spans'][label]-=1

    def _sample(self):
        context = self._component_context
        label = 'periodic_sample'
        if context is not None:
            if context['samples'] == 0:
                label = 'initial_sample'
            context['samples'] += 1
        return self._timed(label, super()._sample)

    def _check_resource_guard(self):
        return self._timed('resource_guard_check', super()._check_resource_guard)

    def _receive(self, started, *args, **kwargs):
        context = self._component_context
        if context is not None:
            context['request_start_ns'] = started
        return super()._receive(started, *args, **kwargs)

    def _stdout_reader(self):
        try:
            while True:
                process_cpu, thread_cpu = time.process_time_ns(), time.thread_time_ns()
                read_started = time.perf_counter_ns()
                raw = self.process.stdout.readline(self.max_message_bytes + 1)
                received = time.perf_counter_ns()  # Exact frozen read -> trusted timestamp adjacency.
                context = self._component_context
                self._record(context, 'stdout_raw_read', read_started, received,
                             time.process_time_ns()-process_cpu, time.thread_time_ns()-thread_cpu)
                if context is not None:
                    with context['lock']:
                        if len(context['raw_arrivals_ns']) < MAX_SPANS:
                            context['raw_arrivals_ns'].append(received)
                if not raw:
                    self._put(('eof', received, None))
                    return
                if len(raw) > self.max_message_bytes or not raw.endswith(b'\n'):
                    self._log('stdout.log', raw[:4096])
                    self._put(('malformed', received, 'oversized or unterminated message'))
                    return
                self._log('stdout.log', raw)
                self._put(('line', received, raw))
        except (OSError, ValueError) as error:
            self._put(('reader_error', time.perf_counter_ns(), str(error)))

    def request(self, *args, **kwargs):
        if not self._component_request_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent requests are forbidden')
        context = dict(lock=threading.Lock(), spans=[], totals={}, samples=0, dropped_spans=0,
                       request_start_ns=None, raw_arrivals_ns=[],active_spans={})
        self._component_context = context
        process_cpu, thread_cpu = time.process_time_ns(), time.thread_time_ns()
        started = time.perf_counter_ns()
        receipt = None
        try:
            receipt = super().request(*args, **kwargs)
            return receipt
        finally:
            ended = time.perf_counter_ns()
            process_delta, thread_delta = time.process_time_ns()-process_cpu, time.thread_time_ns()-thread_cpu
            self._component_context = None
            with context['lock']:
                actual_start = context['request_start_ns']
                elapsed = None if receipt is None else receipt.get('receipt_elapsed_seconds')
                reconstructed = None if actual_start is None or elapsed is None else actual_start+round(elapsed*1e9)
                arrival = next((stamp for stamp in context['raw_arrivals_ns'] if stamp == reconstructed), None)
                self.last_parent_profile = dict(schema='splendorust_component_parent_v1',
                    request_id=None if receipt is None else receipt['request_id'], clock='perf_counter_ns',
                    pid=os.getpid(), child_pid=self.process.pid,
                    process_cpu_clock='process_time_ns', thread_cpu_clock='thread_time_ns',
                    request_start_ns=actual_start, receipt_arrival_ns=arrival,
                    receipt_reconstructed_arrival_ns=reconstructed,
                    spans=list(context['spans']), totals={label:dict(value) for label,value in context['totals'].items()},
                    pending_spans_at_snapshot={label:count for label,count in context['active_spans'].items() if count},
                    raw_arrivals_ns=list(context['raw_arrivals_ns']), dropped_spans=context['dropped_spans'],
                    maximum_spans=MAX_SPANS, wrapper_start_ns=started, wrapper_end_ns=ended,
                    wrapper_process_cpu_ns=process_delta, wrapper_thread_cpu_ns=thread_delta,
                    scope='Nested spans include pre-dispatch and post-arrival work. Raw read can begin before request and includes waiting. Clip/union within trusted window; never subtract overlapping totals. Wrapper CPU bounds are wider than the exact trusted window.')
            self._component_request_lock.release()
