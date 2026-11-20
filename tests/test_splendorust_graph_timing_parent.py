"""Raw-clock, dispatch and failed-writer preservation without models."""
import ast
import inspect
import io
import queue
import threading
import textwrap
from types import SimpleNamespace

import pytest

from tools.splendorust_graph_timing_20261025 import parent_profile as p
from tools.splendorust_component_timing_20261025 import parent_profile as inherited


def worker():
    value = p.ProfiledStudyWorker.__new__(p.ProfiledStudyWorker)
    value._component_context = None
    value._component_request_lock = threading.Lock()
    value._transport_request_lock = threading.Lock()
    value._transport_capture = None
    value.last_parent_profile = None
    value.process = SimpleNamespace(pid=999, stdin=io.BytesIO())
    return value


def context():
    return dict(lock=threading.Lock(), spans=[], totals={}, samples=0, dropped_spans=0,
                request_start_ns=None, raw_arrivals_ns=[], active_spans={})


def test_raw_reader_and_cleanup_are_exact_inherited_methods():
    for name in ('_stdout_reader', '_receive', '_sample', '_check_resource_guard', 'close', '_log', '_put'):
        assert getattr(p.ProfiledStudyWorker, name) is getattr(inherited.ProfiledStudyWorker, name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(p.ProfiledStudyWorker._stdout_reader)))
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While))
    raw = next(node for node in loop.body if isinstance(node, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'raw' for t in node.targets))
    assert ast.unparse(raw.value) == 'self.process.stdout.readline(self.max_message_bytes + 1)'
    assert ast.unparse(loop.body[loop.body.index(raw) + 1]) == 'received = time.perf_counter_ns()'


def test_raw_reader_keeps_arrival_before_logging_and_queueing(monkeypatch):
    child = worker(); child._component_context = context()
    child.process.stdout = io.BytesIO(b'exact\n'); child.max_message_bytes = 1024
    child._queue = queue.Queue(); calls = []
    child._log = lambda name, data: calls.append(('log', data))
    child._put = lambda event: (calls.append(('put', event)), child._queue.put(event))
    ticks = iter([1, 2, 3, 4]); monkeypatch.setattr(inherited.time, 'perf_counter_ns', lambda: next(ticks))
    child._stdout_reader()
    assert calls == [('log', b'exact\n'), ('put', ('line', 2, b'exact\n')), ('put', ('eof', 4, None))]


@pytest.mark.parametrize('status', ['ok', 'timeout', 'worker_error'])
def test_exact_receipt_returned_and_dispatch_markers_only_in_sidecar(monkeypatch, status):
    child = worker(); receipt = {'request_id': 'request-00000001', 'receipt_elapsed_seconds': 10e-9, 'status': status}
    def request(self, *args, **kwargs):
        c = self._component_context; c['request_start_ns'] = 100; c['raw_arrivals_ns'].append(110)
        self._record(c, 'stdin_write', 101, 105, 1, 1)
        self._record(c, 'stdin_write', 106, 107, 1, 1)
        self._record(c, 'stdin_flush', 107, 112, 1, 1)
        return receipt
    monkeypatch.setattr(inherited.CandidateStudyWorker, 'request', request)
    assert child.request({}, 1.) is receipt
    assert set(receipt) == {'request_id', 'receipt_elapsed_seconds', 'status'}
    value = child.last_parent_profile
    assert value['request_id'] == receipt['request_id'] and value['schema'] == p.SCHEMA
    assert value['request_start_ns'] == 100 and value['receipt_arrival_ns'] == 110
    assert value['stdin_write_start_ns'] == 101 and value['stdin_write_end_ns'] == 107
    assert value['dispatch_complete_ns'] == 112  # May follow child read / arrival.
    assert value['child_read_may_precede_dispatch_completion'] is True


def test_pending_flush_is_not_waited_for_or_falsely_marked_complete(monkeypatch):
    child = worker(); entered = threading.Event(); release = threading.Event(); threads = []
    def flush(): entered.set(); assert release.wait(2)
    def request(self, *args, **kwargs):
        self._component_context['request_start_ns'] = 100
        thread = threading.Thread(target=lambda: self._timed('stdin_flush', flush))
        threads.append(thread); thread.start(); assert entered.wait(1)
        return {'request_id': 'request-00000001', 'receipt_elapsed_seconds': None, 'status': 'timeout'}
    monkeypatch.setattr(inherited.CandidateStudyWorker, 'request', request)
    try:
        child.request({}, 1.); value = child.last_parent_profile
        assert value['dispatch_complete_ns'] is None
        assert value['pending_spans_at_snapshot'] == {'stdin_flush': 1}
        release.set(); threads[0].join(2)
        assert value['dispatch_complete_ns'] is None and not value['spans']
    finally:
        release.set()
        for thread in threads: thread.join(2)


def test_concurrent_rejection_does_not_replace_existing_profile():
    child = worker(); value = {'preserved': True}; child.last_parent_profile = value
    child._component_request_lock.acquire()
    with pytest.raises(RuntimeError, match='Concurrent'): child.request({}, 1.)
    assert child.last_parent_profile is value and value == {'preserved': True}


def test_buffers_remain_bounded_and_keep_complete_totals():
    child = worker(); value = context()
    for _ in range(300): child._record(value, 'stdin_write', 10, 20, 3, 2)
    assert len(value['spans']) == p.MAX_SPANS == 256
    assert value['dropped_spans'] == 44
    assert value['totals']['stdin_write']['calls'] == 300
    assert value['totals']['stdin_write']['wall_ns'] == 3000


def test_error_preserves_separate_evidence_and_releases_request_lock(monkeypatch):
    child = worker(); error = RuntimeError('original failure')
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(inherited.CandidateStudyWorker, 'request', fail)
    with pytest.raises(RuntimeError) as caught: child.request({}, 1.)
    assert caught.value is error
    assert child.last_parent_profile['schema'] == p.SCHEMA
    assert child.last_parent_profile['request_id'] is None
    assert child._component_context is None
    assert child._component_request_lock.acquire(blocking=False)


def test_exact_accepted_stdin_bytes_capture_preserves_partial_write_and_flush(monkeypatch):
    import hashlib
    child = worker(); calls = []
    raw = SimpleNamespace(write=lambda data: (calls.append(bytes(data)) or min(2, len(data))),
                          flush=lambda: calls.append('flush'))
    child.process.stdin = p._AcceptedWriteCapture(child, raw)
    def request(self, *args, **kwargs):
        data = b'{"x":1}\n'; remaining = memoryview(data)
        while remaining:
            written = self.process.stdin.write(remaining); remaining = remaining[written:]
        self.process.stdin.flush()
        self._component_context['request_start_ns'] = 100
        return {'request_id': 'request-00000001', 'receipt_elapsed_seconds': None, 'status': 'timeout'}
    monkeypatch.setattr(inherited.CandidateStudyWorker, 'request', request)
    child.request({}, 1.)
    value = child.last_parent_profile
    assert calls == [b'{"x":1}\n', b'x":1}\n', b':1}\n', b'}\n', 'flush']
    assert value['request_raw_sha256'] == hashlib.sha256(b'{"x":1}\n').hexdigest()
    assert value['request_raw_bytes'] == 8 and value['request_raw_complete'] is True
    assert value['request_raw_overflow'] is False


def test_capture_bound_does_not_change_underlying_write_result():
    child = worker(); capture = dict(lock=threading.Lock(), raw=bytearray(), overflow=False, closed=False)
    child._transport_capture = capture
    payload = b'x' * (p.MAX_INPUT_CAPTURE_BYTES + 1)
    proxy = p._AcceptedWriteCapture(child, SimpleNamespace(write=lambda value: len(value)))
    assert proxy.write(payload) == len(payload)
    assert len(capture['raw']) == p.MAX_INPUT_CAPTURE_BYTES and capture['overflow'] is True
    capture['closed'] = True
    assert proxy.write(b'ignored') == 7 and len(capture['raw']) == p.MAX_INPUT_CAPTURE_BYTES
