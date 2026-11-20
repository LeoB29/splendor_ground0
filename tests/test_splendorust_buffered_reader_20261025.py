"""Model-free exact-byte transport, boundary clocks and hostile framing checks."""
from collections import deque
import gzip
import io
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tools.local_player_qualification_128 import clock
from tools.splendorust_buffered_reader_20261025 import reader

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT/'outputs/splendorust_search_scale_opt_20261024_01/workers/operational_g0_challenger/stdout.log.gz'
MAX_MESSAGE = 4*1024*1024


class Chunks:
    def __init__(self, chunks, before_read=None):
        self.chunks = deque(chunks)
        self.requests = []
        self.before_read = before_read
    def read(self, size):
        assert size > 0
        self.requests.append(size)
        if self.before_read:
            self.before_read()
        if not self.chunks:
            return b''
        chunk = self.chunks.popleft()
        if isinstance(chunk, Exception):
            raise chunk
        if len(chunk) > size:
            self.chunks.appendleft(chunk[size:])
        return chunk[:size]


def harness(stream, *, maximum=MAX_MESSAGE):
    worker = reader.BufferedStudyWorker.__new__(reader.BufferedStudyWorker)
    worker.process = SimpleNamespace(stdout=stream)
    worker.max_message_bytes = maximum
    worker._failure = worker._failure_detail = None
    worker._queue = queue.Queue(maxsize=128)
    worker.logged = []
    worker._log = lambda name, raw: worker.logged.append((name, raw))
    worker.absolute_deadline = None
    worker._sample = lambda: {'cpu_seconds': 0., 'rss_bytes': 0}
    worker._check_resource_guard = lambda: None
    return worker


def events(worker):
    result = []
    while not worker._queue.empty():
        result.append(worker._queue.get_nowait())
    return result


def packet(result='ok', request='request-00000001'):
    return (json.dumps({'type': 'response', 'request_id': request, 'result': result},
                       separators=(',', ':'))+'\n').encode()


def test_only_stdout_reader_is_overridden():
    methods = {name for name, value in reader.BufferedStudyWorker.__dict__.items() if callable(value)}
    assert methods == {'_stdout_reader'}
    for name in ('__init__', '_receive', 'request', '_stderr_reader', '_sample', '_put',
                 '_log', '_check_resource_guard', 'close', '__enter__', '__exit__'):
        assert getattr(reader.BufferedStudyWorker, name) is getattr(reader.CandidateStudyWorker, name)


@pytest.mark.parametrize('chunks', [[b'a\nb\n'], [b'a', b'\nb', b'\n'],
                                  [b'a\n', b'b\n'], [b'a', b'\n', b'b', b'\n']])
def test_fragmented_and_coalesced_lines_preserve_exact_bytes(chunks):
    stream = Chunks(chunks)
    worker = harness(stream)
    worker._stdout_reader()
    got = events(worker)
    assert [(kind, raw) for kind, stamp, raw in got] == [('line', b'a\n'), ('line', b'b\n'), ('eof', None)]
    assert worker.logged == [('stdout.log', b'a\n'), ('stdout.log', b'b\n')]
    assert all(size <= reader.READ_CHUNK_BYTES for size in stream.requests)


def test_chunk_arrival_precedes_framing_logs_and_all_coalesced_frames(monkeypatch):
    now = SimpleNamespace(value=0)
    def advance_read():
        now.value += 10
    stream = Chunks([b'a\nb\n'], before_read=advance_read)
    worker = harness(stream)
    monkeypatch.setattr(reader.time, 'perf_counter_ns', lambda: now.value)
    order = []
    original_put = worker._put
    def log(name, raw):
        order.append(('log', raw))
        now.value += 100
    def put(event):
        order.append(('put', event[2]))
        original_put(event)
    worker._log, worker._put = log, put
    worker._stdout_reader()
    got = events(worker)
    assert [event[1] for event in got] == [10, 10, 220]
    assert order == [('log', b'a\n'), ('put', b'a\n'), ('log', b'b\n'), ('put', b'b\n'), ('put', None)]


def test_newline_timestamp_uses_the_chunk_that_completes_partial_frame(monkeypatch):
    stamps = iter([10, 20, 30])
    monkeypatch.setattr(reader.time, 'perf_counter_ns', lambda: next(stamps))
    worker = harness(Chunks([b'ab', b'c\n']))
    worker._stdout_reader()
    assert events(worker) == [('line', 20, b'abc\n'), ('eof', 30, None)]


@pytest.mark.parametrize('length,newline,expected', [(15, True, 'line'), (16, True, 'malformed'),
    (15, False, 'malformed'), (16, False, 'malformed'), (17, False, 'malformed')])
def test_exact_small_message_limit_and_eof(length, newline, expected):
    data = b'x'*length + (b'\n' if newline else b'')
    worker = harness(Chunks([data]), maximum=16)
    worker._stdout_reader()
    got = events(worker)
    assert got[0][0] == expected
    assert worker.logged[0][1] == data[:17]
    if expected == 'line':
        assert got[-1][0] == 'eof'
    else:
        assert len(got) == 1 and got[0][2] == 'oversized or unterminated message'


def test_exact_four_mib_limit_and_bounded_reads():
    data = b'x'*(MAX_MESSAGE-1)+b'\n'
    stream = Chunks([data])
    worker = harness(stream)
    worker._stdout_reader()
    assert [(kind, raw) for kind, stamp, raw in events(worker)] == [('line', data), ('eof', None)]
    assert max(stream.requests) == reader.READ_CHUNK_BYTES
    too_large = harness(Chunks([b'x'*MAX_MESSAGE+b'\n']))
    too_large._stdout_reader()
    assert events(too_large)[0][0] == 'malformed'
    assert too_large.logged == [('stdout.log', b'x'*4096)]


def test_unterminated_eof_logs_only_existing_4096_byte_prefix(monkeypatch):
    stamps = iter([10, 20])
    monkeypatch.setattr(reader.time, 'perf_counter_ns', lambda: next(stamps))
    worker = harness(Chunks([b'x'*5000]))
    worker._stdout_reader()
    assert events(worker) == [('malformed', 20, 'oversized or unterminated message')]
    assert worker.logged == [('stdout.log', b'x'*4096)]


@pytest.mark.parametrize('error', [OSError('broken read'), ValueError('closed read')])
def test_reader_errors_are_preserved_without_partial_frame_log(error):
    worker = harness(Chunks([b'partial', error]))
    worker._stdout_reader()
    got = events(worker)
    assert got[0][0] == 'reader_error' and got[0][2] == str(error)
    assert worker.logged == []


def test_empty_eof_and_blank_lines_keep_original_framing():
    worker = harness(Chunks([b'\n']))
    worker._stdout_reader()
    assert [(kind, raw) for kind, stamp, raw in events(worker)] == [('line', b'\n'), ('eof', None)]
    empty = harness(Chunks([]))
    empty._stdout_reader()
    assert events(empty)[0][0] == 'eof' and empty.logged == []


def test_queue_flood_stays_fail_closed_with_original_logging_order():
    worker = harness(Chunks([b'x\n'*130]))
    worker._stdout_reader()
    assert worker._failure == 'message_flood'
    assert worker._queue.qsize() == 128
    assert worker.logged == [('stdout.log', b'x\n')]*130


@pytest.mark.parametrize('offset,status', [(-1, 'unsolicited'), (0, 'ok'), (1000000000, 'ok'),
                                         (1000000001, 'timeout')])
def test_inherited_deadline_and_unsolicited_checks_cannot_be_bypassed(monkeypatch, offset, status):
    begun = 2000000000
    worker = harness(Chunks([]))
    worker._put(('line', begun+offset, packet()))
    monkeypatch.setattr(reader.time, 'perf_counter_ns', lambda: begun+1000000010)
    seen = []
    receipt = worker._receive(begun, 1., 'response', 'request-00000001', lambda value: seen.append(value))
    assert receipt['status'] == status and receipt['accepted'] is (status == 'ok')
    assert seen == (['ok'] if status == 'ok' else [])


@pytest.mark.parametrize('raw,status', [(b'not-json\n', 'malformed'), (b'\xff\n', 'malformed'),
    (b'{}\n', 'malformed'), (packet(request='wrong'), 'wrong_request'),
    (b'{"type":"error","error":"failure"}\n', 'worker_error')])
def test_decode_and_protocol_rejections_still_use_inherited_receiver(monkeypatch, raw, status):
    worker = harness(Chunks([]))
    worker._put(('line', 10, raw))
    monkeypatch.setattr(reader.time, 'perf_counter_ns', lambda: 20)
    receipt = worker._receive(0, 1., 'response', 'request-00000001', None)
    assert receipt['status'] == status and not receipt['accepted']


def test_log_failure_remains_primary_before_receiver_acceptance(monkeypatch):
    worker = harness(Chunks([packet()]))
    def fail_log(name, raw):
        worker._failure = 'log_limit'
    worker._log = fail_log
    monkeypatch.setattr(reader.time, 'perf_counter_ns', lambda: 10)
    worker._stdout_reader()
    receipt = worker._receive(0, 1., 'response', 'request-00000001', None)
    assert receipt['status'] == 'log_limit' and not receipt['accepted']


class CountedRaw(io.FileIO):
    def __init__(self, fd):
        super().__init__(fd, 'rb', closefd=True)
        self.read_sizes = []
    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


def pipe_fixture(payload, *, sizes=None, keep_open_until_received=False, method=None):
    """Bounded real-pipe test with explicit writer/read-thread cleanup."""
    read_fd, write_fd = os.pipe()
    stream = CountedRaw(read_fd)
    worker = harness(stream)
    release = threading.Event()
    writer_errors = []
    def write():
        try:
            offset = 0
            chunks = sizes or [len(payload)]
            for count in chunks:
                end = min(len(payload), offset+count)
                while offset < end:
                    offset += os.write(write_fd, payload[offset:end])
            while offset < len(payload):
                offset += os.write(write_fd, payload[offset:])
            if keep_open_until_received:
                if not release.wait(5):
                    raise TimeoutError('Reader waited for full chunk or EOF')
        except BaseException as error:
            writer_errors.append(error)
        finally:
            os.close(write_fd)
    writer_thread = threading.Thread(target=write, daemon=True)
    read_thread = threading.Thread(target=worker._stdout_reader if method is None else lambda: method(worker), daemon=True)
    writer_thread.start()
    read_thread.start()
    first = None
    try:
        if keep_open_until_received:
            first = worker._queue.get(timeout=4)
            assert first[0] == 'line' and writer_thread.is_alive()
        release.set()
        writer_thread.join(5)
        read_thread.join(5)
        assert not writer_thread.is_alive() and not read_thread.is_alive()
        assert not writer_errors
        result = ([] if first is None else [first])+events(worker)
        return worker, result, list(stream.read_sizes)
    finally:
        release.set()
        writer_thread.join(5)
        # Closing the writer supplies EOF to a live reader. Join before closing
        # its FileIO so no read competes with close on a shared descriptor.
        read_thread.join(5)
        stream.close()
        assert not writer_thread.is_alive() and not read_thread.is_alive()


def test_real_pipe_returns_short_available_frame_before_writer_closes():
    raw = packet('available without EOF')
    worker, got, sizes = pipe_fixture(raw, keep_open_until_received=True)
    assert got[0][2] == raw and got[-1][0] == 'eof'
    assert sizes[0] == reader.READ_CHUNK_BYTES
    assert worker.logged == [('stdout.log', raw)]


def test_real_pipe_large_partial_writes_preserve_multiple_frames():
    frames = [packet('x'*70000), packet('second'), packet('third')]
    worker, got, sizes = pipe_fixture(b''.join(frames), sizes=[1, 7, 1023, 4095, 65000])
    assert [raw for kind, stamp, raw in got if kind == 'line'] == frames
    assert worker.logged == [('stdout.log', raw) for raw in frames]
    assert got[-1][0] == 'eof' and all(0 < size <= reader.READ_CHUNK_BYTES for size in sizes)


def test_real_pipe_exact_archived_slow_responses_match_original_reader():
    with gzip.open(ARCHIVE, 'rb') as archive:
        raw = archive.read(16*1024*1024+1)
    assert len(raw) <= 16*1024*1024 and raw.endswith(b'\n')
    wanted = {f'request-{index:08d}' for index in (22, 25, 26)}
    frames = [line for line in raw.splitlines(keepends=True)
              if json.loads(line).get('request_id') in wanted]
    assert len(frames) == 3 and [len(frame) for frame in frames] == [5909, 5698, 5564]
    payload = b''.join(frames)
    candidate, got, reads = pipe_fixture(payload)
    baseline, old, old_reads = pipe_fixture(payload, method=clock.PersistentWorker._stdout_reader)
    assert [(kind, raw) for kind, stamp, raw in got] == [(kind, raw) for kind, stamp, raw in old]
    assert candidate.logged == baseline.logged == [('stdout.log', frame) for frame in frames]
    # Counts demonstrate transport mechanism only, not a timing-speed claim.
    assert len(reads) < len(payload)//8
    assert len(old_reads) == len(payload)+1


def test_inherited_timeout_closes_real_pipe_stalled_before_newline(tmp_path):
    # This helper has no game/model/native role: it exercises the unchanged
    # trusted request/cleanup implementation against an actual blocked pipe.
    module = tmp_path/'buffered_stalled_fixture.py'
    module.write_text('''import json, sys, time
from tools.future_training_rules_20261021 import verify_runtime_sources
verify_runtime_sources()
sys.stdout.write(json.dumps({"type":"ready","metadata":{"synthetic":True}})+"\\n")
sys.stdout.flush()
for line in sys.stdin:
    sys.stdout.write('{"type":"response"')
    sys.stdout.flush()
    time.sleep(30)
''', encoding='utf-8')
    class TransportFixture(clock.PersistentWorker):
        _stdout_reader = reader.BufferedStudyWorker._stdout_reader
    worker = TransportFixture('buffered_stalled_fixture', tmp_path/'run', initialization_timeout=5,
        env={'PYTHONPATH':os.pathsep.join([str(tmp_path), str(ROOT/'src'), str(ROOT)])})
    try:
        seen = []
        begun = time.monotonic()
        receipt = worker.request({}, .2, lambda value: seen.append(value))
        assert receipt['status'] == 'timeout' and not receipt['accepted'] and not seen
        assert time.monotonic()-begun < 5
        assert worker.process.poll() is not None
        assert not any(thread.is_alive() for thread in worker._threads)
        assert worker.process.stdout.closed
    finally:
        worker.close()
