"""Model-free byte preservation, corruption, limits and real-clock regressions."""
import gzip
import hashlib
import json
import os
import time

import pytest

from tools.splendorust_one_second_v3_20261024 import common
from tools.splendorust_one_second_v3_20261024.compression import (
    CHUNK, CODEC, StreamingGzipLog, read_log, verify_log, runtime_contract,
)


def manifest(path, raw):
    encoded = path.read_bytes()
    return dict(codec=CODEC, path=path.name, raw_bytes=len(raw),
                raw_sha256=hashlib.sha256(raw).hexdigest(), compressed_bytes=len(encoded),
                compressed_sha256=hashlib.sha256(encoded).hexdigest())


@pytest.mark.parametrize('chunk_size', [1, 7, 4096, 65536])
def test_exact_binary_and_split_utf8_roundtrip_with_flushed_growth(tmp_path, chunk_size):
    raw = bytes(range(256)) * 20 + 'é日本語🔎\r\n'.encode('utf-8') * 60
    path = tmp_path / 'stdout.log.gz'
    writer = StreamingGzipLog(path, max_raw_bytes=len(raw))
    size = 0
    for start in range(0, len(raw), chunk_size):
        part = raw[start:start + chunk_size]
        assert writer.write(part) == len(part)
        assert path.stat().st_size > size
        size = path.stat().st_size
    receipt = writer.close_and_verify()
    assert gzip.decompress(path.read_bytes()) == raw
    assert read_log(path, receipt, max_raw_bytes=len(raw)) == raw
    assert receipt == writer.close_and_verify()
    assert receipt == {**manifest(path, raw), 'status': 'verified'}


def test_empty_log_is_complete_single_member(tmp_path):
    path = tmp_path / 'stderr.log.gz'
    writer = StreamingGzipLog(path, max_raw_bytes=0)
    receipt = writer.close_and_verify()
    assert receipt['raw_bytes'] == 0
    assert gzip.decompress(path.read_bytes()) == b''
    assert read_log(path, receipt, max_raw_bytes=0) == b''


@pytest.mark.parametrize('mutation', ['truncate_one', 'truncate_trailer', 'truncate_half',
                                      'bad_crc', 'extra_member', 'trailing_data'])
def test_rejects_corruption_even_with_matching_compressed_manifest(tmp_path, mutation):
    path = tmp_path / 'stdout.log.gz'
    raw = b'foo\n' * 1000
    writer = StreamingGzipLog(path, max_raw_bytes=len(raw))
    writer.write(raw)
    receipt = writer.close_and_verify()
    encoded = path.read_bytes()
    if mutation == 'truncate_one': encoded = encoded[:-1]
    elif mutation == 'truncate_trailer': encoded = encoded[:-8]
    elif mutation == 'truncate_half': encoded = encoded[:len(encoded) // 2]
    elif mutation == 'bad_crc': encoded = encoded[:-8] + bytes([encoded[-8] ^ 1]) + encoded[-7:]
    elif mutation == 'extra_member': encoded += gzip.compress(b'concealed extra output')
    else: encoded += b'\x00unexpected suffix'
    path.write_bytes(encoded)
    altered = manifest(path, raw)
    with pytest.raises(ValueError, match='gzip|checksum'):
        verify_log(path, altered, max_raw_bytes=len(raw))
    with pytest.raises(ValueError):
        verify_log(path, receipt, max_raw_bytes=len(raw))


def test_bounded_reader_rejects_expansion_and_streams_small_chunks(tmp_path):
    raw = b'x' * (CHUNK * 5 + 19)
    path = tmp_path / 'stdout.log.gz'
    path.write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
    expected = manifest(path, raw)
    sizes = []
    verify_log(path, expected, max_raw_bytes=len(raw), sink=lambda part: sizes.append(len(part)))
    assert sum(sizes) == len(raw) and max(sizes) <= CHUNK and len(sizes) > 1
    lied = {**expected, 'raw_bytes': 100}
    sink = []
    with pytest.raises(ValueError, match='raw byte limit'):
        verify_log(path, lied, max_raw_bytes=100, sink=sink.append)
    assert sum(map(len, sink)) <= 100


@pytest.mark.parametrize('field', ['raw_bytes', 'raw_sha256', 'compressed_bytes', 'compressed_sha256'])
def test_manifest_mismatch_is_rejected(tmp_path, field):
    raw = b'data\n'
    path = tmp_path / 'stdout.log.gz'
    writer = StreamingGzipLog(path, max_raw_bytes=100)
    writer.write(raw)
    receipt = writer.close_and_verify()
    forged = {**receipt, field: receipt[field] + 1 if field.endswith('bytes') else '0' * 64}
    with pytest.raises(ValueError):
        verify_log(path, forged, max_raw_bytes=100)


def test_truncated_inflight_stream_is_preserved_but_never_accepted(tmp_path):
    path = tmp_path / 'stdout.log.gz'
    raw = b'written prefix\n'
    writer = StreamingGzipLog(path, max_raw_bytes=100)
    writer.write(raw)
    inflight = path.read_bytes()
    with pytest.raises(ValueError, match='Truncated'):
        verify_log(path, manifest(path, raw), max_raw_bytes=100)
    writer.close_and_verify()
    assert path.read_bytes().startswith(inflight)


def test_direct_writer_overflow_is_preserved_and_cannot_get_verified_receipt(tmp_path):
    path = tmp_path / 'stdout.log.gz'
    writer = StreamingGzipLog(path, max_raw_bytes=4)
    writer.write(b'1234')
    captured = path.read_bytes()
    with pytest.raises(ValueError, match='byte limit'):
        writer.write(b'5')
    with pytest.raises(ValueError, match='Earlier'):
        writer.close_and_verify()
    assert path.read_bytes() == captured
    assert writer.closed and writer.receipt is None


def test_exclusive_output_and_physical_limits_and_deadline(tmp_path):
    path = tmp_path / 'stdout.log.gz'
    writer = StreamingGzipLog(path, max_raw_bytes=100)
    writer.write(b'a\n')
    receipt = writer.close_and_verify()
    with pytest.raises(FileExistsError):
        StreamingGzipLog(path, max_raw_bytes=100)
    with pytest.raises(ValueError, match='size limit'):
        verify_log(path, receipt, max_raw_bytes=100, max_compressed_bytes=1)
    with pytest.raises(ValueError, match='deadline'):
        verify_log(path, receipt, max_raw_bytes=100, deadline=time.monotonic() - 1)
    other = StreamingGzipLog(tmp_path / 'unfinished.gz', max_raw_bytes=100)
    other.write(b'a\n')
    with pytest.raises(ValueError, match='deadline'):
        other.close_and_verify(deadline=time.monotonic() - 1)
    assert other.closed and other.receipt is None


def test_completed_writer_verifies_against_independent_raw_hash(tmp_path):
    writer = StreamingGzipLog(tmp_path / 'stdout.log.gz', max_raw_bytes=100)
    writer.write(b'actual bytes')
    writer._raw_hash = hashlib.sha256(b'incorrect incoming-byte hash')
    with pytest.raises(ValueError, match='raw_sha256'):
        writer.close_and_verify()
    assert writer.closed and writer.receipt is None


def test_short_write_preserves_partial_file_and_blocks_cleanup_acceptance(tmp_path):
    path = tmp_path / 'stdout.log.gz'
    writer = StreamingGzipLog(path, max_raw_bytes=100)
    original = writer._file
    class PartialFile:
        def write(self, encoded): return original.write(encoded[:-1])
        def close(self): original.close()
    writer._file = PartialFile()
    with pytest.raises(OSError, match='short write'):
        writer.write(b'the required evidence')
    with pytest.raises(ValueError, match='Earlier'):
        writer.close_and_verify()
    assert path.exists() and path.stat().st_size > 0
    assert writer.receipt is None


def test_runtime_contract_binds_loaded_codec_and_independent_reader():
    result = runtime_contract()
    assert result['codec'] == CODEC
    assert result['zlib_runtime_version']
    roles = {row['role'] for row in result['files']}
    assert {'gzip_stdlib', 'compression_stdlib'} <= roles
    assert ('zlib_extension' in roles or 'zlib_extension' in result['modules_without_files'])
    for row in result['files']:
        with open(row['path'], 'rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == row['sha256']
        assert os.path.getsize(row['path']) == row['bytes']


STUB = r'''
import json, sys, time
def emit(value):
    sys.stdout.buffer.write((json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
    sys.stdout.buffer.flush()
emit({'type':'ready','metadata':{'stub':'compression'}})
for line in sys.stdin:
    packet = json.loads(line)
    value = packet['payload']
    if value.get('sleep'):
        time.sleep(value['sleep'])
    if value.get('stderr'):
        sys.stderr.buffer.write(bytes.fromhex(value['stderr']))
        sys.stderr.buffer.flush()
    emit({'type':'response','request_id':packet['request_id'],'result':value})
'''


@pytest.fixture
def real_worker(tmp_path):
    (tmp_path / 'compression_stub.py').write_text(STUB, encoding='utf-8')
    worker = common.StudyWorker('compression_stub', tmp_path / 'worker', native_artifacts=[],
        env={'PYTHONPATH': os.pathsep.join([str(tmp_path), str(common.ROOT)])},
        initialization_timeout=10, max_log_bytes=16 * common.MiB,
        absolute_deadline=time.monotonic() + 30)
    yield worker
    worker.close()


def encode_packet(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n').encode()


def test_real_owned_transport_exact_logs_and_trusted_timestamp(real_worker, monkeypatch):
    worker = real_worker
    payload = {'text': 'é日本語🔎' * 20, 'stderr': bytes(range(256)).hex()}
    original = StreamingGzipLog.write
    def slow_stdout(self, raw):
        if self.path.name == 'stdout.log.gz':
            time.sleep(.15)
        return original(self, raw)
    monkeypatch.setattr(StreamingGzipLog, 'write', slow_stdout)
    receipt = worker.request(payload, 1)
    assert receipt['status'] == 'ok'
    assert receipt['elapsed_seconds'] - receipt['receipt_elapsed_seconds'] >= .13
    assert receipt['received_before_deadline']
    closed = worker.close()
    assert closed['status'] == 'passed' and closed['elapsed_seconds'] < 5
    assert set(closed['log_compression']) == {'stdout.log', 'stderr.log', 'events.jsonl'}
    expected_stdout = encode_packet({'type': 'ready', 'metadata': {'stub': 'compression'}})
    expected_stdout += encode_packet({'type': 'response', 'request_id': 'request-00000001', 'result': payload})
    expected_event = (json.dumps({k: v for k, v in receipt.items() if k != 'result'}, allow_nan=False) + '\n').encode()
    for name, raw in [('stdout.log', expected_stdout), ('stderr.log', bytes(range(256))),
                      ('events.jsonl', expected_event)]:
        path = worker.output_dir / (name + '.gz')
        actual = read_log(path, closed['log_compression'][name], max_raw_bytes=16 * common.MiB)
        assert actual == raw
        assert gzip.decompress(path.read_bytes()) == raw
        assert not (worker.output_dir / name).exists()
    assert worker.process.poll() is not None


def test_transport_raw_cap_preserves_exact_prefix_and_original_failure(real_worker):
    worker = real_worker
    # stderr has not been touched yet; a small synthetic cap exercises exactly
    # the production branch without generating megabytes of evidence.
    worker.max_log_bytes = 4
    worker._log('stderr.log', b'123456')
    assert worker._failure == 'log_limit' and worker._log_sizes['stderr.log'] == 4
    worker.max_log_bytes = 16 * common.MiB
    closed = worker.close()
    expected = closed['log_compression']['stderr.log']
    assert expected['raw_bytes'] == 4
    assert read_log(worker.output_dir / 'stderr.log.gz', expected, max_raw_bytes=4) == b'1234'


def test_compression_io_failure_is_infrastructure_even_on_stderr(real_worker, monkeypatch):
    worker = real_worker
    original = StreamingGzipLog._emit
    def broken(self, encoded):
        if self.path.name == 'stderr.log.gz':
            raise OSError('synthetic disk failure')
        return original(self, encoded)
    monkeypatch.setattr(StreamingGzipLog, '_emit', broken)
    with pytest.raises(ValueError, match='Transport log compression failed'):
        worker._log('stderr.log', b'evidence')
    assert worker._failure == 'log_compression_error'
    worker._primary_failure = {'status': worker._failure, 'detail': worker._failure_detail}
    closed = worker.close()
    assert closed['status'] == 'failed'
    assert any('synthetic disk failure' in error for error in closed['errors'])
    with pytest.raises(RuntimeError, match='cleanup acceptance'):
        worker.__exit__(None, None, None)
    assert (worker.output_dir / 'stderr.log.gz').exists()
