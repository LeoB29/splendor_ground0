"""Closed transport evidence admission and corruption tests without models."""
import hashlib
import json
import time

import pytest

from tools.splendorust_one_second_v3_20261024.compression import StreamingGzipLog
from tools.splendorust_one_second_v3_20261024 import log_audit as a


def fixture_logs(folder, raw=None):
    folder.mkdir()
    raw = raw or {'stdout.log': b'{"type":"response","result":{"x":1}}\n',
                  'stderr.log': b'\xff\x00arbitrary stderr\r\n',
                  'events.jsonl': b'{"status":"timeout","result":null}\n'}
    descriptors = {}
    for name in a.LOG_NAMES:
        stream = StreamingGzipLog(folder / (name + '.gz'), max_raw_bytes=a.MAX_RAW_BYTES)
        data = raw[name]
        if data:
            midpoint = max(1, len(data) // 2)
            stream.write(data[:midpoint])
            stream.write(data[midpoint:])
        descriptors[name] = stream.close_and_verify()
    cleanup = dict(status='passed', log_compression=descriptors)
    (folder / 'cleanup.json').write_text(json.dumps(cleanup), encoding='utf-8')
    return cleanup, raw


def write_manifest(folder, cleanup):
    (folder / 'cleanup.json').write_text(json.dumps(cleanup), encoding='utf-8')


def test_closed_logs_verify_exact_bytes_without_new_packet_semantics(tmp_path):
    folder = tmp_path / 'worker'
    cleanup, raw = fixture_logs(folder)
    calls = []
    result = a.verify_worker_logs(folder, boundary=lambda: calls.append(True),
                                  deadline=time.monotonic() + 5)
    assert result['status'] == 'passed' and result['streams'] == 3
    assert result['raw_bytes'] == sum(map(len, raw.values()))
    assert result['cleanup_sha256'] == hashlib.sha256((folder / 'cleanup.json').read_bytes()).hexdigest()
    assert len(calls) >= 10
    for row in result['decoded_streams']:
        assert row['raw_sha256'] == hashlib.sha256(raw[row['name']]).hexdigest()
    assert set(path.name for path in folder.iterdir()) == {'cleanup.json'} | {
        name + '.gz' for name in a.LOG_NAMES}


def test_empty_streams_are_valid(tmp_path):
    folder = tmp_path / 'worker'
    fixture_logs(folder, {name: b'' for name in a.LOG_NAMES})
    assert a.verify_worker_logs(folder)['raw_bytes'] == 0


@pytest.mark.parametrize('change', ['failed_cleanup', 'unverified_descriptor', 'traversal',
                                   'missing_descriptor', 'wrong_raw_hash', 'too_large_raw'])
def test_manifest_corruption_rejected(tmp_path, change):
    folder = tmp_path / 'worker'
    cleanup, _ = fixture_logs(folder)
    descriptor = cleanup['log_compression']['stdout.log']
    if change == 'failed_cleanup': cleanup['status'] = 'failed'
    elif change == 'unverified_descriptor': descriptor['status'] = 'incomplete'
    elif change == 'traversal': descriptor['path'] = '../stdout.log.gz'
    elif change == 'missing_descriptor': del cleanup['log_compression']['stdout.log']
    elif change == 'wrong_raw_hash': descriptor['raw_sha256'] = '0' * 64
    elif change == 'too_large_raw': descriptor['raw_bytes'] = a.MAX_RAW_BYTES + 1
    write_manifest(folder, cleanup)
    with pytest.raises(ValueError): a.verify_worker_logs(folder)


@pytest.mark.parametrize('change', ['missing_stream', 'extra_stream', 'raw_placeholder',
                                   'truncated', 'trailer_corruption', 'trailing_bytes'])
def test_stream_corruption_or_wrong_inventory_rejected(tmp_path, change):
    folder = tmp_path / 'worker'
    cleanup, _ = fixture_logs(folder)
    stream = folder / 'stdout.log.gz'
    if change == 'missing_stream': stream.unlink()
    elif change == 'extra_stream': (folder / 'extra.log.gz').write_bytes(stream.read_bytes())
    elif change == 'raw_placeholder': (folder / 'stdout.log').write_bytes(b'')
    else:
        data = stream.read_bytes()
        if change == 'truncated': data = data[:-2]
        elif change == 'trailer_corruption': data = data[:-8] + bytes([data[-8] ^ 1]) + data[-7:]
        else: data += b'junk'
        stream.write_bytes(data)
        # Even forged matching physical hashes cannot admit an invalid stream.
        descriptor = cleanup['log_compression']['stdout.log']
        descriptor['compressed_bytes'] = len(data)
        descriptor['compressed_sha256'] = hashlib.sha256(data).hexdigest()
        write_manifest(folder, cleanup)
    with pytest.raises(ValueError): a.verify_worker_logs(folder)


def test_decode_calls_resource_boundary_between_bounded_chunks(tmp_path):
    folder = tmp_path / 'worker'
    fixture_logs(folder, {'stdout.log': b'a' * (3 * 65536),
                         'stderr.log': b'', 'events.jsonl': b''})
    calls = []
    def boundary():
        calls.append(True)
        if len(calls) == 4: raise ValueError('resource gate reached')
    with pytest.raises(ValueError, match='resource gate'): a.verify_worker_logs(folder, boundary=boundary)
    assert len(calls) == 4


def test_expired_deadline_rejects_even_valid_logs(tmp_path):
    folder = tmp_path / 'worker'
    fixture_logs(folder)
    with pytest.raises(ValueError, match='deadline'):
        a.verify_worker_logs(folder, deadline=time.monotonic() - 1)


def test_cleanup_metadata_itself_is_bounded(tmp_path, monkeypatch):
    folder = tmp_path / 'worker'
    fixture_logs(folder)
    monkeypatch.setattr(a, 'MAX_CLEANUP_BYTES', 3)
    with pytest.raises(ValueError, match='manifest size'): a.verify_worker_logs(folder)
