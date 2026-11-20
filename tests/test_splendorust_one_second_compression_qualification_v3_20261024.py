"""Model-free checks of the fixed lossless-log qualification workload."""
import gzip
import hashlib
import time

import pytest

from tools.splendorust_one_second_v3_20261024 import compression_qualification as q


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(q.c, 'ROOT', tmp_path)
    return tmp_path


def test_fixed_qualification_keeps_same_raw_cap_and_actual_model_margin():
    assert q.LIMITS == dict(total_seconds=300.0, evidence_bytes=64 * q.c.MiB,
        source_file_bytes=8 * q.c.MiB, source_total_bytes=16 * q.c.MiB,
        raw_log_bytes=16 * q.c.MiB, maximum_append_flush_seconds=.100,
        maximum_aggregate_compressed_fraction=.75, passes=3)


def test_fixed_inventory_includes_all_six_workers_and_empty_stderr(workspace):
    source = workspace / 'prior'
    for game in range(6):
        folder = source / 'workers' / f'operational_p{game // 2:03d}_g{game}'
        folder.mkdir(parents=True)
        for name in q.LOG_NAMES:
            (folder / name).write_bytes(b'' if name == 'stderr.log' else b'{"data":1}\n')
    items = q.source_inventory(source)
    assert len(items) == 18
    assert len([item for item in items if item['bytes'] == 0]) == 6
    (source / 'workers/operational_p999_g9').mkdir()
    with pytest.raises(ValueError, match='inventory changed'):
        q.source_inventory(source)


def test_boundaries_follow_readline_for_protocol_and_fixed_stderr_chunks():
    raw = b'first\r\r\nsecond\v\x00\xff\nlast'
    assert list(q.raw_chunks(raw, 'stdout.log')) == [b'first\r\r\n', b'second\v\x00\xff\n', b'last']
    assert list(q.raw_chunks(raw, 'events.jsonl')) == list(q.raw_chunks(raw, 'stdout.log'))
    noisy = bytes(range(256)) * 40
    parts = list(q.raw_chunks(noisy, 'stderr.log'))
    assert [len(part) for part in parts] == [4096, 4096, 2048]
    assert b''.join(parts) == noisy


@pytest.mark.parametrize('raw,name', [
    (b'', 'stderr.log'),
    (b'{"message":"caf\xc3\xa9"}\r\n{"final":true}\n', 'stdout.log'),
    (bytes(range(256)) * 40, 'stderr.log'),
])
def test_actual_file_replay_checks_exact_bytes_and_both_reader_hashes(workspace, raw, name):
    source = workspace / name
    source.write_bytes(raw)
    item = q.c.artifact(source)
    destination = workspace / (name + '.gz')
    result = q.replay_file(item, destination, deadline=time.monotonic() + 10)
    assert result['production_manifest']['raw_sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['production_reader'] == result['production_manifest']
    assert result['independent_reader']['exact_bytes_equal'] is True
    assert gzip.decompress(destination.read_bytes()) == raw
    assert q.c.artifact(source) == item
    assert sum(row['bytes'] for row in result['append_flush_timings']) == len(raw)


def test_source_mutation_is_rejected_before_compressed_output(workspace):
    source = workspace / 'stdout.log'
    source.write_bytes(b'original\n')
    item = q.c.artifact(source)
    source.write_bytes(b'changed!\n')
    destination = workspace / 'out.gz'
    with pytest.raises(ValueError, match='Changed frozen input'):
        q.replay_file(item, destination, deadline=time.monotonic() + 10)
    assert not destination.exists()


def test_independent_reader_rejects_wrong_contents_and_expansion(workspace):
    archive = workspace / 'bad.gz'
    archive.write_bytes(gzip.compress(b'wrong'))
    with pytest.raises(ValueError, match='bytes differ'):
        q.independent_gzip(archive, b'right', deadline=time.monotonic() + 10)
    with pytest.raises(ValueError, match='expansion limit'):
        q.independent_gzip(archive, b'w', deadline=time.monotonic() + 10)


def test_budget_and_independent_read_deadlines_are_fail_closed(workspace, monkeypatch):
    with pytest.raises(ValueError, match='time cap'):
        q.check_budget(workspace, time.monotonic() - 1)
    archive = workspace / 'log.gz'
    archive.write_bytes(gzip.compress(b'data'))
    with pytest.raises(ValueError, match='time cap'):
        q.independent_gzip(archive, b'data', deadline=time.monotonic() - 1)
    monkeypatch.setitem(q.LIMITS, 'evidence_bytes', 1)
    with pytest.raises(ValueError, match='evidence cap'):
        q.check_budget(workspace, time.monotonic() + 10)


def test_raw_cap_qualification_keeps_failed_archive_and_rejects_completion(workspace, monkeypatch):
    monkeypatch.setitem(q.LIMITS, 'raw_log_bytes', 2 * 65536)
    rows = q.qualify_raw_cap(workspace, deadline=time.monotonic() + 10)
    assert [row['case'] for row in rows] == ['exact_cap_verified', 'cap_plus_one_rejected']
    assert rows[0]['independent_raw_bytes'] == 2 * 65536
    assert rows[1]['raw_bytes'] == 2 * 65536
    assert 'Raw log byte limit' in rows[1]['error']
    assert (workspace / 'raw_cap_overflow.gz').exists()
    with pytest.raises(EOFError):
        gzip.decompress((workspace / 'raw_cap_overflow.gz').read_bytes())
