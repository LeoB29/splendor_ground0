"""Bounded prefreeze qualification of transport compression, without a model.

All six previous operational workers supply the fixed, read-only byte workload.
The production writer is exercised with the same line/chunk boundaries used by
the transport. The stdlib gzip reader independently checks the production
reader's byte and SHA256 result. No outcome is read or used here.
"""
from __future__ import annotations

import gzip
import hashlib
import io
from pathlib import Path
import sys
import time
import traceback

from . import common as c
from .compression import StreamingGzipLog, runtime_contract, verify_log
from tools.future_training_rules_20261021 import verify_runtime_sources


SOURCE = c.ROOT / 'outputs/splendorust_one_second_20261024_02'
SOURCE_FREEZE_SHA256 = '5b86734afec36093da0916cab4811c7abf076f273fd7c52ce42afc2dc4988f60'
LOG_NAMES = ('stdout.log', 'stderr.log', 'events.jsonl')
LIMITS = dict(total_seconds=300.0, evidence_bytes=64 * c.MiB,
              source_file_bytes=8 * c.MiB, source_total_bytes=16 * c.MiB,
              raw_log_bytes=16 * c.MiB, maximum_append_flush_seconds=0.100,
              maximum_aggregate_compressed_fraction=0.75, passes=3)


def source_inventory(source=SOURCE):
    """Choose exactly the three logs from each of the six already played games."""
    workers = Path(source) / 'workers'
    folders = [workers / f'operational_p{game // 2:03d}_g{game}' for game in range(6)]
    c.require(set(workers.glob('operational_*')) == set(folders),
              'Compression source operational folder inventory changed')
    items = [c.artifact(folder / name) for folder in folders for name in LOG_NAMES]
    c.require(len(items) == 18, 'Compression source quota')
    c.require(all(item['bytes'] <= LIMITS['source_file_bytes'] for item in items),
              'Compression source per-file bound')
    c.require(sum(item['bytes'] for item in items) <= LIMITS['source_total_bytes'],
              'Compression source aggregate bound')
    return items


def raw_chunks(raw, name):
    """Match stdout.readline, stderr.read(4096), and one JSONL event per write."""
    stream = io.BytesIO(raw)
    read = (lambda: stream.read(4096)) if name == 'stderr.log' else stream.readline
    return iter(read, b'')


def check_budget(folder, deadline):
    c.require(time.monotonic() < deadline, 'Compression qualification time cap')
    physical = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
    c.require(physical <= LIMITS['evidence_bytes'], 'Compression qualification evidence cap')
    return physical


def independent_gzip(path, expected_raw, *, deadline):
    """Byte equality as well as hash equality, in bounded decoded chunks."""
    offset = 0
    digest = hashlib.sha256()
    with gzip.open(path, 'rb') as stream:
        while True:
            c.require(time.monotonic() < deadline, 'Independent gzip verification time cap')
            chunk = stream.read(65536)
            if not chunk:
                break
            c.require(offset + len(chunk) <= len(expected_raw), 'Independent gzip expansion limit')
            c.require(chunk == expected_raw[offset:offset + len(chunk)],
                      'Independent gzip bytes differ')
            digest.update(chunk)
            offset += len(chunk)
    c.require(offset == len(expected_raw), 'Independent gzip raw length differs')
    c.require(digest.hexdigest() == hashlib.sha256(expected_raw).hexdigest(),
              'Independent gzip raw hash differs')
    return dict(raw_bytes=offset, raw_sha256=digest.hexdigest(), exact_bytes_equal=True)


def replay_file(item, destination, *, deadline):
    """Leave the file on every failure; expose each actual append duration."""
    source = c.ROOT / item['path']
    c.verify([item])
    raw = source.read_bytes()
    c.require(len(raw) <= LIMITS['source_file_bytes'], 'Compression source grew beyond limit')
    c.require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'Compression source changed on read')
    writer = StreamingGzipLog(destination, max_raw_bytes=LIMITS['raw_log_bytes'])
    timings = []
    primary = None
    try:
        for chunk in raw_chunks(raw, source.name):
            c.require(time.monotonic() < deadline, 'Compression replay time cap')
            started = time.perf_counter_ns()
            written = writer.write(chunk)
            writer.flush()
            elapsed = (time.perf_counter_ns() - started) / 1e9
            timings.append(dict(bytes=len(chunk), seconds=elapsed))
            c.require(written == len(chunk), 'Compression writer short acknowledgement')
        finish_started = time.perf_counter()
        manifest = writer.close_and_verify(deadline=deadline)
        finish_seconds = time.perf_counter() - finish_started
    except BaseException as error:
        primary = error
        raise
    finally:
        if not writer.closed:
            try:
                writer.close_and_verify(deadline=deadline)
            except BaseException:
                if primary is None:
                    raise
    c.require(manifest['raw_bytes'] == len(raw) and manifest['raw_sha256'] == item['sha256'],
              'Production compression changed original bytes/hash')
    verified = verify_log(destination, manifest, max_raw_bytes=LIMITS['raw_log_bytes'], deadline=deadline)
    independent = independent_gzip(destination, raw, deadline=deadline)
    c.verify([item])
    return dict(source=item, archive=c.artifact(destination), production_manifest=manifest,
                production_reader=verified, independent_reader=independent,
                append_flush_timings=timings, close_and_verify_seconds=finish_seconds,
                maximum_append_flush_seconds=max((row['seconds'] for row in timings), default=0.0))


def qualify_raw_cap(folder, *, deadline):
    """Use the unchanged production 16 MiB cap, including rejection at cap + 1."""
    block = b'x' * 65536
    count = LIMITS['raw_log_bytes'] // len(block)
    c.require(count * len(block) == LIMITS['raw_log_bytes'], 'Raw cap block arithmetic')
    rows = []
    for overflow in (False, True):
        path = folder / ('raw_cap_overflow.gz' if overflow else 'raw_cap_exact.gz')
        writer = StreamingGzipLog(path, max_raw_bytes=LIMITS['raw_log_bytes'])
        digest = hashlib.sha256()
        timings = []
        for _ in range(count):
            c.require(time.monotonic() < deadline, 'Raw-cap qualification time cap')
            started = time.perf_counter_ns()
            writer.write(block)
            writer.flush()
            timings.append((time.perf_counter_ns() - started) / 1e9)
            digest.update(block)
        if overflow:
            try:
                writer.write(b'!')
            except ValueError as error:
                overflow_error = str(error)
            else:
                raise ValueError('Production raw cap accepted excess byte')
            try:
                writer.close_and_verify(deadline=deadline)
            except ValueError as error:
                close_error = str(error)
            else:
                raise ValueError('Raw-cap failure accepted as complete archive')
            c.require(writer.closed and writer.receipt is None, 'Overflow file did not close fail-closed')
            c.require(writer.raw_bytes == LIMITS['raw_log_bytes'], 'Overflow consumed excess raw byte')
            rows.append(dict(case='cap_plus_one_rejected', raw_bytes=writer.raw_bytes,
                             raw_sha256=digest.hexdigest(), error=overflow_error, close_error=close_error,
                             archive=c.artifact(path), append_flush_seconds=timings))
        else:
            manifest = writer.close_and_verify(deadline=deadline)
            c.require(manifest['raw_bytes'] == LIMITS['raw_log_bytes'] and
                      manifest['raw_sha256'] == digest.hexdigest(), 'Exact raw-cap archive mismatch')
            # Streaming independent check avoids materializing the full cap.
            length = 0
            independent_hash = hashlib.sha256()
            with gzip.open(path, 'rb') as stream:
                while True:
                    c.require(time.monotonic() < deadline, 'Raw-cap independent read time cap')
                    chunk = stream.read(len(block))
                    if not chunk:
                        break
                    length += len(chunk)
                    c.require(length <= LIMITS['raw_log_bytes'] and chunk == block[:len(chunk)],
                              'Raw-cap independent bytes mismatch')
                    independent_hash.update(chunk)
            c.require(length == LIMITS['raw_log_bytes'] and independent_hash.hexdigest() == digest.hexdigest(),
                      'Raw-cap independent hash mismatch')
            rows.append(dict(case='exact_cap_verified', production_manifest=manifest,
                             independent_raw_bytes=length, independent_raw_sha256=independent_hash.hexdigest(),
                             archive=c.artifact(path), append_flush_seconds=timings))
    return rows


def run():
    folder = c.OUT / 'compression_qualification'
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + LIMITS['total_seconds']
    rows, cap_rows = [], []
    try:
        runtime_sources = verify_runtime_sources()
        c.require(c.file_hash(SOURCE / 'freeze.json') == SOURCE_FREEZE_SHA256,
                  'Compression workload source freeze changed')
        items = source_inventory()
        code = [c.artifact(path) for path in sorted(Path(__file__).parent.glob('*.py'))]
        code += [c.artifact(c.ROOT / 'tools/local_player_qualification_128/clock.py')]
        compression_runtime = runtime_contract()
        plan = dict(schema='lossless_transport_compression_qualification_v3', time=c.now(),
                    limits=LIMITS, source_freeze=c.artifact(SOURCE / 'freeze.json'), sources=items,
                    source_code=code, runtime_sources=runtime_sources, python=sys.version,
                    compression_runtime=compression_runtime,
                    checkpoint_calls=0, game_decisions=0,
                    pass_order=['forward', 'reverse', 'forward'], raw_cap_cases=['exact_16MiB', '16MiB_plus_one'],
                    independent_reader='Python stdlib gzip, bounded decoded chunks and exact byte comparisons',
                    timing_gate_rationale='At most 0.100s for each observed append/flush; unchanged 0.700s '
                    'actual-model qualification and operational gates still apply after freezing.',
                    source_read_only=True, failures_retained=True)
        c.save(folder / 'plan.json', plan)
        # Reserve a conservative worst case before new evidence is created.
        # Two cap files plus three copies of every source, sync-flush overhead
        # and metadata are well below the independent 64 MiB qualification cap.
        worst_case = 2 * LIMITS['raw_log_bytes'] + 3 * sum(item['bytes'] * 2 + 1024 for item in items) + c.MiB
        c.require(worst_case <= LIMITS['evidence_bytes'], 'Qualification worst-case storage reservation')
        for index in range(LIMITS['passes']):
            ordered = list(reversed(items)) if index == 1 else items
            for position, item in enumerate(ordered):
                check_budget(folder, deadline)
                name = f'pass{index}_{position:02d}_{Path(item["path"]).name}.gz'
                row = replay_file(item, folder / name, deadline=deadline)
                rows.append(row)
                c.save(folder / f'pass{index}_{position:02d}.json', row)
                c.require(row['maximum_append_flush_seconds'] <= LIMITS['maximum_append_flush_seconds'],
                          'Compression append/flush timing gate')
                check_budget(folder, deadline)
        cap_rows = qualify_raw_cap(folder, deadline=deadline)
        c.save(folder / 'raw_cap.json', dict(status='passed', cases=cap_rows))
        maximum = max([row['maximum_append_flush_seconds'] for row in rows] +
                      [max(row['append_flush_seconds'], default=0.0) for row in cap_rows])
        raw = sum(row['production_manifest']['raw_bytes'] for row in rows)
        compressed = sum(row['production_manifest']['compressed_bytes'] for row in rows)
        c.require(raw > 0, 'Empty compression workload')
        ratio = compressed / raw
        c.require(len(rows) == 54, 'Compression replay quota')
        c.require(maximum <= LIMITS['maximum_append_flush_seconds'], 'Compression append/flush timing gate')
        c.require(ratio <= LIMITS['maximum_aggregate_compressed_fraction'], 'Compression ratio gate')
        c.verify(items + code)
        c.require(runtime_contract() == compression_runtime, 'Compression runtime changed during qualification')
        receipt = dict(status='passed', time=c.now(), plan=c.artifact(folder / 'plan.json'),
                       elapsed_seconds=time.monotonic() - started, source_logs=len(items),
                       replay_streams=len(rows), append_flush_count=sum(len(row['append_flush_timings']) for row in rows),
                       raw_bytes=raw, compressed_bytes=compressed, compressed_fraction=ratio,
                       maximum_append_flush_seconds=maximum, raw_cap_cases=2,
                       checkpoint_calls=0, game_decisions=0, source_code=code,
                       source_inventory=code + items + [c.artifact(SOURCE / 'freeze.json')],
                       compression_runtime=compression_runtime,
                       evidence_bytes_before_receipt=check_budget(folder, deadline),
                       raw_cap_receipt=c.artifact(folder / 'raw_cap.json'),
                       replay_receipts=[c.artifact(path) for path in sorted(folder.glob('pass*.json'))],
                       original_logs_unchanged=True, independent_bytes_and_hashes_equal=True)
        c.save(folder / 'receipt.json', receipt)
        check_budget(folder, deadline)
        print(c.canonical(receipt), flush=True)
        return receipt
    except BaseException as error:
        c.save(folder / 'failure.json', dict(status='failed', time=c.now(), error=repr(error),
               traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started,
               completed_replay_streams=len(rows), completed_cap_cases=len(cap_rows),
               checkpoint_calls=0, game_decisions=0, partial_files_preserved=True))
        raise


if __name__ == '__main__':
    run()
