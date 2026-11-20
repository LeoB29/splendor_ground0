"""Independent closed-worker transport evidence admission; no model imports.

Compression changes storage only. This audit authenticates every decoded byte
against the separately accumulated raw hash and preserves the existing protocol
and forfeit semantics: malformed or late raw packets are evidence, not new moves.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .compression import verify_log


LOG_NAMES = ('stdout.log', 'stderr.log', 'events.jsonl')
MAX_RAW_BYTES = 16 * 1024**2
# The writer sync-flushes after every nonempty write. This conservative bound
# includes even one-byte writes; the unchanged outer physical evidence cap is
# enforced independently throughout collection and final verification.
MAX_COMPRESSED_BYTES = MAX_RAW_BYTES * 16 + 128
MAX_CLEANUP_BYTES = 1024**2


def _require(value, message):
    if not value:
        raise ValueError('TRANSPORT_LOG_AUDIT: ' + message)


def verify_worker_logs(folder, boundary=None, deadline=None):
    """Verify exactly the three closed streams using bounded streaming decode.

    No expanded log is retained. The optional boundary runs before and after each
    stream and at most 64 KiB apart while decoding. No filesystem writes occur.
    Successful cleanup and complete gzip trailers are required for acceptance;
    an interrupted stream remains preserved but cannot pass this function.
    """
    check = boundary if boundary is not None else lambda: None
    check()
    folder = Path(folder).resolve()
    cleanup_path = folder / 'cleanup.json'
    _require(cleanup_path.stat().st_size <= MAX_CLEANUP_BYTES, 'cleanup manifest size')
    with cleanup_path.open('rb') as stream:
        encoded = stream.read(MAX_CLEANUP_BYTES + 1)
    _require(len(encoded) <= MAX_CLEANUP_BYTES, 'cleanup manifest size')
    cleanup = json.loads(encoded)
    _require(type(cleanup) is dict and cleanup.get('status') == 'passed',
             'owned worker cleanup did not pass')
    descriptors = cleanup.get('log_compression')
    _require(type(descriptors) is dict and set(descriptors) == set(LOG_NAMES),
             'closed log descriptor inventory')
    _require({path.name for path in folder.glob('*.gz')} ==
             {name + '.gz' for name in LOG_NAMES}, 'compressed stream inventory')
    results = []
    for name in LOG_NAMES:
        check()
        _require(not (folder / name).exists(), 'uncompressed stream unexpectedly present')
        path = folder / (name + '.gz')
        _require(path.resolve().parent == folder and path.is_file(), 'stream path escapes worker')
        descriptor = descriptors[name]
        _require(type(descriptor) is dict and descriptor.get('status') == 'verified' and
                 descriptor.get('path') == path.name, 'closed stream descriptor identity/status')
        verified = verify_log(path, descriptor, max_raw_bytes=MAX_RAW_BYTES,
            max_compressed_bytes=MAX_COMPRESSED_BYTES, deadline=deadline,
            sink=lambda _chunk: check())
        results.append({'name': name, **verified})
        check()
    return dict(status='passed', worker=folder.name, streams=len(results),
        raw_bytes=sum(row['raw_bytes'] for row in results),
        compressed_bytes=sum(row['compressed_bytes'] for row in results),
        cleanup_sha256=hashlib.sha256(encoded).hexdigest(),
        max_raw_bytes_per_stream=MAX_RAW_BYTES,
        max_compressed_bytes_per_stream=MAX_COMPRESSED_BYTES,
        decoded_streams=results,
        scope='Lossless raw-byte integrity; original protocol, clocks and forfeit checks unchanged')
