"""Bounded, byte-exact transport evidence compression; no game/model code.

One gzip member is written with level 1 and an explicit sync flush per input
write. The raw hash is accumulated before compression independently of the
decompressor used at close. Readers reject incomplete streams, extra members,
trailing bytes, checksum failures and declared/actual size or hash mismatches.
"""
from __future__ import annotations

import hashlib
import gzip
import io
import os
from pathlib import Path
import time
import zlib


CODEC = 'gzip_single_member_level1_sync_flush'
CHUNK = 65536


def runtime_contract():
    """Pin actual codec and independent qualification-reader implementations."""
    import _compression
    sources = [('zlib_extension', getattr(zlib, '__file__', None)),
               ('gzip_stdlib', getattr(gzip, '__file__', None)),
               ('compression_stdlib', getattr(_compression, '__file__', None))]
    if os.name == 'nt':
        # Conda Python may dynamically link a separately loaded zlib DLL.
        # Query only already loaded modules; never load a guessed library.
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        api.GetModuleHandleW.restype = wintypes.HMODULE
        api.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        api.GetModuleFileNameW.restype = wintypes.DWORD
        for name in ('zlib.dll', 'zlib1.dll'):
            handle = api.GetModuleHandleW(name)
            if handle:
                buffer = ctypes.create_unicode_buffer(32768)
                count = api.GetModuleFileNameW(handle, buffer, len(buffer))
                if not count or count >= len(buffer):
                    raise OSError('Cannot identify loaded zlib runtime library')
                sources.append(('loaded_' + name, buffer.value))
    files, builtin = [], []
    for role, source in sources:
        if source is None:
            builtin.append(role)
            continue
        path = Path(source).resolve(strict=True)
        before = path.stat()
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Compression runtime changed while hashing')
        files.append(dict(role=role, path=str(path), bytes=after.st_size, sha256=digest))
    return dict(schema='lossless_transport_compression_runtime_v1', codec=CODEC,
                zlib_compile_version=zlib.ZLIB_VERSION, zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
                files=files, modules_without_files=builtin)


def _check_time(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise ValueError('Compression verification deadline reached')


def _positive_limit(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('Invalid ' + name)
    return value


def verify_log(path, expected, *, max_raw_bytes, max_compressed_bytes=None,
               deadline=None, sink=None):
    """Verify one gzip stream with <=64 KiB decode chunks, optionally consume.

    The caller supplies trusted independently recorded hashes/lengths. The
    reader never collects the expanded stream; sink receives bounded chunks.
    max_compressed_bytes defaults to a conservative worst-case one-byte-write
    bound, while the outer study continues enforcing its total physical limit.
    """
    path = Path(path)
    maximum = _positive_limit(max_raw_bytes, 'raw byte limit')
    physical_maximum = _positive_limit(
        max_compressed_bytes if max_compressed_bytes is not None else maximum * 16 + 128,
        'compressed byte limit')
    _check_time(deadline)
    if (expected.get('codec') != CODEC or expected.get('path') != path.name or
            type(expected.get('raw_bytes')) is not int or
            not 0 <= expected['raw_bytes'] <= maximum or
            type(expected.get('compressed_bytes')) is not int or
            not 0 <= expected['compressed_bytes'] <= physical_maximum):
        raise ValueError('Invalid compressed log manifest or size limit')
    for field in ('raw_sha256', 'compressed_sha256'):
        digest = expected.get(field)
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in '0123456789abcdef' for character in digest)):
            raise ValueError('Invalid compressed log hash')
    size = path.stat().st_size
    if size != expected['compressed_bytes'] or size > physical_maximum:
        raise ValueError('Compressed log physical size mismatch or limit')
    raw_hash, compressed_hash = hashlib.sha256(), hashlib.sha256()
    raw_bytes = compressed_bytes = 0
    decoder = zlib.decompressobj(wbits=31)
    try:
        with path.open('rb') as source:
            while True:
                _check_time(deadline)
                encoded = source.read(CHUNK)
                if not encoded:
                    break
                compressed_bytes += len(encoded)
                if compressed_bytes > physical_maximum or compressed_bytes > expected['compressed_bytes']:
                    raise ValueError('Compressed log grew beyond its recorded limit')
                compressed_hash.update(encoded)
                if decoder.eof:
                    raise ValueError('Extra gzip member or trailing compressed bytes')
                pending = encoded
                while pending:
                    _check_time(deadline)
                    # One extra byte detects overflow without allocating an
                    # unbounded expansion even for adversarial compressed data.
                    decoded = decoder.decompress(pending, min(CHUNK, maximum - raw_bytes + 1))
                    pending = decoder.unconsumed_tail
                    raw_bytes += len(decoded)
                    if raw_bytes > maximum or raw_bytes > expected['raw_bytes']:
                        raise ValueError('Decoded log exceeds raw byte limit or recorded length')
                    raw_hash.update(decoded)
                    if sink is not None and decoded:
                        sink(decoded)
                    if decoder.eof:
                        if decoder.unused_data or pending:
                            raise ValueError('Extra gzip member or trailing compressed bytes')
                        break
        if not decoder.eof:
            raise ValueError('Truncated gzip stream: trailer/end marker missing')
    except zlib.error as error:
        raise ValueError('Invalid gzip stream or checksum: ' + str(error)) from error
    actual = dict(status='verified', codec=CODEC, path=path.name,
                  raw_bytes=raw_bytes, raw_sha256=raw_hash.hexdigest(),
                  compressed_bytes=compressed_bytes, compressed_sha256=compressed_hash.hexdigest())
    for field in ('raw_bytes', 'raw_sha256', 'compressed_bytes', 'compressed_sha256'):
        if actual[field] != expected[field]:
            raise ValueError('Compressed log integrity mismatch: ' + field)
    _check_time(deadline)
    return actual


def read_log(path, expected, *, max_raw_bytes, deadline=None):
    """Convenience reader bounded by the caller's raw cap; audits prefer sink."""
    output = io.BytesIO()
    verify_log(path, expected, max_raw_bytes=max_raw_bytes, deadline=deadline, sink=output.write)
    return output.getvalue()


class StreamingGzipLog:
    """Exclusive output file; every write becomes visible to evidence accounting."""
    def __init__(self, path, *, max_raw_bytes):
        self.path = Path(path)
        self.max_raw_bytes = _positive_limit(max_raw_bytes, 'raw byte limit')
        self._file = self.path.open('xb')
        self._encoder = zlib.compressobj(level=1, wbits=31)
        self._raw_hash, self._compressed_hash = hashlib.sha256(), hashlib.sha256()
        self.raw_bytes = self.compressed_bytes = 0
        self.closed = False
        self.failure = None
        self.receipt = None

    def _emit(self, encoded):
        if encoded:
            written = self._file.write(encoded)
            if written != len(encoded):
                raise OSError('Compressed log short write')
            self._compressed_hash.update(encoded)
            self.compressed_bytes += len(encoded)

    def write(self, raw):
        if self.closed or self.failure is not None:
            raise ValueError('Compressed log is closed or previously failed')
        if not isinstance(raw, bytes):
            raise TypeError('Transport log input must be bytes')
        if self.raw_bytes + len(raw) > self.max_raw_bytes:
            self.failure = 'Raw log byte limit'
            raise ValueError(self.failure)
        try:
            # Independent raw evidence hash, computed before the codec sees it.
            self._raw_hash.update(raw)
            self.raw_bytes += len(raw)
            self._emit(self._encoder.compress(raw))
            self._emit(self._encoder.flush(zlib.Z_SYNC_FLUSH))
            self._file.flush()
            return len(raw)
        except BaseException as error:
            self.failure = repr(error)
            raise

    def flush(self):
        if self.closed or self.failure is not None:
            raise ValueError('Compressed log is closed or previously failed')
        try:
            self._file.flush()
        except BaseException as error:
            self.failure = repr(error)
            raise

    def close_and_verify(self, deadline=None):
        if self.closed:
            if self.receipt is None:
                raise ValueError('Compressed log closed without verified receipt')
            return self.receipt
        try:
            _check_time(deadline)
            if self.failure is not None:
                raise ValueError('Earlier compressed log write failure: ' + self.failure)
            self._emit(self._encoder.flush(zlib.Z_FINISH))
            self._file.flush()
            os.fsync(self._file.fileno())
        except BaseException as error:
            self.failure = repr(error)
            raise
        finally:
            self._file.close()
            self.closed = True
        expected = dict(codec=CODEC, path=self.path.name, raw_bytes=self.raw_bytes,
                        raw_sha256=self._raw_hash.hexdigest(), compressed_bytes=self.compressed_bytes,
                        compressed_sha256=self._compressed_hash.hexdigest())
        self.receipt = verify_log(self.path, expected, max_raw_bytes=self.max_raw_bytes, deadline=deadline)
        return self.receipt

    def close(self):
        return self.close_and_verify()
