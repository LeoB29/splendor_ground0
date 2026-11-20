"""Bounded, conservative evidence accounting; no player or model operations.

The inherited compressor already serializes all three stream writes, conversion
of empty placeholders, and final close under ``child._log_lock``.  This observer
takes that SAME lock for one finite directory snapshot.  It neither changes the
codec nor writes logs while holding the lock.  Lock waits remain charged to the
caller's original clock; ``boundary`` must check time only, not reenter accounting.

Child JSON publication is independent of that lock.  Each initialization/request
therefore reserves 128 MiB of metadata growth against a fixed starting snapshot.
The reservation is not reset by a guard call and is released only after the
caller has collected the original receipt and sidecars.  This covers files
created after directory enumeration without an unbounded rescan.  The caller
must freeze this resource contract and stop on a reservation violation.

Only a registered current request's authenticated pending-to-final child-sidecar
publication, and the unchanged compressor's empty-log-to-gzip conversion, may
explain a disappearing path.  Deletion, replacement, shrinkage, reparse points,
unknown temporary names, excessive growth and lock/deadline failures remain
fatal.  These byte observations never accept or reinterpret a player response.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading

SCHEMA = 'splendorust_recovery_evidence_accounting_v1'
MiB = 1024**2
GROWTH_RESERVE_BYTES = 128 * MiB
MAX_ENTRIES = 32768
LOG_NAMES = ('stdout.log', 'stderr.log', 'events.jsonl')
REQUEST_ID = re.compile(r'request-[0-9]{8}\Z')
PROFILE_SUFFIX = '_child_profile'
PROFILE_SCHEMA = 'splendorust_boundary_timing_child_v1'
MAX_PROFILE_BYTES = 8192 + 1
MAX_BATCH_BYTES = 512 * 1024 + 1


class AccountingError(ValueError):
    """An accounting/infrastructure error; never a player deadline forfeit."""


def _require(condition, message):
    if not condition:
        raise AccountingError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name)
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _identity(info):
    _require(info.st_ino != 0, 'Unobservable file identity')
    return (info.st_dev, info.st_ino)


def _regular(info, name, *, directory=False):
    _require(not getattr(info, 'st_file_attributes', 0) & 0x400, 'Reparse point: ' + name)
    _require((stat.S_ISDIR if directory else stat.S_ISREG)(info.st_mode), 'Unexpected file type: ' + name)
    _require(info.st_size >= 0, 'Negative file size: ' + name)
    return info


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON identity field')
        result[key] = value
    return result


def _no_constant(value):
    raise AccountingError('Nonfinite sidecar JSON: ' + value)


def publication_identity(request_id, payload, *, pid, quota, graph_identity_sha256=None):
    """Bind the existing transport's exact bytes; this does not send a request."""
    _require(type(request_id) is str and REQUEST_ID.fullmatch(request_id), 'Request identity')
    _integer(pid, 'Child PID', 1)
    _require(type(quota) is int and quota in (16, 64, 128, 256), 'Fixed scientific quota')
    _require(type(payload) is dict, 'Request payload')
    raw = (json.dumps(dict(type='request', request_id=request_id, payload=payload),
                      separators=(',', ':'), allow_nan=False) + '\n').encode()
    _require(len(raw) <= MiB, 'Inherited request byte limit')
    expected = dict(schema=PROFILE_SCHEMA, request_id=request_id, pid=pid, quota=quota,
                    backend='qualified_graph64', payload_sha256=_digest(payload),
                    request_raw_sha256=hashlib.sha256(raw).hexdigest())
    if graph_identity_sha256 is not None:
        _require(type(graph_identity_sha256) is str and re.fullmatch('[0-9a-f]{64}', graph_identity_sha256),
                 'Graph identity digest')
        expected['graph_identity_sha256'] = graph_identity_sha256
    startup = payload.get('operation') in ('entrypoint_checks', 'graph_parity')
    return dict(request_id=request_id, expected=expected,
                maximum_bytes=MAX_BATCH_BYTES if startup else MAX_PROFILE_BYTES,
                pending=request_id+PROFILE_SUFFIX+'.pending', final=request_id+PROFILE_SUFFIX+'.json')


class WorkerAccounting:
    """One registered new worker folder; no historical tree is ever rewritten.

    Register before the constructor creates its directory.  Bind ``child`` from
    the existing aggregate sample hook before the first resource guard.  Finish
    initialization after READY evidence, before graph/entrypoint requests.
    The caller serializes lifecycle methods with its existing request loop.
    """
    def __init__(self, folder, *, growth_reserve_bytes=GROWTH_RESERVE_BYTES, max_entries=MAX_ENTRIES):
        folder = Path(folder)
        _require(folder.is_absolute(), 'Worker folder must be absolute')
        _require(folder.parent.name == 'workers' and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', folder.name),
                 'Registered worker folder naming')
        _require(folder.absolute() == folder.resolve(), 'Worker folder must not traverse a link')
        _require(not folder.exists(), 'Register only a new, absent worker folder')
        _require(type(growth_reserve_bytes) is int and 0 < growth_reserve_bytes <= GROWTH_RESERVE_BYTES,
                 'Fixed bounded metadata reserve')
        _require(type(max_entries) is int and 0 < max_entries <= MAX_ENTRIES, 'Finite entry limit')
        self.folder = folder
        self.growth_reserve_bytes, self.max_entries = growth_reserve_bytes, max_entries
        self._known = {}
        self._root_identity = None
        self._child = self._log_lock = None
        self._scanning = False
        self._state = 'initializing'
        self._base_metadata_bytes = 0
        self._publication = None
        self._preserved_pending = {}
        self._owned_closed = False
        self._last = None
        self._publications = []
        self._rotations = []

    @property
    def active_request_id(self):
        return None if self._publication is None else self._publication['request_id']

    @property
    def state(self):
        return self._state

    def bind_child(self, child):
        _require(Path(child.output_dir).resolve() == self.folder, 'Child folder ownership differs')
        _require(type(child._log_lock) is type(threading.Lock()), 'Original transport log lock required')
        _require(set(child._logs) == set(LOG_NAMES), 'Original three-stream inventory')
        _require(type(child.max_log_bytes) is int and child.max_log_bytes == 128 * MiB,
                 'Inherited raw stream bound changed')
        _require(self._child is None or self._child is child, 'Worker identity cannot be rebound')
        self._child, self._log_lock = child, child._log_lock

    @contextmanager
    def _locked(self, timeout):
        _require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 <= timeout <= 1.,
                 'Bounded log-lock allowance')
        if self._log_lock is None:
            _require(not self.folder.exists() and self._state == 'initializing',
                     'Live worker accounting needs its original log lock')
            yield
            return
        _require(self._log_lock.acquire(timeout=timeout), 'Evidence accounting log-lock deadline')
        try:
            yield
        finally:
            self._log_lock.release()

    def _validate_final(self, path, publication):
        info = _regular(os.stat(path, follow_symlinks=False), path.name)
        _require(info.st_size <= publication['maximum_bytes'], 'Child sidecar exceeds its frozen bound')
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            _require(_identity(opened) == _identity(info), 'Sidecar changed while opening')
            raw = stream.read(publication['maximum_bytes'] + 1)
            after = os.fstat(stream.fileno())
        _require(len(raw) == info.st_size and after.st_size == info.st_size and
                 _identity(after) == _identity(info), 'Sidecar changed while reading')
        _require(raw.endswith(b'\n'), 'Published sidecar is incomplete')
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object, parse_constant=_no_constant)
        _require(type(value) is dict and all(value.get(k) == v for k, v in publication['expected'].items()),
                 'Published sidecar request/source identity differs')
        return info, hashlib.sha256(raw).hexdigest()

    def _publication_replacement(self, relative, previous=None):
        publication = self._publication
        _require(publication is not None and relative == publication['pending'],
                 'Unrelated missing evidence: ' + relative)
        target = self.folder / publication['final']
        info, digest = self._validate_final(target, publication)
        if previous is not None:
            _require(_identity(info) == previous['identity'] and info.st_size >= previous['bytes'],
                     'Pending sidecar was deleted/replaced, not renamed')
        return publication['final'], info, dict(pending=relative, final=publication['final'],
                    sha256=digest, bytes=info.st_size, request_id=publication['request_id'])

    def _scan(self, boundary):
        """One bounded enumeration, while the inherited log lock is held."""
        try:
            root_info = os.stat(self.folder, follow_symlinks=False)
        except FileNotFoundError:
            _require(self._child is None and self._root_identity is None and self._state == 'initializing',
                     'Worker folder disappeared')
            return {}, [], [], 0
        root = _regular(root_info, '.', directory=True)
        ident = _identity(root)
        _require(self._root_identity is None or ident == self._root_identity, 'Worker folder identity changed')
        self._root_identity = ident
        seen, publications, rotations, count = {}, [], [], 0
        pending_dirs = [self.folder]
        processed = 0
        while pending_dirs:
            directory = pending_dirs.pop()
            entries = []
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    count += 1
                    _require(count <= self.max_entries, 'Worker evidence entry bound')
                    entries.append(entry)
            for entry in sorted(entries, key=lambda e: e.name):
                processed += 1
                if processed % 128 == 0:
                    boundary()
                path = Path(entry.path)
                relative = path.relative_to(self.folder).as_posix()
                try:
                    info = os.stat(path, follow_symlinks=False)
                except FileNotFoundError:
                    # No generic retry/skip: only this exact current publication.
                    name, info, proof = self._publication_replacement(relative, self._known.get(relative))
                    publications.append(proof)
                    relative = name
                if stat.S_ISDIR(info.st_mode):
                    _regular(info, relative, directory=True)
                    pending_dirs.append(path)
                    record = dict(identity=_identity(info), bytes=0, directory=True)
                else:
                    _regular(info, relative)
                    if relative.endswith('.pending'):
                        allowed = (self._publication if self._publication is not None and
                                   relative == self._publication['pending'] else self._preserved_pending.get(relative))
                        _require(allowed is not None,
                                 'Unauthenticated pending file: ' + relative)
                        _require(info.st_size <= allowed['maximum_bytes'], 'Pending sidecar byte limit')
                    record = dict(identity=_identity(info), bytes=info.st_size, directory=False)
                if relative in seen:
                    _require(seen[relative] == record, 'Duplicate/changed publication target')
                else:
                    seen[relative] = record
        for relative, old in self._known.items():
            current = seen.get(relative)
            if current is not None:
                _require(old['identity'] == current['identity'] and old['directory'] == current['directory'],
                         'Evidence replaced: ' + relative)
                _require(current['bytes'] >= old['bytes'], 'Evidence shrank: ' + relative)
                continue
            if relative in LOG_NAMES and old['bytes'] == 0 and not old['directory']:
                converted = seen.get(relative + '.gz')
                _require(converted is not None and not converted['directory'], 'Missing empty log has no gzip replacement')
                handle = self._child._logs[relative] if self._child is not None else None
                _require(handle is not None and Path(getattr(handle, 'path', '')).resolve() == self.folder / (relative+'.gz'),
                         'Gzip replacement is not the owned compressor')
                rotations.append(dict(empty_placeholder=relative, compressed=relative+'.gz'))
            elif self._publication is not None and relative == self._publication['pending']:
                name, info, proof = self._publication_replacement(relative, old)
                record = dict(identity=_identity(info), bytes=info.st_size, directory=False)
                _require(name not in seen or seen[name] == record, 'Changed sidecar after enumeration')
                seen[name] = record
                publications.append(proof)
            else:
                raise AccountingError('Evidence disappeared: ' + relative)
        for name in LOG_NAMES:
            plain, zipped = seen.get(name), seen.get(name+'.gz')
            _require(not (plain and zipped), 'Unfinished log conversion under exclusive log lock')
            if plain is not None:
                _require(plain['bytes'] == 0, 'Uncompressed placeholder is nonempty')
            if zipped is not None:
                handle = self._child._logs[name] if self._child is not None else None
                _require(handle is not None and Path(getattr(handle, 'path', '')).resolve() == self.folder/(name+'.gz'),
                         'Unowned compressed evidence')
                _require(zipped['bytes'] <= 128*MiB*16+128, 'Inherited physical compressed-stream bound')
                # _emit can hold buffered bytes; its counter is an upper bound
                # on visible st_size until the original flush completes.
                counter = _integer(getattr(handle, 'compressed_bytes', None), 'Compressed byte counter')
                _require(counter >= zipped['bytes'] and counter <= 128*MiB*16+128,
                         'Compressed size/counter disagreement')
                zipped['bytes'] = counter
        return seen, publications, rotations, count

    def snapshot(self, *, boundary=lambda: None, lock_timeout=1.):
        _require(not self._scanning, 'Accounting must not reenter itself')
        self._scanning = True
        try:
            boundary()
            with self._locked(lock_timeout):
                seen, publications, rotations, count = self._scan(boundary)
                boundary()
                stream_names = set(LOG_NAMES) | {n+'.gz' for n in LOG_NAMES}
                log_bytes = sum(v['bytes'] for k, v in seen.items() if k in stream_names)
                metadata = sum(v['bytes'] for k, v in seen.items() if k not in stream_names)
                active = self._state in ('initializing', 'request')
                if active:
                    _require(metadata <= self._base_metadata_bytes+self.growth_reserve_bytes,
                             'Fixed lifecycle metadata growth reserve exceeded')
                    charged_metadata = self._base_metadata_bytes+self.growth_reserve_bytes
                else:
                    charged_metadata = metadata
                result = dict(schema=SCHEMA, state=self._state, request_id=self.active_request_id,
                    observed_bytes=metadata+log_bytes, accounted_bytes=charged_metadata+log_bytes,
                    metadata_bytes=metadata, compressed_stream_bytes=log_bytes,
                    reserved_bytes=charged_metadata-metadata,
                    fixed_growth_reserve_bytes=self.growth_reserve_bytes if active else 0,
                    base_metadata_bytes=self._base_metadata_bytes if active else None,
                    entries=count, pending_publications=publications, empty_log_conversions=rotations,
                    stream_snapshot_under_original_log_lock=self._log_lock is not None,
                    byte_measurement_is_conservative=True, future_stream_growth_not_precharged=True)
                self._known, self._last = seen, result
                self._publications.extend(publications)
                self._rotations.extend(rotations)
                return dict(result)
        finally:
            self._scanning = False

    def finish_initialization(self, **kwargs):
        _require(self._state == 'initializing', 'Initialization lifecycle order')
        proof = self.snapshot(**kwargs)
        self._state = 'idle'
        self._base_metadata_bytes = proof['metadata_bytes']
        return self.snapshot(**kwargs)

    def begin_request(self, request_id, payload, *, pid, quota, graph_identity_sha256=None,
                      boundary=lambda: None, lock_timeout=1.):
        _require(self._state == 'idle' and not self._owned_closed, 'Overlapping/closed worker request')
        publication = publication_identity(request_id, payload, pid=pid, quota=quota,
                                            graph_identity_sha256=graph_identity_sha256)
        proof = self.snapshot(boundary=boundary, lock_timeout=lock_timeout)
        _require(publication['pending'] not in self._known and publication['final'] not in self._known,
                 'Request sidecar already exists')
        self._base_metadata_bytes = proof['metadata_bytes']
        self._publication, self._state = publication, 'request'
        return dict(schema=SCHEMA, publication=publication, base_metadata_bytes=self._base_metadata_bytes,
                    growth_reserve_bytes=self.growth_reserve_bytes)

    def finish_request(self, *, require_published, **kwargs):
        _require(type(require_published) is bool and self._state == 'request', 'Request lifecycle order')
        publication = self._publication
        proof = self.snapshot(**kwargs)
        target = self.folder/publication['final']
        if target.exists():
            self._validate_final(target, publication)
        else:
            _require(not require_published, 'Successful request lacks its published sidecar')
        _require(not require_published or not (self.folder/publication['pending']).exists(),
                 'Successful publication left a pending sidecar')
        if (self.folder/publication['pending']).exists():
            _require(self._owned_closed and not require_published,
                     'Unfinished sidecar allowed only after failed owned cleanup')
            self._preserved_pending[publication['pending']] = publication
        self._publication = None
        self._state = 'closed' if self._owned_closed else 'idle'
        self._base_metadata_bytes = proof['metadata_bytes']
        return self.snapshot(**kwargs)

    def close(self, **kwargs):
        """After original owned cleanup; active failure reservation stays until collection."""
        if self._child is not None:
            _require(getattr(self._child, '_closed', False), 'Cannot close accounting before original worker cleanup')
        self._owned_closed = True
        if self._state != 'request':
            self._state = 'closed'
        return self.snapshot(**kwargs)


def enforce_cap(total_bytes, cap_bytes):
    """Original strict '< cap' admission; equality is already a failure."""
    _integer(total_bytes, 'Accounted bytes')
    _integer(cap_bytes, 'Evidence cap', 1)
    _require(total_bytes < cap_bytes, 'Study evidence ceiling')
    return total_bytes
