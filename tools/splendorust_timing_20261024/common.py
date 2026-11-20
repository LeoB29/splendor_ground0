"""Timing-study identity and additive observation of the unchanged supervisor."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from tools.splendorust_search_scale_20261024 import common as original

ROOT = original.ROOT
OUT = ROOT / 'outputs/splendorust_timing_20261024_01'
PRIOR = original.OUT
PRIOR_FREEZE_SHA = 'dd9cfdfd71584788e9d3f89c1f90330f4959015e568372c1ef75b1bf2998cd53'
PROTOCOL = ROOT / 'docs/engineering/splendorust_timing_20261024.md'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 2700
STAGES = {'qualification': 600, 'profiling': 1500, 'final': 600}
COLLECTION_CUTOFF_SECONDS = 2100
WRITE_OBSERVER = None
PROFILE = original.PROFILE
NATIVE_COPY = original.NATIVE_COPY
profile_id = original.profile_id
wrapper_id = original.wrapper_id
profile_for = original.profile_for
profile_path = original.profile_path
profile_sha256 = original.profile_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path),
            'bytes': path.stat().st_size}


def verify(items):
    for item in items:
        require(artifact(ROOT / item['path']) == item, 'Changed frozen input: ' + item['path'])


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), 'Output outside timing-study directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER(path, path.stat().st_size)
    return artifact(path)


class StudyWorker(original.StudyWorker):
    """All clocks/ownership/resource checks inherited without changing frequency.

    Optional nested parent timings are diagnostic only. Their sum overlaps the
    trusted move duration and must never be subtracted from that duration.
    """
    def __init__(self, *args, profile_enabled=False, aggregate_sample_guard=None, **kwargs):
        self.profile_enabled = bool(profile_enabled)
        self.aggregate_sample_guard = aggregate_sample_guard
        self._parent_spans = {}
        super().__init__(*args, **kwargs)

    def _timed(self, label, function, *args, **kwargs):
        if not self.profile_enabled:
            return function(*args, **kwargs)
        started = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            row = self._parent_spans.setdefault(label, {'calls': 0, 'wall_ns': 0})
            row['calls'] += 1
            row['wall_ns'] += time.perf_counter_ns() - started

    def _sample(self):
        def sample():
            telemetry = super(StudyWorker, self)._sample()
            if self.aggregate_sample_guard is not None:
                try:
                    self.aggregate_sample_guard(self, telemetry)
                except BaseException as error:
                    if not self._failure:
                        self._failure = 'resource_guard'
                        self._failure_detail = repr(error)
            return telemetry
        return self._timed('process_tree_sample', sample)

    def _check_resource_guard(self):
        return self._timed('resource_guard', super()._check_resource_guard)

    def request(self, *args, **kwargs):
        self._parent_spans = {}
        receipt = super().request(*args, **kwargs)
        if self.profile_enabled:
            receipt['parent_timing'] = {
                'scope': 'nested parent spans including pre-dispatch and post-arrival samples; not all charged to arrival',
                'spans': self._parent_spans,
            }
        return receipt
