"""Immutable study identity and symmetric aggregate resource supervision."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from tools.splendorust_search_scale_20261024 import common as original
from .sampler import BatchedStudyWorker

ROOT = original.ROOT
OUT = ROOT / 'outputs/splendorust_timing_opt_20261024_01'
PRIOR = ROOT / 'outputs/splendorust_timing_20261024_01'
PROBE = ROOT / 'outputs/splendorust_sampler_probe_20261024_01'
PROTOCOL = ROOT / 'docs/engineering/splendorust_timing_opt_20261024.md'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 5400
COLLECTION_CUTOFF_SECONDS = 4500
STAGES = {'sampler': 600, 'qualification32': 1200, 'operational32': 600,
          'qualification64': 900, 'operational64': 600, 'final': 900}
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
    require(path.resolve().is_relative_to(OUT.resolve()), 'Output outside optimization-study directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER(path, path.stat().st_size)
    return artifact(path)


class _AggregateGuard:
    """Apply the same aggregate check to each mode, including construction."""
    def __init__(self, *args, aggregate_sample_guard=None, **kwargs):
        self.aggregate_sample_guard = aggregate_sample_guard
        super().__init__(*args, **kwargs)

    def _sample(self):
        telemetry = super()._sample()
        if self.aggregate_sample_guard is not None:
            try:
                self.aggregate_sample_guard(self, telemetry)
            except BaseException as error:
                # Preserve the parent's trusted failure-receipt path and any
                # earlier, more specific supervision failure.
                if not self._failure:
                    self._failure = 'resource_guard'
                    self._failure_detail = repr(error)
        return telemetry


class BaselineStudyWorker(_AggregateGuard, original.StudyWorker):
    """Original process sampler with the common aggregate envelope."""


class CandidateStudyWorker(_AggregateGuard, BatchedStudyWorker):
    """One fresh snapshot per original poll; identical aggregate envelope."""
