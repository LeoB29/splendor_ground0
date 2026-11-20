"""Immutable study identity and symmetric aggregate resource supervision."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from tools.splendorust_search_scale_20261024 import common as original
from tools.splendorust_timing_opt_20261024.common import CandidateStudyWorker
BaselineStudyWorker = CandidateStudyWorker
from tools.splendorust_boundary_timing_20261025.parent_profile import ProfiledStudyWorker

ROOT = original.ROOT
OUT = ROOT / 'outputs/splendorust_live_boundary_20261025_01'
PRIOR = ROOT / 'outputs/splendorust_boundary_timing_20261025_01'
PRIOR_FREEZE_SHA = '919544571d367be1a3ecfc463063206302c4dbebe048ee6fae232d73c103702e'
PROBE = ROOT / 'outputs/splendorust_sampler_probe_20261024_01'
PROTOCOL = ROOT / 'docs/engineering/splendorust_live_boundary_20261025.md'
GiB, MiB = 1024**3, 1024**2
MAX_EVIDENCE_BYTES = 2*GiB
FINAL_RESERVE_BYTES = 256*MiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
JOB_SECONDS = 21600
COLLECTION_CUTOFF_SECONDS = 19800
STAGES = {'qualification': 600, 'operational': 1800, 'main': 16200, 'final': 1800}
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
    require(path.resolve().is_relative_to(OUT.resolve()), 'Output outside live-boundary reliability study')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER(path, path.stat().st_size)
    return artifact(path)
