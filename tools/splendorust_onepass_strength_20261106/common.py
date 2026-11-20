"""New output boundary and fixed actual bounds, with no host RAM admission gate."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splendorust_onepass_strength_20261106_01'
STUDY_OUT = OUT
PREPARATION = ROOT / 'outputs/splendorust_onepass_strength_preparation_20261106_01'
PREPARED_PLAN = PREPARATION / 'prepared_plan.json'
PREPARED_PLAN_SHA = '740020ffe37041c2dc65eb1e429abb2e3366dc91375b7dd23d06d935664c00ba'
PROTOCOL = ROOT / 'docs/engineering/splendorust_onepass_strength_execution_20261106.md'
PACKAGE = 'tools.splendorust_onepass_strength_20261106'
SCHEMA = 'splendorust_onepass_strength_execution_v1'
SCOPE = PHASE = 'pilot'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = TOTAL_SECONDS = 72 * 3600
COLLECTION_CUTOFF_SECONDS = MODEL_CUTOFF_SECONDS = MODEL_SECONDS = 60 * 3600
FINAL_SECONDS = 12 * 3600
INDEPENDENT_REVIEW_SECONDS = 6 * 3600
INITIAL_AVAILABLE_RAM_BYTES = MINIMUM_AVAILABLE_RAM_BYTES = STARTUP_MINIMUM_AVAILABLE_RAM_BYTES = 0
INITIAL_RAM_STABLE_SECONDS = RAM_WAIT_SECONDS = 0
MAX_INFERENCE_RSS_BYTES = MAX_RSS_BYTES = 2 * GiB
MAX_EVIDENCE_BYTES = 48 * GiB
FINAL_RESERVE_BYTES = 8 * GiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
MINIMUM_FREE_DISK_BYTES = INITIAL_FREE_DISK_BYTES = 5 * GiB
INTERNAL_SECONDS, SECONDS_PER_MOVE = 9., 10.
DEPLOYMENT_TARGET_SECONDS, DESCRIPTIVE_MARGIN_SECONDS = 1., .7
MAX_PLIES = 2000
MAX_LOG_BYTES = GiB
MAX_RESPONSE_BYTES, MAX_REQUEST_BYTES = 8 * MiB, MiB
MAX_WORKERS = 2
MAX_WORKER_LIFECYCLES = MAX_ASSESSMENT_WORKER_LIFECYCLES = 850
MAX_TOTAL_GAMES = MAX_ASSESSMENT_GAMES = 412
MAX_ASSESSMENT_REPLAY_DECISIONS = MAX_REPLAY_DECISIONS = 192
MAX_LIVE_DECISIONS = MAX_TOTAL_GAMES * MAX_PLIES
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
WRITE_OBSERVER = None
STAGES = {'qualification': 4 * 3600, 'assessment_qualification': 4 * 3600,
          'operational': 4 * 3600, 'assessment': 40 * 3600, 'main': 40 * 3600,
          'reproduction': 8 * 3600, 'replay': 8 * 3600, 'final': 12 * 3600}


def require(condition, message):
    if not condition:
        raise ValueError('ONEPASS_STRENGTH: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


sha = file_hash


def artifact(path):
    path = Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
                bytes=path.stat().st_size, sha256=file_hash(path))


def verify(items):
    for item in items:
        require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'Exact artifact fields')
        require(type(item['path']) is str and type(item['bytes']) is int and item['bytes'] >= 0,
                'Strict artifact path and size')
        path = (ROOT / item['path']).resolve()
        require(Path(item['path']).is_absolute() or
                path.is_relative_to(ROOT) and path.relative_to(ROOT).as_posix() == item['path'],
                'Canonical input path')
        require(artifact(path) == item, 'Frozen input changed: ' + item['path'])


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT), 'New study output only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER(path, path.stat().st_size)
    return artifact(path)


def atomic_save(path, value):
    path = Path(path).resolve()
    require(not path.exists(), 'Immutable receipt already exists')
    temporary = path.with_suffix(path.suffix + '.pending')
    save(temporary, value)
    require(not path.exists(), 'Immutable receipt appeared')
    os.rename(temporary, path)
    return artifact(path)


def status(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT) and path.name in ('stage.json', 'progress.json'), 'Mutable status allowlist')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    from tools.splendorust_gpu_live_20261105_03.status_io import atomic_replace_status
    atomic_replace_status(temporary, path)


def nested_artifacts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= set(value):
            yield {key: value[key] for key in ('path', 'bytes', 'sha256')}
        else:
            for child in value.values():
                yield from nested_artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_artifacts(child)
