"""New output boundary and fixed actual bounds, with no host RAM admission gate."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_OUT = ROOT / 'outputs/splendorust_collection4096_20261115_01'
PREVIOUS_OUT = ROOT / 'outputs/splendorust_collection4096_recovery_20261116_01'
PREVIOUS_FREEZE_SHA = 'd1074484d6dbf522288d9ff9354550d7e1c0759925512d976f9582443c301615'
ORIGINAL_FREEZE_SHA = 'ac15c01e9e4e2f36b3de6f525bf6a268c271535496df0e39c553945882a5eb88'
OUT = ROOT / 'outputs/splendorust_collection4096_resume_20261118_01'
STUDY_OUT = OUT
PROTOCOL = ROOT / 'docs/engineering/splendorust_collection4096_resume_20261118.md'
PACKAGE = 'tools.splendorust_collection4096_resume_20261118'
SCHEMA = 'splendorust_collection4096_timeout_resume_v2'
PREPARATION_OUT = ROOT / 'outputs/splendorust_collection_preparation_20261115_01'
PRIOR_OUT = ROOT / 'outputs/splendorust_independent_data_20261107_01'
NEW_GAME_COUNTS = {'train':3072, 'validation':128, 'offline':128}
NEW_GAMES = 3328
CELLS = 208
PLAN = OUT / 'freeze.json'
SELECTION = OUT / 'selection.json'
SCOPE = PHASE = 'pilot'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = TOTAL_SECONDS = 168 * 3600
COLLECTION_CUTOFF_SECONDS = MODEL_CUTOFF_SECONDS = MODEL_SECONDS = 144 * 3600
FINAL_SECONDS = 24 * 3600
INDEPENDENT_REVIEW_SECONDS = 12 * 3600
INITIAL_AVAILABLE_RAM_BYTES = MINIMUM_AVAILABLE_RAM_BYTES = STARTUP_MINIMUM_AVAILABLE_RAM_BYTES = 0
INITIAL_RAM_STABLE_SECONDS = RAM_WAIT_SECONDS = 0
MAX_INFERENCE_RSS_BYTES = MAX_RSS_BYTES = 2 * GiB
MAX_EVIDENCE_BYTES = MAX_OUTPUT = 96 * GiB
FINAL_RESERVE_BYTES = 8 * GiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
MINIMUM_FREE_DISK_BYTES = INITIAL_FREE_DISK_BYTES = MIN_DISK = 5 * GiB
INTERNAL_SECONDS, SECONDS_PER_MOVE = 9., 10.
DEPLOYMENT_TARGET_SECONDS, DESCRIPTIVE_MARGIN_SECONDS = 1., .7
MAX_PLIES = 2000
MAX_LOG_BYTES = GiB
MAX_RESPONSE_BYTES, MAX_REQUEST_BYTES = 8 * MiB, MiB
MAX_WORKERS = 2
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
PARENT_SHA256 = PARENT_CHECKPOINT_SHA
PARENT_CHECKPOINT = ROOT / 'outputs/external_opponent_search_20261018_02/splendorust_model.pt'
COLLECTION_RSS = 8 * GiB
FIT_RSS = 4 * GiB
QUALIFICATION_SECONDS = 6 * 3600
FIT_SECONDS = 4 * 3600
EVALUATION_SECONDS = 6 * 3600
AUDIT_SECONDS = 8 * 3600
ANALYSIS_SECONDS = 12 * 3600
WRITE_OBSERVER = None
STAGES = {'qualification': 6 * 3600, 'collection': 120 * 3600,
          'corpus': 24 * 3600, 'final': 24 * 3600}


def require(condition, message):
    if not condition:
        raise ValueError('COLLECTION4096: ' + message)


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


def resolve(item):
    return (ROOT / item['path']).resolve()


def verify(items, boundary=lambda: None):
    for item in items:
        boundary()
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


atomic_json = status


def fits():
    return []  # This preparation authorizes collection and discarded qualification only.
