"""Qualification identity and prospective bounds; frozen studies stay unchanged."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splendorust_selfplay_qualification_20261101_01'
PILOT_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_20261101_01'
PROTOCOL = ROOT / 'docs/engineering/splendorust_selfplay_qualification_20261101.md'
SCHEMA = 'splendorust_selfplay_engineering_qualification_v1'
PHASE = 'qualification'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 2 * 3600
COLLECTION_CUTOFF_SECONDS = 90 * 60
FINAL_SECONDS = 30 * 60
INDEPENDENT_REVIEW_SECONDS = 20 * 60
INITIAL_AVAILABLE_RAM_BYTES = 4 * GiB
MINIMUM_AVAILABLE_RAM_BYTES = 2 * GiB
STARTUP_MINIMUM_AVAILABLE_RAM_BYTES = 2 * GiB
INITIAL_RAM_STABLE_SECONDS = 60
MAX_INFERENCE_RSS_BYTES = 2 * GiB
MAX_FITTER_RSS_BYTES = 4 * GiB
MAX_RSS_BYTES = MAX_INFERENCE_RSS_BYTES
RAM_WAIT_SECONDS = 15 * 60
INITIAL_FREE_DISK_BYTES = 9 * GiB
MINIMUM_FREE_DISK_BYTES = 5 * GiB
MAX_EVIDENCE_BYTES = 4 * GiB
FINAL_RESERVE_BYTES = GiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
INTERNAL_SECONDS = 9.
SECONDS_PER_MOVE = 10.
DEPLOYMENT_TARGET_SECONDS = 1.
DESCRIPTIVE_MARGIN_SECONDS = .7
MAX_GAMES = 4
MAX_PLIES = 2000
MAX_FIXTURE_DECISIONS = 256
MAX_OPTIMIZER_STEPS = 1
MAX_BATCH_SIZE = 32
MAX_LOG_BYTES = 256 * MiB
MAX_RESPONSE_BYTES = 8 * MiB
MAX_REQUEST_BYTES = MiB
MAX_WORKERS = 2
ACCEPTED_OUT = ROOT / 'outputs/splendorust_search512_20261101_01'
ACCEPTED_FREEZE_SHA = '3b312ace0a3779d12dcbd55263b415d9dedb82e52a087d3379fc3b76b7c2536a'
ACCEPTED_REVIEW_SHA = 'f0cca5543e5c91fff63d8df4e6de9a86bd2b98cb719f64c72cbc19114245988c'
ACCEPTED_CLOSEOUT = ROOT / 'outputs/splendorust_search512_20261101_01_closeout/final_closeout.json'
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
WRITE_OBSERVER = None

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
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=file_hash(path))

def verify(items):
    for item in items:
        require(set(item) == {'path', 'bytes', 'sha256'}, 'Exact artifact fields')
        given = Path(item['path'])
        path = (ROOT / given).resolve()
        require(not given.is_absolute() and path.is_relative_to(ROOT.resolve())
                and path.relative_to(ROOT).as_posix() == item['path'], 'Canonical workspace input')
        require(path.is_file(), 'Missing frozen input: ' + item['path'])
        before = path.stat()
        observed = file_hash(path)
        after = path.stat()
        stable = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        require(stable(before) == stable(after) and after.st_size == item['bytes']
                and observed == item['sha256'], 'Changed frozen input: ' + item['path'])

def verify_external_runtime(items):
    """Only the exact externally located runtime files already admitted in512."""
    require(file_hash(ACCEPTED_OUT/'freeze.json') == ACCEPTED_FREEZE_SHA,
            'Authenticated predecessor runtime inventory')
    expected = sorted((x for x in read(ACCEPTED_OUT/'freeze.json')['inputs']
                       if Path(x['path']).is_absolute()), key=lambda x: x['path'])
    require(len(expected) == 13 and sorted(items, key=lambda x: x['path']) == expected,
            'Exact thirteen inherited external runtime files; no new external paths')
    for item in expected:
        require(set(item) == {'path', 'bytes', 'sha256'}, 'Exact external runtime descriptor')
        path = Path(item['path'])
        require(path.is_absolute() and path.is_file(), 'Existing absolute runtime file')
        before = path.stat()
        observed = file_hash(path)
        after = path.stat()
        stable = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        require(stable(before) == stable(after) and after.st_size == item['bytes']
                and observed == item['sha256'], 'Changed inherited external runtime: ' + item['path'])
    return expected

def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()), 'Qualification evidence path only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER(path, path.stat().st_size)
    return artifact(path)

def atomic_save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    require(not path.exists(), 'Existing immutable receipt')
    save(pending, value)
    require(not path.exists(), 'Existing immutable receipt')
    os.rename(pending, path)
    return artifact(path)

STAGES = {'qualification': 1800, 'collection': 1800, 'fit': 1200, 'final': 1800}
MAX_WORKER_LIFECYCLES = 3
MAX_REPLAY_DECISIONS = 12
MAX_LIVE_DECISIONS = MAX_GAMES * MAX_PLIES + MAX_FIXTURE_DECISIONS + MAX_REPLAY_DECISIONS
