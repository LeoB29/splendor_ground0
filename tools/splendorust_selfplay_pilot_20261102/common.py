"""Qualification identity and prospective bounds; frozen studies stay unchanged."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = os.environ.get('SELFPLAY_JOB_SCOPE', 'qualification')
if SCOPE not in ('qualification', 'pilot'):
    raise ValueError('SELFPLAY_JOB_SCOPE must be qualification or pilot')
QUALIFICATION_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_qualification_20261102_01'
PILOT_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_20261102_01'
OUT = QUALIFICATION_OUT if SCOPE == 'qualification' else PILOT_OUT
PROTOCOL = ROOT / 'docs/engineering/splendorust_selfplay_pilot_20261102.md'
SCHEMA = 'splendorust_selfplay_pilot_v1'
PHASE = SCOPE
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 150 * 60 if SCOPE == 'qualification' else 72 * 3600
COLLECTION_CUTOFF_SECONDS = 120 * 60 if SCOPE == 'qualification' else 60 * 3600
MODEL_CUTOFF_SECONDS = COLLECTION_CUTOFF_SECONDS
FINAL_SECONDS = 30 * 60 if SCOPE == 'qualification' else 12 * 3600
INDEPENDENT_REVIEW_SECONDS = 20 * 60 if SCOPE == 'qualification' else 6 * 3600
INITIAL_AVAILABLE_RAM_BYTES = 4 * GiB
MINIMUM_AVAILABLE_RAM_BYTES = 2 * GiB
STARTUP_MINIMUM_AVAILABLE_RAM_BYTES = 2 * GiB
INITIAL_RAM_STABLE_SECONDS = 60
MAX_INFERENCE_RSS_BYTES = 2 * GiB
MAX_FITTER_RSS_BYTES = 4 * GiB
MAX_FIT_EVIDENCE_BYTES = 512 * MiB if SCOPE == 'qualification' else 2 * GiB
MAX_RSS_BYTES = MAX_INFERENCE_RSS_BYTES
RAM_WAIT_SECONDS = 15 * 60 if SCOPE == 'qualification' else 3 * 3600
MAX_EVIDENCE_BYTES = (4 if SCOPE == 'qualification' else 48) * GiB
FINAL_RESERVE_BYTES = (1 if SCOPE == 'qualification' else 8) * GiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
MINIMUM_FREE_DISK_BYTES = 5 * GiB
INITIAL_FREE_DISK_BYTES = MAX_EVIDENCE_BYTES + MINIMUM_FREE_DISK_BYTES
INTERNAL_SECONDS = 9.
SECONDS_PER_MOVE = 10.
DEPLOYMENT_TARGET_SECONDS = 1.
DESCRIPTIVE_MARGIN_SECONDS = .7
MAX_GAMES = 4 if SCOPE == 'qualification' else 256
MAX_SOURCE_GAMES = MAX_GAMES
MAX_ASSESSMENT_GAMES = 6 if SCOPE == 'qualification' else 406
MAX_TOTAL_GAMES = MAX_SOURCE_GAMES + MAX_ASSESSMENT_GAMES
MAX_PLIES = 2000
MAX_FIXTURE_DECISIONS = 32
MAX_BATCH_SIZE = 32
MAX_OPTIMIZER_STEPS = 1 if SCOPE == 'qualification' else 2 * ((192 * MAX_PLIES + 31) // 32)
MAX_LOG_BYTES = GiB
MAX_RESPONSE_BYTES = 8 * MiB
MAX_REQUEST_BYTES = MiB
MAX_WORKERS = 2
SOURCE_GAMES_PER_WORKER = 8
MAX_SOURCE_WORKER_LIFECYCLES = 3 if SCOPE == 'qualification' else 34
MAX_ASSESSMENT_WORKER_LIFECYCLES = 4 if SCOPE == 'qualification' else 804
MAX_WORKER_LIFECYCLES = MAX_SOURCE_WORKER_LIFECYCLES + MAX_ASSESSMENT_WORKER_LIFECYCLES
MAX_SOURCE_REPLAY_DECISIONS = 12 if SCOPE == 'qualification' else 48
MAX_ASSESSMENT_REPLAY_DECISIONS = 12 if SCOPE == 'qualification' else 96
MAX_REPLAY_DECISIONS = MAX_SOURCE_REPLAY_DECISIONS + MAX_ASSESSMENT_REPLAY_DECISIONS
MAX_SOURCE_LIVE_DECISIONS = MAX_SOURCE_GAMES * MAX_PLIES + MAX_FIXTURE_DECISIONS + MAX_SOURCE_REPLAY_DECISIONS
MAX_LIVE_DECISIONS = MAX_TOTAL_GAMES * MAX_PLIES + MAX_FIXTURE_DECISIONS + MAX_REPLAY_DECISIONS
ACCEPTED_OUT = ROOT / 'outputs/splendorust_search512_20261101_01'
ACCEPTED_FREEZE_SHA = '3b312ace0a3779d12dcbd55263b415d9dedb82e52a087d3379fc3b76b7c2536a'
ACCEPTED_REVIEW_SHA = 'f0cca5543e5c91fff63d8df4e6de9a86bd2b98cb719f64c72cbc19114245988c'
ACCEPTED_CLOSEOUT = ROOT / 'outputs/splendorust_search512_20261101_01_closeout/final_closeout.json'
ENGINEERING_OUT = ROOT / 'outputs/splendorust_selfplay_qualification_20261101_01'
ENGINEERING_FREEZE_SHA = '59af3864e79509a22307386a8d73e80373e4934af10e54f558a7538b2de75736'
RECOVERED_ACCEPTANCE = ROOT / 'outputs/splendorust_selfplay_audit_20261102_02/final_closeout.json'
RECOVERED_ACCEPTANCE_SHA = '6857369b46d7f218ba6fcafb76bdb1e51f96b1fce3c83c11bdac03df24d6f206'
RECOVERED_INDEPENDENT_ACCEPTANCE = ROOT / 'outputs/splendorust_selfplay_audit_20261102_02_closeout/independent_acceptance_01.json'
RECOVERED_INDEPENDENT_ACCEPTANCE_SHA = 'f37b25084efa6c3ce60085f7aafdb0aa4f9dae18a221af97df1a378e867bc9cc'
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
WRITE_OBSERVER = None
STAGES = ({'qualification': 1800, 'collection': 3600, 'fit': 1200,
           'assessment_qualification': 1200, 'operational': 1200, 'assessment': 0, 'reproduction': 600, 'final': 1800}
          if SCOPE == 'qualification' else
          {'qualification': 4 * 3600, 'collection': 34 * 3600, 'fit': 4 * 3600,
           'assessment_qualification': 3600, 'operational': 3600, 'assessment': 16 * 3600,
           'reproduction': 4 * 3600, 'final': 12 * 3600})

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
    require(path.is_relative_to(OUT.resolve()), 'Current frozen self-play scope evidence path only')
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

