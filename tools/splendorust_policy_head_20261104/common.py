"""Qualification identity and prospective bounds; frozen studies stay unchanged."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = os.environ.get('SELFPLAY_JOB_SCOPE', 'pilot')
if SCOPE != 'pilot':
    raise ValueError('Policy-head-only study uses the pilot scope; source collection is disabled')
QUALIFICATION_OUT = ROOT / 'outputs/splendorust_policy_head_qualification_20261104_01'
PILOT_OUT = ROOT / 'outputs/splendorust_policy_head_20261104_01'
OUT = QUALIFICATION_OUT if SCOPE == 'qualification' else PILOT_OUT
PROTOCOL = ROOT / 'docs/engineering/splendorust_policy_head_20261104.md'
SCHEMA = 'splendorust_policy_head_v1'
PHASE = SCOPE
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 72 * 3600
COLLECTION_CUTOFF_SECONDS = 60 * 3600
MODEL_CUTOFF_SECONDS = COLLECTION_CUTOFF_SECONDS
FINAL_SECONDS = 30 * 60 if SCOPE == 'qualification' else 12 * 3600
INDEPENDENT_REVIEW_SECONDS = 20 * 60 if SCOPE == 'qualification' else 6 * 3600
INITIAL_AVAILABLE_RAM_BYTES = 0
MINIMUM_AVAILABLE_RAM_BYTES = 0
STARTUP_MINIMUM_AVAILABLE_RAM_BYTES = 0
INITIAL_RAM_STABLE_SECONDS = 0
MAX_INFERENCE_RSS_BYTES = 2 * GiB
MAX_FITTER_RSS_BYTES = 4 * GiB
MAX_FIT_EVIDENCE_BYTES = 512 * MiB if SCOPE == 'qualification' else 2 * GiB
MAX_RSS_BYTES = MAX_INFERENCE_RSS_BYTES
RAM_WAIT_SECONDS = 15 * 60 if SCOPE == 'qualification' else 3 * 3600
MAX_EVIDENCE_BYTES = (4 if SCOPE == 'qualification' else 48) * GiB
FINAL_RESERVE_BYTES = (1 if SCOPE == 'qualification' else 8) * GiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
MINIMUM_FREE_DISK_BYTES = 5 * GiB
INITIAL_FREE_DISK_BYTES = MINIMUM_FREE_DISK_BYTES
INTERNAL_SECONDS = 9.
SECONDS_PER_MOVE = 10.
DEPLOYMENT_TARGET_SECONDS = 1.
DESCRIPTIVE_MARGIN_SECONDS = .7
MAX_GAMES = 4 if SCOPE == 'qualification' else 256
MAX_SOURCE_GAMES = MAX_GAMES
MAX_ASSESSMENT_GAMES = 6 if SCOPE == 'qualification' else 406
MAX_TOTAL_GAMES = MAX_ASSESSMENT_GAMES
MAX_NEW_SOURCE_GAMES = 0
MAX_PLIES = 2000
MAX_FIXTURE_DECISIONS = 32
MAX_BATCH_SIZE = 32
MAX_OPTIMIZER_STEPS = 1 if SCOPE == 'qualification' else 2 * ((192 * MAX_PLIES + 31) // 32)
MAX_LOG_BYTES = GiB
MAX_RESPONSE_BYTES = 8 * MiB
MAX_REQUEST_BYTES = MiB
MAX_WORKERS = 2
SOURCE_GAMES_PER_WORKER = 8
MAX_SOURCE_WORKER_LIFECYCLES = 0
MAX_PRESERVED_FAILED_SOURCE_WORKER_LIFECYCLES = 0
MAX_ASSESSMENT_WORKER_LIFECYCLES = 4 if SCOPE == 'qualification' else 804
MAX_WORKER_LIFECYCLES = MAX_SOURCE_WORKER_LIFECYCLES + MAX_ASSESSMENT_WORKER_LIFECYCLES + MAX_PRESERVED_FAILED_SOURCE_WORKER_LIFECYCLES
MAX_SOURCE_REPLAY_DECISIONS = 0
MAX_ASSESSMENT_REPLAY_DECISIONS = 12 if SCOPE == 'qualification' else 96
MAX_REPLAY_DECISIONS = MAX_SOURCE_REPLAY_DECISIONS + MAX_ASSESSMENT_REPLAY_DECISIONS
MAX_SOURCE_LIVE_DECISIONS = 0
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
FAILED_QUALIFICATION_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_qualification_20261102_01'
FAILED_WORKFLOW_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_workflow_20261102_01'
FAILED_FREEZE_SHA = '23e3a96214edb313f3c7876541567222633f33dbfd55b6c361ff0a526c4832fd'
CLOCK_DIAGNOSIS = ROOT / 'outputs/splendorust_selfplay_pilot_clock_diagnosis_20261102_01/diagnosis.json'
CLOCK_DIAGNOSIS_SHA = 'e08dc5adfeefdc4956b238bd621bac4238ec3933eb40695f1c8b7b28ac9e6740'
FAILED_PRESERVATION = QUALIFICATION_OUT / 'prelaunch/failed_attempt_preservation.json'
CLOCK_DIAGNOSIS_REVIEW = QUALIFICATION_OUT / 'prelaunch/independent_clock_diagnosis_review.json'
WRITE_OBSERVER = None
STAGES = ({'qualification': 1800, 'collection': 3600, 'fit': 1200,
           'assessment_qualification': 1200, 'operational': 1200, 'assessment': 0, 'reproduction': 600, 'final': 1800}
          if SCOPE == 'qualification' else
          {'qualification': 4 * 3600, 'collection': 0, 'fit': 8 * 3600,
           'assessment_qualification': 2 * 3600, 'operational': 2 * 3600, 'assessment': 40 * 3600,
           'reproduction': 8 * 3600, 'final': 12 * 3600})

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

def verify_failed_attempt_preservation(descriptor):
    """Recheck the stopped attempt without modifying or accepting its evidence."""
    require(descriptor == artifact(FAILED_PRESERVATION), 'Exact failed-attempt inventory')
    value = read(FAILED_PRESERVATION)
    require(value['schema'] == 'selfplay_failed_attempt_preservation_v1'
            and value['original_status'] == 'incomplete'
            and value['full_pilot_started'] is False
            and value['training_examples_reused'] == 0
            and value['checkpoint_reused'] is False
            and value['retrospective_acknowledgements'] == 0,
            'Failure is preserved, never accepted or reused')
    require(value['original_freeze'] == artifact(FAILED_QUALIFICATION_OUT/'freeze.json')
            and value['original_freeze']['sha256'] == FAILED_FREEZE_SHA,
            'Original failed freeze identity')
    expected_roots = [FAILED_QUALIFICATION_OUT, FAILED_WORKFLOW_OUT]
    require([tree['root'] for tree in value['trees']] ==
            [p.relative_to(ROOT).as_posix() for p in expected_roots],
            'Exactly the failed qualification and workflow trees')
    inventory = {item['path']: item for item in value['inputs']}
    require(len(inventory) == len(value['inputs']), 'Unique preservation inputs')
    for root, tree in zip(expected_roots, value['trees']):
        paths = sorted(p.relative_to(ROOT).as_posix() for p in root.rglob('*') if p.is_file())
        require(paths == sorted(tree['paths']) and len(paths) == tree['files']
                and sum(inventory[p]['bytes'] for p in paths) == tree['bytes'],
                'Complete failed evidence inventory unchanged')
    require(read(FAILED_QUALIFICATION_OUT/'completion.json')['status'] == 'incomplete'
            and not (FAILED_QUALIFICATION_OUT/'qualification_acceptance.json').exists(),
            'No retrospective acceptance of the stopped qualification')
    verify(value['inputs'])
    require(all(item in value['inputs'] for item in read(FAILED_QUALIFICATION_OUT/'freeze.json')['inputs']),
            'All original frozen workspace inputs preserved')
    return value

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


SOURCE_OUT = ROOT / 'outputs/splendorust_selfplay_pilot_resume2_20261103_01'
SOURCE_FREEZE_SHA = '12f25c84edaa970dfbab1927a93ec4f8cdf3447e82e189c8811283a2c3ab1e46'
SOURCE_CLOSEOUT_SHA = '33edbb9a8858d6a34cddc1893f6c69efeefead6c0e1eabdc3bb7d5c19a8b9db7'
