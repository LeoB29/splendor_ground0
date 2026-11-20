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
PHASE = os.environ.get('SPLENDOR_SEARCH256_STRENGTH_PHASE','strength')
if PHASE != 'strength':
    raise ValueError('Unknown search256 study phase')
OUT = ROOT / 'outputs/splendorust_search256_recovery_20261030_02'
QUAL_OUT = OUT
PRIOR_ATTEMPT_OUT = ROOT / 'outputs/splendorust_search256_strength_20261029_01'
PRIOR_ATTEMPT_FREEZE_SHA = '66e33176b7894a99d9a3505c3a3ae1a576804baa62e649bf0bce4900ea941523'
PRIOR_ORIGIN = 564433.078
PRIOR_WALL_START = 1793308578.3462226
PRIOR_RAM_WAIT_SECONDS = 1821.0
PRIOR_WORKER_TREES = 608
# All preserved attempt01/02 evidence and their held diagnosis bytes.
PRIOR_EVIDENCE_BYTES = 5115862569
PRIOR = ROOT / 'outputs/splendorust_boundary_timing_20261025_01'
PRIOR_FREEZE_SHA = '919544571d367be1a3ecfc463063206302c4dbebe048ee6fae232d73c103702e'
PROBE = ROOT / 'outputs/splendorust_sampler_probe_20261024_01'
PROTOCOL = ROOT / 'docs/engineering/splendorust_search256_recovery_20261030_02.md'
PREPARED_PROTOCOL = ROOT / 'docs/engineering/splendorust_exploration128_20261029_02_results.md'
GiB, MiB = 1024**3, 1024**2
MAX_EVIDENCE_BYTES = 24*GiB-PRIOR_EVIDENCE_BYTES
FINAL_RESERVE_BYTES = 8*GiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES-FINAL_RESERVE_BYTES
INDEPENDENT_REVIEW_SECONDS = 10800
JOB_SECONDS = 48*3600
COLLECTION_CUTOFF_SECONDS = 36*3600
STAGES = {'recovery_qualification':7200,'main':100800,'main_verification':21600,'final':43200}
RAM_WAIT_SECONDS = 10800
MAX_WORKERS = 260
MAX_GRAPH_BUILDS = MAX_WORKERS
INITIAL_FREE_DISK_BYTES = 32*GiB
SECONDS_PER_MOVE = 10.
INTERNAL_SECONDS = 9.
DEPLOYMENT_TARGET_SECONDS = 1.
DESCRIPTIVE_MARGIN_SECONDS = .7
INITIAL_AVAILABLE_RAM_BYTES = 4*GiB
MINIMUM_AVAILABLE_RAM_BYTES = 2*GiB
STARTUP_MINIMUM_AVAILABLE_RAM_BYTES = 2*GiB
MAX_RSS_BYTES = 2*GiB
ROUTE_QUOTAS = {'frozen':128,'research':256}
ROUTE_CPUCTS = {'frozen':.4,'research':.4}
QUALIFICATION_DECISIONS = 134
QUALIFICATION_ORDINARY_REQUESTS = 0
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
        require(set(item)=={'path','sha256','bytes'},'Exact artifact descriptor fields')
        supplied=Path(item['path'])
        path=supplied if supplied.is_absolute() else (ROOT/supplied).resolve()
        if not supplied.is_absolute():
            require(path.is_relative_to(ROOT.resolve()) and path.relative_to(ROOT.resolve()).as_posix()==item['path'],
                    'Canonical workspace artifact path')
        require(path.is_file(),'Missing frozen input: '+item['path'])
        before=path.stat();observed=file_hash(path);after=path.stat()
        stable=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns)
        require(stable(before)==stable(after) and after.st_size==item['bytes'] and observed==item['sha256'],
                'Changed frozen input: ' + item['path'])


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), 'Output outside fixed256 strength study')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:
        WRITE_OBSERVER(path, path.stat().st_size)
    return artifact(path)


def atomic_save(path,value):
    """Publish a complete exclusive control receipt without a partial-JSON race."""
    path=Path(path);pending=path.with_suffix(path.suffix+'.pending')
    require(not path.exists(),'Control receipt already published')
    result=save(pending,value)
    require(not path.exists(),'Control receipt already published')
    os.rename(pending,path)
    return artifact(path)


def validate_exchange(request,ack,freeze,watchdog,runner_identity):
    from . import placement
    require(type(request['sequence']) is int and request['sequence']>0 and
        request['freeze_sha256']==watchdog['freeze_sha256'] and
        request['runner']==runner_identity and request['watchdog']==dict(pid=watchdog['watchdog_pid'],created=watchdog['watchdog_created']),
        'Placement request ownership/freeze differs')
    require(ack['sequence']==request['sequence'] and ack['request_sha256']==digest(request) and
        ack['freeze_sha256']==request['freeze_sha256'] and ack['placement_mode']==request['placement_mode'],
        'Placement acknowledgement identity differs')
    placement.verify_saved_runtime(request['runner_receipt'],freeze,request['placement_mode'],identity=runner_identity)
    placement.verify_saved_runtime(ack['watchdog_receipt'],freeze,request['placement_mode'],role='watchdog',identity=request['watchdog'])
    require(ack['watchdog_receipt']['monotonic_ns']>=request['runner_receipt']['monotonic_ns'],
        'Placement acknowledgement precedes request')
    require(ack['watchdog_receipt']['monotonic_ns']-request['runner_receipt']['monotonic_ns']<5_000_000_000,
        'Placement acknowledgement exceeds5s')
    return True

# Preserved original attempt02 and authorized single interrupted-move retry.
OLD_OUT = ROOT / 'outputs/splendorust_search256_strength_20261029_02'
OLD_FREEZE_SHA = 'f85737b248e274e6be780461001b4294a1e937cccb70e78ac0e3afcd43878e53'
DIAGNOSIS_OUT = ROOT / 'outputs/splendorust_search256_strength_20261029_02_postinterrupt'
ORIGINAL_MONOTONIC = PRIOR_ORIGIN
ORIGINAL_UNIX = PRIOR_WALL_START
PRIOR_COMPLETED_MAIN_GAMES = 294
CONTINUATION_GAME_INDEX = 294
CONTINUATION_PLY = 50
ORIGINAL_MAIN_MONOTONIC_DEADLINE = 667240.328
ORIGINAL_MAIN_UNIX_DEADLINE = 1793411385.6187253
MAX_COMBINED_WORKERS = 868

# Closed, incomplete RAM-wait recovery is retained as an additional evidence layer.
PREVIOUS_OUT = ROOT / 'outputs/splendorust_search256_recovery_20261030_01'
PREVIOUS_FREEZE_SHA = 'af5174b5659914751776d636bf49cd13d7818cfaed88b4b161daa685da522a52'
PREVIOUS_DIAGNOSIS_OUT = ROOT / 'outputs/splendorust_search256_recovery_20261030_01_postfailure'
