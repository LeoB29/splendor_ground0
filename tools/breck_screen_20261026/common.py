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
OUT = ROOT / 'outputs/breck_screen_20261026_01'
PRIOR = ROOT / 'outputs/splendorust_boundary_timing_20261025_01'
PRIOR_FREEZE_SHA = '919544571d367be1a3ecfc463063206302c4dbebe048ee6fae232d73c103702e'
PROBE = ROOT / 'outputs/splendorust_sampler_probe_20261024_01'
PROTOCOL = ROOT / 'docs/engineering/breck_screen_20261026.md'
GiB, MiB = 1024**3, 1024**2
MAX_EVIDENCE_BYTES = GiB
FINAL_RESERVE_BYTES = 128*MiB
PREFINAL_EVIDENCE_BYTES = MAX_EVIDENCE_BYTES - FINAL_RESERVE_BYTES
JOB_SECONDS = 14400
COLLECTION_CUTOFF_SECONDS = 10800
STAGES = {'qualification': 3600, 'operational': 900, 'main': 6300, 'final': 3600}
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
    require(path.resolve().is_relative_to(OUT.resolve()), 'Output outside Breck admission study')
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
