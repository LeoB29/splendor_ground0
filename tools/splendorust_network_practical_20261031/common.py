"""Bounded practical network execution assessment identity and immutable IO."""
from datetime import datetime, timezone
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_network_practical_20261031_01'
PREVIOUS_OUT=ROOT/'outputs/splendorust_search256_efficiency_20261031_01'
PREVIOUS_FREEZE_SHA='1bc4796d0578fa3082d2f5f940ce366b3f899876ee06551f6750c5e047c555f3'
PREVIOUS_ACCEPTANCE=ROOT/'outputs/splendorust_search256_efficiency_20261031_01_closeout/independent_acceptance_01.json'
PREVIOUS_ACCEPTANCE_SHA='4ad6cbfee5a3192a8a6e59a73ceb01497696e6591f75b23459bae93520a65a2a'
OLD_OUT=ROOT/'outputs/splendorust_search256_strength_20261029_02'
ACCEPTED_OUT=ROOT/'outputs/splendorust_search256_recovery_20261030_02'
ACCEPTED_FREEZE_SHA='6cf5f864e451926cf530c8b78468ac7f496fff159f51cab0d62d29b5c80d8149'
PROTOCOL=ROOT/'docs/engineering/splendorust_network_practical_20261031.md'
PHASE='strength'
GiB,MiB=1024**3,1024**2
JOB_SECONDS=10800
COLLECTION_CUTOFF_SECONDS=7200
STAGES={'qualification':1800,'evaluation':3600,'final':3600}
INDEPENDENT_REVIEW_SECONDS=1800
INITIAL_AVAILABLE_RAM_BYTES=4*GiB
MINIMUM_AVAILABLE_RAM_BYTES=2*GiB
MAX_RSS_BYTES=2*GiB
RAM_WAIT_SECONDS=1800
INITIAL_FREE_DISK_BYTES=5*GiB
MAX_EVIDENCE_BYTES=4*GiB
FINAL_RESERVE_BYTES=GiB
PREFINAL_EVIDENCE_BYTES=MAX_EVIDENCE_BYTES-FINAL_RESERVE_BYTES
MAX_WORKERS=2
SECONDS_PER_MOVE=10.
INTERNAL_SECONDS=9.
VARIANTS=('control','candidate')
MAX_WORKER_LIFECYCLES=2
MAX_GRAPH_CASES=22
NUMERICAL_CASES_PER_WORKER=38
MAX_NUMERICAL_CASES=76
MAX_SCIENTIFIC_QUALIFICATION=134
PROFILE_FORWARD_CALLS=0
EVALUATION_REQUESTS=144
WRITE_OBSERVER=None

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
    require(path.resolve().is_relative_to(OUT.resolve()), 'Output outside fixed practical-execution study')
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


REJECTED_OUT=ROOT/'outputs/splendorust_network_execution_20261031_01'
REJECTED_CLOSEOUT=ROOT/'outputs/splendorust_network_execution_20261031_01_closeout/final_closeout.json'
REJECTED_CLOSEOUT_SHA='7502d6d6563cf3e929283c548a2ba1a5fcc098d7ac7b822626c9665cb9cb9c3d'
CANDIDATE_SOURCE=ROOT/'tools/splendorust_network_execution_20261031/candidate_final_token_ffn.py'
CANDIDATE_SOURCE_SHA='e10ff696e66dfe1cbc788c7968e053d7c07e2b0be3bf8f5ae411734c2906f870'
CANDIDATE_MODULE='tools.splendorust_network_execution_20261031.candidate_final_token_ffn'
CANDIDATE_ID='final_token_only_last_ffn_v1'
