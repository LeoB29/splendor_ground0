"""Separate bounded saved-position256 efficiency diagnostic identity."""
from datetime import datetime, timezone
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_search256_efficiency_20261031_01'
OLD_OUT=ROOT/'outputs/splendorust_search256_strength_20261029_02'
ACCEPTED_OUT=ROOT/'outputs/splendorust_search256_recovery_20261030_02'
ACCEPTED_FREEZE_SHA='6cf5f864e451926cf530c8b78468ac7f496fff159f51cab0d62d29b5c80d8149'
PROTOCOL=ROOT/'docs/engineering/splendorust_search256_efficiency_20261031.md'
PHASE='strength'
GiB,MiB=1024**3,1024**2
JOB_SECONDS=10800
COLLECTION_CUTOFF_SECONDS=7200
STAGES={'qualification':1800,'overhead':1800,'diagnostic':3600,'final':3600}
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
OBSERVER_MODES=('off','on')
EFFICIENCY_OBSERVERS={'observer_off':'off','observer_on':'on'}
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

