"""Fixed study identity, generous finite resources and exclusive evidence IO."""
from datetime import datetime, timezone
import hashlib,json,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_search512_20261101_01'
PROTOCOL=ROOT/'docs/engineering/splendorust_search512_20261101.md'
PHASE='strength'
GiB,MiB=1024**3,1024**2
JOB_SECONDS=72*3600
COLLECTION_CUTOFF_SECONDS=60*3600
STAGES={'qualification':4*3600,'operational':2*3600,'main':48*3600,'replay':4*3600,'final':12*3600}
INDEPENDENT_REVIEW_SECONDS=4*3600
INITIAL_AVAILABLE_RAM_BYTES=4*GiB
MINIMUM_AVAILABLE_RAM_BYTES=2*GiB
STARTUP_MINIMUM_AVAILABLE_RAM_BYTES=2*GiB
MAX_RSS_BYTES=2*GiB
RAM_WAIT_SECONDS=3*3600
INITIAL_FREE_DISK_BYTES=37*GiB
MINIMUM_FREE_DISK_BYTES=5*GiB
MAX_EVIDENCE_BYTES=32*GiB
FINAL_RESERVE_BYTES=8*GiB
PREFINAL_EVIDENCE_BYTES=MAX_EVIDENCE_BYTES-FINAL_RESERVE_BYTES
SECONDS_PER_MOVE=10.
INTERNAL_SECONDS=9.
DEPLOYMENT_TARGET_SECONDS=1.
DESCRIPTIVE_MARGIN_SECONDS=.7
VARIANTS=('control','candidate')
ROUTE_QUOTAS={'control':256,'candidate':512}
ROUTE_CPUCTS={'control':.4,'candidate':.4}
OPERATIONAL_GAMES=6
MAIN_GAMES=400
MAX_GAMES=OPERATIONAL_GAMES+MAIN_GAMES
MAX_PLIES=2000
MAX_LIVE_DECISIONS=MAX_GAMES*MAX_PLIES
MAX_REPLAY_DECISIONS=96
GAMES_PER_BLOCK=10
MAX_WORKERS=2
MAX_WORKER_LIFECYCLES=2*MAX_GAMES+2
MAX_GRAPH_BUILDS=MAX_WORKER_LIFECYCLES
MAX_GRAPH_CASES=11*MAX_WORKER_LIFECYCLES
MAX_SCIENTIFIC_QUALIFICATION=67*MAX_WORKER_LIFECYCLES
QUALIFICATION_DECISIONS=134
QUALIFICATION_ORDINARY_REQUESTS=0
MAX_LOG_BYTES=256*MiB
MAX_RESPONSE_BYTES=8*MiB
WRITE_OBSERVER=None

ACCEPTED_OUT=ROOT/'outputs/splendorust_search256_recovery_20261030_02'
ACCEPTED_FREEZE_SHA='6cf5f864e451926cf530c8b78468ac7f496fff159f51cab0d62d29b5c80d8149'
ACCEPTED_RECEIPT=ROOT/'outputs/splendorust_search256_recovery_20261030_02_closeout/independent_acceptance_01.json'
ACCEPTED_RECEIPT_SHA='40eb825c191ba57fd7e17738f4dd38426404592c7a29997f60c4fcc9ccc7c202'
LIVE_OUT=ROOT/'outputs/splendorust_network_live_20261031_01'
LIVE_FREEZE_SHA='ba428d48f0e2ae70774770da5279607faffba0827e02d78d69fa0cb79187d5b1'
LIVE_CLOSEOUT=ROOT/'outputs/splendorust_network_live_20261031_01_closeout/final_closeout.json'
LIVE_CLOSEOUT_SHA='35ccf54b8b1572fdc21073ab1048ea3bad30f381add40dd518047f391be844f5'
OLD_OUT=ROOT/'outputs/splendorust_search256_strength_20261029_02'
HANDOFF_HELPER=ROOT/'tools/splendorust_network_practical_handoff_20261031.py'
HANDOFF_HELPER_SHA='05262ebfd0744ffab2b9ab3b1696587a48f075933ce70a31a32cca7973ec2aa8'


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def artifact(path):
    path=Path(path).resolve()
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':file_hash(path),'bytes':path.stat().st_size}


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
            'Changed frozen input: '+item['path'])


def save(path,value):
    path=Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()),'Output outside fixed512 study')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    if WRITE_OBSERVER is not None:WRITE_OBSERVER(path,path.stat().st_size)
    return artifact(path)


def atomic_save(path,value):
    path=Path(path);pending=path.with_suffix(path.suffix+'.pending')
    require(not path.exists(),'Control receipt already published')
    save(pending,value)
    require(not path.exists(),'Control receipt already published')
    os.rename(pending,path)
    return artifact(path)
