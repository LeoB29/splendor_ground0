"""Identity, bounds and evidence helpers for the separate live GPU study."""
from pathlib import Path
import hashlib,json,os,datetime

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_gpu_live_20261105_01'
PLAN=OUT/'plan.json'
PROTOCOL=ROOT/'docs/engineering/splendorust_gpu_live_20261105.md'
TOTAL_SECONDS=2*3600
WORKER_SECONDS=100*60
MAX_RSS=8*1024**3
MAX_OUTPUT=2*1024**3
MIN_DISK=1024**3
PRIOR=ROOT/'outputs/splendorust_gpu_probe_20261105_01'
PRIOR_CLOSEOUT_SHA='925446af34c0d0fe361bf8b41dd91e5a28661327ac2686d0c3dddfcad8a471d6'

def require(value,message):
    if not value:raise ValueError('GPU_LIVE: '+message)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def artifact(path):
    p=Path(path).resolve()
    return dict(path=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),bytes=p.stat().st_size,sha256=sha(p))
def resolve(item):return ROOT/Path(item['path'])
def verify(items):
    for item in items:require(artifact(resolve(item))==item,'Frozen input changed: '+item['path'])
def save(path,value):
    path=Path(path).resolve();require(path.is_relative_to(OUT),'Only new study evidence')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    return artifact(path)
def atomic_json(path,value):
    path=Path(path).resolve();require(path.is_relative_to(OUT),'Only mutable new runtime status')
    require(path.name in ('stage.json','active_decisions.json'),'Explicit mutable status names only')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
def nested_artifacts(value):
    if type(value) is dict:
        if set(value)=={'path','bytes','sha256'}:yield value
        else:
            for x in value.values():yield from nested_artifacts(x)
    elif type(value) is list:
        for x in value:yield from nested_artifacts(x)
