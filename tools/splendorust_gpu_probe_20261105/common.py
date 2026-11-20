"""Small, separate GPU probe: immutable inputs and bounded receipts."""
from pathlib import Path
import hashlib,json,os,datetime

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/splendorust_gpu_probe_20261105_01'
PLAN=OUT/'plan.json'
PANEL=OUT/'panel.json'
PROTOCOL=ROOT/'docs/engineering/splendorust_gpu_probe_20261105.md'
TOTAL_SECONDS=45*60
WORKER_SECONDS=30*60
MAX_RSS=4*1024**3
MAX_OUTPUT=100*1024**2
MIN_DISK=1024**3

def require(condition,message):
    if not condition:raise ValueError(message)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def artifact(path):
    path=Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),bytes=path.stat().st_size,sha256=sha(path))

def resolve(item):return ROOT/Path(item['path'])
def verify(items):
    for item in items:require(artifact(resolve(item))==item,'Frozen input changed: '+item['path'])

def save(path,value):
    path=Path(path).resolve()
    require(path.is_relative_to(OUT),'New GPU-probe evidence only')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
    return artifact(path)

def nested_artifacts(value):
    if type(value) is dict:
        if set(value)=={'path','bytes','sha256'}:yield value
        else:
            for x in value.values():yield from nested_artifacts(x)
    elif type(value) is list:
        for x in value:yield from nested_artifacts(x)
