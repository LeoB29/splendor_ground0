"""Identity and evidence helpers for the separately frozen whole-game collector comparison."""
from pathlib import Path
import hashlib, json, os, datetime

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/splendorust_gpu_collector_20261105_01'
PLAN = OUT/'plan.json'
PROTOCOL = ROOT/'docs/engineering/splendorust_gpu_collector_20261105.md'
TOTAL_SECONDS = 12*3600
MODEL_SECONDS = 10*3600
INTERNAL_SECONDS = 60.
OUTER_SECONDS = 75.
# Execution clocks and scientific player wrapper remain unchanged. The new
# scheduling identity is separately recorded by the backend and frozen plan.
EXECUTION_POLICY_ID = 'throughput_fixed512_internal60_outer75_v1'
MAX_RSS = 8*1024**3
MAX_OUTPUT = 12*1024**3
MIN_DISK = 1024**3

def require(value, message):
    if not value: raise ValueError('GPU_COLLECTOR: '+message)
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def artifact(path):
    p=Path(path).resolve()
    return dict(path=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p), bytes=p.stat().st_size, sha256=sha(p))
def resolve(item): return ROOT/Path(item['path'])
def verify(items):
    for item in items: require(artifact(resolve(item))==item, 'Frozen input changed: '+item['path'])
def save(path, value):
    path=Path(path).resolve(); require(path.is_relative_to(OUT), 'Only new study evidence')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
    return artifact(path)
def atomic_json(path,value):
    path=Path(path).resolve(); require(path.is_relative_to(OUT), 'Only new runtime status')
    require(path.name in ('stage.json','active_decisions.json'), 'Explicit mutable status only')
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    from tools.splendorust_gpu_live_20261105_03.status_io import atomic_replace_status
    atomic_replace_status(temporary,path)
def nested_artifacts(value):
    if type(value) is dict:
        if set(value)=={'path','bytes','sha256'}: yield value
        else:
            for x in value.values(): yield from nested_artifacts(x)
    elif type(value) is list:
        for x in value: yield from nested_artifacts(x)
