"""Recovery-only paths, immutable I/O and cheap shared deadline guards."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
import psutil
ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=ROOT/'outputs/splendorust_isolated_reliability_20261026_01'
OUT=ROOT/'outputs/splendorust_reliability_recovery_20261026_01'
PROTOCOL=ROOT/'docs/engineering/splendorust_reliability_recovery_20261026.md'
ORIGINAL_SHA='a7be16152ff398eb5feca4f68419539c5817148d9fe31d74c7441ae6a5b63497'
JOB_SECONDS=1800
GiB=1024**3
MAX_RSS=2*GiB
MAX_OUTPUT=128*1024**2
MIN_DISK=5*GiB
MIN_AVAILABLE=2*GiB

def require(value,message):
    if not value: raise ValueError('RECOVERY: '+message)

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def artifact(path,boundary=lambda:None):
    path=Path(path).resolve();require(path.is_relative_to(ROOT),'Input outside repository')
    total=0;h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(1024*1024): boundary();total+=len(data);h.update(data)
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=total,sha256=h.hexdigest())

def verify(items,boundary=lambda:None):
    for item in items:
        boundary();require(artifact(ROOT/item['path'],boundary)==item,'Changed input: '+item['path'])

def save(path,value):
    path=Path(path).resolve();require(path.is_relative_to(OUT),'Write outside separate recovery output')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    return artifact(path)

def audit_event(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','os.posix_spawn','os.spawn'):
        raise RuntimeError('RECOVERY: no subprocess or model worker permitted inside auditor')
    if event=='open':
        path,mode,flags=args
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing:
            require(not isinstance(path,int),'Raw writable file descriptors forbidden')
            require(Path(path).resolve().is_relative_to(OUT),'Audit attempted to write outside separate output')
    if event in ('os.remove','os.rmdir','os.rename','os.link','os.symlink','os.chmod','os.utime','os.mkdir','os.truncate'):
        # Recovery uses only exclusive new output creation. Never delete or rewrite.
        if event=='os.mkdir':require(Path(args[0]).resolve().is_relative_to(OUT),'Directory mutation outside recovery')
        else:raise RuntimeError('RECOVERY: mutation operation forbidden: '+event)

class Envelope:
    placement_mode='isolated'
    def __init__(self,claim):
        self.start=claim['monotonic_origin'];self.wall_start=claim['unix_origin']
        self.last_resource=-1.;self.phase='startup';self.sequence=0
        self.process=psutil.Process();self.peak_rss=0
    def remaining(self):return min(self.start+JOB_SECONDS-time.monotonic(),self.wall_start+JOB_SECONDS-time.time())
    def guard(self):
        require(self.remaining()>0,'Recovery UTC/monotonic deadline')
        stamp=time.monotonic()
        if stamp-self.last_resource>=1.:
            self.last_resource=stamp
            require('torch' not in sys.modules,'Model runtime imported during saved-only audit')
            require(not self.process.children(recursive=True),'Unexpected auditor child process')
            rss=self.process.memory_info().rss;self.peak_rss=max(self.peak_rss,rss)
            require(rss<=MAX_RSS,'Audit RSS ceiling')
            require(psutil.virtual_memory().available>=MIN_AVAILABLE,'Available RAM reserve')
            require(shutil.disk_usage(ROOT).free>=MIN_DISK,'Free disk reserve')
            require(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<MAX_OUTPUT,'Recovery output ceiling')
    def checkpoint(self,phase,details=None):
        self.guard();self.phase=phase;self.sequence+=1
        save(OUT/'progress'/f'{self.sequence:03d}.json',dict(time=now(),phase=phase,elapsed_seconds=max(time.monotonic()-self.start,time.time()-self.wall_start),details=details))
        print(canonical(dict(phase=phase,remaining_seconds=self.remaining(),details=details)),flush=True)
    def final_guard(self):
        self.last_resource=-1.
        self.guard()
