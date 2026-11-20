"""Exclusive pilot evidence and unchanged scientific identity. No launch at import."""
from datetime import datetime, timezone
import gzip, hashlib, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'tools.splendorust_parallel_throughput_20261115'
STUDY = 'splendorust_parallel_throughput_20261115_v1'
OUT = ROOT/'outputs/splendorust_parallel_throughput_20261115_01'
PREP = ROOT/'outputs/splendorust_parallel_throughput_preparation_20261115_01'
PLAN = OUT/'freeze.json'
PROTOCOL = ROOT/'docs/engineering/splendorust_parallel_throughput_execution_20261115.md'
BRIEF = ROOT/'docs/engineering/splendorust_parallel_throughput_pilot_brief_20261115.md'
ACTIVE_OUT = ROOT/'outputs/splendorust_collection4096_20261115_01'
ACTIVE_FREEZE_SHA = 'ac15c01e9e4e2f36b3de6f525bf6a268c271535496df0e39c553945882a5eb88'
PARENT_CHECKPOINT = ROOT/'outputs/external_opponent_search_20261018_02/splendorust_model.pt'
PARENT_CHECKPOINT_SHA = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
IDENTITY = dict(ruleset_id='splendor_base_qualified_rules_20261018_v1',
 catalogue_sha256='db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312',
 turn_contract='atomic_prereveal_completed_turn_v1', origin='generated_under_declared_rules')
GiB, MiB = 1024**3, 1024**2
TOTAL_SECONDS, MODEL_SECONDS = 6*3600, 16200
MODEL_RSS = AUDIT_RSS = PROCESS_RSS = 2*GiB
AGGREGATE_RSS, MAX_OUTPUT, MIN_DISK = 12*GiB, 10*GiB, 5*GiB
MAX_EVIDENCE_BYTES = MAX_OUTPUT
SECONDS_PER_MOVE, INTERNAL_SECONDS = 120., 119.
COLLECTION_COMPLETE_SECONDS, COLLECTION_INTERNAL_SECONDS = 75., 60.
MAX_PLIES = 2000

def require(value, message):
 if not value: raise ValueError('PARALLEL_THROUGHPUT: '+message)
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def _unique(pairs):
 result={}
 for key,value in pairs:
  require(key not in result,'Duplicate JSON key');result[key]=value
 return result
def read(path):
 path=Path(path)
 with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as stream: raw=stream.read(256*MiB+1)
 require(len(raw)<=256*MiB,'Bounded JSON input')
 return json.loads(raw,object_pairs_hook=_unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
file_hash=sha
def normalized(path):
 raw=str(Path(path).resolve())
 if raw.startswith('\\\\?\\UNC\\'):raw='\\\\'+raw[8:]
 elif raw.startswith('\\\\?\\'):raw=raw[4:]
 return Path(raw)
def artifact(path):
 path=normalized(path)
 return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),bytes=path.stat().st_size,sha256=sha(path))
def resolve(item): return normalized(ROOT/item['path'])
def verify(items,boundary=lambda:None):
 for item in items:
  boundary();require(set(item)=={'path','bytes','sha256'},'Exact artifact descriptor')
  require(artifact(resolve(item))==item,'Changed input: '+item['path'])
def _destination(path):
 path=normalized(path);require(path.is_relative_to(normalized(OUT)) or path.is_relative_to(normalized(PREP)),'Exclusive new pilot output')
 path.parent.mkdir(parents=True,exist_ok=True);return path
def save(path,value):
 path=_destination(path)
 with path.open('x',encoding='utf-8',newline='\n') as stream:
  stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
 return artifact(path)
def save_gzip(path,value):
 path=_destination(path);require(path.suffix=='.gz','Explicit gzip artifact')
 with path.open('xb') as stream:
  with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=3) as z:z.write((canonical(value)+'\n').encode())
  stream.flush();os.fsync(stream.fileno())
 require(read(path)==value,'Lossless compressed roundtrip');return artifact(path)
def atomic_save(path,value):
 path=_destination(path);require(not path.exists(),'Exclusive receipt')
 temporary=path.with_suffix(path.suffix+'.pending');save(temporary,value);os.rename(temporary,path);return artifact(path)
def status(path,value):
 path=_destination(path);require(path.name in ('stage.json','progress.json'),'Mutable status allowlist')
 temporary=path.with_suffix('.tmp')
 with temporary.open('w',encoding='utf-8',newline='\n') as stream:
  stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
 from tools.splendorust_gpu_live_20261105_03.status_io import atomic_replace_status
 atomic_replace_status(temporary,path)
def identity():
 import psutil
 process=psutil.Process();return dict(pid=process.pid,created=process.create_time())
def exact_alive(ident):
 import psutil
 try:
  process=psutil.Process(ident['pid']);return process.create_time()==ident['created'] and process.is_running()
 except psutil.NoSuchProcess:return False
def nested_artifacts(value):
 if isinstance(value,dict):
  if set(value)=={'path','bytes','sha256'}:yield value
  else:
   for item in value.values():yield from nested_artifacts(item)
 elif isinstance(value,list):
  for item in value:yield from nested_artifacts(item)
def check_deadlines(limits):
 require(limits['monotonic_origin']<=time.monotonic()<limits['monotonic_deadline'],'Original monotonic allowance expired')
 require(limits['unix_origin']<=time.time()<limits['unix_deadline'],'Original UTC allowance expired')
