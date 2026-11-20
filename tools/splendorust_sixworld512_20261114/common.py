"""Frozen scientific choices, bounded ownership, exclusive lossless evidence."""
from datetime import datetime,timezone
import gzip,hashlib,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PACKAGE='tools.splendorust_sixworld512_20261114'
STUDY='splendorust_sixworld512_sequential_20261114_v1'
OUT=ROOT/'outputs/splendorust_sixworld512_20261114_01'
PLAN=OUT/'freeze.json'
PROTOCOL=ROOT/'docs/engineering/splendorust_sixworld512_sequential_execution_20261114.md'
PREPARATION_PROTOCOL=ROOT/'docs/engineering/splendorust_sixworld512_sequential_preparation_20261114.md'
PRIOR_OUT=ROOT/'outputs/splendorust_teacher_variability_20261113_01'
PRIOR_FREEZE=PRIOR_OUT/'freeze.json'
PRIOR_FREEZE_SHA='60239d1f7221aa8f25557a6ebae5b18218d24c521fd086dceb0f796ed6aada54'
PRIOR_CLOSEOUT=ROOT/'outputs/splendorust_teacher_variability_closeout_20261114_01/closeout_complete.json'
PRIOR_CLOSEOUT_SHA='3d23c1835568cf60c2d6e3fd3ecd15def3bbf776f7f44aea45e94cc7960d9457'
GRAPH_PLAN=ROOT/'outputs/splendorust_inference_opt_20261025_01/graph_parity_plan.json'
PARENT_CHECKPOINT=ROOT/'outputs/external_opponent_search_20261018_02/splendorust_model.pt'
PARENT_CHECKPOINT_SHA='ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
IDENTITY=dict(ruleset_id='splendor_base_qualified_rules_20261018_v1',
 catalogue_sha256='db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312',
 turn_contract='atomic_prereveal_completed_turn_v1',origin='generated_under_declared_rules')
GiB=1024**3
MiB=1024**2
TOTAL_SECONDS=240*3600
MODEL_SECONDS=216*3600
MODEL_RSS=4*GiB
AUDIT_RSS=8*GiB
MAX_OUTPUT=96*GiB
MIN_DISK=5*GiB
SECONDS_PER_MOVE=120.
INTERNAL_SECONDS=119.
OPERATIONAL_GAMES=6
MAIN_GAMES=3600
MAX_GAMES=3606
GAMES_PER_BLOCK=400
MAX_PLIES=2000
MAX_LIVE_DECISIONS=MAX_GAMES*MAX_PLIES
BLOCKS=tuple(f'block_{i:02d}' for i in range(1,10))
JOBS=('qualification','operational',*BLOCKS,'analysis','acceptance')
MODEL_JOBS=('qualification','operational',*BLOCKS)
CAPS=dict(qualification=6*3600,operational=4*3600,analysis=12*3600,acceptance=12*3600,
 **{r:24*3600 for r in BLOCKS})

def require(value,message):
 if not value:raise ValueError('SIXWORLD512: '+message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def _unique(pairs):
 result={}
 for key,value in pairs:
  require(key not in result,'Duplicate JSON key');result[key]=value
 return result
def read(path):
 path=Path(path)
 with (gzip.open(path,'rb') if path.suffix=='.gz' else path.open('rb')) as f:
  raw=f.read(256*MiB+1)
 require(len(raw)<=256*MiB,'Bounded JSON input')
 return json.loads(raw,object_pairs_hook=_unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
file_hash=sha
def artifact(path):
 path=Path(path).resolve()
 return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),bytes=path.stat().st_size,sha256=sha(path))
def resolve(item):return (ROOT/item['path']).resolve()
def verify(items,boundary=lambda:None):
 for item in items:
  boundary();require(set(item)=={'path','bytes','sha256'},'Exact artifact descriptor')
  require(artifact(resolve(item))==item,'Changed input: '+item['path'])
def _destination(path):
 path=Path(path).resolve();require(path.is_relative_to(OUT),'Exclusive new output boundary')
 path.parent.mkdir(parents=True,exist_ok=True);return path
def save(path,value):
 path=_destination(path)
 with path.open('x',encoding='utf-8',newline='\n') as f:
  f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
 return artifact(path)
def save_gzip(path,value):
 path=_destination(path);require(path.suffix=='.gz','Explicit gzip artifact')
 raw=(canonical(value)+'\n').encode()
 with path.open('xb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=3) as z:z.write(raw)
  f.flush();os.fsync(f.fileno())
 require(read(path)==value,'Lossless gzip roundtrip')
 return artifact(path)
def atomic_save(path,value):
 path=_destination(path);require(not path.exists(),'Exclusive receipt')
 temporary=path.with_suffix(path.suffix+'.pending');save(temporary,value);os.rename(temporary,path);return artifact(path)
def status(path,value):
 path=_destination(path);require(path.name in ('stage.json','progress.json'),'Mutable status allowlist')
 temporary=path.with_suffix('.tmp')
 with temporary.open('w',encoding='utf-8',newline='\n') as f:
  f.write(canonical(value)+'\n');f.flush();os.fsync(f.fileno())
 from tools.splendorust_gpu_live_20261105_03.status_io import atomic_replace_status
 atomic_replace_status(temporary,path)
def identity():
 import psutil
 p=psutil.Process();return dict(pid=p.pid,created=p.create_time())
def exact_alive(ident):
 import psutil
 try:
  p=psutil.Process(ident['pid']);return p.create_time()==ident['created'] and p.is_running()
 except psutil.NoSuchProcess:return False
def nested_artifacts(value):
 if isinstance(value,dict):
  if set(value)=={'path','bytes','sha256'}:yield value
  else:
   for v in value.values():yield from nested_artifacts(v)
 elif isinstance(value,list):
  for v in value:yield from nested_artifacts(v)
def check_deadlines(limits):
 require(limits['monotonic_origin']<=time.monotonic()<limits['monotonic_deadline'],'Original monotonic allowance expired')
 require(limits['unix_origin']<=time.time()<limits['unix_deadline'],'Original UTC allowance expired')
def design():
 return dict(study_id=STUDY,network=PARENT_CHECKPOINT_SHA,pools=[3,6],simulations=512,depth=64,cpuct=.4,
  sampling='legacy_shared_stream_A',dtype='float32',inference_threads=1,inference_batch=1,
  games=MAIN_GAMES,pairs=MAIN_GAMES//2,operational_games=6,interim_games=list(range(400,3601,400)),
  teacher_lower_bound=.48,player_lower_bound=.50,material_harm_upper_bound=.48,
  alpha_per_direction=.025,teacher_margin=.02,old_55pct_gate_applies=False,seat_gate_applies=False,
  conditional_power_threshold=.10,futility_first_games=800,complete_pair_unit=True,
  live_replays_per_block=8,operational_replays=6,one_second_and_700ms_descriptive=True,
  internal_seconds=119.,complete_seconds=120.,timeout_policy='forfeit_game_never_retry_decision',
  max_plies=MAX_PLIES,total_seconds=TOTAL_SECONDS,model_seconds=MODEL_SECONDS,job_caps=CAPS,
  model_rss=MODEL_RSS,audit_rss=AUDIT_RSS,output_cap=MAX_OUTPUT,actual_disk_floor=MIN_DISK,
  concurrent_games=1,available_ram_gate=False,forecast_gate=False,
  exact_corrected_entrypoint=True,lossless_result_compression=True,record_before_dispatch=True,
  no_new_training=True,no_automatic_followup=True,**IDENTITY)
