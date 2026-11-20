"""Owned administrative cancellation, with immutable readiness and no score reads.

There is no cooperative stop hook in the frozen study. We therefore observe a
completed-pair audit gap where possible, terminate only the exact stage owner,
and let its existing parent perform Windows-job cleanup. Any raced or bounded-
fallback tail is preserved as administratively interrupted, never a game loss.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,sys,time
import psutil

ROOT=Path(__file__).resolve().parents[2]
ACTIVE=ROOT/'outputs/splendorust_sixworld512_20261114_02'
ACTIVE_SHA='5bf701f94cf9722f8b016a314f743e5d6dd0d0804c22171515c731b5732ed0de'
PREP=ROOT/'outputs/splendorust_collection_preparation_20261115_01'
POLICY=PREP/'priority_change.json'
OUT=ROOT/'outputs/splendorust_administrative_stop_20261115_01'
FREEZE=OUT/'freeze.json'
PACKAGE='tools.splendorust_administrative_stop_20261115'
ACTIVE_PACKAGE='tools.splendorust_sixworld512_20261114_v2'
TEACHER_SHA='ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'
WAIT_SECONDS=900
CLEANUP_SECONDS=60
BLOCKS=tuple(f'block_{n:02d}' for n in range(1,10))

def require(value,message):
 if not value:raise ValueError('ADMINISTRATIVE_STOP: '+message)
def now():return datetime.now(timezone.utc).isoformat()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def artifact(path):
 path=Path(path).resolve()
 return dict(path=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),bytes=path.stat().st_size,sha256=sha(path))
def resolve(item):return (ROOT/item['path']).resolve()
def verify(items):
 for item in items:
  require(set(item)=={'path','bytes','sha256'} and artifact(resolve(item))==item,'Exact immutable artifact')
def read(path):
 path=Path(path).resolve()
 require(not path.is_relative_to((ACTIVE/'interims').resolve()),'Interim score payloads are forbidden before cancellation')
 with path.open('rb') as stream:raw=stream.read(8*1024**2+1)
 require(len(raw)<=8*1024**2,'Bounded control JSON')
 def unique(pairs):
  result={}
  for key,value in pairs:
   require(key not in result,'Duplicate control JSON key');result[key]=value
  return result
 return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite control JSON')))
def save(path,value):
 path=Path(path).resolve();require(path.is_relative_to(OUT.resolve()),'Only new administrative evidence may be written')
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8',newline='\n') as stream:
  stream.write(canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
 return artifact(path)
def identity():
 p=psutil.Process();return dict(pid=p.pid,created=p.create_time())
def exact_alive(item):
 try:
  p=psutil.Process(item['pid']);return p.create_time()==item['created'] and p.is_running()
 except psutil.NoSuchProcess:return False

def sources():
 return [artifact(p) for p in sorted(Path(__file__).parent.glob('*.py'))]

def validate_readiness(path,expected_sha,policy_item):
 require(sha(path)==expected_sha,'Exact readiness receipt supplied to execute')
 value=read(path);verify([policy_item]);policy=read(resolve(policy_item))
 require(value.get('schema')=='splendor_collection_readiness_20261115_v1' and value.get('status')=='launch_ready'
  and value.get('qualified_preparation') is True,'Only independently qualified launch readiness can cancel')
 require(value.get('policy')==policy_item,'Readiness binds the prior score-blind user policy')
 require(value.get('teacher')==dict(checkpoint_sha256=TEACHER_SHA,pool=3,simulations=512),'Fixed original three-world teacher')
 rows=value.get('requirements');require(type(rows) is list and len(rows)==len(policy['readiness_requirements']),'Every readiness requirement')
 require([r.get('requirement') for r in rows]==policy['readiness_requirements'],'Exact readiness requirements and order')
 for row in rows:
  require(row.get('passed') is True and type(row.get('evidence')) is list and bool(row['evidence']),'Passed readiness with evidence')
  verify(row['evidence'])
 require(type(value.get('source_inputs')) is list and bool(value['source_inputs']),'Bound qualified preparation sources')
 verify(value['source_inputs'])
 return value

def scheduled_inventory():
 """Read only names and bytes of looks; never parse a score or choose on it."""
 rows=[]
 for index,role in enumerate(BLOCKS,1):
  look=ACTIVE/'interims'/f'look_{index*400:06d}.json'
  exited=ACTIVE/'jobs'/role/'actual_exit.json'
  result=ACTIVE/'jobs'/role/'worker/result.json'
  if not (look.exists() and exited.exists() and result.exists()):break
  end=read(exited);worker=read(result)
  require(end['status']=='passed' and end['exit_code']==0 and end['actual_exit_observed'] and end['exact_owned_absence']
   and not end['cleanup_errors'] and end['error'] is None and end['freeze_sha256']==ACTIVE_SHA,'Only completed verified block prefix')
  require(end['result_artifact']==artifact(result) and worker['status']=='passed' and worker['error'] is None
   and worker['identity']==end['identity'] and worker['role']==role and worker['freeze_sha256']==ACTIVE_SHA,'Exact passed block evidence')
  rows.append(dict(games=index*400,look=artifact(look),worker=artifact(result),actual_exit=artifact(exited)))
 return dict(looks=rows,latest_games=rows[-1]['games'] if rows else 0,
  selection='latest contiguous completed independently verified scheduled look; content unopened',scores_opened=False)

def stop_state():
 if (ACTIVE/'failure.json').exists():return dict(action='already_stopped',reason='existing_failure')
 if (ACTIVE/'completion.json').exists():return dict(action='already_stopped',reason='existing_completion')
 stage=read(ACTIVE/'stage.json');role=stage.get('stage')
 if role in ('analysis','acceptance','complete','failed'):
  return dict(action='already_stopped',reason='scientific_stop_or_finalization',stage=role)
 require(role in BLOCKS and stage.get('status')=='running','Only a running main block is cancellable')
 return dict(action='running',stage=role)

def authenticate_owners(role):
 require(role in BLOCKS and sha(ACTIVE/'freeze.json')==ACTIVE_SHA,'Exact frozen active study')
 launcher=read(ACTIVE/'launcher_claim.json');root=read(ACTIVE/'launch/owner.json');stage=read(ACTIVE/'jobs'/role/'owner.json')
 require(root['parent']==launcher['identity'] and stage['parent']==root['identity'],'Exact owned controller hierarchy')
 for row,expected in ((root,'root'),(stage,role)):
  require(row['freeze_sha256']==ACTIVE_SHA and row['windows_job_assigned'] is True and row['role']==expected,'Frozen Windows-job role')
 expected=[str(Path(sys.executable)), '-B','-u','-m',ACTIVE_PACKAGE+'.worker','--role',role,'--freeze-sha256',ACTIVE_SHA]
 require(stage['command']==expected,'Exact active worker command')
 return dict(launcher=launcher['identity'],root=root['identity'],stage=stage['identity'],role=role)

class RequestTracker:
 """Cache high water marks; subsequent probes touch only next request paths."""
 def __init__(self):self.counters={}
 def snapshot(self,folder):
  pending=[];lifetimes=[]
  for purpose in ('live','replay'):
   parent=folder/purpose/'services'
   if not parent.exists():continue
   for life in sorted(parent.glob('life_*')):
    if not life.is_dir():continue
    key=str(life)
    if key not in self.counters:
     numbers=[int(p.stem.split('-')[1]) for p in (life/'links').glob('request-*.json')]
     self.counters[key]=max(numbers,default=0)
    n=self.counters[key]
    while (life/'links'/f'request-{n+1:08d}.json').exists():n+=1
    self.counters[key]=n;next_id=f'request-{n+1:08d}'
    active=(life/'inputs'/f'{next_id}.json.gz').exists()
    if active:pending.append(dict(life=str(life.relative_to(ACTIVE)),request_id=next_id))
    lifetimes.append(dict(life=str(life.relative_to(ACTIVE)),complete_links=n,pending=active))
  return dict(pending=pending,lifetimes=lifetimes)

def pair_audit_gap(role,tracker):
 folder=ACTIVE/'jobs'/role/'worker';progress=folder/'arena/progress.json'
 p=read(progress) if progress.exists() else dict(games_completed=0)
 count=p['games_completed'];require(type(count) is int and 0<=count<=400,'Finite main progress')
 start=BLOCKS.index(role)*400
 candidates=[]
 # A terminal result is fsynced before its several-second independent game
 # audit. The small progress file is updated after that audit, so inspect one
 # game ahead of progress as well as the latest completed pair.
 if count<400 and (count+1)%2==0:candidates.append(count+1)
 if count>0 and count%2==0:candidates.append(count)
 for completed in candidates:
  index=start+completed-1;game=folder/'arena/games'/f'g{index:04d}'/'result.json'
  following=folder/'arena/games'/f'g{index+1:04d}'
  if not game.exists() or (completed<400 and following.exists()):continue
  try:
   ending=read(game)
   okay=ending.get('complete') is True and ending.get('spec',{}).get('game_index')==index
  except (OSError,ValueError,json.JSONDecodeError):continue
  if not okay:continue
  requests=tracker.snapshot(folder)
  if not requests['pending']:
   if completed<400 and following.exists():continue
   return dict(observed=True,completed_games_in_block=completed,last_game_index=index,
    terminal_game=artifact(game),following_game_absent=completed==400 or not following.exists(),requests=requests,
    atomic_boundary_guaranteed=False,score_fields_inspected=False)
 return dict(observed=False,completed_games_in_block=count,requests=tracker.snapshot(folder),atomic_boundary_guaranteed=False)

def process_inventory(owners):
 rows={}
 for name in ('launcher','root','stage'):
  item=owners[name];rows[(item['pid'],item['created'])]=dict(item)
  if name!='root' or not exact_alive(item):continue
  try:
   for p in psutil.Process(item['pid']).children(recursive=True):
    try:rows[(p.pid,p.create_time())]=dict(pid=p.pid,created=p.create_time())
    except psutil.NoSuchProcess:pass
  except psutil.NoSuchProcess:pass
 # Include earlier owners/services even if they are no longer descendants of
 # the current root. A PID reused by another program never matches its birth.
 for pattern in ('jobs/*/owner.json','jobs/*/actual_exit.json','launch/actual_exit.json',
  'jobs/*/worker/*/services/*/transport/initialization.json',
  'jobs/*/worker/*/services/*/native_startup.json'):
  for path in ACTIVE.glob(pattern):
   try:document=read(path)
   except (OSError,ValueError):continue
   items=([document['identity']] if isinstance(document.get('identity'),dict) else [])
   items+=document.get('observed_owned_identities',[])
   items += [dict(pid=r['pid'],created=r['created']) for r in document.get('rows',[]) if 'pid' in r and 'created' in r]
   for item in items:rows[(item['pid'],item['created'])]=dict(pid=item['pid'],created=item['created'])
 return list(rows.values())

def terminate_exact(item):
 """One explicit birth-bound process, never a name or broad PID pattern."""
 try:
  process=psutil.Process(item['pid'])
  if process.create_time()!=item['created'] or not process.is_running():return False
  process.kill();return True
 except psutil.NoSuchProcess:return False

def metadata_inventory():
 rows=[]
 for directory,subdirs,files in os.walk(ACTIVE):
  subdirs.sort()
  for name in sorted(files):
   path=Path(directory)/name;st=path.stat()
   rows.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
 require(len(rows)<1000000,'Bounded preservation inventory')
 return rows

def freeze(tests_path,review_paths):
 require(not FREEZE.exists(),'Separate administrative freeze is exclusive')
 test=read(tests_path);require(test['status']=='passed' and test['exit_code']==0 and test['inputs']==sources()
  and test['passed']==test['collected']>=12 and test['failed']==test['skipped']==test['models']==test['new_games']==0
  and test['tensor_runtime_loaded'] is False,'Current source-bound model-free tests')
 require(bool(review_paths),'At least one independent scoped review')
 for path in review_paths:
  review=read(path);require(review['approved'] is True and not review['blocking_findings'],'Independent cancellation approval');verify(review['inputs'])
  require(artifact(__file__) in review['inputs'],'Actual cancellation implementation reviewed')
 policy=read(POLICY);require(policy['active_freeze_sha256']==ACTIVE_SHA and policy['new_interim_scores_opened_for_this_decision'] is False,'Prior score-blind instruction')
 require(sha(ACTIVE/'freeze.json')==ACTIVE_SHA,'Unchanged active freeze')
 return save(FREEZE,dict(schema='splendor_administrative_stop_freeze_20261115_v1',source_inputs=sources(),policy=artifact(POLICY),
  active_freeze=artifact(ACTIVE/'freeze.json'),wait_seconds=WAIT_SECONDS,cleanup_seconds=CLEANUP_SECONDS,
  tests=artifact(tests_path),reviews=[artifact(p) for p in review_paths],time=now(),
  never_read_interim_payloads=True,never_change_sources=True,never_retry_decisions=True,never_reset_clocks=True,
  administrative_interruption_is_not_a_forfeit=True,atomic_boundary_guaranteed=False))

def execute(freeze_sha,readiness_path,readiness_sha):
 require(sha(FREEZE)==freeze_sha,'Exact administrative freeze')
 frozen=read(FREEZE);verify(frozen['source_inputs']+[frozen['policy'],frozen['active_freeze']])
 validate_readiness(readiness_path,readiness_sha,frozen['policy'])
 claim=save(OUT/'execution_claim.json',dict(identity=identity(),freeze=artifact(FREEZE),readiness=artifact(readiness_path),time=now(),
  reason='User-authorized replacement readiness; independent of study score'))
 initial=scheduled_inventory();save(OUT/'scheduled_before.json',initial)
 tracker=RequestTracker();start=time.monotonic();observations=0;target=None;gap=None
 while True:
  state=stop_state()
  if state['action']=='already_stopped':
   return save(OUT/'cancellation.json',dict(status='not_cancelled_already_stopped',reason=state,
    claim=claim,scheduled=scheduled_inventory(),models=0,new_games=0,time=now()))
  role=state['stage'];owners=authenticate_owners(role)
  if not exact_alive(owners['stage']):
   require(time.monotonic()-start<WAIT_SECONDS+30,'Bounded stale-stage observation');time.sleep(.1);continue
  gap=pair_audit_gap(role,tracker);observations+=1
  if gap['observed'] or time.monotonic()-start>=WAIT_SECONDS:
   current=stop_state()
   if current['action']!='running' or current['stage']!=role:continue
   current_owners=authenticate_owners(role)
   if current_owners!=owners:continue
   target=owners;break
  time.sleep(.1)
 # This durable receipt precedes termination. It documents the unavoidable
 # observe/terminate race; any raced dispatch is excluded and never retried.
 known=process_inventory(target)
 intent=save(OUT/'cancellation_intent.json',dict(time=now(),owners=target,observed_processes=known,
  boundary=gap,wait_elapsed_seconds=time.monotonic()-start,observations=observations,
  method='terminate exact current stage worker; preserve parent Windows-job cleanup',
  reason='replacement_ready_score_blind',bounded_fallback=not gap['observed'],
  active_move_clock_reset=False,interrupted_tail_policy='Preserve as administrative; never synthesize a loss or retry'))
 killed=terminate_exact(target['stage'])
 cleanup_start=time.monotonic()
 while any(exact_alive(i) for i in known) and time.monotonic()-cleanup_start<CLEANUP_SECONDS:
  known_map={(i['pid'],i['created']):i for i in known}
  for item in process_inventory(target):known_map[(item['pid'],item['created'])]=item
  known=list(known_map.values());time.sleep(.1)
 # No broad fallback killing: an unexplained remaining identity is an explicit
 # blocker for replacement launch, never silently treated as cleaned up.
 absent=all(not exact_alive(i) for i in known)
 scheduled=scheduled_inventory();inventory=save(OUT/'preservation_metadata.json',dict(files=metadata_inventory() if absent else [],
  complete=absent,deferred_while_original_writer_alive=not absent,
  mode='read-only file inventory; prior immutable evidence hashes retained in original receipts',time=now()))
 ending=save(OUT/'cancellation.json',dict(status='administratively_cancelled' if absent and killed else 'administrative_cleanup_incomplete',
  intent=intent,claim=claim,termination_sent=killed,exact_owned_absence=absent,observed_processes=known,
  cleanup_elapsed_seconds=time.monotonic()-cleanup_start,scheduled=scheduled,preservation=inventory,
  original_frozen_attempt_complete=False,original_completion_not_fabricated=True,
  all_later_games_preserved_nonconfirmatory=True,interrupted_tail_is_not_a_forfeit=True,
  retried_decisions=0,models=0,new_games=0,time=now(),
  independent_saved_evidence_acceptance_still_required=True))
 require(absent and killed,'Administrative cancellation/cleanup incomplete; do not launch replacement')
 return ending

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','execute'))
 parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--freeze-sha256')
 parser.add_argument('--readiness');parser.add_argument('--readiness-sha256');args=parser.parse_args()
 item=freeze(args.tests,args.reviews) if args.mode=='freeze' else execute(args.freeze_sha256,args.readiness,args.readiness_sha256)
 print(canonical(item),flush=True)

if __name__=='__main__':main()
