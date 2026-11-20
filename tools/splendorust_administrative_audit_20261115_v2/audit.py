"""Reuse passed semantic/replay audits, rebind bytes and check scheduled arithmetic.

No score payload is opened until a qualified administrative cancellation has
selected the latest contiguous verified scheduled look and all owners are gone.
The original attempt remains incomplete. This is a separately reported saved-
evidence acceptance, never a fabricated successful frozen-study completion.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback
from tools.splendorust_administrative_stop_20261115 import cancel as stop
from tools.splendorust_sixworld512_20261114_v2 import common as original
from .preservation import read_inventory

ROOT=stop.ROOT
PACKAGE='tools.splendorust_administrative_audit_20261115_v2'
OUT=ROOT/'outputs/splendorust_administrative_audit_20261115_02'
FREEZE=OUT/'freeze.json'
CAP_SECONDS=3600
MAX_OUTPUT=512*1024**2
FAILED_OUT=ROOT/'outputs/splendorust_administrative_audit_20261115_01'
FAILED_FREEZE_SHA='c9dba5368f3b9075bc39a2606afb3abe7fff637ea301bf0a536d1a1d1b06bae6'
require,artifact,resolve,sha=stop.require,stop.artifact,stop.resolve,stop.sha
read=original.read


def sources():
 return [artifact(p) for p in sorted(Path(__file__).parent.glob('*.py'))]


def save(path,value):
 path=Path(path).resolve();require(path.is_relative_to(OUT.resolve()),'Only separate audit output')
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf8',newline='\n') as stream:
  stream.write(stop.canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
 return artifact(path)


def binding_contract():
 return dict(schema='administrative_saved_audit_contract_20261115_v1',models=0,new_games=0,
  per_worker_seconds=CAP_SECONDS,maximum_workers=2,maximum_output_bytes=MAX_OUTPUT,
  maximum_worker_tree_rss_bytes=8*1024**3,external_real_clock_supervisor=True,
  original_attempt_complete=False,no_training=True,no_retries=True,
  selection='latest contiguous verified scheduled look fixed by score-blind cancellation',
  semantic_verification='reuse successful original independent full-game/service/replay receipts; verify their complete saved dependencies',
  arithmetic='independent per-game points, paired aggregation and original sequential checker',
  tail='all later completed and partial games preserved, excluded from confirmatory inference',
  teacher_default='original EF8A three worlds 512 simulations irrespective of subsequent score reveal')


def prior_preservation():
 """Bind the original launch refusal and unchanged frozen implementation."""
 require(sha(FAILED_OUT/'freeze.json')==FAILED_FREEZE_SHA,'Exact preserved refused finalizer freeze')
 prior=read(FAILED_OUT/'freeze.json');stop.verify(prior['inputs'])
 refusal=stop.read(FAILED_OUT/'launch_refusal.json')
 require(refusal['status']=='failed_preserved_before_worker_launch' and refusal['actual_command_exit_code']==1
  and refusal['error']=='ADMINISTRATIVE_STOP: Bounded control JSON'
  and refusal['launch_claim_exists'] is False and refusal['score_payloads_opened'] is False
  and refusal['models']==refusal['new_games']==0,'Exact preserved pre-worker inventory-reader failure')
 require(not any((FAILED_OUT/name).exists() for name in ('launch_claim.json','jobs','analysis','acceptance','completion.json')),
  'Refused attempt launched no audit workers')
 return dict(original_freeze=artifact(FAILED_OUT/'freeze.json'),refusal=artifact(FAILED_OUT/'launch_refusal.json'),
  files=[artifact(p) for p in sorted(FAILED_OUT.rglob('*')) if p.is_file()],source_inputs=prior['inputs'],
  original_attempt_complete=False,original_worker_launches=0,models=0,new_games=0,
  correction='Dedicated bounded preservation inventory reader; unchanged small control reader and scientific audit')


def freeze(test_path,review_paths):
 require(not FREEZE.exists(),'Separate exclusive audit freeze')
 proof=read(test_path);require(proof['status']=='passed' and proof['inputs']==sources()
  and proof['tests_passed']==proof['tests_collected']>=12 and proof['models']==proof['new_games']==0,
  'Source-bound model-free tests')
 stop.verify(proof['inputs']+[proof['log']])
 reviewed=set()
 for path in review_paths:
  value=read(path);require(value['approved'] and not value['blocking_findings'],'Independent finalizer review')
  stop.verify(value['inputs']);reviewed.update(x['path'] for x in value['inputs'])
 require(artifact(__file__)['path'] in reviewed,'Exact audit implementation reviewed independently')
 require(artifact(Path(__file__).parent/'launcher.py')['path'] in reviewed,'External watchdog reviewed independently')
 require(artifact(Path(__file__).parent/'preservation.py')['path'] in reviewed,'Dedicated large-inventory reader independently reviewed')
 require(sha(original.PLAN)==stop.ACTIVE_SHA,'Exact preserved original freeze')
 previous=prior_preservation()
 return save(FREEZE,dict(contract=binding_contract(),inputs=sources(),tests=artifact(test_path),
  reviews=[artifact(p) for p in review_paths],original_freeze=artifact(original.PLAN),
  cancellation_freeze=artifact(stop.FREEZE),previous_attempt=previous,
  correction_schema='administrative_saved_inventory_reader_correction_20261115_v2',
  time=stop.now(),scores_opened=False))


class Boundary:
 def __init__(self,limits):
  self.mono,self.wall=time.monotonic(),time.time();self.limits=limits;self.calls=0
 def __call__(self):
  self.calls+=1
  require(0<=time.monotonic()-self.mono<CAP_SECONDS and 0<=time.time()-self.wall<CAP_SECONDS,
   'One-hour administrative audit worker clock')
  original.check_deadlines(self.limits)


def admit(expected):
 require(sha(FREEZE)==expected,'Exact administrative audit freeze')
 freeze_value=read(FREEZE);require(freeze_value['contract']==binding_contract(),'Unchanged bounded audit contract')
 stop.verify(freeze_value['inputs']+[freeze_value['original_freeze'],freeze_value['cancellation_freeze']])
 require(freeze_value['previous_attempt']==prior_preservation(),'Refused v1 evidence and sources preserved exactly')
 stop.verify(stop.read(resolve(freeze_value['cancellation_freeze']))['source_inputs'])
 receipt_item=artifact(stop.OUT/'cancellation.json');receipt=stop.read(resolve(receipt_item))
 require(receipt['status']=='administratively_cancelled' and receipt['exact_owned_absence'] is True
  and receipt['original_frozen_attempt_complete'] is False and receipt['retried_decisions']==0,
  'Only completed disclosed administrative cancellation')
 require(all(not stop.exact_alive(i) for i in receipt['observed_processes']),'All original exact owners absent before score reveal')
 require(receipt['scheduled']==stop.scheduled_inventory(),'Latest verified prefix remains exactly the score-blind selection')
 stop.verify([receipt['intent'],receipt['claim'],receipt['preservation']])
 preserve=read_inventory(receipt['preservation'])
 require(preserve['complete'] is True and preserve['files']==stop.metadata_inventory(),
  'All original evidence closed and unchanged since cancellation')
 plan=read(original.PLAN)
 boundary=Boundary(plan['original_limits']);boundary()
 return plan,receipt,receipt_item,boundary


class Custody:
 """Hash every bound dependency once; parse only control evidence, not packets."""
 def __init__(self,boundary):self.boundary=boundary;self.items={};self.parsed=set()
 def verify(self,item,descend=True):
  self.boundary();path=resolve(item)
  require(set(item)=={'path','bytes','sha256'},'Exact evidence descriptor')
  prior=self.items.get(item['path'])
  if prior is not None:require(prior==item,'Consistent repeated evidence binding')
  else:
   before=path.stat();require(artifact(path)==item,'Preserved evidence hash: '+item['path']);after=path.stat()
   require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable closed evidence hash')
   self.items[item['path']]=item
  if not descend or item['path'] in self.parsed:return
  if not path.is_relative_to(original.OUT/'jobs'):return
  relative=path.relative_to(original.OUT).as_posix()
  if '/transport/request-' in relative or '/inputs/request-' in relative:return
  if not (path.suffix=='.json' or path.name=='saved_audit.json.gz'):return
  self.parsed.add(item['path']);value=read(path)
  for child in original.nested_artifacts(value):self.verify(child)
 def read(self,item):
  self.verify(item);return read(resolve(item))


def validated_game_points(game,spec,custody):
 require(game['spec']==spec,'Exact frozen game order')
 value=custody.read(game['game'])
 require(value['complete'] is True and value['spec']==spec and value['reason'] in ('natural','forfeit'),
  'Accepted complete saved game')
 winner=value['winner'];require(winner is None or type(winner) is int and winner in (0,1),'Strict official winner')
 points=1 if winner is None else 2 if winner==spec['candidate_seat'] else 0
 require(type(game['candidate_points2']) is int and game['candidate_points2']==value['candidate_points2']==points,
  'Independent official winner to candidate-point mapping')
 require(game['failed_decisions']==int(value['reason']=='forfeit'),'Only recorded deadline forfeits enter score')
 if value['reason']=='natural':
  require(value['failure'] is None and value['final']['state']['terminal'] is True
   and value['final']['state']['winner']==winner,'Natural game agrees with preserved terminal state')
 else:require(value['failure'] is not None and winner==1-value['failure']['seat'],'Recorded deadline loser preserved')
 return points


def pair_points(games,specs,custody):
 require(len(games)==len(specs) and len(games)%2==0,'Complete planned paired sample')
 scores=[]
 for index in range(0,len(games),2):
  a,b=specs[index:index+2]
  require(a['pair_index']==b['pair_index'] and a['seed_decimal']==b['seed_decimal']
   and a['candidate_seat']==1-b['candidate_seat'],'Same-setup swapped-seat pair')
  scores.append((validated_game_points(games[index],a,custody)+validated_game_points(games[index+1],b,custody))/4)
 return scores


def validate_reuse(result,audit,role,custody):
 replay=custody.read(result['replay_proof']);selected=audit['replay_selection']
 expected=6 if role=='operational' else 8
 require(replay['status']=='passed' and replay['repetitions']==expected and len(replay['checks'])==len(selected)==expected
  and replay['scientific_fields_omitted'] is False,'Passed complete original exact-replay receipt')
 require(len(result['replayed'])==expected,'Complete original replay result list')
 for wanted,performed,checked in zip(selected,result['replayed'],replay['checks'],strict=True):
  require(wanted['replay_id']==performed['replay_id']==checked['replay_id'] and checked['exact'] is True
   and checked['original']==wanted['source'] and checked['reproduced']==performed['source']
   and checked['original']!=checked['reproduced'],'Exact distinct original and reproduction receipt binding')
 services=result['services']+result['replay_services'];proof=custody.read(result['service_audit'])
 require([row['actual_exit'] for row in proof]==services and services,'Every completed service originally audited')
 for row in proof:
  require(row['status']=='passed' and row['forensics']['status']=='passed'
   and row['forensics']['lossless'] is True,'Passed full original wire/forensic verification reused')
  life=custody.read(row['actual_exit'])
  require(life['error'] is None and life['exact_owned_absence'] is True
   and all(not stop.exact_alive(i) for i in [life['identity'],*life['owned_identities']]),'Exact recorded service owners remain absent')
  require(row['requests']==len(life['requests']),'Complete original service request inventory')
 return expected


def rebuild(plan,cancellation,boundary):
 from tools.splendorust_sixworld512_20261114_v2 import worker,sequential,statistical_qualification
 custody=Custody(boundary)
 for item in plan['worker_inputs']+plan['data_inputs']:custody.verify(item,descend=False)
 calibration=custody.read(plan['calibration']);statistical_qualification.verify_qualification(calibration)
 q,qexit=worker.passed_job('qualification',stop.ACTIVE_SHA);custody.verify(qexit)
 qproof=custody.read(q['qualification'])
 require(qproof['verification']['status']=='passed','Reuse passed full original actual-model qualification')
 services=custody.read(q['service_audit'])
 require([s['actual_exit'] for s in services]==q['services']
  and all(s['status']=='passed' and s['forensics']['status']=='passed' for s in services),
  'Passed complete original qualification service receipts')
 selected=cancellation['scheduled']['looks'];count=cancellation['scheduled']['latest_games']
 require([r['games'] for r in selected]==list(range(400,count+1,400)),'Latest contiguous scheduled prefix')
 roles=['operational']+list(original.BLOCKS[:len(selected)])
 pairs=[];games=[];timings={'3':[],'6':[]};receipts=[qexit];replays=0;interims=[];look=None
 for role in roles:
  boundary();result,exit_item=worker.passed_job(role,stop.ACTIVE_SHA);custody.verify(exit_item);receipts.append(exit_item)
  block=custody.read(result['block']);audit=custody.read(block['verified_game_audit'])
  specs=worker.stage_specs(plan,role)
  require(audit['status']=='passed' and audit['official_games']==len(specs)
   and block['status']=='passed' and block['pair_scores']==audit['pair_scores']
   and block['replay_selection']==audit['replay_selection'],'Reuse passed full independent saved-game audit')
  recomputed=pair_points(audit['games'],specs,custody)
  require(recomputed==audit['pair_scores'],'Independent paired arithmetic matches saved semantic audit')
  require(audit['failed_decisions']==sum(g['failed_decisions'] for g in audit['games']),
   'Preserved original forfeit counts')
  replays+=validate_reuse(result,audit,role,custody)
  if role=='operational':
   require(len(specs)==6 and audit['failed_decisions']==0,'Six clean excluded operational games');continue
  require(look is None or look['stop'] is False,'No confirmatory block after statistical stop')
  pairs+=recomputed;games+=audit['games']
  for pool in timings:timings[pool]+=audit['timings'][pool]
  chosen=selected[len(interims)];require(chosen['games']==2*len(pairs),'Prospectively selected cumulative look')
  require(chosen['actual_exit']==exit_item,'Selected prefix exact stage acceptance')
  custody.verify(chosen['look']);look=read(resolve(chosen['look']))
  sequential.verify_analysis(pairs,calibration,look);interims.append(chosen['look'])
 require(len(games)==count,'Entire selected prefix and no tail included')
 failures=[custody.read(g['game'])['failure'] for g in games if g['failed_decisions']]
 counts={str(seat):dict(games=0,points2=0) for seat in (0,1)}
 for game in games:
  row=counts[str(game['spec']['candidate_seat'])];row['games']+=1;row['points2']+=game['candidate_points2']
 return dict(schema='administrative_saved_prefix_acceptance_20261115_v1',status='passed',
  original_frozen_attempt_complete=False,original_attempt_status='administratively_cancelled_incomplete',
  separate_saved_verification=True,game_semantics_recomputed=False,original_semantic_and_replay_receipts_reused=True,
  preserved_dependencies_verified=True,independent_pair_and_sequential_arithmetic=True,
  games=len(games),pairs=len(pairs),pair_scores=pairs,candidate_score=sum(pairs)/len(pairs) if pairs else None,
  terminal_look=look,selected_scheduled_looks=interims,interims=interims,seat_counts=counts,
  timings=timings,failed_decisions=failures,exact_reproductions=replays,operational_games=6,
  original_actual_stage_exits=receipts,bound_evidence=sorted(custody.items.values(),key=lambda item:item['path']),
  all_tail_nonconfirmatory=True,no_tail_games_discarded=True,retried_decisions=0,models=0,new_games=0,
  default_teacher_worlds=3,teacher_choice_independent_of_revealed_result=True,automatic_promotion=False,
  inference_limit='Original simulation-qualified approximate sequential intervals; no new universal optional-stopping claim',
  disclosure='Administrative stopping policy recorded after look400 existed; payloads were not opened by this preparation before cancellation')


def inventory(cancellation,folder,boundary):
 expected=read_inventory(cancellation['preservation'])['files'];actual=stop.metadata_inventory()
 require(actual==expected,'All original closed evidence unchanged before full preservation hashing')
 path=folder/'preserved_evidence.jsonl';count=total=0
 with path.open('xb') as stream:
  for row in actual:
   boundary();source=ROOT/row['path'];before=source.stat();item=artifact(source);after=source.stat()
   require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Stable original evidence during preservation')
   stream.write((stop.canonical(item)+'\n').encode());count+=1;total+=item['bytes']
  stream.flush();os.fsync(stream.fileno())
 require(stop.metadata_inventory()==actual,'Complete original evidence unchanged after full preservation hashing')
 return dict(inventory=artifact(path),files=count,bytes=total,all_later_complete_and_partial_games_preserved=True)


def run(expected,mode):
 require(mode in ('analysis','acceptance'),'Fixed model-free finalization modes')
 owner_path=OUT/'jobs'/mode/'owner.json';started=time.monotonic()
 while not owner_path.exists():
  require(time.monotonic()-started<10,'Bounded assigned-audit ownership handshake');time.sleep(.01)
 owner=read(owner_path)
 require(owner['identity']==stop.identity() and owner['freeze_sha256']==expected and owner['mode']==mode
  and owner['windows_job_assigned'] is True and stop.exact_alive(owner['parent']),
  'Exact external audit supervisor identity and freeze')
 plan,cancellation,cancellation_item,boundary=admit(expected)
 folder=OUT/mode;folder.mkdir(parents=True,exist_ok=False)
 save(folder/'owner.json',dict(identity=stop.identity(),time=stop.now(),freeze=artifact(FREEZE),
  cancellation=cancellation_item,per_worker_seconds=CAP_SECONDS,original_limits=plan['original_limits'],models=0,new_games=0))
 error=None;report=None
 try:
  from tools.future_training_rules_20261021 import verify_runtime_sources
  require(verify_runtime_sources()==plan['corrected_source_receipt'],'Exact corrected source guard before saved evaluation')
  report=rebuild(plan,cancellation,boundary)
  if mode=='analysis':
   preserve=inventory(cancellation,folder,boundary)
   report.update(cancellation=cancellation_item,preservation=preserve)
   save(folder/'report.json',report)
  else:
   prior=read(OUT/'analysis/result.json');require(prior['status']=='passed','Successful separate first audit required')
   require(not stop.exact_alive(prior['identity']),'First audit owner actually exited before independent acceptance')
   baseline=read(OUT/'analysis/report.json')
   expected_body={k:v for k,v in baseline.items() if k not in ('cancellation','preservation')}
   require(report==expected_body,'Independent second saved-evidence arithmetic and custody exactly match')
   require(baseline['cancellation']==cancellation_item,'Exact selected cancellation in both audits')
   preserve=baseline['preservation'];stop.verify([preserve['inventory']])
   seen=set();total=0
   with resolve(preserve['inventory']).open('rb') as stream:
    for line in stream:
     boundary();item=json.loads(line);require(item['path'] not in seen,'Unique full preservation inventory')
     stop.verify([item]);seen.add(item['path']);total+=item['bytes']
   require(len(seen)==preserve['files'] and total==preserve['bytes']
    and seen=={r['path'] for r in stop.metadata_inventory()},'Exact full original preservation inventory')
   save(folder/'independent_acceptance.json',dict(status='accepted_separate_administrative_saved_verification',
    approved=True,blocking_findings=[],analysis=artifact(OUT/'analysis/report.json'),
    cancellation=cancellation_item,preservation=preserve,original_attempt_complete=False,
    latest_scheduled_look=report['interims'][-1] if report['interims'] else None,
    models=0,new_games=0,actual_acceptance_owner_exit_still_required=True))
  boundary();require('torch' not in sys.modules,'Saved evidence audit never loads learned model runtime')
 except BaseException:error=traceback.format_exc()
 save(folder/'result.json',dict(status='passed' if error is None else 'failed',error=error,mode=mode,
  identity=stop.identity(),freeze_sha256=expected,time=stop.now(),models=0,new_games=0,
  actual_owner_exit_still_required=True))
 if error:print(error,flush=True)
 return 0 if error is None else 1


def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','analysis','acceptance'))
 parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*');parser.add_argument('--freeze-sha256');args=parser.parse_args()
 if args.mode=='freeze':print(stop.canonical(freeze(args.tests,args.reviews)));return 0
 return run(args.freeze_sha256,args.mode)


if __name__=='__main__':raise SystemExit(main())
