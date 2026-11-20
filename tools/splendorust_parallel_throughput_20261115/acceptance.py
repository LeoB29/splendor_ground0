"""Independent saved-only acceptance; no new games, searches, fits or retries."""
from pathlib import Path
import time
from . import common as c

def run(folder,plan,owner,boundary,request):
 from tools.future_training_rules_20261021 import verify_runtime_sources
 sources=verify_runtime_sources();c.require(sources==plan['corrected_source_receipt'],'Final independent source guard')
 c.save(Path(folder)/'runtime_sources.json',dict(sources=sources,identity=c.identity()))
 c.verify(plan['inputs'],boundary);c.verify([request['live'],request['collection']],boundary)
 live=c.read(c.resolve(request['live']));collection=c.read(c.resolve(request['collection']))
 c.require(live['status']==collection['status']=='passed','Both full engineering routes passed')
 for item in request['executions']:
  boundary();c.verify([item]);execution=c.read(c.resolve(item))
  c.require(execution['status']=='passed' and execution['exit_code']==0 and execution['error'] is None
   and not execution['cleanup_errors'] and execution['retries']==0,'Each actual owned execution passed')
  c.verify([execution['result']]);value=c.read(c.resolve(execution['result']))
  c.require(value['status']=='passed' and value['identity']==execution['identity'],'Exact actual result identity')
  c.require(all(not c.exact_alive(i) for i in [execution['identity'],*execution['observed_owned_identities']]),'All earlier owned processes actually absent')
 from .live.audit import verify_lane,compare_arms
 from .execution import live_metrics
 byarm={'A1':[],'A4':[]};phases=[];charges={a:0. for a in ('A1','A4','B-control','B-candidate')}
 for item in live['phases']:
  boundary();c.verify([item]);phase=c.read(c.resolve(item))
  started=time.monotonic();checks=[verify_lane(lane,plan,boundary) for lane in phase['lanes']]
  c.require(checks==phase['audits'],'Recomputed exact live phase audits')
  byarm[phase['arm']].extend(row['game'] for row in phase['games']);phases.append(phase)
  charges[phase['arm']]+=time.monotonic()-started
 identity_seconds=0.
 for item in live['identity'].values():c.verify([item]);identity_seconds+=c.read(c.resolve(item))['elapsed_seconds']
 reconstructed_live=live_metrics([dict(result=p) for p in phases],identity_seconds)
 c.require(reconstructed_live==live['metrics'],'Independently reconstructed live timings, throughput and adoption')
 started=time.monotonic();identity=compare_arms(byarm['A1'],byarm['A4'],boundary)
 duration=time.monotonic()-started
 for arm in ('A1','A4'):charges[arm]+=duration/2
 c.require(identity['status']=='passed','Complete exact live identity')
 from .collection.identity import compare_cells
 from .collection import common as cc,audit as ca
 records={}
 for cell in ('X','Y'):
  boundary();left=cc.OUT/'cells'/('control_'+cell);right=cc.OUT/'cells'/('candidate_'+cell)
  started=time.monotonic()
  compare_cells(left,right,plan['collection']['cells'][cell]['specifications'],boundary=boundary)
  duration=time.monotonic()-started
  for arm in ('B-control','B-candidate'):charges[arm]+=duration/2
  for arm,count in [('control',1),('candidate',4)]:
   started=time.monotonic()
   cellfolder=cc.OUT/'cells'/f'{arm}_{cell}'
   proof=ca.audit_cell(plan['collection'],plan['collection']['cells'][cell],cellfolder,
    [str(cellfolder/'players'/f'p{i}'/'worker') for i in range(count)],str(cellfolder/'replays/worker'),boundary=boundary)
   c.require(proof['status']=='passed','Independent reconstructed complete collection audit')
   charges['B-'+arm]+=time.monotonic()-started
   records[(arm,cell)]=dict(folder=cellfolder,started=cc.read(cellfolder/'started.json'),checked=cc.read(cellfolder/'verified.json'))
 from .collection.launch import measure
 reconstructed_collection=measure(records)
 c.require(reconstructed_collection==collection['metrics'],'Independently reconstructed collection raw clocks, intervals and all check charges')
 c.require(live['metrics']['adopt']==(live['metrics']['speedup']>=2 and live['metrics']['A4']['response_seconds']['maximum']<60),'Unchanged live adoption bar')
 c.require(collection['qualifies']==(collection['metrics']['candidate']['verified_games_per_hour']/collection['metrics']['control']['verified_games_per_hour']>=1.3),'Unchanged collection adoption bar')
 return dict(status='passed',sources=sources,live=request['live'],collection=request['collection'],
  live_adopt=live['metrics']['adopt'],collection_adopt=collection['qualifies'],models_run=0,games_run=0,
  reconstructed_live=reconstructed_live,reconstructed_collection=reconstructed_collection,
  final_verification_seconds_by_arm=charges,independent_saved_reconstruction=True,actual_root_exit_still_required=True)
