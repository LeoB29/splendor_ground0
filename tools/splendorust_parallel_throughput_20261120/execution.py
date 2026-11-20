"""Fixed qualified live/collection sequence, invoked only by the released owner."""
from pathlib import Path
import time
from . import common as c

def summarize(values):
 import math,statistics
 c.require(bool(values),'Every timing arm has actual completed decisions')
 values=sorted(values)
 return dict(count=len(values),median=statistics.median(values),p95=values[max(0,math.ceil(.95*len(values))-1)],maximum=max(values))

def live_metrics(phases,identity_seconds):
 result={}
 for arm in ('A1','A4'):
  parts=[row['result'] for row in phases if row['result']['arm']==arm]
  timings=[t for part in parts for proof in part['audits'] for t in proof['timings']]
  duration=sum(part['elapsed_seconds'] for part in parts)+identity_seconds/2
  decisions=sum(proof['decisions'] for part in parts for proof in part['audits'])
  c.require(sum(len(part['games']) for part in parts)==16,'All sixteen live setups verified')
  result[arm]=dict(games=16,decisions=decisions,verified_seconds=duration,games_per_hour=16*3600/duration,
   decisions_per_hour=decisions*3600/duration,response_seconds=summarize(timings),technical_failures=0,
   accounting='Sum of both phase intervals including initialization, graphs, exact replays, actual exits and audits; half of shared identity checks')
 result['speedup']=result['A4']['games_per_hour']/result['A1']['games_per_hour']
 result['adopt']=result['speedup']>=2 and result['A4']['response_seconds']['maximum']<60
 return result

def execute(plan,jobs):
 from .live import scheduler,schedule,audit
 from .collection import launch as collection
 boundary=lambda:jobs.guard(False)
 placement=plan['placement'];root_owner=c.read(c.OUT/'claim.json')
 liveq=jobs.launch_job(c.PACKAGE+'.live.worker',c.OUT/'qualification/live/worker',dict(mode='qualification',kind='qualification',
  model_cpu=placement['model_cpus'][0],controller_cpus=placement['controller_cpus'],original_allowed=placement['original_allowed']))
 collectionq=collection.qualify(plan,jobs,boundary)
 c.save(c.OUT/'qualification_gate.json',dict(status='passed',live=liveq,collection=collectionq,plan=c.artifact(c.PLAN)))
 phases=[];paired={};identity_seconds=0.
 for phase in range(4):
  boundary();arm=schedule.phase_arm(phase)
  c.status(c.OUT/'stage.json',dict(stage='live',phase=phase,arm=arm,time=c.now()))
  item=scheduler.run_stage(c.OUT/'live'/f'phase{phase}',plan,root_owner,boundary,schedule.phase_specs(phase),arm,phase,jobs.launch_job)
  phases.append(item)
  if phase in (1,3):
   indices=list(range(8)) if phase==1 else list(range(16));byarm={a:[] for a in ('A1','A4')}
   for entry in phases:
    byarm[entry['result']['arm']].extend(row['game'] for row in entry['result']['games'])
   start=time.monotonic();proof=audit.compare_subset(byarm['A1'],byarm['A4'],indices,boundary)
   elapsed=time.monotonic()-start;identity_seconds+=elapsed
   paired[str(phase)]=c.save(c.OUT/'live'/f'identity_after_phase{phase}.json',dict(proof=proof,elapsed_seconds=elapsed))
 live=live_metrics(phases,identity_seconds)
 live_receipt=c.save(c.OUT/'live/result.json',dict(status='passed',metrics=live,phases=[x['artifact'] for x in phases],identity=paired))
 c.status(c.OUT/'stage.json',dict(stage='collection',time=c.now()))
 collected=collection.run(plan,jobs,boundary)
 collection_receipt=c.artifact(c.OUT/'collection/result.json')
 final=jobs.launch_job(c.PACKAGE+'.acceptance',c.OUT/'acceptance/worker',dict(mode='acceptance',
  live=live_receipt,collection=collection_receipt,executions=list(jobs.receipts)),model=False)
 return dict(status='verified_pending_actual_root_exit',live=live_receipt,collection=collection_receipt,
  independent_acceptance=final,no_strength_claim=True,no_automatic_followup=True)
