"""Actual detached ownership fixture; no tensors, learned models or game moves."""
import argparse, os, subprocess, sys, time
from pathlib import Path
from . import common as c

def run(folder,plan,owner,boundary,request):
 from tools.future_training_rules_20261021 import verify_runtime_sources
 source=verify_runtime_sources();c.require(source==plan['corrected_source_receipt'],'Fixture startup source guard')
 boundary()
 c.require('torch' not in sys.modules,'Fixture must be model-free')
 c.save(folder/'fixture_proof.json',dict(identity=c.identity(),sources=source,models=0,games=0))
 if request.get('fail'):raise ValueError('Intentional preserved detached fixture failure')
 return dict(status='passed',models=0,games=0)

def fixture_owner(folder,fail):
 from .owner import Jobs
 from .topology import read_topology,choose_placement
 from tools.future_training_rules_20261021 import verify_runtime_sources
 folder=Path(folder);top=read_topology()
 plan=dict(worker_inputs=[c.artifact(__file__),c.artifact(c.ROOT/(c.PACKAGE.replace('.','/')+'/owner.py')),
  c.artifact(c.ROOT/(c.PACKAGE.replace('.','/')+'/runtime.py'))],corrected_source_receipt=verify_runtime_sources(),
  placement=choose_placement(top),limits=dict(process_rss=c.PROCESS_RSS,aggregate_rss=c.AGGREGATE_RSS,
  disk_floor=c.MIN_DISK,evidence_bytes=c.MAX_OUTPUT))
 c.save(folder/'plan.json',plan)
 c.save(folder/'claim.json',dict(identity=c.identity(),monotonic_origin=time.monotonic(),unix_origin=time.time()))
 jobs=Jobs(plan,folder/'plan.json',folder/'claim.json',folder/'stop.json');caught=False
 try:jobs.launch_job(c.PACKAGE+'.fixture',folder/'job/worker',dict(mode='fixture',fail=fail),model=False)
 except ValueError:caught=True
 c.require(caught==fail,'Expected actual owned success/failure')
 executed=c.read(folder/'job/execution.json');worker=c.read(folder/'job/worker/result.json')
 if fail:
  c.require(executed['exit_code']==1 and worker['status']=='failed' and
   'Intentional preserved detached fixture failure' in worker['error'],'The intended negative fixture actually ran')
 else:c.require(executed['exit_code']==0 and worker['status']=='passed','Actual positive fixture ran')
 c.require(not any(c.exact_alive(i) for i in jobs.observed.values()),'All exact fixture children absent')
 c.save(folder/'fixture_owner_complete.json',dict(status='passed',expected_failure_preserved=fail,
  identity=c.identity(),models=0,games=0,observed_identities=list(jobs.observed.values())))

def probe(label):
 c.require(label.isalnum(),'Safe exclusive fixture label');folder=c.PREP/('detached_'+label)
 folder.mkdir(parents=True,exist_ok=False);proof=[]
 for fail in (False,True):
  case=folder/('failure' if fail else 'success');case.mkdir()
  args=[sys.executable,'-B','-u','-m',c.PACKAGE+'.fixture','--owner',str(case)]
  if fail:args.append('--fail')
  with (case/'stdout.log').open('xb') as out,(case/'stderr.log').open('xb') as err:
   flags=getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)
   proc=subprocess.Popen(args,cwd=c.ROOT,stdout=out,stderr=err,creationflags=flags)
   import psutil
   ident=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time())
   try:exitcode=proc.wait(timeout=50)
   except BaseException:
    proc.kill();proc.wait();raise
  c.require(exitcode==0 and not c.exact_alive(ident),'Actual detached fixture owner exit and absence')
  result=c.read(case/'fixture_owner_complete.json');c.require(result['status']=='passed','Detached fixture completed')
  proof.append(dict(identity=ident,exit_code=exitcode,result=c.artifact(case/'fixture_owner_complete.json'),
   execution=c.artifact(case/'job/execution.json'),failure_preserved=fail))
 item=c.save(folder/'result.json',dict(status='passed',detached=True,models=0,games=0,cases=proof,
  inputs=[c.artifact(p) for p in sorted((c.ROOT/c.PACKAGE.replace('.','/')).glob('*.py'))]))
 return item

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--owner');parser.add_argument('--fail',action='store_true')
 parser.add_argument('--probe');args=parser.parse_args()
 if args.owner:fixture_owner(args.owner,args.fail)
 else:print(c.canonical(probe(args.probe)))

if __name__=='__main__':main()
