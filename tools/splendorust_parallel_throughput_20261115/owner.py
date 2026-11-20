"""Bounded detached owner and reusable exact-owned-job launcher.

Import and preparation never launch the pilot. A later explicit authorization
receipt plus complete collection acceptance/exit are mandatory at launch.
"""
import argparse, os, shutil, subprocess, sys, threading, time, traceback
from pathlib import Path
import psutil
from . import common as c
from tools.splendorust_collection4096_20261115.owner import child_environment, output_bytes, wait_for_absence
from tools.splendorust_gpu_probe_20261105.control import require_absent, finish_owned_process

def metric_arm(folder):
 parts=c.normalized(folder).relative_to(c.OUT).parts if c.normalized(folder).is_relative_to(c.OUT) else ()
 if len(parts)>1 and parts[0]=='live' and parts[1] in ('phase0','phase1','phase2','phase3'):
  return 'A1' if parts[1] in ('phase0','phase3') else 'A4'
 if len(parts)>2 and parts[:2]==('collection','cells'):
  return 'B-control' if parts[2].startswith('control_') else 'B-candidate'
 return 'qualification_or_shared_verification'

def owned_tree(ident):
 try:
  parent=psutil.Process(ident['pid'])
  c.require(parent.create_time()==ident['created'],'Exact original process before tree snapshot')
  return [parent]+parent.children(recursive=True)
 except psutil.NoSuchProcess:return []

def claim_from_release(release,plan_item):
 c.require(release['identity']==c.identity() and release['windows_job_assigned'] and release['plan']==plan_item,
  'Exact root ownership, release and plan')
 c.require(c.exact_alive(release['parent']),'Exact supervising launcher is alive')
 return dict(identity=c.identity(),monotonic_origin=release['monotonic_origin'],unix_origin=release['unix_origin'],plan=plan_item)

class Jobs:
 def __init__(self,plan,plan_path,claim_path,stop_path=None):
  self.plan,self.plan_path,self.claim_path=plan,Path(plan_path),Path(claim_path)
  self.claim=c.read(claim_path);self.stop_path=Path(stop_path) if stop_path else c.OUT/'stop.json'
  self.stop=threading.Event();self.lock=threading.RLock();self.observed={};self.running={}
  self.cpu={};self.peak=0;self.last_disk=-float('inf');self.receipts=[]
  self.job_peaks={};self.job_cpu={}
  self.arm_peaks={}
 def abort(self,reason):
  self.stop.set()
  with self.lock:
   if not self.stop_path.exists():c.save(self.stop_path,dict(time=c.now(),reason=reason,retries=0))
 def guard(self,model=True):
  limit=c.MODEL_SECONDS if model else c.TOTAL_SECONDS
  c.check_deadlines(dict(monotonic_origin=self.claim['monotonic_origin'],unix_origin=self.claim['unix_origin'],
   monotonic_deadline=self.claim['monotonic_origin']+limit,unix_deadline=self.claim['unix_origin']+limit))
  c.require(not self.stop.is_set(),'Fail-stop; peer failure preserved')
  with self.lock:
   own_rss=psutil.Process().memory_info().rss
   c.require(own_rss<=self.plan['limits']['process_rss'],'Actual root individual RSS')
   total=own_rss;arm_totals={}
   for label,ident in list(self.running.items()):
    if not c.exact_alive(ident):continue
    job_total=0;job_cpu=self.job_cpu.setdefault(label,{})
    for process in owned_tree(ident):
     try:
      key=(process.pid,process.create_time());item=dict(pid=key[0],created=key[1])
      self.observed[key]=item;rss=process.memory_info().rss
      c.require(rss<=self.plan['limits']['process_rss'],'Actual 2GiB per-process RSS bound')
      total+=rss;job_total+=rss;cpu=process.cpu_times();self.cpu[key]=max(self.cpu.get(key,0),cpu.user+cpu.system)
      job_cpu[key]=max(job_cpu.get(key,0),cpu.user+cpu.system)
     except psutil.NoSuchProcess:pass
    self.job_peaks[label]=max(self.job_peaks.get(label,0),job_total)
    arm=metric_arm(Path(label));arm_totals[arm]=arm_totals.get(arm,0)+job_total
   for arm,rss in arm_totals.items():self.arm_peaks[arm]=max(self.arm_peaks.get(arm,0),rss)
   self.peak=max(self.peak,total);c.require(total<=self.plan['limits']['aggregate_rss'],'Actual aggregate RSS bound')
   if time.monotonic()-self.last_disk>=10:
    self.last_disk=time.monotonic();c.require(shutil.disk_usage(self.plan_path.parent).free>=self.plan['limits']['disk_floor'],'Actual disk floor')
    c.require(output_bytes(self.plan_path.parent)<=self.plan['limits']['evidence_bytes'],'Actual new-evidence bound')
 def launch_job(self,module,folder,request,model=True):
  from tools.local_player_qualification_128.clock import _WindowsJob
  from .runtime import MODULES
  c.require(module in MODULES,'Fixed worker entry point')
  folder=c.normalized(folder)
  c.require(folder.is_relative_to(c.normalized(c.OUT)) or folder.is_relative_to(c.normalized(c.PREP)),'Fresh new-pilot owned worker')
  folder.mkdir(parents=True,exist_ok=False);container=folder.parent
  c.require(not (container/'owner.json').exists(),'Single-use owner container')
  args=[sys.executable,'-B','-u','-m',c.PACKAGE+'.runtime','--module',module,'--output',str(folder),
   '--plan',str(self.plan_path),'--plan-sha256',c.sha(self.plan_path)]
  begun,wall=time.monotonic(),time.time();proc=job=ident=None;error=None;cleanup=[];status='failed';local={}
  limit=c.MODEL_SECONDS if model else c.TOTAL_SECONDS
  try:
   self.guard(model)
   with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
    # A child must inherit the original mask before its startup map validation.
    with self.lock:
     controller=psutil.Process();previous=controller.cpu_affinity()
     try:
      controller.cpu_affinity(self.plan['placement']['original_allowed'])
      proc=subprocess.Popen(args,cwd=c.ROOT,env=child_environment(),stdout=out,stderr=err,
       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
     finally:controller.cpu_affinity(previous)
     ident=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
     self.running[str(folder)]=ident
     c.atomic_save(container/'owner.json',dict(identity=ident,module=module,request=request,
      plan=c.artifact(self.plan_path),claim=c.artifact(self.claim_path),windows_job_assigned=True,
      model_work=model,monotonic_origin=self.claim['monotonic_origin'],unix_origin=self.claim['unix_origin'],
      monotonic_deadline=self.claim['monotonic_origin']+limit,unix_deadline=self.claim['unix_origin']+limit))
    while proc.poll() is None:
     self.guard(model)
     if module==c.PACKAGE+'.collection.worker':
      from .collection.worker import active_registry_paths
      from tools.splendorust_gpu_collector_20261105.control import check_registry
      from tools.splendorust_gpu_live_20261105_03.status_io import read_status_json
      for path in active_registry_paths(folder,request):
       if path.exists():check_registry(read_status_json(path),ident,time.time(),time.perf_counter())
     for p in owned_tree(ident):
      try:local[(p.pid,p.create_time())]=dict(pid=p.pid,created=p.create_time())
      except psutil.NoSuchProcess:pass
     # Original per-move clocks remain independently enforced in each route.
     time.sleep(.25)
    self.guard(model);c.require(proc.returncode==0,'Worker failed; no retry')
    result=c.read(folder/'result.json');c.require(result['status']=='passed' and result['identity']==ident,'Exact worker completion')
    wait_for_absence(list(local.values())+[ident])
    for item in list(local.values())+[ident]:require_absent(item)
    status='passed'
  except BaseException:
   error=traceback.format_exc();self.abort(error)
  finally:
   if job:
    try:job.close()
    except BaseException as exc:cleanup.append(repr(exc))
   if proc:
    try:finish_owned_process(proc,ident)
    except BaseException as exc:cleanup.append(repr(exc))
   wait_for_absence(list(local.values())+([ident] if ident else []))
   for item in list(local.values())+([ident] if ident else []):
    try:require_absent(item)
    except BaseException as exc:cleanup.append(repr(exc))
   with self.lock:self.running.pop(str(folder),None)
  if cleanup:status='failed';self.abort('Exact cleanup failure')
  receipt=c.save(container/'execution.json',dict(status=status,module=module,identity=ident,
   exit_code=None if proc is None else proc.returncode,observed_owned_identities=list(local.values()),
   cleanup_errors=cleanup,error=error,request=request,monotonic_start=begun,monotonic_end=time.monotonic(),
   unix_start=wall,unix_end=time.time(),result=c.artifact(folder/'result.json') if (folder/'result.json').exists() else None,
   retries=0,model_work=model,peak_tree_rss_bytes=self.job_peaks.get(str(folder),0),
   cpu_seconds=sum(self.job_cpu.get(str(folder),{}).values())))
  with self.lock:self.receipts.append(receipt)
  c.require(status=='passed','Failed owned worker: '+str(folder));return receipt

def launch_admission(plan,authorization):
 """No launch based only on this brief or a completion marker."""
 from .preparation import design
 c.require(plan['design']==design() and all(plan.get(k)==v for k,v in c.IDENTITY.items()),'Unchanged engineering design and corrected rules metadata')
 c.require(authorization.get('explicit_user_launch') is True and authorization.get('machine_quiet_confirmed') is True,
  'Later explicit user launch and quiet-machine confirmation required')
 c.require(authorization.get('plan')==c.artifact(c.PLAN),'Authorization binds exact prepared plan')
 c.require(c.sha(c.ACTIVE_OUT/'freeze.json')==c.ACTIVE_FREEZE_SHA,'Current collection freeze unchanged')
 # The collection acceptance entrypoint records independent final checks; the
 # exact receipt names are bound by preparation, never guessed from a marker.
 for expected in plan['admission']['required_receipts']:
  path=c.ROOT/expected['path'];c.require(path.exists(),'Collection still incomplete: '+expected['path'])
  value=c.read(path)
  for key,want in expected['equals'].items():c.require(value.get(key)==want,'Collection acceptance/exit: '+key)
 for relative in ('claim.json','launcher_claim.json'):
  path=c.ACTIVE_OUT/relative
  if path.exists():
   value=c.read(path);ident=value.get('identity',value)
   if 'pid' in ident and 'created' in ident:require_absent(ident)
 for path in c.ACTIVE_OUT.rglob('owner.json'):
  value=c.read(path)
  if 'identity' in value:require_absent(value['identity'])
 observed=c.read(c.ACTIVE_OUT/'observed_owner_exit.json')
 c.require(observed['freeze']==c.artifact(c.ACTIVE_OUT/'freeze.json'),'Accepted collection exact freeze')
 c.verify([observed[k] for k in ('root_execution','completion','independent_acceptance')])
 independent=c.read(c.resolve(observed['independent_acceptance']))
 c.require(independent['status']=='passed' and independent['error'] is None,'Independent collection acceptance')
 root=c.read(c.resolve(observed['root_execution']))
 c.require(root['status']=='passed' and root['worker_exit_code']==0 and not root['cleanup_errors'],'Actual collection root exit')
 for ident in [root['identity'],*root['observed_owned_identities']]:require_absent(ident)
 external=c.read(c.ROOT/'outputs/splendorust_collection4096_launch_20261115_01/observed_launcher_exit.json')
 require_absent(external['launcher_identity'])
 from .schedule import registry_inventory,seeds_in
 targets={int(s['seed_decimal']) for s in plan['live_schedule']+plan['collection']['games']['specifications']+plan['collection']['engineering_games']}
 registry_inventory(targets)
 c.verify(plan['inputs'])
 from .topology import verify_startup
 return verify_startup(plan['topology'],plan['placement'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--authorization',required=True);args=parser.parse_args()
 plan=c.read(c.PLAN);authorization=c.read(args.authorization)
 c.require(authorization.get('explicit_user_launch') is True and authorization.get('machine_quiet_confirmed') is True
  and authorization['plan']==c.artifact(c.PLAN),'Exact explicit future user launch')
 launch=c.OUT/'launch/root_release.json';started=time.monotonic()
 while not launch.exists():
  c.require(time.monotonic()-started<20,'Bounded external root ownership handshake');time.sleep(.01)
 release=c.read(launch)
 claim=claim_from_release(release,c.artifact(c.PLAN))
 launcher=c.read(c.OUT/'launcher_claim.json')
 c.require(launcher['identity']==release['parent'] and launcher['plan']==c.artifact(c.PLAN)
  and launcher['authorization']==c.artifact(args.authorization),'Successful original launcher admission')
 psutil.Process().cpu_affinity(plan['placement']['controller_cpus'])
 c.require(not (c.OUT/'claim.json').exists(),'No duplicate pilot or reset clock')
 c.save(c.OUT/'claim.json',claim)
 jobs=Jobs(plan,c.PLAN,c.OUT/'claim.json');error=None
 try:
  from .execution import execute
  payload=execute(plan,jobs)
  c.save(c.OUT/'complete.json',dict(status='passed',payload=payload,identity=c.identity(),time=c.now()))
 except BaseException:
  error=traceback.format_exc();jobs.abort(error)
  c.save(c.OUT/'failure.json',dict(status='failed',error=error,identity=c.identity(),time=c.now()))
 finally:
  c.save(c.OUT/'owner_final.json',dict(error=error,observed_owned_identities=list(jobs.observed.values()),
   peak_aggregate_rss=jobs.peak,cpu_seconds=sum(jobs.cpu.values()),arm_peak_rss=jobs.arm_peaks,
   arm_cpu_seconds={arm:sum(sum(values.values()) for path,values in jobs.job_cpu.items() if metric_arm(Path(path))==arm)
    for arm in ('A1','A4','B-control','B-candidate','qualification_or_shared_verification')},
   metric_scope='Owned process trees sampled at250ms; CPU totals approximate and RSS peaks sampled',
   jobs=jobs.receipts,identity=c.identity()))
 if error:raise SystemExit(1)

if __name__=='__main__':main()
