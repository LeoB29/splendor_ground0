"""Later explicit detached launch with actual root process-handle exit receipt."""
import argparse, subprocess, sys, time, traceback
import psutil
from . import common as c
from .owner import launch_admission,child_environment
from tools.splendorust_collection4096_20261115.owner import wait_for_absence
from tools.splendorust_gpu_probe_20261105.control import finish_owned_process,require_absent

def launch(authorization):
 from tools.local_player_qualification_128.clock import _WindowsJob
 plan=c.read(c.PLAN);launch_admission(plan,c.read(authorization))
 c.require(not (c.OUT/'launcher_claim.json').exists(),'Single-use explicit launch')
 folder=c.OUT/'launch';folder.mkdir(parents=True,exist_ok=False)
 started,wall=time.monotonic(),time.time();me=c.identity()
 c.save(c.OUT/'launcher_claim.json',dict(identity=me,plan=c.artifact(c.PLAN),authorization=c.artifact(authorization),
  monotonic_origin=started,unix_origin=wall))
 proc=job=ident=None;error=None;cleanup=[];observed={}
 try:
  args=[sys.executable,'-B','-u','-m',c.PACKAGE+'.owner','--authorization',str(authorization)]
  with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
   proc=subprocess.Popen(args,cwd=c.ROOT,env=child_environment(),stdin=subprocess.DEVNULL,stdout=out,stderr=err,
    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
   ident=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
   c.save(folder/'root_release.json',dict(identity=ident,parent=me,windows_job_assigned=True,
    monotonic_origin=started,unix_origin=wall,plan=c.artifact(c.PLAN)))
   while proc.poll() is None:
    c.check_deadlines(dict(monotonic_origin=started,unix_origin=wall,monotonic_deadline=started+c.TOTAL_SECONDS,unix_deadline=wall+c.TOTAL_SECONDS))
    try:
     for p in [psutil.Process(ident['pid'])]+psutil.Process(ident['pid']).children(recursive=True):
      try:observed[(p.pid,p.create_time())]=dict(pid=p.pid,created=p.create_time())
      except psutil.NoSuchProcess:pass
    except psutil.NoSuchProcess:pass
    time.sleep(.5)
   c.require(proc.returncode==0,'Actual pilot owner failure preserved')
   completion=c.read(c.OUT/'complete.json');result=completion['payload']
   c.require(completion['status']=='passed' and result['status']=='verified_pending_actual_root_exit','Completed result requires independent acceptance')
   c.verify([result['independent_acceptance']]);accept=c.read(c.resolve(result['independent_acceptance']))
   c.require(accept['status']=='passed' and accept['exit_code']==0 and not accept['cleanup_errors'],'Actual independent acceptance exit')
 except BaseException:error=traceback.format_exc()
 finally:
  if job:
   try:job.close()
   except BaseException as exc:cleanup.append(repr(exc))
  if proc:
   try:finish_owned_process(proc,ident)
   except BaseException as exc:cleanup.append(repr(exc))
  wait_for_absence(list(observed.values()))
  for item in list(observed.values())+([ident] if ident else []):
   try:require_absent(item)
   except BaseException as exc:cleanup.append(repr(exc))
 status='passed' if error is None and not cleanup else 'failed'
 c.save(c.OUT/'observed_owner_exit.json',dict(status=status,error=error,cleanup_errors=cleanup,root_identity=ident,
  actual_root_exit=None if proc is None else proc.returncode,observed_owned_identities=list(observed.values()),
  launcher_identity=me,launcher_exit_still_requires_observation=True,time=c.now()))
 if status=='passed':
  try:
   c.check_deadlines(dict(monotonic_origin=started,unix_origin=wall,monotonic_deadline=started+c.TOTAL_SECONDS,unix_deadline=wall+c.TOTAL_SECONDS))
   from .report import write
   write()
  except BaseException:
   c.save(c.OUT/'report_failure.json',dict(status='finalization_failed_after_verified_root',error=traceback.format_exc(),time=c.now(),
    original_verified_root_preserved=True,no_new_model_or_game=True));return 1
 return 0 if status=='passed' else 1

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--authorization',required=True);args=parser.parse_args()
 raise SystemExit(launch(args.authorization))
