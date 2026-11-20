"""Real-clock external supervisor for the two model-free acceptance processes."""
import argparse,os,subprocess,sys,time,traceback
import psutil
from . import audit as a


def output_bytes():
 total=0
 for directory,_,files in os.walk(a.OUT):
  for name in files:total+=(a.Path(directory)/name).stat().st_size
 return total


def guard(limits,origin):
 a.original.check_deadlines(limits)
 a.require(time.monotonic()-origin<a.CAP_SECONDS,'Actual external per-pass audit clock')
 a.require(output_bytes()<a.MAX_OUTPUT,'Actual 512MiB separate audit evidence bound')


def run_job(mode,expected,limits):
 from tools.local_player_qualification_128.clock import _WindowsJob
 from tools.splendorust_sixworld512_20261114_v2.control import cleanup_owned
 folder=a.OUT/'jobs'/mode;folder.mkdir(parents=True,exist_ok=False)
 started=time.monotonic();identity=None;proc=job=None;observed={};error=None;cleanup=[];peak=0
 args=[sys.executable,'-B','-u','-m',a.PACKAGE+'.audit',mode,'--freeze-sha256',expected]
 try:
  guard(limits,started)
  with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
   env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
   proc=subprocess.Popen(args,cwd=a.ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
   child=psutil.Process(proc.pid);identity=dict(pid=proc.pid,created=child.create_time());job=_WindowsJob(proc.pid)
   a.save(folder/'owner.json',dict(identity=identity,parent=a.stop.identity(),args=args,mode=mode,
    freeze_sha256=expected,windows_job_assigned=True,per_worker_seconds=a.CAP_SECONDS,original_limits=limits))
   last_check=-float('inf')
   while proc.poll() is None:
    a.original.check_deadlines(limits);a.require(time.monotonic()-started<a.CAP_SECONDS,'Actual one-hour child clock')
    if time.monotonic()-last_check>=1:
     last_check=time.monotonic();guard(limits,started);used=0
     try:
      parent=psutil.Process(identity['pid']);a.require(parent.create_time()==identity['created'],'Exact audit owner')
      for current in [parent,*parent.children(recursive=True)]:
       try:
        item=dict(pid=current.pid,created=current.create_time());observed[(item['pid'],item['created'])]=item
        used+=current.memory_info().rss
       except psutil.NoSuchProcess:pass
     except psutil.NoSuchProcess:pass
     peak=max(peak,used);a.require(used<=8*1024**3,'Actual audit process-tree RSS bound')
    time.sleep(.1)
   proc.wait(timeout=5);guard(limits,started)
   a.require(proc.returncode==0,'Administrative saved audit failed; preserve without retry')
   result=a.read(a.OUT/mode/'result.json')
   a.require(result['status']=='passed' and result['identity']==identity and result['freeze_sha256']==expected,
    'Successful exact saved-audit process')
 except BaseException:error=traceback.format_exc()
 finally:cleanup_owned(proc,job,identity,observed,cleanup)
 absent=identity is not None and all(not a.stop.exact_alive(i) for i in [identity,*observed.values()])
 item=a.save(folder/'actual_exit.json',dict(status='passed' if error is None and not cleanup and absent else 'failed',
  identity=identity,exit_code=None if proc is None else proc.returncode,actual_exit_observed=proc is not None and proc.returncode is not None,
  exact_owned_absence=absent,observed_owned_identities=list(observed.values()),peak_rss_bytes=peak,
  error=error,cleanup_errors=cleanup,freeze_sha256=expected,mode=mode,elapsed_seconds=time.monotonic()-started,
  models=0,new_games=0,time=a.stop.now()))
 a.require(error is None and not cleanup and absent,'Failed independent audit retained; no retry')
 return item


def run(expected):
 plan,receipt,cancel_item,boundary=a.admit(expected)
 a.save(a.OUT/'launch_claim.json',dict(identity=a.stop.identity(),freeze=a.artifact(a.FREEZE),
  cancellation=cancel_item,original_limits=plan['original_limits'],time=a.stop.now(),models=0,new_games=0))
 exits=[];error=None
 try:
  for mode in ('analysis','acceptance'):exits.append(run_job(mode,expected,plan['original_limits']))
  a.save(a.OUT/'completion.json',dict(status='separate_administrative_verification_passed',actual_worker_exits=exits,
   acceptance=a.artifact(a.OUT/'acceptance/independent_acceptance.json'),original_attempt_complete=False,
   launcher_identity=a.stop.identity(),launcher_exit_still_requires_observation=True,models=0,new_games=0))
 except BaseException:error=traceback.format_exc()
 if error:a.save(a.OUT/'failure.json',dict(error=error,actual_worker_exits=exits,time=a.stop.now()));return 1
 return 0


if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);args=parser.parse_args()
 raise SystemExit(run(args.freeze_sha256))
