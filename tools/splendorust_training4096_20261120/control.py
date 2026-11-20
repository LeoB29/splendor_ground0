"""Exclusive processes, durable actual exits, dual clocks and actual resource checks."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c

def environment():
    env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',RAYON_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return env

def output_bytes():
    used=0
    for path in c.OUT.rglob('*'):
        try:
            if path.is_file():used+=path.stat().st_size
        except FileNotFoundError:
            c.require(path.name in ('stage.tmp','progress.tmp','owner.json.pending','all_checkpoints_committed.json.pending'),
                'Only known atomic receipt/status rename may vanish')
    return used

def bounds(limits):
    c.check_deadlines(limits)
    c.require(shutil.disk_usage(c.OUT).free>=c.MIN_DISK,'Actual free disk floor')

def cleanup_owned(proc,job,ident,observed,errors):
    from tools.splendorust_collection4096_resume_20261118_v2.ownership import observe_job,finish_exact_process
    if job and job.handle is not None:
        try:observe_job(job,observed)
        except BaseException as exc:errors.append(repr(exc))
    if job:
        try:job.close()
        except BaseException as exc:errors.append(repr(exc))
    if proc:
        try:finish_exact_process(proc,ident)
        except BaseException as exc:errors.append(repr(exc))
    targets=list(observed.values())+([ident] if ident else [])
    until=time.monotonic()+5.
    while any(c.exact_alive(i) for i in targets) and time.monotonic()<until:time.sleep(.05)
    if any(c.exact_alive(i) for i in targets):errors.append('Exact native-owned process remains after bounded cleanup')


def launch(command,folder,limits,metadata,rss=c.AUDIT_RSS):
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_collection4096_resume_20261118_v2.ownership import observe_job,ownership_proofs,METHOD,verify_observations
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    proc=job=ident=None;observed={};errors=[];error=None;peak=0;last_scan=0
    started,wall_started=time.monotonic(),time.time()
    try:
        bounds(limits)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(command,cwd=c.ROOT,env=environment(),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            ident=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            # Record the launched process while it awaits its ownership handshake.
            observe_job(job,observed)
            c.atomic_save(folder/'owner.json',dict(identity=ident,parent=c.identity(),command=command,
                windows_job_assigned=True,limits=limits,rss_cap=rss,**metadata))
            while proc.poll() is None:
                bounds(limits);used=observe_job(job,observed);peak=max(peak,used)
                c.require(used<=rss,'Actual aggregate native-owned RSS bound')
                if time.monotonic()-last_scan>=30:
                    cap=c.MAX_OUTPUT if metadata.get('role') in ('root','analysis','acceptance') else c.MAX_OUTPUT-c.FINAL_RESERVE
                    c.require(output_bytes()<cap,'Actual new-evidence bound including final reserve');last_scan=time.monotonic()
                time.sleep(.5)
            proc.wait();bounds(limits)
    except BaseException:error=traceback.format_exc()
    finally:cleanup_owned(proc,job,ident,observed,errors)
    if error is None:
        try:
            bounds(limits)
            cap=c.MAX_OUTPUT if metadata.get('role') in ('root','analysis','acceptance') else c.MAX_OUTPUT-c.FINAL_RESERVE
            c.require(output_bytes()<cap,'Actual final evidence bound')
        except BaseException:error=traceback.format_exc()
    code=None if proc is None else proc.returncode
    absent=ident is not None and not c.exact_alive(ident) and all(not c.exact_alive(i) for i in observed.values())
    result_artifact=None
    if error is None and code==0 and metadata.get('role') in (*c.JOBS,'root'):
        try:
            result_path=c.OUT/'completion.json' if metadata['role']=='root' else folder/'worker/result.json'
            result_artifact=c.artifact(result_path)
        except BaseException:error=traceback.format_exc()
    passed=error is None and not errors and code==0 and absent
    receipt=dict(status='passed' if passed else 'failed',identity=ident,
        exit_code=code,actual_exit_observed=code is not None,exact_owned_absence=absent,
        observed_owned_identities=list(observed.values()),ownership_method=METHOD,
        job_membership_observations=ownership_proofs(job),peak_rss_bytes=peak,
        monotonic_origin=started,monotonic_end=time.monotonic(),unix_origin=wall_started,unix_end=time.time(),
        elapsed_seconds=time.monotonic()-started,limits=limits,
        cleanup_errors=errors,error=error,result_artifact=result_artifact,time=c.now(),**metadata)
    if passed:
        try:verify_observations(receipt)
        except BaseException:
            receipt['status']='failed';receipt['error']=traceback.format_exc();passed=False
    item=c.save(folder/'actual_exit.json',receipt)
    c.require(passed,'Child failed; actual exit and failure preserved: '+folder.name)
    return item


def passed_job(role,boundary=lambda:None):
    from tools.splendorust_collection4096_resume_20261118_v2.ownership import verify_observations
    c.require(role in c.JOBS,'Known fixed worker role')
    folder=c.OUT/'jobs'/role
    item=c.artifact(folder/'actual_exit.json');execution=c.read(c.resolve(item));result=c.read(folder/'worker/result.json')
    boundary();verify_observations(execution)
    owner=c.read(folder/'owner.json');validate_limits(owner,role)
    c.require(execution['limits']==owner['limits'] and execution['identity']==owner['identity']
        and owner['role']==role and owner['freeze_sha256']==c.sha(c.PLAN),'Actual exit binds original owned identity and clocks')
    c.require(abs(execution['elapsed_seconds']-(execution['monotonic_end']-execution['monotonic_origin']))<.01,
        'Actual elapsed arithmetic')
    for clock in ('monotonic','unix'):
        c.require(execution['limits'][clock+'_origin']<=execution[clock+'_origin']<=execution[clock+'_end']<execution['limits'][clock+'_deadline'],
            'Actual exit within original stage clocks')
    c.require(execution['result_artifact']==c.artifact(folder/'worker/result.json'),'Actual exit binds exact result')
    c.require(execution['status']=='passed' and type(execution['exit_code']) is int and execution['exit_code']==0
        and execution['actual_exit_observed'] and execution['exact_owned_absence'] and not execution['cleanup_errors']
        and execution['error'] is None and execution['role']==role and execution['freeze_sha256']==c.sha(c.PLAN),'Actual clean prior execution: '+role)
    c.require(result['status']=='passed' and result['error'] is None and result['role']==role
        and result['identity']==execution['identity'] and result['freeze_sha256']==c.sha(c.PLAN),'Passed exact prior result: '+role)
    c.require(not c.exact_alive(execution['identity']) and all(not c.exact_alive(i) for i in execution['observed_owned_identities']),'Prior actual owned absence')
    return item,result


def handshake(folder,expected,role):
    start=time.monotonic();path=Path(folder)/'owner.json'
    while not path.exists():
        c.require(time.monotonic()-start<30,'Bounded ownership handshake');time.sleep(.02)
    owner=c.read(path)
    c.require(owner['identity']==c.identity() and owner['windows_job_assigned'] is True and
        owner['freeze_sha256']==expected and owner['role']==role,'Exact frozen owned role')
    c.require(c.exact_alive(owner['parent']),'Exact live owner')
    validate_limits(owner,role)
    return owner

def validate_limits(owner,role):
    c.require(role in (*c.JOBS,'root'),'Bounded registered role')
    launcher=c.read(c.OUT/'launcher_claim.json');global_limits=launcher['limits']
    c.require(launcher['freeze']==c.artifact(c.PLAN),'Original launch clock binds current freeze')
    if role=='root':
        c.require(owner['parent']==launcher['identity'] and owner['limits']==global_limits,'Exact original launcher owner and clocks')
    else:
        root=c.read(c.OUT/'claim.json')
        c.require(root['limits']==global_limits and owner['parent']==root['identity'],'Exact original root owner and clocks')
    for clock in ('monotonic','unix'):
        origin,deadline=clock+'_origin',clock+'_deadline';limits=owner['limits']
        c.require(abs(global_limits[deadline]-global_limits[origin]-c.TOTAL_SECONDS)<.001,'Frozen overall clock span')
        c.require(global_limits[origin]<=limits[origin]<limits[deadline]<=global_limits[deadline],'No clock reset or extension')
        cap=c.TOTAL_SECONDS if role=='root' else c.CAPS[role]
        c.require(limits[deadline]-limits[origin]<=cap+.001,'Frozen role duration')
        if role in c.MODEL_JOBS:c.require(limits[deadline]<=global_limits[origin]+c.MODEL_SECONDS,'Original model cutoff')
    expected=c.AUDIT_RSS+512*1024**2 if role=='root' else c.MODEL_RSS if role in c.MODEL_JOBS else c.AUDIT_RSS
    c.require(owner['rss_cap']==expected,'Frozen owned RSS limit')

class Boundary:
    def __init__(self,folder,owner):self.folder=Path(folder);self.owner=owner;self.last=0;self.calls=0
    def __call__(self):
        self.calls+=1;c.check_deadlines(self.owner['limits'])
        if time.monotonic()-self.last>=5:
            bounds(self.owner['limits']);c.require(c.exact_alive(self.owner['parent']),'Exact owner remains alive')
            c.require(psutil.Process().memory_info().rss<=self.owner['rss_cap'],'Actual worker RSS bound')
            target=c.TRAIN_CPU if self.owner['role'] in c.MODEL_JOBS and (self.folder/'startup.json').exists() else c.SUPERVISOR_CPU
            c.require(psutil.Process().cpu_affinity()==[target],'Own worker stays on its declared physical-core role')
            c.status(self.folder/'boundary/progress.json',dict(time=c.now(),role=self.owner['role'],calls=self.calls))
            self.last=time.monotonic()

def clipped_limits(parent,seconds,model=False):
    mono,wall=time.monotonic(),time.time()
    return dict(monotonic_origin=mono,unix_origin=wall,
        monotonic_deadline=min(mono+seconds,parent['monotonic_deadline'],
            parent['monotonic_origin']+c.MODEL_SECONDS if model else parent['monotonic_deadline']),
        unix_deadline=min(wall+seconds,parent['unix_deadline'],
            parent['unix_origin']+c.MODEL_SECONDS if model else parent['unix_deadline']))
