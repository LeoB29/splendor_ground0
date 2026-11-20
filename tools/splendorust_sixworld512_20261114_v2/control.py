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
    # Windows directory enumeration carries file size, avoiding one expensive
    # extra filesystem request per immutable decision artifact in long runs.
    used=0;pending=[c.OUT]
    while pending:
        directory=pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):used+=entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    c.require(entry.name in ('stage.tmp','progress.tmp'),'Only transient status rename may vanish')
    return used

def bounds(limits):
    c.check_deadlines(limits)
    c.require(shutil.disk_usage(c.OUT).free>=c.MIN_DISK,'Actual free disk floor')

def observe_absence(identities,limits,seconds=5.):
    """Bounded passive observation of asynchronous cleanup, with original clock."""
    started=time.monotonic();samples=0
    while True:
        c.check_deadlines(limits);samples+=1
        absent=all(not c.exact_alive(i) for i in identities)
        if absent or time.monotonic()-started>=seconds:
            return dict(exact_owned_absence=absent,elapsed_seconds=time.monotonic()-started,samples=samples)
        time.sleep(.05)

def cleanup_owned(proc,job,ident,observed,errors):
    """Wait for asynchronous Windows job cleanup; never target a reused PID."""
    if job:
        try:job.close()
        except BaseException as exc:errors.append(repr(exc))
    if proc:
        try:
            if proc.poll() is None and ident is not None and c.exact_alive(ident):
                try:
                    parent=psutil.Process(ident['pid'])
                    for child in parent.children(recursive=True):
                        try:
                            created=child.create_time()
                            observed[(child.pid,created)]=dict(pid=child.pid,created=created)
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                try:proc.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    if c.exact_alive(ident):
                        try:psutil.Process(ident['pid']).kill()
                        except psutil.NoSuchProcess:pass
                    proc.wait(timeout=5.)
            else:proc.wait(timeout=5.)
        except BaseException as exc:errors.append(repr(exc))
    targets=list(observed.values())+([ident] if ident else [])
    until=time.monotonic()+5.
    while any(c.exact_alive(i) for i in targets) and time.monotonic()<until:time.sleep(.05)
    for item in targets:
        if c.exact_alive(item):
            try:psutil.Process(item['pid']).kill()
            except psutil.NoSuchProcess:pass
            except BaseException as exc:errors.append(repr(exc))
    until=time.monotonic()+5.
    while any(c.exact_alive(i) for i in targets) and time.monotonic()<until:time.sleep(.05)
    if any(c.exact_alive(i) for i in targets):errors.append('Exact owned process remains after bounded cleanup')

def launch(command,folder,limits,metadata,rss=c.AUDIT_RSS):
    """Child waits for owner handshake; OS job contains its entire lifetime."""
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_full_corpus_recovery_20261111.owner import observe_tree
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    proc=job=ident=None;observed={};errors=[];error=None;peak=0;last_scan=0
    started=time.monotonic()
    try:
        bounds(limits)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(command,cwd=c.ROOT,env=environment(),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            ident=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time())
            job=_WindowsJob(proc.pid)
            c.atomic_save(folder/'owner.json',dict(identity=ident,parent=c.identity(),command=command,
                windows_job_assigned=True,limits=limits,rss_cap=rss,**metadata))
            while proc.poll() is None:
                bounds(limits);used=observe_tree(ident,observed);peak=max(peak,used)
                c.require(used<=rss,'Actual child-tree RSS bound')
                if time.monotonic()-last_scan>=30:
                    c.require(output_bytes()<c.MAX_OUTPUT,'Actual new-evidence size bound');last_scan=time.monotonic()
                time.sleep(.5)
            proc.wait();bounds(limits)
    except BaseException:error=traceback.format_exc()
    finally:cleanup_owned(proc,job,ident,observed,errors)
    code=None if proc is None else proc.returncode
    absent=ident is not None and not c.exact_alive(ident) and all(not c.exact_alive(i) for i in observed.values())
    result_artifact=None
    if error is None and code==0 and metadata.get('role') in (*c.JOBS,'root'):
        try:
            result_path=c.OUT/'completion.json' if metadata['role']=='root' else folder/'worker/result.json'
            result_artifact=c.artifact(result_path)
        except BaseException:error=traceback.format_exc()
    passed=error is None and not errors and code==0 and absent
    item=c.save(folder/'actual_exit.json',dict(status='passed' if passed else 'failed',identity=ident,
        exit_code=code,actual_exit_observed=code is not None,exact_owned_absence=absent,
        observed_owned_identities=list(observed.values()),peak_rss_bytes=peak,elapsed_seconds=time.monotonic()-started,
        cleanup_errors=errors,error=error,result_artifact=result_artifact,time=c.now(),**metadata))
    c.require(passed,'Child failed; actual exit and failure preserved: '+folder.name)
    return item

def handshake(folder,expected,role):
    start=time.monotonic();path=Path(folder)/'owner.json'
    while not path.exists():
        c.require(time.monotonic()-start<30,'Bounded ownership handshake');time.sleep(.02)
    owner=c.read(path)
    c.require(owner['identity']==c.identity() and owner['windows_job_assigned'] is True and
        owner['freeze_sha256']==expected and owner['role']==role,'Exact frozen owned role')
    c.require(c.exact_alive(owner['parent']),'Exact live owner')
    return owner

class Boundary:
    def __init__(self,folder,owner):self.folder=Path(folder);self.owner=owner;self.last=0;self.calls=0
    def __call__(self):
        self.calls+=1;c.check_deadlines(self.owner['limits'])
        if time.monotonic()-self.last>=5:
            bounds(self.owner['limits']);c.require(c.exact_alive(self.owner['parent']),'Exact owner remains alive')
            c.require(psutil.Process().memory_info().rss<=self.owner['rss_cap'],'Actual worker RSS bound')
            c.status(self.folder/'boundary/progress.json',dict(time=c.now(),role=self.owner['role'],calls=self.calls))
            self.last=time.monotonic()

def clipped_limits(parent,seconds,model=False):
    mono,wall=time.monotonic(),time.time()
    return dict(monotonic_origin=mono,unix_origin=wall,
        monotonic_deadline=min(mono+seconds,parent['monotonic_deadline'],
            parent['monotonic_origin']+c.MODEL_SECONDS if model else parent['monotonic_deadline']),
        unix_deadline=min(wall+seconds,parent['unix_deadline'],
            parent['unix_origin']+c.MODEL_SECONDS if model else parent['unix_deadline']))
