"""One actual process owner for the separate saved-only recovery audit."""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c
from tools.splendorust_search128_resume_20261029.guardian import cleanup_owned
from tools.local_player_qualification_128.clock import _WindowsJob

def deadline_remaining(monotonic_deadline, unix_deadline):
    return min(monotonic_deadline-time.monotonic(), unix_deadline-time.time())

def observe(command, label, freeze, freeze_sha, deadlines, *, permit_children=False):
    """Always close the owned tree, even when the post-spawn setup itself fails."""
    started=time.monotonic(); process=job=None; identity=None; reason=None; cleanup=None
    streams=[]; peak=0; code=None
    last_deadlines=(started,time.time())
    def remaining():
        nonlocal last_deadlines
        candidate=deadlines()
        c.require(len(candidate)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in candidate),
                  'Owned deadline callback schema')
        last_deadlines=candidate
        return deadline_remaining(*last_deadlines)
    try:
        c.require(remaining()>0, 'No time before owned spawn')
        for suffix in ('stdout','stderr'):
            path=c.OUT/(label+'_'+suffix+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            streams.append(path.open('xb'))
        env=dict(os.environ,PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
            PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',SPLENDOR_SEARCH128_PHASE='strength')
        process=subprocess.Popen(command,cwd=c.ROOT,stdin=subprocess.DEVNULL,
            stdout=streams[0],stderr=streams[1],env=env,
            # The qualified detached bootstrap avoids an implicit conhost child
            # for the otherwise child-free saved-evidence reader.
            creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
        # All post-spawn operations are within the cleanup-protected try.
        child=psutil.Process(process.pid)
        identity=dict(pid=child.pid,created=child.create_time())
        job=_WindowsJob(process.pid)
        child.cpu_affinity(freeze['placement']['role_masks']['isolated'])
        child.nice(psutil.NORMAL_PRIORITY_CLASS)
        c.atomic_save(c.OUT/(label+'_launch.json'),dict(**identity,owner_pid=os.getpid(),
            freeze_sha256=freeze_sha,command=command,time=c.now(),
            observed_by='owned Process handle',permit_children=permit_children))
        next_sample=0.
        while process.poll() is None:
            c.require(remaining()>0,'Owned process shared real-clock deadline')
            current=time.monotonic()
            if current>=next_sample:
                try:
                    c.require(child.create_time()==identity['created'],'Owned root identity changed')
                    rss=child.memory_info().rss;peak=max(peak,rss)
                    c.require(rss<=c.MAX_RSS_BYTES,'Owned supervisor/reader RSS exceeded2GiB')
                    c.require(shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Free disk reserve')
                    if not permit_children:
                        c.require(not child.children(recursive=True),'Model-free reviewer created child process')
                except (psutil.NoSuchProcess,psutil.ZombieProcess):
                    c.require(process.poll() is not None,'Owned process disappeared without handle exit')
                next_sample=current+2.
            time.sleep(.1)
        code=process.wait(timeout=0)
        c.require(remaining()>0,'Owned exit after shared deadline')
    except BaseException as error:
        reason=repr(error)
    finally:
        # Cleanup has at most five seconds. It cannot make a late result acceptable.
        # A malformed newly published claim can make deadlines() raise. Cleanup
        # must still run using the last already validated bound, not call it again.
        end_mono,end_wall=last_deadlines
        cleanup=cleanup_owned(process,job,streams,
            monotonic_origin=end_mono-c.JOB_SECONDS,unix_origin=end_wall-c.JOB_SECONDS,
            root_identity=identity)
        if process is not None:
            code=process.poll()
        if cleanup['errors']:
            reason=reason or 'Owned cleanup failed: '+repr(cleanup['errors'])
    return dict(pid=None if process is None else process.pid,
        created=None if identity is None else identity['created'],exit_code=code,
        observed_by='owned Process handle',owner_pid=os.getpid(),freeze_sha256=freeze_sha,
        elapsed_seconds=time.monotonic()-started,cleanup_reason=reason,
        cleanup_receipt=cleanup,peak_rss_bytes=peak)


def run(sha):
    c.require(c.file_hash(c.OUT/'freeze.json')==sha,'Exact separate audit freeze')
    frozen=c.read(c.OUT/'freeze.json')
    review=c.read(c.OUT/'prelaunch_review.json')
    c.require(review['approved'] is True and review['freeze_sha256']==sha,'Independent exact prelaunch approval')
    c.verify_pins(frozen['implementation_inputs'])
    c.require(not any((c.OUT/name).exists() for name in ('owner_clock.json','component.json','owner_failure.json')),
              'One audit launch only; preserve failures')
    c.require(psutil.virtual_memory().available>=4*c.GiB,'Initial saved-audit4GiB available RAM reserve')
    c.require(shutil.disk_usage(c.ROOT).free>=5*c.GiB,'Saved-audit free disk reserve')
    final=c.read(c.SOURCE/'stages/final_start.json')
    mono,wall=time.monotonic(),time.time();ends=c.deadline_pair(mono,wall,final)
    c.require(min(ends[0]-mono,ends[1]-wall)>=900,'At least15minutes remain for bounded saved-only audit')
    clock=dict(time=c.now(),monotonic_origin=mono,unix_origin=wall,deadline_monotonic=ends[0],deadline_unix=ends[1],
        maximum_seconds=c.LIMIT,freeze_sha256=sha,source_freeze_sha256=c.SOURCE_SHA,
        owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),
        scope='new separately authorized saved-evidence audit; original expired review is not reset',
        original_review_clock=c.artifact(c.SOURCE/'postrun/review_owner_clock.json'))
    c.save(c.OUT/'owner_clock.json',clock)
    source_freeze=c.read(c.SOURCE/'freeze.json')
    script=Path(__file__).with_name('audit.py')
    command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.audit']
    terminal=observe(command+['component','--freeze-sha256',sha],'reader',source_freeze,sha,lambda:ends)
    component=c.OUT/'component.json'
    c.save(c.OUT/'reader_os_exit.json',dict(terminal,script_sha256=c.file_hash(script),
        component_sha256=c.file_hash(component) if component.exists() else None))
    c.require(terminal['exit_code']==0 and terminal['cleanup_reason'] is None,'New reader actual exit0 required')
    c.guard_clock(clock)
    terminal=observe(command+['accept','--freeze-sha256',sha],'acceptance',source_freeze,sha,lambda:ends)
    c.save(c.OUT/'acceptance_os_exit.json',dict(terminal,script_sha256=c.file_hash(script)))
    c.require(terminal['exit_code']==0 and terminal['cleanup_reason'] is None,'New acceptance actual exit0 required')
    accepted=c.read(c.OUT/'acceptance.json')
    c.require(accepted['recovered_verification'] is True and accepted['original_acceptance'] is False and
        accepted['source_attempt_accepted'] is False and not(c.OUT/'acceptance_invalidated.json').exists(),
        'Separate recovered verification only')
    c.guard_clock(clock)
    c.save(c.OUT/'owner_completion.json',dict(status='recovered_verification_pending_observed_owner_exit0',
        time=c.now(),audit_freeze_sha256=sha,source_freeze_sha256=c.SOURCE_SHA,reader_exit_code=0,
        acceptance_exit_code=0,original_acceptance=False,source_attempt_accepted=False,
        acceptance=c.artifact(c.OUT/'acceptance.json'),clock=clock,elapsed_seconds=time.monotonic()-mono,
        actual_owner_exit_must_be_observed=True,model_calls=0,new_games=0))
    c.guard_clock(clock)
    print(c.canonical(dict(status='recovered_verification_pending_observed_owner_exit0',elapsed_seconds=time.monotonic()-mono)),flush=True)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-sha256',required=True);args=p.parse_args()
    try:return run(args.freeze_sha256)
    except BaseException as exc:
        for name in ('owner_failure.json','acceptance_invalidated.json'):
            if not(c.OUT/name).exists():c.save(c.OUT/name,dict(status='no_recovered_acceptance',time=c.now(),
                error=repr(exc),traceback=traceback.format_exc(),source_attempt_unchanged=True))
        print(c.canonical(dict(status='failed',error=repr(exc))),flush=True);return 1


if __name__=='__main__':raise SystemExit(main())

