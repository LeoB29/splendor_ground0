"""Own every actual exit through play and independent acceptance, on one clock."""
import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c, placement
from .guardian import cleanup_owned
from tools.local_player_qualification_128.clock import _WindowsJob

PACKAGE = 'tools.splendorust_search128_resume_20261029'


def independent_reader_route(freeze_sha):
    module = PACKAGE + '.independent_review'
    return (c.ROOT / (module.replace('.', '/') + '.py'),
            [sys.executable, '-B', '-u', '-m', module, '--freeze-sha256', freeze_sha])


def deadline_remaining(monotonic_deadline, unix_deadline):
    return min(monotonic_deadline-time.monotonic(), unix_deadline-time.time())


def acceptance_deadlines(claim, completion, mono, wall):
    c.require(completion['freeze_sha256']==claim['freeze_sha256'], 'Completion freeze identity')
    c.require(completion['independent_review_maximum_seconds']==c.INDEPENDENT_REVIEW_SECONDS,
              'Frozen review allowance')
    c.require(all(type(x) in (int,float) and math.isfinite(x) for x in
        (mono,wall,claim['monotonic_origin'],claim['unix_origin'],
         completion['independent_acceptance_deadline_monotonic'],
         completion['independent_acceptance_deadline_unix'])), 'Finite acceptance clocks')
    return (min(mono+c.INDEPENDENT_REVIEW_SECONDS,
                completion['independent_acceptance_deadline_monotonic'],
                claim['monotonic_origin']+c.JOB_SECONDS),
            min(wall+c.INDEPENDENT_REVIEW_SECONDS,
                completion['independent_acceptance_deadline_unix'],
                claim['unix_origin']+c.JOB_SECONDS))


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


def invalidate(reason):
    path=c.OUT/'postrun/acceptance_invalidated.json'
    if not path.exists():
        c.atomic_save(path,dict(accepted=False,time=c.now(),reason=reason,owner_pid=os.getpid()))


def run(freeze_sha):
    freeze_path=c.OUT/'freeze.json'
    c.require(c.file_hash(freeze_path)==freeze_sha,'Exact frozen attempt')
    freeze=c.read(freeze_path)
    review=c.read(c.OUT/'prelaunch_review.json')
    c.require(review.get('approved') is True and review.get('freeze_sha256')==freeze_sha,
              'Independent exact prelaunch approval')
    from .runner import validate_freeze_contract
    validate_freeze_contract(freeze)
    c.require(not any((c.OUT/p).exists() for p in ('owner_launch.json','guardian_claim.json',
        'guardian_os_exit.json','postrun/independent_review_component.json','postrun/independent_completion_review.json')),
        'One owned launch only; preserve previous attempts')
    launch_placement=placement.configure_execution(freeze,'isolated',role='watchdog')
    c.atomic_save(c.OUT/'launcher_placement.json',dict(launch_placement,
        logical_processors=launch_placement['process_mask'],priority='Normal',scope='retained_launch_owner',
        freeze_sha256=freeze_sha))
    start,wall=time.monotonic(),time.time()
    c.atomic_save(c.OUT/'owner_launch.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),
        time=c.now(),freeze_sha256=freeze_sha,owner_monotonic_origin=start,owner_unix_origin=wall,
        guardian_claim_handshake_seconds=60,automatic_retries=0))
    claim_cache=None
    def guardian_deadlines():
        nonlocal claim_cache
        path=c.OUT/'guardian_claim.json'
        if claim_cache is None and path.exists():
            claim_cache=c.read(path)
            c.require(claim_cache['freeze_sha256']==freeze_sha,'Guardian claim freeze')
        if claim_cache is None:return start+60,wall+60
        return claim_cache['monotonic_origin']+c.JOB_SECONDS,claim_cache['unix_origin']+c.JOB_SECONDS
    result=observe([sys.executable,'-B','-u','-m',PACKAGE+'.guardian','--freeze-sha256',freeze_sha],
        'guardian',freeze,freeze_sha,guardian_deadlines,permit_children=True)
    c.atomic_save(c.OUT/'guardian_os_exit.json',dict(result,guardian_pid=result['pid'],guardian_created=result['created']))
    c.require(result['exit_code']==0 and result['cleanup_reason'] is None,'Guardian actual exit must pass')
    claim=c.read(c.OUT/'guardian_claim.json')
    c.require((claim['guardian_pid'],claim['guardian_created'])==(result['pid'],result['created']),
              'Actual guardian identity must match original clock owner')
    completion=c.read(c.OUT/'completion.json')
    c.require(completion['status'] == 'amended_recovered_verified',
              'Controller did not complete its declared scientific checks')
    if completion['status']=='not_admitted_verified':
        # A scientifically checked negative admission is retained, but cannot be
        # submitted to the full400-game acceptance reader or called a benchmark.
        end_mono,end_wall=acceptance_deadlines(claim,completion,time.monotonic(),time.time())
        c.require(deadline_remaining(end_mono,end_wall)>0,'Negative disposition shared clock')
        c.atomic_save(c.OUT/'owner_completion.json',dict(status='not_admitted',time=c.now(),
            freeze_sha256=freeze_sha,study_status=completion['status'],guardian_exit_code=0,
            independent_full_study_review_run=False,accepted_strength_result=False,
            elapsed_seconds=time.monotonic()-start,automatic_retries=0))
        c.require(deadline_remaining(end_mono,end_wall)>0,'Negative disposition publication clock')
        return 0
    mono,unix=time.monotonic(),time.time()
    mono_end,wall_end=acceptance_deadlines(claim,completion,mono,unix)
    c.require(deadline_remaining(mono_end,wall_end)>0,'No time remains for independent verification')
    clock=dict(review_monotonic_origin=mono,review_unix_origin=unix,
        deadline_monotonic=mono_end,deadline_unix=wall_end,freeze_sha256=freeze_sha,
        owner_pid=os.getpid(),owner_created=psutil.Process().create_time(),time=c.now(),
        guardian_claim_sha256=c.file_hash(c.OUT/'guardian_claim.json'),
        completion_sha256=c.file_hash(c.OUT/'completion.json'))
    c.atomic_save(c.OUT/'postrun/review_owner_clock.json',clock)
    deadlines=lambda:(mono_end,wall_end)
    script,command=independent_reader_route(freeze_sha)
    result=observe(command,'postrun/independent_reader',freeze,freeze_sha,deadlines)
    component=c.OUT/'postrun/independent_review_component.json'
    c.atomic_save(c.OUT/'postrun/independent_review_os_exit.json',dict(result,
        reader_pid=result['pid'],reader_created=result['created'],script_sha256=c.file_hash(script),
        component_sha256=c.file_hash(component) if component.exists() else None))
    c.require(result['exit_code']==0 and result['cleanup_reason'] is None,'Independent reader actual exit must pass')
    c.require(deadline_remaining(*deadlines())>0,'No shared time for acceptance')
    result=observe(command+['--accept-exit'],'postrun/independent_acceptance',freeze,freeze_sha,deadlines)
    c.atomic_save(c.OUT/'postrun/independent_acceptance_os_exit.json',dict(result,
        acceptance_pid=result['pid'],acceptance_created=result['created'],script_sha256=c.file_hash(script)))
    c.require(result['exit_code']==0 and result['cleanup_reason'] is None,'Acceptance actual exit must pass')
    accepted=c.read(c.OUT/'postrun/independent_completion_review.json')
    c.require(accepted['accepted'] is True and accepted.get('amended_recovered_benchmark') is True and
              accepted.get('original_attempt_remains_incomplete') is True and
              accepted.get('first_recovery_remains_incomplete') is True and
              not(c.OUT/'postrun/acceptance_invalidated.json').exists(),
              'Final independent acceptance missing or invalidated')
    c.require(deadline_remaining(*deadlines())>0,'Owner acceptance guard before publication')
    c.atomic_save(c.OUT/'owner_completion.json',dict(status='passed',time=c.now(),freeze_sha256=freeze_sha,
        amended_recovered_benchmark=True,original_attempt_remains_incomplete=True,
        first_recovery_remains_incomplete=True,
        study_status=completion['status'],guardian_exit_code=0,reader_exit_code=0,acceptance_exit_code=0,
        elapsed_seconds=time.monotonic()-start,charged_job_seconds=max(time.monotonic()-claim['monotonic_origin'],
            time.time()-claim['unix_origin']),accepted_review_sha256=c.file_hash(c.OUT/'postrun/independent_completion_review.json')))
    c.require(deadline_remaining(*deadlines())>0,'Owner acceptance guard after publication')
    return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args()
    try:return run(args.freeze_sha256)
    except BaseException as error:
        # Preserve a published component/acceptance but invalidate it if any owner
        # check fails. No inferred success or automatic new clock follows a crash.
        invalidate(repr(error))
        failure=c.OUT/'owner_failure.json'
        if not failure.exists():
            c.atomic_save(failure,dict(status='not_accepted',time=c.now(),error=repr(error),
                traceback=traceback.format_exc(),freeze_sha256=args.freeze_sha256))
        return 1


if __name__=='__main__':raise SystemExit(main())
