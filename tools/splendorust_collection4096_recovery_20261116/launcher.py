"""Durable hidden owner supervision, with a separate model-free lifecycle probe.

Production is started explicitly with --freeze-sha256 after the previous study
has actually stopped. The probe launches only short Python fixture processes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c, clocks
from tools.splendorust_gpu_probe_20261105.control import require_absent, finish_owned_process


def identity(pid=None):
    process=psutil.Process(os.getpid() if pid is None else pid)
    return dict(pid=process.pid,created=process.create_time())


def limits(seconds):
    mono,wall=time.monotonic(),time.time()
    return dict(monotonic_origin=mono,unix_origin=wall,
        monotonic_deadline=mono+seconds,unix_deadline=wall+seconds)


def require_clock(clock):
    if 'recovery_anchor' in clock:
        clocks.check(clock)
        return
    c.require(clock['monotonic_origin'] <= time.monotonic() < clock['monotonic_deadline'] and
        clock['unix_origin'] <= time.time() < clock['unix_deadline'],'Original external-owner dual clock')


def cleanup_owned(proc,job,child,observed,errors):
    """Closing a Windows job initiates termination; wait for actual disappearance."""
    if job:
        try:job.close()
        except BaseException as caught:errors.append(repr(caught))
    if proc:
        try:finish_owned_process(proc,child)
        except BaseException as caught:errors.append(repr(caught))
    unique={(row['pid'],row['created']):row for row in [*observed.values(),*([child] if child else [])]}
    deadline=time.monotonic()+5
    pending=list(unique.values())
    while pending and time.monotonic()<deadline:
        alive=[]
        for row in pending:
            try:require_absent(row)
            except ValueError:alive.append(row)
        pending=alive
        if pending:time.sleep(.02)
    for row in pending:
        try:require_absent(row)
        except BaseException as caught:errors.append(repr(caught))


def supervise(args,folder,clock,metadata,*,rss,production=False):
    """The child remains blocked until its exact Windows job identity is durable."""
    from tools.local_player_qualification_128.clock import _WindowsJob
    from . import owner
    folder=Path(folder).resolve();c.require(folder.is_relative_to(c.OUT),'Only new owner/probe evidence')
    folder.mkdir(parents=True,exist_ok=False)
    parent=identity();proc=job=child=None;observed={};peak=0;error=None;cleanup=[]
    started_mono,started_wall=time.monotonic(),time.time()
    try:
        require_clock(clock)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=c.ROOT,env=owner.child_environment(),stdin=subprocess.DEVNULL,
                stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
            child=identity(proc.pid);job=_WindowsJob(proc.pid)
            c.save(folder/'owner.json',dict(identity=child,parent=parent,args=args,windows_job_assigned=True,
                limits=clock,maximum_rss_bytes=rss,**metadata))
            c.save(folder/'release.json',dict(identity=child,freeze_sha256=metadata.get('freeze_sha256'),
                owner=c.artifact(folder/'owner.json'),time=c.now()))
            while proc.poll() is None:
                require_clock(clock)
                peak=max(peak,owner.observe_tree(child,observed))
                c.require(peak<=rss,'Actual complete external-owner tree RSS bound')
                if production:
                    import shutil
                    c.require(shutil.disk_usage(c.OUT).free>=c.MIN_DISK,'Actual disk floor during owner supervision')
                time.sleep(.25)
            proc.wait(timeout=5)
            require_clock(clock)
            if proc.returncode != 0:error='Owned process exited '+str(proc.returncode)
    except BaseException:error=traceback.format_exc()
    finally:cleanup_owned(proc,job,child,observed,cleanup)
    result=dict(schema='collection4096_external_owner_execution_v1',status='passed' if error is None and not cleanup else 'failed',
        identity=child,parent=parent,worker_exit_code=None if proc is None else proc.returncode,
        observed_owned_identities=list(observed.values()),cleanup_errors=cleanup,error=error,
        peak_rss_bytes=peak,limits=clock,monotonic_start=started_mono,unix_start=started_wall,
        monotonic_end=time.monotonic(),unix_end=time.time(),time=c.now(),metadata=metadata,
        windows_job_assigned=job is not None,models=0 if not production else None)
    c.save(folder/'execution.json',result)
    return result


def fixture_child(folder,exit_code):
    """A real fixture owner leaves a child for its exact Windows-job cleanup."""
    folder=Path(folder).resolve()
    c.require(folder.is_relative_to(c.OUT/'prelaunch') and exit_code in (0,7),'Only separate fixture execution')
    started=time.monotonic()
    while not (folder/'release.json').exists():
        c.require(time.monotonic()-started<10,'Bounded fixture release');time.sleep(.01)
    owned=c.read(folder/'owner.json');release=c.read(folder/'release.json')
    me=identity();c.require(owned['identity']==release['identity']==me and owned['windows_job_assigned'] is True,
        'Fixture exact assigned and released owner')
    grandchild=subprocess.Popen([sys.executable,'-B','-c','import time; time.sleep(25)'],
        cwd=c.ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),close_fds=True)
    c.save(folder/'fixture_child.json',dict(identity=me,grandchild=identity(grandchild.pid),expected_exit=exit_code,
        models=0,games=0,torch_loaded='torch' in sys.modules))
    time.sleep(1.5)
    return exit_code


def probe(label):
    c.require(label.isalnum(),'Simple exclusive fixture label')
    folder=c.OUT/'prelaunch'/('launcher_probe_'+label)
    c.require(not folder.exists(),'Exclusive separate lifecycle fixture')
    folder.mkdir(parents=True)
    before={name:(c.OUT/name).exists() for name in ('claim.json','launcher_claim.json','completion.json')}
    source_inputs=[c.artifact(__file__),c.artifact(c.__file__)]
    from . import owner
    source_inputs.append(c.artifact(owner.__file__))
    me=identity();c.save(folder/'probe_claim.json',dict(identity=me,time=c.now(),inputs=source_inputs,models=0,games=0))
    results=[]
    for code in (0,7):
        child_folder=folder/('success' if code==0 else 'failure')
        clock=limits(30)
        result=supervise([sys.executable,'-B','-u','-m',c.PACKAGE+'.launcher','--fixture-child',str(child_folder),
            '--fixture-exit',str(code)],child_folder,clock,
            dict(role='model_free_lifecycle_fixture',expected_exit=code,freeze_sha256=None),rss=256*c.MiB)
        receipt=c.read(child_folder/'fixture_child.json')
        c.require(result['worker_exit_code']==code and result['status']==('passed' if code==0 else 'failed')
            and not result['cleanup_errors'] and result['windows_job_assigned'] is True,
            'Actual expected fixture exit preserved with exact cleanup')
        c.require(receipt['identity']==result['identity'] and receipt['grandchild'] in result['observed_owned_identities']
            and receipt['models']==receipt['games']==0 and receipt['torch_loaded'] is False,
            'Real descendants observed and no model work in either fixture')
        for item in result['observed_owned_identities']+[result['identity'],receipt['grandchild']]:require_absent(item)
        results.append(c.artifact(child_folder/'execution.json'))
    c.verify(source_inputs)
    c.require(before=={name:(c.OUT/name).exists() for name in before},'Probe never claims or completes production')
    c.require('torch' not in sys.modules,'Probe loads no models')
    c.save(folder/'probe_result.json',dict(status='passed',identity=me,inputs=source_inputs,
        actual_fixture_executions=results,production_files_unchanged=True,models=0,games=0,
        expected_failure_preserved=True,actual_probe_owner_exit_still_required=True,time=c.now()))
    return 0


def observed_probe(label):
    """Hold the detached supervisor's process handle until its real exit is read."""
    c.require(label.isalnum(),'Simple exclusive observed fixture label')
    folder=c.OUT/'prelaunch'/('launcher_probe_'+label)
    c.require(not folder.exists(),'Never overwrite an earlier probe')
    from . import owner
    clock=limits(90);parent=identity();source_inputs=[c.artifact(__file__),c.artifact(c.__file__),c.artifact(owner.__file__)]
    c.save(c.OUT/'prelaunch'/('probe_observer_'+label+'.json'),dict(identity=parent,limits=clock,
        inputs=source_inputs,models=0,games=0,time=c.now()))
    with (c.OUT/'prelaunch'/('probe_'+label+'_stdout.log')).open('xb') as stdout,\
         (c.OUT/'prelaunch'/('probe_'+label+'_stderr.log')).open('xb') as stderr:
        proc=subprocess.Popen([sys.executable,'-B','-u','-m',c.PACKAGE+'.launcher','--probe-owner',label],
            cwd=c.ROOT,env=owner.child_environment(),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
            close_fds=True,creationflags=getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0))
        child=identity(proc.pid)
        try:
            while proc.poll() is None:require_clock(clock);time.sleep(.1)
            code=proc.wait(timeout=5);require_absent(child);require_clock(clock)
            result=c.read(folder/'probe_result.json')
            c.require(code==0 and result['status']=='passed' and result['identity']==child,'Actual detached probe exit zero')
            c.verify(source_inputs+result['inputs']+result['actual_fixture_executions'])
            error=None
        except BaseException:
            error=traceback.format_exc();finish_owned_process(proc,child);code=proc.returncode
    receipt=dict(status='passed' if error is None else 'failed',identity=child,parent=parent,
        actual_probe_owner_exit=code,actual_probe_owner_absent=True,error=error,inputs=source_inputs,
        detached=True,models=0,games=0,time=c.now())
    c.save(c.OUT/'prelaunch'/('observed_probe_'+label+'.json'),receipt)
    return 0 if error is None else 1


def production(expected):
    from . import owner
    owner.launch_readiness()
    c.require(c.sha(c.PLAN)==expected,'Exact frozen collection before exclusive launch')
    plan=owner.validate_plan(c.read(c.PLAN))
    c.verify(plan['inputs'])
    c.require(not (c.OUT/'launcher_claim.json').exists(),'No duplicate recovery launcher')
    clock=clocks.create_limits();clocks.check(clock,model=True);me=identity()
    c.save(c.OUT/'launcher_claim.json',dict(identity=me,freeze=c.artifact(c.PLAN),time=c.now(),
        **clock,total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,automatic_retries=0))
    result=supervise([sys.executable,'-B','-u','-m',c.PACKAGE+'.owner','run','--freeze-sha256',expected],
        c.OUT/'launch',clock,dict(role='root',freeze_sha256=expected),
        rss=c.COLLECTION_RSS+c.FIT_RSS,production=True)
    error=None
    try:
        c.require(result['status']=='passed' and result['worker_exit_code']==0 and not result['cleanup_errors'],
            'Actual successful root exit and complete cleanup required')
        completion=c.read(c.OUT/'completion.json');execution=c.read(c.OUT/'execution.json')
        c.require(completion['status']==execution['status']=='verified_pending_actual_owner_exit'
            and execution['error'] is None and execution['freeze_sha256']==expected,
            'Matching completed independent study receipts')
        c.verify([completion['execution'],completion['independent_acceptance']])
        independent=c.read(c.resolve(completion['independent_acceptance']))
        c.require(independent['status']=='passed' and independent['error'] is None,
            'Independent final worker acceptance required')
        require_clock(clock);c.verify(plan['inputs'])
        c.save(c.OUT/'observed_owner_exit.json',dict(status='accepted_completed_verified',freeze=c.artifact(c.PLAN),
            launcher_identity=me,root_execution=c.artifact(c.OUT/'launch/execution.json'),
            completion=c.artifact(c.OUT/'completion.json'),independent_acceptance=completion['independent_acceptance'],
            actual_root_exit=0,exact_root_identity_absent=True,owned_cleanup_verified=True,
            launcher_exit_still_requires_observation=True,time=c.now(),new_followup_started=False))
    except BaseException:error=traceback.format_exc()
    if error:
        c.save(c.OUT/'launcher_failure.json',dict(status='incomplete',error=error,
            root_execution=c.artifact(c.OUT/'launch/execution.json'),time=c.now(),no_retry=True))
        return 1
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-sha256');group.add_argument('--probe');group.add_argument('--probe-owner');group.add_argument('--fixture-child')
    parser.add_argument('--fixture-exit',type=int,choices=(0,7))
    args=parser.parse_args()
    if args.fixture_child:
        c.require(args.fixture_exit in (0,7),'Explicit fixture exit');return fixture_child(args.fixture_child,args.fixture_exit)
    c.require(args.fixture_exit is None,'Fixture exit is private to fixture children')
    if args.probe_owner:return probe(args.probe_owner)
    return observed_probe(args.probe) if args.probe else production(args.freeze_sha256)


if __name__=='__main__':raise SystemExit(main())
