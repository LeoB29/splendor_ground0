"""Durable actual-exit observer, independent of a retained interactive tool session."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c


def run_child(args, folder, *, metadata, environment=None, limits=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_full_corpus_recovery_20261111.owner import cleanup_owned, observe_tree
    proc=job=ident=None;observed={};errors=[];error=None
    try:
        if limits:c.check_deadlines(limits)
        with (folder/'stdout.log').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
            proc=subprocess.Popen(args,cwd=c.ROOT,env=environment,stdout=stdout,stderr=stderr,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            ident=dict(pid=proc.pid,created=psutil.Process(proc.pid).create_time());job=_WindowsJob(proc.pid)
            c.save(folder/'child.json',dict(identity=ident,launcher=c.identity(),args=args,windows_job_assigned=True,**metadata))
            while proc.poll() is None:
                if limits:c.check_deadlines(limits)
                observe_tree(ident,observed);time.sleep(.25)
            proc.wait()
            if limits:c.check_deadlines(limits)
    except BaseException:error=traceback.format_exc()
    finally:cleanup_owned(proc,job,ident,observed,errors)
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    if ident:require_absent(ident)
    code=None if proc is None else proc.returncode
    receipt=c.save(folder/'actual_exit.json',dict(identity=ident,launcher=c.identity(),exit_code=code,
        actual_exit_observed=proc is not None and code is not None,exact_process_absent=ident is not None,
        observed_owned_identities=list(observed.values()),error=error,cleanup_errors=errors,time=c.now(),**metadata))
    c.require(error is None and not errors,'Launcher failure preserved after exact owned cleanup')
    return code,receipt


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze-sha256');group.add_argument('--detached-fixture',action='store_true')
    parser.add_argument('--fixture-label',choices=('01','02'),default='02')
    args=parser.parse_args()
    if args.detached_fixture:
        label=args.fixture_label;mono,wall=time.monotonic(),time.time();original=c.original_limits()
        limits=dict(monotonic_origin=mono,unix_origin=wall,
            monotonic_deadline=min(mono+60,original['monotonic_deadline']),unix_deadline=min(wall+60,original['unix_deadline']))
        c.save(c.OUT/f'prelaunch/detached_launcher_identity{label}.json',dict(identity=c.identity(),fixture=True,time=c.now()))
        code,_=run_child([sys.executable,'-B','-c','import time; time.sleep(5)'],c.OUT/f'prelaunch/detached_fixture{label}',
                         metadata=dict(fixture=True,model_calls=0,new_games=0,launch_limits=limits),limits=limits)
        return code
    c.load_plan(args.freeze_sha256,verify_inputs=False);c.assert_stopped()
    c.require(not (c.OUT/'claim.json').exists(),'No duplicate finalization owner')
    mono,wall=time.monotonic(),time.time();original=c.original_limits()
    limits=dict(monotonic_origin=mono,unix_origin=wall,
        monotonic_deadline=min(mono+c.TOTAL_SECONDS,original['monotonic_deadline']),
        unix_deadline=min(wall+c.TOTAL_SECONDS,original['unix_deadline']))
    c.save(c.OUT/'launcher_claim.json',dict(identity=c.identity(),freeze=c.artifact(c.PLAN),time=c.now(),limits=limits))
    from .owner import child_environment
    command=[sys.executable,'-B','-u','-m',c.PACKAGE+'.owner','run','--freeze-sha256',args.freeze_sha256]
    code,receipt=run_child(command,c.OUT/'launch',metadata=dict(plan_sha256=args.freeze_sha256,launch_limits=limits),
                           environment=child_environment(c.JOBS[0]),limits=limits)
    return code


if __name__=='__main__': raise SystemExit(main())
