"""The only worker entrypoint exposes saved record checks, never model execution."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
from . import common as c


def bootstrap(mode, expected, folder):
    c.validate_job_name(mode)
    c.require(folder == c.OUT/'jobs'/mode/'worker', 'Fixed new saved-only worker directory')
    plan=c.load_plan(expected, verify_inputs=False);claim=c.read(c.OUT/'claim.json')
    c.require(claim['freeze']==c.artifact(c.PLAN), 'Exact new owner freeze')
    origin=time.monotonic()
    while not (folder.parent/'owner.json').exists():
        c.check_deadlines(c.original_limits()); c.require(time.monotonic()-origin<20,'Bounded owner handshake');time.sleep(.01)
    owner=c.read(folder.parent/'owner.json');ident=c.identity()
    c.require(owner['identity']==ident and owner['mode']==mode and owner['windows_job_assigned'] is True
              and owner['plan_sha256']==expected and owner['global_claim']==c.artifact(c.OUT/'claim.json'),
              'Exact fresh worker ownership and mode')
    c.require(owner['monotonic_deadline']==min(owner['monotonic_origin']+c.JOB_CAPS[mode],claim['monotonic_deadline'])
              and owner['unix_deadline']==min(owner['unix_origin']+c.JOB_CAPS[mode],claim['unix_deadline']),
              'Exact clipped worker clocks')
    boundary=c.Boundary(claim,worker_owner=owner);boundary()
    c.verify(plan['new_source_inputs'],boundary=boundary)
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()  # Exactly once for each worker invocation.
    c.require(sources==plan['corrected_runtime_sources'],'Frozen corrected runtime sources')
    c.save(folder/'runtime_sources.json',dict(sources=sources,identity=ident,mode=mode,new_games=0,model_calls=0))
    return plan,boundary


def dispatch(mode, plan, folder, boundary):
    c.validate_job_name(mode)
    if mode==c.JOBS[0]:
        from .saved import run_independent
        return run_independent(boundary)
    if mode==c.JOBS[1]:
        from .saved import run_reader
        return run_reader(boundary)
    from .audit import final
    return final(plan,folder,boundary)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=c.JOBS,required=True)
    parser.add_argument('--freeze-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.output).resolve()
    plan,boundary=bootstrap(args.mode,args.freeze_sha256,folder)
    error=None;result=None;status='failed'
    try:
        result=dispatch(args.mode,plan,folder,boundary);boundary()
        c.require('torch' not in sys.modules,'No model runtime loaded');status='passed'
    except BaseException:error=traceback.format_exc()
    c.save(folder/'result.json',dict(status=status,mode=args.mode,identity=c.identity(),time=c.now(),
        result=result,error=error,plan_sha256=args.freeze_sha256,boundary_calls=boundary.calls,
        model_calls=0,new_games=0,optimizer_calls=0,runtime_sources=c.artifact(folder/'runtime_sources.json')))
    if error: print(error,file=sys.stderr,flush=True)
    return 0 if status=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
