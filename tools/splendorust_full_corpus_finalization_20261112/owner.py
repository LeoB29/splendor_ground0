"""Bounded owner of three saved-only checks, never a model or game runner."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c


def child_environment(mode):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               FULL_CORPUS_MATCHUP='mixed_terminal',
               INDEPENDENT_DATA_REPETITION='1' if mode == c.JOBS[0] else '0')
    return env


def run_job(claim, plan_sha, mode, boundary):
    c.validate_job_name(mode)
    from tools.local_player_qualification_128.clock import _WindowsJob
    from tools.splendorust_full_corpus_recovery_20261111.owner import observe_tree, cleanup_owned
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    folder = c.OUT / 'jobs' / mode; folder.mkdir(parents=True, exist_ok=False)
    worker = folder / 'worker'; worker.mkdir()
    mono, wall = time.monotonic(), time.time()
    limits = dict(monotonic_deadline=min(mono+c.JOB_CAPS[mode], claim['monotonic_deadline']),
                  unix_deadline=min(wall+c.JOB_CAPS[mode], claim['unix_deadline']))
    args = [sys.executable, '-B', '-u', '-m', c.PACKAGE+'.worker', '--mode', mode,
            '--freeze-sha256', plan_sha, '--output', str(worker)]
    proc = job = ident = None; observed = {}; cleanup = []; error = None; status = 'failed'; peak = 0
    c.status(dict(time=c.now(), stage='saved_finalization', job=mode, status='starting',
                  main_games=1200, exact_replays=463, new_games=0, model_calls=0))
    last_deep = -float('inf')
    try:
        boundary()
        with (worker/'stdout.log').open('xb') as stdout, (worker/'stderr.log').open('xb') as stderr:
            proc = subprocess.Popen(args, cwd=c.ROOT, env=child_environment(mode), stdout=stdout, stderr=stderr,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            ident = dict(pid=proc.pid, created=psutil.Process(proc.pid).create_time()); job = _WindowsJob(proc.pid)
            c.save(folder/'owner.json', dict(identity=ident, args=args, mode=mode,
                created_monotonic=mono, created_unix=wall, monotonic_origin=mono, unix_origin=wall,
                **limits, maximum_seconds=c.JOB_CAPS[mode], maximum_rss_bytes=c.MAX_RSS,
                windows_job_assigned=True, plan_sha256=plan_sha, model_work=False,
                global_claim=c.artifact(c.OUT/'claim.json'), source_reader_clock=c.artifact(c.SOURCE/'reader_clock.json')))
            while proc.poll() is None:
                boundary(); c.check_deadlines(limits)
                used=observe_tree(ident, observed); peak=max(peak, used)
                c.require(used <= c.MAX_RSS, 'Actual owned model-free worker RSS cap')
                if time.monotonic()-last_deep >= 10:
                    last_deep=time.monotonic(); c.require(c.output_bytes(c.OUT) <= c.MAX_NEW_BYTES, 'New audit evidence cap')
                time.sleep(.25)
            boundary(); c.check_deadlines(limits)
            c.require(proc.returncode == 0, 'Saved worker failed; no automatic retry')
            result=c.read(worker/'result.json')
            c.require(result['status']=='passed' and result['error'] is None
                      and result['mode']==mode and result['identity']==ident,
                      'Exact completed saved-only worker role and identity')
            c.require(result['model_calls']==result['new_games']==result['optimizer_calls']==0,
                      'No new scientific work in finalization')
            for item in [*observed.values(), ident]: require_absent(item)
            status='passed'
    except BaseException: error=traceback.format_exc()
    finally: cleanup_owned(proc, job, ident, observed, cleanup)
    if cleanup: status='failed'
    receipt=c.save(folder/'execution.json', dict(status=status, mode=mode, identity=ident,
        worker_exit_code=None if proc is None else proc.returncode, error=error, cleanup_errors=cleanup,
        observed_owned_identities=list(observed.values()), monotonic_origin=mono, unix_origin=wall,
        monotonic_end=time.monotonic(), unix_end=time.time(), ended_monotonic=time.monotonic(),
        ended_unix=time.time(), **limits, maximum_seconds=c.JOB_CAPS[mode], peak_rss_bytes=peak,
        plan_sha256=plan_sha, retries=0, model_calls=0, new_games=0, optimizer_calls=0))
    c.require(status=='passed', 'Failed saved-only job preserved: '+mode)
    return receipt


def run(expected):
    plan=c.load_plan(expected); c.assert_stopped()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()
    c.require(sources==plan['corrected_runtime_sources'], 'Exact corrected runtime at owner startup')
    mono, wall=time.monotonic(),time.time(); original=c.original_limits()
    launch=c.read(c.OUT/'launcher_claim.json')
    c.require(launch['freeze']==c.artifact(c.PLAN),'Exact bounded standalone launcher')
    c.check_deadlines(launch['limits'])
    claim=dict(identity=c.identity(), freeze=c.artifact(c.PLAN), plan_sha256=expected,
        monotonic_origin=mono, unix_origin=wall,
        monotonic_deadline=min(mono+c.TOTAL_SECONDS, original['monotonic_deadline'],launch['limits']['monotonic_deadline']),
        unix_deadline=min(wall+c.TOTAL_SECONDS, original['unix_deadline'],launch['limits']['unix_deadline']),
        launcher_claim=c.artifact(c.OUT/'launcher_claim.json'),
        source_claim=c.artifact(c.SOURCE/'claim.json'), original_reader=c.artifact(c.SOURCE/'reader_clock.json'),
        source_attempt_status='incomplete', source_actual_owner_exit='unknown',
        original_attempt_status='incomplete', new_games=0, model_calls=0, optimizer_calls=0)
    c.save(c.OUT/'claim.json', claim); boundary=c.Boundary(claim); receipts=[]; error=None
    try:
        for mode in c.JOBS: receipts.append(run_job(claim, expected, mode, boundary))
        boundary(); c.verify(plan['new_source_inputs'], boundary=boundary)
        accepted=c.read(c.OUT/'jobs/final_acceptance/worker/independent_acceptance.json')
        c.require(accepted['status']=='passed', 'Final independent saved-evidence acceptance')
        status='verified_pending_actual_owner_exit'
    except BaseException:
        error=traceback.format_exc(); status='incomplete'
    receipt=c.save(c.OUT/'execution.json', dict(status=status, identity=c.identity(), jobs=receipts,
        error=error, monotonic_end=time.monotonic(), unix_end=time.time(), plan_sha256=expected,
        new_games=0, model_calls=0, optimizer_calls=0, source_actual_owner_exit='unknown',
        source_attempt_status='incomplete', original_attempt_status='incomplete'))
    c.status(dict(time=c.now(), status=status, stage=status, completed_jobs=len(receipts)))
    if error:
        c.save(c.OUT/'failure.json', dict(status='incomplete', execution=receipt, error=error)); return 1
    c.save(c.OUT/'completion.json', dict(status=status, execution=receipt,
        acceptance=c.artifact(c.OUT/'jobs/final_acceptance/worker/independent_acceptance.json'),
        actual_owner_exit_still_requires_observation=True, no_automatic_followup=True))
    return 0


def freeze(test_path, review_paths):
    c.assert_stopped(); c.check_deadlines(c.original_limits())
    tests=c.read(test_path)
    c.require(tests['status']=='passed' and tests['failures']==tests['errors']==tests['skipped']==0,
              'Fresh source-bound tests passed without skips')
    c.require(tests['actual_exit_observed'] is True and tests['exit_code']==0
              and tests['torch_loaded'] is False and tests['passed']==tests['collected']>0,
              'Actual model-free qualification exit and complete test collection')
    reviews=[c.read(path) for path in review_paths]
    c.require(len(reviews)>=3 and all(x['approved'] is True and x['blocking_findings']==[] for x in reviews),
              'Three independent source-bound reviews approve')
    c.require(len({r['reviewer'] for r in reviews}) == len(reviews)
              and all(isinstance(r['reviewer'],str) and r['reviewer'] for r in reviews),
              'Review receipts belong to distinct independent reviewers')
    source_paths=sorted((c.ROOT/'tools'/c.PACKAGE.split('.')[-1]).glob('*.py'))+[c.PROTOCOL]
    source_inputs=[c.artifact(path) for path in source_paths]
    c.require(tests['source_inputs']==source_inputs and all(r['source_inputs']==source_inputs for r in reviews),
              'Tests and reviews bind the exact final source')
    previous=c.read(c.SOURCE/'freeze.json')
    inherited=previous['inputs']
    extra=[c.SOURCE/'freeze.json', c.SOURCE/'claim.json', c.SOURCE/'reader_clock.json',
           c.SOURCE/'stages/final_start.json', c.OUT/'inspection.json', c.OUT/'source_inventory.jsonl',
           c.OUT/'inventory_helper_execution.json', c.DETACHED_ACCEPTANCE,
           Path(test_path), *map(Path,review_paths)]
    dependencies=[]
    for path in (c.OUT/'inspection.json',c.OUT/'inventory_helper_execution.json',
                 c.DETACHED_ACCEPTANCE):
        dependencies += c.prior.nested_artifacts(c.read(path))
    inputs={x['path']:x for x in inherited+source_inputs+dependencies+[tests['harness'],tests['worker_receipt']]+[c.artifact(path) for path in extra]}
    c.verify(list(inputs.values()), boundary=lambda:c.check_deadlines(c.original_limits()))
    inventory_exit=c.read(c.OUT/'inventory_helper_execution.json')
    c.require(inventory_exit.get('status')=='passed' and inventory_exit.get('exit_code')==0,
              'Source preservation helper actually exited successfully')
    c.require(inventory_exit['inventory']==c.artifact(c.OUT/'source_inventory.jsonl')
              and inventory_exit['inspection']==c.artifact(c.OUT/'inspection.json')
              and inventory_exit['identity_absent'] is True,
              'Actual helper exit binds this exact source preservation inventory')
    c.require(c.read(c.OUT/'inspection.json')['source_inventory']==inventory_exit['inventory'],
              'Inspected stopped source binds the same completed inventory')
    with (c.OUT/'source_inventory.jsonl').open(encoding='utf-8') as stream:
        import json
        source_bytes=sum(json.loads(line)['bytes'] for line in stream)
    preserved_bytes=c.read(c.SOURCE/'reuse.json')['preserved_bytes']
    c.require(source_bytes+preserved_bytes+c.MAX_NEW_BYTES<=c.MAX_COMBINED_BYTES,
              'Immutable source plus original plus all new allowance stays within unchanged evidence cap')
    detached=c.read(c.DETACHED_ACCEPTANCE)
    c.require(detached['status']=='passed' and detached['actual_exit_code']==0
              and detached['launcher_absent'] is True and detached['child_absent'] is True
              and detached['launcher_source']==c.artifact(Path(__file__).with_name('launcher.py')),
              'Actual hidden detached launcher path qualified against its exact source')
    c.verify([detached['actual_exit']])
    from tools.future_training_rules_20261021 import verify_runtime_sources
    plan=dict(contract=c.contract(), inputs=sorted(inputs.values(),key=lambda x:x['path']),
        new_source_inputs=source_inputs, source_freeze=c.artifact(c.SOURCE/'freeze.json'),
        source_inventory=c.artifact(c.OUT/'source_inventory.jsonl'),
        corrected_runtime_sources=verify_runtime_sources(), tests=c.artifact(test_path),
        reviews=[c.artifact(path) for path in review_paths])
    c.save(c.PLAN,plan); return c.artifact(c.PLAN)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('freeze','run'))
    parser.add_argument('--freeze-sha256');parser.add_argument('--tests');parser.add_argument('--reviews',nargs='*')
    args=parser.parse_args()
    if args.operation=='freeze': print(c.canonical(freeze(args.tests,args.reviews)));return 0
    return run(args.freeze_sha256)


if __name__=='__main__': raise SystemExit(main())
