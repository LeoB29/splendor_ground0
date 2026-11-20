"""One owned, bounded qualification -> fixed pilot transition; no retries."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from . import common as c
from tools.local_player_qualification_128.clock import _WindowsJob

OUT = c.ROOT / 'outputs/splendorust_selfplay_pilot_workflow_20261102_02'
WORKFLOW_SECONDS = 76 * 3600
HELPER_SECONDS = 600
ADMISSION_SECONDS = 30 * 60
JOB_SECONDS = {'qualification': 150 * 60, 'pilot': 72 * 3600}
PACKAGE = 'tools.splendorust_policy_head_20261104'


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(OUT.resolve()) or
              path in (c.QUALIFICATION_OUT/'qualification_acceptance.json', c.PILOT_OUT/'final_closeout.json'),
              'Workflow writes only new receipts in its fixed destinations')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(c.canonical(value)+'\n'); stream.flush(); os.fsync(stream.fileno())
    return c.artifact(path)


def scope_out(scope):
    c.require(scope in JOB_SECONDS, 'Exact two workflow scopes')
    return c.QUALIFICATION_OUT if scope == 'qualification' else c.PILOT_OUT


def child_environment(scope):
    scope_out(scope)
    env = os.environ.copy()
    env.update(SELFPLAY_JOB_SCOPE=scope, PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
        SPLENDOR_SEARCH256_STRENGTH_PHASE='strength', SPLENDOR_SEARCH512_STRENGTH_PHASE='strength')
    return env


def identities(value):
    """Only exact recorded process identities, never names or bare PID guesses."""
    result = set()
    if isinstance(value, dict):
        if type(value.get('pid')) is int and type(value.get('created')) in (int, float):
            result.add((value['pid'], value['created']))
        for prefix in ('guardian', 'watchdog', 'runner'):
            if type(value.get(prefix+'_pid')) is int and type(value.get(prefix+'_created')) in (int, float):
                result.add((value[prefix+'_pid'], value[prefix+'_created']))
        for child in value.values(): result.update(identities(child))
    elif isinstance(value, list):
        for child in value: result.update(identities(child))
    return result


def require_absent(items):
    for item in items:
        try: process = psutil.Process(item['pid'])
        except psutil.NoSuchProcess: continue
        c.require(process.create_time() != item['created'], 'Exact previous owned process still alive')


def accept_scope(scope, expected_sha, guardian_exit, *, boundary=lambda:None):
    boundary()
    out = scope_out(scope)
    c.require(guardian_exit['exit_code'] == 0 and guardian_exit['status'] == 'completed'
              and not guardian_exit['cleanup_errors'] and guardian_exit['scope'] == scope
              and guardian_exit['operation'] == 'guardian'
              and guardian_exit['freeze_sha256'] == expected_sha,
              'Actual owning guardian handle must exit successfully')
    c.require(c.file_hash(out/'freeze.json') == expected_sha, 'Accepted scope freeze unchanged')
    freeze = c.read(out/'freeze.json')
    c.verify_failed_attempt_preservation(freeze['failed_attempt_preservation'])
    boundary()
    for item in freeze['inputs']: boundary(); c.verify([item])
    review = c.read(out/'independent_review.json'); completion = c.read(out/'completion.json')
    c.require(freeze['scope'] == scope and freeze['phase'] == scope
              and review['approved'] is True and review['freeze_sha256'] == expected_sha
              and completion['status'] == 'completed_verified' and completion['failure'] is None
              and completion['freeze_sha256'] == expected_sha
              and review['coverage'] == completion['coverage'], 'Complete independently reconciled scope')
    coverage = review['coverage']
    c.require(coverage['completed_games'] == (4 if scope == 'qualification' else 256)
              and coverage['qualification_only'] is (scope == 'qualification')
              and coverage['pilot_data'] is (scope == 'pilot'), 'No qualification/pilot sample substitution')
    records = [guardian_exit, c.read(out/'guardian_claim.json'), c.read(out/'guardian_owner.json'),
               c.read(out/'launch.json'), c.read(out/'fit_execution.json')]
    for name in ('execution.json','watchdog_execution.json','independent_review_os_exit.json'):
        boundary()
        item = c.read(out/name)
        c.require(item['status'] == 'completed' and item['exit_code'] == 0
                  and item['freeze_sha256'] == expected_sha and not item['cleanup_errors']
                  and not item['cleanup_receipt']['unconfirmed'], 'All actual owner/reader exits passed')
        records.append(item)
    c.require(records[-1]['reviewer'] == review['reviewer'], 'Accepted reader exact identity')
    for worker_root in (out/'workers', out/'assessment/workers'):
        for folder in sorted(worker_root.iterdir()):
            boundary()
            if not folder.is_dir(): continue
            for name in ('initialization.json','cleanup.json'):
                path=folder/name
                c.require(path.is_file(), 'Every declared lifecycle has startup and cleanup receipts')
                records.append(c.read(path))
    known = sorted(set().union(*(identities(x) for x in records)))
    absent = [dict(pid=pid, created=created) for pid,created in known]
    boundary()
    require_absent(absent)
    operational = sorted((out/'assessment/games/operational').glob('g*/result.json'))
    main = sorted((out/'assessment/games/main').glob('g*/result.json'))
    c.require(len(operational) == 6 and len(main) == (0 if scope == 'qualification' else 400),
              'Complete fixed operational and independent assessment inventory')
    value = dict(status='accepted_completed_verified', scope=scope, time=c.now(),
        freeze=c.artifact(out/'freeze.json'), independent_review=c.artifact(out/'independent_review.json'),
        completion=c.artifact(out/'completion.json'), guardian_exit_observation=guardian_exit,
        exact_owner_identities_absent=absent, completed_source_games=coverage['completed_games'],
        assessment_operational_games=6, independent_assessment_games=len(main),
        no_pilot_data=scope == 'qualification', coverage=coverage, reference_promoted=False,
        no_retries=True, no_replaced_games=True)
    path = out/('qualification_acceptance.json' if scope == 'qualification' else 'final_closeout.json')
    boundary(); save(path, value); boundary()
    return value


class Owner:
    def __init__(self, plan):
        self.plan=plan; self.mono=time.monotonic(); self.wall=time.time(); self.events=0
        save(OUT/'claim.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
            monotonic_origin=self.mono,unix_origin=self.wall,maximum_seconds=WORKFLOW_SECONDS,
            plan=c.artifact(OUT/'plan.json'),automatic_retries=0))

    def guard(self):
        c.require(time.monotonic()<self.mono+WORKFLOW_SECONDS and time.time()<self.wall+WORKFLOW_SECONDS,
                  'Original complete workflow clock')
        c.require(psutil.Process().memory_info().rss < 512*c.MiB, 'Model-free workflow owner memory bound')

    def remaining(self):
        return min(self.mono+WORKFLOW_SECONDS-time.monotonic(), self.wall+WORKFLOW_SECONDS-time.time())

    def event(self, status, **value):
        self.events+=1
        save(OUT/'progress'/f'{self.events:03d}.json',dict(time=c.now(),status=status,**value))

    def admit(self, scope):
        self.guard(); started=time.monotonic(); wall=time.time(); stable=None; minimum=None
        reserve=(9 if scope=='qualification' else 53)*c.GiB
        while True:
            self.guard(); current=time.monotonic(); available=psutil.virtual_memory().available
            c.require(current-started < ADMISSION_SECONDS and time.time()-wall < ADMISSION_SECONDS,
                      'Bounded workflow initial memory admission expired')
            c.require(shutil.disk_usage(c.ROOT).free >= reserve, 'Frozen initial free disk requirement')
            minimum=available if minimum is None else min(minimum,available)
            if available < 4*c.GiB: stable=None
            elif stable is None: stable=current
            if stable is not None and current-stable >= 60: break
            time.sleep(.5)
        save(OUT/f'{scope}_resource_admission.json',dict(passed=True,time=c.now(),
            initial_available_ram_bytes=4*c.GiB,stable_seconds=60,minimum_available_bytes=minimum,
            elapsed_seconds=time.monotonic()-started,within_experimental_clock=False,
            owner_rechecks_resources_inside_original_job=True))

    def process(self, scope, operation, arguments, maximum, *, freeze_sha=None):
        self.guard(); c.require(maximum<=JOB_SECONDS[scope] if operation=='guardian' else maximum<=HELPER_SECONDS,
                                'Exact prospective operation limit')
        begun,wall=time.monotonic(),time.time(); process=job=None; identity=None; errors=[]; failure=None
        status='failed'; stdout=stderr=None
        folder=OUT/(scope+'_'+operation);folder.mkdir(exist_ok=False)
        try:
            stdout=(folder/'stdout.log').open('xb');stderr=(folder/'stderr.log').open('xb')
            process=subprocess.Popen([sys.executable,'-B','-u','-m',PACKAGE+'.'+arguments[0],*arguments[1:]],
                cwd=c.ROOT,env=child_environment(scope),stdout=stdout,stderr=stderr,
                creationflags=getattr(subprocess,'DETACHED_PROCESS',0))
            identity=dict(pid=process.pid,created=psutil.Process(process.pid).create_time())
            job=_WindowsJob(process.pid)
            save(folder/'owner.json',dict(identity=identity,scope=scope,operation=operation,
                monotonic_origin=begun,unix_origin=wall,maximum_seconds=maximum,freeze_sha256=freeze_sha))
            self.event('running',scope=scope,operation=operation,identity=identity)
            while process.poll() is None:
                self.guard()
                c.require(time.monotonic()-begun < maximum and time.time()-wall < maximum,
                          'Owned operation original real-clock deadline')
                c.require(sum(p.stat().st_size for p in folder.glob('*.log'))<4*c.MiB,'Bounded workflow logs')
                time.sleep(.25)
            c.require(process.returncode==0,'Owned operation failed; preserve and stop without retry')
            self.guard(); status='completed'
        except BaseException: failure=traceback.format_exc()
        finally:
            observed=[]
            if process is not None:
                try:
                    root=psutil.Process(process.pid)
                    c.require(root.create_time()==identity['created'],'Owned PID identity before cleanup')
                    observed=[(p,dict(pid=p.pid,created=p.create_time())) for p in [root]+root.children(recursive=True)]
                except psutil.NoSuchProcess: pass
                except BaseException as error: errors.append(repr(error))
            if job is not None:
                try: job.close()
                except BaseException as error: errors.append(repr(error))
            cleanup_end=time.monotonic()+min(5.,max(0.,self.remaining()))
            for child,ident in reversed(observed):
                try:
                    if child.create_time()==ident['created']: child.kill()
                except psutil.NoSuchProcess: pass
                except BaseException as error: errors.append(repr(error))
            if process is not None:
                try: process.wait(timeout=max(0.,cleanup_end-time.monotonic()))
                except BaseException as error: errors.append(repr(error))
            for stream in (stdout,stderr):
                if stream is not None: stream.close()
            try: require_absent([ident for _,ident in observed])
            except BaseException as error: errors.append(repr(error))
            if errors: status='failed'
            receipt=dict(status=status,scope=scope,operation=operation,freeze_sha256=freeze_sha,
                identity=identity,exit_code=None if process is None else process.returncode,
                observed_by='retained workflow Popen handle',cleanup_errors=errors,error=failure,
                monotonic_start=begun,unix_start=wall,monotonic_end=time.monotonic(),unix_end=time.time())
            save(folder/'exit.json',receipt)
        c.require(status=='completed','Automatic workflow stopped; see preserved '+str(folder/'exit.json'))
        return receipt


def run_sequence(plan, owner):
    qsha=plan['qualification_freeze']['sha256']
    owner.admit('qualification')
    owner.process('qualification','admission',['preparation','validate-launch','--freeze-sha256',qsha],HELPER_SECONDS,freeze_sha=qsha)
    receipt=owner.process('qualification','guardian',['qualification_watchdog','--freeze-sha256',qsha],JOB_SECONDS['qualification'],freeze_sha=qsha)
    owner.guard(); accept_scope('qualification',qsha,receipt,boundary=owner.guard); owner.guard()
    owner.event('qualification_accepted')
    owner.process('pilot','freeze',['preparation','freeze','--tests',str(c.ROOT/plan['tests']['path']),
        '--reviews',*[str(c.ROOT/x['path']) for x in plan['reviews']]],HELPER_SECONDS)
    psha=c.file_hash(c.PILOT_OUT/'freeze.json')
    owner.admit('pilot')
    owner.process('pilot','admission',['preparation','validate-launch','--freeze-sha256',psha],HELPER_SECONDS,freeze_sha=psha)
    receipt=owner.process('pilot','guardian',['qualification_watchdog','--freeze-sha256',psha],JOB_SECONDS['pilot'],freeze_sha=psha)
    owner.guard(); result=accept_scope('pilot',psha,receipt,boundary=owner.guard); owner.guard()
    owner.event('pilot_accepted',acceptance=c.artifact(c.PILOT_OUT/'final_closeout.json'))
    return result


def validate_plan(plan):
    c.require(set(plan)=={'schema','qualification_freeze','tests','reviews','workflow_seconds','automatic_retries'},
              'Exact workflow plan fields')
    c.require(plan['schema']=='selfplay_pilot_automatic_workflow_v1' and plan['workflow_seconds']==WORKFLOW_SECONDS
              and plan['automatic_retries']==0 and len(plan['reviews'])>=3,'Fixed conditional workflow')
    c.require(plan['qualification_freeze']['path']==(c.QUALIFICATION_OUT/'freeze.json').relative_to(c.ROOT).as_posix(),
              'Exact intended fresh qualification output')
    c.verify([plan['qualification_freeze'],plan['tests'],*plan['reviews']])
    freeze=c.read(c.QUALIFICATION_OUT/'freeze.json')
    c.require(freeze['phase']=='qualification' and freeze['schema']==c.SCHEMA
              and freeze['model_free_tests']==plan['tests'] and freeze['implementation_reviews']==plan['reviews'],
              'Exact reviewed source/test admission travels into pilot')
    c.verify(freeze['inputs'])
    for path in (OUT/'claim.json',c.QUALIFICATION_OUT/'guardian_claim.json',c.PILOT_OUT/'freeze.json',c.PILOT_OUT/'guardian_claim.json'):
        c.require(not path.exists(),'No duplicate workflow or implicit resume')
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    c.require(c.SCOPE=='qualification','Workflow process scope stays qualification; children have isolated scopes')
    c.require(c.file_hash(OUT/'plan.json')==args.plan_sha256,'Exact workflow plan identity')
    plan=validate_plan(c.read(OUT/'plan.json'));owner=Owner(plan)
    try:
        run_sequence(plan,owner)
        owner.guard()
        save(OUT/'completion.json',dict(status='completed_verified',time=c.now(),pilot_closeout=c.artifact(c.PILOT_OUT/'final_closeout.json')))
        return 0
    except BaseException:
        save(OUT/'failure.json',dict(status='stopped_preserved',time=c.now(),error=traceback.format_exc(),automatic_retries=0))
        return 1


if __name__=='__main__': raise SystemExit(main())
