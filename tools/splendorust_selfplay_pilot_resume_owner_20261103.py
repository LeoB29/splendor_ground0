"""Separately owned clean-prefix continuation; estimates never admit or stop work."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from tools.local_player_qualification_128.clock import _WindowsJob
from tools.splendorust_selfplay_pilot_resume_20261103 import common as c,workflow as w,preparation,continuation
from tools.splendorust_selfplay_pilot_resume_20261103.qualification_watchdog import inherited_origins

OUT=c.ROOT/'outputs/splendorust_selfplay_pilot_resume_owner_20261103_01'
SOURCE=Path(__file__).resolve()
TEST_SOURCE=c.ROOT/'tests/test_splendorust_selfplay_pilot_resume_owner_20261103.py'
AUTHORIZATION=c.PILOT_OUT/'prelaunch/user_authorization.json'
WORKFLOW_SECONDS=76*3600
OUTER_MONO=900586.578
OUTER_WALL=1793644734.6388845
PILOT_MONO=904900.656
PILOT_WALL=1793649048.764625
HELPER_SECONDS=w.HELPER_SECONDS
ADMISSION_SECONDS=w.ADMISSION_SECONDS
JOB_SECONDS=w.JOB_SECONDS
PACKAGE='tools.splendorust_selfplay_pilot_resume_20261103'
child_environment=w.child_environment
require_absent=w.require_absent


def save(path,value):
    path=Path(path).resolve()
    c.require(path.is_relative_to(OUT.resolve()),'Retained continuation owner writes only separate receipts')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(c.canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    return c.artifact(path)


def original_clocks_open(*,model_required):
    c.require(c.SCOPE=='pilot' and c.JOB_SECONDS==72*3600 and c.COLLECTION_CUTOFF_SECONDS==60*3600,
              'Continuation owner retains original pilot scope and limits')
    now,wall=time.monotonic(),time.time()
    cap=c.COLLECTION_CUTOFF_SECONDS if model_required else c.JOB_SECONDS
    c.require(OUTER_MONO<=now<OUTER_MONO+WORKFLOW_SECONDS and OUTER_WALL<=wall<OUTER_WALL+WORKFLOW_SECONDS and
              PILOT_MONO<=now<PILOT_MONO+cap and PILOT_WALL<=wall<PILOT_WALL+cap,
              'Actual original outer and pilot clocks remain open without reset')


def validate_outer_clock(claim):
    c.require(claim['monotonic_origin']==OUTER_MONO and claim['unix_origin']==OUTER_WALL and
              claim['maximum_seconds']==WORKFLOW_SECONDS and claim['automatic_retries']==0,
              'Exact inherited76-hour outer origins')
    original_clocks_open(model_required=False)


def validate_plan(plan):
    c.require(set(plan)=={'schema','authorization','continuation','prefix_review','pilot_freeze','owner_source','tests','reviews','automatic_retries'} and
              plan['schema']=='selfplay_pilot_retained_resume_v1' and plan['automatic_retries']==0,
              'Exact separately owned no-retry continuation plan')
    original_clocks_open(model_required=True)
    c.require(plan['authorization']==c.artifact(AUTHORIZATION) and plan['owner_source']==c.artifact(SOURCE) and
              plan['pilot_freeze']==c.artifact(c.PILOT_OUT/'freeze.json') and
              plan['continuation']==c.artifact(continuation.MANIFEST) and plan['prefix_review']==c.artifact(continuation.PREFIX_REVIEW),
              'Exact explicit authorization, owner and frozen continuation inputs')
    c.verify([plan['authorization'],plan['continuation'],plan['prefix_review'],plan['pilot_freeze'],plan['owner_source'],plan['tests'],*plan['reviews']])
    authorization=c.read(AUTHORIZATION)
    c.require(authorization['schema']=='selfplay_pilot_resume_authorization_v1' and authorization['resume_authorized'] is True and
              authorization['resource_forecasts_advisory'] is True and authorization['science_unchanged'] is True and
              authorization['actual_original_clocks_retained'] is True and authorization['failed_decision_retries']==0 and
              authorization['original_completed_games_preserved']==32,'Explicit advisory-forecast clean-prefix continuation approval')
    boundary=lambda:original_clocks_open(model_required=True)
    manifest=continuation.verify_continuation(plan['continuation'],boundary=boundary)
    continuation.verify_prefix_review(plan['prefix_review'],boundary=boundary)
    outer_path=continuation.OLD_LAUNCH/'claim.json';outer=c.read(outer_path)
    c.require(c.artifact(outer_path) in manifest['inputs'] and outer['plan']==manifest['original_launch_plan'],
              'Entire original launcher and its original plan remain authenticated')
    c.verify([outer['plan']]);validate_outer_clock(outer)
    freeze=c.read(c.PILOT_OUT/'freeze.json')
    c.require(freeze['phase']==freeze['scope']=='pilot' and freeze['continuation']==plan['continuation'] and
              freeze['saved_prefix_review']==plan['prefix_review'] and freeze['source_continuation']==manifest['source_continuation'] and
              freeze['job_seconds']==72*3600 and freeze['collection_cutoff_seconds']==60*3600 and
              freeze['resource_forecast_policy']=='advisory_only' and freeze['resource_forecasts_may_stop'] is False,
              'Frozen same-sample continuation and actual limits with advisory forecasts')
    c.require(inherited_origins(freeze)==(PILOT_MONO,PILOT_WALL),'Guardian shares original pilot origins')
    require_absent(manifest['exact_owner_identities_absent'])
    tests=preparation.require_test_receipt(c.ROOT/plan['tests']['path'])
    c.require(plan['owner_source'] in tests['inputs'] and c.artifact(TEST_SOURCE) in tests['inputs'],
              'Passed retained owner tests bind exact source')
    c.require(len(plan['reviews'])>=1,'Independent retained owner review')
    for item in plan['reviews']:
        review=preparation.require_review(c.ROOT/item['path'])
        c.require(plan['owner_source'] in review['inputs'] and plan['tests'] in review['inputs'],
                  'Independent owner review binds tested source')
    for path in (OUT/'claim.json',OUT/'failure.json',OUT/'completion.json',c.PILOT_OUT/'guardian_claim.json',
                 c.PILOT_OUT/'launch.json',c.PILOT_OUT/'prelaunch_review.json',c.PILOT_OUT/'final_closeout.json'):
        c.require(not path.exists(),'No duplicate continuation owner, admission, or implicit retry')
    c.require('torch' not in sys.modules,'Retained owner is model-free')
    boundary();return plan


class Owner:
    def __init__(self,plan):
        self.plan=plan;self.events=0
        outer=c.read(continuation.OLD_LAUNCH/'claim.json')
        validate_outer_clock(outer)
        self.mono=outer['monotonic_origin'];self.wall=outer['unix_origin']
        self.guard()
        save(OUT/'claim.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
            monotonic_origin=self.mono,unix_origin=self.wall,maximum_seconds=WORKFLOW_SECONDS,
            original_pilot_monotonic_origin=PILOT_MONO,original_pilot_unix_origin=PILOT_WALL,
            current_monotonic_start=time.monotonic(),current_unix_start=time.time(),
            plan=c.artifact(OUT/'plan.json'),prior_claim=c.artifact(continuation.OLD_LAUNCH/'claim.json'),
            automatic_retries=0,inherited_original_clocks=True))

    def guard(self):
        original_clocks_open(model_required=False)
        c.require(psutil.Process().memory_info().rss<512*c.MiB,'Model-free retained owner memory bound')

    def remaining(self):
        return min(OUTER_MONO+WORKFLOW_SECONDS-time.monotonic(),OUTER_WALL+WORKFLOW_SECONDS-time.time(),
                   PILOT_MONO+c.JOB_SECONDS-time.monotonic(),PILOT_WALL+c.JOB_SECONDS-time.time())

    def event(self, status, **value):
        self.events+=1
        save(OUT/'progress'/f'{self.events:03d}.json',dict(time=c.now(),status=status,**value))

    def admit(self,scope):
        self.guard();c.require(scope=='pilot','Only explicit same-pilot continuation')
        available=psutil.virtual_memory().available;free=shutil.disk_usage(c.ROOT).free
        c.require(free>=c.MINIMUM_FREE_DISK_BYTES,'Actual maintained free-disk floor')
        save(OUT/'pilot_resource_admission.json',dict(passed=True,time=c.now(),
            observed_available_ram_bytes=available,observed_free_disk_bytes=free,
            ram_reserve_policy='advisory_only',ram_wait_seconds=0.,stable_seconds=0.,
            initial_available_ram_bytes=0,actual_rss_caps_unchanged=True,
            actual_disk_floor_bytes=c.MINIMUM_FREE_DISK_BYTES,original_clocks_preserved=True))

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


def run_sequence(plan,owner):
    sha=plan['pilot_freeze']['sha256']
    owner.admit('pilot')
    owner.process('pilot','admission',['preparation','validate-launch','--freeze-sha256',sha],HELPER_SECONDS,freeze_sha=sha)
    owner.guard();original_clocks_open(model_required=True)
    receipt=owner.process('pilot','guardian',['qualification_watchdog','--freeze-sha256',sha],JOB_SECONDS['pilot'],freeze_sha=sha)
    owner.guard();result=w.accept_scope('pilot',sha,receipt,boundary=owner.guard);owner.guard()
    owner.event('pilot_accepted',acceptance=c.artifact(c.PILOT_OUT/'final_closeout.json'))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    c.require(c.file_hash(OUT/'plan.json')==args.plan_sha256,'Exact separately frozen retained resume plan')
    plan=validate_plan(c.read(OUT/'plan.json'));owner=Owner(plan)
    try:
        run_sequence(plan,owner);owner.guard()
        save(OUT/'completion.json',dict(status='completed_verified',time=c.now(),
            pilot_closeout=c.artifact(c.PILOT_OUT/'final_closeout.json'),original_attempt_remains_incomplete=True,
            original_clocks_preserved=True,forecast_policy='advisory_only'))
        return 0
    except BaseException:
        save(OUT/'failure.json',dict(status='stopped_preserved',time=c.now(),error=traceback.format_exc(),
            automatic_retries=0,original_attempt_remains_incomplete=True))
        return 1


if __name__=='__main__':raise SystemExit(main())
