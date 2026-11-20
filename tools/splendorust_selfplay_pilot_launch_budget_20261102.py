"""Explicitly approved20-hour-stage pilot launch, preserving the stopped workflow.

Scientific modules and original ownership method bodies remain unchanged. Only
this wrapper's receipt destination and inherited outer-origin constructor differ.
The already accepted qualification is read, never relaunched or republished.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from tools.local_player_qualification_128.clock import _WindowsJob
from tools.splendorust_selfplay_pilot_budget_20261102 import common as c, workflow as w, preparation

OUT=c.ROOT/'outputs/splendorust_selfplay_pilot_launch_budget_20261102_01'
OLD_OUT=c.ROOT/'outputs/splendorust_selfplay_pilot_workflow_20261102_02'
SOURCE=Path(__file__).resolve()
TEST_SOURCE=c.ROOT/'tests/test_splendorust_selfplay_pilot_launch_budget_20261102.py'
WORKFLOW_SOURCE=c.ROOT/'tools/splendorust_selfplay_pilot_v2_20261102/workflow.py'
WORKFLOW_SOURCE_SHA='be5caa3fbbdd09d43ff57338d5535de3d81dcad418d4d6c1e21e565055b593aa'
QUALIFICATION_ACCEPTANCE_SHA='d47a4a087133ba66820d37afc4813f98429c1d6379d7b06d3266acaed77b1352'
ORIGINAL_MONOTONIC_ORIGIN=900586.578
ORIGINAL_UNIX_ORIGIN=1793644734.6388845
WORKFLOW_SECONDS=w.WORKFLOW_SECONDS
HELPER_SECONDS=w.HELPER_SECONDS
ADMISSION_SECONDS=w.ADMISSION_SECONDS
JOB_SECONDS=w.JOB_SECONDS
PACKAGE=w.PACKAGE
ACCEPTANCE_RESERVE_SECONDS=600
AUTHORIZATION='ok start the full pilot'
BUDGET_APPROVAL='Approve 20-hour stage and launch (Recommended)'
BUDGET_AUTHORIZATION=c.ROOT/'outputs/splendorust_selfplay_pilot_launch_preparation_20261102_01/authorization_20h.json'
BUDGET_AUTHORIZATION_SHA='8654c520a24bb2ef96ee10b390520263b155e143d708d50077e3d9433f3c75ff'
child_environment=w.child_environment
require_absent=w.require_absent


def save(path,value):
    path=Path(path).resolve()
    c.require(path.is_relative_to(OUT.resolve()),'Launcher writes only its separate new receipts')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(c.canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())
    return c.artifact(path)


def validate_original_clock(claim):
    c.require(claim['monotonic_origin']==ORIGINAL_MONOTONIC_ORIGIN and
              claim['unix_origin']==ORIGINAL_UNIX_ORIGIN and
              claim['maximum_seconds']==WORKFLOW_SECONDS and claim['automatic_retries']==0,
              'Exact unchanged original76-hour workflow origins')
    mono,wall=time.monotonic(),time.time()
    c.require(ORIGINAL_MONOTONIC_ORIGIN<=mono<ORIGINAL_MONOTONIC_ORIGIN+WORKFLOW_SECONDS and
              ORIGINAL_UNIX_ORIGIN<=wall<ORIGINAL_UNIX_ORIGIN+WORKFLOW_SECONDS,
              'Original outer clock valid without reset or reboot-origin reversal')


def prior_paths():
    return dict(claim=OLD_OUT/'claim.json',plan=OLD_OUT/'plan.json',failure=OLD_OUT/'failure.json',
        hold=OLD_OUT/'pilot_freeze/USER_REQUESTED_HOLD.json',hold_validation=OLD_OUT/'pilot_freeze/validation.json',
        qualification_guardian_exit=OLD_OUT/'qualification_guardian/exit.json',
        qualification_admission_exit=OLD_OUT/'qualification_admission/exit.json',
        qualification_accepted_progress=OLD_OUT/'progress/003.json')


def validate_plan(plan):
    c.require(c.SCOPE=='qualification','Launcher stays model-free qualification scope; child has isolated pilot scope')
    fields={'schema','authorization','prior_workflow','qualification_acceptance','pilot_freeze',
        'workflow_source','launcher_source','tests','reviews','workflow_seconds','pilot_seconds',
        'acceptance_reserve_seconds','automatic_retries','budget_authorization','resource_only_equivalence'}
    c.require(set(plan)==fields and plan['schema']=='selfplay_pilot_retained_budget_launch_v1',
              'Exact separately versioned pilot-only launch plan')
    c.require(plan['authorization']==AUTHORIZATION and plan['workflow_seconds']==WORKFLOW_SECONDS and
              plan['pilot_seconds']==JOB_SECONDS['pilot'] and plan['acceptance_reserve_seconds']==ACCEPTANCE_RESERVE_SECONDS and
              plan['automatic_retries']==0,'Explicit user release and unchanged bounded pilot')
    c.require(plan['budget_authorization']==c.artifact(BUDGET_AUTHORIZATION) and
              plan['budget_authorization']['sha256']==BUDGET_AUTHORIZATION_SHA,
              'Exact explicit authorization for the prospective20-hour assessment stage')
    approval=c.read(BUDGET_AUTHORIZATION)
    c.require(approval['schema']=='selfplay_pilot_budget_amendment_user_authorization_v1' and
              approval['approval']==BUDGET_APPROVAL and approval['authorized'] is True and
              approval['assessment_stage_seconds']==20*3600 and approval['total_job_seconds']==72*3600 and
              approval['model_cutoff_seconds']==60*3600 and approval['other_scientific_settings_unchanged'] is True and
              approval['same_fresh_samples'] is True and approval['prior_hold_preserved_as_history'] is True and
              approval['requires_separate_version_and_freeze'] is True,
              'Only the unlaunched assessment-stage ceiling is amended')
    c.verify([approval['proposal'],plan['resource_only_equivalence']])
    expected=prior_paths()
    c.require(set(plan['prior_workflow'])==set(expected),'Exact preserved prior workflow inputs')
    for key,path in expected.items():
        c.require(plan['prior_workflow'][key]['path']==path.relative_to(c.ROOT).as_posix(),
                  'Exact stopped workflow evidence path')
    c.require(plan['workflow_source']==c.artifact(WORKFLOW_SOURCE) and
              plan['workflow_source']['sha256']==WORKFLOW_SOURCE_SHA and
              plan['launcher_source']==c.artifact(SOURCE),'Original ownership source and separate wrapper identity')
    c.require(plan['qualification_acceptance']['path']==(c.QUALIFICATION_OUT/'qualification_acceptance.json').relative_to(c.ROOT).as_posix()
              and plan['qualification_acceptance']['sha256']==QUALIFICATION_ACCEPTANCE_SHA and
              plan['pilot_freeze']['path']==(c.PILOT_OUT/'freeze.json').relative_to(c.ROOT).as_posix(),
              'Exact accepted qualification and distinct already frozen pilot')
    c.verify([*plan['prior_workflow'].values(),plan['qualification_acceptance'],plan['pilot_freeze'],
              plan['workflow_source'],plan['launcher_source'],plan['tests'],*plan['reviews'],
              plan['budget_authorization'],plan['resource_only_equivalence']])
    prior={key:c.read(path) for key,path in expected.items()}
    claim=prior['claim'];validate_original_clock(claim)
    c.require(claim['plan']==plan['prior_workflow']['plan'],'Original workflow claim binds its original plan')
    hold=prior['hold'];validation=prior['hold_validation'];failure=prior['failure'];progress=prior['qualification_accepted_progress']
    c.require(hold['status']=='hold_installed' and hold['automatic_release'] is False and
              hold['pilot_started'] is False and hold['frozen_files_changed'] is False and
              hold['workflow_identity']=={'pid':claim['pid'],'created':claim['created']} and
              hold['workflow_source_sha256']==WORKFLOW_SOURCE_SHA and validation['passed'] is True and
              validation['workflow_source_sha256']==WORKFLOW_SOURCE_SHA and all(validation['checks'].values()),
              'Exact intentional user hold remains preserved')
    c.require(failure['status']=='stopped_preserved' and failure['automatic_retries']==0 and
              'FileExistsError' in failure['error'] and 'pilot_freeze' in failure['error'] and
              'folder.mkdir(exist_ok=False)' in failure['error'] and progress['status']=='qualification_accepted',
              'Old workflow stopped only at reserved pilot freeze after qualification acceptance')
    c.require(not (OLD_OUT/'completion.json').exists() and
              {x.name for x in (OLD_OUT/'pilot_freeze').iterdir()}=={'USER_REQUESTED_HOLD.json','validation.json'},
              'Old workflow remains stopped and no pilot helper was launched there')
    acceptance=c.read(c.ROOT/plan['qualification_acceptance']['path'])
    c.require(prior['qualification_guardian_exit']==acceptance['guardian_exit_observation'],
              'Prior actual guardian exit exactly matches accepted qualification')
    for key,operation in [('qualification_guardian_exit','guardian'),('qualification_admission_exit','admission')]:
        receipt=prior[key]
        c.require(receipt['scope']=='qualification' and receipt['operation']==operation and
                  receipt['status']=='completed' and receipt['exit_code']==0 and not receipt['cleanup_errors'] and
                  receipt['freeze_sha256']==acceptance['freeze']['sha256'],'Prior actual helper exits passed')
    c.require(prior['plan']['qualification_freeze']==acceptance['freeze'] and
              datetime.fromisoformat(acceptance['time'])<=datetime.fromisoformat(progress['time'])<=datetime.fromisoformat(failure['time']),
              'Successful qualification acceptance precedes intentional hold stop')
    preflight=preparation.accepted_new_entrypoint()
    c.require(preflight['acceptance']==plan['qualification_acceptance'] and
              preflight['resource_only_equivalence']==plan['resource_only_equivalence'] and
              preflight['budget_authorization']==plan['budget_authorization'],
              'Read-only accepted entrypoint binds approved resource-only equivalence')
    preparation.verify_resource_only_equivalence(plan['resource_only_equivalence'])
    preparation.require_review(c.ROOT/preflight['resource_only_review']['path'])
    c.verify([acceptance['completion']])
    completion=c.read(c.ROOT/acceptance['completion']['path'])
    c.require(completion['status']=='completed_verified' and completion['failure'] is None and
              completion['coverage']==acceptance['coverage'],'Preserved qualification completion and coverage')
    known=w.identities(list(prior.values()))|w.identities(acceptance['exact_owner_identities_absent'])
    require_absent([dict(pid=pid,created=created) for pid,created in sorted(known)])
    pfreeze=c.read(c.PILOT_OUT/'freeze.json')
    c.require(pfreeze['scope']==pfreeze['phase']=='pilot' and pfreeze['schema']==c.SCHEMA and
              pfreeze['new_entrypoint_preflight']==preflight and
              pfreeze['resource_only_equivalence']==plan['resource_only_equivalence'] and
              pfreeze['resource_only_review']==preflight['resource_only_review'] and
              pfreeze['budget_authorization']==plan['budget_authorization'] and
              pfreeze['job_seconds']==72*3600 and pfreeze['collection_cutoff_seconds']==60*3600 and
              pfreeze['stage_seconds']['assessment']==20*3600 and pfreeze['stage_seconds']['final']==12*3600,
              'Frozen pilot changes only approved assessment allowance, preserving total/model/final limits')
    preparation.require_test_receipt(c.ROOT/pfreeze['model_free_tests']['path'])
    c.require(len(pfreeze['implementation_reviews'])>=3,'Amended pilot keeps independent implementation reviews')
    for item in pfreeze['implementation_reviews']:preparation.require_review(c.ROOT/item['path'])
    tests=preparation.require_test_receipt(c.ROOT/plan['tests']['path'])
    c.require(plan['launcher_source'] in tests['inputs'] and c.artifact(TEST_SOURCE) in tests['inputs'],
              'Passed wrapper tests bind this exact launcher and focused test source')
    c.require(len(plan['reviews'])>=1,'Independent wrapper review required')
    for item in plan['reviews']:
        review=preparation.require_review(c.ROOT/item['path'])
        c.require(plan['launcher_source'] in review['inputs'] and plan['tests'] in review['inputs'],
                  'Independent wrapper review binds exact source and passed test receipt')
    for path in (OUT/'claim.json',OUT/'completion.json',OUT/'failure.json',c.PILOT_OUT/'guardian_claim.json',
                 c.PILOT_OUT/'launch.json',c.PILOT_OUT/'prelaunch_review.json',c.PILOT_OUT/'final_closeout.json'):
        c.require(not path.exists(),'No duplicate launcher, pilot, admission or implicit resume')
    validate_original_clock(claim)
    c.require('torch' not in sys.modules,'No model runtime in retained launcher')
    return plan


class Owner:
    def __init__(self, plan):
        self.plan=plan; self.events=0
        old=c.read(c.ROOT/plan['prior_workflow']['claim']['path'])
        validate_original_clock(old)
        self.mono=old['monotonic_origin']; self.wall=old['unix_origin']
        self.guard()
        save(OUT/'claim.json',dict(time=c.now(),pid=os.getpid(),created=psutil.Process().create_time(),
            monotonic_origin=self.mono,unix_origin=self.wall,maximum_seconds=WORKFLOW_SECONDS,
            current_monotonic_start=time.monotonic(),current_unix_start=time.time(),
            inherited_original_workflow_clock=True,prior_claim=plan['prior_workflow']['claim'],
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


def run_sequence(plan,owner):
    psha=plan['pilot_freeze']['sha256']
    owner.admit('pilot')
    owner.process('pilot','admission',['preparation','validate-launch','--freeze-sha256',psha],HELPER_SECONDS,freeze_sha=psha)
    owner.guard()
    c.require(owner.remaining()>=JOB_SECONDS['pilot']+ACCEPTANCE_RESERVE_SECONDS,
              'Original outer clock retains full72-hour pilot plus600-second acceptance reserve')
    receipt=owner.process('pilot','guardian',['qualification_watchdog','--freeze-sha256',psha],JOB_SECONDS['pilot'],freeze_sha=psha)
    owner.guard();result=w.accept_scope('pilot',psha,receipt,boundary=owner.guard);owner.guard()
    owner.event('pilot_accepted',acceptance=c.artifact(c.PILOT_OUT/'final_closeout.json'))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha256',required=True);args=parser.parse_args()
    c.require(c.file_hash(OUT/'plan.json')==args.plan_sha256,'Exact pilot-only launch plan identity')
    plan=validate_plan(c.read(OUT/'plan.json'));owner=Owner(plan)
    try:
        run_sequence(plan,owner);owner.guard()
        save(OUT/'completion.json',dict(status='completed_verified',time=c.now(),
            pilot_closeout=c.artifact(c.PILOT_OUT/'final_closeout.json'),
            prior_workflow_remains_stopped=True,original_outer_clock_preserved=True))
        return 0
    except BaseException:
        save(OUT/'failure.json',dict(status='stopped_preserved',time=c.now(),error=traceback.format_exc(),
            automatic_retries=0,prior_workflow_remains_stopped=True))
        return 1


if __name__=='__main__':raise SystemExit(main())
