"""Owned Stage B watchdog carrying forward closed Stage A budgets."""
import argparse
import os
import subprocess
import sys
import time
import traceback

import psutil

from . import common as c
from .runner import read_study_plan, study_guard, load_prior, STUDY_SECONDS, OWNED_SECONDS
from tools.local_player_qualification_128.clock import _WindowsJob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    started, wall_started = time.monotonic(), time.time()
    c.require(not any((c.OUT/name).exists() for name in
        ('launch.json','watchdog.json','watchdog_claim.json','execution.json','runner_stdout.log','runner_stderr.log')),
        'No automatic relaunch or clock reset')
    c.save(c.OUT/'watchdog_claim.json',{'time':c.now(),'monotonic_origin':started,'unix_origin':wall_started,
        'freeze_sha256':args.freeze_sha256,'watchdog_pid':os.getpid(),'watchdog_created':psutil.Process().create_time()})
    stdout = stderr = process = job = plan = prior = None
    allowance = c.JOB_SECONDS
    status = 'failed'
    try:
        c.require(c.file_hash(c.OUT/'freeze.json') == args.freeze_sha256, 'Frozen source mismatch')
        freeze = c.read(c.OUT/'freeze.json')
        plan = read_study_plan(freeze)
        study_guard(plan, reserve=c.FINAL_RESERVE)
        prior = load_prior()
        c.require(prior == freeze['prior_stage_a'], 'Closed Stage A budget changed')
        allowance = min(c.JOB_SECONDS,OWNED_SECONDS-prior['owned_seconds'])
        c.require(allowance > c.FINAL_RESERVE and c.JOB_SECONDS == 3600 and freeze['job_seconds'] == 3600 and
                  freeze['final_reserve_seconds'] == 300, 'Stage B fixed clock limits')
        stdout = (c.OUT/'runner_stdout.log').open('xb')
        stderr = (c.OUT/'runner_stderr.log').open('xb')
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT), PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
            DIAGNOSTIC_START_MONOTONIC=str(started), DIAGNOSTIC_START_UNIX=str(wall_started))
        study_guard(plan,reserve=c.FINAL_RESERVE)
        c.require(time.monotonic() < started+allowance-c.FINAL_RESERVE and
                  time.time() < wall_started+allowance-c.FINAL_RESERVE, 'Owned startup consumed work allowance')
        process = subprocess.Popen([sys.executable,'-B','-u','-m','tools.move_evaluation_inference_opt_20261023.runner',
            'run','--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=environment,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        job = _WindowsJob(process.pid)
        c.save(c.OUT/'watchdog.json',{'time':c.now(),'watchdog_pid':os.getpid(),
            'watchdog_created':psutil.Process().create_time(),'runner_pid':process.pid,
            'runner_created':psutil.Process(process.pid).create_time(),'freeze_sha256':args.freeze_sha256,
            'study_plan':freeze['study_plan'],'scope':'owned_runner_and_descendants','job_seconds':3600,'owned_job_allowance_seconds':allowance,'prior_stage_a':prior,
            'monotonic_origin':started,'unix_origin':wall_started,
            'study_monotonic_deadline':plan['monotonic_origin']+STUDY_SECONDS,
            'study_unix_deadline':plan['unix_origin']+STUDY_SECONDS})
        while process.poll() is None:
            if time.monotonic() >= started+allowance or time.time() >= wall_started+allowance:
                status = 'watchdog_deadline'
                break
            if time.monotonic() >= plan['monotonic_origin']+STUDY_SECONDS or time.time() >= plan['unix_origin']+STUDY_SECONDS:
                status = 'watchdog_study_deadline'
                break
            time.sleep(.1)
        if process.poll() is not None:
            status = 'completed' if process.returncode == 0 else 'runner_failed'
    except BaseException:
        c.save(c.OUT/'watchdog_error.json',{'time':c.now(),'error':traceback.format_exc()})
        raise
    finally:
        cleanup_errors = []
        if job is not None:
            try:
                job.close()
            except BaseException as error:
                cleanup_errors.append('Job closure: '+type(error).__name__)
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
            except BaseException as error:
                cleanup_errors.append('Owned runner closure: '+type(error).__name__)
        for stream in (stdout,stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as error:
                    cleanup_errors.append('Log closure: '+type(error).__name__)
        mono_end, wall_end = time.monotonic(), time.time()
        charged = max(mono_end-started,wall_end-wall_started)
        study_elapsed = None if plan is None else max(mono_end-plan['monotonic_origin'],wall_end-plan['unix_origin'])
        if charged >= allowance:
            status = 'watchdog_deadline'
        if study_elapsed is not None and study_elapsed >= STUDY_SECONDS:
            status = 'watchdog_study_deadline'
        c.save(c.OUT/'execution.json',{'status':'cleanup_failed' if cleanup_errors else status,
            'exit_code':None if process is None else process.returncode,'time':c.now(),
            'freeze_sha256':args.freeze_sha256,
            'study_plan':None if plan is None else freeze['study_plan'],
            'monotonic_origin':started,'unix_origin':wall_started,'monotonic_end':mono_end,'unix_end':wall_end,
            'outer_elapsed_seconds':mono_end-started,'wall_elapsed_seconds':wall_end-wall_started,
            'charged_owned_job_seconds':charged,'study_elapsed_seconds':study_elapsed,
            'prior_stage_a':prior,
            'cumulative_owned_job_seconds':None if prior is None else prior['owned_seconds']+charged,
            'remaining_owned_job_seconds':None if prior is None else max(0.,OWNED_SECONDS-prior['owned_seconds']-charged),
            'cleanup_errors':cleanup_errors,
            'cleanup':'Windows job closed; owned descendants terminated' if not cleanup_errors else 'Incomplete; inspect cleanup_errors'})


if __name__ == '__main__':
    main()
