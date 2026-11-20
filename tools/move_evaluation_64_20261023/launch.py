"""Windowless owning watchdog; clocks include startup and never reset."""
import argparse
import os
import subprocess
import sys
import time
import traceback

import psutil

from . import common as c
from tools.local_player_qualification_128.clock import _WindowsJob

JOB_SECONDS = 28800


def ensure_unlaunched():
    c.require(not any((c.OUT/name).exists() for name in
        ('launch.json','watchdog.json','watchdog_claim.json','execution.json','runner_stdout.log','runner_stderr.log')),
        'No automatic relaunch or clock reset')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    started, wall_started = time.monotonic(), time.time()
    ensure_unlaunched()
    c.save(c.OUT/'watchdog_claim.json', dict(time=c.now(),monotonic_origin=started,unix_origin=wall_started,
        freeze_sha256=args.freeze_sha256,watchdog_pid=os.getpid(),watchdog_created=psutil.Process().create_time()))
    stdout = stderr = process = job = None
    status = 'failed'
    try:
        c.require(c.file_hash(c.OUT/'freeze.json') == args.freeze_sha256, 'Execution freeze changed')
        freeze = c.read(c.OUT/'freeze.json')
        c.require(freeze['job_seconds'] == JOB_SECONDS and freeze['pilot_seconds'] == 3600 and
                  freeze['qualification_seconds'] == 10800 and freeze['collection_cutoff_seconds'] == 21600 and
                  freeze['automatic_retries'] == 0, 'Fixed resource windows changed')
        stdout = (c.OUT/'runner_stdout.log').open('xb')
        stderr = (c.OUT/'runner_stderr.log').open('xb')
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT),PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        process = subprocess.Popen([sys.executable,'-B','-u','-m','tools.move_evaluation_64_20261023.runner',
            'run','--freeze-sha256',args.freeze_sha256],cwd=c.ROOT,env=environment,stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        job = _WindowsJob(process.pid)
        c.save(c.OUT/'watchdog.json', dict(time=c.now(),watchdog_pid=os.getpid(),
            watchdog_created=psutil.Process().create_time(),runner_pid=process.pid,
            runner_created=psutil.Process(process.pid).create_time(),freeze_sha256=args.freeze_sha256,
            scope='owned_runner_and_descendants',job_seconds=JOB_SECONDS,
            monotonic_origin=started,unix_origin=wall_started))
        while process.poll() is None:
            if time.monotonic() >= started+JOB_SECONDS or time.time() >= wall_started+JOB_SECONDS:
                status = 'watchdog_deadline'; break
            time.sleep(.1)
        if process.poll() is not None:
            status = 'completed' if process.returncode == 0 else 'runner_failed'
    except BaseException:
        c.save(c.OUT/'watchdog_error.json',dict(time=c.now(),error=traceback.format_exc()))
        raise
    finally:
        errors = []
        if job is not None:
            try: job.close()
            except BaseException as error: errors.append('Job closure: '+type(error).__name__)
        if process is not None:
            try:
                if process.poll() is None: process.kill()
                process.wait(timeout=10)
            except BaseException as error: errors.append('Runner closure: '+type(error).__name__)
        for stream in (stdout,stderr):
            if stream is not None:
                try: stream.close()
                except BaseException as error: errors.append('Log closure: '+type(error).__name__)
        mono_end, wall_end = time.monotonic(),time.time()
        charged = max(mono_end-started,wall_end-wall_started)
        if charged >= JOB_SECONDS: status = 'watchdog_deadline'
        c.save(c.OUT/'execution.json',dict(status='cleanup_failed' if errors else status,
            exit_code=None if process is None else process.returncode,time=c.now(),freeze_sha256=args.freeze_sha256,
            monotonic_origin=started,unix_origin=wall_started,monotonic_end=mono_end,unix_end=wall_end,
            outer_elapsed_seconds=mono_end-started,wall_elapsed_seconds=wall_end-wall_started,
            charged_job_seconds=charged,cleanup_errors=errors,
            cleanup='Windows job closed; owned descendants terminated' if not errors else 'Incomplete cleanup'))


if __name__ == '__main__':
    main()
