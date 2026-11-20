"""Windowless outer watchdog; kills the owned runner tree at the one-hour cap."""
import argparse
import os
import subprocess
import sys
import time
import traceback

import psutil

from . import common as c
from tools.local_player_qualification_128.clock import _WindowsJob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    c.require(c.file_hash(c.OUT/'freeze.json') == args.freeze_sha256, 'Frozen source mismatch')
    c.require(not (c.OUT/'launch.json').exists(), 'No automatic relaunch')
    started = time.monotonic()
    wall_started = time.time()
    stdout = (c.OUT/'runner_stdout.log').open('xb')
    stderr = (c.OUT/'runner_stderr.log').open('xb')
    process, job, status = None, None, 'failed'
    try:
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT), PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                           DIAGNOSTIC_START_MONOTONIC=str(started), DIAGNOSTIC_START_UNIX=str(wall_started))
        process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.move_evaluation_profile_20261023.runner',
            'run', '--freeze-sha256', args.freeze_sha256], cwd=c.ROOT, env=environment, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        job = _WindowsJob(process.pid)
        c.save(c.OUT/'watchdog.json', {'time': c.now(), 'watchdog_pid': os.getpid(),
            'watchdog_created': psutil.Process().create_time(), 'runner_pid': process.pid,
            'runner_created': psutil.Process(process.pid).create_time(), 'freeze_sha256': args.freeze_sha256,
            'scope': 'owned_runner_and_descendants', 'job_seconds': 3600,
            'monotonic_origin': started, 'unix_origin': wall_started})
        # Includes startup/source admission, even if launch.json never appears.
        deadline = started+3600
        while process.poll() is None:
            if time.monotonic() >= deadline or time.time() >= wall_started+3600:
                status = 'watchdog_deadline'
                break
            time.sleep(.1)
        if process.poll() is not None:
            status = 'completed' if process.returncode == 0 else 'runner_failed'
    except BaseException:
        c.save(c.OUT/'watchdog_error.json', {'time': c.now(), 'error': traceback.format_exc()})
        raise
    finally:
        if job is not None:
            job.close()
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        stdout.close(); stderr.close()
        c.save(c.OUT/'execution.json', {'status': status, 'exit_code': None if process is None else process.returncode,
               'time': c.now(), 'outer_elapsed_seconds': time.monotonic()-started,
               'cleanup': 'Windows job closed; owned descendants terminated'})


if __name__ == '__main__':
    main()
