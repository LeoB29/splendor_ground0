"""Job-isolated administrative supervisor; no automatic retry or game logic."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/matched_strength_recovery_20261022_01'
SOURCE = ROOT / 'outputs/matched_strength_20261020_01'
PYTHON = Path('C:/Users/leoba/anaconda3/python.exe').resolve()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def in_any_job(pid=None):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x0400, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        result = wintypes.BOOL()
        if not kernel.IsProcessInJob(handle, None, ctypes.byref(result)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(result.value)
    finally:
        if pid is not None:
            kernel.CloseHandle(handle)


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def identity(process):
    return {'pid': process.pid, 'created': process.create_time(), 'executable': process.exe(),
            'command': process.cmdline(), 'affinity': process.cpu_affinity(),
            'in_any_windows_job': in_any_job(process.pid)}


def probe(path):
    assert Path(sys.executable).resolve() == PYTHON
    owner = identity(psutil.Process())
    assert not owner['in_any_windows_job'], 'Launcher must escape all inherited Windows jobs'
    save(path, {'time': now(), 'identity': owner, 'model_requests': 0, 'probe_only': True})
    time.sleep(15)
    assert not in_any_job()
    save(path.with_name(path.stem + '.finished.json'),
         {'time': now(), 'identity': owner, 'job_isolated_after_15_seconds': True, 'model_requests': 0})


def close_owned_tree(child, authenticated_identity=None):
    """Always return cleanup evidence, including failures; never act on reused IDs."""
    captured, errors = [], []
    try:
        if child.poll() is None:
            parent = psutil.Process(child.pid)
            if (authenticated_identity is not None and parent.pid == authenticated_identity['pid']
                    and parent.create_time() == authenticated_identity['created']):
                parent.suspend()
                for process in parent.children(recursive=True):
                    try:
                        captured.append({'pid': process.pid, 'created': process.create_time()})
                    except psutil.NoSuchProcess:
                        pass
            else:
                errors.append('Parent identity unavailable or changed; skipped PID-based tree capture')
            child.kill()
    except BaseException as error:
        errors.append('Parent termination/capture: ' + type(error).__name__)
        try:
            if child.poll() is None:
                child.kill()
        except BaseException as inner:
            errors.append('Owned parent kill: ' + type(inner).__name__)
    for item in captured:
        try:
            process = psutil.Process(item['pid'])
            if process.create_time() == item['created']:
                process.kill()
        except psutil.NoSuchProcess:
            pass
        except BaseException as error:
            errors.append('Owned descendant kill: ' + type(error).__name__)
    try:
        child.wait(timeout=10)
    except BaseException as error:
        errors.append('Owned parent wait: ' + type(error).__name__)
    survivors = []
    deadline = time.monotonic() + 5
    while True:
        survivors = []
        for item in captured:
            try:
                process = psutil.Process(item['pid'])
                if process.create_time() == item['created'] and process.is_running():
                    survivors.append(item)
            except psutil.NoSuchProcess:
                pass
            except BaseException as error:
                errors.append('Descendant closure check: ' + type(error).__name__)
                survivors.append(item)
        if not survivors or time.monotonic() >= deadline:
            break
        time.sleep(.05)
    return {'parent_closed': child.poll() is not None, 'captured_descendants': captured,
            'surviving_descendants': survivors, 'errors': errors}


def run(expected):
    owner = identity(psutil.Process())
    assert not owner['in_any_windows_job'], 'Launcher must escape all inherited Windows jobs'
    assert Path(sys.executable).resolve() == PYTHON
    data = (BASE / 'freeze.json').read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    frozen = json.loads(data)
    assert owner['affinity'] == frozen['topology']['allowed_logical_processors']
    assert not (BASE / 'launch.json').exists()
    save(BASE / 'launcher_identity.json', {'time': now(), 'identity': owner,
         'automatic_retries': 0, 'freeze_sha256': expected, 'detachment_method': 'WMI_breakaway_no_window'})
    command = [str(PYTHON), '-B', '-m', 'tools.matched_strength_recovery_20261022.runner',
               'run', '--freeze-sha256', expected]
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT))
    started, epoch = time.monotonic(), time.time()
    original = json.loads((SOURCE / 'launch.json').read_text())
    failure = cleanup = child = child_identity = None
    try:
        with (BASE / 'runner.stdout.log').open('xb') as stdout, (BASE / 'runner.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            child_identity = identity(psutil.Process(child.pid))
            assert not child_identity['in_any_windows_job'], 'Runner inherited a Windows job'
            assert Path(child_identity['executable']).resolve() == PYTHON
            assert child_identity['command'] == command
            save(BASE / 'runner_process.json', {'time': now(), 'identity': child_identity,
                'supervisor_identity': owner, 'automatic_retries': 0, 'freeze_sha256': expected})
            while child.poll() is None:
                # Independent outer ceiling includes startup verification; never extends runner budgets.
                remaining = min(43200 - max(time.monotonic() - started, time.time() - epoch, 0),
                    original['maximum_seconds'] - max(time.monotonic() - original['monotonic_start'],
                        time.time() - original['epoch_seconds'], 0))
                if remaining <= 0:
                    raise TimeoutError('Recovery or original outer execution cap reached')
                try:
                    child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
    except BaseException:
        failure = traceback.format_exc()
        if child is not None:
            cleanup = close_owned_tree(child, child_identity)
    save(BASE / 'runner_execution.json', {'time': now(), 'started_epoch_seconds': epoch,
         'started_monotonic': started, 'pid': child.pid if child else None,
         'exit_code': child.returncode if child else None,
         'elapsed_seconds': time.monotonic() - started, 'error': failure,
         'forced_cleanup': cleanup,
         'freeze_sha256': expected, 'automatic_retries': 0,
         'original_frozen_status': 'incomplete', 'scope': 'amended_recovered_benchmark'})
    return child.returncode if failure is None and child is not None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probe-path', type=Path)
    group.add_argument('--freeze-sha256')
    args = parser.parse_args()
    if args.probe_path:
        probe(args.probe_path)
    else:
        raise SystemExit(run(args.freeze_sha256))
