"""One bounded, authenticated final analysis; no launch occurs on import."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

from tools.matched_strength_20261020 import common as c
from tools.operational_final_validation_repair_20261020 import ProcessTree, identity, closed

CPU, ANALYSIS_SECONDS, FINAL_SECONDS = 2, 3600, 600
MAX_LOG = 2 * 1024**2
ANALYZER = c.ROOT / 'tools/analyze_matched_strength_20261020.py'
CONTRACT = c.PREPARATION / 'completion_contract.json'
LIVE_SUPERVISOR_FILES = {'runner.stdout.log', 'runner.stderr.log', 'runner_execution.json'}


def command():
    return [sys.executable, '-B', str(ANALYZER), '--base', str(c.OUTPUT),
            '--output', str(c.OUTPUT / 'independent_analysis.json')]


def contract():
    helper = Path(os.environ['SystemRoot']) / 'System32/conhost.exe'
    return {'schema': 'matched_strength_completion_contract_v1',
        'analyzer_command': command(), 'cpu': CPU,
        'maximum_seconds': ANALYSIS_SECONDS, 'final_reserve_seconds': FINAL_SECONDS,
        'maximum_log_bytes': MAX_LOG, 'maximum_process_tree_rss_bytes': c.MAX_WORKER_RSS,
        'console_helper': {'executable': str(helper.resolve()), 'name': 'conhost.exe',
            'sha256': c.file_hash(helper), 'maximum_distinct_helpers': 1},
        'analysis_attempts': 1, 'automatic_retries': 0, 'creationflags': 'CREATE_NO_WINDOW'}


def verify_report(report, freeze_sha256):
    cfg = c.config()
    c.require(report['status'] == 'passed' and report['final_acceptance'] is False and
              report['outcomes_opened'] is True and report['freeze_sha256'] == freeze_sha256 and
              report['strength_analysis_games'] == 2*cfg['pair_count'] and
              report['completed_games'] == 2*cfg['pair_count'] and
              report['completed_pairs'] == cfg['pair_count'] and
              type(report['statistics']) is dict, 'Independent fixed-sample report scope')


def supervise(argv, log_path, plan, *, guard, maximum_seconds=ANALYSIS_SECONDS):
    """The same authenticated OS-helper exception as the accepted repair launcher."""
    child = tree = owner = None
    cleanup = {'errors': [], 'parent_closed': False, 'descendants_closed': False}
    error = None
    started, epoch = time.monotonic(), time.time()
    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    try:
        # The hidden console helper must inherit the same single CPU as its parent.
        process.cpu_affinity([CPU])
        with Path(log_path).open('xb') as log:
            child = subprocess.Popen(argv, cwd=c.ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT),
                     'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            owner = identity(psutil.Process(child.pid))
            tree = ProcessTree(owner, plan)
            process.cpu_affinity(original_affinity)
            while child.poll() is None:
                guard()
                c.require(max(time.monotonic()-started, time.time()-epoch, 0.) < maximum_seconds,
                          'Independent analysis time allowance exhausted')
                c.require(Path(log_path).stat().st_size <= MAX_LOG, 'Independent analysis log ceiling')
                try:
                    tree.sample()
                except psutil.NoSuchProcess:
                    c.require(child.poll() is not None, 'Live analyzer identity unavailable')
                    break
                try:
                    child.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
            guard()
            c.require(Path(log_path).stat().st_size <= MAX_LOG, 'Independent analysis log ceiling')
            c.require(max(time.monotonic()-started, time.time()-epoch, 0.) < maximum_seconds,
                      'Independent analysis exceeded time allowance')
    except BaseException:
        error = traceback.format_exc()[-16384:]
    finally:
        restore_errors = []
        try:
            process.cpu_affinity(original_affinity)
        except BaseException as exc:
            restore_errors.append('Affinity restore: '+type(exc).__name__)
        try:
            if tree is not None:
                cleanup = tree.cleanup(child)
                # Windows may retain a just-terminated identity after wait().
                # Observe closure; never retry or relaunch the work.
                closure_deadline = time.monotonic()+5
                while time.monotonic() < closure_deadline:
                    cleanup['parent_closed'] = closed(owner)
                    cleanup['descendants_closed'] = all(closed(x) for x in tree.observed.values())
                    if cleanup['parent_closed'] and cleanup['descendants_closed']:
                        break
                    time.sleep(.05)
            elif child is not None:
                # Popen owns this child; do not act on an unauthenticated PID.
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                cleanup['errors'].append('Analyzer identity was never authenticated')
        except BaseException as exc:
            cleanup['errors'].append('Closure failure: '+type(exc).__name__)
        cleanup['errors'].extend(restore_errors)
    try:
        guard()
        c.require(max(time.monotonic()-started, time.time()-epoch, 0.) < maximum_seconds,
                  'Independent analysis and closure exceeded allowance')
    except BaseException:
        error = error or traceback.format_exc()[-16384:]
    return {'time': c.now(), 'command': argv, 'started_epoch_seconds': epoch,
        'started_monotonic': started, 'elapsed_seconds': time.monotonic()-started,
        'wall_elapsed_seconds': time.time()-epoch, 'exit_code': child.returncode if child else None,
        'analyzer_identity': owner, 'error': error, 'cleanup': cleanup,
        'observed_descendants': list(tree.observed.values()) if tree else [],
        'peak_sampled_combined_rss_bytes': tree.peak_rss if tree else None,
        'automatic_retries': 0}


def complete(deadline):
    """Called once by the runner after collection, under its original global clock."""
    base = c.OUTPUT
    try:
        c.require('torch' not in sys.modules, 'Completion supervisor must remain model-free')
        for name in ('independent_analysis.json', 'analysis_execution.json', 'finalization.json',
                     'finalization_validation.json', 'completion_hold.json'):
            c.require(not (base/name).exists(), 'Completion already attempted; preserve existing evidence')
        frozen = c.read(base/'freeze.json')
        freeze_sha = c.file_hash(base/'freeze.json')
        c.verify(frozen['inputs'])
        c.require(c.read(CONTRACT) == contract(), 'Frozen completion contract changed')
        collection = c.read(base/'completion.json')
        c.require(collection['status'] == 'completed_pending_independent_analysis' and
                  collection['failure'] is None and collection['worker_processes_closed'] is True and
                  collection['completed_games'] == 2*c.config()['pair_count'], 'Collection incomplete')
        from tools.matched_strength_20261020.memory import wait_for_memory
        wait_for_memory(base, deadline, name='analysis', reserve=FINAL_SECONDS,
                        request_seconds=ANALYSIS_SECONDS)
        plan = {**c.read(CONTRACT), 'runner_identity': identity(psutil.Process())}
        c.save_new(base/'analysis_dispatch.json', {'time': c.now(), 'plan': plan,
            'freeze': c.artifact(base/'freeze.json'), 'launch': c.artifact(base/'launch.json'),
            'memory_admission': c.artifact(base/'memory_admissions/analysis.json')})
        def guard():
            receipt = c.boundary(base, deadline, memory_headroom=False)
            c.require(receipt['remaining_seconds'] > FINAL_SECONDS, 'Protected finalization reserve reached')
        execution = supervise(command(), base/'analysis.log', plan, guard=guard)
        execution.update(dispatch=c.artifact(base/'analysis_dispatch.json'),
                         log=c.artifact(base/'analysis.log'))
        c.save_new(base/'analysis_execution.json', execution)
        clean = execution['cleanup']
        c.require(execution['exit_code'] == 0 and execution['error'] is None and not clean['errors'] and
                  clean['parent_closed'] and clean['descendants_closed'] and
                  closed(execution['analyzer_identity']) and
                  all(closed(x) for x in execution['observed_descendants']), 'Independent analysis did not close successfully')
        report = c.read(base/'independent_analysis.json')
        verify_report(report, freeze_sha)
        c.verify(frozen['inputs'])
        c.boundary(base, deadline, memory_headroom=False)
        sources = [c.artifact(p) for p in sorted(base.rglob('*')) if p.is_file() and
                   p.relative_to(base).as_posix() not in LIVE_SUPERVISOR_FILES]
        c.save_new(base/'finalization.json', {'schema': 'matched_strength_finalization_v1',
            'time': c.now(), 'status': 'accepted', 'freeze_sha256': freeze_sha,
            'games': 2*c.config()['pair_count'], 'pairs': c.config()['pair_count'],
            'timeouts_counted_as_losses': True, 'fresh_sample_only': True,
            'analysis_and_descendants_closed': True, 'outcomes_opened': True,
            'larger_data_deferred': True, 'artifacts': sources,
            'excluded_live_supervisor_files': sorted(LIVE_SUPERVISOR_FILES),
            'exclusion_reason': 'Parent launch logs/exit receipt may finish after this child-analysis acceptance'})
        c.verify(sources)
        c.require(c.file_hash(base/'freeze.json') == freeze_sha, 'Freeze changed during finalization')
        c.boundary(base, deadline, memory_headroom=False)
        c.save_new(base/'finalization_validation.json', {'time': c.now(), 'status': 'passed',
            'finalization': c.artifact(base/'finalization.json'), 'bound_artifacts': len(sources),
            'games': 2*c.config()['pair_count'], 'pairs': c.config()['pair_count'],
            'outcomes_opened': True, 'larger_data_deferred': True})
        return 0
    except BaseException:
        if not (base/'completion_hold.json').exists():
            c.save_new(base/'completion_hold.json', {'time': c.now(), 'status': 'preserved_hold',
                'traceback': traceback.format_exc()[-16384:], 'results_accepted': False,
                'automatic_retries': 0})
        return 1
