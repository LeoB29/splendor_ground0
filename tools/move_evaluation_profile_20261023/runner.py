"""Bounded, serial performance diagnosis of immutable scientific code."""
import argparse
import cProfile
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import psutil

from tools.local_player_qualification_128.clock import processor_topology
from . import common as c


def clamp_timeout(limit, deadline, *, clock=time.monotonic):
    remaining = deadline-clock()
    c.require(remaining > 0, 'Diagnostic stage deadline reached')
    return min(limit, remaining)


def verify_runtime(freeze):
    c.verify(freeze['inputs'])
    for item in freeze['profiling_runtime']:
        path = Path(item['absolute_path'])
        c.require(path.stat().st_size == item['bytes'] and c.file_hash(path) == item['sha256'], 'Profiling runtime changed')
    c.require(c.file_hash(sys.executable) == freeze['python_executable_sha256'], 'Python executable changed')


def prepare():
    c.require(not (c.OUT/'freeze.json').exists(), 'Diagnostic freeze already exists')
    c.require(c.file_hash(c.OLD/'freeze.json') == c.OLD_FREEZE_SHA, 'Original freeze changed')
    old = c.read(c.OLD/'freeze.json')
    c.verify(old['inputs'])
    c.require(c.read(c.OLD/'failure.json')['status'] == 'held_failure_preserved', 'Original failure missing')
    c.require(c.read(c.OLD/'execution.json')['status'] == 'runner_failed', 'Original closure missing')
    c.require(shutil.disk_usage(c.ROOT).free >= 6*c.GiB, 'Initial disk reserve')
    topology = processor_topology()
    c.require(topology['selected_logical_processor'] == 0, 'CPU0 unavailable')
    lock = c.read(c.PREP/'candidate_lock.json')
    roots = [lock['roots'][i] for i in c.ROOT_INDICES]
    c.require([(r['game'], r['ply'], r['fit']) for r in roots] ==
              [(25, 8, 161001), (57, 14, 161002), (58, 4, 161003)], 'Predetermined root identities')
    c.require(all(len(r['candidates']['actions']) == 4 for r in roots), 'Fixed four candidates')
    paths = set((c.ROOT/'tools/move_evaluation_profile_20261023').glob('*.py'))
    paths.update(c.ROOT.glob('tests/test_move_evaluation_profile_*_20261023.py'))
    paths.update(c.OUT.glob('prelaunch_*.json'))
    paths.update([c.PROTOCOL, c.OLD/'freeze.json', c.OLD/'entrypoint_qualification.json',
        c.OLD/'workers/selection/r00/continuations.jsonl.gz', c.OLD/'failure.json', c.OLD/'execution.json',
        c.OUT/'prelaunch_tests.json'])
    tests = c.read(c.OUT/'prelaunch_tests.json')
    c.require(tests['status'] == 'passed' and tests['exit_code'] == 0, 'Prelaunch tests missing')
    inputs = {a['path']: a for a in old['inputs']}
    for path in paths:
        a = c.artifact(path)
        inputs[a['path']] = a
    import _lsprof, pstats, profile
    runtime = []
    for module in (cProfile, pstats, profile, _lsprof):
        # Conda Windows builds link _lsprof into the interpreter DLL.
        if getattr(module, '__file__', None):
            p = Path(module.__file__).resolve()
        else:
            c.require(module.__spec__.origin == 'built-in' and os.name == 'nt', 'Unknown built-in profiler runtime')
            import ctypes
            buffer = ctypes.create_unicode_buffer(32768)
            size = ctypes.windll.kernel32.GetModuleFileNameW(ctypes.c_void_p(ctypes.pythonapi._handle), buffer, len(buffer))
            c.require(0 < size < len(buffer), 'Cannot locate interpreter DLL for built-in profiler')
            p = Path(buffer.value).resolve()
        runtime.append({'absolute_path': str(p), 'bytes': p.stat().st_size, 'sha256': c.file_hash(p),
                        'module': module.__name__})
    freeze = {'schema': 'fixed_evaluator_performance_diagnosis_v1', 'time': c.now(),
        'authorization': 'Owner approved the recommended separate bounded performance diagnosis.',
        'inputs': sorted(inputs.values(), key=lambda a: a['path']), 'profiling_runtime': runtime,
        'python': sys.version, 'python_executable_sha256': c.file_hash(sys.executable),
        'topology': topology, 'native_binary': old['native_binary'],
        'root_indices': list(c.ROOT_INDICES), 'roots': [{k: r[k] for k in ('id', 'game', 'ply', 'fit')} for r in roots],
        'job_seconds': c.JOB_SECONDS, 'final_reserve_seconds': c.FINAL_RESERVE,
        'request_seconds': c.REQUEST_SECONDS, 'audit_reserve_seconds_per_root': 660,
        'original_qualification_seconds': 120, 'original_failure_unchanged': True,
        'max_evidence_bytes': c.GiB, 'max_tree_rss_bytes': 2*c.GiB,
        'automatic_retries': 0, 'larger_data_deferred': True}
    c.save(c.OUT/'freeze.json', freeze)
    print(c.canonical({'freeze_sha256': c.file_hash(c.OUT/'freeze.json'), 'inputs': len(inputs), 'model_calls': 0}), flush=True)


class Envelope:
    def __init__(self, launch):
        self.start = launch['monotonic']
        self.wall_start = launch['unix_time']
        self.end = self.start+c.JOB_SECONDS
        self.work_end = self.end-c.FINAL_RESERVE
        self.waited = 0.
        self.admitted = False
        self.admissions = 0

    def guard(self, deadline, *, memory=True):
        c.require(time.monotonic() < min(deadline, self.end), 'Diagnostic stage/global deadline')
        c.require(time.time() < self.wall_start+c.JOB_SECONDS, 'Diagnostic wall deadline')
        c.require(shutil.disk_usage(c.ROOT).free >= 5*c.GiB, 'Free disk reserve')
        used = sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())
        c.require(used < c.GiB, 'New evidence size cap')
        if memory:
            c.require(psutil.virtual_memory().available >= 2*c.GiB, 'RAM shortage during active request')

    def admit(self, deadline):
        self.guard(deadline, memory=False)
        start = time.monotonic()
        available = psutil.virtual_memory().available
        stable = start if available >= 2*c.GiB else None
        fast = self.admitted and stable is not None
        minimum = available
        while not fast:
            self.guard(deadline, memory=False)
            current = time.monotonic()
            available = psutil.virtual_memory().available
            minimum = min(minimum, available)
            if available < 2*c.GiB:
                stable = None
            elif stable is None:
                stable = current
            c.require(self.waited+current-start < 300, 'Cumulative memory admission allowance exhausted')
            if stable is not None and current-stable >= 60:
                break
            time.sleep(.5)
        elapsed = time.monotonic()-start
        self.waited += elapsed
        self.admitted = True
        self.admissions += 1
        c.save(c.OUT/'memory_admissions'/f'{self.admissions:03d}.json', {
            'status': 'passed', 'time': c.now(), 'fast_path': fast, 'wait_seconds': elapsed,
            'cumulative_wait_seconds': self.waited, 'minimum_available_bytes': minimum})


def perform(name, mode, payload, limit, deadline, envelope, freeze):
    c.require('torch' not in sys.modules, 'Supervisor must remain model-free')
    envelope.admit(deadline)
    folder = c.OUT/'workers'/name
    print(c.canonical({'stage': name, 'event': 'starting', 'time': c.now()}), flush=True)
    with c.StudyWorker('tools.move_evaluation_profile_20261023.worker', folder,
        native_artifact=freeze['native_binary'], affinity=[0], initialization_timeout=clamp_timeout(60, deadline),
        absolute_deadline=deadline, max_message_bytes=32*1024**2, max_request_bytes=64*1024**2,
        max_log_bytes=256*1024**2, max_rss_bytes=2*c.GiB,
        env={'PYTHONPATH': str(c.ROOT/'src')+os.pathsep+str(c.ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
             'CUDA_VISIBLE_DEVICES': '', 'STUDY_MODE': mode, 'STUDY_WORKER_DIR': str(folder),
             'RAYON_NUM_THREADS': '1', 'TOKIO_WORKER_THREADS': '1'},
        resource_guard=lambda: envelope.guard(deadline)) as child:
        receipt = child.request(payload, clamp_timeout(limit, deadline))
        c.require(receipt['status'] == 'ok' and receipt['accepted'], 'Request failed: '+c.canonical({
            k: receipt[k] for k in ('status', 'detail', 'receipt_elapsed_seconds')}))
        initialization = child.initialization
    before = {(p['pid'], p['created']): p['cpu_seconds'] for p in initialization['telemetry']['process_tree']}
    cpu = {}
    for p in receipt['telemetry']['process_tree']:
        cpu[p['role']] = p['cpu_seconds']-before.get((p['pid'], p['created']), 0.)
    record = {'name': name, 'mode': mode, 'result': receipt['result'],
        'receipt': c.artifact(folder/(receipt['request_id']+'.json')),
        'initialization_seconds': initialization['elapsed_seconds'],
        'complete_response_seconds': receipt['receipt_elapsed_seconds'],
        'sampled_cpu_deltas_by_role': cpu,
        'cpu_scope': 'final receipt sample minus initialization sample; differs slightly from request endpoints',
        'request_tree_cpu_seconds': receipt['telemetry'].get('request_cpu_seconds'),
        'peak_tree_rss_bytes': receipt['telemetry']['peak_sampled_rss_bytes']}
    c.save(c.OUT/'measurements'/(name.replace('/', '_')+'.json'), record)
    print(c.canonical({'stage': name, 'event': 'completed', 'seconds': record['complete_response_seconds']}), flush=True)
    return record


def run(expected_sha):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    # The outer owner starts the single clock before spawning this runner.
    # Source admission and Python import/startup consume that same allowance.
    launch = {'time': c.now(), 'monotonic': float(os.environ['DIAGNOSTIC_START_MONOTONIC']),
              'unix_time': float(os.environ['DIAGNOSTIC_START_UNIX']),
              'pid': os.getpid(), 'created': psutil.Process().create_time(), 'freeze_sha256': expected_sha}
    c.save(c.OUT/'launch.json', launch)
    envelope = Envelope(launch)
    completed = []
    try:
        c.require(c.file_hash(c.OUT/'freeze.json') == expected_sha, 'Diagnostic freeze mismatch')
        freeze = c.read(c.OUT/'freeze.json')
        verify_runtime(freeze)
        envelope.guard(envelope.end, memory=False)
        c.require(processor_topology() == freeze['topology'], 'Topology changed')
        c.save(c.OUT/'supervisor_runtime_sources.json', {'sources': verify_runtime_sources(), 'time': c.now()})
        from .analyze import summarize_profile, compare_scientific_runs, inspect_old_prefix
        qualification = perform('qualification', 'qualify', {'operation': 'entrypoint_checks'},
                                300, envelope.work_end, envelope, freeze)
        c.require(c.digest(qualification['result']['scientific']) == c.digest(c.read(c.OLD/'entrypoint_qualification.json')),
                  'New wrapper qualification differs from accepted fixtures')
        c.save(c.OUT/'entrypoint_qualification.json', {'status': 'passed', 'same_scientific_result_as_original': True,
            'measurement': c.artifact(c.OUT/'measurements/qualification.json')})
        lock = c.read(c.PREP/'candidate_lock.json')
        for index in c.ROOT_INDICES:
            row = lock['roots'][index]
            payload = c.selector_payload(row)
            # Reserve the complete audit ceiling plus initialization before any
            # new root's selections, never borrow from final authentication.
            select_end = envelope.work_end-660
            envelope.guard(select_end)
            plain = perform(f'r{index:02d}/plain', 'plain', payload, c.REQUEST_SECONDS, select_end, envelope, freeze)
            profiled = perform(f'r{index:02d}/profile', 'profile', payload, c.REQUEST_SECONDS, select_end, envelope, freeze)
            p, q = plain['result']['scientific'], profiled['result']['scientific']
            comparison = compare_scientific_runs(p, q)
            old_prefix = inspect_old_prefix(c.OLD/'workers/selection/r00/continuations.jsonl.gz', c.ROOT/p['trace']['path']) if index == 0 else None
            a = profiled['result']['measurement']['profile']
            profile_summary = summarize_profile(c.ROOT/a['path'], a)
            c.save(c.OUT/'profiles'/f'r{index:02d}.json', profile_summary)
            c.save(c.OUT/'comparisons'/f'r{index:02d}.json', {'pair': comparison, 'old_failed_prefix': old_prefix})
            audit = perform(f'r{index:02d}/audit', 'audit', {'selector': payload, 'result': p,
                'trace': c.static_artifact(p['trace'])}, c.REQUEST_SECONDS, envelope.work_end, envelope, freeze)
            completed.append({'index': index, 'id': row['id'], 'fit': row['fit'],
                'plain': c.artifact(c.OUT/'measurements'/f'r{index:02d}_plain.json'),
                'profiled': c.artifact(c.OUT/'measurements'/f'r{index:02d}_profile.json'),
                'audit': c.artifact(c.OUT/'measurements'/f'r{index:02d}_audit.json'),
                'profile_summary': c.artifact(c.OUT/'profiles'/f'r{index:02d}.json'),
                'comparison': c.artifact(c.OUT/'comparisons'/f'r{index:02d}.json'),
                'plain_seconds': plain['complete_response_seconds'], 'profiled_seconds': profiled['complete_response_seconds'],
                'audit_seconds': audit['complete_response_seconds']})
            c.save(c.OUT/'completed_roots'/f'r{index:02d}.json', completed[-1])
        envelope.guard(envelope.end)
        c.require(len(completed) == 3, 'Fixed trio incomplete')
        verify_runtime(freeze)
        envelope.guard(envelope.end)
        inventory = [c.artifact(p) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name not in
                     ('runner_stdout.log', 'runner_stderr.log', 'watchdog.json', 'launcher_stdout.log', 'launcher_stderr.log')]
        bound = c.save(c.OUT/'final_inventory.json', {'artifacts': inventory, 'time': c.now()})
        envelope.guard(envelope.end)
        c.save(c.OUT/'completion.json', {'status': 'completed_diagnosis_only', 'roots': completed, 'time': c.now(),
            'elapsed_seconds': time.monotonic()-launch['monotonic'], 'final_inventory': bound,
            'original_120_second_qualification': 'failed_unchanged', 'larger_data_deferred': True})
    except BaseException:
        c.save(c.OUT/'failure.json', {'status': 'held_failure_preserved', 'time': c.now(),
            'elapsed_seconds': time.monotonic()-launch['monotonic'], 'error': traceback.format_exc(),
            'fully_completed_root_indices': [r['index'] for r in completed], 'automatic_retry': False,
            'original_120_second_qualification': 'failed_unchanged', 'fixed_trio_complete': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('prepare', 'run'))
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare() if args.operation == 'prepare' else run(args.freeze_sha256)


if __name__ == '__main__':
    main()
