"""Saved-only selection-batch validation; never alters a production backend."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
PRIOR = ROOT / 'outputs/simulation_speed_probe_128_02'
PREVIOUS = ROOT / 'outputs/simulation_speed_validation_128_01'
CONFIG = {
    'roots': [515, 517, 465, 251, 277, 382],
    'selection_worlds': 128, 'selection_continuations': 7040, 'selection_turns': 243536,
    'checked_worlds': [0, 17, 35, 53, 71, 89, 107, 127],
    'assessment_checked_worlds': [0, 127],
    'maximum_active_seconds': 4 * 3600, 'admission_seconds': 180,
    'physical_pressure_warning_bytes': 512 * 1024**2,
    'severe_physical_pressure_warning_bytes': 128 * 1024**2,
    'physical_memory_policy': 'telemetry_only_no_stop_or_pause',
    'minimum_commit_headroom_bytes': 6 * 1024**3,
    'minimum_disk_free_bytes': 5 * 1024**3,
    'maximum_worker_rss_bytes': 512 * 1024**2,
    'maximum_parent_rss_bytes': 256 * 1024**2,
    'maximum_group_rss_bytes': 1280 * 1024**2,
    'maximum_worker_private_bytes': 512 * 1024**2,
    'maximum_parent_private_bytes': 256 * 1024**2,
    'maximum_group_private_bytes': 1280 * 1024**2,
    'threads': 1, 'production_enabled': False,
}
CORE_FIELDS = ('reason', 'winner', 'turns', 'trace_sha256', 'final_state_sha256')
SCHEDULE = [
    {'key': 'checks', 'backend': 'checked', 'workers': 1, 'repeat': None},
    {'key': 'r0_w1_reference', 'backend': 'reference', 'workers': 1, 'repeat': 0},
    {'key': 'r0_w1_prototype', 'backend': 'prototype', 'workers': 1, 'repeat': 0},
    {'key': 'r0_w2_prototype', 'backend': 'prototype', 'workers': 2, 'repeat': 0},
    {'key': 'r0_w2_reference', 'backend': 'reference', 'workers': 2, 'repeat': 0},
    {'key': 'r1_w1_prototype', 'backend': 'prototype', 'workers': 1, 'repeat': 1},
    {'key': 'r1_w1_reference', 'backend': 'reference', 'workers': 1, 'repeat': 1},
    {'key': 'r1_w2_reference', 'backend': 'reference', 'workers': 2, 'repeat': 1},
    {'key': 'r1_w2_prototype', 'backend': 'prototype', 'workers': 2, 'repeat': 1},
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def verify(ref):
    path = (ROOT / ref['path']).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != ref['sha256']:
        raise ValueError('Changed or outside artifact: ' + str(path))
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f'.pending_{time.time_ns()}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    # Windows rename fails if final already exists; incomplete temp evidence stays.
    temporary.rename(path)


def replace_status(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def safe_output(path):
    path = Path(path).resolve()
    if path.parent != ROOT / 'outputs' or not path.name.startswith('simulation_speed_validation_128_') or path == PREVIOUS:
        raise ValueError('Use a new direct outputs/simulation_speed_validation_128_* directory, never the prior_01')
    return path


def code_identity():
    paths = sorted((ROOT / 'src/splendor_ai').rglob('*.py'))
    for namespace in ('simulation_speed_probe_128_v2', 'simulation_speed_validation_128', 'simulation_speed_validation_128_v2'):
        paths += sorted((ROOT / 'tools' / namespace).glob('*.py'))
        paths += [ROOT / 'tools' / namespace / 'PROTOCOL.md']
    return [artifact(p) for p in paths]


def child_env():
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + str(ROOT)
    return env


def process_options():
    return {'cwd': ROOT, 'env': child_env(), 'creationflags':
            getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)}


def process_identity(pid):
    import psutil
    process = psutil.Process(pid)
    return {'pid': pid, 'create_time': process.create_time()}


def identity_alive(identity):
    import psutil
    try:
        process = psutil.Process(identity['pid'])
        return process.is_running() and process.create_time() == identity['create_time']
    except psutil.NoSuchProcess:
        return False


def require_parent(identity):
    if not identity_alive(identity):
        raise RuntimeError('Validation controller lease expired')


def require_no_prior_workers(output):
    import psutil
    for path in (Path(output) / 'attempts').glob('*/root_*_process.json'):
        if identity_alive(read(path)):
            raise RuntimeError('A prior validation worker is still alive; wait for its controller lease to expire')
    # Cover a controller crash between Popen and the durable PID receipt. Inspect
    # command lines only for Python processes; never print unrelated commands.
    for process in psutil.process_iter(['name']):
        if not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            if command_is_own_worker(process.cmdline(), output):
                raise RuntimeError('A prior validation worker has no exit yet; do not overlap attempts')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def command_is_own_worker(command, output):
    if len(command) != 6 or command[1:4] != ['-m', 'tools.simulation_speed_validation_128_v2.collector', 'worker'] or command[4] != '--job':
        return False
    path = Path(command[5]).resolve()
    return path.is_relative_to(Path(output).resolve() / 'attempts') and path.name.startswith('root_') and path.name.endswith('_job.json')


def memory():
    from tools.simulation_speed_probe_128_v2.probe import memory as measured
    result = measured()
    result['physical_pressure'] = result['available_ram_bytes'] < CONFIG['physical_pressure_warning_bytes']
    result['severe_physical_pressure'] = result['available_ram_bytes'] < CONFIG['severe_physical_pressure_warning_bytes']
    return result


def resource_ok(info, disk, role='parent', group_rss=0, group_private=0):
    cap = CONFIG['maximum_parent_rss_bytes' if role == 'parent' else 'maximum_worker_rss_bytes']
    private_cap = CONFIG['maximum_parent_private_bytes' if role == 'parent' else 'maximum_worker_private_bytes']
    return (info['commit_headroom_bytes'] >= CONFIG['minimum_commit_headroom_bytes']
            and max(info['process_rss_bytes'], info.get('process_peak_rss_bytes') or 0) <= cap
            and info['process_private_bytes'] <= private_cap
            and group_rss <= CONFIG['maximum_group_rss_bytes']
            and group_private <= CONFIG['maximum_group_private_bytes']
            and disk >= CONFIG['minimum_disk_free_bytes'])


def operational_guard(output, role='parent', group_rss=0, group_private=0, waiting=False):
    from tools.simulation_speed_probe_128_v2.probe import permitted_stage
    info = memory()
    admission_info = {**info, 'commit_headroom_bytes': max(info['commit_headroom_bytes'], CONFIG['minimum_commit_headroom_bytes'])} if waiting else info
    if not resource_ok(admission_info, shutil.disk_usage(ROOT).free, role, group_rss, group_private):
        raise RuntimeError('Resource hold: ' + json.dumps(info))
    if (Path(output) / 'abort.json').exists():
        raise RuntimeError('Validation abort requested')
    current = ROOT / 'outputs/structured_proposer_128'
    complete = read(current / 'final_status.json').get('complete') is True if (current / 'final_status.json').exists() else False
    if not permitted_stage(read(current / 'progress.json').get('stage'), complete):
        raise RuntimeError('Current study entered a protected collection/evaluation stage')
    return info


def admission_headroom(workers):
    return (CONFIG['minimum_commit_headroom_bytes'] + CONFIG['maximum_parent_private_bytes']
            + workers * CONFIG['maximum_worker_private_bytes'])


def pressure_summary(samples):
    """Observed host-pressure occupancy, not a claim to measure paging directly."""
    if not samples:
        return {'samples': 0, 'conditional_on_host_pressure': True}
    available = [row['available_ram_bytes'] for row in samples]
    result = {'samples': len(samples), 'minimum_available_ram_bytes': min(available),
              'sampled_span_seconds': samples[-1]['elapsed']-samples[0]['elapsed'],
              'conditional_on_host_pressure': True}
    for label, threshold in [('low',CONFIG['physical_pressure_warning_bytes']),
                             ('severe',CONFIG['severe_physical_pressure_warning_bytes'])]:
        result[label+'_sample_fraction'] = sum(value < threshold for value in available)/len(samples)
        result[label+'_sampled_seconds'] = sum(samples[i+1]['elapsed']-row['elapsed']
                    for i,row in enumerate(samples[:-1]) if row['available_ram_bytes'] < threshold)
    return result


@contextmanager
def policy_scope(old, policy):
    previous = old._RAW
    old._RAW = policy
    try:
        yield
    finally:
        old._RAW = previous


def trajectory_with_policy(old, policy, state, action=None, include_trace=False):
    with policy_scope(old, policy):
        return old.trajectory(state, action, include_trace=include_trace)


def cases(record, checked=False):
    """Keep outcome position distinct from a sparse assessment candidate index."""
    for split in (('selection', 'assessment') if checked else ('selection',)):
        block = record[split]
        indices = list(range(len(record['union']['actions']))) if split == 'selection' else block['candidate_indices']
        if len(indices) != len(set(indices)) or any(i < 0 or i >= len(record['union']['actions']) for i in indices):
            raise ValueError('Malformed candidate indices')
        if len(block['worlds']) != 128:
            raise ValueError('Expected a complete saved128-world block')
        worlds = CONFIG['assessment_checked_worlds'] if split == 'assessment' else (CONFIG['checked_worlds'] if checked else range(128))
        for world_index in worlds:
            world = block['worlds'][world_index]
            if len(world['outcomes']) != len(indices):
                raise ValueError('Outcome/candidate count mismatch')
            for offset, candidate in enumerate(indices):
                expected = {key: world['outcomes'][offset][key] for key in CORE_FIELDS}
                if expected['reason'] not in ('completed', 'forced_pass_deadlock', 'max_turns', 'repetition_cutoff', 'stalled', 'no_progress_cutoff'):
                    raise ValueError('Unknown termination reason')
                yield split, world_index, candidate, world, expected


def preflight(output):
    output = safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'preflight.json').exists():
        raise FileExistsError(output / 'preflight.json')
    identity = code_identity()
    operational_guard(output)
    if memory()['commit_headroom_bytes'] < admission_headroom(1):
        raise RuntimeError('Insufficient commitment headroom for focused tests')
    command = [sys.executable, '-m', 'pytest', str(Path(__file__).parent), '-q', '-p', 'no:cacheprovider']
    started = time.perf_counter()
    with (output / 'preflight.stdout').open('x', encoding='utf-8') as stdout, (output / 'preflight.stderr').open('x', encoding='utf-8') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180, **process_options())
    if result.returncode or code_identity() != identity:
        raise RuntimeError('Focused checks failed or code changed; preserve logs')
    receipt = {'passed': True, 'code': identity, 'elapsed': time.perf_counter() - started,
               'stdout': artifact(output / 'preflight.stdout'), 'stderr': artifact(output / 'preflight.stderr')}
    save(output / 'preflight.json', receipt)
    return {'passed': True, 'elapsed': receipt['elapsed']}


def prepare(output):
    output = safe_output(output)
    checks = read(output / 'preflight.json')
    if checks.get('passed') is not True or checks['code'] != code_identity():
        raise ValueError('Matching successful preflight required')
    from tools.simulation_speed_probe_128_v2 import probe
    _, prior = probe.load_plan(PRIOR)
    result = read(PRIOR / 'rollout.json')
    if result['binding'] != prior['binding'] or result['all_exact'] is not True:
        raise ValueError('Prior full-trace check missing')
    selected = [r for r in prior['selected'] if r['support_band'] in ('median', 'maximum')]
    if [r['spec']['index'] for r in selected] != CONFIG['roots']:
        raise ValueError('Fixed selection changed')
    counts = {'selection_continuations': 0, 'selection_turns': 0, 'checked_selection': 0, 'checked_assessment': 0}
    for item in selected:
        record = read(verify(item['root']))
        for _, _, _, _, expected in cases(record):
            counts['selection_continuations'] += 1
            counts['selection_turns'] += expected['turns']
        for split, *_ in cases(record, checked=True):
            counts['checked_' + split] += 1
    if any(counts[k] != CONFIG[k] for k in ('selection_continuations', 'selection_turns')) or counts['checked_selection'] != 440:
        raise ValueError('Fixed workload counts changed')
    from tools.simulation_speed_validation_128 import collector as previous
    previous.load_plan(PREVIOUS)
    previous.require_no_prior_workers(PREVIOUS)
    if list((PREVIOUS / 'attempts').glob('*/result.json')):
        raise ValueError('Amendment expects no complete predecessor blocks; never select prior timings')
    previous_attempts = [artifact(path) for path in sorted((PREVIOUS / 'attempts').glob('*/*.json'))]
    amendment = {'previous_plan': artifact(PREVIOUS / 'plan.json'), 'previous_attempts': previous_attempts,
                 'previous_active_seconds': previous.active_used(PREVIOUS),
                 'reason': 'Owner authorized continuing through physical-memory dips; commitment and own bounded memory remain enforced.',
                 'all_prior_evidence_preserved': True, 'prior_partial_timings_reused': False,
                 'fixed_workload_model_schedule_unchanged': True}
    plan = {'config': CONFIG, 'schedule': SCHEDULE, 'selected': selected, 'counts': counts,
            'code': code_identity(), 'preflight': artifact(output / 'preflight.json'),
            'prior_plan': artifact(PRIOR / 'plan.json'), 'prior_result': artifact(PRIOR / 'rollout.json'),
            'checkpoint': prior['checkpoint'], 'production_enabled': False, 'resource_amendment': amendment,
            'time': datetime.now(timezone.utc).isoformat()}
    plan['binding'] = digest(plan)
    save(output / 'plan.json', plan)
    return {'binding': plan['binding'], 'counts': counts}


def load_plan(output):
    output = safe_output(output)
    plan = read(output / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'binding'}) != plan['binding']:
        raise ValueError('Changed plan')
    if plan['config'] != CONFIG or plan['schedule'] != SCHEDULE or plan['code'] != code_identity():
        raise ValueError('Changed frozen code/configuration')
    for key in ('preflight', 'prior_plan', 'prior_result', 'checkpoint'):
        verify(plan[key])
    for item in plan['selected']:
        verify(item['root'])
    amendment = plan['resource_amendment']
    verify(amendment['previous_plan'])
    current = [artifact(path) for path in sorted((PREVIOUS / 'attempts').glob('*/*.json'))]
    if current != amendment['previous_attempts']:
        raise ValueError('Predecessor attempt evidence changed after the resource amendment')
    from tools.simulation_speed_validation_128 import collector as previous
    if previous.active_used(PREVIOUS) != amendment['previous_active_seconds']:
        raise ValueError('Predecessor active time changed')
    return output, plan


class WorkerGuard:
    def __init__(self, output, seconds, pulse, parent):
        self.output, self.started, self.seconds, self.pulse = output, time.perf_counter(), seconds, pulse
        self.parent = parent
        self.last, self.snapshots = -1., []
        self(force=True)

    def __call__(self, force=False):
        elapsed = time.perf_counter() - self.started
        if elapsed >= self.seconds:
            raise RuntimeError('Worker reached remaining cumulative time budget')
        if not force and elapsed - self.last < 1:
            return
        self.last = elapsed
        require_parent(self.parent)
        info = operational_guard(self.output, 'worker')
        if force or not self.snapshots or elapsed - self.snapshots[-1]['elapsed'] >= 10:
            self.snapshots.append({'elapsed': elapsed, **info})
        replace_status(self.pulse, {'elapsed': elapsed, 'memory': info, 'pid': os.getpid()})


def worker(job_path):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started, cpu_started = time.perf_counter(), time.process_time()
    job = read(job_path)
    output, plan = load_plan(job['output'])
    attempt = Path(job_path).parent
    guard = WorkerGuard(output, job['remaining_seconds'], attempt / f'root_{job["root"]}_pulse.json', job['parent'])
    from tools.simulation_speed_probe_128_v2 import probe, encoder
    torch, old = probe.initialize()
    original = old._RAW
    checked = job['backend'] == 'checked'
    tensor_checks = 0
    def check(state, legal, actual):
        nonlocal tensor_checks
        expected = encoder.reference_tensors(state, legal)
        if not all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)):
            raise ValueError('Intermediate input mismatch')
        tensor_checks += 1
    policy = (probe.GuardedOriginal(original, guard) if job['backend'] == 'reference'
              else encoder.Fast035(original, guard, check if checked else None))
    item = next(row for row in plan['selected'] if row['spec']['index'] == job['root'])
    record, state = probe.root_state(item, old)
    actions = [old.decode_action(a) for a in record['union']['actions']]
    rows, cached_key, world = [], None, None
    work_started, work_cpu = time.perf_counter(), time.process_time()
    for split, world_index, candidate, saved, expected in cases(record, checked):
        guard()
        key = (split, world_index)
        if key != cached_key:
            world = old.determinize_for_observer(state, state.current_player, random.Random(saved['seed']))
            if old.digest(old.serialize_state_snapshot(world)) != saved['state_sha256']:
                raise ValueError('Historical world identity changed')
            cached_key = key
        result = trajectory_with_policy(old, policy, world, actions[candidate])
        if result != expected:
            raise ValueError(f'Full continuation mismatch: {job["root"]}/{split}/{world_index}/{candidate}')
        rows.append({'split': split, 'world_index': world_index, 'candidate_index': candidate, **result})
    work_ended = time.perf_counter()
    work_timing = {'wall': work_ended - work_started, 'cpu': time.process_time() - work_cpu}
    expected_checks = sum(max(row['turns']-1, 0) for row in rows) if checked else 0
    if tensor_checks != expected_checks:
        raise ValueError('Exact input callback coverage differs from every subsequent decision')
    guard(force=True)
    payload = {'binding': plan['binding'], 'root': job['root'], 'backend': job['backend'],
               'phase': item['spec']['phase'], 'support_band': item['support_band'], 'exact': True,
               'continuations': len(rows), 'turns': sum(r['turns'] for r in rows), 'tensor_checks': tensor_checks,
               'rows': rows, 'rows_digest': digest(rows), 'work_timing': work_timing,
               'work_interval': [work_started, work_ended],
               'worker_timing_before_final_write': {'wall': time.perf_counter()-started, 'cpu': time.process_time()-cpu_started},
               'memory': guard.snapshots, 'physical_pressure': pressure_summary(guard.snapshots),
               'pid': os.getpid(), 'production_enabled': False}
    save(attempt / f'root_{job["root"]}_result.json', payload)


def active_used(output):
    plan_path = Path(output) / 'plan.json'
    total = read(plan_path)['resource_amendment']['previous_active_seconds'] if plan_path.exists() else 0.
    for attempt in (Path(output) / 'attempts').glob('*'):
        if (attempt / 'result.json').exists():
            total += read(attempt / 'result.json')['active_seconds']
        elif (attempt / 'failure.json').exists():
            total += read(attempt / 'failure.json')['active_seconds']
        elif (attempt / 'pulse.json').exists():
            # Conservative allowance for a crash between one-second heartbeats.
            total += read(attempt / 'pulse.json')['active_seconds'] + 5
        else:
            total += 5
    return total


def admit(output, workers, seconds, history, pulse):
    threshold = admission_headroom(workers)
    start, consecutive = time.perf_counter(), 0
    while time.perf_counter() - start < min(CONFIG['admission_seconds'], seconds):
        info = operational_guard(output, waiting=True)
        history.append({'elapsed': time.perf_counter()-start, **info})
        pulse()
        enough = info['commit_headroom_bytes'] >= threshold
        consecutive = consecutive + 1 if enough else 0
        if consecutive == 5:
            operational_guard(output)
            return
        time.sleep(1)
    raise RuntimeError('Bounded memory admission expired')


def stage_attempt(output, plan, stage, remaining):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    attempts = output / 'attempts'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'{stage["key"]}_{time.time_ns()}'
    attempt.mkdir()
    save(attempt / 'identity.json', {'binding': plan['binding'], 'stage': stage})
    start, histories, active, handles = time.perf_counter(), [], [], []
    admission, finished, pending = [], [], list(CONFIG['roots'])
    timed_start = None
    try:
        def admission_pulse():
            replace_status(attempt / 'pulse.json', {'active_seconds': time.perf_counter()-start, 'waiting_for_memory': True})
            replace_status(output / 'progress.json', {'binding': plan['binding'], 'stage': stage['key'],
                           'attempt': attempt.name, 'waiting_for_memory': True, 'pid': os.getpid()})
        admit(output, stage['workers'], remaining, admission, admission_pulse)
        save(attempt / 'admission.json', admission)
        # Full measured boundary includes job serialization, process startup,
        # model/root loading, all work, result writes, process exit and receipts.
        timed_start = time.perf_counter()
        max_active, overlap_seconds, previous_count, previous_time = 0, 0., 0, timed_start
        while pending or active:
            elapsed = time.perf_counter() - start
            if elapsed >= remaining:
                raise RuntimeError('Four-hour cumulative active limit reached')
            now = time.perf_counter()
            if previous_count == 2:
                overlap_seconds += now - previous_time
            previous_time = now
            own = memory()
            group = own['process_rss_bytes']
            group_private = own['process_private_bytes']
            child_memory = []
            for child in active:
                if child['process'].poll() is None:
                    try:
                        info = psutil.Process(child['process'].pid).memory_info()
                    except psutil.NoSuchProcess:
                        continue  # The exit/receipt loop below still validates this child.
                    child_memory.append({'pid': child['process'].pid, 'rss': info.rss, 'peak_rss': getattr(info, 'peak_wset', info.rss), 'private_bytes': info.private})
                    group += info.rss
                    group_private += info.private
                    if max(info.rss, getattr(info, 'peak_wset', info.rss)) > CONFIG['maximum_worker_rss_bytes']:
                        raise RuntimeError('Child memory cap exceeded')
                    if info.private > CONFIG['maximum_worker_private_bytes']:
                        raise RuntimeError('Child private commitment cap exceeded')
            operational_guard(output, group_rss=group, group_private=group_private)
            histories.append({'elapsed': elapsed, 'parent': own, 'children': child_memory, 'group_rss_bytes': group, 'group_private_bytes': group_private})
            for child in list(active):
                code = child['process'].poll()
                if code is None:
                    continue
                for stream in child['streams']:
                    stream.close()
                active.remove(child)
                if code:
                    raise RuntimeError(f'Worker root{child["root"]} exited{code}; inspect preserved log')
                ref = artifact(attempt / f'root_{child["root"]}_result.json')
                row = read(verify(ref))
                if row['binding'] != plan['binding'] or row['root'] != child['root'] or row['backend'] != stage['backend'] or row['exact'] is not True:
                    raise ValueError('Mismatched worker receipt')
                finished.append(ref)
            while pending and len(active) < stage['workers']:
                index = pending.pop(0)
                path = attempt / f'root_{index}_job.json'
                save(path, {'output': str(output), 'root': index, 'backend': stage['backend'], 'parent': process_identity(os.getpid()),
                            'remaining_seconds': remaining - (time.perf_counter()-start)})
                stdout = (attempt / f'root_{index}.stdout').open('x', encoding='utf-8')
                stderr = (attempt / f'root_{index}.stderr').open('x', encoding='utf-8')
                handles += [stdout, stderr]
                process = subprocess.Popen([sys.executable, '-m', 'tools.simulation_speed_validation_128_v2.collector', 'worker', '--job', str(path)],
                                           stdout=stdout, stderr=stderr, **process_options())
                active.append({'root': index, 'process': process, 'streams': [stdout, stderr]})
                save(attempt / f'root_{index}_process.json', process_identity(process.pid))
            max_active = max(max_active, len(active))
            previous_count = len(active)
            replace_status(attempt / 'pulse.json', {'active_seconds': time.perf_counter()-start,
                           'finished_roots': len(finished), 'active_roots': [c['root'] for c in active]})
            replace_status(output / 'progress.json', {'binding': plan['binding'], 'stage': stage['key'],
                           'attempt': attempt.name, 'finished_roots': len(finished), 'active_roots': [c['root'] for c in active]})
            if pending or active:
                time.sleep(.5)
        finished.sort(key=lambda ref: CONFIG['roots'].index(read(verify(ref))['root']))
        records = [read(verify(ref)) for ref in finished]
        counts = sum(r['continuations'] for r in records)
        turns = sum(r['turns'] for r in records)
        expected = plan['counts']['checked_selection'] + plan['counts']['checked_assessment'] if stage['backend'] == 'checked' else CONFIG['selection_continuations']
        if counts != expected or (stage['backend'] != 'checked' and turns != CONFIG['selection_turns']):
            raise ValueError('Incomplete exact workload')
        # Final group sample remains inside the net-time boundary.
        final = operational_guard(output)
        histories.append({'elapsed': time.perf_counter()-start, 'parent': final, 'children': [], 'group_rss_bytes': final['process_rss_bytes'], 'group_private_bytes': final['process_private_bytes']})
        save(attempt / 'resources.json', histories)
        payload = {'binding': plan['binding'], 'stage': stage, 'roots': finished, 'exact': True,
                   'continuations': counts, 'turns': turns, 'tensor_checks': sum(r['tensor_checks'] for r in records),
                   'worker_work_wall_sum': sum(r['work_timing']['wall'] for r in records),
                   'worker_work_cpu_sum': sum(r['work_timing']['cpu'] for r in records),
                   'peak_sampled_group_rss_bytes': max(s['group_rss_bytes'] for s in histories),
                   'peak_sampled_group_private_bytes': max(s['group_private_bytes'] for s in histories),
                   'physical_pressure': pressure_summary([{'elapsed': s['elapsed'], **s['parent']} for s in histories]),
                   'maximum_active_children': max_active, 'sampled_two_process_overlap_seconds': overlap_seconds,
                   'two_work_interval_overlap_seconds': interval_overlap([r['work_interval'] for r in records]),
                   'resources': artifact(attempt / 'resources.json'),
                   'net_wall_seconds': time.perf_counter()-timed_start, 'active_seconds': time.perf_counter()-start,
                   'net_boundary': 'before job serialization through all worker exits, result verification and resource write; excludes admission and final summary write'}
        payload['receipt_sha256'] = digest(payload)
        save(attempt / 'result.json', payload)
        return artifact(attempt / 'result.json')
    except BaseException as error:
        for child in active:
            if child['process'].poll() is None:
                child['process'].terminate()
        for child in active:
            child['process'].wait(timeout=30)
        if not (attempt / 'admission.json').exists():
            save(attempt / 'admission.json', admission)
        save(attempt / 'failure.json', {'binding': plan['binding'], 'stage': stage, 'error': repr(error),
                                      'active_seconds': time.perf_counter()-start, 'partial_memory': histories})
        raise
    finally:
        for stream in handles:
            if not stream.closed:
                stream.close()


def interval_overlap(intervals):
    points = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    active, previous, overlap = 0, None, 0.
    for stamp, delta in points:
        if previous is not None and active >= 2:
            overlap += stamp - previous
        active += delta
        previous = stamp
    return overlap


def completed_stages(output, plan):
    completed = {}
    for path in sorted((output / 'attempts').glob('*/result.json')):
        row = read(path)
        if row['binding'] != plan['binding'] or row['exact'] is not True or row['stage'] not in SCHEDULE:
            raise ValueError('Invalid completed stage')
        if row.get('receipt_sha256') != digest({k: v for k, v in row.items() if k != 'receipt_sha256'}):
            raise ValueError('Completed stage receipt changed')
        records = [read(verify(ref)) for ref in row['roots']]
        if [r['root'] for r in records] != CONFIG['roots']:
            raise ValueError('Completed stage lacks the exact ordered six roots')
        checked = row['stage']['backend'] == 'checked'
        for item, result in zip(plan['selected'], records, strict=True):
            expected_rows = [{'split': split, 'world_index': world, 'candidate_index': candidate, **expected}
                             for split, world, candidate, _, expected in cases(read(verify(item['root'])), checked)]
            if (result['binding'] != plan['binding'] or result['backend'] != row['stage']['backend'] or result['exact'] is not True
                    or result['rows'] != expected_rows or result['rows_digest'] != digest(expected_rows)
                    or result['continuations'] != len(expected_rows) or result['turns'] != sum(r['turns'] for r in expected_rows)
                    or result['tensor_checks'] != (sum(max(r['turns']-1,0) for r in expected_rows) if checked else 0)):
                raise ValueError('Malformed or incomplete exact root receipt')
        if (row['continuations'] != sum(r['continuations'] for r in records)
                or row['turns'] != sum(r['turns'] for r in records)
                or row['tensor_checks'] != sum(r['tensor_checks'] for r in records)):
            raise ValueError('Completed stage counts differ from root receipts')
        for ref in row['roots']:
            verify(ref)
        verify(row['resources'])
        key = row['stage']['key']
        if key in completed:
            raise ValueError('Duplicate complete stage; do not select a timing')
        completed[key] = artifact(path)
    return completed


def run(output):
    output, plan = load_plan(output)
    if (output / 'final.json').exists():
        raise FileExistsError(output / 'final.json')
    # An OS lock is released automatically after a crash; no stale PID guess.
    lock = (output / 'run.lock').open('a+b')
    import msvcrt
    if lock.tell() == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        require_no_prior_workers(output)
        from tools.simulation_speed_validation_128 import collector as previous
        previous.require_no_prior_workers(PREVIOUS)
        complete = completed_stages(output, plan)
        for stage in SCHEDULE:
            if stage['key'] not in complete:
                remaining = CONFIG['maximum_active_seconds'] - active_used(output)
                if remaining <= 0:
                    raise RuntimeError('Cumulative time budget exhausted')
                complete[stage['key']] = stage_attempt(output, plan, stage, remaining)
        receipt = {'binding': plan['binding'], 'complete': True, 'all_exact': True,
                   'stages': [complete[s['key']] for s in SCHEDULE], 'active_seconds': active_used(output),
                   'previous_active_seconds': plan['resource_amendment']['previous_active_seconds'],
                   'timing_caveat': 'All matched timing blocks retained. Host physical-memory pressure may cause paging or competition; net wall results are conditional on this host load.',
                   'production_enabled': False, 'scope': 'saved selection batches only; no fresh sources/proposals/independent assessment timing'}
        save(output / 'final.json', receipt)
        replace_status(output / 'progress.json', {'binding': plan['binding'], 'complete': True, 'stage': 'complete'})
        return receipt
    finally:
        lock.close()


def diagnostic(job_path):
    """Tiny no-model subprocess fixture for hidden launch and controller death."""
    import ctypes
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    job = read(job_path)
    info = {'identity': process_identity(os.getpid()), 'console_window': int(ctypes.windll.kernel32.GetConsoleWindow()),
            'priority': int(psutil.Process().nice()), 'threads': {k: os.environ.get(k) for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    save(job['ready'], info)
    if 'parent' not in job:
        return
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < 12:
            require_parent(job['parent'])
            time.sleep(.1)
        raise RuntimeError('Diagnostic lease did not expire')
    except RuntimeError as error:
        save(job['result'], {'lease_expired': str(error) == 'Validation controller lease expired'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'prepare', 'run', 'resume', 'status', 'worker', 'diagnostic'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    try:
        if args.command in ('worker', 'diagnostic'):
            if args.job is None: parser.error('--job required')
            {'worker': worker, 'diagnostic': diagnostic}[args.command](args.job)
            return
        if args.output is None: parser.error('--output required')
        if args.command == 'status':
            out = safe_output(args.output)
            result = read(out / 'final.json') if (out / 'final.json').exists() else read(out / 'progress.json')
        else:
            result = {'preflight': preflight, 'prepare': prepare, 'run': run, 'resume': run}[args.command](args.output)
        print(json.dumps(result), flush=True)
    except BaseException:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
