"""Full historical collector acceptance; no production experiment is modified."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.simulation_speed_validation_128_v2 import collector as util

read, save, artifact, verify, digest = util.read, util.save, util.artifact, util.verify, util.digest
OLD = ROOT / 'outputs/large_corpus_optimization_128'
VALIDATION = ROOT / 'outputs/simulation_speed_validation_128_02'
MAIN = ROOT / 'outputs/structured_proposer_128'
MODULE = 'tools.simulation_backend_integration_128.integration'
CONFIG = {
    'historical_roots': [517, 251, 382],
    'selection': 'previously_frozen_maximum_canonical_support_root_each_phase',
    'stage': 'test', 'worlds_per_block': 128, 'workers': 1, 'threads': 1,
    'maximum_active_seconds': 7200,
    'minimum_commit_headroom_bytes': 6 * 1024**3,
    'maximum_worker_memory_bytes': 2 * 1024**3,
    'maximum_parent_memory_bytes': 256 * 1024**2,
    'minimum_disk_free_bytes': 5 * 1024**3,
    'physical_memory_policy': 'telemetry_only_no_stop_or_pause',
    'interruption': 'after_selection_seal_before_assessment_on_middle_fast_root',
    'resume_unit': 'complete_authenticated_job_only_partial_root_restarts_whole_in_new_attempt',
    'production_enabled': False,
}
# Fixed before any new fixture outcomes. Alternate order across phases; this is
# an integration check, not another powered throughput benchmark.
SCHEDULE = [
    {'key': 'reference_0', 'kind': 'collect', 'backend': 'reference', 'index': 0},
    {'key': 'fast035_0', 'kind': 'collect', 'backend': 'fast035', 'index': 0},
    {'key': 'interrupt_1', 'kind': 'interrupt', 'backend': 'fast035', 'index': 1},
    {'key': 'fast035_1', 'kind': 'collect', 'backend': 'fast035', 'index': 1},
    {'key': 'reference_1', 'kind': 'collect', 'backend': 'reference', 'index': 1},
    {'key': 'reference_2', 'kind': 'collect', 'backend': 'reference', 'index': 2},
    {'key': 'fast035_2', 'kind': 'collect', 'backend': 'fast035', 'index': 2},
    *[{'key': f'audit_{i}', 'kind': 'audit', 'backend': 'reference', 'index': i}
      for i in range(3)],
    {'key': 'summary', 'kind': 'summary', 'backend': 'reference', 'index': None},
]


def safe_output(path):
    path = Path(path).resolve()
    if path.parent != ROOT / 'outputs' or not path.name.startswith('simulation_backend_integration_128_'):
        raise ValueError('Use a separate direct outputs/simulation_backend_integration_128_* directory')
    return path


def code_identity():
    # Freeze all imported repository code conservatively, including unchanged
    # original collector/auditor and recovery. Outputs and handoffs are excluded.
    paths = []
    for folder in ('src/splendor_ai', 'scripts/analysis', 'tools'):
        paths.extend((ROOT / folder).rglob('*.py'))
    paths.append(Path(__file__).with_name('PROTOCOL.md'))
    return [artifact(p) for p in sorted(set(paths))]


def phase_allowed(stage, complete=False):
    return complete is True or stage in ('learning_data_validation', 'learning_fits', 'learning_fit_audits')


def operational_guard(output, role='parent', admission=False):
    if (Path(output) / 'abort.json').exists():
        raise RuntimeError('Integration abort requested; preserve all evidence')
    stage = read(MAIN / 'progress.json').get('stage')
    complete = read(MAIN / 'final_status.json').get('complete') is True if (MAIN / 'final_status.json').exists() else False
    if not phase_allowed(stage, complete):
        raise RuntimeError('Do not overlap main source generation/evaluation: ' + str(stage))
    info = util.memory()
    cap = CONFIG['maximum_' + role + '_memory_bytes']
    if max(info['process_rss_bytes'], info.get('process_peak_rss_bytes') or 0,
           info['process_private_bytes']) > cap:
        raise RuntimeError('Integration process exceeded its own memory allocation')
    reserve = CONFIG['minimum_commit_headroom_bytes']
    if admission:
        reserve += CONFIG['maximum_worker_memory_bytes'] + CONFIG['maximum_parent_memory_bytes']
    if info['commit_headroom_bytes'] < reserve:
        raise RuntimeError('Insufficient system commitment headroom: ' + str(info['commit_headroom_bytes']))
    if shutil.disk_usage(ROOT).free < CONFIG['minimum_disk_free_bytes']:
        raise RuntimeError('Five-GiB disk reserve reached')
    return {'epoch': time.time(), 'main_stage': stage, **info}


def prepare(output, checks):
    output = safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        raise FileExistsError(output / 'plan.json')
    check = read(checks)
    if not check['passed'] or check['code'] != code_identity():
        raise ValueError('Matching successful focused preflight required')
    _, validation = util.load_plan(VALIDATION)
    stages = util.completed_stages(VALIDATION, validation)
    if len(stages) != 9 or not read(VALIDATION / 'final.json')['all_exact']:
        raise ValueError('Completed authenticated speed validation required')
    selected = [x for x in validation['selected'] if x['support_band'] == 'maximum']
    if [x['spec']['index'] for x in selected] != CONFIG['historical_roots']:
        raise ValueError('Historical metadata selection changed')
    specs = []
    for i, item in enumerate(selected):
        record = read(verify(item['root']))
        if record['spec'] != item['spec'] or digest(record['snapshot']) != item['spec']['snapshot_sha256']:
            raise ValueError('Historical snapshot identity changed')
        specs.append({**item['spec'], 'index': i, 'fixture_snapshot': record['snapshot']})
    registry = read(OLD / 'models_lock.json')
    for ref in registry['models'].values():
        verify(ref)
    protected = [artifact(p) for folder in ('structured_proposer_128', 'structured_proposer_128_recovery_01')
                 for p in sorted((ROOT / 'tools' / folder).glob('*')) if p.is_file()]
    protected += [artifact(MAIN / 'freeze.json'), artifact(MAIN / 'launch_seal.json')]
    plan = {'config': CONFIG, 'schedule': SCHEDULE, 'selected': selected, 'specs': specs,
            'models': registry['models'], 'registry': artifact(OLD / 'models_lock.json'),
            'validation_plan': artifact(VALIDATION / 'plan.json'),
            'validation_final': artifact(VALIDATION / 'final.json'),
            'checkpoint': validation['checkpoint'], 'preflight': artifact(checks),
            'code': code_identity(), 'protected_main': protected, 'epoch': time.time(),
            'fresh_current_outcomes_read': False, 'production_enabled': False}
    plan['binding'] = digest(plan)
    save(output / 'plan.json', plan)
    return {'binding': plan['binding'], 'historical_roots': CONFIG['historical_roots']}


def preflight(output):
    output = safe_output(output)
    operational_guard(output, admission=True)
    attempt = output / ('preflight_' + str(time.time_ns()))
    attempt.mkdir(parents=True)
    identity = code_identity()
    start = time.perf_counter()
    with (attempt / 'stdout.txt').open('x') as stdout, (attempt / 'stderr.txt').open('x') as stderr:
        result = subprocess.run([sys.executable, '-m', 'pytest', str(Path(__file__).parent),
                                 '-q', '-p', 'no:cacheprovider', '--basetemp', str(attempt / 'pytest_tmp')],
                                stdout=stdout, stderr=stderr, timeout=180, **util.process_options())
    receipt = {'passed': result.returncode == 0 and identity == code_identity(),
               'code': identity, 'elapsed_seconds': time.perf_counter()-start,
               'stdout': artifact(attempt / 'stdout.txt'), 'stderr': artifact(attempt / 'stderr.txt')}
    save(attempt / 'checks.json', receipt)
    if not receipt['passed']:
        raise RuntimeError('Preflight failed; see ' + str(attempt))
    return {'passed': True, 'checks': str(attempt / 'checks.json'), 'elapsed_seconds': receipt['elapsed_seconds']}


def launch(output):
    output, plan = load_plan(output)
    with exclusive(output):
        no_prior_workers(output)
        operational_guard(output, admission=True)
        runtime = output / ('runtime_' + str(time.time_ns()))
        runtime.mkdir()
        from splendor_ai.alphazero.windowless_workers_160 import detached_launch
        process = detached_launch([sys.executable, '-u', '-m', MODULE, 'run', '--output', str(output)],
                                  runtime / 'stdout.txt', runtime / 'stderr.txt', ROOT)
        save(runtime / 'process.json', {**process, **util.process_identity(process['pid']),
                                      'binding': plan['binding'], 'epoch': time.time()})
    return {'launched': True, 'pid': process['pid'], 'runtime': str(runtime), 'binding': plan['binding']}


def load_plan(output):
    output = safe_output(output)
    plan = read(output / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'binding'}) != plan['binding']:
        raise ValueError('Plan binding changed')
    if plan['config'] != CONFIG or plan['schedule'] != SCHEDULE or plan['code'] != code_identity():
        raise ValueError('Frozen code or configuration changed')
    for key in ('registry', 'validation_plan', 'validation_final', 'checkpoint', 'preflight'):
        verify(plan[key])
    for ref in [*plan['models'].values(), *plan['protected_main'], *[r['root'] for r in plan['selected']]]:
        verify(ref)
    return output, plan


def canonical_record(record):
    """Drop only enumerated operational fields; preserve all other science."""
    value = deepcopy(record)
    for key in ('time', 'timing', 'assessment_started_at_ns', 'selection_artifact'):
        del value[key]
    for proposal in value['proposals'].values():
        del proposal['timing']
    for block in ('selection', 'assessment'):
        del value[block]['timing']
    for key in ('time', 'created_at_ns'):
        del value['selection_lock'][key]
    # Actual raw selection digest is independently checked before normalization.
    value['selection_lock']['selection_sha256'] = digest(value['selection'])
    if 'proof_timing' in value:
        del value['proof_timing']['wall']; del value['proof_timing']['cpu']
    return value


def authenticate_root(ref, spec, binding, base):
    record = read(verify(ref))
    if Path(verify(ref)) != Path(base) / 'test/roots' / f'state_{spec["index"]:04d}.json':
        raise ValueError('Wrong root path')
    if record['manifest'] != binding or record['stage'] != 'test' or record['spec'] != spec or 'failure' in record:
        raise ValueError('Invalid completed root')
    lock = record['selection_lock']
    if lock['selection_sha256'] != digest(record['selection']):
        raise ValueError('Raw selection digest differs')
    seal = read(verify(record['selection_artifact']))
    if seal['selection_lock'] != lock or seal['selection'] != record['selection']:
        raise ValueError('Immutable selection seal differs')
    if lock['identity_digest'] != digest(seal['identity']):
        raise ValueError('Selection identity differs')
    if not lock['created_at_ns'] <= record['assessment_started_at_ns']:
        raise ValueError('Selection not sealed before assessment')
    if len(record['selection']['worlds']) != 128 or len(record['assessment']['worlds']) != 128:
        raise ValueError('Incomplete world blocks')
    if set(record['selection']['world_seeds']) & set(record['assessment']['world_seeds']):
        raise ValueError('World blocks overlap')
    return record


class WorkerGuard:
    def __init__(self, job, pulse):
        self.job, self.pulse, self.start, self.last, self.samples = job, pulse, time.perf_counter(), -1., []
        self(force=True)

    def __call__(self, force=False):
        elapsed = time.perf_counter() - self.start
        if elapsed >= self.job['remaining_seconds']:
            raise RuntimeError('Integration active time ceiling reached')
        if not force and elapsed - self.last < 1:
            return
        util.require_parent(self.job['parent'])
        info = operational_guard(self.job['output'], 'worker')
        self.last = elapsed
        if force or not self.samples or elapsed - self.samples[-1]['elapsed'] >= 10:
            self.samples.append({'elapsed': elapsed, **info})
        util.replace_status(self.pulse, {'elapsed': elapsed, 'pid': os.getpid(), 'memory': info})


@contextmanager
def guarded_budget(common, guard):
    original = common.budget_boundary
    def boundary(base=None):
        guard(); original(base)
    common.budget_boundary = boundary
    try:
        yield
    finally:
        common.budget_boundary = original


class PlannedInterruption(RuntimeError):
    pass


@contextmanager
def audit_receipt_destination(common, original_path, destination):
    """Route only the auditor's final receipt into this attempt, not its inputs.

    Replay/calculations stay unmodified. A failed audit can therefore be rerun
    fully in a new attempt even if its earlier final receipt was already saved.
    """
    original_save, original_artifact = common.old.save_new, common.artifact
    original_path, destination = Path(original_path).resolve(), Path(destination).resolve()
    if original_path == destination or original_path.exists() or destination.exists():
        raise ValueError('Audit output must be new and separately isolated')
    counts = {'writes': 0, 'artifacts': 0}
    def redirected_save(path, payload):
        matched = Path(path).resolve() == original_path
        if matched:
            counts['writes'] += 1
            if counts['writes'] != 1:
                raise ValueError('Repeated audit receipt write')
        target = destination if matched else path
        return original_save(target, payload)
    def redirected_artifact(path, **metadata):
        matched = Path(path).resolve() == original_path
        if matched:
            counts['artifacts'] += 1
            if counts['artifacts'] != 1:
                raise ValueError('Repeated audit receipt artifact')
        target = destination if matched else path
        return original_artifact(target, **metadata)
    common.old.save_new, common.artifact = redirected_save, redirected_artifact
    try:
        yield
    finally:
        common.old.save_new, common.artifact = original_save, original_artifact
    if counts != {'writes': 1, 'artifacts': 1}:
        raise ValueError('Audit did not save and return exactly one receipt')


def worker(job_path):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    started, cpu = time.perf_counter(), time.process_time()
    job = read(job_path)
    output, plan = load_plan(job['output'])
    task = job['task']
    if task not in SCHEDULE:
        raise ValueError('Unscheduled job')
    attempt = Path(job_path).resolve().parent
    if attempt.parent != output / 'attempts' or Path(job_path).name != 'job.json':
        raise ValueError('Worker job must be inside its isolated attempt directory')
    guard = WorkerGuard(job, attempt / 'pulse.json')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from tools.large_corpus_optimization_128 import common as c, engine, verify as audit
    from tools.simulation_backend_integration_128.backend import use_backend
    binding, specs = plan['binding'], plan['specs']
    base = attempt / 'data'
    base.mkdir()
    registry = {'manifest': binding, 'models': plan['models']}
    save(base / 'models_lock.json', registry)
    sample = audit.sample_plan(specs, 'test', binding)
    if not all(x['stress'] and all(len(x['replay'][b]) == 128 for b in ('selection', 'assessment')) for x in sample['selections']):
        raise ValueError('Small integration panel must receive full independent replay')
    save(base / 'test/audit_sample.json', sample)
    guard(force=True)
    loading = time.perf_counter()
    engine.initialize_models(base=base, registry=registry)
    load_seconds = time.perf_counter() - loading
    guard(force=True)
    original_raw, original_block = c.old._RAW, engine.run_block
    result = {'binding': binding, 'task': task, 'base': str(base), 'pid': os.getpid(),
              'loaded_models': len(engine._MODELS), 'loading_seconds': load_seconds,
              'console_window': int(__import__('ctypes').windll.kernel32.GetConsoleWindow())}
    if result['loaded_models'] != 9 or result['console_window'] != 0:
        raise ValueError('Worker must load all nine models and have no console window')
    with guarded_budget(c, guard):
        if task['kind'] in ('collect', 'interrupt'):
            spec = specs[task['index']]
            original_save = c.old.save_new
            def fault(path, payload):
                original_save(path, payload)
                if Path(path) == engine.selection_path('test', spec['index'], base):
                    raise PlannedInterruption('Deliberate interruption after durable selection seal')
            if task['kind'] == 'interrupt':
                c.old.save_new = fault
            try:
                with use_backend(engine, task['backend'], boundary=guard) as usage:
                    try:
                        ref = engine.collect('test', spec, binding, base)
                    except PlannedInterruption:
                        if task['kind'] != 'interrupt':
                            raise
                    else:
                        if task['kind'] == 'interrupt':
                            raise ValueError('Interruption did not fire')
                result['backend_usage'] = dict(usage)
            finally:
                c.old.save_new = original_save
            expected_blocks = 1 if task['kind'] == 'interrupt' else 2
            if usage['blocks_attempted'] != expected_blocks or usage['blocks_completed'] != expected_blocks:
                raise ValueError('Backend did not run the expected collection blocks')
            if task['kind'] == 'interrupt':
                root = engine.root_path('test', spec['index'], base)
                record = read(root)
                seal = engine.selection_path('test', spec['index'], base)
                if 'failure' not in record or 'assessment' in record or not seal.exists() or len(record['selection']['worlds']) != 128:
                    raise ValueError('Interrupted evidence did not preserve full selection and omit assessment')
                # The old collector correctly refuses overwriting its one-shot root.
                try:
                    engine.collect('test', spec, binding, base)
                except ValueError:
                    pass
                else:
                    raise ValueError('Collector overwrote interrupted evidence')
                result.update(expected_interruption=True, partial_root=artifact(root), selection=artifact(seal))
            else:
                record = authenticate_root(ref, spec, binding, base)
                if usage['candidate_worlds_completed'] != ref['continuations']:
                    raise ValueError('Backend candidate-world accounting differs')
                if [b['candidate_indices'] for b in usage['blocks']] != [record[b]['candidate_indices'] for b in ('selection', 'assessment')] or any(b['worlds'] != 128 for b in usage['blocks']):
                    raise ValueError('Backend block identities differ')
                result.update(root=ref, science_sha256=digest(canonical_record(record)),
                              row=engine.statistics_row(record),
                              continuations=ref['continuations'], collection_seconds=record['timing']['wall'])
        elif task['kind'] == 'audit':
            done = completed(output, plan)
            collected = done[f'fast035_{task["index"]}']['result']
            original_base = Path(collected['base'])
            relative = Path('test/audits') / f'state_{task["index"]:04d}.json'
            with audit_receipt_destination(c, original_base / relative, base / relative):
                result['audit'] = audit.audit('test', specs[task['index']], collected['root'], binding, original_base)
            if verify(result['audit']) != (base / relative).resolve():
                raise ValueError('Auditor returned wrong attempt receipt')
            receipt = read(verify(result['audit']))
            if not receipt['passed'] or not receipt['full_inner_replay']:
                raise ValueError('Independent complete audit missing')
            result['replayed_continuations'] = receipt['replayed_continuations']
        elif task['kind'] == 'summary':
            done = completed(output, plan)
            refs = [done[f'audit_{i}']['result']['audit'] for i in range(3)]
            result['summary'] = audit.summarize(refs, specs, binding, base)
            if result['summary']['statistics'] is not None or not result['summary']['full_inner_replay']:
                raise ValueError('Invalid small-panel summary')
        else:
            raise ValueError('Unknown task')
    if c.old._RAW is not original_raw or engine.run_block is not original_block:
        raise ValueError('Backend leaked beyond continuation scope')
    guard(force=True)
    result.update(restored_reference=True, memory=guard.samples,
                  wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
    save(attempt / 'result.json', result)


def completed(output, plan):
    found = {}
    for task in SCHEDULE:
        path = Path(output) / 'completed' / (task['key'] + '.json')
        if not path.exists():
            continue
        receipt = read(path)
        if receipt['binding'] != plan['binding'] or receipt['task'] != task or receipt['exit_code'] != 0:
            raise ValueError('Invalid job completion')
        result = read(verify(receipt['result']))
        if result['binding'] != plan['binding'] or result['task'] != task or not result['restored_reference']:
            raise ValueError('Changed job result')
        if task['kind'] == 'collect':
            root = authenticate_root(result['root'], plan['specs'][task['index']], plan['binding'], result['base'])
            if digest(canonical_record(root)) != result['science_sha256']:
                raise ValueError('Changed scientific record')
        elif task['kind'] == 'interrupt':
            verify(result['partial_root']); verify(result['selection'])
            if not result['expected_interruption']:
                raise ValueError('Missing prescribed interruption')
        elif task['kind'] == 'audit':
            replay = read(verify(result['audit']))
            if not replay['passed'] or not replay['full_inner_replay']:
                raise ValueError('Incomplete independent replay')
            for key in ('root', 'selection_artifact', 'sample_artifact'):
                verify(replay[key])
        found[task['key']] = {'receipt': receipt, 'result': result}
    for i in range(3):
        keys = [f'{backend}_{i}' for backend in ('reference', 'fast035')]
        if all(k in found for k in keys):
            left, right = [found[k]['result'] for k in keys]
            if left['science_sha256'] != right['science_sha256'] or left['row'] != right['row']:
                raise ValueError('Reference/fast scientific outputs differ')
    return found


def active_used(output):
    total = 0.
    for path in (Path(output) / 'attempts').glob('*/start.json'):
        end = path.with_name('end.json')
        total += read(end)['seconds'] if end.exists() else max(0., time.time() - read(path)['epoch'])
    return total


def no_prior_workers(output):
    for path in (Path(output) / 'attempts').glob('*/process.json'):
        if util.identity_alive(read(path)):
            raise RuntimeError('Prior integration worker still alive')
    import psutil
    for process in psutil.process_iter(['name']):
        if not (process.info['name'] or '').lower().startswith('python'):
            continue
        try:
            command = process.cmdline()
            if len(command) == 6 and command[1:5] == ['-m', MODULE, 'worker', '--job'] and Path(command[5]).resolve().is_relative_to(Path(output) / 'attempts'):
                raise RuntimeError('Unreceipted integration worker is still alive')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


@contextmanager
def exclusive(output):
    import msvcrt
    path = Path(output) / 'run.lock'
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def run_job(output, plan, task):
    operational_guard(output, admission=True)
    remaining = CONFIG['maximum_active_seconds'] - active_used(output)
    if remaining <= 0:
        raise RuntimeError('Integration active time exhausted')
    attempt = output / 'attempts' / (task['key'] + '_' + str(time.time_ns()))
    attempt.mkdir(parents=True)
    job = {'output': str(output), 'task': task, 'parent': util.process_identity(os.getpid()),
           'remaining_seconds': remaining}
    save(attempt / 'job.json', job)
    save(attempt / 'start.json', {'epoch': time.time(), 'binding': plan['binding'], 'task': task})
    start = time.perf_counter()
    process, samples = None, []
    already_completed = len(completed(output, plan))
    try:
        with (attempt / 'stdout.txt').open('x') as stdout, (attempt / 'stderr.txt').open('x') as stderr:
            process = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--job', str(attempt / 'job.json')],
                                       stdout=stdout, stderr=stderr, **util.process_options())
            save(attempt / 'process.json', util.process_identity(process.pid))
            last_sample = -10.
            while process.poll() is None:
                info = operational_guard(output)
                import psutil
                try:
                    child = psutil.Process(process.pid).memory_info()
                    child_memory = {'rss': child.rss, 'peak_rss': getattr(child, 'peak_wset', 0),
                                    'private': getattr(child, 'private', 0)}
                    if max(child_memory.values()) > CONFIG['maximum_worker_memory_bytes']:
                        raise RuntimeError('Worker exceeded memory cap during its lifecycle')
                except psutil.NoSuchProcess:
                    child_memory = None
                if time.perf_counter() - start >= remaining:
                    raise RuntimeError('Integration active time ceiling reached')
                if time.perf_counter() - start - last_sample >= 10:
                    last_sample = time.perf_counter() - start
                    samples.append({'elapsed': last_sample, 'child_memory': child_memory, **info})
                util.replace_status(output / 'progress.json', {'binding': plan['binding'], 'stage': task['key'],
                                    'pid': os.getpid(), 'worker_pid': process.pid, 'epoch': time.time(),
                                    'completed': already_completed, 'total': len(SCHEDULE)})
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError('Integration worker failed; preserved ' + str(attempt))
        receipt = {'binding': plan['binding'], 'task': task, 'exit_code': process.returncode,
                   'net_wall_seconds': time.perf_counter() - start, 'result': artifact(attempt / 'result.json'),
                   'job': artifact(attempt / 'job.json'), 'parent_memory': samples}
        # Successful worker exit plus authenticated full result is the only reuse unit.
        save(output / 'completed' / (task['key'] + '.json'), receipt)
        completed(output, plan)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        save(attempt / 'end.json', {'seconds': time.perf_counter() - start, 'epoch': time.time()})


def finish(output, plan):
    done = completed(output, plan)
    if set(done) != {x['key'] for x in SCHEDULE}:
        raise ValueError('Incomplete integration')
    pairs = []
    for i in range(3):
        reference, fast = (done[f'{backend}_{i}'] for backend in ('reference', 'fast035'))
        if reference['result']['science_sha256'] != fast['result']['science_sha256'] or reference['result']['row'] != fast['result']['row']:
            raise ValueError('Reference/fast scientific outputs differ')
        pairs.append({'index': i, 'historical_index': CONFIG['historical_roots'][i], 'exact': True,
                      'continuations': fast['result']['continuations'],
                      'reference_net_seconds': reference['receipt']['net_wall_seconds'],
                      'fast_net_seconds': fast['receipt']['net_wall_seconds']})
    reference_seconds = sum(x['reference_net_seconds'] for x in pairs)
    fast_seconds = sum(x['fast_net_seconds'] for x in pairs)
    return {'binding': plan['binding'], 'complete': True, 'all_exact': True, 'pairs': pairs,
            'independent_full_replay': True, 'partial_evidence_preserved': True,
            'net_reference_seconds': reference_seconds, 'net_fast_seconds': fast_seconds,
            'net_wall_ratio': reference_seconds / fast_seconds,
            'net_time_saved_fraction': 1 - fast_seconds / reference_seconds,
            'time_comparison': 'one fixed three-root panel; host-load conditional; no new speed certainty claim',
            'active_seconds': active_used(output),
            'optional_backend_accepted': fast_seconds < reference_seconds,
            'production_enabled': False, 'main_unchanged': True,
            'summary': done['summary']['result']['summary'],
            'completions': [artifact(output / 'completed' / (x['key'] + '.json')) for x in SCHEDULE]}


def run(output):
    output, plan = load_plan(output)
    with exclusive(output):
        no_prior_workers(output)
        done = completed(output, plan)
        # Explicitly log the recovery planner's authenticated skip decisions.
        save(output / 'sessions' / (str(time.time_ns()) + '.json'),
             {'pid': os.getpid(), 'binding': plan['binding'], 'reused': list(done),
              'pending': [x['key'] for x in SCHEDULE if x['key'] not in done], 'epoch': time.time()})
        try:
            for task in SCHEDULE:
                if task['key'] not in done:
                    run_job(output, plan, task)
                    done = completed(output, plan)
            result = finish(output, plan)
            load_plan(output)  # Includes protected main files, after all work.
            if not (output / 'final.json').exists():
                save(output / 'final.json', result)
            util.replace_status(output / 'progress.json', {'stage': 'complete', 'completed': len(SCHEDULE),
                                'total': len(SCHEDULE), 'epoch': time.time(), 'binding': plan['binding']})
            return {'complete': True, 'all_exact': result['all_exact'], 'net_wall_ratio': result['net_wall_ratio']}
        except BaseException:
            save(output / 'failures' / (str(time.time_ns()) + '.json'), {'traceback': traceback.format_exc(), 'epoch': time.time()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preflight', 'prepare', 'launch', 'run', 'worker', 'status'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job', type=Path)
    parser.add_argument('--checks', type=Path)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.job); return
    if args.command == 'preflight':
        result = preflight(args.output)
    elif args.command == 'prepare':
        result = prepare(args.output, args.checks)
    elif args.command == 'launch':
        result = launch(args.output)
    elif args.command == 'run':
        result = run(args.output)
    else:
        path = safe_output(args.output)
        result = read(path / ('final.json' if (path / 'final.json').exists() else 'progress.json'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
