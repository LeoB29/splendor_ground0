"""Historical real-collector integration with fixed worker placement and order."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.simulation_backend_integration_128 import integration as h
from tools.simulation_backend_diagnosis_128 import topology

util = h.util
read, save, artifact, verify, digest = h.read, h.save, h.artifact, h.verify, h.digest
PREVIOUS = ROOT / 'outputs/simulation_backend_integration_128_01'
MODULE = 'tools.simulation_backend_integration_pinned_128.integration'
CONFIG = {**h.CONFIG, 'repeats': 2, 'logical_processor': 0,
          'placement': 'same_single_P_logical_processor_before_models_for_every_job',
          'order': 'r0_roots_012_reference_fast;r1_roots_210_fast_reference',
          'interruption': 'after_all_12_clean_timings_middle_fast_selection_seal',
          'timing_pool': 'only_12_clean_collections_no_fault_recovery_audit_or_summary'}
SCHEDULE = [
    {'key': f'r{repeat}_{backend}_{index}', 'kind': 'collect', 'backend': backend,
     'index': index, 'repeat': repeat, 'timed': True}
    for repeat, indices, backends in ((0, (0, 1, 2), ('reference', 'fast035')),
                                      (1, (2, 1, 0), ('fast035', 'reference')))
    for index in indices for backend in backends
] + [
    {'key': 'interrupt_1', 'kind': 'interrupt', 'backend': 'fast035', 'index': 1, 'timed': False},
    {'key': 'recovery_1', 'kind': 'collect', 'backend': 'fast035', 'index': 1, 'timed': False},
    *[{'key': f'audit_{i}', 'kind': 'audit', 'backend': 'reference', 'index': i, 'timed': False}
      for i in range(3)],
    {'key': 'summary', 'kind': 'summary', 'backend': 'reference', 'index': None, 'timed': False},
]
_ORIGINAL_GUARD = h.WorkerGuard


def safe_output(path):
    path = Path(path).resolve()
    if path.parent != ROOT / 'outputs' or not path.name.startswith('simulation_backend_integration_pinned_128_'):
        raise ValueError('Use a separate direct outputs/simulation_backend_integration_pinned_128_* directory')
    return path


def code_identity():
    return [artifact(p) for p in sorted(Path(__file__).parent.glob('*.py'))] + [
        artifact(Path(__file__).with_name('PROTOCOL.md'))]


def authenticate_historical_plan(plan):
    if digest({k: v for k, v in plan.items() if k != 'binding'}) != plan['binding']:
        raise ValueError('Historical plan binding changed')
    # Verify the historical listed bytes, not a newly enumerated tools tree.
    for ref in plan['code']:
        verify(ref)
    for key in ('registry', 'validation_plan', 'validation_final', 'checkpoint', 'preflight'):
        verify(plan[key])
    for ref in [*plan['models'].values(), *plan['protected_main'], *[x['root'] for x in plan['selected']]]:
        verify(ref)


def prepare(output, checks):
    output = safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        raise FileExistsError(output / 'plan.json')
    check = read(checks)
    if not check['passed'] or check['code'] != code_identity():
        raise ValueError('Matching successful focused preflight required')
    previous = read(PREVIOUS / 'plan.json')
    authenticate_historical_plan(previous)
    done = h.completed(PREVIOUS, previous)
    final = read(PREVIOUS / 'final.json')
    if len(done) != 11 or not final['complete'] or not final['all_exact']:
        raise ValueError('Completed exact prior integration required')
    mapping = topology.cpu_sets()
    matches = [x for x in mapping if x['group'] == 0 and x['logical_processor_index'] == CONFIG['logical_processor']]
    if len(matches) != 1 or any(x['group'] != 0 for x in mapping) or matches[0]['efficiency_class'] != max(x['efficiency_class'] for x in mapping):
        raise ValueError('Expected single-group CPU 0 in highest efficiency class')
    if [x['spec']['index'] for x in previous['selected']] != CONFIG['historical_roots']:
        raise ValueError('Historical panel changed')
    inherited = {r['path']: r for r in previous['code']}
    inherited.update({r['path']: r for r in (artifact(Path(h.__file__)), artifact(Path(topology.__file__)))})
    plan = {'config': CONFIG, 'schedule': SCHEDULE, 'specs': previous['specs'],
            'selected': previous['selected'], 'models': previous['models'],
            'protected_main': previous['protected_main'], 'inherited_code': list(inherited.values()),
            'prior_plan': artifact(PREVIOUS / 'plan.json'), 'prior_final': artifact(PREVIOUS / 'final.json'),
            'prior_completions': [artifact(PREVIOUS / 'completed' / (k + '.json')) for k in done],
            'topology': mapping, 'target_cpu': matches[0], 'code': code_identity(),
            'preflight': artifact(checks), 'epoch': time.time(),
            'fresh_current_outcomes_read': False, 'production_enabled': False}
    plan['binding'] = digest(plan)
    save(output / 'plan.json', plan)
    return {'binding': plan['binding'], 'jobs': len(SCHEDULE), 'timed_collections': 12,
            'target_cpu': plan['target_cpu']}


def load_plan(output):
    output = safe_output(output)
    plan = read(output / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'binding'}) != plan['binding']:
        raise ValueError('Plan binding changed')
    if plan['config'] != CONFIG or plan['schedule'] != SCHEDULE or plan['code'] != code_identity():
        raise ValueError('Frozen code/configuration changed')
    for ref in [*plan['inherited_code'], *plan['models'].values(), *plan['protected_main'],
                *plan['prior_completions'], plan['prior_plan'], plan['prior_final'], plan['preflight'],
                *[x['root'] for x in plan['selected']]]:
        verify(ref)
    authenticate_historical_plan(read(verify(plan['prior_plan'])))
    return output, plan


@contextmanager
def pin_worker(cpu, receipt_path, *, process=None, current_processor=None):
    """Pin only the calling worker; publish restoration evidence even on error."""
    if process is None:
        import psutil
        process = psutil.Process()
    current_processor = current_processor or topology.current_processor
    before = list(process.cpu_affinity())
    state = {'pid': process.pid, 'target': cpu, 'before': before, 'entered': False,
             'restored': False, 'body_completed': False}
    try:
        if cpu not in before:
            raise ValueError('Requested CPU is outside original own-worker affinity')
        process.cpu_affinity([cpu])
        state['during'] = list(process.cpu_affinity())
        state['entry_processor'] = current_processor()
        if state['during'] != [cpu] or state['entry_processor'] != cpu:
            raise ValueError('Worker did not enter exact prescribed CPU placement')
        state['entered'] = True
        yield state
        state['body_completed'] = True
    finally:
        try:
            process.cpu_affinity(before)
            state['after'] = list(process.cpu_affinity())
            state['restored'] = state['after'] == before
            if not state['restored']:
                raise ValueError('Own worker affinity restoration failed')
        finally:
            save(receipt_path, state)


@contextmanager
def controller_scope():
    """Temporary dispatch adaptation; never edit any frozen file/global guard."""
    replacements = {'MODULE': MODULE, 'SCHEDULE': SCHEDULE, 'load_plan': load_plan,
                    'completed': completed, 'finish': finish}
    original = {name: getattr(h, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(h, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(h, name, value)


def worker(job_path):
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    job = read(job_path)
    output, plan = load_plan(job['output'])
    attempt = Path(job_path).resolve().parent
    if job['task'] not in SCHEDULE or attempt.parent != output / 'attempts' or Path(job_path).name != 'job.json':
        raise ValueError('Invalid isolated worker job')
    if topology.cpu_sets() != plan['topology']:
        raise ValueError('Host CPU topology changed')
    cpu = plan['target_cpu']['logical_processor_index']
    payloads, samples = [], []
    result_path = attempt / 'result.json'
    original_save = h.save
    def staged_result(path, payload):
        if Path(path).resolve() == result_path.resolve():
            if payloads:
                raise ValueError('Repeated final worker result')
            payloads.append(payload)
            return
        return original_save(path, payload)
    class PinnedGuard(_ORIGINAL_GUARD):
        def __call__(self, force=False):
            previous = self.last
            super().__call__(force)
            if self.last != previous:
                affinity, actual = list(process.cpu_affinity()), topology.current_processor()
                samples.append({'elapsed': self.last, 'processor': actual, 'affinity': affinity})
                if affinity != [cpu] or actual != cpu:
                    raise ValueError('Worker placement changed during collection/audit')
    def worker_completions(path, current_plan):
        found = completed(path, current_plan)
        # The original auditor's fixed lookup remains directed at the first
        # clean fast collection. Aliases are local and never saved as receipts.
        return {**found, **{f'fast035_{i}': found[f'r0_fast035_{i}']
                          for i in range(3) if f'r0_fast035_{i}' in found}}
    affinity_path = attempt / 'affinity.json'
    with pin_worker(cpu, affinity_path, process=process) as affinity:
        with controller_scope():
            previous = h.WorkerGuard, h.save, h.completed
            h.WorkerGuard, h.save, h.completed = PinnedGuard, staged_result, worker_completions
            try:
                h.worker(job_path)
            finally:
                h.WorkerGuard, h.save, h.completed = previous
    if len(payloads) != 1 or not samples:
        raise ValueError('Missing original-worker result or CPU samples')
    result = payloads[0]
    result.update(affinity=artifact(affinity_path), cpu_samples=samples,
                  target_cpu=plan['target_cpu'], affinity_restored=affinity['restored'])
    save(result_path, result)


def validate_prefix(output):
    keys = [x['key'] for x in SCHEDULE]
    paths = list((Path(output) / 'completed').glob('*.json'))
    present = {p.stem for p in paths}
    if present != set(keys[:len(present)]):
        raise ValueError('Completion inventory must be an exact schedule prefix')
    return len(present)


def same_artifact_identity(left, right):
    """Ignore optional producer metadata, never the artifact location or bytes."""
    return {key: left[key] for key in ('path', 'sha256')} == {
        key: right[key] for key in ('path', 'sha256')}


def validate_scientific_agreement(found):
    for index in range(3):
        results = [value['result'] for value in found.values()
                   if value['result']['task']['kind'] == 'collect' and value['result']['task']['index'] == index]
        if results and any(x['science_sha256'] != results[0]['science_sha256'] or x['row'] != results[0]['row'] for x in results):
            raise ValueError('Four-way reference/fast/repeat or recovery scientific outputs differ')


def completed(output, plan):
    found = {}
    count = validate_prefix(output)
    for task in SCHEDULE[:count]:
        receipt = read(Path(output) / 'completed' / (task['key'] + '.json'))
        result = read(verify(receipt['result']))
        job_path = verify(receipt['job'])
        job = read(job_path)
        attempt = job_path.parent
        if (receipt['binding'] != plan['binding'] or receipt['task'] != task or receipt['exit_code'] != 0
                or result['binding'] != plan['binding'] or result['task'] != task or job['task'] != task
                or Path(job['output']).resolve() != Path(output).resolve()
                or attempt.parent != Path(output) / 'attempts'
                or verify(receipt['result']) != attempt / 'result.json'
                or Path(result['base']) != attempt / 'data'
                or not result['restored_reference'] or not result['affinity_restored']
                or result['loaded_models'] != 9 or result['console_window'] != 0):
            raise ValueError('Invalid authenticated job completion')
        placement = read(verify(result['affinity']))
        cpu = plan['target_cpu']['logical_processor_index']
        if (verify(result['affinity']) != attempt / 'affinity.json'
                or placement['pid'] != result['pid'] or placement['target'] != cpu
                or not all(placement[x] for x in ('entered', 'restored', 'body_completed'))
                or placement['before'] != placement['after'] or placement['during'] != [cpu]
                or placement['entry_processor'] != cpu or result['target_cpu'] != plan['target_cpu']
                or not result['cpu_samples']
                or any(x['processor'] != cpu or x['affinity'] != [cpu] for x in result['cpu_samples'])):
            raise ValueError('Invalid worker placement/restoration evidence')
        if task['kind'] == 'collect':
            root = h.authenticate_root(result['root'], plan['specs'][task['index']], plan['binding'], result['base'])
            if digest(h.canonical_record(root)) != result['science_sha256']:
                raise ValueError('Changed scientific record')
            # Statistics come from the original worker implementation; compare
            # every completed backend/repeat, and the later whole-root recovery.
            usage = result['backend_usage']
            if (usage['backend'] != task['backend'] or usage['blocks_attempted'] != 2
                    or usage['blocks_completed'] != 2
                    or usage['candidate_worlds_completed'] != result['continuations']
                    or [x['candidate_indices'] for x in usage['blocks']] != [root[b]['candidate_indices'] for b in ('selection', 'assessment')]
                    or any(x['worlds'] != 128 or not x['completed'] for x in usage['blocks'])):
                raise ValueError('Changed backend block usage')
        elif task['kind'] == 'interrupt':
            partial, seal = read(verify(result['partial_root'])), read(verify(result['selection']))
            if (not result['expected_interruption'] or 'failure' not in partial or 'assessment' in partial
                    or partial['manifest'] != plan['binding'] or partial['spec'] != plan['specs'][1]
                    or len(partial['selection']['worlds']) != 128
                    or partial['selection'] != seal['selection']
                    or partial['selection_lock'] != seal['selection_lock']
                    or seal['selection_lock']['selection_sha256'] != digest(seal['selection'])
                    or result['backend_usage']['blocks_completed'] != 1):
                raise ValueError('Missing prescribed interruption evidence')
        elif task['kind'] == 'audit':
            replay = read(verify(result['audit']))
            original = found[f'r0_fast035_{task["index"]}']['result']
            if (not replay['passed'] or not replay['full_inner_replay']
                    or not same_artifact_identity(replay['root'], original['root'])
                    or verify(result['audit']) != Path(result['base']) / 'test/audits' / f'state_{task["index"]:04d}.json'):
                raise ValueError('Independent replay identity differs')
            for key in ('root', 'selection_artifact', 'sample_artifact'):
                verify(replay[key])
        elif task['kind'] == 'summary':
            if result['summary']['statistics'] is not None or not result['summary']['full_inner_replay']:
                raise ValueError('Invalid original summary')
        found[task['key']] = {'receipt': receipt, 'result': result}
    validate_scientific_agreement(found)
    return found


def finish(output, plan):
    done = completed(output, plan)
    if set(done) != {x['key'] for x in SCHEDULE}:
        raise ValueError('Incomplete pinned integration')
    pairs, repeats = [], []
    for repeat in range(2):
        for index in range(3):
            reference, fast = (done[f'r{repeat}_{backend}_{index}'] for backend in ('reference', 'fast035'))
            pairs.append({'repeat': repeat, 'index': index, 'historical_index': CONFIG['historical_roots'][index],
                          'exact': True, 'continuations': fast['result']['continuations'],
                          'reference_net_seconds': reference['receipt']['net_wall_seconds'],
                          'fast_net_seconds': fast['receipt']['net_wall_seconds'],
                          'reference_cpu_seconds': reference['result']['cpu_seconds'],
                          'fast_cpu_seconds': fast['result']['cpu_seconds'],
                          'reference_collection_seconds': reference['result']['collection_seconds'],
                          'fast_collection_seconds': fast['result']['collection_seconds']})
        selected = [x for x in pairs if x['repeat'] == repeat]
        reference = sum(x['reference_net_seconds'] for x in selected)
        fast = sum(x['fast_net_seconds'] for x in selected)
        repeats.append({'repeat': repeat, 'net_reference_seconds': reference, 'net_fast_seconds': fast,
                        'net_wall_ratio': reference / fast, 'net_time_saved_fraction': 1 - fast / reference})
    reference = sum(x['net_reference_seconds'] for x in repeats)
    fast = sum(x['net_fast_seconds'] for x in repeats)
    return {'binding': plan['binding'], 'complete': True, 'all_exact': True,
            'pairs': pairs, 'repeats': repeats, 'clean_timed_collections': 12,
            'independent_full_replay': True, 'partial_evidence_preserved': True,
            'whole_root_recovery_exact': True, 'all_worker_affinities_restored': True,
            'target_cpu': plan['target_cpu'], 'net_reference_seconds': reference,
            'net_fast_seconds': fast, 'net_wall_ratio': reference / fast,
            'net_time_saved_fraction': 1 - fast / reference,
            'optional_backend_accepted': fast < reference,
            'time_comparison': 'fixed CPU 0 and two reversed orders; host-load conditional; one logical processor only',
            'active_seconds': h.active_used(output), 'production_enabled': False, 'main_unchanged': True,
            'summary': done['summary']['result']['summary'],
            'completions': [artifact(Path(output) / 'completed' / (x['key'] + '.json')) for x in SCHEDULE]}


def run(output):
    with controller_scope():
        return h.run(output)


def launch(output):
    with controller_scope():
        return h.launch(output)


def preflight(output):
    output = safe_output(output)
    h.operational_guard(output, admission=True)
    attempt = output / ('preflight_' + str(time.time_ns()))
    attempt.mkdir(parents=True)
    identity, started = code_identity(), time.perf_counter()
    with (attempt / 'stdout.txt').open('x') as stdout, (attempt / 'stderr.txt').open('x') as stderr:
        check = subprocess.run([sys.executable, '-m', 'pytest', str(Path(h.__file__).parent),
                                str(Path(__file__).parent), '-q', '-p', 'no:cacheprovider',
                                '--basetemp', str(attempt / 'pytest_tmp')],
                               stdout=stdout, stderr=stderr, timeout=180, **util.process_options())
    receipt = {'passed': check.returncode == 0 and identity == code_identity(), 'code': identity,
               'elapsed_seconds': time.perf_counter()-started,
               'stdout': artifact(attempt / 'stdout.txt'), 'stderr': artifact(attempt / 'stderr.txt')}
    save(attempt / 'checks.json', receipt)
    if not receipt['passed']:
        raise RuntimeError('Preflight failed; see ' + str(attempt))
    return {'passed': True, 'checks': str(attempt / 'checks.json'), 'elapsed_seconds': receipt['elapsed_seconds']}


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
        output = safe_output(args.output)
        result = read(output / ('final.json' if (output / 'final.json').exists() else 'progress.json'))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
