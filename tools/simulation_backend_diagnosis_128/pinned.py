"""Controlled P/E placement counterfactual; never an automatic adoption test.

Replays the same exact saved integration work through diagnose.execute. Only
this diagnostic worker's affinity changes, with its original mask restored on
exit. No original experiment or frozen source is modified. Prepare binds the
topology, fixed workload and balanced order before any pinned timing results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.simulation_backend_diagnosis_128 import diagnose as d, topology

ROOT, h, util = d.ROOT, d.h, d.util
read, save, artifact, verify, digest = d.read, d.save, d.artifact, d.verify, d.digest
NORMAL = ROOT / 'outputs/simulation_backend_diagnosis_128_01'
MODULE = 'tools.simulation_backend_diagnosis_128.pinned'
CONFIG = {'root_indices': [517, 382], 'world_indices': [0, 42, 85, 127],
          'source': 'integration', 'block': 'selection', 'repeats': 2,
          'variants': ['reference', 'fast035'], 'maximum_seconds': 1800,
          'threads': 1, 'workers': 1, 'production_enabled': False,
          'affinity': 'own_worker_only_lowest_logical_processor_per_efficiency_class'}


def safe_output(path):
    path = Path(path).resolve()
    if path.parent != ROOT / 'outputs' or not path.name.startswith('simulation_backend_diagnosis_pinned_128_'):
        raise ValueError('Separate pinned diagnostic output required')
    return path


def prepare(output):
    output = safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'plan.json').exists():
        raise FileExistsError('Preserve existing pinned plan')
    h.operational_guard(output, admission=True)
    _, prior = d.load_plan(NORMAL)
    cases = [c for c in prior['cases'] if c['source'] == 'integration'
             and c['block'] == 'selection' and c['root_index'] in CONFIG['root_indices']]
    if [c['root_index'] for c in cases] != CONFIG['root_indices']:
        raise ValueError('Pinned historical case inventory changed')
    mapping = topology.cpu_sets()
    classes = sorted({row['efficiency_class'] for row in mapping}, reverse=True)
    if len(mapping) != 20 or {row['group'] for row in mapping} != {0} or len(classes) != 2:
        raise ValueError('Expected verified single-group heterogeneous 20-processor host')
    cores = [min((row for row in mapping if row['efficiency_class'] == cls),
                 key=lambda row: row['logical_processor_index']) for cls in classes]
    schedule = []
    for repeat in range(CONFIG['repeats']):
        for core in (cores if repeat == 0 else list(reversed(cores))):
            for case in (cases if repeat == 0 else list(reversed(cases))):
                variants = CONFIG['variants'] if repeat == 0 else list(reversed(CONFIG['variants']))
                for variant in variants:
                    schedule.append({'repeat': repeat, 'case': case['key'], 'variant': variant,
                                     'processor': core['logical_processor_index'],
                                     'efficiency_class': core['efficiency_class']})
    plan = {'config': CONFIG, 'cases': cases, 'schedule': schedule, 'topology': mapping,
            'selected_cores': cores, 'prior_plan': artifact(NORMAL / 'plan.json'),
            'prior_binding': prior['binding'], 'models': prior['models'],
            'code': artifact(Path(__file__)), 'epoch': time.time(),
            'interpretation': 'controlled placement only; no universal throughput or adoption claim'}
    plan['binding'] = digest(plan)
    save(output / 'plan.json', plan)
    return {'binding': plan['binding'], 'timing_blocks': len(schedule), 'selected_cores': cores}


def load_plan(output):
    output = safe_output(output)
    plan = read(output / 'plan.json')
    if digest({k: v for k, v in plan.items() if k != 'binding'}) != plan['binding'] or plan['config'] != CONFIG:
        raise ValueError('Pinned plan changed')
    verify(plan['code']); verify(plan['prior_plan'])
    _, prior = d.load_plan(NORMAL)
    if prior['binding'] != plan['prior_binding'] or prior['models'] != plan['models']:
        raise ValueError('Pinned model/dependency binding changed')
    for case in plan['cases']:
        verify(case['root'])
    return output, plan


def worker(output, job_path):
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    output, plan = load_plan(output)
    if Path(job_path).resolve() != output / 'job.json':
        raise ValueError('Pinned worker job must belong to its output')
    job = read(job_path)
    if Path(job['output']).resolve() != output:
        raise ValueError('Pinned job output differs')
    boundary = d.Boundary(job, output / 'pulse.json')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from tools.large_corpus_optimization_128 import common as c, engine
    if topology.cpu_sets() != plan['topology']:
        raise ValueError('CPU topology changed since pinned preparation')
    engine.initialize_models(base=output, registry={'manifest': plan['binding'], 'models': plan['models']})
    boundary()
    if len(engine._MODELS) != 9:
        raise ValueError('Nine-model lifecycle missing')
    original_affinity = process.cpu_affinity()
    if not {r['logical_processor_index'] for r in plan['selected_cores']} <= set(original_affinity):
        raise ValueError('Selected processors outside original worker affinity')
    original_policy, original_block = engine.old._RAW, engine.run_block
    cases, rows = {case['key']: case for case in plan['cases']}, []
    save(output / 'worker_ready.json', {'pid': os.getpid(), 'model_count': len(engine._MODELS),
                                       'original_affinity': original_affinity, 'topology': plan['topology']})
    try:
        with h.guarded_budget(c, boundary):
            for sequence, task in enumerate(plan['schedule']):
                process.cpu_affinity([task['processor']])
                time.sleep(0)  # Yield before confirming the newly constrained placement.
                if process.cpu_affinity() != [task['processor']] or topology.current_processor() != task['processor']:
                    raise ValueError('Diagnostic worker placement was not applied')
                boundary.label = f'pinned_{sequence}_cpu{task["processor"]}_{task["variant"]}'
                boundary()
                row = d.execute(cases[task['case']], task['variant'], CONFIG['world_indices'],
                                engine, output, boundary)
                row.update(sequence=sequence, repeat=task['repeat'], processor=task['processor'],
                           efficiency_class=task['efficiency_class'],
                           processor_after=topology.current_processor())
                if row['processor_after'] != task['processor']:
                    raise ValueError('Diagnostic worker left its pinned processor')
                save(output / 'timings' / f'{sequence:03d}.json', row)
                rows.append(row)
                util.replace_status(output / 'progress.json', {'stage': 'timings', 'completed': len(rows),
                                    'total': len(plan['schedule']), 'pid': os.getpid(), 'epoch': time.time()})
    finally:
        process.cpu_affinity(original_affinity)
        restored = process.cpu_affinity() == original_affinity
        save(output / 'affinity_restored.json', {'pid': os.getpid(), 'restored': restored,
                                               'affinity': process.cpu_affinity(), 'epoch': time.time()})
        if not restored:
            raise RuntimeError('Diagnostic worker affinity restoration failed')
    if engine.old._RAW is not original_policy or engine.run_block is not original_block:
        raise ValueError('Pinned backend scope leaked')
    for case in cases:
        if len({row['science_sha256'] for row in rows if row['case'] == case}) != 1:
            raise ValueError('Pinned variants/core classes did not execute identical work')
    totals = []
    for core in plan['selected_cores']:
        for repeat in range(CONFIG['repeats']):
            group = [row for row in rows if row['processor'] == core['logical_processor_index']
                     and row['repeat'] == repeat]
            totals.append({'processor': core['logical_processor_index'], 'efficiency_class': core['efficiency_class'],
                           'repeat': repeat, **{variant: {field: sum(row[field] for row in group
                            if row['variant'] == variant) for field in ('wall', 'cpu', 'turns', 'continuations')}
                            for variant in CONFIG['variants']}})
    load_plan(output)
    save(output / 'result.json', {'binding': plan['binding'], 'all_exact': True, 'timing_rows': len(rows),
                                'totals': totals, 'core_samples': boundary.samples,
                                'memory': boundary.inner.samples, 'affinity_restored': True,
                                'main_unchanged': True, 'production_enabled': False})


def run(output):
    output, plan = load_plan(output)
    with h.exclusive(output):
        h.operational_guard(output, admission=True)
        if (output / 'job.json').exists():
            raise FileExistsError('One-shot pinned diagnosis; preserve all prior attempts')
        save(output / 'job.json', {'output': str(output), 'parent': util.process_identity(os.getpid()),
                                   'remaining_seconds': CONFIG['maximum_seconds']})
        started, process = time.perf_counter(), None
        command = [str(Path(sys.executable).with_name('pythonw.exe')), '-m', MODULE, 'worker',
                   '--output', str(output), '--job', str(output / 'job.json')]
        try:
            with (output / 'stdout.txt').open('x') as stdout, (output / 'stderr.txt').open('x') as stderr:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr, **util.process_options())
                save(output / 'process.json', util.process_identity(process.pid))
                while process.poll() is None:
                    h.operational_guard(output)
                    import psutil
                    try:
                        info = psutil.Process(process.pid).memory_info()
                        if max(info.rss, getattr(info, 'peak_wset', 0), getattr(info, 'private', 0)) > h.CONFIG['maximum_worker_memory_bytes']:
                            raise RuntimeError('Pinned worker memory allocation exceeded')
                    except psutil.NoSuchProcess:
                        pass
                    if time.perf_counter() - started > CONFIG['maximum_seconds']:
                        raise RuntimeError('Thirty-minute pinned diagnostic limit')
                    time.sleep(2)
                if process.returncode:
                    raise RuntimeError('Pinned worker failed; preserved stderr and timing evidence')
            result = read(output / 'result.json')
            if result['binding'] != plan['binding'] or not result['all_exact'] or not result['affinity_restored']:
                raise ValueError('Invalid pinned worker result')
            save(output / 'finished.json', {'binding': plan['binding'], 'elapsed': time.perf_counter() - started,
                                           'result': artifact(output / 'result.json'), 'exit_code': process.returncode})
            return {'complete': True, 'all_exact': True, 'elapsed': time.perf_counter() - started,
                    'totals': result['totals']}
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'worker'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.output, args.job)
    else:
        print(json.dumps({'prepare': prepare, 'run': run}[args.command](args.output)), flush=True)


if __name__ == '__main__':
    main()
