"""Small crossed replay: saved workloads, raw/guarded reference and fast encoder."""
from __future__ import annotations
import argparse
from collections import Counter
import cProfile
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.simulation_backend_integration_128 import integration as h
util = h.util
read, save, artifact, verify, digest = h.read, h.save, h.artifact, h.verify, h.digest
PREVIOUS = ROOT / 'outputs/simulation_backend_integration_128_01'
MODULE = 'tools.simulation_backend_diagnosis_128.diagnose'
CONFIG = {'root_indices': [517, 251, 382], 'world_indices': [0, 42, 85, 127],
          'repeats': 2, 'variants': ['reference', 'guarded_reference', 'fast035'],
          'maximum_seconds': 1800, 'threads': 1, 'workers': 1,
          'profile_roots': [517, 382], 'profile_worlds': [0],
          'production_enabled': False, 'affinity': 'unchanged'}


def safe_output(path):
    path = Path(path).resolve()
    if path.parent != ROOT / 'outputs' or not path.name.startswith('simulation_backend_diagnosis_128_'):
        raise ValueError('Separate diagnosis output required')
    return path


def authenticate_old_plan(plan):
    if digest({k:v for k,v in plan.items() if k != 'binding'}) != plan['binding']:
        raise ValueError('Historical plan binding differs')
    # Verify every frozen dependency. Newly added diagnosis files do not mutate
    # historical code and are not required to exist in the old inventory.
    for ref in plan['code']:
        verify(ref)


def prepare(output):
    output = safe_output(output)
    output.mkdir(parents=True, exist_ok=True)
    h.operational_guard(output, admission=True)
    old_plan = read(PREVIOUS / 'plan.json')
    authenticate_old_plan(old_plan)
    done = h.completed(PREVIOUS, old_plan)
    final = read(PREVIOUS / 'final.json')
    if not final['complete'] or not final['all_exact'] or final['optional_backend_accepted']:
        raise ValueError('Expected completed correctness-pass/speed-fail integration')
    cases = []
    for i, selected in enumerate(old_plan['selected']):
        for source, ref, blocks in (
            ('historical', selected['root'], ('selection',)),
            ('integration', done[f'reference_{i}']['result']['root'], ('selection','assessment'))):
            record = read(verify(ref))
            for block in blocks:
                cases.append({'key': f'{source}_{selected["spec"]["index"]}_{block}',
                              'source': source, 'root_index': selected['spec']['index'],
                              'root': ref, 'block': block})
    schedule = []
    for repeat in range(2):
        ordered = cases if repeat == 0 else list(reversed(cases))
        for case in ordered:
            offset = cases.index(case) % 3
            variants = CONFIG['variants'][offset:] + CONFIG['variants'][:offset]
            if repeat == 1: variants = list(reversed(variants))
            for variant in variants:
                schedule.append({'repeat': repeat, 'case': case['key'], 'variant': variant})
    code = [artifact(p) for p in sorted(Path(__file__).parent.glob('*.py'))]
    code.append(artifact(Path(__file__).with_name('PROTOCOL.md')))
    plan = {'config': CONFIG, 'cases': cases, 'schedule': schedule,
            'prior_plan': artifact(PREVIOUS/'plan.json'), 'prior_final': artifact(PREVIOUS/'final.json'),
            'models': old_plan['models'], 'protected_main': old_plan['protected_main'],
            'inherited_code': old_plan['code'], 'code': code, 'epoch': time.time()}
    plan['binding'] = digest(plan)
    save(output / 'plan.json', plan)
    return {'binding': plan['binding'], 'timing_blocks': len(schedule), 'cases': len(cases)}


def load_plan(output):
    output = safe_output(output); plan = read(output/'plan.json')
    if digest({k:v for k,v in plan.items() if k != 'binding'}) != plan['binding'] or plan['config'] != CONFIG:
        raise ValueError('Changed diagnostic plan')
    for ref in [*plan['code'], *plan['inherited_code'], *plan['models'].values(),
                *plan['protected_main'], plan['prior_plan'], plan['prior_final'], *[c['root'] for c in plan['cases']]]:
        verify(ref)
    return output, plan


class Boundary:
    def __init__(self, job, pulse):
        self.inner = h.WorkerGuard(job, pulse)
        self.samples, self.label = [], None
    def __call__(self):
        previous = self.inner.last
        self.inner()
        if self.inner.last != previous:
            from .topology import current_processor
            self.samples.append({'elapsed': self.inner.last, 'label': self.label,
                                 'processor': current_processor()})


def profile_rows(profile):
    rows = []
    for (file,line,name), (primitive,total,own,cumulative,_) in pstats.Stats(profile).stats.items():
        rows.append({'file': file, 'line': line, 'name': name, 'calls': total,
                     'self_seconds': own, 'cumulative_seconds': cumulative})
    return {'by_self': sorted(rows,key=lambda x:-x['self_seconds'])[:45],
            'by_cumulative': sorted(rows,key=lambda x:-x['cumulative_seconds'])[:45]}


def execute(case, variant, worlds, engine, base, boundary, profile=False):
    from tools.simulation_backend_integration_128.backend import use_backend
    from tools.simulation_speed_probe_128_v2.probe import GuardedOriginal
    record = read(verify(case['root']))
    state = engine.old.deserialize_state_snapshot(record['snapshot'])
    saved = record[case['block']]
    indices = saved['candidate_indices']
    actions = [engine.old.decode_action(record['union']['actions'][i]) for i in indices]
    seeds = [saved['world_seeds'][i] for i in worlds]
    expected = {'candidate_indices': indices, 'world_seeds': seeds,
                'worlds': [saved['worlds'][i] for i in worlds]}
    block = {'candidate_indices': list(indices), 'world_seeds': seeds, 'worlds': []}
    original, profiler, histogram = engine.old._RAW, cProfile.Profile() if profile else None, Counter()
    # Profiling callbacks are absent from the primary timing experiment.
    if variant == 'guarded_reference':
        engine.old._RAW = GuardedOriginal(original, boundary)
    elif profile and variant == 'reference':
        class Observed:
            def choose_action(self, env, state, legal_actions=None):
                histogram[len(legal_actions)] += 1
                return original.choose_action(env,state,legal_actions)
        engine.old._RAW = Observed()
    def observe(state, actions, inputs): histogram[len(actions)] += 1
    try:
        with use_backend(engine, 'fast035' if variant == 'fast035' else 'reference', boundary=boundary,
                         on_position=observe if profile and variant == 'fast035' else None) as usage:
            if profiler: profiler.enable()
            wall,cpu = time.perf_counter(),time.process_time()
            engine.run_block(state, actions, seeds, block, base)
            cpu,wall = time.process_time()-cpu,time.perf_counter()-wall
            if profiler: profiler.disable()
    finally:
        engine.old._RAW = original
        if profiler: profiler.disable()
    actual = {k:v for k,v in block.items() if k != 'timing'}
    if actual != expected or usage['candidate_worlds_completed'] != len(actions)*len(worlds):
        raise ValueError('Saved continuation mismatch')
    result = {'case': case['key'], 'source': case['source'], 'root_index': case['root_index'],
              'block': case['block'], 'variant': variant, 'world_indices': worlds,
              'continuations': len(actions)*len(worlds),
              'turns': sum(o['turns'] for w in block['worlds'] for o in w['outcomes']),
              'exact': True, 'science_sha256': digest(actual), 'wall':wall, 'cpu':cpu}
    if profile: result.update(profile=profile_rows(profiler), legal_action_histogram=dict(histogram))
    return result


def worker(output, job_path):
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    output,plan = load_plan(output)
    job = read(job_path); boundary = Boundary(job, output/'pulse.json')
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    from tools.large_corpus_optimization_128 import common as c, engine
    from .topology import cpu_sets, current_processor
    registry = {'manifest':plan['binding'], 'models':plan['models']}
    engine.initialize_models(base=output,registry=registry)
    if len(engine._MODELS) != 9: raise ValueError('Nine-model lifecycle missing')
    save(output/'worker_ready.json', {'pid':os.getpid(), 'affinity': psutil.Process().cpu_affinity(),
                                     'topology':cpu_sets(), 'processor':current_processor(),
                                     'model_count':len(engine._MODELS)})
    cases = {x['key']:x for x in plan['cases']}
    rows, profiles = [], []
    with h.guarded_budget(c, boundary):
        for n, task in enumerate(plan['schedule']):
            boundary.label = f'timing_{n}_{task["case"]}_{task["variant"]}'
            boundary()
            row = execute(cases[task['case']],task['variant'],CONFIG['world_indices'],engine,output,boundary)
            row['repeat'] = task['repeat']; row['sequence'] = n
            save(output/'timings'/f'{n:03d}.json',row); rows.append(row)
            util.replace_status(output/'progress.json', {'stage':'timings', 'completed':n+1,
                                'total':len(plan['schedule']), 'pid':os.getpid(),'epoch':time.time()})
        for case in plan['cases']:
            if case['block'] != 'selection' or case['root_index'] not in CONFIG['profile_roots']: continue
            for variant in ('reference','fast035'):
                boundary.label = 'profile_' + case['key'] + '_' + variant
                row = execute(case,variant,CONFIG['profile_worlds'],engine,output,boundary,profile=True)
                save(output/'profiles'/f'{case["key"]}_{variant}.json',row); profiles.append(row)
    # Every variant and repeat must retain the same expected scientific record.
    for case in cases:
        if len({r['science_sha256'] for r in rows if r['case']==case}) != 1:
            raise ValueError('Crossed comparisons are not the same work')
    totals = []
    for source in ('historical','integration'):
        for block in ('selection','assessment'):
            for repeat in range(2):
                group = [r for r in rows if r['source']==source and r['block']==block and r['repeat']==repeat]
                if not group: continue
                totals.append({'source':source,'block':block,'repeat':repeat,
                               **{v:{k:sum(r[k] for r in group if r['variant']==v) for k in ('wall','cpu','continuations','turns')}
                                  for v in CONFIG['variants']}})
    load_plan(output)
    save(output/'result.json', {'binding':plan['binding'],'all_exact':True,'totals':totals,
                              'timing_rows':len(rows),'profile_rows':len(profiles),
                              'core_samples':boundary.samples,'memory':boundary.inner.samples,
                              'main_unchanged':True,'production_enabled':False})


def run(output):
    output,plan = load_plan(output)
    h.operational_guard(output,admission=True)
    if (output/'job.json').exists(): raise FileExistsError('One-shot diagnosis; preserve prior attempts')
    job = {'output':str(output),'parent':util.process_identity(os.getpid()),
           'remaining_seconds':CONFIG['maximum_seconds']}
    save(output/'job.json',job)
    started = time.perf_counter()
    command = [str(Path(sys.executable).with_name('pythonw.exe')), '-m',MODULE,'worker','--output',str(output),
               '--job',str(output/'job.json')]
    process = None
    try:
        with (output/'stdout.txt').open('x') as stdout,(output/'stderr.txt').open('x') as stderr:
            process = subprocess.Popen(command,stdout=stdout,stderr=stderr,**util.process_options())
            save(output/'process.json',util.process_identity(process.pid))
            while process.poll() is None:
                h.operational_guard(output)
                if time.perf_counter()-started > CONFIG['maximum_seconds']:
                    raise RuntimeError('Thirty-minute diagnostic limit')
                time.sleep(2)
            if process.returncode: raise RuntimeError('Diagnostic worker failed; inspect stderr')
        result = read(output/'result.json')
        save(output/'finished.json',{'binding':plan['binding'],'elapsed':time.perf_counter()-started,
                                    'result':artifact(output/'result.json'),'exit_code':process.returncode})
        return {'complete':True,'all_exact':result['all_exact'],'elapsed':time.perf_counter()-started,'totals':result['totals']}
    finally:
        if process is not None and process.poll() is None:
            process.terminate(); process.wait(timeout=10)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','worker'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.command=='worker': worker(args.output,args.job); return
    print(json.dumps({'prepare':prepare,'run':run}[args.command](args.output)),flush=True)


if __name__=='__main__':
    main()
