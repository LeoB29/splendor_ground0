"""Bounded saved-128-world input and rollout profiling, never production use.

Commands require an already frozen alternative-learner study. The coordinator
must separately release execution; importing/preparing this module starts none.
Only the completed large_corpus_optimization_128 panel supplies positions/worlds.
"""
from __future__ import annotations

import argparse
import cProfile
import ctypes
import hashlib
import json
import os
from pathlib import Path
import pstats
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
OLD = ROOT / 'outputs/large_corpus_optimization_128'
CHECKPOINT = ROOT / 'outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt'
CHECKPOINT_SHA = '4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd'
CONFIG = {
    'selection_rule': 'within_each_phase_canonical_support_min_median_max_then_index',
    'input_repetitions': 20, 'input_warmups': 2,
    'world_indices': [0, 17, 35, 53, 71, 89, 107, 127],
    'candidate_indices': 'first_and_last_union_candidate',
    'rollout_roots': 'median_support_root_each_phase',
    'micro_active_seconds': 180, 'rollout_active_seconds': 900,
    'minimum_available_ram_bytes': 2 * 1024**3,
    'minimum_commit_headroom_bytes': 6 * 1024**3,
    'maximum_process_rss_bytes': 2 * 1024**3,
    'minimum_disk_free_bytes': 5 * 1024**3,
    'readiness_only': {'maximum_seconds': 60,
                       'minimum_available_ram_bytes': 512 * 1024**2,
                       'minimum_commit_headroom_bytes': 2 * 1024**3,
                       'maximum_process_rss_bytes': 768 * 1024**2},
    'threads': 1, 'production_enabled': False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)


def code_identity():
    # Freeze shared dependencies read-only. No predecessor source is edited.
    paths = sorted((ROOT / 'src/splendor_ai').rglob('*.py'))
    paths += sorted(Path(__file__).parent.glob('*.py'))
    paths += [Path(__file__).parent / 'PROTOCOL.md']
    return [artifact(path) for path in paths]


def choose_specs(specs):
    selected = []
    for phase in range(3):
        ordered = sorted((s for s in specs if s['phase'] == phase),
                         key=lambda s: (s['canonical_support'], s['index']))
        if len(ordered) < 3:
            raise ValueError('Insufficient completed source coverage')
        for label, offset in [('minimum', 0), ('median', len(ordered)//2), ('maximum', len(ordered)-1)]:
            selected.append({'support_band': label, 'spec': ordered[offset]})
    if len({s['spec']['index'] for s in selected}) != 9:
        raise ValueError('Repeated profiling source')
    return selected


def safe_output(path, fresh=False):
    path = Path(path).resolve()
    if path.parent != ROOT / 'outputs' or not path.name.startswith('simulation_speed_probe_128_'):
        raise ValueError('Use a new, direct outputs/simulation_speed_probe_128_* directory')
    if fresh and path.exists():
        raise FileExistsError(path)
    return path


def prepare(output, alternative_freeze):
    """Metadata/hashes only: no model, game engine, inference or simulation."""
    output = safe_output(output, fresh=True)
    alt = Path(alternative_freeze).resolve()
    alt_payload = read(alt)
    if len(alt_payload.get('binding', '')) != 64:
        raise ValueError('Alternative study must already have its immutable freeze binding')
    status = read(OLD/'final_status.json')
    if not status.get('complete') or not status.get('all_prescribed_audits_pass'):
        raise ValueError('Historical source study is incomplete')
    seal = read(OLD/'selection_seal.json')
    selection = read(verify(seal['selection']))
    collection = read(OLD/'main_collection_lock.json')
    refs = {r['index']: r for r in collection['files']}
    selected = choose_specs(selection['main'])
    for item in selected:
        index = item['spec']['index']
        reference = refs[index]
        verify(reference)
        item['root'] = {k: reference[k] for k in ('path', 'sha256')}
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Original035 checkpoint changed')
    plan = {'config': CONFIG, 'source': artifact(OLD/'selection.json'),
            'collection': artifact(OLD/'main_collection_lock.json'),
            'source_final': artifact(OLD/'final_status.json'),
            'source_manifest': status['manifest'], 'selected': selected,
            'alternative_freeze': artifact(alt), 'checkpoint': artifact(CHECKPOINT),
            'code': code_identity(), 'no_strength_outcomes_used_for_selection': True,
            'no_structured_fresh_data_read': True, 'production_enabled': False}
    plan['binding'] = digest(plan)
    save(output/'plan.json', plan)
    return {'output': str(output), 'binding': plan['binding'], 'roots': 9}


def load_plan(output):
    output = safe_output(output)
    plan = read(output/'plan.json')
    body = {k: v for k, v in plan.items() if k != 'binding'}
    if digest(body) != plan['binding'] or plan['config'] != CONFIG or plan['code'] != code_identity():
        raise ValueError('Changed speed-probe plan/code')
    for key in ('source', 'collection', 'source_final', 'alternative_freeze', 'checkpoint'):
        verify(plan[key])
    return output, plan


def memory():
    import psutil
    process = psutil.Process()
    info = process.memory_info()
    result = {'available_ram_bytes': psutil.virtual_memory().available,
              'process_rss_bytes': info.rss, 'process_peak_rss_bytes': getattr(info, 'peak_wset', None),
              'process_private_bytes': getattr(info, 'private', None)}
    if os.name != 'nt':
        raise RuntimeError('This operational probe is specified for the Windows host')
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
            ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
             'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')] + [
            ('HandleCount',ctypes.c_ulong),('ProcessCount',ctypes.c_ulong),('ThreadCount',ctypes.c_ulong)]
    status = PerformanceInfo(); status.cb = ctypes.sizeof(status)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(status), status.cb):
        raise OSError('Cannot establish memory headroom')
    result['commit_headroom_bytes'] = (status.CommitLimit-status.CommitTotal)*status.PageSize
    return result


def resource_ok(info, disk_free, policy=None):
    policy = CONFIG if policy is None else policy
    return (info['available_ram_bytes'] >= policy['minimum_available_ram_bytes']
            and info['commit_headroom_bytes'] >= policy['minimum_commit_headroom_bytes']
            and info['process_rss_bytes'] <= policy['maximum_process_rss_bytes']
            and disk_free >= CONFIG['minimum_disk_free_bytes'])


def permitted_stage(phase, complete=False):
    return complete is True or phase in ('learning_data_validation', 'learning_fits', 'learning_fit_audits')


class Boundary:
    def __init__(self, output, seconds, policy=None):
        self.output, self.started, self.seconds = output, time.perf_counter(), seconds
        self.policy = CONFIG if policy is None else policy
        self.last_check, self.snapshots = -1., []
        self(force=True)

    def __call__(self, force=False):
        elapsed = time.perf_counter() - self.started
        if elapsed > self.seconds or (self.output/'abort.json').exists():
            raise RuntimeError('Probe time/abort hold; preserve evidence')
        if not force and elapsed - self.last_check < 1:
            return
        self.last_check = elapsed
        info = memory()
        if not resource_ok(info, shutil.disk_usage(ROOT).free, self.policy):
            raise RuntimeError('Insufficient memory/storage headroom: ' + json.dumps(info))
        # Only operational status is read. Never inspect running study outcomes.
        running = ROOT/'outputs/structured_proposer_128'
        complete = (read(running/'final_status.json').get('complete') is True
                    if (running/'final_status.json').exists() else False)
        phase = read(running/'progress.json').get('stage')
        if not permitted_stage(phase, complete):
            raise RuntimeError('Do not compete with the running study evaluation stage: ' + str(phase))
        if force or not self.snapshots or elapsed - self.snapshots[-1]['elapsed'] >= 10:
            self.snapshots.append({'elapsed': elapsed, **info})


def initialize():
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from splendor_ai.alphazero import run_afterstate_advantage as old
    old.initialize_worker()
    return torch, old


def root_state(item, old):
    record = read(verify(item['root']))
    if record['spec'] != item['spec'] or old.digest(record['snapshot']) != item['spec']['snapshot_sha256']:
        raise ValueError('Historical root/source identity changed')
    return record, old.deserialize_state_snapshot(record['snapshot'])


def timed(function):
    wall, cpu = time.perf_counter(), time.process_time()
    value = function()
    return value, {'wall': time.perf_counter()-wall, 'cpu': time.process_time()-cpu}


def profile_rows(profile):
    stats = pstats.Stats(profile)
    values = []
    for (file, line, name), (primitive, total, own, cumulative, callers) in stats.stats.items():
        values.append({'file': file, 'line': line, 'function': name,
                       'primitive_calls': primitive, 'calls': total,
                       'self_seconds': own, 'cumulative_seconds': cumulative})
    return sorted(values, key=lambda row: row['self_seconds'], reverse=True)[:60]


def readiness(output):
    """Load-only memory accounting; no neural forward, timing test or rollout."""
    output, plan = load_plan(output)
    if (output/'readiness.json').exists():
        raise FileExistsError(output/'readiness.json')
    policy = CONFIG['readiness_only']
    guard = Boundary(output, policy['maximum_seconds'], policy)
    torch, old = initialize()
    guard(force=True)
    from . import encoder
    item = max(plan['selected'], key=lambda row: (row['spec']['canonical_support'], -row['spec']['index']))
    record, state = root_state(item, old)
    legal = old.SplendorEnv().legal_actions(state)
    expected, actual = encoder.reference_tensors(state, legal), encoder.tensors(state, legal)
    if not all(torch.equal(a,b) for a,b in zip(expected,actual,strict=True)):
        raise ValueError('Readiness input differs from reference')
    guard(force=True)
    result = {'binding': plan['binding'], 'root': item['spec']['index'],
              'legal_actions': len(legal), 'memory': guard.snapshots,
              'model_forward_calls': 0, 'simulated_moves': 0, 'production_enabled': False,
              'elapsed': time.perf_counter()-guard.started,
              'limitations': 'Only dependency/model/input working set; excludes neural activation and trajectory peaks. Main probe resource guard unchanged.'}
    save(output/'readiness.json', result)
    return result


def micro(output):
    output, plan = load_plan(output)
    if (output/'micro.json').exists():
        raise FileExistsError(output/'micro.json')
    guard = Boundary(output, CONFIG['micro_active_seconds'])
    torch, old = initialize()
    from . import encoder
    guard(force=True)
    rows, exact = [], 0
    profiler = cProfile.Profile()
    for item in plan['selected']:
        guard()
        record, state = root_state(item, old)
        env = old.SplendorEnv()
        legal = env.legal_actions(state)
        expected, actual = encoder.reference_tensors(state, legal), encoder.tensors(state, legal)
        if not all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)):
            raise ValueError('Input tensor mismatch at historical root ' + str(item['spec']['index']))
        with torch.no_grad():
            reference_scores = old._RAW._model(*expected)[0]
            actual_scores = old._RAW._model(*actual)[0]
        if not torch.equal(reference_scores, actual_scores):
            raise ValueError('Same-shape model outputs changed')
        exact += 1
        totals = {mode: {'wall': 0., 'cpu': 0.} for mode in ('reference', 'prototype', 'legal', 'forward')}
        funcs = {'reference': lambda: encoder.reference_tensors(state, legal),
                 'prototype': lambda: encoder.tensors(state, legal)}
        for _ in range(CONFIG['input_warmups']):
            for fn in funcs.values(): fn()
        for repetition in range(CONFIG['input_repetitions']):
            guard()
            order = ('reference','prototype') if repetition % 2 == 0 else ('prototype','reference')
            for mode in order:
                _, duration = timed(funcs[mode])
                for kind in ('wall', 'cpu'): totals[mode][kind] += duration[kind]
            _, duration = timed(lambda: env.legal_actions(state))
            for kind in ('wall', 'cpu'): totals['legal'][kind] += duration[kind]
            with torch.no_grad():
                _, duration = timed(lambda: old._RAW._model(*expected))
            for kind in ('wall', 'cpu'): totals['forward'][kind] += duration[kind]
        profiler.enable()
        encoder.reference_tensors(state, legal)
        profiler.disable()
        rows.append({'index': item['spec']['index'], 'phase': item['spec']['phase'],
                     'support_band': item['support_band'], 'canonical_support': item['spec']['canonical_support'],
                     'legal_actions': len(legal), 'exact_tensors': True, 'exact_scores': True,
                     'input_speed_ratio': totals['reference']['wall']/totals['prototype']['wall'], 'timing': totals})
        del record
    ref = sum(row['timing']['reference']['wall'] for row in rows)
    new = sum(row['timing']['prototype']['wall'] for row in rows)
    result = {'binding': plan['binding'], 'rows': rows, 'exact_root_count': exact,
              'all_exact': exact == 9, 'input_speed_ratio': ref/new,
              'profile_reference_input': profile_rows(profiler), 'memory': guard.snapshots,
              'elapsed': time.perf_counter()-guard.started, 'production_enabled': False,
              'limitations': 'Root-only reconnaissance. No trajectory speed or broad equivalence claim.'}
    save(output/'micro.json', result)
    return {key: result[key] for key in ('all_exact', 'input_speed_ratio', 'elapsed')}


class GuardedOriginal:
    def __init__(self, original, guard): self.original, self.guard = original, guard
    def choose_action(self, *args, **kwargs):
        self.guard()
        return self.original.choose_action(*args, **kwargs)


def rollout(output):
    output, plan = load_plan(output)
    screening = read(output/'micro.json')
    if screening.get('binding') != plan['binding'] or screening.get('all_exact') is not True:
        raise ValueError('Exact input micro-check required first')
    if (output/'rollout.json').exists():
        raise FileExistsError(output/'rollout.json')
    guard = Boundary(output, CONFIG['rollout_active_seconds'])
    torch, old = initialize()
    from . import encoder
    original = old._RAW
    tensor_checks = 0
    def exact_position(state, legal, actual):
        nonlocal tensor_checks
        expected = encoder.reference_tensors(state, legal)
        if not all(torch.equal(a,b) for a,b in zip(actual,expected,strict=True)):
            raise ValueError('Intermediate trajectory input mismatch')
        tensor_checks += 1
    policies = {'reference': GuardedOriginal(original, guard),
                'prototype': encoder.Fast035(original, guard),
                'checked_prototype': encoder.Fast035(original, guard, exact_position)}
    rows, profiler, sequence = [], cProfile.Profile(), 0
    try:
        for item in [r for r in plan['selected'] if r['support_band'] == 'median']:
            record, state = root_state(item, old)
            actions = [old.decode_action(a) for a in record['union']['actions']]
            indices = sorted({0, len(actions)-1})
            for world_index in CONFIG['world_indices']:
                guard()
                saved = record['selection']['worlds'][world_index]
                world = old.determinize_for_observer(state,state.current_player,random.Random(saved['seed']))
                if old.digest(old.serialize_state_snapshot(world)) != saved['state_sha256']:
                    raise ValueError('Historical determinization changed')
                for candidate in indices:
                    expected = saved['outcomes'][candidate]
                    expected_core = {k: expected[k] for k in ('reason','winner','turns','trace_sha256','final_state_sha256')}
                    # Exact all-intermediate input checks are a separate untimed
                    # pass, so equivalence overhead cannot masquerade as speed.
                    old._RAW = policies['checked_prototype']
                    checked = old.trajectory(world, actions[candidate])
                    if checked != expected_core:
                        raise ValueError('Prototype trace differs from historical128-world reference')
                    order = ('reference','prototype') if sequence % 2 == 0 else ('prototype','reference')
                    timing = {}
                    for mode in order:
                        old._RAW = policies[mode]
                        actual, duration = timed(lambda: old.trajectory(world, actions[candidate]))
                        if actual != expected_core:
                            raise ValueError('Timed continuation differs from saved reference')
                        timing[mode] = duration
                    if world_index == CONFIG['world_indices'][0] and candidate == indices[0]:
                        old._RAW = policies['reference']
                        profiler.enable()
                        profiled = old.trajectory(world, actions[candidate])
                        profiler.disable()
                        if profiled != expected_core:
                            raise ValueError('Profiled continuation changed')
                    rows.append({'root': item['spec']['index'], 'phase': item['spec']['phase'],
                                 'world_index': world_index, 'candidate_index': candidate,
                                 'turns': expected['turns'], 'exact': True, 'timing': timing})
                    sequence += 1
                    save(output/'continuations'/f'job_{sequence:03d}.json', rows[-1])
    finally:
        old._RAW = original
        profiler.disable()
    reference_time = sum(row['timing']['reference']['wall'] for row in rows)
    prototype_time = sum(row['timing']['prototype']['wall'] for row in rows)
    result = {'binding': plan['binding'], 'rows': rows, 'all_exact': True,
              'intermediate_tensor_checks': tensor_checks, 'trajectory_speed_ratio': reference_time/prototype_time,
              'reference_profile': profile_rows(profiler), 'memory': guard.snapshots,
              'elapsed': time.perf_counter()-guard.started, 'production_enabled': False,
              'limitations': '48 selected continuations from completed128-world roots, one CPU worker; not full128-root, multiworker or universal correctness evidence.'}
    save(output/'rollout.json', result)
    return {key: result[key] for key in ('all_exact','intermediate_tensor_checks','trajectory_speed_ratio','elapsed')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'readiness', 'micro', 'rollout'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alternative-freeze', type=Path)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            if args.alternative_freeze is None: parser.error('--alternative-freeze required')
            result = prepare(args.output, args.alternative_freeze)
        else:
            result = {'readiness': readiness, 'micro': micro, 'rollout': rollout}[args.command](args.output)
        print(json.dumps(result, sort_keys=True))
    except Exception as error:
        output = safe_output(args.output)
        if output.exists():
            marker = output/f'{args.command}_hold_{time.time_ns()}.json'
            save(marker, {'command': args.command, 'error': repr(error), 'production_enabled': False})
        raise


if __name__ == '__main__':
    main()
