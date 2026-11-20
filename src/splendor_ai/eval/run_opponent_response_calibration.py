"""Experiment 141: fixed-move calibration of two opponent-response estimators."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np

from splendor_ai.eval import run_independent_world_root_validation as base
from splendor_ai.eval import run_independent_confirmation_root_policy as prior139
from splendor_ai.eval import run_continuation_policy_crossover as prior140

PROTOCOL = Path('docs/engineering/opponent_response_calibration_141.md')
OUTPUT = Path('outputs/opponent_response_calibration_141')
WRAPPER = Path('run_opponent_response_calibration_141.py')
TEST = Path('tests/test_opponent_response_calibration_141.py')
VERIFIER = Path('scripts/analysis/verify_opponent_response_calibration_141.py')
CELLS = base.CELLS
CONDITIONS = ('sampled_min', 'policy_reply')
CONTRASTS = ('sampled_min_gain', 'policy_reply_gain', 'policy_minus_min')
STATES, PER_CELL, BLOCKS, WORLDS, REPLICATES, WORKERS = 120, 20, 16, 3, 4, 10
MAX_CONTINUATIONS = 230_400
EXCLUSION_HASHES = {
 'outputs/independent_world_root_validation_138/state_manifest.json': '5bcc900a2a03c0aa3fde95c036819164617cde2f65a0e450f8502e6371dbdee9',
 'outputs/independent_world_root_validation_138/preflight.json': 'bb3f9c4201592e113a00e78dd11ac91a6486ded33de006e64d699e7f30ed3d8f',
 'outputs/independent_confirmation_root_policy_139/state_manifest.json': '3acddd3c887786ba853c2d08568111a311127f1d448f85aca459de52bc9c6bdb',
 'outputs/independent_confirmation_root_policy_139/preflight.json': '698403fd4d670090490821a088ac5cdc99832726f93092c113b3a460e522e17e',
 'outputs/continuation_policy_crossover_140/state_manifest.json': 'abbdc0fb67b264d6fd999b90efbd2cb9aa24ff66ae390d5288c5d180296a52ac',
 'outputs/continuation_policy_crossover_140/preparation_manifest.json': 'e9329ec1409fefddb7bc298410db2d1dc193f276d9d0f6bdbbd07221b6c6ee21',
}
digest, file_hash = base.digest, base.file_hash
read_json, atomic_write = base.read_json, base.atomic_write


def input_identity():
    expected = {**base.SOURCE_HASHES, **EXCLUSION_HASHES}
    sources = {p: file_hash(Path(p)) for p in expected}
    if sources != expected:
        raise ValueError('141 source/checkpoint/exclusion artifact changed')
    files = sorted(Path('src/splendor_ai').rglob('*.py')) + [WRAPPER, TEST, VERIFIER]
    return {'format': 'opponent_response_calibration_141_v1', 'sources': sources,
            'protocol_sha256': file_hash(PROTOCOL),
            'dependencies': {p.as_posix(): file_hash(p) for p in files},
            'workers': WORKERS, 'torch_threads': 1}


def excluded_games(source):
    ids = {s['sample_index']: s['source_seed'] for s in source['selected']}
    excluded = set()
    for name in EXCLUSION_HASHES:
        record = read_json(Path(name))
        excluded.update(s['source_seed'] for s in record.get('states', []))
        excluded.update(s['source_seed'] for s in record.get('probes', []))
        if 'runtime_probe' in record:
            probe = record['runtime_probe']
            excluded.add(ids[probe.get('source_sample_index', probe.get('sample_index'))])
    return excluded


def select_states(source, excluded):
    def rank(s, prefix):
        key = f"{prefix}|{s['source_seed']}|{s['turn_index']}|{s['actor_seat']}|{s['phase']}"
        return hashlib.sha256(key.encode()).hexdigest(), s['source_seed'], s['turn_index']
    selected, probes = [], []
    for cell in CELLS:
        eligible = [s for s in source['selected'] if s['selection_cell'] == cell and s['source_seed'] not in excluded]
        if len(eligible) <= PER_CELL:
            raise ValueError('141 insufficient unused games for panel and probe')
        chosen = sorted(eligible, key=lambda s: rank(s, '141-selection-v1'))[:PER_CELL]
        selected.extend(chosen)
        used = {s['source_seed'] for s in chosen}
        probes.append(min((s for s in eligible if s['source_seed'] not in used), key=lambda s: rank(s, '141-probe-v1')))
    if len({s['source_seed'] for s in selected+probes}) != STATES+6:
        raise ValueError('141 duplicate source games')
    return selected, probes


def build_manifest(source, old_result, identity):
    excluded = excluded_games(source)
    selected, probes = select_states(source, excluded)
    def bind(rows):
        return [{**r, 'schedule_index': i} for i, r in enumerate(prior140.bind_samples(rows, old_result))]
    return {'identity': identity, 'states': bind(selected), 'probes': bind(probes),
            'state_count': STATES, 'excluded_source_seeds': sorted(excluded),
            'selection_outcome_blind': True, 'worlds_per_state': BLOCKS*WORLDS}


def worlds_for(sample, block, probe=False):
    if not 0 <= block < BLOCKS:
        raise ValueError('141 invalid world block')
    index = (1_439_000_000 if probe else 1_430_000_000)+sample['sample_index']*1000+block*10
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    return base.deterministic_belief_range(state, sample['actor_seat'], root_index=index, count=WORLDS)


def continuation_seed(sample_index, block, world, replicate, probe=False):
    if not (0 <= sample_index < 300 and 0 <= block < BLOCKS and 0 <= world < WORLDS and 0 <= replicate < REPLICATES):
        raise ValueError('141 invalid continuation seed slot')
    return (1_430_000_000_000_000_000 + int(probe)*9_000_000_000_000_000
            + sample_index*1_000_000 + block*1000 + world*10 + replicate)


class TrackingEnv(base.TrackingEnv):
    def __init__(self):
        super().__init__()
        self.actions, self.visits = [], Counter()
        self.no_progress, self.max_no_progress, self.max_visits = 0, 0, 0

    def step(self, state, action):
        signature = base.state_signature(state)
        self.visits[signature] += 1
        self.max_visits = max(self.max_visits, self.visits[signature])
        if state.turn_index >= 400 or self.max_visits >= 4 or self.no_progress >= 60:
            raise ValueError('141 incompatible absolute-turn/repetition/progress boundary')
        self.actions.append(repr(base.canonical_successor_action_key(state, action)))
        after = super().step(state, action)
        self.no_progress = 0 if prior140.is_progress_transition(state, after, action) else self.no_progress+1
        self.max_no_progress = max(self.max_no_progress, self.no_progress)
        return after


def observe(state, action, player, seed, scorer, retain_actions=False):
    env = TrackingEnv()
    result = base.rollout_score_margin_loop_safe(env, scorer, state, action, player, seed, .1, .5, 3, 400, 20., 64)
    end = env.last_state
    if (end is None or not result.terminal or not result.resolved or result.hard_cap_reached
            or result.loop_adjudicated or result.fallback_triggers or not math.isfinite(result.value)
            or end.turn_index >= 400 or env.max_no_progress >= 60):
        raise ValueError('141 non-natural or incompatible continuation; preserve, do not replace')
    out = {**asdict(result), **base.terminal_record(end, player), 'seed': seed,
           'final_turn_index': end.turn_index, 'max_repetition_visits': env.max_visits,
           'max_no_progress': env.max_no_progress, 'action_sequence_sha256': digest(env.actions),
           'immediate_reply': env.actions[1] if len(env.actions) > 1 else None}
    if retain_actions:
        out['actions'] = env.actions
    return out


def structure_block(sample, block, scorer, probe=False):
    worlds = worlds_for(sample, block, probe)
    branches = prior139.prepare_branches(worlds, base.root_keys(sample), scorer)
    prepared, counts = [], Counter()
    for b in branches:
        replies = sorted(repr(base.canonical_successor_action_key(b['after'], r.action)) for r in b['replies'])
        prepared.append({'world': b['world'], 'root': repr(b['root']),
                         'info': digest(b['information_set']), 'replies': replies,
                         'terminal_root': b['after'].terminal})
        if not b['after'].terminal:
            counts['sampled_min'] += len(replies)*REPLICATES
            counts['policy_reply'] += REPLICATES
    return {'block': block, 'world_hashes': [digest(base.serialize_state_snapshot(w)) for w in worlds],
            'branches': prepared, 'expected_continuations': {c: counts[c] for c in CONDITIONS}}


def evaluate_block(sample, block, scorer, probe=False):
    started, cpu_start = time.perf_counter(), time.process_time()
    worlds = worlds_for(sample, block, probe)
    keys = base.root_keys(sample)
    branches = prior139.prepare_branches(worlds, keys, scorer)
    leaves, policy, counts = [], [], {c: Counter() for c in CONDITIONS}
    times, cpus = Counter(), Counter()
    order = CONDITIONS if (sample['schedule_index']+block) % 2 == 0 else CONDITIONS[::-1]
    for condition in order:
        tick, cpu = time.perf_counter(), time.process_time()
        for b in branches:
            w, root, after = b['world'], b['root'], b['after']
            common = {'world': w, 'root': repr(root), 'information_set_sha256': digest(b['information_set'])}
            if after.terminal:
                endpoint = base.terminal_record(after, sample['actor_seat'])
                margin = max(-1., min(1., (endpoint['final_scores'][sample['actor_seat']]-endpoint['final_scores'][1-sample['actor_seat']])/20.))
                row = {**common, 'reply': repr(('terminal',)), 'terminal_root': True,
                       'margin': margin, 'wdl': endpoint['wdl'], 'returns': [endpoint]}
                (leaves if condition == 'sampled_min' else policy).append(row)
                counts[condition]['terminal_roots'] += 1
                continue
            replies = b['replies'] if condition == 'sampled_min' else (None,)
            reply_keys = {repr(base.canonical_successor_action_key(after, r.action)) for r in b['replies']}
            counts[condition]['fewer_than_four_reply_branches'] += int(len(b['replies']) < 4)
            for reply in replies:
                if condition == 'sampled_min':
                    start_state, action = after, reply.action
                else:
                    start_state = worlds[w]
                    action = base.actions_by_canonical_successor_key(start_state, base.SplendorEnv().legal_actions(start_state))[root]
                returns = []
                for rep in range(REPLICATES):
                    seed = continuation_seed(sample['sample_index'], block, w, rep, probe)
                    outcome = observe(start_state, action, sample['actor_seat'], seed, scorer)
                    returns.append(outcome)
                    counts[condition]['continuations'] += 1
                    counts[condition]['resolved_continuations'] += int(outcome['resolved'])
                    counts[condition]['natural_continuations'] += int(outcome['terminal'])
                    counts[condition]['hard_caps'] += int(outcome['hard_cap_reached'])
                    counts[condition]['adjudications'] += int(outcome['loop_adjudicated'])
                    counts[condition]['fallback_triggers'] += outcome['fallback_triggers']
                    if condition == 'policy_reply':
                        counts[condition]['immediate_reply_outside_min_set'] += int(outcome['immediate_reply'] not in reply_keys)
                row = {**common, 'reply': repr(base.canonical_successor_action_key(after, reply.action)) if reply else None,
                       'margin': statistics.fmean(r['value'] for r in returns),
                       'wdl': statistics.fmean(r['wdl'] for r in returns), 'returns': returns}
                (leaves if condition == 'sampled_min' else policy).append(row)
        times[condition], cpus[condition] = time.perf_counter()-tick, time.process_time()-cpu
    records = [{**r, 'root': ast.literal_eval(r['root']), 'reply': ast.literal_eval(r['reply']),
                'info': r['information_set_sha256']} for r in leaves]
    resolved, _, _ = base.resolve_records(records, keys)
    values = {'sampled_min': {k: {'margin': v['margin'], 'wdl': v['wdl_under_margin']} for k, v in resolved.items()},
              'policy_reply': {repr(k): {m: statistics.fmean(r[m] for r in policy if r['root'] == repr(k))
                                        for m in ('margin', 'wdl')} for k in keys}}
    gains = {c: {m: values[c][sample['expanded_action']][m]-values[c][sample['champion_action']][m]
                 for m in ('margin', 'wdl')} for c in CONDITIONS}
    return {'block': block, 'world_hashes': [digest(base.serialize_state_snapshot(w)) for w in worlds],
            'values': values, 'gains': gains, 'leaves': leaves, 'policy_returns': policy,
            'telemetry': {c: dict(t) for c, t in counts.items()}, 'elapsed_by_condition': dict(times),
            'cpu_by_condition': dict(cpus), 'elapsed_seconds': time.perf_counter()-started,
            'cpu_seconds': time.process_time()-cpu_start}


def verify_sample(sample, scorer):
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    if digest(base.serialize_state_snapshot(state)) != digest(sample['state_snapshot']) or state.current_player != sample['actor_seat']:
        raise ValueError('141 snapshot/seat mismatch')
    def inventory(s):
        return sorted([c.card_id for t in (1, 2, 3) for c in s.hidden_tier_decks[t]+s.visible_tier_cards[t]]
                      + [c.card_id for p in s.players for c in p.purchased_cards+p.reserved_cards])
    info, cards = base.canonical_public_belief_key(state, state.current_player), inventory(state)
    altered = state.copy_shallow()
    for tier in (1, 2, 3):
        altered.hidden_tier_decks[tier] = list(reversed(altered.hidden_tier_decks[tier]))
    changed = {**sample, 'state_snapshot': base.serialize_state_snapshot(altered)}
    structures = []
    for block in range(BLOCKS):
        for w, x in zip(worlds_for(sample, block), worlds_for(changed, block), strict=True):
            if base.canonical_public_belief_key(w, state.current_player) != info or inventory(w) != cards:
                raise ValueError('141 world changes observer information or inventory')
            if digest(base.serialize_state_snapshot(w)) != digest(base.serialize_state_snapshot(x)):
                raise ValueError('141 world depends on true hidden deck order')
        structures.append(structure_block(sample, block, scorer))
    return {'sample_index': sample['sample_index'], 'sample_sha256': digest(sample), 'blocks': structures}


def sanitized_probe(sample):
    block = evaluate_block(sample, 0, base._SCORER, probe=True)
    projection = {k: v for k, v in block.items() if k not in ('elapsed_seconds', 'cpu_seconds', 'elapsed_by_condition', 'cpu_by_condition')}
    repeat = None
    if sample['selection_cell'] == 'seat0|late':
        other = evaluate_block(sample, 0, base._SCORER, probe=True)
        repeat = digest(projection) == digest({k: v for k, v in other.items() if k in projection})
        if not repeat:
            raise ValueError('141 probe nondeterminism')
    world = worlds_for(sample, 0, True)[0]
    root = ast.literal_eval(sample['champion_action'])
    action = base.actions_by_canonical_successor_key(world, base.SplendorEnv().legal_actions(world))[root]
    seed = continuation_seed(sample['sample_index'], 0, 0, 0, True)
    observed = observe(world, action, sample['actor_seat'], seed, base._SCORER, retain_actions=True)
    bot = prior140.ScorerPolicy(base._SCORER, seed, True)
    end, trace, reason = prior140.play_from_state(world, root, (bot, bot))
    if reason != 'completed' or observed['actions'] != [t['action'] for t in trace] or base.terminal_record(end, sample['actor_seat']) != {k: observed[k] for k in ('wdl', 'winner', 'final_scores', 'final_card_counts')}:
        raise ValueError('141 policy arm differs from 140 weak continuation')
    return {'sample_index': sample['sample_index'], 'selection_cell': sample['selection_cell'],
            'elapsed_seconds': block['elapsed_seconds'], 'cpu_seconds': block['cpu_seconds'],
            'telemetry': block['telemetry'], 'opaque_digest': digest(projection),
            'repeat_identical': repeat, 'matches_140_action_sequence': True, 'outcomes_retained': False}


def preflight(identity, manifest):
    path = OUTPUT/'preflight.json'
    prepared_identity = {'inputs': identity, 'manifest_sha256': digest(manifest)}
    if path.exists():
        cached = read_json(path)
        if cached['identity'] != prepared_identity:
            raise ValueError('141 cached preflight identity mismatch')
        return cached
    base.initialize_worker()
    started, structures = time.perf_counter(), []
    for i, sample in enumerate(manifest['states'], 1):
        structures.append(verify_sample(sample, base._SCORER))
        if i % 10 == 0:
            print(f'[141:prepare] structures={i}/{STATES}', flush=True)
    probes = []
    for sample in manifest['probes']:
        probe = sanitized_probe(sample)
        probes.append(probe)
        print(f"[141:prepare] timing_probes={len(probes)}/6 seconds={probe['elapsed_seconds']:.2f}", flush=True)
    projection = 2*PER_CELL*BLOCKS*sum(p['elapsed_seconds'] for p in probes)
    expected = {c: sum(b['expected_continuations'][c] for s in structures for b in s['blocks']) for c in CONDITIONS}
    different = sum(s['champion_action'] != s['expanded_action'] for s in manifest['states'])
    passed = projection <= 86_400 and sum(expected.values()) <= MAX_CONTINUATIONS and different >= 12
    result = {'identity': prepared_identity, 'status': 'prepared', 'passed': passed,
              'structures': structures, 'timing_probes': probes, 'projected_serial_seconds': projection,
              'expected_continuations': expected, 'different_fixed_moves': different,
              'elapsed_seconds': time.perf_counter()-started, 'protected_continuations': 0,
              'precision_scenarios': [{'state_contrast_sd': sd, 'normal_ci_halfwidth': 1.96*sd/math.sqrt(STATES)} for sd in (.05, .1, .2)]}
    atomic_write(path, result)
    return result


def evaluate_state(sample):
    start = time.perf_counter()
    blocks = [evaluate_block(sample, i, base._SCORER) for i in range(BLOCKS)]
    gains = {c: {m: statistics.fmean(b['gains'][c][m] for b in blocks) for m in ('margin', 'wdl')} for c in CONDITIONS}
    keep = ('sample_index', 'source_seed', 'turn_index', 'actor_seat', 'phase', 'selection_cell', 'schedule_index', 'champion_action', 'expanded_action')
    return {**{k: sample[k] for k in keep}, 'sample_sha256': digest(sample), 'blocks': blocks,
            'gains': gains, 'elapsed_seconds': time.perf_counter()-start}


def validate_saved(payload, sample, identity, structure):
    row = payload['row']
    if payload['identity'] != identity or row['sample_sha256'] != digest(sample):
        raise ValueError('141 saved identity mismatch')
    for key in ('sample_index', 'source_seed', 'turn_index', 'actor_seat', 'phase', 'selection_cell', 'schedule_index', 'champion_action', 'expanded_action'):
        if key in sample and row.get(key) != sample[key]:
            raise ValueError('141 saved source metadata mismatch')
    if len(row['blocks']) != BLOCKS or {b['block'] for b in row['blocks']} != set(range(BLOCKS)):
        raise ValueError('141 incomplete/duplicate blocks')
    refs = {b['block']: b for b in structure['blocks']}
    for b in row['blocks']:
        ref = refs[b['block']]
        if b['world_hashes'] != ref['world_hashes']:
            raise ValueError('141 world identity mismatch')
        for c in CONDITIONS:
            t = b['telemetry'][c]
            if not (t.get('continuations', 0) == t.get('resolved_continuations', 0) == t.get('natural_continuations', 0) == ref['expected_continuations'][c]):
                raise ValueError('141 continuation accounting mismatch')
            if any(t.get(k, 0) for k in ('hard_caps', 'adjudications', 'fallback_triggers')):
                raise ValueError('141 incompatible continuation')
            if set(b['values'][c]) != {sample['champion_action'], sample['expanded_action']}:
                raise ValueError('141 root choices changed')
            for m in ('margin', 'wdl'):
                gain = b['values'][c][sample['expanded_action']][m]-b['values'][c][sample['champion_action']][m]
                if not math.isfinite(gain) or abs(gain-b['gains'][c][m]) > 1e-12:
                    raise ValueError('141 block gain mismatch')
    for c in CONDITIONS:
        for m in ('margin', 'wdl'):
            if abs(statistics.fmean(b['gains'][c][m] for b in row['blocks'])-row['gains'][c][m]) > 1e-12:
                raise ValueError('141 state mean mismatch')
    return row


def vector(gains, metric):
    a, b = (gains[c][metric] for c in CONDITIONS)
    return [a, b, b-a]


def paired_summary(rows, metric='margin', samples=50_000):
    ordered = sorted(rows, key=lambda r: (r['selection_cell'], r['source_seed']))
    matrix = np.array([vector(r['gains'], metric) for r in ordered])
    rng, boot = np.random.default_rng(141300000), np.zeros((samples, 3))
    for cell in sorted({r['selection_cell'] for r in ordered}):
        ids = np.array([i for i, r in enumerate(ordered) if r['selection_cell'] == cell])
        draws = rng.choice(ids, size=(samples, len(ids)), replace=True)
        boot += matrix[draws].sum(axis=1)/len(ordered)
    ci = np.quantile(boot, [.025, .975], axis=0)
    return {name: {'mean': float(matrix[:, i].mean()), 'ci95': ci[:, i].tolist()} for i, name in enumerate(CONTRASTS)}


def classify(summary, valid=True):
    if not valid:
        return 'opponent_response_calibration_integrity_or_dose_failed'
    low, high = summary['policy_minus_min']['ci95']
    if low > .025 or high < -.025:
        return 'material_response_rule_effect_supported'
    if low > 0 or high < 0:
        return 'response_rule_difference_detected_materiality_uncertain'
    if low >= -.025 and high <= .025:
        return 'response_rule_difference_bounded_below_material_size'
    return 'response_rule_contrast_inconclusive'


def summarize(rows):
    summary, secondary = paired_summary(rows), paired_summary(rows, 'wdl')
    telemetry, elapsed, cpu = {}, {}, {}
    for c in CONDITIONS:
        t = Counter()
        for row in rows:
            for b in row['blocks']:
                t.update(b['telemetry'][c])
        telemetry[c] = dict(t)
        elapsed[c] = sum(b['elapsed_by_condition'][c] for r in rows for b in r['blocks'])
        cpu[c] = sum(b['cpu_by_condition'][c] for r in rows for b in r['blocks'])
    gates = {'120_distinct_games': len(rows) == len({r['source_seed'] for r in rows}) == STATES,
             'balanced_cells': Counter(r['selection_cell'] for r in rows) == Counter({c: PER_CELL for c in CELLS}),
             'different_move_dose': sum(r['champion_action'] != r['expanded_action'] for r in rows) >= 12,
             'maximum_budget': sum(t.get('continuations', 0) for t in telemetry.values()) <= MAX_CONTINUATIONS}
    for c, t in telemetry.items():
        gates[c+'_accounting'] = t.get('continuations', 0) == t.get('natural_continuations', 0) == t.get('resolved_continuations', 0)
        gates[c+'_no_cutoffs_or_fallbacks'] = not any(t.get(k, 0) for k in ('hard_caps', 'adjudications', 'fallback_triggers'))
    block_var = np.mean([np.var([vector(b['gains'], 'margin') for b in r['blocks']], axis=0, ddof=1) for r in rows], axis=0)
    state_var = np.mean([np.var([vector(r['gains'], 'margin') for r in rows if r['selection_cell'] == cell], axis=0, ddof=1) for cell in CELLS], axis=0)
    variance = {name: {'within_block_variance': float(block_var[i]), 'observed_state_variance': float(state_var[i]),
                       'between_state_moment_estimate': float(state_var[i]-block_var[i]/BLOCKS)} for i, name in enumerate(CONTRASTS)}
    controls = {name: ('positive' if summary[name]['ci95'][0] > 0 else 'negative' if summary[name]['ci95'][1] < 0 else 'uncertain') for name in CONTRASTS[:2]}
    half = (summary['policy_minus_min']['ci95'][1]-summary['policy_minus_min']['ci95'][0])/2
    return {'summary': summary, 'secondary_wdl': secondary, 'absolute_gain_status_secondary': controls,
            'primary_ci_halfwidth': half, 'precision_target_met': half <= .025,
            'variance_for_future_design': variance, 'telemetry': telemetry,
            'aggregate_wall_by_condition': elapsed, 'aggregate_cpu_by_condition': cpu,
            'integrity_gates': gates, 'valid': all(gates.values()), 'decision': classify(summary, all(gates.values())),
            'rows': [{k: v for k, v in r.items() if k != 'blocks'} for r in rows],
            'bootstrap': {'samples': 50_000, 'seed': 141300000, 'units': 'source_games_within_six_cells'}}


def run(launch=False):
    with base.process_lock(OUTPUT/'process.lock'):
        identity = input_identity()
        manifest = build_manifest(read_json(base.SELECTION), read_json(base.RESULT086), identity)
        path = OUTPUT/'state_manifest.json'
        if path.exists():
            if read_json(path) != manifest:
                raise ValueError('141 manifest changed')
        else:
            atomic_write(path, manifest)
        if not launch:
            prepared = preflight(identity, manifest)
            print(base.json.dumps({k: prepared[k] for k in ('passed', 'projected_serial_seconds', 'expected_continuations', 'different_fixed_moves')}), flush=True)
            return prepared
        prepared = read_json(OUTPUT/'preflight.json')
        expected = {'inputs': identity, 'manifest_sha256': digest(manifest)}
        if prepared['identity'] != expected or not prepared['passed']:
            raise ValueError('141 unpassed or changed preparation')
        launch_identity = {**expected, 'preflight_sha256': file_hash(OUTPUT/'preflight.json')}
        if (OUTPUT/'result.json').exists():
            final = read_json(OUTPUT/'result.json')
            if final['identity'] != launch_identity:
                raise ValueError('141 completed result identity mismatch')
            return final
        structures = {s['sample_index']: s for s in prepared['structures']}
        rows, pending = [], []
        for s in manifest['states']:
            path = OUTPUT/'states'/f"state_{s['schedule_index']:03d}.json"
            if path.exists():
                rows.append(validate_saved(read_json(path), s, launch_identity, structures[s['sample_index']]))
            else:
                pending.append(s)
        started = time.perf_counter()
        atomic_write(OUTPUT/'run_identity.json', {'identity': launch_identity, 'pid': os.getpid(), 'started_unix': time.time()})
        print(f'[141:run] completed={len(rows)}/{STATES} pending={len(pending)} workers={WORKERS}', flush=True)
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=base.initialize_worker) as executor:
            futures = {executor.submit(evaluate_state, s): s for s in pending}
            for future in as_completed(futures):
                s = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    atomic_write(OUTPUT/'failure.json', {'sample_index': s['sample_index'], 'error': repr(error), 'identity': launch_identity})
                    for f in futures:
                        f.cancel()
                    raise
                payload = {'identity': launch_identity, 'row': row}
                validate_saved(payload, s, launch_identity, structures[s['sample_index']])
                atomic_write(OUTPUT/'states'/f"state_{s['schedule_index']:03d}.json", payload)
                rows.append(row)
                print(f'[141:run] completed={len(rows)}/{STATES} elapsed={time.perf_counter()-started:.1f}s', flush=True)
        if input_identity() != identity:
            raise ValueError('141 frozen inputs changed during evaluation')
        rows.sort(key=lambda r: r['schedule_index'])
        final = {'identity': launch_identity, 'status': 'complete', **summarize(rows),
                 'elapsed_this_session_seconds': time.perf_counter()-started,
                 'automatic_continuation': False, 'training': False, 'promotion': False,
                 'full_game_arena': False, 'strong_player_search': False, 'research_champion': '084', 'fast_champion': '035'}
        atomic_write(OUTPUT/'result.json', final)
        print(f"[141:complete] decision={final['decision']}", flush=True)
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    run(launch=parser.parse_args().launch)


if __name__ == '__main__':
    main()
