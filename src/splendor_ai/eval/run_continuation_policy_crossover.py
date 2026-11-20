"""Experiment 140: frozen move pairs finished by different policies."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np

from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.eval import run_independent_world_root_validation as base
from splendor_ai.eval import run_independent_confirmation_root_policy as prior139
from splendor_ai.eval.run_paired_public_belief_resolver_gameplay import (
    FrozenPublicBeliefResolverBot, ResolverGameplayConfig, _frozen_contract_matches_049,
)
from splendor_ai.training.run_rollout_action_credit import _sample_scored_action

PROTOCOL = Path('docs/engineering/continuation_policy_crossover_140.md')
OUTPUT = Path('outputs/continuation_policy_crossover_140')
WRAPPER = Path('run_continuation_policy_crossover_140.py')
TEST = Path('tests/test_continuation_policy_crossover_140.py')
VERIFIER = Path('scripts/analysis/verify_continuation_policy_crossover_140.py')
CONDITIONS = ('stochastic_035', 'greedy_035', 'resolver_084')
CELLS = base.CELLS
SIZES = (60, 48, 36, 24)
WORLDS, WORKERS = 3, 10
CONTRASTS = ('stochastic_gain', 'strong_gain', 'strong_minus_stochastic',
             'greedy_gain', 'strong_minus_greedy', 'greedy_minus_stochastic')
digest, file_hash = base.digest, base.file_hash
read_json, atomic_write = base.read_json, base.atomic_write


class ContinuationIntegrityError(RuntimeError):
    pass


def resolver_config():
    cfg = ResolverGameplayConfig(output_dir=OUTPUT, smoke_seed=1_400_000,
        direct_seed_start=1_400_100, belief_root_index_start=1_420_000_000,
        continuation_seed_start=1_400_000_000_000_000_000)
    if not _frozen_contract_matches_049(cfg):
        raise ValueError('140 continuation resolver differs from exact 084 contract')
    return cfg


def input_identity():
    sources = {name: file_hash(Path(name)) for name in base.SOURCE_HASHES}
    if sources != base.SOURCE_HASHES:
        raise ValueError('140 source/checkpoint identity mismatch')
    files = sorted(Path('src/splendor_ai').rglob('*.py'))+[WRAPPER, TEST, VERIFIER]
    resolver_config()
    return {'format': 'continuation_policy_crossover_140_v1', 'sources': sources,
            'protocol_sha256': file_hash(PROTOCOL),
            'dependencies': {p.as_posix(): file_hash(p) for p in files},
            'workers': WORKERS, 'torch_threads': 1}


def select_candidates(source):
    excluded = {s['source_seed'] for s in base.select_states(source)+prior139.select_states(source)}
    def rank(sample, prefix):
        key = f"{prefix}|{sample['source_seed']}|{sample['turn_index']}|{sample['actor_seat']}|{sample['phase']}"
        return hashlib.sha256(key.encode()).hexdigest(), sample['source_seed'], sample['turn_index']
    maximum, probes = [], []
    for cell in CELLS:
        eligible = [s for s in source['selected'] if s['selection_cell'] == cell and s['source_seed'] not in excluded]
        if len(eligible) < 11:
            raise ValueError('140 insufficient disjoint source states')
        chosen = sorted(eligible, key=lambda s: rank(s, '140-selection-v1'))[:10]
        maximum.extend(chosen)
        selected_ids = {s['source_seed'] for s in chosen}
        probes.append(min((s for s in eligible if s['source_seed'] not in selected_ids),
                          key=lambda s: rank(s, '140-probe-v1')))
    if len({s['source_seed'] for s in maximum+probes}) != 66:
        raise ValueError('140 source games are not distinct')
    return maximum, probes, sorted(excluded)


def bind_samples(samples, result):
    old_rows = {r['sample_index']: r for r in result['rows']}
    bound = []
    for index, sample in enumerate(samples):
        old = old_rows[sample['sample_index']]
        for key in ('source_seed', 'turn_index', 'actor_seat', 'phase', 'champion_root_keys', 'expanded_root_keys'):
            if sample[key] != old[key]:
                raise ValueError(f'140 original choice binding mismatch: {key}')
        keep = ('sample_index', 'source_seed', 'turn_index', 'actor_seat', 'phase', 'selection_cell', 'state_snapshot')
        bound.append({**{k: sample[k] for k in keep}, 'maximum_schedule_index': index,
            'champion_action': sample['champion_root_keys'][old['champion_selected_index']],
            'expanded_action': sample['expanded_root_keys'][old['expanded_selected_index']]})
    return bound


def worlds_for(sample, probe=False):
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    index = (1_409_000_000 if probe else 1_400_000_000)+sample['sample_index']*10
    return base.deterministic_belief_range(state, sample['actor_seat'], root_index=index, count=1 if probe else WORLDS)


def policy_slot(sample_index, world, probe=False):
    if not (0 <= sample_index < 300 and 0 <= world < (1 if probe else WORLDS)):
        raise ValueError('140 invalid policy seed slot')
    return sample_index*3+world+(10_000 if probe else 0)


class ScorerPolicy:
    def __init__(self, scorer, seed, stochastic):
        self.scorer, self.rng, self.stochastic = scorer, random.Random(seed), stochastic

    def choose_action(self, env, state, legal_actions):
        scores = self.scorer.score_actions(state, legal_actions)
        if not self.stochastic:
            return scores[0].action
        return _sample_scored_action(scores, rng=self.rng, epsilon=.1, temperature=.5, top_k=3)


def make_bots(condition, slot, scorer):
    if condition not in CONDITIONS:
        raise ValueError('140 unknown continuation policy')
    if condition == 'resolver_084':
        return tuple(FrozenPublicBeliefResolverBot(resolver_config(), 1_400_000+slot, seat,
                      scorer=scorer, continuation_workers=1) for seat in (0, 1))
    bot = ScorerPolicy(scorer, 1_410_000_000_000_000_000+slot*1_000_000,
                       stochastic=condition == 'stochastic_035')
    return bot, bot


def play_from_state(state, forced_key, bots, env=None):
    """Native match termination semantics, resumed snapshot and one forced move."""
    env = env or base.SplendorEnv()
    visits, trace = {}, []
    no_progress, forced_pass_streak = 0, 0
    while not state.terminal:
        signature = state_signature(state)
        seen = visits.get(signature, 0)+1
        visits[signature] = seen
        if seen >= 4:
            raise ContinuationIntegrityError('repetition_cutoff')
        if state.turn_index >= 400:
            raise ContinuationIntegrityError('max_turns')
        legal = env.legal_actions(state)
        if not legal:
            raise ContinuationIntegrityError('stalled')
        if not trace:
            mapping = base.actions_by_canonical_successor_key(state, legal)
            if forced_key not in mapping:
                raise ContinuationIntegrityError('illegal_forced_root')
            action = mapping[forced_key]
        else:
            action = bots[state.current_player].choose_action(env, state, legal)
            if action is None or action not in legal:
                raise ContinuationIntegrityError('illegal_continuation_action')
        after = env.step(state, action)
        trace.append({'turn_index': state.turn_index, 'actor_seat': state.current_player,
                      'action': repr(base.canonical_successor_action_key(state, action)),
                      'forced': len(trace) == 0,
                      'before_sha256': digest(base.serialize_state_snapshot(state)),
                      'after_sha256': digest(base.serialize_state_snapshot(after))})
        forced_pass_streak = forced_pass_streak+1 if base.is_forced_pass_turn(legal) else 0
        if forced_pass_streak >= 2:
            return after, trace, 'forced_pass_deadlock'
        no_progress = 0 if is_progress_transition(state, after, action) else no_progress+1
        if no_progress >= 60:
            raise ContinuationIntegrityError('no_progress_cutoff')
        state = after
    return state, trace, 'completed'


def evaluate_trajectory(sample, world_index, condition, root_action, probe=False):
    start, cpu_start = time.perf_counter(), time.process_time()
    worlds = worlds_for(sample, probe)
    world = worlds[world_index]
    slot = policy_slot(sample['sample_index'], world_index, probe)
    bots = make_bots(condition, slot, base._SCORER)
    initial = base.serialize_state_snapshot(world)
    try:
        end, trace, reason = play_from_state(world, ast.literal_eval(root_action), bots)
        endpoint = base.terminal_record(end, sample['actor_seat'])
        if end.terminal and end.winner != endpoint['winner']:
            raise ContinuationIntegrityError('engine_winner_tiebreak_mismatch')
        raw_margin = endpoint['final_scores'][sample['actor_seat']]-endpoint['final_scores'][1-sample['actor_seat']]
        telemetry = [bot.telemetry for bot in bots] if condition == 'resolver_084' else []
        for t in telemetry:
            if (t['continuations'] != 192*t['completed_searches']
                    or t['continuations'] != t['resolved_continuations'] or t['hard_caps']):
                raise ContinuationIntegrityError('084_inner_continuation_integrity')
        return {'sample_index': sample['sample_index'], 'sample_sha256': digest(sample),
                'world_index': world_index, 'condition': condition, 'root_action': root_action,
                'probe': probe, 'policy_slot': slot, 'world_snapshot': initial,
                'world_sha256': digest(initial), 'trace': trace,
                'final_state': base.serialize_state_snapshot(end), 'termination_reason': reason,
                **endpoint, 'raw_score_margin': raw_margin, 'margin': max(-1., min(1., raw_margin/20.)),
                'outer_turns': len(trace), 'resolver_telemetry': telemetry,
                'elapsed_seconds': time.perf_counter()-start, 'cpu_seconds': time.process_time()-cpu_start}
    finally:
        for bot in {id(b): b for b in bots}.values():
            close = getattr(bot, 'close', None)
            if close:
                close()


def deterministic_projection(row):
    return {k: v for k, v in row.items() if k not in ('elapsed_seconds', 'cpu_seconds')}


def run_probe(sample):
    results = {}
    for condition in CONDITIONS:
        row = evaluate_trajectory(sample, 0, condition, sample['champion_action'], probe=True)
        results[condition] = {'elapsed_seconds': row['elapsed_seconds'], 'cpu_seconds': row['cpu_seconds'],
            'termination_reason': row['termination_reason'], 'outer_turns': row['outer_turns'],
            'resolver_telemetry': row['resolver_telemetry'],
            'deterministic_digest': digest(deterministic_projection(row))}
    repeated = None
    if sample['selection_cell'] == 'seat0|late':
        row = evaluate_trajectory(sample, 0, 'resolver_084', sample['champion_action'], probe=True)
        repeated = {'identical': digest(deterministic_projection(row)) == results['resolver_084']['deterministic_digest'],
                    'elapsed_seconds': row['elapsed_seconds'], 'cpu_seconds': row['cpu_seconds']}
        if not repeated['identical']:
            raise ContinuationIntegrityError('084_probe_repeat_mismatch')
    return {'sample_index': sample['sample_index'], 'sample_sha256': digest(sample),
            'selection_cell': sample['selection_cell'], 'phase': sample['phase'],
            'timings_and_integrity': results, 'repeat': repeated, 'outcomes_retained': False}


def choose_panel_size(probes, ceiling=129_600.):
    phase_times = {phase: max(sum(p['timings_and_integrity'][c]['elapsed_seconds'] for c in CONDITIONS)
                              for p in probes if p['phase'] == phase) for phase in ('early', 'mid', 'late')}
    projections = {str(n): 1.5*(n/3)*2*WORLDS*sum(phase_times.values()) for n in SIZES}
    n = next((n for n in SIZES if projections[str(n)] <= ceiling), None)
    return n, phase_times, projections


def select_prefix(maximum, n):
    if n not in SIZES:
        raise ValueError('140 unsupported panel size')
    return [s for cell in CELLS for s in [r for r in maximum if r['selection_cell'] == cell][:n//6]]


def verify_structure(sample, probe=False):
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    if digest(base.serialize_state_snapshot(state)) != digest(sample['state_snapshot']) or state.current_player != sample['actor_seat']:
        raise ValueError('140 snapshot round-trip or actor mismatch')
    observer_key = base.canonical_public_belief_key(state, state.current_player)
    def inventory(s):
        return sorted([c.card_id for t in (1, 2, 3) for c in s.hidden_tier_decks[t]+s.visible_tier_cards[t]]
                      + [c.card_id for p in s.players for c in p.purchased_cards+p.reserved_cards])
    changed = state.copy_shallow()
    for tier in (1, 2, 3):
        changed.hidden_tier_decks[tier] = list(reversed(changed.hidden_tier_decks[tier]))
    other = {**sample, 'state_snapshot': base.serialize_state_snapshot(changed)}
    worlds = worlds_for(sample, probe)
    for world, reordered in zip(worlds, worlds_for(other, probe), strict=True):
        if base.canonical_public_belief_key(world, state.current_player) != observer_key or inventory(world) != inventory(state):
            raise ValueError('140 sampled world changes observer information or card inventory')
        if digest(base.serialize_state_snapshot(world)) != digest(base.serialize_state_snapshot(reordered)):
            raise ValueError('140 world depends on actual hidden deck ordering')
        legal = base.actions_by_canonical_successor_key(world, base.SplendorEnv().legal_actions(world))
        if not set(base.root_keys(sample)).issubset(legal):
            raise ValueError('140 original root not legal in sampled world')
    return {'sample_index': sample['sample_index'], 'probe': probe,
            'world_hashes': [digest(base.serialize_state_snapshot(w)) for w in worlds]}


def validate_saved(payload, identity, sample, world_index, condition, action, structure):
    if payload.get('identity') != identity:
        raise ValueError('140 saved trajectory identity mismatch')
    row = payload['row']
    if (row['sample_sha256'] != digest(sample) or row['sample_index'] != sample['sample_index']
            or row['world_index'] != world_index or row['condition'] != condition
            or row['root_action'] != action or row['probe']
            or row['policy_slot'] != policy_slot(sample['sample_index'], world_index)):
        raise ValueError('140 saved trajectory binding mismatch')
    if row['world_sha256'] != structure['world_hashes'][world_index] or digest(row['world_snapshot']) != row['world_sha256']:
        raise ValueError('140 saved starting world mismatch')
    if row['termination_reason'] not in ('completed', 'forced_pass_deadlock'):
        raise ValueError('140 cutoff outcome cannot be evaluated')
    trace = row['trace']
    if not trace or trace[0]['action'] != action or not trace[0]['forced'] or any(t['forced'] for t in trace[1:]):
        raise ValueError('140 forced-root trajectory mismatch')
    previous = row['world_sha256']
    for step in trace:
        if step['before_sha256'] != previous:
            raise ValueError('140 broken trace chain')
        previous = step['after_sha256']
    if previous != digest(row['final_state']) or len(trace) != row['outer_turns']:
        raise ValueError('140 incomplete final trace')
    final_state = base.deserialize_state_snapshot(row['final_state'])
    if row['termination_reason'] == 'completed' and not final_state.terminal:
        raise ValueError('140 nonterminal natural endpoint')
    endpoint = base.terminal_record(final_state, sample['actor_seat'])
    if any(row[k] != v for k, v in endpoint.items()):
        raise ValueError('140 terminal scoring mismatch')
    margin = row['final_scores'][sample['actor_seat']]-row['final_scores'][1-sample['actor_seat']]
    if row['raw_score_margin'] != margin or row['margin'] != max(-1., min(1., margin/20.)):
        raise ValueError('140 normalized margin mismatch')
    telemetry = row['resolver_telemetry']
    if condition == 'resolver_084':
        if len(telemetry) != 2:
            raise ValueError('140 missing resolver telemetry')
        for seat, t in enumerate(telemetry):
            decisions = sum(step['actor_seat'] == seat for step in trace[1:])
            if (t['actor_seat'] != seat or t['engine_seed'] != 1_400_000+row['policy_slot']
                    or t['decisions'] != decisions or t['decisions'] != t['completed_searches']+t['fallbacks']
                    or t['continuations'] != 192*t['completed_searches']
                    or t['continuations'] != t['resolved_continuations'] or t['hard_caps']
                    or not 0 <= t['switches'] <= t['completed_searches']):
                raise ValueError('140 saved resolver telemetry mismatch')
    elif telemetry:
        raise ValueError('140 unexpected resolver telemetry')
    return row


def trajectory_name(sample, world, condition, action):
    return f"s{sample['sample_index']:03d}_w{world}_{condition}_{digest(action)[:16]}.json"


def summarize_state(sample, trajectories):
    by_key = {(r['world_index'], r['condition'], r['root_action']): r for r in trajectories}
    expected = {(w, c, a) for w in range(WORLDS) for c in CONDITIONS
                for a in dict.fromkeys((sample['champion_action'], sample['expanded_action']))}
    if len(by_key) != len(trajectories) or set(by_key) != expected:
        raise ValueError('140 incomplete state trajectory set')
    gains = {c: {metric: statistics.fmean(by_key[w, c, sample['expanded_action']][metric]
                                          - by_key[w, c, sample['champion_action']][metric] for w in range(WORLDS))
                 for metric in ('margin', 'wdl', 'raw_score_margin')} for c in CONDITIONS}
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    private = sum(p.reserved_card_source(i) != base.RESERVE_SOURCE_VISIBLE
                  for p in state.players for i in range(len(p.reserved_cards)))
    return {**{k: v for k, v in sample.items() if k != 'state_snapshot'}, 'gains': gains,
            'existing_blind_reserves': private}


def contrast_vector(row, metric):
    weak, greedy, strong = (row['gains'][c][metric] for c in CONDITIONS)
    return [weak, strong, strong-weak, greedy, strong-greedy, greedy-weak]


def paired_summary(rows, metric='margin', samples=50_000):
    ordered = sorted(rows, key=lambda r: (r['selection_cell'], r['source_seed']))
    matrix = np.array([contrast_vector(r, metric) for r in ordered])
    rng, boot = np.random.default_rng(140300000), np.zeros((samples, len(CONTRASTS)))
    for cell in sorted({r['selection_cell'] for r in ordered}):
        ids = np.array([i for i, r in enumerate(ordered) if r['selection_cell'] == cell])
        draws = rng.choice(ids, size=(samples, len(ids)), replace=True)
        boot += matrix[draws].sum(axis=1)/len(ordered)
    ci = np.quantile(boot, [.025, .975], axis=0)
    return {name: {'mean': float(matrix[:, i].mean()), 'ci95': ci[:, i].tolist()} for i, name in enumerate(CONTRASTS)}


def classify(summary, valid=True):
    if not valid:
        return 'continuation_policy_crossover_integrity_or_dose_failed'
    weak, strong, change = (summary[k] for k in ('stochastic_gain', 'strong_gain', 'strong_minus_stochastic'))
    if weak['mean'] <= 0:
        return 'stochastic_control_gain_not_reproduced'
    if weak['ci95'][0] <= 0:
        return 'stochastic_control_gain_uncertain'
    if change['mean'] <= -.025 and change['ci95'][1] < 0:
        return 'continuation_policy_mismatch_supported'
    if strong['ci95'][0] > 0 and change['ci95'][0] > -.025:
        return 'advantage_survives_stronger_continuations'
    return 'continuation_policy_crossover_inconclusive'


def summarize(manifest, records):
    n = manifest['state_count']
    rows = [summarize_state(s, [r for r in records if r['sample_index'] == s['sample_index']]) for s in manifest['states']]
    telemetry = {}
    for condition in CONDITIONS:
        group = [r for r in records if r['condition'] == condition]
        count = Counter()
        for r in group:
            count[r['termination_reason']] += 1
            count['outer_turns'] += r['outer_turns']
            for t in r['resolver_telemetry']:
                for key in ('decisions', 'completed_searches', 'switches', 'fallbacks', 'continuations',
                            'resolved_continuations', 'hard_caps', 'loop_adjudications', 'rollout_fallback_triggers'):
                    count[key] += t[key]
        telemetry[condition] = {'outer_continuations': len(group), **dict(count),
            'aggregate_wall_seconds': sum(r['elapsed_seconds'] for r in group),
            'aggregate_cpu_seconds': sum(r['cpu_seconds'] for r in group)}
    strong = telemetry['resolver_084']
    phase_searches = {p: sum(t['completed_searches'] for r in records if r['condition'] == 'resolver_084'
                            and next(s for s in manifest['states'] if s['sample_index'] == r['sample_index'])['phase'] == p
                            for t in r['resolver_telemetry']) for p in ('early', 'mid', 'late')}
    gates = {'selected_state_count': len(rows) == n, 'distinct_source_games': len({r['source_seed'] for r in rows}) == n,
             'balanced_cells': Counter(r['selection_cell'] for r in rows) == Counter({c: n//6 for c in CELLS}),
             'exact_outer_accounting': len(records) == manifest['expected_outer_continuations'],
             'maximum_outer_work': len(records) <= n*18,
             'all_outer_endpoints_valid': all(r['termination_reason'] in ('completed', 'forced_pass_deadlock') for r in records),
             'strong_inner_accounting': strong.get('continuations', 0) == 192*strong.get('completed_searches', 0)
                 == strong.get('resolved_continuations', 0),
             'no_inner_hard_caps': strong.get('hard_caps', 0) == 0,
             'bounded_inner_adjudications': strong.get('loop_adjudications', 0)/max(1, strong.get('continuations', 0)) <= .001,
             'strong_operator_exercised_all_phases': all(v > 0 for v in phase_searches.values()),
             'strong_operator_switched': strong.get('switches', 0) > 0}
    summary = paired_summary(rows)
    breakdowns = {field: {str(value): {'states': len(group),
                    **{c: statistics.fmean(r['gains'][c]['margin'] for r in group) for c in CONDITIONS}}
                    for value in sorted({r[field] for r in rows})
                    for group in [[r for r in rows if r[field] == value]]}
                  for field in ('phase', 'actor_seat', 'existing_blind_reserves')}
    return {'valid': all(gates.values()), 'decision': classify(summary, all(gates.values())),
            'summary': summary, 'secondary_wdl': paired_summary(rows, 'wdl'),
            'secondary_raw_margin': paired_summary(rows, 'raw_score_margin'),
            'integrity_gates': gates, 'telemetry': telemetry, 'source_phase_searches': phase_searches,
            'rows': rows, 'breakdowns': breakdowns,
            'bootstrap': {'samples': 50_000, 'seed': 140300000, 'units': 'source_games_within_six_cells'}}


def prepare(identity):
    source, decisions = read_json(base.SELECTION), read_json(base.RESULT086)
    maximum, probes, excluded = select_candidates(source)
    maximum, probes = bind_samples(maximum, decisions), bind_samples(probes, decisions)
    preparation = {'identity': identity, 'maximum_panel': maximum, 'probes': probes, 'excluded_source_seeds': excluded}
    path = OUTPUT/'preparation_manifest.json'
    if path.exists():
        if read_json(path) != preparation:
            raise ValueError('140 preparation manifest changed')
    else:
        atomic_write(path, preparation)
    prep_id = {'inputs': identity, 'preparation_manifest_sha256': digest(preparation)}
    preflight_path = OUTPUT/'preflight.json'
    if preflight_path.exists():
        result = read_json(preflight_path)
        if result['identity'] != prep_id:
            raise ValueError('140 preflight identity mismatch')
        return result
    structures = [verify_structure(s) for s in maximum]
    probe_structures = [verify_structure(s, True) for s in probes]
    print('[140:prepare] all 66 source structures verified; starting outcome-blind timing probes', flush=True)
    timings, pending = [], []
    for sample in probes:
        path = OUTPUT/'probes'/f"probe_{sample['sample_index']:03d}.json"
        if path.exists():
            payload = read_json(path)
            if payload['identity'] != prep_id or payload['row']['sample_sha256'] != digest(sample):
                raise ValueError('140 cached probe identity changed')
            timings.append(payload['row'])
        else:
            pending.append(sample)
    with ProcessPoolExecutor(max_workers=6, initializer=base.initialize_worker) as executor:
        futures = {executor.submit(run_probe, s): s for s in pending}
        try:
            for future in as_completed(futures):
                sample, row = futures[future], future.result()
                atomic_write(OUTPUT/'probes'/f"probe_{sample['sample_index']:03d}.json", {'identity': prep_id, 'row': row})
                timings.append(row)
                print(f'[140:prepare] timing probes complete={len(timings)}/6', flush=True)
        except BaseException as error:
            for future in futures:
                future.cancel()
            atomic_write(OUTPUT/'preparation_failure.json', {'identity': prep_id,
                         'error_type': type(error).__name__, 'reason': str(error),
                         'completed_probes': len(timings), 'protected_continuations': 0})
            raise
    n, phase_times, projections = choose_panel_size(timings)
    repeated = [p['repeat'] for p in timings if p['repeat'] is not None]
    probe_inner = [t for p in timings for c in CONDITIONS for t in p['timings_and_integrity'][c]['resolver_telemetry']]
    inner_n = sum(t['continuations'] for t in probe_inner)
    loops = sum(t['loop_adjudications'] for t in probe_inner)
    passed = n is not None and len(repeated) == 1 and repeated[0]['identical'] and inner_n > 0 and loops/inner_n <= .001
    result = {'identity': prep_id, 'passed': passed, 'status': 'prepared' if passed else 'preparation_failed',
              'selected_state_count': n, 'phase_serial_seconds': phase_times, 'projected_serial_seconds': projections,
              'timing_probes': timings, 'maximum_structures': structures, 'probe_structures': probe_structures,
              'protected_continuations': 0, 'probe_inner_continuations': inner_n,
              'probe_inner_adjudications': loops}
    if n is not None:
        states = select_prefix(maximum, n)
        manifest = {'identity': identity, 'state_count': n, 'states': states,
                    'excluded_source_seeds': excluded, 'probe_source_seeds': [s['source_seed'] for s in probes],
                    'expected_outer_continuations': sum(len(base.root_keys(s))*WORLDS*len(CONDITIONS) for s in states),
                    'worlds_per_state': WORLDS, 'selection_based_only_on_ids_and_timing': True}
        atomic_write(OUTPUT/'state_manifest.json', manifest)
        result['state_manifest_sha256'] = digest(manifest)
        result['precision_scenarios'] = [{'state_contrast_sd': sd, 'normal_ci95_half_width': 1.96*sd/math.sqrt(n)}
                                         for sd in (.05, .1, .2)]
    atomic_write(preflight_path, result)
    print({k: result[k] for k in ('status', 'passed', 'selected_state_count', 'projected_serial_seconds')}, flush=True)
    return result


def run(launch=False):
    with base.process_lock(OUTPUT/'process.lock'):
        identity = input_identity()
        if not launch:
            return prepare(identity)
        prepared = read_json(OUTPUT/'preflight.json')
        preparation = read_json(OUTPUT/'preparation_manifest.json')
        if (prepared['identity'] != {'inputs': identity, 'preparation_manifest_sha256': digest(preparation)}
                or not prepared['passed']):
            raise ValueError('140 preflight failed or identities changed')
        manifest = read_json(OUTPUT/'state_manifest.json')
        if digest(manifest) != prepared['state_manifest_sha256'] or manifest['identity'] != identity:
            raise ValueError('140 selected manifest changed')
        launch_id = {'inputs': identity, 'state_manifest_sha256': digest(manifest),
                     'preflight_sha256': file_hash(OUTPUT/'preflight.json')}
        final_path = OUTPUT/'result.json'
        if final_path.exists():
            result = read_json(final_path)
            if result['identity'] != launch_id:
                raise ValueError('140 completed result identity changed')
            return result
        structures = {s['sample_index']: s for s in prepared['maximum_structures']}
        pending, records = [], []
        for sample in manifest['states']:
            for w in range(WORLDS):
                order = CONDITIONS if (sample['maximum_schedule_index']+w) % 2 == 0 else CONDITIONS[::-1]
                for c in order:
                    for action in dict.fromkeys((sample['champion_action'], sample['expanded_action'])):
                        path = OUTPUT/'trajectories'/trajectory_name(sample, w, c, action)
                        if path.exists():
                            records.append(validate_saved(read_json(path), launch_id, sample, w, c, action, structures[sample['sample_index']]))
                        else:
                            pending.append((sample, w, c, action))
        started = time.perf_counter()
        total = manifest['expected_outer_continuations']
        atomic_write(OUTPUT/'run_identity.json', {'identity': launch_id, 'pid': os.getpid(), 'started_unix': time.time()})
        print(f'[140:run] complete={len(records)}/{total} workers={WORKERS}', flush=True)
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=base.initialize_worker) as executor:
            futures = {executor.submit(evaluate_trajectory, *job): job for job in pending}
            try:
                for future in as_completed(futures):
                    sample, w, c, action = futures[future]
                    payload = {'identity': launch_id, 'row': future.result()}
                    validate_saved(payload, launch_id, sample, w, c, action, structures[sample['sample_index']])
                    atomic_write(OUTPUT/'trajectories'/trajectory_name(sample, w, c, action), payload)
                    records.append(payload['row'])
                    print(f'[140:run] complete={len(records)}/{total} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            except BaseException as error:
                for future in futures:
                    future.cancel()
                atomic_write(OUTPUT/'integrity_failure.json', {'identity': launch_id, 'error_type': type(error).__name__,
                             'reason': str(error), 'completed_trajectories': len(records), 'no_scientific_conclusion': True})
                raise
        if input_identity() != identity:
            raise ValueError('140 frozen inputs changed during execution')
        result = {'identity': launch_id, 'status': 'complete', **summarize(manifest, records),
                  'state_count': manifest['state_count'], 'elapsed_this_session_seconds': time.perf_counter()-started,
                  'training': False, 'promotion': False, 'automatic_continuation': False,
                  'full_game_arena': False, 'research_champion': '084', 'fast_champion': '035'}
        atomic_write(final_path, result)
        print(f"[140:complete] decision={result['decision']}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    run(parser.parse_args().launch)


if __name__ == '__main__':
    main()
