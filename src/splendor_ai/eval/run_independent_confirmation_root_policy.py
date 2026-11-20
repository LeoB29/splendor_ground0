"""Frozen propose/check/test policy experiment 139; no model training."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np

from splendor_ai.eval import run_independent_world_root_validation as base

PROTOCOL = Path('docs/engineering/independent_confirmation_root_policy_139.md')
OUTPUT = Path('outputs/independent_confirmation_root_policy_139')
WRAPPER = Path('run_independent_confirmation_root_policy_139.py')
TEST = Path('tests/test_independent_confirmation_root_policy_139.py')
VERIFIER = Path('scripts/analysis/verify_independent_confirmation_139.py')
CELLS = base.CELLS
CHECKS = ('same_world_check', 'fresh_world_check')
CONDITIONS = (*CHECKS, 'test')
METRICS = ('margin', 'wdl_under_margin', 'wdl_min')
POLICIES = ('champion', 'unfiltered', 'same_world', 'independent')
CONTRASTS = (('independent', 'champion'), ('independent', 'unfiltered'),
             ('independent', 'same_world'), ('same_world', 'champion'),
             ('unfiltered', 'champion'))
BLOCKS, WORKERS, MAX_CONTINUATIONS = 8, 10, 138_240
digest, file_hash = base.digest, base.file_hash
read_json, atomic_write = base.read_json, base.atomic_write


def input_identity():
    sources = {name: file_hash(Path(name)) for name in base.SOURCE_HASHES}
    if sources != base.SOURCE_HASHES:
        raise ValueError('139 frozen source/checkpoint mismatch')
    dependencies = sorted(Path('src/splendor_ai').rglob('*.py')) + [WRAPPER, TEST, VERIFIER]
    return {'format': 'independent_confirmation_root_policy_139_v1', 'sources': sources,
            'protocol_sha256': file_hash(PROTOCOL),
            'dependencies': {p.as_posix(): file_hash(p) for p in dependencies},
            'workers': WORKERS, 'torch_threads': 1}


def select_states(source):
    excluded = {r['source_seed'] for r in base.select_states(source)}
    selected = []
    for cell in CELLS:
        rows = [r for r in source['selected']
                if r['selection_cell'] == cell and r['source_seed'] not in excluded]
        def rank(r):
            text = f"139-selection-v1|{r['source_seed']}|{r['turn_index']}|{r['actor_seat']}|{r['phase']}"
            return hashlib.sha256(text.encode()).hexdigest(), r['source_seed'], r['turn_index']
        if len(rows) < 10:
            raise ValueError('139 insufficient unexposed source cell')
        selected.extend(sorted(rows, key=rank)[:10])
    if len({r['source_seed'] for r in selected}) != 60:
        raise ValueError('139 duplicate source games')
    return selected


def build_manifest(source, result, identity):
    selected = select_states(source)
    decisions = {r['sample_index']: r for r in result['rows']}
    rows = []
    for i, sample in enumerate(selected):
        old = decisions[sample['sample_index']]
        for key in ('source_seed', 'turn_index', 'actor_seat', 'phase',
                    'expanded_root_keys', 'champion_root_keys'):
            if old[key] != sample[key]:
                raise ValueError(f'139 original proposal binding mismatch: {key}')
        keep = ('sample_index', 'source_seed', 'turn_index', 'actor_seat', 'phase',
                'selection_cell', 'state_snapshot')
        rows.append({**{k: sample[k] for k in keep}, 'schedule_index': i,
                     'champion_action': sample['champion_root_keys'][old['champion_selected_index']],
                     'expanded_action': sample['expanded_root_keys'][old['expanded_selected_index']],
                     'source_root_count': len(sample['expanded_root_keys'])})
    return {'identity': identity, 'states': rows, 'state_count': 60,
            'selection_outcome_blind': True, 'disjoint_from_138': True,
            'excluded_138_source_seeds': sorted(s['source_seed'] for s in base.select_states(source))}


def worlds_for(sample, condition, block):
    if condition not in CONDITIONS or block not in range(BLOCKS):
        raise ValueError('139 invalid world slot')
    if condition == CHECKS[0]:
        return base.worlds_for(sample, 'original_worlds', 0)
    index = (1_390_100_000 if condition == CHECKS[1] else 1_390_200_000)
    index += sample['sample_index'] * 100 + block * 10
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    return base.deterministic_belief_range(state, sample['actor_seat'], root_index=index, count=3)


def continuation_seed(sample_index, condition, block, world, replicate):
    if not (0 <= sample_index < 300 and condition in CONDITIONS and 0 <= block < BLOCKS
            and 0 <= world < 3 and 0 <= replicate < 4):
        raise ValueError('139 invalid continuation slot')
    return (1_390_000_000_000_000_000 + int(condition == 'test') * 100_000_000
            + sample_index * 100_000 + block * 1_000 + world * 10 + replicate)


def screen(gains, different=True):
    if len(gains) != BLOCKS or not all(math.isfinite(g) for g in gains):
        raise ValueError('139 screen needs eight finite check gains')
    mean = statistics.fmean(gains)
    se = statistics.stdev(gains) / math.sqrt(BLOCKS)
    lower = mean - 1.895 * se
    return {'mean_gain': mean, 'standard_error': se, 'screening_lower_score': lower,
            'switch': bool(different and mean >= .025 and lower > 0)}


def decisions_from_checks(sample, blocks):
    if len(blocks) != 16 or {(b['condition'], b['block']) for b in blocks} != {
            (c, i) for c in CHECKS for i in range(BLOCKS)}:
        raise ValueError('139 incomplete/contaminated check blocks')
    different = sample['champion_action'] != sample['expanded_action']
    return {policy: screen([b['gains']['margin'] for b in sorted(blocks, key=lambda x: x['block'])
                            if b['condition'] == condition], different)
            for policy, condition in zip(('same_world', 'independent'), CHECKS)}


def policy_gains(test_gain, decisions):
    # No screening takes place here; these decisions were frozen before testing.
    return {'champion': 0., 'unfiltered': test_gain,
            'same_world': test_gain if decisions['same_world']['switch'] else 0.,
            'independent': test_gain if decisions['independent']['switch'] else 0.}


def prepare_branches(worlds, keys, scorer):
    """Reuse 138 transitions/replies; enforce semantic sets, not ranking order.

    A shared minimizing reply needs the same alternatives in every information
    history. Their presentation order does not affect resolution or seed slots.
    """
    branches, action_sets = [], {}
    for world_index, world in enumerate(worlds):
        for branch in base.prepare_branches((world,), keys, scorer):
            replies = frozenset(base.canonical_successor_action_key(branch['after'], r.action)
                                for r in branch['replies'])
            if action_sets.setdefault(branch['information_set'], replies) != replies:
                raise ValueError('139 unstable opponent information-set reply membership')
            branches.append({**branch, 'world': world_index})
    return branches


def evaluate_block(sample, condition, block, scorer):
    worlds = worlds_for(sample, condition, block)
    keys = base.root_keys(sample)
    branches = prepare_branches(worlds, keys, scorer)
    records, telemetry = [], Counter()
    for branch in branches:
        after, info = branch['after'], branch['information_set']
        if after.terminal:
            endpoint = base.terminal_record(after, sample['actor_seat'])
            margin = (endpoint['final_scores'][sample['actor_seat']]
                      - endpoint['final_scores'][1-sample['actor_seat']]) / 20.
            records.append({'world': branch['world'], 'root': branch['root'], 'info': info,
                            'reply': ('terminal',), 'margin': max(-1., min(1., margin)),
                            'wdl': endpoint['wdl'], 'returns': [endpoint], 'terminal_root': True})
            telemetry['terminal_roots'] += 1
            continue
        telemetry['opponent_scoring_calls'] += 1
        telemetry['expected_continuations'] += len(branch['replies']) * 4
        telemetry['fewer_than_four_reply_branches'] += int(len(branch['replies']) < 4)
        for reply in branch['replies']:
            returns = []
            for rep in range(4):
                seed = continuation_seed(sample['sample_index'], condition, block, branch['world'], rep)
                outcome = base.observe_rollout(after, reply.action, sample['actor_seat'], seed, scorer)
                returns.append(outcome)
                telemetry['continuations'] += 1
                telemetry['resolved_continuations'] += int(outcome['resolved'])
                telemetry['loop_adjudications'] += int(outcome['loop_adjudicated'])
                telemetry['fallback_triggers'] += outcome['fallback_triggers']
                telemetry['hard_caps'] += int(outcome['hard_cap_reached'])
                telemetry[outcome['termination_reason']] += 1
            records.append({'world': branch['world'], 'root': branch['root'], 'info': info,
                            'reply': base.canonical_successor_action_key(after, reply.action),
                            'margin': statistics.fmean(r['value'] for r in returns),
                            'wdl': statistics.fmean(r['wdl'] for r in returns), 'returns': returns})
    values, changes, choices = base.resolve_records(records, keys)
    gains = {m: values[sample['expanded_action']][m] - values[sample['champion_action']][m] for m in METRICS}
    leaves = [{**{k: v for k, v in r.items() if k not in ('root', 'reply', 'info')},
               'root': repr(r['root']), 'reply': repr(r['reply']),
               'information_set_sha256': digest(r['info'])} for r in records]
    return {'condition': condition, 'block': block, 'values': values, 'gains': gains,
            'world_hashes': [digest(base.serialize_state_snapshot(w)) for w in worlds],
            'telemetry': dict(telemetry), 'wdl_reply_changes': changes,
            'opponent_choices': choices, 'leaves': leaves}


def evaluate_stage(sample, stage):
    start = time.perf_counter()
    conditions = CHECKS if stage == 'checks' else ('test',)
    blocks = []
    elapsed = Counter()
    for block in range(BLOCKS):
        order = conditions if (sample['schedule_index']+block) % 2 == 0 else conditions[::-1]
        for condition in order:
            tick = time.perf_counter()
            blocks.append(evaluate_block(sample, condition, block, base._SCORER))
            elapsed[condition] += time.perf_counter()-tick
    row = {'sample_index': sample['sample_index'], 'schedule_index': sample['schedule_index'],
           'sample_sha256': digest(sample), 'stage': stage, 'blocks': blocks,
           'elapsed_by_condition': dict(elapsed), 'elapsed_seconds': time.perf_counter()-start}
    if stage == 'checks':
        row['decisions'] = decisions_from_checks(sample, blocks)
    return row


def verify_sample(sample, scorer):
    state = base.deserialize_state_snapshot(sample['state_snapshot'])
    # JSON persists numeric dictionary keys as strings; compare canonical JSON.
    if digest(base.serialize_state_snapshot(state)) != digest(sample['state_snapshot']) or state.current_player != sample['actor_seat']:
        raise ValueError('139 snapshot/seat mismatch')
    info = base.canonical_public_belief_key(state, state.current_player)
    def inventory(s):
        return sorted([c.card_id for t in (1, 2, 3) for c in s.hidden_tier_decks[t]+s.visible_tier_cards[t]]
                      + [c.card_id for p in s.players for c in p.purchased_cards+p.reserved_cards])
    altered = state.copy_shallow()
    for t in (1, 2, 3):
        altered.hidden_tier_decks[t] = list(reversed(altered.hidden_tier_decks[t]))
    altered_sample = {**sample, 'state_snapshot': base.serialize_state_snapshot(altered)}
    result = {'sample_index': sample['sample_index'], 'blocks': []}
    for condition in CONDITIONS:
        for block in range(BLOCKS):
            worlds = worlds_for(sample, condition, block)
            changed = worlds_for(altered_sample, condition, block)
            for w, x in zip(worlds, changed, strict=True):
                if base.canonical_public_belief_key(w, state.current_player) != info or inventory(w) != inventory(state):
                    raise ValueError('139 observer information or card inventory changed')
                if base.serialize_state_snapshot(w) != base.serialize_state_snapshot(x):
                    raise ValueError('139 hidden deck ordering affects sampling')
            branches = prepare_branches(worlds, base.root_keys(sample), scorer)
            if condition == CHECKS[0] and block == 0:
                for root in base.root_keys(sample):
                    legacy = base.audit085._prepare_single_root(worlds, root, base.ExpandedRootReplicationConfig(), scorer)
                    current = [b for b in branches if b['root'] == root]
                    for a, b in zip(legacy, current, strict=True):
                        if (base.state_signature(a['after_root']) != base.state_signature(b['after'])
                                or a['information_set'] != b['information_set']
                                or [r.action for r in a['responses']] != [r.action for r in b['replies']]):
                            raise ValueError('139 original 086 branch reproduction mismatch')
            result['blocks'].append({'condition': condition, 'block': block,
                'world_hashes': [digest(base.serialize_state_snapshot(w)) for w in worlds],
                'expected_continuations': sum(len(b['replies'])*4 for b in branches)})
    return result


def preflight(identity, manifest):
    path = OUTPUT / 'preflight.json'
    prepared_identity = {'inputs': identity, 'manifest_sha256': digest(manifest)}
    if path.exists():
        cached = read_json(path)
        if cached['identity'] != prepared_identity:
            raise ValueError('139 cached preflight changed')
        return cached
    base.initialize_worker()
    start, structures = time.perf_counter(), []
    for i, sample in enumerate(manifest['states']):
        structures.append(verify_sample(sample, base._SCORER))
        if (i+1) % 10 == 0:
            print(f'[139:preflight] structural_states={i+1}/60', flush=True)
    source = read_json(base.SELECTION)
    used = {s['source_seed'] for s in manifest['states']} | set(manifest['excluded_138_source_seeds'])
    probe = min((s for s in source['selected'] if s['source_seed'] not in used), key=lambda s: s['sample_index'])
    state = base.deserialize_state_snapshot(probe['state_snapshot'])
    action = base._SCORER.score_actions(state, base.SplendorEnv().legal_actions(state))[0].action
    tick = time.perf_counter()
    for i in range(8):
        base.observe_rollout(state, action, probe['actor_seat'], 1_399_000_000_000_000_000+i, base._SCORER)
    seconds = time.perf_counter()-tick
    projected = seconds/8*MAX_CONTINUATIONS*2
    expected = sum(b['expected_continuations'] for s in structures for b in s['blocks'])
    payload = {'identity': prepared_identity, 'status': 'prepared', 'structures': structures,
               'expected_primary_continuations': expected, 'protected_continuations': 0,
               'runtime_probe': {'sample_index': probe['sample_index'], 'continuations': 8,
                                 'seconds': seconds, 'conservative_projected_cpu_seconds': projected},
               'passed': projected <= 129_600 and expected <= MAX_CONTINUATIONS,
               'elapsed_seconds': time.perf_counter()-start,
               'precision_scenarios': [{'state_difference_sd': sd, 'normal_ci95_half_width': 1.96*sd/math.sqrt(60)}
                                       for sd in (.05, .1, .2)]}
    atomic_write(path, payload)
    return payload


def validate_saved(payload, sample, identity, stage, structure):
    if payload.get('identity') != identity or payload['row']['sample_sha256'] != digest(sample):
        raise ValueError('139 saved identity mismatch')
    row = payload['row']
    conditions = CHECKS if stage == 'checks' else ('test',)
    expected = {(c, b) for c in conditions for b in range(BLOCKS)}
    if (row['stage'] != stage or len(row['blocks']) != len(expected)
            or {(b['condition'], b['block']) for b in row['blocks']} != expected):
        raise ValueError('139 incomplete/duplicate stage blocks')
    reference = {(b['condition'], b['block']): b for b in structure['blocks']}
    for b in row['blocks']:
        ref = reference[b['condition'], b['block']]
        t = b['telemetry']
        if (b['world_hashes'] != ref['world_hashes']
                or t.get('continuations', 0) != ref['expected_continuations']
                or t.get('continuations', 0) != t.get('resolved_continuations', 0)):
            raise ValueError('139 prepared world/accounting mismatch')
        if set(b['values']) != {sample['champion_action'], sample['expanded_action']}:
            raise ValueError('139 original action keys changed')
        for m in METRICS:
            gain = b['values'][sample['expanded_action']][m]-b['values'][sample['champion_action']][m]
            if not math.isfinite(gain) or abs(gain-b['gains'][m]) > 1e-12:
                raise ValueError('139 block gain mismatch')
    if stage == 'checks':
        if row['decisions'] != decisions_from_checks(sample, row['blocks']):
            raise ValueError('139 saved screening decision mismatch')
    elif 'decisions' in row:
        raise ValueError('139 test worker received/produced decisions')
    return row


def frozen_choices(check_rows, samples, identity):
    if (len(check_rows) != 60 or len(samples) != 60
            or {r['sample_index'] for r in check_rows} != {s['sample_index'] for s in samples}):
        raise ValueError('139 cannot test before all 60 checks complete')
    by_id = {r['sample_index']: r for r in check_rows}
    return {'identity': identity, 'all_choices_frozen_before_test': True,
            'choices': [{'sample_index': s['sample_index'], 'sample_sha256': digest(s),
                         'check_row_sha256': digest(by_id[s['sample_index']]),
                         'decisions': by_id[s['sample_index']]['decisions']}
                        for s in samples]}


def make_rows(samples, checks, tests):
    check_by_id = {r['sample_index']: r for r in checks}
    test_by_id = {r['sample_index']: r for r in tests}
    rows = []
    for sample in samples:
        check, test = check_by_id[sample['sample_index']], test_by_id[sample['sample_index']]
        test_gaps = {m: statistics.fmean(b['gains'][m] for b in test['blocks']) for m in METRICS}
        gains = {m: policy_gains(test_gaps[m], check['decisions']) for m in METRICS}
        values = {m: {p: statistics.fmean(b['values'][sample['champion_action']][m] for b in test['blocks'])
                         + gains[m][p] for p in POLICIES} for m in METRICS}
        state = base.deserialize_state_snapshot(sample['state_snapshot'])
        private = sum(p.reserved_card_source(i) != base.RESERVE_SOURCE_VISIBLE
                      for p in state.players for i in range(len(p.reserved_cards)))
        rows.append({**{k: v for k, v in sample.items() if k != 'state_snapshot'},
                     'decisions': check['decisions'], 'policy_gains': gains,
                     'policy_values': values, 'test_expanded_minus_champion': test_gaps,
                     'existing_blind_reserves': private})
    return rows


def paired_summary(rows, samples=50_000):
    ordered = sorted(rows, key=lambda r: (r['selection_cell'], r['source_seed']))
    values = np.array([[r['policy_gains']['margin'][a]-r['policy_gains']['margin'][b]
                        for a, b in CONTRASTS] for r in ordered])
    rng, boot = np.random.default_rng(139300000), np.zeros((samples, len(CONTRASTS)))
    for cell in sorted({r['selection_cell'] for r in ordered}):
        ids = np.array([i for i, r in enumerate(ordered) if r['selection_cell'] == cell])
        draws = rng.choice(ids, size=(samples, len(ids)), replace=True)
        boot += values[draws].sum(axis=1)/len(ordered)
    ci = np.quantile(boot, [.025, .975], axis=0)
    return {f'{a}_minus_{b}': {'mean': float(values[:, i].mean()), 'ci95': ci[:, i].tolist()}
            for i, (a, b) in enumerate(CONTRASTS)}


def classify(summary, switches, valid=True):
    if not valid:
        return 'independent_confirmation_integrity_failed'
    if switches < 6:
        return 'insufficient_independent_switch_dose'
    default = summary['independent_minus_champion']
    default_pass = default['mean'] >= .01 and default['ci95'][0] > 0
    if default_pass and all(summary[f'independent_minus_{p}']['ci95'][0] > 0
                            for p in ('unfiltered', 'same_world')):
        return 'independent_confirmation_policy_improvement_supported'
    if default['ci95'][1] <= 0:
        return 'no_positive_default_improvement_supported'
    if default_pass:
        return 'default_improvement_supported_confirmation_benefit_unproven'
    return 'independent_confirmation_policy_inconclusive'


def summarize(rows, checks, tests):
    summary = paired_summary(rows)
    telemetry, timing = {}, {}
    for condition in CONDITIONS:
        counter = Counter()
        for r in checks+tests:
            for b in r['blocks']:
                if b['condition'] == condition:
                    counter.update(b['telemetry'])
        telemetry[condition] = dict(counter)
        timing[condition] = sum(r['elapsed_by_condition'].get(condition, 0.) for r in checks+tests)
    gates = {'sixty_states': len(rows) == 60, 'sixty_source_games': len({r['source_seed'] for r in rows}) == 60,
             'balanced_cells': Counter(r['selection_cell'] for r in rows) == Counter({c: 10 for c in CELLS}),
             'maximum_budget': sum(t.get('continuations', 0) for t in telemetry.values()) <= MAX_CONTINUATIONS}
    for c, t in telemetry.items():
        n = t.get('continuations', 0)
        gates[f'{c}_accounting'] = n == t.get('expected_continuations', 0) == t.get('resolved_continuations', 0)
        gates[f'{c}_no_hard_caps'] = t.get('hard_caps', 0) == 0
        gates[f'{c}_bounded_adjudication'] = t.get('loop_adjudications', 0)/max(1, n) <= .001
    switches = {p: sum(r['decisions'][p]['switch'] for r in rows) for p in ('same_world', 'independent')}
    secondary = {m: {f'{a}_minus_{b}': statistics.fmean(r['policy_gains'][m][a]-r['policy_gains'][m][b] for r in rows)
                     for a, b in CONTRASTS} for m in METRICS[1:]}
    breakdowns = {}
    for field in ('phase', 'actor_seat', 'existing_blind_reserves'):
        breakdowns[field] = {}
        for value in sorted({r[field] for r in rows}):
            group = [r for r in rows if r[field] == value]
            breakdowns[field][str(value)] = {'states': len(group),
                'independent_switches': sum(r['decisions']['independent']['switch'] for r in group),
                **{f'{a}_minus_{b}': statistics.fmean(r['policy_gains']['margin'][a]-r['policy_gains']['margin'][b] for r in group)
                   for a, b in CONTRASTS}}
    return {'summary': summary, 'switches': switches, 'valid': all(gates.values()),
            'decision': classify(summary, switches['independent'], all(gates.values())),
            'integrity_gates': gates, 'telemetry': telemetry, 'aggregate_condition_wall_seconds': timing,
            'policy_values': {m: {p: statistics.fmean(r['policy_values'][m][p] for r in rows) for p in POLICIES} for m in METRICS},
            'secondary_wdl_contrasts': secondary, 'breakdowns': breakdowns,
            'different_original_choices': sum(r['champion_action'] != r['expanded_action'] for r in rows),
            'screening_rule_disagreements': sum(r['decisions']['same_world']['switch'] != r['decisions']['independent']['switch'] for r in rows),
            'rows': rows, 'bootstrap': {'samples': 50_000, 'seed': 139300000, 'units': 'distinct_games_within_six_cells'}}


def run_stage(samples, identity, structures, stage):
    pending, rows = [], []
    for sample in samples:
        path = OUTPUT / stage / f"state_{sample['schedule_index']:02d}.json"
        if path.exists():
            rows.append(validate_saved(read_json(path), sample, identity, stage, structures[sample['sample_index']]))
        else:
            pending.append(sample)
    tick = time.perf_counter()
    print(f'[139:{stage}] completed={len(rows)}/60 pending={len(pending)} workers={WORKERS}', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=base.initialize_worker) as executor:
        futures = {executor.submit(evaluate_stage, s, stage): s for s in pending}
        for future in as_completed(futures):
            sample = futures[future]
            payload = {'identity': identity, 'row': future.result()}
            validate_saved(payload, sample, identity, stage, structures[sample['sample_index']])
            atomic_write(OUTPUT / stage / f"state_{sample['schedule_index']:02d}.json", payload)
            rows.append(payload['row'])
            print(f'[139:{stage}] completed={len(rows)}/60 elapsed={time.perf_counter()-tick:.1f}s', flush=True)
    return sorted(rows, key=lambda r: r['schedule_index'])


def run(launch=False):
    with base.process_lock(OUTPUT / 'process.lock'):
        identity = input_identity()
        manifest = build_manifest(read_json(base.SELECTION), read_json(base.RESULT086), identity)
        path = OUTPUT / 'state_manifest.json'
        if path.exists():
            if read_json(path) != manifest:
                raise ValueError('139 existing manifest changed')
        else:
            atomic_write(path, manifest)
        if not launch:
            result = preflight(identity, manifest)
            print({k: result[k] for k in ('status', 'passed', 'expected_primary_continuations', 'runtime_probe')}, flush=True)
            return result
        prepared = read_json(OUTPUT / 'preflight.json')
        prepared_identity = {'inputs': identity, 'manifest_sha256': digest(manifest)}
        if prepared['identity'] != prepared_identity or not prepared['passed']:
            raise ValueError('139 preparation failed or changed')
        launch_identity = {**prepared_identity, 'preflight_sha256': file_hash(OUTPUT / 'preflight.json')}
        final_path = OUTPUT / 'result.json'
        if final_path.exists():
            final = read_json(final_path)
            if final['identity'] != launch_identity:
                raise ValueError('139 completed identity changed')
            return final
        structures = {r['sample_index']: r for r in prepared['structures']}
        started = time.perf_counter()
        atomic_write(OUTPUT / 'run_identity.json', {'identity': launch_identity, 'pid': os.getpid(), 'started_unix': time.time()})
        checks = run_stage(manifest['states'], launch_identity, structures, 'checks')
        choices = frozen_choices(checks, manifest['states'], launch_identity)
        choice_path = OUTPUT / 'frozen_choices.json'
        if choice_path.exists():
            if read_json(choice_path) != choices:
                raise ValueError('139 all-state choices changed')
        else:
            atomic_write(choice_path, choices)
        test_identity = {**launch_identity, 'frozen_choices_sha256': file_hash(choice_path)}
        print('[139:seal] all 60 choices frozen; opening independent test stage', flush=True)
        tests = run_stage(manifest['states'], test_identity, structures, 'test')
        if input_identity() != identity or file_hash(choice_path) != test_identity['frozen_choices_sha256']:
            raise ValueError('139 protected inputs/choices changed during evaluation')
        rows = make_rows(manifest['states'], checks, tests)
        final = {'identity': launch_identity, 'frozen_choices_sha256': test_identity['frozen_choices_sha256'],
                 'status': 'complete', **summarize(rows, checks, tests),
                 'elapsed_this_session_seconds': time.perf_counter()-started,
                 'training': False, 'promotion': False, 'full_game_arena': False,
                 'automatic_continuation': False, 'research_champion': '084', 'fast_champion': '035'}
        atomic_write(final_path, final)
        print(f"[139:complete] decision={final['decision']}", flush=True)
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    run(parser.parse_args().launch)


if __name__ == '__main__':
    main()
