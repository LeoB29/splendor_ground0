"""Independently replay 140 traces and reconstruct estimates; no policy calls."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

from splendor_ai.cfr.public_belief_resolving import deterministic_belief_range
from splendor_ai.diagnostics import state_signature, is_progress_transition
from splendor_ai.encoding import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.termination import is_forced_pass_turn
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot

ROOT = Path('outputs/continuation_policy_crossover_140')
CONDITIONS = ('stochastic_035', 'greedy_035', 'resolver_084')
NAMES = ('stochastic_gain', 'strong_gain', 'strong_minus_stochastic',
         'greedy_gain', 'strong_minus_greedy', 'greedy_minus_stochastic')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def near(x, y):
    assert math.isfinite(x) and math.isfinite(y) and abs(x-y) < 1e-12, (x, y)


def replay(row, actor):
    env = SplendorEnv()
    state = deserialize_state_snapshot(row['world_snapshot'])
    assert sha(serialize_state_snapshot(state)) == row['world_sha256']
    visits, passes, idle = Counter(), 0, 0
    trace = row['trace']
    assert trace and len(trace) == row['outer_turns']
    for index, step in enumerate(trace):
        assert not state.terminal and state.turn_index < 400 and passes < 2
        visits[state_signature(state)] += 1
        assert visits[state_signature(state)] < 4
        assert step['before_sha256'] == sha(serialize_state_snapshot(state))
        assert step['turn_index'] == state.turn_index and step['actor_seat'] == state.current_player
        assert step['forced'] == (index == 0)
        if index == 0:
            assert step['action'] == row['root_action']
        legal = env.legal_actions(state)
        key = ast.literal_eval(step['action'])
        mapping = actions_by_canonical_successor_key(state, legal)
        assert key in mapping
        action = mapping[key]
        after = env.step(state, action)
        assert sha(serialize_state_snapshot(after)) == step['after_sha256']
        passes = passes+1 if is_forced_pass_turn(legal) else 0
        if passes < 2:
            idle = 0 if is_progress_transition(state, after, action) else idle+1
            assert idle < 60
        state = after
    assert sha(serialize_state_snapshot(state)) == sha(row['final_state'])
    reason = 'forced_pass_deadlock' if passes >= 2 else 'completed'
    assert reason == row['termination_reason'] and (passes >= 2 or state.terminal)
    scores, cards = [p.score for p in state.players], [len(p.purchased_cards) for p in state.players]
    winner = (int(scores[1] > scores[0]) if scores[0] != scores[1]
              else int(cards[1] < cards[0]) if cards[0] != cards[1] else None)
    assert not state.terminal or state.winner == winner
    assert row['winner'] == winner and row['final_scores'] == scores and row['final_card_counts'] == cards
    values = {'raw_score_margin': scores[actor]-scores[1-actor],
              'wdl': 0. if winner is None else 1. if winner == actor else -1.}
    values['margin'] = max(-1., min(1., values['raw_score_margin']/20.))
    for name, value in values.items():
        near(row[name], value)
    if row['condition'] == 'resolver_084':
        assert len(row['resolver_telemetry']) == 2
        for seat, t in enumerate(row['resolver_telemetry']):
            assert t['actor_seat'] == seat and t['engine_seed'] == 1_400_000+row['policy_slot']
            assert t['decisions'] == sum(step['actor_seat'] == seat for step in trace[1:])
            assert t['decisions'] == t['completed_searches']+t['fallbacks']
            assert t['fallbacks'] == sum(t['fallback_reasons'].values())
            assert t['continuations'] == t['resolved_continuations'] == t['completed_searches']*192
            assert t['hard_caps'] == 0 and 0 <= t['switches'] <= t['completed_searches']
            for name in ('decisions', 'completed_searches', 'switches', 'fallbacks'):
                assert t[name] == sum(p[name] for p in t['by_phase'].values())
    else:
        assert row['resolver_telemetry'] == []
    return values


def reconstruct_summary(rows, metric, draws=50_000):
    ordered = sorted(rows, key=lambda r: (r['selection_cell'], r['source_seed']))
    vectors = []
    for row in ordered:
        a, b, c = [row['gains'][policy][metric] for policy in CONDITIONS]
        vectors.append([a, c, c-a, b, c-b, b-a])
    values = np.asarray(vectors)
    boot = np.zeros((draws, 6))
    generator = np.random.default_rng(140300000)
    for cell in sorted({r['selection_cell'] for r in ordered}):
        indices = [i for i, r in enumerate(ordered) if r['selection_cell'] == cell]
        selections = generator.choice(indices, (draws, len(indices)), replace=True)
        boot += np.sum(values[selections], axis=1)/len(ordered)
    bounds = np.quantile(boot, [.025, .975], axis=0)
    return {name: {'mean': float(np.mean(values[:, i])), 'ci95': bounds[:, i].tolist()}
            for i, name in enumerate(NAMES)}


def main():
    final = read(ROOT/'result.json')
    assert final['status'] == 'complete'
    identity = final['identity']
    manifest, preflight = read(ROOT/'state_manifest.json'), read(ROOT/'preflight.json')
    preparation = read(ROOT/'preparation_manifest.json')
    assert sha(manifest) == identity['state_manifest_sha256'] == preflight['state_manifest_sha256']
    assert file_sha(ROOT/'preflight.json') == identity['preflight_sha256'] and preflight['passed']
    assert preflight['identity'] == {'inputs': identity['inputs'], 'preparation_manifest_sha256': sha(preparation)}
    assert manifest['identity'] == preparation['identity'] == identity['inputs']
    for group in ('sources', 'dependencies'):
        for filename, expected in identity['inputs'][group].items():
            assert file_sha(Path(filename)) == expected
    assert file_sha(Path('docs/engineering/continuation_policy_crossover_140.md')) == identity['inputs']['protocol_sha256']
    n = manifest['state_count']
    assert n == final['state_count'] == preflight['selected_state_count']
    phase_times = {p: max(sum(r['timings_and_integrity'][c]['elapsed_seconds'] for c in CONDITIONS)
                         for r in preflight['timing_probes'] if r['phase'] == p) for p in ('early', 'mid', 'late')}
    projections = {size: 1.5*(size/3)*6*sum(phase_times.values()) for size in (60, 48, 36, 24)}
    assert n == next(size for size, seconds in projections.items() if seconds <= 129600)
    states = manifest['states']
    assert len(states) == len({s['source_seed'] for s in states}) == n
    assert set(s['source_seed'] for s in states).isdisjoint(manifest['excluded_source_seeds']+manifest['probe_source_seeds'])
    assert set(Counter(s['selection_cell'] for s in states).values()) == {n//6}
    final_rows = {r['sample_index']: r for r in final['rows']}
    rows, counts, all_paths = [], defaultdict(Counter), []
    phase_searches = Counter()
    for sample in states:
        source = deserialize_state_snapshot(sample['state_snapshot'])
        worlds = deterministic_belief_range(source, sample['actor_seat'],
                                           root_index=1_400_000_000+sample['sample_index']*10, count=3)
        values = {}
        for w, world in enumerate(worlds):
            for condition in CONDITIONS:
                for action in dict.fromkeys((sample['champion_action'], sample['expanded_action'])):
                    path = ROOT/'trajectories'/f"s{sample['sample_index']:03d}_w{w}_{condition}_{sha(action)[:16]}.json"
                    all_paths.append(path)
                    payload = read(path)
                    assert payload['identity'] == identity
                    record = payload['row']
                    assert record['sample_index'] == sample['sample_index'] and record['sample_sha256'] == sha(sample)
                    assert record['condition'] == condition and record['world_index'] == w
                    assert record['root_action'] == action and not record['probe']
                    assert record['policy_slot'] == sample['sample_index']*3+w
                    assert record['world_sha256'] == sha(serialize_state_snapshot(world))
                    values[w, condition, action] = replay(record, sample['actor_seat'])
                    count = counts[condition]
                    count['outer_continuations'] += 1
                    count['outer_turns'] += record['outer_turns']
                    count[record['termination_reason']] += 1
                    count['aggregate_cpu_seconds'] += record['cpu_seconds']
                    count['aggregate_wall_seconds'] += record['elapsed_seconds']
                    for t in record['resolver_telemetry']:
                        for name in ('decisions', 'completed_searches', 'switches', 'fallbacks', 'continuations',
                                     'resolved_continuations', 'hard_caps', 'loop_adjudications', 'rollout_fallback_triggers'):
                            count[name] += t[name]
                        phase_searches[sample['phase']] += t['completed_searches']
        gains = {c: {m: statistics.fmean(values[w, c, sample['expanded_action']][m]
                                         - values[w, c, sample['champion_action']][m] for w in range(3))
                     for m in ('margin', 'wdl', 'raw_score_margin')} for c in CONDITIONS}
        for c in CONDITIONS:
            for m in gains[c]:
                near(gains[c][m], final_rows[sample['sample_index']]['gains'][c][m])
        rows.append({**sample, 'gains': gains})
    assert len(all_paths) == len(set(all_paths)) == manifest['expected_outer_continuations']
    assert set(all_paths) == set((ROOT/'trajectories').glob('*.json'))
    for c, count in counts.items():
        assert set(count) == set(final['telemetry'][c])
        for k, value in count.items():
            # Different summation order can introduce rounding only in timings.
            if k.startswith('aggregate_'):
                assert math.isclose(value, final['telemetry'][c][k], rel_tol=1e-12)
            else:
                near(value, final['telemetry'][c][k])
    assert dict(phase_searches) == final['source_phase_searches']
    comparisons = 0
    for metric, target in (('margin', 'summary'), ('wdl', 'secondary_wdl'), ('raw_score_margin', 'secondary_raw_margin')):
        estimates = reconstruct_summary(rows, metric)
        for name in NAMES:
            near(estimates[name]['mean'], final[target][name]['mean'])
            for x, y in zip(estimates[name]['ci95'], final[target][name]['ci95'], strict=True):
                near(x, y)
            comparisons += 3
    report = {'passed': True, 'result_sha256': file_sha(ROOT/'result.json'),
              'verifier_sha256': file_sha(Path(__file__)), 'states': n, 'replayed_trajectories': len(all_paths),
              'replayed_turns': sum(c['outer_turns'] for c in counts.values()),
              'mean_and_ci_values_verified': comparisons, 'tolerance': 1e-12,
              'policy_calls': 0, 'training': False}
    path = ROOT/'independent_verification.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
