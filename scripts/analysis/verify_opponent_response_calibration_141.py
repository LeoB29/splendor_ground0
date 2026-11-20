"""Independent endpoint/statistical reconstruction of 141, without policy calls."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path('outputs/opponent_response_calibration_141')
ARMS = ('sampled_min', 'policy_reply')
NAMES = ('sampled_min_gain', 'policy_reply_gain', 'policy_minus_min')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def near(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-12, (a, b)


def reconstruct_block(block, sample, structure):
    assert block['block'] == structure['block'] and block['world_hashes'] == structure['world_hashes']
    roots = {sample['champion_action'], sample['expanded_action']}
    refs = {(r['world'], r['root']): r for r in structure['branches']}
    assert set(refs) == {(w, r) for w in range(3) for r in roots}
    values, counters = {}, {}
    for arm, field in zip(ARMS, ('leaves', 'policy_returns')):
        seen, groups, counts = set(), defaultdict(list), Counter()
        for leaf in block[field]:
            w, root, reply = leaf['world'], leaf['root'], leaf['reply']
            ref = refs[w, root]
            assert leaf['information_set_sha256'] == ref['info']
            terminal = ref['terminal_root']
            assert leaf.get('terminal_root', False) == terminal
            expected_reply = "('terminal',)" if terminal else reply
            if arm == 'sampled_min':
                assert reply == "('terminal',)" if terminal else reply in ref['replies']
            elif not terminal:
                assert reply is None
            key = (w, root, expected_reply)
            assert key not in seen
            seen.add(key)
            outcomes = leaf['returns']
            assert len(outcomes) == (1 if terminal else 4)
            ms, ws = [], []
            for rep, outcome in enumerate(outcomes):
                scores, cards = outcome['final_scores'], outcome['final_card_counts']
                assert len(scores) == len(cards) == 2
                if scores[0] != scores[1]:
                    winner = int(scores[1] > scores[0])
                elif cards[0] != cards[1]:
                    winner = int(cards[1] < cards[0])
                else:
                    winner = None
                wdl = 0. if winner is None else 1. if winner == sample['actor_seat'] else -1.
                margin = max(-1., min(1., (scores[sample['actor_seat']]-scores[1-sample['actor_seat']])/20.))
                assert outcome['winner'] == winner
                near(outcome['wdl'], wdl)
                if not terminal:
                    near(outcome['value'], margin)
                    expected_seed = 1_430_000_000_000_000_000+sample['sample_index']*1_000_000+block['block']*1000+w*10+rep
                    assert outcome['seed'] == expected_seed
                    assert outcome['terminal'] and outcome['resolved']
                    assert not outcome['hard_cap_reached'] and not outcome['loop_adjudicated'] and outcome['fallback_triggers'] == 0
                    assert outcome['final_turn_index'] < 400 and outcome['max_repetition_visits'] < 4 and outcome['max_no_progress'] < 60
                    assert len(outcome['action_sequence_sha256']) == 64
                    expected_turns = outcome['final_turn_index']-sample['turn_index']-(2 if arm == 'sampled_min' else 1)
                    assert outcome['turns'] == expected_turns
                    counts['continuations'] += 1
                    counts['resolved_continuations'] += 1
                    counts['natural_continuations'] += 1
                    if arm == 'policy_reply':
                        assert outcome['immediate_reply'] is not None
                        counts['immediate_reply_outside_min_set'] += int(outcome['immediate_reply'] not in ref['replies'])
                ms.append(margin)
                ws.append(wdl)
            near(leaf['margin'], statistics.fmean(ms))
            near(leaf['wdl'], statistics.fmean(ws))
            groups[root, ref['info']].append((w, reply, statistics.fmean(ms), statistics.fmean(ws)))
        expected_seen = set()
        for (w, root), ref in refs.items():
            if ref['terminal_root']:
                expected_seen.add((w, root, "('terminal',)"))
                counts['terminal_roots'] += 1
            else:
                counts['fewer_than_four_reply_branches'] += int(len(ref['replies']) < 4)
                expected_seen.update((w, root, r) for r in (ref['replies'] if arm == 'sampled_min' else (None,)))
        assert seen == expected_seen
        assert counts['continuations'] == structure['expected_continuations'][arm]
        reported_counts = block['telemetry'][arm]
        for k in counts.keys() | reported_counts.keys():
            assert counts[k] == reported_counts.get(k, 0), (arm, k)
        counters[arm] = counts
        vals = {r: {'margin': 0., 'wdl': 0.} for r in roots}
        for (root, info), entries in sorted(groups.items()):
            if arm == 'sampled_min':
                replies = sorted({x[1] for x in entries})
                worlds = {x[0] for x in entries}
                candidates = {reply: sorted([x for x in entries if x[1] == reply]) for reply in replies}
                assert all(len(xs) == len(worlds) and {x[0] for x in xs} == worlds for xs in candidates.values())
                selected = min(replies, key=lambda reply: (sum(x[2] for x in candidates[reply])/len(worlds), reply))
                chosen = candidates[selected]
            else:
                chosen = entries
            vals[root]['margin'] += sum(x[2] for x in chosen)/3
            vals[root]['wdl'] += sum(x[3] for x in chosen)/3
        for root in roots:
            for metric in ('margin', 'wdl'):
                near(vals[root][metric], block['values'][arm][root][metric])
        values[arm] = vals
    gains = {arm: {m: values[arm][sample['expanded_action']][m]-values[arm][sample['champion_action']][m]
                   for m in ('margin', 'wdl')} for arm in ARMS}
    for arm in ARMS:
        for metric in ('margin', 'wdl'):
            near(gains[arm][metric], block['gains'][arm][metric])
    return gains, counters


def main():
    final = read(ROOT/'result.json')
    assert final['status'] == 'complete'
    original_sha, identity = sha(ROOT/'result.json'), final['identity']
    manifest, prepared = read(ROOT/'state_manifest.json'), read(ROOT/'preflight.json')
    assert canonical(manifest) == identity['manifest_sha256']
    assert sha(ROOT/'preflight.json') == identity['preflight_sha256']
    assert prepared['identity'] == {'inputs': identity['inputs'], 'manifest_sha256': identity['manifest_sha256']}
    assert manifest['identity'] == identity['inputs'] and prepared['passed']
    for path, expected in {**identity['inputs']['sources'], **identity['inputs']['dependencies']}.items():
        assert sha(Path(path)) == expected, path
    assert sha(Path('docs/engineering/opponent_response_calibration_141.md')) == identity['inputs']['protocol_sha256']
    source = read(Path('outputs/expanded_root_regret_stability_replication_086/selection.json'))
    old = {s['sample_index']: s for s in read(Path('outputs/expanded_root_regret_stability_replication_086/result.json'))['rows']}
    by_id = {s['sample_index']: s for s in source['selected']}
    excluded = set()
    for path in identity['inputs']['sources']:
        if 'state_manifest.json' in path or 'preparation_manifest.json' in path or 'preflight.json' in path:
            data = read(Path(path))
            excluded.update(s['source_seed'] for s in data.get('states', []))
            excluded.update(s['source_seed'] for s in data.get('probes', []))
            if 'runtime_probe' in data:
                p = data['runtime_probe']
                excluded.add(by_id[p.get('source_sample_index', p.get('sample_index'))]['source_seed'])
    assert sorted(excluded) == manifest['excluded_source_seeds']
    def rank(s, prefix):
        text = f"{prefix}|{s['source_seed']}|{s['turn_index']}|{s['actor_seat']}|{s['phase']}"
        return hashlib.sha256(text.encode()).hexdigest(), s['source_seed'], s['turn_index']
    expected_ids, expected_probes = [], []
    for cell in [f'seat{s}|{p}' for s in (0, 1) for p in ('early', 'mid', 'late')]:
        available = [r for r in source['selected'] if r['selection_cell'] == cell and r['source_seed'] not in excluded]
        selected = sorted(available, key=lambda s: rank(s, '141-selection-v1'))[:20]
        expected_ids.extend(r['sample_index'] for r in selected)
        remaining = [r for r in available if r['sample_index'] not in expected_ids]
        expected_probes.append(min(remaining, key=lambda s: rank(s, '141-probe-v1'))['sample_index'])
    assert expected_ids == [r['sample_index'] for r in manifest['states']]
    assert expected_probes == [r['sample_index'] for r in manifest['probes']]
    references = {s['sample_index']: s for s in prepared['structures']}
    reported_rows = {r['sample_index']: r for r in final['rows']}
    rows, counters, wall, cpu = [], {c: Counter() for c in ARMS}, Counter(), Counter()
    for sample in manifest['states']:
        idx = sample['sample_index']
        original, decision = by_id[idx], old[idx]
        assert sample['champion_action'] == original['champion_root_keys'][decision['champion_selected_index']]
        assert sample['expanded_action'] == original['expanded_root_keys'][decision['expanded_selected_index']]
        for key in ('source_seed', 'turn_index', 'actor_seat', 'phase', 'selection_cell', 'state_snapshot'):
            assert canonical(sample[key]) == canonical(original[key])
        payload = read(ROOT/'states'/f"state_{sample['schedule_index']:03d}.json")
        assert payload['identity'] == identity
        row, ref = payload['row'], references[idx]
        assert row['sample_sha256'] == ref['sample_sha256'] == canonical(sample)
        assert len(row['blocks']) == 16 and {b['block'] for b in row['blocks']} == set(range(16))
        frozen_blocks = {b['block']: b for b in ref['blocks']}
        gaps = []
        for b in sorted(row['blocks'], key=lambda b: b['block']):
            gains, counts = reconstruct_block(b, sample, frozen_blocks[b['block']])
            gaps.append(gains)
            for c in ARMS:
                counters[c].update(counts[c])
                wall[c] += b['elapsed_by_condition'][c]
                cpu[c] += b['cpu_by_condition'][c]
        means = {c: {m: statistics.fmean(g[c][m] for g in gaps) for m in ('margin', 'wdl')} for c in ARMS}
        for c in ARMS:
            for m in ('margin', 'wdl'):
                near(means[c][m], row['gains'][c][m])
                near(means[c][m], reported_rows[idx]['gains'][c][m])
        rows.append({'source_seed': sample['source_seed'], 'cell': sample['selection_cell'], 'gains': means, 'blocks': gaps})
    assert len(rows) == len({r['source_seed'] for r in rows}) == 120
    rows.sort(key=lambda r: (r['cell'], r['source_seed']))
    reconstructed = {}
    for metric, key in (('margin', 'summary'), ('wdl', 'secondary_wdl')):
        matrix = np.array([[r['gains'][ARMS[0]][metric], r['gains'][ARMS[1]][metric],
                            r['gains'][ARMS[1]][metric]-r['gains'][ARMS[0]][metric]] for r in rows])
        rng, boot = np.random.default_rng(141300000), np.zeros((50_000, 3))
        for start in range(0, 120, 20):
            assert len({r['cell'] for r in rows[start:start+20]}) == 1
            ids = rng.integers(0, 20, size=(50_000, 20))+start
            boot += matrix[ids].sum(axis=1)/120
        intervals = np.percentile(boot, [2.5, 97.5], axis=0)
        for i, name in enumerate(NAMES):
            near(final[key][name]['mean'], matrix[:, i].mean())
            for j in range(2):
                near(final[key][name]['ci95'][j], intervals[j, i])
        reconstructed[key] = intervals
        if metric == 'margin':
            block_matrix = np.array([[[g[ARMS[0]][metric], g[ARMS[1]][metric], g[ARMS[1]][metric]-g[ARMS[0]][metric]] for g in r['blocks']] for r in rows])
            within = block_matrix.var(axis=1, ddof=1).mean(axis=0)
            between = np.array([matrix[i:i+20].var(axis=0, ddof=1) for i in range(0, 120, 20)]).mean(axis=0)
            for i, name in enumerate(NAMES):
                v = final['variance_for_future_design'][name]
                near(v['within_block_variance'], within[i])
                near(v['observed_state_variance'], between[i])
                near(v['between_state_moment_estimate'], between[i]-within[i]/16)
    for c in ARMS:
        for key in counters[c].keys() | final['telemetry'][c].keys():
            assert counters[c][key] == final['telemetry'][c].get(key, 0)
        assert counters[c]['continuations'] == prepared['expected_continuations'][c]
        near(final['aggregate_wall_by_condition'][c], wall[c])
        near(final['aggregate_cpu_by_condition'][c], cpu[c])
    lo, hi = reconstructed['summary'][:, 2]
    expected_decision = ('material_response_rule_effect_supported' if lo > .025 or hi < -.025 else
                         'response_rule_difference_detected_materiality_uncertain' if lo > 0 or hi < 0 else
                         'response_rule_difference_bounded_below_material_size' if lo >= -.025 and hi <= .025 else
                         'response_rule_contrast_inconclusive')
    assert final['valid'] and all(final['integrity_gates'].values()) and final['decision'] == expected_decision
    assert sum(s['champion_action'] != s['expanded_action'] for s in manifest['states']) >= 12
    assert sum(counters[c]['continuations'] for c in ARMS) <= 230_400
    near(final['primary_ci_halfwidth'], (hi-lo)/2)
    assert final['precision_target_met'] == ((hi-lo)/2 <= .025)
    for i, name in enumerate(NAMES[:2]):
        low, high = reconstructed['summary'][:, i]
        status = 'positive' if low > 0 else 'negative' if high < 0 else 'uncertain'
        assert final['absolute_gain_status_secondary'][name] == status
    assert sha(ROOT/'result.json') == original_sha
    record = {'passed': True, 'result_sha256': original_sha, 'states': 120, 'blocks': 1920,
              'continuations': {c: counters[c]['continuations'] for c in ARMS},
              'all_18_mean_and_ci_numbers_reconstructed': True, 'selection_bindings_and_structures_verified': True,
              'variance_and_classification_reconstructed': True, 'tolerance': 1e-12,
              'policy_calls_or_training': False, 'full_trajectory_replay': False}
    path = ROOT/'independent_verification.json'
    if path.exists():
        assert read(path) == record
    else:
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
