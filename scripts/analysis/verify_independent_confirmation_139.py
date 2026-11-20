"""Read-only independent reconstruction of completed experiment 139 evidence.

Uses no experiment implementation, resolver, screening or summary helpers.
Writes only a verification record after every check passes; never simulates play.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path('outputs/independent_confirmation_root_policy_139')
METRICS = ('margin', 'wdl_under_margin', 'wdl_min')
PAIRS = (('independent', 'champion'), ('independent', 'unfiltered'),
         ('independent', 'same_world'), ('same_world', 'champion'), ('unfiltered', 'champion'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def near(x, y):
    assert math.isfinite(x) and math.isfinite(y) and abs(x-y) < 1e-12, (x, y)


def reconstruct_block(block, sample, counts):
    groups, seen = defaultdict(list), set()
    for leaf in block['leaves']:
        key = (leaf['root'], leaf['information_set_sha256'], leaf['reply'], leaf['world'])
        assert key not in seen
        seen.add(key)
        margins, wdls = [], []
        for outcome in leaf['returns']:
            scores, cards = outcome['final_scores'], outcome['final_card_counts']
            if scores[0] != scores[1]:
                winner = int(scores[1] > scores[0])
            elif cards[0] != cards[1]:
                winner = int(cards[1] < cards[0])
            else:
                winner = None
            wdl = 0. if winner is None else (1. if winner == sample['actor_seat'] else -1.)
            margin = float(np.clip((scores[sample['actor_seat']]-scores[1-sample['actor_seat']])/20., -1., 1.))
            assert outcome['winner'] == winner
            near(outcome['wdl'], wdl)
            if not leaf.get('terminal_root', False):
                near(outcome['value'], margin)
                assert outcome['resolved'] and not outcome['hard_cap_reached']
                assert outcome['terminal'] or outcome['loop_adjudicated']
                counts['continuations'] += 1
                counts['adjudications'] += int(outcome['loop_adjudicated'])
            else:
                counts['terminal_root_returns'] += 1
            margins.append(margin)
            wdls.append(wdl)
        near(leaf['margin'], statistics.fmean(margins))
        near(leaf['wdl'], statistics.fmean(wdls))
        groups[leaf['root'], leaf['information_set_sha256']].append(
            (leaf['reply'], leaf['world'], statistics.fmean(margins), statistics.fmean(wdls)))
    values = {key: {m: 0. for m in METRICS} for key in {sample['champion_action'], sample['expanded_action']}}
    for (root, info), leaves in groups.items():
        replies = sorted({x[0] for x in leaves})
        worlds = {x[1] for x in leaves}
        reply_rows = {r: [x for x in leaves if x[0] == r] for r in replies}
        assert all(len(xs) == len(worlds) and {x[1] for x in xs} == worlds for xs in reply_rows.values())
        # Arithmetic and lexical tie-breaking match the frozen mathematical rule.
        selected_margin = min(replies, key=lambda r: (sum(x[2] for x in reply_rows[r])/len(worlds), r))
        selected_wdl = min(replies, key=lambda r: (sum(x[3] for x in reply_rows[r])/len(worlds), r))
        values[root]['margin'] += sum(x[2] for x in reply_rows[selected_margin])/3
        values[root]['wdl_under_margin'] += sum(x[3] for x in reply_rows[selected_margin])/3
        values[root]['wdl_min'] += sum(x[3] for x in reply_rows[selected_wdl])/3
    for root in values:
        assert {w for r, i, a, w in seen if r == root} == {0, 1, 2}
        for metric in METRICS:
            near(values[root][metric], block['values'][root][metric])
    gains = {m: values[sample['expanded_action']][m]-values[sample['champion_action']][m] for m in METRICS}
    for m in METRICS:
        near(gains[m], block['gains'][m])
    return gains, values


def main():
    final = read(ROOT/'result.json')
    assert final['status'] == 'complete'
    final_sha = file_sha(ROOT/'result.json')
    manifest, choices = read(ROOT/'state_manifest.json'), read(ROOT/'frozen_choices.json')
    identity = final['identity']
    assert canonical_sha(manifest) == identity['manifest_sha256']
    assert file_sha(ROOT/'preflight.json') == identity['preflight_sha256']
    assert file_sha(ROOT/'frozen_choices.json') == final['frozen_choices_sha256']
    assert choices['identity'] == identity and choices['all_choices_frozen_before_test']
    for name, expected in identity['inputs']['sources'].items():
        assert file_sha(Path(name)) == expected
    for name, expected in identity['inputs']['dependencies'].items():
        assert file_sha(Path(name)) == expected
    assert file_sha(Path('docs/engineering/independent_confirmation_root_policy_139.md')) == identity['inputs']['protocol_sha256']
    sealed = {r['sample_index']: r for r in choices['choices']}
    expected_rows = {r['sample_index']: r for r in final['rows']}
    rows, counts = [], defaultdict(int)
    for sample in manifest['states']:
        filename = f"state_{sample['schedule_index']:02d}.json"
        check, test = read(ROOT/'checks'/filename), read(ROOT/'test'/filename)
        assert check['identity'] == identity
        assert test['identity'] == {**identity, 'frozen_choices_sha256': final['frozen_choices_sha256']}
        assert check['row']['sample_sha256'] == test['row']['sample_sha256'] == canonical_sha(sample)
        assert sealed[sample['sample_index']]['check_row_sha256'] == canonical_sha(check['row'])
        assert sealed[sample['sample_index']]['decisions'] == check['row']['decisions']
        assert len(check['row']['blocks']) == 16 and len(test['row']['blocks']) == 8
        assert {(b['condition'], b['block']) for b in check['row']['blocks']} == {
            (c, i) for c in ('same_world_check', 'fresh_world_check') for i in range(8)}
        assert {(b['condition'], b['block']) for b in test['row']['blocks']} == {('test', i) for i in range(8)}
        check_gains = defaultdict(list)
        for block in check['row']['blocks']:
            gaps, _ = reconstruct_block(block, sample, counts)
            check_gains[block['condition']].append((block['block'], gaps['margin']))
        switches = {}
        for policy, condition in (('same_world', 'same_world_check'), ('independent', 'fresh_world_check')):
            gains = [g for _, g in sorted(check_gains[condition])]
            mean = math.fsum(gains)/8
            se = math.sqrt(sum((g-mean)**2 for g in gains)/7/8)
            lower = mean-1.895*se
            switches[policy] = sample['champion_action'] != sample['expanded_action'] and mean >= .025 and lower > 0
            decision = check['row']['decisions'][policy]
            assert decision['switch'] == switches[policy]
            near(decision['mean_gain'], mean)
            near(decision['standard_error'], se)
            near(decision['screening_lower_score'], lower)
        test_gaps, test_values = [], []
        for block in test['row']['blocks']:
            gap, value = reconstruct_block(block, sample, counts)
            test_gaps.append(gap)
            test_values.append(value)
        policy_gains = {}
        expected = expected_rows[sample['sample_index']]
        assert expected['decisions'] == check['row']['decisions']
        for m in METRICS:
            gap = sum(g[m] for g in test_gaps)/8
            near(expected['test_expanded_minus_champion'][m], gap)
            policy_gains[m] = {'champion': 0., 'unfiltered': gap,
                              **{p: gap if switch else 0. for p, switch in switches.items()}}
            default = sum(v[sample['champion_action']][m] for v in test_values)/8
            for p, gain in policy_gains[m].items():
                near(expected['policy_gains'][m][p], gain)
                near(expected['policy_values'][m][p], default+gain)
        rows.append({'source_seed': sample['source_seed'], 'selection_cell': sample['selection_cell'],
                     'gains': policy_gains, 'switches': switches})
    assert len(rows) == len({r['source_seed'] for r in rows}) == 60
    assert not {r['source_seed'] for r in rows} & set(manifest['excluded_138_source_seeds'])
    rows.sort(key=lambda r: (r['selection_cell'], r['source_seed']))
    matrix = np.array([[r['gains']['margin'][a]-r['gains']['margin'][b] for a, b in PAIRS] for r in rows])
    rng, boot = np.random.default_rng(139300000), np.zeros((50_000, len(PAIRS)))
    for start in range(0, 60, 10):
        assert len({r['selection_cell'] for r in rows[start:start+10]}) == 1
        indices = rng.integers(0, 10, size=(50_000, 10))+start
        boot += matrix[indices].sum(axis=1)/60
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    for i, (a, b) in enumerate(PAIRS):
        reported = final['summary'][f'{a}_minus_{b}']
        near(reported['mean'], matrix[:, i].mean())
        for j in range(2):
            near(reported['ci95'][j], intervals[j, i])
        for m in METRICS[1:]:
            near(final['secondary_wdl_contrasts'][m][f'{a}_minus_{b}'],
                 statistics.fmean(r['gains'][m][a]-r['gains'][m][b] for r in rows))
    for p in ('same_world', 'independent'):
        assert final['switches'][p] == sum(r['switches'][p] for r in rows)
    assert counts['continuations'] == sum(t.get('continuations', 0) for t in final['telemetry'].values())
    assert file_sha(ROOT/'result.json') == final_sha
    record = {'passed': True, 'result_sha256': final_sha, 'states': len(rows), **dict(counts),
              'all_primary_means_and_ci_endpoints_reproduced': True,
              'screening_and_held_out_policy_routing_reproduced': True,
              'tolerance': 1e-12, 'simulation_or_training': False}
    path = ROOT/'independent_verification.json'
    if path.exists():
        assert read(path) == record
    else:
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
