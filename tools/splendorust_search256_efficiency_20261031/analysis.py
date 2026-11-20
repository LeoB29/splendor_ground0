"""Exact saved256 equality, paired observer overhead and descriptive components.

No playing-strength, deployment, scheduling-cause or population latency claim.
All measured requests are retained; a failure cannot be removed from a gate.
"""
from __future__ import annotations

from collections import Counter
import math
import statistics
from . import panel as p

BOOTSTRAP_SEED = 20261031256
BOOTSTRAP_DRAWS = 10000
OVERHEAD_LIMITS = dict(paired_median_ratio_maximum=1.05,
                      paired_median_extra_seconds_maximum=.050,
                      cluster_bootstrap_upper95_median_ratio_maximum=1.10)
COMPONENT_LABELS = ('compiled_forward', 'inference_bridge', 'callback_total',
                    'native_exchange', 'native_read_wait', 'native_write', 'native_flush')


def require(ok, message):
    if not ok:
        raise ValueError('SEARCH256_EFFICIENCY_ANALYSIS: ' + message)


def number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), name + ' finite number')
    return value


def distribution(values, *, timing=True):
    ordered = sorted(number(x, 'timing') for x in values)
    def quantile(q):
        if not ordered:
            return None
        index = (len(ordered) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    result = dict(count=len(ordered), minimum=ordered[0] if ordered else None,
        mean=statistics.fmean(ordered) if ordered else None, median=quantile(.5),
        p95=quantile(.95), maximum=ordered[-1] if ordered else None)
    if timing:
        result.update(completed_over_700ms=sum(x > .7 for x in ordered),
                      completed_over_900ms=sum(x > .9 for x in ordered),
                      completed_over_one_second=sum(x > 1. for x in ordered))
    return result


def signed_distribution(values):
    require(bool(values) and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            'nonempty finite signed paired differences')
    ordered = sorted(values)
    return dict(count=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                minimum=ordered[0], maximum=ordered[-1])


def panel_roots(panel):
    require(panel.get('schema') == p.SCHEMA and panel.get('selection') == p.selection_rule(),
            'exact panel schema and prespecified selection')
    roots = panel.get('roots')
    require(type(roots) is list and len(roots) == 18, 'all eighteen fixed roots')
    require(Counter(r['group'] for r in roots) == Counter(systematic=12, tail=6),
            'systematic and targeted populations separate')
    require(len({r['game_index'] for r in roots}) == 18, 'eighteen distinct source games')
    by_id = {}
    for root in roots:
        name = root['root_id']
        require(name == root['panel_id'] and name not in by_id, 'unique root identity aliases')
        require(root['progress'] == p.progress(root['ply'], root['game_plies']) and
                type(root['seat']) is int and root['seat'] in (0, 1), 'root progress/seat')
        require(root['original_result256']['simulations_quota'] == 256 and
                root['original_result256']['observer_sha256'] == p.digest(root['observer']),
                'full original result observer/quota binding')
        require(root['observer_seed'] == int(p.digest(root['observer'])[:16], 16) and
                root['trace']['observer_seed'] == root['observer_seed'], 'original deterministic seed')
        require(root['action'] == root['original_result256']['action'] and
                root['trace'] == root['original_result256']['diagnostics'], 'full original output binding')
        by_id[name] = root
    for progress in p.PROGRESS:
        for seat in (0, 1):
            require(Counter(r['group'] for r in roots if r['progress'] == progress and r['seat'] == seat) ==
                    Counter(systematic=2, tail=1), 'fixed two controls and one tail per cell')
    return by_id


def validate_layout(panel, matrix, phase):
    roots = panel_roots(panel)
    require(p.canonical(matrix) == p.canonical(p.request_matrix(panel, phase)),
            'complete ordered fixed matrix; no omission, duplicate, reorder or substitution')
    return roots


def validate_row(row, spec, panel, *, boundary=lambda: None):
    boundary()
    roots = panel_roots(panel)
    require(type(row) is dict, 'saved request object')
    observed = row.get('spec', {key: row.get(key) for key in spec})
    require(p.canonical(observed) == p.canonical(spec), 'exact scheduled request identity')
    root = roots[spec['root_id']]
    require(p.canonical(row.get('observer')) == p.canonical(root['observer']), 'exact saved lawful input')
    require(p.canonical(row.get('before')) == p.canonical(root['before']), 'exact saved before')
    clock = row.get('clock')
    require(type(clock) is dict and clock.get('deadline_seconds') == 10. and
            type(row.get('completed')) is bool and row['completed'] == (clock.get('status') == 'ok'),
            'unchanged trusted completion and ten-second guard')
    if not row['completed']:
        require(clock.get('accepted') is False, 'failed request is never accepted')
        return dict(status='failed_request_preserved', accepted=False, full_exact_equality=False,
                    index=spec['index'], panel_id=spec['panel_id'], result_sha256=None)
    require(clock.get('accepted') is True and clock.get('received_before_deadline') is True and
            0 < number(clock.get('receipt_elapsed_seconds'), 'trusted complete-line arrival') <= 10.,
            'accepted trusted research clock')
    require(p.canonical(row.get('after')) == p.canonical(root['after']), 'exact saved after')
    result = clock.get('result')
    require(p.canonical(result) == p.canonical(root['original_result256']),
            'exact full historical256 result including numerical arrays, seeds and all scientific trace')
    require(p.canonical(row.get('action')) == p.canonical(result['action']) and
            p.canonical(row.get('trace')) == p.canonical(result['diagnostics']), 'result/action/trace agreement')
    return dict(status='passed', accepted=True, full_exact_equality=True,
                index=spec['index'], panel_id=spec['panel_id'], result_sha256=p.digest(result))


def _validated(rows, panel, phase):
    rows = list(rows)
    matrix = p.request_matrix(panel, phase)
    validate_layout(panel, matrix, phase)
    require(len(rows) == len(matrix), 'complete fixed measured requests')
    checks = [validate_row(row, spec, panel) for row, spec in zip(rows, matrix, strict=True)]
    return rows, matrix, checks


def _pairs(rows, matrix):
    grouped = {}
    for row, spec in zip(rows, matrix, strict=True):
        key = (spec['panel_id'], spec['repeat'])
        cell = grouped.setdefault(key, {})
        require(spec['mode'] not in cell, 'unique paired mode')
        cell[spec['mode']] = row
    result = []
    for (panel_id, repeat), pair in grouped.items():
        require(set(pair) == set(p.MODES), 'complete OFF/ON pair')
        off, on = (number(pair[mode]['clock']['receipt_elapsed_seconds'], 'paired arrival', positive=True)
                   for mode in p.MODES)
        result.append(dict(panel_id=panel_id, repeat=repeat, off_seconds=off, on_seconds=on,
                           ratio=on/off, extra_seconds=on-off))
    return result


def cluster_bootstrap_upper(pairs):
    import numpy as np
    roots = sorted({r['panel_id'] for r in pairs})
    require(len(roots) == 6 and len(pairs) == 24, 'six overhead clusters of four paired repeats')
    values = []
    for root in roots:
        cluster = sorted((r for r in pairs if r['panel_id'] == root), key=lambda r: r['repeat'])
        require([r['repeat'] for r in cluster] == list(range(4)), 'every repeated pair retained in its cluster')
        values.append([number(r['ratio'], 'paired ratio', positive=True) for r in cluster])
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(0, 6, size=(BOOTSTRAP_DRAWS, 6), dtype=np.int64)
    medians = np.median(values[indices].reshape(BOOTSTRAP_DRAWS, 24), axis=1)
    return dict(upper_one_sided_95=float(np.quantile(medians, .95, method='linear')),
                positions=6, pairs_per_position=4, replicates=BOOTSTRAP_DRAWS,
                seed=BOOTSTRAP_SEED, generator='PCG64', quantile='linear', numpy_version=np.__version__,
                resampling_unit='saved position retaining all four paired repeats')


def analyze_overhead(rows, panel):
    rows, matrix, checks = _validated(rows, panel, 'overhead')
    exact = all(r['accepted'] and r['full_exact_equality'] for r in checks)
    if not exact:
        return dict(schema='search256_efficiency_overhead_v1', passed=False,
            all_exact=False, failed_requests=[r for r in checks if not r['accepted']],
            gates=None, full_diagnostic_authorized=False, limits=OVERHEAD_LIMITS,
            failure_removal_or_replacement=False)
    pairs = _pairs(rows, matrix)
    median_ratio = statistics.median(r['ratio'] for r in pairs)
    median_extra = statistics.median(r['extra_seconds'] for r in pairs)
    bootstrap = cluster_bootstrap_upper(pairs)
    gates = dict(paired_median_ratio=median_ratio <= OVERHEAD_LIMITS['paired_median_ratio_maximum'],
                 paired_median_extra_seconds=median_extra <= OVERHEAD_LIMITS['paired_median_extra_seconds_maximum'],
                 clustered_upper_ratio=bootstrap['upper_one_sided_95'] <= OVERHEAD_LIMITS['cluster_bootstrap_upper95_median_ratio_maximum'])
    return dict(schema='search256_efficiency_overhead_v1', passed=all(gates.values()),
        all_exact=True, requests=48, positions=6, paired_repeats=24,
        paired_median_ratio=median_ratio, paired_median_extra_seconds=median_extra,
        paired_extra_seconds=signed_distribution([r['extra_seconds'] for r in pairs]),
        bootstrap=bootstrap, gates=gates, limits=OVERHEAD_LIMITS, pairs=pairs,
        full_diagnostic_authorized=all(gates.values()), failure_removal_or_replacement=False,
        scope='qualified on these six positions and host conditions; not a universal observer guarantee')


overhead_gate = analyze_overhead


def component_metrics(row):
    """Read validated disjoint exclusive spans; nested inclusive costs never sum."""
    sidecar = row.get('component_profile')
    require(type(sidecar) is dict and sidecar.get('schema') == 'splendorust_search256_efficiency_v1',
            'instrumentation sidecar required')
    require(sidecar.get('observer_mode') == 'on' and sidecar.get('profile_enabled') is True and
            sidecar.get('request_id') == row['clock']['request_id'] and sidecar.get('quota') == 256 and
            sidecar.get('route') == 'research' and sidecar.get('result_sha256') == p.digest(row['clock']['result']),
            'component profile belongs to exact successful256 request')
    profiles = sidecar.get('profiles')
    require(type(profiles) is list and len(profiles) == 1, 'one ordinary handler profile')
    profile = profiles[0]
    totals = profile['totals']
    require(set(totals) == set(COMPONENT_LABELS), 'complete declared timing label schema')
    values = {name: number(profile[name], name) for name in
              ('handler_wall_ns', 'handler_thread_cpu_ns', 'native_cpu_ns',
               'handler_unattributed_wall_ns', 'handler_unattributed_thread_cpu_ns')}
    for label in COMPONENT_LABELS:
        item = totals[label]
        require(type(item['calls']) is int and item['calls'] >= 0 and item['failures'] == 0,
                'successful exact instrumentation calls')
        for dimension in ('wall', 'thread_cpu'):
            exclusive = number(item[f'exclusive_{dimension}_ns'], 'exclusive ' + label)
            inclusive = number(item[f'inclusive_{dimension}_ns'], 'inclusive ' + label)
            require(exclusive <= inclusive, 'exclusive span within inclusive span')
            values[f'{label}_exclusive_{dimension}_ns'] = exclusive
            values[f'{label}_inclusive_{dimension}_ns'] = inclusive
        values[label + '_calls'] = item['calls']
    for dimension in ('wall', 'thread_cpu'):
        total = sum(values[f'{label}_exclusive_{dimension}_ns'] for label in COMPONENT_LABELS)
        require(total + values[f'handler_unattributed_{dimension}_ns'] == values[f'handler_{dimension}_ns'],
                'disjoint nested spans plus residual exactly partition handler ' + dimension)
    return values


def analyze_diagnostic(rows, panel):
    rows, matrix, checks = _validated(rows, panel, 'diagnostic')
    require(all(x['accepted'] and x['full_exact_equality'] for x in checks),
            'full diagnostic requires all144 diagnostic requests; overhead admission is checked separately')
    roots = panel_roots(panel)
    pairs = _pairs(rows, matrix)
    groups = {}
    by_position = []
    for group in p.GROUPS:
        selected = [(row, spec) for row, spec in zip(rows, matrix, strict=True) if spec['group'] == group]
        on = [component_metrics(row) for row, spec in selected if spec['mode'] == 'on']
        mode_timings = {mode: distribution([r['clock']['receipt_elapsed_seconds'] for r, s in selected if s['mode'] == mode])
                        for mode in p.MODES}
        group_pairs = [r for r in pairs if roots[r['panel_id']]['group'] == group]
        metrics = {(key[:-3] + '_seconds' if key.endswith('_ns') else key):
                   distribution([row[key] / (1e9 if key.endswith('_ns') else 1) for row in on], timing=False)
                   for key in on[0]}
        groups[group] = dict(positions=12 if group == 'systematic' else 6,
            timings=mode_timings, paired_median_ratio=statistics.median(x['ratio'] for x in group_pairs),
            paired_extra_seconds=signed_distribution([x['extra_seconds'] for x in group_pairs]),
            components=metrics, component_time_unit='seconds; fields ending _calls remain counts',
            source_selection='outcome-blind SHA controls' if group == 'systematic' else 'explicit historical slow-tail controls')
    for name, root in roots.items():
        selected = [(row, spec) for row, spec in zip(rows, matrix, strict=True) if spec['root_id'] == name]
        by_position.append(dict(panel_id=name, group=root['group'], progress=root['progress'], seat=root['seat'],
            historical_seconds=root['original_receipt_seconds'],
            timings={mode: distribution([r['clock']['receipt_elapsed_seconds'] for r, s in selected if s['mode'] == mode])
                     for mode in p.MODES}))
    return dict(schema='search256_efficiency_diagnostic_v1', passed=True, all_exact=True,
        requests=144, positions=18, groups=groups, by_position=by_position,
        full_historical_output_comparisons=144, numerical_player_change=False,
        strength_test=False, deployment_qualified=False, causal_host_bottleneck_proven=False,
        limits=['The two panel groups are not pooled into a population latency estimate.',
                'Inclusive nested spans overlap; only exclusive spans and unattributed residual partition wall time.',
                'Native read wait includes native search, communication and waiting; it does not isolate Rust compute.',
                'Thread CPU and native process CPU are separate dimensions and are not added to elapsed wall time.',
                'Wall minus sampled CPU does not identify scheduling, paging or any particular host cause.',
                '700ms and one second are descriptive, while the original9s/10s guards remain binding.'])
