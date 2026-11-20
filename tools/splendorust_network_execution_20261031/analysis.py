"""Exact saved-output validation and the fixed, position-clustered speed screen.

No model invocation, timing-overhead subtraction, population latency estimate,
strength claim or deployment qualification is performed here. Invalid or
incomplete evidence raises; only a complete exact comparison can have a valid
negative timing result.
"""
from __future__ import annotations

from collections import Counter
import math
import statistics

from . import common as c
from tools.splendorust_search256_efficiency_20261031.analysis import panel_roots


SCHEMA = 'splendorust_network_execution_evaluation_v1'
BOOTSTRAP_SEED = 20261031257
BOOTSTRAP_DRAWS = 10000
VARIANTS = ('control', 'candidate')
GROUPS = ('systematic', 'tail')
REPEATS = 4
THRESHOLDS = {
    'systematic_paired_median_ratio_maximum': .95,
    'systematic_one_sided_95_upper_ratio_exclusive_maximum': 1.,
    'tail_paired_median_ratio_maximum': 1.05,
}


def require(condition, message):
    if not condition:
        raise ValueError('NETWORK_EXECUTION_ANALYSIS: ' + message)


def _number(value, name, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), name + ' finite valid number')
    return value


def evaluation_matrix(panel):
    """Four rotated repetitions, two orders per root, fixed before localization."""
    by_id = panel_roots(panel)
    roots = list(by_id.values())
    matrix = []
    for repeat in range(REPEATS):
        indices = list(range(repeat, len(roots))) + list(range(repeat))
        for position, panel_index in enumerate(indices):
            root = roots[panel_index]
            variants = VARIANTS if (panel_index + repeat) % 2 == 0 else VARIANTS[::-1]
            for order, variant in enumerate(variants):
                matrix.append(dict(index=len(matrix), pair_index=repeat*18+position,
                    root_id=root['root_id'], panel_id=root['panel_id'], variant=variant,
                    repeat=repeat, panel_index=panel_index, group=root['group'], order=order))
    require(len(matrix) == 144, 'fixed144 request matrix')
    return matrix


def _validate_row(row, spec, root):
    require(type(row) is dict and c.canonical(row.get('spec')) == c.canonical(spec),
            'complete ordered request identity')
    require(row.get('variant') == spec['variant'] and row.get('root_id') == spec['root_id'] and
            row.get('route') == 'research', 'row variant/root/unchanged256 route')
    for name in ('observer', 'before', 'after', 'action', 'trace'):
        require(c.canonical(row.get(name)) == c.canonical(root[name]),
                'exact historical ' + name)
    require(type(row.get('completed')) is bool and row['completed'],
            'every prescribed request completed; no failure removal or replacement')
    clock = row.get('clock')
    require(type(clock) is dict and clock.get('status') == 'ok' and
            clock.get('accepted') is True and clock.get('received_before_deadline') is True,
            'accepted complete response before unchanged research guard')
    require(type(clock.get('deadline_seconds')) in (int, float) and
            clock['deadline_seconds'] == 10., 'unchanged ten-second complete-response guard')
    elapsed = _number(clock.get('receipt_elapsed_seconds'), 'complete-response seconds', positive=True)
    require(elapsed <= 10., 'accepted complete response within ten seconds')
    result = clock.get('result')
    require(c.canonical(result) == c.canonical(root['original_result256']),
            'complete original scientific result including all numerical outputs and search work')
    require(c.canonical(row['action']) == c.canonical(result['action']) and
            c.canonical(row['trace']) == c.canonical(result['diagnostics']),
            'action and trace bind the full result')
    return elapsed


def _distribution(values):
    ordered = sorted(_number(x, 'duration', positive=True) for x in values)
    require(bool(ordered), 'nonempty timing group')
    location = (len(ordered)-1)*.95
    lo, hi = math.floor(location), math.ceil(location)
    return dict(count=len(ordered), minimum=ordered[0], mean=statistics.fmean(ordered),
        median=statistics.median(ordered), p95=ordered[lo]+(ordered[hi]-ordered[lo])*(location-lo),
        maximum=ordered[-1], completed_over_700ms=sum(x > .7 for x in ordered),
        completed_over_900ms=sum(x > .9 for x in ordered),
        completed_over_one_second=sum(x > 1. for x in ordered))


def _paired_summary(pairs):
    ratios = [x['ratio'] for x in pairs]
    differences = [x['candidate_minus_control_seconds'] for x in pairs]
    return dict(pairs=len(pairs), paired_median_ratio=statistics.median(ratios),
        paired_mean_ratio=statistics.fmean(ratios),
        paired_median_extra_seconds=statistics.median(differences),
        paired_mean_extra_seconds=statistics.fmean(differences))


def cluster_bootstrap_upper(pairs):
    """Resample12 roots, each retaining its four paired ratios; never iid moves."""
    import numpy as np
    require(type(pairs) is list and len(pairs) == 48, 'all48 systematic pairs')
    roots = sorted({x['panel_id'] for x in pairs})
    require(len(roots) == 12, 'exactly12 systematic position clusters')
    values = []
    for name in roots:
        cluster = sorted((x for x in pairs if x['panel_id'] == name), key=lambda x: x['repeat'])
        require([x['repeat'] for x in cluster] == list(range(REPEATS)),
                'four distinct repetitions retained per cluster')
        require(all(x.get('group') == 'systematic' for x in cluster),
                'tail positions never enter systematic bootstrap')
        values.append([_number(x['ratio'], 'paired ratio', positive=True) for x in cluster])
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = generator.integers(0, 12, size=(BOOTSTRAP_DRAWS, 12), dtype=np.int64)
    samples = np.median(values[indices].reshape(BOOTSTRAP_DRAWS, 48), axis=1)
    return dict(upper_one_sided_95=float(np.quantile(samples, .95, method='linear')),
        positions=12, pairs_per_position=4, replicates=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED, generator='PCG64', quantile='linear', numpy_version=np.__version__,
        resampling_unit='saved position retaining all four paired repetitions',
        statistic='median of candidate/control complete-response ratios',
        degenerate_resamples_retained=True)


def analyze_evaluation(rows, panel, matrix):
    """Read all144 exact observations before computing the predetermined gates."""
    roots = panel_roots(panel)
    expected_matrix = evaluation_matrix(panel)
    require(type(matrix) is list and c.canonical(matrix) == c.canonical(expected_matrix),
            'exact complete counterbalanced matrix; no omission, duplicate or reorder')
    rows = list(rows)
    require(len(rows) == 144, 'all144 prescribed observations required')
    durations = [_validate_row(row, spec, roots[spec['root_id']])
                 for row, spec in zip(rows, matrix, strict=True)]
    require(Counter(s['group'] for s in matrix) == Counter(systematic=96, tail=48),
            'fixed12 systematic and6 tail positions kept separate')
    pairs = []
    for offset in range(0, 144, 2):
        two = matrix[offset:offset+2]
        by_variant = {s['variant']: (rows[offset+j], durations[offset+j]) for j, s in enumerate(two)}
        require(set(by_variant) == set(VARIANTS), 'each pair contains both execution variants')
        control, candidate = (by_variant[x] for x in VARIANTS)
        require(c.canonical(control[0]['clock']['result']) == c.canonical(candidate[0]['clock']['result']),
                'paired full scientific outputs exact')
        spec = two[0]
        pairs.append(dict(pair_index=spec['pair_index'], panel_id=spec['panel_id'],
            group=spec['group'], repeat=spec['repeat'], control_seconds=control[1],
            candidate_seconds=candidate[1], ratio=candidate[1]/control[1],
            candidate_minus_control_seconds=candidate[1]-control[1]))
    systematic_pairs = [x for x in pairs if x['group'] == 'systematic']
    bootstrap = cluster_bootstrap_upper(systematic_pairs)
    groups = {}
    for group in GROUPS:
        group_pairs = [x for x in pairs if x['group'] == group]
        groups[group] = dict(positions=12 if group == 'systematic' else 6,
            **_paired_summary(group_pairs),
            timings={variant: _distribution([x[variant+'_seconds'] for x in group_pairs])
                     for variant in VARIANTS},
            source_selection='fixed systematic controls' if group == 'systematic'
                             else 'deliberately selected historical slow tails')
    gates = dict(
        systematic_paired_median=groups['systematic']['paired_median_ratio'] <= THRESHOLDS['systematic_paired_median_ratio_maximum'],
        systematic_clustered_upper=bootstrap['upper_one_sided_95'] < THRESHOLDS['systematic_one_sided_95_upper_ratio_exclusive_maximum'],
        tail_paired_median=groups['tail']['paired_median_ratio'] <= THRESHOLDS['tail_paired_median_ratio_maximum'])
    useful = all(gates.values())
    by_position = []
    for name, root in roots.items():
        selected = [x for x in pairs if x['panel_id'] == name]
        require(len(selected) == REPEATS, 'four complete pairs per root')
        by_position.append(dict(panel_id=name, group=root['group'], progress=root['progress'],
            seat=root['seat'], **_paired_summary(selected),
            timings={variant: _distribution([x[variant+'_seconds'] for x in selected])
                     for variant in VARIANTS}))
    return dict(schema=SCHEMA, status='accepted_speedup' if useful else 'valid_no_useful_speedup',
        passed=useful, useful_speedup=useful, comparison_complete=True, all_exact=True,
        requests=144, positions=18, paired_repeats=72, full_historical_output_comparisons=144,
        groups=groups, by_position=by_position, pairs=pairs, bootstrap=bootstrap,
        gates=gates, thresholds=dict(THRESHOLDS), matrix_sha256=c.digest(matrix),
        panel_sha256=c.digest(panel), full_observations_sha256=c.digest(rows),
        failure_removal_or_replacement=False, overhead_subtracted=False,
        deployment_qualified=False, strength_test=False, reference_promoted=False,
        limits=['A fixed-panel implementation feasibility screen; full study acceptance is a separate verification.',
                'The12 systematic roots and6 historical slow tails are reported separately, without a pooled population estimate.',
                'Repeated observations remain position clusters and are not independent sampled positions.',
                'Exact equality on the prescribed evidence is not universal bitwise equivalence.',
                'One second and700ms are descriptive; unchanged research guards remain binding.'])
