"""Pure saved-row equality and descriptive timing analysis; never calls a player.

This selected repeat panel is not a reliability sample or an admission test.
Scientific/transport/placement validation belongs to the separate collector audit.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import statistics

SCHEMA = 'search128_repeat_analysis_20261029_v1'
ROUTE_QUOTA = {'research': 128, 'frozen': 64}
GROUPS = ('slow', 'systematic')
SEGMENTS = ('input_delivery', 'input_read_to_handler', 'whole_handler',
            'handler_to_write', 'response_delivery')


def require(ok, message):
    if not ok:
        raise ValueError('SEARCH128_REPEAT_ANALYSIS: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            name + ' finite nonnegative number')
    return value


def distribution(values):
    values = sorted(number(x, 'timing') for x in values)
    return dict(count=len(values), minimum=values[0] if values else None,
                median=statistics.median(values) if values else None,
                mean=statistics.fmean(values) if values else None,
                maximum=values[-1] if values else None,
                above_700ms=sum(x > .7 for x in values),
                above_900ms=sum(x > .9 for x in values),
                above_1s=sum(x > 1. for x in values))


def panel_roots(panel):
    require(type(panel) is dict and type(panel.get('roots')) is list,
            'Panel roots list')
    roots = panel['roots']
    require(len(roots) == 20 and Counter(r.get('group') for r in roots) ==
            Counter(slow=10, systematic=10), 'Exactly ten slow and ten systematic roots')
    by_id = {}
    for root in roots:
        name = root.get('root_id')
        require(type(name) is str and name and name not in by_id, 'Unique root identity')
        require(type(root.get('observer')) is dict and
                type(root.get('original_result128')) is dict, 'Original observer/result present')
        require(root['original_result128'].get('simulations_quota') == 128 and
                root['original_result128'].get('observer_sha256') == digest(root['observer']),
                'Original128 result observer/quota binding')
        original = number(root.get('original_receipt_seconds'), 'Original trusted arrival')
        require(0 < original <= 10., 'Original research arrival bound')
        b = root.get('original_boundary')
        require(type(b) is dict and type(b.get('partition_ns')) is dict and
                set(b['partition_ns']) == set(SEGMENTS), 'Original exact boundary partition')
        require(all(type(x) is int and x >= 0 for x in b['partition_ns'].values()),
                'Original integer boundary spans')
        require(abs(sum(b['partition_ns'].values()) / 1e9 - original) <= 1e-9,
                'Original boundary agrees with original trusted arrival')
        by_id[name] = root
    return by_id


def validate_layout(panel, matrix):
    roots = panel_roots(panel)
    require(type(matrix) is list and len(matrix) == 120, 'Fixed120 request matrix')
    cells = set()
    for sequence, spec in enumerate(matrix):
        require(type(spec) is dict and type(spec.get('sequence')) is int and
                spec['sequence'] == sequence, 'Exact contiguous matrix sequence')
        root = roots.get(spec.get('root_id'))
        require(root is not None and spec.get('group') == root['group'], 'Matrix root/group')
        route = spec.get('route')
        require(route in ROUTE_QUOTA and type(spec.get('quota')) is int and
                spec['quota'] == ROUTE_QUOTA[route], 'Exact matrix route/quota')
        repeat = spec.get('repeat')
        require(type(repeat) is int and repeat in (0, 1, 2), 'Three fixed repeat indices')
        cell = (spec['root_id'], route, repeat)
        require(cell not in cells, 'No duplicate matrix cell')
        cells.add(cell)
    require(cells == {(root, route, repeat) for root in roots for route in ROUTE_QUOTA
                      for repeat in range(3)}, 'All120 fixed cells without replacement')
    return roots


def validate_repeat_row(row, spec, cache, panel, *, boundary=lambda: None):
    """Immediate exact result comparison; a rejected request is never accepted here."""
    boundary()
    roots = panel_roots(panel)
    require(type(row) is dict and canonical(row.get('spec')) == canonical(spec),
            'Saved row is the exact scheduled request')
    root = roots.get(spec.get('root_id'))
    require(root is not None and spec.get('group') == root['group'] and
            spec.get('route') in ROUTE_QUOTA and type(spec.get('quota')) is int and
            spec['quota'] == ROUTE_QUOTA[spec['route']], 'Declared row root/route/quota')
    require(row.get('route') == spec['route'] and row.get('quota') == spec['quota'],
            'Outer row route/quota binding')
    require(canonical(row.get('observer')) == canonical(root['observer']),
            'Same lawful saved observer')
    clock = row.get('clock')
    require(type(clock) is dict and type(row.get('completed')) is bool and
            row['completed'] == (clock.get('status') == 'ok'), 'Original completion status')
    require(type(clock.get('deadline_seconds')) in (int, float) and
            clock['deadline_seconds'] == 10., 'Unshortened10s ordinary guard')
    if not row['completed']:
        require(clock.get('accepted') is False, 'Failed row remains rejected')
        return dict(status='failed_request_preserved', sequence=spec['sequence'],
                    accepted=False, historical128_comparison=False,
                    cross_repeat_comparison=False, result_sha256=None)
    require(clock.get('accepted') is True and clock.get('received_before_deadline') is True,
            'Completed original trusted clock')
    require(number(clock.get('receipt_elapsed_seconds'), 'Trusted arrival') <= 10.,
            'Actual research complete-response guard')
    result = clock.get('result')
    require(type(result) is dict and result.get('simulations_quota') == spec['quota'] and
            result.get('observer_sha256') == digest(root['observer']),
            'Raw result quota and observer binding')
    require(canonical(row.get('action')) == canonical(result.get('action')) and
            canonical(row.get('trace')) == canonical(result.get('diagnostics')),
            'Saved action and full trace are raw result values')
    encoded = canonical(result)
    historical = spec['route'] == 'research'
    if historical:
        require(encoded == canonical(root['original_result128']),
                'Exact full historical128 result including trace/capture/identity')
    key = (spec['root_id'], spec['route'])
    previous = cache.get(key)
    if previous is not None:
        require(previous['result'] == encoded, 'Exact full result across same-root repeats')
    else:
        cache[key] = dict(result=encoded, first_sequence=spec['sequence'])
    return dict(status='passed', sequence=spec['sequence'], accepted=True,
                historical128_comparison=historical,
                cross_repeat_comparison=previous is not None,
                first_sequence=cache[key]['first_sequence'],
                result_sha256=hashlib.sha256(encoded.encode()).hexdigest())


def verify_records(records, panel, matrix, *, complete, boundary=lambda: None):
    """Fresh-cache exact saved-prefix readback; no model or transition execution."""
    validate_layout(panel, matrix)
    require(type(complete) is bool, 'Explicit completion scope')
    records = list(records)
    require(len(records) <= 120 and (not complete or len(records) == 120),
            'Bounded fixed prefix or complete120')
    cache, proofs, failed = {}, [], False
    for index, row in enumerate(records):
        boundary()
        require(not failed, 'No continuation after first failed ordinary request')
        proof = validate_repeat_row(row, matrix[index], cache, panel, boundary=boundary)
        proofs.append(proof)
        failed = not proof['accepted']
    require(not complete or not failed, 'Complete panel cannot contain a failed request')
    return dict(status='complete_panel_exact' if complete else 'prefix_exact',
                planned_requests=120, attempted_requests=len(records),
                completed_requests=sum(p['accepted'] for p in proofs),
                failed_requests=sum(not p['accepted'] for p in proofs),
                historical128_comparisons=sum(p['historical128_comparison'] for p in proofs),
                cross_repeat_comparisons=sum(p['cross_repeat_comparison'] for p in proofs),
                result_cells=len(cache), historical64_comparisons=0,
                historical64_targets_assumed=False,
                exact_full_results=True, proofs=proofs)


def boundary_measurement(row):
    """Five additive wall spans. Other overlapping/CPU spans are not subtracted."""
    clock, parent, child = row['clock'], row['parent_profile'], row['child_profile']
    require(len(child.get('profiles', [])) == 1, 'One ordinary handler profile')
    handle = child['profiles'][0]
    require(handle.get('handle_failed') is False and handle.get('pid') == child.get('pid'),
            'Successful handler profile identity')
    require(parent.get('request_id') == child.get('request_id') == clock.get('request_id'),
            'Boundary request identities')
    require(parent.get('child_pid') == child.get('pid'), 'Parent/child PID binding')
    stamps = [parent.get('request_start_ns'), child.get('request_read_complete_ns'),
              handle.get('handle_start_ns'), handle.get('handle_end_ns'),
              child.get('pre_write_ns'), parent.get('receipt_arrival_ns')]
    require(all(type(x) is int and x > 0 for x in stamps) and stamps == sorted(stamps),
            'Complete ordered existing boundary markers')
    pieces = dict(zip(SEGMENTS, (b-a for a, b in zip(stamps, stamps[1:]))))
    require(abs(sum(pieces.values()) / 1e9 - clock['receipt_elapsed_seconds']) <= 1e-9,
            'Exact nonoverlapping trusted-arrival partition')
    trace = row['trace']
    phases = trace.get('phases')
    require(type(phases) is list and type(trace.get('inference_calls')) is int and
            trace['inference_calls'] >= 0, 'Saved callback counts')
    callbacks = [p.get('neural_callback_count') for p in phases]
    require(all(type(n) is int and n >= 0 for n in callbacks) and
            sum(callbacks) == trace['inference_calls'], 'Full-phase callback count reconciliation')
    return dict(receipt_seconds=clock['receipt_elapsed_seconds'],
                handler_seconds=pieces['whole_handler'] / 1e9,
                partition_ns=pieces, inference_calls=trace['inference_calls'],
                phase_count=len(phases), phase_callback_counts=callbacks,
                response_raw_bytes=parent.get('response_raw_bytes'))


def analyze(records, panel, matrix, *, complete, boundary=lambda: None):
    records = list(records)
    verification = verify_records(records, panel, matrix, complete=complete, boundary=boundary)
    roots = panel_roots(panel)
    measurements, failures = [], []
    for row in records:
        boundary()
        if row['completed']:
            measurements.append(dict(**row['spec'], **boundary_measurement(row)))
        else:
            failures.append(dict(spec=row['spec'], status=row['clock']['status'],
                detail=row['clock'].get('detail'), receipt_elapsed_seconds=row['clock'].get('receipt_elapsed_seconds'),
                original_rejected_clock_retained=True, replacement_or_retry=False))
    groups = []
    for group in GROUPS:
        for route, quota in ROUTE_QUOTA.items():
            rr = [r for r in measurements if r['group'] == group and r['route'] == route]
            groups.append(dict(group=group, route=route, quota=quota,
                expected_requests=30, completed_requests=len(rr),
                arrival=distribution(r['receipt_seconds'] for r in rr),
                handler=distribution(r['handler_seconds'] for r in rr),
                by_repeat=[dict(repeat=rep, completed_requests=sum(r['repeat'] == rep for r in rr),
                    arrival=distribution(r['receipt_seconds'] for r in rr if r['repeat'] == rep),
                    handler=distribution(r['handler_seconds'] for r in rr if r['repeat'] == rep))
                    for rep in range(3)]))
    cells = []
    for root_id, root in roots.items():
        for route, quota in ROUTE_QUOTA.items():
            rr = sorted((r for r in measurements if r['root_id'] == root_id and r['route'] == route),
                        key=lambda r: r['repeat'])
            arrival = distribution(r['receipt_seconds'] for r in rr)
            handler = distribution(r['handler_seconds'] for r in rr)
            original = None
            if route == 'research':
                old_arrival = root['original_receipt_seconds']
                old_handler = root['original_boundary']['partition_ns']['whole_handler'] / 1e9
                original = dict(receipt_seconds=old_arrival, handler_seconds=old_handler,
                    new_median_minus_original_seconds=None if not rr else arrival['median'] - old_arrival,
                    new_median_to_original_ratio=None if not rr else arrival['median'] / old_arrival,
                    new_handler_median_minus_original_seconds=None if not rr else handler['median'] - old_handler,
                    temporal_and_extreme_selection_comparison=True, causal_effect_estimate=False)
            cells.append(dict(root_id=root_id, group=root['group'], route=route, quota=quota,
                expected_repetitions=3, complete_repetitions=len(rr) == 3, measurements=rr,
                arrival=arrival, handler=handler, original128_comparison=original,
                historical64_target_available=False if route == 'frozen' else None))
    recurrence = []
    for group in GROUPS:
        cc = [r for r in cells if r['group'] == group and r['route'] == 'research']
        done = [r for r in cc if r['complete_repetitions']]
        recurrence.append(dict(group=group, roots=10, complete_roots=len(done),
            incomplete_roots=10-len(done),
            roots_by_number_of_three_arrivals_above_1s={str(n): sum(r['arrival']['above_1s'] == n for r in done) for n in range(4)},
            roots_by_number_of_three_handlers_above_1s={str(n): sum(r['handler']['above_1s'] == n for r in done) for n in range(4)},
            full_panel_interpretation_available=complete))
    return dict(schema=SCHEMA, complete=complete, planned_requests=120,
        attempted_requests=len(records), completed_requests=len(measurements),
        failed_requests=len(failures), exact_verification=verification,
        groups=groups, roots=cells, recurrence=recurrence, failures=failures,
        timing_metric='Trusted complete-line arrival before parent decoding/validation',
        quantiles='Median of observed values; no tail quantile estimate from three repeats',
        thresholds_seconds=[.7, .9, 1.], thresholds_are_descriptive=True,
        research_guards=dict(internal_seconds=9., complete_response_seconds=10.),
        new_timing_admission_gate=False, reliability_estimate=False,
        strength_claim=False, causal_claim=False, automatic_promotion=False,
        interpretation=[
            'Recurrence is reported as0/1/2/3 of three responses above1s for every complete root.',
            'Original extremes were selected after observing timing; old/new differences include regression to the mean and temporal conditions.',
            'The selected20 roots are not representative of future games or timeout risk.',
            'Handler wall time includes native search, inference, inner transport and bookkeeping; it is not CPU or inference time.',
            'No worker-tree CPU subtraction, overlapping interval addition, or scheduling attribution is performed.',
            'Historical64 same-root outputs are not assumed. Both quotas must reproduce their own complete outputs across repeats.',
            'Incomplete prefixes retain failures and cannot establish full-panel recurrence or admission.'])


def verify_saved_analysis(saved, records, panel, matrix, *, complete, boundary=lambda: None):
    """Re-read supplied saved rows using a fresh equality cache and pure analysis.

    The caller authenticates bytes/paths and lifecycle receipts. This is a fresh
    recomputation using the frozen analysis, not independent model execution.
    """
    recomputed = analyze(records, panel, matrix, complete=complete, boundary=boundary)
    require(canonical(saved) == canonical(recomputed), 'Saved analysis equals fresh saved-row readback')
    return dict(status='passed', analysis_sha256=digest(recomputed),
                complete=complete, attempted_requests=recomputed['attempted_requests'],
                historical128_comparisons=recomputed['exact_verification']['historical128_comparisons'],
                cross_repeat_comparisons=recomputed['exact_verification']['cross_repeat_comparisons'],
                historical64_comparisons=0, model_calls=0, new_games=0,
                scientific_transition_executions=0, same_frozen_analysis_recomputed=True,
                outer_file_hash_clock_source_ownership_checks_required=True)
