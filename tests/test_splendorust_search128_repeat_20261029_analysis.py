"""Model-free repeat analysis/equality regressions; synthetic data are labelled."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.splendorust_search128_repeat_20261029 import analysis as a


def result(observer, quota):
    return dict(simulations_quota=quota, observer_sha256=a.digest(observer),
                profile_id='synthetic_' + str(quota),
                action={'synthetic_root': observer['root']},
                diagnostics=dict(inference_calls=quota+1, inference_rows=quota+1,
                    phases=[dict(neural_callback_count=quota+1,
                                 inference=[{'tokens': [0.0], 'outputs': [[.25]]}])]))


def fixture():
    roots = []
    for i in range(20):
        observer = {'root': i}
        arrival = 2. if i < 10 else .6
        roots.append(dict(root_id='r'+str(i), group='slow' if i < 10 else 'systematic',
            observer=observer, original_result128=result(observer, 128),
            original_receipt_seconds=arrival,
            original_boundary=dict(partition_ns=dict(input_delivery=50000000,
                input_read_to_handler=100000, whole_handler=round(arrival*1e9)-100300000,
                handler_to_write=200000, response_delivery=50000000))))
    panel = {'roots': roots}
    matrix = []
    for rep in range(3):
        for i, root in enumerate(roots):
            routes = ('research', 'frozen') if (i+rep) % 2 == 0 else ('frozen', 'research')
            for route in routes:
                matrix.append(dict(sequence=len(matrix), root_id=root['root_id'],
                    group=root['group'], repeat=rep, route=route, quota=a.ROUTE_QUOTA[route]))
    records = [row_for(spec, roots[int(spec['root_id'][1:])]) for spec in matrix]
    return panel, matrix, records


def row_for(spec, root, seconds=.6, handler=None):
    seconds_ns = round(seconds*1e9)
    handler_ns = round((seconds*.8 if handler is None else handler)*1e9)
    start = 10000000000 + spec['sequence']*20000000000
    read = start+1000000
    hs = read+100000
    he = hs+handler_ns
    prewrite = he+200000
    arrival = start+seconds_ns
    assert prewrite <= arrival
    request = f"request-{spec['sequence']:08}"
    output = result(root['observer'], spec['quota'])
    return dict(spec=deepcopy(spec), observer=deepcopy(root['observer']),
        route=spec['route'], quota=spec['quota'], completed=True,
        action=deepcopy(output['action']), trace=deepcopy(output['diagnostics']),
        clock=dict(status='ok', accepted=True, received_before_deadline=True,
            deadline_seconds=10., request_id=request, receipt_elapsed_seconds=seconds,
            elapsed_seconds=seconds+.1, result=output),
        parent_profile=dict(request_id=request, child_pid=123, request_start_ns=start,
                            receipt_arrival_ns=arrival, response_raw_bytes=12345),
        child_profile=dict(request_id=request, pid=123, request_read_complete_ns=read,
            pre_write_ns=prewrite, profiles=[dict(pid=123, handle_failed=False,
                handle_start_ns=hs, handle_end_ns=he)]))


def test_complete_fixed_workload_exact_comparison_counts_and_no_extra_gate():
    panel, matrix, rows = fixture()
    report = a.analyze(rows, panel, matrix, complete=True)
    v = report['exact_verification']
    assert (v['completed_requests'], v['historical128_comparisons'],
            v['cross_repeat_comparisons'], v['result_cells']) == (120, 60, 80, 40)
    assert v['historical64_comparisons'] == 0
    assert all(g['completed_requests'] == 30 for g in report['groups'])
    assert all(c['complete_repetitions'] for c in report['roots'])
    assert report['recurrence'][0]['roots_by_number_of_three_arrivals_above_1s'] == {
        '0': 10, '1': 0, '2': 0, '3': 0}
    assert not report['new_timing_admission_gate']
    assert not report['reliability_estimate'] and not report['causal_claim']


def test_recurrence_reports_zero_one_two_three_without_automatic_verdict():
    panel, matrix, rows = fixture()
    roots = {r['root_id']: r for r in panel['roots']}
    for i, spec in enumerate(matrix):
        root_index = int(spec['root_id'][1:])
        if spec['route'] == 'research' and root_index < 4:
            late = spec['repeat'] < root_index
            rows[i] = row_for(spec, roots[spec['root_id']], 1.2 if late else .6,
                              1.1 if late else .5)
    report = a.analyze(rows, panel, matrix, complete=True)
    recurrence = report['recurrence'][0]
    expected = {'0': 7, '1': 1, '2': 1, '3': 1}
    assert recurrence['roots_by_number_of_three_arrivals_above_1s'] == expected
    assert recurrence['roots_by_number_of_three_handlers_above_1s'] == expected
    assert 'decision' not in recurrence and 'qualified' not in report


def test_descriptive_thresholds_are_strict_and_failed_clock_is_not_created():
    d = a.distribution([.7, .9, 1., 1.0000001])
    assert (d['above_700ms'], d['above_900ms'], d['above_1s']) == (3, 2, 1)
    assert d['median'] == .95
    panel, matrix, rows = fixture()
    rows[0] = row_for(matrix[0], panel['roots'][0], 2., 1.8)
    checked = a.verify_records(rows[:1], panel, matrix, complete=False)
    assert checked['completed_requests'] == 1  # 1s is not this study's guard.


@pytest.mark.parametrize('change', ['trace', 'action', 'capture', 'identity'])
def test_historical128_checks_complete_result_not_action_only(change):
    panel, matrix, rows = fixture()
    row = rows[0]
    if change == 'trace':
        row['clock']['result']['diagnostics']['inference_calls'] += 1
        row['trace'] = deepcopy(row['clock']['result']['diagnostics'])
    elif change == 'action':
        row['clock']['result']['action'] = {'changed': True}
        row['action'] = {'changed': True}
    elif change == 'capture':
        row['clock']['result']['diagnostics']['phases'][0]['inference'][0]['outputs'][0][0] = .3
        row['trace'] = deepcopy(row['clock']['result']['diagnostics'])
    else:
        row['clock']['result']['profile_id'] += '_changed'
    with pytest.raises(ValueError, match='historical128'):
        a.validate_repeat_row(row, matrix[0], {}, panel)


def test_frozen64_uses_own_first_repeat_without_fabricated_historical64():
    panel, matrix, rows = fixture()
    frozen = [r for r in rows if r['spec']['root_id'] == 'r0' and r['route'] == 'frozen']
    cache = {}
    first = a.validate_repeat_row(frozen[0], frozen[0]['spec'], cache, panel)
    assert not first['historical128_comparison'] and not first['cross_repeat_comparison']
    frozen[1]['clock']['result']['profile_id'] = 'changed64'
    with pytest.raises(ValueError, match='across same-root repeats'):
        a.validate_repeat_row(frozen[1], frozen[1]['spec'], cache, panel)


def test_failure_preserved_terminal_only_and_not_counted_as_success():
    panel, matrix, rows = fixture()
    failure = rows[2]
    failure['completed'] = False
    failure['clock'].update(status='worker_error', accepted=False,
        received_before_deadline=True, receipt_elapsed_seconds=.8,
        detail='STRENGTH_INTERNAL_DEADLINE: 9s', result=None)
    failure['action'] = failure['trace'] = None
    prefix = a.analyze(rows[:3], panel, matrix, complete=False)
    assert (prefix['completed_requests'], prefix['failed_requests']) == (2, 1)
    assert prefix['failures'][0]['original_rejected_clock_retained']
    assert not prefix['recurrence'][0]['full_panel_interpretation_available']
    with pytest.raises(ValueError, match='continuation'):
        a.verify_records(rows[:4], panel, matrix, complete=False)
    with pytest.raises(ValueError):
        a.verify_records(rows, panel, matrix, complete=True)


def test_illegal_packet_can_be_retained_without_becoming_completed():
    panel, matrix, rows = fixture()
    row = rows[0]
    row['completed'] = False
    row['clock'].update(status='illegal', accepted=False, detail='scientific mismatch')
    cache = {}
    proof = a.validate_repeat_row(row, matrix[0], cache, panel)
    assert not proof['accepted'] and proof['result_sha256'] is None and cache == {}


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'quota_bool', 'repeat_bool', 'sequence', 'group'])
def test_matrix_corruption_rejected(mutation):
    panel, matrix, rows = fixture()
    if mutation == 'duplicate':
        matrix[1] = dict(matrix[0], sequence=1)
    elif mutation == 'missing':
        matrix.pop()
    elif mutation == 'quota_bool':
        matrix[0]['quota'] = True
    elif mutation == 'repeat_bool':
        matrix[0]['repeat'] = False
    elif mutation == 'sequence':
        matrix[0]['sequence'] = 1
    else:
        matrix[0]['group'] = 'systematic'
    with pytest.raises(ValueError):
        a.verify_records(rows, panel, matrix, complete=True)


@pytest.mark.parametrize('mutation', ['observer', 'route', 'row_spec', 'shortened_clock', 'trace_binding'])
def test_row_binding_corruption_rejected(mutation):
    panel, matrix, rows = fixture()
    row = rows[0]
    if mutation == 'observer': row['observer']['root'] = 1000
    elif mutation == 'route': row['route'] = 'frozen'
    elif mutation == 'row_spec': row['spec']['sequence'] = 99
    elif mutation == 'shortened_clock': row['clock']['deadline_seconds'] = 1.
    else: row['trace']['inference_calls'] = 2
    with pytest.raises(ValueError):
        a.validate_repeat_row(row, matrix[0], {}, panel)


def test_boundary_exact_partition_does_not_use_full_elapsed_or_overlapping_span():
    panel, matrix, rows = fixture()
    rows[0]['clock']['elapsed_seconds'] = 10.
    rows[0]['child_profile']['post_flush_ns'] = rows[0]['parent_profile']['receipt_arrival_ns'] + 1000000
    measured = a.boundary_measurement(rows[0])
    assert measured['receipt_seconds'] == .6
    assert sum(measured['partition_ns'].values()) == 600000000
    assert measured['handler_seconds'] == .48
    assert 'cpu_wait_seconds' not in measured


@pytest.mark.parametrize('mutation', ['missing', 'reversed', 'arrival', 'identity', 'callbacks'])
def test_boundary_counterexamples(mutation):
    panel, matrix, rows = fixture()
    row = rows[0]
    if mutation == 'missing': del row['child_profile']['pre_write_ns']
    elif mutation == 'reversed': row['child_profile']['profiles'][0]['handle_end_ns'] = 1
    elif mutation == 'arrival': row['clock']['receipt_elapsed_seconds'] += .001
    elif mutation == 'identity': row['child_profile']['request_id'] = 'wrong'
    else: row['trace']['inference_calls'] += 1
    with pytest.raises(ValueError): a.boundary_measurement(row)


def test_partial_empty_distribution_has_no_invented_zero_or_full_recurrence():
    panel, matrix, rows = fixture()
    report = a.analyze([], panel, matrix, complete=False)
    assert report['groups'][0]['arrival']['mean'] is None
    assert report['roots'][0]['original128_comparison']['new_median_minus_original_seconds'] is None
    assert report['recurrence'][0]['incomplete_roots'] == 10
    assert report['recurrence'][0]['roots_by_number_of_three_arrivals_above_1s']['0'] == 0


def test_saved_analysis_recomputed_with_fresh_cache_and_tamper_rejected():
    panel, matrix, rows = fixture()
    report = a.analyze(rows, panel, matrix, complete=True)
    receipt = a.verify_saved_analysis(report, rows, panel, matrix, complete=True)
    assert receipt['status'] == 'passed' and receipt['model_calls'] == 0
    report['groups'][0]['arrival']['mean'] += .001
    with pytest.raises(ValueError, match='fresh saved-row readback'):
        a.verify_saved_analysis(report, rows, panel, matrix, complete=True)


def test_guard_failure_propagates_without_success():
    panel, matrix, rows = fixture()
    def stop(): raise RuntimeError('review deadline')
    with pytest.raises(RuntimeError, match='review deadline'):
        a.analyze(rows, panel, matrix, complete=True, boundary=stop)


def test_actual_saved128_result_and_boundary_schema():
    """A real immutable accepted result exercises field assumptions, without inference."""
    path = Path(__file__).resolve().parents[1] / (
        'outputs/splendorust_search128_strength_20261028_01/games/main/p016_g33/moves/0017.json')
    old = json.loads(path.read_text())
    from tools.splendorust_search128_saved_timing_20261029 import extract
    panel, matrix, _ = fixture()
    root = panel['roots'][0]
    root.update(observer=old['observer'], original_result128=old['clock']['result'],
                original_receipt_seconds=old['clock']['receipt_elapsed_seconds'],
                original_boundary=extract(old))
    row = dict(old, spec=matrix[0], route='research')
    proof = a.validate_repeat_row(row, matrix[0], {}, panel)
    assert proof['historical128_comparison']
    measured = a.boundary_measurement(row)
    assert measured['inference_calls'] == 129
    assert measured['partition_ns']['whole_handler'] == 1887434100
    assert measured['receipt_seconds'] == 2.0121378
