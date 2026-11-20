"""Prospective threshold, missing-evidence and timing-attribution counterexamples."""
from copy import deepcopy
import math

import pytest

from tools.splendorust_minimal_logging_20261028 import analysis as a
from tools.splendorust_minimal_logging_20261028 import logging_probe as probe


def refresh(r):
    value = r['logging_profile']
    value['child_profile_sha256'] = a.c.digest(r.get('child_profile'))
    value['summary'] = probe.validate_profile(value, expected_enabled=r['spec']['logging_timing_enabled'],
                                               child_profile=r.get('child_profile'))
    return r


def row(index, arm, case, mode, elapsed=.4, stage='overhead'):
    route, cpu = arm.rsplit('_cpu', 1)
    markers = dict(zip(a.TIMESTAMPS, (100_000_000, 105_000_000)))
    if not mode: markers = dict.fromkeys(a.TIMESTAMPS)
    r = dict(spec=dict(index=index, sequence=index, round=0, stage=stage, arm_id=arm,
                         route=route, logical_cpu=int(cpu), case_id=case, logging_timing_enabled=mode),
        arm_id=arm, clock=dict(status='ok', accepted=True, deadline_seconds=1.,
                              receipt_elapsed_seconds=elapsed, result={'diagnostics': {}}),
        child_profile=dict(profiles=[dict(handle_start_ns=0, handle_end_ns=300_000_000)],
                           pre_write_ns=300_000_000, post_flush_ns=310_000_000),
        logging_profile=dict(schema=probe.SCHEMA, instrumented=mode, start_record=0, end_record=1,
            record_operations_unchanged=True, synchronous_fsync_retained=True,
            persistence_disposition='after_successful_response_flush',
            timing_metadata_written_after_response_or_error_emission=True, status='recorded',
            records=[dict(sequence=0,
            event='complete_result_before_worker_validation', completed=True,
            error=None, timestamps=markers)]))
    return refresh(r)


def actual_rows():
    rows = []
    for arm in a.ARM_IDS:
        for case in a.CASE_IDS:
            for mode in (False, True):
                for _ in range(4):rows.append(row(len(rows), arm, case, mode))
    return rows


def replay_rows():
    return [dict(case_id=case, logging_timing_enabled=mode, elapsed_seconds=.020,
                 bytes_verified=True) for case in a.CASE_IDS for mode in (False, True) for _ in range(16)]


def test_original_actual_gate_and_new_saved_gate_are_distinct():
    assert a.ACTUAL_GATE == dict(maximum_mean_ratio=1.05, maximum_added_mean_seconds=.010,
                               maximum_added_p95_seconds=.025)
    assert a.REPLAY_GATE == dict(maximum_added_mean_seconds=.005, maximum_added_p95_seconds=.025)
    assert a.overhead_gate(actual_rows())['passed'] is True
    assert a.replay_overhead_gate(replay_rows())['passed'] is True


def test_actual_absolute_mean_gate_even_when_ratio_passes():
    rows = actual_rows()
    for r in rows:
        r['clock']['receipt_elapsed_seconds'] = .5 + (.011 if r['spec']['logging_timing_enabled'] else 0)
    gate = a.overhead_gate(rows)
    assert not gate['passed']
    assert all(g['checks']['mean_ratio'] and not g['checks']['mean'] for g in gate['per_arm'].values())


def test_actual_relative_gate_even_when_absolute_passes():
    rows = actual_rows()
    for r in rows:r['clock']['receipt_elapsed_seconds'] = .100 + (.006 if r['spec']['logging_timing_enabled'] else 0)
    gate = a.overhead_gate(rows)
    assert not gate['passed']
    assert all(g['checks']['mean'] and not g['checks']['mean_ratio'] for g in gate['per_arm'].values())


def test_one_tail_cannot_be_averaged_away():
    rows = actual_rows(); next(r for r in rows if r['spec']['logging_timing_enabled'])['clock']['receipt_elapsed_seconds'] = .426
    gate = a.overhead_gate(rows)
    assert not gate['passed'] and not gate['per_arm']['frozen_cpu0']['checks']['p95']
    assert gate['per_arm']['frozen_cpu0']['checks']['mean']
    assert gate['per_arm']['frozen_cpu0']['on']['p95'] == .426


def test_equal_one_second_is_accepted_by_unchanged_original_clock():
    rows = actual_rows()
    for r in rows:r['clock']['receipt_elapsed_seconds'] = 1.
    assert a.overhead_gate(rows)['passed']
    rows[-1]['clock']['receipt_elapsed_seconds'] = 1.000001
    assert not a.overhead_gate(rows)['passed']


def test_one_failed_clock_never_imputed_or_ignored():
    rows = actual_rows(); rows[-1]['clock'].update(status='worker_error', accepted=False, result=None)
    gate = a.overhead_gate(rows)
    assert gate['complete'] and not gate['passed']
    assert gate['per_arm']['research_cpu2']['on']['n'] == 7
    assert not gate['per_arm']['research_cpu2']['evaluated']


def test_partial_gate_is_explicit_and_detailed_rows_cannot_rehabilitate_it():
    rows = actual_rows()[:-1]
    rows.extend(row(64+i, 'research_cpu2', 'gold_returns', True, .001, 'diagnostic') for i in range(32))
    gate = a.overhead_gate(rows)
    assert gate['attempted'] == 63 and not gate['complete'] and not gate['passed']


def test_duplicate_cell_is_rejected_even_if_global_count_matches():
    rows = actual_rows(); rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match='overrepresented'):a.overhead_gate(rows)


@pytest.mark.parametrize('value', [True, float('nan'), float('inf')])
def test_bad_numeric_arrival_rejected(value):
    rows = actual_rows(); rows[0]['clock']['receipt_elapsed_seconds'] = value
    with pytest.raises(ValueError):a.overhead_gate(rows)


def test_shortened_original_allowance_rejected():
    rows = actual_rows(); rows[0]['clock']['deadline_seconds'] = .9
    with pytest.raises(ValueError, match='one-second'):a.overhead_gate(rows)


def test_saved_replay_byte_proof_is_mandatory():
    rows = replay_rows(); rows[3]['bytes_verified'] = False
    assert not a.replay_overhead_gate(rows)['passed']
    rows = replay_rows()
    for r in rows:
        if r['logging_timing_enabled']:r['elapsed_seconds'] += .00501
    assert not a.replay_overhead_gate(rows)['passed']


def test_off_and_missing_logging_are_absence_not_zero_cost():
    off = a.logging_measurements(row(0, 'frozen_cpu0', 'private_reserves', False))
    assert off['available'] is False and off['total_recording_seconds'] is None
    missing = row(0, 'frozen_cpu0', 'private_reserves', True);missing['logging_profile'] = None
    assert not a.logging_measurements(missing)['available']


def test_total_only_attribution_retains_nonsearch_residual_label():
    measured = a.measurements(row(0, 'frozen_cpu0', 'private_reserves', True))
    log = measured['logging']
    assert math.isclose(log['wholly_in_handler_recording_seconds'], .005)
    assert log['measurement_scope'] == 'two_timestamp_total_recorder_only'
    assert log['component_costs_available'] is False
    assert 'component_distributions' not in log and all('components_seconds' not in record for record in log['records'])
    assert math.isclose(measured['handler_residual_after_measured_logging_seconds'], .295)


def test_failure_logging_separate_from_prevalidation():
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    r['logging_profile']['records'][0]['event'] = 'worker_decision_failed'
    measured = a.logging_measurements(refresh(r))
    assert measured['in_handler_prevalidation_recording_seconds'] == 0
    assert measured['in_handler_failure_evidence_seconds'] == .005


def test_outside_handler_record_not_subtracted_from_handler():
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    r['child_profile']['profiles'][0]['handle_end_ns'] = 90_000_000
    measured = a.measurements(refresh(r))
    assert measured['logging']['total_recording_seconds'] == .005
    assert measured['logging']['wholly_in_handler_recording_seconds'] == 0
    assert measured['handler_residual_after_measured_logging_seconds'] == .09


def test_corrupt_or_fabricated_timestamps_rejected():
    r = row(0, 'frozen_cpu0', 'private_reserves', False)
    r['logging_profile']['records'][0]['timestamps']['start_ns'] = 0
    with pytest.raises(ValueError, match='OFF'):a.logging_measurements(r)
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    r['logging_profile']['records'][0]['timestamps']['end_ns'] = 0
    with pytest.raises(ValueError, match='timestamp'):a.logging_measurements(r)


def test_forged_summary_and_missing_boundaries_never_become_zero():
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    r['logging_profile']['summary']['record_count'] = 0
    with pytest.raises(ValueError, match='summary'):a.logging_measurements(r)
    r = row(0, 'frozen_cpu0', 'private_reserves', True);r['child_profile'] = None
    measured = a.logging_measurements(refresh(r))
    assert measured['handler_bounds_available'] is False
    assert measured['wholly_in_handler_recording_seconds'] is None


def test_missing_logging_cannot_pass_overhead_even_with_timely_response():
    rows = actual_rows();rows[-1]['logging_profile'] = None
    assert not a.overhead_gate(rows)['passed']


def test_before_error_persistence_is_valid_evidence_not_a_passing_request():
    rows = actual_rows();r = rows[-1]
    r['clock'].update(status='worker_error', accepted=False, receipt_elapsed_seconds=.95,
                      detail='ONE_SECOND_INTERNAL_DEADLINE: complete response exceeded 0.900s', result=None)
    r['child_profile']['outcome'] = 'failed_before_successful_emission'
    r['logging_profile']['persistence_disposition'] = 'after_failure_before_error_emission'
    r['logging_profile']['timing_metadata_written_after_response_or_error_emission'] = False
    r['logging_profile']['records'][0]['event'] = 'worker_decision_failed'
    refresh(r)
    evidence = a.logging_measurements(r)
    assert evidence['measurement_accepted'] is True
    assert evidence['in_handler_failure_evidence_seconds'] == .005
    assert a.overhead_gate(rows)['passed'] is False
    measured = a.measurements(r)
    assert measured['failed_request'] and measured['deadline_failed']


def test_persistence_label_cannot_disagree_with_saved_boolean():
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    r['logging_profile']['persistence_disposition'] = 'after_failure_before_error_emission'
    with pytest.raises(ValueError, match='persistence'):a.logging_measurements(r)


def test_complete_analysis_binds_exact_matrix_and_actual_gate():
    rows = actual_rows()
    rows.extend(row(64+i, arm, case, True, stage='diagnostic')
                for i,(arm,case) in enumerate((arm,case) for arm in a.ARM_IDS for case in a.CASE_IDS for _ in range(4)))
    matrix = dict(requests=[r['spec'] for r in rows]);panel = dict(positions=[dict(case_id=p) for p in a.CASE_IDS])
    result = a.analyze(rows, panel, matrix, True)
    assert result['attempted'] == 96 and result['overhead_gate']['attempted'] == 64
    assert result['logging_timing_available'] == 64 and result['core_placement_verdict'] is None
    assert result['statistical_inference'] is False and result['admission_passed'] is False


def test_legacy_component_marker_rejected_instead_of_silently_reused():
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    r['logging_profile']['records'][0]['timestamps']['synced_ns'] = 104_000_000
    with pytest.raises(ValueError):a.logging_measurements(r)


def test_partial_total_is_unavailable_not_zero_or_completed_cost():
    r = row(0, 'frozen_cpu0', 'private_reserves', True)
    record = r['logging_profile']['records'][0]
    record.update(completed=False, error=dict(type='OSError', detail='retained partial marker'))
    record['timestamps']['end_ns'] = None
    r['logging_profile']['status'] = 'log_failure'
    measured = a.logging_measurements(refresh(r))
    assert not measured['available'] and not measured['measurement_accepted']
    assert measured['total_recording_seconds'] is None
    assert measured['wholly_in_handler_recording_seconds'] is None
    assert measured['reason'] == 'incomplete_recorder_span'
