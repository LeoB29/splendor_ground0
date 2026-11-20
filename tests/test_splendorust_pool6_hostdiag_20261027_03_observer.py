"""Model-free observer contract, fault preservation, native self-counter probes."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import psutil
import pytest

from tools.splendorust_pool6_hostdiag_20261027_03 import host_observer as h


TARGETS = [dict(role='frozen:python', pid=10001, created=1000.),
           dict(role='research:search', pid=10002, created=1001.)]


class Backend:
    def __init__(self, targets=TARGETS):
        self.targets = targets
        self.calls = []
        self.counter = 0
        self.context = 0
        self.closed = False
        self.fail_sample = None
        self.fail_context = None
        self.reset_sample = False
        self.reset_context = False
        self.fail_close = False
        self.fail_placement = False

    def placement(self):
        self.calls.append('placement')
        if self.fail_placement:
            raise OSError('placement unavailable')
        return dict(pid=1234, created=2000., thread_id=987,
                    process_mask=h.SUPERVISOR_MASK, thread_mask=h.SUPERVISOR_MASK,
                    processor_group=0, priority_class=h.NORMAL_PRIORITY)

    def snapshot(self):
        self.calls.append('snapshot')
        self.counter += 1
        if self.counter == self.fail_sample:
            raise OSError('sample unavailable')
        start = time.perf_counter_ns()
        count = 1 if self.reset_sample and self.counter > 1 else self.counter*10
        return dict(start_ns=start, end_ns=time.perf_counter_ns(),
                    system=dict(available_physical_bytes=12345, total_physical_bytes=99999,
                                commit_bytes=100, commit_limit_bytes=500, commit_headroom_bytes=400,
                                idle_100ns=count, kernel_100ns=count*2, user_100ns=count),
                    processes=[dict(identity=target, created_filetime_100ns=h.FILETIME_EPOCH+int(target['created']*10_000_000),
                                    kernel_100ns=count, user_100ns=count*2,
                                    working_set_bytes=500, private_bytes=600, page_faults=count,
                                    io={name: count for name, _ in h.IO._fields_}) for target in self.targets])

    def context_switches(self):
        self.calls.append('context')
        self.context += 1
        if self.context == self.fail_context:
            raise OSError('context unsupported')
        start = time.perf_counter_ns()
        count = 0 if self.reset_context and self.context > 1 else self.context*3
        return dict(start_ns=start, end_ns=time.perf_counter_ns(),
                    processes=[dict(identity=target, live_thread_context_switch_sum=count) for target in self.targets])

    def close(self):
        self.calls.append('close')
        self.closed = True
        return dict(handles_closed=not self.fail_close,
                    errors=[dict(type='OSError', detail='close failed')] if self.fail_close else [])


def window(enabled=True, backend=None, delay=0):
    backend = Backend() if backend is None else backend
    observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(7, enabled, binding=dict(route='research', request_id=4))
    start = time.perf_counter_ns()
    if delay:
        time.sleep(delay)
    end = time.perf_counter_ns()
    receipt = observer.finish_window(token)
    close = observer.close()
    return receipt, close, backend, dict(request_start_ns=start, receipt_arrival_ns=end)


def test_enabled_boundaries_and_actual_context_counts():
    value, close, backend, clock = window()
    proof = h.validate_receipt(value, 7, True, TARGETS, clock_bounds=clock)
    assert proof['accepted'] and proof['sample_count'] == 2
    assert [r['net_change'] for r in value['context_switch_deltas']] == [3, 3]
    assert backend.calls.count('context') == 2 and backend.calls.count('snapshot') == 2
    assert h.validate_close(close, TARGETS, expected_windows=1)['thread_joined']
    assert close['os_exit_code'] is None and not close['os_child_created']


def test_disabled_thread_is_parked_with_no_counter_reads():
    value, close, backend, clock = window(False, delay=.12)
    h.validate_receipt(value, 7, False, TARGETS, clock_bounds=clock)
    assert 'snapshot' not in backend.calls and 'context' not in backend.calls
    assert value['samples'] == [] and value['context_before'] is None
    h.validate_close(close, TARGETS)


def test_periodic_timestamps_do_not_claim_exact_intervals():
    value, _, _, clock = window(delay=.13)
    h.validate_receipt(value, 7, True, TARGETS, clock_bounds=clock)
    assert len(value['samples']) >= 3
    middle = value['samples'][1]
    assert middle['kind'] == 'periodic'
    assert middle['scheduled_ns'] == value['ready_ns'] + 100_000_000
    assert middle['lateness_ns'] == max(0, middle['start_ns']-middle['scheduled_ns'])


def test_missing_arrival_preserves_original_failure_window():
    value, _, _, clock = window()
    clock['receipt_arrival_ns'] = None
    assert h.validate_receipt(value, 7, True, TARGETS, clock_bounds=clock)['accepted']


def test_missed_intervals_cannot_be_hidden():
    value, _, _, _ = window(delay=.13)
    value['missed_intervals'] += 1
    with pytest.raises(ValueError, match='Missed interval'):
        h.validate_receipt(value, 7, True, TARGETS)


def test_derived_summary_cannot_change_counter_scope_or_values():
    value, _, _, _ = window()
    assert value['summary']['scope'] == 'observer_bracketing_window_not_exact_move_or_handler'
    value['summary']['processes'][0]['live_thread_context_switch_net_change'] += 1
    with pytest.raises(ValueError, match='derived summary'):
        h.validate_receipt(value, 7, True, TARGETS)


def test_zero_allowance_finish_and_close_preserve_partial_error():
    backend = Backend()
    observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(0, False)
    value = observer.finish_window(token, timeout_seconds=0)
    close = observer.close(timeout_seconds=0)
    assert value['status'] in ('disabled', 'observer_failed')
    assert close['thread_joined'] is not close['thread_alive']
    if not close['thread_joined']:
        assert not close['handles_closed']
        observer.thread.join(2)
        backend.close()


@pytest.mark.parametrize('field', ['request_start_ns', 'receipt_arrival_ns'])
def test_clock_not_bracketed_rejected(field):
    value, _, _, clock = window()
    clock[field] = value['done_ns'] + 1
    with pytest.raises(ValueError, match='enclose'):
        h.validate_receipt(value, 7, True, TARGETS, clock_bounds=clock)


@pytest.mark.parametrize('change', ['identity', 'negative', 'creation', 'io_reset', 'fault_reset'])
def test_native_counter_tampering_or_epoch_reset_rejected(change):
    value, _, _, _ = window()
    last = value['samples'][-1]
    if change == 'identity': last['processes'][0]['identity'] = dict(TARGETS[0], created=1.)
    if change == 'negative': last['processes'][0]['private_bytes'] = -1
    if change == 'creation': last['processes'][0]['created_filetime_100ns'] += 1
    if change == 'io_reset': last['processes'][0]['io']['read_bytes'] = 0
    if change == 'fault_reset': last['processes'][0]['page_faults'] = 0
    with pytest.raises(ValueError):
        h.validate_receipt(value, 7, True, TARGETS)


@pytest.mark.parametrize('change', ['negative', 'reset', 'identity', 'delta'])
def test_context_switch_invalid_read_is_not_zero_filled(change):
    value, _, _, _ = window()
    last = value['context_after']['processes'][0]
    if change == 'negative': last['live_thread_context_switch_sum'] = -1
    if change == 'reset': last['live_thread_context_switch_sum'] = 0
    if change == 'identity': last['identity'] = dict(TARGETS[0], pid=999)
    if change == 'delta': value['context_switch_deltas'][0]['net_change'] += 1
    with pytest.raises(ValueError):
        h.validate_receipt(value, 7, True, TARGETS)


def test_constant_wrong_creation_epoch_cannot_pass_saved_identity_validation():
    value, _, _, _ = window()
    for sample in value['samples']:
        sample['processes'][0]['created_filetime_100ns'] += 20_000_000
    with pytest.raises(ValueError, match='creation epoch'):
        h.validate_receipt(value, 7, True, TARGETS)


def test_counter_failure_preserves_partial_receipt_and_stops_future_windows():
    backend = Backend(); backend.fail_sample = 2
    observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(1, True)
    value = observer.finish_window(token)
    assert value['status'] == 'observer_failed' and len(value['samples']) == 1
    assert value['context_before'] is not None and value['context_after'] is None
    assert value['placement_end'] == value['placement_start']
    assert not h.validate_receipt(value, 1, True, TARGETS)['accepted']
    with pytest.raises(ValueError, match='unavailable'):
        observer.start_window(2, True)
    close = observer.close()
    assert close['prior_error'] and close['thread_joined'] and close['handles_closed']


def test_required_context_capability_failure_prevents_request():
    backend = Backend(); backend.fail_context = 1
    observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(0, True)
    assert token['status'] == 'observer_failed'
    value = observer.finish_window(token)
    assert value['samples'] == [] and value['status'] == 'observer_failed'
    observer.close()


def test_context_aggregate_decrease_is_signed_not_a_reset_or_failure():
    backend = Backend(); backend.reset_context = True
    value, close, _, clock = window(backend=backend)
    assert value['status'] == 'observed' and value['context_switch_deltas'][0]['net_change'] == -3
    assert h.validate_receipt(value, 7, True, TARGETS, clock_bounds=clock)['measurement_accepted']
    assert value['summary']['processes'][0]['interval_total_context_switches'] is None
    assert close['prior_error'] is None


def test_sample_reset_preserved_as_failure():
    backend = Backend(); backend.reset_sample = True
    value, _, _, _ = window(backend=backend)
    assert value['status'] == 'observer_failed'
    assert 'reset' in value['errors'][0]['detail']
    proof = h.verify_forensic_receipt(value, 7, True, TARGETS)
    assert not proof['measurement_accepted'] and proof['native_counter_issues']


@pytest.mark.parametrize('before,after', [(1388, 1387), (9, 9), (10, 20)])
def test_all_aggregate_signs_have_unknown_interval_totals(before, after):
    start = dict(start_ns=1, end_ns=2, processes=[dict(identity=TARGETS[0], live_thread_context_switch_sum=before)])
    end = dict(start_ns=3, end_ns=4, processes=[dict(identity=TARGETS[0], live_thread_context_switch_sum=after)])
    change = h._context_delta(start, end, [TARGETS[0]])[0]
    assert change['net_change'] == after-before
    assert change['interval_total_context_switches'] is None
    assert change['interval_total_context_switches_reason'] == h.CONTEXT_UNAVAILABLE


@pytest.mark.parametrize('key,value', [('net_change', 0), ('interval_total_context_switches', 3),
                                     ('interval_total_context_switches_reason', 'exact_total')])
def test_signed_change_and_unavailable_total_cannot_be_corrupted(key, value):
    receipt, _, _, _ = window()
    receipt['context_switch_deltas'][0][key] = value
    with pytest.raises(ValueError, match='delta differs'):
        h.validate_receipt(receipt, 7, True, TARGETS)


def test_context_api_failure_stays_fatal_but_native_summary_survives():
    backend = Backend(); backend.fail_context = 2
    receipt, close, _, clock = window(backend=backend)
    assert receipt['status'] == 'observer_failed' and receipt['context_after'] is None
    assert receipt['placement_end'] == receipt['placement_start']
    assert receipt['summary']['processes'][0]['cpu_seconds'] > 0
    assert receipt['summary']['processes'][0]['live_thread_context_switch_net_change'] is None
    proof = h.verify_forensic_receipt(receipt, 7, True, TARGETS, clock_bounds=clock)
    assert not proof['measurement_accepted'] and 'context_after' in proof['missing_fields']
    assert h.validate_close(close, TARGETS)['prior_observer_error']


def test_end_placement_is_attempted_even_after_counter_failure():
    backend = Backend(); backend.fail_sample = 2
    original = backend.placement
    def placement():
        if backend.counter >= 2:
            raise OSError('end placement unreadable')
        return original()
    backend.placement = placement
    receipt, _, _, _ = window(backend=backend)
    assert receipt['status'] == 'observer_failed' and receipt['placement_end'] is None
    assert [x['detail'] for x in receipt['errors']] == ['sample unavailable', 'end placement unreadable']
    proof = h.verify_forensic_receipt(receipt, 7, True, TARGETS)
    assert not proof['measurement_accepted'] and 'placement_end' in proof['missing_fields']


def test_forensic_validation_never_turns_infrastructure_failure_into_success():
    backend = Backend(); backend.fail_sample = 2
    receipt, _, _, _ = window(backend=backend)
    assert h.validate_receipt(receipt, 7, True, TARGETS)['accepted'] is False
    receipt['status'] = 'observed'
    with pytest.raises(ValueError):
        h.verify_forensic_receipt(receipt, 7, True, TARGETS)


def test_failed_forensic_identity_corruption_still_rejected():
    backend = Backend(); backend.fail_sample = 2
    receipt, _, _, _ = window(backend=backend)
    receipt['samples'][0]['processes'][0]['created_filetime_100ns'] += 20_000_000
    with pytest.raises(ValueError, match='creation epoch'):
        h.verify_forensic_receipt(receipt, 7, True, TARGETS)


@pytest.mark.parametrize('corruption', ['lateness', 'missed_total'])
def test_present_failed_cadence_metadata_is_still_verified(corruption):
    backend = Backend(); backend.fail_context = 2
    receipt, _, _, _ = window(backend=backend, delay=.13)
    assert h.verify_forensic_receipt(receipt, 7, True, TARGETS)['complete_cadence_equations_verified']
    if corruption == 'lateness':
        receipt['samples'][1]['lateness_ns'] += 1
    else:
        receipt['missed_intervals'] += 1
    with pytest.raises(ValueError, match='Forensic'):
        h.verify_forensic_receipt(receipt, 7, True, TARGETS)


def test_actual_saved_v1_window28_preserved_as_failed_with_missing_end_placement():
    path = Path(__file__).resolve().parents[1] / 'outputs/splendorust_pool6_hostdiag_20261027_01/host_observations/0028.json'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'beecb6277011a37c826ff1b4181cceae4825482e0b4701023cf74b5614e92bd3'
    value = json.loads(raw); original = copy.deepcopy(value)
    proof = h.verify_legacy_forensic_receipt(value, 28, True, value['targets'])
    assert proof['schema_verified'] == h.LEGACY_SCHEMA and not proof['measurement_accepted']
    assert 'placement_end' in proof['missing_fields'] and 'summary' in proof['missing_fields']
    assert value == original and value['status'] == 'observer_failed' and value['placement_end'] is None
    affected = next(x for x in proof['derived_live_thread_net_changes_for_explanation_only'] if x['identity']['pid'] == 23768)
    assert affected['net_change'] == -1 and affected['interval_total_context_switches'] is None
    with pytest.raises(ValueError, match='identity differs'):
        h.validate_receipt(value, 28, True, value['targets'])
    assert path.read_bytes() == raw


def test_actual_saved_negative_values_qualify_only_in_explicit_new_synthetic_receipt():
    path = Path(__file__).resolve().parents[1] / 'outputs/splendorust_pool6_hostdiag_20261027_01/host_observations/0028.json'
    old = json.loads(path.read_bytes())
    class SavedAggregatesBackend(Backend):
        def __init__(self):
            super().__init__(old['targets'])
        def context_switches(self):
            context = super().context_switches()
            side = 'context_before' if self.context == 1 else 'context_after'
            for target, source in zip(context['processes'], old[side]['processes']):
                target['live_thread_context_switch_sum'] = source['total_context_switches']
            return context
    backend = SavedAggregatesBackend()
    observer = h.HostObserver(backend.targets, _backend=backend)
    token = observer.start_window(0, True, binding=dict(test_only='new synthetic window using old raw aggregate values'))
    new = observer.finish_window(token); observer.close()
    assert new['schema'] == h.SCHEMA and new['status'] == 'observed'
    assert h.validate_receipt(new, 0, True, backend.targets)['measurement_accepted']
    assert next(x for x in new['context_switch_deltas'] if x['identity']['pid'] == 23768)['net_change'] == -1
    assert old['status'] == 'observer_failed' and old['placement_end'] is None


def test_active_window_token_lifecycle():
    observer = h.HostObserver(TARGETS, _backend=Backend())
    token = observer.start_window(0, False)
    with pytest.raises(ValueError, match='active'):
        observer.start_window(1, False)
    with pytest.raises(ValueError, match='token'):
        observer.finish_window(dict(token, token=3))
    observer.finish_window(token)
    with pytest.raises(ValueError, match='token'):
        observer.finish_window(token)
    observer.close()
    with pytest.raises(ValueError, match='already closed'):
        observer.close()


def test_persistent_thread_handles_multiple_windows_without_dropped_records():
    backend = Backend(); observer = h.HostObserver(TARGETS, _backend=backend)
    tids = []
    for index, enabled in enumerate((False, True, True, False, True)):
        receipt = observer.finish_window(observer.start_window(index, enabled))
        h.validate_receipt(receipt, index, enabled, TARGETS)
        tids.append(receipt['placement_start']['thread_id'])
    assert len(set(tids)) == 1
    h.validate_close(observer.close(), TARGETS, expected_windows=5)


def test_buffer_limit_failure_is_explicit_not_truncation():
    observer = h.HostObserver(TARGETS, _backend=Backend())
    observer.total_bytes = h.MAX_BUFFER_BYTES
    value = observer.finish_window(observer.start_window(0, True))
    assert value['status'] == 'observer_failed' and value['sample_count'] == 2
    assert value['errors'][-1]['type'] == 'OverflowError'
    close = observer.close()
    with pytest.raises(ValueError, match='bounds'):
        h.validate_close(close, TARGETS)


def test_sample_count_limit_does_not_drop_prior_samples():
    observer = h.HostObserver(TARGETS, _backend=Backend())
    token = observer.start_window(0, True)
    current = observer.active
    current['samples'] = current['samples'] * h.MAX_SAMPLES
    value = observer.finish_window(token)
    assert value['status'] == 'observer_failed' and len(value['samples']) == h.MAX_SAMPLES
    assert any('sample cap' in error['detail'] for error in value['errors'])
    observer.close()


def test_native_handle_close_failure_is_not_clean_exit():
    backend = Backend(); backend.fail_close = True
    _, close, _, _ = window(backend=backend)
    assert close['status'] == 'observer_failed' and not close['handles_closed']
    with pytest.raises(ValueError, match='close failed'):
        h.validate_close(close, TARGETS)


@pytest.mark.parametrize('key,value', [('thread_joined', False), ('thread_alive', True),
                                     ('handles_closed', False), ('os_exit_code', 0)])
def test_false_close_claim_rejected(key, value):
    _, close, _, _ = window()
    close[key] = value
    with pytest.raises(ValueError):
        h.validate_close(close, TARGETS)


def test_thread_placement_failure_closes_handles_before_raise():
    backend = Backend(); backend.fail_placement = True
    with pytest.raises(RuntimeError, match='startup failed'):
        h.HostObserver(TARGETS, _backend=backend)
    assert backend.closed


def test_join_timeout_does_not_close_handles_used_by_running_thread():
    backend = Backend(); original = backend.snapshot
    release, entered = threading.Event(), threading.Event()
    def blocked():
        entered.set(); release.wait(2)
        return original()
    backend.snapshot = blocked
    observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(0, True, timeout_seconds=.01)
    assert token['status'] == 'observer_failed' and entered.is_set()
    value = observer.finish_window(token, timeout_seconds=.01)
    close = observer.close(timeout_seconds=.01)
    assert value['status'] == close['status'] == 'observer_failed'
    assert not close['handles_closed'] and not backend.closed
    release.set(); observer.thread.join(2); backend.close()
    assert not observer.thread.is_alive()


@pytest.mark.parametrize('targets', [[], TARGETS*2, [dict(TARGETS[0], pid=0)],
                                     [dict(TARGETS[0], created=float('nan'))],
                                     [dict(TARGETS[0], extra=1)]])
def test_invalid_identities_rejected_before_thread(targets):
    with pytest.raises(ValueError):
        h.HostObserver(targets, _backend=Backend())


def test_source_has_no_process_launch_model_or_file_writes():
    source = Path(h.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(any(word in str(name) for word in ('torch', 'numpy', 'subprocess', 'adapter', 'worker')) for name in imports)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                   node.func.attr in ('write_text', 'write_bytes', 'Popen', 'setpriority', 'SetThreadAffinityMask')
                   for node in ast.walk(tree))


@pytest.mark.skipif(os.name != 'nt', reason='Windows native self-counter capability')
def test_native_self_counters_and_runtime_owner_identity_without_models():
    identity = dict(role='native_fixture_self', pid=os.getpid(), created=psutil.Process().create_time())
    backend = h.WindowsCounters([identity])
    try:
        before = backend.context_switches()
        first = backend.snapshot(); second = backend.snapshot()
        after = backend.context_switches()
        h._check_snapshots([first, second], [identity])
        assert h._context_delta(before, after, [identity])[0]['interval_total_context_switches'] is None
    finally:
        assert backend.close()['handles_closed']
    evidence = h.runtime_identity()
    assert evidence['capability_preflight_passed']
    assert set(evidence['modules']) == {'kernel32.dll', 'kernelbase.dll', 'ntdll.dll'}
    assert len(evidence['psutil_files']) == 3
    assert all(Path(record['path']).is_file() and len(record['sha256']) == 64 for record in evidence['psutil_files'])


@pytest.mark.skipif(os.name != 'nt', reason='Windows native process identity')
def test_native_creation_mismatch_fails_closed():
    identity = dict(role='self', pid=os.getpid(), created=psutil.Process().create_time()+2)
    with pytest.raises(ValueError, match='identity changed'):
        h.WindowsCounters([identity])


V2_WINDOW193 = Path(__file__).resolve().parents[1] / 'outputs/splendorust_pool6_hostdiag_20261027_02/host_observations/0193.json'
V2_WINDOW193_SHA = '20f523b2baceaba611044eaed20947e6dc1057c627ee27b206520f118a97bfc1'


def projected_v2_failure_for_test_only():
    """Distinct synthetic v3 envelope; no claim that v2 actually completed."""
    raw = V2_WINDOW193.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V2_WINDOW193_SHA
    previous = json.loads(raw)
    projected = copy.deepcopy(previous)
    projected.update(schema=h.SCHEMA, semantics=h.SEMANTICS, status='observed', errors=[])
    projected['binding'] = dict(projected['binding'],
        test_only_projection='v3 numeric policy on exact saved v2 samples; not historical acceptance',
        original_status=previous['status'], original_sha256=V2_WINDOW193_SHA)
    projected['context_switch_deltas'] = h._context_delta(projected['context_before'], projected['context_after'], projected['targets'])
    projected['host_cpu_availability'] = h.host_cpu_availability(projected['samples'])
    projected['summary'] = h.summarize_window(projected)
    projected['summary_absence_reason'] = None
    return raw, previous, projected


def test_actual_v2_failure_stays_failed_with_exact_raw_counter_regression():
    raw, previous, projected = projected_v2_failure_for_test_only()
    proof = h.verify_previous_forensic_receipt(previous, 193, True, previous['targets'])
    assert not proof['measurement_accepted'] and proof['schema_verified'] == h.PREVIOUS_SCHEMA
    assert previous['status'] == 'observer_failed' and previous['summary'] is None
    with pytest.raises(ValueError, match='identity differs'):
        h.validate_receipt(previous, 193, True, previous['targets'])
    assert projected['samples'] == previous['samples']
    assert h.validate_receipt(projected, 193, True, projected['targets'], clock_bounds=dict(
        request_start_ns=378493135214800, receipt_arrival_ns=378493602223200))['measurement_accepted']
    cpu = projected['host_cpu_availability']
    assert cpu['valid'] is False and not cpu['available'] and cpu['busy_fraction'] is None
    assert cpu['offending_pairs'] == [dict(before_index=7, after_index=8,
        before_start_ns=378493810884400, after_start_ns=378493825174500,
        delta_100ns=dict(idle=-4062500, kernel=-3593750, user=156250, total=-3437500),
        reasons=['idle_counter_decreased', 'kernel_counter_decreased'])]
    summary = projected['summary']
    assert summary['host_cpu_availability'] == cpu and summary['system_cpu_busy_fraction'] is None
    assert summary['minimum_available_physical_bytes'] > 0
    assert summary['minimum_commit_headroom_bytes'] > 0
    python = next(p for p in summary['processes'] if p['identity']['role'] == 'frozen:python')
    search = next(p for p in summary['processes'] if p['identity']['role'] == 'frozen:search')
    assert python['cpu_seconds'] == .3125 and search['cpu_seconds'] == .03125 and search['page_faults'] == 77
    assert python['live_thread_context_switch_net_change'] == 346
    assert search['live_thread_context_switch_net_change'] == 133
    assert all(p['interval_total_context_switches'] is None for p in summary['processes'])
    assert V2_WINDOW193.read_bytes() == raw


def cpu_samples(counters):
    return [dict(start_ns=i*10+1, end_ns=i*10+2,
                 system=dict(zip(('idle_100ns', 'kernel_100ns', 'user_100ns'), counter)))
            for i, counter in enumerate(counters)]


def test_every_offending_adjacent_pair_retained_despite_endpoint_recovery():
    samples = cpu_samples([(100, 200, 100), (90, 190, 105), (150, 250, 150),
                           (151, 251, 140), (200, 300, 200)])
    original = copy.deepcopy(samples)
    result = h.host_cpu_availability(samples)
    assert result['valid'] is False and result['available'] is False
    assert [(p['before_index'], p['after_index']) for p in result['offending_pairs']] == [(0, 1), (2, 3)]
    assert result['busy_fraction'] is None and samples == original
    # Overall endpoints would give an ordinary, plausible fraction; forbidden.
    assert h.host_cpu_availability([samples[0], samples[-1]])['available']


@pytest.mark.parametrize('last,reason', [((30, 25, 30), 'idle_delta_exceeds_kernel_delta'),
                                      ((50, 30, 10), 'idle_delta_exceeds_total_delta')])
def test_impossible_numeric_host_utilization_unavailable_even_with_nondecreasing_totals(last, reason):
    result = h.host_cpu_availability(cpu_samples([(10, 20, 10), last]))
    assert result['valid'] is False and not result['available'] and result['busy_fraction'] is None
    assert reason in result['offending_pairs'][0]['reasons']


def test_zero_ticks_and_insufficient_samples_are_explicitly_unavailable():
    result = h.host_cpu_availability(cpu_samples([(10, 20, 10), (10, 20, 10)]))
    assert result['valid'] is True and not result['available'] and result['reason'] == 'no_elapsed_cpu_ticks'
    assert result['offending_pairs'] == []
    for samples in ([], cpu_samples([(10, 20, 10)])):
        result = h.host_cpu_availability(samples)
        assert result['valid'] is None and not result['available'] and result['reason'] == 'insufficient_samples'


def test_valid_host_cpu_computation_uses_kernel_including_idle():
    result = h.host_cpu_availability(cpu_samples([(10, 20, 10), (15, 30, 15)]))
    assert result['valid'] and result['available'] and result['busy_fraction'] == 2/3
    assert result['offending_pairs'] == [] and result['reason'] is None


@pytest.mark.parametrize('invalid', [True, 1.5, '10', None, -1])
def test_host_cpu_scalar_format_remains_fatal(invalid):
    receipt, _, _, _ = window()
    receipt['samples'][-1]['system']['kernel_100ns'] = invalid
    with pytest.raises(ValueError):
        h.host_cpu_availability(receipt['samples'])
    with pytest.raises(ValueError):
        h.validate_receipt(receipt, 7, True, TARGETS)


@pytest.mark.parametrize('bad', ['bool_time', 'negative_time', 'overlap', 'missing_api_value'])
def test_host_cpu_bad_clock_or_missing_api_value_is_not_optional(bad):
    samples = cpu_samples([(10, 20, 10), (15, 30, 15)])
    if bad == 'bool_time': samples[1]['start_ns'] = True
    if bad == 'negative_time': samples[1]['start_ns'] = -1
    if bad == 'overlap': samples[1]['start_ns'] = samples[0]['end_ns']-1
    if bad == 'missing_api_value': del samples[1]['system']['user_100ns']
    with pytest.raises((ValueError, KeyError)):
        h.host_cpu_availability(samples)


@pytest.mark.parametrize('part', ['top_level', 'summary', 'offending_pairs', 'busy_fraction'])
def test_host_cpu_availability_cannot_be_forged(part):
    _, _, projected = projected_v2_failure_for_test_only()
    if part == 'top_level': projected['host_cpu_availability']['available'] = True
    if part == 'summary': projected['summary']['host_cpu_availability']['valid'] = True
    if part == 'offending_pairs': projected['host_cpu_availability']['offending_pairs'] = []
    if part == 'busy_fraction': projected['summary']['system_cpu_busy_fraction'] = .1
    with pytest.raises(ValueError):
        h.validate_receipt(projected, 193, True, projected['targets'])


def test_optional_global_counter_anomaly_does_not_stop_later_windows_or_process_summary():
    class GlobalResetBackend(Backend):
        def snapshot(self):
            row = super().snapshot()
            if self.counter == 2:
                row['system']['kernel_100ns'] = 1
            return row
    observer = h.HostObserver(TARGETS, _backend=GlobalResetBackend())
    first = observer.finish_window(observer.start_window(0, True))
    assert first['status'] == 'observed' and first['errors'] == []
    assert first['host_cpu_availability']['valid'] is False
    assert first['summary']['processes'][0]['cpu_seconds'] > 0
    assert h.validate_receipt(first, 0, True, TARGETS)['measurement_accepted']
    second = observer.finish_window(observer.start_window(1, True))
    assert h.validate_receipt(second, 1, True, TARGETS)['measurement_accepted']
    assert second['host_cpu_availability']['available']
    assert h.validate_close(observer.close(), TARGETS, expected_windows=2)['prior_observer_error'] is None


@pytest.mark.parametrize('bad', ['pid', 'cpu', 'fault', 'io', 'creation_epoch'])
def test_optional_host_cpu_never_masks_invalid_process_evidence(bad):
    _, _, projected = projected_v2_failure_for_test_only()
    process = projected['samples'][-1]['processes'][0]
    if bad == 'pid': process['identity']['pid'] += 1
    if bad == 'cpu': process['kernel_100ns'] = 0
    if bad == 'fault': process['page_faults'] = 0
    if bad == 'io': process['io']['read_bytes'] = 0
    if bad == 'creation_epoch': process['created_filetime_100ns'] += 20_000_000
    with pytest.raises(ValueError):
        h.validate_receipt(projected, 193, True, projected['targets'])
