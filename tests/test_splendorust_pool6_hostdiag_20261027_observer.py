"""Model-free observer contract, fault preservation, native self-counter probes."""
import ast
import copy
import os
from pathlib import Path
import threading
import time

import psutil
import pytest

from tools.splendorust_pool6_hostdiag_20261027 import host_observer as h


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
                    processes=[dict(identity=target, total_context_switches=count) for target in self.targets])

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
    assert [r['total_context_switches'] for r in value['context_switch_deltas']] == [3, 3]
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
    value['summary']['processes'][0]['total_context_switches'] += 1
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


@pytest.mark.parametrize('change', ['identity', 'negative', 'creation', 'system_reset', 'io_reset', 'fault_reset'])
def test_native_counter_tampering_or_epoch_reset_rejected(change):
    value, _, _, _ = window()
    last = value['samples'][-1]
    if change == 'identity': last['processes'][0]['identity'] = dict(TARGETS[0], created=1.)
    if change == 'negative': last['processes'][0]['private_bytes'] = -1
    if change == 'creation': last['processes'][0]['created_filetime_100ns'] += 1
    if change == 'system_reset': last['system']['kernel_100ns'] = 0
    if change == 'io_reset': last['processes'][0]['io']['read_bytes'] = 0
    if change == 'fault_reset': last['processes'][0]['page_faults'] = 0
    with pytest.raises(ValueError):
        h.validate_receipt(value, 7, True, TARGETS)


@pytest.mark.parametrize('change', ['negative', 'reset', 'identity', 'delta'])
def test_context_switch_invalid_read_is_not_zero_filled(change):
    value, _, _, _ = window()
    last = value['context_after']['processes'][0]
    if change == 'negative': last['total_context_switches'] = -1
    if change == 'reset': last['total_context_switches'] = 0
    if change == 'identity': last['identity'] = dict(TARGETS[0], pid=999)
    if change == 'delta': value['context_switch_deltas'][0]['total_context_switches'] += 1
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


def test_context_reset_preserved_as_failure():
    backend = Backend(); backend.reset_context = True
    value, close, _, _ = window(backend=backend)
    assert value['status'] == 'observer_failed'
    assert 'reset' in value['errors'][0]['detail']
    assert close['prior_error']


def test_sample_reset_preserved_as_failure():
    backend = Backend(); backend.reset_sample = True
    value, _, _, _ = window(backend=backend)
    assert value['status'] == 'observer_failed'
    assert 'reset' in value['errors'][0]['detail']


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
    assert 'sample cap' in value['errors'][-1]['detail']
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
        assert h._context_delta(before, after, [identity])[0]['total_context_switches'] >= 0
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
