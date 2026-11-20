"""Model-free OFF-only observer: no counter capture, exact lifecycle and proofs."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time

import pytest

from tools.splendorust_pool6_core_compare_20261027_02 import host_observer as h


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [dict(role=f'{route}_cpu{cpu}:{role}', pid=10000+i, created=1000.+i)
           for i, (route, cpu, role) in enumerate(
               (route, cpu, role) for route in ('frozen', 'research')
               for cpu in (0, 2) for role in ('python', 'search', 'reference'))]


class Backend:
    def __init__(self):
        self.calls = []
        self.placement_count = 0
        self.fail_placement = None
        self.fail_close = False

    def placement(self):
        self.calls.append('placement')
        self.placement_count += 1
        if self.placement_count == self.fail_placement:
            raise OSError('placement unavailable')
        return dict(pid=1234, created=2000., thread_id=987,
                    process_mask=h.SUPERVISOR_MASK, thread_mask=h.SUPERVISOR_MASK,
                    processor_group=0, priority_class=h.NORMAL_PRIORITY)

    def snapshot(self):
        self.calls.append('snapshot')
        raise AssertionError('Disabled window must never sample counters')

    def context_switches(self):
        self.calls.append('context')
        raise AssertionError('Disabled window must never read context switches')

    def close(self):
        self.calls.append('close')
        return dict(handles_closed=not self.fail_close,
                    errors=[dict(type='OSError', detail='close failed')] if self.fail_close else [])


def window(*, backend=None, delay=0):
    backend = Backend() if backend is None else backend
    observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(7, False, binding=dict(arm_id='research_cpu2', request_id=4))
    start = time.perf_counter_ns()
    if delay:
        time.sleep(delay)
    end = time.perf_counter_ns()
    value = observer.finish_window(token)
    close = observer.close()
    return value, close, backend, dict(request_start_ns=start, receipt_arrival_ns=end)


def test_disabled_window_parks_past_sampling_cadence_and_has_no_counter_reads():
    value, close, backend, clock = window(delay=.12)
    proof = h.validate_receipt(value, 7, False, TARGETS, clock_bounds=clock)
    assert proof['accepted'] and proof['sample_count'] == 0
    assert value['samples'] == [] and value['missed_intervals'] == 0
    assert value['context_before'] is value['context_after'] is value['context_switch_deltas'] is None
    assert value['summary'] is None and value['summary_absence_reason'] == 'disabled'
    assert value['host_cpu_availability'] == h.host_cpu_availability([])
    assert value['host_cpu_availability']['available'] is False
    assert backend.calls == ['placement', 'placement', 'placement', 'close']
    assert h.validate_close(close, TARGETS, expected_windows=1)['thread_joined']
    assert close['os_exit_code'] is None and close['os_child_created'] is False


def test_all_192_prospective_windows_stay_off_and_within_bound():
    backend = Backend(); observer = h.HostObserver(TARGETS, _backend=backend)
    try:
        for sequence in range(192):
            token = observer.start_window(sequence, False, binding=dict(sequence=sequence))
            stamp = time.perf_counter_ns()
            value = observer.finish_window(token)
            assert token['token'] == sequence+1
            assert h.validate_receipt(value, sequence, False, TARGETS,
                clock_bounds=dict(request_start_ns=stamp, receipt_arrival_ns=stamp))['accepted']
    finally:
        close = observer.close()
    assert h.validate_close(close, TARGETS, expected_windows=192)['handles_closed']
    assert close['recorded_bytes'] < h.MAX_BUFFER_BYTES
    assert backend.calls.count('placement') == 1+192*2
    assert set(backend.calls) == {'placement', 'close'}


@pytest.mark.parametrize('enabled', [True, 0, 1, None, 'false', [], {}])
def test_on_or_non_boolean_request_fails_before_any_window_mutation(enabled):
    backend = Backend(); observer = h.HostObserver(TARGETS, _backend=backend)
    try:
        calls = list(backend.calls)
        with pytest.raises(ValueError, match='disabled windows only'):
            observer.start_window(0, enabled)
        assert observer.active is None and observer.next_token == 0 and backend.calls == calls
        token = observer.start_window(0, False)
        value = observer.finish_window(token)
        assert h.validate_receipt(value, 0, False, TARGETS)['accepted']
    finally:
        close = observer.close()
    assert h.validate_close(close, TARGETS, expected_windows=1)['thread_joined']


@pytest.mark.parametrize('field,replacement', [
    ('enabled', True), ('sample_count', 1), ('sample_count', False),
    ('samples', [{}]), ('missed_intervals', 1), ('missed_intervals', False),
    ('context_before', {}), ('context_after', {}), ('context_switch_deltas', []),
    ('summary', {}), ('summary_absence_reason', None),
    ('host_cpu_availability', {}), ('status', 'observed'),
    ('thread_alive_at_finish', False)])
def test_disabled_receipt_cannot_claim_observation_or_forge_availability(field, replacement):
    value, _, _, _ = window()
    value[field] = replacement
    with pytest.raises(ValueError):
        h.validate_receipt(value, 7, False, TARGETS)
    with pytest.raises(ValueError):
        h.verify_forensic_receipt(value, 7, False, TARGETS)


@pytest.mark.parametrize('enabled', [True, 0, None])
def test_saved_validator_requires_literal_false_expectation(enabled):
    value, _, _, _ = window()
    with pytest.raises(ValueError, match='disabled windows only'):
        h.validate_receipt(value, 7, enabled, TARGETS)
    with pytest.raises(ValueError, match='disabled windows only'):
        h.verify_forensic_receipt(value, 7, enabled, TARGETS)


@pytest.mark.parametrize('field', ['request_start_ns', 'receipt_arrival_ns'])
def test_original_clock_bracket_check_remains(field):
    value, _, _, clock = window()
    clock[field] = value['done_ns']+1
    with pytest.raises(ValueError, match='enclose'):
        h.validate_receipt(value, 7, False, TARGETS, clock_bounds=clock)


def test_missing_arrival_does_not_rewrite_original_failed_clock():
    value, _, _, clock = window()
    clock['receipt_arrival_ns'] = None
    original = copy.deepcopy(clock)
    assert h.validate_receipt(value, 7, False, TARGETS, clock_bounds=clock)['accepted']
    assert clock == original  # Observer proof alone never accepts a player receipt.


@pytest.mark.parametrize('change', ['target', 'sequence', 'start_mask', 'end_thread', 'schema'])
def test_off_identity_and_placement_proofs_remain_strict(change):
    value, _, _, _ = window()
    if change == 'target': value['targets'][0]['created'] += 1
    if change == 'sequence': value['sequence'] += 1
    if change == 'start_mask': value['placement_start']['process_mask'] |= 4
    if change == 'end_thread': value['placement_end']['thread_id'] += 1
    if change == 'schema': value['schema'] = 'splendorust_core_compare_host_observer_20261027_v1'
    with pytest.raises(ValueError):
        h.validate_receipt(value, 7, False, TARGETS)


def test_missing_end_placement_stays_failed_and_missing_not_invented():
    backend = Backend(); backend.fail_placement = 3
    value, close, _, clock = window(backend=backend)
    assert value['status'] == 'observer_failed' and value['placement_end'] is None
    proof = h.verify_forensic_receipt(value, 7, False, TARGETS, clock_bounds=clock)
    assert not proof['measurement_accepted'] and 'placement_end' in proof['missing_fields']
    assert h.validate_receipt(value, 7, False, TARGETS, clock_bounds=clock)['accepted'] is False
    assert h.validate_close(close, TARGETS)['prior_observer_error']['detail'] == 'placement unavailable'
    assert set(backend.calls) == {'placement', 'close'}
    value['samples'] = [{}]; value['sample_count'] = 1
    with pytest.raises(ValueError, match='captured counters'):
        h.verify_forensic_receipt(value, 7, False, TARGETS)


@pytest.mark.parametrize('field,replacement', [('thread_joined', False), ('handles_closed', False),
                                            ('completed_windows', 2), ('targets', [])])
def test_close_proof_cannot_be_forged(field, replacement):
    _, close, _, _ = window()
    close[field] = replacement
    with pytest.raises(ValueError):
        h.validate_close(close, TARGETS, expected_windows=1)


def test_real_backend_cleanup_error_is_preserved():
    backend = Backend(); backend.fail_close = True
    value, close, _, _ = window(backend=backend)
    assert h.validate_receipt(value, 7, False, TARGETS)['accepted']
    assert close['status'] == 'observer_failed' and close['errors']
    with pytest.raises(ValueError, match='close failed'):
        h.validate_close(close, TARGETS)


def test_zero_allowance_stop_does_not_extend_clock_and_retains_truthful_join():
    backend = Backend(); observer = h.HostObserver(TARGETS, _backend=backend)
    token = observer.start_window(0, False)
    value = observer.finish_window(token, timeout_seconds=0)
    close = observer.close(timeout_seconds=0)
    assert value['status'] in ('disabled', 'observer_failed')
    assert close['thread_joined'] is not close['thread_alive']
    if not close['thread_joined']:
        assert not close['handles_closed'] and close['errors']
        observer.thread.join(2)
        backend.close()
    assert not any(call in backend.calls for call in ('snapshot', 'context'))


def test_wrong_window_token_cannot_finish_and_active_window_cannot_overlap():
    backend = Backend(); observer = h.HostObserver(TARGETS, _backend=backend)
    try:
        token = observer.start_window(0, False)
        with pytest.raises(ValueError, match='window active'):
            observer.start_window(1, False)
        with pytest.raises(ValueError, match='token differs'):
            observer.finish_window(dict(token, token=token['token']+1))
        observer.finish_window(token)
    finally:
        close = observer.close()
    assert h.validate_close(close, TARGETS, expected_windows=1)['thread_joined']


def test_target_inventory_remains_exactly_bounded():
    backend = Backend()
    with pytest.raises(ValueError, match='target count'):
        h.HostObserver(TARGETS+[dict(role='extra', pid=20000, created=5000.)], _backend=backend)
    assert not backend.calls
    duplicate = copy.deepcopy(TARGETS); duplicate[-1]['pid'] = duplicate[0]['pid']
    with pytest.raises(ValueError, match='Duplicate'):
        h.HostObserver(duplicate, _backend=backend)


def _function_nodes(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}, {
        node.name: {child.name: child for child in node.body if isinstance(child, ast.FunctionDef)}
        for node in tree.body if isinstance(node, ast.ClassDef)}


def test_execution_finish_close_and_native_capability_paths_are_exact_previous_source():
    old_path = ROOT/'tools/splendorust_pool6_core_compare_20261027/host_observer.py'
    new_path = Path(h.__file__)
    old_functions, old_classes = _function_nodes(old_path)
    new_functions, new_classes = _function_nodes(new_path)
    for name in ('__init__', '_run', '_sample', 'finish_window', 'close'):
        assert ast.dump(old_classes['HostObserver'][name]) == ast.dump(new_classes['HostObserver'][name])
    old_start = old_classes['HostObserver']['start_window']
    new_start = copy.deepcopy(new_classes['HostObserver']['start_window'])
    new_start.body.pop(0)  # Only the prospective literal-False guard was added.
    assert ast.dump(old_start) == ast.dump(new_start)
    for name in old_classes['WindowsCounters']:
        assert ast.dump(old_classes['WindowsCounters'][name]) == ast.dump(new_classes['WindowsCounters'][name])
    for name in ('runtime_identity', 'validate_close', '_verify_failed_receipt', 'host_cpu_availability',
                 'verify_legacy_forensic_receipt', 'verify_previous_forensic_receipt'):
        assert ast.dump(old_functions[name]) == ast.dump(new_functions[name])
    for name in ('validate_receipt', 'verify_forensic_receipt'):
        updated = copy.deepcopy(new_functions[name])
        index = 1 if name == 'verify_forensic_receipt' else 0
        updated.body.pop(index)
        assert ast.dump(old_functions[name]) == ast.dump(updated)


def test_placement_copy_is_byte_identical_and_predecessor_pins_remain_unchanged():
    old = ROOT/'tools/splendorust_pool6_core_compare_20261027/placement.py'
    new = ROOT/'tools/splendorust_pool6_core_compare_20261027_02/placement.py'
    assert old.read_bytes() == new.read_bytes()
    receipt = ROOT/'outputs/splendorust_pool6_core_compare_20261027_02/prelaunch/observer_placement_original_pins.json'
    for item in json.loads(receipt.read_text())['sources']:
        raw = (ROOT/item['path']).read_bytes()
        assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']


def test_prior_failed_windows_are_read_only_forensics_not_lean_measurements():
    cases = [
        ('outputs/splendorust_pool6_hostdiag_20261027_01/host_observations/0028.json',
         'beecb6277011a37c826ff1b4181cceae4825482e0b4701023cf74b5614e92bd3', 28,
         h.verify_legacy_forensic_receipt),
        ('outputs/splendorust_pool6_hostdiag_20261027_02/host_observations/0193.json',
         '20f523b2baceaba611044eaed20947e6dc1057c627ee27b206520f118a97bfc1', 193,
         h.verify_previous_forensic_receipt)]
    for relative, digest, sequence, validator in cases:
        path = ROOT/relative; raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        value = json.loads(raw); before = copy.deepcopy(value)
        proof = validator(value, sequence, True, value['targets'])
        assert not proof['measurement_accepted'] and value['status'] == 'observer_failed'
        assert value == before and path.read_bytes() == raw
        with pytest.raises(ValueError, match='disabled windows only'):
            h.validate_receipt(value, sequence, True, value['targets'])
