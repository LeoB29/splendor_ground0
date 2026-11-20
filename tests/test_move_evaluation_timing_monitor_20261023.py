"""Countable, model-free tests of selected Python-region instrumentation."""
import sys
import threading
import time

import pytest

from tools.move_evaluation_timing_20261023.monitor import SelectedTimer, TimingIntegrityError


def _rows(measurement):
    return {row['label']: row for row in measurement['rows']}


def _clean(targets):
    monitoring = sys.monitoring
    assert monitoring.get_tool(SelectedTimer.TOOL_ID) is None
    assert monitoring.get_events(SelectedTimer.TOOL_ID) == 0
    for function in targets.values():
        assert monitoring.get_local_events(SelectedTimer.TOOL_ID, function.__code__) == 0
    monitoring.use_tool_id(SelectedTimer.TOOL_ID, 'test_cleanup_inspection')
    try:
        for event in (monitoring.events.PY_START, monitoring.events.PY_RETURN, monitoring.events.PY_UNWIND):
            assert monitoring.register_callback(SelectedTimer.TOOL_ID, event, None) is None
    finally:
        monitoring.free_tool_id(SelectedTimer.TOOL_ID)


def test_32_calls_and_unselected_descendant_time_partition():
    def unselected(value):
        return sum(range(value))
    def selected(value):
        return unselected(value)
    def root():
        values = [selected(i+1) for i in range(32)]
        time.sleep(.002)  # This unselected wait belongs to the root's scope.
        return values
    targets = {'scientific_body': root, 'selected': selected}
    expected = root()
    timer = SelectedTimer(targets)
    actual, measurement = timer.run(root)
    assert actual == expected
    rows = _rows(measurement)
    assert rows['scientific_body']['calls'] == rows['scientific_body']['returns'] == 1
    assert rows['selected']['calls'] == rows['selected']['returns'] == 32
    assert rows['selected']['unwinds'] == 0
    assert rows['scientific_body']['exclusive_ns'] >= 1_000_000
    assert rows['selected']['inclusive_ns'] == rows['selected']['exclusive_ns'] > 0
    assert measurement['events'] == 66
    assert measurement['exclusive_sum_ns'] == measurement['root_region_ns']
    assert measurement['body_ns'] == measurement['root_region_ns']+measurement['unassigned_body_ns']
    assert 0 <= measurement['unassigned_body_ns'] <= max(5_000_000, measurement['body_ns']//100)
    assert measurement['body_seconds'] == measurement['body_ns']/1e9
    assert measurement['balanced'] is True and measurement['callback_count'] >= 66
    assert measurement['callback_body_ns'] >= 0
    assert measurement['interpretation']['instrumentation_overhead_subtracted'] is False
    assert measurement['schema'] == 'selected_python_regions_timing_v1'
    _clean(targets)


def test_selected_nonroot_recursion_counts_and_inclusive_overlap():
    def recurse(depth):
        return 1 if not depth else recurse(depth-1)+1
    def root():
        return recurse(12)
    targets = {'scientific_body': root, 'recursive': recurse}
    actual, measurement = SelectedTimer(targets).run(root)
    row = _rows(measurement)['recursive']
    assert actual == 13 and row['calls'] == row['returns'] == 13
    assert row['inclusive_ns'] > row['exclusive_ns'] > 0
    assert measurement['exclusive_sum_ns'] == measurement['root_region_ns']
    _clean(targets)


def test_selected_exception_caught_by_root_has_unwind_and_balanced_stack():
    def child():
        raise LookupError('expected fixture')
    def root():
        for _ in range(32):
            try:
                child()
            except LookupError:
                pass
        return 'unchanged'
    targets = {'scientific_body': root, 'child': child}
    scientific, measurement = SelectedTimer(targets).run(root)
    assert scientific == 'unchanged'
    assert _rows(measurement)['child']['calls'] == _rows(measurement)['child']['unwinds'] == 32
    assert _rows(measurement)['child']['returns'] == 0
    assert measurement['events'] == 66
    _clean(targets)


def test_unselected_exception_does_not_pop_selected_stack():
    def unselected():
        raise LookupError('not a target')
    def root():
        try:
            unselected()
        except LookupError:
            return 42
    targets = {'scientific_body': root}
    result, measurement = SelectedTimer(targets).run(root)
    assert result == 42 and measurement['events'] == 2
    assert measurement['callback_count'] > measurement['events']
    _clean(targets)


def test_application_exception_identity_is_preserved_and_cleanup_always_runs():
    original = RuntimeError('original application failure')
    def child():
        raise original
    def root():
        return child()
    targets = {'scientific_body': root, 'child': child}
    timer = SelectedTimer(targets)
    with pytest.raises(RuntimeError) as caught:
        timer.run(root)
    assert caught.value is original
    assert timer.last_failure['phase'] == 'application'
    assert timer.last_failure['open_regions'] == []
    rows = {row['label']: row for row in timer.last_failure['rows']}
    assert rows['scientific_body']['unwinds'] == rows['child']['unwinds'] == 1
    _clean(targets)


@pytest.mark.parametrize('kind', ['generator', 'coroutine', 'async_generator'])
def test_generator_and_coroutine_targets_rejected(kind):
    def generator():
        yield 1
    async def coroutine():
        return 1
    async def async_generator():
        yield 1
    functions = locals()
    with pytest.raises(TimingIntegrityError, match='Generator and coroutine'):
        SelectedTimer({'scientific_body': functions[kind]})


def test_tool_id_collision_does_not_release_or_replace_other_tool():
    def root():
        return 1
    monitoring = sys.monitoring
    monitoring.use_tool_id(SelectedTimer.TOOL_ID, 'another_tool')
    try:
        timer = SelectedTimer({'scientific_body': root})
        with pytest.raises(TimingIntegrityError, match='occupied'):
            timer.run(root)
        assert monitoring.get_tool(SelectedTimer.TOOL_ID) == 'another_tool'
        assert timer.last_failure['phase'] == 'installation'
    finally:
        monitoring.free_tool_id(SelectedTimer.TOOL_ID)


def test_duplicate_code_labels_rejected():
    def root():
        return 1
    with pytest.raises(TimingIntegrityError, match='same code object'):
        SelectedTimer({'scientific_body': root, 'alias': root})


def test_manifest_has_unique_code_and_source_identity():
    def root():
        return {'stable': True}
    _, measurement = SelectedTimer({'scientific_body': root}).run(root)
    entry = measurement['manifest'][0]
    assert set(entry) == {'label', 'source_path', 'source_sha256', 'qualname', 'line', 'code_sha256'}
    assert entry['qualname'] == root.__qualname__ and entry['line'] == root.__code__.co_firstlineno
    assert entry['source_path'].replace('\\', '/').endswith('/tests/test_move_evaluation_timing_monitor_20261023.py')
    assert len(entry['source_sha256']) == len(entry['code_sha256']) == 64


def test_bound_method_targets_supported():
    class Example:
        def child(self, x):
            return x+1
        def root(self):
            return self.child(4)
    obj = Example()
    result, measurement = SelectedTimer({'scientific_body': obj.root, 'child': obj.child}).run(obj.root)
    assert result == 5 and _rows(measurement)['child']['calls'] == 1


def test_root_recursion_is_rejected_even_with_balanced_python_calls():
    def root(n):
        return root(n-1) if n else 1
    targets = {'scientific_body': root}
    timer = SelectedTimer(targets)
    with pytest.raises(TimingIntegrityError, match='root'):
        timer.run(root, 1)
    assert timer.last_failure['phase'] == 'integrity_validation'
    _clean(targets)


def test_single_use_even_after_success():
    def root():
        return 1
    timer = SelectedTimer({'scientific_body': root})
    timer.run(root)
    with pytest.raises(TimingIntegrityError, match='single-use'):
        timer.run(root)


def test_wrong_root_callable_rejected_before_install():
    def root():
        return 1
    def other():
        return 1
    with pytest.raises(TimingIntegrityError, match='does not match'):
        SelectedTimer({'scientific_body': root}).run(other)
    _clean({'scientific_body': root})


def test_other_thread_selected_execution_is_rejected():
    result = []
    def selected():
        result.append(1)
    def root():
        thread = threading.Thread(target=selected)
        thread.start()
        thread.join()
        return 'application still returned'
    targets = {'scientific_body': root, 'selected': selected}
    timer = SelectedTimer(targets)
    with pytest.raises(TimingIntegrityError, match='another thread'):
        timer.run(root)
    assert result == [1]
    _clean(targets)


def test_nonmain_construction_rejected():
    errors = []
    def root():
        return None
    def attempt():
        try:
            SelectedTimer({'scientific_body': root})
        except TimingIntegrityError as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=attempt)
    thread.start(); thread.join()
    assert len(errors) == 1 and 'main thread' in errors[0]


def test_integrity_failure_does_not_leave_tool_installed(monkeypatch):
    def root():
        return 42
    targets = {'scientific_body': root}
    timer = SelectedTimer(targets)
    real = timer._on_return
    def dropped_return(code, offset, result):
        if code is not root.__code__:
            real(code, offset, result)
    monkeypatch.setattr(timer, '_on_return', dropped_return)
    with pytest.raises(TimingIntegrityError, match='did not close'):
        timer.run(root)
    assert timer.last_failure['open_regions'] == ['scientific_body']
    _clean(targets)


def test_missing_root_and_nonpython_callable_rejected():
    with pytest.raises(TimingIntegrityError, match='root label'):
        SelectedTimer({'other': test_missing_root_and_nonpython_callable_rejected})
    with pytest.raises(TimingIntegrityError, match='ordinary Python'):
        SelectedTimer({'scientific_body': len})


def test_no_c_events_enabled_during_root():
    observed = []
    def root():
        observed.append(sys.monitoring.get_events(SelectedTimer.TOOL_ID))
        observed.append(sys.monitoring.get_local_events(SelectedTimer.TOOL_ID, root.__code__))
        return 'ok'
    result, _ = SelectedTimer({'scientific_body': root}).run(root)
    assert result == 'ok'
    assert observed == [sys.monitoring.events.PY_UNWIND,
                        sys.monitoring.events.PY_START | sys.monitoring.events.PY_RETURN]


def test_selected_function_that_catches_its_own_exception_returns_normally():
    def selected():
        try:
            raise ValueError('caught inside this region')
        except ValueError:
            return 7
    def root():
        return selected()
    targets = {'scientific_body': root, 'selected': selected}
    result, measurement = SelectedTimer(targets).run(root)
    assert result == 7
    assert _rows(measurement)['selected']['returns'] == 1
    assert _rows(measurement)['selected']['unwinds'] == 0
    _clean(targets)


def test_boundary_gap_and_counter_corruption_are_rejected():
    def root():
        return 1
    timer = SelectedTimer({'scientific_body': root})
    _, measurement = timer.run(root)
    with pytest.raises(TimingIntegrityError, match='boundary discrepancy'):
        timer._measurement(measurement['body_ns']+100_000_000)
    timer._rows['scientific_body']['exclusive_ns'] -= 1
    with pytest.raises(TimingIntegrityError, match='exactly partition'):
        timer._measurement(measurement['body_ns'])


def test_partial_install_failure_cleans_registered_callbacks(monkeypatch):
    def root():
        return 1
    targets = {'scientific_body': root}
    timer = SelectedTimer(targets)
    original = sys.monitoring.set_local_events
    def fail_nonzero(tool, code, events):
        if events:
            raise RuntimeError('synthetic install failure')
        return original(tool, code, events)
    monkeypatch.setattr(sys.monitoring, 'set_local_events', fail_nonzero)
    with pytest.raises(RuntimeError, match='synthetic install failure'):
        timer.run(root)
    assert timer.last_failure['phase'] == 'installation'
    _clean(targets)
