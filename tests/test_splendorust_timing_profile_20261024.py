"""Model-free invariance and accounting tests for opt-in timing wrappers."""
from types import SimpleNamespace

import pytest

from tools.splendorust_timing_20261024.profiling import (
    DecisionProfiler, calibrate_timer_overhead, instrument_adapters,
)


def test_disabled_is_zero_clock_and_returns_same_function():
    def forbidden():
        raise AssertionError('Disabled instrumentation read a clock')
    profiler = DecisionProfiler(False, wall_clock=forbidden, cpu_clock=forbidden)
    function = lambda value: value
    assert profiler.wrap('unused', function) is function
    with profiler.span('unused'):
        pass
    with instrument_adapters({'not_an_adapter': object()}, profiler):
        pass
    assert profiler.snapshot()['interval_count'] == 0


def test_nested_exclusive_accounting_and_interval_bound():
    wall = iter([0, 10, 20, 40, 70, 80, 90])
    cpu = iter([100, 110, 120, 140, 150, 160])
    profiler = DecisionProfiler(True, max_intervals=1,
                                wall_clock=lambda: next(wall), cpu_clock=lambda: next(cpu))
    with profiler.span('outer'):
        with profiler.span('inner'):
            pass
    with profiler.span('inner'):
        pass
    snapshot = profiler.snapshot()
    assert snapshot['totals']['outer']['wall_ns'] == 60
    assert snapshot['totals']['outer']['exclusive_wall_ns'] == 40
    assert snapshot['totals']['outer']['cpu_ns'] == 40
    assert snapshot['totals']['outer']['exclusive_cpu_ns'] == 30
    assert snapshot['totals']['inner']['count'] == 2
    assert snapshot['totals']['inner']['wall_ns'] == 30
    assert snapshot['interval_count'] == 3 and snapshot['omitted_intervals'] == 2
    assert snapshot['intervals'][0]['parent'] == 0


def test_wrapper_preserves_result_identity_arguments_and_exception():
    profiler = DecisionProfiler(True)
    expected = object()
    seen = []
    def original(*args, **kwargs):
        seen.append((args, kwargs))
        return expected
    assert profiler.wrap('delegate', original)(1, option=2) is expected
    assert seen == [((1,), {'option': 2})]
    error = ValueError('preserve this exact failure')
    def failing():
        raise error
    with pytest.raises(ValueError) as captured:
        profiler.wrap('failing', failing)()
    assert captured.value is error
    assert profiler.snapshot()['totals']['failing']['failures'] == 1


class Layer:
    def __init__(self):
        self.calls = []
    def forward(self, value):
        self.calls.append(value)
        return value
    def __call__(self, value):
        return self.forward(value)


class Stream:
    def __init__(self):
        self.writes, self.flushes, self.reads = [], 0, 0
    def write(self, value):
        self.writes.append(value)
        return len(value)
    def flush(self):
        self.flushes += 1
    def readline(self):
        self.reads += 1
        return 'unchanged\n'


class Adapter:
    def __init__(self, model):
        self.model = model
        self.proc = SimpleNamespace(stdin=Stream(), stdout=Stream())
    def _infer(self, value):
        return self.model.blocks(value)
    def exchange(self, value):
        self.proc.stdin.write('request\n')
        self.proc.stdin.flush()
        assert self.proc.stdout.readline() == 'unchanged\n'
        return self._infer(value)
    def choose(self, value):
        return self.exchange(value)


def fake_adapters():
    model = SimpleNamespace(**{name: Layer() for name in ('project', 'blocks', 'norm', 'policy', 'value')})
    return {'search': Adapter(model), 'reference': Adapter(model)}


def test_shared_model_not_double_wrapped_and_all_objects_restore():
    adapters = fake_adapters()
    profiler = DecisionProfiler(True)
    expected = object()
    originals = [(adapter, dict(vars(adapter)), adapter.proc.stdin, adapter.proc.stdout)
                 for adapter in adapters.values()]
    with instrument_adapters(adapters, profiler):
        assert adapters['search'].choose(expected) is expected
        assert adapters['reference'].choose(expected) is expected
    totals = profiler.snapshot()['totals']
    assert totals['complete_choose']['count'] == 2
    assert totals['model_blocks']['count'] == 2
    assert totals['native_read_wait']['count'] == 2
    assert totals['native_write']['count'] == 2
    for adapter, before, stdin, stdout in originals:
        assert vars(adapter) == before
        assert adapter.proc.stdin is stdin and adapter.proc.stdout is stdout
        assert stdin.writes == ['request\n'] and stdin.flushes == 1
        assert stdout.reads == 1
    assert 'forward' not in vars(adapters['search'].model.blocks)


def test_instrumentation_restores_module_functions_on_exception_and_rejects_nesting():
    from tools.splendorust_search_scale_20261024 import adapters as frozen
    from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
    before = (frozen.digest, Opponent._infer)
    adapters = fake_adapters()
    with pytest.raises(RuntimeError, match='forced failure'):
        with instrument_adapters(adapters, DecisionProfiler(True)):
            with pytest.raises(RuntimeError, match='Nested or concurrent'):
                with instrument_adapters(adapters, DecisionProfiler(True)):
                    pass
            raise RuntimeError('forced failure')
    assert (frozen.digest, Opponent._infer) == before
    # The rejected nested scope and exception must both leave the lock usable.
    with instrument_adapters(adapters, DecisionProfiler(True)):
        pass


def test_failed_partial_setup_restores_earlier_patches():
    from tools.splendorust_search_scale_20261024 import adapters as frozen
    before = frozen.digest
    with pytest.raises(AttributeError):
        with instrument_adapters({'broken': object()}, DecisionProfiler(True)):
            pass
    assert frozen.digest is before
    with instrument_adapters(fake_adapters(), DecisionProfiler(True)):
        pass


def test_calibration_is_bounded_and_not_deadline_adjustment():
    with pytest.raises(ValueError):
        calibrate_timer_overhead(2001)
    result = calibrate_timer_overhead(3)
    assert result['iterations'] == 3
    assert result['total_wall_ns'] > 0
    assert result['never_subtract_from_qualification_or_deadline_time']
    assert result['does_not_estimate_entire_request_instrumentation_effect']
