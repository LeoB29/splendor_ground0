"""Model-free safety and attribution checks; no owned process is launched."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import psutil
from tools.splendorust_sampler_probe_20261024 import probe as p, idle_worker as helper


@pytest.mark.parametrize('size', [1024, 16384, 65536])
def test_only_fixed_synthetic_sizes_are_accepted(size):
    raw = json.dumps(dict(type='request', request_id='request-1', payload=dict(operation='synthetic_echo', bytes=size))).encode()+b'\n'
    assert helper.parse_echo_request(raw) == ('request-1', size)


@pytest.mark.parametrize('payload', [dict(operation='select', bytes=1024),
    dict(operation='synthetic_echo', bytes=True), dict(operation='synthetic_echo', bytes=65537),
    dict(operation='synthetic_echo', bytes=1024, observer={}), dict(operation='synthetic_echo', bytes=-1),
    dict(operation='synthetic_echo', bytes=1024., script='x'), []])
def test_echo_cannot_become_an_inference_or_arbitrary_payload_endpoint(payload):
    raw = json.dumps(dict(type='request', request_id='request-1', payload=payload)).encode()+b'\n'
    with pytest.raises(ValueError):
        helper.parse_echo_request(raw)


def test_echo_rejects_oversized_unterminated_or_unexpected_envelopes():
    for raw in (b'{}', b'x'*4097+b'\n', b'[]\n', b'{"type":"ready"}\n'):
        with pytest.raises((ValueError, TypeError)):
            helper.parse_echo_request(raw)


def test_instrumentation_restores_bindings_and_delegates_once(monkeypatch):
    calls = []
    def fake_resolve(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return self
    monkeypatch.setattr(Path, 'resolve', fake_resolve)
    original = psutil.Process.cpu_times
    timings = p.Timings()
    value = Path('example')
    with p.instrument(timings):
        assert value.resolve(strict=False) is value
        assert psutil.Process.cpu_times is not original
    assert calls == [(value, (), {'strict': False})]
    assert Path.resolve is fake_resolve
    assert psutil.Process.cpu_times is original
    assert timings.snapshot()['totals']['path.resolve']['calls'] == 1


def test_restoration_and_failed_interval_survive_exceptions(monkeypatch):
    marker = RuntimeError('synthetic failure')
    def failure(self, *args, **kwargs):
        raise marker
    monkeypatch.setattr(Path, 'resolve', failure)
    profiler = p.Timings()
    with pytest.raises(RuntimeError) as caught:
        with p.instrument(profiler):
            Path('example').resolve()
    assert caught.value is marker
    assert Path.resolve is failure
    assert profiler.snapshot()['totals']['path.resolve']['failures'] == 1


def test_exclusive_spans_do_not_double_count_nested_work():
    timings = p.Timings()
    with timings.span('outer'):
        with timings.span('inner'):
            pass
    totals = timings.snapshot()['totals']
    assert totals['outer']['wall_ns'] >= totals['inner']['wall_ns']
    assert totals['outer']['exclusive_wall_ns'] == totals['outer']['wall_ns']-totals['inner']['wall_ns']


def test_raw_readline_mechanism_is_bounded_and_content_preserving():
    result = p.raw_readline_mechanism()
    assert result['payload_bytes'] == result['bytes_returned'] == 1025
    assert result['raw_read_calls'] >= 1
    assert result['model_calls'] == 0


def test_fixed_schedule_has_20_profiled_and_10_baseline_samples():
    schedule = p.fixed_schedule()
    assert len(schedule) == 30
    assert [row['index'] for row in schedule] == list(range(30))
    assert sum(row['mode'] == 'baseline' for row in schedule) == 10
    assert sum(row['mode'] == 'profiled' for row in schedule) == 20


def test_summary_separates_two_tree_overhead_and_modes():
    rows = []
    for scenario in ('one_idle_tree', 'two_idle_trees'):
        for spec in p.fixed_schedule():
            profile = None if spec['mode'] == 'baseline' else dict(totals={'process.children': dict(
                calls=1, wall_ns=10, exclusive_wall_ns=8, cpu_ns=4, exclusive_cpu_ns=3)})
            rows.append(dict(spec, scenario=scenario, wall_ns=100 if scenario == 'one_idle_tree' else 200,
                             profile=profile))
    summary = p.summarize(rows)
    assert summary['scenarios']['one_idle_tree']['modes']['baseline']['count'] == 10
    assert summary['scenarios']['two_idle_trees']['modes']['profiled']['count'] == 20
    assert summary['scenarios']['one_idle_tree']['profiled_labels']['process.children']['calls'] == 20
    with pytest.raises(ValueError):
        p.summarize(rows[:-1])


def test_probe_bound_uses_both_clocks(monkeypatch, tmp_path):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'OUT', tmp_path/'out')
    p.OUT.mkdir()
    monkeypatch.setattr(p.shutil, 'disk_usage', lambda _: SimpleNamespace(free=8*p.GiB))
    monkeypatch.setattr(p.time, 'monotonic', lambda: 100.)
    monkeypatch.setattr(p.time, 'time', lambda: 1000.)
    bound = p.Bound(dict(monotonic_origin=100., unix_origin=1000.))
    assert bound.remaining() == 300.
    bound.guard()
    monkeypatch.setattr(p.time, 'time', lambda: 1300.)
    with pytest.raises(ValueError, match='five-minute'):
        bound.guard()


def test_cleanup_allowance_is_clamped_to_the_probe_not_prior_study(monkeypatch):
    monkeypatch.setattr(p.time, 'monotonic', lambda: 399.)
    monkeypatch.setattr(p.time, 'time', lambda: 1299.)
    result = p.cleanup_owned(None, None, (), monotonic_origin=100., unix_origin=1000.)
    assert result['effective_allowance_seconds'] == 1.
    assert result['errors'] == []


def test_identity_covers_pid_creation_parent_role_path_and_affinity():
    row = dict(pid=1, created=2., parent_pid=3, role='python', executable='python.exe', affinity=[4])
    assert p.identity({'process_tree': [row]}) == [(1, 2., 3, 'python', 'python.exe', (4,))]

