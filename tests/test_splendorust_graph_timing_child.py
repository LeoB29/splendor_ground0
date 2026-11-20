"""Model-free preservation, partial evidence, and actual-envelope checks."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from tools.splendorust_graph_timing_20261025 import child_profile as cp, worker


class Clock:
    def __init__(self, step): self.value = 0; self.step = step
    def __call__(self): self.value += self.step; return self.value


def test_bounded_aggregates_preserve_last_inference_and_failed_call():
    p = cp.ChildProfiler(wall_clock=Clock(10), cpu_clock=Clock(1), pid=123)
    for i in range(1000):
        frame = p.begin('inference'); p.end(frame, failed=i == 999)
    p.finish(failed=True)
    value = p.snapshot(adapter=SimpleNamespace(callbacks=[{'batch': 2}]))
    assert value['totals']['inference'] == [1000, 10000, 1000, 10, 1, 10000, 1000]
    counts = value['partial_counters']
    assert counts['inference_calls_returned'] == 999 and counts['inference_calls_failed'] == 1
    assert counts['last_inference_end_ns'] == value['handle_end_ns'] - 10
    assert counts['last_inference_wall_ns'] == 10 and counts['last_inference_cpu_ns'] == 1
    assert counts['last_exchange_callback_hash_count'] == 1
    assert counts['last_exchange_callback_rows'] == 2
    assert len(cp.validate_payload_size(value)) < 4096
    assert set(value['totals']) == set(cp.LABELS)


def test_profile_wraps_current_graph_binding_and_restores_it_before_graph_close():
    eager, graph = [], []
    class Opponent:
        def _infer(self, tokens, batch): eager.append(tokens); return ['eager']
    def graph_forward(self, tokens, batch): graph.append((tokens, batch)); return tokens
    Opponent._infer = graph_forward
    original = SimpleNamespace(Opponent=Opponent)
    adapters = SimpleNamespace(digest=lambda value: value)
    bridge = SimpleNamespace(encode_observation=lambda value: value)
    frozen = SimpleNamespace(decode_observer=lambda value: value)
    adapter = SimpleNamespace(proc=SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO('reply\n')))
    bindings = cp.InstrumentedBindings().install({'search': adapter}, worker_module=frozen,
        adapter_module=adapters, original_module=original, bridge_module=bridge)
    try:
        tokens = [1, 2]
        bindings.active = cp.ChildProfiler()
        assert Opponent()._infer(tokens, 1) is tokens
        p = bindings.active; bindings.active = None; p.finish()
        assert p.snapshot()['totals']['inference'][0] == 1
        assert graph == [(tokens, 1)] and eager == []
    finally:
        bindings.close()
    assert Opponent._infer is graph_forward


def context(tmp_path=None, quota=64):
    return dict(quota=quota, component_bindings=cp.InstrumentedBindings(), component_profiles=[],
        component_backend='qualified_graph64' if quota == 64 else 'unchanged_eager32',
        component_backend_identity='graph' if quota == 64 else None,
        request_started=None, request_read_complete_ns=None, graph_parity_passed=True,
        adapters={}, metadata={'exact': True}, folder=tmp_path)


@pytest.mark.parametrize('quota', [32, 64])
def test_ordinary_handler_returns_same_scientific_object_and_separate_profile(monkeypatch, quota):
    ctx = context(quota=quota); science = {'action': {'x': 1}, 'diagnostics': [1, 2]}
    before = deepcopy(science)
    def handle(context, request):
        assert context['component_bindings'].active is not None
        return science
    monkeypatch.setattr(worker, '_GRAPH_HANDLE' if quota == 64 else '_EAGER_HANDLE', handle)
    assert worker.handle(ctx, {'player': 'search'}) is science
    assert science == before and len(ctx['component_profiles']) == 1
    assert ctx['component_profiles'][0]['handle_failed'] is False
    assert ctx['component_bindings'].active is None


def test_graph_parity_is_required_before_any_profiled_decision(monkeypatch):
    ctx = context(); ctx['graph_parity_passed'] = False
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', lambda *a: pytest.fail('unqualified graph invoked'))
    with pytest.raises(ValueError, match='exact graph parity'): worker.handle(ctx, {'player': 'search'})
    assert ctx['component_profiles'] == []


def test_graph_parity_uses_frozen_handler_and_no_inference_profile(monkeypatch):
    ctx = context(); ctx['graph_parity_passed'] = False
    result = {'passed': True}; calls = []
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', lambda c, r: (calls.append(r) or result))
    request = {'operation': 'graph_parity', 'fixtures': [1]}
    assert worker.handle(ctx, request) is result and calls == [request]
    assert ctx['component_profiles'] == []


@pytest.mark.parametrize('quota', [32, 64])
def test_recursive_checker_routes_every_ordinary_callback_with_graph_gate(monkeypatch, quota):
    ctx = context(quota=quota); original = worker.frozen.handle
    def eager(c, request):
        if request == {'operation': 'entrypoint_checks'}:
            rows = [worker.frozen.handle(c, {'case': i}) for i in range(52)]
            return {'actual': rows}
        assert c['component_bindings'].active is not None
        return {'case': request['case']}
    monkeypatch.setattr(worker, '_EAGER_HANDLE', eager)
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', eager)
    result = worker.handle(ctx, {'operation': 'entrypoint_checks'})
    assert result == {'actual': [{'case': i} for i in range(52)]}
    assert len(ctx['component_profiles']) == 52
    assert worker.frozen.handle is original


@pytest.mark.parametrize('quota', [32, 64])
def test_original_26_rejection_counterexamples_are_profiled_and_fit_actual_batch(quota):
    from tools.splendorust_search_scale_20261024 import qualification
    from tools.move_evaluation_20261023.qualification import _private
    ctx = context(quota=quota)
    # The checker has37 ordinary successes before its26 actual rejections.
    # Its15 direct reference_choose reproductions bypass this worker handle.
    marker = cp.ChildProfiler(); marker.finish()
    ctx['component_profiles'] = [marker.snapshot() for _ in range(37)]
    evidence = []
    result = qualification.worker_rejections(lambda request: worker.handle(ctx, request),
                                            _private(), evidence_callback=evidence.append)
    assert len(result) == 26 and len(ctx['component_profiles']) == 63
    assert all(profile['handle_failed'] for profile in ctx['component_profiles'][37:])
    assert sum(row['event'] == 'rejected' for row in evidence) == 26
    assert all(profile['totals']['inference'][0] == 0 for profile in ctx['component_profiles'])
    assert ctx['component_bindings'].active is None


def test_internal_error_keeps_partial_counter_and_original_exception(monkeypatch):
    ctx = context(); error = TimeoutError('same original .9 failure')
    def fail(c, request):
        profile = c['component_bindings'].active
        frame = profile.begin('inference'); profile.end(frame)
        raise error
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', fail)
    with pytest.raises(TimeoutError) as caught: worker.handle(ctx, {'player': 'search'})
    assert caught.value is error
    profile = ctx['component_profiles'][0]
    assert profile['handle_failed'] is True
    assert profile['partial_counters']['inference_calls_returned'] == 1
    assert ctx['component_bindings'].active is None


def test_malformed_unhashable_player_preserves_original_validation_error(monkeypatch):
    ctx = context(); error = ValueError('request allowlist')
    def fail(*args): raise error
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', fail)
    with pytest.raises(ValueError) as caught: worker.handle(ctx, {'player': []})
    assert caught.value is error and ctx['component_profiles'][0]['handle_failed'] is True


@pytest.mark.parametrize('new_exchange', [False, True])
def test_failure_counters_do_not_reuse_previous_request_callback_list(monkeypatch, new_exchange):
    ctx = context(); stale = [{'batch': 1}] * 65
    adapter = SimpleNamespace(callbacks=stale); ctx['adapters']['search'] = adapter
    def fail(c, request):
        if new_exchange: adapter.callbacks = [{'batch': 2}]
        raise ValueError('current request failed')
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', fail)
    with pytest.raises(ValueError): worker.handle(ctx, {'player': 'search'})
    counters = ctx['component_profiles'][0]['partial_counters']
    assert counters['last_exchange_observed_in_handle'] is new_exchange
    assert counters['last_exchange_callback_hash_count'] == (1 if new_exchange else 0)
    assert counters['last_exchange_callback_rows'] == (2 if new_exchange else 0)
    assert len(stale) == 65
    if not new_exchange: assert adapter.callbacks is stale


def test_recursive_failure_restores_frozen_handler(monkeypatch):
    ctx = context(); before = worker.frozen.handle
    def fail(c, request):
        if request == {'operation': 'entrypoint_checks'}: return worker.frozen.handle(c, {'case': 0})
        raise LookupError('preserve')
    monkeypatch.setattr(worker, '_EAGER_HANDLE', fail); monkeypatch.setattr(worker, '_GRAPH_HANDLE', fail)
    with pytest.raises(LookupError): worker.handle(ctx, {'operation': 'entrypoint_checks'})
    assert worker.frozen.handle is before and len(ctx['component_profiles']) == 1


def test_publication_is_complete_exclusive_and_bounded(tmp_path):
    value = {'schema': cp.SCHEMA, 'small': True}
    target = cp.publish_profile(tmp_path, 'request-00000001', value)
    assert json.loads(target.read_text()) == value
    assert not target.with_suffix('.pending').exists()
    with pytest.raises(FileExistsError): cp.publish_profile(tmp_path, 'request-00000001', {'changed': True})
    assert json.loads(target.read_text()) == value
    with pytest.raises(ValueError, match='identity'): cp.publish_profile(tmp_path, '../escape', value)
    with pytest.raises(ValueError, match='bounded'): cp.publish_profile(tmp_path, 'request-00000002', {'x': 'x' * 8192})
    assert not (tmp_path / 'request-00000002_child_profile.pending').exists()


def serve_fixture(monkeypatch, tmp_path, *, handle, payload=None, quota=64):
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl, 'threadpool_limits', lambda **kwargs: nullcontext())
    ctx = context(tmp_path, quota); closed = []; sidecars = []; saved = []
    packet = {'type': 'request', 'request_id': 'request-00000001',
              'payload': payload if payload is not None else {'player': 'search'}}
    raw = (json.dumps(packet, separators=(',', ':')) + '\n').encode()
    stdout = io.StringIO()
    monkeypatch.setattr(worker, 'initialize', lambda: ctx)
    monkeypatch.setattr(worker, 'handle', handle)
    monkeypatch.setattr(worker, 'close_context', lambda c: closed.append(True))
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(worker.sys, 'stdout', stdout)
    monkeypatch.setattr(worker.frozen, '_save', lambda path, value: saved.append((path, value)))
    def publish(folder, rid, value, **kwargs):
        sidecars.append((value, stdout.getvalue(), kwargs))
    monkeypatch.setattr(worker, 'publish_profile', publish)
    return ctx, stdout, sidecars, saved, closed, raw


def test_success_scientific_bytes_unchanged_and_sidecar_only_after_flush(monkeypatch, tmp_path):
    science = {'action': {'x': 1}, 'diagnostics': {'same': [2]}}
    ctx, output, sidecars, saved, closed, raw = serve_fixture(monkeypatch, tmp_path,
        handle=lambda c, p: science)
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.)
    assert worker.serve() == 0
    expected = json.dumps({'type': 'response', 'request_id': 'request-00000001', 'result': science},
                          separators=(',', ':'), allow_nan=False) + '\n'
    assert output.getvalue().endswith(expected)
    evidence, emitted, options = sidecars[0]
    assert emitted.endswith(expected) and evidence['outcome'] == 'response_flushed'
    assert evidence['pre_write_ns'] <= evidence['post_flush_ns']
    assert evidence['result_sha256'] == cp.digest(science)
    assert evidence['request_raw_sha256'] == hashlib.sha256(raw).hexdigest()
    assert evidence['serialized_packet_utf8_sha256'] == hashlib.sha256(expected.encode()).hexdigest()
    assert 'timing_profile' not in science and not saved and closed == [True]


def test_original_internal_clock_is_not_reset_or_relaxed(monkeypatch, tmp_path):
    starts = []
    ctx, output, sidecars, saved, closed, raw = serve_fixture(monkeypatch, tmp_path,
        handle=lambda c, p: (starts.append(c['request_started']) or {'same': 1}))
    ticks = iter([100., 100.901]); monkeypatch.setattr(worker.time, 'perf_counter', lambda: next(ticks))
    assert worker.serve() == 1 and starts == [100.]
    assert [json.loads(line)['type'] for line in output.getvalue().splitlines()] == ['ready', 'error']
    evidence, emitted, _ = sidecars[0]
    assert evidence['outcome'] == 'failed_before_successful_emission'
    assert evidence['pre_write_ns'] is None and evidence['post_flush_ns'] is None
    assert 'error' not in emitted and '0.900s' in saved[0][1]['error']
    assert closed == [True]


@pytest.mark.parametrize('payload', [{'operation': 'entrypoint_checks'}, {'operation': 'graph_parity', 'fixtures': []}])
def test_only_existing_startup_operations_keep_original_exemption(monkeypatch, tmp_path, payload):
    ctx, output, sidecars, saved, closed, raw = serve_fixture(monkeypatch, tmp_path,
        handle=lambda c, p: {'same': True}, payload=payload)
    ticks = iter([100.]); monkeypatch.setattr(worker.time, 'perf_counter', lambda: next(ticks))
    assert worker.serve() == 0
    assert sidecars[0][2] == {'batch': True}


def test_sidecar_failure_after_response_does_not_emit_second_packet(monkeypatch, tmp_path):
    ctx, output, sidecars, saved, closed, raw = serve_fixture(monkeypatch, tmp_path,
        handle=lambda c, p: {'same': True})
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.)
    def fail(*a, **k): raise OSError('disk failure')
    monkeypatch.setattr(worker, 'publish_profile', fail)
    assert worker.serve() == 1
    assert [json.loads(line)['type'] for line in output.getvalue().splitlines()] == ['ready', 'response']
    assert saved[-1][0].name == 'component_sidecar_error.json'
    assert closed == [True]


def test_disk_failure_cannot_add_second_packet_or_replace_internal_error(monkeypatch, tmp_path):
    error = TimeoutError('original internal deadline')
    def fail_handler(*args): raise error
    ctx, output, sidecars, saved, closed, raw = serve_fixture(monkeypatch, tmp_path, handle=fail_handler)
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.)
    def fail_sidecar(*a, **k): raise OSError('sidecar disk unavailable')
    monkeypatch.setattr(worker, 'publish_profile', fail_sidecar)
    assert worker.serve() == 1
    packets = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [p['type'] for p in packets] == ['ready', 'error']
    assert packets[-1]['error'] == repr(error)
    assert saved[-1][1]['primary_error'] == repr(error)


def test_success_followed_by_total_sidecar_disk_failure_has_only_one_response(monkeypatch, tmp_path):
    ctx, output, sidecars, saved, closed, raw = serve_fixture(monkeypatch, tmp_path,
        handle=lambda c, p: {'same': True})
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.)
    def fail(*a, **k): raise OSError('disk unavailable')
    monkeypatch.setattr(worker, 'publish_profile', fail)
    monkeypatch.setattr(worker.frozen, '_save', fail)
    assert worker.serve() == 1
    assert [json.loads(line)['type'] for line in output.getvalue().splitlines()] == ['ready', 'response']
    assert closed == [True]


@pytest.mark.parametrize('quota,backend', [(32, 'unchanged_eager32'), (64, 'qualified_graph64')])
def test_initialize_uses_exact_backend_once_preserves_readiness(monkeypatch, tmp_path, quota, backend):
    ctx = context(tmp_path, quota); ctx['frozen_forward'] = SimpleNamespace(identity={'same': 'graph'})
    calls = []; identity = ctx['metadata']
    monkeypatch.setenv('STUDY_COMPONENT_PROFILE', '1'); monkeypatch.setenv('STUDY_SEARCH_SIMULATIONS', str(quota))
    monkeypatch.setattr(worker, '_GRAPH_INITIALIZE', lambda: (calls.append('graph') or ctx))
    monkeypatch.setattr(worker, '_EAGER_INITIALIZE', lambda: (calls.append('eager') or ctx))
    bindings = SimpleNamespace(install=lambda adapters: (calls.append('install') or bindings))
    monkeypatch.setattr(worker, 'InstrumentedBindings', lambda: bindings)
    saved = []; monkeypatch.setattr(worker.frozen, '_save', lambda path, value: saved.append(value))
    assert worker.initialize() is ctx
    assert calls == [('graph' if quota == 64 else 'eager'), 'install']
    assert ctx['metadata'] is identity and 'component_timing_mode' not in identity
    assert saved[0]['backend'] == backend


def test_close_restores_profiler_before_graph_identity_guard(monkeypatch):
    order = []; ctx = context()
    ctx['component_bindings'] = SimpleNamespace(close=lambda: order.append('profile'))
    monkeypatch.setattr(worker.graph_worker, 'close_context', lambda c: order.append('graph'))
    worker.close_context(ctx)
    assert order == ['profile', 'graph']
