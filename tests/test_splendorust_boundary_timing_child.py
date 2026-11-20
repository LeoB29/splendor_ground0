"""Model-free boundary observer preservation, exact transport and failure checks."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
from types import SimpleNamespace
import pytest

from tools.splendorust_boundary_timing_20261025 import child_profile as cp, worker


def context(tmp_path=None, quota=64):
    return dict(quota=quota, boundary_handle_active=False, component_profiles=[],
        component_backend='qualified_graph64' if quota == 64 else 'unchanged_eager32',
        component_backend_identity='graph' if quota == 64 else None,
        request_started=None, request_read_complete_ns=None, graph_parity_passed=True,
        adapters={}, metadata={'exact': True}, folder=tmp_path)


@pytest.mark.parametrize('quota', [32, 64])
def test_ordinary_handle_uses_two_clocks_no_model_hooks_and_preserves_science(monkeypatch, quota):
    ctx=context(quota=quota); science={'action': {'x': 1}, 'diagnostics': [1, 2]}
    before=deepcopy(science); ticks=iter([100, 200])
    monkeypatch.setattr(worker.time, 'perf_counter_ns', lambda: next(ticks))
    def original(c, request):
        assert c['boundary_handle_active'] is True
        assert 'component_bindings' not in c
        return science
    monkeypatch.setattr(worker, '_GRAPH_HANDLE' if quota==64 else '_EAGER_HANDLE', original)
    assert worker.handle(ctx, {'player': 'search'}) is science and science==before
    profile=ctx['component_profiles'][0]
    assert profile['handle_start_ns']==100 and profile['handle_end_ns']==200
    assert profile['handle_failed'] is False and ctx['boundary_handle_active'] is False
    assert set(profile)=={'schema','pid','clock','measurement_units','request_read_complete_ns',
        'handle_start_ns','handle_end_ns','handle_failed','times_are_diagnostic_not_trusted_deadline_receipts'}


def test_original_failure_has_only_available_handle_boundaries(monkeypatch):
    ctx=context(); error=TimeoutError('unchanged internal failure')
    def original(*args): raise error
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', original)
    with pytest.raises(TimeoutError) as caught: worker.handle(ctx, {'player': 'search'})
    assert caught.value is error and not ctx['boundary_handle_active']
    profile=ctx['component_profiles'][0]
    assert profile['handle_failed'] is True
    assert profile['handle_start_ns'] <= profile['handle_end_ns']
    assert 'partial_counters' not in profile


@pytest.mark.parametrize('quota', [32,64])
def test_actual_checker_recursion_keeps_37_handles_26_rejects_and_15_direct_references(monkeypatch, quota):
    ctx=context(quota=quota); original=worker.frozen.handle; direct=[]
    def eager(c, request):
        if request=={'operation':'entrypoint_checks'}:
            successes=[worker.frozen.handle(c, {'case': i}) for i in range(37)]
            for i in range(26):
                with pytest.raises(ValueError, match='expected rejection'):
                    worker.frozen.handle(c, {'reject': i})
            direct.extend(range(15))
            return {'successful_decisions': len(successes)+len(direct)}
        if 'reject' in request: raise ValueError('expected rejection')
        return {'case':request['case']}
    monkeypatch.setattr(worker, '_EAGER_HANDLE', eager)
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', eager)
    assert worker.handle(ctx, {'operation':'entrypoint_checks'})=={'successful_decisions':52}
    assert len(ctx['component_profiles'])==63 and len(direct)==15
    assert sum(p['handle_failed'] for p in ctx['component_profiles'])==26
    assert worker.frozen.handle is original and not ctx['boundary_handle_active']


def test_record_bound_stops_before_call_and_preserves_prior_records(monkeypatch):
    ctx=context(); ctx['component_profiles']=[{'preserved':True}]*64
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', lambda *a: pytest.fail('over-bound scientific call'))
    with pytest.raises(ValueError, match='count bound'): worker.handle(ctx, {'player':'search'})
    assert len(ctx['component_profiles'])==64 and not ctx['boundary_handle_active']


@pytest.mark.parametrize('quota,backend',[(32,'unchanged_eager32'),(64,'qualified_graph64')])
def test_initialize_calls_exact_original_once_no_rebinding_and_preserves_ready(monkeypatch,tmp_path,quota,backend):
    ctx=context(tmp_path,quota); ctx['frozen_forward']=SimpleNamespace(identity={'same':'graph'})
    calls=[]; metadata=ctx['metadata']
    monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1'); monkeypatch.setenv('STUDY_SEARCH_SIMULATIONS',str(quota))
    monkeypatch.setattr(worker,'_GRAPH_INITIALIZE',lambda:(calls.append('graph') or ctx))
    monkeypatch.setattr(worker,'_EAGER_INITIALIZE',lambda:(calls.append('eager') or ctx))
    saved=[]; monkeypatch.setattr(worker.frozen,'_save',lambda path,value:saved.append((path,value)))
    assert worker.initialize() is ctx and calls==['graph' if quota==64 else 'eager']
    assert ctx['metadata'] is metadata and 'component_bindings' not in ctx
    assert saved[0][0].name=='boundary_timing_identity.json' and saved[0][1]['backend']==backend
    assert 'InstrumentedBindings' not in vars(worker)


def test_close_does_not_wrap_or_alter_graph_cleanup(monkeypatch):
    ctx=context(); calls=[]
    monkeypatch.setattr(worker.graph_worker,'close_context',lambda c:calls.append(c))
    worker.close_context(ctx)
    assert calls==[ctx]


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


def test_malformed_unhashable_player_preserves_original_validation_error(monkeypatch):
    ctx = context(); error = ValueError('request allowlist')
    def fail(*args): raise error
    monkeypatch.setattr(worker, '_GRAPH_HANDLE', fail)
    with pytest.raises(ValueError) as caught: worker.handle(ctx, {'player': []})
    assert caught.value is error and ctx['component_profiles'][0]['handle_failed'] is True


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
    assert saved[-1][0].name == 'boundary_sidecar_error.json'
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



@pytest.mark.parametrize('quota', [32, 64])
def test_original_26_rejection_counterexamples_keep_boundary_records_and_batch_bound(quota):
    from tools.splendorust_search_scale_20261024 import qualification
    from tools.move_evaluation_20261023.qualification import _private
    ctx=context(quota=quota)
    ctx['component_profiles']=[{'prior_success':True} for _ in range(37)]
    evidence=[]
    result=qualification.worker_rejections(lambda request:worker.handle(ctx,request),
                                          _private(), evidence_callback=evidence.append)
    assert len(result)==26 and len(ctx['component_profiles'])==63
    assert all(p['handle_failed'] for p in ctx['component_profiles'][37:])
    assert sum(row['event']=='rejected' for row in evidence)==26
    assert not ctx['boundary_handle_active']


def test_failed_response_flush_retains_prewrite_boundary_without_claiming_flush(monkeypatch,tmp_path):
    ctx,output,sidecars,saved,closed,raw=serve_fixture(monkeypatch,tmp_path,handle=lambda c,p:{'same':True})
    monkeypatch.setattr(worker.time,'perf_counter',lambda:100.)
    class FailOnceFlush(io.StringIO):
        def __init__(self):super().__init__();self.flush_count=0
        def flush(self):
            self.flush_count+=1
            if self.flush_count==2:raise OSError('original response flush failed')
    output=FailOnceFlush();monkeypatch.setattr(worker.sys,'stdout',output)
    assert worker.serve()==1
    value=sidecars[0][0]
    assert value['outcome']=='failed_before_successful_emission'
    assert value['pre_write_ns'] is not None and value['post_flush_ns'] is None
    assert 'original response flush failed' in value['error']
    assert closed==[True]
