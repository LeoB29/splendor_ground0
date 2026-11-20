"""Model-free checks of the additive timing worker and trusted-clock delegation."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
import sys
from types import SimpleNamespace

import pytest

from tools.splendorust_timing_20261024 import common, worker
from tools.splendorust_search_scale_20261024 import worker as frozen_worker
from tools.splendorust_search_scale_20261024 import qualification


class Layer:
    def forward(self, value):
        return value


class Adapter:
    def __init__(self, action, profile_id):
        self.action, self.profile_id = action, profile_id
        self.model = SimpleNamespace(**{name: Layer() for name in ('project', 'blocks', 'norm', 'policy', 'value')})
        self.proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO())
        self.request_deadline = None
        self.seen = []
        self.closed = False
    def _infer(self, *args):
        raise AssertionError('No model invocation allowed')
    def exchange(self, *args):
        raise AssertionError('No native invocation allowed')
    def choose(self, observer, *, diagnostic_capture):
        self.seen.append((deepcopy(observer), diagnostic_capture, self.request_deadline))
        return self.action, {'profile_id': self.profile_id, 'unchanged_scientific_marker': [3, 5, 7]}
    def close(self):
        self.closed = True


def request_context(tmp_path, *, profile_enabled=False):
    state = qualification.reference._initial()
    observer = worker.rules.observation_payload(state)
    action = worker.rules.CorrectedEnv().legal_actions(state)[0]
    adapters = {arm: Adapter(action, common.profile_id(quota))
                for arm, quota in [('search', 32), ('reference', 16)]}
    context = {'adapters': adapters, 'folder': tmp_path, 'quota': 32,
               'diagnostic_capture': False, 'request_started': 100.,
               'profile_enabled': profile_enabled}
    return context, {'player': 'search', 'observer': observer}


@pytest.mark.parametrize('mode, expected', [('0', 'baseline'), ('1', 'profiled')])
def test_initialize_binds_mode_and_calls_corrected_runtime_once(monkeypatch, tmp_path, mode, expected):
    calls, saves = [], []
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setenv('STUDY_WORKER_DIR', str(tmp_path / 'outputs' / 'new_worker'))
    monkeypatch.setenv('STUDY_SEARCH_SIMULATIONS', '32')
    monkeypatch.setenv('STUDY_TIMING_PROFILE', mode)
    monkeypatch.setattr(worker, 'verify_runtime_sources', lambda: calls.append('runtime') or {'authenticated': True})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        set_num_threads=lambda count: calls.append(('threads', count)),
        set_num_interop_threads=lambda count: calls.append(('interop', count))))
    adapters = {'search': object(), 'reference': object()}
    def initialize_adapters(folder, quota):
        calls.append(('adapters', quota))
        return adapters, {'identity': 'unchanged'}
    monkeypatch.setattr(worker, 'initialize_adapters', initialize_adapters)
    monkeypatch.setattr(worker, '_save', lambda path, value: saves.append((path.name, value)))
    context = worker.initialize()
    assert calls == ['runtime', ('threads', 1), ('interop', 1), ('adapters', 32)]
    assert context['adapters'] is adapters and context['profile_enabled'] is (mode == '1')
    assert context['metadata']['timing_mode'] == expected
    assert context['metadata']['sources'] == {'authenticated': True}
    assert context['metadata']['ruleset_id'] == worker.RULESET_ID
    assert context['metadata']['catalogue_sha256'] == worker.CATALOGUE_SHA256
    assert [name for name, _ in saves] == (
        ['runtime_sources.json', 'timer_overhead.json', 'player_identity.json']
        if mode == '1' else ['runtime_sources.json', 'player_identity.json'])
    if mode == '1':
        calibration = next(value for name, value in saves if name == 'timer_overhead.json')
        assert calibration['iterations'] == 200
        assert calibration['never_subtract_from_qualification_or_deadline_time']
        assert calibration['does_not_estimate_entire_request_instrumentation_effect']


@pytest.mark.parametrize('profile_enabled', [False, True])
def test_worker_returns_exact_frozen_scientific_response_and_deadline(monkeypatch, tmp_path, profile_enabled):
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.25)
    baseline, request = request_context(tmp_path / 'baseline')
    context, same_request = request_context(tmp_path / 'new', profile_enabled=profile_enabled)
    assert request == same_request
    expected = frozen_worker.handle(baseline, request)
    actual = worker.handle(context, request)
    profile = actual.pop('timing_profile', None)
    assert actual == expected
    assert context['adapters']['search'].seen == baseline['adapters']['search'].seen
    assert context['adapters']['search'].seen[0][2] == 100. + worker.INTERNAL_SECONDS
    assert context['adapters']['search'].request_deadline is None
    assert '_profiler' not in context
    assert 'choose' not in vars(context['adapters']['search'])
    if profile_enabled:
        assert profile['enabled']
        assert {'worker_handle', 'decode_observer', 'legal_actions', 'complete_choose',
                'response_validation_serialization'} <= profile['totals'].keys()
    else:
        assert profile is None


def test_qualification_strips_only_timing_field_and_records_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.25)
    context, request = request_context(tmp_path, profile_enabled=True)
    evidence = []
    context['diagnostic_capture'] = True
    context['evidence_callback'] = evidence.append
    actual = worker.qualification_request(context, request)
    assert 'timing_profile' not in actual
    assert actual['diagnostics']['unchanged_scientific_marker'] == [3, 5, 7]
    assert context['adapters']['search'].seen[0][1] is True
    assert evidence[0]['event'] == 'complete_result_before_worker_validation'
    assert evidence[0]['diagnostics'] == actual['diagnostics']
    assert evidence[1]['event'] == 'qualification_timing_profile'
    assert evidence[1]['profile']['totals']['complete_choose']['count'] == 1


@pytest.mark.parametrize('profile_enabled', [False, True])
def test_worker_never_resets_earlier_request_clock(monkeypatch, tmp_path, profile_enabled):
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: 100.95)
    context, request = request_context(tmp_path, profile_enabled=profile_enabled)
    with pytest.raises(TimeoutError, match='complete response exceeded'):
        worker.handle(context, request)
    assert context['adapters']['search'].seen[0][2] == 100. + worker.INTERNAL_SECONDS
    assert context['adapters']['search'].request_deadline is None
    assert '_profiler' not in context
    if profile_enabled:
        assert context['_failed_timing_profile']['totals']['worker_handle']['failures'] == 1


@pytest.mark.parametrize('profile_enabled', [False, True])
def test_serving_charges_complete_profile_serialization_to_original_clock(monkeypatch, tmp_path, profile_enabled):
    context, payload = request_context(tmp_path, profile_enabled=profile_enabled)
    context['metadata'] = {'mode': profile_enabled}
    monkeypatch.setattr(worker, 'initialize', lambda: context)
    result = {'diagnostics': {'untouched': True}}
    if profile_enabled:
        result['timing_profile'] = {'extra_payload_still_charged': True}
    monkeypatch.setattr(worker, 'handle', lambda context, request: result)
    times = iter([100., 100.901])
    monkeypatch.setattr(worker.time, 'perf_counter', lambda: next(times))
    raw = json.dumps({'type': 'request', 'request_id': 'first', 'payload': payload}).encode() + b'\n'
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw)))
    output = io.StringIO()
    monkeypatch.setattr(worker.sys, 'stdout', output)
    monkeypatch.setitem(sys.modules, 'threadpoolctl', SimpleNamespace(threadpool_limits=lambda **kw: nullcontext()))
    assert worker.serve() == 1
    packets = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [packet['type'] for packet in packets] == ['ready', 'error']
    assert 'response serialization exceeded' in packets[-1]['error']
    assert context['request_started'] == 100.
    assert all(adapter.closed for adapter in context['adapters'].values())
    error = json.loads((tmp_path / 'worker_error.json').read_text())
    assert error['request_id'] == 'first'


@pytest.mark.parametrize('profile_enabled', [False, True])
def test_parent_request_delegates_once_without_clock_or_validation_changes(monkeypatch, profile_enabled):
    instance = object.__new__(common.StudyWorker)
    instance.profile_enabled = profile_enabled
    instance._parent_spans = {'stale_previous_request': {'calls': 1, 'wall_ns': 1}}
    receipt = {'status': 'ok', 'deadline_seconds': 1., 'receipt_elapsed_seconds': .5,
               'result': {'diagnostics': {'unchanged': True}}}
    calls = []
    validator = object()
    def request(self, payload, timeout_seconds, validate=None):
        calls.append((self, payload, timeout_seconds, validate, deepcopy(self._parent_spans)))
        self._parent_spans['current'] = {'calls': 2, 'wall_ns': 5}
        return receipt
    monkeypatch.setattr(common.original.StudyWorker, 'request', request)
    actual = instance.request({'move': 'unchanged'}, 1., validate=validator)
    assert actual is receipt
    assert calls == [(instance, {'move': 'unchanged'}, 1., validator, {})]
    assert actual['deadline_seconds'] == 1. and actual['receipt_elapsed_seconds'] == .5
    assert actual['result'] == {'diagnostics': {'unchanged': True}}
    if profile_enabled:
        assert actual['parent_timing']['spans'] == {'current': {'calls': 2, 'wall_ns': 5}}
    else:
        assert 'parent_timing' not in actual
