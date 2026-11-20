"""Model-free regressions for the new observer-only adapter and its checker."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.splendorust_proxy_calibration_20261113 import adapters as a, qualification as q

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'outputs/splendorust_decision_quality_20261027_02/qualification_rows/0002.json'


@pytest.fixture
def saved():
    return json.loads(SAVED.read_text(encoding='utf-8'))


def test_profile_search_dynamics_and_original_control_identity():
    for quota in (16, *a.QUOTAS):
        p = a.profile(quota)
        assert (p['iterations'], p['world_pool'], p['depth'], p['cpuct']) == (quota, 3, 64, .4)
        assert p['native_binary_sha256'] == a.NATIVE_SHA256
        assert p['internal_budget_seconds'] == 119 and p['complete_budget_seconds'] == 120
        assert p['origin'] == 'generated_under_declared_rules'
    for quota in a.CONTROL_QUOTAS:
        p = a.profile(quota, control=True)
        assert p['native_binary_sha256'] == a.CONTROL_NATIVE_SHA256
    for quota in (True, 0, 1024, 6400):
        with pytest.raises(ValueError):
            a.profile_id(quota)


def test_seed_families_canonical_and_disjoint(saved):
    observer = saved['observer']
    values = {label: a.observer_seed(observer, label) for label in a.SEED_LABELS}
    assert values['A'] == int(digest(observer)[:16], 16)
    assert len(set(values.values())) == 4
    assert all(0 <= value < 2**64 for value in values.values())
    assert values == {label: a.observer_seed(deepcopy(observer), label) for label in a.SEED_LABELS}
    for label in ('C', 'teacher', None, 0):
        with pytest.raises(ValueError):
            a.observer_seed(observer, label)


def test_plan_has_original_controls_high_quota_and_every_seed_route(saved):
    plan = q.build_plan(saved['observer'])
    assert plan['complete_decisions'] == 70 and plan['graph_cases'] == 11
    assert len(plan['parity_requests']) == 18
    assert {x['seed_label'] for x in plan['parity_requests']} == set(a.SEED_LABELS)
    assert {x['quota'] for x in plan['parity_requests'] if x['player'] == 'control'} == set(a.CONTROL_QUOTAS)
    assert plan['original_suite']['actual_complete_decisions']['total'] == 52


def test_quota_dispatch_uses_unchanged_native_request_and_compound_resolver(saved):
    instance = object.__new__(a.ResearchOpponent)
    instance.profile = a.profile(64)
    instance.admission_guard = instance.request_deadline = instance.decision_guard = None
    calls = []
    def exchange(request):
        calls.append(deepcopy(request))
        instance.last_exchange_telemetry = None
        instance.callback_bits = []
        return dict(kind='result', action=request['legal'][0])
    instance.exchange = exchange
    _, trace = instance.choose(saved['observer'], quota=2048, seed_label='R', telemetry=False)
    assert calls[0]['iterations'] == 2048
    assert calls[0]['seed'] == a.observer_seed(saved['observer'], 'R')
    assert set(calls[0]) == {'op', 'observation', 'legal', 'seed', 'iterations'}
    assert trace['profile_id'] == a.profile_id(2048)
    assert instance.active_deadline is None and not instance.capture
    instance.request_deadline = time.perf_counter() - 1
    with pytest.raises(TimeoutError):
        instance.choose(saved['observer'], quota=2048)
    assert len(calls) == 1


def root_fixture(saved, quota=2048):
    observer = saved['observer']
    native, choices = bridge.encode_observation(observer), bridge.main_actions(observer)
    action = q.original.safe_choices(native, choices)[0]
    index = q.original.canonical_index(action)
    visits, values = [0] * 67, [None] * 67
    visits[index], values[index] = quota, .5
    telemetry = dict(visits=visits, q=values, root_value=.5, network_value=.5, selected_edge_value=.5)
    return telemetry, native, choices, action, quota, [0.] * 83


def test_high_quota_exact_work_and_missing_values(saved):
    args = root_fixture(saved)
    checked = q.validate_root_statistics(*args)
    assert checked['visited_edges'] == 1
    assert checked['missing_safe_edges'] > 0
    assert checked['dynamic_fpu_absolute_residual'] == 0
    altered = list(deepcopy(args))
    missing = altered[0]['q'].index(None)
    altered[0]['q'][missing] = 0.
    with pytest.raises(ValueError, match='unvisited Q must be missing'):
        q.validate_root_statistics(*altered)
    altered = list(deepcopy(args))
    altered[0]['visits'][checked['selected_index']] -= 1
    with pytest.raises(ValueError, match='root visits equal completed quota'):
        q.validate_root_statistics(*altered)
    altered = list(deepcopy(args))
    altered[0]['network_value'] = .75
    with pytest.raises(ValueError, match='dynamic-FPU'):
        q.validate_root_statistics(*altered)


def adapt_saved_for_checker_test(saved):
    # Synthetic identity adaptation is confined to this model-free unit test.
    # No historical record or historical target is rewritten/relabelled on disk.
    result = deepcopy(saved['clock']['result'])
    p = a.profile(64)
    result.update(profile_id=p['profile_id'], wrapper_id=p['wrapper_id'], profile_sha256='a' * 64)
    result['diagnostics']['profile_id'] = p['profile_id']
    result['research'].update(schema=a.SCHEMA, internal_seconds=a.INTERNAL_SECONDS)
    kwargs = dict(quota=64, seed_label='A', profile_id=p['profile_id'], profile_sha256='a' * 64,
        wrapper_id=p['wrapper_id'], native_binary_sha256=p['native_binary_sha256'])
    return result, kwargs


def test_saved_accepted_scientific_trace_and_mutated_work_are_checked(saved):
    result, kwargs = adapt_saved_for_checker_test(saved)
    check = q.validate_result(result, saved['observer'], **kwargs)
    assert check['status'] == 'passed' and check['simulations'] == 64
    altered = deepcopy(result)
    altered['diagnostics']['phases'][0]['simulations'] = 63
    with pytest.raises(ValueError, match='actual fixed simulation quota'):
        q.validate_result(altered, saved['observer'], **kwargs)
    altered = deepcopy(result)
    altered['research']['seed_label'] = 'R'
    with pytest.raises(ValueError, match='research seed/clock/native identity'):
        q.validate_result(altered, saved['observer'], **kwargs)
    altered = deepcopy(result)
    altered['research']['phases'][0]['callback_bits'][0]['output_float32_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='root bit hash coupling'):
        q.validate_result(altered, saved['observer'], **kwargs)


def test_root_network_mean_is_not_initial_value(saved):
    result, kwargs = adapt_saved_for_checker_test(saved)
    root = result['research']['phases'][0]['telemetry']
    raw83 = result['diagnostics']['phases'][0]['inference'][0]['output83']
    initial = q.initial_network_win_credit(raw83)
    assert abs(root['network_value'] - initial) > q.dynamic_fpu_rounding_bound(64)
    assert q.validate_result(result, saved['observer'], **kwargs)['status'] == 'passed'


def test_actual_ordinary_rejects_hidden_or_wrong_catalogue_input_before_model(saved):
    context = dict(adapters={'search': None}, qualification_active=False)
    request = dict(player='search', quota=64, seed_label='A', telemetry=True, observer=saved['observer'])
    with pytest.raises(ValueError, match='allowlist'):
        a.ordinary(context, dict(request, snapshot={}))
    altered = deepcopy(request)
    altered['observer']['catalogue_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        a.ordinary(context, altered)
    with pytest.raises(ValueError, match='graph parity'):
        a.handle(dict(graph_parity_passed=False), request)


def test_saved_graph_output_evidence_recomputes_exact_comparisons():
    path = ROOT / 'outputs/splendorust_decision_quality_20261027_02/workers/qualification_research/graph_parity.json'
    plan_path = ROOT / 'outputs/splendorust_inference_opt_20261025_01/graph_parity_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert q.verify_graph_evidence(path, plan['fixtures'])['fixtures'] == 11


def test_expanded_bound_retains_old_limit_and_rejects_unplanned_work():
    from tools.splendorust_decision_quality_20261027_v2.audit import dynamic_fpu_rounding_bound as old_bound
    for quota in (16, 64, 128, 256, 512, 1024):
        assert q.dynamic_fpu_rounding_bound(quota) == old_bound(quota)
    assert q.dynamic_fpu_rounding_bound(2048) > q.dynamic_fpu_rounding_bound(1024)
    for quota in (True, 0, 2049):
        with pytest.raises(ValueError):
            q.dynamic_fpu_rounding_bound(quota)
