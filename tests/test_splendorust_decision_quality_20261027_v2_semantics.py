"""Saved-response and arithmetic regressions; no model, search or game calls."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools.splendorust_decision_quality_20261027_v2 import audit as a, analysis


PINS = {
    0: '7bd9653569b0c85bd7525b64ed002f48b2af9fc80707c6d4eb514bfb6dde04d5',
    1: 'd22c939ee28eebc2ec6910e4af65e11757670cc991f34ae3914bec28dd57bd27',
    2: 'd2789734dfe5f8975a4301c8def8731d8187432fe1ccc89552f088513dd9368d',
}


@pytest.fixture(scope='module')
def saved_rows():
    rows = []
    for index, expected in PINS.items():
        path = a.ROOT / f'outputs/splendorust_decision_quality_20261027_01/qualification_rows/{index:04d}.json'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        rows.append(json.loads(raw))
    return rows


def identity(result):
    return {**{key: result[key] for key in ('profile_id', 'profile_sha256', 'wrapper_id')},
            'native_binary_sha256': result['research']['native_binary_sha256']}


def validate_saved_response(row, result=None):
    result = row['clock']['result'] if result is None else result
    return a.validate_research_result(result, row['observer'], quota=64, seed_label='A',
        telemetry_required=row['spec']['telemetry'], **identity(result))


def test_actual_failed_response_checks_semantics_without_reclassifying_history(saved_rows):
    frozen, off, on = saved_rows
    before = deepcopy(saved_rows)
    assert validate_saved_response(off)['status'] == 'passed'
    checked = validate_saved_response(on)
    root = checked['root_statistic_checks'][0]
    assert root['initial_network_win_credit'] == 0.4875371530652046
    assert root['post_search_root_node_value'] == 0.5192456841468811
    assert a.f32(root['dynamic_fpu_expected_node_mean']) == root['post_search_root_node_value']
    assert root['dynamic_fpu_absolute_residual'] <= root['dynamic_fpu_absolute_rounding_bound']
    assert on['clock']['status'] == 'illegal' and on['clock']['accepted'] is False
    assert on['completed'] is False and on['action'] is None and on['trace'] is None
    assert saved_rows == before
    assert a.scientific_core(frozen['clock']['result']) == a.scientific_core(off['clock']['result'])
    assert a.scientific_core(off['clock']['result']) == a.scientific_core(on['clock']['result'])
    assert off['clock']['result']['diagnostics'] == on['clock']['result']['diagnostics']
    assert off['clock']['result']['research']['phases'][0]['callback_bits'] == \
           on['clock']['result']['research']['phases'][0]['callback_bits']


@pytest.mark.parametrize('change', ['initial_in_postsearch_field', 'shift_postsearch', 'wrong_root_mean',
                                   'invent_unvisited_q', 'lost_callback_bit'])
def test_actual_saved_response_corruptions_fail(saved_rows, change):
    row = saved_rows[2]
    result = deepcopy(row['clock']['result'])
    telemetry = result['research']['phases'][0]['telemetry']
    if change == 'initial_in_postsearch_field':
        telemetry['network_value'] = a.f32(a.initial_network_win_credit(
            result['diagnostics']['phases'][0]['inference'][0]['output83']))
    elif change == 'shift_postsearch':
        telemetry['network_value'] = a.f32(telemetry['network_value'] + 8*a.dynamic_fpu_rounding_bound(64))
    elif change == 'wrong_root_mean':
        telemetry['root_value'] = a.f32(telemetry['root_value'] + .01)
    elif change == 'invent_unvisited_q':
        telemetry['q'][next(i for i,n in enumerate(telemetry['visits']) if n == 0)] = 0.
    elif change == 'lost_callback_bit':
        result['research']['phases'][0]['callback_bits'].pop()
    with pytest.raises(ValueError):
        validate_saved_response(row, result)


def synthetic_node(saved_rows, quota, signed, backups):
    observer = saved_rows[2]['observer']
    native = a.bridge.encode_observation(observer)
    choices = a.bridge.main_actions(observer)
    action = saved_rows[2]['clock']['result']['diagnostics']['phases'][0]['action']
    selected = a.original.canonical_index(action)
    raw = [0.]*83
    raw[81] = a.f32(signed)
    initial = a.initial_network_win_credit(raw)
    # Literal native serial recurrence, with an intentionally different backup
    # mean. The former all-zero fixture concealed the semantic mistake.
    node = initial
    for index, value in enumerate(backups):
        node = ((index+1)*node + value)/(index+2)
    mean = a.f32(sum(backups)/quota)
    visits = [0]*67; qs = [None]*67
    visits[selected] = quota; qs[selected] = mean
    telemetry = dict(visits=visits, q=qs, root_value=mean,
                     network_value=a.f32(node), selected_edge_value=mean)
    return telemetry, native, choices, action, raw


@pytest.mark.parametrize('quota', [16, 64, 256, 1024])
@pytest.mark.parametrize('signed,cycle', [(-.75, (.125,.875,.625)), (.875, (0.,.125,.25)),
                                       (-1., (1.,)), (1., (0.,))])
def test_dynamic_fpu_recurrence_distinguishes_initial_and_postsearch(saved_rows, quota, signed, cycle):
    backups = [cycle[i % len(cycle)] for i in range(quota)]
    telemetry, native, choices, action, raw = synthetic_node(saved_rows, quota, signed, backups)
    checked = a.validate_root_statistics(telemetry, native, choices, action, quota, raw)
    assert checked['initial_network_win_credit'] != checked['post_search_root_node_value']
    assert checked['dynamic_fpu_absolute_residual'] <= a.dynamic_fpu_rounding_bound(quota)
    telemetry['network_value'] = a.f32(checked['initial_network_win_credit'])
    with pytest.raises(ValueError, match='dynamic-FPU'):
        a.validate_root_statistics(telemetry, native, choices, action, quota, raw)


def test_declared_rounding_bound_includes_f32_and_f64_terms():
    low, high = a.dynamic_fpu_rounding_bound(16), a.dynamic_fpu_rounding_bound(1024)
    assert 2**-23 < low < high < 2**-23+1e-12
    for bad in (0,1025,True,64.):
        with pytest.raises(ValueError):
            a.dynamic_fpu_rounding_bound(bad)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), 1.0001, -1.0001, .123456789])
def test_initial_head_requires_signed_finite_float32(value):
    raw = [0.]*83; raw[81] = value
    with pytest.raises(ValueError):
        a.initial_network_win_credit(raw)


def test_analysis_labels_saved_native_field_and_derived_initial_separately(saved_rows):
    on = saved_rows[2]
    before = deepcopy(on)
    measured = analysis._measure(on)  # A pure value extraction, not acceptance.
    assert measured['initial_network_win_credit'] == 0.4875371530652046
    assert measured['post_search_root_node_value'] == 0.5192456841468811
    assert measured['upstream_network_value_semantics'] == 'post_search_dynamic_fpu_running_mean'
    assert 'root_network_win_credit' not in measured
    assert on == before and on['clock']['accepted'] is False


def test_original_failed_receipt_is_preserved_without_acceptance(saved_rows):
    row = saved_rows[2]; before = deepcopy(row)
    assert a.check_clock(row['clock'], limit=30.) == .4252405
    assert row == before and row['clock']['status'] == 'illegal' and not row['completed']
