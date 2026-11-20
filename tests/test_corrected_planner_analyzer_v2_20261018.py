"""Read-only receipt regression tests; never run the analyzer or a player."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from tools import analyze_corrected_planner_qualification_20261018 as original
from tools import analyze_corrected_planner_qualification_20261018_v2 as repaired
from tools.analyze_external_strength_screen_20261018_02 import check_clock


BASE = repaired.ROOT / 'outputs/corrected_planner_qualification_20261018_01'


def real_receipt(role='worker'):
    index = '000' if role == 'worker' else '023'
    folder = 'attempts' if role == 'worker' else 'audits'
    key = 'initialization' if role == 'worker' else 'audit_initialization'
    row = repaired.read(BASE / 'rows' / (index + '.json'))
    value = repaired.read(BASE / folder / index / 'initialization.json')
    return value, row[key], repaired.read(BASE / 'freeze.json'), 'tools.corrected_planner_qualification_20261018.' + role


@pytest.mark.parametrize('role', ['worker', 'audit_worker'])
def test_real_initialization_schema_keeps_all_clock_fields_and_validates_metadata(role):
    value, saved, freeze, module = real_receipt(role)
    assert set(value) - set(saved) == repaired.INITIALIZATION_METADATA | {'result'}
    assert {k: value[k] for k in saved} == saved
    with pytest.raises(ValueError, match='Copied clock'):
        original.require({k: v for k, v in value.items() if k != 'result'} == saved, 'Copied clock receipt differs')
    repaired.check_initialization_copy(value, saved, freeze, module)
    check_clock(value, None, 60, freeze['topology']['selected_logical_processor'])


@pytest.mark.parametrize('field', sorted(repaired.INITIALIZATION_METADATA))
def test_missing_supervisor_field_fails(field):
    value, saved, freeze, module = real_receipt()
    del value[field]
    with pytest.raises(ValueError, match='Missing initialization'):
        repaired.check_initialization_copy(value, saved, freeze, module)


@pytest.mark.parametrize('damage', [
    'worker_module', 'executable', 'extra_argument', 'affinity', 'topology',
    'thread_count', 'missing_thread_variable', 'extra_thread_variable',
    'identity_pid', 'identity_creation', 'boolean_pid', 'extra_identity',
    'telemetry_executable', 'ready_pid', 'missing_python', 'duplicate_python',
    'unknown_metadata', 'copied_clock', 'copied_result',
])
def test_mutated_metadata_or_copied_clock_is_rejected(damage):
    value, saved, freeze, module = real_receipt()
    if damage == 'worker_module': value['command'][-1] += '_different'
    elif damage == 'executable': value['command'][0] += '.different'
    elif damage == 'extra_argument': value['command'].append('--different')
    elif damage == 'affinity': value['affinity'] = [999]
    elif damage == 'topology': value['topology']['selected_core_index'] += 1
    elif damage == 'thread_count': value['thread_environment']['OMP_NUM_THREADS'] = '2'
    elif damage == 'missing_thread_variable': del value['thread_environment']['OMP_NUM_THREADS']
    elif damage == 'extra_thread_variable': value['thread_environment']['EXTRA'] = '1'
    elif damage == 'identity_pid': value['identity']['pid'] += 1
    elif damage == 'identity_creation': value['identity']['created'] += 1
    elif damage == 'boolean_pid': value['identity']['pid'] = True
    elif damage == 'extra_identity': value['identity']['extra'] = True
    elif damage == 'telemetry_executable':
        value['telemetry']['process_tree'][0]['executable'] += '.different'
        saved['telemetry'] = deepcopy(value['telemetry'])
    elif damage == 'ready_pid': value['result']['pid'] += 1
    elif damage == 'missing_python':
        value['telemetry']['process_tree'] = [p for p in value['telemetry']['process_tree'] if p['role'] != 'python']
        saved['telemetry'] = deepcopy(value['telemetry'])
    elif damage == 'duplicate_python':
        value['telemetry']['process_tree'].append(deepcopy(value['telemetry']['process_tree'][0]))
        saved['telemetry'] = deepcopy(value['telemetry'])
    elif damage == 'unknown_metadata': value['unrecognized_supervisor_field'] = True
    elif damage == 'copied_clock': saved['receipt_elapsed_seconds'] += .001
    elif damage == 'copied_result': saved['result'] = value['result']
    with pytest.raises(ValueError):
        repaired.check_initialization_copy(value, saved, freeze, module)


def test_timeout_initialization_uses_same_metadata_contract_without_ready_payload():
    value, saved, freeze, module = real_receipt()
    value.update(status='timeout', accepted=False, received_before_deadline=False,
                 receipt_elapsed_seconds=None, result=None)
    saved = {k: v for k, v in value.items() if k not in repaired.INITIALIZATION_METADATA | {'result'}}
    repaired.check_initialization_copy(value, saved, freeze, module)
    check_clock(value, None, 60, freeze['topology']['selected_logical_processor'])


def test_original_clock_still_rejects_late_response_with_matching_row_copy():
    value, saved, freeze, module = real_receipt()
    value['receipt_elapsed_seconds'] = 61.
    value['elapsed_seconds'] = 62.
    saved.update(receipt_elapsed_seconds=61., elapsed_seconds=62.)
    repaired.check_initialization_copy(value, saved, freeze, module)
    with pytest.raises(ValueError): check_clock(value, None, 60, freeze['topology']['selected_logical_processor'])


def test_core_qualification_and_arithmetic_functions_remain_identical():
    functions = ('cases_for', 'budget_recommendation', 'check_completion_flags', 'quantile', 'description', 'main')
    def nodes(module):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    before, after = nodes(original), nodes(repaired)
    for name in functions: assert before[name] == after[name]


def test_real_request_receipt_schema_has_no_initialization_exclusion():
    row = repaired.read(BASE / 'rows/023.json')
    for folder, field, allowance in (('attempts', 'clock', 120), ('audits', 'audit_clock', 600)):
        value = repaired.read(BASE / folder / '023/request-00000001.json')
        assert not (repaired.INITIALIZATION_METADATA & value.keys())
        assert {k: v for k, v in value.items() if k != 'result'} == row[field]
        check_clock(value, 'request-00000001', allowance, row[field]['telemetry']['affinity'][0])
