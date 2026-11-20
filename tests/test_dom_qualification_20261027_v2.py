"""Model-free qualification inventory and corruption tests; no learned forwards."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.dom_screen_20261027_v2 import adapter, qualification as q, reference
from tools.external_strength_screen_20261018 import rules


@pytest.fixture(scope='module')
def plan():
    return q.build_plan()


def synthetic_output(_features):
    return np.zeros(77, dtype=np.float64)


@pytest.fixture(scope='module')
def rows(plan):
    result = []
    for spec in plan['requests']:
        chosen = adapter.decide(spec['observer'], synthetic_output)
        result.append({**deepcopy(spec), 'clock': {'status': 'ok', 'result': chosen,
                      'receipt_elapsed_seconds': .125, 'received_before_deadline': True},
                      'action': chosen['action'], 'after': q._step(spec['before'], chosen['action'])})
    return result


def test_fixed_inventory_and_truthful_hidden_variants(plan):
    checked = q.validate_plan(plan)
    assert checked == {'status': 'plan_validated', 'cases': 64, 'ordinary_timed_requests': 256,
                       'distinct_hidden_pairs': 60}
    assert {c['case_id'] for c in plan['cases'] if not c['hidden_state_changed']} == {
        'forced_pass', 'forced_pass_swapped', 'single_buy', 'single_buy_swapped'}
    assert plan['bundled_startup']['complete_handler_decisions'] == 64
    assert plan['bundled_startup']['reference_decisions'] == 64
    assert plan['bundled_startup']['reference_forwards_maximum'] == 320
    assert plan['numeric_tolerances'] == {'features': {'atol': 1e-7, 'rtol': 0.0},
                                          'logits': {'atol': 1e-6, 'rtol': 1e-6}}


def test_all_77_enabled_witnesses_and_independent_semantics(plan):
    result = q.validate_model_free_contract(plan)
    assert result['enabled_index_witnesses'] == result['independent_mapping_transitions'] == 77
    assert result['static_mapping']['cards'] == 90 and result['static_mapping']['nobles'] == 10
    assert [r['winner'] for r in result['terminal_ties']] == [None, 1]
    assert result['actual_player_qualified'] is False


def test_all_case_source_features_and_masks_exact_without_model(plan):
    for case in plan['cases']:
        prepared = adapter.prepare(case['observer'])
        assert np.array_equal(prepared['features'], reference.encode(case['observer']))
        chosen = adapter.decide(case['observer'], synthetic_output)
        for phase in chosen['diagnostics']['phases']:
            assert np.array_equal(phase['features'], reference.encode_phase(phase['context']))


@pytest.mark.parametrize('mutation', ['order', 'private', 'hash', 'count', 'tolerance', 'missing_mapping'])
def test_frozen_plan_corruptions_rejected(plan, mutation):
    changed = deepcopy(plan)
    if mutation == 'order': changed['requests'][0], changed['requests'][1] = changed['requests'][1], changed['requests'][0]
    elif mutation == 'private': changed['cases'][0]['hidden_variant']['state']['bank_tokens']['white'] += 1
    elif mutation == 'hash': changed['cases'][0]['observer_sha256'] = '0'*64
    elif mutation == 'count': changed['timed_request_count'] -= 1
    elif mutation == 'tolerance': changed['numeric_tolerances']['logits']['atol'] = 1e-3
    elif mutation == 'missing_mapping': changed['mapping_witnesses'].pop()
    with pytest.raises(ValueError): q.validate_plan(changed)


def test_mapping_transition_corruption_rejected(plan):
    changed = deepcopy(plan)
    changed['mapping_witnesses'][0]['action']['action_type'] = 'PASS'
    with pytest.raises(ValueError, match='mapping witness action changed'):
        q.validate_model_free_contract(changed)


def test_complete_synthetic_panel_and_exact_boundary(rows, plan):
    result = q.validate_qualification_rows(rows, plan)
    assert result['qualified'] is True and result['requests'] == 256 and result['repeat_groups'] == 64
    changed = deepcopy(rows)
    changed[0]['clock']['receipt_elapsed_seconds'] = .7
    result = q.validate_qualification_rows(changed, plan)
    assert result['qualified'] is False and result['timing_margin_failures'][0]['index'] == 0


@pytest.mark.parametrize('mutation', ['timeout', 'late_receipt', 'no_receipt', 'negative_time', 'nan',
                                     'order', 'incomplete', 'action', 'successor'])
def test_timed_receipt_corruption_rejected(rows, plan, mutation):
    changed = deepcopy(rows)
    if mutation == 'timeout': changed[0]['clock']['status'] = 'timeout'
    elif mutation == 'late_receipt': changed[0]['clock']['received_before_deadline'] = False
    elif mutation == 'no_receipt': changed[0]['clock']['receipt_elapsed_seconds'] = None
    elif mutation == 'negative_time': changed[0]['clock']['receipt_elapsed_seconds'] = -1
    elif mutation == 'nan': changed[0]['clock']['receipt_elapsed_seconds'] = float('nan')
    elif mutation == 'order': changed[0]['index'] = 1
    elif mutation == 'incomplete': changed.pop()
    elif mutation == 'action': changed[0]['action'] = {}
    elif mutation == 'successor': changed[0]['after']['state']['turn_index'] += 1
    with pytest.raises(ValueError): q.validate_qualification_rows(changed, plan)


def test_hidden_repeat_scientific_change_rejected(rows, plan):
    changed = deepcopy(rows)
    logits = np.zeros(77, dtype=np.float64)
    logits[:] = .25  # Same selected move, different scientific output must still reject.
    changed[1]['clock']['result'] = adapter.decide(changed[1]['observer'], lambda _: logits)
    with pytest.raises(ValueError, match='hidden/referee/repetition dependence'):
        q.validate_qualification_rows(changed, plan)


@pytest.mark.parametrize('actual', [[float('nan')], [float('inf')], [True], [], [.001]])
def test_numeric_parity_rejects_bad_data(actual):
    with pytest.raises(ValueError):
        q.assert_numeric_close(actual, [0.], atol=1e-6, rtol=1e-6, size=1, label='probe')


def _callback(context):
    from tools.dom_screen_20261027_v2 import worker
    return lambda payload: worker.handle(context, payload)


def _context(plan, folder):
    return {'qualification_plan': plan, 'folder': folder,
            'policy': SimpleNamespace(predict=synthetic_output),
            'diagnostic_capture': True, 'component_profiles': []}


def test_actual_handler_bundle_with_mocked_model_functions(plan, tmp_path, monkeypatch):
    # Synthetic zeros are test data; no artifact is loaded or neural layer executed.
    monkeypatch.setattr(reference, 'load_policy', lambda: object())
    monkeypatch.setattr(reference, 'predict', lambda features, policy=None: synthetic_output(features))
    context = _context(plan, tmp_path)
    result = q.run_entrypoint_checks(context, _callback(context))
    assert result['status'] == 'dom_entrypoint_checks_passed'
    assert result['complete_handler_decisions'] == result['reference_decisions'] == 64
    assert 64 <= result['reference_forwards'] <= 320
    assert result['rejection_count'] == 13
    events = [json.loads(line) for line in (tmp_path/'dom_entrypoint_forensics.jsonl').read_text().splitlines()]
    assert len(events) == 64*2 + result['reference_forwards'] + 13*2
    assert [row['event'] for row in events[:3]] == ['started', 'returned', 'reference_returned']
    assert not context['component_profiles']
    def forbidden(*args, **kwargs): raise AssertionError('Receipt reread must not call a model')
    monkeypatch.setattr(reference, 'load_policy', forbidden)
    monkeypatch.setattr(reference, 'predict', forbidden)
    verified = q.validate_entrypoint_result(result, plan, tmp_path/'dom_entrypoint_forensics.jsonl')
    assert verified['saved_reference_forwards_checked'] == result['reference_forwards'] and verified['new_model_forwards'] == 0
    for field, value in [('complete_handler_decisions', 63), ('reference_decisions', 63), ('reference_forwards', 63),
                         ('plan_sha256', '0'*64), ('rejection_count', 12)]:
        corrupt = deepcopy(result)
        corrupt[field] = value
        with pytest.raises(ValueError):
            q.validate_entrypoint_result(corrupt, plan, tmp_path/'dom_entrypoint_forensics.jsonl')
    corrupt = deepcopy(result)
    corrupt['forensics']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='forensic binding'):
        q.validate_entrypoint_result(corrupt, plan, tmp_path/'dom_entrypoint_forensics.jsonl')
    # Even recomputed transport metadata cannot make corrupted source values valid.
    events[2]['source']['logits'][0] = .125
    changed_bytes = ''.join(json.dumps(event) + '\n' for event in events).encode()
    (tmp_path/'dom_entrypoint_forensics.jsonl').write_bytes(changed_bytes)
    corrupt = deepcopy(result)
    corrupt['forensics'].update(bytes=len(changed_bytes), sha256=q.hashlib.sha256(changed_bytes).hexdigest())
    with pytest.raises(ValueError, match='actual source phase logits differs'):
        q.validate_entrypoint_result(corrupt, plan, tmp_path/'dom_entrypoint_forensics.jsonl')


def test_near_tie_source_disagreement_fails_despite_numeric_tolerance(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(reference, 'load_policy', lambda: object())
    supported = np.flatnonzero(adapter.prepare(plan['cases'][0]['observer'])['legal_mask'])
    assert len(supported) >= 2
    reference_logits = np.zeros(77, dtype=np.float64)
    reference_logits[supported[1]] = 1e-7
    monkeypatch.setattr(reference, 'predict', lambda features, policy=None: reference_logits)
    context = _context(plan, tmp_path)
    with pytest.raises(ValueError, match='exact masked first-maximum'):
        q.run_entrypoint_checks(context, _callback(context))
    events = [json.loads(line) for line in (tmp_path/'dom_entrypoint_forensics.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['started', 'returned', 'reference_returned', 'entrypoint_failed']
    assert events[1]['result']['diagnostics']['logits'] == [0.]*77


def test_handler_failure_keeps_started_forensic_request(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(reference, 'load_policy', lambda: object())
    def fail(_request): raise ValueError('synthetic handler failure')
    with pytest.raises(ValueError, match='synthetic handler failure'):
        q.run_entrypoint_checks(_context(plan, tmp_path), fail)
    events = [json.loads(line) for line in (tmp_path/'dom_entrypoint_forensics.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events] == ['started', 'entrypoint_failed']


def test_forensic_no_overwrite(tmp_path):
    first = q._Forensics(tmp_path)
    first.write({'event': 'preserve'})
    first.close()
    with pytest.raises(FileExistsError): q._Forensics(tmp_path)


@pytest.mark.parametrize('field,value', [('deadline_seconds',301),('reference_forwards_maximum',321),
                                        ('ordinary_timed_requests',1),('forensic_max_records',513)])
def test_startup_resource_bounds_are_part_of_plan(plan,field,value):
    changed=deepcopy(plan);changed['bundled_startup'][field]=value
    with pytest.raises(ValueError,match='bundled workload/forensic bounds'):q.validate_plan(changed)


def test_all_auxiliary_tokens_nobles_and_atomic_index_semantics(plan):
    for index in range(60,76):
        witness=plan['mapping_witnesses'][index]
        result=q._witness_result(witness['observer'],index)
        q._assert_phase_semantics(witness['before'],result)
        corrupt=deepcopy(result)
        selected=next(p for p in corrupt['diagnostics']['phases'] if p['selected_index']==index)
        selected['selected_index']=60+(index-60+1)%6 if index<66 else 66+(index-66+1)%10
        with pytest.raises(ValueError,match='source (discard|noble)'):
            q._assert_phase_semantics(witness['before'],corrupt)


def test_auxiliary_reference_failure_preserves_completed_phase_prefix(plan,tmp_path,monkeypatch):
    monkeypatch.setattr(reference,'load_policy',lambda:object())
    calls=0
    target_calls=None
    for case in plan['cases']:
        length=len(adapter.decide(case['observer'],synthetic_output)['diagnostics']['phases'])
        if length>1:
            target_calls=calls+2
            break
        calls+=length
    assert target_calls is not None
    calls=0
    def source(features,policy=None):
        nonlocal calls
        calls+=1
        if calls==target_calls:raise ValueError('synthetic auxiliary reference failure')
        return synthetic_output(features)
    monkeypatch.setattr(reference,'predict',source)
    with pytest.raises(ValueError,match='synthetic auxiliary reference failure'):
        q.run_entrypoint_checks(_context(plan,tmp_path),_callback(_context(plan,tmp_path)))
    events=[json.loads(line) for line in (tmp_path/'dom_entrypoint_forensics.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='entrypoint_failed'
    assert events[-2]['event']=='reference_returned' and events[-2]['phase_index']==0
    assert events[-3]['event']=='returned' and len(events[-3]['result']['diagnostics']['phases'])>1


def test_worker_initialization_source_guard_precedes_weights_and_preserves_failure(tmp_path,monkeypatch):
    from tools.dom_screen_20261027_v2 import worker,network,numerical_runtime
    events=[]
    monkeypatch.setattr(worker,'verify_runtime_sources',lambda:events.append('rules') or {'source':'synthetic'})
    monkeypatch.setattr(worker.c,'OUT',tmp_path)
    folder=tmp_path/'worker';folder.mkdir()
    monkeypatch.setenv('STUDY_WORKER_DIR',str(folder));monkeypatch.setenv('STUDY_BOUNDARY_PROFILE','1')
    monkeypatch.setattr(worker.c,'read',lambda path:{'numpy_blas_runtime':'synthetic'})
    monkeypatch.setattr(numerical_runtime,'verify_runtime',lambda *args,**kwargs:events.append('numeric') or {'contract':{},'status':'passed'})
    monkeypatch.setattr(worker,'save_worker',lambda path,value:events.append(path.name))
    def stop():
        events.append('weights')
        raise ValueError('synthetic stopped before weight load')
    monkeypatch.setattr(network,'FrozenPolicy',stop)
    with pytest.raises(ValueError,match='synthetic stopped before weight load'):worker.initialize()
    assert events==['rules','numeric','numerical_runtime.json','weights']
