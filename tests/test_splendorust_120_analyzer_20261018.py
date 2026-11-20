"""Synthetic qualification cases and recorded receipt schemas; no model execution."""
from copy import deepcopy
import json

import pytest

from tools import analyze_splendorust_120_qualification_20261018 as a
from tools.analyze_external_strength_screen_20261018_02 import check_clock, check_diagnostics


def calibration(quota, fixture, seconds=95., status='completed'):
    return {'stage': 'calibration', 'quota': quota, 'fixture': fixture, 'kind': 'original',
            'correct': status == 'completed', 'result': {'status': status} if status != 'timeout' else None,
            'clock': {'status': 'timeout' if status == 'timeout' else 'ok',
                      'receipt_elapsed_seconds': None if status == 'timeout' else seconds}}


def quota_rows(quota, seconds=95.):
    return [calibration(quota, fixture, seconds) for fixture in a.CALIBRATION_FIXTURES]


@pytest.mark.parametrize('count', [1, 2, 3, 4])
def test_calibration_selects_last_full_pass_and_stops_immediately(count):
    rows = [r for q in a.LADDER[:count] for r in quota_rows(q)]
    if count < 4: rows.append(calibration(a.LADDER[count], 0, 96.001))
    result = a.reconstruct_selection(rows)
    assert result['selected_quota'] == a.LADDER[count-1]
    assert result['attempted'] == count*7 + int(count < 4)


@pytest.mark.parametrize('status,seconds', [('completed', 96.000001), ('timeout', None), ('internal_budget_exhausted', 108.)])
def test_no_passing_quota_is_a_valid_negative_calibration(status, seconds):
    result = a.reconstruct_selection([calibration(256, 0, seconds, status)])
    assert result['selected_quota'] is None and result['attempted'] == 1
    assert a.final_gate(None, []) == {'qualified': False, 'verified_decisions': 0, 'privacy_pairs_matched': 0}


def test_exact_margin_and_total_validation_time_are_distinguished():
    row = calibration(256, 0, 96.)
    row['clock']['elapsed_seconds'] = 97.
    assert a.calibration_gate(row)
    row['clock']['receipt_elapsed_seconds'] += .000001
    assert not a.calibration_gate(row)


@pytest.mark.parametrize('damage', ['extra_after_failure', 'skip_lower', 'missing_fixture', 'reorder_fixture', 'hidden_calibration', 'truncate_ladder'])
def test_calibration_rejects_scope_or_order_changes(damage):
    rows = quota_rows(256) + [calibration(1024, 0, 97.)]
    if damage == 'extra_after_failure': rows.append(calibration(1024, 3))
    elif damage == 'skip_lower': rows = quota_rows(1024)
    elif damage == 'missing_fixture': rows.pop(2)
    elif damage == 'reorder_fixture': rows[0], rows[1] = rows[1], rows[0]
    elif damage == 'hidden_calibration': rows[0]['kind'] = 'hidden_variant'
    elif damage == 'truncate_ladder': rows.pop()
    with pytest.raises(ValueError): a.reconstruct_selection(rows)


def final_rows():
    return [{'stage': 'qualification', 'quota': 1024, 'fixture': i//2,
             'kind': 'hidden_variant' if i%2 else 'original', 'correct': True,
             'result': {'action': {'fixture': i//2}, 'diagnostics': {'same': True}}} for i in range(46)]


def test_full_final_gate_and_timing_failure_are_operational_not_strength_claims():
    rows = final_rows()
    assert a.final_gate(1024, rows) == {'qualified': True, 'verified_decisions': 46, 'privacy_pairs_matched': 23}
    rows[3]['correct'] = False; rows[3]['result'] = {'status': 'internal_budget_exhausted'}
    assert a.final_gate(1024, rows) == {'qualified': False, 'verified_decisions': 45, 'privacy_pairs_matched': 22}


@pytest.mark.parametrize('damage', ['action', 'diagnostics', 'missing', 'quota', 'order'])
def test_final_gate_rejects_repeat_or_schedule_corruption(damage):
    rows = final_rows()
    if damage == 'action': rows[1]['result']['action']['fixture'] = 99
    elif damage == 'diagnostics': rows[1]['result']['diagnostics']['same'] = False
    elif damage == 'missing': rows.pop()
    elif damage == 'quota': rows[1]['quota'] = 4096
    else: rows[0], rows[2] = rows[2], rows[0]
    with pytest.raises(ValueError): a.final_gate(1024, rows)


def initialization():
    base = a.ROOT / 'outputs/external_strength_screen_20261018_02'
    value = a.read(base / 'workers/qualification/initialization.json')
    freeze = a.read(base / 'freeze.json')
    freeze['executable'] = value['command'][0]
    value['command'][-1] = a.MODULE
    saved = {k: v for k, v in value.items() if k not in a.METADATA and k != 'result'}
    return value, saved, freeze


def test_real_enriched_initialization_and_frozen_clock_contract():
    value, saved, freeze = initialization()
    a.check_initialization(value, saved, freeze)
    check_clock(value, None, 60, freeze['topology']['selected_logical_processor'])
    a.process_identity([value, deepcopy(value)], freeze, a.read(a.ROOT / a.BASE_PROFILE))


@pytest.mark.parametrize('damage', ['command', 'affinity', 'topology', 'threads', 'identity', 'ready_pid', 'extra', 'clock'])
def test_initialization_metadata_and_copied_clock_mutations_fail(damage):
    value, saved, freeze = initialization(); saved = deepcopy(saved)
    if damage == 'command': value['command'][-1] += '_other'
    elif damage == 'affinity': value['affinity'] = [999]
    elif damage == 'topology': value['topology'] = {**value['topology'], 'extra': True}
    elif damage == 'threads': value['thread_environment']['OMP_NUM_THREADS'] = '2'
    elif damage == 'identity': value['identity']['created'] += 1.
    elif damage == 'ready_pid': value['result']['pid'] += 1
    elif damage == 'extra': value['unknown_supervisor_field'] = True
    else: saved['receipt_elapsed_seconds'] += .001
    with pytest.raises(ValueError): a.check_initialization(value, saved, freeze)


def test_process_replacement_native_hash_and_fake_helper_are_rejected(monkeypatch):
    first, _, freeze = initialization(); profile = a.read(a.ROOT / a.BASE_PROFILE)
    for damage in ('replacement', 'native_hash', 'helper'):
        second = deepcopy(first)
        if damage == 'replacement': second['telemetry']['process_tree'][0]['created'] += 1.
        elif damage == 'native_hash': next(p for p in second['telemetry']['process_tree'] if p['role'] == 'rust')['executable_sha256'] = '0'*64
        else: next(p for p in second['telemetry']['process_tree'] if p['role'].startswith('console:'))['executable'] = 'C:/fake/conhost.exe'
        with pytest.raises(ValueError): a.process_identity([first, second], freeze, profile)


def test_exact_profile_changes_retain_all_model_and_native_artifacts():
    baseline = a.read(a.ROOT / a.BASE_PROFILE)
    value = a.expected_profile(baseline, 6400)
    assert {k for k in baseline if baseline[k] != value[k]} == {'schema', 'profile_id', 'iterations', 'internal_budget_seconds'}
    assert set(value)-set(baseline) == {'base_profile', 'external_budget_seconds'}
    assert value['artifacts'] == baseline['artifacts'] and value['depth'] == 64 and value['world_pool'] == 3
    with pytest.raises(ValueError): a.expected_profile(baseline, True)


def test_diagnostic_counts_reject_incomplete_large_quota_without_inference():
    base = a.ROOT / 'outputs/external_strength_screen_20261018_02'
    fixture = a.read(base / 'dynamic_qualification/002.json')
    value = fixture['clock']['result']; observer = fixture['observer']
    profile = a.expected_profile(a.read(a.ROOT / a.BASE_PROFILE), 6400)
    diagnostic = deepcopy(value['diagnostics']); diagnostic['profile_id'] = profile['profile_id']
    for phase in diagnostic['phases']:
        if not phase['heuristic_phase']: phase.update(simulations=6400, leaf_calls=6401)
    diagnostic['inference_calls'] = diagnostic['inference_rows'] = 6401
    from tools.external_strength_screen_20261018 import rules
    snapshot = a.read(base / 'fixtures.json')[0]['snapshot']
    legal = [rules.serialize_action(v) for v in rules.legal_actions(rules.deserialize_snapshot(snapshot))]
    assert check_diagnostics(diagnostic, observer, value['action'], legal, profile)['simulations'] == 6400
    diagnostic['phases'][0]['simulations'] = 6399
    with pytest.raises(ValueError): check_diagnostics(diagnostic, observer, value['action'], legal, profile)


@pytest.mark.parametrize('status', ['completed', 'internal_budget_exhausted', 'timeout'])
def test_full_row_receipts_result_binding_and_dictionary_transition(tmp_path, monkeypatch, status):
    """Exercise all three real runner row paths using existing receipt-shaped data."""
    from tools.external_strength_screen_20261018 import rules, audit as ledger
    base = a.ROOT / 'outputs/external_strength_screen_20261018_02'
    old = a.read(base / 'dynamic_qualification/002.json')
    fixture = a.read(base / 'fixtures.json')[0]
    init, saved_init, freeze = initialization()
    profile = a.expected_profile(a.read(a.ROOT / a.BASE_PROFILE), 256)
    clock = deepcopy(old['clock']); clock['request_id'] = 'request-00000001'
    result = clock['result']; result['status'] = 'completed'
    profile_path = tmp_path / 'profiles/q00256.json'
    profile_path.parent.mkdir(); profile_path.write_text(json.dumps(profile), encoding='utf-8')
    result['profile_sha256'] = a.file_hash(profile_path)
    result['diagnostics']['profile_id'] = profile['profile_id']
    for phase in result['diagnostics']['phases']:
        if not phase['heuristic_phase']: phase.update(simulations=256, leaf_calls=257)
    result['diagnostics']['inference_calls'] = result['diagnostics']['inference_rows'] = 257
    if status == 'internal_budget_exhausted':
        result = {k: v for k, v in result.items() if k not in ('action', 'diagnostics', 'status')}
        result.update(status=status, internal_budget_seconds=108, opponent_id='splendorust',
                      profile_id=profile['profile_id'], reason='Internal fixed-work deadline exceeded')
        clock['result'] = result
    if status == 'timeout':
        clock.update(status='timeout', accepted=False, result=None, receipt_elapsed_seconds=None, received_before_deadline=False)
    # Temporary test evidence lives outside the repository; source identity checks remain real.
    monkeypatch.setattr(a, 'artifact', lambda p: {'path': str(p), 'sha256': a.file_hash(p), 'bytes': p.stat().st_size})
    row = {'stage': 'calibration', 'quota': 256, 'fixture': 0, 'kind': 'original',
           'root_id': fixture['source']['root_id'], 'observer_sha256': a.digest(old['observer']),
           'profile': a.artifact(profile_path), 'correct': status == 'completed',
           'initialization': saved_init, 'clock': {k: v for k, v in clock.items() if k != 'result'}}
    if status != 'timeout': row.update(result=result, result_sha256=a.digest(result))
    if status == 'completed':
        after = rules.serialize_snapshot(rules.CorrectedEnv().step(rules.deserialize_snapshot(fixture['snapshot']),
                                                                  rules.deserialize_action(result['action'])))
        row.update(post_snapshot=after, ledger_sha256=a.digest(after))
    name = 'q00256_00'
    folder = tmp_path / 'workers/calibration' / name; folder.mkdir(parents=True)
    for filename, value in (('initialization.json', init), ('request-00000001.json', clock)):
        (folder / filename).write_text(json.dumps(value), encoding='utf-8')
    row_path = tmp_path / 'rows/calibration' / (name + '.json'); row_path.parent.mkdir(parents=True)
    row_path.write_text(json.dumps(row), encoding='utf-8')
    audit = a.Audit(tmp_path)
    audit.launch = {'epoch_seconds': a.time.time()}
    audit.fixtures = [fixture]; audit.observers = [{'original': old['observer']}]
    audit.freeze_doc = freeze; audit.cpu = freeze['topology']['selected_logical_processor']
    audit.profiles = {256: profile}; audit.rules = rules; audit.ledger = ledger
    audit.check_clock = check_clock; audit.check_diagnostics = check_diagnostics
    assert audit.row('calibration', name, 256, 0, 'original') == row
    assert audit.completed_transitions == int(status == 'completed')
    if status == 'completed':
        row['post_snapshot']['state']['bank_tokens']['gold'] += 1
        row_path.write_text(json.dumps(row), encoding='utf-8')
        with pytest.raises(ValueError, match='successor'):
            audit.row('calibration', name, 256, 0, 'original')
