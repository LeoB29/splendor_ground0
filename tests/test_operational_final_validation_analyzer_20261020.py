"""Synthetic admission/provenance regressions; no numerical audit or model work."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_live_game_operational_final_validation_20261020 as a


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def receipt(prior=None, low=False):
    prior = prior or {'count': 458, 'charged_seconds': 1097., 'receipts': []}
    rows = []
    for i in range(14 if low else 13):
        good = not low or i > 0
        rows.append({'epoch_seconds': 1000.+10*i, 'monotonic_seconds': 2000.+10*i,
            'elapsed_seconds': 10.*i, 'available_physical_bytes': 2*1024**3 if good else 1,
            'available_commit_bytes': 4*1024**3, 'admissible': good,
            'consecutive_admissible_samples': i if low else i+1, 'remaining_seconds': 30000.-10*i})
    end = rows[-1]['monotonic_seconds']+.1; wall_end = rows[-1]['epoch_seconds']+.1
    charge = max(end-2000., wall_end-1000.)
    return {'schema': 'operational_memory_admission_v2', 'supplemental': True,
        'name': 'analysis_resume_02', 'status': 'passed', 'failure_category': None, 'failure_detail': None,
        'initial': False, 'stability_required': True, 'maximum_wait_seconds': 7200,
        'maximum_cumulative_wait_seconds': 14400, 'minimum_available_physical_bytes': 2*1024**3,
        'minimum_available_commit_bytes': 4*1024**3, 'sample_interval_seconds': 10, 'stable_seconds': 120,
        'required_consecutive_samples': 13, 'reserve_seconds': 600, 'request_seconds': 0,
        'initialization_seconds': 60, 'no_worker_during_admission': True, 'outcomes_closed': True,
        'prior_admission_count': prior['count'], 'prior_gate_inputs': prior['receipts'],
        'started_epoch_seconds': 1000., 'finished_epoch_seconds': wall_end,
        'started_monotonic': 2000., 'finished_monotonic': end, 'elapsed_seconds': end-2000.,
        'charged_seconds': charge, 'cumulative_before_seconds': prior['charged_seconds'],
        'cumulative_after_seconds': prior['charged_seconds']+charge, 'readings': rows}


LAUNCH = {'epoch_seconds': 900., 'monotonic_start': 1900.}


@pytest.mark.parametrize('low', [False, True])
def test_stability_is_mandatory_and_low_reading_restarts_streak(low):
    value = receipt(low=low)
    assert a.check_supplemental(value, {'count': 458, 'charged_seconds': 1097., 'receipts': []}, LAUNCH) == value['finished_epoch_seconds']


@pytest.mark.parametrize('damage', ['fast', 'streak', 'spacing', 'short', 'threshold', 'reserve', 'charge',
    'seed', 'count', 'inputs', 'episode', 'cumulative', 'initial', 'supplemental', 'wall_deadline', 'mono_deadline',
    'extra', 'status', 'boolean_memory'])
def test_supplemental_rejects_relaxed_admission_or_refunded_time(damage):
    prior = {'count': 458, 'charged_seconds': 1097., 'receipts': []}; value = receipt(prior)
    if damage == 'fast': value['readings'] = value['readings'][:1]
    elif damage == 'streak': value['readings'][-1]['consecutive_admissible_samples'] = 12
    elif damage == 'spacing': value['readings'][1]['monotonic_seconds'] -= 1
    elif damage == 'short':
        value['readings'][-1]['monotonic_seconds'] -= .01
        value['readings'][-1]['elapsed_seconds'] -= .01
    elif damage == 'threshold': value['readings'][-1]['available_commit_bytes'] -= 1
    elif damage == 'reserve': value['readings'][0]['remaining_seconds'] = 659
    elif damage == 'charge': value['charged_seconds'] = 0
    elif damage == 'seed': value['cumulative_before_seconds'] = 0
    elif damage == 'count': value['prior_admission_count'] -= 1
    elif damage == 'inputs': value['prior_gate_inputs'] = [{}]
    elif damage == 'episode': value['maximum_wait_seconds'] += 1
    elif damage == 'cumulative': value['cumulative_after_seconds'] = 14400
    elif damage == 'initial': value['initial'] = True
    elif damage == 'supplemental': value['supplemental'] = False
    elif damage == 'wall_deadline': value['finished_epoch_seconds'] = LAUNCH['epoch_seconds']+42600
    elif damage == 'mono_deadline': value['finished_monotonic'] = LAUNCH['monotonic_start']+42600
    elif damage == 'extra': value['readings'].append(deepcopy(value['readings'][-1]))
    elif damage == 'status': value['status'] = 'preserved_hold'
    else: value['readings'][0]['available_physical_bytes'] = True
    with pytest.raises(ValueError): a.check_supplemental(value, prior, LAUNCH)


def test_backward_wall_change_preserves_full_monotonic_charge():
    value = receipt()
    for row in value['readings'][1:]: row['epoch_seconds'] -= 50.
    value['finished_epoch_seconds'] -= 50.
    value['charged_seconds'] = value['elapsed_seconds']
    value['cumulative_after_seconds'] = 1097.+value['charged_seconds']
    a.check_supplemental(value, {'count': 458, 'charged_seconds': 1097., 'receipts': []}, LAUNCH)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    source = tmp_path/'source'; repair = tmp_path/'repair'
    source.mkdir(); repair.mkdir()
    monkeypatch.setattr(a, 'ROOT', tmp_path); monkeypatch.setattr(a, 'SOURCE', source); monkeypatch.setattr(a, 'REPAIR', repair)
    monkeypatch.setattr(a.time, 'time', lambda: 1200.); monkeypatch.setattr(a.time, 'monotonic', lambda: 2200.)
    tools = tmp_path/'tools'; tools.mkdir(); modules = []
    for name in ('analyze_live_game_operational_recovery_20261020_02.py',
                 'analyze_live_game_operational_recovery_20261020.py', 'analyze_live_game_operational_20261019.py'):
        p = tools/name; p.write_text('# frozen numerical code\n'); modules.append(p)
    frozen = modules[0]; monkeypatch.setattr(a, 'FROZEN', frozen); monkeypatch.setattr(a, 'FROZEN_SHA', a.sha(frozen))
    own = tools/'wrapper.py'; own.write_text('# separate wrapper\n'); monkeypatch.setattr(a, '__file__', str(own))
    test = tmp_path/'tests/test_operational_final_validation_analyzer_20261020.py'; test.parent.mkdir(); test.write_text('# test\n')
    write(source/'freeze.json', {'inputs': [a.artifact(p) for p in modules]})
    monkeypatch.setattr(a, 'ORIGINAL_FREEZE_SHA', a.sha(source/'freeze.json'))
    launch = {**LAUNCH, 'maximum_seconds': 43200, 'freeze': a.artifact(source/'freeze.json'), 'outcomes_closed': True}
    write(source/'launch.json', launch)
    write(source/'coordinator_hold.json', {'status': 'preserved_hold'})
    for name in ('coordinator_analyzer_failure_01', 'coordinator_failure_01'): write(source/'sealed'/(name+'.json'), {'closed': True})
    write(source/'independent_analysis_execution_01.json', {'exit_code': 1, 'error': 'Hold', 'cleanup_error': None,
        'analyzer_process_closed': True, 'owned_descendants_closed': True, 'output': None,
        'log': {'bytes': 0}, 'frozen_analyzer': a.artifact(frozen)})
    for name, charge in (('initial', 120.), ('analysis', .5)):
        write(source/'memory_admissions'/(name+'.json'), {'status': 'passed', 'charged_seconds': charge})
    monkeypatch.setattr(a, 'PRIOR_GATES', 2)
    snapshot = {'schema': 'operational_final_validation_source_inventory_v1', 'source_base': 'source',
        'files': [a.artifact(p) for p in sorted(source.rglob('*')) if p.is_file()], 'outcomes_closed': True}
    write(repair/'source_inventory.json', snapshot)
    freeze = {'schema': 'operational_final_validation_repair_freeze_v1', 'outcomes_closed': True, 'larger_data_deferred': True,
        'original_freeze': a.artifact(source/'freeze.json'), 'original_launch': a.artifact(source/'launch.json'),
        'source_inventory': a.artifact(repair/'source_inventory.json'), 'inputs': [a.artifact(p) for p in
            (source/'freeze.json', source/'launch.json', repair/'source_inventory.json', frozen, own, test)]}
    write(repair/'repair_freeze.json', freeze)
    return a.Validation(a.sha(repair/'repair_freeze.json')), source, repair


@pytest.mark.parametrize('damage', [None, 'changed', 'added', 'removed', 'old_output', 'repair_freeze'])
def test_authentication_keeps_completed_source_and_failure_immutable(synthetic, damage):
    value, source, repair = synthetic
    if damage == 'changed': write(source/'coordinator_hold.json', {'status': 'changed'})
    elif damage == 'added': write(source/'extra.json', {})
    elif damage == 'removed': (source/'coordinator_hold.json').unlink()
    elif damage == 'old_output': write(source/'independent_analysis.json', {})
    elif damage == 'repair_freeze': write(repair/'repair_freeze.json', {})
    if damage is None:
        value.authenticate()
        assert value.prior['count'] == 2 and value.prior['charged_seconds'] == 120.5
    else:
        with pytest.raises((ValueError, FileNotFoundError)): value.authenticate()


@pytest.mark.parametrize('damage', [None, 'log', 'early_start'])
def test_supplemental_receipt_and_readings_precede_second_process(synthetic, monkeypatch, damage):
    value, _, repair = synthetic; value.authenticate()
    admission = receipt(value.prior); log = repair/'memory_admission.jsonl'
    log.write_text(''.join(json.dumps(row)+'\n' for row in admission['readings']))
    admission['readings_log'] = {'path': 'memory_admission.jsonl', 'sha256': a.sha(log), 'bytes': log.stat().st_size}
    write(repair/'memory_admission.json', admission)
    if damage == 'log': log.write_text('{}\n')
    import psutil
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(create_time=lambda: 1000. if damage == 'early_start' else 1200.))
    if damage is None: value.supplemental()
    else:
        with pytest.raises(ValueError): value.supplemental()


def test_run_calls_frozen_numerical_algorithm_once_without_subclass_or_global_mutation(monkeypatch):
    calls = []; accepted = {'synthetic': True}; original = object()
    class FrozenAudit:
        marker = original
        def __init__(self, source): calls.append(('construct', source))
        def run(self): calls.append('numerical'); return accepted
    value = a.Validation('synthetic'); value.launch = LAUNCH
    monkeypatch.setattr(value, 'authenticate', lambda: calls.append('authenticate'))
    monkeypatch.setattr(value, 'supplemental', lambda: calls.append('admission'))
    monkeypatch.setattr(a, 'boundary', lambda launch: None)
    monkeypatch.setattr(a.importlib, 'import_module', lambda name: SimpleNamespace(Audit=FrozenAudit))
    monkeypatch.setattr(value, 'finish', lambda result: calls.append(('finish', result)) or result)
    assert value.run() is accepted and FrozenAudit.marker is original
    assert calls == ['authenticate', 'admission', ('construct', a.SOURCE), 'numerical', ('finish', accepted)]


@pytest.mark.parametrize('damage', [None, 'source', 'old_input', 'new_input', 'admission', 'seed_result', 'numerical_failure'])
def test_after_audit_rechecks_preservation_and_adds_separate_combined_waiting(synthetic, damage):
    value, source, repair = synthetic; value.authenticate()
    value.admission = receipt(value.prior)
    write(repair/'memory_admission.json', value.admission)
    (repair/'memory_admission.jsonl').write_text('synthetic-bound-log\n')
    value.admission_artifact = a.artifact(repair/'memory_admission.json')
    value.admission_log_artifact = a.artifact(repair/'memory_admission.jsonl')
    waiting = {'admissions': 2, 'stability_required_admissions': 1, 'charged_seconds': 120.5}
    result = {'status': 'passed', 'outcomes_closed': True, 'strength_analysis_games': 0,
        'freeze_sha256': a.ORIGINAL_FREEZE_SHA, 'memory_waiting': deepcopy(waiting),
        'source_preservation': {'files': 3, 'all_hashes_and_sizes_match': True}}
    if damage == 'source': write(source/'coordinator_hold.json', {'status': 'changed'})
    elif damage == 'old_input': a.FROZEN.write_text('# changed old source\n')
    elif damage == 'new_input': Path(a.__file__).write_text('# changed wrapper\n')
    elif damage == 'admission': write(repair/'memory_admission.json', {})
    elif damage == 'seed_result': result['memory_waiting']['charged_seconds'] = 0.
    elif damage == 'numerical_failure': result['status'] = 'failed'
    if damage is None:
        result = value.finish(result)
        assert result['memory_waiting'] == waiting
        assert result['combined_memory_waiting'] == {'admissions': 3, 'stability_required_admissions': 2,
            'charged_seconds': value.admission['cumulative_after_seconds']}
        assert result['final_validation_repair']['frozen_numerical_algorithm_unchanged'] is True
        assert result['final_validation_repair']['original_coordinator_hold_retained'] is True
    else:
        with pytest.raises(ValueError): value.finish(result)


@pytest.mark.parametrize('clock', ['wall', 'monotonic'])
def test_original_deadline_retains_six_hundred_second_final_reserve(monkeypatch, clock):
    monkeypatch.setattr(a.time, 'time', lambda: LAUNCH['epoch_seconds']+(42600 if clock == 'wall' else 1))
    monkeypatch.setattr(a.time, 'monotonic', lambda: LAUNCH['monotonic_start']+(42600 if clock == 'monotonic' else 1))
    with pytest.raises(ValueError): a.boundary(LAUNCH)


def test_failure_is_written_only_in_new_repair_directory(synthetic, monkeypatch, capsys):
    _, source, repair = synthetic; before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(a.sys, 'argv', ['wrapper', '--repair-freeze-sha256', 'unused'])
    def fail(self): raise ValueError('SEALED_SYNTHETIC_DETAIL')
    monkeypatch.setattr(a.Validation, 'run', fail)
    with pytest.raises(ValueError): a.main()
    assert 'SEALED_SYNTHETIC_DETAIL' in a.read(repair/'sealed/independent_analysis_failure_01.json')['traceback']
    assert not capsys.readouterr().out
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
