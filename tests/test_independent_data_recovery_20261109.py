"""Independent recovery counterexamples: no neural runtime or scientific work."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest
import psutil

from tools.splendorust_independent_data_recovery_20261109 import common as c
from tools.splendorust_independent_data_recovery_20261109 import preflight, runtime

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')


def valid_summary():
    return dict(tests_collected=300, tests_passed=300, failures=0, skipped=0,
                tensor_runtime_loaded=False, models=0, games=0)


def test_full_predecessor_and_new_recovery_suites_are_selected():
    paths = preflight.test_paths()
    assert len([p for p in paths if '20261107' in p.name]) == 10
    assert Path(__file__).resolve() in paths
    assert len(paths) == len(set(paths))


def test_valid_complete_test_summary_is_accepted():
    assert preflight.complete_summary(valid_summary()) is True


@pytest.mark.parametrize('key,value', [
    ('tests_passed', 299), ('tests_collected', 0), ('failures', 1), ('skipped', 1),
    ('tensor_runtime_loaded', True), ('models', 1), ('games', 1),
    ('tests_passed', 300.0), ('models', False), ('tensor_runtime_loaded', 0),
])
def test_partial_skipped_or_type_aliased_test_proof_is_rejected(key, value):
    proof = valid_summary()
    proof[key] = value
    with pytest.raises(ValueError):
        preflight.complete_summary(proof)


def clocks():
    return (
        dict(monotonic_origin=100., unix_origin=1000.),
        dict(monotonic=120., unix_time=1020., monotonic_deadline=180., unix_deadline=1080.),
        dict(monotonic_origin=140., unix_origin=1040., monotonic_deadline=170., unix_deadline=1070.),
    )


def test_resumed_worker_within_original_clocks_is_admissible():
    runtime.check_deadlines(*clocks(), 150., 1050.)


@pytest.mark.parametrize('mono,wall', [(170., 1050.), (150., 1070.), (99., 1050.), (150., 999.)])
def test_either_original_clock_or_worker_clock_rejects_expiry(mono, wall):
    with pytest.raises(ValueError):
        runtime.check_deadlines(*clocks(), mono, wall)


def test_recovery_downtime_cannot_be_recovered_by_a_fresh_worker_clock():
    claim, phase, worker = clocks()
    worker.update(monotonic_origin=170., unix_origin=1070.,
                  monotonic_deadline=10000., unix_deadline=10000.)
    with pytest.raises(ValueError, match='phase'):
        runtime.check_deadlines(claim, phase, worker, 181., 1081.)


def test_recovery_model_work_cannot_use_saved_only_finalization_reserve():
    claim, phase, worker = clocks()
    mono, wall = claim['monotonic_origin'] + c.MODEL_SECONDS, claim['unix_origin'] + c.MODEL_SECONDS
    phase.update(monotonic_deadline=mono+100., unix_deadline=wall+100.)
    worker.update(monotonic_deadline=mono+100., unix_deadline=wall+100.)
    with pytest.raises(ValueError, match='model cutoff'):
        runtime.check_deadlines(claim, phase, worker, mono, wall, model=True)
    runtime.check_deadlines(claim, phase, worker, mono, wall, model=False)


def test_saved_only_work_still_cannot_extend_original_overall_deadline():
    claim, phase, worker = clocks()
    mono, wall = claim['monotonic_origin'] + c.TOTAL_SECONDS, claim['unix_origin'] + c.TOTAL_SECONDS
    phase.update(monotonic_deadline=mono+100., unix_deadline=wall+100.)
    worker.update(monotonic_deadline=mono+100., unix_deadline=wall+100.)
    with pytest.raises(ValueError, match='overall'):
        runtime.check_deadlines(claim, phase, worker, mono, wall, model=False)


def test_new_output_helpers_cannot_write_into_preserved_attempt():
    forbidden = c.SOURCE_OUT / 'recovery_must_never_create_this.json'
    assert not forbidden.exists()
    with pytest.raises(ValueError, match='New study output only'):
        c.save(forbidden, {'forbidden': True})
    assert not forbidden.exists()


def test_recovery_sources_preserve_fixed_clock_and_resource_ceilings():
    from tools.splendorust_independent_data_20261107 import common as previous
    for name in ('TOTAL_SECONDS', 'MODEL_SECONDS', 'MAX_EVIDENCE_BYTES',
                 'PREFINAL_EVIDENCE_BYTES', 'FINAL_RESERVE_BYTES', 'MIN_DISK',
                 'SECONDS_PER_MOVE', 'INTERNAL_SECONDS', 'MAX_TOTAL_GAMES'):
        assert getattr(c, name) == getattr(previous, name)
    assert c.STAGES['strength'] == previous.STAGES['strength'] == 40 * 3600
    assert c.SOURCE_OUT == previous.OUT
    assert c.OUT != c.SOURCE_OUT
    assert c.INITIAL_AVAILABLE_RAM_BYTES == c.MINIMUM_AVAILABLE_RAM_BYTES == 0
    assert c.RAM_WAIT_SECONDS == 0


@pytest.mark.parametrize('rep', [0, 1])
def test_actual_committed_broad_checkpoint_bytes_import_without_tensor_runtime(rep):
    """Exercise the real new importer against saved final96, not a synthetic clone."""
    from tools.splendorust_independent_data_recovery_20261109.strength import checkpoint
    manifest, entries, raw = checkpoint.source_model_bytes(rep)
    assert manifest['metadata']['fit_id'] == f'r{rep}_broad_joint'
    assert manifest['metadata']['source_training_games'] == 1024
    assert manifest['metadata']['checkpoint_step'] == 96
    assert sum(len(value) for value in raw.values()) == sum(entry['bytes'] for entry in entries)
    assert len(raw) == len(entries) > 0
    assert 'torch' not in sys.modules


def test_combined_evidence_accounting_includes_preserved_attempt(tmp_path):
    from tools.splendorust_independent_data_recovery_20261109.strength import envelope
    (tmp_path / 'new.json').write_bytes(b'new evidence')
    with patch.object(c, 'OUT', tmp_path), patch.object(c, 'preserved_bytes', return_value=123456):
        assert envelope.bytes_used() == 123456 + len(b'new evidence')


def test_new_small_tree_cannot_ignore_predecessor_bytes_at_original_cap(tmp_path):
    from tools.splendorust_independent_data_recovery_20261109.strength import envelope
    (tmp_path / 'new.json').write_bytes(b'ab')
    mono, wall = time.monotonic(), time.time()
    account = envelope._AccountingEnvelope(dict(monotonic_origin=mono, unix_origin=wall))
    account.stage = 'strength'
    with patch.object(c, 'OUT', tmp_path), \
         patch.object(c, 'preserved_bytes', return_value=c.PREFINAL_EVIDENCE_BYTES-1):
        with pytest.raises(ValueError, match='evidence ceiling'):
            account.guard()


def write_fixture(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(c.canonical(value), encoding='utf-8')
    return c.artifact(path)


@pytest.fixture
def reuse_fixture(tmp_path, monkeypatch):
    """An actual on-disk, hash-bound stopped prefix with no scientific payload."""
    source, output = tmp_path / 'preserved', tmp_path / 'recovery'
    monkeypatch.setattr(c, 'SOURCE_OUT', source)
    monkeypatch.setattr(c, 'SOURCE_PLAN', source / 'freeze.json')
    monkeypatch.setattr(c, 'OUT', output)
    monkeypatch.setattr(c, 'PLAN', output / 'freeze.json')
    original_freeze = write_fixture(c.SOURCE_PLAN, {'fixture': 'original freeze'})
    monkeypatch.setattr(c, 'SOURCE_FREEZE_SHA', original_freeze['sha256'])
    mono, wall = time.monotonic()-3600., time.time()-3600.
    old_claim = dict(pid=999999, created=1., freeze=original_freeze,
                     monotonic_origin=mono, unix_origin=wall,
                     total_seconds=c.TOTAL_SECONDS, model_seconds=c.MODEL_SECONDS)
    original_claim = write_fixture(source / 'claim.json', old_claim)
    phase = dict(stage='strength', claim=original_claim, monotonic=mono+1000.,
                 unix_time=wall+1000., cap_seconds=c.STAGES['strength'],
                 monotonic_deadline=mono+1000.+c.STAGES['strength'],
                 unix_deadline=wall+1000.+c.STAGES['strength'])
    original_phase = write_fixture(source / 'stages/strength_start.json', phase)
    from tools.splendorust_independent_data_recovery_20261109.recovery import expected_source_executions
    passed = []
    for index, path in enumerate(expected_source_executions()):
        passed.append(write_fixture(path, dict(
            status='passed', worker_exit_code=0, error=None, cleanup_errors=[],
            observed_owned_identities=[], identity=dict(pid=900000+index, created=1.))))
    failed = write_fixture(source / 'segments/qualification_r0/execution.json', dict(
        status='failed', worker_exit_code=1, cleanup_errors=[],
        observed_owned_identities=[], identity=dict(pid=999998, created=1.)))
    original_execution = write_fixture(source / 'execution.json', dict(status='failed', segments=passed))
    inventory = write_fixture(output / 'original_inventory.jsonl', {'fixture': 'no scientific data'})
    reuse = dict(schema='independent_data_preserved_prefix_v1', original_freeze=original_freeze,
                 original_claim=original_claim, original_strength_phase=original_phase,
                 original_execution=original_execution, failed_execution=failed,
                 passed_executions=passed, old_inventory=inventory,
                 preserved_files=160, preserved_bytes=123456)
    reuse_item = write_fixture(output / 'reuse.json', reuse)
    freeze = write_fixture(c.PLAN, {'fixture': 'recovery freeze'})
    claim = dict(pid=os.getpid(), created=psutil.Process().create_time(),
                 original_claim=original_claim, reuse=reuse_item, freeze=freeze,
                 plan_sha256=freeze['sha256'], automatic_retries=0,
                 **{key: old_claim[key] for key in ('monotonic_origin', 'unix_origin',
                                                   'total_seconds', 'model_seconds')})
    write_fixture(output / 'claim.json', claim)
    return reuse, claim


def test_complete_preserved_prefix_and_inherited_claim_are_accepted(reuse_fixture):
    from tools.splendorust_independent_data_recovery_20261109 import recovery
    reuse, claim = reuse_fixture
    assert recovery.validate_reuse(reuse) == reuse
    assert recovery.validate_recovery_claim(claim) == claim


def test_recovery_cannot_drop_one_predecessor_receipt(reuse_fixture):
    from tools.splendorust_independent_data_recovery_20261109 import recovery
    reuse, _ = reuse_fixture
    changed = dict(reuse, passed_executions=reuse['passed_executions'][:-1])
    with pytest.raises(ValueError, match='prefix'):
        recovery.validate_reuse(changed)


def test_started_inference_worker_disqualifies_predecision_recovery(reuse_fixture):
    from tools.splendorust_independent_data_recovery_20261109 import recovery
    reuse, _ = reuse_fixture
    worker = c.SOURCE_OUT / 'strength/r0/workers/qualification_candidate'
    worker.mkdir(parents=True)
    with pytest.raises(ValueError, match='retried'):
        recovery.validate_reuse(reuse)


def test_failed_qualification_cannot_be_relabelled_passed(reuse_fixture):
    from tools.splendorust_independent_data_recovery_20261109 import recovery
    reuse, _ = reuse_fixture
    path = c.resolve(reuse['failed_execution'])
    changed = c.read(path)
    changed.update(status='passed', worker_exit_code=0)
    reuse['failed_execution'] = write_fixture(path, changed)
    with pytest.raises(ValueError, match='failed qualification'):
        recovery.validate_reuse(reuse)


@pytest.mark.parametrize('key', ['monotonic_origin', 'unix_origin', 'total_seconds', 'model_seconds'])
def test_recovery_claim_cannot_restart_or_extend_any_origin_or_allowance(reuse_fixture, key):
    from tools.splendorust_independent_data_recovery_20261109 import recovery
    _, claim = reuse_fixture
    changed = dict(claim)
    changed[key] += 1
    with pytest.raises(ValueError, match='Inherited unchanged'):
        recovery.validate_recovery_claim(changed)


def test_recovered_phase_rebinds_owner_without_restarting_phase_clock(reuse_fixture):
    from tools.splendorust_independent_data_recovery_20261109 import recovery
    reuse, claim = reuse_fixture
    original = c.read(c.resolve(reuse['original_strength_phase']))
    recovered = c.read(c.resolve(recovery.recover_phase(claim)))
    assert recovered['claim'] == c.artifact(c.OUT / 'claim.json')
    assert recovered['original_phase'] == reuse['original_strength_phase']
    for key in ('stage', 'monotonic', 'unix_time', 'monotonic_deadline', 'unix_deadline', 'cap_seconds'):
        assert type(recovered[key]) is type(original[key]) and recovered[key] == original[key]


@pytest.mark.parametrize('mode', ['passed', 'failed'])
def test_actual_recovery_owned_process_exit_and_failure_preservation(reuse_fixture, monkeypatch, mode):
    from tools.splendorust_independent_data_recovery_20261109 import owner, recovery
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    _, claim = reuse_fixture
    phase = recovery.recover_phase(claim)
    monkeypatch.setenv('INDEPENDENT_DATA_PROCESS_PROBE', '1')
    call = lambda: owner.run_job(claim, phase, claim['plan_sha256'], c.PACKAGE+'.process_probe',
                                 mode, c.OUT/'probe', seconds=20, rss=c.GiB, model=False)
    if mode == 'passed':
        call()
    else:
        with pytest.raises(ValueError, match='Failed worker preserved'):
            call()
    actual = c.read(c.OUT/'probe/execution.json')
    assert actual['status'] == ('passed' if mode == 'passed' else 'failed')
    assert actual['worker_exit_code'] == (0 if mode == 'passed' else 1)
    assert actual['cleanup_errors'] == [] and actual['retries'] == 0
    assert actual['observed_owned_identities']
    for identity in actual['observed_owned_identities'] + [actual['identity']]:
        require_absent(identity)


def test_expired_original_phase_prevents_even_a_probe_process_launch(reuse_fixture):
    from tools.splendorust_independent_data_recovery_20261109 import owner, recovery
    _, claim = reuse_fixture
    item = recovery.recover_phase(claim)
    value = c.read(c.resolve(item))
    value.update(monotonic_deadline=time.monotonic()-1, unix_deadline=time.time()-1)
    phase = write_fixture(c.OUT/'stages/expired_fixture.json', value)
    with patch.object(owner.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError, match='Failed worker preserved'):
            owner.run_job(claim, phase, claim['plan_sha256'], c.PACKAGE+'.process_probe',
                          'passed', c.OUT/'probe', seconds=20, rss=c.GiB, model=False)
        launch.assert_not_called()
    actual = c.read(c.OUT/'probe/execution.json')
    assert actual['status'] == 'failed' and actual['worker_exit_code'] is None
    assert actual['identity'] is None and actual['retries'] == 0


def complete_test_receipt():
    files = [c.artifact(path) for path in preflight.test_paths()]
    runner = c.artifact(c.ROOT/'tools/splendorust_independent_data_recovery_20261109/preflight.py')
    return dict(schema=preflight.SCHEMA, full_suite=True, exit_code=0,
                observed_process_exit_code=0, suite_summary=valid_summary(),
                runner=runner, inputs=files+[runner], test_files=files, models=0, games=0)


def test_production_admission_accepts_complete_recovery_test_proof():
    from tools.splendorust_independent_data_recovery_20261109 import owner
    owner.validate_test_receipt(complete_test_receipt())


@pytest.mark.parametrize('scope,key,value', [
    ('summary', 'failures', False), ('summary', 'skipped', 0.0),
    ('summary', 'models', False), ('summary', 'games', 0.0),
    ('outer', 'models', False), ('outer', 'games', 0.0),
    ('outer', 'observed_process_exit_code', 1),
])
def test_production_admission_rejects_partial_type_aliased_or_bad_exit_proofs(scope, key, value):
    from tools.splendorust_independent_data_recovery_20261109 import owner
    proof = complete_test_receipt()
    (proof['suite_summary'] if scope == 'summary' else proof)[key] = value
    with pytest.raises(ValueError):
        owner.validate_test_receipt(proof)


def test_production_admission_cannot_omit_recovery_tests_from_declared_suite():
    from tools.splendorust_independent_data_recovery_20261109 import owner
    proof = complete_test_receipt()
    proof['test_files'] = [x for x in proof['test_files'] if 'recovery' not in x['path']]
    with pytest.raises(ValueError):
        owner.validate_test_receipt(proof)


@pytest.mark.parametrize('change', ['none', 'changed_bytes', 'omitted_failure', 'extra_file', 'duplicate_path'])
def test_preservation_reader_detects_original_attempt_changes(tmp_path, monkeypatch, change):
    from tools.splendorust_independent_data_recovery_20261109 import audit
    source, output = tmp_path/'source', tmp_path/'recovery'
    monkeypatch.setattr(c, 'SOURCE_OUT', source)
    monkeypatch.setattr(c, 'OUT', output)
    first = write_fixture(source/'failure.json', {'status': 'failed'})
    second = write_fixture(source/'strength/r0/partial_export.json', {'partial': True})
    entries = [first, second]
    if change == 'changed_bytes':
        write_fixture(source/'failure.json', {'status': 'falsely_passed'})
    elif change == 'omitted_failure':
        entries = [second]
    elif change == 'extra_file':
        write_fixture(source/'undisclosed_retry.json', {'retry': True})
    elif change == 'duplicate_path':
        entries.append(first)
    output.mkdir()
    inventory = output/'original_inventory.jsonl'
    inventory.write_text(''.join(c.canonical(item)+'\n' for item in entries), encoding='utf-8')
    reuse = dict(old_inventory=c.artifact(inventory), preserved_files=len(entries),
                 preserved_bytes=sum(item['bytes'] for item in entries))
    if change == 'none':
        assert audit.verify_preserved(reuse)['all_original_bytes_preserved'] is True
    else:
        with pytest.raises(ValueError):
            audit.verify_preserved(reuse)


@pytest.mark.parametrize('mode,allowed', [('audit', True), ('final', False)])
def test_saved_reader_allows_only_audit_tensor_deserialization(tmp_path, monkeypatch, mode, allowed):
    """A fake module import proves dispatch policy without importing Torch."""
    from tools.splendorust_independent_data_recovery_20261109 import saved_worker, audit
    folder = tmp_path/('training_audit' if mode == 'audit' else 'final_audit')/'worker'
    monkeypatch.setattr(c, 'OUT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['saved_worker', '--mode', mode,
                                    '--plan-sha256', 'a'*64, '--output', str(folder)])
    monkeypatch.setattr(runtime, 'bootstrap', lambda *args, **kwargs: ({}, lambda: None, {}))

    def saved_decode(*args, **kwargs):
        monkeypatch.setitem(sys.modules, 'torch', object())
        return dict(status='all_training_saved_verified', new_optimizer_calls=0,
                    recovery_model_calls=0)

    monkeypatch.setattr(saved_worker, 'training_audit', saved_decode)
    monkeypatch.setattr(audit, 'final', saved_decode)
    result = saved_worker.main()
    assert result == (0 if allowed else 1)
    receipt = c.read(folder/'result.json')
    assert receipt['status'] == ('passed' if allowed else 'failed')
    assert receipt['new_models'] == receipt['new_games'] == receipt['new_optimizer_calls'] == 0

