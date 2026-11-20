"""Counterexamples for truthful saved-only receipt substitution and read-only use."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import common as c, saved


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def passed(new=True):
    identity = dict(pid=100, created=200.)
    mode = saved.INDEPENDENT if new else 'independent'
    execution = dict(status='passed', worker_exit_code=0, cleanup_errors=[], error=None,
                     identity=identity, observed_owned_identities=[identity], mode=mode, retries=0)
    owner = dict(identity=identity, windows_job_assigned=True)
    result = dict(status='passed', error=None, mode=mode, model_calls=0, new_games=0, optimizer_calls=0)
    result.update(dict(identity=identity) if new else dict(pid=100, created=200.))
    return execution, owner, result


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('worker_exit_code', None),
                                        ('worker_exit_code', 1), ('worker_exit_code', False),
                                        ('cleanup_errors', ['owned process alive']), ('error', 'interrupted'),
                                        ('mode', 'independent_mixed_terminal_r0'), ('retries', 1)])
def test_worker_result_alone_cannot_replace_observed_clean_exit(field, value):
    execution, owner, result = passed()
    execution[field] = value
    with pytest.raises(ValueError):
        saved.validate_passed_execution(execution, owner, result, mode=saved.INDEPENDENT, new=True)


@pytest.mark.parametrize('field', ['model_calls', 'new_games', 'optimizer_calls'])
def test_saved_continuation_rejects_scientific_work(field):
    execution, owner, result = passed()
    result[field] = 1
    with pytest.raises(ValueError):
        saved.validate_passed_execution(execution, owner, result, mode=saved.INDEPENDENT, new=True)


def test_wrong_owner_identity_and_missing_windows_ownership_are_rejected():
    execution, owner, result = passed()
    owner = deepcopy(owner); owner['identity']['created'] += 1
    with pytest.raises(ValueError):
        saved.validate_passed_execution(execution, owner, result, mode=saved.INDEPENDENT, new=True)
    execution, owner, result = passed()
    owner['windows_job_assigned'] = False
    with pytest.raises(ValueError):
        saved.validate_passed_execution(execution, owner, result, mode=saved.INDEPENDENT, new=True)


@pytest.mark.parametrize('new', [True, False])
def test_both_explicit_source_result_schemas_accept_actual_clean_exit(new):
    execution, owner, result = passed(new)
    assert saved.validate_passed_execution(execution, owner, result, mode=execution['mode'], new=new)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    out, source, original = (tmp_path/name for name in ('new', 'source', 'original'))
    monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'SOURCE', source)
    monkeypatch.setattr(c, 'ORIGINAL', original)
    monkeypatch.setattr(saved, 'require_absent', lambda identity: None)
    return out, source, original


def test_historical_receipt_must_not_come_from_finalization_or_another_root(roots):
    out, _, _ = roots
    with pytest.raises(ValueError, match='preserved source'):
        saved.completed_source(out/'segments'/saved.INDEPENDENT, lambda: None)


def new_independent_fixture(roots, monkeypatch):
    out, source, _ = roots
    folder = out/'jobs'/saved.INDEPENDENT
    execution, owner, result = passed()
    put(out/'freeze.json', dict(saved_only=True))
    sha = c.file_hash(out/'freeze.json')
    execution.update(plan_sha256=sha, monotonic_origin=10., monotonic_end=20.,
                     unix_origin=100., unix_end=110., monotonic_deadline=30., unix_deadline=120.)
    owner.update(plan_sha256=sha, created_monotonic=11., monotonic_deadline=30., unix_deadline=120.)
    put(source/'freeze.json', dict(corrected_source_receipt='fixture'))
    put(source/'claim.json', {})
    put(source/'reader_clock.json', {})
    put(out/'claim.json', dict(freeze=c.artifact(out/'freeze.json'), source_claim=c.artifact(source/'claim.json'),
                              original_reader=c.artifact(source/'reader_clock.json'),
                              monotonic_deadline=30., unix_deadline=120.))
    owner.update(global_claim=c.artifact(out/'claim.json'), source_reader_clock=c.artifact(source/'reader_clock.json'),
                 model_work=False)
    result['plan_sha256'] = sha
    coverage = dict(status='verified_amended_full_corpus_assessment', matchup='mixed_terminal', repetition=1)
    result['result'] = coverage
    put(source/'strength/mixed_terminal/r1/verification_summary.json', coverage)
    put(folder/'worker/runtime_sources.json', dict(sources='fixture', identity=result['identity'],
                                                  mode=saved.INDEPENDENT, model_calls=0, new_games=0))
    result['runtime_sources'] = c.artifact(folder/'worker/runtime_sources.json')
    for name, doc in [('owner.json', owner), ('execution.json', execution), ('worker/result.json', result)]:
        put(folder/name, doc)
    monkeypatch.setattr(saved, 'interrupted_source', lambda boundary: dict(actual_exit_code=None))
    clock = dict(monotonic_deadline=40., unix_deadline=130.)
    monkeypatch.setattr(saved, 'source_clocks', lambda: ({}, clock, clock))
    return folder


def test_missing_new_execution_is_not_accepted_from_completed_result(roots, monkeypatch):
    folder = new_independent_fixture(roots, monkeypatch)
    (folder/'execution.json').unlink()
    with pytest.raises(FileNotFoundError):
        saved.completed_new_independent(lambda: None)


def test_new_receipt_retains_true_path_and_does_not_claim_source_exit(roots, monkeypatch):
    folder = new_independent_fixture(roots, monkeypatch)
    proof = saved.completed_new_independent(lambda: None)
    assert proof['execution'] == c.artifact(folder/'execution.json')
    assert proof['result'] == c.artifact(folder/'worker/result.json')
    assert proof['source'] == 'saved_only_finalization'
    assert proof['replaced_historical_exit'] is False
    assert proof['interrupted_source_exit_still_unknown'] is True


@pytest.mark.parametrize('case', ['wrong_plan', 'different_coverage', 'extended_clock', 'end_at_deadline'])
def test_new_substitution_cannot_change_freeze_coverage_or_deadlines(roots, monkeypatch, case):
    folder = new_independent_fixture(roots, monkeypatch)
    path = folder/('worker/result.json' if case == 'different_coverage' else 'execution.json')
    value = c.read(path)
    if case == 'wrong_plan': value['plan_sha256'] = '0'*64
    elif case == 'different_coverage': value['result']['repetition'] = 0
    elif case == 'extended_clock': value['monotonic_deadline'] = 41.
    else: value['monotonic_end'] = value['monotonic_deadline']
    put(path, value)
    with pytest.raises(ValueError):
        saved.completed_new_independent(lambda: None)


def test_only_disclosed_missing_slot_can_be_substituted(roots, monkeypatch):
    _, source, _ = roots
    names = ['saved_mixed_terminal_r0', 'saved_mixed_terminal_r1', saved.INDEPENDENT]
    for name in names[:-1]: put(source/'segments'/name/'execution.json', {})
    monkeypatch.setattr(saved, 'completed_source', lambda folder, boundary: ('source', folder.name))
    monkeypatch.setattr(saved, 'completed_new_independent', lambda boundary: ('new', saved.INDEPENDENT))
    assert saved.receipt_mapping(names, lambda: None) == [('source', names[0]), ('source', names[1]), ('new', names[2])]
    (source/'segments'/names[0]/'execution.json').unlink()
    with pytest.raises(ValueError, match='only the disclosed'):
        saved.receipt_mapping(names, lambda: None)


def test_source_terminal_receipt_cannot_be_hidden_by_new_mapping(roots, monkeypatch):
    _, source, _ = roots
    put(source/'segments'/saved.INDEPENDENT/'execution.json', dict(status='failed'))
    with pytest.raises(ValueError, match='only the disclosed'):
        saved.receipt_mapping([saved.INDEPENDENT], lambda: None)


def test_duplicate_logical_slot_rejected_before_mapping(roots):
    with pytest.raises(ValueError, match='One exact'):
        saved.receipt_mapping([saved.INDEPENDENT, saved.INDEPENDENT], lambda: None)


def test_incomplete_source_does_not_infer_exit_from_process_absence(roots, monkeypatch):
    _, source, _ = roots
    folder = source/'segments'/saved.INDEPENDENT
    put(source/'freeze.json', {})
    put(source/'claim.json', dict(pid=10, created=20.))
    put(source/'stages/final_start.json', {})
    put(source/'reader_clock.json', {})
    claim = dict(mode='independent', matchup='mixed_terminal', repetition=1, block_start=None,
                 segment_id=saved.INDEPENDENT, freeze_sha256=saved.SOURCE_FREEZE_SHA256,
                 claim=c.artifact(source/'claim.json'), phase_clock=c.artifact(source/'stages/final_start.json'),
                 reader_clock=c.artifact(source/'reader_clock.json'))
    put(folder/'claim.json', claim)
    put(folder/'owner.json', dict(identity=dict(pid=11, created=21.), windows_job_assigned=True,
                                  segment_claim=c.artifact(folder/'claim.json')))
    proof = saved.interrupted_source(lambda: None)
    assert proof['exact_processes_absent'] and proof['actual_exit_code'] is None
    put(folder/'result.json', dict(status='passed'))
    with pytest.raises(ValueError, match='terminal receipt'):
        saved.interrupted_source(lambda: None)


def test_saved_envelope_clips_frozen_reader_to_actual_new_worker_deadline(monkeypatch):
    claim = dict(monotonic_origin=0., unix_origin=100., total_seconds=432000)
    phase = dict(monotonic_deadline=70., unix_deadline=170.)
    reader = dict(monotonic_deadline=60., unix_deadline=160.)
    monkeypatch.setattr(saved, 'source_clocks', lambda: (claim, phase, reader))
    monkeypatch.setattr(saved.time, 'monotonic', lambda: 10.)
    monkeypatch.setattr(saved.time, 'time', lambda: 110.)
    class Guard:
        limits = dict(monotonic_deadline=50., unix_deadline=150.)
        def __call__(self): pass
    envelope = saved.ReadOnlyEnvelope(Guard())
    assert envelope.deadline == 50. and envelope.wall_deadline == 150.
    assert envelope.remaining() == 40. and envelope.effective_deadline() == 50.
    assert envelope.worker_accounting == {} and envelope.resident == {}
    with pytest.raises(ValueError, match='Only existing'):
        envelope.close_worker(c.OUT/'workers/new_worker')
    with pytest.raises(ValueError, match='Only existing'):
        envelope.close_worker(c.SOURCE/'strength/mixed_terminal/r1/workers/x', child=object())


def test_actual_existing_closure_is_read_only_and_original_clocks_authenticate():
    claim, phase, reader = saved.source_clocks()
    assert reader['monotonic_deadline'] <= phase['monotonic_deadline'] <= claim['monotonic_origin']+432000
    envelope = saved.ReadOnlyEnvelope(lambda: None)
    folder = c.SOURCE/'strength/mixed_terminal/r1/workers/qualification_control'
    before = c.artifact(folder/'cleanup.json')
    assert envelope.close_worker(folder) is None
    assert c.artifact(folder/'cleanup.json') == before


@pytest.mark.parametrize('source,name', [
    ('original', 'qualification_mixed_terminal_r0'),
    ('original', 'main_block_mixed_terminal_r0_g020'),
    ('recovery', 'prepare_mixed_terminal_r0'),
    ('recovery', 'replay_mixed_terminal_r0'),
    ('recovery', 'saved_mixed_terminal_r1'),
    ('recovery', 'independent_mixed_terminal_r0'),
    ('recovery', 'aggregate_mixed_terminal_r0')])
def test_actual_saved_receipt_shapes_map_without_relabeling(source, name):
    folder = (c.ORIGINAL if source == 'original' else c.SOURCE)/'segments'/name
    before = c.artifact(folder/'execution.json')
    proof = saved.completed_source(folder, lambda: None)
    assert proof['execution'] == before
    assert proof['logical_segment'] == name
    assert proof['source'] == ('original' if source == 'original' else 'amended_recovery')
    assert c.artifact(folder/'execution.json') == before


def test_independent_calls_frozen_verifier_without_writing_or_new_results(roots, monkeypatch):
    _, source, _ = roots
    monkeypatch.setenv('FULL_CORPUS_MATCHUP', 'mixed_terminal')
    monkeypatch.setenv('INDEPENDENT_DATA_REPETITION', '1')
    from tools.splendorust_full_corpus_recovery_20261111.strength import control
    coverage = dict(status='verified_amended_full_corpus_assessment', matchup='mixed_terminal', repetition=1,
                    main_games=200, operational_games=6)
    put(source/'strength/mixed_terminal/r1/verification_summary.json', coverage)
    before = {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    sentinel = object(); calls = []
    monkeypatch.setattr(saved, '_context', lambda repetition: (None, {'frozen': True}))
    monkeypatch.setattr(saved, 'interrupted_source', lambda boundary: None)
    monkeypatch.setattr(saved, 'ReadOnlyEnvelope', lambda boundary: sentinel)
    def verify(envelope, freeze, *, writing):
        calls.append((envelope, freeze, writing)); return coverage
    monkeypatch.setattr(control, 'verify_saved', verify)
    assert saved.run_independent(lambda: None) == coverage
    assert calls == [(sentinel, {'frozen': True}, False)]
    assert before == {p.relative_to(source): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert not c.OUT.exists()
