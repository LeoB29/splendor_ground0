"""Meaningful acceptance-harness checks without simulation or model loading."""
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import integration as h


def example_record():
    return {'time': 'a', 'timing': {'wall': 1}, 'assessment_started_at_ns': 10,
            'selection_artifact': {'path': 'a', 'sha256': 'a'},
            'proposals': {'anchor': {'timing': {'wall': 1}, 'logits': [1, 2]}},
            'selection': {'timing': {'wall': 1}, 'worlds': [{'winner': 0, 'trace_sha256': 'trace'}]},
            'assessment': {'timing': {'wall': 1}, 'worlds': [{'winner': 1, 'final_state_sha256': 'end'}]},
            'selection_lock': {'time': 'a', 'created_at_ns': 1, 'selection_sha256': 'a', 'choices': [0]},
            'proof_timing': {'wall': 2, 'cpu': 1, 'replayed_continuations': 3},
            'summary': {'scores': [0.5]}, 'unknown_future_science': 'preserved'}


def test_comparison_excludes_only_explicit_operational_differences():
    a = example_record(); b = deepcopy(a)
    b['time'] = 'b'; b['timing']['wall'] = 9; b['assessment_started_at_ns'] = 20
    b['selection_artifact'] = {'path': 'other', 'sha256': 'other'}
    b['selection']['timing']['wall'] = 7; b['assessment']['timing']['wall'] = 8
    b['selection_lock'].update(time='b', created_at_ns=9, selection_sha256='different_raw_timing_hash')
    b['proposals']['anchor']['timing']['wall'] = 5
    b['proof_timing'].update(wall=8, cpu=5)
    assert h.canonical_record(a) == h.canonical_record(b)
    assert a == example_record(), 'Normalization must not modify original evidence'


@pytest.mark.parametrize('field', ['trace', 'final', 'logits', 'choice', 'summary', 'proof_count', 'unknown'])
def test_comparison_detects_substantive_changes(field):
    a = example_record(); b = deepcopy(a)
    if field == 'trace': b['selection']['worlds'][0]['trace_sha256'] = 'wrong'
    if field == 'final': b['assessment']['worlds'][0]['final_state_sha256'] = 'wrong'
    if field == 'logits': b['proposals']['anchor']['logits'][0] = 3
    if field == 'choice': b['selection_lock']['choices'][0] = 1
    if field == 'summary': b['summary']['scores'][0] = 0.6
    if field == 'proof_count': b['proof_timing']['replayed_continuations'] = 4
    if field == 'unknown': b['unknown_future_science'] = 'changed'
    assert h.canonical_record(a) != h.canonical_record(b)


@pytest.mark.parametrize('stage,complete,expected', [
    ('learning_fit_audits', False, True), ('learning_fits', False, True),
    ('main_collection', False, False), ('source_main', False, False),
    (None, False, False), ('complete', True, True)])
def test_main_phase_firewall(stage, complete, expected):
    assert h.phase_allowed(stage, complete) is expected


def test_resource_policy_tolerates_low_physical_ram_but_not_commit_pressure(monkeypatch, tmp_path):
    monkeypatch.setattr(h, 'MAIN', tmp_path)
    h.save(tmp_path / 'progress.json', {'stage': 'learning_fit_audits'})
    info = {'available_ram_bytes': 1, 'process_rss_bytes': 10, 'process_peak_rss_bytes': 20,
            'process_private_bytes': 30, 'commit_headroom_bytes': 10 * 1024**3}
    monkeypatch.setattr(h.util, 'memory', lambda: dict(info))
    monkeypatch.setattr(h.shutil, 'disk_usage', lambda _: type('D', (), {'free': 6 * 1024**3})())
    assert h.operational_guard(tmp_path, admission=True)['available_ram_bytes'] == 1
    info['commit_headroom_bytes'] = 5 * 1024**3
    with pytest.raises(RuntimeError, match='commitment'): h.operational_guard(tmp_path)


def test_resume_dispatches_only_missing_jobs(monkeypatch, tmp_path):
    plan = {'binding': 'fixture'}
    completed_keys = {x['key']: {} for x in h.SCHEDULE[:3]}
    calls = []
    monkeypatch.setattr(h, 'load_plan', lambda _: (tmp_path, plan))
    monkeypatch.setattr(h, 'exclusive', lambda _: nullcontext())
    monkeypatch.setattr(h, 'no_prior_workers', lambda _: None)
    monkeypatch.setattr(h, 'completed', lambda *args: dict(completed_keys))
    def dispatch(output, actual, task):
        assert task['key'] not in completed_keys
        calls.append(task['key']); completed_keys[task['key']] = {}
    monkeypatch.setattr(h, 'run_job', dispatch)
    monkeypatch.setattr(h, 'finish', lambda *args: {'all_exact': True, 'net_wall_ratio': 1.2})
    h.run(tmp_path)
    assert calls == [x['key'] for x in h.SCHEDULE[3:]]
    h.run(tmp_path)
    assert calls == [x['key'] for x in h.SCHEDULE[3:]], 'Second resume must not duplicate complete work'


def test_corrupt_completion_is_rejected(tmp_path):
    task = h.SCHEDULE[0]
    h.save(tmp_path / 'completed' / (task['key'] + '.json'),
           {'binding': 'wrong', 'task': task, 'exit_code': 0})
    with pytest.raises(ValueError, match='completion'): h.completed(tmp_path, {'binding': 'expected'})


def test_output_never_points_to_main_or_prior_validation():
    for path in (h.MAIN, h.OLD, h.VALIDATION, h.ROOT / 'tools', h.ROOT / 'outputs/simulation_backend_integration_128_01/child'):
        with pytest.raises(ValueError): h.safe_output(path)


def test_failed_and_incomplete_attempts_are_charged(monkeypatch, tmp_path):
    h.save(tmp_path / 'attempts/failed/start.json', {'epoch': 2})
    h.save(tmp_path / 'attempts/failed/end.json', {'seconds': 4})
    h.save(tmp_path / 'attempts/orphan/start.json', {'epoch': 10})
    monkeypatch.setattr(h.time, 'time', lambda: 20)
    assert h.active_used(tmp_path) == 14


def test_audit_receipt_crash_gap_retries_in_new_attempt_and_restores(tmp_path):
    original = tmp_path / 'collector/audits/receipt.json'
    saved = []
    def write(path, value):
        h.save(path, value); saved.append(Path(path))
    def ref(path, **metadata):
        return {'path': str(path), **metadata}
    common = SimpleNamespace(old=SimpleNamespace(save_new=write), artifact=ref)
    first, retry = tmp_path / 'attempt1/audit.json', tmp_path / 'attempt2/audit.json'
    with pytest.raises(RuntimeError, match='crash'):
        with h.audit_receipt_destination(common, original, first):
            common.old.save_new(original, {'passed': True})
            assert common.artifact(original, index=0) == {'path': str(first), 'index': 0}
            raise RuntimeError('crash before job result')
    assert first.exists() and not original.exists()
    assert common.old.save_new is write and common.artifact is ref
    with h.audit_receipt_destination(common, original, retry):
        common.old.save_new(original, {'passed': True, 'fresh_replay': True})
        assert common.artifact(original, index=0) == {'path': str(retry), 'index': 0}
        assert common.artifact(tmp_path / 'unchanged_root.json')['path'] == str(tmp_path / 'unchanged_root.json')
    assert h.read(first) == {'passed': True}
    assert h.read(retry)['fresh_replay'] and saved == [first, retry]
    assert common.old.save_new is write and common.artifact is ref
