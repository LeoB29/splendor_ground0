"""Actual recovery strength route: inherited clocks, source data and exact science."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault('INDEPENDENT_DATA_REPETITION', '0')
from tools.splendorust_independent_data_recovery_20261109 import common as root
from tools.splendorust_independent_data_recovery_20261109.strength import common as c
from tools.splendorust_independent_data_recovery_20261109.strength import control, checkpoint, design, envelope, transport
from tools.splendorust_independent_data_20261107 import fit as original_fit, schedule as original_schedule

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'tools/splendorust_independent_data_20261107/strength'
NEW = ROOT / 'tools/splendorust_independent_data_recovery_20261109/strength'


def test_only_declared_recovery_files_differ_from_frozen_strength():
    changed = {'common.py', 'checkpoint.py', 'design.py', 'transport.py', 'control.py', 'envelope.py'}
    assert {p.name for p in OLD.glob('*.py')} == {p.name for p in NEW.glob('*.py')}
    actual = {p.name for p in OLD.glob('*.py') if p.read_bytes() != (NEW / p.name).read_bytes()}
    assert actual == changed


def test_network_search_rules_and_statistics_unchanged():
    for name in ('adapters.py', 'graph.py', 'worker.py', 'qualification.py', 'scientific.py',
                 'corpus.py', 'analysis.py', 'accounting.py'):
        assert (OLD / name).read_bytes() == (NEW / name).read_bytes(), name
    assert design.reservation is original_schedule
    assert transport.OBSERVER_MODULE == root.PACKAGE + '.strength.worker'
    assert c.SOURCE_OUT == root.SOURCE_OUT
    assert c.STUDY_OUT == root.OUT and c.OUT.is_relative_to(root.OUT)
    assert not c.OUT.is_relative_to(c.SOURCE_OUT)


def test_controller_sole_code_change_is_phase_translation():
    before = ast.parse((OLD / 'control.py').read_text())
    after = ast.parse((NEW / 'control.py').read_text())
    for tree in (before, after):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_segment_start')
        fn.body = [n for n in fn.body if not (isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Name) and t.id == 'expected_phase' for t in n.targets))]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


def phase_fixture(monkeypatch, tmp_path, mode):
    out = tmp_path / 'recovery'
    monkeypatch.setattr(root, 'ROOT', tmp_path); monkeypatch.setattr(root, 'OUT', out)
    monkeypatch.setattr(root, 'WRITE_OBSERVER', None)
    monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'OUT', out / 'strength/r0')
    monkeypatch.setattr(c, 'STUDY_OUT', out)
    claim = root.save(out / 'claim.json', {'monotonic_origin': 1000., 'unix_origin': 100000.})
    stage = c.STAGE_MAP['main' if mode == 'main_block' else 'final' if mode == 'saved' else mode]
    cap = root.STAGES[stage]
    cutoff = root.JOB_SECONDS if stage == 'final' else root.MODEL_SECONDS
    clock = {'stage': stage, 'claim': claim, 'monotonic': 2000., 'unix_time': 101000.,
             'cap_seconds': cap, 'monotonic_deadline': min(2000.+cap, 1000.+cutoff),
             'unix_deadline': min(101000.+cap, 100000.+cutoff), 'original_phase': {'saved': True}}
    descriptor = root.save(out / 'stages' / (stage + '_start.json'), clock)
    root.save(c.OUT / 'freeze.json', {'fixture': True})
    monkeypatch.setattr(control.time, 'monotonic', lambda: 2001.)
    monkeypatch.setattr(control.time, 'time', lambda: 101001.)
    env = SimpleNamespace(claim=claim, phase_receipt=descriptor, stage=stage, start=1000., wall_start=100000.,
        deadline=clock['monotonic_deadline'], wall_deadline=clock['unix_deadline'], segment_claim=None)
    return env, clock


@pytest.mark.parametrize('mode', ['qualification', 'operational', 'main_block', 'replay', 'saved'])
def test_actual_recovery_segment_accepts_shared_phase_without_reset(monkeypatch, tmp_path, mode):
    env, clock = phase_fixture(monkeypatch, tmp_path, mode)
    item = control._segment_start(env, {}, mode, 0 if mode == 'main_block' else None)
    value = root.read(root.resolve(item))
    assert value['phase'] == mode and value['root_phase_clock'] == env.phase_receipt
    assert value['monotonic_deadline'] == clock['monotonic_deadline']
    assert value['unix_deadline'] == clock['unix_deadline']
    assert value['original_monotonic_origin'] == 1000.
    assert value['deadline_reset'] is False and value['active_decision'] is False


@pytest.mark.parametrize('mutation', ['wrong_phase', 'wrong_claim', 'clock_extended', 'expired'])
def test_actual_recovery_start_rejects_phase_identity_and_clock_corruption(monkeypatch, tmp_path, mutation):
    env, clock = phase_fixture(monkeypatch, tmp_path, 'qualification')
    if mutation == 'wrong_phase': env.stage = 'qualification'
    elif mutation == 'wrong_claim': env.claim = {**env.claim, 'sha256': '0' * 64}
    elif mutation == 'clock_extended': env.deadline += 1
    else: monkeypatch.setattr(control.time, 'monotonic', lambda: clock['monotonic_deadline'])
    with pytest.raises(ValueError): control._segment_start(env, {}, 'qualification', None)
    assert not (c.OUT / 'segments').exists()


def test_original_checkpoint_source_never_uses_recovery_output(monkeypatch, tmp_path):
    original = tmp_path / 'preserved'
    recovery = tmp_path / 'new'
    monkeypatch.setattr(c, 'SOURCE_OUT', original)
    monkeypatch.setattr(c, 'STUDY_OUT', recovery)
    monkeypatch.setattr(c, 'artifact', lambda path: {'path': str(path)})
    expected = {'path': str(original / 'fits/r0_broad_joint/worker/checkpoints/096/checkpoint_manifest.json')}
    monkeypatch.setattr(checkpoint, 'committed_fits', lambda **kwargs: {
        'fits': [{'spec': {'fit_id': 'r0_broad_joint'}, 'final_checkpoint': expected}]})
    assert checkpoint.commit_descriptor() == {'path': str(original / 'all_fits_committed.json')}
    assert checkpoint.source_descriptor(0) == expected


def test_commitment_calls_original_fit_validator_and_preserves_original_freeze(monkeypatch, tmp_path):
    original, recovery = tmp_path / 'preserved', tmp_path / 'new'
    monkeypatch.setattr(c, 'ROOT', tmp_path); monkeypatch.setattr(c, 'SOURCE_OUT', original)
    monkeypatch.setattr(c, 'STUDY_OUT', recovery)
    descriptor = lambda path: {'path': Path(path).relative_to(tmp_path).as_posix(), 'bytes': 1, 'sha256': 'a' * 64}
    value = {'schema': 'independent_data_all_fits_committed_v1', 'study_id': checkpoint.SOURCE_STUDY,
             'freeze': descriptor(original / 'freeze.json'), 'selection': descriptor(original / 'selection.json'),
             'source_receipt': {}, 'fits': []}
    selection = {'fixed': True}
    monkeypatch.setattr(c, 'artifact', descriptor); monkeypatch.setattr(c, 'verify', lambda *args, **kwargs: None)
    monkeypatch.setattr(c, 'read', lambda path: deepcopy(value if Path(path).name == 'all_fits_committed.json' else selection))
    seen = []
    monkeypatch.setattr(original_fit, 'validate_commitment', lambda selected, item, **kwargs:
                        seen.append((selected, item)) or deepcopy(value))
    assert checkpoint.committed_fits() == value
    assert seen == [(selection, descriptor(original / 'all_fits_committed.json'))]


def test_preserved_evidence_counts_towards_same_output_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(root, 'OUT', tmp_path)
    monkeypatch.setattr(root, 'preserved_bytes', lambda: 12345)
    (tmp_path / 'new.bin').write_bytes(b'abcdef')
    assert envelope.bytes_used() == 12351
