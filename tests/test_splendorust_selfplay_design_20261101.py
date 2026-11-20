"""Prospective sample isolation and exclusive-evidence boundary regressions."""
from copy import deepcopy
import pytest
from tools.splendorust_selfplay_20261101 import common as c, design
from tools.future_training_rules_20261021 import decode_training_snapshot

def test_four_fresh_corrected_engineering_starts_and_disjoint_splits():
    specs = design.schedule()
    assert len({s['seed_decimal'] for s in specs}) == 4
    assert [s['split'] for s in specs] == ['fixture_train','fixture_train','fixture_validation','fixture_assessment']
    for item in design.initial_states():
        assert design.validate_spec(item['spec']) == item['spec']
        state = decode_training_snapshot(item['initial'])
        assert state.turn_index == 0 and not state.terminal
        assert item['spec']['purpose'] == 'engineering_qualification_only'

@pytest.mark.parametrize('key,value', [('game_index', True),('split','train'),('seed_decimal','42'),
                                      ('teacher_simulations',1024),('purpose','single_generation_learning_pilot')])
def test_cannot_relabel_or_substitute_qualification_game(key, value):
    spec = deepcopy(design.schedule()[0]); spec[key] = value
    with pytest.raises(ValueError):
        design.validate_spec(spec)

def test_evidence_is_exclusive_and_stays_in_declared_output(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'qualification')
    p = c.OUT/'receipt.json'
    item = c.save(p, {'verified':True})
    c.verify([item])
    with pytest.raises(FileExistsError):
        c.save(p, {'verified':False})
    with pytest.raises(ValueError):
        c.save(tmp_path/'frozen_other_study'/'receipt.json', {})
    p.write_text('{"verified":false}', encoding='utf-8')
    with pytest.raises(ValueError):
        c.verify([item])

def test_artifact_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    with pytest.raises(ValueError):
        c.verify([{'path':'../outside.json','bytes':0,'sha256':'0'*64}])

def test_exact_predecessor_external_runtime_inventory_remains_authenticated():
    from pathlib import Path
    items = [x for x in c.read(c.ACCEPTED_OUT/'freeze.json')['inputs'] if Path(x['path']).is_absolute()]
    assert len(c.verify_external_runtime(items)) == 13
    wrong = deepcopy(items)
    wrong[0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Exact thirteen'):
        c.verify_external_runtime(wrong)
    with pytest.raises(ValueError, match='Exact thirteen'):
        c.verify_external_runtime(items[:-1])
    with pytest.raises(ValueError, match='Canonical workspace'):
        c.verify([items[0]])
