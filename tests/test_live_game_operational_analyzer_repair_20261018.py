"""Separate v2 regressions: authoritative starting envelope and repair lineage."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import analyze_live_game_operational_20261018 as original
from tools import analyze_live_game_operational_20261018_v2 as repaired
from tools.analyze_external_strength_screen_20261018_02 import initial_snapshot
from tools.external_strength_screen_20261018 import rules, audit as ledger


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class InitialComparisonPassed(Exception): pass


def initial_audit(module, tmp_path, monkeypatch, seed, damage=None):
    catalogue = original.read(original.ROOT/'outputs/external_strength_screen_20261018_02/catalogue.json')
    spec = {**original.schedule()[0], 'seed': str(seed)}
    snapshot = rules.serialize_snapshot(rules.CorrectedEnv(seed=int(seed)).initial_state())
    assert set(snapshot) == {'schema', 'ruleset_id', 'catalogue_sha256', 'state'}
    assert snapshot['ruleset_id'] == repaired.RULESET
    value = {**spec, 'initial': snapshot, 'initial_sha256': original.digest(snapshot)}
    if damage == 'bank': value['initial']['state']['bank_tokens']['white'] += 1
    elif damage == 'deck': value['initial']['state']['hidden_tier_decks']['1'].reverse()
    elif damage == 'nobles': value['initial']['state']['nobles'].reverse()
    elif damage == 'envelope': value['initial']['ruleset'] = value['initial'].pop('ruleset_id')
    elif damage == 'hash': value['initial_sha256'] = '0'*64
    elif damage == 'schedule': value['fit'] = 161002
    elif damage == 'state_and_hash':
        value['initial']['state']['bank_tokens']['gold'] += 1
        value['initial_sha256'] = original.digest(value['initial'])
    path = tmp_path/'sealed/games/000/initial.json'; write(path, value)
    old_read = module.read
    def reader(path):
        if Path(path).name == 'result.json': raise InitialComparisonPassed()
        return old_read(path)
    monkeypatch.setattr(module, 'read', reader)
    audit = module.Audit(tmp_path); audit.initial_snapshot = initial_snapshot
    audit.catalogue = catalogue; audit.ledger = ledger
    return audit, spec


@pytest.mark.parametrize('seed', [0, 17, *[int(s['seed']) for s in original.schedule()[:3]]])
def test_authoritative_literal_envelope_and_entire_initial_world_pass(tmp_path, monkeypatch, seed):
    audit, spec = initial_audit(repaired, tmp_path, monkeypatch, seed)
    with pytest.raises(InitialComparisonPassed): audit.game(spec)


def test_preserved_original_bug_is_reproduced_only_on_synthetic_initial_fixture(tmp_path, monkeypatch):
    audit, spec = initial_audit(original, tmp_path, monkeypatch, 17)
    with pytest.raises(ValueError, match='Independent paired initial world'): audit.game(spec)


@pytest.mark.parametrize('damage', ['bank', 'deck', 'nobles', 'envelope', 'hash', 'schedule', 'state_and_hash'])
def test_initial_world_corruption_is_still_rejected(tmp_path, monkeypatch, damage):
    audit, spec = initial_audit(repaired, tmp_path, monkeypatch, 17, damage)
    with pytest.raises(ValueError, match='Independent paired initial world'): audit.game(spec)


def methods(module):
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    return {node.name: node for node in next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Audit').body
            if isinstance(node, ast.FunctionDef)}


def test_scientific_game_check_changes_only_the_envelope_key_and_other_gates_remain_identical():
    before, after = methods(original), methods(repaired)
    changed = 0
    for node in ast.walk(before['game']):
        if isinstance(node, ast.Constant) and node.value == 'ruleset': node.value = 'ruleset_id'; changed += 1
    assert changed == 1 and ast.dump(before['game']) == ast.dump(after['game'])
    for name in set(before)-{'__init__', 'freeze', 'game', 'run'}:
        assert ast.dump(before[name]) == ast.dump(after[name]), name
    # In the final report only a separate repair-authentication descriptor is added.
    final_return = next(n for n in after['run'].body if isinstance(n, ast.Return))
    index = next(i for i, key in enumerate(final_return.value.keys) if key.value == 'analyzer_repair')
    del final_return.value.keys[index]; del final_return.value.values[index]
    assert ast.dump(before['run']) == ast.dump(after['run'])


def repair_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(repaired, 'ROOT', tmp_path)
    source = tmp_path/'tools/analyze_live_game_operational_20261018_v2.py'
    monkeypatch.setattr(repaired, '__file__', str(source))
    base = tmp_path/'outputs/test'; folder = base/'analyzer_repair_02'
    old = tmp_path/repaired.ORIGINAL_ANALYZER; write(old, {'preserved': 'source'})
    monkeypatch.setattr(repaired, 'ORIGINAL_SHA256', repaired.file_hash(old))
    write(base/'freeze.json', {'preserved': 'original freeze'})
    write(base/'launch.json', {'epoch_seconds': 1792356838.6393518, 'maximum_seconds': 43200})
    write(base/'completion.json', {'preserved': 'completion'})
    write(base/'final_input_check.json', {'preserved': 'input check'})
    failure_names = ('independent_analysis_execution_01.log', 'independent_analysis_execution_01.json', 'sealed/independent_analysis_failure_01.json')
    for name in failure_names: write(base/name, {'preserved': name})
    inputs = [source, tmp_path/repaired.REPAIR_TEST, tmp_path/repaired.REPAIR_PROTOCOL,
              folder/'tests.json', folder/'tests.log', folder/'analyzer.diff']
    for path in inputs: write(path, {'returncode': 0} if path.name == 'tests.json' else {'content': str(path)})
    value = {'schema': 'live_operational_analyzer_repair_v2', 'original_freeze': repaired.artifact(base/'freeze.json'),
        'original_analyzer': repaired.artifact(old), 'original_launch': repaired.artifact(base/'launch.json'),
        'original_completion': repaired.artifact(base/'completion.json'),
        'original_final_input_check': repaired.artifact(base/'final_input_check.json'),
        'original_deadline_epoch_seconds': 1792356838.6393518+43200, 'maximum_seconds': 43200, 'final_reserve_seconds': 600,
        'preserved_failure': [repaired.artifact(base/n) for n in failure_names], 'inputs': [repaired.artifact(p) for p in inputs]}
    write(folder/'freeze.json', value)
    return base, folder, value, inputs


def test_separate_freeze_authenticates_v2_tests_protocol_originals_and_failures(tmp_path, monkeypatch):
    base, folder, value, inputs = repair_fixture(tmp_path, monkeypatch)
    result = repaired.verify_repair_freeze(base, repaired.file_hash(folder/'freeze.json'))
    assert result['all_hashes_and_sizes_match'] is True and result['files'] == 6


@pytest.mark.parametrize('damage', ['no_hash', 'wrong_hash', 'source', 'missing_protocol', 'duplicate', 'failure',
                                   'original', 'deadline', 'reserve', 'completion', 'failed_tests'])
def test_repair_authentication_rejects_tampering_or_new_execution_allowance(tmp_path, monkeypatch, damage):
    base, folder, value, inputs = repair_fixture(tmp_path, monkeypatch)
    if damage == 'source': write(inputs[0], {'tampered': True})
    elif damage == 'missing_protocol': value['inputs'] = [i for i in value['inputs'] if i['path'] != repaired.REPAIR_PROTOCOL]
    elif damage == 'duplicate': value['inputs'].append(deepcopy(value['inputs'][0]))
    elif damage == 'failure': write(base/'independent_analysis_execution_01.json', {'tampered': True})
    elif damage == 'original': write(tmp_path/repaired.ORIGINAL_ANALYZER, {'tampered': True})
    elif damage == 'deadline': value['original_deadline_epoch_seconds'] += 1
    elif damage == 'reserve': value['final_reserve_seconds'] = 0
    elif damage == 'completion': write(base/'completion.json', {'tampered': True})
    elif damage == 'failed_tests':
        write(folder/'tests.json', {'returncode': 1})
        value['inputs'] = [repaired.artifact(p) for p in inputs]
    write(folder/'freeze.json', value)
    expected = None if damage == 'no_hash' else '0'*64 if damage == 'wrong_hash' else repaired.file_hash(folder/'freeze.json')
    with pytest.raises(ValueError): repaired.verify_repair_freeze(base, expected)


def test_v2_failure_preserves_original_failure_and_uses_new_name(tmp_path, monkeypatch, capsys):
    original_failure = tmp_path/'sealed/independent_analysis_failure_01.json'; write(original_failure, {'preserved': True})
    before = original_failure.read_bytes()
    monkeypatch.setattr(repaired.sys, 'argv', ['analyzer', '--base', str(tmp_path), '--repair-freeze-sha256', '0'*64])
    def fail(self): raise ValueError('SEALED_V2_FIXTURE')
    monkeypatch.setattr(repaired.Audit, 'run', fail)
    with pytest.raises(ValueError): repaired.main()
    assert original_failure.read_bytes() == before
    assert 'SEALED_V2_FIXTURE' in repaired.read(tmp_path/'sealed/independent_analysis_failure_02.json')['traceback']
    captured = capsys.readouterr(); assert 'SEALED_V2_FIXTURE' not in captured.out+captured.err
    assert not (tmp_path/'independent_analysis_v2.json').exists()


def test_v2_success_default_is_separate_and_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(repaired.sys, 'argv', ['analyzer', '--base', str(tmp_path), '--repair-freeze-sha256', '0'*64])
    monkeypatch.setattr(repaired.Audit, 'run', lambda self: {'status': 'passed', 'outcomes_closed': True})
    monkeypatch.setattr(repaired.Audit, 'boundary', lambda self: None)
    repaired.main()
    output = tmp_path/'independent_analysis_v2.json'
    assert output.exists() and not (tmp_path/'independent_analysis.json').exists()
    before = output.read_bytes()
    with pytest.raises(ValueError, match='Never overwrite'): repaired.main()
    assert output.read_bytes() == before
