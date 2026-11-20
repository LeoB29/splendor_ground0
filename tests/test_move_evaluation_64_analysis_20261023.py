"""Model-free rejection and exact bootstrap-stream checks for A64/B64."""
from copy import deepcopy

import pytest

from tools.move_evaluation_64_20261023 import analysis as a
from tools.move_evaluation_20261023 import audit as independent


def fixture():
    roots, rows = [], []
    for index in range(18):
        fit = a.FITS[index//6]
        aa = [(index+j)%3 for j in range(32)]
        bb = [(index+2*j)%3 for j in range(32)]
        roots.append({'id': f'r{index}', 'fit': fit, 'control_choice': 0,
                      'candidates': {'actions': [{'action': 0}, {'action': 1}]}})
        rows.append({'id': f'r{index}', 'fit': fit, 'control_choice': 0, 'alternative_choice': 1,
                     'a_scores_halfpoints': aa, 'b_scores_halfpoints': bb,
                     'differences_halfpoints': [b-a for a, b in zip(aa, bb)]})
    for index in (1, 7, 13):
        rows[index].update(alternative_choice=0, a_scores_halfpoints=None, b_scores_halfpoints=None,
                           differences_halfpoints=[0]*32)
    return rows, {'roots': roots}


def test_exact_primary_and_independent_stream_keeps_structural_zero_draws():
    rows, lock = fixture()
    primary = a.calculate(rows, lock)
    reference = independent.statistics(rows, lock, checks_complete=True)
    assert all(value == reference[key] for key, value in primary.items())
    assert reference['structural_zero_roots'] == 3
    assert reference['denominator'] == 1152
    assert reference['terminal_branches'] == 960
    assert reference['population_strength_claim'] is False


@pytest.mark.parametrize('damage', ['missing', 'order', 'duplicate', 'fit', 'extra', 'control', 'choice_bool',
                                    'choice_range', 'short', 'boolean', 'float', 'arithmetic', 'imputed', 'breadth'])
def test_invalid_or_incomplete_data_rejected_before_bootstrap(damage):
    rows, lock = fixture()
    if damage == 'missing': rows.pop()
    elif damage == 'order': rows[0], rows[1] = rows[1], rows[0]
    elif damage == 'duplicate': lock['roots'][1]['id'] = lock['roots'][0]['id']
    elif damage == 'fit': rows[0]['fit'] = 161002
    elif damage == 'extra': rows[0]['new_field'] = 'forbidden'
    elif damage == 'control': lock['roots'][0]['control_choice'] = 1
    elif damage == 'choice_bool': rows[0]['alternative_choice'] = True
    elif damage == 'choice_range': rows[0]['alternative_choice'] = 2
    elif damage == 'short': rows[0]['differences_halfpoints'].pop()
    elif damage == 'boolean': rows[0]['differences_halfpoints'][0] = True
    elif damage == 'float': rows[0]['a_scores_halfpoints'][0] = 0.0
    elif damage == 'arithmetic': rows[0]['differences_halfpoints'][0] = 2
    elif damage == 'imputed': rows[1]['a_scores_halfpoints'] = [0]*32
    else:
        for row in rows:
            row.update(alternative_choice=0, a_scores_halfpoints=None, b_scores_halfpoints=None,
                       differences_halfpoints=[0]*32)
    with pytest.raises(ValueError):
        a.validate_rows(rows, lock)


def test_exact_comparison_rejects_changed_bootstrap_stream(monkeypatch, tmp_path):
    from tools.matched_strength_20261020.statistics import runtime_contract
    from tools.move_evaluation_64_20261023 import common as c
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    c.save(tmp_path/'statistics_runtime.json', runtime_contract())
    rows, lock = fixture()
    monkeypatch.setattr(a, 'calculate', lambda *args: {'bootstrap_integer_totals_sha256': 'primary'})
    monkeypatch.setattr(independent, 'statistics', lambda *args, **kwargs: {'bootstrap_integer_totals_sha256': 'different'})
    with pytest.raises(ValueError, match='stream'):
        a.calculate_and_verify(rows, lock)


def test_runtime_change_blocks_calculation(monkeypatch, tmp_path):
    from tools.move_evaluation_64_20261023 import common as c
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path)
    c.save(tmp_path/'statistics_runtime.json', {'changed': True})
    monkeypatch.setattr(a, 'calculate', lambda *args: pytest.fail('Runtime guard must precede calculation'))
    rows, lock = fixture()
    with pytest.raises(ValueError, match='runtime changed'):
        a.calculate_and_verify(rows, lock)
