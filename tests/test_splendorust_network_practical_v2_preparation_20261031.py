"""Fail-closed preparation tests; no torch imports, models or games."""
from pathlib import Path
import json
import pytest
from tools.splendorust_network_practical_v2_20261031 import common as c, preparation as p


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path / 'out')
    c.OUT.mkdir()
    return c.OUT


def test_exclusive_artifact_cannot_be_replaced(store):
    path = store / 'immutable.json'
    first = c.save(path, {'value': 1})
    with pytest.raises(FileExistsError):
        c.save(path, {'value': 2})
    c.verify([first])


def test_changed_pinned_input_is_rejected(store):
    path = store / 'input.json'
    pin = c.save(path, {'value': 1})
    path.write_text('{"value":2}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed frozen input'):
        c.verify([pin])


def test_output_must_remain_inside_new_attempt(store):
    with pytest.raises(ValueError, match='Output outside'):
        c.save(store.parent / 'elsewhere.json', {'bad': True})


def test_existing_preparation_must_be_identical(store):
    path = store / 'panel.json'
    first = p._same_or_save(path, {'fixed': True})
    assert p._same_or_save(path, {'fixed': True}) == first
    with pytest.raises(ValueError, match='Never replace'):
        p._same_or_save(path, {'fixed': False})


def test_failed_test_receipt_does_not_admit_freeze(store):
    path = store / 'tests.json'
    c.save(path, {'exit_code': 1, 'models': 0, 'games': 0})
    with pytest.raises(ValueError, match='Passed actual'):
        p._passed_tests(path)


def test_passed_test_receipt_requires_current_sources(store):
    source = store / 'source.py'
    source.write_text('fixed = 1\n', encoding='utf-8')
    log = store / 'tests.log'
    log.write_text('passed\n', encoding='utf-8')
    path = store / 'tests.json'
    c.save(path, {'exit_code': 0, 'models': 0, 'games': 0,
                  'inputs': [c.artifact(source)], 'log': c.artifact(log)})
    assert p._passed_tests(path)['exit_code'] == 0
    source.write_text('fixed = 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed frozen input'):
        p._passed_tests(path)


def test_review_with_blocker_rejected(store):
    path = store / 'review.json'
    c.save(path, {'approved': True, 'blocking_findings': ['clock reset']})
    with pytest.raises(ValueError, match='Independent implementation review'):
        p._approved_review(path)


def test_atomic_publication_is_exclusive(store):
    path = store / 'control.json'
    c.atomic_save(path, {'once': True})
    assert c.read(path) == {'once': True}
    with pytest.raises(ValueError, match='already published'):
        c.atomic_save(path, {'twice': True})


def test_numerical_limits_are_predeclared_and_head_specific():
    assert p.TOLERANCES == {'policy_atol':1e-5,'policy_rtol':1e-6,'value_atol':1e-6,'value_rtol':0.0}



def test_new_namespace_and_all_resource_limits_unchanged():
    from tools.splendorust_network_practical_20261031 import common as prior
    assert c.OUT != prior.OUT and c.FAILED_OUT == prior.OUT
    for name in ('JOB_SECONDS','COLLECTION_CUTOFF_SECONDS','STAGES','MAX_WORKERS',
                 'MAX_WORKER_LIFECYCLES','SECONDS_PER_MOVE','INTERNAL_SECONDS',
                 'MAX_GRAPH_CASES','MAX_NUMERICAL_CASES','MAX_SCIENTIFIC_QUALIFICATION',
                 'EVALUATION_REQUESTS','INITIAL_AVAILABLE_RAM_BYTES','MINIMUM_AVAILABLE_RAM_BYTES',
                 'RAM_WAIT_SECONDS','MAX_RSS_BYTES','MAX_EVIDENCE_BYTES','FINAL_RESERVE_BYTES'):
        assert getattr(c,name) == getattr(prior,name)


def test_actual_same_saved_assessment_and_tolerances():
    old=c.read(c.FAILED_OUT/'freeze.json')
    selected=p.verify_reused_assessment(old)
    assert selected == c.read(c.FAILED_OUT/'panel.json')


@pytest.mark.parametrize('changed_name',['panel.json','numerical_plan.json','evaluation_matrix.json','qualification_plan.json','selection_rule.json'])
def test_no_reselection_or_threshold_change(tmp_path,monkeypatch,changed_name):
    old=c.read(c.FAILED_OUT/'freeze.json')
    for name in p.REUSED_NAMES:
        (tmp_path/name).write_bytes((c.FAILED_OUT/name).read_bytes())
    monkeypatch.setattr(c,'OUT',tmp_path)
    (tmp_path/changed_name).write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='exact saved input bytes'):
        p.verify_reused_assessment(old)


def test_tolerance_reinterpretation_rejected(tmp_path,monkeypatch):
    old=c.read(c.FAILED_OUT/'freeze.json')
    old['numerical_tolerances']['value_atol']=1e-5
    with pytest.raises(ValueError,match='No result-informed tolerance change'):
        p.verify_reused_assessment(old)
