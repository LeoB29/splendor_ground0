"""Admission and immutable-evidence regressions; no model execution."""
from copy import deepcopy
import pytest
from tools.splendorust_network_live_20261031 import common as c,preparation as p


def test_prior_accepted_record_does_not_claim_bit_exact_or_deployment():
    closed=c.read(c.PRACTICAL_CLOSEOUT)
    assert c.file_hash(c.PRACTICAL_CLOSEOUT)==c.PRACTICAL_CLOSEOUT_SHA
    assert closed['useful_speedup'] and not closed['full_numerical_outputs_exact']
    assert not closed['deployment_qualified'] and not closed['strength_equivalence_established']
    assert c.file_hash(c.CANDIDATE_SOURCE)==c.CANDIDATE_SOURCE_SHA


def test_new_limits_keep_original_move_guard_and_terminal_safety():
    assert (c.INTERNAL_SECONDS,c.SECONDS_PER_MOVE)==(9.,10.)
    assert c.MAX_PLIES==2000 and c.MAX_LIVE_DECISIONS==6*c.MAX_PLIES
    assert c.MAX_WORKER_LIFECYCLES==4 and len(c.LIFECYCLES)==4
    assert 0<c.COLLECTION_CUTOFF_SECONDS<c.JOB_SECONDS
    assert c.PREFINAL_EVIDENCE_BYTES+c.FINAL_RESERVE_BYTES==c.MAX_EVIDENCE_BYTES


def test_preparation_does_not_replace_existing_evidence(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path)
    monkeypatch.setattr(c,'ROOT',tmp_path)
    path=tmp_path/'held.json';p.same_or_save(path,{'state':'original'})
    before=path.read_bytes()
    with pytest.raises(ValueError,match='Never replace'):
        p.same_or_save(path,{'state':'replacement'})
    assert path.read_bytes()==before


@pytest.mark.parametrize('changed',[
    {'approved':False},{'blocking_findings':['missing raw response']},{'inputs':[]}])
def test_review_cannot_approve_missing_or_failed_checks(tmp_path,monkeypatch,changed):
    record=dict(approved=True,blocking_findings=[],inputs=[{'placeholder':True}])
    record.update(changed);monkeypatch.setattr(c,'read',lambda _:record)
    with pytest.raises(ValueError,match='Independent current-source review'):
        p.require_review(tmp_path/'review.json')


def test_review_rejects_changed_source_hash(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('original')
    monkeypatch.setattr(c,'ROOT',tmp_path)
    descriptor=c.artifact(source)
    record=dict(approved=True,blocking_findings=[],inputs=[descriptor])
    monkeypatch.setattr(c,'read',lambda _:record)
    source.write_text('changed')
    with pytest.raises(ValueError,match='Changed frozen input'):
        p.require_review(tmp_path/'review.json')


@pytest.mark.parametrize('patch',[
    {'exit_code':1},{'models':1},{'games':1},{'inputs':[]}])
def test_admission_requires_actual_passed_model_free_tests(tmp_path,monkeypatch,patch):
    record=dict(exit_code=0,models=0,games=0,inputs=[{'placeholder':True}],log={})
    record.update(patch);monkeypatch.setattr(c,'read',lambda _:record)
    with pytest.raises(ValueError,match='Passed actual model-free tests'):
        p.require_tests(tmp_path/'tests.json')


def test_freeze_never_overwrites_existing_attempt(tmp_path,monkeypatch):
    # Import only after control is available in the completed new implementation.
    monkeypatch.setattr(c,'OUT',tmp_path)
    path=tmp_path/'freeze.json';path.write_text('preserved freeze')
    with pytest.raises(ValueError,match='One exclusive'):
        p.freeze()
    assert path.read_text()=='preserved freeze'
