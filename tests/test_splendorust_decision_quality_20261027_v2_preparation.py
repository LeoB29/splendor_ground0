"""Model-free rejection tests for immutable preparation and resource admission."""
from copy import deepcopy
import pytest

from tools.splendorust_decision_quality_20261027_v2 import preparation as p


def test_accepted_design_fixed_workload_is_held():
    plan,matrix,panel,seal=p.load_design()
    assert plan['resources']['job_seconds']==7200
    assert len(matrix['requests'])==156 and panel['positions']==24
    assert seal['execution_freeze'] is False


def test_changed_workload_rejected_even_if_summary_is_unchanged(monkeypatch):
    read=p.c.read
    def changed(path):
        value=read(path)
        if path.name=='prepared_request_matrix.json':
            value=deepcopy(value);value['requests'].pop()
        return value
    monkeypatch.setattr(p.c,'read',changed)
    with pytest.raises(ValueError,match='Fixed156'):
        p.load_design()


def test_forecast_cannot_override_hard_time_cap(monkeypatch):
    def fake(path):
        if path.name=='performance.json':
            return {'players':{'challenger':{'completed_arrival_seconds':{'max':4.0}}}}
        return {'artifacts':[{'bytes':1000}]}
    monkeypatch.setattr(p.c,'read',fake)
    matrix={'requests':[{'quota':1024} for _ in range(120)]}
    with pytest.raises(ValueError,match='does not fit'):
        p.resource_forecast(matrix,{})


def test_forecast_cannot_override_reserved_evidence(monkeypatch):
    def fake(path):
        if path.name=='performance.json':
            return {'players':{'challenger':{'completed_arrival_seconds':{'max':0.1}}}}
        return {'artifacts':[{'bytes':1024**4}]}
    monkeypatch.setattr(p.c,'read',fake)
    matrix={'requests':[{'quota':1024} for _ in range(120)]}
    with pytest.raises(ValueError,match='does not fit'):
        p.resource_forecast(matrix,{})


def test_original_failed_attempt_is_bound_without_reclassification():
    failed, items = p.preserved_attempt_inputs()
    assert len(failed['inputs']) == 1776
    assert len(items) == 1776 + 145 + 1
    assert p.c.read(p.FAILED/'completion.json')['status'] == 'incomplete'


def test_changed_rejected_response_cannot_be_relabelled_preserved(monkeypatch):
    artifact = p.c.artifact
    target = p.FAILED/'qualification_rows/0002.json'
    def changed(path):
        value = artifact(path)
        if path == target:
            value = dict(value, sha256='0'*64)
        return value
    monkeypatch.setattr(p.c, 'artifact', changed)
    with pytest.raises(ValueError, match='Changed frozen input'):
        p.preserved_attempt_inputs()


def test_failed_completion_cannot_be_upgraded_by_new_attempt(monkeypatch):
    read = p.c.read
    def changed(path):
        value = read(path)
        if path == p.FAILED/'completion.json':
            value = dict(value, status='completed_verified')
        return value
    monkeypatch.setattr(p.c, 'read', changed)
    with pytest.raises(ValueError, match='must remain incomplete'):
        p.preserved_attempt_inputs()
