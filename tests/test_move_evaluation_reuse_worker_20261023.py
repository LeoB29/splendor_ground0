"""Model-free coverage of optimized dispatch and independent reference replay."""
import json
import sys
from types import SimpleNamespace

import pytest

from tools.move_evaluation_reuse_20261023 import common as c, worker as w
import tools.move_evaluation_reuse_20261023 as package
import tools.move_evaluation_20261023 as original


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path/'workspace'
    out = root/'outputs/reuse'
    out.mkdir(parents=True)
    monkeypatch.setattr(c, 'ROOT', root)
    monkeypatch.setattr(c, 'OUT', out)
    def artifact(path):
        import hashlib
        return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(c, 'artifact', artifact)
    return root, out


@pytest.mark.parametrize('mode', ['plain','timed','qualify_plain','qualify_timed','audit'])
def test_initialize_guard_once_before_models_and_explicit_policy(sandbox, monkeypatch, mode):
    _, out = sandbox
    from tools import future_training_rules_20261021 as guard
    import tools.corrected_planner_qualification_20261018 as qpackage
    from tools.move_evaluation_reuse_20261023.runner import implementation_metadata
    calls = []
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: calls.append('guard') or {'verified': True})
    local = SimpleNamespace(initialize=lambda: calls.append('models') or {'historical_models': True})
    monkeypatch.setitem(sys.modules, 'tools.corrected_planner_qualification_20261018.local', local)
    monkeypatch.setattr(qpackage, 'local', local, raising=False)
    def reference(folder):
        calls.append('original_reference');return object()
    def reuse(folder):
        calls.append('per_decision_reuse');return object()
    old = SimpleNamespace(GreedyOpponent=reference, PROFILE_ID='entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1')
    new = SimpleNamespace(GreedyOpponent=reuse, IMPLEMENTATION_ID=c.IMPLEMENTATION_ID)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.greedy_adapter', old)
    monkeypatch.setattr(original, 'greedy_adapter', old, raising=False)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_reuse_20261023.greedy_adapter', new)
    monkeypatch.setattr(package, 'greedy_adapter', new, raising=False)
    folder = out/mode
    monkeypatch.setenv('STUDY_MODE', mode)
    monkeypatch.setenv('STUDY_WORKER_DIR', str(folder))
    ctx = w.initialize()
    chosen = 'original_reference' if mode == 'audit' else 'per_decision_reuse'
    assert calls == ['guard','models',chosen]
    assert ctx['implementation'] == implementation_metadata(mode)
    assert json.loads((folder/'implementation.json').read_text()) == ctx['implementation']
    assert ctx['used'] is False


def test_initialize_boundary_precedes_guard_and_models(sandbox, monkeypatch):
    root, _ = sandbox
    from tools import future_training_rules_20261021 as guard
    calls=[]
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: calls.append('guard'))
    monkeypatch.setenv('STUDY_MODE','plain')
    monkeypatch.setenv('STUDY_WORKER_DIR',str(root/'outside'))
    with pytest.raises(ValueError, match='Worker output boundary'):w.initialize()
    assert not calls


def fake_scientific_modules(monkeypatch):
    evaluator=SimpleNamespace(PROFILE='unchanged',select_raw=object())
    monkeypatch.setitem(sys.modules,'tools.move_evaluation_20261023.evaluator',evaluator)
    monkeypatch.setattr(original,'evaluator',evaluator,raising=False)
    return evaluator


@pytest.mark.parametrize('mode',['qualify_plain','qualify_timed'])
def test_actual_qualification_receives_optimized_opponent_unchanged(sandbox,monkeypatch,mode):
    _,out=sandbox
    evaluator=fake_scientific_modules(monkeypatch)
    opponent=object();calls=[]
    def qualify(policy,raw):
        calls.append((policy,raw));return {'accepted':'same'}
    qualification=SimpleNamespace(run_entrypoint_checks=qualify)
    monkeypatch.setitem(sys.modules,'tools.move_evaluation_20261023.qualification',qualification)
    monkeypatch.setattr(original,'qualification',qualification,raising=False)
    result=w.scientific_body({'mode':mode,'opponent':opponent},{'operation':'entrypoint_checks'})
    assert calls==[(opponent,evaluator.select_raw)] and result=={'accepted':'same'}


def test_audit_rejects_optimized_policy_before_evidence_or_replay(sandbox,monkeypatch):
    fake_scientific_modules(monkeypatch)
    with pytest.raises(ValueError,match='Original reference audit required'):
        w.scientific_body({'mode':'audit','implementation':{'active_opponent':'per_decision_reuse'}},{})


def test_audit_uses_original_policy_and_exact_trace(sandbox,monkeypatch):
    root,out=sandbox
    fake_scientific_modules(monkeypatch)
    path=out/'continuations.jsonl.gz';path.write_bytes(b'synthetic')
    checked=[];calls=[];opponent=object()
    monkeypatch.setattr(c,'verify',lambda items: checked.extend(items))
    def audit(selector,result,trace,policy,*,boundary):
        calls.append((selector,result,trace,policy));return {'status':'passed_synthetic'}
    module=SimpleNamespace(audit_selection=audit)
    monkeypatch.setitem(sys.modules,'tools.move_evaluation_20261023.audit',module)
    monkeypatch.setattr(original,'audit',module,raising=False)
    payload={'selector':{'fixed':True},'result':{'exact':True},'trace':c.artifact(path)}
    value=w.scientific_body({'mode':'audit','opponent':opponent,'implementation':{'active_opponent':'original_reference'}},payload)
    assert checked==[payload['trace']]
    assert calls==[(payload['selector'],payload['result'],path,opponent)]
    assert value=={'status':'passed_synthetic'}


@pytest.mark.parametrize('mode',['plain','timed'])
def test_handle_science_provenance_and_single_use(sandbox,monkeypatch,mode):
    _,out=sandbox;calls=[]
    def body(ctx,payload):calls.append(payload);return {'decision':'unchanged'}
    monkeypatch.setattr(w,'scientific_body',body)
    monkeypatch.setattr(w,'targets',lambda ctx:{'scientific_body':body})
    ctx={'mode':mode,'folder':out,'used':False,'implementation':{'independent':'provenance'}}
    result=w.handle(ctx,{'same':True})
    assert result['scientific']=={'decision':'unchanged'}
    assert result['implementation']==ctx['implementation']
    assert result['measurement']['body_seconds']>0
    with pytest.raises(ValueError,match='Exactly one request'):w.handle(ctx,{})
    assert calls==[{'same':True}]


def test_failure_keeps_original_exception_and_timer_record(sandbox,monkeypatch):
    _,out=sandbox
    failure=RuntimeError('synthetic optimizer failure')
    def body(ctx,payload):raise failure
    monkeypatch.setattr(w,'scientific_body',body)
    monkeypatch.setattr(w,'targets',lambda ctx:{'scientific_body':body})
    ctx={'mode':'timed','folder':out,'used':False,'implementation':{}}
    with pytest.raises(RuntimeError) as observed:w.handle(ctx,{})
    assert observed.value is failure
    assert json.loads((out/'timing_failure.json').read_text())['phase']=='application'
    assert ctx['used'] is True
