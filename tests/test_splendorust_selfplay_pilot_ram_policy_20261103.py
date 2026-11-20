"""Actual frozen worker handlers, with no model initialization or game calls."""
import ast
import hashlib
import io
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

os.environ.setdefault('SELFPLAY_JOB_SCOPE', 'pilot')
from tools.splendorust_selfplay_pilot_ram_policy_20261103 import patch

ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / 'outputs/splendorust_selfplay_pilot_resume_20261103_01/workers/collect_006/entrypoint_clock.json'


def module(role, corrected):
    source = (ROOT / patch.FROZEN_WORKERS[role]['path']).read_bytes()
    if corrected:
        source = patch.corrected_worker_source(role, source)
    loaded = ModuleType('isolated_ram_policy_worker_' + role)
    loaded.__package__ = 'tools.splendorust_selfplay_pilot_resume_20261103' + ('.assessment' if role == 'assessment' else '')
    loaded.__file__ = str(ROOT / patch.FROZEN_WORKERS[role]['path'])
    exec(compile(source, loaded.__file__, 'exec'), loaded.__dict__)
    return loaded


def setup_context(worker, tmp_path):
    callbacks, stream = [], io.StringIO()
    worker._forensic_recorder = lambda path: (callbacks.append, stream)
    route = 'candidate' if worker.__name__.endswith('source') else 'control'
    context = dict(graph_parity_passed=True, diagnostic_capture=False, entrypoint_attempted=False,
                   entrypoint_passed=False, request_started=42., adapters={'search':SimpleNamespace(admission_guard=None)},
                   folder=tmp_path, route=route, boundary_handle_active=False, component_profiles=[])
    return context, callbacks, stream


@pytest.mark.parametrize('role', ['source', 'assessment'])
def test_actual_old_boundary_reproduces_saved_failure_before_suite(role, tmp_path, monkeypatch):
    import psutil
    worker = module(role, False)
    context, records, stream = setup_context(worker, tmp_path)
    suite = Mock()
    monkeypatch.setattr(worker.corpus, 'run_entrypoint_checks', suite)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda:SimpleNamespace(available=1024**3))
    saved = json.loads(FAILED.read_text())
    assert saved['status'] == 'worker_error' and saved['request_id'] == 'request-00000002'
    assert saved['received_before_deadline'] is True
    with pytest.raises(ValueError) as error:
        worker.handle(context, {'operation':'entrypoint_checks'})
    assert repr(error.value) == saved['detail']
    suite.assert_not_called()
    assert not context['entrypoint_passed'] and context['entrypoint_attempted']
    assert records[-1]['event'] == 'entrypoint_failed' and stream.closed
    assert context['request_started'] == 42. and context['adapters']['search'].admission_guard is None


@pytest.mark.parametrize('role', ['source', 'assessment'])
@pytest.mark.parametrize('available', [0, 1, 1024**3, 2*1024**3-1])
def test_corrected_actual_boundary_invokes_complete_suite_without_ram_gate(role, available, tmp_path, monkeypatch):
    import psutil
    worker = module(role, True)
    context, records, stream = setup_context(worker, tmp_path)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda:SimpleNamespace(available=available))
    calls = []
    def suite(adapters, *, request_callback, boundary, evidence_callback, route):
        for _ in range(67):
            boundary()
            adapters['search'].admission_guard()
        calls.append(route)
        return {'counts':{'complete_decisions':67}}
    monkeypatch.setattr(worker.corpus, 'run_entrypoint_checks', suite)
    assert worker.handle(context, {'operation':'entrypoint_checks'}) == {'counts':{'complete_decisions':67}}
    assert calls == [context['route']] and context['entrypoint_passed'] and stream.closed
    assert context['request_started'] == 42. and not context['diagnostic_capture']
    assert context['adapters']['search'].admission_guard is None
    with pytest.raises(ValueError, match='Single actual qualification prefix'):
        worker.handle(context, {'operation':'entrypoint_checks'})


@pytest.mark.parametrize('role', ['source', 'assessment'])
@pytest.mark.parametrize('count', [0, 66, 68])
def test_full_scientific_suite_count_still_required(role, count, tmp_path, monkeypatch):
    worker = module(role, True)
    context, records, stream = setup_context(worker, tmp_path)
    monkeypatch.setattr(worker.corpus, 'run_entrypoint_checks', lambda *a, **kw:{'counts':{'complete_decisions':count}})
    with pytest.raises(ValueError, match='Complete actual entrypoint suite'):
        worker.handle(context, {'operation':'entrypoint_checks'})
    assert not context['entrypoint_passed'] and stream.closed


@pytest.mark.parametrize('role', ['source', 'assessment'])
def test_actual_ordinary_nine_second_complete_response_guard_remains(role):
    worker = module(role, True)
    action = 'legal'
    instance = SimpleNamespace(request_deadline=None,last_training_capture={'unchanged':True})
    def choose(observer, diagnostic_capture):
        assert instance.request_deadline == 109.
        return action, {'profile_id':'id'}
    instance.choose = choose
    worker.decode_observer = lambda x:'state'
    worker.rules = SimpleNamespace(CorrectedEnv=lambda:SimpleNamespace(legal_actions=lambda state:[action]),
                                   serialize_action=lambda x:x)
    worker.time = SimpleNamespace(perf_counter=lambda:109.)
    route = 'candidate' if role == 'source' else 'control'
    quota = worker.adapters.route_quota(route)
    context = dict(request_started=100.,diagnostic_capture=False,route=route,adapters={'search':instance},
        execution_identities={quota:{'profile_id':'id','profile_sha256':'sha','wrapper_id':'wrapper'}})
    with pytest.raises(TimeoutError, match='complete response exceeded9s'):
        worker._ordinary(context, {'player':'search','observer':{}})
    assert instance.request_deadline is None


@pytest.mark.parametrize('role', ['source', 'assessment'])
def test_only_available_ram_line_changes_and_all_other_bodies_identical(role):
    original = (ROOT / patch.FROZEN_WORKERS[role]['path']).read_bytes()
    corrected = patch.corrected_worker_source(role, original)
    assert corrected.replace(patch.NEW_LINE,patch.OLD_LINE) == original
    before = ast.parse(original); after = ast.parse(corrected)
    old = {n.name:n for n in before.body if isinstance(n,ast.FunctionDef)}
    new = {n.name:n for n in after.body if isinstance(n,ast.FunctionDef)}
    assert old.keys()==new.keys()
    assert [name for name in old if ast.dump(old[name])!=ast.dump(new[name])] == ['handle']
    assert hashlib.sha256((ROOT / patch.FROZEN_WORKERS[role]['path']).read_bytes()).hexdigest()==patch.FROZEN_WORKERS[role]['sha256']
    with pytest.raises(ValueError, match='identity mismatch'):
        patch.corrected_worker_source(role, original+b'\n')
    with pytest.raises(ValueError, match='identity mismatch'):
        patch.corrected_worker_source(role, corrected)


@pytest.mark.parametrize('kind', ['deadline','rss','disk','evidence'])
def test_actual_external_envelope_guards_remain_fatal(kind, monkeypatch):
    from tools.splendorust_selfplay_pilot_resume_20261103 import qualification_control as control
    e=control.Envelope({'monotonic_origin':0.,'unix_origin':0.})
    e.remaining=lambda:0. if kind=='deadline' else 100.
    e.accounted_bytes=lambda:control.c.PREFINAL_EVIDENCE_BYTES if kind=='evidence' else 0
    e.stage='collection'
    monkeypatch.setattr(control.shutil,'disk_usage',lambda p:SimpleNamespace(free=0 if kind=='disk' else 6*control.c.GiB))
    if kind=='rss':
        proc=SimpleNamespace(pid=123,create_time=lambda:1.,children=lambda recursive:[],
                             memory_info=lambda:SimpleNamespace(rss=2*control.c.GiB+1))
        e.resident={(123,1.):proc}
    with pytest.raises(ValueError, match={'deadline':'real-clock deadline','rss':'RSS exceeded','disk':'Free disk','evidence':'evidence ceiling'}[kind]):
        e.guard()


def test_patch_is_pure_preparation_and_rejects_unknown_role():
    with pytest.raises(ValueError, match='Exact source or assessment'):
        patch.corrected_worker_source('unexpected',b'')
    tree=ast.parse(Path(patch.__file__).read_text())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in
                   ('write_text','write_bytes','Popen','run','start') for n in ast.walk(tree))
