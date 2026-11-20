"""Guard sequencing, arm dispatch and preserved partial traces without models."""
import gzip
import json
from pathlib import Path
import pytest
from tools.move_evaluation_64_20261023 import worker as w
from tools.move_evaluation_64_20261023 import evaluator as e


def test_guard_once_saved_before_model_initialization_and_no_a_native(monkeypatch, tmp_path):
    from tools import future_training_rules_20261021 as guard
    from tools.corrected_planner_qualification_20261018 import local
    events = []
    monkeypatch.setenv('STUDY_MODE', 'qualify'); monkeypatch.setenv('STUDY_ARM', 'A')
    monkeypatch.setenv('STUDY_WORKER_DIR', str(tmp_path))
    monkeypatch.setattr(w.c, 'OUT', tmp_path)
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: events.append('guard') or ['receipt'])
    monkeypatch.setattr(w.c, 'save', lambda path, value: events.append(Path(path).name) or {})
    monkeypatch.setattr(local, 'initialize', lambda: events.append('models') or {})
    context = w.initialize()
    assert events.count('guard') == 1 and events.index('runtime_sources.json') < events.index('models')
    assert context['opponent'] is None and context['arm'] == 'A'


def test_selection_failure_closes_and_retains_partial_gzip(monkeypatch, tmp_path):
    monkeypatch.setattr(w.c, 'ROOT', tmp_path)
    def fail(payload, opponent, **kwargs):
        kwargs['trace_sink']({'partial': 1})
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(e, 'choose', fail)
    context = {'mode': 'select', 'arm': 'A', 'folder': tmp_path, 'used': False, 'opponent': None}
    with pytest.raises(RuntimeError, match='synthetic failure'): w.handle(context, {})
    with gzip.open(tmp_path/'continuations.jsonl.gz', 'rt') as f:
        assert json.loads(f.readline()) == {'partial': 1}
    with pytest.raises(ValueError, match='one request'): w.handle(context, {})


def test_deterministic_trace_bytes_and_exact_record_count(monkeypatch, tmp_path):
    monkeypatch.setattr(w.c, 'ROOT', tmp_path)
    artifacts = []
    for name in ('a.gz', 'b.gz'):
        trace = w.Trace(tmp_path/name)
        trace.sink({'x': 1}); trace.sink({'x': 2})
        artifacts.append(trace.finalize())
    assert artifacts[0]['sha256'] == artifacts[1]['sha256']
    assert artifacts[0]['records'] == artifacts[1]['records'] == 2


def test_assessment_rejects_referee_or_seed_payload_before_model(monkeypatch):
    context = {'mode': 'assess', 'used': False, 'opponent': None}
    monkeypatch.setattr(e, 'decode_observer', lambda _: pytest.fail('invalid payload reached decoder'))
    with pytest.raises(ValueError, match='observer-only'):
        w.handle(context, {'observer': {}, 'player': 'opponent', 'world_seed': 3})
