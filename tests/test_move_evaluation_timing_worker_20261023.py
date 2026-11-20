"""Model-free contract tests of the separately versioned timing worker."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.move_evaluation_timing_20261023 import common as c, worker as w
import tools.move_evaluation_20261023 as frozen_package


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path/'workspace'
    out = root/'outputs/timing'
    out.mkdir(parents=True)
    monkeypatch.setattr(c, 'ROOT', root)
    monkeypatch.setattr(c, 'OUT', out)
    def artifact(path):
        path = Path(path).resolve()
        return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size}
    def verify(items):
        for item in items:
            assert artifact(root/item['path']) == item, 'frozen artifact mismatch'
    monkeypatch.setattr(c, 'artifact', artifact)
    monkeypatch.setattr(c, 'verify', verify)
    return root, out


@pytest.fixture
def fake_evaluator(monkeypatch):
    calls = []
    rows = [{'world_index': 0, 'candidate_index': 0, 'steps': [{'actor': 0, 'forced': True}]},
            {'world_index': 0, 'candidate_index': 1, 'steps': [{'actor': 0, 'forced': True}]}]
    decision = {'action': {'sentinel': 'unchanged'}, 'record': {'worlds': [{'seed': 'fixed'}], 'choice': 0}}
    def choose(payload, opponent, *, boundary, trace_sink):
        calls.append((deepcopy(payload), opponent))
        boundary()
        for row in rows:
            trace_sink(deepcopy(row))
        return deepcopy(decision)
    def select_raw(observer):
        return None
    fake = SimpleNamespace(choose=choose, select_raw=select_raw, PROFILE='unchanged-scientific-profile')
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.evaluator', fake)
    monkeypatch.setattr(frozen_package, 'evaluator', fake, raising=False)
    return fake, calls, rows, decision


def context(out, mode, name=None):
    folder = out/(name or mode)
    folder.mkdir()
    return {'mode': mode, 'folder': folder, 'used': False, 'opponent': object()}


@pytest.fixture
def selected_fake_targets(monkeypatch, fake_evaluator):
    targets = {'scientific_body': w.scientific_body, 'choose': fake_evaluator[0].choose,
               'trace_sink': w.Trace.sink, 'trace_finalize': w.Trace.finalize}
    monkeypatch.setattr(w, 'targets', lambda _: dict(targets))
    return targets


def test_plain_timed_science_and_gzip_equal_to_original_layout(sandbox, fake_evaluator, selected_fake_targets):
    root, out = sandbox
    fake, calls, rows, decision = fake_evaluator
    payload = {'observer': {'lawful': True}}
    plain = w.handle(context(out, 'plain'), payload)
    timed = w.handle(context(out, 'timed'), payload)
    assert len(calls) == 2
    first, second = deepcopy(plain['scientific']), deepcopy(timed['scientific'])
    assert first['decision'] == second['decision'] == decision
    one, two = root/first['trace'].pop('path'), root/second['trace'].pop('path')
    assert first == second
    assert one.read_bytes() == two.read_bytes()
    # Reproduce the immutable worker's exact gzip/header/write recipe.
    original = out/'original_recipe.gz'
    with original.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
            for row in rows:
                compressed.write((c.canonical(row)+'\n').encode())
    assert original.read_bytes() == one.read_bytes()
    assert [json.loads(row) for row in gzip.decompress(one.read_bytes()).splitlines()] == rows
    measurement = timed['measurement']
    counts = {row['label']: row['calls'] for row in measurement['rows']}
    assert counts == {'scientific_body': 1, 'choose': 1, 'trace_sink': 2, 'trace_finalize': 1}
    assert measurement['exclusive_sum_ns'] == measurement['root_region_ns']
    assert measurement['scientific_body_seconds'] == measurement['body_seconds']
    assert plain['measurement']['schema'] == 'ordinary_body_wall_time_v1'
    assert not list(out.rglob('*.pstats'))


@pytest.mark.parametrize('mode', ['plain', 'timed'])
def test_one_request_per_worker(sandbox, fake_evaluator, selected_fake_targets, mode):
    _, out = sandbox
    ctx = context(out, mode)
    w.handle(ctx, {'observer': {}})
    with pytest.raises(ValueError, match='Exactly one request'):
        w.handle(ctx, {'observer': {}})
    assert len(fake_evaluator[1]) == 1


def test_failed_timed_science_preserves_error_partial_trace_and_timer_failure(sandbox, fake_evaluator, monkeypatch):
    _, out = sandbox
    original = RuntimeError('synthetic scientific failure')
    def fail(payload, opponent, *, boundary, trace_sink):
        trace_sink({'preserved_partial': True})
        raise original
    fake_evaluator[0].choose = fail
    monkeypatch.setattr(w, 'targets', lambda _: {'scientific_body': w.scientific_body, 'choose': fail,
                                                'trace_finalize': w.Trace.finalize})
    ctx = context(out, 'timed')
    with pytest.raises(RuntimeError) as caught:
        w.handle(ctx, {'observer': {}})
    assert caught.value is original and ctx['used'] is True
    assert gzip.decompress((ctx['folder']/'continuations.jsonl.gz').read_bytes()) == b'{"preserved_partial":true}\n'
    failure = json.loads((ctx['folder']/'timing_failure.json').read_text())
    assert failure['phase'] == 'application' and failure['exception_type'] == 'RuntimeError'
    assert not failure['open_regions']
    rows = {row['label']: row for row in failure['rows']}
    assert rows['choose']['unwinds'] == rows['scientific_body']['unwinds'] == 1
    assert rows['trace_finalize']['returns'] == 1
    with pytest.raises(ValueError, match='Exactly one request'):
        w.handle(ctx, {'observer': {}})


def test_trace_file_is_exclusive_and_closed_after_fsync(sandbox, monkeypatch):
    _, out = sandbox
    path = out/'continuations.jsonl.gz'
    trace = w.Trace(path)
    trace.sink({'a': 1})
    events = []
    original_fsync = w.os.fsync
    def fsync(fd):
        assert not trace.raw.closed
        assert trace.compressed.fileobj is None
        events.append('fsync')
        original_fsync(fd)
    monkeypatch.setattr(w.os, 'fsync', fsync)
    descriptor = trace.finalize()
    assert trace.raw.closed and events == ['fsync'] and descriptor['records'] == 1
    assert descriptor['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        w.Trace(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('mode', ['qualify_plain', 'qualify_timed'])
def test_qualification_uses_exact_frozen_entrypoint(sandbox, fake_evaluator, monkeypatch, mode):
    _, out = sandbox
    expected = {'status': 'passed', 'fixture': 'no-model-sentinel'}
    calls = []
    def qualify(opponent, raw):
        calls.append((opponent, raw))
        return deepcopy(expected)
    fake = SimpleNamespace(run_entrypoint_checks=qualify)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.qualification', fake)
    monkeypatch.setattr(frozen_package, 'qualification', fake, raising=False)
    monkeypatch.setattr(w, 'targets', lambda _: {'scientific_body': w.scientific_body, 'entrypoint_checks': qualify})
    ctx = context(out, mode)
    result = w.handle(ctx, {'operation': 'entrypoint_checks'})
    assert result['scientific'] == expected
    assert calls == [(ctx['opponent'], fake_evaluator[0].select_raw)]
    assert not list(ctx['folder'].glob('*.gz'))
    if mode == 'qualify_timed':
        assert {row['label']: row['calls'] for row in result['measurement']['rows']} == {
            'scientific_body': 1, 'entrypoint_checks': 1}


@pytest.mark.parametrize('payload', [{}, [], {'operation': 'wrong'}, {'operation': 'entrypoint_checks', 'extra': 1}])
def test_qualification_payload_exact(sandbox, fake_evaluator, payload):
    _, out = sandbox
    with pytest.raises(ValueError, match='Qualification payload'):
        w.handle(context(out, 'qualify_plain'), payload)


@pytest.mark.parametrize('payload', [{}, [], {'selector': {}, 'result': {}, 'trace': {}, 'extra': 1}])
def test_audit_payload_exact(sandbox, fake_evaluator, payload):
    _, out = sandbox
    with pytest.raises(ValueError, match='Audit payload'):
        w.handle(context(out, 'audit'), payload)


@pytest.mark.parametrize('kind', ['old_output', 'traversal', 'wrong_name', 'absolute_outside'])
def test_audit_escape_rejected_before_authentication(sandbox, fake_evaluator, monkeypatch, kind):
    root, out = sandbox
    paths = {'old_output': 'outputs/old/continuations.jsonl.gz',
             'traversal': 'outputs/timing/../../continuations.jsonl.gz',
             'wrong_name': 'outputs/timing/trace.gz',
             'absolute_outside': str(root/'elsewhere/continuations.jsonl.gz')}
    called = []
    monkeypatch.setattr(c, 'verify', lambda _: called.append(True))
    with pytest.raises(ValueError, match='Audit path boundary'):
        w.handle(context(out, 'audit'), {'selector': {}, 'result': {},
            'trace': {'path': paths[kind], 'bytes': 0, 'sha256': '0'*64}})
    assert called == []


def test_audit_symlink_escape_rejected(sandbox, fake_evaluator):
    root, out = sandbox
    elsewhere = root/'elsewhere'
    elsewhere.mkdir()
    try:
        (out/'link').symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not permit directory symlinks')
    with pytest.raises(ValueError, match='Audit path boundary'):
        w.handle(context(out, 'audit'), {'selector': {}, 'result': {},
            'trace': {'path': 'outputs/timing/link/continuations.jsonl.gz', 'bytes': 0, 'sha256': '0'*64}})


def test_audit_forwards_only_authenticated_trace(sandbox, fake_evaluator, monkeypatch):
    root, out = sandbox
    path = out/'continuations.jsonl.gz'
    path.write_bytes(gzip.compress(b'{"synthetic":true}\n', mtime=0))
    calls = []
    def audit(selector, result, trace_path, opponent, *, boundary):
        calls.append((selector, result, trace_path, opponent))
        return {'status': 'passed_synthetic'}
    fake = SimpleNamespace(audit_selection=audit)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.audit', fake)
    monkeypatch.setattr(frozen_package, 'audit', fake, raising=False)
    ctx = context(out, 'audit')
    payload = {'selector': {'lawful': True}, 'result': {'complete': True}, 'trace': c.artifact(path)}
    result = w.handle(ctx, payload)
    assert result['scientific'] == {'status': 'passed_synthetic'}
    assert result['measurement']['schema'] == 'ordinary_body_wall_time_v1'
    assert calls == [(payload['selector'], payload['result'], path, ctx['opponent'])]


def test_tampered_trace_rejected(sandbox, fake_evaluator):
    _, out = sandbox
    path = out/'continuations.jsonl.gz'
    path.write_bytes(b'original')
    descriptor = c.artifact(path)
    path.write_bytes(b'changed')
    with pytest.raises(AssertionError, match='artifact mismatch'):
        w.handle(context(out, 'audit'), {'selector': {}, 'result': {}, 'trace': descriptor})


def test_initialize_rejects_old_or_outside_output_before_loading(sandbox, monkeypatch):
    root, _ = sandbox
    monkeypatch.setenv('STUDY_WORKER_DIR', str(root/'outputs/old'))
    monkeypatch.setenv('STUDY_MODE', 'plain')
    with pytest.raises(ValueError, match='Worker output boundary'):
        w.initialize()


@pytest.mark.parametrize('mode', ['plain', 'timed', 'qualify_plain', 'qualify_timed', 'audit'])
def test_initialize_source_guard_once_then_models_and_native_object(sandbox, monkeypatch, mode):
    _, out = sandbox
    from tools import future_training_rules_20261021 as guard
    import tools.corrected_planner_qualification_20261018 as qualified_package
    calls = []
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: calls.append('guard') or {'source': 'verified'})
    fake_local = SimpleNamespace(initialize=lambda: calls.append('models') or {'fixed': True})
    monkeypatch.setitem(sys.modules, 'tools.corrected_planner_qualification_20261018.local', fake_local)
    monkeypatch.setattr(qualified_package, 'local', fake_local, raising=False)
    opponent = object()
    def constructor(folder):
        calls.append(('opponent', folder))
        return opponent
    fake_greedy = SimpleNamespace(GreedyOpponent=constructor)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.greedy_adapter', fake_greedy)
    monkeypatch.setattr(frozen_package, 'greedy_adapter', fake_greedy, raising=False)
    folder = out/mode
    monkeypatch.setenv('STUDY_WORKER_DIR', str(folder))
    monkeypatch.setenv('STUDY_MODE', mode)
    ctx = w.initialize()
    assert calls == ['guard', 'models', ('opponent', folder)]
    assert ctx == {'mode': mode, 'folder': folder, 'used': False, 'opponent': opponent}
    assert json.loads((folder/'runtime_sources.json').read_text())['sources'] == {'source': 'verified'}
    assert json.loads((folder/'models.json').read_text()) == {'fixed': True}


def test_unknown_mode_rejected_before_models(sandbox, monkeypatch):
    _, out = sandbox
    from tools import future_training_rules_20261021 as guard
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: {'source': 'verified'})
    monkeypatch.setenv('STUDY_WORKER_DIR', str(out/'unknown'))
    monkeypatch.setenv('STUDY_MODE', 'unknown')
    with pytest.raises(ValueError, match='Timing mode'):
        w.initialize()


def test_common_save_exclusive_and_contained(sandbox):
    root, out = sandbox
    path = out/'receipt.json'
    c.save(path, {'first': True})
    with pytest.raises(FileExistsError):
        c.save(path, {'second': True})
    assert json.loads(path.read_text()) == {'first': True}
    with pytest.raises(ValueError, match='Timing output boundary'):
        c.save(root/'outside.json', {})


def _distinct_function(name):
    scope = {}
    exec(compile('def '+name+'(*args, **kwargs):\n    return None\n', __file__, 'exec'), scope)
    return scope[name]


@pytest.fixture
def target_dependencies(monkeypatch):
    import tools.corrected_planner_qualification_20261018 as qualified_package
    import tools.external_strength_screen_20261018 as rules_package
    import tools.external_strength_screen_20261018_02 as bridge_package
    def install(package, name, value):
        monkeypatch.setitem(sys.modules, package.__name__+'.'+name, value)
        monkeypatch.setattr(package, name, value, raising=False)
    evaluator = SimpleNamespace(**{name: _distinct_function('eval_'+name) for name in (
        'choose', 'trajectory', 'sample_world', 'decode_observer', 'select_raw')})
    qualification = SimpleNamespace(**{name: _distinct_function(name) for name in (
        'run_entrypoint_checks', 'verify_actual_trajectory')})
    class SharedAttention:
        def forward(self, *args):
            return 'same implementation for all four model instances'
    models = [SharedAttention() for _ in range(3)]
    raw = SharedAttention()
    q = SimpleNamespace(score_actions=_distinct_function('raw_score'),
        proposal_for_root=_distinct_function('proposal'), _RAW=SimpleNamespace(_model=raw),
        registered_model=lambda name: (models[(161001, 161002, 161003).index(int(name.removeprefix('anchor_')))], {}))
    class Greedy:
        def choose(self, observer):
            return None
        def _infer(self, tokens, batch):
            return None
    class Native:
        def exchange(self, request):
            return None
    greedy = SimpleNamespace(GreedyOpponent=Greedy, NativeOnly=Native)
    class Env:
        def legal_actions(self, state):
            return []
        def step(self, state, action):
            return state
    rules = SimpleNamespace(CorrectedEnv=Env, state_from_observation=_distinct_function('decode'),
                            observation_payload=_distinct_function('encode'))
    bridge = SimpleNamespace(**{name: _distinct_function(name) for name in (
        '_state_and_legal', 'encode_observation', 'main_actions', 'resolve_compound')})
    install(frozen_package, 'evaluator', evaluator)
    install(frozen_package, 'qualification', qualification)
    install(frozen_package, 'greedy_adapter', greedy)
    install(qualified_package, 'local', q)
    install(rules_package, 'rules', rules)
    install(bridge_package, 'bridge', bridge)
    return q, models, raw, greedy


def test_target_identity_combines_shared_attention_and_manual_entity_infer(target_dependencies):
    q, models, raw, greedy = target_dependencies
    selected = w.targets({})
    assert len(selected) == len({id(fn.__code__) for fn in selected.values()})
    assert selected['shared_attention_forward'].__code__ is raw.forward.__code__
    assert all(model.forward.__code__ is raw.forward.__code__ for model in models)
    assert selected['entity_infer'] is greedy.GreedyOpponent._infer
    assert [label for label in selected if 'forward' in label] == ['shared_attention_forward']
    assert selected['trace_sink'] is w.Trace.sink and selected['trace_finalize'] is w.Trace.finalize


def test_different_proposal_forward_rejected(target_dependencies):
    _, models, _, _ = target_dependencies
    models[1].forward = _distinct_function('unexpected_forward')
    with pytest.raises(ValueError, match='common proposal forward'):
        w.targets({})


def test_different_raw_forward_rejected(target_dependencies):
    _, _, raw, _ = target_dependencies
    raw.forward = _distinct_function('different_raw')
    with pytest.raises(ValueError, match='shared raw/proposal'):
        w.targets({})


def test_alias_target_code_rejected(target_dependencies):
    q, _, _, _ = target_dependencies
    q.proposal_for_root = q.score_actions
    with pytest.raises(ValueError, match='Ambiguous selected code'):
        w.targets({})
