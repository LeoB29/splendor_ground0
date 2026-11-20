"""No-model tests of the separate profiling wrapper and evidence boundaries."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pstats
import sys
from types import SimpleNamespace

import pytest

from tools.move_evaluation_profile_20261023 import common as c, worker as w
import tools.move_evaluation_20261023 as frozen_package


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / 'workspace'
    out = root / 'outputs/profile'
    out.mkdir(parents=True)
    monkeypatch.setattr(c, 'ROOT', root)
    monkeypatch.setattr(c, 'OUT', out)
    def artifact(path):
        path = Path(path).resolve()
        return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'bytes': path.stat().st_size}
    def verify(items):
        for item in items:
            assert artifact(root / item['path']) == item, 'frozen artifact mismatch'
    monkeypatch.setattr(c, 'artifact', artifact)
    monkeypatch.setattr(c, 'verify', verify)
    return root, out


@pytest.fixture
def fake_evaluator(monkeypatch):
    calls = []
    rows = [{'world_index': 0, 'candidate_index': 0, 'steps': [{'actor': 0, 'forced': True}]},
            {'world_index': 0, 'candidate_index': 1, 'steps': [{'actor': 0, 'forced': True}]}]
    decision = {'action': {'sentinel': 'fixed'}, 'record': {'worlds': [{'seed': 'fixed'}], 'choice': 0}}
    def choose(payload, opponent, *, boundary, trace_sink):
        calls.append((deepcopy(payload), opponent))
        boundary()
        for row in rows:
            trace_sink(deepcopy(row))
        return deepcopy(decision)
    fake = SimpleNamespace(choose=choose, select_raw=lambda _: None, PROFILE='unchanged-profile')
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.evaluator', fake)
    monkeypatch.setattr(frozen_package, 'evaluator', fake, raising=False)
    return fake, calls, rows, decision


def context(out, mode, name=None):
    folder = out / (name or mode)
    folder.mkdir()
    return {'mode': mode, 'folder': folder, 'opponent': object(), 'used': False}


def test_profile_and_plain_have_identical_science_and_gzip(sandbox, fake_evaluator):
    root, out = sandbox
    fake, calls, rows, decision = fake_evaluator
    payload = {'observer': {'lawful': True}}
    plain = w.handle(context(out, 'plain'), payload)
    profiled = w.handle(context(out, 'profile'), payload)
    assert len(calls) == 2
    a, b = deepcopy(plain['scientific']), deepcopy(profiled['scientific'])
    assert a['decision'] == b['decision'] == decision
    assert a['trace']['path'] != b['trace']['path']
    assert (root / a['trace']['path']).read_bytes() == (root / b['trace']['path']).read_bytes()
    for value in (a, b):
        trace_path = root / value['trace'].pop('path')
        assert [json.loads(line) for line in gzip.decompress(trace_path.read_bytes()).splitlines()] == rows
        assert value['trace']['records'] == 2
    assert a == b
    assert plain['measurement']['profile'] is None
    profile = profiled['measurement']['profile']
    stats = pstats.Stats(str(root / profile['path']))
    assert stats.total_calls > 0
    assert any(key[2] == 'scientific_body' for key in stats.stats)
    assert not any(key[2] == 'dump_stats' for key in stats.stats)
    assert profiled['measurement']['scientific_body_seconds'] >= 0
    assert profiled['measurement']['profile_export_seconds'] >= 0


@pytest.mark.parametrize('mode', ['plain', 'profile'])
def test_fresh_worker_accepts_exactly_one_request(sandbox, fake_evaluator, mode):
    _, out = sandbox
    ctx = context(out, mode)
    w.handle(ctx, {'observer': {}})
    with pytest.raises(ValueError, match='Exactly one request'):
        w.handle(ctx, {'observer': {}})
    assert len(fake_evaluator[1]) == 1


def test_failed_science_remains_used_and_has_no_profiler_success(sandbox, fake_evaluator):
    _, out = sandbox
    fake, _, _, _ = fake_evaluator
    def fail(*args, **kwargs):
        kwargs['trace_sink']({'partial': True})
        raise ValueError('deliberate model-free failure')
    fake.choose = fail
    ctx = context(out, 'profile')
    with pytest.raises(ValueError, match='deliberate'):
        w.handle(ctx, {'observer': {}})
    assert ctx['used'] is True
    assert not (ctx['folder'] / 'scientific.pstats').exists()
    assert gzip.decompress((ctx['folder'] / 'continuations.jsonl.gz').read_bytes())
    with pytest.raises(ValueError, match='Exactly one request'):
        w.handle(ctx, {'observer': {}})


def test_profile_export_never_overwrites(sandbox, fake_evaluator):
    _, out = sandbox
    ctx = context(out, 'profile')
    p = ctx['folder'] / 'scientific.pstats'
    p.write_bytes(b'preserved')
    with pytest.raises(ValueError, match='No profile overwrite'):
        w.handle(ctx, {'observer': {}})
    assert p.read_bytes() == b'preserved'


def test_qualification_is_profiled_and_scientific_result_untouched(sandbox, fake_evaluator, monkeypatch):
    root, out = sandbox
    expected = {'status': 'entrypoint_counterexamples_passed', 'fixtures': ['synthetic'], 'hash': 'known'}
    calls = []
    def qualify(opponent, raw):
        calls.append((opponent, raw))
        return deepcopy(expected)
    fake = SimpleNamespace(run_entrypoint_checks=qualify)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.qualification', fake)
    monkeypatch.setattr(frozen_package, 'qualification', fake, raising=False)
    ctx = context(out, 'qualify')
    result = w.handle(ctx, {'operation': 'entrypoint_checks'})
    assert result['scientific'] == expected and len(calls) == 1
    assert calls[0] == (ctx['opponent'], fake_evaluator[0].select_raw)
    path = root / result['measurement']['profile']['path']
    assert path.name == 'scientific.pstats' and path.is_file()
    assert any(key[2] == 'qualify' for key in pstats.Stats(str(path)).stats)


@pytest.mark.parametrize('payload', [{}, {'operation': 'entrypoint_checks', 'extra': True},
                                    {'operation': 'wrong'}, []])
def test_qualification_payload_is_exact(sandbox, fake_evaluator, payload):
    _, out = sandbox
    with pytest.raises(ValueError, match='Qualification payload'):
        w.handle(context(out, 'qualify'), payload)


@pytest.mark.parametrize('payload', [{}, [], {'selector': {}, 'result': {}, 'trace': {}, 'extra': 0}])
def test_audit_payload_is_exact(sandbox, fake_evaluator, payload):
    _, out = sandbox
    with pytest.raises(ValueError, match='Audit payload'):
        w.handle(context(out, 'audit'), payload)


@pytest.mark.parametrize('path_kind', ['outside', 'traversal', 'wrong_name', 'absolute_outside'])
def test_audit_path_escape_rejected_before_read(sandbox, fake_evaluator, monkeypatch, path_kind):
    root, out = sandbox
    paths = {'outside': 'outputs/old/continuations.jsonl.gz',
             'traversal': 'outputs/profile/../old/continuations.jsonl.gz',
             'wrong_name': 'outputs/profile/trace.gz',
             'absolute_outside': str(root / 'elsewhere/continuations.jsonl.gz')}
    called = []
    monkeypatch.setattr(c, 'verify', lambda _: called.append(True))
    payload = {'selector': {}, 'result': {}, 'trace': {'path': paths[path_kind], 'bytes': 0, 'sha256': '0'*64}}
    with pytest.raises(ValueError, match='Audit path boundary'):
        w.handle(context(out, 'audit'), payload)
    assert called == []


def test_audit_symlink_escape_rejected(sandbox, fake_evaluator):
    root, out = sandbox
    target = root / 'outside'
    target.mkdir()
    link = out / 'escaped'
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('Host does not permit directory symlinks')
    payload = {'selector': {}, 'result': {}, 'trace': {
        'path': 'outputs/profile/escaped/continuations.jsonl.gz', 'bytes': 0, 'sha256': '0'*64}}
    with pytest.raises(ValueError, match='Audit path boundary'):
        w.handle(context(out, 'audit'), payload)


def test_audit_forwards_verified_trace_and_no_profiler(sandbox, fake_evaluator, monkeypatch):
    root, out = sandbox
    source = out / 'original'
    source.mkdir()
    path = source / 'continuations.jsonl.gz'
    path.write_bytes(gzip.compress(b'{"synthetic":true}\n', mtime=0))
    calls = []
    def audit(selector, result, traces_path, opponent, *, boundary):
        calls.append((selector, result, traces_path, opponent))
        return {'status': 'passed_synthetic'}
    fake = SimpleNamespace(audit_selection=audit)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.audit', fake)
    monkeypatch.setattr(frozen_package, 'audit', fake, raising=False)
    ctx = context(out, 'audit')
    payload = {'selector': {'id': 'one'}, 'result': {'id': 'one'}, 'trace': c.artifact(path)}
    result = w.handle(ctx, payload)
    assert calls == [(payload['selector'], payload['result'], path, ctx['opponent'])]
    assert result['scientific'] == {'status': 'passed_synthetic'}
    assert result['measurement']['profile'] is None


def test_tampered_trace_fails_before_independent_audit(sandbox, fake_evaluator):
    _, out = sandbox
    path = out / 'continuations.jsonl.gz'
    path.write_bytes(b'original')
    descriptor = c.artifact(path)
    path.write_bytes(b'tampered')
    with pytest.raises(AssertionError, match='artifact mismatch'):
        w.handle(context(out, 'audit'), {'selector': {}, 'result': {}, 'trace': descriptor})


def test_common_save_is_exclusive_and_contained(sandbox):
    root, out = sandbox
    path = out / 'receipt.json'
    c.save(path, {'first': True})
    with pytest.raises(FileExistsError):
        c.save(path, {'second': True})
    assert json.loads(path.read_text()) == {'first': True}
    with pytest.raises(ValueError, match='Diagnostic output boundary'):
        c.save(root / 'outside.json', {})


def test_initialize_rejects_outside_before_rule_model_loading(sandbox, monkeypatch):
    root, _ = sandbox
    monkeypatch.setenv('STUDY_WORKER_DIR', str(root / 'elsewhere'))
    monkeypatch.setenv('STUDY_MODE', 'plain')
    with pytest.raises(ValueError, match='Worker output boundary'):
        w.initialize()


def test_initialize_authenticates_once_and_uses_fresh_objects(sandbox, monkeypatch):
    _, out = sandbox
    from tools import future_training_rules_20261021 as guard
    import tools.corrected_planner_qualification_20261018 as qualified_package
    calls = []
    monkeypatch.setattr(guard, 'verify_runtime_sources', lambda: calls.append('rules') or {'source': 'hash'})
    fake_local = SimpleNamespace(initialize=lambda: calls.append('models') or {'profile': 'same'})
    monkeypatch.setitem(sys.modules, 'tools.corrected_planner_qualification_20261018.local', fake_local)
    monkeypatch.setattr(qualified_package, 'local', fake_local, raising=False)
    created = []
    def opponent(folder):
        value = object()
        created.append((folder, value))
        calls.append('opponent')
        return value
    fake_greedy = SimpleNamespace(GreedyOpponent=opponent)
    monkeypatch.setitem(sys.modules, 'tools.move_evaluation_20261023.greedy_adapter', fake_greedy)
    monkeypatch.setattr(frozen_package, 'greedy_adapter', fake_greedy, raising=False)
    folder = out / 'newworker'
    monkeypatch.setenv('STUDY_WORKER_DIR', str(folder))
    monkeypatch.setenv('STUDY_MODE', 'plain')
    ctx = w.initialize()
    assert calls == ['rules', 'models', 'opponent']
    assert ctx['opponent'] is created[0][1] and ctx['used'] is False
    assert json.loads((folder / 'runtime_sources.json').read_text())['sources'] == {'source': 'hash'}
    assert json.loads((folder / 'models.json').read_text()) == {'profile': 'same'}


@pytest.mark.parametrize('limit,deadline,current,expected', [(600, 1000, 1, 600),
                                                          (600, 1000, 995, 5),
                                                          (60, 1000, 999.75, .25)])
def test_request_cap_clamps_to_remaining_stage(limit, deadline, current, expected):
    from tools.move_evaluation_profile_20261023.runner import clamp_timeout
    assert clamp_timeout(limit, deadline, clock=lambda: current) == expected


@pytest.mark.parametrize('current', [1000, 1001])
def test_no_request_grace_after_stage(current):
    from tools.move_evaluation_profile_20261023.runner import clamp_timeout
    with pytest.raises(ValueError, match='stage deadline'):
        clamp_timeout(600, 1000, clock=lambda: current)


def test_perform_clamps_init_then_request_and_preserves_cpu_identities(sandbox, monkeypatch):
    _, out = sandbox
    from tools.move_evaluation_profile_20261023 import runner as r
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    seen, current = [], [95.0]
    real_clamp = r.clamp_timeout
    monkeypatch.setattr(r, 'clamp_timeout', lambda limit, deadline: real_clamp(limit, deadline, clock=lambda: current[0]))
    envelope = SimpleNamespace(admit=lambda deadline: seen.append(('admit', deadline)),
                               guard=lambda deadline: None)
    class FakeWorker:
        def __init__(self, module, folder, **kwargs):
            self.output_dir = folder
            folder.mkdir(parents=True)
            seen.append(('init', module, kwargs))
            self.initialization = {'elapsed_seconds': 2., 'telemetry': {'process_tree': [
                {'pid': 1, 'created': 10., 'cpu_seconds': 2.},
                {'pid': 2, 'created': 20., 'cpu_seconds': 1.}]}}
            current[0] += 2
        def __enter__(self):
            return self
        def __exit__(self, *_):
            seen.append(('close',))
        def request(self, payload, seconds):
            seen.append(('request', deepcopy(payload), seconds))
            receipt = {'request_id': 'request-00000001', 'status': 'ok', 'accepted': True,
                'result': {'scientific': 'unchanged'}, 'detail': None, 'receipt_elapsed_seconds': .5,
                'telemetry': {'process_tree': [
                    {'pid': 1, 'created': 10., 'cpu_seconds': 4.5, 'role': 'python'},
                    {'pid': 2, 'created': 20., 'cpu_seconds': 2., 'role': 'rust'}],
                    'request_cpu_seconds': 3.45, 'peak_sampled_rss_bytes': 12345}}
            (self.output_dir / 'request-00000001.json').write_text(json.dumps(receipt))
            return receipt
    monkeypatch.setattr(c, 'StudyWorker', FakeWorker)
    payload = {'observer': {'public': True}}
    result = r.perform('root/plain', 'plain', payload, 600, 100, envelope, {'native_binary': {'bound': True}})
    assert seen[0] == ('admit', 100)
    init = seen[1][2]
    assert init['initialization_timeout'] == 5 and init['absolute_deadline'] == 100
    assert init['affinity'] == [0] and init['max_rss_bytes'] == 2*c.GiB
    assert init['env']['STUDY_WORKER_DIR'].startswith(str(out))
    assert seen[2] == ('request', payload, 3)
    assert seen[3] == ('close',)
    assert result['sampled_cpu_deltas_by_role'] == {'python': 2.5, 'rust': 1.}
    assert result['request_tree_cpu_seconds'] == 3.45
    assert 'differs slightly' in result['cpu_scope']


@pytest.mark.parametrize('status', ['timeout', 'worker_error', 'resource_guard'])
def test_perform_failure_closes_once_and_never_retries(sandbox, monkeypatch, status):
    _, out = sandbox
    from tools.move_evaluation_profile_20261023 import runner as r
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    calls = []
    class FakeWorker:
        def __init__(self, module, folder, **kwargs):
            calls.append('created')
        def __enter__(self):
            return self
        def __exit__(self, *_):
            calls.append('closed')
        def request(self, payload, seconds):
            calls.append('request')
            return {'status': status, 'accepted': False, 'detail': 'preserved failure', 'receipt_elapsed_seconds': None}
    monkeypatch.setattr(c, 'StudyWorker', FakeWorker)
    env = SimpleNamespace(admit=lambda _: None, guard=lambda _: None)
    with pytest.raises(ValueError, match='Request failed'):
        r.perform('root/plain', 'plain', {}, 600, float('inf'), env, {'native_binary': {}})
    assert calls == ['created', 'request', 'closed']
    assert not (out / 'measurements/root_plain.json').exists()
