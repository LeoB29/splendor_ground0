"""Model-free regressions reconstructed from the actual failed handoff records."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from tools import splendorust_network_practical_handoff_20261031 as fixed

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'tools/splendorust_network_practical_20261031'
SAVED = ROOT / 'outputs/splendorust_network_practical_20261031_01'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def original_function(name, common, clock):
    """Execute the exact frozen function AST without rebinding its real module."""
    tree = ast.parse((SOURCE / 'control.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(c=common, time=clock)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE / 'control.py'), 'exec'), namespace)
    return namespace[name]


def augmentation_fields():
    """Derive the saved-only fields from the actual constructor, not a guessed mock."""
    tree = ast.parse((SOURCE / 'transport.py').read_text(encoding='utf-8'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
             isinstance(n.func, ast.Name) and n.func.id == '_new_json' and
             'initialization.json' in ast.unparse(n)]
    assert len(calls) == 1
    value = calls[0].args[1]
    assert isinstance(value, ast.Dict) and value.keys[0] is None
    assert ast.unparse(value.values[0]) == 'receipt'
    fields = {key.value for key in value.keys if key is not None}
    assert fields == {'command', 'identity', 'affinity', 'topology', 'thread_environment'}
    return fields


@pytest.fixture(params=['control', 'candidate'])
def actual_worker(request, tmp_path):
    name = request.param
    original = read(SAVED / 'workers' / name / 'initialization.json')
    folder = tmp_path / 'workers' / name
    folder.mkdir(parents=True)
    (folder / 'initialization.json').write_bytes((SAVED / 'workers' / name / 'initialization.json').read_bytes())
    saved_identity = original['identity']
    live = {key: deepcopy(value) for key, value in original.items() if key not in augmentation_fields()}
    assert 'identity' not in live and live['status'] == 'ok'
    child = SimpleNamespace(identity=deepcopy(saved_identity), initialization=live,
        process=SimpleNamespace(pid=saved_identity['pid'], poll=lambda: None),
        _ps=SimpleNamespace(pid=saved_identity['pid'], create_time=lambda: saved_identity['created']),
        output_dir=folder, _closed=False, _counter=3, _failure=None, _lock=threading.Lock(),
        absolute_deadline=read(SAVED / 'stages/qualification_start.json')['monotonic_deadline'])
    origin = read(SAVED / 'guardian_claim.json')
    admission = read(SAVED / 'evaluation_admission.json')
    envelope = SimpleNamespace(start=origin['monotonic_origin'], wall_start=origin['unix_origin'],
        deadline=admission['effective_monotonic_deadline'])
    last = read(SAVED / 'workers' / name / 'parent_profiles/request-00000003.json')['capture_snapshot_ns']
    clock = SimpleNamespace(monotonic=lambda: admission['monotonic'] + .001,
        time=lambda: admission['unix_time'] + .001, perf_counter_ns=lambda: last + 1)
    written = []
    def save(path, value):
        # Assert identity/publication checks precede the deadline mutation.
        assert child.absolute_deadline == read(SAVED / 'stages/qualification_start.json')['monotonic_deadline']
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle)
        written.append((path, value))
        return {'path': str(path)}
    common = SimpleNamespace(OUT=tmp_path, read=read, require=require, save=save,
        now=lambda: 'model-free saved-failure fixture', COLLECTION_CUTOFF_SECONDS=7200)
    return name, child, envelope, common, clock, written


def test_exact_original_failure_reproduced_from_actual_unaugmented_ready(actual_worker):
    name, child, envelope, common, clock, written = actual_worker
    failure = read(SAVED / 'failure.json')
    assert "KeyError: 'identity'" in failure['error']
    with pytest.raises(KeyError, match='identity'):
        original_function('advance_idle_worker', common, clock)(child, envelope)
    assert not written
    # The original code changed this idle attribute before raising; no request ran.
    assert child.absolute_deadline == envelope.deadline


def test_corrected_actual_identity_reaches_unchanged_saved_reader(actual_worker, monkeypatch):
    name, child, envelope, common, clock, written = actual_worker
    monkeypatch.setattr(fixed, 'time', clock)
    identity = deepcopy(child.identity)
    fixed.advance_idle_worker(child, envelope, common=common)
    assert child.absolute_deadline == envelope.deadline and len(written) == 1
    path, receipt = written[0]
    assert receipt['worker_pid'] == identity['pid'] and receipt['worker_created'] == identity['created']
    assert receipt['next_request_id'] == 'request-00000004' and receipt['active_request'] is False
    assert receipt['request_allowance_unchanged'] == 10.
    # The reader consumes augmented disk receipts; its init['identity'] is correct.
    for relative in ['stages/qualification_start.json', 'stages/evaluation_start.json', 'evaluation_admission.json',
                     f'workers/{name}/parent_profiles/request-00000003.json']:
        target = common.OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((SAVED / relative).read_bytes())
    first_future_clock = {'parent_profile': {'request_start_ns': clock.perf_counter_ns() + 1}}
    assert original_function('verify_phase_transition', common, clock)(name, envelope, [first_future_clock]) == receipt
    assert 'identity' not in child.initialization  # No production-object schema patch.


@pytest.mark.parametrize('corruption', ['live_created', 'live_pid', 'saved_created', 'saved_ready_pid',
    'live_ready_pid', 'changed_live_receipt', 'reused_pid', 'missing_identity'])
def test_identity_fault_rejects_before_receipt_or_phase_mutation(actual_worker, monkeypatch, corruption):
    name, child, envelope, common, clock, written = actual_worker
    monkeypatch.setattr(fixed, 'time', clock)
    before = child.absolute_deadline
    if corruption == 'live_created': child.identity['created'] += 1
    elif corruption == 'live_pid': child.identity['pid'] += 1
    elif corruption == 'live_ready_pid': child.initialization['result']['pid'] += 1
    elif corruption == 'changed_live_receipt': child.initialization['receipt_elapsed_seconds'] += 1
    elif corruption == 'reused_pid': child._ps.create_time = lambda: child.identity['created'] + 1
    elif corruption == 'missing_identity': del child.identity
    else:
        path = child.output_dir / 'initialization.json'; saved = read(path)
        if corruption == 'saved_created': saved['identity']['created'] += 1
        else: saved['result']['pid'] += 1
        path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises((ValueError, AttributeError, KeyError)):
        fixed.advance_idle_worker(child, envelope, common=common)
    assert child.absolute_deadline == before and not written


@pytest.mark.parametrize('state', ['active', 'failed', 'unqualified', 'closed', 'exited', 'renewed_clock'])
def test_existing_request_and_original_clock_guards_remain(actual_worker, monkeypatch, state):
    name, child, envelope, common, clock, written = actual_worker
    monkeypatch.setattr(fixed, 'time', clock)
    before = child.absolute_deadline
    if state == 'active': child._lock.acquire()
    elif state == 'failed': child._failure = 'saved failure'
    elif state == 'unqualified': child._counter = 2
    elif state == 'closed': child._closed = True
    elif state == 'exited': child.process.poll = lambda: 0
    else: envelope.deadline = envelope.start + 7201
    with pytest.raises(ValueError): fixed.advance_idle_worker(child, envelope, common=common)
    assert child.absolute_deadline == before and not written


def test_receipt_publication_failure_does_not_change_idle_deadline(actual_worker, monkeypatch):
    name, child, envelope, common, clock, written = actual_worker
    monkeypatch.setattr(fixed, 'time', clock)
    before = child.absolute_deadline
    def fail(*args): raise OSError('simulated exclusive publication failure')
    common.save = fail
    with pytest.raises(OSError): fixed.advance_idle_worker(child, envelope, common=common)
    assert child.absolute_deadline == before and not written


def test_actual_saved_attempt_has_no_started_ordinary_request():
    summary = read(SAVED / 'qualification_summary.json')
    assert summary == dict(accepted=True, graph_cases=22, numerical_cases=76, rejected_at=None,
        scientific_decisions=134, startup_parent_requests=6, status='qualified', workers=2)
    assert read(SAVED / 'completion.json')['status'] == 'incomplete'
    assert not list((SAVED / 'observations').glob('*.json'))
    assert not list((SAVED / 'worker_phase_transitions').glob('*.json'))
    for name in ('control', 'candidate'):
        folder = SAVED / 'workers' / name
        assert sorted(path.name for path in folder.glob('request-????????.json')) == [f'request-{i:08d}.json' for i in (1, 2, 3)]
        assert not (folder / 'request-00000004.json').exists()
        closed = read(SAVED / 'closed_workers' / f'{name}.json')
        assert closed['raw']['requests'] == closed['raw']['startup_requests'] == 3
        assert closed['raw']['failed_requests'] == 0


def test_every_identity_access_is_live_or_augmented_disk_not_live_ready_receipt():
    tree = ast.parse((SOURCE / 'control.py').read_text(encoding='utf-8'))
    offending = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript) and
                 isinstance(node.value, ast.Attribute) and node.value.attr == 'initialization' and
                 isinstance(node.slice, ast.Constant) and node.slice.value == 'identity']
    assert len(offending) == 1 and offending[0].lineno == 429
    corrected = ast.parse(Path(fixed.__file__).read_text(encoding='utf-8'))
    assert not any(isinstance(node, ast.Subscript) and ast.unparse(node.value) == 'child.initialization' and
                   isinstance(node.slice, ast.Constant) and node.slice.value == 'identity'
                   for node in ast.walk(corrected))
