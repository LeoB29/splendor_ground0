"""Model-free lifecycle and strict deadline regressions for the new owner."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.splendorust_proxy_calibration_20261113 import common as c, control, service


def test_clipped_limits_never_extend_original_model_or_wall_clock(monkeypatch):
    monkeypatch.setattr(control.time, 'monotonic', lambda: 1000.)
    monkeypatch.setattr(control.time, 'time', lambda: 5000.)
    monkeypatch.setattr(c, 'MODEL_SECONDS', 200.)
    parent = dict(monotonic_origin=900., unix_origin=4900., monotonic_deadline=1800., unix_deadline=5800.)
    limits = control.clipped_limits(parent, 500, model=True)
    assert limits['monotonic_deadline'] == 1100
    assert limits['unix_deadline'] == 5100
    ordinary = control.clipped_limits(parent, 500, model=False)
    assert ordinary['monotonic_deadline'] == 1500 and ordinary['unix_deadline'] == 5500


@pytest.mark.parametrize('mono,wall', [(5., 12.), (12., 5.), (-1., 5.), (5., -1.)])
def test_both_original_clocks_and_origins_are_required(monkeypatch, mono, wall):
    monkeypatch.setattr(c.time, 'monotonic', lambda: mono)
    monkeypatch.setattr(c.time, 'time', lambda: wall)
    with pytest.raises(ValueError, match='dual clock'):
        c.check_deadlines(dict(monotonic_origin=0., unix_origin=0., monotonic_deadline=10., unix_deadline=10.))


def test_cleanup_does_not_kill_reused_pid(monkeypatch):
    ident = dict(pid=777, created=1.)
    killed = []
    class Process:
        returncode = 0
        def poll(self): return self.returncode
        def wait(self, timeout): return self.returncode
    monkeypatch.setattr(c, 'exact_alive', lambda value: False)
    monkeypatch.setattr(control.psutil, 'Process', lambda pid: SimpleNamespace(kill=lambda: killed.append(pid)))
    errors = []
    control.cleanup_owned(Process(), None, ident, {(777, 1.): ident}, errors)
    assert not errors and not killed


def test_owned_handshake_binds_freeze_exact_child_and_live_parent(monkeypatch, tmp_path):
    ident, parent = dict(pid=1, created=1.), dict(pid=2, created=2.)
    owner = dict(identity=ident, parent=parent, windows_job_assigned=True, freeze_sha256='a'*64, role='diagnostic')
    (tmp_path / 'owner.json').write_text(json.dumps(owner), encoding='utf-8')
    monkeypatch.setattr(c, 'identity', lambda: ident)
    monkeypatch.setattr(c, 'exact_alive', lambda value: value == parent)
    assert control.handshake(tmp_path, 'a'*64, 'diagnostic') == owner
    with pytest.raises(ValueError, match='Exact frozen owned role'):
        control.handshake(tmp_path, 'b'*64, 'diagnostic')
    monkeypatch.setattr(c, 'exact_alive', lambda value: False)
    with pytest.raises(ValueError, match='Exact live owner'):
        control.handshake(tmp_path, 'a'*64, 'diagnostic')


def fake_service(monkeypatch, tmp_path, *, elapsed=0.):
    from tools import future_training_rules_20261021 as guard
    from tools.splendorust_proxy_calibration_20261113 import adapters
    out = tmp_path / 'study'
    out.mkdir()
    freeze = out / 'freeze.json'
    plan = dict(worker_inputs=[], corrected_source_receipt=guard.SOURCE_HASHES,
                research_profiles={}, control_profiles={})
    freeze.write_text(c.canonical(plan), encoding='utf-8')
    monkeypatch.setattr(c, 'OUT', out)
    monkeypatch.setattr(c, 'PLAN', freeze)
    monkeypatch.setenv('PROXY_SERVICE_FOLDER', str(out / 'service'))
    monkeypatch.setenv('PROXY_FREEZE_SHA256', c.sha(freeze))
    monkeypatch.setenv('PROXY_ROLE', 'diagnostic')
    packet = dict(type='request', request_id='test-request', payload=dict(player='search'))
    raw = (c.canonical(packet)+'\n').encode()
    stream = io.StringIO()
    monkeypatch.setattr(service.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw)))
    monkeypatch.setattr(service.sys, 'stdout', stream)
    calls = dict(runtime=0, initialize=0, handle=0, close=0, timer_started=0, timer_canceled=0)
    clock = [0.]
    monkeypatch.setattr(service.time, 'perf_counter', lambda: clock[0])
    def runtime():
        calls['runtime'] += 1
        return dict(guard.SOURCE_HASHES)
    monkeypatch.setattr(guard, 'verify_runtime_sources', runtime)
    class Timer:
        def __init__(self, seconds, callback):
            assert seconds == 120
            self.daemon = False
        def start(self): calls['timer_started'] += 1
        def cancel(self): calls['timer_canceled'] += 1
    monkeypatch.setattr(service.threading, 'Timer', Timer)
    def initialize(*args, **kwargs):
        calls['initialize'] += 1
        return dict(decision_guard=kwargs['decision_guard'])
    def handle(context, payload):
        calls['handle'] += 1
        with context['decision_guard']():
            clock[0] += elapsed
            return dict(ok=True)
    def close(context): calls['close'] += 1
    monkeypatch.setattr(adapters, 'initialize', initialize)
    monkeypatch.setattr(adapters, 'handle', handle)
    monkeypatch.setattr(adapters, 'close_context', close)
    return out, stream, calls


def test_service_one_startup_guard_and_finite_decision_watchdog(monkeypatch, tmp_path):
    out, stream, calls = fake_service(monkeypatch, tmp_path)
    assert service.serve() == 0
    assert calls == dict(runtime=1, initialize=1, handle=1, close=1, timer_started=1, timer_canceled=1)
    packets = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [p['type'] for p in packets] == ['ready', 'response']
    receipt = c.read(out / 'service/service_exit.json')
    assert receipt['status'] == 'passed' and receipt['guarded_decisions'] == 1


def test_internal_complete_serialized_response_deadline_is_not_reset(monkeypatch, tmp_path):
    out, stream, calls = fake_service(monkeypatch, tmp_path, elapsed=120.)
    assert service.serve() == 1
    packets = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [p['type'] for p in packets] == ['ready', 'error']
    assert 'serialized internal response deadline' in packets[-1]['error']
    assert c.read(out / 'service/service_exit.json')['status'] == 'failed'
    assert calls['close'] == 1 and calls['handle'] == 1


def test_failed_post_decision_check_preserves_raw_action_and_emits_no_response(monkeypatch, tmp_path):
    from tools.splendorust_proxy_calibration_20261113 import adapters
    out, stream, calls = fake_service(monkeypatch, tmp_path)
    event = dict(event='complete_result_before_worker_validation', arm='search', observer={'test_only': True},
                 action={'test_only_completed_action': True}, diagnostics={'test_only_callbacks': ['a'*64]})
    def rejected(context, payload):
        calls['handle'] += 1
        context['evidence_callback'](event)
        context['evidence_callback'](dict(event='worker_research_decision_failed', error='synthetic post-decision rejection'))
        raise ValueError('synthetic post-decision rejection')
    monkeypatch.setattr(adapters, 'handle', rejected)
    assert service.serve() == 1
    packets = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [p['type'] for p in packets] == ['ready', 'error']
    raw = (out / 'service/ordinary_forensics.jsonl').read_text(encoding='utf-8').splitlines()
    rows = [json.loads(line) for line in raw]
    assert len(rows) == 2 and rows[0]['sequence'] == 0 and rows[0]['previous_sha256'] is None
    assert rows[0]['event'] == dict(transport_request_id='test-request', event=event)
    assert rows[1]['previous_sha256'] == c.digest(rows[0])
    assert rows[1]['sequence'] == 1 and rows[1]['event']['transport_request_id'] == 'test-request'
    assert c.read(out / 'service/service_exit.json')['status'] == 'failed'
    assert 'synthetic post-decision rejection' in c.read(out / 'service/failure.json')['error']
    assert calls['close'] == 1 and calls['handle'] == 1


def test_environment_fixes_threads_and_disables_accelerator_visibility():
    env = control.environment()
    assert env['CUDA_VISIBLE_DEVICES'] == '' and env['PYTHONDONTWRITEBYTECODE'] == '1'
    assert all(env[key] == '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                         'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'))
    assert 'KMP_DUPLICATE_LIB_OK' not in env


def owned_console_fixture():
    """Use the actual accepted host's console-parent topology, adding control."""
    import os
    from tools.splendorust_proxy_calibration_20261113 import adapters
    from tools.splendorust_proxy_calibration_20261113 import ownership
    path = c.ROOT / 'outputs/splendorust_decision_quality_20261027_02/workers/qualification_research/initialization.json'
    recorded = c.read(path)['telemetry']['process_tree']
    parent = next(row for row in recorded if row['role'] == 'python')
    ident = dict(pid=parent['pid'], created=parent['created'])
    native_pids = {row['role']: row['pid'] for row in recorded if row['role'] in ('search', 'reference')}
    native_pids['control'] = 999900
    documents = {name: adapters.profile(16 if name == 'reference' else 64, control=name == 'control')
                 for name in ownership.NATIVE_ROLES}
    binaries = {name: next(item for item in document['artifacts'] if item['path'] == document['native_binary'])
                for name, document in documents.items()}
    console = dict(path=str((Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve()),
                   bytes=12345, sha256='c' * 64)
    rows = []
    for old in recorded:
        if old['role'] == 'python':
            continue
        name = 'console:service' if old['role'] == 'console:python' else old['role']
        item = console if name.startswith('console:') else binaries[name]
        rows.append(dict(pid=old['pid'], created=old['created'], parent_pid=old['parent_pid'], role=name,
            executable=str(c.resolve(item)), executable_sha256=item['sha256'], executable_bytes=item['bytes'], affinity=[0]))
    rows += [dict(pid=native_pids['control'], created=1793099505.1, parent_pid=ident['pid'], role='control',
                  executable=str(c.resolve(binaries['control'])), executable_sha256=binaries['control']['sha256'],
                  executable_bytes=binaries['control']['bytes'], affinity=[0]),
             dict(pid=999901, created=1793099505.2, parent_pid=native_pids['control'], role='console:control',
                  executable=str(c.resolve(console)), executable_sha256=console['sha256'], executable_bytes=console['bytes'], affinity=[0])]
    return ident, native_pids, rows, binaries, console


def test_real_windows_console_topology_is_allowed_and_retained():
    from tools.splendorust_proxy_calibration_20261113 import ownership as o
    ident, pids, rows, binaries, console = owned_console_fixture()
    book = o.make_book(ident, pids, rows, rows, binaries, console)
    proof = o.validate_book(book, binaries, console)
    assert proof['native_processes'] == 3 and proof['observed_owned_descendants'] == 7
    assert len(book['all_owned_identities']) == 7


def test_console_can_inherit_host_affinity_before_service_is_pinned():
    import os
    from tools.splendorust_proxy_calibration_20261113 import ownership as o
    ident, pids, rows, binaries, console = owned_console_fixture()
    for row in rows:
        if row['role'].startswith('console:'):
            row['affinity'] = list(range(os.cpu_count()))
    book = o.make_book(ident, pids, rows, rows, binaries, console)
    assert o.validate_book(book, binaries, console)['observed_owned_descendants'] == 7


def test_native_replacement_is_rejected_but_all_console_lifetimes_are_retained():
    from copy import deepcopy
    from tools.splendorust_proxy_calibration_20261113 import ownership as o
    ident, pids, rows, binaries, console = owned_console_fixture()
    end = deepcopy(rows)
    next(row for row in end if row['role'] == 'search')['created'] += 1
    with pytest.raises(ValueError, match='exact native identities'):
        o.make_book(ident, pids, rows, end, binaries, console)
    end = deepcopy(rows)
    changed = next(row for row in end if row['role'] == 'console:control')
    changed['pid'] += 100
    changed['created'] += 1
    book = o.make_book(ident, pids, rows, end, binaries, console)
    assert len(book['all_owned_identities']) == 8
    altered = deepcopy(book)
    altered['all_owned_identities'].pop()
    with pytest.raises(ValueError, match='cleanup union'):
        o.validate_book(altered, binaries, console)


@pytest.mark.parametrize('mutation', ['console_parent', 'console_executable', 'native_hash', 'affinity'])
def test_ownership_rejects_unknown_or_changed_descendants(mutation):
    from tools.splendorust_proxy_calibration_20261113 import ownership as o
    ident, pids, rows, binaries, console = owned_console_fixture()
    con = next(row for row in rows if row['role'] == 'console:service')
    native = next(row for row in rows if row['role'] == 'search')
    if mutation == 'console_parent': con['parent_pid'] = 777777
    if mutation == 'console_executable': con['executable'] = str(Path(con['executable']).with_name('other.exe'))
    if mutation == 'native_hash': native['executable_sha256'] = 'a' * 64
    if mutation == 'affinity': native['affinity'] = [2]
    with pytest.raises(ValueError):
        o.validate_snapshot(ident, pids, rows, binaries, console)


def test_actual_inspection_uses_every_native_and_console(monkeypatch):
    import psutil
    from tools.splendorust_proxy_calibration_20261113 import ownership as o
    ident, pids, rows, binaries, console = owned_console_fixture()
    children = [SimpleNamespace(pid=row['pid'], ppid=lambda row=row: row['parent_pid'],
        create_time=lambda row=row: row['created'], exe=lambda row=row: row['executable'],
        cpu_affinity=lambda: [0]) for row in rows]
    parent = SimpleNamespace(create_time=lambda: ident['created'], is_running=lambda: True,
        children=lambda recursive: children)
    monkeypatch.setattr(psutil, 'Process', lambda pid: parent)
    artifacts = {str(c.resolve(item)): item for item in [*binaries.values(), console]}
    monkeypatch.setattr(c, 'artifact', lambda path: artifacts[str(Path(path).resolve())])
    actual = o.inspect_service_tree(ident, pids, binaries, console)
    assert len(actual) == 7 and {row['pid'] for row in actual} == {row['pid'] for row in rows}


def test_evidence_size_scan_tolerates_only_status_rename(monkeypatch):
    class Vanished:
        def __init__(self, name): self.name = name
        def is_file(self): return True
        def stat(self): raise FileNotFoundError(self.name)
    monkeypatch.setattr(c, 'OUT', SimpleNamespace(rglob=lambda pattern: [Vanished('progress.tmp')]))
    assert control.output_bytes() == 0
    monkeypatch.setattr(c, 'OUT', SimpleNamespace(rglob=lambda pattern: [Vanished('result.json')]))
    with pytest.raises(ValueError, match='Only transient'):
        control.output_bytes()
