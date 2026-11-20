import hashlib
import json

from tools.matched_strength_recovery_20261022 import launch


class Child:
    pid = 17

    def __init__(self, wait_error=False):
        self.exited = False
        self.wait_error = wait_error

    def poll(self):
        return 1 if self.exited else None

    def kill(self):
        self.exited = True

    def wait(self, timeout):
        if self.wait_error:
            raise TimeoutError('synthetic wait failure')
        return 1


class Process:
    def __init__(self, pid, created, children=()):
        self.pid, self.created, self.descendants = pid, created, children
        self.killed = False
        self.suspended = False

    def create_time(self):
        return self.created

    def suspend(self):
        self.suspended = True

    def children(self, recursive):
        return self.descendants

    def kill(self):
        self.killed = True

    def is_running(self):
        return not self.killed


def test_cap_cleanup_closes_only_authenticated_owned_tree(monkeypatch):
    descendant = Process(18, 2)
    parent = Process(17, 1, [descendant])
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: {17: parent, 18: descendant}[pid])
    result = launch.close_owned_tree(Child(), {'pid': 17, 'created': 1})
    assert result['parent_closed'] and not result['surviving_descendants']
    assert result['errors'] == [] and descendant.killed


def test_reused_descendant_pid_is_never_killed(monkeypatch):
    old = Process(18, 2)
    replacement = Process(18, 3)
    parent = Process(17, 1, [old])
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent if pid == 17 else replacement)
    result = launch.close_owned_tree(Child(), {'pid': 17, 'created': 1})
    assert not replacement.killed and not result['surviving_descendants']


def test_wait_failure_returns_durable_cleanup_error_instead_of_raising(monkeypatch):
    parent = Process(17, 1)
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent)
    result = launch.close_owned_tree(Child(wait_error=True), {'pid': 17, 'created': 1})
    assert result['parent_closed']
    assert result['errors'] == ['Owned parent wait: TimeoutError']


def test_reused_parent_pid_is_not_suspended_or_enumerated(monkeypatch):
    descendant = Process(18, 3)
    replacement = Process(17, 2, [descendant])
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: replacement)
    result = launch.close_owned_tree(Child(), {'pid': 17, 'created': 1})
    assert result['parent_closed'] and not replacement.suspended and not descendant.killed
    assert result['captured_descendants'] == [] and result['errors']


def test_unauthenticated_parent_uses_only_owned_handle(monkeypatch):
    parent = Process(17, 1)
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent)
    result = launch.close_owned_tree(Child())
    assert result['parent_closed'] and not parent.suspended and result['errors']


def test_popen_failure_preserves_execution_receipt_without_retry(monkeypatch, tmp_path):
    base = tmp_path / 'recovery'; base.mkdir()
    source = tmp_path / 'source'; source.mkdir()
    freeze = json.dumps({'topology': {'allowed_logical_processors': [2]}}).encode()
    (base / 'freeze.json').write_bytes(freeze)
    (source / 'launch.json').write_text('{}')
    monkeypatch.setattr(launch, 'BASE', base)
    monkeypatch.setattr(launch, 'SOURCE', source)
    monkeypatch.setattr(launch.psutil, 'Process', lambda: object())
    monkeypatch.setattr(launch, 'identity', lambda process: {'in_any_windows_job': False, 'affinity': [2]})
    calls = []
    def fail_start(*args, **kwargs):
        calls.append(args)
        raise OSError('synthetic launch failure')
    monkeypatch.setattr(launch.subprocess, 'Popen', fail_start)
    assert launch.run(hashlib.sha256(freeze).hexdigest()) == 1
    receipt = json.loads((base / 'runner_execution.json').read_text())
    assert len(calls) == 1 and receipt['pid'] is None and receipt['automatic_retries'] == 0
    assert 'synthetic launch failure' in receipt['error']
