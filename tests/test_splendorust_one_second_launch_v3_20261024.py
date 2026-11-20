"""Model-free watchdog cleanup ownership and clock counterexamples."""
from types import SimpleNamespace

from tools.splendorust_one_second_v3_20261024 import launch


class Child:
    def __init__(self, pid=11, created=1.0):
        self.pid, self.created, self.killed = pid, created, False
        self.descendants = []
        self.waits = []
    def create_time(self): return self.created
    def children(self, recursive=False): return self.descendants
    def kill(self): self.killed = True
    def wait(self, timeout):
        self.waits.append(timeout)
        return 0


class Handle:
    pid = 11
    returncode = 0
    def poll(self): return self.returncode
    def kill(self): self.returncode = -1
    def wait(self, timeout): return self.returncode


def test_watchdog_confirms_exit_without_instant_visibility_check(monkeypatch):
    parent = Child()
    parent.descendants = [Child(12), Child(13)]
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent)
    job = SimpleNamespace(closed=False)
    job.close = lambda: setattr(job, 'closed', True)
    result = launch.cleanup_owned(Handle(), job, (), monotonic_origin=launch.time.monotonic(),
        unix_origin=launch.time.time(), root_identity={'pid': 11, 'created': 1.0})
    assert job.closed and not result['errors'] and len(result['confirmed']) == 3
    assert all(child.killed for child in [parent] + parent.descendants)


def test_watchdog_cleanup_waits_share_one_remaining_utc_budget(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(launch.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(launch.time, 'time', lambda: clock[0] + 1000)
    parent = Child()
    parent.descendants = [Child(12), Child(13)]
    def wait(timeout):
        parent.waits.append(timeout)
        if timeout <= 0:
            raise launch.psutil.TimeoutExpired(timeout, pid=parent.pid)
        clock[0] += min(.03, timeout)
        return 0
    for child in [parent] + parent.descendants:
        child.wait = wait
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent)
    result = launch.cleanup_owned(Handle(), None, (), monotonic_origin=100,
        unix_origin=1100 - launch.c.JOB_SECONDS + .05,
        root_identity={'pid': 11, 'created': 1.0})
    assert abs(result['effective_allowance_seconds'] - .05) < 1e-9
    assert result['elapsed_seconds'] <= .05 + 1e-9
    assert parent.waits[-1] == 0.0 and result['unconfirmed']
    assert result['errors']


def test_watchdog_expired_global_clock_grants_no_cleanup_extension(monkeypatch):
    parent = Child()
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent)
    result = launch.cleanup_owned(Handle(), None, (),
        monotonic_origin=launch.time.monotonic()-launch.c.JOB_SECONDS-1,
        unix_origin=launch.time.time(), root_identity={'pid': 11, 'created': 1.0})
    assert parent.killed and parent.waits == [0.0]
    assert result['effective_allowance_seconds'] == 0.0


def test_watchdog_preserves_unowned_reused_runner_pid(monkeypatch):
    replacement = Child(created=2.0)
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: replacement)
    result = launch.cleanup_owned(Handle(), None, (), monotonic_origin=launch.time.monotonic(),
        unix_origin=launch.time.time(), root_identity={'pid': 11, 'created': 1.0})
    assert not replacement.killed and not replacement.waits
    assert result['errors'] and result['observed_owned_identities'] == 0


def test_watchdog_termination_access_denied_race_uses_confirmed_exit(monkeypatch):
    parent = Child()
    def already_exited(): raise launch.psutil.AccessDenied(parent.pid)
    parent.kill = already_exited
    monkeypatch.setattr(launch.psutil, 'Process', lambda pid: parent)
    result = launch.cleanup_owned(Handle(), None, (), monotonic_origin=launch.time.monotonic(),
        unix_origin=launch.time.time(), root_identity={'pid': 11, 'created': 1.0})
    assert not result['errors'] and len(result['confirmed']) == 1
