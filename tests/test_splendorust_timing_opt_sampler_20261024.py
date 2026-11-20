"""Model-free Windows sampler equivalence, identity and API failure tests."""
from copy import deepcopy
import ctypes
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from tools.splendorust_timing_opt_20261024 import sampler as s


class FakeAPI:
    def __init__(self, rows, *, create_error=0, first_error=0, next_error=0, close_error=0):
        self.rows = rows
        self.create_error, self.first_error, self.next_error, self.close_error = create_error, first_error, next_error, close_error
        self.error = 0
        self.created, self.closed, self.first, self.next, self.index = 0, 0, 0, 0, 0

    def CreateToolhelp32Snapshot(self, flags, pid):
        assert flags == 2 and pid == 0
        self.created += 1
        self.error = self.create_error
        return s.INVALID_HANDLE_VALUE if self.create_error else 321

    def _write(self, pointer):
        entry = pointer._obj
        assert entry.dwSize == ctypes.sizeof(s.PROCESSENTRY32W)
        pid, parent, threads = self.rows[self.index]
        entry.th32ProcessID, entry.th32ParentProcessID, entry.cntThreads = pid, parent, threads
        return 1

    def Process32FirstW(self, handle, pointer):
        assert handle == 321
        self.first += 1
        if self.first_error or not self.rows:
            self.error = self.first_error or 18
            return 0
        return self._write(pointer)

    def Process32NextW(self, handle, pointer):
        assert handle == 321
        self.next += 1
        self.index += 1
        if self.next_error or self.index >= len(self.rows):
            self.error = self.next_error or 18
            return 0
        return self._write(pointer)

    def CloseHandle(self, handle):
        assert handle == 321
        self.closed += 1
        self.error = self.close_error
        return int(not self.close_error)


def use_api(monkeypatch, api):
    monkeypatch.setattr(s, '_get_error', lambda: api.error)
    monkeypatch.setattr(s, '_clear_error', lambda: setattr(api, 'error', 0))
    return api


def test_snapshot_returns_only_declared_metadata_and_closes_once(monkeypatch):
    api = use_api(monkeypatch, FakeAPI([(10, 1, 4), (11, 10, 2)]))
    assert s.capture_process_snapshot(_api=api) == {10: {'parent_pid': 1, 'threads': 4}, 11: {'parent_pid': 10, 'threads': 2}}
    assert (api.created, api.first, api.next, api.closed) == (1, 1, 2, 1)


@pytest.mark.parametrize('kwargs,operation,closed', [
    ({'create_error': 5}, 'CreateToolhelp32Snapshot', 0),
    ({'first_error': 5}, 'Process32FirstW', 1),
    ({'first_error': 18}, 'Process32FirstW', 1),
    ({'next_error': 5}, 'Process32NextW', 1),
    ({'close_error': 6}, 'CloseHandle', 1),
])
def test_snapshot_api_faults_never_retry_or_return_partial_data(monkeypatch, kwargs, operation, closed):
    api = use_api(monkeypatch, FakeAPI([(10, 1, 4)], **kwargs))
    with pytest.raises(OSError, match=operation):
        s.capture_process_snapshot(_api=api)
    assert api.created == 1 and api.closed == closed


def test_snapshot_keeps_primary_error_and_reports_cleanup_failure(monkeypatch):
    api = use_api(monkeypatch, FakeAPI([(10, 1, 4)], next_error=5, close_error=6))
    with pytest.raises(OSError, match='Process32NextW') as caught:
        s.capture_process_snapshot(_api=api)
    assert any('CloseHandle' in note for note in caught.value.__notes__)


@pytest.mark.parametrize('rows,maximum', [([(10, 1, 4), (10, 1, 4)], 32768), ([(10, 1, 4), (11, 10, 1)], 1)])
def test_duplicate_and_overlarge_snapshots_are_rejected_and_closed(monkeypatch, rows, maximum):
    api = use_api(monkeypatch, FakeAPI(rows))
    with pytest.raises(ValueError):
        s.capture_process_snapshot(_api=api, max_entries=maximum)
    assert api.closed == 1


@pytest.mark.parametrize('maximum', [0, -1, True, 1., 32769])
def test_snapshot_bound_cannot_be_disabled(maximum):
    with pytest.raises(ValueError):
        s.capture_process_snapshot(max_entries=maximum, _api=object())


class Process:
    def __init__(self, pid, parent, path, *, created=None, threads=2, rss=100, affinity=None):
        self.pid, self.parent, self.path = pid, parent, path
        self.created = float(pid) if created is None else created
        self.thread_count, self.rss = threads, rss
        self.affinity = [4] if affinity is None else affinity
        self.descendants = []
        self.calls = {}

    def _record(self, name):
        self.calls[name] = self.calls.get(name, 0)+1

    def children(self, recursive):
        assert recursive is True
        self._record('children')
        return self.descendants

    def create_time(self):
        self._record('create_time')
        return self.created

    def exe(self):
        self._record('exe')
        return str(self.path)

    def ppid(self):
        self._record('ppid')
        return self.parent

    def num_threads(self):
        self._record('num_threads')
        return self.thread_count

    def memory_info(self):
        self._record('memory_info')
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        self._record('cpu_times')
        return SimpleNamespace(user=1., system=.5)

    def cpu_affinity(self, value=None):
        self._record('cpu_affinity')
        if value is not None:
            self.affinity = list(value)
        return list(self.affinity)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    paths = {name: tmp_path/(name+'.exe') for name in ('python', 'search', 'reference', 'console')}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    processes = {
        100: Process(100, 1, paths['python']),
        101: Process(101, 100, paths['search']),
        102: Process(102, 100, paths['reference']),
        103: Process(103, 100, paths['console']),
        104: Process(104, 101, paths['console']),
        105: Process(105, 102, paths['console']),
    }
    processes[100].descendants = [processes[pid] for pid in (104, 105, 103, 101, 102)]
    def process(pid):
        if pid not in processes:
            raise psutil.NoSuchProcess(pid)
        return processes[pid]
    monkeypatch.setattr(s.psutil, 'Process', process)
    snapshots = []
    def capture():
        snapshots.append(True)
        return {pid: dict(parent_pid=proc.parent, threads=proc.thread_count) for pid, proc in processes.items()}
    monkeypatch.setattr(s, 'capture_process_snapshot', capture)
    def worker(cls=s.BatchedStudyWorker):
        value = object.__new__(cls)
        value._ps = processes[100]
        value._last_telemetry = {}
        value._failure = value._failure_detail = None
        value._tree_cpu, value._owners, value._verified = {}, {}, {}
        value._python, value._console = paths['python'].resolve(), paths['console'].resolve()
        value._natives = {paths['search'].resolve(): 'search', paths['reference'].resolve(): 'reference'}
        value._expected = {path.resolve(): s.frozen.file_hash(path) for path in paths.values()}
        value.affinity, value._initializing = [4], False
        value._peak_rss, value.max_rss_bytes = 0, 2*1024**3
        return value
    return SimpleNamespace(processes=processes, paths=paths, worker=worker, snapshots=snapshots, capture=capture)


def test_old_and_batched_telemetry_are_identical_on_fixed_process_tree(setup):
    old = setup.worker(s.frozen.StudyWorker)
    expected = old._sample()
    for process in setup.processes.values():
        process.calls.clear()
    optimized = setup.worker()
    actual = optimized._sample()
    assert actual == expected
    assert optimized._failure is None
    assert len(setup.snapshots) == 1
    assert {row['role'] for row in actual['process_tree']} == {'python', 'search', 'reference', 'console:python', 'console:search', 'console:reference'}
    for process in setup.processes.values():
        assert process.calls.get('ppid', 0) == process.calls.get('num_threads', 0) == 0
        assert process.calls['exe'] == process.calls['memory_info'] == process.calls['cpu_times'] == process.calls['cpu_affinity'] == 1
        assert process.calls['create_time'] >= 4


def test_every_poll_repeats_discovery_snapshot_and_mutable_checks(setup):
    worker = setup.worker()
    first = worker._sample()
    setup.processes[101].thread_count = 8
    setup.processes[101].rss = 200
    second = worker._sample()
    assert len(setup.snapshots) == 2
    assert setup.processes[100].calls['children'] == 2
    assert second['observed_threads'] == first['observed_threads']+6
    assert second['rss_bytes'] == first['rss_bytes']+100
    assert setup.processes[101].calls['exe'] == setup.processes[101].calls['memory_info'] == 2


def test_snapshot_failure_becomes_preserved_supervisor_failure(setup, monkeypatch):
    worker = setup.worker()
    prior = worker._sample()
    def fail():
        raise OSError(5, 'Tool Help denied')
    monkeypatch.setattr(s, 'capture_process_snapshot', fail)
    assert worker._sample() == prior
    assert worker._failure == 'process_tree_unobservable'
    assert 'denied' in worker._failure_detail


@pytest.mark.parametrize('when', ['before', 'after', 'during'])
def test_pid_reuse_cannot_receive_previous_snapshot_metadata(setup, monkeypatch, when):
    # Isolate the changed process; a surviving console of an unrecognized
    # parent would also fail the unchanged later ownership checks.
    setup.processes[100].descendants = [proc for proc in setup.processes[100].descendants if proc.pid != 104]
    counts = {}
    def fresh(proc):
        counts[proc.pid] = counts.get(proc.pid, 0)+1
        changed_at = {'before': 1, 'after': 2, 'during': 3}[when]
        return proc.pid, proc.created+(1 if proc.pid == 101 and counts[proc.pid] >= changed_at else 0)
    monkeypatch.setattr(s, 'fresh_identity', fresh)
    worker = setup.worker()
    worker._sample()
    assert worker._failure in ('process_tree_unobservable', 'process_tree_limit')
    assert 'identity changed' in worker._failure_detail


def test_missing_live_owned_pid_is_fail_closed(setup, monkeypatch):
    setup.processes[100].descendants = [proc for proc in setup.processes[100].descendants if proc.pid != 104]
    snapshot = setup.capture()
    snapshot.pop(101)
    monkeypatch.setattr(s, 'capture_process_snapshot', lambda: snapshot)
    worker = setup.worker()
    worker._sample()
    assert worker._failure == 'process_tree_limit'
    assert 'missing' in worker._failure_detail


def test_truly_exited_process_can_be_skipped_without_stale_metadata(setup, monkeypatch):
    setup.processes[100].descendants = [setup.processes[101]]
    original = s.fresh_identity
    def fresh(proc):
        if proc.pid == 101:
            raise psutil.NoSuchProcess(101)
        return original(proc)
    monkeypatch.setattr(s, 'fresh_identity', fresh)
    worker = setup.worker()
    sample = worker._sample()
    assert worker._failure is None
    assert [row['pid'] for row in sample['process_tree']] == [100]


@pytest.mark.parametrize('metadata', [dict(parent_pid=-1, threads=2), dict(parent_pid=100, threads=True),
    dict(parent_pid=100, threads=-1), dict(parent_pid=100, threads=2, extra=0), None])
def test_malformed_owned_metadata_is_rejected(setup, monkeypatch, metadata):
    snapshot = setup.capture()
    snapshot[101] = metadata
    monkeypatch.setattr(s, 'capture_process_snapshot', lambda: snapshot)
    worker = setup.worker()
    worker._sample()
    assert worker._failure == 'process_tree_limit'


def test_unchanged_affinity_limit_still_rejects_changes(setup):
    setup.processes[101].affinity = [5]
    worker = setup.worker()
    worker._sample()
    assert worker._failure == 'affinity_changed'


def test_unchanged_memory_limit_still_rejects_excess(setup):
    worker = setup.worker()
    worker.max_rss_bytes = 500
    worker._sample()
    assert worker._failure == 'rss_limit'


def test_unchanged_ownership_check_rejects_helper_replacement(setup):
    worker = setup.worker()
    worker._owners = {'search': (101, 99.)}
    worker._sample()
    assert worker._failure == 'process_tree_limit'
    assert 'replacement' in worker._failure_detail


def test_unchanged_path_and_binary_authentication_still_rejects_unknown_executable(setup, tmp_path):
    path = tmp_path/'unknown.exe'
    path.write_bytes(b'not approved')
    setup.processes[101].path = path
    worker = setup.worker()
    worker._sample()
    assert worker._failure == 'process_tree_limit'


def test_console_startup_affinity_repair_retains_old_behavior(setup):
    setup.processes[104].affinity = [5]
    worker = setup.worker()
    worker._initializing = True
    sample = worker._sample()
    assert worker._failure is None
    row = next(row for row in sample['process_tree'] if row['pid'] == 104)
    assert row['startup_affinity_before'] == [5] and row['affinity'] == [4]


def test_clock_reader_and_cleanup_methods_are_inherited_unchanged():
    for name in ('request', '_receive', '_stdout_reader', '_stderr_reader', 'close', '_log'):
        assert getattr(s.BatchedStudyWorker, name) is getattr(s.frozen.StudyWorker, name)

