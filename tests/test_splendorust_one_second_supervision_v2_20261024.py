"""Model-free counterexamples for the new three-process ownership/RSS sampler."""
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys
import time

import pytest

from tools.splendorust_one_second_v2_20261024 import common


class Process:
    def __init__(self, pid, parent, executable, *, rss=100, created=1, affinity=None):
        self.pid, self.parent, self.executable = pid, parent, executable
        self.rss, self.created = rss, created
        self.affinity = affinity or [0]
        self.descendants = []

    def create_time(self): return self.created
    def exe(self): return str(self.executable)
    def ppid(self): return self.parent
    def memory_info(self): return SimpleNamespace(rss=self.rss)
    def cpu_times(self): return SimpleNamespace(user=1, system=.5)
    def num_threads(self): return 1
    def cpu_affinity(self): return self.affinity
    def children(self, recursive=False): return list(self.descendants)


def sampler(monkeypatch):
    python, search, greedy = map(lambda s: Path(s).resolve(), ['stub_python.exe', 'stub_search.exe', 'stub_greedy.exe'])
    root = Process(10, 1, python)
    root.descendants = [Process(11, 10, search), Process(12, 10, greedy)]
    worker = object.__new__(common.StudyWorker)
    worker._ps, worker._python = root, python
    worker._natives, worker._console = {search: 'search', greedy: 'greedy'}, None
    worker._expected = {p: 'verified_hash' for p in [python, search, greedy]}
    worker._verified, worker._owners, worker._tree_cpu = {}, {}, {}
    worker._failure, worker._failure_detail = None, None
    worker._last_telemetry, worker._peak_rss = {}, 0
    worker._initializing, worker.affinity = False, [0]
    worker.max_rss_bytes = 1000
    monkeypatch.setattr(common, 'file_hash', lambda path: 'verified_hash')
    return worker, root


def test_owns_exact_separate_search_and_greedy_native_children(monkeypatch):
    worker, root = sampler(monkeypatch)
    result = worker._sample()
    assert worker._failure is None
    assert set(worker._owners) == {'python', 'search', 'greedy'}
    assert result['rss_bytes'] == 300 and result['cpu_seconds'] == 4.5
    assert result['resource_scope'] == 'sampled_worker_and_all_descendants'


def test_combined_memory_limit_is_not_only_python_memory(monkeypatch):
    worker, root = sampler(monkeypatch)
    root.descendants[0].rss = 700
    root.descendants[1].rss = 700
    result = worker._sample()
    assert result['rss_bytes'] == 1500
    assert worker._failure == 'rss_limit'


def test_native_replacement_with_same_executable_is_rejected(monkeypatch):
    worker, root = sampler(monkeypatch)
    worker._sample()
    root.descendants[0].created = 2
    worker._sample()
    assert worker._failure == 'process_tree_limit'
    assert 'replacement' in worker._failure_detail.lower()


def test_native_cannot_spawn_an_additional_allowed_executable(monkeypatch):
    worker, root = sampler(monkeypatch)
    root.descendants.append(Process(13, 11, root.descendants[1].executable))
    worker._sample()
    assert worker._failure == 'process_tree_limit'


def test_descendant_cpu_affinity_change_is_rejected(monkeypatch):
    worker, root = sampler(monkeypatch)
    root.descendants[0].affinity = [1]
    worker._sample()
    assert worker._failure == 'affinity_changed'


def test_changed_native_executable_is_rejected_before_accepting_telemetry(monkeypatch):
    worker, root = sampler(monkeypatch)
    monkeypatch.setattr(common, 'file_hash', lambda path: 'wrong_hash' if path == root.descendants[0].executable else 'verified_hash')
    worker._sample()
    assert worker._failure == 'process_tree_limit'
    assert 'executable' in worker._failure_detail


def cleanup_worker(tmp_path, *, primary=None):
    worker = object.__new__(common.StudyWorker)
    worker._closed, worker._ps, worker._job, worker.process = False, None, None, None
    worker._tree_cpu = {(12345, 1.0): 0.0}
    worker._threads, worker._logs = [], {}
    worker.output_dir = tmp_path
    worker.absolute_deadline = time.monotonic() + 10
    if primary is not None:
        worker._primary_failure = primary
    return worker


class ExitedButVisibleProcess:
    pid = 12345
    def create_time(self): return 1.0
    def kill(self): pass
    def wait(self, timeout): return 0
    def is_running(self):
        raise AssertionError('Visibility must not decide exited-process completion')


def test_cleanup_accepts_actual_exit_when_windows_process_remains_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: ExitedButVisibleProcess())
    worker = cleanup_worker(tmp_path)
    result = worker.close()
    assert result['status'] == 'passed' and not result['alive']
    assert result['confirmed'][0]['state'] == 'exit_wait_confirmed'
    assert result['confirmed'][0]['exit_status'] == 0
    assert worker.close() is result


def test_cleanup_accepts_confirmed_exit_after_termination_race(monkeypatch, tmp_path):
    class AlreadyExited(ExitedButVisibleProcess):
        def kill(self):
            raise common.psutil.AccessDenied(self.pid)
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: AlreadyExited())
    worker = cleanup_worker(tmp_path)
    assert worker.close()['status'] == 'passed'


def test_cleanup_failure_is_secondary_to_original_worker_failure(monkeypatch, tmp_path):
    class StillRunning(ExitedButVisibleProcess):
        def wait(self, timeout):
            raise common.psutil.TimeoutExpired(timeout, pid=self.pid)
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: StillRunning())
    primary = {'status': 'worker_error', 'detail': 'original checker failure'}
    worker = cleanup_worker(tmp_path, primary=primary)
    result = worker.close()
    assert result['status'] == 'failed' and result['primary_failure'] == primary
    assert len(result['alive']) == 1
    saved = json.loads((tmp_path / 'cleanup.json').read_text())
    assert saved['primary_failure']['detail'] == 'original checker failure'


def test_primary_player_failure_cannot_finish_game_with_unconfirmed_cleanup(monkeypatch, tmp_path):
    class StillRunning(ExitedButVisibleProcess):
        def wait(self, timeout):
            raise common.psutil.TimeoutExpired(timeout, pid=self.pid)
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: StillRunning())
    primary = {'status': 'timeout', 'detail': 'original one-second timeout'}
    worker = cleanup_worker(tmp_path, primary=primary)
    # Immediate request-path cleanup preserves the timeout receipt for audit.
    assert worker.close()['primary_failure'] == primary
    # A normal exit would otherwise let play_game accept a forfeit and start
    # the next game while an old process remained unconfirmed.
    with pytest.raises(RuntimeError, match='cleanup acceptance failed'):
        with worker:
            pass
    assert json.loads((tmp_path / 'cleanup.json').read_text())['primary_failure'] == primary


def test_active_checker_exception_remains_primary_when_cleanup_also_fails(monkeypatch, tmp_path):
    class StillRunning(ExitedButVisibleProcess):
        def wait(self, timeout):
            raise common.psutil.TimeoutExpired(timeout, pid=self.pid)
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: StillRunning())
    worker = cleanup_worker(tmp_path)
    with pytest.raises(ValueError, match='original checker exception'):
        with worker:
            raise ValueError('original checker exception')
    saved = json.loads((tmp_path / 'cleanup.json').read_text())
    assert saved['status'] == 'failed'
    assert 'original checker exception' in saved['primary_failure']['detail']


def test_confirmed_cleanup_allows_original_player_forfeit(monkeypatch, tmp_path):
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: ExitedButVisibleProcess())
    worker = cleanup_worker(tmp_path, primary={'status': 'timeout', 'detail': 'preserved timeout'})
    with worker:
        worker.close()
    assert worker.cleanup_receipt['status'] == 'passed'


def test_bookkeeping_failure_after_player_failure_stops_acceptance(monkeypatch, tmp_path):
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: ExitedButVisibleProcess())
    worker = cleanup_worker(tmp_path, primary={'status': 'timeout', 'detail': 'preserved timeout'})
    def rejected():
        raise ValueError('bookkeeping failed')
    worker.study_after_close = rejected
    with pytest.raises(RuntimeError, match='cleanup acceptance failed'):
        with worker:
            worker.close()
    saved = json.loads((tmp_path / 'cleanup_callback_error.json').read_text())
    assert saved['primary_failure']['status'] == 'timeout'


def test_cleanup_failure_without_primary_rejects_acceptance(monkeypatch, tmp_path):
    class StillRunning(ExitedButVisibleProcess):
        def wait(self, timeout):
            raise common.psutil.TimeoutExpired(timeout, pid=self.pid)
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: StillRunning())
    worker = cleanup_worker(tmp_path)
    with pytest.raises(RuntimeError, match='cleanup failed'):
        worker.close()
    assert json.loads((tmp_path / 'cleanup.json').read_text())['status'] == 'failed'


def test_cleanup_wait_is_clamped_by_outer_stage_and_global_deadline(monkeypatch, tmp_path):
    waits = []
    class Process(ExitedButVisibleProcess):
        def wait(self, timeout):
            waits.append(timeout)
            return 0
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: Process())
    worker = cleanup_worker(tmp_path)
    worker.absolute_deadline = time.monotonic() - .001
    result = worker.close()
    assert waits == [0.0] and result['effective_allowance_seconds'] == 0.0


def test_reused_pid_is_not_terminated(monkeypatch, tmp_path):
    class Replacement(ExitedButVisibleProcess):
        def create_time(self): return 2.0
        def kill(self): raise AssertionError('Unowned replacement must survive')
    monkeypatch.setattr(common.psutil, 'Process', lambda pid: Replacement())
    worker = cleanup_worker(tmp_path)
    result = worker.close()
    assert result['status'] == 'passed'
    assert result['confirmed'][0]['state'] == 'original_identity_absent'


def test_cleanup_terminates_and_waits_for_actual_owned_child(tmp_path):
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        worker = cleanup_worker(tmp_path)
        worker.process = process
        worker._ps = common.psutil.Process(process.pid)
        worker._tree_cpu = {(process.pid, worker._ps.create_time()): 0.0}
        result = worker.close()
        assert result['status'] == 'passed'
        assert process.poll() is not None
        assert result['elapsed_seconds'] < 5.5
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
