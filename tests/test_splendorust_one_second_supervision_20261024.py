"""Model-free counterexamples for the new three-process ownership/RSS sampler."""
from pathlib import Path
from types import SimpleNamespace

from tools.splendorust_one_second_20261024 import common


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
