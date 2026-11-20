"""Symmetric constructor-time aggregate supervision and immutable outputs."""
from pathlib import Path
import pytest

from tools.splendorust_timing_opt_20261024 import common as c
from tools.splendorust_timing_opt_20261024.sampler import BatchedStudyWorker


@pytest.fixture
def stub_workers(monkeypatch):
    calls = []
    def initialize(self, *args, **kwargs):
        self._failure = self._failure_detail = None
        self.initial_args, self.initial_kwargs = args, kwargs
        calls.append(('constructor', type(self).__name__))
        self.initial_sample = self._sample()
    def sample(self):
        calls.append(('sample', type(self).__name__))
        return {'rss_bytes': 123, 'process_tree': []}
    monkeypatch.setattr(c.original.StudyWorker, '__init__', initialize)
    monkeypatch.setattr(c.original.StudyWorker, '_sample', sample)
    monkeypatch.setattr(BatchedStudyWorker, '_sample', sample)
    return calls


@pytest.mark.parametrize('cls', [c.BaselineStudyWorker, c.CandidateStudyWorker])
def test_aggregate_guard_is_installed_before_base_constructor_samples(stub_workers, cls):
    observed = []
    def guard(child, telemetry):
        observed.append((child, telemetry))
        assert stub_workers[-1] == ('sample', cls.__name__)
    child = cls('worker.module', Path('fixture'), aggregate_sample_guard=guard, max_rss_bytes=987)
    assert observed == [(child, {'rss_bytes': 123, 'process_tree': []})]
    assert child.initial_args == ('worker.module', Path('fixture'))
    assert child.initial_kwargs == {'max_rss_bytes': 987}
    assert child._failure is None


@pytest.mark.parametrize('cls', [c.BaselineStudyWorker, c.CandidateStudyWorker])
def test_constructor_resource_error_enters_trusted_failure_receipt_path(stub_workers, cls):
    def guard(child, telemetry):
        raise ValueError('aggregate RSS fixture')
    child = cls(aggregate_sample_guard=guard)
    assert child.initial_sample == {'rss_bytes': 123, 'process_tree': []}
    assert child._failure == 'resource_guard'
    assert 'aggregate RSS fixture' in child._failure_detail
    assert child.request.__func__ is c.original.StudyWorker.request
    assert child._receive.__func__ is c.original.StudyWorker._receive


@pytest.mark.parametrize('cls', [c.BaselineStudyWorker, c.CandidateStudyWorker])
def test_resource_failure_does_not_overwrite_specific_existing_fault(stub_workers, cls):
    child = cls()
    child._failure, child._failure_detail = 'affinity_changed', 'original detailed fault'
    def guard(child, telemetry):
        raise RuntimeError('secondary aggregate failure')
    child.aggregate_sample_guard = guard
    child._sample()
    assert (child._failure, child._failure_detail) == ('affinity_changed', 'original detailed fault')


def test_both_modes_call_same_aggregate_guard_once_per_sample(stub_workers):
    observed = []
    def guard(child, telemetry):
        observed.append((type(child).__name__, telemetry['rss_bytes']))
    a, b = c.BaselineStudyWorker(aggregate_sample_guard=guard), c.CandidateStudyWorker(aggregate_sample_guard=guard)
    a._sample()
    b._sample()
    assert observed == [('BaselineStudyWorker', 123), ('CandidateStudyWorker', 123),
                        ('BaselineStudyWorker', 123), ('CandidateStudyWorker', 123)]


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'outputs/new_attempt')
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    return tmp_path


def test_output_boundary_prevents_writing_previous_or_external_evidence(workspace):
    for path in (workspace/'outputs/previous/result.json', workspace/'external.json', c.OUT/'../previous/result.json'):
        with pytest.raises(ValueError, match='boundary|outside|Outside'):
            c.save(path, {'value': 1})
        assert not path.exists()


def test_outputs_are_exclusive_and_write_observer_receives_actual_bytes(workspace, monkeypatch):
    observed = []
    monkeypatch.setattr(c, 'WRITE_OBSERVER', lambda path, size: observed.append((path, size)))
    path = c.OUT/'nested/result.json'
    receipt = c.save(path, {'value': 1})
    assert c.read(path) == {'value': 1}
    assert receipt == c.artifact(path)
    assert observed == [(path, path.stat().st_size)]
    with pytest.raises(FileExistsError):
        c.save(path, {'value': 2})
    assert c.read(path) == {'value': 1}


def test_hash_verification_rejects_mutated_evidence(workspace):
    path = c.OUT/'record.json'
    receipt = c.save(path, {'value': 1})
    c.verify([receipt])
    path.write_text('{"value":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='Changed frozen'):
        c.verify([receipt])

