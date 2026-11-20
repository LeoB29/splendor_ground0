"""Evidence boundaries and inherited clock/resource contracts for the new reader."""
from pathlib import Path
import pytest

from tools.splendorust_buffered_reader_20261025 import common as c, launch
from tools.splendorust_timing_opt_20261024 import common as prior
from tools.splendorust_timing_opt_20261024.sampler import BatchedStudyWorker


@pytest.mark.parametrize('klass', [c.BaselineStudyWorker, c.CandidateStudyWorker])
def test_constructor_keeps_aggregate_guard_and_failure_path(monkeypatch, klass):
    def initialize(child, *args, **kwargs):
        child._failure = child._failure_detail = None
        child.sample = child._sample()
    monkeypatch.setattr(BatchedStudyWorker, '__init__', initialize)
    monkeypatch.setattr(BatchedStudyWorker, '_sample', lambda child: {'rss_bytes': 123})
    def fail(child, telemetry):
        assert telemetry['rss_bytes'] == 123
        raise ValueError('combined resident memory')
    child = klass(aggregate_sample_guard=fail)
    assert child._failure == 'resource_guard'
    assert 'combined resident memory' in child._failure_detail
    assert child.request.__func__ is prior.CandidateStudyWorker.request
    assert child._receive.__func__ is prior.CandidateStudyWorker._receive


def test_only_reader_behavior_is_replaced():
    for name in ('__init__', 'request', '_receive', '_sample', '_check_resource_guard', '_log', 'close'):
        assert getattr(c.CandidateStudyWorker, name) is getattr(c.BaselineStudyWorker, name)
    assert c.CandidateStudyWorker._stdout_reader is not c.BaselineStudyWorker._stdout_reader


def test_new_outputs_cannot_overwrite_old_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    monkeypatch.setattr(c, 'OUT', tmp_path/'outputs/new')
    monkeypatch.setattr(c, 'WRITE_OBSERVER', None)
    for forbidden in (tmp_path/'outputs/old/result.json', c.OUT/'../old/result.json'):
        with pytest.raises(ValueError, match='outside'):
            c.save(forbidden, {'changed': True})
        assert not forbidden.exists()
    target = c.OUT/'result.json'
    receipt = c.save(target, {'original': True})
    with pytest.raises(FileExistsError):
        c.save(target, {'replacement': True})
    c.verify([receipt])


@pytest.mark.parametrize('clock', ['monotonic', 'time'])
def test_cleanup_respects_either_original_clock(monkeypatch, clock):
    origins = {'monotonic': 100., 'time': 1000.}
    for name, origin in origins.items():
        now = origin+c.JOB_SECONDS-(.5 if name == clock else 100.)
        monkeypatch.setattr(launch.time, name, lambda value=now: value)
    receipt = launch.cleanup_owned(None, None, (), monotonic_origin=100., unix_origin=1000.)
    assert receipt['effective_allowance_seconds'] == .5
    assert receipt['errors'] == []


def test_launcher_has_no_historical_worker_or_strength_launch():
    # The copied watchdog must dispatch this frozen qualification, not a prior run.
    source = Path(launch.__file__).read_text(encoding='utf-8')
    assert "'tools.splendorust_buffered_reader_20261025.runner'" in source
    assert "'tools.splendorust_timing_opt_20261024.runner'" not in source
    assert 'CREATE_NO_WINDOW' in source
