"""Model-free failure-preservation checks for combined resident-worker guards."""
import pytest

from tools.splendorust_timing_20261024 import common


def instance(profile_enabled, guard, *, failure=None, detail=None):
    child = object.__new__(common.StudyWorker)
    child.profile_enabled = profile_enabled
    child.aggregate_sample_guard = guard
    child._parent_spans = {}
    child._failure, child._failure_detail = failure, detail
    return child


@pytest.mark.parametrize("profile_enabled", [False, True])
def test_aggregate_violation_sets_failure_without_escaping_receipt_path(monkeypatch, profile_enabled):
    telemetry = {"rss_bytes": 100, "cpu_seconds": .2, "process_tree": []}
    calls = []
    def guard(child, observed):
        calls.append((child, observed))
        raise ValueError("Combined resident workers/helpers RSS exceeded 2GiB")
    monkeypatch.setattr(common.original.StudyWorker, "_sample", lambda child: telemetry)
    child = instance(profile_enabled, guard)
    observed = child._sample()
    assert observed is telemetry
    assert calls == [(child, telemetry)]
    assert child._failure == "resource_guard"
    assert "Combined resident" in child._failure_detail
    if profile_enabled:
        assert child._parent_spans["process_tree_sample"]["calls"] == 1


@pytest.mark.parametrize("profile_enabled", [False, True])
def test_aggregate_violation_keeps_original_failure_primary(monkeypatch, profile_enabled):
    def guard(child, observed):
        raise ValueError("secondary combined-memory failure")
    monkeypatch.setattr(common.original.StudyWorker, "_sample", lambda child: {"rss_bytes": 100})
    child = instance(profile_enabled, guard, failure="process_tree_limit", detail="original source identity failure")
    assert child._sample() == {"rss_bytes": 100}
    assert child._failure == "process_tree_limit"
    assert child._failure_detail == "original source identity failure"


@pytest.mark.parametrize("profile_enabled", [False, True])
def test_successful_aggregate_check_preserves_exact_telemetry(monkeypatch, profile_enabled):
    telemetry = {"rss_bytes": 100, "process_tree": [{"pid": 123}]}
    seen = []
    monkeypatch.setattr(common.original.StudyWorker, "_sample", lambda child: telemetry)
    child = instance(profile_enabled, lambda child, observed: seen.append(observed))
    assert child._sample() is telemetry
    assert seen == [telemetry]
    assert child._failure is None and child._failure_detail is None
