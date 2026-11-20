"""Recovery-only tests; frozen1272-test evidence remains unchanged."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("recovery157_for_test", "scripts/analysis/recover_fast_selector_transfer_157.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def windows_error(code):
    error = PermissionError("synthetic sharing failure")
    error.winerror = code
    return error


def test_progress_retry_is_bounded_and_scoped(monkeypatch):
    calls, sleeps = [], []
    def writer(path, payload):
        calls.append((path, payload))
        if len(calls) < 3:
            raise windows_error(5)
    monkeypatch.setattr(r, "ORIGINAL_WRITE", writer)
    monkeypatch.setattr(r.time, "sleep", sleeps.append)
    value = {"complete": 20}
    r.progress_write(r.OUTPUT/"progress.json", value)
    assert len(calls) == 3 and sleeps == [.1, .1]
    assert all(payload is value for _, payload in calls)
    assert r.mutable_progress(r.OUTPUT/"fits"/"treatment_157001"/"audit_progress.json")
    assert not r.mutable_progress(Path("outputs/rollout_selector_champion_156/progress.json"))


def test_immutable_error_and_other_errors_not_retried(monkeypatch):
    calls = []
    def writer(path, payload):
        calls.append(path)
        raise windows_error(5)
    monkeypatch.setattr(r, "ORIGINAL_WRITE", writer)
    with pytest.raises(PermissionError):
        r.progress_write(r.OUTPUT/"fits"/"result.json", {})
    assert len(calls) == 1
    def unrelated(path, payload):
        raise windows_error(99)
    monkeypatch.setattr(r, "ORIGINAL_WRITE", unrelated)
    with pytest.raises(PermissionError):
        r.progress_write(r.OUTPUT/"progress.json", {})


def test_retry_exhaustion_stops_and_authorization_required(monkeypatch):
    calls = []
    def writer(path, payload):
        calls.append(path)
        raise windows_error(32)
    monkeypatch.setattr(r, "ORIGINAL_WRITE", writer)
    monkeypatch.setattr(r.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        r.progress_write(r.OUTPUT/"progress.json", {})
    assert len(calls) == 51
    with pytest.raises(AssertionError, match="authorization"):
        r.resume(False)
