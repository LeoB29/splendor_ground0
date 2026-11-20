import json

import pytest

from tools.live_game_operational_recovery_20261020 import admission as a


class Clock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        assert 0 < seconds <= 10
        self.sleeps.append(seconds)
        self.value += seconds


def configure_clock(monkeypatch, sequence):
    clock = Clock()
    values = iter(sequence)
    latest = sequence[-1]

    def read():
        nonlocal latest
        latest = next(values, latest)
        return {"available_physical": latest[0], "available_page": latest[1]}

    monkeypatch.setattr(a.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(a.time, "sleep", clock.sleep)
    monkeypatch.setattr(a, "memory", read)
    monkeypatch.setattr(a, "emit", lambda row: None)
    calls = []
    monkeypatch.setattr(a, "guard", lambda base, deadline: calls.append(clock.value))
    return clock, calls


GOOD = (a.PHYSICAL_BYTES, a.COMMIT_BYTES)


def test_exact_threshold_requires_thirty_seconds(tmp_path, monkeypatch):
    clock, calls = configure_clock(monkeypatch, [GOOD])
    result = a.wait_for_admission(tmp_path, 10000.0)
    assert result["status"] == "passed"
    assert result["elapsed_seconds"] == 30
    assert len(result["readings"]) == len(calls) == 4
    assert clock.sleeps == [10, 10, 10]
    assert json.loads((tmp_path / "admission.json").read_text()) == result


@pytest.mark.parametrize("bad", [(a.PHYSICAL_BYTES - 1, a.COMMIT_BYTES),
                                (a.PHYSICAL_BYTES, a.COMMIT_BYTES - 1)])
def test_bad_reading_resets_stability(tmp_path, monkeypatch, bad):
    configure_clock(monkeypatch, [GOOD, GOOD, bad, GOOD, GOOD, GOOD, GOOD])
    result = a.wait_for_admission(tmp_path, 10000.0)
    assert result["status"] == "passed"
    assert result["elapsed_seconds"] == 60
    assert [r["consecutive_admissible_samples"] for r in result["readings"]] == [1, 2, 0, 1, 2, 3, 4]


def test_ten_minute_limit_preserves_readings(tmp_path, monkeypatch):
    clock, calls = configure_clock(monkeypatch, [(0, a.COMMIT_BYTES)])
    with pytest.raises(a.ResourceHold, match="wait exhausted"):
        a.wait_for_admission(tmp_path, 10000.0)
    result = json.loads((tmp_path / "admission.json").read_text())
    assert result["status"] == "preserved_hold"
    assert result["elapsed_seconds"] == 600
    assert len(result["readings"]) == len(calls) == 60
    assert sum(clock.sleeps) == 600


def test_recovery_deadline_bounds_wait(tmp_path, monkeypatch):
    clock, _ = configure_clock(monkeypatch, [(0, a.COMMIT_BYTES)])
    with pytest.raises(a.ResourceHold):
        a.wait_for_admission(tmp_path, 1025.0)
    assert clock.value == 1025
    assert json.loads((tmp_path / "admission.json").read_text())["elapsed_seconds"] == 25


def test_other_resource_failure_is_not_retried(tmp_path, monkeypatch):
    clock, _ = configure_clock(monkeypatch, [GOOD])

    def fail(base, deadline):
        raise a.ResourceHold("disk reserve")

    monkeypatch.setattr(a, "guard", fail)
    with pytest.raises(a.ResourceHold, match="disk reserve"):
        a.wait_for_admission(tmp_path, 10000.0)
    result = json.loads((tmp_path / "admission.json").read_text())
    assert result["status"] == "preserved_hold"
    assert not result["readings"] and not clock.sleeps


def test_no_reentry_or_overwrite(tmp_path, monkeypatch):
    configure_clock(monkeypatch, [GOOD])
    a.wait_for_admission(tmp_path, 10000.0)
    original = (tmp_path / "admission.json").read_bytes()
    with pytest.raises(a.ResourceHold, match="already attempted"):
        a.wait_for_admission(tmp_path, 10000.0)
    assert (tmp_path / "admission.json").read_bytes() == original


@pytest.mark.parametrize("bad", [(-1, a.COMMIT_BYTES), (True, a.COMMIT_BYTES),
                                (a.PHYSICAL_BYTES, 1.2)])
def test_invalid_readings_preserve_failure(tmp_path, monkeypatch, bad):
    configure_clock(monkeypatch, [bad])
    with pytest.raises(a.ResourceHold, match="Invalid"):
        a.wait_for_admission(tmp_path, 10000.0)
    assert json.loads((tmp_path / "admission.json").read_text())["failure_category"] == "ResourceHold"


@pytest.mark.parametrize("remaining,allowed", [(21779.99, False), (21780, True)])
def test_guard_respects_smaller_wall_clock_reserve(tmp_path, monkeypatch, remaining, allowed):
    from tools.live_game_operational_recovery_20261020 import common as c

    def boundary(base, deadline, *, memory_headroom):
        assert base == tmp_path and deadline == 10**12 and memory_headroom is False
        return {"remaining_seconds": remaining}

    monkeypatch.setattr(c, "boundary", boundary)
    if allowed:
        a.guard(tmp_path, 10**12)
    else:
        with pytest.raises(a.ResourceHold, match="work reserve"):
            a.guard(tmp_path, 10**12)
