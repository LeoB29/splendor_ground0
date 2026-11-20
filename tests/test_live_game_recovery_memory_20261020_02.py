import hashlib
import json
from pathlib import Path

import pytest

from tools.live_game_operational_recovery_20261020_02 import memory as m

GOOD = (m.PHYSICAL_BYTES, m.COMMIT_BYTES)
LOW = (m.PHYSICAL_BYTES - 1, m.COMMIT_BYTES)


class Clock:
    def __init__(self):
        self.mono, self.epoch = 1000., 100000.
        self.sleeps = []

    def sleep(self, seconds):
        assert 0 < seconds <= 10
        self.sleeps.append(seconds)
        self.mono += seconds
        self.epoch += seconds


def configure(monkeypatch, values):
    clock = Clock()
    source = iter(values)
    latest = values[-1]

    def reading():
        nonlocal latest
        latest = next(source, latest)
        return {"available_physical": latest[0], "available_page": latest[1]}

    monkeypatch.setattr(m.time, "monotonic", lambda: clock.mono)
    monkeypatch.setattr(m.time, "time", lambda: clock.epoch)
    monkeypatch.setattr(m.time, "sleep", clock.sleep)
    monkeypatch.setattr(m, "memory", reading)
    monkeypatch.setattr(m, "emit", lambda row: None)
    monkeypatch.setattr(m, "guard", lambda base, deadline, reserve, request_seconds:
                        {"remaining_seconds": deadline-clock.mono})
    return clock


def initial(base, deadline=50000.):
    return m.wait_for_memory(base, deadline, name="initial", reserve=21600,
                             request_seconds=120, initial=True)


def move(base, name="games/g021/t0011", deadline=50000.):
    return m.wait_for_memory(base, deadline, name=name, reserve=21600, request_seconds=120)


def receipt(base, name):
    return json.loads((base/"memory_admissions"/(name+".json")).read_text())


def test_initial_requires_full_two_minute_stability(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [GOOD])
    result = initial(tmp_path)
    assert result["status"] == "passed" and result["initial"] and result["stability_required"]
    assert result["elapsed_seconds"] == result["charged_seconds"] == 120
    assert len(result["readings"]) == 13 and clock.sleeps == [10]*12
    assert result == receipt(tmp_path, "initial")
    data = (tmp_path/result["readings_log"]["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == result["readings_log"]["sha256"]
    assert len(data) == result["readings_log"]["bytes"]
    assert [json.loads(x) for x in data.splitlines()] == result["readings"]


def test_healthy_between_moves_is_immediate_and_charged(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [GOOD])
    initial(tmp_path)
    result = move(tmp_path)
    assert result["status"] == "passed" and not result["stability_required"]
    assert len(result["readings"]) == 1 and result["elapsed_seconds"] == 0
    assert result["prior_admission_count"] == 1
    assert result["cumulative_before_seconds"] == result["cumulative_after_seconds"] == 120
    assert clock.sleeps == [10]*12


@pytest.mark.parametrize("bad", [LOW, (m.PHYSICAL_BYTES, m.COMMIT_BYTES-1)])
def test_low_ram_waits_then_resumes_only_after_full_window(tmp_path, monkeypatch, bad):
    configure(monkeypatch, [GOOD]*13+[bad]+[GOOD]*5+[bad]+[GOOD]*13)
    initial(tmp_path)
    result = move(tmp_path)
    assert result["status"] == "passed" and result["stability_required"]
    assert result["elapsed_seconds"] == 190
    assert result["cumulative_after_seconds"] == 310
    assert [r["consecutive_admissible_samples"] for r in result["readings"]][:7] == [0, 1, 2, 3, 4, 5, 0]


def test_episode_limit_keeps_failed_receipt_and_does_not_retry(tmp_path, monkeypatch):
    configure(monkeypatch, [LOW])
    monkeypatch.setattr(m, "MAXIMUM_WAIT_SECONDS", 25)
    with pytest.raises(m.ResourceHold, match="allowance exhausted"):
        initial(tmp_path)
    result = receipt(tmp_path, "initial")
    assert result["status"] == "preserved_hold" and result["elapsed_seconds"] == 25
    original = (tmp_path/"memory_admissions/initial.json").read_bytes()
    with pytest.raises(m.ResourceHold, match="already attempted"):
        initial(tmp_path)
    assert (tmp_path/"memory_admissions/initial.json").read_bytes() == original


def test_cumulative_budget_includes_initial_wait(tmp_path, monkeypatch):
    configure(monkeypatch, [GOOD]*13+[LOW])
    initial(tmp_path)
    monkeypatch.setattr(m, "CUMULATIVE_WAIT_SECONDS", 145)
    with pytest.raises(m.ResourceHold, match="allowance exhausted"):
        move(tmp_path)
    result = receipt(tmp_path, "games/g021/t0011")
    assert result["elapsed_seconds"] == 25 and result["cumulative_after_seconds"] == 145


def test_failed_prior_gate_precludes_another_worker(tmp_path, monkeypatch):
    configure(monkeypatch, [LOW])
    monkeypatch.setattr(m, "MAXIMUM_WAIT_SECONDS", 10)
    with pytest.raises(m.ResourceHold):
        initial(tmp_path)
    with pytest.raises(m.ResourceHold, match="Prior memory"):
        move(tmp_path)
    assert not (tmp_path/"memory_admissions/games/g021/t0011.json").exists()


def test_global_reserve_bounds_wait_and_cannot_be_refunded(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [LOW])
    with pytest.raises(m.ResourceHold, match="global work reserve"):
        initial(tmp_path, deadline=1000+21600+60+120+25)
    assert clock.mono == 1025
    assert receipt(tmp_path, "initial")["elapsed_seconds"] == 25


def test_guard_work_is_not_added_to_available_wait(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [LOW])
    def guard(base, deadline, reserve, request_seconds):
        remaining = deadline-clock.mono
        clock.mono += 3
        clock.epoch += 3
        return {"remaining_seconds": remaining}
    monkeypatch.setattr(m, "guard", guard)
    with pytest.raises(m.ResourceHold):
        initial(tmp_path, deadline=1000+21600+60+120+15)
    # No sleep extends beyond the protected deadline; a later guard may diagnose it.
    assert sum(clock.sleeps) == 10


def test_forward_wall_clock_consumes_wait_budget(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [LOW])
    def guard(*args):
        clock.epoch += 7200
        return {"remaining_seconds": 30000}
    monkeypatch.setattr(m, "guard", guard)
    with pytest.raises(m.ResourceHold, match="allowance exhausted"):
        initial(tmp_path)
    result = receipt(tmp_path, "initial")
    assert result["charged_seconds"] == 7200 and result["elapsed_seconds"] == 0
    assert not clock.sleeps


def test_backward_wall_clock_does_not_refund_elapsed_wait(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [GOOD])
    def sleep(seconds):
        clock.mono += seconds
        clock.epoch -= seconds
    monkeypatch.setattr(m.time, "sleep", sleep)
    result = initial(tmp_path)
    assert result["charged_seconds"] == result["elapsed_seconds"] == 120


@pytest.mark.parametrize("bad", [(-1, m.COMMIT_BYTES), (True, m.COMMIT_BYTES),
                                (m.PHYSICAL_BYTES, 1.5)])
def test_invalid_reading_fails_without_wait(tmp_path, monkeypatch, bad):
    clock = configure(monkeypatch, [bad])
    with pytest.raises(m.ResourceHold, match="Invalid memory"):
        initial(tmp_path)
    assert not clock.sleeps and not receipt(tmp_path, "initial")["readings"]


def test_disk_failure_is_never_treated_as_ram_wait(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [GOOD])
    def fail(*args):
        raise m.ResourceHold("disk reserve")
    monkeypatch.setattr(m, "guard", fail)
    with pytest.raises(m.ResourceHold, match="disk reserve"):
        initial(tmp_path)
    assert not clock.sleeps and receipt(tmp_path, "initial")["failure_detail"] == "disk reserve"


@pytest.mark.parametrize("name", ["../initial", "/initial", "games/g21/t0011",
                                "games/g021/../../../escape", "initial.json"])
def test_names_cannot_escape_or_alias_receipts(tmp_path, monkeypatch, name):
    configure(monkeypatch, [GOOD])
    with pytest.raises(m.ResourceHold, match="Invalid memory admission name"):
        m.wait_for_memory(tmp_path, 50000, name=name, reserve=600, request_seconds=120)
    assert not list(tmp_path.rglob("*"))


def test_initial_gate_cannot_be_skipped(tmp_path, monkeypatch):
    configure(monkeypatch, [GOOD])
    with pytest.raises(m.ResourceHold, match="Initial memory admission"):
        move(tmp_path)


def test_analysis_and_audit_have_distinct_receipts(tmp_path, monkeypatch):
    configure(monkeypatch, [GOOD])
    initial(tmp_path)
    a = m.wait_for_memory(tmp_path, 50000, name="audits/games/g021/t0010", reserve=600, request_seconds=600)
    b = m.wait_for_memory(tmp_path, 50000, name="analysis", reserve=600, request_seconds=0)
    assert a["prior_admission_count"] == 1 and b["prior_admission_count"] == 2
    assert a["request_seconds"] == 600 and b["request_seconds"] == 0


def test_slow_final_log_flush_cannot_pass_over_wait_cap(tmp_path, monkeypatch):
    clock = configure(monkeypatch, [GOOD])
    actual_fsync = m.os.fsync
    calls = 0
    def flush(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            clock.mono += m.MAXIMUM_WAIT_SECONDS
            clock.epoch += m.MAXIMUM_WAIT_SECONDS
        actual_fsync(fd)
    monkeypatch.setattr(m.os, "fsync", flush)
    with pytest.raises(m.ResourceHold, match="final admission check"):
        initial(tmp_path)
    result = receipt(tmp_path, "initial")
    assert result["status"] == "preserved_hold" and result["charged_seconds"] == 7320


def test_final_log_flush_failure_cannot_record_passed(tmp_path, monkeypatch):
    configure(monkeypatch, [GOOD])
    actual_fsync = m.os.fsync
    calls = 0
    def flush(fd):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic failed log flush")
        actual_fsync(fd)
    monkeypatch.setattr(m.os, "fsync", flush)
    with pytest.raises(OSError, match="failed log flush"):
        initial(tmp_path)
    result = receipt(tmp_path, "initial")
    assert result["status"] == "preserved_hold" and result["failure_category"] == "OSError"


@pytest.mark.parametrize("request_duration", [0, 120, 600])
@pytest.mark.parametrize("margin", [-.01, 0])
def test_guard_uses_returned_dual_clock_and_full_work_reserve(tmp_path, monkeypatch, request_duration, margin):
    from tools.live_game_operational_recovery_20261020_02 import common as c
    def boundary(base, deadline, *, memory_headroom):
        assert base == tmp_path and deadline == 10**12 and memory_headroom is False
        return {"remaining_seconds": 600+60+request_duration+margin}
    monkeypatch.setattr(c, "boundary", boundary)
    if margin < 0:
        with pytest.raises(m.ResourceHold, match="global work reserve"):
            m.guard(tmp_path, 10**12, 600, request_duration)
    else:
        assert m.guard(tmp_path, 10**12, 600, request_duration)["remaining_seconds"] == 600+60+request_duration
