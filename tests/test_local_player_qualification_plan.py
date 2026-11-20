import pytest

from tools.local_player_qualification_128.plan import build_cases, summarize


def entries():
    return [{"fit": fit} for fit in (161001, 161002, 161003) for _ in range(6)]


def test_every_prespecified_case_once_and_six_declared_repeats():
    cases = build_cases(entries())
    assert len(cases) == 78
    assert len({(c["root_index"], c["breadth"], c["allowance_seconds"])
                for c in cases if c["kind"] != "repeat"}) == 72
    repeats = [c for c in cases if c["kind"] == "repeat"]
    assert len(repeats) == 6
    assert all(c["hidden_variant"] for c in repeats)
    assert [c["index"] for c in cases] == list(range(78))


def test_short_panel_or_missing_fit_fails():
    with pytest.raises(ValueError):
        build_cases(entries()[:-1])
    with pytest.raises(ValueError):
        build_cases([{"fit": 161001}] * 18)


def test_timeout_is_not_a_completion_time():
    case = {"kind": "core", "breadth": True, "allowance_seconds": 120}
    group = summarize([
        {"case": case, "correct": True,
         "clock": {"status": "ok", "receipt_elapsed_seconds": 8.0}},
        {"case": case, "correct": False,
         "clock": {"status": "timeout", "receipt_elapsed_seconds": None}},
    ])
    row = next(r for r in group if r["kind"] == "core" and r["candidates"] == 8
               and r["allowance_seconds"] == 120)
    assert row["attempted"] == 2 and row["verified_completions"] == 1
    assert row["completion_mean_seconds"] == 8.0
    assert row["statuses"] == {"ok": 1, "timeout": 1}


def test_shared_lease_prevents_competing_execution_and_releases(tmp_path, monkeypatch):
    from tools.local_player_qualification_128 import common
    monkeypatch.setattr(common, "ROOT", tmp_path)
    (tmp_path / "outputs").mkdir()
    with common.execution_lease():
        with pytest.raises(OSError):
            with common.execution_lease():
                pytest.fail("Competing qualification acquired the lease")
    with common.execution_lease():
        pass
