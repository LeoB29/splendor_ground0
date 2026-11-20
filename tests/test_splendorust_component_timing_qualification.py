"""Model-free predeclared schedule, exact-output and instrumentation gates."""
from copy import deepcopy
import pytest

from tools.splendorust_component_timing_20261025 import qualification as q


def science(index, quota):
    return {"action": {"index": index}, "trace": {"quota": quota, "callbacks": ["exact"]}, "after": {"index": index}}


def plan_fixture():
    fixtures = [{"fixture_id": f"development/{i}", "snapshot": {"index": i},
                 "historical_records": [{"quota": 64, "science": science(i, 64)}]} for i in range(24)]
    return {"panels": {"development": fixtures}, "source_artifacts": [],
            "overhead_schedule": q.make_schedule(fixtures, stage="overhead"),
            "diagnostic_schedule": q.make_schedule(fixtures, stage="diagnostic")}


def synthetic_rows(plan, stage):
    return [{**spec, "time": .5 if spec["mode"] == "baseline" else .51,
             "science": science(spec["fixture_index"], spec["quota"])} for spec in plan[stage+"_schedule"]]


def fake_completed(row, snapshot, *, quota):
    assert snapshot == {"index": row["fixture_index"]} and quota == row["quota"]
    return row["time"], row["science"]


def test_prespecified_overhead_roots_first_and_reversed_passes():
    plan = plan_fixture()
    assert len(plan["overhead_schedule"]) == 48 and len(plan["diagnostic_schedule"]) == 192
    assert [r["fixture_index"] for r in plan["overhead_schedule"][:12:2]] == [0, 1, 2, 23, 15, 3]
    for stage in ("overhead", "diagnostic"):
        rows = plan[stage+"_schedule"]
        assert [r["index"] for r in rows] == list(range(len(rows)))
        assert all(r["quota"] == 64 for r in rows[:len(rows)//2])
        for quota in q.QUOTAS:
            passes = [[(r["fixture_index"], r["mode"]) for r in rows if r["quota"] == quota and r["pass_index"] == p]
                      for p in range(2)]
            assert passes[1] == list(reversed(passes[0]))


@pytest.mark.parametrize("baseline,profiled,failed_check", [
    ([.1]*20, [.115]*20, "mean_ratio_at_most_1_10"),
    ([.5]*20, [.527]*20, "added_mean_at_most_025s"),
    ([.5]*20, [.45]*18+[.56]*2, "added_p95_at_most_050s"),
])
def test_each_overhead_constraint_is_enforced(baseline, profiled, failed_check):
    result = q.overhead_gate(baseline, profiled)
    assert not result["qualified"] and not result["checks"][failed_check]


def test_exact_absolute_boundary_and_soft_margin_are_accepted():
    assert q.overhead_gate([.5]*20, [.525]*20)["qualified"]
    result = q.overhead_gate([.8]*20, [.8]*20)
    assert result["qualified"] and result["profiled"]["above_diagnostic_margin"] == 20
    assert result["margin_exceedances_are_diagnostic_only"]


@pytest.mark.parametrize("baseline,profiled", [([1.01],[.5]),([.5],[1.01]),([0],[0]),([.5],[True]),([.5],[float("nan")]),([.5],[])])
def test_hard_clock_and_invalid_samples_stop(baseline, profiled):
    with pytest.raises(ValueError): q.overhead_gate(baseline, profiled)


def test_both_quota_gates_required_no_pooling(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = plan_fixture(); rows = synthetic_rows(plan, "overhead")
    for row in rows:
        if row["quota"] == 32 and row["mode"] == "profiled": row["time"] = .53
    result = q.validate_panel(rows, plan)
    assert result["per_quota"]["64"]["qualified"] and not result["per_quota"]["32"]["qualified"]
    assert not result["qualified"] and not result["permits_diagnostic"]


def test_exact_science_groups_and_history_both_stages(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = plan_fixture()
    overhead = q.validate_panel(synthetic_rows(plan, "overhead"), plan)
    assert overhead["qualified"] and overhead["permits_diagnostic"]
    assert overhead["exact_scientific_groups"] == 12 and overhead["historical_comparisons"] == {"64": 24, "32": 0}
    rows = synthetic_rows(plan, "diagnostic")
    for row in rows: row["time"] = .9
    diagnostic = q.validate_panel(rows, plan, stage="diagnostic")
    assert diagnostic["qualified"] and diagnostic["exact_scientific_groups"] == 48
    assert not diagnostic["permits_diagnostic"] and not diagnostic["permits_fresh_games"]


@pytest.mark.parametrize("component", ["action", "trace", "after"])
def test_repeat_or_mode_change_stops(monkeypatch, component):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = plan_fixture(); rows = synthetic_rows(plan, "overhead")
    changed = next(r for r in rows if r["quota"] == 32 and r["mode"] == "profiled")
    changed["science"][component] = {"changed": True}
    with pytest.raises(ValueError, match="mode/repeat"):
        q.validate_panel(rows, plan)


def test_consistently_wrong_historical_trace_stops(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = plan_fixture(); rows = synthetic_rows(plan, "overhead")
    for row in rows:
        if row["quota"] == 64 and row["fixture_index"] == 0: row["science"]["trace"] = {"wrong": True}
    with pytest.raises(ValueError, match="historical"):
        q.validate_panel(rows, plan)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "order", "quota", "mode", "holdout"])
def test_schedule_changes_stop(monkeypatch, kind):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = plan_fixture(); rows = synthetic_rows(plan, "overhead")
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[1] = deepcopy(rows[0])
    elif kind == "order": rows[0], rows[1] = rows[1], rows[0]
    elif kind == "quota": rows[0]["quota"] = 32
    elif kind == "mode": rows[0]["mode"] = "candidate"
    else: plan["panels"]["holdout"] = []
    with pytest.raises(ValueError): q.validate_panel(rows, plan)


def test_only_profile_field_is_removed_from_copy_for_scientific_audit(monkeypatch):
    from tools.splendorust_component_timing_20261025 import analyze
    monkeypatch.setattr(analyze, "validate_profile_row", lambda row: None)
    seen = []
    monkeypatch.setattr(q.prior, "_completed", lambda row, snapshot, quota: seen.append(row) or (.5, {}))
    row = {"mode": "profiled", "clock": {"result": {"timing_profile": {"x": 1}, "diagnostics": {"exact": 2}}}}
    q._completed(row, {}, quota=64)
    assert seen[0]["clock"]["result"] == {"diagnostics": {"exact": 2}}
    assert "timing_profile" in row["clock"]["result"]


def test_real_plan_uses_no_reserved_positions_or_fresh_games():
    plan = q.build_plan()
    assert set(plan["panels"]) == {"development"}
    assert plan["reserved_panel_used"] is False and plan["no_fresh_games"] is True
    assert plan["raw_reader_optimized_sampler_both_modes"] is True
    assert len(plan["overhead_schedule"]) == 48 and len(plan["diagnostic_schedule"]) == 192
    assert len(plan["source_artifacts"]) == 91
