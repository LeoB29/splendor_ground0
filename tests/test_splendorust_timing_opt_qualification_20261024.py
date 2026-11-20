"""Model-free gate and schedule tests for the single sampler candidate."""
from copy import deepcopy
import pytest

from tools.splendorust_timing_opt_20261024 import qualification as q


def test_large_baseline_margin_exceedance_does_not_disqualify_fast_candidate():
    result = q.timing_gate([.8, .9], [.5, .6])
    assert result["qualified"] and result["baseline_margin_exceedances_preserved"] == 2


@pytest.mark.parametrize("baseline,candidate,check", [
    ([.5]*20, [.46]*20, "mean_reduction_at_least_ten_percent"),
    ([.8]*20, [.71]*20, "all_candidate_within_margin"),
    ([.5]*20, [.1]*18+[.6]*2, "candidate_p95_no_worse"),
])
def test_each_gate_is_independently_enforced(baseline, candidate, check):
    result = q.timing_gate(baseline, candidate)
    assert not result["qualified"] and not result["checks"][check]


def test_exact_ten_percent_and_equal_p95_boundary_passes():
    assert q.timing_gate([.5]*20, [.45]*20)["qualified"]
    result = q.timing_gate([.5]*20, [.1]*18+[.5]*2)
    assert result["qualified"] and result["checks"]["candidate_p95_no_worse"]


@pytest.mark.parametrize("baseline,candidate", [([1.01],[.4]),([.5],[1.01]),([0],[0]),([.5],[float("nan")]),([.5],[True])])
def test_deadline_and_invalid_timing_stop_instead_of_becoming_gate_failure(baseline, candidate):
    with pytest.raises(ValueError):
        q.timing_gate(baseline, candidate)


@pytest.mark.parametrize("after", [None, [], "missing"])
def test_completed_request_must_have_snapshot_for_independent_transition(after):
    row = {"completed": True, "clock": {"status": "ok"}, "after": after}
    with pytest.raises(ValueError, match="completed next snapshot required"):
        q._completed(row, {}, quota=32)


def miniature_plan():
    root = {"fixture_id": "dev", "snapshot": {"root": "dev"},
            "historical32": {"action": {"move": 1}, "trace": {"callbacks": ["exact"]}},
            "preserved_variants": [{"snapshot": {"root": "dev_hidden"}}]}
    holdout = {"fixture_id": "holdout", "snapshot": {"root": "held"}}
    panels = {"development": [root], "holdout": [holdout]}
    return {"panels": panels, "stage32_schedule": q._schedule(panels, quota=32, modes=q.MODES),
            "stage64_holdout_schedule": q._schedule({"holdout": [holdout]}, quota=64, modes=("candidate",))}


def synthetic_rows(plan, schedule_name="stage32_schedule"):
    return [{**spec, "time": .5 if spec["mode"] == "baseline" else .4,
             "science": {"action": {"move": 1}, "trace": {"callbacks": ["exact"]}}}
            for spec in plan[schedule_name]]


def fake_completed(row, snapshot, **kwargs):
    return row["time"], row["science"]


def test_schedule_has_exact_variant_both_modes_repeats_and_independent_panels():
    plan = miniature_plan()
    specs = plan["stage32_schedule"]
    assert len(specs) == 12
    assert [row["index"] for row in specs] == list(range(12))
    assert len([row for row in specs if row["variant"] == "preserved/0"]) == 4
    assert q.snapshot_for_request(plan, next(row for row in specs if row["variant"] == "preserved/0")) == {"root": "dev_hidden"}


def test_both_panels_must_pass_and_failed_holdout_cannot_be_pooled_away(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = miniature_plan()
    rows = synthetic_rows(plan)
    assert q.validate_32(rows, plan)["qualified"]
    for row in rows:
        if row["panel"] == "holdout" and row["mode"] == "candidate":
            row["time"] = .46
    result = q.validate_32(rows, plan)
    assert result["panels"]["development"]["qualified"]
    assert not result["panels"]["holdout"]["qualified"]
    assert not result["permits64_qualification"]


def test_hidden_variant_or_callback_change_fails_exact_reproduction(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = miniature_plan()
    rows = synthetic_rows(plan)
    changed = next(row for row in rows if row["variant"] == "preserved/0")
    changed["science"]["trace"]["callbacks"] = ["different"]
    with pytest.raises(ValueError, match="scientific output|historical32"):
        q.validate_32(rows, plan)


@pytest.mark.parametrize("kind", ["missing", "reordered", "duplicate", "mode"])
def test_schedule_corruption_is_not_accepted(monkeypatch, kind):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = miniature_plan()
    rows = synthetic_rows(plan)
    if kind == "missing": rows.pop()
    elif kind == "reordered": rows[0], rows[1] = rows[1], rows[0]
    elif kind == "duplicate": rows[1] = deepcopy(rows[0])
    else: rows[0]["mode"] = "unapproved"
    with pytest.raises(ValueError):
        q.validate_32(rows, plan)


def test_64_holdout_requires_exact_repeat_and_margin(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = miniature_plan()
    rows = synthetic_rows(plan, "stage64_holdout_schedule")
    assert q.validate_64_holdout(rows, plan)["qualified"]
    rows[1]["time"] = .701
    assert not q.validate_64_holdout(rows, plan)["qualified"]
    rows[1]["science"]["trace"]["callbacks"] = ["changed"]
    with pytest.raises(ValueError, match="repeated scientific output"):
        q.validate_64_holdout(rows, plan)
