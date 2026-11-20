"""Model-free selection, role preservation and science/deadline regressions."""
from copy import deepcopy
import gzip
import json

import pytest

from tools.splendorust_boundary_timing_20261025 import qualification as q


def fixtures():
    result = []
    for i in range(104):
        quota = 64 if i < 68 else 32
        result.append({"fixture_id": f"saved/{i}.json", "panel_id": f"g{i}_p0", "quota": quota,
                       "player_role": q.ROLES[quota], "backend": q.BACKENDS[quota],
                       "historical_completed": i not in (0, 1), "case": i < 20 or i == 68,
                       "snapshot": {"original": i}, "historical_records": []})
    return result


def plan_and_rows(stage="overhead"):
    roots = fixtures()
    plan = {"panels": {"saved": roots}, "overhead_indices": q.choose_overhead(roots), "source_artifacts": []}
    schedule = q.make_schedule(roots, stage=stage, overhead_indices=plan["overhead_indices"])
    plan[stage+"_schedule"] = schedule
    rows = []
    for spec in schedule:
        science = {"action": {"root": spec["fixture_index"]}, "trace": [spec["fixture_index"]],
                   "after": {"next": spec["fixture_index"]}, "result": {"fixed": spec["quota"]}}
        roots[spec["fixture_index"]]["historical_records"] = [
            {"kind": "completed_accepted", "quota": spec["quota"], "science": science}]
        rows.append({**spec, "completed": True, "fake_science": deepcopy(science),
                     "clock": {"receipt_elapsed_seconds": .3 if spec["mode"] == "baseline" else .31}})
    return plan, rows


def mocked_science(monkeypatch):
    from tools.splendorust_boundary_timing_20261025 import analyze

    def completed(row, snapshot, *, quota):
        if not row["completed"]:
            raise ValueError("failed request")
        elapsed = row["clock"]["receipt_elapsed_seconds"]
        if elapsed > 1:
            raise ValueError("hard clock")
        return elapsed, row["fake_science"]
    monkeypatch.setattr(q, "_scientific_completed", completed)
    monkeypatch.setattr(analyze, "validate_profile_row", lambda row: None)


def test_overhead_ranking_excludes_preknown_cases_but_not_new_timing():
    roots = fixtures()
    indices = q.choose_overhead(roots)
    assert len(indices) == len(set(indices)) == 12
    assert all(roots[i]["historical_completed"] and not roots[i]["case"] for i in indices)
    assert [roots[i]["quota"] for i in indices] == [64]*6 + [32]*6
    for root in roots:
        root["new_timing"] = 999
        root["game_winner"] = "ignored"
    assert q.choose_overhead(roots) == indices


@pytest.mark.parametrize("stage,count,groups", [("overhead", 48, 12), ("diagnostic", 416, 104)])
def test_original_roles_exact_repetitions_and_reverse_order(stage, count, groups):
    roots = fixtures()
    schedule = q.make_schedule(roots, stage=stage)
    assert len(schedule) == count
    assert len({r["fixture_id"] for r in schedule}) == groups
    for quota in q.QUOTAS:
        rows = [r for r in schedule if r["quota"] == quota]
        first = [r for r in rows if r["pass_index"] == 0]
        second = [r for r in rows if r["pass_index"] == 1]
        assert [(r["fixture_id"], r["mode"]) for r in second] == [
            (r["fixture_id"], r["mode"]) for r in reversed(first)]
        assert all(r["player_role"] == q.ROLES[quota] for r in rows)
    assert all(r["quota"] == 64 for r in schedule[:count//2] if stage == "overhead")


def test_graph32_and_quota_relabelling_rejected():
    roots = fixtures()
    roots[70]["backend"] = "graph64"
    with pytest.raises(ValueError, match="identity"):
        q.make_schedule(roots, stage="diagnostic")
    roots = fixtures()
    roots[0]["quota"] = 32
    with pytest.raises(ValueError, match="inventory"):
        q.make_schedule(roots, stage="diagnostic")


def test_snapshot_is_copied_and_bound_to_role_quota():
    plan, rows = plan_and_rows()
    spec = rows[0]
    snapshot = q.snapshot_for_request(plan, spec)
    snapshot["original"] = "changed"
    assert q.snapshot_for_request(plan, spec)["original"] != "changed"
    with pytest.raises(ValueError, match="identity"):
        q.snapshot_for_request(plan, {**spec, "quota": 32})


def test_complete_panel_enforces_full_result_and_transition(monkeypatch):
    mocked_science(monkeypatch)
    plan, rows = plan_and_rows()
    result = q.validate_panel(rows, plan)
    assert result["qualified"] and result["historical_comparisons"] == {
        "completed_accepted": 48, "late_unaccepted_science_only": 0}
    assert result["exact_scientific_groups"] == 12
    for field in ("action", "trace", "after", "result"):
        changed = deepcopy(rows)
        changed[1]["fake_science"][field] = "mutation"
        with pytest.raises(ValueError, match="scientific output|historical"):
            q.validate_panel(changed, plan)


def test_late_history_comparison_never_invents_historical_transition(monkeypatch):
    mocked_science(monkeypatch)
    plan, rows = plan_and_rows()
    index = rows[0]["fixture_index"]
    item = plan["panels"]["saved"][index]["historical_records"][0]
    item["kind"] = "late_unaccepted_science_only"
    item["science"].pop("after")
    result = q.validate_panel(rows, plan)
    assert result["historical_comparisons"]["late_unaccepted_science_only"] == 4
    assert result["historical_failures_reclassified"] is False
    assert result["permits_fresh_games"] is False


def test_failure_or_truncated_schedule_cannot_pass(monkeypatch):
    mocked_science(monkeypatch)
    plan, rows = plan_and_rows()
    with pytest.raises(ValueError, match="complete unchanged"):
        q.validate_panel(rows[:-1], plan)
    rows[0]["completed"] = False
    with pytest.raises(ValueError, match="failed request"):
        q.validate_panel(rows, plan)


def test_overhead_gate_all_three_requirements_and_no_margin_gate():
    assert q.overhead_gate([.75]*12, [.76]*12)["qualified"]
    assert not q.overhead_gate([.1]*12, [.12]*12)["qualified"]
    assert not q.overhead_gate([.4]*12, [.43]*12)["qualified"]
    base = [.3]*12
    candidate = [.3]*11 + [.5]
    result = q.overhead_gate(base, candidate)
    assert result["checks"]["mean_ratio_at_most_1_10"]
    assert result["checks"]["added_mean_at_most_025s"]
    assert not result["checks"]["added_p95_at_most_050s"]
    with pytest.raises(ValueError, match="hard clock"):
        q.overhead_gate([.3], [1.001])


def test_selected_source_change_rejected(tmp_path, monkeypatch):
    path = tmp_path/"evidence.json"
    path.write_text('{}')
    monkeypatch.setattr(q, "ROOT", tmp_path)
    item = {"path": "evidence.json", "bytes": 2, "sha256": "original"}
    monkeypatch.setattr(q, "_artifact", lambda path: {**item, "sha256": "modified"})
    with pytest.raises(ValueError, match="source changed"):
        q.verify_sources({"source_artifacts": [item]})


def test_late_packet_is_bound_unique_and_never_accepted(tmp_path, monkeypatch):
    folder = tmp_path/"worker"
    folder.mkdir()
    path = folder/"stdout.log.gz"
    result = {"action": {"move": 1}, "diagnostics": {"calls": 65}}
    packet = {"type": "response", "request_id": "request-16", "result": result}
    path.write_bytes(gzip.compress((json.dumps(packet)+'\n').encode()))
    item = {"path": "worker/stdout.log.gz", "bytes": path.stat().st_size, "sha256": "bound"}
    monkeypatch.setattr(q, "ROOT", tmp_path)
    monkeypatch.setattr(q, "_artifact", lambda path: item)
    monkeypatch.setattr(q.audit, "validate_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(q.audit, "audit_decision", lambda *args, **kwargs: None)
    row = {"game_index": 66, "ply": 29, "quota": 64, "completed": False, "before": {}, "observer": {},
           "clock": {"status": "timeout", "request_id": "request-16"}, "worker_clock_file": "worker/request-16.json"}
    history = q._late_history(row, {}, {item["path"]: item})
    assert history["kind"] == "late_unaccepted_science_only"
    assert history["historical_clock_accepted"] is False
    assert history["historical_transition_available"] is False
    assert "after" not in history["science"]
    path.write_bytes(gzip.compress(((json.dumps(packet)+'\n')*2).encode()))
    with pytest.raises(ValueError, match="unique"):
        q._late_history(row, {}, {item["path"]: item})


def test_lighter_observer_cannot_reselect_or_relax_frozen_design():
    plan, _ = plan_and_rows()
    plan.update(overhead_domain=q.OVERHEAD_DOMAIN, diagnostic_schedule=q.make_schedule(fixtures(), stage="diagnostic"),
        overhead_gate={"maximum_added_mean_seconds": .025}, deadline_seconds=1., internal_seconds=.9,
        diagnostic_margin_seconds=.7, margin_is_admission_gate=False, no_fresh_games=True)
    previous = deepcopy(plan)
    q.verify_frozen_design(plan, previous)
    for key in ("panels", "overhead_indices", "overhead_schedule", "diagnostic_schedule", "overhead_gate",
                "deadline_seconds", "internal_seconds"):
        changed = deepcopy(plan); changed[key] = "changed"
        with pytest.raises(ValueError, match="previous frozen design changed"):
            q.verify_frozen_design(changed, previous)
