"""Model-free failure, selection and split-gate checks for the reader candidate."""
from copy import deepcopy
import json

import pytest

from tools.splendorust_buffered_reader_20261025 import qualification as q


def science(fixture_id, quota):
    return {"action": {"choice": fixture_id}, "trace": {"quota": quota, "callbacks": ["exact"]},
            "after": {"successor": fixture_id}}


def synthetic_plan():
    panels = {}
    for panel, count in (("development", 24), ("holdout", 12)):
        historical_quota = 64 if panel == "development" else 32
        panels[panel] = [{"fixture_id": f"{panel}/{i}", "snapshot": {"root": f"{panel}/{i}"},
                          "historical": {str(historical_quota): science(f"{panel}/{i}", historical_quota)},
                          "historical_records": [{"quota": historical_quota,
                                                   "science": science(f"{panel}/{i}", historical_quota)}]}
                         for i in range(count)]
    return {"panels": panels, "timed_requests": q.make_schedule(panels), "source_artifacts": []}


def synthetic_rows(plan):
    return [{**spec, "time": .5 if spec["mode"] == "baseline" else .45,
             "science": science(spec["fixture_id"], spec["quota"])} for spec in plan["timed_requests"]]


def fake_completed(row, snapshot, *, quota):
    assert snapshot == {"root": row["fixture_id"]}
    assert row["quota"] == quota
    return row["time"], row["science"]


def test_schedule_first_three_roots_two_modes_both_quotas_reversed_passes():
    plan = synthetic_plan()
    rows = plan["timed_requests"]
    assert len(rows) == 288
    assert [r["index"] for r in rows] == list(range(288))
    assert [r["fixture_index"] for r in rows[:6]] == [0, 0, 1, 1, 2, 2]
    assert [r["mode"] for r in rows[:6]] == ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"]
    assert all(r["quota"] == 64 for r in rows[:144])
    assert all(r["quota"] == 32 for r in rows[144:])
    for panel in plan["panels"]:
        for quota in q.QUOTAS:
            passes = [[(r["fixture_id"], r["mode"]) for r in rows
                       if r["panel"] == panel and r["quota"] == quota and r["pass_index"] == p]
                      for p in range(2)]
            assert passes[1] == list(reversed(passes[0]))


def test_snapshot_is_copied_and_unapproved_variant_rejected():
    plan = synthetic_plan()
    spec = plan["timed_requests"][0]
    snapshot = q.snapshot_for_request(plan, spec)
    snapshot["root"] = "changed"
    assert q.snapshot_for_request(plan, spec)["root"] == spec["fixture_id"]
    with pytest.raises(ValueError, match="original states"):
        q.snapshot_for_request(plan, {**spec, "variant": "hidden"})


def test_gate_accepts_exact_five_percent_and_preserves_slow_baseline():
    assert q.timing_gate([.5]*20, [.475]*20)["qualified"]
    result = q.timing_gate([.8, .9], [.6, .7])
    assert result["qualified"] and result["baseline_margin_exceedances_preserved"] == 2


@pytest.mark.parametrize("baseline,candidate,failed_check", [
    ([.5]*20, [.476]*20, "mean_reduction_at_least_five_percent"),
    ([.8]*20, [.701]*20, "all_candidate_within_margin"),
    ([.5]*20, [.1]*18+[.6]*2, "candidate_p95_no_worse"),
])
def test_gate_requirements_independently_fail(baseline, candidate, failed_check):
    result = q.timing_gate(baseline, candidate)
    assert not result["qualified"] and not result["checks"][failed_check]


@pytest.mark.parametrize("baseline,candidate", [
    ([1.00001], [.4]), ([.5], [1.00001]), ([0], [0]), ([.5], [float("nan")]),
    ([.5], [True]), ([.5], [-.1]), ([.5], []),
])
def test_invalid_or_hard_deadline_samples_stop(baseline, candidate):
    with pytest.raises(ValueError):
        q.timing_gate(baseline, candidate)


def test_exact_science_historical_counts_and_four_separate_gates(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan()
    result = q.validate_panel(synthetic_rows(plan), plan)
    assert result["qualified"] and result["permits_fresh_operational_games"]
    assert result["exact_scientific_groups"] == 72
    assert result["historical_comparisons"] == {"64": 96, "32": 48}
    assert all(result["panels"][p][str(quota)]["qualified"]
               for p in plan["panels"] for quota in q.QUOTAS)


@pytest.mark.parametrize("panel,quota", [("development", 64), ("development", 32), ("holdout", 64), ("holdout", 32)])
def test_one_failed_split_cannot_be_pooled_away(monkeypatch, panel, quota):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan()
    rows = synthetic_rows(plan)
    for row in rows:
        if row["panel"] == panel and row["quota"] == quota and row["mode"] == "candidate":
            row["time"] = .476
    result = q.validate_panel(rows, plan)
    assert not result["qualified"] and not result["permits_fresh_operational_games"]
    assert not result["panels"][panel][str(quota)]["qualified"]
    assert sum(gate["qualified"] for group in result["panels"].values() for gate in group.values()) == 3


@pytest.mark.parametrize("component", ["action", "trace", "after"])
def test_any_mode_or_repeat_scientific_change_stops(monkeypatch, component):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan()
    rows = synthetic_rows(plan)
    # Development32 has no historical32 target, so this specifically exercises
    # the mode/repeat equality gate instead of a historical-output shortcut.
    changed = next(r for r in rows if r["quota"] == 32 and r["mode"] == "candidate")
    changed["science"][component] = {"changed": True}
    with pytest.raises(ValueError, match="mode/repeat"):
        q.validate_panel(rows, plan)


def test_all_modes_repeating_same_wrong_historical_answer_still_stop(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan()
    rows = synthetic_rows(plan)
    for row in rows:
        if row["quota"] == 64 and row["fixture_id"] == "development/0":
            row["science"]["trace"] = {"consistently_wrong": True}
    with pytest.raises(ValueError, match="historical"):
        q.validate_panel(rows, plan)


@pytest.mark.parametrize("kind", ["missing", "reordered", "duplicate", "quota", "mode", "schedule"])
def test_schedule_corruption_stops(monkeypatch, kind):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan()
    rows = synthetic_rows(plan)
    if kind == "missing": rows.pop()
    elif kind == "reordered": rows[0], rows[1] = rows[1], rows[0]
    elif kind == "duplicate": rows[1] = deepcopy(rows[0])
    elif kind == "quota": rows[0]["quota"] = 32
    elif kind == "mode": rows[0]["mode"] = "fallback"
    else: plan["timed_requests"][0]["quota"] = 32
    with pytest.raises(ValueError):
        q.validate_panel(rows, plan)


@pytest.mark.parametrize("after", [None, [], "missing"])
def test_completed_transition_cannot_be_omitted(after):
    row = {"arm": "search", "completed": True, "clock": {"status": "ok"}, "after": after}
    with pytest.raises(ValueError, match="completed next snapshot"):
        q._completed(row, {}, quota=64)


def test_both_quotas_must_use_original_search_scientific_arm():
    with pytest.raises(ValueError, match="scientific search arm"):
        q._completed({"arm": "reference"}, {}, quota=32)


def selection_candidates():
    result = []
    for cell_index, (phase, seat) in enumerate(q.saved.CELLS):
        ply = (0 if phase == "0..19" else 20 if phase == "20..39" else 40)+seat
        for choice in range(5):
            pair = cell_index*5+choice
            result.append({"phase": phase, "seat": seat, "ply": ply, "pair_index": pair,
                           "game_index": pair*2, "fixture_id": f"fixture/{pair}", "timing": choice,
                           "winner": choice % 2, "action": {"irrelevant": choice}})
    return result


def materialize_metadata(row):
    return {"fixture_id": row["fixture_id"], "observer_sha256": f"observer/{row['pair_index']}"}


def test_hash_selection_is_order_timing_action_outcome_blind_and_excludes_previous_roots():
    candidates = selection_candidates()
    args = dict(excluded_pairs={0}, excluded_observers={"observer/6"}, read_fixture=materialize_metadata)
    chosen = q.select_holdout_metadata(candidates, **args)
    changed = list(reversed(deepcopy(candidates)))
    for row in changed:
        row.update(timing=10000, winner="different", action={"different": True})
    assert q.select_holdout_metadata(changed, **args) == chosen
    assert len(chosen) == len({row["pair_index"] for row in chosen}) == 12
    assert not {0, 6} & {row["pair_index"] for row in chosen}
    assert all(sum(row["phase"] == phase and row["seat"] == seat for row in chosen) == 2
               for phase, seat in q.saved.CELLS)


def test_insufficient_independent_holdout_roots_do_not_trigger_fallback():
    with pytest.raises(ValueError, match="insufficient reserved roots"):
        q.select_holdout_metadata(selection_candidates(), excluded_pairs=set(range(4)), excluded_observers=set(),
                                  read_fixture=materialize_metadata)


def test_frozen_source_change_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"old": True}))
    monkeypatch.setattr(q, "_artifact", lambda p: {"path": "source.json", "bytes": p.stat().st_size,
                                                 "sha256": q.saved.digest(p.read_text())})
    plan = {"source_artifacts": [q._artifact(path)]}
    q.verify_sources(plan)
    path.write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="historical evidence changed"):
        q.verify_sources(plan)


def test_real_saved_plan_is_corrected_complete_and_excludes_all_diagnosis_observers():
    # Saved JSON, hashes and independent ledger checks only: no model imports or
    # worker/native launches. This catches corpus identity/schema integration.
    plan = q.build_plan()
    assert plan["ruleset_id"] == q.guard.RULESET_ID
    assert len(plan["timed_requests"]) == 288
    assert [x["fixture_id"].rsplit("/", 1)[-1] for x in plan["panels"]["development"][:3]] == ["0042.json", "0048.json", "0050.json"]
    assert len({x["pair_index"] for x in plan["panels"]["holdout"]}) == 12
    assert not {x["pair_index"] for x in plan["panels"]["holdout"]} & set(plan["selection"]["excluded_main_pairs"])
    assert not {x["observer_sha256"] for x in plan["panels"]["holdout"]} & set(plan["selection"]["excluded_observers"])
    assert all(len(x["provenance"]) == 3*len(x["historical_records"]) for panel in plan["panels"].values() for x in panel)
    assert all("receipt_elapsed_seconds" not in x for x in plan["panels"]["holdout"])
    assert plan["historical_matching_scope"]["matching_records"] == 41
    both_quotas = [x for x in plan["panels"]["development"] if set(x["historical"]) == {"32", "64"}]
    assert len(both_quotas) == 3
