"""Model-free fixed-scope, freshness, exact science and split-gate tests."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.splendorust_inference64_20261025 import qualification as q


def test_imported_absolute_artifact_reference_is_canonical_without_hash_or_size_change(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "ROOT", tmp_path)
    source = {"path": str(tmp_path / "old" / "source.json"), "bytes": 123, "sha256": "a"*64}
    before = deepcopy(source)
    canonical = q.canonical_artifact_reference(source)
    assert canonical == {"path": "old/source.json", "bytes": 123, "sha256": "a"*64}
    assert source == before
    assert q.canonical_artifact_reference(canonical) == canonical
    with pytest.raises(ValueError, match="escaped workspace"):
        q.canonical_artifact_reference({**source, "path": str(tmp_path.parent / "outside.json")})


def candidates():
    result = []
    for cell, (phase, seat) in enumerate(q.saved.CELLS):
        for alternative in range(4):
            corpus = f"corpus{cell}"
            ply = (0 if cell < 2 else 20 if cell < 4 else 40)+seat
            result.append({"corpus": corpus, "source_stage": "operational", "pair_index": alternative,
                           "pair_key": f"{corpus}/operational/{alternative}", "game_index": 2*alternative,
                           "ply": ply, "seat": seat, "phase": phase, "fixture_id": f"{corpus}/root{alternative}",
                           "time": 3, "action": "unread", "winner": "unread"})
    return result


def materialize(row):
    return {"fixture_id": row["fixture_id"], "observer_sha256": f"observer:{row['fixture_id']}"}


def test_selection_blind_to_timings_actions_outcomes_permutation_and_namespaced_pair_ids():
    rows = candidates()
    excluded = {"observer:corpus0/root0"}
    chosen = q.select_holdout_metadata(rows, excluded_observers=excluded, read_fixture=materialize)
    altered = list(reversed(deepcopy(rows)))
    for row in altered:
        row.update(time=9999, action="changed", winner="changed")
    assert chosen == q.select_holdout_metadata(altered, excluded_observers=excluded, read_fixture=materialize)
    assert len(chosen) == len({row["pair_key"] for row in chosen}) == 12
    assert len({row["pair_index"] for row in chosen}) < 12
    assert not {row["observer_sha256"] for row in chosen} & excluded
    assert all(sum(row["phase"] == phase and row["seat"] == seat for row in chosen) == 2
               for phase, seat in q.saved.CELLS)


def test_two_roots_in_same_namespaced_pair_never_count_as_independent_pairs():
    rows = candidates()
    for row in rows:
        if row["corpus"] == "corpus0":
            row["pair_key"] = "only_one_pair"
    with pytest.raises(ValueError, match="no fallback"):
        q.select_holdout_metadata(rows, excluded_observers=set(), read_fixture=materialize)


def synthetic_plan():
    panels = {panel: [{"fixture_id": f"{panel}/{i}", "snapshot": {"root": i},
                       "historical_records": [{"quota": 64 if panel == "development" else 32,
                                               "science": {"id": f"{panel}/{i}", "quota": 64 if panel == "development" else 32}}]}
                      for i in range(count)] for panel, count in (("development", 24), ("holdout", 12))}
    return {"panels": panels, "timed_requests": q.make_schedule(panels), "source_artifacts": []}


def synthetic_rows(plan):
    return [{**row, "time": .5 if row["mode"] == "baseline" else .45,
             "science": {"id": row["fixture_id"], "quota": 64}} for row in plan["timed_requests"]]


def fake_completed(row, snapshot):
    assert row["quota"] == 64 and snapshot == {"root": row["fixture_index"]}
    return row["time"], row["science"]


def test_exact144_schedule_all64_with_two_reversed_paired_passes_and_original_slow_roots_first():
    plan = synthetic_plan(); schedule = plan["timed_requests"]
    assert len(schedule) == 144 and {row["quota"] for row in schedule} == {64}
    assert [row["index"] for row in schedule] == list(range(144))
    assert [row["fixture_index"] for row in schedule[:6]] == [0, 0, 1, 1, 2, 2]
    assert sum(row["panel"] == "development" for row in schedule) == 96
    assert sum(row["panel"] == "holdout" for row in schedule) == 48
    for panel in plan["panels"]:
        passes = [[(row["fixture_id"], row["mode"]) for row in schedule
                   if row["panel"] == panel and row["pass_index"] == p] for p in (0, 1)]
        assert passes[1] == list(reversed(passes[0]))


def test_historical32_provenance_is_not_relabelled_as64(monkeypatch):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan(); result = q.validate_panel(synthetic_rows(plan), plan)
    assert result["qualified"] and result["exact_scientific_groups"] == 36
    assert result["historical_comparisons"] == {"64": 96}
    assert result["unchanged_eager32_reference"]
    assert set(result["panels"]["development"]) == set(result["panels"]["holdout"]) == {"64"}


@pytest.mark.parametrize("panel", ["development", "holdout"])
def test_either64_split_failure_blocks_operations_without_pooling(monkeypatch, panel):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan(); rows = synthetic_rows(plan)
    for row in rows:
        if row["panel"] == panel and row["mode"] == "candidate":
            row["time"] = .476
    result = q.validate_panel(rows, plan)
    assert not result["qualified"] and not result["permits_fresh_operational_games"]
    assert not result["panels"][panel]["64"]["qualified"]


@pytest.mark.parametrize("kind", ["missing", "order", "wrong_quota", "changed_candidate", "wrong_history"])
def test_schedule_and_exact_science_corruption_stop(monkeypatch, kind):
    monkeypatch.setattr(q, "_completed", fake_completed)
    plan = synthetic_plan(); rows = synthetic_rows(plan)
    if kind == "missing": rows.pop()
    elif kind == "order": rows[0], rows[1] = rows[1], rows[0]
    elif kind == "wrong_quota": rows[0]["quota"] = 32
    elif kind == "changed_candidate":
        row = next(row for row in rows if row["panel"] == "holdout" and row["mode"] == "candidate")
        row["science"] = {"changed_callback": True}
    else:
        for row in rows:
            if row["panel"] == "development": row["science"] = {"consistently_wrong": True}
    with pytest.raises(ValueError):
        q.validate_panel(rows, plan)


def test_direct_completed32_is_rejected_before_any_audit():
    with pytest.raises(ValueError, match="must be64"):
        q._completed({"quota": 32}, {})


@pytest.mark.parametrize("after", [None, [], "missing"])
def test_missing_after_cannot_bypass_independent_transition(after):
    with pytest.raises(ValueError, match="completed next snapshot"):
        q._completed({"quota": 64, "arm": "search", "completed": True, "clock": {"status": "ok"}, "after": after}, {})


def test_gate_limits_unchanged_and_slow_baseline_preserved():
    assert q.timing_gate([.5]*10, [.475]*10)["qualified"]
    assert not q.timing_gate([.5]*10, [.476]*10)["qualified"]
    assert not q.timing_gate([.8]*10, [.701]*10)["qualified"]
    assert not q.timing_gate([.5]*20, [.1]*18+[.6]*2)["qualified"]
    assert q.timing_gate([.8], [.6])["baseline_margin_exceedances_preserved"] == 1
    with pytest.raises(ValueError): q.timing_gate([1.00001], [.3])


def test_parity_plan_is_exact_prior_frozen_plan_with_no_reselection():
    original = json.loads((q.PRIOR / "graph_parity_plan.json").read_text())
    plan = q.build_parity_plan()
    assert plan == original and len(plan["fixtures"]) == 11
    plan["fixtures"][0]["tokens"][0] = 9999
    assert q.build_parity_plan() == original


def test_fresh_ops_candidate64_against_eager32_only():
    schedule = q.operational_schedule()
    assert len(schedule) == 6 and len({row["seed_decimal"] for row in schedule}) == 3
    assert q.OPS_DOMAIN == "splendorust_inference64_ops_20261025_v1"
    for pair in range(3):
        rows = schedule[2*pair:2*pair+2]
        assert rows[0]["seed_decimal"] == rows[1]["seed_decimal"]
        assert {row["challenger_seat"] for row in rows} == {0, 1}
        assert all(row["challenger_mode"] == "candidate" and row["challenger_quota"] == 64 and
                   row["opponent_mode"] == "baseline" and row["opponent_quota"] == 32 for row in rows)


def test_real_saved_plan_fresh_roots_distinct_namespaced_pairs_corrected_history_and_disclosure():
    # Saved JSON, source hashes and independent transition audits; no model,
    # subprocess, fresh game or trained data is generated.
    plan = q.build_plan(); holdout = plan["panels"]["holdout"]
    assert len(plan["timed_requests"]) == 144 and len(holdout) == 12
    assert len(plan["selection"]["candidate_pairs"]) == 14
    assert len({row["pair_key"] for row in holdout}) == 12
    assert not {row["observer_sha256"] for row in holdout} & set(plan["selection"]["excluded_observers"])
    previous = json.loads((q.PRIOR / "qualification_plan.json").read_text())
    prior_observers = {row["observer_sha256"] for panel in previous["panels"].values() for row in panel}
    assert prior_observers <= set(plan["selection"]["excluded_observers"])
    assert len(plan["selection"]["excluded_main_pairs"]) == 95
    assert all(row["pair_index"] not in plan["selection"]["excluded_main_pairs"]
               for row in holdout if row["source_stage"] == "main")
    assert sum(row["source_stage"] == "main" for row in holdout) == 4
    assert plan["historical_matching_scope"]["matching_records"] == 353
    assert all(set(row["historical"]) == {"32"} for row in holdout)
    assert 4*sum(sum(h["quota"] == 64 for h in row["historical_records"])
                 for panel in plan["panels"].values() for row in panel) == 724
    assert plan["ruleset_id"] == q.guard.RULESET_ID and plan["catalogue_sha256"] == q.guard.CATALOGUE_SHA256
    assert plan["scope_selected_after_preceding_joint_result"]
    assert not plan["selection"]["wholly_untouched_source_games_claimed"]
    assert plan["selection"]["maximum_roots_per_namespaced_pair"] == 1
    # The actual freezer recursively imports artifact triples, including old
    # index references nested under selection metadata. Verifying only the
    # top-level source_artifacts misses absolute-versus-relative mismatches.
    from tools.splendorust_inference64_20261025 import common, runner
    nested = list(runner.nested_artifacts(plan)) + list(runner.nested_artifacts(q.build_parity_plan()))
    assert nested and all(not Path(item["path"]).is_absolute() for item in nested)
    unique = {}
    for item in nested:
        assert unique.setdefault(item["path"], item) == item
    common.verify(list(unique.values()))
