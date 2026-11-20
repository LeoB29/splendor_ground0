"""No model calls: deterministic selection, strict parity and split gates."""
from copy import deepcopy
import hashlib
import json

import pytest

from tools.splendorust_inference_opt_20261025 import qualification as q


def metadata_candidates():
    rows = []
    for cell, (phase, seat) in enumerate(q.saved.CELLS):
        for choice in range(5):
            pair = cell*5+choice
            ply = (0 if cell < 2 else 20 if cell < 4 else 40)+seat
            rows.append(dict(phase=phase, seat=seat, pair_index=pair, game_index=pair*2,
                             ply=ply, fixture_id=f"fixture/{pair}", timing=choice, action=choice, outcome=choice))
    return rows


def materialize(row):
    return {"fixture_id": row["fixture_id"], "observer_sha256": f"observer/{row['pair_index']}"}


def test_selection_blind_to_timing_action_outcome_and_input_order():
    rows = metadata_candidates()
    kwargs = dict(excluded_pairs={0}, excluded_observers={"observer/6"}, read_fixture=materialize)
    selected = q.select_holdout_metadata(rows, **kwargs)
    altered = list(reversed(deepcopy(rows)))
    for row in altered:
        row.update(timing=999, action="different", outcome="different")
    assert selected == q.select_holdout_metadata(altered, **kwargs)
    assert len(selected) == len({r["pair_index"] for r in selected}) == 12
    assert not {0, 6} & {r["pair_index"] for r in selected}
    assert all(sum(r["phase"] == phase and r["seat"] == seat for r in selected) == 2
               for phase, seat in q.saved.CELLS)


def test_no_fallback_when_independent_reserved_data_insufficient():
    with pytest.raises(ValueError, match="insufficient fresh"):
        q.select_holdout_metadata(metadata_candidates(), excluded_pairs=set(range(4)),
                                  excluded_observers=set(), read_fixture=materialize)


def test_prior_exclusions_include_reserved_and_hidden_nested_manifests():
    observer1, observer2 = "a"*64, "b"*64
    pairs, observers = q.prior_panel_exclusions([
        {"panels": {"holdout": [{"pair_index": 2, "observer_sha256": observer1}]}},
        {"selection": {"excluded_main_pairs": [3], "excluded_observers": [observer2]}},
        {"fixtures": [{"hidden_variant": {"observer_sha256": observer2}}]},
    ])
    assert pairs == {2, 3} and observers == {observer1, observer2}


@pytest.mark.parametrize("bad", [{"excluded_main_pairs": [True]}, {"excluded_observers": ["invalid"]}])
def test_bad_prior_identity_cannot_silently_reduce_exclusions(bad):
    with pytest.raises(ValueError, match="identity"):
        q.prior_panel_exclusions([bad])


def synthetic_plan():
    panels = {panel: [{"fixture_id": f"{panel}/{i}", "snapshot": {"root": i},
                       "historical_records": [{"quota": 64 if panel == "development" else 32,
                                               "science": {"id": f"{panel}/{i}"}}]}
                      for i in range(count)] for panel, count in (("development", 24), ("holdout", 12))}
    return {"panels": panels, "timed_requests": q.make_schedule(panels), "source_artifacts": []}


def synthetic_rows(plan):
    return [{**spec, "elapsed": .5 if spec["mode"] == "baseline" else .45,
             "science": {"id": spec["fixture_id"]}} for spec in plan["timed_requests"]]


def fake_completed(row, snapshot, *, quota):
    assert snapshot == {"root": row["fixture_index"]} and row["quota"] == quota
    return row["elapsed"], row["science"]


def test_schedule_both_modes_quotas_repeats_first_three_slow_roots():
    plan = synthetic_plan()
    schedule = plan["timed_requests"]
    assert len(schedule) == 288 and [r["index"] for r in schedule] == list(range(288))
    assert [r["fixture_index"] for r in schedule[:6]] == [0, 0, 1, 1, 2, 2]
    assert all(r["quota"] == 64 for r in schedule[:144])
    assert all(r["quota"] == 32 for r in schedule[144:])
    for panel in plan["panels"]:
        for quota in q.QUOTAS:
            passes = [[(r["fixture_id"], r["mode"]) for r in schedule
                       if r["panel"] == panel and r["quota"] == quota and r["pass_index"] == i]
                      for i in (0, 1)]
            assert passes[1] == list(reversed(passes[0]))


def test_delegated_scientific_validator_preserves_exact_groups_history_and_separate_gates(monkeypatch):
    monkeypatch.setattr(q.prior, "_completed", fake_completed)
    plan = synthetic_plan()
    result = q.validate_panel(synthetic_rows(plan), plan)
    assert result["qualified"] and result["exact_scientific_groups"] == 72
    assert result["historical_comparisons"] == {"64": 96, "32": 48}
    assert result["permits_fresh_operational_games"]


@pytest.mark.parametrize("panel,quota", [("development", 64), ("development", 32), ("holdout", 64), ("holdout", 32)])
def test_each_split_independently_blocks_operational_games(monkeypatch, panel, quota):
    monkeypatch.setattr(q.prior, "_completed", fake_completed)
    plan = synthetic_plan()
    rows = synthetic_rows(plan)
    for row in rows:
        if row["panel"] == panel and row["quota"] == quota and row["mode"] == "candidate":
            row["elapsed"] = .476
    result = q.validate_panel(rows, plan)
    assert not result["qualified"] and not result["permits_fresh_operational_games"]
    assert not result["panels"][panel][str(quota)]["qualified"]


@pytest.mark.parametrize("mutation", ["history", "repeat", "order", "missing"])
def test_exact_science_history_and_complete_schedule_remain_mandatory(monkeypatch, mutation):
    monkeypatch.setattr(q.prior, "_completed", fake_completed)
    plan = synthetic_plan()
    rows = synthetic_rows(plan)
    if mutation == "history":
        for row in rows:
            row["science"] = {"wrong": True}
    elif mutation == "repeat":
        row = next(r for r in rows if r["panel"] == "development" and r["quota"] == 32 and r["mode"] == "candidate")
        row["science"] = {"wrong": True}
    elif mutation == "order": rows[0], rows[1] = rows[1], rows[0]
    else: rows.pop()
    with pytest.raises(ValueError):
        q.validate_panel(rows, plan)


def callback_event(value):
    tokens = [float(value)] + [0.] * (31*48-1)
    outputs = [[float(value)] * 83]
    return {"arm": "search", "event": "result", "operation": "ordinary_worker_request", "case_id": f"case{value}",
            "request": {"observer": {"ruleset_id": q.guard.RULESET_ID, "catalogue_sha256": q.guard.CATALOGUE_SHA256}},
            "result": {"diagnostics": {"phases": [{"inference": [{"batch": 1, "tokens": tokens,
                       "input_sha256": q.saved.digest(tokens), "outputs": outputs,
                       "output_sha256": q.saved.digest(outputs)}]}]}}}


def forensic_lines(events):
    previous, lines = None, []
    for index, event in enumerate(events):
        row = {"sequence": index, "previous_sha256": previous, "event": event}
        lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()+b"\r\n")
        previous = q.saved.digest(row)
    return lines


def test_parity_source_order_dedup_nonzero_first_callback_only_and_exact_line_provenance():
    events = [callback_event(value) for value in (0, 1, 1, 2, 3, 4, 5, 6, 7)]
    # A leaf callback cannot supply a selected root.
    events[1]["result"]["diagnostics"]["phases"][0]["inference"].append(
        callback_event(99)["result"]["diagnostics"]["phases"][0]["inference"][0])
    lines = forensic_lines(events)
    roots = q.select_parity_roots(lines)
    assert [row["tokens"][0] for row in roots] == [1., 2., 3., 4., 5., 6.]
    assert [row["source_line_1based"] for row in roots] == [2, 4, 5, 6, 7, 8]
    assert roots[0]["source_line_sha256"] == hashlib.sha256(lines[1]).hexdigest()
    assert roots[0]["historical_outputs"] == [[1.] * 83]


@pytest.mark.parametrize("bad", ["shape", "batch", "nan", "bool", "input_hash", "output_hash", "rules"])
def test_invalid_recorded_callback_rejected(bad):
    events = [callback_event(value) for value in range(1, 7)]
    callback = events[0]["result"]["diagnostics"]["phases"][0]["inference"][0]
    if bad == "shape": callback["tokens"].pop()
    elif bad == "batch": callback["batch"] = 2
    elif bad == "nan": callback["tokens"][0] = float("nan")
    elif bad == "bool": callback["tokens"][0] = True
    elif bad == "input_hash": callback["input_sha256"] = "wrong"
    elif bad == "output_hash": callback["output_sha256"] = "wrong"
    else: events[0]["request"]["observer"]["ruleset_id"] = "legacy"
    # NaN cannot even be represented in the authentic source's canonical hash.
    if bad == "nan":
        lines = forensic_lines([callback_event(value) for value in range(1, 7)])
        row = json.loads(lines[0]); row["event"] = events[0]
        lines[0] = json.dumps(row).encode()+b"\n"
    else: lines = forensic_lines(events)
    with pytest.raises(ValueError):
        q.select_parity_roots(lines)


def test_forensic_chain_after_selected_six_is_still_validated():
    lines = forensic_lines([callback_event(value) for value in range(1, 8)])
    row = json.loads(lines[-1]); row["previous_sha256"] = "tampered"
    lines[-1] = json.dumps(row).encode()+b"\n"
    with pytest.raises(ValueError, match="chain"):
        q.select_parity_roots(lines)


def test_missing_distinct_parity_cases_do_not_fall_back_to_zeros():
    with pytest.raises(ValueError, match="six distinct"):
        q.select_parity_roots(forensic_lines([callback_event(1)]*6))


def test_real_parity_inputs_composites_zeros_and_source_provenance():
    plan = q.build_parity_plan()
    fixtures = plan["fixtures"]
    assert len(fixtures) == 11 and all(set(row) == {"fixture_id", "batch", "tokens"} for row in fixtures)
    assert [row["batch"] for row in fixtures] == [1]*6+[2, 8, 1, 2, 8]
    assert [plan["provenance_by_fixture"][row["fixture_id"]]["source_line_1based"] for row in fixtures[:6]] == [7, 23, 39, 55, 71, 87]
    assert fixtures[6]["tokens"] == fixtures[0]["tokens"]+fixtures[1]["tokens"]
    assert fixtures[7]["tokens"] == sum([fixtures[index]["tokens"] for index in [0, 1, 2, 3, 4, 5, 0, 1]], [])
    assert all(len(row["tokens"]) == row["batch"]*31*48 and not any(row["tokens"]) for row in fixtures[8:])
    assert not plan["reserved_roots_used"] and plan["exact_float32_bitwise_outputs_required"]
    assert not plan["relative_or_absolute_tolerance_permitted"]


def test_fresh_operational_pairs_swap_seats_hold_players_and_use_new_namespace():
    schedule = q.operational_schedule()
    assert len(schedule) == 6 and len({r["seed_decimal"] for r in schedule}) == 3
    assert q.OPS_DOMAIN == "splendorust_inference_graph_ops_20261025_v1"
    for pair in range(3):
        rows = schedule[2*pair:2*pair+2]
        assert rows[0]["seed_decimal"] == rows[1]["seed_decimal"]
        assert {r["challenger_seat"] for r in rows} == {0, 1}
        assert all(r["challenger_quota"] == 64 and r["opponent_quota"] == 32 and
                   r["challenger_mode"] == r["opponent_mode"] == "candidate" for r in rows)
        assert rows[0]["seed_decimal"] == str(int(q.saved.digest([q.OPS_DOMAIN, "operational", pair]), 16))


def test_real_plan_freshness_corrected_sources_complete_history_and_saved_replay():
    # Reading archived JSON and independently checking transitions only.
    plan = q.build_plan()
    holdout = plan["panels"]["holdout"]
    assert len(plan["timed_requests"]) == 288 and len(holdout) == 12
    assert len({row["pair_index"] for row in holdout}) == 12
    assert not {row["pair_index"] for row in holdout} & set(plan["selection"]["excluded_main_pairs"])
    assert not {row["observer_sha256"] for row in holdout} & set(plan["selection"]["excluded_observers"])
    assert len(plan["selection"]["excluded_main_pairs"]) == 83
    assert all(set(row["historical"]) == {"32", "64"} for row in plan["panels"]["development"])
    assert plan["historical_matching_scope"]["matching_records"] == 161
    assert plan["historical_matching_scope"]["additional_component_baseline_rows"] == 120
    assert all(len(row["provenance"]) == 3*len(row["historical_records"])
               for panel in plan["panels"].values() for row in panel)
    assert plan["ruleset_id"] == q.guard.RULESET_ID and plan["catalogue_sha256"] == q.guard.CATALOGUE_SHA256
    assert not plan["selection"]["ranking_uses_timing"] and not plan["selection"]["ranking_uses_actions_or_outcomes"]
