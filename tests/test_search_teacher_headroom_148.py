import copy
from dataclasses import replace
import json
import runpy

import pytest

from splendor_ai.alphazero import run_search_teacher_headroom as exp
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(scope="module")
def models():
    exp.initialize_worker()
    return exp._VALUE


def fake_games(panel="compute"):
    records = []
    for i in range(exp.CONFIG["games_per_panel"]):
        trace = []
        for arm in exp.arms(panel):
            budget = exp.CONFIG["seconds_by_arm"].get(arm, .003)
            row = {"arm": arm, "player": i%2 if arm == "long" else 1-i%2,
                   "decision_seconds": budget+.002, "cpu_seconds": budget*.9}
            if arm != "raw":
                row.update(allowance_seconds=budget, search_seconds=budget+.001,
                           simulations=256 if arm == "long" else 64, max_depth=4 if arm == "long" else 2)
            trace.append(row)
        records.append({"panel": panel, "index": i, "seed": exp.schedule(panel, i)[0],
                        "candidate_seat": i%2, "probe": False, "winner": 0,
                        "candidate_points": float(i%2 == 0), "valid_natural": True,
                        "termination_reason": "completed", "final_scores": [15, 12], "turns": 60,
                        "trace": trace})
    return records


def test_seed_panels_and_preflight_are_disjoint_and_pair_seats():
    sets = []
    for panel in exp.PANELS:
        seeds = {exp.schedule(panel, i)[0] for i in range(400)}
        assert len(seeds) == 200
        assert all(exp.schedule(panel, 2*i)[0] == exp.schedule(panel, 2*i+1)[0] for i in range(200))
        assert all(exp.schedule(panel, i)[1] == i%2 for i in range(400))
        sets.extend([seeds, {exp.schedule(panel, i, True)[0] for i in range(2)}])
    assert sum(map(len, sets)) == len(set.union(*sets))
    assert exp.schedule("compute", 399) == (1480000199, 1)
    assert exp.schedule("competence", 399) == (1481000199, 1)
    for panel, i, probe in [("unknown", 0, False), ("compute", -1, False), ("compute", 400, False), ("compute", 2, True)]:
        with pytest.raises(ValueError):
            exp.schedule(panel, i, probe)


@pytest.mark.parametrize("panel", exp.PANELS)
def test_pair_bootstrap_summary_and_independent_statistics(panel):
    records = fake_games(panel)
    # A pure seat effect must have zero uncertainty when resampling whole pairs.
    result = exp.summarize(panel, records)
    assert result["candidate_score_share"] == .5 and result["paired_ci95"] == [.5, .5]
    assert result["decision"] == "improvement_not_established"
    assert all(result["budget_passed_by_arm"].values())
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    audit = runpy.run_path(str(exp.VERIFIER))
    assert audit["independent_primary"](panel, records) == (.5, [.5, .5])
    assert audit["independent_gate"](panel, records, .5, [.5, .5]) == result["decision"]


@pytest.mark.parametrize("problem", ["seed", "duplicate", "outcome", "arm", "panel", "probe"])
def test_accounting_and_treatment_corruption_rejected(problem):
    records = fake_games()
    if problem == "seed":
        records[1]["seed"] += 1
    elif problem == "duplicate":
        records[1]["index"] = 0
    elif problem == "outcome":
        records[1]["candidate_points"] = .5
    elif problem == "arm":
        records[1]["trace"][0]["arm"] = "short"
    elif problem == "panel":
        records[1]["panel"] = "competence"
    else:
        records[1]["probe"] = True
    with pytest.raises(ValueError):
        exp.summarize("compute", records)


@pytest.mark.parametrize("problem", ["cutoff", "long_budget", "short_budget", "underuse", "allowance"])
def test_budget_or_termination_failure_blocks_winning_candidate(problem):
    records = fake_games()
    for r in records:
        r["winner"], r["candidate_points"] = r["candidate_seat"], 1.
    if problem == "cutoff":
        records[0].update(valid_natural=False, winner=None, candidate_points=.5)
    elif problem in ("long_budget", "short_budget"):
        index = 0 if problem == "long_budget" else 1
        for r in records:
            r["trace"][index]["decision_seconds"] *= 1.5
    elif problem == "underuse":
        records[0]["trace"][0]["search_seconds"] = .4
    else:
        records[0]["trace"][0]["allowance_seconds"] = .4
    result = exp.summarize("compute", records)
    assert result["decision"] == "invalid_integrity_or_budget"
    assert not result["practical_55_percent_ruled_out"]
    independent = runpy.run_path(str(exp.VERIFIER))["independent_gate"]
    assert independent("compute", records, result["candidate_score_share"], result["paired_ci95"]) == result["decision"]


def test_decision_boundaries_and_missing_outcomes():
    assert exp.decision_label(.55, [.501, .599], True) == "material_superiority"
    assert exp.decision_label(.54, [.501, .579], True) == "positive_below_material_threshold"
    assert exp.decision_label(.44, [.39, .49], True) == "candidate_weaker"
    assert exp.decision_label(.56, [.50, .62], True) == "improvement_not_established"
    for values in ([1.]*399, [1.]*399+[float("nan")], [1.]*399+[.25]):
        with pytest.raises(ValueError):
            exp.paired_statistics("compute", values)


@pytest.mark.parametrize("problem", [None, "failed_strength", "failed_audit", "changed_result", "wrong_binding", "incomplete"])
def test_conditional_stage_requires_exact_complete_audited_success(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    summary = {"decision": "material_superiority", "integrity_and_budget_passed": True,
               "candidate_score_share": .6, "paired_ci95": [.55, .65]}
    if problem == "failed_strength":
        summary.update(decision="improvement_not_established", paired_ci95=[.49, .65])
    result = {"manifest": "bound", "summary": summary}
    path = tmp_path / "compute/result.json"
    exp.atomic_write(path, result)
    audit = {"manifest": "bound", "panel": "compute", "passed": True, "games": 400,
             "result_sha256": exp.file_hash(path)}
    if problem == "failed_audit":
        audit["passed"] = False
    elif problem == "changed_result":
        result["tampered"] = True
        exp.atomic_write(path, result)
    elif problem == "wrong_binding":
        audit["manifest"] = "other"
    elif problem == "incomplete":
        audit["games"] = 398
    exp.atomic_write(tmp_path / "compute/verification.json", audit)
    if problem in (None, "failed_strength"):
        assert exp.audited_panel_passed("compute", "bound") == (problem is None)
    else:
        with pytest.raises(ValueError):
            exp.audited_panel_passed("compute", "bound")


def test_competence_does_not_start_after_compute_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "audited_panel_passed", lambda panel, binding: False)
    with pytest.raises(ValueError, match="Conditional competence blocked"):
        exp.run_panel("competence", {})
    assert not (tmp_path / "competence").exists()


def test_same_model_and_search_configuration_for_both_allowances(models):
    long = exp.ArenaBot("long", 2, [])
    short = exp.ArenaBot("short", 2, [])
    assert long.search.config == short.search.config == SearchConfig(**exp.CONFIG["search"])
    # Both constructors receive the same immutable policy/value object.
    assert exp.CONFIG["seconds_by_arm"] == {"long": 1.6, "short": .4}
    assert exp.ArenaBot("raw", 2, []).search is None


def test_timed_search_reproduces_fixed_count_without_hidden_deck_leak(models):
    env = SplendorEnv(seed=1489000101)
    state = env.initial_state()
    altered = copy.deepcopy(state)
    for deck in altered.hidden_tier_decks.values():
        deck.reverse()
    config = SearchConfig(**exp.CONFIG["search"])
    timed = exp.BudgetSearch(models, config).run(env, state, self_play=False, seed=93, seconds=.02)
    fixed = AlphaZeroSearch(models, replace(config, simulations=timed.simulations)).run(env, altered, self_play=False, seed=93)
    assert exp.search_record(timed) == exp.search_record(fixed) and timed.action == fixed.action


@pytest.mark.parametrize("panel", exp.PANELS)
def test_actual_budget_routes_full_replay_and_tamper_detection(tmp_path, monkeypatch, models, panel):
    monkeypatch.setitem(exp.CONFIG, "max_turns", 4)
    monkeypatch.setitem(exp.CONFIG, "seconds_by_arm", {"long": .02, "short": .005})
    exp.play_one(panel, 0, "test", tmp_path, probe=True)
    path = tmp_path / panel / "games/game_0000.json"
    record = exp.read_json(path)
    assert not record["valid_natural"] and record["winner"] is None and record["candidate_points"] == .5
    for row in record["trace"]:
        if row["arm"] != "raw":
            assert row["allowance_seconds"] == exp.CONFIG["seconds_by_arm"][row["arm"]]
    audit = runpy.run_path(str(exp.VERIFIER))["audit_game"]
    report = audit(path, "test", panel, probe=True)
    assert report["searches_reproduced"]["long"] == 2
    if panel == "compute":
        assert report["searches_reproduced"]["short"] == 2
    else:
        assert report["raw_actions_reproduced"] == 2
    record["trace"][0]["raw_network_value"] = .987
    exp.atomic_write(path, record)
    with pytest.raises(AssertionError):
        audit(path, "test", panel, probe=True)
