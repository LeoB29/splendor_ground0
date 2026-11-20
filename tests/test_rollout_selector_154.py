from concurrent.futures import ThreadPoolExecutor
import importlib.util
import random

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import run_rollout_selector as p
from splendor_ai.engine.state import RESERVE_SOURCE_DECK


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def audit():
    return module("audit154_test", "scripts/analysis/verify_rollout_selector_154.py")


def test_frozen_namespaces_and_budget():
    sets = [{p.world_seed(stage, i, w) for i in range(p.stage_spec(stage)[2]) for w in range(p.stage_spec(stage)[1])}
            for stage in ("discovery", "evaluation", "probe_discovery", "probe_evaluation")]
    sets += [set(range(1540000000, 1540004096)), {1549000000, 1549000001}, {1543000000}, set(range(1549300000,1549300768))]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    assert min(min(s) for s in sets) > 1529105003
    assert 768*(4*32+2*128) == p.CONFIG["main_cap"] == 294912
    for args in (("test", 0, 0), ("discovery", 768, 0), ("evaluation", 0, 128), ("discovery", -1, 0)):
        with pytest.raises(ValueError):
            p.world_seed(*args)


def test_catalog_world_independent_of_true_order_and_private_identity(audit):
    state = p.old.SplendorEnv(seed=154).initial_state()
    state.players[1].reserve_card(state.hidden_tier_decks[2].pop(), RESERVE_SOURCE_DECK)
    state.deck_counts[2] -= 1
    for seed in range(4):
        expected = p.old.determinize_for_observer(state, 0, random.Random(seed))
        rebuilt = audit.independent_world(state, seed)
        assert p.old.serialize_state_snapshot(expected) == p.old.serialize_state_snapshot(rebuilt)
        changed = state.copy_shallow()
        for cards in changed.hidden_tier_decks.values():
            cards.reverse()
        changed.players[1].reserved_cards[0], changed.hidden_tier_decks[2][0] = changed.hidden_tier_decks[2][0], changed.players[1].reserved_cards[0]
        assert p.old.serialize_state_snapshot(audit.independent_world(changed, seed)) == p.old.serialize_state_snapshot(expected)


@pytest.mark.parametrize("n,values,expected", [(1, [0.], 0), (2, [1., 1.], 0), (4, [0., .5, .5, -.5], 1), (3, [0., -.5, -.25], 0)])
def test_selection_ties_and_coincident_dedup(audit, n, values, expected):
    samples = [{"index": 0, "actions": [{"a": i} for i in range(n)], "keys": [str(i) for i in range(n)]}]
    records = [{"action_indices": list(range(n)), "worlds": [{"outcomes": [{"wdl": v} for v in values]}]*4}]
    rows = p.select_choices(samples, records)
    assert rows == audit.independent_selection(samples, records) and rows[0]["choice"] == expected
    assert p.action_indices("evaluation", samples[0], expected) == ([0] if expected == 0 else [0, expected])
    for choice in (None, -1, n):
        with pytest.raises(ValueError):
            p.action_indices("evaluation", samples[0], choice)


def test_independent_source_selection_keeps_first_feasible_prefix(audit, monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setitem(p.CONFIG, "quota", 1)
    monkeypatch.setitem(p.CONFIG, "states", 6)
    initial = p.old.SplendorEnv(seed=154).initial_state()
    states = []
    for cell in range(6):
        state = initial.copy_shallow()
        state.turn_index = (cell//2)*20+cell%2
        state.current_player = cell%2
        states.append(state)
    monkeypatch.setattr(p.old, "reconstruct_source", lambda record: [states[record["index"]%6]])
    monkeypatch.setattr(p.old, "candidates", lambda *args: {"actions": [{}], "keys": ["stub"]})
    records, summaries = [], []
    for i in range(8):
        r = {"index": i, "seed": 1540000000+i, "reason": "completed"}
        p.old.save_new(p.source_path(i), r)
        records.append(r)
        summaries.append({"index": i, "eligible": [int(c == i%6) for c in range(6)], "sha256": p.old.file_hash(p.source_path(i))})
    panel = p.select_panel(records)
    assignments, prefix = audit.independent_matching(summaries, 1)
    assert [s["source_index"] for s in panel["samples"]] == list(range(6))
    assert panel["filtering"] == prefix and assignments == {i: i for i in range(6)}


def test_augmenting_match_can_reassign_previous_game(audit):
    # First game can serve every cell; later five cannot serve cell 5.
    source = [{"index": 0, "eligible": [1]*6, "sha256": "0"}]
    source += [{"index": i+1, "eligible": [int(c == i) for c in range(6)], "sha256": str(i+1)} for i in range(5)]
    assignments, _ = audit.independent_matching(source, 1)
    assert assignments[0] == 5 and len(assignments) == 6
    buckets, cells = [[] for _ in range(6)], {}
    for r in source:
        cells[r["index"]] = [c for c in range(6) if r["eligible"][c]]
        p.old.place_source(r["index"], cells, buckets, 1)
    assert assignments == {i: c for c, group in enumerate(buckets) for i in group}


def test_no_outcome_based_exclusions_or_prefetch_overrun(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(p.old, "initialize_worker", lambda: None)
    monkeypatch.setitem(p.CONFIG, "source_limit", 16)
    def task(i):
        p.old.save_new(p.source_path(i), {"index": i})
        return {"index": i, "reason": "completed", "sha256": "stub"}
    monkeypatch.setattr(p, "source_task", task)
    def select(records):
        assert [next(records)["index"] for _ in range(3)] == [0, 1, 2]
        return {"filtering": [0, 1, 2]}
    monkeypatch.setattr(p, "select_panel", select)
    p.generate_panel()
    lock = p.old.read_json(tmp_path/"sources_lock.json")
    assert len(lock["files"]) == 8 and lock["unused_prefetched"] == 5
    monkeypatch.setattr(p, "source_task", lambda i: {"index": i, "reason": "max_turns" if i == 7 else "completed"})
    with pytest.raises(ValueError, match="prefetched"):
        p.generate_panel()


def test_full_source_replay_and_tampering(audit, monkeypatch, tmp_path):
    env_class = p.old.SplendorEnv
    class RichEnv(env_class):
        def initial_state(self):
            state = super().initial_state()
            for player in state.players:
                player.bonuses = {c: 5 for c in player.bonuses}
                player.score = 14
            return state
    class FakePolicy:
        def score_actions(self, state, legal):
            return [10. if a.action_type.name.startswith("BUY") else 0. for a in legal]
        def choose_action(self, env, state, legal=None):
            legal = env.legal_actions(state) if legal is None else legal
            scores = self.score_actions(state, legal)
            return legal[max(range(len(legal)), key=lambda j: scores[j])]
    monkeypatch.setattr(p.old, "SplendorEnv", RichEnv)
    monkeypatch.setattr(p.old, "_RAW", FakePolicy())
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    info = p.source_task(0, True)
    checked = audit.source_check(0, info, True)
    assert checked["moves"] == 2
    info["turns"] += 1
    with pytest.raises(AssertionError):
        audit.source_check(0, info, True)


def test_collection_replay_identity_and_reuse(audit, monkeypatch, tmp_path):
    helpers = module("positions154_test", "tests/test_afterstate_advantage_149.py")
    state = helpers.positions()[2]
    action = next(a for a in p.old.SplendorEnv().legal_actions(state) if a.noble_id is not None)
    sample = {"index": 0, "snapshot": p.old.serialize_state_snapshot(state), "actions": [p.old.serialize_action(action)]}
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    info = p.collect_task("probe_discovery", sample, None, "bound")
    rebuilt = audit.target_check("probe_discovery", sample, None, "bound", info)
    assert rebuilt["continuations"] == 4 and rebuilt["moves"] == 4
    assert p.collect_task("probe_discovery", sample, None, "bound") == info
    with pytest.raises(AssertionError):
        p.collect_task("probe_discovery", sample, None, "other")
    info["continuations"] += 1
    with pytest.raises(AssertionError):
        audit.target_check("probe_discovery", sample, None, "bound", info)


def statistic_rows():
    return [{"source_seed": i+1, "cell": i%6, "evaluation_gain": (i%5-1)/32,
             "discovery_gain": (i%5+1)/16, "choice": int(i%3 != 0), "candidate_count": 1 if i == 0 else 4} for i in range(60)]


def test_bootstrap_is_independent_stratified_source_game_arithmetic(audit):
    rows = statistic_rows()
    result = p.statistics(rows)
    audit.compare(result, audit.independent_statistics(rows))
    assert result["states"] == 60 and result["baseline_only_states"] == 1 and result["coincident_choices"] == 20
    with pytest.raises(AssertionError):
        p.statistics(rows+[rows[0]])


@pytest.mark.parametrize("gate", ("material_gain", "positive_lower_bound", "precision"))
def test_each_statistical_gate_is_required(monkeypatch, gate):
    rows = statistic_rows()
    for row in rows:
        row["evaluation_gain"] = .019 if gate == "material_gain" else .03
    bounds = {"material_gain": [.01, .025], "positive_lower_bound": [0., .039], "precision": [.001, .06]}[gate]
    monkeypatch.setattr(p.np, "quantile", lambda *args, **kwargs: np.asarray(bounds))
    result = p.statistics(rows)
    assert not result["statistical_pass"]
    assert [k for k, v in result["statistical_gates"].items() if not v] == [gate]


def test_no_silent_resume_or_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    path = tmp_path/"stage.json"
    p.save_identical_or_new(path, {"a": 1})
    p.save_identical_or_new(path, {"a": 1})
    with pytest.raises(AssertionError):
        p.save_identical_or_new(path, {"a": 2})
    p.old.save_new(tmp_path/"failure_run_1.json", {})
    with pytest.raises(ValueError, match="Preserved failure"):
        p.run(resume=True)


@pytest.mark.parametrize("passed", (False, True))
def test_complete_pipeline_locks_choices_before_evaluation_and_always_audits(monkeypatch, tmp_path, passed):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "check_identity", lambda: "bound")
    monkeypatch.setattr(p.old, "initialize_worker", lambda: None)
    for name in ("probe_audit.json", "timing_probe.json", "sources_lock.json"):
        p.old.save_new(tmp_path/name, {})
    p.old.save_new(tmp_path/"preflight.json", {"passed": True, "manifest": "bound", "probe_audit_sha256": p.old.file_hash(tmp_path/"probe_audit.json"), "timing_sha256": p.old.file_hash(tmp_path/"timing_probe.json")})
    sample = {"index": 0, "actions": [{}], "keys": ["x"]}
    monkeypatch.setattr(p, "generate_panel", lambda: {"samples": [sample]})
    events = []
    def collect(stage, samples, choices, upstream):
        events.append(stage)
        if stage == "evaluation":
            assert (tmp_path/"selection_lock.json").exists() and (tmp_path/"discovery_audit.json").exists()
            assert choices == [0] and upstream == p.old.file_hash(tmp_path/"selection_lock.json")
        p.old.save_new(tmp_path/f"{stage}_targets_lock.json", {})
    monkeypatch.setattr(p, "collect", collect)
    monkeypatch.setattr(p, "read_records", lambda *args: [{"action_indices": [0], "worlds": [{"outcomes": [{"wdl": 1.}]}]}])
    monkeypatch.setattr(p, "per_state", lambda *args: [])
    monkeypatch.setattr(p, "statistics", lambda *args: {"statistical_pass": passed, "gain": 0., "ci95": [0.,0.]})
    def audit(stage):
        events.append(stage+"_audit")
        value = {"passed": True}
        if stage == "panel": value["panel_lock_sha256"] = p.old.file_hash(tmp_path/"panel_lock.json")
        if stage == "discovery": value["selection_sha256"] = p.old.file_hash(tmp_path/"selection_lock.json")
        if stage == "final":
            value.update(main_continuations=0, **{field: p.old.file_hash(tmp_path/name) for field, name in (("result_sha256", "result.json"), ("state_results_sha256", "state_results.json"), ("evaluation_targets_sha256", "evaluation_targets_lock.json"))})
        p.old.save_new(tmp_path/f"{stage}_audit.json", value)
        return value
    monkeypatch.setattr(p, "audit_stage", audit)
    p.run()
    assert events == ["panel_audit", "discovery", "discovery_audit", "evaluation", "final_audit"]
    status = p.old.read_json(tmp_path/"final_status.json")
    assert status["complete"] and status["advance_gate_passed"] == passed
    assert not any(status[k] for k in ("arena", "training", "promotion", "automatic_successor"))
