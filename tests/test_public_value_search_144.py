from dataclasses import replace
import copy
import json
import runpy

import pytest
import torch

from splendor_ai.alphazero import run_public_value_search as exp
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig, _Edge, _Node, _PathStep
from splendor_ai.encoding import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(scope="module")
def inference():
    exp.initialize_worker()
    return exp._VALUE


def test_public_evaluator_exact_policy_logits_and_complete_attention(inference):
    env = SplendorEnv(seed=1449000100)
    state = env.initial_state()
    for _ in range(36):
        actions = env.legal_actions(state)
        scores, _, row, keys, _ = inference.full_prediction(state, actions)
        assert tuple(scores.tolist()) == exp._RAW.score_actions(state, actions)
        reps = list(actions_by_canonical_successor_key(state, actions).values())
        pred = inference.predict_one(state, reps[:3])
        for k in range(min(3, len(reps))):
            assert pred.logits[k] == float(scores[row["groups"] == k].max())
        assert pred.value == inference.public_value(state)
        state = env.step(state, exp._RAW.choose_action(env, state, actions))
        if state.terminal:
            break


def test_public_value_is_antisymmetric_clipped_and_143_exact(inference):
    env = SplendorEnv(seed=1449000101)
    state = env.initial_state()
    state.players[0].score = 3
    state.players[0].bonuses["white"] = 2
    state.players[1].tokens["gold"] = 1
    expected = float((exp.simple_features(state).double() @ inference.coefficients).clamp(-1, 1))
    assert inference.public_value(state) == expected != 0
    state.current_player = 1
    assert inference.public_value(state) == -expected
    state.players[1].score = 100
    assert inference.public_value(state) == 1
    state.current_player = 0
    assert inference.public_value(state) == -1


def test_hidden_orders_do_not_affect_value_policy_or_fixed_search(inference):
    env = SplendorEnv(seed=1449000102)
    state = env.initial_state()
    altered = copy.deepcopy(state)
    for deck in altered.hidden_tier_decks.values():
        deck.reverse()
    legal = env.legal_actions(state)
    assert inference.predict_one(state, legal) == inference.predict_one(altered, legal)
    config = SearchConfig(simulations=4, root_noise_fraction=0)
    a = AlphaZeroSearch(inference, config).run(env, state, self_play=False, seed=71)
    b = AlphaZeroSearch(inference, config).run(env, altered, self_play=False, seed=71)
    assert a.root_edges == b.root_edges and a.action == b.action


def test_timed_public_search_reproduces_fixed_count(inference):
    env = SplendorEnv(seed=1449000103)
    state = env.initial_state()
    timed = exp.BudgetSearch(inference, SearchConfig(**exp.CONFIG["search"])).run(env, state, self_play=False, seed=81, seconds=.04)
    fixed = AlphaZeroSearch(inference, replace(SearchConfig(**exp.CONFIG["search"]), simulations=timed.simulations)).run(env, state, self_play=False, seed=81)
    assert exp.search_record(timed) == exp.search_record(fixed)
    assert timed.action == fixed.action


def test_leaf_value_backup_changes_player_perspective():
    node = _Node(player_id=0, edges={})
    edge = _Edge(prior=1)
    AlphaZeroSearch._backup([_PathStep(node, edge, 0, 1)], .75)
    assert edge.q == -.75 and edge.visits == 1
    edge2 = _Edge(prior=1)
    AlphaZeroSearch._backup([_PathStep(node, edge2, 0, 0)], .75)
    assert edge2.q == .75


def fake_games():
    return [{"index": i, "seed": exp.schedule(i)[0], "candidate_seat": i%2, "probe": False,
             "winner": 0, "candidate_points": float(i%2 == 0), "valid_natural": True,
             "termination_reason": "completed", "final_scores": [15, 12], "turns": 60,
             "trace": [{"candidate": True, "decision_seconds": .402, "search_seconds": .401,
                        "cpu_seconds": .3, "simulations": 100, "max_depth": 5},
                       {"candidate": False, "decision_seconds": .002, "cpu_seconds": .001}]}
            for i in range(400)]


def test_paired_units_and_complete_summary_json_serialization():
    result = exp.summarize(fake_games())
    assert result["candidate_score_share"] == .5
    assert result["paired_ci95"] == [.5, .5]
    assert result["decision"] == "improvement_not_established"
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    independent = runpy.run_path("scripts/analysis/verify_public_value_search_144.py")["independent_primary"]
    assert independent(fake_games()) == (.5, [.5, .5])


@pytest.mark.parametrize("problem", ["duplicate", "seed", "seat", "outcome"])
def test_bad_schedule_and_accounting_rejected(problem):
    records = fake_games()
    if problem == "duplicate":
        records[0]["index"] = 1
    elif problem == "seed":
        records[1]["seed"] += 1
    elif problem == "seat":
        records[1]["candidate_seat"] = 0
    else:
        records[1]["candidate_points"] = .5
    with pytest.raises(ValueError):
        exp.summarize(records)


@pytest.mark.parametrize("problem", ["cutoff", "budget"])
def test_integrity_failure_cannot_pass(problem):
    records = fake_games()
    for r in records:
        r["winner"] = r["candidate_seat"]
        r["candidate_points"] = 1.
    if problem == "cutoff":
        records[0]["valid_natural"] = False
    else:
        for r in records:
            r["trace"][0]["decision_seconds"] = .6
    result = exp.summarize(records)
    assert result["decision"] == "invalid_integrity_or_budget"
    json.dumps(result, allow_nan=False)


def test_frozen_decision_boundaries():
    assert exp.decision_label(.55, [.501, .599], True) == "material_search_system_improvement"
    assert exp.decision_label(.54, [.501, .579], True) == "positive_improvement_below_material_threshold"
    assert exp.decision_label(.44, [.39, .49], True) == "search_system_worse_than_raw_035"
    assert exp.decision_label(.56, [.50, .62], True) == "improvement_not_established"


def test_cutoff_saved_without_false_winner_and_audit_detects_tampering(tmp_path, monkeypatch, inference):
    monkeypatch.setitem(exp.CONFIG, "max_turns", 2)
    monkeypatch.setitem(exp.CONFIG, "decision_seconds", .01)
    exp.play_one(0, "test", tmp_path, probe=True)
    path = tmp_path / "games/game_0000.json"
    record = exp.read_json(path)
    assert not record["valid_natural"] and record["winner"] is None
    assert record["candidate_points"] == .5
    audit = runpy.run_path("scripts/analysis/verify_public_value_search_144.py")["audit_game"]
    assert audit(path, "test", probe=True)["decisions"] == 2
    record["trace"][0]["state_hash"] = "tampered"
    exp.atomic_write(path, record)
    with pytest.raises(AssertionError):
        audit(path, "test", probe=True)
