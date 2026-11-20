import copy
from dataclasses import replace
import json
import runpy

import pytest

from splendor_ai.alphazero import run_public_value_ablation as exp
from splendor_ai.alphazero.network import NetworkPrediction
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(scope="module")
def models():
    exp.initialize_worker()
    return exp._MODELS


def test_switch_changes_only_value_and_preserves_inference_work():
    class Spy:
        calls = 0

        def predict_one(self, state, actions):
            self.calls += 1
            return NetworkPrediction((1., -2.), .75)

    base = Spy()
    public = exp.LeafModeInference(base, "public").predict_one(None, None)
    neutral = exp.LeafModeInference(base, "neutral").predict_one(None, None)
    assert base.calls == 2
    assert public.logits == neutral.logits == (1., -2.)
    assert (public.value, neutral.value) == (.75, 0.)
    with pytest.raises(ValueError):
        exp.LeafModeInference(base, "unknown")


def test_real_inference_exact_priors_and_scalar_on_game_positions(models):
    env = SplendorEnv(seed=1459000100)
    state = env.initial_state()
    nonzero = 0
    for _ in range(30):
        legal = env.legal_actions(state)
        public = models["public"].predict_one(state, legal)
        neutral = models["neutral"].predict_one(state, legal)
        expected = models["public"].base.predict_one(state, legal)
        assert public == expected
        assert neutral.logits == public.logits and neutral.value == 0
        nonzero += public.value != 0
        state = env.step(state, legal[max(range(len(legal)), key=lambda i: public.logits[i])])
        if state.terminal:
            break
    assert nonzero > 0


@pytest.mark.parametrize("arm", exp.ARMS)
def test_hidden_order_and_timed_fixed_equivalence(models, arm):
    env = SplendorEnv(seed=1459000101)
    state = env.initial_state()
    altered = copy.deepcopy(state)
    for deck in altered.hidden_tier_decks.values():
        deck.reverse()
    config = SearchConfig(**exp.CONFIG["search"])
    timed = exp.BudgetSearch(models[arm], config).run(env, state, self_play=False, seed=93, seconds=.02)
    fixed_config = replace(config, simulations=timed.simulations)
    fixed = AlphaZeroSearch(models[arm], fixed_config).run(env, altered, self_play=False, seed=93)
    assert exp.search_record(timed) == exp.search_record(fixed)
    assert timed.action == fixed.action


@pytest.mark.parametrize("arm", exp.ARMS)
@pytest.mark.parametrize("winner,expected", [(None, 0.), (0, -1.), (1, 1.)])
def test_both_arms_recognize_terminal_values_before_leaf_inference(arm, winner, expected):
    class MustNotInfer:
        def predict_one(self, state, actions):
            raise AssertionError("Terminal value must bypass learned/neutral inference")

    env = SplendorEnv(seed=1459000102)
    state = env.initial_state()
    state.current_player, state.terminal, state.winner = 1, True, winner
    search = exp.BudgetSearch(exp.LeafModeInference(MustNotInfer(), arm))
    value, path = search._select_and_evaluate(env, state, root_player=0, search_seed=0, simulation=0)
    assert value == expected and path == []


def fake_games():
    return [{"index": i, "seed": exp.schedule(i)[0], "public_seat": i%2, "probe": False,
             "winner": 0, "public_points": float(i%2 == 0), "valid_natural": True,
             "termination_reason": "completed", "final_scores": [15, 12], "turns": 60,
             "trace": [{"arm": arm, "player": i%2 if arm == "public" else 1-i%2,
                        "decision_seconds": .402, "search_seconds": .401,
                        "cpu_seconds": .3, "simulations": 64, "max_depth": 3} for arm in exp.ARMS]}
            for i in range(800)]


def test_whole_pair_bootstrap_and_entire_summary_serialization():
    result = exp.summarize(fake_games())
    assert result["public_score_share"] == .5 and result["paired_ci95"] == [.5, .5]
    assert result["decision"] == "value_benefit_not_established"
    assert result["mean_decision_time_ratio"] == 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    independent = runpy.run_path("scripts/analysis/verify_public_value_ablation_145.py")["independent_primary"]
    assert independent(fake_games()) == (.5, [.5, .5])


@pytest.mark.parametrize("problem", ["seed", "duplicate", "outcome", "arm"])
def test_schedule_accounting_and_treatment_mismatch_rejected(problem):
    records = fake_games()
    if problem == "seed":
        records[1]["seed"] += 1
    elif problem == "duplicate":
        records[1]["index"] = 0
    elif problem == "outcome":
        records[1]["public_points"] = .5
    else:
        records[1]["trace"][0]["arm"] = "neutral"
    with pytest.raises(ValueError):
        exp.summarize(records)


@pytest.mark.parametrize("problem", ["cutoff", "public_budget", "neutral_budget", "time_ratio"])
def test_any_integrity_or_either_arm_budget_failure_blocks_success(problem):
    records = fake_games()
    for r in records:
        r["winner"], r["public_points"] = r["public_seat"], 1.
    if problem == "cutoff":
        records[0].update(valid_natural=False, winner=None, public_points=.5)
    else:
        for r in records:
            if problem == "time_ratio":
                r["trace"][0]["decision_seconds"] = .43  # Both arm budgets pass; ratio fails.
            else:
                index = 0 if problem == "public_budget" else 1
                r["trace"][index]["decision_seconds"] = .6
    result = exp.summarize(records)
    assert result["decision"] == "invalid_integrity_or_time_match"
    assert not result["practical_55_percent_ruled_out"]
    json.dumps(result, allow_nan=False)


def test_decision_boundaries_and_seed_pair_contract():
    assert exp.decision_label(.55, [.501, .599], True) == "material_public_value_benefit"
    assert exp.decision_label(.54, [.501, .579], True) == "positive_value_benefit_below_material_threshold"
    assert exp.decision_label(.44, [.39, .49], True) == "public_value_worse_than_neutral"
    assert exp.decision_label(.56, [.50, .62], True) == "value_benefit_not_established"
    assert len({exp.schedule(i)[0] for i in range(800)}) == 400
    assert exp.schedule(0) == (1450000000, 0) and exp.schedule(799) == (1450000399, 1)
    with pytest.raises(ValueError):
        exp.schedule(800)


def test_cutoff_kept_and_both_arm_audit_detects_tampering(tmp_path, monkeypatch, models):
    monkeypatch.setitem(exp.CONFIG, "max_turns", 4)
    monkeypatch.setitem(exp.CONFIG, "decision_seconds", .01)
    exp.play_one(0, "test", tmp_path, probe=True)
    path = tmp_path / "games/game_0000.json"
    record = exp.read_json(path)
    assert not record["valid_natural"] and record["winner"] is None and record["public_points"] == .5
    audit = runpy.run_path("scripts/analysis/verify_public_value_ablation_145.py")["audit_game"]
    result = audit(path, "test", probe=True)
    assert result["roots_audited"] == result["searches_reproduced"] == {"public": 2, "neutral": 2}
    record["trace"][1]["raw_network_value"] = .987
    exp.atomic_write(path, record)
    with pytest.raises(AssertionError):
        audit(path, "test", probe=True)
