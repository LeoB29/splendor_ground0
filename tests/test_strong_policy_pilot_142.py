"""Scientific invariants of the isolated strong-policy learning pilot."""
from dataclasses import replace
import copy
import runpy

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import strong_policy as bridge
from splendor_ai.alphazero import run_strong_policy_pilot as pilot
from splendor_ai.alphazero.network import NetworkPrediction
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.encoding import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(scope="module")
def inference():
    torch.set_num_threads(1)
    return bridge.StrongPolicyInference(pilot.CHECKPOINT)


def test_original_logits_and_greedy_action_exactly_retained(inference):
    original = LegalActionAttentionScorerBot(pilot.CHECKPOINT)
    env = SplendorEnv(seed=1429000011)
    state = env.initial_state()
    for _ in range(50):
        legal = env.legal_actions(state)
        scores, _, _, _, _ = inference.full_prediction(state, legal)
        assert tuple(scores.tolist()) == original.score_actions(state, legal)
        assert legal[int(scores.argmax())] == original.choose_action(env, state, legal)
        state = env.step(state, legal[int(scores.argmax())])
        if state.terminal:
            break


def test_subset_search_prediction_keeps_complete_attention_context(inference):
    env = SplendorEnv(seed=1429000012)
    state = env.initial_state()
    full = env.legal_actions(state)
    scores, value, row, keys, _ = inference.full_prediction(state, full)
    grouped = bridge.grouped_logits(scores, row["groups"], row["group_count"]).tolist()
    representatives = list(actions_by_canonical_successor_key(state, full).values())
    prediction = inference.predict_one(state, representatives[:3])
    assert prediction.logits == tuple(grouped[:3])
    assert prediction.value == value


def test_hidden_deck_order_does_not_enter_model(inference):
    env = SplendorEnv(seed=1429000013)
    state = env.initial_state()
    altered = copy.deepcopy(state)
    for deck in altered.hidden_tier_decks.values():
        deck.reverse()
    left = inference.full_prediction(state)
    right = inference.full_prediction(altered)
    assert torch.equal(left[0], right[0])
    assert left[1] == right[1]
    for key in ("observation", "features", "groups"):
        assert torch.equal(left[2][key], right[2][key])


def test_canonical_max_does_not_reward_alias_multiplicity():
    scores = torch.tensor([1., 3., 2.], requires_grad=True)
    actual = bridge.grouped_logits(scores, torch.tensor([0, 0, 1]), 2)
    assert actual.tolist() == [3., 2.]
    actual.sum().backward()
    assert scores.grad.tolist() == [0., 1., 1.]


def test_value_only_learning_preserves_every_policy_parameter(inference):
    model = copy.deepcopy(inference.model)
    before = pilot.policy_hash(model)
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("value_head."))
    obs = torch.randn(4, model.config.observation_size)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss = model.value_head(model.state_encoder(obs)).square().mean()
    loss.backward()
    optimizer.step()
    assert before == pilot.policy_hash(model)


class Uniform:
    def predict_one(self, state, actions):
        return NetworkPrediction(tuple(0. for _ in actions), .2)


def test_timed_search_matches_fixed_search_at_same_simulation_count(monkeypatch):
    env = SplendorEnv(seed=1429000014)
    state = env.initial_state()
    config = SearchConfig(simulations=2, root_noise_fraction=0)
    fixed = AlphaZeroSearch(Uniform(), config).run(env, state, self_play=False, seed=77)
    clock = iter([0., .25, .5, .75])
    monkeypatch.setattr(bridge.time, "perf_counter", lambda: next(clock))
    timed = bridge.BudgetSearch(Uniform(), config).run(env, state, self_play=False, seed=77, seconds=.5)
    assert timed.simulations == 2
    assert timed.root_edges == fixed.root_edges
    assert timed.action == fixed.action
    assert timed.search_root_value == fixed.search_root_value


def test_expired_budget_still_completes_one_simulation(monkeypatch):
    env = SplendorEnv(seed=1429000015)
    clock = iter([0., 1., 2.])
    monkeypatch.setattr(bridge.time, "perf_counter", lambda: next(clock))
    result = bridge.BudgetSearch(Uniform()).run(env, env.initial_state(), self_play=False, seconds=.1)
    assert result.simulations == sum(e.visits for e in result.root_edges) == 1


@pytest.mark.parametrize("seconds", [0, -1])
def test_invalid_time_budget_rejected(seconds):
    env = SplendorEnv(seed=1)
    with pytest.raises(ValueError):
        bridge.BudgetSearch(Uniform()).run(env, env.initial_state(), self_play=False, seconds=seconds)


def test_game_split_is_disjoint_complete_and_reproducible():
    train, val = pilot.split_games(128, 142)
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(128))
    assert (train, val) == pilot.split_games(128, 142)
    assert len(val) == 26


def fake_arena():
    return [{"index": i, "seed": 100 + i//2, "candidate_seat": i % 2,
             "candidate_points": float(i % 2), "valid_natural": True,
             "trace": [{"model_index": p, "decision_seconds": .401, "cpu_seconds": .3,
                        "simulations": 100, "max_depth": 5} for p in (0, 1)]}
            for i in range(400)]


def test_bootstrap_resamples_seed_pairs_not_individual_games():
    summary = pilot.arena_summary(fake_arena())
    assert summary["candidate_score_share"] == .5
    assert summary["paired_ci95"] == [.5, .5]
    assert summary["decision"] == "improvement_not_established"


def test_seat_pair_mismatch_rejected():
    games = fake_arena()
    games[1]["seed"] += 1
    with pytest.raises(ValueError):
        pilot.arena_summary(games)


@pytest.mark.parametrize("failure", ["cutoff", "time"])
def test_integrity_failure_cannot_be_called_a_win(failure):
    games = fake_arena()
    for game in games:
        game["candidate_points"] = 1.
    if failure == "cutoff":
        games[0]["valid_natural"] = False
    else:
        for game in games:
            game["trace"][1]["decision_seconds"] = .5
    assert pilot.arena_summary(games)["decision"] == "invalid_integrity_or_time_match"


def test_cutoff_game_is_retained_without_fake_outcome_labels(tmp_path, monkeypatch, inference):
    monkeypatch.setattr(pilot, "_MODELS", [inference])
    monkeypatch.setitem(pilot.CONFIG, "max_turns", 1)
    pilot.play_game("bootstrap", 0, "test", tmp_path)
    record = pilot.read_json(tmp_path / "bootstrap/game_0000.json")
    data = torch.load(tmp_path / "bootstrap/game_0000.pt", weights_only=True)
    assert not record["valid_natural"]
    assert record["winner"] is None
    assert data["rows"][0]["value_target"] is None


def test_joint_loss_padding_and_gradient_are_finite(inference):
    model = copy.deepcopy(inference.model)
    env = SplendorEnv(seed=1429000016)
    state = env.initial_state()
    rows = []
    for _ in range(3):
        row, _, actions = bridge.encode_position(state, model.config)
        row["policy_target"] = torch.full((row["group_count"],), 1 / row["group_count"])
        row["value_target"] = 1.
        rows.append(row)
        state = env.step(state, actions[0])
    loss, policy, value = pilot.joint_loss(model, rows)
    assert all(torch.isfinite(x) for x in (loss, policy, value))
    loss.backward()
    assert model.state_encoder[0].weight.grad is not None
    assert model.score_head[0].weight.grad is not None
    assert model.value_head[0].weight.grad is not None


def test_independent_audit_handles_json_key_roundtrip_and_rejects_changed_state(tmp_path, monkeypatch, inference):
    monkeypatch.setattr(pilot, "_MODELS", [inference])
    pilot.play_game("bootstrap", 910, "test", tmp_path)
    audit = runpy.run_path("scripts/analysis/verify_strong_policy_pilot_142.py")["audit_game"]
    path = tmp_path / "bootstrap/game_0910.json"
    record = audit(path, pilot.CONFIG, inference.model.config)
    record["final_state"]["turn_index"] += 1
    pilot.atomic_write(path, record)
    with pytest.raises(AssertionError):
        audit(path, pilot.CONFIG, inference.model.config)
