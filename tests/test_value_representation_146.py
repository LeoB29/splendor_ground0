import copy
import json

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import run_value_representation as exp
from splendor_ai.alphazero import run_value_representation_arena as arena
from splendor_ai.alphazero.search import AlphaZeroSearch, SearchConfig
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def test_zero_residual_is_exact_public_inference_and_same_search():
    env = SplendorEnv(seed=1_469_900_010)
    state = env.initial_state()
    legal = env.legal_actions(state)
    public = exp.PublicValueInference()
    for flag in (False, True):
        residual = exp.ResidualInference(exp.ResidualValue(flag), public)
        assert residual.predict_one(state, legal) == public.predict_one(state, legal)
        config = SearchConfig(simulations=8, mode="real", root_noise_fraction=0., temperature=0.)
        a = AlphaZeroSearch(residual, config).run(env, state, self_play=False, seed=555)
        b = AlphaZeroSearch(public, config).run(env, state, self_play=False, seed=555)
        assert arena.search_record(a) == arena.search_record(b)


def test_trainability_is_only_initial_difference_and_policy_is_isolated():
    frozen, free = exp.ResidualValue(False), exp.ResidualValue(True)
    assert all(torch.equal(v, free.state_dict()[k]) for k, v in frozen.state_dict().items())
    source = exp.StrongPolicyInference(exp.CHECKPOINT).model
    policy_before = exp.policy_hash(source)
    observations = torch.randn(8, 295)
    simple = torch.zeros(8, 15)
    for model, should_change in ((frozen, False), (free, True)):
        before = copy.deepcopy(model.encoder.state_dict())
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            (model(observations, simple)-.5).square().mean().backward()
            optimizer.step()
        changed = any(not torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items())
        assert changed == should_change
        assert all((p.grad is not None) == should_change for p in model.encoder.parameters())
    assert exp.policy_hash(source) == policy_before


def test_zero_deployment_reuses_exact_nonzero_scalar():
    env = SplendorEnv(seed=1_469_900_011)
    state = env.initial_state()
    base = exp.PublicValueInference()
    candidate = exp.ResidualInference(exp.ResidualValue(True), base)
    nonzero = 0
    for _ in range(12):
        legal = env.legal_actions(state)
        reference = base.predict_one(state, legal)
        assert candidate.predict_one(state, legal) == reference
        nonzero += reference.value != 0.
        state = env.step(state, legal[0])
    assert nonzero > 0


def test_baseline_double_precision_and_clipping():
    model = exp.ResidualValue(False)
    simple = torch.randn(10, 15)
    obs = torch.randn(10, 295)
    with torch.no_grad():
        expected = (simple.double() @ model.coefficients).clamp(-1, 1)
        assert torch.equal(model(obs, simple), expected)
        model.head[-1].bias.fill_(20.)
        assert torch.equal(model(obs, simple), torch.ones(10, dtype=torch.float64))


@pytest.mark.parametrize("indices", [[4608], [0, 4609], [-1], []])
def test_old_protected_data_cannot_enter_training(indices):
    with pytest.raises(ValueError, match="development"):
        exp.load_development(indices)


def test_primary_is_game_balanced_and_excludes_late(monkeypatch):
    data = exp.Data([0, 1], torch.zeros(5, 295), torch.zeros(5, 15),
                    torch.tensor([1., 0., 0., 0., 1.]), torch.tensor([0, 0, 1, 2, 40]),
                    torch.tensor([1, 4]), torch.tensor([0, 1, 5]))
    monkeypatch.setattr(exp, "predictions", lambda model, d: torch.zeros(5, dtype=torch.float64))
    assert exp.validation_loss(None, data) == .5


def successful_stats():
    result = {}
    for phase in exp.PHASES:
        result[phase] = {"games": 512, "contrasts": {
            c: {"relative_reduction": .04, "ci95": [-.05, -.02]} for c in ("public", "frozen")}}
    return result


def test_gate_success_is_json_serializable():
    gate = exp.prediction_gate(successful_stats(), 512, {"passed": np.bool_(True)})
    assert gate["passed"]
    json.dumps(gate, allow_nan=False)


@pytest.mark.parametrize("failure", ["public_margin", "frozen_margin", "uncertainty", "late", "overall", "coverage", "invalid", "cost"])
def test_each_preregistered_requirement_blocks_arena(failure):
    stats, valid, cost = successful_stats(), 512, {"passed": True}
    if failure == "public_margin":
        stats["early_middle"]["contrasts"]["public"]["relative_reduction"] = .029
    elif failure == "frozen_margin":
        stats["early_middle"]["contrasts"]["frozen"]["relative_reduction"] = .009
    elif failure == "uncertainty":
        stats["early_middle"]["contrasts"]["public"]["ci95"][1] = 0.
    elif failure in ("late", "overall"):
        stats[failure]["contrasts"]["public"]["ci95"][1] = .01
    elif failure == "coverage":
        stats["late"]["games"] = 511
    elif failure == "invalid":
        valid = 511
    else:
        cost["passed"] = False
    assert not exp.prediction_gate(stats, valid, cost)["passed"]


def test_statistics_resample_games_and_preserve_paired_differences(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "bootstrap_samples", 1000)
    rows = {"early_middle": [[1., .8, .79, .75]]*8}
    actual = exp.statistics(rows)["early_middle"]
    assert actual["games"] == 8
    assert np.allclose(actual["contrasts"]["public"]["ci95"], [-.05, -.05])
    json.dumps(actual, allow_nan=False)


def test_arena_is_fresh_seat_paired_and_out_of_prediction_namespace():
    assert arena.schedule(0) == (1_461_000_000, 0)
    assert arena.schedule(1) == (1_461_000_000, 1)
    assert arena.schedule(799) == (1_461_000_399, 1)
    with pytest.raises(ValueError):
        arena.schedule(800)
    assert exp.CONFIG["test_seed_start"] + exp.CONFIG["test_games"] < arena.CONFIG["seed_start"]


def test_arena_refuses_failed_prediction_gate(monkeypatch):
    monkeypatch.setattr(exp, "check_lock", lambda manifest: None)
    monkeypatch.setattr(arena, "read_json", lambda path: {"gate": {"passed": False}, "passed": True})
    with pytest.raises(ValueError, match="prediction gate"):
        arena.run({})


def test_fit_selects_baseline_on_no_validation_improvement(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "max_updates", 4)
    monkeypatch.setitem(exp.CONFIG, "validation_interval", 1)
    monkeypatch.setitem(exp.CONFIG, "patience", 2)
    train = exp.Data([0], torch.zeros(2, 295), torch.zeros(2, 15), torch.zeros(2),
                     torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([0, 2]))
    val = copy.deepcopy(train)
    val.ids = [4096]
    model, report = exp.fit("trainable", train, val)
    assert report["selected_step"] == 0 and report["completed_updates"] == 2
    assert torch.count_nonzero(model.head[-1].weight) == 0
