import copy
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import random

import pytest
import torch

from splendor_ai.alphazero import run_shared_cards as p
from splendor_ai.alphazero.shared_cards_151 import catalog_tensor, reconstruct, restructure, SharedCardModel
from splendor_ai.alphazero.afterstate_149 import features
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK
from splendor_ai.training.replay import serialize_state_snapshot


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return load_module("audit151_test", "scripts/analysis/verify_shared_cards_151.py")


@pytest.fixture(scope="module")
def positions():
    return load_module("positions149_test", "tests/test_afterstate_advantage_149.py").positions()


def synthetic(n=4, seed=151):
    rng = torch.Generator().manual_seed(seed)
    flat = torch.randn(n, 4, 1085, generator=rng)
    q = torch.randint(-8, 9, (n, 4), generator=rng).float()/8
    mask = torch.tensor([[False, True, True, True]]*n)
    return p.structured_data({"afterstate": flat, "mask": mask, "q": q, "y": q-q[:, :1]})


def test_catalog_and_parameter_budget():
    torch.set_num_threads(1)
    assert catalog_tensor().shape == (90, 14)
    assert len({tuple(row) for row in catalog_tensor().tolist()}) == 90
    assert sum(t.numel() for t in SharedCardModel().parameters()) == 327297
    assert abs(327297/320001-1) < .10


@pytest.mark.parametrize("index", range(5))
def test_all_legal_actions_lossless_without_mutation(positions, index):
    state = positions[index]
    before = serialize_state_snapshot(state)
    for action in SplendorEnv().legal_actions(state):
        flat = features(state, action, "afterstate")
        context, cards = restructure(flat)
        assert context.shape == (455,) and cards.shape == (90, 22)
        assert torch.equal(reconstruct(context, cards), flat)
    assert before == serialize_state_snapshot(state)


def test_projection_cannot_see_hidden_identity_or_future_order(positions):
    state = positions[0].copy_shallow()
    state.players[1].reserve_card(state.hidden_tier_decks[2].pop(), RESERVE_SOURCE_DECK)
    state.deck_counts[2] -= 1
    for action in SplendorEnv().legal_actions(state):
        expected = restructure(features(state, action, "afterstate"))
        for seed in range(3):
            world = p.determinize_for_observer(state, 0, random.Random(seed))
            actual = restructure(features(world, action, "afterstate"))
            assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_token_order_invariance_baseline_zero_and_card_gradients():
    torch.set_num_threads(1)
    data = synthetic()
    model = SharedCardModel()
    before = p.forward(model, data)
    permuted = model(data["context"], data["cards"][..., torch.randperm(90), :])
    torch.testing.assert_close(before, permuted, rtol=1e-5, atol=2e-7)
    assert torch.equal(before[:, 0], torch.zeros(4))
    (before[:, 1:]-1).square().mean().backward()
    assert model.card_encoder[0].weight.grad.abs().sum() > 0
    assert model.query.weight.grad.abs().sum() > 0


def test_inverse_rejects_schema_or_action_corruption():
    data = synthetic()
    with pytest.raises(ValueError, match="schema"):
        restructure(torch.zeros(1084))
    corrupt = data["cards"].clone()
    corrupt[0, 0, 0, 21] += 1
    with pytest.raises(ValueError, match="target"):
        reconstruct(data["context"], corrupt)
    corrupt = data["cards"].flip(-2)
    with pytest.raises(ValueError, match="canonical"):
        reconstruct(data["context"], corrupt)


def test_independent_loop_matches_training_and_selected_final_tensors(audit):
    train, val = synthetic(), synthetic(3, 152)
    main = p.fit(149001, train, val, updates=8)
    other = audit.independent_fit(149001, train, val, total=8)
    audit.a150.compare(main, other)
    assert main["evaluation_rng_unchanged"]
    assert len(main["history"]) == 9
    bad = copy.deepcopy(other)
    bad["final_state"][next(iter(bad["final_state"]))].add_(1)
    with pytest.raises(AssertionError):
        audit.a150.compare(main, bad)


def test_earliest_tie_remains_initial_model():
    data = synthetic()
    data["mask"][:] = False
    result = p.fit(149002, data, data, updates=3)
    assert result["selected_update"] == 0
    assert all(row["validation_mse"] == 0 for row in result["history"])


def screen_case(n=12):
    q = torch.tensor([[0., .25, -.5, 1.]]*n)
    mask = torch.tensor([[False, True, True, False]]*n)
    rows = {arm: [{"seed": seed, "selected_update": 256,
                  "advantages": (q if arm == "shared" else torch.zeros_like(q)).tolist(),
                  "choices": [1 if arm == "shared" else 0]*n}
                 for seed in (149001, 149002, 149003)] for arm in ("flat", "shared")}
    return {"q": q, "y": q, "mask": mask}, rows


def test_independent_statistics_and_all_seeds_retained(audit):
    data, rows = screen_case()
    result = p.development_statistics(data, rows)
    assert result["development_passed"]  # late selected updates are allowed
    assert result["arms"]["shared"]["means"]["wdl_gain_over_035"] == .25
    audit.a150.compare(result, audit.independent_development_statistics(data, rows), tolerance=1e-12)
    fresh = p.fresh_statistics(data, rows)
    assert fresh["fresh_passed"]
    audit.verify_fresh_statistics(data, rows, fresh)
    rows["shared"][0]["choices"] = [0]*12
    assert p.metrics(data, rows)["arms"]["shared"]["means"]["wdl_gain_over_035"] == pytest.approx(1/6)


@pytest.mark.parametrize("criterion", ["mse_versus_flat", "mse_versus_zero", "wdl_versus_035", "wdl_versus_flat"])
def test_each_development_gate_is_required(monkeypatch, criterion):
    shared = {"mse": .05, "wdl_gain_over_035": .03}
    flat = {"mse": .1, "wdl_gain_over_035": .01}
    zero = .1
    if criterion == "mse_versus_flat":
        flat["mse"] = .05
    elif criterion == "mse_versus_zero":
        zero = .05
    elif criterion == "wdl_versus_035":
        shared["wdl_gain_over_035"] = .019
    else:
        flat["wdl_gain_over_035"] = .03
    monkeypatch.setattr(p, "metrics", lambda *args: {"zero_mse": zero, "arms": {"shared": {"means": shared}, "flat": {"means": flat}}})
    result = p.development_statistics(None, None)
    assert not result["development_passed"]
    assert [k for k, v in result["development_gates"].items() if not v] == [criterion]


def test_fresh_precision_failure_is_not_rescued(audit):
    data, rows = screen_case(12)
    data["q"][::2, 1] = -.125
    data["y"] = data["q"].clone()
    result = p.fresh_statistics(data, rows)
    assert result["contrasts"]["shared_minus_035"]["mean"] > .02
    assert result["contrasts"]["shared_minus_035"]["half_width"] > .02
    assert not result["fresh_passed"]
    audit.verify_fresh_statistics(data, rows, result)


@pytest.mark.parametrize("dev_pass,audit_pass", [(False, True), (True, False), (True, True)])
def test_run_cannot_reach_new_simulations_without_development_and_audit_pass(monkeypatch, tmp_path, dev_pass, audit_pass):
    events = []
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "check_identity", lambda: "bound")
    monkeypatch.setattr(p, "development_data", lambda: {"train": {}, "validation": {}})
    monkeypatch.setattr(p, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(p, "fit_task", lambda *args: {"synthetic": True})
    monkeypatch.setattr(p, "predictions", lambda *args: {})
    monkeypatch.setattr(p, "development_statistics", lambda *args: {"development_passed": dev_pass})
    p.old.save_new(tmp_path/"preflight.json", {"manifest": "bound"})
    p.old.save_new(tmp_path/"development_lock.json", {})

    def simulated_audit(*args, **kwargs):
        events.append("audit")
        p.old.save_new(tmp_path/"development_audit.json", {"passed": audit_pass,
            "result_sha256": p.old.file_hash(tmp_path/"development_result.json"),
            "selection_sha256": p.old.file_hash(tmp_path/"selection_lock.json"),
            "predictions_sha256": p.old.file_hash(tmp_path/"validation_predictions.json")})

    def fresh(*args):
        assert events == ["audit"] and dev_pass and audit_pass
        events.append("fresh")
        result = {"fresh_passed": False}
        p.old.save_new(tmp_path/"fresh_result.json", result)
        p.old.save_new(tmp_path/"fresh_audit.json", {"passed": True, "result_sha256": p.old.file_hash(tmp_path/"fresh_result.json")})
        return result

    monkeypatch.setattr(p.subprocess, "run", simulated_audit)
    monkeypatch.setattr(p, "conditional_fresh", fresh)
    if not audit_pass:
        with pytest.raises(ValueError, match="Development audit"):
            p.run()
        assert not (tmp_path/"final_status.json").exists()
    else:
        p.run()
        assert p.old.read_json(tmp_path/"final_status.json")["fresh_stage_ran"] == dev_pass
    assert events == (["audit", "fresh"] if dev_pass and audit_pass else ["audit"])
    assert not (tmp_path/"targets").exists()


def test_continuation_namespaces_and_budget_enforcement():
    source = set(range(1510000000, 1510000000+2048))
    test = {1511000000+s*1000+w for s in range(384) for w in range(128)}
    probe = {1519100000+s*1000+w for s in range(6) for w in range(4)}
    prior = {p.old.world_seed(split, s, w) for split in ("train", "validation", "test")
             for s in range(p.old.CONFIG["states"][split]) for w in range(p.old.CONFIG["worlds"][split])}
    assert not source & test and not source & probe and not test & probe
    assert not prior & (source | test | probe)
    with pytest.raises(ValueError, match="budget"):
        p.collect({"split": "probe", "samples": [{"index": 0, "actions": [None]*25}]})
    with pytest.raises(ValueError, match="panel"):
        p.collect({"split": "train", "samples": []})


def test_collector_matches_independent_replay_with_new_seeds(monkeypatch, audit, positions):
    class FirstLegal:
        def choose_action(self, env, state, legal=None):
            return (legal if legal is not None else env.legal_actions(state))[0]

    monkeypatch.setattr(p.old, "_RAW", FirstLegal())
    state = positions[2]
    action = next(a for a in SplendorEnv().legal_actions(state) if a.noble_id is not None)
    sample = {"index": 2, "snapshot": serialize_state_snapshot(state), "actions": [p.old.serialize_action(action)]}
    record = p.evaluate_sample("probe", sample)
    assert len(record["worlds"]) == 4
    for i, saved in enumerate(record["worlds"]):
        assert saved["seed"] == 1519102000+i
        world = p.determinize_for_observer(state, state.current_player, random.Random(saved["seed"]))
        assert saved["outcomes"][0] == audit.a149.replay_branch(world, action, state.current_player)
