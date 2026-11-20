import importlib.util
import random

import pytest
import torch

from splendor_ai.alphazero import run_public_feature_advantage as p
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.engine.actions import ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_DECK


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def audit():
    return module("audit153_test", "scripts/analysis/verify_public_feature_advantage_153.py")


@pytest.fixture(scope="module")
def helpers():
    return module("helpers149_test153", "tests/test_afterstate_advantage_149.py")


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("actor", (0, 1))
def test_public_effects_match_all_actions_including_nobles_payments_returns(audit, helpers, index, actor):
    state = helpers.positions()[index]
    if actor:
        state.players.reverse()
        state.current_player = 1
        state.start_player = 1-state.start_player
    before = p.old.serialize_state_snapshot(state)
    for action in SplendorEnv().legal_actions(state):
        phi = p.public_features(state, action)
        assert torch.equal(phi, torch.tensor(audit.public_effects(state, action), dtype=torch.float64))
        assert p.basis(phi).tolist() == audit.polynomial(phi.tolist())
    assert p.old.serialize_state_snapshot(state) == before


def test_private_pending_counts_and_hidden_invariance(audit):
    state = SplendorEnv(seed=153).initial_state()
    state.current_player = 1
    state.players[0].reserve_card(state.hidden_tier_decks[2].pop(), RESERVE_SOURCE_DECK)
    state.deck_counts[2] -= 1
    action = next(a for a in SplendorEnv().legal_actions(state) if a.action_type == ActionType.RESERVE_DECK)
    expected = p.public_features(state, action)
    assert expected[11] == .2 and expected[13] == 0.  # Both seats now have one reserve.
    for seed in range(4):
        world = determinize_for_observer(state, 1, random.Random(seed))
        assert torch.equal(expected, p.public_features(world, action))
        assert expected.tolist() == audit.public_effects(world, action)


def synthetic(n=31, width=135):
    rng = torch.Generator().manual_seed(153)
    x = torch.randn(n, 4, width, generator=rng, dtype=torch.float64)*.1
    x -= x[:, :1].clone()
    mask = torch.tensor([[False, True, True, True]]*n)
    mask[0] = False
    mask[1, 2:] = False
    x[0] = 0.
    x[1, 2:] = 0.
    q = torch.randn(n, 4, generator=rng, dtype=torch.float64)*.2
    y = q-q[:, :1]
    return {"x": x, "y": y, "q": q, "mask": mask}


def test_state_weighting_ridge_scale_and_baseline_only_denominator(audit):
    # Two states: first has two alternate observations x=y=1, second has none.
    # Objective .5*(w-1)^2 + .5*w^2 has unique solution w=.5.
    data = {"x": torch.tensor([[[0.], [1.], [1.], [90.]], [[0.], [80.], [80.], [80.]]], dtype=torch.float64),
            "y": torch.tensor([[0., 1., 1., 70.], [0., 60., 60., 60.]], dtype=torch.float64),
            "mask": torch.tensor([[False, True, True, False], [False, False, False, False]])}
    main, _ = p.solve(data, .5)
    other, _ = audit.independent_solve(data, .5)
    assert float(main) == .5
    torch.testing.assert_close(main, other, atol=1e-12, rtol=0)
    assert p.losses(torch.zeros_like(data["y"]), data["y"], data["mask"]).tolist() == [1., 0.]


def test_folds_group_games_balance_cells_and_ignore_outcomes(audit):
    samples = [{"source_seed": i+10000, "cell": i%6, "snapshot": {"a": i}} for i in range(72)]
    rows = p.fold_map(samples)
    assert rows == audit.independent_folds(samples)
    reversed_rows = p.fold_map(list(reversed(samples)))
    assert {r["source_seed"]: r["fold"] for r in rows} == {r["source_seed"]: r["fold"] for r in reversed_rows}
    for cell in range(6):
        counts = [sum(r["fold"] == f and r["cell"] == cell for r in rows) for f in range(5)]
        assert max(counts)-min(counts) == 1
    with pytest.raises(ValueError, match="source game"):
        p.fold_map(samples+[samples[0]])


def test_all_26_solves_and_oof_metric_reproduce(audit):
    torch.set_num_threads(1)
    data = synthetic()
    folds = [{"fold": i%5} for i in range(len(data["y"]))]
    result = p.fit_cv(data, folds)
    assert len(result["fits"]) == 26
    for saved in result["fits"]:
        subset = data if saved["fold"] == "all" else p.subset(data, torch.tensor([r["fold"] != saved["fold"] for r in folds]))
        w, _ = audit.independent_solve(subset, saved["penalty"])
        torch.testing.assert_close(w, saved["coefficients"], atol=1e-9, rtol=1e-8)
    for row, pred in zip(result["cv"], result["oof_predictions"]):
        expected = audit.independent_metric(data, {"advantages": pred.tolist(), "choices": [0]*len(pred)})["mse"]
        assert abs(expected-row["oof_mse"]) < 1e-12
    assert p.choose_penalty([{"penalty": .1, "oof_mse": .5}, {"penalty": 1., "oof_mse": .5}]) == 1.


def test_baseline_ties_padding_no_clipping_and_independent_prediction(audit):
    data = {"x": torch.tensor([[[0.], [0.], [4.], [7.]], [[0.], [3.], [0.], [0.]]], dtype=torch.float64),
            "mask": torch.tensor([[False, True, False, False], [False, True, False, False]])}
    w = torch.ones(1, dtype=torch.float64)
    row = p.prediction(data, w)
    assert row == audit.independent_prediction(data, w)
    assert row["choices"] == [0, 1] and row["advantages"][1][1] == 3.


def test_collision_counts_keep_baseline_only_and_aliases(audit):
    data = {"phi": torch.tensor([[[0.]*15, [0.]*15, [1.]*15, [0.]*15], [[0.]*15]*4], dtype=torch.float64),
            "mask": torch.tensor([[False, True, True, False], [False, False, False, False]]), "y": torch.zeros(2, 4)}
    result = p.collisions(data)
    assert result == audit.independent_collisions(data)
    assert result["aliased_pairs"] == result["zero_difference_alternates"] == result["baseline_only_states"] == 1
    assert result["real_pairs"] == 3 and result["real_alternates"] == 2


@pytest.mark.parametrize("gate", ("mse_versus_C", "mse_versus_zero", "wdl_versus_035", "wdl_versus_C"))
def test_each_gate_required(gate):
    treatment, control, zero = {"mse": .05, "wdl_gain_over_035": .03}, {"mse": .1, "wdl_gain_over_035": .01}, .1
    if gate == "mse_versus_C": control["mse"] = .05
    elif gate == "mse_versus_zero": zero = .05
    elif gate == "wdl_versus_035": treatment["wdl_gain_over_035"] = .019
    else: control["wdl_gain_over_035"] = .03
    result = p.gates(treatment, control, zero)
    assert not result["development_passed"]
    assert [k for k, v in result["development_gates"].items() if not v] == [gate]


def test_target_arithmetic_first_eight_and_tampering(audit, tmp_path):
    sample = {"index": 0, "actor": 1, "actions": [{}, {}]}
    worlds = [{"seed": 1490100000+i, "outcomes": [
        {"winner": None, "wdl": 0., "reason": "completed"},
        {"winner": 1 if i < 4 or i >= 8 else 0, "wdl": 1. if i < 4 or i >= 8 else -1., "reason": "completed"}]} for i in range(32)]
    path = tmp_path/"targets/train/state_0000.json"
    p.old.save_new(path, {"split": "train", "index": 0, "sample_sha256": p.old.digest(sample), "worlds": worlds})
    lock = {"files": {"0": {"sha256": p.old.file_hash(path)}}}
    assert audit.target_row(tmp_path, "train", sample, lock, 8) == [0.]*4
    sample["actor"] = 0
    with pytest.raises(AssertionError):
        audit.target_row(tmp_path, "train", sample, lock, 8)


def test_failure_still_audited_then_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "OUTPUT", tmp_path/"153")
    monkeypatch.setattr(p.parent, "OUTPUT", tmp_path/"152")
    monkeypatch.setattr(p, "check_identity", lambda: "frozen")
    data = synthetic(31)
    data["phi"] = torch.zeros(31, 4, 15, dtype=torch.float64)
    monkeypatch.setattr(p, "load_data", lambda split: data)
    p.old.save_new(p.OUTPUT/"fold_map.json", [{"fold": i%5} for i in range(31)])
    p.old.save_new(p.OUTPUT/"freeze.json", {"manifest": "frozen", "fold_map_sha256": p.old.file_hash(p.OUTPUT/"fold_map.json")})
    p.old.save_new(p.OUTPUT/"data_lock.json", {"freeze_sha256": p.old.file_hash(p.OUTPUT/"freeze.json")})
    p.old.save_new(p.OUTPUT/"preflight_audit.json", {"passed": True, "manifest": "frozen", "data_lock_sha256": p.old.file_hash(p.OUTPUT/"data_lock.json")})
    control = {"advantages": torch.zeros_like(data["y"]).tolist(), "choices": [0]*31}
    p.old.save_new(p.parent.OUTPUT/"validation_predictions.json", {"C": [{"seed": s, "selected_update": 1, **control} for s in (149001,149002,149003)]})
    monkeypatch.setattr(p, "gates", lambda *args: {"development_passed": False})
    called = []

    def audit_stage(stage):
        called.append(stage)
        assert not p.old.read_json(p.OUTPUT/"result.json")["development_passed"]
        value = {f"{field}_sha256": p.old.file_hash(p.OUTPUT/name) for field, name in (
            ("result", "result.json"), ("selection", "selection_lock.json"), ("predictions", "validation_predictions.json"), ("training_predictions", "training_predictions.json"))}
        p.old.save_new(p.OUTPUT/"final_audit.json", value)
        return value

    monkeypatch.setattr(p, "audit_stage", audit_stage)
    p.run()
    final = p.old.read_json(p.OUTPUT/"final_status.json")
    assert called == ["final"] and final["complete"] and not final["development_passed"]
    assert not any(final[k] for k in ("fresh_performance_test", "arena", "promotion", "automatic_successor", "new_simulations", "neural_refits"))


def test_no_protected_split_or_overwrite(tmp_path):
    with pytest.raises(ValueError, match="no protected"):
        p.load_data("test")
    path = tmp_path/"immutable.json"
    p.old.save_new(path, {"a": 1})
    with pytest.raises(FileExistsError):
        p.old.save_new(path, {"a": 2})
