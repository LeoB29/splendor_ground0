import copy
import json

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import run_value_representation as prior
from splendor_ai.alphazero import run_value_training_exposure as exp
from splendor_ai.alphazero import run_value_training_exposure_arena as arena
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def toy_data():
    training = exp.Data([0], torch.zeros(2, 295), torch.zeros(2, 15), torch.zeros(2),
                        torch.tensor([0, 1]), torch.tensor([2]), torch.tensor([0, 2]))
    validation = copy.deepcopy(training)
    validation.ids = [4096]
    return training, validation


def test_recipe_except_exposure_and_test_namespace_is_preserved():
    changed = {k for k in prior.CONFIG if k in exp.CONFIG and prior.CONFIG[k] != exp.CONFIG[k]}
    assert changed == {"max_updates", "test_seed_start", "probe_seed", "bootstrap_seed"}
    assert exp.CONFIG["max_updates"] == 16384 and exp.CONFIG["training_seed"] == 146010
    assert exp.CONFIG["early_stopping"] is False and exp.CONFIG["prefix_updates"] == 640
    for flag in (False, True):
        a, b = prior.ResidualValue(flag), exp.ResidualValue(flag)
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_zero_inference_remains_exact_in_nonzero_public_positions():
    env = SplendorEnv(seed=1_479_900_010)
    state = env.initial_state()
    base = exp.PublicValueInference()
    models = [exp.ResidualInference(exp.ResidualValue(flag), base) for flag in (False, True)]
    nonzero = 0
    for _ in range(12):
        legal = env.legal_actions(state)
        prediction = base.predict_one(state, legal)
        assert all(model.predict_one(state, legal) == prediction for model in models)
        nonzero += prediction.value != 0
        state = env.step(state, legal[0])
    assert nonzero > 0


def test_fit_runs_full_budget_when_validation_stays_flat(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "max_updates", 8)
    monkeypatch.setitem(exp.CONFIG, "validation_interval", 1)
    model, report, terminal = exp.fit("trainable", *toy_data(), enforce_prefix=False)
    assert report["completed_updates"] == 8 and report["selected_step"] == 0
    assert len(report["history"]) == 9
    assert report["final_coverage"] == {"samples": 1024, "unique_positions": 2, "unique_games": 1}
    assert report["selected_coverage"]["samples"] == 0
    assert any(not torch.equal(v, terminal[k]) for k, v in model.state_dict().items())


def test_later_material_validation_improvement_can_be_selected(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "max_updates", 6)
    monkeypatch.setitem(exp.CONFIG, "validation_interval", 1)
    losses = iter([1., 1., 1., 1., 1., .9, .8995])
    monkeypatch.setattr(exp, "validation_loss", lambda *args: next(losses, .9))
    _, report, _ = exp.fit("frozen", *toy_data(), enforce_prefix=False)
    assert report["completed_updates"] == 6 and report["selected_step"] == 5
    assert not report["history"][-1]["selected"]  # Preserve the .001 threshold.


def test_production_fit_requires_prefix_check(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "max_updates", 1)
    monkeypatch.setitem(exp.CONFIG, "validation_interval", 1)
    with pytest.raises(ValueError, match="prefix"):
        exp.fit("frozen", *toy_data())


def test_146_prefix_identity_checks_exact_selected_tensors():
    for name in ("frozen", "trainable"):
        report = exp.read_json(exp.SHORT_ROOT / f"{name}_training.json")
        state = exp.load_short(name).state_dict()
        assert exp.verify_prefix(name, report["history"], state, 128)["passed"]
        wrong = copy.deepcopy(state)
        wrong["head.2.bias"][0] += .01
        with pytest.raises(ValueError, match="checkpoint"):
            exp.verify_prefix(name, report["history"], wrong, 128)
        wrong_history = copy.deepcopy(report["history"])
        wrong_history[1]["validation_game_mse"] += .001
        with pytest.raises(ValueError, match="history"):
            exp.verify_prefix(name, wrong_history, state, 128)


@pytest.mark.parametrize("ids", [[4608], [-1], [], [0, 5119]])
def test_consumed_protected_ids_are_excluded(ids):
    with pytest.raises(ValueError, match="development"):
        exp.load_development(ids)


def gate_stats():
    return {phase: {"games": 512, "contrasts": {
        "trainable_long_minus_"+ref: {"relative_reduction": .04, "ci95": [-.05, -.02]}
        for ref in exp.PRIMARY_REFERENCES}} for phase in exp.PHASES}


def test_gate_passes_and_serializes():
    result = exp.prediction_gate(gate_stats(), 512, {"passed": np.bool_(True)})
    assert result["passed"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failure", ["public_margin", "frozen_margin", "uncertainty", "no_exposure_gain",
                                     "late_public", "late_short", "overall_frozen", "invalid", "coverage", "cost"])
def test_each_required_gate_blocks_gameplay(failure):
    stats, valid, cost = gate_stats(), 512, {"passed": True}
    primary = stats["early_middle"]["contrasts"]
    if failure == "public_margin":
        primary["trainable_long_minus_public"]["relative_reduction"] = .029
    elif failure == "frozen_margin":
        primary["trainable_long_minus_frozen_long"]["relative_reduction"] = .009
    elif failure == "uncertainty":
        primary["trainable_long_minus_public"]["ci95"][1] = 0.
    elif failure == "no_exposure_gain":
        primary["trainable_long_minus_trainable_short"]["ci95"][1] = 0.
    elif failure == "late_public":
        stats["late"]["contrasts"]["trainable_long_minus_public"]["ci95"][1] = .01
    elif failure == "late_short":
        stats["late"]["contrasts"]["trainable_long_minus_trainable_short"]["ci95"][1] = .01
    elif failure == "overall_frozen":
        stats["overall"]["contrasts"]["trainable_long_minus_frozen_long"]["ci95"][1] = .01
    elif failure == "invalid":
        valid = 511
    elif failure == "coverage":
        stats["late"]["games"] = 511
    else:
        cost["passed"] = False
    assert not exp.prediction_gate(stats, valid, cost)["passed"]


def test_six_models_and_factorial_contrasts_use_paired_games(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "bootstrap_samples", 1000)
    result = exp.statistics({"early_middle": [[1., .9, .89, .88, .87, .80]]*8})["early_middle"]
    contrasts = result["contrasts"]
    assert set(result["mse"]) == set(exp.NAMES)
    assert np.allclose(contrasts["trainable_long_minus_trainable_short"]["ci95"], [-.08, -.08])
    assert np.allclose(contrasts["frozen_long_minus_frozen_short"]["ci95"], [-.02, -.02])
    assert np.allclose(contrasts["exposure_by_trainability_interaction"]["ci95"], [-.06, -.06])
    json.dumps(result, allow_nan=False)


def test_selected_long_model_can_equal_short_control(monkeypatch):
    monkeypatch.setattr(exp, "load_value", exp.load_short)
    models = exp.evaluation_models()
    assert len(models) == 4
    for a, b in ((0, 2), (1, 3)):
        assert all(torch.equal(v, models[b].state_dict()[k]) for k, v in models[a].state_dict().items())


def test_short_control_tampering_breaks_lock(tmp_path, monkeypatch):
    output, short = tmp_path / "new", tmp_path / "short"
    output.mkdir()
    short.mkdir()
    path = short / "frozen.pt"
    path.write_bytes(b"original")
    monkeypatch.setattr(exp, "OUTPUT", output)
    monkeypatch.setattr(exp, "SHORT_ROOT", short)
    exp.atomic_write(output / "selection_lock.json", {"manifest": exp.digest({}), "files": {},
        "short_controls": {"frozen.pt": exp.file_hash(path)}})
    assert exp.check_lock({})
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="short control"):
        exp.check_lock({})


def test_arena_schedule_and_failed_gate_guard(monkeypatch):
    assert arena.schedule(0) == (1_471_000_000, 0)
    assert arena.schedule(799) == (1_471_000_399, 1)
    with pytest.raises(ValueError):
        arena.schedule(800)
    assert exp.CONFIG["test_seed_start"]+512 < arena.CONFIG["seed_start"]
    monkeypatch.setattr(exp, "check_lock", lambda manifest: None)
    monkeypatch.setattr(arena, "read_json", lambda path: {"gate": {"passed": False}, "passed": True})
    with pytest.raises(ValueError, match="prediction gate"):
        arena.run({})
