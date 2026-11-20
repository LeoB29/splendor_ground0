"""Whole-game generalization, unchanged-policy and sealed-test invariants."""
import copy
from pathlib import Path
import runpy

import numpy as np
import pytest
import torch

from splendor_ai.alphazero import run_value_generalization_scale as exp
from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from splendor_ai.engine.env import SplendorEnv


@pytest.fixture(scope="module")
def inference():
    torch.set_num_threads(1)
    return StrongPolicyInference(exp.CHECKPOINT)


def toy_data(lengths=(1, 3), ids=None):
    n = torch.tensor(lengths)
    total = sum(lengths)
    targets = torch.tensor([float(2*(i % 2)-1) for i in range(total)])
    simple = torch.zeros(total, 15)
    simple[:, 0] = targets
    return exp.ValueData(list(range(len(n))) if ids is None else ids, torch.zeros(total, 256), simple,
                         targets, n, torch.cat((torch.zeros(1, dtype=torch.long), n.cumsum(0)[:-1])))


def test_fixed_schedule_and_nested_small_corpus():
    small = exp.small_indices()
    assert len(small) == len(set(small)) == 256
    assert set(small) <= set(range(4096))
    assert small == exp.small_indices()
    assert all(exp.split_for(i) == "training" for i in small)
    assert [exp.split_for(i) for i in (4095, 4096, 4607, 4608, 5119)] == ["training", "validation", "validation", "test", "test"]
    with pytest.raises(ValueError):
        exp.split_for(5120)


def test_primary_metric_weights_games_equally():
    data = toy_data()
    predictions = data.y.clone()
    predictions[1:] += 1
    assert float(exp.game_mse(predictions, data)) == pytest.approx(.5)
    assert float((predictions-data.y).square().mean()) == pytest.approx(.75)


def test_minibatches_sample_games_before_positions():
    data = toy_data((1, 100))
    sampled = exp.sample_batch_indices(data, torch.Generator().manual_seed(19), 10000)
    assert .48 < float((sampled == 0).float().mean()) < .52
    assert sampled.min() >= 0 and sampled.max() < 101


def test_public_baseline_is_antisymmetric_and_ignores_hidden_decks():
    state = SplendorEnv(seed=1439000010).initial_state()
    state.players[0].score = 7
    state.players[1].tokens["gold"] = 2
    before = exp.simple_features(state)
    altered = copy.deepcopy(state)
    for deck in altered.hidden_tier_decks.values():
        deck.reverse()
    assert torch.equal(exp.simple_features(altered), before)
    altered.players.reverse()
    assert torch.equal(exp.simple_features(altered), -before)
    assert len(before) == 15


def test_selection_retains_zero_and_requires_prespecified_improvement():
    assert exp.update_selection(1.1, 1., 0) == (False, 1., 1)
    assert exp.update_selection(.9995, 1., 3) == (False, 1., 4)
    assert exp.update_selection(.99, 1., 3) == (True, .99, 0)


def test_pairing_of_model_errors_is_preserved_in_bootstrap(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "bootstrap_samples", 2000)
    data = np.array([[1., 1., .9, .8], [1., .4, .3, .5], [1., .7, .6, .8]])
    stats = exp.summarize_game_errors(data)
    assert stats["contrasts"]["large_minus_small"]["ci95"] == pytest.approx([-.1, -.1])


def test_neutral_result_cannot_be_claimed_as_improvement(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "bootstrap_samples", 1000)
    stats = exp.summarize_game_errors(np.ones((10, 4)))
    assert not stats["large_beats_neutral"]
    assert exp.classify_result(stats, 512) == "value_generalization_not_established"


def test_invalid_test_completion_overrides_positive_statistics(monkeypatch):
    monkeypatch.setitem(exp.CONFIG, "bootstrap_samples", 1000)
    stats = exp.summarize_game_errors(np.tile([1., .9, .7, .8], (10, 1)))
    assert stats["large_material_improvement"]
    assert exp.classify_result(stats, 511) == "invalid_test_completion"
    assert exp.classify_result(stats, 512) == "material_value_improvement_and_scale_benefit"


def test_ridge_is_selected_on_validation_and_predicts_bounded_values(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    data = toy_data((10, 10))
    result = exp.fit_simple(data, data, {"test": True})
    assert result["selected_index"] == min(range(5), key=lambda i: result["candidates"][i]["validation_game_mse"])
    assert result["validation_game_mse"] < .01
    assert len(result["coefficients"]) == 15


def test_training_saves_diagnostics_and_preserves_all_policy_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    for key, value in {"max_updates": 8, "validation_interval": 2, "patience": 2, "batch_size": 4}.items():
        monkeypatch.setitem(exp.CONFIG, key, value)
    report = exp.train_head("small", toy_data(), toy_data(ids=[4096, 4097]), {"test": True})
    assert report["status"] == "complete"
    assert report["completed_updates"] <= 8
    assert report["initial_policy_sha256"] == report["final_policy_sha256"]
    assert report["trainable_parameters"] == 33025
    assert (tmp_path / "small_training.json").exists()
    assert (tmp_path / "small.pt").exists()
    assert report["selected_validation_game_mse"] <= report["history"][0]["validation_game_mse"]


def prepare_fake_lock_files(root):
    for name in ("small.pt", "large.pt", "small_training.json", "large_training.json", "simple.json", "development_summary.json"):
        (root / name).write_text("sealed", encoding="utf-8")


def test_test_data_must_not_exist_before_selection_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    manifest = {"test": True}
    monkeypatch.setattr(exp, "identity", lambda: manifest)
    prepare_fake_lock_files(tmp_path)
    (tmp_path / "games").mkdir()
    (tmp_path / "games/game_4608.json").write_text("forbidden", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Protected games exist"):
        exp.seal_selection(manifest)


def test_selection_lock_detects_model_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    manifest = {"test": True}
    monkeypatch.setattr(exp, "identity", lambda: manifest)
    prepare_fake_lock_files(tmp_path)
    exp.seal_selection(manifest)
    exp.check_selection_lock(manifest)
    (tmp_path / "large.pt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Selected model"):
        exp.check_selection_lock(manifest)


def test_complete_probe_game_reconstructs_035_behavior_and_all_value_inputs(tmp_path, monkeypatch, inference):
    monkeypatch.setattr(exp, "_INFERENCE", inference)
    exp.collect_one(0, "probe", root=tmp_path, seed_override=1439000020)
    audit = runpy.run_path("scripts/analysis/verify_value_generalization_scale_143.py")
    audit["initialize"]()
    result = audit["audit_one"](0, tmp_path, protected=False)
    assert result["valid"] and result["positions"] > 10
    path = tmp_path / "games/game_0000.json"
    record = exp.read_json(path)
    record["final_state"]["turn_index"] += 1
    exp.atomic_write(path, record)
    with pytest.raises(AssertionError):
        audit["audit_one"](0, tmp_path, protected=False)


def test_cutoff_games_have_no_fabricated_training_labels(tmp_path, monkeypatch, inference):
    monkeypatch.setattr(exp, "_INFERENCE", inference)
    monkeypatch.setitem(exp.CONFIG, "max_turns", 1)
    exp.collect_one(0, "probe", root=tmp_path, seed_override=1439000021)
    record = exp.read_json(tmp_path / "games/game_0000.json")
    data = torch.load(tmp_path / "games/game_0000.pt", weights_only=True)
    assert not record["valid_natural"] and record["winner"] is None
    assert data["targets"] is None


def test_saved_replay_tensors_are_not_overwritten(tmp_path):
    path = tmp_path / "data.pt"
    exp.tensor_save(path, {"x": torch.tensor([1.])})
    with pytest.raises(FileExistsError):
        exp.tensor_save(path, {"x": torch.tensor([2.])})
    assert torch.load(path, weights_only=True)["x"].item() == 1.
