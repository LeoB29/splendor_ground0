"""Model-free contracts and tiny tensor fixtures; never load EF8A or play games."""
from copy import deepcopy
import math
from pathlib import Path
import random

import pytest

from tools.splendorust_learning_diagnostic_20261106 import training as t


def test_startup_verifies_once(monkeypatch):
    calls = []
    monkeypatch.setattr(t, "_STARTUP", None)
    monkeypatch.setattr(t.guard, "verify_runtime_sources", lambda: calls.append(True) or {"ok": True})
    first = t.startup()
    first["runtime_sources"]["ok"] = False
    assert t.startup()["runtime_sources"] == {"ok": True}
    assert calls == [True]


@pytest.mark.parametrize("args", [(2, "dense", "joint"), (True, "dense", "joint"),
                                  (0, "other", "joint"), (0, "dense", "policy_only")])
def test_rejects_extra_cells(args):
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        t.fit_identity(*args)


def test_fixed_work_inventory():
    assert 4 * 3 * 4 * 2 + 4 * 3 * 4 == 144
    assert t.STEPS * t.BATCH == 21376
    assert t.STEPS * 8 + 2 + 8 == 5354
    assert t.CHECKPOINTS == (0, 334, 667, 668)
    assert t.EVALUATION_STEPS == (0, 334, 668)


def test_tensor_tree_preserves_types_and_signed_zero():
    import torch
    assert t.tensor_tree({1: 2}) != t.tensor_tree({"1": 2})
    assert t.tensor_tree([1]) != t.tensor_tree((1,))
    assert t.tensor_tree(-0.) != t.tensor_tree(0.)
    assert t.tensor_tree(torch.tensor([-0.])) != t.tensor_tree(torch.tensor([0.]))
    assert t.tensor_tree(torch.tensor([1.])) != t.tensor_tree(torch.tensor([1], dtype=torch.int32))
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        t.tensor_tree(torch.tensor([float("nan")]))


def test_rng_round_trip():
    import numpy as np
    import torch
    previous = t.rng_state()
    try:
        t.seed_rng(0)
        state = t.rng_state()
        expected = (random.random(), float(np.random.random()), float(torch.rand(())))
        t.restore_rng(state)
        assert t.tensor_tree(t.rng_state()) == t.tensor_tree(state)
        assert (random.random(), float(np.random.random()), float(torch.rand(()))) == expected
    finally:
        t.restore_rng(previous)


def fixture_model(monkeypatch):
    """Tiny two-input arithmetic fixture, not the study architecture/checkpoint."""
    import torch

    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(2, 3)
            self.policy = torch.nn.Linear(3, 81)
            self.value = torch.nn.Linear(3, 2)

    def forward(model, tokens):
        h = model.shared(tokens)
        return torch.cat((model.policy(h), model.value(h).tanh()), dim=1)

    monkeypatch.setattr(t.accepted, "tensor_forward", forward)
    model = Fixture()
    batch = {"tokens": torch.ones(32, 2), "mask": torch.ones(32, 81, dtype=torch.bool),
             "policy": torch.full((32, 81), 1 / 81.), "value": torch.tensor([[1., -1.]] * 32),
             "weights": torch.ones(32)}
    return model, batch


@pytest.mark.parametrize("scope", ["joint", "policy"])
def test_tiny_step_has_exact_scope_and_loss(monkeypatch, scope):
    import torch
    model, batch = fixture_model(monkeypatch)
    for name, p in model.named_parameters():
        p.requires_grad_(scope == "joint" or name.startswith("policy."))
    parameters = t.scoped_parameters(model, scope)
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(.9, .999), eps=1e-8,
                                  weight_decay=1e-4, foreach=False, fused=False)
    before = t.accepted.group_digests(model)
    output = t.step(model, optimizer, batch, scope)
    assert output["before_group_sha256"] == before
    assert output["after_group_sha256"] != before
    assert output["optimizer_calls"] == output["forward_backward_calls"] == 1
    assert len(output["outputs83"]) == 32
    assert math.isfinite(output["loss"])
    if scope == "policy":
        assert output["before_group_sha256"]["shared"] == output["after_group_sha256"]["shared"]
        assert output["before_group_sha256"]["value"] == output["after_group_sha256"]["value"]


def test_exact_final_step_on_tiny_tensor_fixture(monkeypatch):
    import torch
    model, batch = fixture_model(monkeypatch)
    optimizer = t.make_optimizer(model, "joint")
    t.step(model, optimizer, batch, "joint")
    saved = deepcopy({"state": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": t.rng_state()})
    expected = t.step(model, optimizer, batch, "joint")
    final = t.tensor_tree({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": t.rng_state()})
    replay_model = deepcopy(model)
    replay_model.load_state_dict(saved["state"])
    replay_opt = t.make_optimizer(replay_model, "joint")
    replay_opt.load_state_dict(saved["optimizer"])
    t.restore_rng(saved["rng"])
    actual = t.step(replay_model, replay_opt, batch, "joint")
    assert {k: v for k, v in expected.items() if k != "seconds"} == {k: v for k, v in actual.items() if k != "seconds"}
    assert t.tensor_tree({"model": replay_model.state_dict(), "optimizer": replay_opt.state_dict(), "rng": t.rng_state()}) == final


@pytest.mark.parametrize("scope,probes", [("joint", 2), ("policy", 1)])
def test_gradient_probe_is_disposable(monkeypatch, scope, probes):
    model, batch = fixture_model(monkeypatch)
    original = t.tensor_tree(model.state_dict())
    result = t.gradient_probe(model, scope, batch)
    assert result["optimizer_calls"] == 0
    assert result["backward_probes"] == probes
    assert result["weights_unchanged"] is True
    assert t.tensor_tree(model.state_dict()) == original
    assert all(p.grad is None for p in model.parameters())


def test_zero_shared_gradient_has_undefined_cosine(monkeypatch):
    import torch
    model, batch = fixture_model(monkeypatch)
    with torch.no_grad():
        model.policy.weight.zero_()
        model.value.weight.zero_()
    result = t.gradient_probe(model, "joint", batch)
    assert result["policy"]["shared_norm"] == result["value"]["shared_norm"] == 0.
    assert result["shared_cosine"] is None and result["undefined_cosine"] is True


def test_parameter_displacement_counts_bit_changes():
    import torch
    original = {"shared.weight": torch.tensor([0., 2.]), "policy.weight": torch.tensor([1.]),
                "value.weight": torch.tensor([3.])}
    state = deepcopy(original)
    state["shared.weight"][0] = -0.
    state["policy.weight"][0] = 2.
    result = t.parameter_displacement(state, original)
    assert result["shared"]["changed_float32_values"] == 1
    assert result["shared"]["l2"] == 0.
    assert result["policy"]["l2"] == 1.
    assert result["all"]["changed_float32_values"] == 2


def test_training_uses_actual_new_collator(monkeypatch):
    calls = []
    monkeypatch.setattr(t.data, "collate", lambda examples, weights: calls.append((examples, weights)) or "batch")
    assert t.selected_batch({"train_examples": ["a", "b"], "train_weights": [1., 1.]}, [1]) == "batch"
    assert calls == [(["b"], [1.])]


def test_prediction_retains_source_root_hash():
    row = {"game_index": 4, "ply": 8, "split": "train", "root_input_sha256": "a" * 64,
           "row_sha256": "b" * 64, "mask": [True] * 81, "policy": [1.] + [0.] * 80,
           "value": [1., -1.]}
    saved = t.prediction_row(row, [0.] * 83)
    assert saved["row_sha256"] == "b" * 64
    assert saved["policy_target81"] == row["policy"]


def test_saved_hashes_match_accepted_model_hashes(monkeypatch):
    model, _ = fixture_model(monkeypatch)
    hashes = t.state_mapping_hashes(model.state_dict())
    assert hashes.pop("all") == t.accepted.state_digest(model)
    assert hashes == t.accepted.group_digests(model)


def test_optimizer_archive_rejects_wrong_step_and_membership(monkeypatch):
    model, batch = fixture_model(monkeypatch)
    optimizer = t.make_optimizer(model, "joint")
    t.step(model, optimizer, batch, "joint")
    payload = {"metadata": {"scope": "joint", "checkpoint_step": 1},
               "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    t.verify_optimizer_payload(payload)
    bad = deepcopy(payload)
    bad["metadata"]["checkpoint_step"] = 2
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        t.verify_optimizer_payload(bad)
    bad = deepcopy(payload)
    bad["optimizer"]["param_groups"][0]["params"].reverse()
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        t.verify_optimizer_payload(bad)


def test_checkpoint_roundtrip_and_false_metadata_rejected(monkeypatch, tmp_path):
    model, batch = fixture_model(monkeypatch)
    optimizer = t.make_optimizer(model, "joint")
    t.step(model, optimizer, batch, "joint")
    monkeypatch.setattr(t, "ROOT", tmp_path)
    monkeypatch.setattr(t.accepted, "ROOT", tmp_path)
    monkeypatch.setattr(t.accepted_data, "ROOT", tmp_path)
    monkeypatch.setattr(t.accepted, "validate_architecture", lambda model: None)
    metadata = {"scope": "joint", "historical_warm_start_sha256": t.accepted.PARENT_SHA256,
                "historical_data_relabelled": False}
    item = t.save_checkpoint(model, optimizer, 1, metadata, tmp_path / "good")
    payload, manifest = t.checkpoint_payload(item)
    assert payload["metadata"]["checkpoint_step"] == 1
    assert manifest["tree_sha256"]["state_dict"] == t.digest(t.tensor_tree(model.state_dict()))
    # A false model hash cannot pass just because the JSON and archive agree.
    import torch
    payload["metadata"]["weights_sha256"] = "0" * 64
    path = tmp_path / "bad.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    manifest["checkpoint"] = t.artifact(path)
    manifest["metadata"] = payload["metadata"]
    fake = t.save(tmp_path / "bad_manifest.json", manifest)
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        t.checkpoint_payload(fake)


def test_no_entrypoint_loads_parent_before_startup():
    import ast
    tree = ast.parse(Path(t.__file__).read_text(encoding="utf-8"))
    for name in ("run_qualification", "run_fit", "run_diagnostics"):
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        startup_line = min(n.lineno for n in calls if isinstance(n.func, ast.Name) and n.func.id == "engine")
        later = [n.lineno for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in
                 ("load_parent", "qualify_data", "load_selected")]
        assert all(line > startup_line for line in later)
