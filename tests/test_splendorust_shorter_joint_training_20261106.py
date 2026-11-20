"""Model-free contracts and tiny tensor fixtures; never load EF8A or play games."""
from copy import deepcopy
import math
from pathlib import Path
import random

import pytest

from tools.splendorust_shorter_joint_20261106 import training as t


def test_startup_verifies_once(monkeypatch):
    calls = []
    monkeypatch.setattr(t, "_STARTUP", None)
    monkeypatch.setattr(t.guard, "verify_runtime_sources", lambda: calls.append(True) or {"ok": True})
    first = t.startup()
    first["runtime_sources"]["ok"] = False
    assert t.startup()["runtime_sources"] == {"ok": True}
    assert calls == [True]


@pytest.mark.parametrize("args", [(2, "broad", "joint"), (True, "broad", "joint"),
                                  (0, "dense", "joint"), (0, "broad", "policy"),
                                  (0, "other", "joint"), (0, "broad", "policy_only")])
def test_rejects_extra_cells(args):
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        t.fit_identity(*args)


def test_fixed_work_inventory():
    assert t.STEPS * t.BATCH == 3072
    assert t.STEPS * 2 + 2 + 2 == 196
    assert t.CHECKPOINTS == (0, 95, 96)
    assert t.EVALUATION_STEPS == (0, 96)
    assert t.fit_identity(1, "broad", "joint")["fit_id"] == "r1_broad_joint"


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
    metadata = {"scope": "joint", "study_id": t.STUDY, "historical_warm_start_sha256": t.accepted.PARENT_SHA256,
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


def test_original_math_and_rng_source_are_exact():
    proof = t.source_equivalence()
    assert proof["status"] == "shorter_schedule_math_rng_sampling_source_verified"
    assert proof["model_calls"] == proof["new_games"] == 0


def test_diagnostic_entrypoint_cannot_execute_gradient_probe():
    import ast
    tree = ast.parse(Path(t.__file__).read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_diagnostics")
    names = {n.func.id for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not names & {"gradient_probe", "step", "make_optimizer"}
    assert t.EVALUATION_STEPS == (0, 96)


def test_actual_joint_optimizer_owns_all_three_groups(monkeypatch):
    model, batch = fixture_model(monkeypatch)
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = t.make_optimizer(model, "joint")
    assert all(p.requires_grad for p in model.parameters())
    assert [id(p) for g in optimizer.param_groups for p in g["params"]] == [id(p) for p in model.parameters()]
    result = t.step(model, optimizer, batch, "joint")
    assert all(result["before_group_sha256"][g] != result["after_group_sha256"][g]
               for g in ("shared", "policy", "value"))
    assert all(p.grad is not None for p in model.parameters())


def test_96_batch_loader_rejects_repeated_roots_and_extra_probes():
    loaded = {"train_examples": [None] * 3072, "train_weights": [1.] * 3072,
              "heldout_examples": [None] * 3557,
              "batches": [list(range(i, i + 32)) for i in range(0, 3072, 32)], "probe_batches": []}
    assert t.validate_loaded(loaded) is loaded
    bad = deepcopy(loaded); bad["batches"][1] = bad["batches"][0]
    with pytest.raises(ValueError):
        t.validate_loaded(bad)
    bad = deepcopy(loaded); bad["probe_batches"] = [bad["batches"][0]]
    with pytest.raises(ValueError):
        t.validate_loaded(bad)


def test_run_fit_replays_actual_95_to_96_control_flow(monkeypatch, tmp_path):
    """Run the real fit orchestration with inert model/optimizer test doubles."""
    examples = [{"game_index": i // 16, "ply": i, "row_sha256": str(i)} for i in range(3072)]
    loaded = {"train_examples": examples, "train_weights": [1.] * 3072,
              "heldout_examples": [None] * 3557, "probe_batches": [],
              "batches": [list(range(i, i + 32)) for i in range(0, 3072, 32)],
              "source_receipt": {}, "selection_cell": {"breadth": "broad"}}
    saved, calls, restored, checkpoints = {}, [], [], []
    monkeypatch.setattr(t, "engine", lambda folder: None)
    monkeypatch.setattr(t.data, "load_selected", lambda *a, **k: loaded)
    monkeypatch.setattr(t, "seed_rng", lambda rep: 202611012 + rep)
    monkeypatch.setattr(t.accepted, "load_parent", lambda: object())
    monkeypatch.setattr(t, "make_optimizer", lambda *a: object())
    monkeypatch.setattr(t.accepted, "group_digests", lambda model: {"shared": "s", "policy": "p", "value": "v"})
    monkeypatch.setattr(t, "selected_batch", lambda loaded, indices: indices)
    def step(model, optimizer, batch, scope, **kwargs):
        calls.append(list(batch))
        return {"marker": batch[0], "optimizer_calls": 1, "seconds": 0.}
    monkeypatch.setattr(t, "step", step)
    def save(path, value):
        key = str(path); saved[key] = deepcopy(value); return {"path": key}
    monkeypatch.setattr(t, "save", save)
    monkeypatch.setattr(t, "read_json", lambda item, **kwargs: deepcopy(saved[item["path"]]))
    def save_checkpoint(model, optimizer, index, metadata, folder, **kwargs):
        checkpoints.append(index)
        return save(Path(folder) / "manifest.json", {"step": index, "tree_sha256": {"state": str(index)}})
    monkeypatch.setattr(t, "save_checkpoint", save_checkpoint)
    def load_checkpoint(item, **kwargs):
        restored.append(saved[item["path"]]["step"])
        return object(), object(), {}
    monkeypatch.setattr(t, "load_checkpoint", load_checkpoint)
    result = t.run_fit({}, 1, "broad", "joint", tmp_path)
    assert len(calls) == 97 and calls[:96] == loaded["batches"]
    assert calls[-1] == loaded["batches"][95]
    assert restored == [95] and checkpoints == [0, 95, 96, 96]
    assert set(result["checkpoints"]) == {"0", "95", "96"}
    replay = saved[result["replay"]["path"]]
    assert replay["from_checkpoint"] == result["checkpoints"]["95"]
    assert replay["expected_checkpoint"] == result["checkpoints"]["96"]
    assert result["optimizer_calls"] == 96 and result["disposable_optimizer_calls"] == 1
    assert result["record_presentations"] == 3072 and result["study_id"] == t.STUDY


def test_saved_original_prefixes_authenticate_all192_records_without_models():
    proof = t.historical_prefix_bindings()
    assert proof["compared_steps_per_fit"] == 96 and proof["model_calls"] == 0
    assert proof["saved_scientific_audit"]["sha256"] == "8707600941066c1284ed5eed916035a3e4d5fc242c6599b16fcc39dd8d231f0a"
    assert [fit["fit_id"] for fit in proof["fits"]] == ["r0_broad_joint", "r1_broad_joint"]
    new_sha = "1" * 64
    total = 0
    for fit in proof["fits"]:
        assert len(fit["updates"]) == 96
        for index, item in enumerate(fit["updates"], start=1):
            original = t.read_json(item)
            assert original["step"] == index and original["fit_id"] == fit["fit_id"]
            shorter = deepcopy(original)
            shorter.update(seconds=original["seconds"] + 1., selection_sha256=new_sha)
            assert len(t.verify_historical_update(shorter, original, new_sha, proof["selection_sha256"])) == 64
            total += 1
    assert total == 192


@pytest.mark.parametrize("mutate", [
    lambda row: row["outputs83"][0].__setitem__(0, row["outputs83"][0][0] + .125),
    lambda row: row.__setitem__("loss", row["loss"] + .125),
    lambda row: row.__setitem__("gradient_norm_before_clip", row["gradient_norm_before_clip"] + .125),
    lambda row: row["after_group_sha256"].__setitem__("shared", "0" * 64),
    lambda row: row["before_group_sha256"].__setitem__("value", "0" * 64),
    lambda row: row["example_indices"].reverse(),
    lambda row: row.__setitem__("clipped", not row["clipped"]),
    lambda row: row.__setitem__("step", 2),
    lambda row: row.__setitem__("selection_sha256", "0" * 64),
    lambda row: row.__setitem__("seconds", float("nan")),
    lambda row: row.__setitem__("extra_unverified_metadata", True),
])
def test_prefix_rejects_every_mathematical_or_identity_mutation(mutate):
    proof = t.historical_prefix_bindings()
    original = t.read_json(proof["fits"][0]["updates"][0])
    shorter = deepcopy(original); shorter["selection_sha256"] = "1" * 64
    mutate(shorter)
    with pytest.raises(ValueError):
        t.verify_historical_update(shorter, original, "1" * 64, proof["selection_sha256"])


def test_prefix_comparison_preserves_float_signed_zero():
    original = {"seconds": 0., "selection_sha256": "a" * 64, "outputs83": [0.]}
    shorter = {**original, "selection_sha256": "b" * 64, "outputs83": [-0.]}
    with pytest.raises(ValueError, match="historical first-pass"):
        t.verify_historical_update(shorter, original, "b" * 64, "a" * 64)


def test_prefix_refuses_substituted_original_audit(monkeypatch):
    actual = t.artifact
    monkeypatch.setattr(t, "artifact", lambda path: {**actual(path), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="accepted original saved scientific audit"):
        t.historical_prefix_bindings()
