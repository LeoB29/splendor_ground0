import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from splendor_ai.alphazero import run_early_learning as p


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_150", Path("scripts/analysis/verify_early_learning_150.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_data(n=12, seed=150):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 8, generator=generator)
    q = torch.randint(-8, 9, (n, 4), generator=generator).float()/8
    mask = torch.ones(n, 4, dtype=torch.bool)
    mask[:, 0] = False
    mask[0] = False
    mask[1, 3] = False
    return {"control": x, "afterstate": x*.5, "mask": mask, "q": q, "y": q-q[:, :1]}


def test_grid_adds_exactly_missing_127_checks():
    checks = [i for i in range(4097) if p.validate_grid(i)]
    original = list(range(0, 4097, 128))
    assert len(checks) == 160
    assert set(checks)-set(original) == set(range(1, 128))
    assert set(original).issubset(checks)


def test_added_validation_preserves_original_full_trajectory():
    train, val = synthetic_data(), synthetic_data(5, 151)
    dense = p.fit_dense("control", 150001, train, val, updates=129)
    original = p.old.fit_model("control", 150001, train, val, updates=129)
    p.verify_original(dense, original, total=129)
    assert dense["evaluation_rng_unchanged"]
    assert len(dense["history"]) == 130
    changed = copy.deepcopy(dense)
    changed["history"][0]["validation_mse"] += .01
    with pytest.raises(ValueError, match="history"):
        p.verify_original(changed, original, total=129)
    changed = copy.deepcopy(dense)
    key = next(iter(changed["final_state"]))
    changed["final_state"][key].add_(1)
    with pytest.raises(ValueError, match="Tensor mismatch"):
        p.verify_original(changed, original, total=129)


def test_independent_loop_reproduces_dense_history_and_tensors(audit):
    train, val = synthetic_data(), synthetic_data(5, 151)
    main = p.fit_dense("afterstate", 150002, train, val, updates=8)
    independent = audit.independent_fit("afterstate", 150002, train, val, total=8)
    audit.compare(main, independent)


def test_zero_loss_retains_earliest_tie():
    data = synthetic_data()
    data["mask"][:] = False
    fit = p.fit_dense("control", 150003, data, data, updates=3)
    assert fit["selected_updates"] == {"coarse": 0, "dense": 0}
    assert all(h["validation_mse"] == 0 for h in fit["history"])


def screen_case():
    q = torch.tensor([[0., .5, -.25, 1.], [0., .25, -.5, -1.]])
    mask = torch.tensor([[False, True, True, False]]*2)
    rows = {arm: [{"seed": seed,
                  "coarse": {"selected_update": 0, "advantages": torch.zeros_like(q).tolist(), "choices": [0, 0]},
                  "dense": {"selected_update": 4, "advantages": q.tolist(), "choices": [1, 1]}}
                 for seed in p.CONFIG["seeds"]] for arm in p.CONFIG["arms"]}
    return {"q": q, "y": q, "mask": mask}, rows


def test_screen_can_pass_and_matches_separate_statistics(audit):
    data, rows = screen_case()
    result = p.statistics(data["q"], data["y"], data["mask"], rows)
    audit.compare(result, audit.independent_stats(data, rows), tolerance=1e-12)
    assert result["any_development_screen_passed"]
    assert result["zero_mse"] == .15625
    assert result["arms"]["afterstate"]["means"]["dense"]["wdl_gain_over_035"] == .375
    assert not result["confirmatory"]


@pytest.mark.parametrize("failure", ["early_seeds", "mse_versus_coarse", "mse_versus_zero", "wdl_versus_035", "wdl_versus_coarse"])
def test_each_screen_criterion_is_required(failure, audit):
    data, rows = screen_case()
    for seeds in rows.values():
        for row in seeds:
            if failure == "early_seeds":
                row["dense"]["selected_update"] = 128
            elif failure == "mse_versus_coarse":
                row["coarse"]["advantages"] = (data["y"]+.01).tolist()
                row["dense"]["advantages"] = (data["y"]+.02).tolist()
            elif failure == "mse_versus_zero":
                row["dense"]["advantages"] = torch.zeros_like(data["y"]).tolist()
                row["coarse"]["advantages"] = torch.ones_like(data["y"]).tolist()
            elif failure == "wdl_versus_035":
                row["dense"]["choices"] = [0, 0]
                row["coarse"]["choices"] = [2, 2]
            else:
                row["coarse"]["choices"] = [1, 1]
    result = p.statistics(data["q"], data["y"], data["mask"], rows)
    audit.compare(result, audit.independent_stats(data, rows), tolerance=1e-12)
    assert not result["any_development_screen_passed"]
    assert not result["arms"]["afterstate"]["development_screen"][failure]


def test_prediction_ties_keep_baseline_and_use_mask(tmp_path, monkeypatch):
    monkeypatch.setitem(p.CONFIG, "seeds", [150001])
    data = synthetic_data(3)
    rows = {}
    for arm in p.CONFIG["arms"]:
        net = p.AdvantageModel(8)
        for parameter in net.parameters():
            parameter.data.zero_()
        path = tmp_path / f"{arm}.pt"
        p.old.tensor_save(path, {"width": 8, "coarse_state": net.state_dict(), "dense_state": net.state_dict(),
                                "selected_updates": {"coarse": 0, "dense": 0}})
        rows[f"{arm}_150001"] = {"path": str(path), "sha256": p.old.file_hash(path)}
    prediction = p.predictions(data, rows)
    for arm in prediction:
        assert prediction[arm][0]["dense"]["choices"] == [0, 0, 0]


def test_frozen_file_mutation_detected(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("original", encoding="utf-8")
    files = {str(path): p.old.file_hash(path)}
    p.verify_files(files)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        p.verify_files(files)
