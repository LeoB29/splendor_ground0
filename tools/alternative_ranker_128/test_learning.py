"""Target/weight firewalls, deterministic fitting and integrity checks."""
import copy
import json
import numpy as np
import pytest
from . import learning as h
from .test_features import example


def row(name, outsiders):
    obs, actions = example()
    count = 3+outsiders
    actions = np.array([actions[i % 3] for i in range(count)], dtype=np.float32)
    return {"id": name, "pair": name, "split": "train", "observation": obs,
            "features": actions, "members": [[i] for i in range(count)],
            "old_sums_wdl": [0, 0, 0]+list(range(outsiders)),
            "outsider_start": 3, "outsider_support": outsiders}


def dataset(tmp_path):
    import torch
    rows = {"old": [row("old0", 0), row("old1", 1), row("old2", 2)],
            "added": [row("add0", 3), row("add1", 2)]}
    refs = {}
    for pool, values in rows.items():
        # PyTorch safe loader accepts tensors and plain Python, not NumPy arrays.
        for r in values:
            for key in ("observation", "features"):
                r[key] = torch.tensor(r[key])
        path = tmp_path/f"{pool}.pt"
        torch.save(values, path)
        refs[pool] = h.artifact(path)
    return h.prepare_dataset(refs, tmp_path/"data", expected_counts={k: len(v) for k, v in rows.items()}, fixture=True)


def test_targets_are_centered_and_heldout_or_512_labels_rejected():
    X, y = h.row_examples(row("x", 3))
    assert X.shape == (3, 465)
    np.testing.assert_array_equal(y, np.array([-1, 0, 1])/256)
    for alteration in ({"split": "validation"}, {"assessment": {}}, {"teacher_worlds": 512},
                       {"old_sums_wdl": [0, 0, 0, 129]}):
        with pytest.raises(ValueError):
            h.row_examples({**row("x", 1), **alteration})
    X, y = h.row_examples(row("empty", 0))
    assert X.shape == (0, 465) and len(y) == 0


def test_pools_have_half_mass_roots_equal_and_all_membership_retained(tmp_path):
    ref = dataset(tmp_path)
    data, arrays = h.load_dataset(ref)
    assert len(data["roots"]) == 5 and data["summary"]["old"]["support_counts"]["empty"] == 1
    assert arrays["X"].shape == (8, 465)
    mass = {pool: [] for pool in ("old", "added")}
    for r in data["roots"]:
        if r["outsiders"]:
            mass[r["pool"]].append(arrays["weight"][r["start"]:r["stop"]].sum())
    np.testing.assert_allclose(mass["old"], [2, 2])
    np.testing.assert_allclose(mass["added"], [2, 2])
    np.testing.assert_allclose(arrays["weight"].sum(), 8, rtol=0, atol=2e-15)
    with pytest.raises(FileExistsError):
        h.prepare_dataset(data["source_files"], tmp_path/"data", expected_counts={"old": 3, "added": 2}, fixture=True)


def test_ridge_matches_direct_weighted_standardized_solution():
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(7)
    X = rng.normal(size=(29, 465)).astype(np.float32)
    X[:, 0] = 1
    y = X[:, 1]*.1-X[:, 4]*.05
    w = np.linspace(.5, 1.5, len(X))
    fitted = h.fit_model("ridge", X, y, w)
    with threadpool_limits(limits=1):
        x = X.astype(np.float64)
        mean = np.average(x, axis=0, weights=w)
        variance = np.average((x-mean)**2, axis=0, weights=w)
        scale = np.where(variance > 1e-12, np.sqrt(variance), 1)
        z = (x-mean)/scale
        ym = np.average(y, weights=w)
        coef = np.linalg.solve(z.T @ (w[:, None]*z)+100*np.eye(465), z.T@(w*(y-ym)))
    np.testing.assert_allclose(fitted["coef"], coef, atol=1e-13)
    np.testing.assert_allclose(h.predict_scores(fitted, X), z@coef+ym, atol=1e-12)


def test_both_fixed_models_replay_and_roundtrip(tmp_path):
    ref = dataset(tmp_path)
    settings = copy.deepcopy(h.CONFIG)
    settings["hgb"].update(max_iter=5, min_samples_leaf=1)
    models_ref = h.fit_and_save(ref, tmp_path/"models", settings)
    lock = json.loads(h.verify(models_ref).read_text())
    _, arrays = h.load_dataset(ref)
    assert set(lock["models"]) == {"ridge", "hgb"}
    for recipe, record in lock["models"].items():
        model = h.load_model(record["model"], expected_recipe=recipe, expected_dataset=ref)
        assert record["identical_second_fit"]
        assert h.parameter_digest(model) == record["parameter_sha256"]
        assert h.prediction_digest(model, arrays["X"]) == record["all_train_prediction_sha256"]
        assert len(h.predict_scores(model, np.empty((0, 465), np.float32))) == 0
    with pytest.raises(ValueError, match="recipe"):
        h.load_model(lock["models"]["ridge"]["model"], expected_recipe="hgb")


def test_changed_dataset_bytes_and_source_identity_fail(tmp_path):
    ref = dataset(tmp_path)
    data = json.loads(h.verify(ref).read_text())
    with h.verify(data["files"]["X"]).open("r+b") as handle:
        handle.write(b"BAD!")
    with pytest.raises(ValueError, match="digest"):
        h.load_dataset(ref)
    with pytest.raises(ValueError, match="identities"):
        h.prepare_dataset(data["source_files"], tmp_path/"bad_production")


def test_replay_disagreement_is_not_silently_selected(tmp_path, monkeypatch):
    ref = dataset(tmp_path)
    settings = copy.deepcopy(h.CONFIG)
    settings["recipes"] = ["ridge"]
    original = h.fit_model
    calls = 0
    def different(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            result["coef"][0] += .01
        return result
    monkeypatch.setattr(h, "fit_model", different)
    with pytest.raises(ValueError, match="replay differs"):
        h.fit_and_save(ref, tmp_path/"bad_fit", settings)
    assert not (tmp_path/"bad_fit/models.json").exists()
