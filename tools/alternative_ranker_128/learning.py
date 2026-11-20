"""Two fixed, regularized TRAIN-only ranking regressors.

Each target is the original128-world score centered within the recorded
outsiders of its root. Canonical groups, rather than concrete aliases, receive
weight. This module never opens a fresh panel, source game or outcome artifact.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from .features import FEATURE_CONFIG, FEATURE_SIZE, group_features

ROOT = Path(__file__).resolve().parents[2]
COUNTS = {"old": 9461, "added": 11520}
TRAIN_REFS = {
    "old": {"path": "outputs/full_teacher_fit_128/train.pt",
            "sha256": "289a0540ce82c395c8edca111224a35343cb4d76a5ee68cef72c67d935d542df"},
    "added": {"path": "outputs/independent_game_scaling_128/added.pt",
              "sha256": "58ef6ca7f7e92f5c786df0ed5b531500dc9828fa0cc9c109564c23b460c1252a"},
}
CONFIG = {
    "version": "alternative_ranker_fixed_recipes_v1",
    "recipes": ["ridge", "hgb"], "pooled_fit_per_recipe": 1,
    "feature_config": FEATURE_CONFIG,
    "target": "original128_score_minus_unweighted_root_outsider_score_mean",
    "target_units": "expected_score; old_sums_wdl_div256_then_root_centered",
    "root_inclusion": "all_TRAIN_roots_with_at_least_one_recorded_outsider; singleton_target_zero; empty_roots_metadata_only",
    "sample_weight": "old_and_added_each_half_total; equal_nonempty_root_mass_within_pool; equal_outsider_mass_within_root; rescale_sum_to_sample_count",
    "feature_centering": "no_root_centering; ridge_weighted_global_standardization_only",
    "ridge": {"alpha": 100.0, "fit_intercept": True,
              "standardization": "weighted_TRAIN_population_mean_and_variance; scale1_if_variance_le1e-12",
              "solver": "streamed_float64_gram_positive_definite_solve"},
    "hgb": {"loss": "squared_error", "learning_rate": .05, "max_iter": 250,
            "max_leaf_nodes": 8, "max_depth": None, "min_samples_leaf": 64,
            "l2_regularization": 10., "max_bins": 255, "categorical_features": None,
            "monotonic_cst": None, "interaction_cst": None, "warm_start": False,
            "early_stopping": False, "validation_fraction": None,
            "random_state": 2801000000},
    "cpu_threads": 1, "chunk_size": 2048,
    "hyperparameter_selection": "none; no_CV_no_sweep_no_early_stopping",
    "repeatability": "two_deterministic_fits_per_recipe; exact_parameter_and_all_TRAIN_prediction_digests",
    "production_data": TRAIN_REFS,
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest()}


def verify(ref):
    path = Path(ref["path"])
    if not path.is_absolute():
        path = ROOT / path
    actual = artifact(path)
    if actual["sha256"] != ref["sha256"]:
        raise ValueError(f"Artifact digest changed: {path}")
    return path


def _save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _plain_ref(ref):
    return {key: ref[key] for key in ("path", "sha256")}


def _as_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def row_examples(row):
    """Extract only recorded outsiders. No inference or full-legal expansion."""
    if row.get("split") != "train":
        raise ValueError("Only TRAIN rows are permitted")
    forbidden = {"assessment", "assessment_outcomes", "fresh_outcomes", "fresh_means",
                 "teacher_worlds", "teacher_sums_wdl", "teacher_outsider"}
    if forbidden.intersection(row):
        raise ValueError("Fresh or alternate-label fields forbidden")
    sums = row["old_sums_wdl"]
    if not 1 <= len(sums) <= 8 or len(row["members"]) != len(sums):
        raise ValueError("Invalid recorded teacher support")
    if any(type(v) is not int or not -128 <= v <= 128 for v in sums):
        raise ValueError("Expected original128 integer WDL sums")
    start = min(3, len(sums))
    if row.get("outsider_start", start) != start or row.get("outsider_support", len(sums)-start) != len(sums)-start:
        raise ValueError("Outsider support metadata disagrees")
    if len(sums) == start:
        return np.empty((0, FEATURE_SIZE), np.float32), np.empty(0, np.float64)
    features = group_features(row["observation"], row["features"], row["members"][start:])
    # The constant1/2 in expected score cancels exactly under root centering.
    values = np.asarray(sums[start:], dtype=np.float64) / 256.
    return features, values - values.mean()


def _emit_pool(rows, pool, expected_count, handles, records, seen):
    if len(rows) != expected_count:
        raise ValueError("TRAIN row count differs")
    counts = Counter()
    offset = records[-1]["stop"] if records else 0
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Missing or repeated TRAIN row identity")
        seen.add(identifier)
        X, y = row_examples(row)
        k = len(y)
        counts["empty" if k == 0 else "singleton" if k == 1 else "multiple"] += 1
        handles["X"].write(X.tobytes(order="C"))
        handles["y"].write(y.tobytes(order="C"))
        records.append({"id": identifier, "pool": pool, "pair": str(row.get("pair", "")),
                        "start": offset, "stop": offset+k, "outsiders": k})
        offset += k
    return {"rows": len(rows), "samples": sum(r["outsiders"] for r in records if r["pool"] == pool),
            "eligible_roots": len(rows)-counts["empty"],
            "support_counts": {name: counts[name] for name in ("empty", "singleton", "multiple")}}


def prepare_dataset(datarefs, base, *, expected_counts=None, fixture=False):
    """Authenticate and mmap one inherited tensor pool at a time, emit outsiders.

    New files live only under the explicitly supplied new dataset directory.
    A failed attempt is preserved and must use a different destination on retry.
    """
    expected_counts = dict(COUNTS if expected_counts is None else expected_counts)
    refs = {key: _plain_ref(value) for key, value in datarefs.items()}
    if set(refs) != set(COUNTS) or set(expected_counts) != set(COUNTS):
        raise ValueError("Both and only old/added pools required")
    if not fixture and (refs != TRAIN_REFS or expected_counts != COUNTS):
        raise ValueError("Production TRAIN identities changed")
    base = Path(base).resolve()
    if fixture and base.is_relative_to(ROOT / "outputs/alternative_ranker_128"):
        raise ValueError("Fixture cannot target production")
    base.mkdir(parents=True, exist_ok=False)
    records, summary, seen = [], {}, set()
    handles = {name: (base/f"{name}.bin").open("xb") for name in ("X", "y")}
    try:
        import torch
        torch.set_num_threads(1)
        for pool in ("old", "added"):
            path = verify(refs[pool])
            rows = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            summary[pool] = _emit_pool(rows, pool, expected_counts[pool], handles, records, seen)
            del rows
            gc.collect()
            verify(refs[pool])
    finally:
        for handle in handles.values():
            handle.close()
    n = sum(value["samples"] for value in summary.values())
    if not n or any(not value["eligible_roots"] for value in summary.values()):
        raise ValueError("Each pool must contain a nonempty outsider root")
    weights = np.memmap(base/"weight.bin", dtype=np.float64, mode="w+", shape=(n,))
    for row in records:
        k = row["outsiders"]
        if k:
            weights[row["start"]:row["stop"]] = n*.5 / (summary[row["pool"]]["eligible_roots"]*k)
    weights.flush()
    del weights
    value = {"schema": "alternative_ranker_dataset_v1", "fixture": fixture,
             "feature_config": FEATURE_CONFIG, "target": CONFIG["target"],
             "weight_policy": CONFIG["sample_weight"], "source_files": refs,
             "expected_counts": expected_counts, "summary": summary, "roots": records,
             "shape": [n, FEATURE_SIZE],
             "files": {name: artifact(base/f"{name}.bin") for name in ("X", "y", "weight")},
             "dtypes": {"X": "float32", "y": "float64", "weight": "float64"},
             "fresh_outcomes_read": False, "validation_read": False, "new_simulations": 0}
    _save_json(base/"dataset.json", value)
    return artifact(base/"dataset.json")


def load_dataset(ref):
    data = _json(verify(ref))
    if (data["schema"] != "alternative_ranker_dataset_v1" or data["feature_config"] != FEATURE_CONFIG
            or data["target"] != CONFIG["target"] or data["weight_policy"] != CONFIG["sample_weight"]
            or data["dtypes"] != {"X": "float32", "y": "float64", "weight": "float64"}
            or data["fresh_outcomes_read"] or data["validation_read"] or data["new_simulations"]):
        raise ValueError("Dataset contract changed")
    n, width = data["shape"]
    if type(n) is not int or n <= 0 or width != FEATURE_SIZE:
        raise ValueError("Invalid dataset shape")
    arrays = {}
    for name, dtype in (("X", np.float32), ("y", np.float64), ("weight", np.float64)):
        path = verify(data["files"][name])
        shape = (n, width) if name == "X" else (n,)
        if path.stat().st_size != int(np.prod(shape))*np.dtype(dtype).itemsize:
            raise ValueError("Unexpected dataset byte count")
        arrays[name] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
    cursor = 0
    seen, counts, eligible = set(), Counter(), Counter()
    for row in data["roots"]:
        pool, lo, hi = row["pool"], row["start"], row["stop"]
        if (pool not in COUNTS or row["id"] in seen or lo != cursor
                or hi-lo != row["outsiders"] or not 0 <= hi-lo <= 5):
            raise ValueError("Invalid dataset root membership")
        seen.add(row["id"]); counts[pool] += 1; eligible[pool] += hi > lo
        cursor = hi
        if hi > lo:
            wanted = n*.5/(data["summary"][pool]["eligible_roots"]*(hi-lo))
            if not np.all(arrays["weight"][lo:hi] == wanted) or abs(arrays["y"][lo:hi].sum()) > 1e-12:
                raise ValueError("Weight or centered target changed")
    if cursor != n or dict(counts) != data["expected_counts"]:
        raise ValueError("Dataset sample or root coverage differs")
    for pool in COUNTS:
        if eligible[pool] != data["summary"][pool]["eligible_roots"]:
            raise ValueError("Eligible root denominator differs")
    for lo in range(0, n, CONFIG["chunk_size"]):
        if not all(np.isfinite(value[lo:lo+CONFIG["chunk_size"]]).all() for value in arrays.values()):
            raise ValueError("Nonfinite TRAIN data")
    return data, arrays


def _fit_ridge(X, y, weight, settings):
    from scipy.linalg import solve
    n, width = X.shape
    chunk = settings["chunk_size"]
    mass = float(np.sum(weight, dtype=np.float64))
    sums, squares = np.zeros(width), np.zeros(width)
    for lo in range(0, n, chunk):
        x = np.asarray(X[lo:lo+chunk], dtype=np.float64)
        w = np.asarray(weight[lo:lo+chunk])
        sums += np.sum(x*w[:, None], axis=0)
        squares += np.sum(x*x*w[:, None], axis=0)
    mean = sums/mass
    variance = np.maximum(0., squares/mass-mean*mean)
    scale = np.where(variance > 1e-12, np.sqrt(variance), 1.)
    ymean = float(np.dot(weight, y)/mass)
    gram, rhs = np.zeros((width, width)), np.zeros(width)
    for lo in range(0, n, chunk):
        z = (np.asarray(X[lo:lo+chunk], dtype=np.float64)-mean)/scale
        w = np.asarray(weight[lo:lo+chunk])
        gram += z.T @ (w[:, None]*z)
        rhs += z.T @ (w*(y[lo:lo+chunk]-ymean))
    gram.flat[::width+1] += settings["ridge"]["alpha"]
    coef = solve(gram, rhs, assume_a="pos", check_finite=True)
    return {"recipe": "ridge", "mean": mean, "scale": scale,
            "coef": coef, "intercept": ymean}


def fit_model(recipe, X, y, weight, settings=None):
    from threadpoolctl import threadpool_limits
    settings = CONFIG if settings is None else settings
    if recipe not in settings["recipes"]:
        raise ValueError("Unknown fixed recipe")
    with threadpool_limits(limits=1):
        if recipe == "ridge":
            return _fit_ridge(X, y, weight, settings)
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(**settings["hgb"])
        model.fit(X, y, sample_weight=weight)
        if model.n_iter_ != settings["hgb"]["max_iter"]:
            raise ValueError("Unexpected boosted-tree early stopping")
        return {"recipe": "hgb", "model": model}


def predict_scores(payload, X):
    from threadpoolctl import threadpool_limits
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != FEATURE_SIZE or not np.isfinite(X).all():
        raise ValueError("Invalid ranker prediction features")
    if not len(X):
        return np.empty(0, dtype=np.float64)
    with threadpool_limits(limits=1):
        if payload["recipe"] == "ridge":
            result = ((X-payload["mean"])/payload["scale"]) @ payload["coef"] + payload["intercept"]
        elif payload["recipe"] == "hgb":
            result = payload["model"].predict(X)
        else:
            raise ValueError("Unknown ranker payload")
    result = np.asarray(result, dtype=np.float64)
    if result.shape != (len(X),) or not np.isfinite(result).all():
        raise ValueError("Invalid ranker predictions")
    return result


def _array_digest(value):
    value = np.asarray(value)
    if value.dtype.names:
        # Structured-node dtypes contain padding bytes: hash semantic fields.
        return {name: _array_digest(value[name]) for name in value.dtype.names}
    contiguous = np.ascontiguousarray(value)
    return {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest()}


def parameter_digest(payload):
    if payload["recipe"] == "ridge":
        return digest({key: _array_digest(payload[key]) for key in ("mean", "scale", "coef", "intercept")})
    model = payload["model"]
    return digest({"baseline": _array_digest(model._baseline_prediction), "iterations": model.n_iter_,
                   "nodes": [[{name: _array_digest(getattr(tree, name)) for name in
                               ("nodes", "binned_left_cat_bitsets", "raw_left_cat_bitsets")}
                              for tree in iteration] for iteration in model._predictors],
                   "bins": [_array_digest(x) for x in model._bin_mapper.bin_thresholds_]})


def prediction_digest(payload, X, chunk=2048):
    h = hashlib.sha256()
    for lo in range(0, len(X), chunk):
        h.update(predict_scores(payload, X[lo:lo+chunk]).tobytes())
    return h.hexdigest()


def fit_and_save(dataset_ref, base, config=None):
    """Two fixed pooled models, each fit twice; keep the first fit, never select."""
    settings = CONFIG if config is None else config
    data, arrays = load_dataset(dataset_ref)
    if settings != CONFIG and not data["fixture"]:
        raise ValueError("Production recipe changed")
    if not data["fixture"] and (data["source_files"] != TRAIN_REFS or data["expected_counts"] != COUNTS):
        raise ValueError("Unrecognized production TRAIN data")
    base = Path(base).resolve()
    if data["fixture"] and base.is_relative_to(ROOT / "outputs/alternative_ranker_128"):
        raise ValueError("Fixture cannot target production")
    base.mkdir(parents=True, exist_ok=False)
    results = {}
    for recipe in settings["recipes"]:
        started = time.perf_counter()
        payload = fit_model(recipe, arrays["X"], arrays["y"], arrays["weight"], settings)
        params = parameter_digest(payload)
        predictions = prediction_digest(payload, arrays["X"], settings["chunk_size"])
        payload["contract"] = {"dataset": _plain_ref(dataset_ref), "config": settings,
                               "feature_config": FEATURE_CONFIG}
        model_path = base/f"{recipe}.pkl"
        with model_path.open("xb") as handle:
            pickle.dump(payload, handle, protocol=5)
        del payload
        gc.collect()
        replay = fit_model(recipe, arrays["X"], arrays["y"], arrays["weight"], settings)
        if parameter_digest(replay) != params or prediction_digest(replay, arrays["X"], settings["chunk_size"]) != predictions:
            raise ValueError("Deterministic fit replay differs; no model may be selected")
        del replay
        gc.collect()
        results[recipe] = {"model": artifact(model_path), "parameter_sha256": params,
                           "all_train_prediction_sha256": predictions, "identical_second_fit": True,
                           "selected_fit": "first", "elapsed_seconds": time.perf_counter()-started}
    result = {"schema": "alternative_ranker_models_v1", "dataset": _plain_ref(dataset_ref),
              "config": settings, "feature_config": FEATURE_CONFIG, "models": results,
              "fresh_outcomes_read": False, "validation_read": False,
              "new_simulations": 0, "hyperparameter_selection": False}
    _save_json(base/"models.json", result)
    return artifact(base/"models.json")


def load_model(ref, *, expected_recipe=None, expected_dataset=None):
    # These are our own newly created, hash-authenticated local artifacts.
    with verify(ref).open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("recipe") not in CONFIG["recipes"] or payload.get("contract", {}).get("feature_config") != FEATURE_CONFIG:
        raise ValueError("Unknown ranker model")
    if expected_recipe is not None and payload["recipe"] != expected_recipe:
        raise ValueError("Ranker recipe differs")
    if expected_dataset is not None and payload["contract"]["dataset"] != _plain_ref(expected_dataset):
        raise ValueError("Ranker TRAIN identity differs")
    return payload
