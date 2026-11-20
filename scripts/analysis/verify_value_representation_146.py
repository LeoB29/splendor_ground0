"""146 audit: sealed inputs, deterministic fits, all new trajectories and estimates."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import runpy
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_value_representation as exp

_AUDIT = _MODELS = None


def initialize():
    global _AUDIT, _MODELS
    torch.set_num_threads(1)
    _AUDIT = runpy.run_path("scripts/analysis/verify_value_generalization_scale_143.py")
    _AUDIT["initialize"]()
    _MODELS = [exp.load_value(n) for n in ("frozen", "trainable")]


def audit_game(index, binding, summary, selection_sha):
    path = exp.OUTPUT / "test" / "games" / f"game_{index:04d}.json"
    record = exp.read_json(path)
    assert exp.file_hash(path) == summary[path.name]
    assert record["seed"] == 1_460_000_000 + index
    assert record["binding"] == binding and record["split"] == "protected_test" and record["experiment"] == 146
    checked = _AUDIT["audit_one"](index, exp.OUTPUT / "test", protected=False)
    if not checked["valid"]:
        return checked, None
    data = torch.load(path.with_suffix(".pt"), weights_only=True)
    saved = torch.load(exp.OUTPUT / "predictions" / path.with_suffix(".pt").name, weights_only=True)
    assert saved["index"] == index and saved["selection_sha256"] == selection_sha
    with torch.no_grad():
        baseline = torch.matmul(data["simple"].to(torch.float64), _MODELS[0].coefficients).clamp(-1, 1)
        estimates = [torch.zeros_like(baseline), baseline]
        for model in _MODELS:
            correction = model.head(model.encoder(data["observations"])).flatten().to(torch.float64)
            estimates.append(torch.clamp(baseline + correction, min=-1., max=1.))
        actual = torch.stack(estimates, dim=1)
    assert torch.equal(actual, saved["values"])
    error = (actual.numpy() - data["targets"].numpy()[:, None])**2
    turns = data["turns"].numpy()
    masks = {"early_middle": turns < 40, "overall": turns >= 0, "late": turns >= 40,
             "early": turns < 20, "middle": (turns >= 20) & (turns < 40)}
    return checked, {p: error[mask].mean(axis=0).tolist() for p, mask in masks.items() if mask.any()}


def verify_training():
    train = exp.load_development(list(range(4096)))
    val = exp.load_development(list(range(4096, 4608)))
    results = {}
    initial = exp.ResidualValue(False)
    for name in ("frozen", "trainable"):
        report = exp.read_json(exp.OUTPUT / f"{name}_training.json")
        assert report["training_ids"] == list(range(4096)) and report["validation_ids"] == list(range(4096, 4608))
        best, step, stale = report["history"][0]["validation_game_mse"], 0, 0
        for n, item in enumerate(report["history"][1:], 1):
            assert stale < 4 and item["step"] == n*128
            selected = item["validation_game_mse"] < best-.001
            assert item["selected"] == selected
            if selected:
                best, step, stale = item["validation_game_mse"], item["step"], 0
            else:
                stale += 1
        assert stale == 4 or report["completed_updates"] == 4096
        assert report["selected_step"] == step and report["selected_validation_game_mse"] == best
        actual, replay = exp.fit(name, train, val)
        assert all(report[key] == value for key, value in replay.items())
        saved = exp.load_value(name)
        assert all(torch.equal(value, saved.state_dict()[key]) for key, value in actual.state_dict().items())
        encoder_changed = any(not torch.equal(value, initial.encoder.state_dict()[key]) for key, value in saved.encoder.state_dict().items())
        assert not encoder_changed if name == "frozen" else encoder_changed == (step > 0)
        assert torch.equal(saved.coefficients, initial.coefficients)
        results[name] = {"updates_reproduced": report["completed_updates"], "selected_step": step,
                         "encoder_changed": encoder_changed, "all_selected_tensors_exact": True}
        print(json.dumps({"stage": "audit_training", "name": name, **results[name]}), flush=True)
    return results


def independent_stats(rows, actual):
    # Same seeded whole-game bootstrap, different chunking and aggregation.
    for phase, data in rows.items():
        values = np.array(data, dtype=np.float64)
        means = values.mean(0)
        assert actual[phase]["games"] == len(values)
        for i, name in enumerate(exp.NAMES):
            assert abs(means[i]-actual[phase]["mse"][name]) < 1e-12
        rng = np.random.default_rng(146999)
        bootstrap = np.concatenate([values[rng.integers(0, len(values), (1000, len(values)))].mean(axis=1)
                                    for _ in range(50)])
        for i, comparator in ((1, "public"), (2, "frozen")):
            assert abs(actual[phase]["contrasts"][comparator]["difference"] - (means[3]-means[i])) < 1e-12
            assert abs(actual[phase]["contrasts"][comparator]["relative_reduction"] - (1-means[3]/means[i])) < 1e-12
            delta = bootstrap[:, 3]-bootstrap[:, i]
            ci = np.percentile(delta, [2.5, 97.5])
            assert np.allclose(ci, actual[phase]["contrasts"][comparator]["ci95"], atol=1e-12, rtol=0)


def verify(finalize_result=True):
    started = time.perf_counter()
    torch.set_num_threads(1)
    manifest = exp.read_json(exp.OUTPUT / "manifest.json")
    assert exp.identity() == manifest
    binding = exp.digest(manifest)
    exp.check_lock(manifest)
    result = exp.read_json(exp.OUTPUT / "result.json")
    summary = exp.read_json(exp.OUTPUT / "test_summary.json")
    assert result["manifest"] == summary["manifest"] == binding
    assert result["test_summary_sha256"] == exp.file_hash(exp.OUTPUT / "test_summary.json")
    assert result["selection_sha256"] == exp.file_hash(exp.OUTPUT / "selection_lock.json")
    training = verify_training()
    rows = {}
    with ProcessPoolExecutor(max_workers=8, initializer=initialize) as pool:
        jobs = [pool.submit(audit_game, i, binding, summary["files"], result["selection_sha256"]) for i in range(512)]
        for n, job in enumerate(as_completed(jobs), 1):
            check, errors = job.result()
            rows[check["index"]] = (check, errors)
            if n % 64 == 0:
                print(json.dumps({"stage": "audit_prediction_games", "completed": n}), flush=True)
    ordered = [rows[i] for i in range(512)]
    errors = {p: [r[1][p] for r in ordered if r[1] and p in r[1]] for p in exp.PHASES}
    for phase in errors:
        assert np.allclose(errors[phase], result["game_error_rows"][phase], atol=1e-12, rtol=0)
    independent_stats(errors, result["statistics"])
    cost = exp.read_json(exp.OUTPUT / "cost.json")
    for name, values in cost["seconds"].items():
        assert len(values) == cost["states"]*3 and all(np.isfinite(v) and v > 0 for v in values)
    for label, percentile in (("median", 50), ("p95", 95)):
        ratio = float(np.percentile(cost["seconds"]["trainable"], percentile)/np.percentile(cost["seconds"]["public"], percentile))
        assert ratio == cost["ratio"][label]
    assert cost["passed"] == (cost["ratio"]["median"] <= 1.5 and cost["ratio"]["p95"] <= 2.)
    valid = sum(r[0]["valid"] for r in ordered)
    assert valid == result["valid_games"] == summary["valid_games"]
    s = result["statistics"]
    em = s["early_middle"]["contrasts"]
    checks = {
        "all_test_games_natural": valid == 512,
        "phase_coverage": all(s[p]["games"] == 512 for p in ("early_middle", "overall", "late")),
        "early_middle_public": em["public"]["relative_reduction"] >= .03 and em["public"]["ci95"][1] < 0,
        "early_middle_frozen": em["frozen"]["relative_reduction"] >= .01 and em["frozen"]["ci95"][1] < 0,
        "nonregression": all(s[p]["contrasts"][c]["ci95"][1] < .01 for p in ("overall", "late") for c in ("public", "frozen")),
        "inference_cost": cost["passed"],
    }
    assert result["gate"] == {"checks": checks, "passed": all(checks.values())}
    base = exp.PublicValueInference()
    models = [exp.ResidualInference(exp.load_value(name), base) for name in ("frozen", "trainable")]
    policy_positions = 0
    for state, legal in exp.probe_states():
        expected = base.predict_one(state, legal).logits
        assert all(model.predict_one(state, legal).logits == expected for model in models)
        policy_positions += 1
    assert cost["source_policy_sha256"] == exp.policy_hash(base.model)
    assert exp.identity() == manifest
    report = {"passed": True, "manifest": binding, "result_sha256": exp.file_hash(exp.OUTPUT / "result.json"),
              "training_reproduction": training, "games": len(rows), "valid_games": valid,
              "positions": sum(r[0]["positions"] for r in ordered), "policy_positions": policy_positions,
              "all_behavior_features_labels_predictions_verified": True,
              "independent_statistics_and_gate_verified": True, "dependencies": len(manifest["dependencies"]),
              "elapsed_seconds": time.perf_counter()-started}
    exp.atomic_write(exp.OUTPUT / "prediction_verification.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)
    if finalize_result:
        finalize()


def finalize():
    manifest = exp.read_json(exp.OUTPUT / "manifest.json")
    assert exp.identity() == manifest
    exp.check_lock(manifest)
    result = exp.read_json(exp.OUTPUT / "result.json")
    audit = exp.read_json(exp.OUTPUT / "prediction_verification.json")
    assert audit["passed"] and audit["result_sha256"] == exp.file_hash(exp.OUTPUT / "result.json")
    arena = exp.OUTPUT / "arena"
    if result["gate"]["passed"]:
        arena_status = exp.read_json(arena / "final_status.json")
        assert arena_status["status"] == "complete"
        decision = arena_status["decision"]
    else:
        assert not (arena / "games").exists() and not (arena / "manifest.json").exists()
        decision = "prediction_gate_failed_stop"
    exp.atomic_write(exp.OUTPUT / "final_status.json", {"status": "complete", "decision": decision,
        "manifest": exp.digest(manifest), "result_sha256": audit["result_sha256"],
        "verification_sha256": exp.file_hash(exp.OUTPUT / "prediction_verification.json"),
        "conditional_arena_ran": result["gate"]["passed"], "promotion": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="store_true")
    args = parser.parse_args()
    verify(finalize_result=not args.prediction)
