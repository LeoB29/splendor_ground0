"""Experiment 150: development-only dense early selection on fixed 149 fits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from splendor_ai.alphazero import run_afterstate_advantage as old
from splendor_ai.alphazero.afterstate_149 import AdvantageModel

OUTPUT = Path("outputs/early_learning_150")
PRIOR = old.OUTPUT
PROTOCOL = Path("docs/engineering/early_learning_150.md")
VERIFIER = Path("scripts/analysis/verify_early_learning_150.py")
PRIOR_SHA = "05fa10e2ea15b92b3670f6ba5876fd22cf6fe63979c2fc20c770e5e69cc7d188"
PRIOR_AUDIT_SHA = "8b67ea968a3b09ec226a681ccc212dd3496b00b5a03d0ed787942d01e1ebedbc"
CONFIG = {
    "arms": ["control", "afterstate"], "seeds": [149001, 149002, 149003],
    "updates": 4096, "early_end": 128, "coarse_interval": 128,
    "batch_states": 64, "batch_seed_offset": 149000,
    "learning_rate": .0003, "weight_decay": .0001, "gradient_norm": 1.,
    "workers": 6, "torch_threads": 1, "states": {"train": 768, "validation": 192},
    "minimum_early_seeds": 2, "minimum_mse_reduction": .03,
    "minimum_validation_wdl_gain": .02,
    "new_continuations": 0, "fresh_test": False, "arena": False,
    "development_only": True, "automatic_successor": False,
}


def validate_grid(update, total=4096):
    return update <= CONFIG["early_end"] or update % CONFIG["coarse_interval"] == 0 or update == total


def verify_files(files):
    for name, sha in files.items():
        if old.file_hash(Path(name)) != sha:
            raise ValueError(f"Frozen dependency/input changed: {name}")


def prior_inventory():
    """Follow explicit old inventories; never expand an old dynamic identity."""
    if old.file_hash(PRIOR / "manifest.json") != PRIOR_SHA:
        raise ValueError("149 manifest changed")
    if old.file_hash(PRIOR / "audit.json") != PRIOR_AUDIT_SHA:
        raise ValueError("149 successful audit changed")
    manifest = old.read_json(PRIOR / "manifest.json")
    for field in ("dependencies", "inputs"):
        verify_files(manifest[field])
    ancestor = old.read_json(old.PRIOR)
    for field in ("dependencies", "inputs"):
        verify_files(ancestor[field])
    audit = old.read_json(PRIOR / "audit.json")
    final = old.read_json(PRIOR / "final_status.json")
    if not (audit["passed"] and final["complete"] and final["audit_sha256"] == PRIOR_AUDIT_SHA
            and audit["result_sha256"] == final["result_sha256"]):
        raise ValueError("149 audit was not closed successfully")
    selected = old.read_json(PRIOR / "selection_lock.json")
    dev_path = PRIOR / "development_lock.json"
    if selected["manifest"] != old.digest(manifest) or old.file_hash(dev_path) != selected["development_lock"]:
        raise ValueError("149 selection binding changed")
    dev = old.read_json(dev_path)
    files = [PRIOR / name for name in ("manifest.json", "audit.json", "final_status.json",
                                      "selection_lock.json", "development_lock.json")]
    for split in CONFIG["states"]:
        for field, ext in (("panel", "json"), ("features", "pt")):
            path = PRIOR / f"{split}_{field}.{ext}"
            if old.file_hash(path) != dev[split][field]:
                raise ValueError("149 development binding changed")
            files.append(path)
        path = PRIOR / f"{split}_targets_lock.json"
        if old.file_hash(path) != selected["targets"][split]:
            raise ValueError("149 target lock changed")
        lock = old.read_json(path)
        if not lock["all_natural"] or len(lock["files"]) != CONFIG["states"][split]:
            raise ValueError("Incomplete development targets")
        files.append(path)
        for index, info in lock["files"].items():
            path = PRIOR / "targets" / split / f"state_{int(index):04d}.json"
            if not info["natural"] or old.file_hash(path) != info["sha256"]:
                raise ValueError("149 audited target changed")
            files.append(path)
    expected = {f"{arm}_{seed}" for arm in CONFIG["arms"] for seed in CONFIG["seeds"]}
    if set(selected["models"]) != expected:
        raise ValueError("Original six models are required")
    for info in selected["models"].values():
        path = Path(info["path"])
        if old.file_hash(path) != info["sha256"]:
            raise ValueError("Original model changed")
        files.append(path)
    return {"files": {p.as_posix(): old.file_hash(p) for p in files},
            "dependencies": manifest["dependencies"], "inputs": manifest["inputs"],
            "ancestor_dependencies": ancestor["dependencies"], "ancestor_inputs": ancestor["inputs"]}


def identity():
    files = [Path(__file__).relative_to(Path.cwd()), PROTOCOL, VERIFIER,
             Path("run_early_learning_150.py"), Path("tests/test_early_learning_150.py")]
    return {"format": "early_learning_150_v1", "config": CONFIG,
            "dependencies": {p.as_posix(): old.file_hash(p) for p in files},
            "prior": prior_inventory(), "python": sys.version,
            "torch": str(torch.__version__), "numpy": old.np.__version__}


def check_identity():
    saved = old.read_json(OUTPUT / "manifest.json")
    if saved != identity():
        raise ValueError("150 frozen identity changed")
    return old.digest(saved)


def load_development():
    data, sources = {}, set()
    for split, n in CONFIG["states"].items():
        panel = old.read_json(PRIOR / f"{split}_panel.json")
        if panel["split"] != split or len(panel["samples"]) != n:
            raise ValueError("Wrong development panel")
        for i, sample in enumerate(panel["samples"]):
            if sample["index"] != i or sample["source_seed"] in sources:
                raise ValueError("Source-game reuse or panel reorder")
            sources.add(sample["source_seed"])
        features = torch.load(PRIOR / f"{split}_features.pt", weights_only=True)
        q, features["y"] = old.target_data(panel, root=PRIOR)
        features["q"] = q
        if features["mask"].shape != (n, 4) or features["mask"][:, 0].any():
            raise ValueError("Invalid candidate mask")
        for arm in CONFIG["arms"]:
            if features[arm].shape != (n, 4, 1085) or not torch.isfinite(features[arm]).all():
                raise ValueError("Invalid features")
        data[split] = features
    return data


def require_equal_weights(left, right, label):
    if left.keys() != right.keys():
        raise ValueError(f"Tensor keys differ: {label}")
    for key in left:
        if not torch.equal(left[key], right[key]):
            raise ValueError(f"Tensor mismatch: {label}/{key}")


def fit_dense(arm, seed, training, validation, *, updates=None):
    """Preserve 149's operations and RNG streams, adding only evaluation."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = AdvantageModel(training[arm].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator().manual_seed(seed + CONFIG["batch_seed_offset"])
    total = CONFIG["updates"] if updates is None else updates
    best = {kind: float("inf") for kind in ("coarse", "dense")}
    chosen, weights, history = {}, {}, []
    for update in range(total + 1):
        if validate_grid(update, total):
            global_rng, batch_rng = torch.get_rng_state(), generator.get_state()
            model.eval()
            with torch.no_grad():
                val = float(old.state_loss(model(validation[arm]), validation["y"], validation["mask"]).mean())
                train = float(old.state_loss(model(training[arm]), training["y"], training["mask"]).mean())
            if not old.np.isfinite(val) or not old.np.isfinite(train):
                raise ValueError("Nonfinite fit")
            history.append({"update": update, "validation_mse": val, "training_mse": train})
            for kind in ("coarse", "dense"):
                allowed = kind == "dense" or update % CONFIG["coarse_interval"] == 0 or update == total
                if allowed and val < best[kind]:
                    best[kind], chosen[kind], weights[kind] = val, update, copy.deepcopy(model.state_dict())
            if not torch.equal(global_rng, torch.get_rng_state()) or not torch.equal(batch_rng, generator.get_state()):
                raise ValueError("Added evaluation consumed RNG")
        if update == total:
            break
        model.train()
        indices = torch.randint(len(training["y"]), (CONFIG["batch_states"],), generator=generator)
        pred = model(training[arm][indices])
        loss = old.state_loss(pred, training["y"][indices], training["mask"][indices]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_norm"], error_if_nonfinite=True)
        optimizer.step()
    return {"arm": arm, "seed": seed, "width": training[arm].shape[-1], "selected_updates": chosen,
            "coarse_state": weights["coarse"], "dense_state": weights["dense"],
            "final_state": model.state_dict(), "history": history,
            "parameters": sum(p.numel() for p in model.parameters()), "evaluation_rng_unchanged": True}


def verify_original(saved, reference, *, total=4096):
    for key in ("arm", "seed", "width", "parameters"):
        if saved[key] != reference[key]:
            raise ValueError(f"Original metadata mismatch: {key}")
    coarse = [r for r in saved["history"] if r["update"] % 128 == 0 or r["update"] == total]
    if coarse != reference["history"] or saved["selected_updates"]["coarse"] != reference["selected_update"]:
        raise ValueError("Original coarse history or selected update mismatch")
    require_equal_weights(saved["coarse_state"], reference["selected_state"], "original_selected")
    require_equal_weights(saved["final_state"], reference["final_state"], "original_final")


def fit_task(arm, seed, training, validation):
    started = time.perf_counter()
    fitted = fit_dense(arm, seed, training, validation)
    reference = torch.load(PRIOR / "models" / f"{arm}_{seed}.pt", weights_only=True)
    verify_original(fitted, reference)
    path = OUTPUT / "models" / f"{arm}_{seed}.pt"
    old.tensor_save(path, fitted)
    return {"path": path.as_posix(), "sha256": old.file_hash(path), "seconds": time.perf_counter()-started,
            "selected_updates": fitted["selected_updates"], "original_reproduced": True}


def predictions(data, model_rows):
    rows = {}
    for arm in CONFIG["arms"]:
        rows[arm] = []
        for seed in CONFIG["seeds"]:
            info = model_rows[f"{arm}_{seed}"]
            if old.file_hash(Path(info["path"])) != info["sha256"]:
                raise ValueError("New model changed")
            saved = torch.load(info["path"], weights_only=True)
            model = AdvantageModel(saved["width"])
            row = {"seed": seed}
            for kind in ("coarse", "dense"):
                model.load_state_dict(saved[f"{kind}_state"])
                model.eval()
                with torch.no_grad():
                    pred = model(data[arm])
                    legal = data["mask"].clone()
                    legal[:, 0] = True
                    choice = pred.masked_fill(~legal, -torch.inf).argmax(-1)
                row[kind] = {"selected_update": saved["selected_updates"][kind],
                             "advantages": pred.tolist(), "choices": choice.tolist()}
            rows[arm].append(row)
    return rows


def statistics(q, y, mask, rows):
    q, y = q.to(torch.float64), y.to(torch.float64)
    zero_mse = float(old.state_loss(torch.zeros_like(y), y, mask).mean())
    result = {"development_only": True, "validation_reused_for_selection": True,
              "confirmatory": False, "source_games": len(q), "zero_mse": zero_mse, "arms": {}}
    for arm, seeds in rows.items():
        individual = []
        for row in seeds:
            scores = {"seed": row["seed"]}
            for kind in ("coarse", "dense"):
                pred = torch.tensor(row[kind]["advantages"], dtype=torch.float64)
                choice = torch.tensor(row[kind]["choices"], dtype=torch.int64)
                scores[kind] = {"selected_update": row[kind]["selected_update"],
                    "mse": float(old.state_loss(pred, y, mask).mean()),
                    "wdl_gain_over_035": float((q[torch.arange(len(q)), choice]-q[:, 0]).mean()),
                    "baseline_fraction": float((choice == 0).double().mean())}
            scores["move_change_fraction"] = sum(a != b for a, b in zip(row["coarse"]["choices"], row["dense"]["choices"])) / len(q)
            individual.append(scores)
        means = {kind: {metric: sum(r[kind][metric] for r in individual)/len(individual)
                       for metric in ("mse", "wdl_gain_over_035", "baseline_fraction")} for kind in ("coarse", "dense")}
        early = sum(1 <= r["dense"]["selected_update"] <= 127 for r in individual)
        reductions = {"versus_coarse": 1-means["dense"]["mse"]/means["coarse"]["mse"],
                      "versus_zero": 1-means["dense"]["mse"]/zero_mse}
        gates = {"early_seeds": early >= CONFIG["minimum_early_seeds"],
                 "mse_versus_coarse": reductions["versus_coarse"] >= CONFIG["minimum_mse_reduction"],
                 "mse_versus_zero": reductions["versus_zero"] >= CONFIG["minimum_mse_reduction"],
                 "wdl_versus_035": means["dense"]["wdl_gain_over_035"] >= CONFIG["minimum_validation_wdl_gain"],
                 "wdl_versus_coarse": means["dense"]["wdl_gain_over_035"] > means["coarse"]["wdl_gain_over_035"]}
        result["arms"][arm] = {"per_seed": individual, "means": means, "early_selected_seeds": early,
            "relative_mse_reduction": reductions, "development_screen": gates,
            "screen_passed": all(gates.values()),
            "move_change_fraction": sum(r["move_change_fraction"] for r in individual)/len(individual)}
    result["any_development_screen_passed"] = any(r["screen_passed"] for r in result["arms"].values())
    return result


def prepare():
    started = time.perf_counter()
    manifest = identity()
    old.save_new(OUTPUT / "manifest.json", manifest)
    data = load_development()
    old.save_new(OUTPUT / "preflight.json", {"passed": True, "manifest": old.digest(manifest),
        "time": old.now(), "seconds": time.perf_counter()-started,
        "states": {s: len(d["y"]) for s, d in data.items()}, "new_continuations": 0, "new_fits": 0})
    print({"stage": "inputs_frozen", "manifest": old.digest(manifest)}, flush=True)


def run():
    binding = check_identity()
    preflight = old.read_json(OUTPUT / "preflight.json")
    if not preflight["passed"] or preflight["manifest"] != binding:
        raise ValueError("Preflight missing or changed")
    started = time.perf_counter()
    old.save_new(OUTPUT / "launch.json", {"started": old.now(), "manifest": binding})
    data = load_development()
    rows = {}
    print({"stage": "six_fixed_fits_started", "new_validation_checks_per_fit": 127}, flush=True)
    with ProcessPoolExecutor(max_workers=CONFIG["workers"]) as pool:
        jobs = {pool.submit(fit_task, arm, seed, data["train"], data["validation"]): f"{arm}_{seed}"
                for arm in CONFIG["arms"] for seed in CONFIG["seeds"]}
        for job in as_completed(jobs):
            rows[jobs[job]] = job.result()
            print({"stage": "fit_and_original_reproduction_complete", "model": jobs[job], **rows[jobs[job]]}, flush=True)
    old.save_new(OUTPUT / "selection_lock.json", {"manifest": binding, "locked_at": old.now(), "models": rows})
    pred = predictions(data["validation"], rows)
    old.save_new(OUTPUT / "validation_predictions.json", pred)
    result = statistics(data["validation"]["q"], data["validation"]["y"], data["validation"]["mask"], pred)
    old.save_new(OUTPUT / "result.json", result)
    print({"stage": "development_analysis_complete", "seconds": time.perf_counter()-started,
           "any_screen_passed": result["any_development_screen_passed"]}, flush=True)
    subprocess.run([sys.executable, str(VERIFIER)], check=True)
    audit = old.read_json(OUTPUT / "audit.json")
    if not audit["passed"] or audit["result_sha256"] != old.file_hash(OUTPUT / "result.json"):
        raise ValueError("Audit failed")
    check_identity()
    old.save_new(OUTPUT / "final_status.json", {"complete": True, "finished": old.now(),
        "seconds": time.perf_counter()-started, "development_only": True,
        "development_screen_passed": result["any_development_screen_passed"],
        "result_sha256": old.file_hash(OUTPUT / "result.json"), "audit_sha256": old.file_hash(OUTPUT / "audit.json"),
        "new_continuations": 0, "fresh_test": False, "arena": False, "promotion": False,
        "automatic_successor": False})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    with old.process_lock(OUTPUT / "run.lock"):
        try:
            prepare() if args.mode == "prepare" else run()
        except Exception:
            old.atomic_write(OUTPUT / f"failure_{args.mode}.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
