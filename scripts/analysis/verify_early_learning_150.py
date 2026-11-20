"""Independent 150 target aggregation, training loop and descriptive statistics."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import math
import time

import torch

from splendor_ai.alphazero import run_early_learning as p
from splendor_ai.alphazero.afterstate_149 import AdvantageModel


def compare(actual, expected, path="root", tolerance=0.):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", tolerance)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}/{i}", tolerance)
    elif isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected), path
    elif isinstance(expected, float):
        assert math.isfinite(actual) and abs(actual-expected) <= tolerance, (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def read_development():
    output, sources = {}, set()
    for split, size in (("train", 768), ("validation", 192)):
        panel = p.old.read_json(p.PRIOR / f"{split}_panel.json")
        assert len(panel["samples"]) == size and panel["split"] == split
        arrays = torch.load(p.PRIOR / f"{split}_features.pt", weights_only=True)
        values = []
        for i, sample in enumerate(panel["samples"]):
            assert sample["index"] == i and sample["source_seed"] not in sources
            sources.add(sample["source_seed"])
            record = p.old.read_json(p.PRIOR / "targets" / split / f"state_{i:04d}.json")
            assert record["sample_sha256"] == p.old.digest(sample)
            assert record["index"] == i and record["split"] == split
            worlds = record["worlds"]
            assert len(worlds) == (32 if split == "train" else 64)
            n = len(sample["actions"])
            for world in worlds:
                assert len(world["outcomes"]) == n
                for outcome in world["outcomes"]:
                    assert outcome["reason"] in ("completed", "forced_pass_deadlock")
                    assert outcome["wdl"] in (-1., 0., 1.)
            means = [sum(w["outcomes"][j]["wdl"] for w in worlds)/len(worlds) for j in range(n)]
            values.append(means + [means[0]]*(4-n))
            assert arrays["mask"][i].tolist() == [0 < j < n for j in range(4)]
        arrays["q"] = torch.tensor(values, dtype=torch.float32)
        arrays["y"] = arrays["q"] - arrays["q"][:, :1]
        output[split] = arrays
    return output


def loss(scores, labels, mask):
    counts = mask.sum(-1)
    return ((scores-labels).square()*mask).sum(-1) / counts.clamp_min(1)


def independent_fit(arm, seed, train, valid, total=4096):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    net = AdvantageModel(train[arm].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.0001)
    stream = torch.Generator().manual_seed(seed+149000)
    history, selected, states = [], {}, {}
    minima = {"coarse": math.inf, "dense": math.inf}
    for step in range(total+1):
        if step < 129 or step % 128 == 0 or step == total:
            rng_before = torch.get_rng_state()
            sampler_before = stream.get_state()
            net.eval()
            with torch.no_grad():
                validation_loss = float(loss(net(valid[arm]), valid["y"], valid["mask"]).mean())
                training_loss = float(loss(net(train[arm]), train["y"], train["mask"]).mean())
            history.append(dict(update=step, validation_mse=validation_loss, training_mse=training_loss))
            candidates = ("coarse", "dense") if step % 128 == 0 or step == total else ("dense",)
            for method in candidates:
                if validation_loss < minima[method]:
                    minima[method] = validation_loss
                    selected[method] = step
                    states[method] = copy.deepcopy(net.state_dict())
            assert torch.equal(rng_before, torch.get_rng_state())
            assert torch.equal(sampler_before, stream.get_state())
        if step < total:
            net.train()
            batch = torch.randint(len(train["y"]), (64,), generator=stream)
            prediction = net(train[arm][batch])
            objective = loss(prediction, train["y"][batch], train["mask"][batch]).mean()
            opt.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            opt.step()
    return {"arm": arm, "seed": seed, "width": train[arm].shape[-1], "selected_updates": selected,
            "coarse_state": states["coarse"], "dense_state": states["dense"], "final_state": net.state_dict(),
            "history": history, "parameters": sum(t.numel() for t in net.parameters()),
            "evaluation_rng_unchanged": True}


def audit_fit(arm, seed, train, valid, info):
    from pathlib import Path
    assert p.old.file_hash(Path(info["path"])) == info["sha256"]
    rebuilt = independent_fit(arm, seed, train, valid)
    saved = torch.load(info["path"], weights_only=True)
    compare(rebuilt, saved)
    original = torch.load(p.PRIOR / "models" / f"{arm}_{seed}.pt", weights_only=True)
    assert [r for r in rebuilt["history"] if r["update"] % 128 == 0] == original["history"]
    assert rebuilt["selected_updates"]["coarse"] == original["selected_update"]
    compare(rebuilt["coarse_state"], original["selected_state"])
    compare(rebuilt["final_state"], original["final_state"])
    net = AdvantageModel(saved["width"])
    pred = {"seed": seed}
    for method in ("coarse", "dense"):
        net.load_state_dict(rebuilt[f"{method}_state"])
        net.eval()
        with torch.no_grad():
            advantages = net(valid[arm]).tolist()
        choices = [max([0]+[j for j in range(1, 4) if valid["mask"][i, j]], key=lambda j: scores[j])
                   for i, scores in enumerate(advantages)]
        pred[method] = {"selected_update": rebuilt["selected_updates"][method],
                        "advantages": advantages, "choices": choices}
    return arm, pred


def independent_stats(data, rows):
    q, y, masks = (data[key].tolist() for key in ("q", "y", "mask"))
    n = len(q)
    zero = sum(sum(target[j]**2 for j in range(4) if mask[j])/max(sum(mask), 1)
               for target, mask in zip(y, masks))/n
    result = dict(development_only=True, validation_reused_for_selection=True, confirmatory=False,
                  source_games=n, zero_mse=zero, arms={})
    for arm, seeds in rows.items():
        individuals = []
        for seed in seeds:
            row = {"seed": seed["seed"]}
            for method in ("coarse", "dense"):
                model = seed[method]
                errors, gains = [], []
                for i in range(n):
                    errors.append(sum((model["advantages"][i][j]-y[i][j])**2 for j in range(4) if masks[i][j])/max(sum(masks[i]), 1))
                    gains.append(q[i][model["choices"][i]]-q[i][0])
                row[method] = {"selected_update": model["selected_update"], "mse": sum(errors)/n,
                    "wdl_gain_over_035": sum(gains)/n, "baseline_fraction": model["choices"].count(0)/n}
            row["move_change_fraction"] = sum(c != d for c, d in zip(seed["coarse"]["choices"], seed["dense"]["choices"]))/n
            individuals.append(row)
        mean = {method: {key: sum(r[method][key] for r in individuals)/len(individuals)
                        for key in ("mse", "wdl_gain_over_035", "baseline_fraction")} for method in ("coarse", "dense")}
        early = sum(0 < r["dense"]["selected_update"] < 128 for r in individuals)
        changes = {"versus_coarse": 1-mean["dense"]["mse"]/mean["coarse"]["mse"], "versus_zero": 1-mean["dense"]["mse"]/zero}
        gates = {"early_seeds": early >= 2, "mse_versus_coarse": changes["versus_coarse"] >= .03,
                 "mse_versus_zero": changes["versus_zero"] >= .03,
                 "wdl_versus_035": mean["dense"]["wdl_gain_over_035"] >= .02,
                 "wdl_versus_coarse": mean["dense"]["wdl_gain_over_035"] > mean["coarse"]["wdl_gain_over_035"]}
        result["arms"][arm] = {"per_seed": individuals, "means": mean, "early_selected_seeds": early,
            "relative_mse_reduction": changes, "development_screen": gates, "screen_passed": all(gates.values()),
            "move_change_fraction": sum(r["move_change_fraction"] for r in individuals)/len(individuals)}
    result["any_development_screen_passed"] = any(r["screen_passed"] for r in result["arms"].values())
    return result


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    binding = p.check_identity()
    data = read_development()
    main_data = p.load_development()
    for split in data:
        compare(data[split], main_data[split])
    selection = p.old.read_json(p.OUTPUT / "selection_lock.json")
    assert selection["manifest"] == binding
    predictions = p.old.read_json(p.OUTPUT / "validation_predictions.json")
    result = p.old.read_json(p.OUTPUT / "result.json")
    compare(independent_stats(data["validation"], predictions), result, tolerance=1e-12)
    print({"audit": "all_reused_development_hashes_target_aggregation_and_statistics_passed"}, flush=True)
    models = selection["models"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(audit_fit, arm, seed, data["train"], data["validation"], models[f"{arm}_{seed}"])
                for arm in ("control", "afterstate") for seed in (149001, 149002, 149003)]
        for job in as_completed(jobs):
            arm, pred = job.result()
            expected = next(r for r in predictions[arm] if r["seed"] == pred["seed"])
            compare(pred, expected)
            print({"audit": "independent_full_fit_and_predictions_reproduced", "arm": arm, "seed": pred["seed"]}, flush=True)
    p.check_identity()
    p.old.save_new(p.OUTPUT / "audit.json", {"passed": True, "seconds": time.perf_counter()-started,
        "independent_full_fits": 6, "original_full_fits_reproduced": 6, "history_rows_per_fit": 160,
        "development_source_games": 960, "reused_target_files_verified": 960,
        "new_continuations": 0, "protected_test_accessed": False,
        "result_sha256": p.old.file_hash(p.OUTPUT / "result.json"),
        "predictions_sha256": p.old.file_hash(p.OUTPUT / "validation_predictions.json"),
        "selection_sha256": p.old.file_hash(p.OUTPUT / "selection_lock.json")})


if __name__ == "__main__":
    main()
