"""Separate-process 151 feature, fit, statistics and conditional-rollout audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import importlib.util
from pathlib import Path
import random
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_shared_cards as p
from splendor_ai.alphazero.afterstate_149 import CARDS, features
from splendor_ai.engine.constants import TOKEN_COLORS


def load_auditor(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a150 = load_auditor("previous_audit150", "scripts/analysis/verify_early_learning_150.py")
a149 = load_auditor("previous_audit149", "scripts/analysis/verify_afterstate_advantage_149.py")


def verify_features(panel, data):
    for i, sample in enumerate(panel["samples"]):
        assert data["mask"][i].tolist() == [0 < j < len(sample["actions"]) for j in range(4)]
        state = p.old.deserialize_state_snapshot(sample["snapshot"])
        flat = [features(state, p.old.decode_action(action), "afterstate") for action in sample["actions"]]
        flat += [flat[0]]*(4-len(flat))
        flat = torch.stack(flat)
        assert torch.equal(flat, data["flat"][i]), (i, "original_feature")
        for j, vector in enumerate(flat):
            assert torch.equal(torch.cat((vector[:295], vector[925:])), data["context"][i, j])
            for k, card in enumerate(CARDS):
                row = [card.cost.get(c, 0)/7 for c in TOKEN_COLORS]
                row += [float(card.bonus_color == c) for c in TOKEN_COLORS]
                row += [card.points/5]+[float(card.tier == t) for t in (1, 2, 3)]
                row += vector[295+6*k:301+6*k].tolist()+[float(vector[835+k]), float(vector[-101+k])]
                assert torch.equal(torch.tensor(row, dtype=torch.float32), data["cards"][i, j, k]), (i, j, k)


def independent_fit(seed, train, val, total=4096):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = p.SharedCardModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    rng = torch.Generator().manual_seed(seed+149000)
    history, best, selected, weights = [], float("inf"), 0, None
    for step in range(total+1):
        if step < 129 or step % 128 == 0 or step == total:
            before, batches = torch.get_rng_state(), rng.get_state()
            model.eval()
            with torch.no_grad():
                prediction = model(val["context"], val["cards"])
                v = float(a150.loss(prediction, val["y"], val["mask"]).mean())
                prediction = model(train["context"], train["cards"])
                t = float(a150.loss(prediction, train["y"], train["mask"]).mean())
            history.append({"update": step, "validation_mse": v, "training_mse": t})
            if v < best:
                best, selected, weights = v, step, copy.deepcopy(model.state_dict())
            assert torch.equal(before, torch.get_rng_state()) and torch.equal(batches, rng.get_state())
        if step < total:
            model.train()
            batch = torch.randint(len(train["y"]), (64,), generator=rng)
            output = model(train["context"][batch], train["cards"][batch])
            objective = a150.loss(output, train["y"][batch], train["mask"][batch]).mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if total == 4096 and (step+1) % 512 == 0:
                print({"audit": "refit_progress", "seed": seed, "updates": step+1}, flush=True)
    return {"seed": seed, "parameters": 327297, "selected_update": selected,
            "selected_state": weights, "final_state": model.state_dict(), "history": history,
            "evaluation_rng_unchanged": True}


def audit_fit(seed, train, val, info):
    assert p.old.file_hash(Path(info["path"])) == info["sha256"]
    fitted = independent_fit(seed, train, val)
    a150.compare(fitted, torch.load(info["path"], weights_only=True))
    return seed


def independent_predictions(data, models):
    result = {"flat": [], "shared": []}
    for arm in result:
        for seed in (149001, 149002, 149003):
            if arm == "shared":
                info = models[str(seed)]
                assert p.old.file_hash(Path(info["path"])) == info["sha256"]
                saved = torch.load(info["path"], weights_only=True)
                model = p.SharedCardModel()
                model.load_state_dict(saved["selected_state"])
                update = saved["selected_update"]
            else:
                saved = torch.load(p.parent.OUTPUT/"models"/f"afterstate_{seed}.pt", weights_only=True)
                model = p.old.AdvantageModel(1085)
                model.load_state_dict(saved["dense_state"])
                update = saved["selected_updates"]["dense"]
            model.eval()
            with torch.no_grad():
                values = (model(data["flat"]) if arm == "flat" else model(data["context"], data["cards"])).tolist()
            choices = [max([0]+[j for j in range(1, 4) if data["mask"][i, j]], key=lambda j: v[j]) for i, v in enumerate(values)]
            result[arm].append({"seed": seed, "selected_update": update, "advantages": values, "choices": choices})
    return result


def independent_metrics(data, predictions):
    n = len(data["q"])
    q, y, masks = (data[k].tolist() for k in ("q", "y", "mask"))
    zero = sum(sum(t[j]**2 for j in range(4) if m[j])/max(sum(m), 1) for t, m in zip(y, masks))/n
    result = {"zero_mse": zero, "arms": {}}
    for arm in predictions:
        rows = []
        for model in predictions[arm]:
            errors = [sum((model["advantages"][i][j]-y[i][j])**2 for j in range(4) if masks[i][j])/max(sum(masks[i]), 1) for i in range(n)]
            gains = [q[i][model["choices"][i]]-q[i][0] for i in range(n)]
            rows.append({"seed": model["seed"], "selected_update": model["selected_update"],
                "mse": sum(errors)/n, "wdl_gain_over_035": sum(gains)/n, "baseline_fraction": model["choices"].count(0)/n})
        result["arms"][arm] = {"per_seed": rows,
            "means": {k: sum(r[k] for r in rows)/len(rows) for k in ("mse", "wdl_gain_over_035", "baseline_fraction")}}
    return result


def independent_development_statistics(data, predictions):
    result = independent_metrics(data, predictions)
    s, f = (result["arms"][a]["means"] for a in ("shared", "flat"))
    reduction = {"versus_flat": 1-s["mse"]/f["mse"], "versus_zero": 1-s["mse"]/result["zero_mse"]}
    gates = {"mse_versus_flat": reduction["versus_flat"] >= .03,
             "mse_versus_zero": reduction["versus_zero"] >= .03,
             "wdl_versus_035": s["wdl_gain_over_035"] >= .02,
             "wdl_versus_flat": s["wdl_gain_over_035"] > f["wdl_gain_over_035"]}
    return {**result, "relative_mse_reduction": reduction, "development_gates": gates,
            "development_passed": all(gates.values()), "development_only": True, "confirmatory": False}


def verify_fresh_statistics(data, predictions, saved):
    n = len(data["q"])
    q, y, masks = (data[k].tolist() for k in ("q", "y", "mask"))
    per_state = []
    for i in range(n):
        comparisons = []
        for k in range(3):
            s, f = predictions["shared"][k], predictions["flat"][k]
            sq, fq = q[i][s["choices"][i]], q[i][f["choices"][i]]
            se, fe = [sum((m["advantages"][i][j]-y[i][j])**2 for j in range(4) if masks[i][j])/max(sum(masks[i]), 1) for m in (s, f)]
            comparisons.append([sq-q[i][0], sq-fq, se-fe])
        per_state.append(np.mean(comparisons, axis=0))
    matrix = np.asarray(per_state)
    rng = np.random.default_rng(151999)
    resampled = np.concatenate([matrix[rng.integers(n, size=(1000, n))].mean(1) for _ in range(50)])
    interval = np.quantile(resampled, [.025, .975], axis=0)
    names = ("shared_minus_035", "shared_minus_flat", "mse_shared_minus_flat")
    contrasts = {name: {"mean": float(matrix[:, i].mean()), "ci95": interval[:, i].tolist(),
                       "half_width": float((interval[1, i]-interval[0, i])/2)} for i, name in enumerate(names)}
    gates = {name: contrasts[name]["mean"] >= .02 and contrasts[name]["ci95"][0] > 0 and contrasts[name]["half_width"] <= .02 for name in names[:2]}
    gates["mse"] = contrasts[names[2]]["ci95"][1] < 0
    expected = {**independent_metrics(data, predictions), "contrasts": contrasts, "fresh_gates": gates,
                "fresh_passed": all(gates.values()), "conditional_move_evidence_only": True}
    a150.compare(expected, saved, tolerance=1e-12)


def verify_development():
    data = p.development_data()
    originals = a150.read_development()
    for split in data:
        for key in ("mask", "q", "y"):
            assert torch.equal(originals[split][key], data[split][key])
        assert torch.equal(originals[split]["afterstate"], data[split]["flat"])
        panel = p.old.read_json(p.old.OUTPUT/f"{split}_panel.json")
        verify_features(panel, data[split])
    selection = p.old.read_json(p.OUTPUT/"selection_lock.json")
    assert selection["manifest"] == p.check_identity()
    assert selection["development_lock_sha256"] == p.old.file_hash(p.OUTPUT/"development_lock.json")
    rows = independent_predictions(data["validation"], selection["models"])
    a150.compare(rows, p.old.read_json(p.OUTPUT/"validation_predictions.json"))
    # Fixed flat controls must reproduce their prior validation predictions exactly.
    previous = p.old.read_json(p.parent.OUTPUT/"validation_predictions.json")["afterstate"]
    for current, old in zip(rows["flat"], previous):
        assert current["seed"] == old["seed"]
        a150.compare({k: current[k] for k in ("selected_update", "advantages", "choices")}, old["dense"])
    a150.compare(independent_development_statistics(data["validation"], rows),
                  p.old.read_json(p.OUTPUT/"development_result.json"), tolerance=1e-12)
    print({"audit": "development_features_controls_predictions_statistics_passed"}, flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(audit_fit, seed, data["train"], data["validation"], selection["models"][str(seed)]) for seed in (149001,149002,149003)]
        for job in as_completed(jobs):
            print({"audit": "full_fit_reproduced", "seed": job.result()}, flush=True)
    return {"source_states": 960, "complete_fits_reproduced": 3, "history_rows_per_fit": 160,
            "result_sha256": p.old.file_hash(p.OUTPUT/"development_result.json"),
            "predictions_sha256": p.old.file_hash(p.OUTPUT/"validation_predictions.json"),
            "selection_sha256": p.old.file_hash(p.OUTPUT/"selection_lock.json")}


def audit_sample(split, sample):
    state = p.old.deserialize_state_snapshot(sample["snapshot"])
    path = p.OUTPUT/"targets"/split/f"state_{sample['index']:04d}.json"
    record = p.old.read_json(path)
    lock = p.old.read_json(p.OUTPUT/f"{split}_targets_lock.json")
    assert p.old.file_hash(path) == lock["files"][str(sample["index"])]["sha256"]
    assert record["sample_sha256"] == p.old.digest(sample)
    expected_n = 4 if split == "probe" else 128
    assert len(record["worlds"]) == expected_n
    count, moves = 0, 0
    for i, saved in enumerate(record["worlds"]):
        seed = (1519100000 if split == "probe" else 1511000000)+sample["index"]*1000+i
        assert saved["seed"] == seed
        world = p.determinize_for_observer(state, state.current_player, random.Random(seed))
        assert saved["world_sha256"] == p.old.digest(p.old.serialize_state_snapshot(world))
        assert len(saved["outcomes"]) == len(sample["actions"])
        for j, action in enumerate(sample["actions"]):
            rebuilt = a149.replay_branch(world, p.old.decode_action(action), state.current_player)
            assert rebuilt == saved["outcomes"][j], (split, sample["index"], i, j)
            assert rebuilt["reason"] in p.old.VALID_NATURAL_TERMINATIONS
            count += 1
            moves += rebuilt["turns"]
    return count, moves


def verify_fresh():
    dev = p.old.read_json(p.OUTPUT/"development_result.json")
    audit = p.old.read_json(p.OUTPUT/"development_audit.json")
    assert dev["development_passed"] and audit["passed"]
    assert audit["result_sha256"] == p.old.file_hash(p.OUTPUT/"development_result.json")
    p.old.initialize_worker()
    panel = p.old.read_json(p.OUTPUT/"test_panel.json")
    assert p.old.digest(p.fresh_panel(audit=True)) == p.old.digest(panel)
    assert len(panel["samples"]) == len({s["source_seed"] for s in panel["samples"]}) == 384
    selection_path = p.OUTPUT/"selection_lock.json"
    for source in panel["filtering"]:
        assert Path(source["path"]).stat().st_mtime_ns > selection_path.stat().st_mtime_ns
    data = torch.load(p.OUTPUT/"test_data.pt", weights_only=True)
    verify_features(panel, data)
    choices_path = p.OUTPUT/"test_choices_lock.json"
    choices = p.old.read_json(choices_path)
    for field, path in (("selection_sha256", selection_path), ("panel_sha256", p.OUTPUT/"test_panel.json"), ("data_sha256", p.OUTPUT/"test_data.pt")):
        assert choices[field] == p.old.file_hash(path)
    rows = independent_predictions(data, p.old.read_json(selection_path)["models"])
    a150.compare(rows, choices["predictions"])
    for path in (p.OUTPUT/"targets"/"test").glob("state_*.json"):
        assert path.stat().st_mtime_ns > choices_path.stat().st_mtime_ns
    data["q"], data["y"] = p.old.target_data(panel, root=p.OUTPUT)
    means = []
    for sample in panel["samples"]:
        record = p.old.read_json(p.OUTPUT/"targets"/"test"/f"state_{sample['index']:04d}.json")
        assert len(record["worlds"]) == 128
        row = [sum(w["outcomes"][j]["wdl"] for w in record["worlds"])/128 for j in range(len(sample["actions"]))]
        means.append(row+[row[0]]*(4-len(row)))
    independent_q = torch.tensor(means, dtype=torch.float32)
    assert torch.equal(independent_q, data["q"])
    assert torch.equal(independent_q-independent_q[:, :1], data["y"])
    verify_fresh_statistics(data, rows, p.old.read_json(p.OUTPUT/"fresh_result.json"))
    probe = p.old.read_json(p.OUTPUT/"probe_panel.json")
    assert probe == p.old.read_json(p.old.OUTPUT/"probe_panel.json")
    count = moves = 0
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        jobs = [pool.submit(audit_sample, split, sample) for split, block in (("probe", probe), ("test", panel)) for sample in block["samples"]]
        for i, job in enumerate(as_completed(jobs), 1):
            c, m = job.result()
            count += c
            moves += m
            if i % 8 == 0 or i == len(jobs):
                print({"audit": "fresh_continuations", "states": i, "total": len(jobs)}, flush=True)
    assert count <= 196608+96
    return {"continuations": count, "moves": moves, "fresh_source_states": 384,
            "result_sha256": p.old.file_hash(p.OUTPUT/"fresh_result.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("development", "fresh"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    p.check_identity()
    details = verify_development() if args.stage == "development" else verify_fresh()
    p.check_identity()
    p.old.save_new(p.OUTPUT/f"{args.stage}_audit.json", {"passed": True, "seconds": time.perf_counter()-started, **details})


if __name__ == "__main__":
    main()
