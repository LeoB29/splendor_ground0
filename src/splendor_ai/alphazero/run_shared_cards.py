"""Frozen 151 shared-card pilot, with a gated fresh conditional evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import run_early_learning as parent
from splendor_ai.alphazero.shared_cards_151 import SharedCardModel, restructure, reconstruct, PARAMETERS
from splendor_ai.bots.hidden_information import determinize_for_observer

old = parent.old
OUTPUT = Path("outputs/shared_cards_151")
PROTOCOL = Path("docs/engineering/shared_cards_151.md")
VERIFIER = Path("scripts/analysis/verify_shared_cards_151.py")
PARENT_SHA = "8a1fe332890fcc939c69ca08feaf40a27eb9b739562a4b89af4a2f4f60484f57"
AUDIT_SHA = "26a3cff7145f60715556515a5830efd60ad916da371975e6bdedf740697dc0ad"
CONFIG = {
    "seeds": [149001, 149002, 149003], "updates": 4096, "batch_states": 64,
    "batch_seed_offset": 149000, "learning_rate": .0003, "weight_decay": .0001,
    "gradient_norm": 1., "validation_grid": "0..128 then every128 through4096",
    "fit_workers": 3, "collection_workers": 8, "torch_threads": 1,
    "parameters": PARAMETERS, "flat_parameters": 320001,
    "development_states": {"train": 768, "validation": 192},
    "dev_mse_reduction": .03, "minimum_wdl_gain": .02,
    "fresh_states": 384, "fresh_worlds": 128, "source_limit": 2048,
    "source_seed_start": 1_510_000_000, "world_seed_start": 1_511_000_000,
    "probe_world_start": 1_519_100_000, "probe_worlds": 4,
    "bootstrap_seed": 151999, "bootstrap_samples": 50000, "max_ci_half_width": .02,
    "fresh_continuation_limit": 196608, "probe_continuation_limit": 96,
    "no_arena": True, "no_promotion": True, "automatic_successor": False,
}


def parent_inputs():
    if old.file_hash(parent.OUTPUT/"manifest.json") != PARENT_SHA:
        raise ValueError("150 manifest changed")
    if old.file_hash(parent.OUTPUT/"audit.json") != AUDIT_SHA:
        raise ValueError("150 audit changed")
    binding = parent.check_identity()
    audit = old.read_json(parent.OUTPUT/"audit.json")
    final = old.read_json(parent.OUTPUT/"final_status.json")
    selection_path = parent.OUTPUT/"selection_lock.json"
    if not (final["complete"] and audit["passed"] and final["audit_sha256"] == AUDIT_SHA
            and old.file_hash(selection_path) == audit["selection_sha256"]
            and final["result_sha256"] == audit["result_sha256"] == old.file_hash(parent.OUTPUT/"result.json")
            and audit["predictions_sha256"] == old.file_hash(parent.OUTPUT/"validation_predictions.json")):
        raise ValueError("150 was not closed and bound successfully")
    selection = old.read_json(selection_path)
    if selection["manifest"] != binding:
        raise ValueError("150 selection mismatch")
    paths = [parent.OUTPUT/n for n in ("manifest.json", "audit.json", "final_status.json", "selection_lock.json",
                                      "result.json", "validation_predictions.json")]
    for seed in CONFIG["seeds"]:
        row = selection["models"][f"afterstate_{seed}"]
        path = Path(row["path"])
        if old.file_hash(path) != row["sha256"]:
            raise ValueError("Flat control changed")
        paths.append(path)
    paths.append(old.OUTPUT/"probe_panel.json")
    return {"binding": binding, "files": {p.as_posix(): old.file_hash(p) for p in paths}}


def identity():
    paths = [Path(__file__).relative_to(Path.cwd()), Path("src/splendor_ai/alphazero/shared_cards_151.py"),
             PROTOCOL, VERIFIER, Path("run_shared_cards_151.py"), Path("tests/test_shared_cards_151.py")]
    return {"format": "shared_cards_151_v1", "config": CONFIG, "parent": parent_inputs(),
            "dependencies": {p.as_posix(): old.file_hash(p) for p in paths},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def check_identity():
    manifest = old.read_json(OUTPUT/"manifest.json")
    if manifest != identity():
        raise ValueError("151 frozen identity changed")
    return old.digest(manifest)


def structured_data(flat_data):
    context, cards = restructure(flat_data["afterstate"])
    if not torch.equal(reconstruct(context, cards), flat_data["afterstate"]):
        raise ValueError("Information was lost in restructuring")
    return {"context": context, "cards": cards, "flat": flat_data["afterstate"],
            **{k: flat_data[k] for k in ("mask", "q", "y") if k in flat_data}}


def development_data():
    lock = old.read_json(OUTPUT/"development_lock.json")
    result = {}
    for split in CONFIG["development_states"]:
        path = OUTPUT/f"{split}_data.pt"
        if old.file_hash(path) != lock[split]:
            raise ValueError("151 development arrays changed")
        result[split] = torch.load(path, weights_only=True)
    return result


def forward(model, data, indices=None):
    return model(data["context"], data["cards"]) if indices is None else model(data["context"][indices], data["cards"][indices])


def fit(seed, train, validation, *, updates=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = SharedCardModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed+149000)
    best, selected, weights, history = float("inf"), 0, None, []
    total = CONFIG["updates"] if updates is None else updates
    for update in range(total+1):
        if parent.validate_grid(update, total):
            rng, batch_rng = torch.get_rng_state(), generator.get_state()
            model.eval()
            with torch.no_grad():
                val = float(old.state_loss(forward(model, validation), validation["y"], validation["mask"]).mean())
                training = float(old.state_loss(forward(model, train), train["y"], train["mask"]).mean())
            if not np.isfinite(val) or not np.isfinite(training):
                raise ValueError("Nonfinite training")
            history.append({"update": update, "validation_mse": val, "training_mse": training})
            if val < best:
                best, selected, weights = val, update, copy.deepcopy(model.state_dict())
            if not torch.equal(rng, torch.get_rng_state()) or not torch.equal(batch_rng, generator.get_state()):
                raise ValueError("Evaluation changed training RNG")
        if update == total:
            break
        model.train()
        indices = torch.randint(len(train["y"]), (64,), generator=generator)
        objective = old.state_loss(forward(model, train, indices), train["y"][indices], train["mask"][indices]).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if updates is None and (update+1) % 512 == 0:
            print({"stage": "fit_progress", "seed": seed, "updates": update+1}, flush=True)
    return {"seed": seed, "parameters": PARAMETERS, "selected_update": selected,
            "selected_state": weights, "final_state": model.state_dict(), "history": history,
            "evaluation_rng_unchanged": True}


def fit_task(seed, train, validation):
    start = time.perf_counter()
    fitted = fit(seed, train, validation)
    path = OUTPUT/"models"/f"shared_{seed}.pt"
    old.tensor_save(path, fitted)
    return {"path": path.as_posix(), "sha256": old.file_hash(path),
            "selected_update": fitted["selected_update"], "seconds": time.perf_counter()-start}


def predictions(data, models):
    result = {"flat": [], "shared": []}
    for seed in CONFIG["seeds"]:
        for arm in ("flat", "shared"):
            if arm == "flat":
                saved = torch.load(parent.OUTPUT/"models"/f"afterstate_{seed}.pt", weights_only=True)
                model = old.AdvantageModel(saved["width"])
                model.load_state_dict(saved["dense_state"])
                selected = saved["selected_updates"]["dense"]
            else:
                info = models[str(seed)]
                if old.file_hash(Path(info["path"])) != info["sha256"]:
                    raise ValueError("Shared model changed")
                saved = torch.load(info["path"], weights_only=True)
                model = SharedCardModel()
                model.load_state_dict(saved["selected_state"])
                selected = saved["selected_update"]
            model.eval()
            with torch.no_grad():
                pred = model(data["flat"]) if arm == "flat" else forward(model, data)
                legal = data["mask"].clone()
                legal[:, 0] = True
                choice = pred.masked_fill(~legal, -torch.inf).argmax(-1)
            result[arm].append({"seed": seed, "selected_update": selected,
                                "advantages": pred.tolist(), "choices": choice.tolist()})
    return result


def metrics(data, rows):
    q, y, mask = data["q"].double(), data["y"].double(), data["mask"]
    result = {"zero_mse": float(old.state_loss(torch.zeros_like(y), y, mask).mean()), "arms": {}}
    for arm in rows:
        per_seed = []
        for row in rows[arm]:
            pred = torch.tensor(row["advantages"], dtype=torch.float64)
            choice = torch.tensor(row["choices"])
            per_seed.append({"seed": row["seed"], "selected_update": row["selected_update"],
                "mse": float(old.state_loss(pred, y, mask).mean()),
                "wdl_gain_over_035": float((q[torch.arange(len(q)), choice]-q[:, 0]).mean()),
                "baseline_fraction": float((choice == 0).double().mean())})
        result["arms"][arm] = {"per_seed": per_seed,
            "means": {key: sum(row[key] for row in per_seed)/len(per_seed)
                      for key in ("mse", "wdl_gain_over_035", "baseline_fraction")}}
    return result


def development_statistics(data, rows):
    result = metrics(data, rows)
    shared, flat = (result["arms"][arm]["means"] for arm in ("shared", "flat"))
    reduction = {"versus_flat": 1-shared["mse"]/flat["mse"], "versus_zero": 1-shared["mse"]/result["zero_mse"]}
    gates = {"mse_versus_flat": reduction["versus_flat"] >= .03,
             "mse_versus_zero": reduction["versus_zero"] >= .03,
             "wdl_versus_035": shared["wdl_gain_over_035"] >= .02,
             "wdl_versus_flat": shared["wdl_gain_over_035"] > flat["wdl_gain_over_035"]}
    return {**result, "relative_mse_reduction": reduction, "development_gates": gates,
            "development_passed": all(gates.values()), "development_only": True, "confirmatory": False}


def fresh_statistics(data, rows):
    # One prespecified source-game bootstrap, retaining all three paired seeds.
    q, y, mask = data["q"].numpy().astype(float), data["y"].numpy().astype(float), data["mask"].numpy()
    picked, error = {}, {}
    for arm in rows:
        picked[arm] = np.stack([q[np.arange(len(q)), row["choices"]] for row in rows[arm]])
        error[arm] = np.stack([(((np.asarray(row["advantages"])-y)**2*mask).sum(-1)/np.maximum(mask.sum(-1), 1)) for row in rows[arm]])
    matrix = np.stack(((picked["shared"]-q[:, 0]).mean(0), (picked["shared"]-picked["flat"]).mean(0),
                       (error["shared"]-error["flat"]).mean(0)), -1)
    rng = np.random.default_rng(151999)
    bootstrap = np.concatenate([matrix[rng.integers(len(q), size=(1000, len(q)))].mean(1) for _ in range(50)])
    interval = np.quantile(bootstrap, [.025, .975], axis=0)
    contrasts = {}
    for i, name in enumerate(("shared_minus_035", "shared_minus_flat", "mse_shared_minus_flat")):
        contrasts[name] = {"mean": float(matrix[:, i].mean()), "ci95": interval[:, i].tolist(),
                           "half_width": float((interval[1, i]-interval[0, i])/2)}
    gates = {name: row["mean"] >= .02 and row["ci95"][0] > 0 and row["half_width"] <= .02
             for name, row in contrasts.items() if not name.startswith("mse")}
    gates["mse"] = contrasts["mse_shared_minus_flat"]["ci95"][1] < 0
    return {**metrics(data, rows), "contrasts": contrasts, "fresh_gates": gates,
            "fresh_passed": all(gates.values()), "conditional_move_evidence_only": True}


def fresh_panel(audit=False):
    assigned, cells, positions, filtering = [[] for _ in range(6)], {}, {}, []
    quota = CONFIG["fresh_states"]//6
    for index in range(CONFIG["source_limit"]):
        path = OUTPUT/"sources"/f"game_{index:04d}.json"
        seed = CONFIG["source_seed_start"]+index
        if audit:
            record = old.read_json(path)
            if record["seed"] != seed or record["index"] != index:
                raise ValueError("Fresh source namespace changed")
        else:
            record = {"seed": seed, "index": index, **old.trajectory(old.SplendorEnv(seed=seed).initial_state(), include_trace=True)}
            old.save_new(path, record)
        if record["reason"] not in old.VALID_NATURAL_TERMINATIONS:
            raise ValueError("Non-natural source: stop, do not replace")
        states = old.reconstruct_source(record, verify_raw=audit)
        eligible = [[s for s in states if old.eligible(s, cell)] for cell in range(6)]
        cells[index] = [cell for cell in range(6) if eligible[cell]]
        positions[index] = {cell: min(eligible[cell], key=lambda s: old.digest(["151-position-v1", index, s.turn_index])) for cell in cells[index]}
        filtering.append({"source_index": index, "eligible_by_cell": [len(x) for x in eligible],
                          "path": path.as_posix(), "sha256": old.file_hash(path)})
        old.place_source(index, cells, assigned, quota)
        if all(len(x) == quota for x in assigned):
            break
    if [len(x) for x in assigned] != [quota]*6:
        raise ValueError("Frozen fresh-source schedule exhausted")
    game_cell = {game: cell for cell, games in enumerate(assigned) for game in games}
    if len(game_cell) != CONFIG["fresh_states"]:
        raise ValueError("Fresh source-game reuse")
    samples = []
    for source_index, cell in sorted(game_cell.items()):
        state = positions[source_index][cell]
        samples.append({"index": len(samples), "source_index": source_index, "cell": cell,
            "source_seed": CONFIG["source_seed_start"]+source_index, "turn": state.turn_index, "actor": state.current_player,
            "snapshot": old.serialize_state_snapshot(state), **old.candidates(state, "151-test", source_index)})
    return {"split": "test", "samples": samples, "filtering": filtering, "counts": [quota]*6}


def evaluate_sample(split, sample):
    count = CONFIG["probe_worlds"] if split == "probe" else CONFIG["fresh_worlds"]
    start = CONFIG["probe_world_start"] if split == "probe" else CONFIG["world_seed_start"]
    state = old.deserialize_state_snapshot(sample["snapshot"])
    worlds = []
    for index in range(count):
        seed = start+sample["index"]*1000+index
        world = determinize_for_observer(state, state.current_player, random.Random(seed))
        outcomes = []
        for payload in sample["actions"]:
            row = old.trajectory(world, old.decode_action(payload))
            row["wdl"] = old.outcome_value(row["winner"], state.current_player) if row["reason"] in old.VALID_NATURAL_TERMINATIONS else None
            outcomes.append(row)
        worlds.append({"seed": seed, "world_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": outcomes})
    return {"split": split, "index": sample["index"], "sample_sha256": old.digest(sample), "worlds": worlds}


def collect_task(split, sample):
    record = evaluate_sample(split, sample)
    path = OUTPUT/"targets"/split/f"state_{sample['index']:04d}.json"
    old.save_new(path, record)
    natural = all(o["wdl"] is not None for w in record["worlds"] for o in w["outcomes"])
    return {"index": sample["index"], "sha256": old.file_hash(path), "natural": natural,
            "continuations": sum(len(w["outcomes"]) for w in record["worlds"]),
            "moves": sum(o["turns"] for w in record["worlds"] for o in w["outcomes"])}


def collect(panel):
    split = panel["split"]
    if split not in ("probe", "test") or len({s["index"] for s in panel["samples"]}) != len(panel["samples"]):
        raise ValueError("Invalid collection panel")
    worlds = CONFIG["probe_worlds"] if split == "probe" else CONFIG["fresh_worlds"]
    if sum(len(s["actions"])*worlds for s in panel["samples"]) > CONFIG["probe_continuation_limit" if split == "probe" else "fresh_continuation_limit"]:
        raise ValueError("Frozen continuation budget exceeded")
    files = {}
    with ProcessPoolExecutor(max_workers=8, initializer=old.initialize_worker) as pool:
        jobs = [pool.submit(collect_task, split, sample) for sample in panel["samples"]]
        for i, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            files[str(row["index"])] = row
            if i % 8 == 0 or i == len(jobs):
                print({"stage": split+"_continuations", "states": i, "total": len(jobs)}, flush=True)
    summary = {"files": files, "all_natural": all(r["natural"] for r in files.values())}
    old.save_new(OUTPUT/f"{split}_targets_lock.json", summary)
    if not summary["all_natural"]:
        raise ValueError("Invalid continuation: no exclusions or extensions")


def conditional_fresh(models):
    old.initialize_worker()
    probe = old.read_json(old.OUTPUT/"probe_panel.json")
    old.save_new(OUTPUT/"probe_panel.json", probe)
    started = time.perf_counter()
    collect(probe)
    seconds = time.perf_counter()-started
    counts = old.read_json(OUTPUT/"probe_targets_lock.json")
    n = sum(r["continuations"] for r in counts["files"].values())
    old.save_new(OUTPUT/"timing_probe.json", {"seconds": seconds, "continuations": n,
        "outcome_blind": True, "rough_collection_seconds": seconds/n*196608,
        "caveat": "Small probe includes worker startup; audit is additional"})
    print({"stage": "conditional_timing", "seconds": seconds, "continuations": n, "rough_collection_seconds": seconds/n*196608}, flush=True)
    panel = fresh_panel()
    old.save_new(OUTPUT/"test_panel.json", panel)
    arrays = structured_data(old.feature_data(panel))
    old.tensor_save(OUTPUT/"test_data.pt", arrays)
    pred = predictions(arrays, models)
    old.save_new(OUTPUT/"test_choices_lock.json", {"locked_at": old.now(), "predictions": pred,
        "selection_sha256": old.file_hash(OUTPUT/"selection_lock.json"), "panel_sha256": old.file_hash(OUTPUT/"test_panel.json"),
        "data_sha256": old.file_hash(OUTPUT/"test_data.pt")})
    collect(panel)
    arrays["q"], arrays["y"] = old.target_data(panel, root=OUTPUT)
    result = fresh_statistics(arrays, pred)
    old.save_new(OUTPUT/"fresh_result.json", result)
    subprocess.run([sys.executable, str(VERIFIER), "fresh"], check=True)
    return result


def prepare():
    started = time.perf_counter()
    manifest = identity()
    old.save_new(OUTPUT/"manifest.json", manifest)
    data = parent.load_development()
    lock = {}
    for split, arrays in data.items():
        path = OUTPUT/f"{split}_data.pt"
        old.tensor_save(path, structured_data(arrays))
        lock[split] = old.file_hash(path)
    old.save_new(OUTPUT/"development_lock.json", lock)
    old.save_new(OUTPUT/"preflight.json", {"passed": True, "manifest": old.digest(manifest), "seconds": time.perf_counter()-started,
        "time": old.now(), "parameters": PARAMETERS, "new_fits": 0, "new_continuations": 0})
    print({"stage": "frozen_preflight_passed", "binding": old.digest(manifest), "parameters": PARAMETERS}, flush=True)


def run():
    binding = check_identity()
    if old.read_json(OUTPUT/"preflight.json")["manifest"] != binding:
        raise ValueError("Preflight mismatch")
    started = time.perf_counter()
    old.save_new(OUTPUT/"launch.json", {"started": old.now(), "manifest": binding})
    data = development_data()
    models = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(fit_task, seed, data["train"], data["validation"]): seed for seed in CONFIG["seeds"]}
        for job in as_completed(jobs):
            models[str(jobs[job])] = job.result()
            print({"stage": "fit_complete", "seed": jobs[job], **models[str(jobs[job])]}, flush=True)
    old.save_new(OUTPUT/"selection_lock.json", {"manifest": binding, "locked_at": old.now(), "models": models,
        "development_lock_sha256": old.file_hash(OUTPUT/"development_lock.json")})
    pred = predictions(data["validation"], models)
    old.save_new(OUTPUT/"validation_predictions.json", pred)
    result = development_statistics(data["validation"], pred)
    old.save_new(OUTPUT/"development_result.json", result)
    print({"stage": "development_complete", "passed": result["development_passed"], "seconds": time.perf_counter()-started}, flush=True)
    subprocess.run([sys.executable, str(VERIFIER), "development"], check=True)
    dev_audit = old.read_json(OUTPUT/"development_audit.json")
    if (not dev_audit["passed"] or dev_audit["result_sha256"] != old.file_hash(OUTPUT/"development_result.json")
            or dev_audit["selection_sha256"] != old.file_hash(OUTPUT/"selection_lock.json")
            or dev_audit["predictions_sha256"] != old.file_hash(OUTPUT/"validation_predictions.json")):
        raise ValueError("Development audit failed")
    check_identity()
    fresh = conditional_fresh(models) if result["development_passed"] else None
    if fresh is not None:
        fresh_audit = old.read_json(OUTPUT/"fresh_audit.json")
        if not fresh_audit["passed"] or fresh_audit["result_sha256"] != old.file_hash(OUTPUT/"fresh_result.json"):
            raise ValueError("Fresh audit failed")
    check_identity()
    old.save_new(OUTPUT/"final_status.json", {"complete": True, "finished": old.now(), "seconds": time.perf_counter()-started,
        "development_passed": result["development_passed"], "fresh_stage_ran": fresh is not None,
        "fresh_passed": None if fresh is None else fresh["fresh_passed"],
        "development_result_sha256": old.file_hash(OUTPUT/"development_result.json"),
        "development_audit_sha256": old.file_hash(OUTPUT/"development_audit.json"),
        "fresh_result_sha256": None if fresh is None else old.file_hash(OUTPUT/"fresh_result.json"),
        "fresh_audit_sha256": None if fresh is None else old.file_hash(OUTPUT/"fresh_audit.json"),
        "arena": False, "promotion": False, "automatic_successor": False})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    with old.process_lock(OUTPUT/"run.lock"):
        try:
            prepare() if args.mode == "prepare" else run()
        except Exception:
            old.atomic_write(OUTPUT/f"failure_{args.mode}.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
