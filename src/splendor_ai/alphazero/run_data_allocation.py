"""152: frozen development comparison of source breadth and target replication."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch

from splendor_ai.alphazero import run_shared_cards as parent
from splendor_ai.alphazero import run_value_generalization_scale as corpus
from splendor_ai.alphazero.strong_policy import StrongPolicyInference

old = parent.old
OUTPUT = Path("outputs/data_allocation_152")
PROTOCOL = Path("docs/engineering/data_allocation_152.md")
VERIFIER = Path("scripts/analysis/verify_data_allocation_152.py")
PARENT_SHA = "27c1b1f65edc448aa11a72606598a560494a5e62a2d32cd9af417fac8ae15b1c"
PARENT_AUDIT_SHA = "fc84b9616bb3c60abdbe82d638d8c048aec62c56bc8091f2f4311c46c2537eec"
CONFIG = {
    "arms": {"A": [768, 32], "B": [768, 8], "C": [3072, 8]},
    "new_states": 2304, "new_worlds": 8, "source_limit": 8192,
    "source_seed_start": 1_520_000_000, "world_seed_start": 1_521_000_000,
    "probe_seed_start": 1_529_100_000, "probe_worlds": 4,
    "new_continuations_max": 73728, "probe_continuations_max": 96,
    "workers": 8, "source_batch": 8, "fit_workers": 2,
    "training_seeds": [149001, 149002, 149003], "updates": 4096,
    "batch_states": 64, "batch_seed_offset": 149000, "parameters": 327297,
    "optimizer": "unchanged151_AdamW_lr0.0003_decay0.0001_clip1",
    "validation_grid": "0..128 then every128 through4096, earliest strict minimum",
    "source_behavior": "unchanged143_first20_epsilon0.2_top3_temperature0.5_offset143003",
    "minimum_mse_reduction": .03, "minimum_wdl_gain": .02,
    "validation_states": 192, "validation_worlds": 64,
    "development_only": True, "fresh_performance_test": False,
    "arena": False, "promotion": False, "automatic_successor": False,
}
_SOURCE = None


def parent_inputs():
    if old.file_hash(parent.OUTPUT/"manifest.json") != PARENT_SHA or old.file_hash(parent.OUTPUT/"development_audit.json") != PARENT_AUDIT_SHA:
        raise ValueError("151 manifest/audit changed")
    binding = parent.check_identity()
    audit = old.read_json(parent.OUTPUT/"development_audit.json")
    final = old.read_json(parent.OUTPUT/"final_status.json")
    if not final["complete"] or not audit["passed"] or final["development_audit_sha256"] != PARENT_AUDIT_SHA:
        raise ValueError("151 closure is not audited")
    for field, name in (("result_sha256", "development_result.json"), ("predictions_sha256", "validation_predictions.json"), ("selection_sha256", "selection_lock.json")):
        if audit[field] != old.file_hash(parent.OUTPUT/name):
            raise ValueError("151 audit binding changed")
    selection = old.read_json(parent.OUTPUT/"selection_lock.json")
    if selection["manifest"] != binding or selection["development_lock_sha256"] != old.file_hash(parent.OUTPUT/"development_lock.json"):
        raise ValueError("151 selection/data binding changed")
    paths = [parent.OUTPUT/name for name in ("manifest.json", "development_audit.json", "final_status.json", "development_result.json", "validation_predictions.json", "selection_lock.json", "development_lock.json")]
    lock = old.read_json(parent.OUTPUT/"development_lock.json")
    for split in ("train", "validation"):
        path = parent.OUTPUT/f"{split}_data.pt"
        if old.file_hash(path) != lock[split]:
            raise ValueError("151 data changed")
        paths.append(path)
    for info in selection["models"].values():
        path = Path(info["path"])
        if old.file_hash(path) != info["sha256"]:
            raise ValueError("151 control changed")
        paths.append(path)
    paths.append(old.OUTPUT/"probe_panel.json")
    return {"binding": binding, "files": {p.as_posix(): old.file_hash(p) for p in paths}}


def identity():
    paths = [Path(__file__).relative_to(Path.cwd()), PROTOCOL, VERIFIER,
             Path("run_data_allocation_152.py"), Path("tests/test_data_allocation_152.py")]
    return {"format": "data_allocation_152_v1", "config": CONFIG, "parent": parent_inputs(),
            "dependencies": {p.as_posix(): old.file_hash(p) for p in paths},
            "python": sys.version, "torch": str(torch.__version__)}


def check_identity():
    manifest = old.read_json(OUTPUT/"manifest.json")
    if manifest != identity():
        raise ValueError("152 frozen identity changed")
    return old.digest(manifest)


def initialize_worker():
    global _SOURCE
    old.initialize_worker()
    _SOURCE = StrongPolicyInference(old.CHECKPOINT)


def source_record(index):
    seed = CONFIG["source_seed_start"]+index
    env = old.SplendorEnv(seed=seed)
    state = env.initial_state()
    bot = corpus.CollectionBot(_SOURCE, seed)
    seen, forced, idle, trace = Counter(), 0, 0, []
    hasher = hashlib.sha256()
    reason, winner = "completed", None
    while not state.terminal:
        seen[old.state_signature(state)] += 1
        if seen[old.state_signature(state)] >= 4:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= 400:
            reason = "max_turns"
            break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"
            break
        action = bot.choose_action(env, state, legal)
        payload = old.serialize_action(action)
        hasher.update(old.digest(payload).encode())
        trace.append({**bot.trace[-1], "turn": state.turn_index, "actor": state.current_player, "action": payload})
        successor = env.step(state, action)
        forced = forced+1 if old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(state, successor, action) else idle+1
        state = successor
        if forced >= 2:
            reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
            break
        if idle >= 60:
            reason = "no_progress_cutoff"
            break
    if reason == "completed":
        winner = state.winner
    final = old.serialize_state_snapshot(state)
    return {"index": index, "seed": seed, "reason": reason, "winner": winner,
            "turns": len(trace), "trace": trace, "trace_sha256": hasher.hexdigest(),
            "final_state": final, "final_state_sha256": old.digest(final)}


def source_task(index):
    record = source_record(index)
    path = OUTPUT/"sources"/f"game_{index:04d}.json"
    old.save_new(path, record)
    return {"index": index, "sha256": old.file_hash(path), "reason": record["reason"]}


def select_panel(records):
    """First feasible prefix; input schedule is ordered and outcome-independent."""
    assigned, cells, positions, filtering = [[] for _ in range(6)], {}, {}, []
    quota = CONFIG["new_states"]//6
    for record in records:
        index = len(filtering)
        if record["index"] != index or record["seed"] != CONFIG["source_seed_start"]+index:
            raise ValueError("Source schedule changed")
        if record["reason"] not in old.VALID_NATURAL_TERMINATIONS:
            raise ValueError("Non-natural source: no replacement")
        states = old.reconstruct_source(record)
        eligible = [[s for s in states if old.eligible(s, cell)] for cell in range(6)]
        cells[index] = [cell for cell in range(6) if eligible[cell]]
        positions[index] = {cell: min(eligible[cell], key=lambda s: old.digest(["152-position-v1", index, s.turn_index])) for cell in cells[index]}
        path = OUTPUT/"sources"/f"game_{index:04d}.json"
        filtering.append({"source_index": index, "eligible_by_cell": [len(v) for v in eligible], "path": path.as_posix(), "sha256": old.file_hash(path)})
        old.place_source(index, cells, assigned, quota)
        if len(filtering) % 128 == 0:
            print({"stage": "source_matching", "examined": len(filtering), "counts": [len(v) for v in assigned]}, flush=True)
        if all(len(v) == quota for v in assigned):
            break
    if [len(v) for v in assigned] != [quota]*6:
        raise ValueError("Frozen source schedule exhausted")
    choices = {game: cell for cell, games in enumerate(assigned) for game in games}
    if len(choices) != CONFIG["new_states"]:
        raise ValueError("Source-game reuse")
    samples = []
    for index, cell in sorted(choices.items()):
        state = positions[index][cell]
        samples.append({"index": len(samples), "source_index": index, "cell": cell,
            "source_seed": CONFIG["source_seed_start"]+index, "turn": state.turn_index,
            "actor": state.current_player, "snapshot": old.serialize_state_snapshot(state),
            **old.candidates(state, "152-new", index)})
    return {"split": "new", "samples": samples, "filtering": filtering, "counts": [quota]*6}


def generate_panel():
    attempted = {}
    with ProcessPoolExecutor(max_workers=8, initializer=initialize_worker) as pool:
        def records():
            for start in range(0, CONFIG["source_limit"], 8):
                # At most seven unused records in the final dispatched batch.
                rows = list(pool.map(source_task, range(start, min(start+8, CONFIG["source_limit"]))))
                for row in rows:
                    attempted[str(row["index"])] = row
                if any(row["reason"] not in old.VALID_NATURAL_TERMINATIONS for row in rows):
                    raise ValueError("Non-natural attempted source, including prefetched suffix")
                for row in rows:
                    yield old.read_json(OUTPUT/"sources"/f"game_{row['index']:04d}.json")
        panel = select_panel(records())
    old.save_new(OUTPUT/"sources_lock.json", {"files": attempted, "selected_prefix": len(panel["filtering"]),
        "unused_prefetched": len(attempted)-len(panel["filtering"])})
    return panel


def world_seed(split, sample, world):
    if split not in ("probe", "new"):
        raise ValueError("152 has no performance-test split")
    return CONFIG["probe_seed_start" if split == "probe" else "world_seed_start"]+sample*1000+world


def evaluate_sample(split, sample):
    n = CONFIG["probe_worlds"] if split == "probe" else CONFIG["new_worlds"]
    state = old.deserialize_state_snapshot(sample["snapshot"])
    worlds = []
    for i in range(n):
        seed = world_seed(split, sample["index"], i)
        world = parent.determinize_for_observer(state, state.current_player, random.Random(seed))
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
    outcomes = [o for w in record["worlds"] for o in w["outcomes"]]
    return {"index": sample["index"], "sha256": old.file_hash(path),
            "natural": all(o["wdl"] is not None for o in outcomes), "continuations": len(outcomes),
            "moves": sum(o["turns"] for o in outcomes), "reasons": dict(Counter(o["reason"] for o in outcomes))}


def collect(panel):
    split = panel["split"]
    if split not in ("probe", "new"):
        raise ValueError("No performance test authorized")
    worlds, cap = (4, 96) if split == "probe" else (8, 73728)
    if len({s["index"] for s in panel["samples"]}) != len(panel["samples"]) or sum(len(s["actions"])*worlds for s in panel["samples"]) > cap:
        raise ValueError("Continuation panel/budget violation")
    files = {}
    with ProcessPoolExecutor(max_workers=8, initializer=old.initialize_worker) as pool:
        jobs = [pool.submit(collect_task, split, sample) for sample in panel["samples"]]
        for i, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            files[str(row["index"])] = row
            if i % 32 == 0 or i == len(jobs):
                print({"stage": split+"_continuations", "states": i, "total": len(jobs)}, flush=True)
    lock = {"files": files, "all_natural": all(v["natural"] for v in files.values())}
    old.save_new(OUTPUT/f"{split}_targets_lock.json", lock)
    if not lock["all_natural"]:
        raise ValueError("Invalid continuation: no replacement or exclusions")


def first_eight_targets(panel):
    lock = old.read_json(old.OUTPUT/"train_targets_lock.json")
    values = []
    for sample in panel["samples"]:
        path = old.OUTPUT/"targets"/"train"/f"state_{sample['index']:04d}.json"
        if old.file_hash(path) != lock["files"][str(sample["index"])]["sha256"]:
            raise ValueError("Reused target changed")
        record = old.read_json(path)
        if record["sample_sha256"] != old.digest(sample) or len(record["worlds"]) != 32:
            raise ValueError("Wrong original training targets")
        worlds = record["worlds"][:8]
        if any(o["reason"] not in old.VALID_NATURAL_TERMINATIONS for w in worlds for o in w["outcomes"]):
            raise ValueError("Invalid reused target")
        q = torch.tensor([[o["wdl"] for o in w["outcomes"]] for w in worlds], dtype=torch.float32).mean(0)
        values.append(torch.cat((q, q[:1].repeat(4-len(q)))))
    q = torch.stack(values)
    return q, q-q[:, :1]


def concatenate_data(small, added):
    if small.keys() != added.keys():
        raise ValueError("Training schema mismatch")
    return {k: torch.cat((small[k], added[k])) for k in small}


def load_data(name, lock_name="development_lock.json"):
    path = OUTPUT/f"{name}_data.pt"
    if old.file_hash(path) != old.read_json(OUTPUT/lock_name)[name]:
        raise ValueError("152 data changed")
    return torch.load(path, weights_only=True)


def exposure(seed, n, selected, total=4096):
    rng = torch.Generator().manual_seed(seed+149000)
    seen, rows = set(), []
    for step in range(total+1):
        if parent.parent.validate_grid(step, total) or step == selected:
            rows.append({"update": step, "draws": step*64, "unique_states": len(seen), "unique_games": len(seen)})
        if step < total:
            seen.update(torch.randint(n, (64,), generator=rng).tolist())
    return {"available_states": n, "selected": next(v for v in rows if v["update"] == selected), "final": rows[-1], "history": rows}


def fit_task(arm, seed):
    start = time.perf_counter()
    data, val = load_data(arm), load_data("validation")
    fit = parent.fit(seed, data, val)
    payload = {"arm": arm, "fit": fit, "exposure": exposure(seed, len(data["y"]), fit["selected_update"])}
    path = OUTPUT/"models"/f"{arm}_{seed}.pt"
    old.tensor_save(path, payload)
    return {"path": path.as_posix(), "sha256": old.file_hash(path), "selected_update": fit["selected_update"], "seconds": time.perf_counter()-start}


def predictions(data, models):
    rows = {}
    for arm in ("A", "B", "C"):
        rows[arm] = []
        for seed in CONFIG["training_seeds"]:
            info = models[arm][str(seed)]
            if old.file_hash(Path(info["path"])) != info["sha256"]:
                raise ValueError("Selected model changed")
            payload = torch.load(info["path"], weights_only=True)
            fit = payload if arm == "A" else payload["fit"]
            model = parent.SharedCardModel()
            model.load_state_dict(fit["selected_state"])
            model.eval()
            with torch.no_grad():
                pred = parent.forward(model, data)
                legal = data["mask"].clone()
                legal[:, 0] = True
                choices = pred.masked_fill(~legal, -torch.inf).argmax(-1)
            rows[arm].append({"seed": seed, "selected_update": fit["selected_update"], "advantages": pred.tolist(), "choices": choices.tolist()})
    return rows


def statistics(data, rows):
    metrics = parent.metrics(data, rows)
    a, b, c = (metrics["arms"][arm]["means"] for arm in ("A", "B", "C"))
    contrasts = {name: {"mse_difference": x["mse"]-y["mse"], "wdl_difference": x["wdl_gain_over_035"]-y["wdl_gain_over_035"]}
                 for name, x, y in (("C_minus_A", c, a), ("C_minus_B", c, b), ("B_minus_A", b, a))}
    reduction = {"versus_A": 1-c["mse"]/a["mse"], "versus_zero": 1-c["mse"]/metrics["zero_mse"]}
    gates = {"mse_versus_A": reduction["versus_A"] >= .03, "mse_versus_zero": reduction["versus_zero"] >= .03,
             "wdl_versus_035": c["wdl_gain_over_035"] >= .02, "wdl_versus_A": c["wdl_gain_over_035"] > a["wdl_gain_over_035"]}
    return {**metrics, "contrasts": contrasts, "relative_mse_reduction_C": reduction,
            "development_gates": gates, "development_passed": all(gates.values()),
            "C_better_than_B_on_both_means": c["mse"] < b["mse"] and c["wdl_gain_over_035"] > b["wdl_gain_over_035"],
            "development_only": True, "confirmatory": False, "fresh_performance_test": False}


def audit_stage(stage):
    subprocess.run([sys.executable, str(VERIFIER), stage], check=True)
    audit = old.read_json(OUTPUT/f"{stage}_audit.json")
    if not audit["passed"] or audit["manifest"] != check_identity():
        raise ValueError("Audit binding failed")
    return audit


def prepare():
    started = time.perf_counter()
    manifest = identity()
    old.save_new(OUTPUT/"manifest.json", manifest)
    original = parent.development_data()
    small = dict(original["train"])
    small["q"], small["y"] = first_eight_targets(old.read_json(old.OUTPUT/"train_panel.json"))
    for name, data in (("B", small), ("validation", original["validation"])):
        old.tensor_save(OUTPUT/f"{name}_data.pt", data)
    old.save_new(OUTPUT/"initial_data_lock.json", {name: old.file_hash(OUTPUT/f"{name}_data.pt") for name in ("B", "validation")})
    probe = old.read_json(old.OUTPUT/"probe_panel.json")
    old.save_new(OUTPUT/"probe_panel.json", probe)
    timing = time.perf_counter()
    collect(probe)
    elapsed = time.perf_counter()-timing
    count = sum(v["continuations"] for v in old.read_json(OUTPUT/"probe_targets_lock.json")["files"].values())
    old.save_new(OUTPUT/"timing_probe.json", {"seconds": elapsed, "continuations": count, "outcome_blind": True,
        "rough_label_seconds": elapsed/count*73728, "caveat": "Small six-state probe includes startup; source generation, fits and replay audit are additional"})
    audit_stage("probe")
    old.save_new(OUTPUT/"preflight.json", {"passed": True, "manifest": check_identity(), "time": old.now(),
        "seconds": time.perf_counter()-started, "new_main_labels": 0, "new_fits": 0, "source_panel_pending": True,
        "probe_audit_sha256": old.file_hash(OUTPUT/"probe_audit.json"),
        "initial_data_lock_sha256": old.file_hash(OUTPUT/"initial_data_lock.json"),
        "timing_sha256": old.file_hash(OUTPUT/"timing_probe.json")})
    print({"stage": "preflight_passed", "binding": check_identity(), "rough_label_seconds": elapsed/count*73728}, flush=True)


def run():
    binding = check_identity()
    preflight = old.read_json(OUTPUT/"preflight.json")
    if not preflight["passed"] or preflight["manifest"] != binding:
        raise ValueError("Preflight not passed")
    for key, name in (("probe_audit_sha256", "probe_audit.json"), ("initial_data_lock_sha256", "initial_data_lock.json"), ("timing_sha256", "timing_probe.json")):
        if preflight[key] != old.file_hash(OUTPUT/name):
            raise ValueError("Preflight artifact changed")
    started = time.perf_counter()
    old.save_new(OUTPUT/"launch.json", {"manifest": binding, "started": old.now(), "preflight_sha256": old.file_hash(OUTPUT/"preflight.json")})
    initialize_worker()
    panel = generate_panel()
    old.save_new(OUTPUT/"new_panel.json", panel)
    added = parent.structured_data(old.feature_data(panel))
    old.tensor_save(OUTPUT/"new_features.pt", added)
    old.save_new(OUTPUT/"panel_lock.json", {"manifest": binding, "time": old.now(),
        "panel_sha256": old.file_hash(OUTPUT/"new_panel.json"), "features_sha256": old.file_hash(OUTPUT/"new_features.pt"),
        "sources_sha256": old.file_hash(OUTPUT/"sources_lock.json")})
    audit_stage("panel")
    old.save_new(OUTPUT/"labels_launch.json", {"time": old.now(), "panel_audit_sha256": old.file_hash(OUTPUT/"panel_audit.json")})
    collect(panel)
    added["q"], added["y"] = old.target_data(panel, root=OUTPUT)
    broad = concatenate_data(load_data("B", "initial_data_lock.json"), added)
    old.tensor_save(OUTPUT/"C_data.pt", broad)
    old.save_new(OUTPUT/"development_lock.json", {name: old.file_hash(OUTPUT/f"{name}_data.pt") for name in ("B", "C", "validation")})
    del broad, added
    models = {"A": old.read_json(parent.OUTPUT/"selection_lock.json")["models"], "B": {}, "C": {}}
    for arm in ("B", "C"):
        print({"stage": "training_arm", "arm": arm}, flush=True)
        with ProcessPoolExecutor(max_workers=CONFIG["fit_workers"]) as pool:
            jobs = {pool.submit(fit_task, arm, seed): seed for seed in CONFIG["training_seeds"]}
            for job in as_completed(jobs):
                seed = jobs[job]
                models[arm][str(seed)] = job.result()
                print({"stage": "fit_complete", "arm": arm, "seed": seed, **models[arm][str(seed)]}, flush=True)
    old.save_new(OUTPUT/"selection_lock.json", {"manifest": binding, "locked_at": old.now(), "models": models,
        "development_lock_sha256": old.file_hash(OUTPUT/"development_lock.json"),
        "new_targets_lock_sha256": old.file_hash(OUTPUT/"new_targets_lock.json"), "panel_lock_sha256": old.file_hash(OUTPUT/"panel_lock.json")})
    val = load_data("validation")
    rows = predictions(val, models)
    old.save_new(OUTPUT/"validation_predictions.json", rows)
    result = statistics(val, rows)
    old.save_new(OUTPUT/"result.json", result)
    print({"stage": "development_complete", "passed": result["development_passed"], "seconds": time.perf_counter()-started}, flush=True)
    audit = audit_stage("final")
    for field, name in (("result_sha256", "result.json"), ("selection_sha256", "selection_lock.json"), ("predictions_sha256", "validation_predictions.json")):
        if audit[field] != old.file_hash(OUTPUT/name):
            raise ValueError("Final audit output mismatch")
    old.save_new(OUTPUT/"final_status.json", {"complete": True, "finished": old.now(), "seconds": time.perf_counter()-started,
        "development_passed": result["development_passed"], "result_sha256": old.file_hash(OUTPUT/"result.json"),
        "audit_sha256": old.file_hash(OUTPUT/"final_audit.json"), "development_only": True,
        "new_main_continuations": audit["new_continuations"], "fresh_performance_test": False,
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
