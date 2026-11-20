"""Independent source behavior, target arithmetic, full-fit and result audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
from pathlib import Path
import random
import time

import torch

from splendor_ai.alphazero import run_data_allocation as p

spec = importlib.util.spec_from_file_location("audit151_dependency", "scripts/analysis/verify_shared_cards_151.py")
a151 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a151)
compare = a151.a150.compare


def verify_initial():
    original = a151.a150.read_development()
    val = p.load_data("validation", "initial_data_lock.json")
    small = p.load_data("B", "initial_data_lock.json")
    for split, data in (("train", small), ("validation", val)):
        panel = p.old.read_json(p.old.OUTPUT/f"{split}_panel.json")
        a151.verify_features(panel, data)
        assert torch.equal(data["flat"], original[split]["afterstate"])
        assert torch.equal(data["mask"], original[split]["mask"])
        if split == "validation":
            for key in ("q", "y"):
                assert torch.equal(data[key], original[split][key])
        else:
            values = []
            for sample in panel["samples"]:
                saved = p.old.read_json(p.old.OUTPUT/"targets"/"train"/f"state_{sample['index']:04d}.json")
                assert len(saved["worlds"]) == 32 and saved["sample_sha256"] == p.old.digest(sample)
                for j in range(8):
                    assert saved["worlds"][j]["seed"] == p.old.world_seed("train", sample["index"], j)
                row = [sum(saved["worlds"][w]["outcomes"][j]["wdl"] for w in range(8))/8 for j in range(len(sample["actions"]))]
                values.append(row+[row[0]]*(4-len(row)))
            q = torch.tensor(values, dtype=torch.float32)
            assert torch.equal(q, small["q"]) and torch.equal(q-q[:, :1], small["y"])
    return small, val


def audit_source_record(record):
    """Independent behavior RNG/action selection, including nonselected sources."""
    index = record["index"]
    assert record["seed"] == 1520000000+index
    env = p.old.SplendorEnv(seed=record["seed"])
    state = env.initial_state()
    rng = random.Random(record["seed"]+143003)
    seen, passes, idle, hasher = Counter(), 0, 0, hashlib.sha256()
    for step, row in enumerate(record["trace"]):
        assert not state.terminal and state.turn_index < 400
        seen[p.old.state_signature(state)] += 1
        assert seen[p.old.state_signature(state)] < 4
        assert row["state_hash"] == p.old.digest(p.old.serialize_state_snapshot(state))
        assert row["turn"] == state.turn_index and row["actor"] == state.current_player
        legal = env.legal_actions(state)
        scores = p.old._RAW.score_actions(state, legal)
        best = selected = max(range(len(scores)), key=lambda j: scores[j])
        if state.turn_index < 20 and rng.random() < .2:
            top = sorted(range(len(scores)), key=lambda j: (-scores[j], j))[:3]
            weights = torch.softmax(torch.tensor(scores, dtype=torch.float32)[top]/.5, 0).tolist()
            selected = rng.choices(top, weights=weights, k=1)[0]
        assert (row["greedy_row"], row["selected_row"]) == (best, selected)
        action = legal[selected]
        assert p.old.serialize_action(action) == row["action"]
        hasher.update(p.old.digest(row["action"]).encode())
        after = env.step(state, action)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if p.old.is_progress_transition(state, after, action) else idle+1
        state = after
        if step < len(record["trace"])-1:
            assert passes < 2 and idle < 60
    assert record["turns"] == len(record["trace"])
    assert hasher.hexdigest() == record["trace_sha256"]
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == p.old.digest(record["final_state"]) == record["final_state_sha256"]
    if record["reason"] == "completed":
        assert state.terminal and idle < 60
        winner = state.winner
    else:
        assert record["reason"] == "forced_pass_deadlock" and passes == 2
        winner = p.old.adjudicate_scores(state)
    assert winner == record["winner"]
    return record["turns"]


def audit_source_task(index, info):
    path = p.OUTPUT/"sources"/f"game_{index:04d}.json"
    assert p.old.file_hash(path) == info["sha256"]
    record = p.old.read_json(path)
    assert record["index"] == index and record["reason"] == info["reason"]
    return audit_source_record(record)


def panel_integrity():
    lock = p.old.read_json(p.OUTPUT/"panel_lock.json")
    assert lock["manifest"] == p.check_identity()
    for field, name in (("panel_sha256", "new_panel.json"), ("features_sha256", "new_features.pt"), ("sources_sha256", "sources_lock.json")):
        assert lock[field] == p.old.file_hash(p.OUTPUT/name)
    sources = p.old.read_json(p.OUTPUT/"sources_lock.json")
    for index, info in sources["files"].items():
        assert p.old.file_hash(p.OUTPUT/"sources"/f"game_{int(index):04d}.json") == info["sha256"]
    assert set(sources["files"]) == {str(i) for i in range(len(sources["files"]))}
    assert len(sources["files"]) <= 8192 and 0 <= sources["unused_prefetched"] < 8
    assert sources["selected_prefix"]+sources["unused_prefetched"] == len(sources["files"])
    return lock, sources


def verify_panel():
    lock, sources = panel_integrity()
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        jobs = [pool.submit(audit_source_task, int(i), info) for i, info in sources["files"].items()]
        moves = 0
        for j, job in enumerate(as_completed(jobs), 1):
            moves += job.result()
            if j % 128 == 0 or j == len(jobs):
                print({"audit": "source_replay", "games": j, "total": len(jobs)}, flush=True)
    p.old.initialize_worker()
    rebuilt = p.select_panel(p.old.read_json(p.OUTPUT/"sources"/f"game_{i:04d}.json") for i in range(len(sources["files"])))
    panel = p.old.read_json(p.OUTPUT/"new_panel.json")
    assert p.old.digest(rebuilt) == p.old.digest(panel)
    assert len(panel["filtering"]) == sources["selected_prefix"]
    samples = panel["samples"]
    assert len(samples) == len({s["source_seed"] for s in samples}) == 2304
    assert Counter(s["cell"] for s in samples) == {c: 384 for c in range(6)}
    prior = {s["source_seed"] for split in ("train", "validation") for s in p.old.read_json(p.old.OUTPUT/f"{split}_panel.json")["samples"]}
    assert not prior & {s["source_seed"] for s in samples}
    data = torch.load(p.OUTPUT/"new_features.pt", weights_only=True)
    assert set(data) == {"flat", "context", "cards", "mask"}
    a151.verify_features(panel, data)
    # This stage must be complete before new main target collection exists.
    assert not (p.OUTPUT/"targets"/"new").exists()
    return {"source_games_replayed": len(sources["files"]), "source_moves": moves,
            "new_states": 2304, "panel_lock_sha256": p.old.file_hash(p.OUTPUT/"panel_lock.json")}


def audit_sample(split, sample, info):
    path = p.OUTPUT/"targets"/split/f"state_{sample['index']:04d}.json"
    assert p.old.file_hash(path) == info["sha256"]
    record = p.old.read_json(path)
    assert record["index"] == sample["index"] and record["split"] == split
    assert record["sample_sha256"] == p.old.digest(sample)
    count = 4 if split == "probe" else 8
    assert len(record["worlds"]) == count
    state = p.old.deserialize_state_snapshot(sample["snapshot"])
    n = moves = 0
    reasons = Counter()
    for w, saved in enumerate(record["worlds"]):
        seed = (1529100000 if split == "probe" else 1521000000)+sample["index"]*1000+w
        assert saved["seed"] == seed
        world = p.parent.determinize_for_observer(state, state.current_player, random.Random(seed))
        assert saved["world_sha256"] == p.old.digest(p.old.serialize_state_snapshot(world))
        assert len(saved["outcomes"]) == len(sample["actions"])
        for j, payload in enumerate(sample["actions"]):
            action = p.old.decode_action(payload)
            if w == 0:
                assert torch.equal(a151.features(state, action, "afterstate"), a151.features(world, action, "afterstate"))
            rebuilt = a151.a149.replay_branch(world, action, state.current_player)
            assert rebuilt == saved["outcomes"][j], (split, sample["index"], w, j)
            assert rebuilt["reason"] in p.old.VALID_NATURAL_TERMINATIONS
            n += 1
            moves += rebuilt["turns"]
            reasons[rebuilt["reason"]] += 1
    assert info["natural"] and info["continuations"] == n and info["moves"] == moves and info["reasons"] == dict(reasons)
    return {"continuations": n, "moves": moves, "reasons": dict(reasons)}


def replay_targets(panel):
    lock = p.old.read_json(p.OUTPUT/f"{panel['split']}_targets_lock.json")
    assert lock["all_natural"] and set(lock["files"]) == {str(s["index"]) for s in panel["samples"]}
    count = moves = 0
    reasons = Counter()
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        jobs = [pool.submit(audit_sample, panel["split"], s, lock["files"][str(s["index"])]) for s in panel["samples"]]
        for i, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            count += row["continuations"]
            moves += row["moves"]
            reasons.update(row["reasons"])
            if i % 32 == 0 or i == len(jobs):
                print({"audit": panel["split"]+"_replay", "states": i, "total": len(jobs)}, flush=True)
    return {"continuations": count, "moves": moves, "reasons": dict(reasons)}


def verify_probe():
    verify_initial()
    panel = p.old.read_json(p.OUTPUT/"probe_panel.json")
    assert panel == p.old.read_json(p.old.OUTPUT/"probe_panel.json")
    results = replay_targets(panel)
    assert results["continuations"] <= 96
    return {**results, "initial_states_checked": 960,
            "initial_data_lock_sha256": p.old.file_hash(p.OUTPUT/"initial_data_lock.json"),
            "probe_targets_lock_sha256": p.old.file_hash(p.OUTPUT/"probe_targets_lock.json")}


def independent_exposure(seed, n, selected, total=4096):
    generator = torch.Generator().manual_seed(seed+149000)
    seen = torch.zeros(n, dtype=torch.bool)
    rows = []
    for step in range(total+1):
        if step <= 128 or step % 128 == 0 or step in (selected, total):
            unique = int(seen.sum())
            rows.append({"update": step, "draws": step*64, "unique_states": unique, "unique_games": unique})
        if step != total:
            seen[torch.randint(n, (64,), generator=generator)] = True
    return {"available_states": n, "selected": next(r for r in rows if r["update"] == selected), "final": rows[-1], "history": rows}


def audit_fit(arm, seed, info):
    assert p.old.file_hash(Path(info["path"])) == info["sha256"]
    train, val = p.load_data(arm), p.load_data("validation")
    saved = torch.load(info["path"], weights_only=True)
    assert saved["arm"] == arm
    rebuilt = a151.independent_fit(seed, train, val)
    compare(rebuilt, saved["fit"])
    compare(independent_exposure(seed, len(train["y"]), rebuilt["selected_update"]), saved["exposure"])
    assert info["selected_update"] == rebuilt["selected_update"]
    return arm, seed


def independent_predictions(data, models):
    rows = {}
    for arm in ("A", "B", "C"):
        rows[arm] = []
        for seed in (149001,149002,149003):
            info = models[arm][str(seed)]
            assert p.old.file_hash(Path(info["path"])) == info["sha256"]
            saved = torch.load(info["path"], weights_only=True)
            fit = saved if arm == "A" else saved["fit"]
            model = p.parent.SharedCardModel()
            model.load_state_dict(fit["selected_state"])
            model.eval()
            with torch.no_grad():
                pred = model(data["context"], data["cards"]).tolist()
            choices = [max([0]+[j for j in range(1,4) if data["mask"][i,j]], key=lambda j: pred[i][j]) for i in range(len(pred))]
            rows[arm].append({"seed": seed, "selected_update": fit["selected_update"], "advantages": pred, "choices": choices})
    return rows


def independent_statistics(data, rows):
    result = a151.independent_metrics(data, rows)
    a, b, c = (result["arms"][x]["means"] for x in ("A", "B", "C"))
    contrasts = {}
    for name, first, second in (("C_minus_A", c, a), ("C_minus_B", c, b), ("B_minus_A", b, a)):
        contrasts[name] = {"mse_difference": first["mse"]-second["mse"], "wdl_difference": first["wdl_gain_over_035"]-second["wdl_gain_over_035"]}
    reduction = {"versus_A": 1-c["mse"]/a["mse"], "versus_zero": 1-c["mse"]/result["zero_mse"]}
    gates = {"mse_versus_A": reduction["versus_A"] >= .03, "mse_versus_zero": reduction["versus_zero"] >= .03,
             "wdl_versus_035": c["wdl_gain_over_035"] >= .02, "wdl_versus_A": c["wdl_gain_over_035"] > a["wdl_gain_over_035"]}
    return {**result, "contrasts": contrasts, "relative_mse_reduction_C": reduction, "development_gates": gates,
            "development_passed": all(gates.values()), "C_better_than_B_on_both_means": c["mse"] < b["mse"] and c["wdl_gain_over_035"] > b["wdl_gain_over_035"],
            "development_only": True, "confirmatory": False, "fresh_performance_test": False}


def verify_final():
    panel_integrity()
    launch = p.old.read_json(p.OUTPUT/"launch.json")
    assert launch["manifest"] == p.check_identity() and launch["preflight_sha256"] == p.old.file_hash(p.OUTPUT/"preflight.json")
    preflight = p.old.read_json(p.OUTPUT/"preflight.json")
    assert preflight["passed"] and preflight["manifest"] == launch["manifest"]
    for key, name in (("probe_audit_sha256", "probe_audit.json"), ("initial_data_lock_sha256", "initial_data_lock.json"), ("timing_sha256", "timing_probe.json")):
        assert preflight[key] == p.old.file_hash(p.OUTPUT/name)
    panel_audit = p.old.read_json(p.OUTPUT/"panel_audit.json")
    assert panel_audit["passed"] and panel_audit["panel_lock_sha256"] == p.old.file_hash(p.OUTPUT/"panel_lock.json")
    label_launch = p.old.read_json(p.OUTPUT/"labels_launch.json")
    assert label_launch["panel_audit_sha256"] == p.old.file_hash(p.OUTPUT/"panel_audit.json")
    initial_audit = p.old.read_json(p.OUTPUT/"probe_audit.json")
    assert initial_audit["initial_data_lock_sha256"] == p.old.file_hash(p.OUTPUT/"initial_data_lock.json")
    assert initial_audit["probe_targets_lock_sha256"] == p.old.file_hash(p.OUTPUT/"probe_targets_lock.json")
    small, val = verify_initial()
    broad = p.load_data("C")
    added = torch.load(p.OUTPUT/"new_features.pt", weights_only=True)
    for key in small:
        assert torch.equal(broad[key][:768], small[key])
    for key in added:
        assert torch.equal(broad[key][768:], added[key])
    panel = p.old.read_json(p.OUTPUT/"new_panel.json")
    values = []
    target_lock = p.old.read_json(p.OUTPUT/"new_targets_lock.json")
    assert target_lock["all_natural"]
    for sample in panel["samples"]:
        path = p.OUTPUT/"targets"/"new"/f"state_{sample['index']:04d}.json"
        assert path.stat().st_mtime_ns > (p.OUTPUT/"labels_launch.json").stat().st_mtime_ns
        assert p.old.file_hash(path) == target_lock["files"][str(sample["index"])]["sha256"]
        record = p.old.read_json(path)
        assert record["sample_sha256"] == p.old.digest(sample) and len(record["worlds"]) == 8
        row = [sum(w["outcomes"][j]["wdl"] for w in record["worlds"])/8 for j in range(len(sample["actions"]))]
        values.append(row+[row[0]]*(4-len(row)))
    q = torch.tensor(values, dtype=torch.float32)
    assert torch.equal(broad["q"][768:], q) and torch.equal(broad["y"][768:], q-q[:, :1])
    assert len(broad["y"]) == 3072
    selection = p.old.read_json(p.OUTPUT/"selection_lock.json")
    assert selection["manifest"] == p.check_identity()
    for field, name in (("development_lock_sha256", "development_lock.json"), ("new_targets_lock_sha256", "new_targets_lock.json"), ("panel_lock_sha256", "panel_lock.json")):
        assert selection[field] == p.old.file_hash(p.OUTPUT/name)
    assert selection["models"]["A"] == p.old.read_json(p.parent.OUTPUT/"selection_lock.json")["models"]
    rows = independent_predictions(val, selection["models"])
    compare(rows, p.old.read_json(p.OUTPUT/"validation_predictions.json"))
    compare(rows["A"], p.old.read_json(p.parent.OUTPUT/"validation_predictions.json")["shared"])
    compare(independent_statistics(val, rows), p.old.read_json(p.OUTPUT/"result.json"), tolerance=1e-12)
    print({"audit": "data_targets_frozen_control_predictions_statistics_passed"}, flush=True)
    replay = replay_targets(panel)
    assert replay["continuations"] <= 73728
    del broad, added, small, val
    for arm in ("B", "C"):
        print({"audit": "full_refits", "arm": arm}, flush=True)
        with ProcessPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(audit_fit, arm, seed, selection["models"][arm][str(seed)]) for seed in (149001,149002,149003)]
            for job in as_completed(jobs):
                print({"audit": "fit_reproduced", "arm_seed": job.result()}, flush=True)
    return {"new_continuations": replay["continuations"], "new_moves": replay["moves"], "termination_counts": replay["reasons"],
            "full_fits_reproduced": 6, "history_rows_per_fit": 160,
            "result_sha256": p.old.file_hash(p.OUTPUT/"result.json"), "selection_sha256": p.old.file_hash(p.OUTPUT/"selection_lock.json"),
            "predictions_sha256": p.old.file_hash(p.OUTPUT/"validation_predictions.json"),
            "panel_audit_sha256": p.old.file_hash(p.OUTPUT/"panel_audit.json"), "probe_audit_sha256": p.old.file_hash(p.OUTPUT/"probe_audit.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("probe", "panel", "final"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    binding = p.check_identity()
    result = {"probe": verify_probe, "panel": verify_panel, "final": verify_final}[args.stage]()
    assert p.check_identity() == binding
    p.old.save_new(p.OUTPUT/f"{args.stage}_audit.json", {"passed": True, "manifest": binding, "seconds": time.perf_counter()-started, **result})


if __name__ == "__main__":
    main()
