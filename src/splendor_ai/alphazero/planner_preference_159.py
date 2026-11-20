"""159: independent-world replication of two fixed planners' preferences.

Conditional rollout objectives are measurements, never gameplay-quality labels.
All selection records are sealed before any evaluation task is dispatched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import math
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.stats import t as student_t
import torch

from splendor_ai.alphazero import learned_continuation_158 as h
from splendor_ai.alphazero import run_learned_continuation as parent
from splendor_ai.encoding.public_belief import canonical_public_belief_key

old = h.old
OUTPUT = Path("outputs/planner_preference_159")
REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = Path("docs/engineering/planner_preference_159.md")
VERIFIER = Path("scripts/analysis/verify_planner_preference_159.py")
INVENTORY = [Path("src/splendor_ai/alphazero/planner_preference_159.py"),
             Path("run_planner_preference_159.py"), PROTOCOL, VERIFIER,
             Path("tests/test_planner_preference_159.py")]
PARENT_BINDING = "06fcd7d298a3099b11206ff3cc5c4343b5c9c1fb01b3ef975c87fafa015d40ab"
GAMES_LOCK_SHA = "b535ff50e868020d8e1392f9ae95526e53f7cf1e866cdf6dec50299708b0b06a"
RESULT_SHA = "2abae0ec098780c23dcdfd9d82e78044e99d5d2e36ca0021f8e7ee373d3c4fb4"
CONFIG = {"states": 180, "cells": 36, "quota": 5, "workers": 8,
          "selection_worlds": 32, "evaluation_worlds": 128,
          "bootstrap_replicates": 50000, "bootstrap_seed": 1593000000,
          "maximum_half_width": .025, "material_score_difference": .025,
          "planning_root_sd": .14, "selection_replay_worlds": [0, 16],
          "evaluation_replay_worlds": [0, 64], "main_continuation_cap": 138240,
          "training": False, "arena": False, "promotion": False,
          "automatic_successor": False, "automatic_resume": False,
          "scheduled_monitoring": False}
POLICIES = ("GG", "LL")


def phase(turn):
    return 0 if turn < 20 else 1 if turn < 40 else 2


def cell_id(model_id, game_phase, actor, source_planner):
    return ((h.MODELS.index(model_id)*3+game_phase)*2+actor)*2+POLICIES.index(source_planner)


def event(stage, **values):
    print(old.digest(values)[:8], old.now(), stage, values, flush=True)


def identity():
    assert parent.validate_freeze() == PARENT_BINDING
    assert old.file_hash(h.OUTPUT/"main_games_lock.json") == GAMES_LOCK_SHA
    assert old.file_hash(h.OUTPUT/"result.json") == RESULT_SHA
    return {"format": "planner_preference_159_v1", "config": CONFIG,
            "parent_binding": PARENT_BINDING, "source_games_lock_sha256": GAMES_LOCK_SHA,
            "source_result_sha256": RESULT_SHA,
            "source_status": "158_owner_stopped_full_replay_incomplete_19_of_402",
            "models": h.selected_models(),
            "dependencies": {p.as_posix(): old.file_hash(p) for p in INVENTORY},
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "torch": str(torch.__version__)}


def check_identity(base=OUTPUT):
    manifest = old.read_json(Path(base)/"manifest.json")
    assert manifest == identity(), "159 frozen identity changed"
    return old.digest(manifest)


def select_panel():
    """Hash-ranked capacity matching uses only source identities and turn/actor.

    A pair can supply exactly one position. Neither terminal outcome, root means,
    chosen action nor rollout outcomes participate in matching or root ranking.
    """
    lock = old.read_json(h.OUTPUT/"main_games_lock.json")
    assert old.file_hash(h.OUTPUT/"main_games_lock.json") == GAMES_LOCK_SHA
    assert lock["manifest"] == PARENT_BINDING
    eligible, best, game_hashes = {}, {}, {}
    for pair in range(201):
        available = {}
        for seat in (0, 1):
            source_index = pair*2+seat
            path = h.OUTPUT/"main/games"/f"game_{source_index:04d}.json"
            sha = old.file_hash(path)
            assert sha == lock["files"][str(source_index)]
            game_hashes[path.as_posix()] = sha
            game = old.read_json(path)
            spec = game["spec"]
            assert game["manifest"] == PARENT_BINDING and "failure" not in game
            assert spec == parent.schedule("main")[source_index]
            for move in game["trace"]:
                actor, turn = move["actor"], move["turn"]
                source_planner = "LL" if actor == seat else "GG"
                cell = cell_id(spec["model_id"], phase(turn), actor, source_planner)
                row = {"pair": pair, "model_id": spec["model_id"], "phase": phase(turn),
                       "actor": actor, "turn": turn, "source_planner": source_planner,
                       "cell": cell, "source_index": source_index,
                       "source_game_path": path.as_posix(), "source_game_sha256": sha,
                       "source_root_path": h.root_path("main", source_index, turn).as_posix(),
                       "source_root_sha256": move["root_sha256"], "source_spec": spec}
                rank = old.digest(["159-position-v1", pair, source_index, turn])
                if cell not in available or rank < available[cell][0]:
                    available[cell] = (rank, row)
        eligible[pair] = sorted(available, key=lambda c: old.digest(["159-cell-v1", pair, c]))
        best[pair] = {cell: row for cell, (_, row) in available.items()}
    assigned = [[] for _ in range(CONFIG["cells"])]
    examined = []
    for pair in sorted(eligible, key=lambda p: old.digest(["159-pair-v1", p])):
        examined.append(pair)
        old.place_source(pair, eligible, assigned, CONFIG["quota"])
        if all(len(rows) == CONFIG["quota"] for rows in assigned):
            break
    assert all(len(rows) == CONFIG["quota"] for rows in assigned), "Fixed source panel cannot fill; no relaxed strata"
    rows = []
    for cell, pairs in enumerate(assigned):
        for pair in sorted(pairs):
            sample = dict(best[pair][cell])
            source = Path(sample["source_root_path"])
            assert old.file_hash(source) == sample["source_root_sha256"]
            root = old.read_json(source)
            assert root["manifest"] == PARENT_BINDING and "failure" not in root
            assert root["snapshot"]["turn_index"] == sample["turn"]
            sample.update(index=len(rows), snapshot=root["snapshot"], candidates=root["candidates"],
                          source_schedule=root["schedule"])
            rows.append(sample)
    assert len(rows) == len({s["pair"] for s in rows}) == CONFIG["states"]
    return {"samples": rows, "examined_pairs": examined,
            "eligibility": {str(k): v for k, v in eligible.items()},
            "counts": [len(v) for v in assigned], "source_game_hashes": game_hashes}


def stream(sample, stage):
    assert stage in ("selection", "evaluation")
    return old.digest(["159-root-stream-v1", stage, sample["pair"]])


def world_seeds(sample, stage):
    state = old.deserialize_state_snapshot(sample["snapshot"])
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    return [f"159-world-v1|{stage}|{stream(sample, stage)}|{belief}|{i:04d}"
            for i in range(CONFIG[f"{stage}_worlds"])]


def simulate(sample, stage, binding, selection=None):
    start, cpu = time.perf_counter(), time.process_time()
    state = old.deserialize_state_snapshot(sample["snapshot"])
    actions = [old.decode_action(a) for a in sample["candidates"]["actions"]]
    assert 1 <= len(actions) <= 4 and all(a in old.SplendorEnv().legal_actions(state) for a in actions)
    if stage == "selection":
        assert selection is None
        indices, same = list(range(len(actions))), False
    else:
        assert stage == "evaluation" and selection["index"] == sample["index"]
        indices = list(dict.fromkeys([selection["choices"]["LL"], selection["choices"]["GG"]]))
        same = len(indices) == 1
    record = {"manifest": binding, "index": sample["index"], "pair": sample["pair"],
              "stage": stage, "stream": stream(sample, stage), "action_indices": indices,
              "selection_sha256": None, "worlds": []}
    if stage == "evaluation":
        record["same_choice"] = same
    for wi, seed in enumerate([] if same else world_seeds(sample, stage)):
        world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
        row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)),
               "outcomes": {policy: [] for policy in POLICIES}}
        record["worlds"].append(row)
        for policy in POLICIES:
            for index in indices:
                action = actions[index]
                assert action in old.SplendorEnv().legal_actions(world)
                result = (old.trajectory(world, action) if policy == "GG" else
                          h.trajectory(world, action, record["stream"], wi, sample["model_id"]))
                result["wdl"] = (old.outcome_value(result["winner"], state.current_player)
                                 if result["reason"] in old.VALID_NATURAL_TERMINATIONS else None)
                row["outcomes"][policy].append(result)
                if result["wdl"] is None:
                    raise ValueError(f"Invalid159 continuation: {result}")
    if stage == "selection":
        record["means"] = {policy: [sum(w["outcomes"][policy][j]["wdl"] for w in record["worlds"])
                                      /len(record["worlds"]) for j in range(len(actions))] for policy in POLICIES}
        record["choices"] = {policy: max(indices, key=lambda j: record["means"][policy][j]) for policy in POLICIES}
    record["timing"] = {"wall": time.perf_counter()-start, "cpu": time.process_time()-cpu}
    return record


def record_path(base, stage, index):
    return Path(base)/stage/f"root_{index:03d}.json"


def collect_task(base, sample, stage, binding):
    path = record_path(base, stage, sample["index"])
    assert not path.exists(), "No automatic rerun/reuse of159 samples"
    selection = None
    if stage == "evaluation":
        selection_path = record_path(base, "selection", sample["index"])
        lock = old.read_json(Path(base)/"selection_lock.json")
        assert lock["manifest"] == binding
        assert lock["files"][str(sample["index"])] == old.file_hash(selection_path)
        selection = old.read_json(selection_path)
    try:
        result = simulate(sample, stage, binding, selection)
        if selection is not None:
            result["selection_sha256"] = old.file_hash(selection_path)
        old.save_new(path, result)
    except BaseException:
        old.save_new(Path(base)/"failures"/f"{stage}_{sample['index']:03d}.json",
                     {"time": old.now(), "sample": sample["index"], "stage": stage,
                      "traceback": traceback.format_exc()})
        raise
    return {"index": sample["index"], "sha256": old.file_hash(path),
            "wall": result["timing"]["wall"], "cpu": result["timing"]["cpu"]}


def auditor():
    name = "audit159_runtime"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPOSITORY/VERIFIER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def audit_task(base, sample, stage, binding):
    a = auditor()
    record = old.read_json(record_path(base, stage, sample["index"]))
    assert record["manifest"] == binding
    selection = None if stage == "selection" else old.read_json(record_path(base, "selection", sample["index"]))
    if stage == "selection":
        a.verify_sample(sample)
    else:
        assert record["selection_sha256"] == old.file_hash(record_path(base, "selection", sample["index"]))
    result = a.verify_record(sample, record, stage, selection=selection, replay=True)
    path = Path(base)/f"{stage}_audit"/f"root_{sample['index']:03d}.json"
    old.save_new(path, {"manifest": binding, "record_sha256": old.file_hash(record_path(base, stage, sample["index"])),
                        "audit": result})
    return {"index": sample["index"], "sha256": old.file_hash(path)}


def run_tasks(base, samples, stage, binding, audit=False):
    target = audit_task if audit else collect_task
    name = stage+"_audit" if audit else stage
    results = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=h.initialize_worker) as pool:
        futures = [pool.submit(target, base, sample, stage, binding) for sample in samples]
        try:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                event(name, completed=len(results), total=len(samples), elapsed=time.perf_counter()-start,
                      last_index=result["index"])
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    assert sorted(r["index"] for r in results) == list(range(len(samples)))
    old.save_new(Path(base)/f"{name}_lock.json", {"manifest": binding,
                 "files": {str(r["index"]): r["sha256"] for r in results},
                 "elapsed_seconds": time.perf_counter()-start,
                 "worker_cpu_seconds": sum(r.get("cpu", 0.) for r in results)})


def record_metrics(selection, evaluation):
    ci, gi = selection["choices"]["LL"], selection["choices"]["GG"]
    result = {"changed": ci != gi}
    deltas = {}
    for policy in POLICIES:
        result[f"in_sample_d_{policy}"] = (selection["means"][policy][ci]-selection["means"][policy][gi])/2
        if ci == gi:
            deltas[policy] = np.zeros(CONFIG["evaluation_worlds"])
        else:
            c, g = evaluation["action_indices"].index(ci), evaluation["action_indices"].index(gi)
            deltas[policy] = np.array([(w["outcomes"][policy][c]["wdl"]-w["outcomes"][policy][g]["wdl"])/2
                                       for w in evaluation["worlds"]])
        result[f"d_{policy}"] = float(deltas[policy].mean())
        result[f"first_half_d_{policy}"] = float(deltas[policy][:len(deltas[policy])//2].mean())
        result[f"second_half_d_{policy}"] = float(deltas[policy][len(deltas[policy])//2:].mean())
    result["interaction"] = result["d_LL"]-result["d_GG"]
    a = result["first_half_d_LL"]-result["first_half_d_GG"]
    b = result["second_half_d_LL"]-result["second_half_d_GG"]
    result.update(first_half_interaction=a, second_half_interaction=b,
                  interaction_crossproduct=a*b)
    return result


def classification(estimates):
    g, l = (estimates[f"d_{p}"]["simultaneous_ci95"] for p in POLICIES)
    m = CONFIG["material_score_difference"]
    precise = all(estimates[f"d_{p}"]["half_width"] <= CONFIG["maximum_half_width"] for p in POLICIES)
    if not precise:
        label = "inconclusive_precision_target_not_met"
    elif l[0] > m and g[1] < -m:
        label = "material_conditional_preference_reversal"
    elif l[1] < -m and g[0] > m:
        label = "material_reverse_direction_conditional_preference_reversal"
    elif l[0] > m and g[0] > m:
        label = "learned_selected_moves_better_under_both_conditional_objectives"
    elif l[1] < -m and g[1] < -m:
        label = "greedy_selected_moves_better_under_both_conditional_objectives"
    elif all(-m < bound[0] and bound[1] < m for bound in (g, l)):
        label = "small_average_fixed_choice_effects_only"
    else:
        label = "inconclusive_for_material_classification"
    return {"label": label, "precision_target_met": precise,
            "caution": "Conditional mean effects only; no gameplay oracle, local equivalence or causal explanation of158."}


def statistics(rows):
    assert len(rows) == CONFIG["states"] == len({r["pair"] for r in rows})
    assert Counter(r["cell"] for r in rows) == {i: CONFIG["quota"] for i in range(CONFIG["cells"])}
    keys = ["d_GG", "d_LL", "interaction", "interaction_crossproduct", "changed"]
    values = np.array([[[r[k] for k in keys] for r in rows if r["cell"] == cell]
                       for cell in range(CONFIG["cells"])], dtype=float)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    batches = []
    for start in range(0, CONFIG["bootstrap_replicates"], 1000):
        n = min(1000, CONFIG["bootstrap_replicates"]-start)
        drawn = rng.integers(CONFIG["quota"], size=(n, CONFIG["cells"], CONFIG["quota"]))
        sampled = values[np.arange(CONFIG["cells"])[None, :, None], drawn].mean(axis=2)
        observed = values.mean(axis=1)
        calibrated = observed[None]+(sampled-observed[None])*math.sqrt(CONFIG["quota"]/(CONFIG["quota"]-1))
        batches.append(calibrated.mean(axis=1))
    boot = np.concatenate(batches)
    estimates = {}
    for i, key in enumerate(keys):
        estimate = float(values[:, :, i].mean())
        if key in ("d_GG", "d_LL"):
            components = values[:, :, i].var(axis=1, ddof=1)/CONFIG["quota"]
            se = float(math.sqrt(components.sum())/CONFIG["cells"])
            df = (None if se == 0 else float(components.sum()**2 /
                  ((components**2).sum()/(CONFIG["quota"]-1))))
            half_width = 0. if se == 0 else float(student_t.ppf(.9875, df)*se)
            interval = [estimate-half_width, estimate+half_width]
            estimates[key] = {"mean": estimate, "simultaneous_ci95": interval,
                              "half_width": half_width, "standard_error": se, "degrees_of_freedom": df,
                              "method": "Bonferroni_pair_of_97.5_percent_marginal_Welch_Satterthwaite_t_intervals"}
        else:
            estimates[key] = {"mean": estimate, "exploratory_ci95": np.quantile(boot[:, i], [.025, .975]).tolist()}
    strata = {}
    for name in ("phase", "model_id", "actor", "source_planner"):
        strata[name] = {str(value): {"roots": len(part), **{k: float(np.mean([r[k] for r in part])) for k in keys}}
                        for value in sorted({r[name] for r in rows})
                        for part in [[r for r in rows if r[name] == value]]}
    return {"roots": len(rows), "independent_source_pairs": len(rows), "estimates": estimates,
            "classification": classification(estimates), "descriptive_subgroups": strata,
            "units": "win1_draw0.5_loss0; action difference learned-selected minus greedy-selected",
            "uncertainty": "Source-pair stratified resampling, conditional on3 fixed models and the balanced visited-state panel."}


def tests():
    folder = OUTPUT/"preparation"
    folder.mkdir(parents=True, exist_ok=True)
    before = identity()
    start = time.perf_counter()
    paths = ["tests/test_planner_preference_159.py", "tests/test_learned_continuation_158.py",
             "tests/test_rollout_selector_154.py"]
    with (folder/"tests.txt").open("x", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q", *paths], stdout=log, stderr=subprocess.STDOUT)
    assert identity() == before
    old.save_new(folder/"tests.json", {"identity": before, "exit_code": completed.returncode,
                 "paths": paths, "full_repository": False, "elapsed_seconds": time.perf_counter()-start,
                 "log_sha256": old.file_hash(folder/"tests.txt")})
    print((folder/"tests.txt").read_text(encoding="utf-8")[-6000:], flush=True)
    return completed.returncode


def prepare():
    assert not (OUTPUT/"manifest.json").exists()
    manifest = identity()
    evidence = old.read_json(OUTPUT/"preparation/tests.json")
    assert evidence["identity"] == manifest and evidence["exit_code"] == 0
    assert evidence["log_sha256"] == old.file_hash(OUTPUT/"preparation/tests.txt")
    panel = select_panel()
    h.initialize_worker()
    source_checks = [auditor().verify_sample(sample) for sample in panel["samples"]]
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"panel.json", panel)
    old.save_new(OUTPUT/"source_preflight.json", {"passed": True, "manifest": old.digest(manifest),
                 "checks": source_checks, "all_selected_prefixes_verified": True})
    old.save_new(OUTPUT/"freeze.json", {"time": old.now(), "manifest": old.digest(manifest),
                 "panel_sha256": old.file_hash(OUTPUT/"panel.json"),
                 "source_preflight_sha256": old.file_hash(OUTPUT/"source_preflight.json"),
                 "tests_sha256": old.file_hash(OUTPUT/"preparation/tests.json"), "owner_authorized": True})
    event("prepared", roots=len(panel["samples"]), counts=panel["counts"], binding=old.digest(manifest))


def validate_freeze():
    binding = check_identity()
    frozen = old.read_json(OUTPUT/"freeze.json")
    assert frozen["manifest"] == binding and frozen["owner_authorized"]
    assert frozen["panel_sha256"] == old.file_hash(OUTPUT/"panel.json")
    assert frozen["source_preflight_sha256"] == old.file_hash(OUTPUT/"source_preflight.json")
    assert old.read_json(OUTPUT/"source_preflight.json")["passed"]
    assert frozen["tests_sha256"] == old.file_hash(OUTPUT/"preparation/tests.json")
    evidence = old.read_json(OUTPUT/"preparation/tests.json")
    assert evidence["identity"] == old.read_json(OUTPUT/"manifest.json") and evidence["exit_code"] == 0
    assert evidence["log_sha256"] == old.file_hash(OUTPUT/"preparation/tests.txt")
    return binding


def finish(samples, binding):
    audit_counts = Counter()
    for stage in ("selection", "evaluation", "selection_audit", "evaluation_audit"):
        lock = old.read_json(OUTPUT/f"{stage}_lock.json")
        assert lock["manifest"] == binding and set(lock["files"]) == {str(i) for i in range(len(samples))}
        for i, sha in lock["files"].items():
            assert old.file_hash(record_path(OUTPUT, stage, int(i))) == sha
        if stage.endswith("_audit"):
            source_stage = stage.removesuffix("_audit")
            collection = old.read_json(OUTPUT/f"{source_stage}_lock.json")
            for sample in samples:
                checked = old.read_json(record_path(OUTPUT, stage, sample["index"]))
                assert checked["manifest"] == binding
                assert checked["record_sha256"] == collection["files"][str(sample["index"])]
                for key in ("continuations", "moves", "replayed_branches"):
                    audit_counts[key] += checked["audit"][key]
    assert audit_counts["continuations"] <= CONFIG["main_continuation_cap"]
    assert audit_counts["replayed_branches"] <= 4320
    for path, sha in old.read_json(OUTPUT/"panel.json")["source_game_hashes"].items():
        assert old.file_hash(Path(path)) == sha, "Source game changed during159"
    for sample in samples:
        assert old.file_hash(Path(sample["source_root_path"])) == sample["source_root_sha256"]
    rows = []
    for sample in samples:
        selection = old.read_json(record_path(OUTPUT, "selection", sample["index"]))
        evaluation = old.read_json(record_path(OUTPUT, "evaluation", sample["index"]))
        auditor().a158.compare(record_metrics(selection, evaluation), auditor().independent_metrics(selection, evaluation))
        rows.append({**{k: sample[k] for k in ("index", "pair", "cell", "phase", "actor", "model_id", "source_planner")},
                     **record_metrics(selection, evaluation)})
    stats = statistics(rows)
    auditor().a158.compare(stats, auditor().independent_statistics(rows))
    result = {"manifest": binding, "time": old.now(), **stats, "rows": rows,
              "operational": dict(audit_counts),
              "audit_scope": "All source prefixes/proposals, all world hashes/outcomes/arithmetic; exactly2 worlds/stage/root independently replayed. Not full trajectory replay.",
              "parent_caveat": "158 full-game integrity replay was stopped by owner at19/402; unchanged and incomplete."}
    assert validate_freeze() == binding
    old.save_new(OUTPUT/"result.json", result)
    g, l = [result["estimates"][f"d_{p}"] for p in POLICIES]
    with (OUTPUT/"report.md").open("x", encoding="utf-8") as handle:
        handle.write("# Experiment159: fixed-move preference replication\n\n"
                     f"{len(rows)} distinct source pairs;32 selection worlds,128 independent evaluation worlds.\n\n"
                     "Learned-selected minus greedy-selected move score:\n\n"
                     f"- Greedy futures: {100*g['mean']:+.3f} percentage points, simultaneous95% interval "
                     f"[{100*g['simultaneous_ci95'][0]:+.3f}, {100*g['simultaneous_ci95'][1]:+.3f}] points.\n"
                     f"- Learned futures: {100*l['mean']:+.3f} percentage points, simultaneous95% interval "
                     f"[{100*l['simultaneous_ci95'][0]:+.3f}, {100*l['simultaneous_ci95'][1]:+.3f}] points.\n\n"
                     f"Classification: {result['classification']['label']}. Precision target met: {result['classification']['precision_target_met']}.\n\n"
                     "Neither continuation policy is ground truth. These are conditional average effects, not actual game win rates; cancellation and model bias remain possible. "
                     "Subgroup and cross-half interaction results are exploratory.158 remains incompletely replayed. "
                     "159 uses complete artifact/arithmetic checks and a fixed sampled continuation replay, not full replay. No training, promotion, successor or scheduled monitoring.\n")
    old.save_new(OUTPUT/"final_status.json", {"complete": True, "time": old.now(), "manifest": binding,
                 "result_sha256": old.file_hash(OUTPUT/"result.json"),
                 "prespecified_audit_pass": True, "full_trajectory_replay": False,
                 "classification": result["classification"], "automatic_successor": False})
    event("complete", classification=result["classification"])


def run():
    with old.process_lock(OUTPUT/"run.lock"):
        binding = validate_freeze()
        assert not (OUTPUT/"launch.json").exists(), "One-shot launch; no automatic resume"
        samples = old.read_json(OUTPUT/"panel.json")["samples"]
        old.save_new(OUTPUT/"launch.json", {"time": old.now(), "pid": os.getpid(), "manifest": binding,
                     "owner_authorized": True, "roots": len(samples)})
        try:
            run_tasks(OUTPUT, samples, "selection", binding)
            run_tasks(OUTPUT, samples, "selection", binding, audit=True)
            assert validate_freeze() == binding
            run_tasks(OUTPUT, samples, "evaluation", binding)
            run_tasks(OUTPUT, samples, "evaluation", binding, audit=True)
            finish(samples, binding)
        except BaseException:
            old.save_new(OUTPUT/"failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


def launch():
    binding = validate_freeze()
    folder = OUTPUT/"preparation"
    with old.process_lock(folder/"launcher.lock"):
        assert not (folder/"process.json").exists() and not (OUTPUT/"launch.json").exists()
        command = [sys.executable, "-u", str(REPOSITORY/"run_planner_preference_159.py"), "run"]
        flags = 0
        startup = None
        if os.name == "nt":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_BREAKAWAY_FROM_JOB
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
        with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=REPOSITORY, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       creationflags=flags, startupinfo=startup)
        old.save_new(folder/"process.json", {"time": old.now(), "pid": process.pid,
                     "command": command, "manifest": binding, "creationflags": flags})
        event("detached_launch", pid=process.pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tests", "prepare", "launch", "run"))
    args = parser.parse_args()
    return {"tests": tests, "prepare": prepare, "launch": launch, "run": run}[args.command]()
