"""Fresh fourth-slot proposal screen; isolated from all frozen source modules."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import run_afterstate_advantage as old
from splendor_ai.alphazero import run_rollout_selector_gameplay as baseline
from splendor_ai.alphazero import windowless_workers_160 as windows
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/learned_proposal_screen"
ARMS = ("reference", "raw_control", "learned_161001", "learned_161002", "learned_161003")
MODELS = {
    161001: (563, "8d83d69b9691569f187beb93f2636635ab419c835442d02f0eb6fc511866fa62"),
    161002: (488, "b06e37d3640adb846de562daaaae4d526bf586a848a1e1783f2bc4d2c3899c96"),
    161003: (563, "8d4a258f625003e0fad9241560db0f606c23d1d8fab3a9c1afd3bb0c2f11a66a"),
}
CONFIG = {
    "states": {"probe": 18, "main": 288}, "quota": {"probe": 3, "main": 48},
    "source_limit": {"probe": 128, "main": 1024},
    "source_seed": {"probe": 2039000000, "main": 2030000000},
    "world_seed": {"probe_selection": 2039100000, "probe_evaluation": 2039200000,
                   "main_selection": 2031000000, "main_evaluation": 2032000000},
    "worlds": {"selection": 32, "evaluation": 128}, "privacy_seed": 2039300000,
    "bootstrap_seed": 2033000000, "bootstrap_samples": 50000, "bootstrap_chunk": 1000,
    "position_tag": "learned-proposal-position-v1", "stream_tag": "learned-proposal-root-v1",
    "workers": 8, "torch_threads": 1, "device": "cpu", "source_batch": 8,
    "maximum_union": 8, "maximum_choices": 5, "maximum_root_continuations": 896,
    "main_continuation_cap": 258048, "probe_continuation_cap": 16128,
    "maximum_forecast_hours": 8., "disk_reserve_bytes": 5 * 1024**3,
    "training": False, "full_inner_replay": True, "automatic_retry": False,
    "promotion": False, "automatic_successor": False, "scheduled_monitoring": False,
}
EXTRAS = (
    "run_learned_proposal_screen.py", "scripts/analysis/learned_proposal_screen.py",
    "scripts/analysis/verify_learned_proposal_screen.py", "tests/test_learned_proposal_screen.py",
    "scripts/analysis/verify_rollout_selector_154.py", "scripts/analysis/verify_afterstate_advantage_149.py",
    "docs/engineering/learned_proposal_screen.md", "docs/strategy/next_after_saved_halving.md",
)
_MODELS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def artifact(path, **metadata):
    path = Path(path)
    return {"path": path.resolve().relative_to(ROOT).as_posix(), "sha256": old.file_hash(path), **metadata}


def verified(item):
    path = ROOT / item["path"]
    require(path.resolve().is_relative_to(ROOT), "Artifact outside repository")
    require(old.file_hash(path) == item["sha256"], f"Changed artifact: {path}")
    return old.read_json(path)


def model_path(seed):
    update, expected = MODELS[seed]
    path = ROOT / "outputs/teacher_budget_161/fits" / f"world128_{seed}" / f"checkpoint_{update:04d}.pt"
    require(old.file_hash(path) == expected, "161 checkpoint changed")
    return path


def initialize_models():
    old.initialize_worker()
    _MODELS.clear()
    for seed in MODELS:
        _MODELS[seed] = StrongPolicyInference(model_path(seed)).model


def projection(proposals):
    """All scientific fields, excluding measurements of evaluation latency."""
    return {key: value for key, value in proposals.items() if key != "timing"}


def assemble_slots(group_keys, representatives, reference_keys, raw_rank, learned_scores):
    require(1 <= len(reference_keys) <= 4 and len(set(reference_keys)) == len(reference_keys), "Bad reference support")
    require(len(group_keys) == len(representatives) == len(set(group_keys)), "Bad canonical support")
    require(set(raw_rank) == set(group_keys) and len(raw_rank) == len(group_keys), "Raw ranking coverage")
    require(set(reference_keys).issubset(group_keys), "Unknown reference key")
    arm_keys = {"reference": list(reference_keys)}
    if len(reference_keys) < 4:
        require(len(reference_keys) == len(group_keys), "Short support discarded legal groups")
        arm_keys.update({arm: list(reference_keys) for arm in ARMS[1:]})
    else:
        common = reference_keys[:3]
        remaining = [key for key in group_keys if key not in common]
        arm_keys["raw_control"] = common + [next(key for key in raw_rank if key in remaining)]
        for seed in MODELS:
            values = learned_scores[str(seed)]
            require(len(values) == len(group_keys) and np.isfinite(values).all(), "Invalid learned group scores")
            best = max((j for j, key in enumerate(group_keys) if key in remaining), key=lambda j: values[j])
            arm_keys[f"learned_{seed}"] = common + [group_keys[best]]
    union = []
    for arm in ARMS:
        for key in arm_keys[arm]:
            if key not in union:
                union.append(key)
    require(len(union) <= 8, "Union exceeds frozen cap")
    by_key = dict(zip(group_keys, representatives, strict=True))
    return {"union_keys": union, "union_actions": [by_key[key] for key in union],
            "arm_indices": {arm: [union.index(key) for key in arm_keys[arm]] for arm in ARMS},
            "arm_keys": arm_keys}


def proposals(state, stream):
    wall, cpu = time.perf_counter(), time.process_time()
    env = old.SplendorEnv()
    legal = env.legal_actions(state)
    schedule = baseline.root_schedule(state, stream)
    reference = baseline.candidate_set(state, schedule["candidate_nonce"])
    raw_scores = old._RAW.score_actions(state, legal)
    require(len(raw_scores) == len(legal) and np.isfinite(raw_scores).all(), "Invalid035 scores")
    groups = old.actions_by_canonical_successor_key(state, legal)
    keys = list(groups)
    key_strings = [repr(key) for key in keys]
    representatives = {repr(key): old.serialize_action(action) for key, action in groups.items()}
    # The top two preserve exact035 concrete actions; others keep first-legal representatives.
    representatives.update(zip(reference["keys"], reference["actions"], strict=True))
    raw_rank = []
    for index in sorted(range(len(legal)), key=lambda i: (-raw_scores[i], i)):
        key = repr(old.canonical_successor_action_key(state, legal[index]))
        if key not in raw_rank:
            raw_rank.append(key)
    raw_seconds = time.perf_counter() - wall
    logits, scores, timings = {}, {}, {}
    for seed, model in _MODELS.items():
        started = time.perf_counter()
        with torch.no_grad():
            row, encoded_keys, _ = encode_position(state, model.config, legal)
            values, _ = model(row["observation"][None], row["features"][None],
                              torch.ones((1, len(legal)), dtype=torch.bool))
        require(encoded_keys == keys and torch.isfinite(values).all().item(), "Legal-group encoding mismatch")
        logits[str(seed)] = values[0].tolist()
        scores[str(seed)] = [float(torch.amax(values[0][row["groups"] == j])) for j in range(len(keys))]
        timings[str(seed)] = time.perf_counter() - started
    require(set(logits) == {str(seed) for seed in MODELS}, "All three frozen fits required")
    slots = assemble_slots(key_strings, [representatives[key] for key in key_strings],
                           reference["keys"], raw_rank, scores)
    require([slots["union_actions"][j] for j in slots["arm_indices"]["reference"]] == reference["actions"],
            "Original reference actions changed")
    return {"schedule": schedule, "reference": reference, "group_keys": key_strings,
            "representatives": [representatives[key] for key in key_strings],
            "raw_rank": raw_rank, "raw_scores": list(map(float, raw_scores)),
            "learned_logits": logits, "learned_group_scores": scores,
            "legal_actions": [old.serialize_action(a) for a in legal], **slots,
            "timing": {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu,
                       "raw_proposal_wall": raw_seconds, "learned_extra_wall": timings}}


def path_for(stage, kind, index):
    require(stage in ("probe", "main"), "Unknown stage")
    require(type(index) is int and index >= 0, "Invalid index")
    return OUTPUT / stage / kind / f"record_{index:04d}.json"


def source_seed(stage, index):
    require(stage in ("probe", "main") and 0 <= index < CONFIG["source_limit"][stage], "Source index outside budget")
    return CONFIG["source_seed"][stage] + index


def source_task(stage, index, binding):
    seed = source_seed(stage, index)
    wall, cpu = time.perf_counter(), time.process_time()
    record = old.trajectory(old.SplendorEnv(seed=seed).initial_state(), include_trace=True)
    path = path_for(stage, "sources", index)
    old.save_new(path, {"manifest": binding, "stage": stage, "seed": seed, "index": index, **record,
                        "timing": {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}})
    require(record["reason"] in old.VALID_NATURAL_TERMINATIONS, "Non-natural source; no replacement")
    return artifact(path, index=index, turns=record["turns"])


def select_panel(stage, source_items):
    buckets, cells, positions, filtering = [[] for _ in range(6)], {}, {}, []
    quota = CONFIG["quota"][stage]
    for item in source_items:
        record = verified(item)
        index = len(filtering)
        require(record["index"] == item["index"] == index and record["seed"] == source_seed(stage, index), "Source order")
        require(record["reason"] in old.VALID_NATURAL_TERMINATIONS, "Invalid source")
        states = old.reconstruct_source(record)
        eligible = [[s for s in states if old.eligible(s, c)] for c in range(6)]
        cells[index] = [c for c in range(6) if eligible[c]]
        positions[index] = {c: min(eligible[c], key=lambda s: old.digest([CONFIG["position_tag"], stage, index, s.turn_index]))
                            for c in cells[index]}
        filtering.append({"source_index": index, "eligible_by_cell": [len(v) for v in eligible],
                          "sha256": item["sha256"]})
        old.place_source(index, cells, buckets, quota)
        if all(len(bucket) == quota for bucket in buckets):
            break
    require([len(b) for b in buckets] == [quota] * 6, "Fixed source ceiling cannot fill panel")
    assignments = sorted((index, c) for c, bucket in enumerate(buckets) for index in bucket)
    samples = []
    for index, (source_index, cell) in enumerate(assignments):
        state = positions[source_index][cell]
        samples.append({"index": index, "stage": stage, "source_index": source_index,
                        "source_seed": source_seed(stage, source_index), "cell": cell,
                        "actor": state.current_player, "turn": state.turn_index,
                        "stream": old.digest([CONFIG["stream_tag"], stage, source_index]),
                        "snapshot": old.serialize_state_snapshot(state)})
    require(len(samples) == CONFIG["states"][stage], "Panel size mismatch")
    return {"samples": samples, "filtering": filtering, "counts": [quota] * 6}


def proposal_task(stage, sample, binding):
    state = old.deserialize_state_snapshot(sample["snapshot"])
    result = proposals(state, sample["stream"])
    path = path_for(stage, "proposals", sample["index"])
    old.save_new(path, {"manifest": binding, "sample_sha256": old.digest(sample), "index": sample["index"], **result})
    return artifact(path, index=sample["index"])


def world_seed(stage, kind, index, world):
    require(kind in ("selection", "evaluation") and stage in ("probe", "main"), "World stage")
    require(0 <= index < CONFIG["states"][stage] and 0 <= world < CONFIG["worlds"][kind], "World outside cap")
    return CONFIG["world_seed"][f"{stage}_{kind}"] + index * 1000 + world


def choose_arms(record, offered):
    indices = record["action_indices"]
    require(indices == list(range(len(offered["union_actions"]))) and len(record["worlds"]) == 32, "Selection support/budget")
    totals = [0] * len(indices)
    for world in record["worlds"]:
        require(len(world["outcomes"]) == len(indices), "Incomplete world")
        for j, outcome in enumerate(world["outcomes"]):
            require(outcome["reason"] in old.VALID_NATURAL_TERMINATIONS and outcome["wdl"] in (-1., 0., 1.), "Invalid outcome")
            totals[j] += outcome["wdl"]
    return {"choices": {arm: max(offered["arm_indices"][arm], key=lambda j: totals[j]) for arm in ARMS},
            "union_selection_wdl": [value / 32 for value in totals]}


def collect_task(stage, kind, sample, proposal_item, choices, binding):
    wall, cpu = time.perf_counter(), time.process_time()
    offered = verified(proposal_item)
    require(offered["manifest"] == binding and offered["sample_sha256"] == old.digest(sample), "Proposal dependency")
    actions = offered["union_actions"]
    if kind == "selection":
        require(choices is None, "Selection received future choices")
        indices = list(range(len(actions)))
    else:
        require(set(choices) == set(ARMS), "Missing locked choices")
        indices = sorted(set(choices.values()))
        require(len(indices) <= 5 and all(0 <= i < len(actions) for i in indices), "Invalid evaluation choices")
    root = old.deserialize_state_snapshot(sample["snapshot"])
    path = path_for(stage, kind, sample["index"])
    require(not path.exists(), "Existing target; no automatic retry")
    worlds = []
    record = {"manifest": binding, "index": sample["index"], "sample_sha256": old.digest(sample),
              "proposal": proposal_item, "action_indices": indices, "worlds": worlds,
              "kind": kind, "stage": stage, "locked_choices": choices}
    try:
        for wi in range(CONFIG["worlds"][kind]):
            seed = world_seed(stage, kind, sample["index"], wi)
            world = old.determinize_for_observer(root, root.current_player, random.Random(seed))
            outcomes = []
            worlds.append({"seed": seed, "world_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": outcomes})
            for ai in indices:
                action = old.decode_action(actions[ai])
                require(action in old.SplendorEnv().legal_actions(world), "Proposed action illegal in sampled world")
                value = old.trajectory(world, action)
                value["wdl"] = old.outcome_value(value["winner"], root.current_player) if value["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                outcomes.append(value)
                require(value["wdl"] is not None, "Invalid continuation; no exclusion/replacement")
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
        old.save_new(path, record)
    return artifact(path, index=sample["index"], continuations=len(indices) * len(worlds),
                    moves=sum(o["turns"] for w in worlds for o in w["outcomes"]))


def per_root(sample, offered, selected, evaluation):
    require(evaluation["action_indices"] == sorted(set(selected["choices"].values())), "Evaluation choice mismatch")
    require(len(evaluation["worlds"]) == 128, "Evaluation world count")
    lookup = {ai: j for j, ai in enumerate(evaluation["action_indices"])}
    scores = {}
    for arm in ARMS:
        j = lookup[selected["choices"][arm]]
        scores[arm] = sum((w["outcomes"][j]["wdl"] + 1) / 2 for w in evaluation["worlds"]) / 128
    learned = sum(scores[arm] for arm in ARMS[2:]) / 3
    return {"index": sample["index"], "source_seed": sample["source_seed"], "cell": sample["cell"],
            "scores": scores, "learned_mean": learned, "primary": learned - scores["reference"],
            "attribution": learned - scores["raw_control"], "choices": selected["choices"],
            "canonical_support": offered["reference"]["canonical_support"],
            "legal_support": offered["reference"]["legal_support"],
            "arm_indices": offered["arm_indices"], "proposal_timing": offered["timing"],
            "fourth_raw_ranks": {arm: (offered["raw_rank"].index(offered["arm_keys"][arm][-1]) + 1)
                                 for arm in ARMS}}


def statistics(rows):
    require(len(rows) == 288 and len({r["source_seed"] for r in rows}) == 288, "Independent source count")
    arrays = [np.array([[r["primary"], r["attribution"]] for r in rows if r["cell"] == cell]) for cell in range(6)]
    require(all(a.shape == (48, 2) and np.isfinite(a).all() for a in arrays), "Balanced paired endpoint")
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = np.empty((50000, 2))
    for start in range(0, 50000, 1000):
        boot[start:start + 1000] = sum(a[rng.integers(48, size=(1000, 48))].mean(axis=1) for a in arrays) / 6
    means = np.stack(arrays).mean(axis=(0, 1))
    ci = np.quantile(boot, [.025, .975], axis=0)
    result = {key: {"delta_pp": float(100 * means[j]), "ci95_pp": (100 * ci[:, j]).tolist()}
              for j, key in enumerate(("primary", "attribution"))}
    result["conditional_learning_contribution_established"] = bool((ci[0] > 0).all())
    result["classification"] = "positive_conditional_learning_contribution" if (ci[0] > 0).all() else "learning_contribution_not_established"
    return result


def auditor():
    return importlib.import_module("scripts.analysis.verify_learned_proposal_screen")


def identity():
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / path for path in EXTRAS]
    return {"config": CONFIG, "arms": list(ARMS), "models": {str(k): list(v) for k, v in MODELS.items()},
            "dependencies": {path.relative_to(ROOT).as_posix(): old.file_hash(path) for path in paths},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    base = ROOT / "outputs/teacher_budget_161"
    final = old.read_json(base / "final_status.json")
    require(final["complete"] and final["all_prescribed_audits_pass"]
            and final["manifest"] == "f8f1fda4884cdcde518ad079430528b95abaf19e5b47946630dd5c4e5f22eeeb", "161 not complete/audited")
    result = verified(final["result"])
    require(result["manifest"] == final["manifest"], "Wrong161 result")
    manifest = old.read_json(base / "manifest.json")
    require(old.digest(manifest) == final["manifest"], "161 manifest changed")
    for name, expected in manifest["dependencies"].items():
        require(old.file_hash(ROOT / name) == expected, f"Changed161 dependency: {name}")
    selected = old.read_json(base / "selected_lock.json")
    require(selected["manifest"] == final["manifest"], "Wrong selected161 binding")
    fit_lock = verified(selected["fits_lock"])
    audit_lock = verified(selected["fits_audit_lock"])
    evidence = [artifact(base / name) for name in ("final_status.json", "manifest.json", "selected_lock.json")]
    evidence += [final["result"], final["report"], selected["fits_lock"], selected["fits_audit_lock"]]
    models = {}
    for seed, (update, expected) in MODELS.items():
        item = selected["models"][f"world128_{seed}"]
        require(item["path"] == model_path(seed).relative_to(ROOT).as_posix()
                and item["sha256"] == expected and item["update"] == update, "Wrong selected model")
        fit_item = next(x for x in fit_lock["files"] if x["arm"] == "world128" and x["seed"] == seed)
        audit_item = next(x for x in audit_lock["files"] if x["arm"] == "world128" and x["seed"] == seed)
        fit, audit = verified(fit_item), verified(audit_item)
        chosen = min(fit["curves"], key=lambda row: (row["pair_loss"], row["update"]))
        require(audit["complete_fit_bitwise_identical"] and audit["result_sha256"] == fit_item["sha256"]
                and audit["selected"] == fit["selected_update"] == chosen["update"] == update
                and chosen["checkpoint_sha256"] == expected, "161 fit selection/audit mismatch")
        evidence.extend((fit_item, audit_item))
        models[str(seed)] = item
    require(old.file_hash(old.CHECKPOINT) == old.CHECKPOINT_SHA, "035 checkpoint changed")
    # Check recorded inventories, not dynamic glob-based identities of completed studies.
    panel = old.read_json(ROOT / "outputs/student_generalization_panel/manifest.json")
    for name, expected in panel["identity"]["dependencies"].items():
        require(old.file_hash(ROOT / name) == expected, f"Completed panel dependency changed: {name}")
    evidence.extend([artifact(ROOT / "outputs/student_generalization_panel/manifest.json"), artifact(old.CHECKPOINT)])
    return {"models": models, "files": evidence}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    targets = ["tests/test_learned_proposal_screen.py", "tests/test_rollout_selector_154.py",
               "tests/test_fast_selector_transfer_157.py", "tests/test_windowless_workers_160.py"]
    code = "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(" + repr(["-q", *targets]) + "))"
    started = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as stdout, (folder / "stderr.txt").open("x", encoding="utf-8") as stderr:
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=stdout, stderr=stderr,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Identity changed during checks")
    old.save_new(folder / "checks.json", {"identity": before, "exit_code": result.returncode,
                 "targets": targets, "seconds": time.perf_counter() - started,
                 "logs": [artifact(folder / name) for name in ("stdout.txt", "stderr.txt")]})
    return result.returncode


def prepare(checks_path):
    require(not (OUTPUT / "manifest.json").exists(), "Already prepared")
    evidence = old.read_json(checks_path)
    require(evidence["identity"] == identity() and evidence["exit_code"] == 0, "Current passing checks required")
    for item in evidence["logs"]:
        require(old.file_hash(ROOT / item["path"]) == item["sha256"], "Check log changed")
    parents = parent_evidence()
    manifest = {"identity": identity(), "parent": parents, "checks": artifact(checks_path), "owner_authorized": True}
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": artifact(OUTPUT / "manifest.json"),
                 "binding": old.digest(manifest), "time": old.now(), "owner_authorized": True})
    print({"prepared": True, "binding": old.digest(manifest)}, flush=True)


def validate_freeze():
    freeze = old.read_json(OUTPUT / "freeze.json")
    manifest = verified(freeze["manifest"])
    require(old.digest(manifest) == freeze["binding"] and freeze["owner_authorized"], "Freeze mismatch")
    # JSON converts tuples to lists, so compare canonical digests.
    require(old.digest(manifest["identity"]) == old.digest(identity()), "Frozen code/config/runtime changed")
    checks = verified(manifest["checks"])
    require(checks["exit_code"] == 0 and old.digest(checks["identity"]) == old.digest(identity()), "Check identity mismatch")
    for item in [*checks["logs"], *manifest["parent"]["files"], *manifest["parent"]["models"].values()]:
        require(old.file_hash(ROOT / item["path"]) == item["sha256"], "Frozen evidence changed")
    return freeze["binding"]


def event(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    old.atomic_write(OUTPUT / "progress.json", row)
    print(row, flush=True)


def worker_initialize(stage, binding):
    folder = OUTPUT / "workers" / stage
    folder.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt", folder / f"{os.getpid()}_stderr.txt")
    initialize_models()
    require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "Worker arithmetic config")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, **health})


def dispatch(kind, arguments):
    functions = {"source": source_task, "proposal": proposal_task, "collect": collect_task,
                 "source_audit": auditor().source_check, "proposal_audit": auditor().proposal_check,
                 "continuation_audit": auditor().continuation_check}
    return functions[kind](*arguments)


def run_jobs(stage, kind, arguments, binding, upstream):
    require(validate_freeze() == binding, "Stage freeze mismatch")
    stage_launch = OUTPUT / f"{stage}_launch.json"
    dependencies = [artifact(path) for path in upstream]
    old.save_new(stage_launch, {"manifest": binding, "time": old.now(), "jobs": len(arguments),
                               "dependencies": dependencies, "kind": kind})
    runtime = windows.configure_spawn()
    started = time.perf_counter()
    files = []
    event(stage, completed=0, total=len(arguments))
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, binding)) as pool:
        jobs = [pool.submit(dispatch, kind, args) for args in arguments]
        try:
            for job in as_completed(jobs):
                files.append(job.result())
                if len(files) % 16 == 0 or len(files) == len(jobs):
                    event(stage, completed=len(files), total=len(jobs))
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    files.sort(key=lambda row: row["index"])
    require(len(files) == len(arguments) and len({r["index"] for r in files}) == len(files), "Incomplete/duplicate jobs")
    for item in [*files, *dependencies]:
        require(old.file_hash(ROOT / item["path"]) == item["sha256"], "Stage artifact changed")
    require(validate_freeze() == binding, "Stage changed frozen evidence")
    lock = {"manifest": binding, "launch": artifact(stage_launch), "files": files,
            "runtime": runtime, "elapsed_seconds": time.perf_counter() - started}
    old.save_new(OUTPUT / f"{stage}_lock.json", lock)
    return files


def verify_stage(stage, binding):
    lock = old.read_json(OUTPUT / f"{stage}_lock.json")
    require(lock["manifest"] == binding, "Stage binding")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["jobs"] == len(lock["files"]), "Launch accounting")
    for item in [*lock["files"], *launch["dependencies"]]:
        require(old.file_hash(ROOT / item["path"]) == item["sha256"], "Stage hash mismatch")
    return lock


def build_stage(stage, binding):
    started = time.perf_counter()
    sources = run_jobs(f"{stage}_sources", "source",
                      [(stage, i, binding) for i in range(CONFIG["source_limit"][stage])], binding, [OUTPUT / "freeze.json"])
    audits = run_jobs(f"{stage}_sources_audit", "source_audit",
                     [(stage, item, binding) for item in sources], binding, [OUTPUT / f"{stage}_sources_lock.json"])
    panel = select_panel(stage, sources)
    old.save_new(OUTPUT / stage / "panel.json", {"manifest": binding, **panel})
    auditor().verify_matching(stage, panel, audits, sources)
    old.save_new(OUTPUT / stage / "panel_audit.json", {"manifest": binding, "passed": True,
                 "panel": artifact(OUTPUT / stage / "panel.json"), "source_audit": artifact(OUTPUT / f"{stage}_sources_audit_lock.json")})
    samples = panel["samples"]
    offered = run_jobs(f"{stage}_proposals", "proposal", [(stage, sample, binding) for sample in samples],
                       binding, [OUTPUT / stage / "panel_audit.json"])
    run_jobs(f"{stage}_proposals_audit", "proposal_audit",
             [(stage, s, item, binding) for s, item in zip(samples, offered)], binding,
             [OUTPUT / f"{stage}_proposals_lock.json", OUTPUT / stage / "panel_audit.json"])
    discovery = run_jobs(f"{stage}_selection", "collect",
                         [(stage, "selection", s, item, None, binding) for s, item in zip(samples, offered)], binding,
                         [OUTPUT / f"{stage}_proposals_audit_lock.json", OUTPUT / stage / "panel.json"])
    run_jobs(f"{stage}_selection_audit", "continuation_audit",
             [(stage, "selection", s, p, d, None, binding) for s, p, d in zip(samples, offered, discovery)], binding,
             [OUTPUT / f"{stage}_selection_lock.json", OUTPUT / f"{stage}_proposals_audit_lock.json"])
    choices = []
    for sample, p, d in zip(samples, offered, discovery):
        record = choose_arms(verified(d), verified(p))
        require(record == auditor().independent_choice(verified(d), verified(p)), "Independent selection mismatch")
        choices.append({"index": sample["index"], "sample_sha256": old.digest(sample), **record})
    choices_path = OUTPUT / stage / "choices.json"
    old.save_new(choices_path, {"manifest": binding, "selection_audit": artifact(OUTPUT / f"{stage}_selection_audit_lock.json"),
                              "rows": choices, "time": old.now()})
    evaluation = run_jobs(f"{stage}_evaluation", "collect",
                          [(stage, "evaluation", s, p, c["choices"], binding) for s, p, c in zip(samples, offered, choices)],
                          binding, [choices_path, OUTPUT / f"{stage}_proposals_lock.json", OUTPUT / stage / "panel.json"])
    run_jobs(f"{stage}_evaluation_audit", "continuation_audit",
             [(stage, "evaluation", s, p, e, c["choices"], binding) for s, p, e, c in zip(samples, offered, evaluation, choices)],
             binding, [OUTPUT / f"{stage}_evaluation_lock.json", choices_path])
    count = sum(item["continuations"] for item in [*discovery, *evaluation])
    require(count <= CONFIG[f"{stage}_continuation_cap"], "Continuation cap exceeded")
    old.save_new(OUTPUT / stage / "complete.json", {"manifest": binding, "time": old.now(), "roots": len(samples),
                 "continuations": count, "elapsed_seconds": time.perf_counter() - started,
                 "choices": artifact(choices_path), "full_source_and_continuation_replay": True})


def operational_decision(binding):
    complete = old.read_json(OUTPUT / "probe/complete.json")
    require(complete["manifest"] == binding and complete["full_source_and_continuation_replay"], "Probe incomplete")
    total_cpu = total_wall = bytes_per_continuation = 0
    count = complete["continuations"]
    for name in ("selection", "evaluation", "selection_audit", "evaluation_audit"):
        lock = verify_stage(f"probe_{name}", binding)
        for item in lock["files"]:
            record = verified(item)
            total_cpu += record["timing"]["cpu"]
            total_wall += record["timing"]["wall"]
            bytes_per_continuation += (ROOT / item["path"]).stat().st_size / count
    source_seconds = sum(verify_stage(f"probe_{name}", binding)["elapsed_seconds"] for name in ("sources", "sources_audit"))
    # Use conservative CPU/task-wall maximum, maximum continuation count, and25%overhead.
    forecast = 1.25 * (max(total_cpu, total_wall) / count * CONFIG["main_continuation_cap"] / 8
                       + source_seconds * CONFIG["source_limit"]["main"] / CONFIG["source_limit"]["probe"])
    projected_bytes = int(bytes_per_continuation * CONFIG["main_continuation_cap"] * 2 + 512 * 1024**2)
    free = shutil.disk_usage(ROOT).free
    reasonable = forecast <= CONFIG["maximum_forecast_hours"] * 3600 and free - projected_bytes >= CONFIG["disk_reserve_bytes"]
    value = {"manifest": binding, "time": old.now(), "outcome_blind": True,
             "forecast_seconds": forecast, "forecast_hours": forecast / 3600, "reasonable": reasonable,
             "free_bytes": free, "projected_main_bytes": projected_bytes,
             "probe": artifact(OUTPUT / "probe/complete.json"),
             "caveat": "Small-probe cost estimate, not a runtime guarantee; no strength used."}
    old.save_new(OUTPUT / "operational_decision.json", value)
    require(reasonable, "Probe forecast exceeds8hours or disk reserve; main held for owner review")
    event("probe_passed_main_authorized", forecast_hours=forecast / 3600)


def finish(binding):
    for stage in ("probe", "main"):
        for name in ("sources", "sources_audit", "proposals", "proposals_audit", "selection", "selection_audit", "evaluation", "evaluation_audit"):
            verify_stage(f"{stage}_{name}", binding)
    samples = old.read_json(OUTPUT / "main/panel.json")["samples"]
    offers = old.read_json(OUTPUT / "main_proposals_lock.json")["files"]
    choices = old.read_json(OUTPUT / "main/choices.json")["rows"]
    evaluations = old.read_json(OUTPUT / "main_evaluation_lock.json")["files"]
    rows = [per_root(s, verified(p), c, verified(e)) for s, p, c, e in zip(samples, offers, choices, evaluations)]
    result = {"manifest": binding, "time": old.now(), "states": len(rows), **statistics(rows),
              "scope": "Fresh no-private-reserve035 positions; one-move035 continuations, not whole-game strength.",
              "all_prescribed_audits_pass": True, "training": False, "full_inner_replay": True,
              "descriptive": descriptive(rows)}
    old.save_new(OUTPUT / "state_results.json", rows)
    old.save_new(OUTPUT / "result.json", result)
    auditor().final_audit(binding)
    audit = old.read_json(OUTPUT / "final_audit.json")
    require(audit["passed"] and audit["result_sha256"] == old.file_hash(OUTPUT / "result.json"), "Final independent audit failed")
    text = ("# Learned fourth-proposal screen\n\n"
            f"Learned minus original random-slot reference: {result['primary']['delta_pp']:+.4f} percentage points; "
            f"paired95% {result['primary']['ci95_pp']}.\n\n"
            f"Learned minus035-ranked control: {result['attribution']['delta_pp']:+.4f} percentage points; "
            f"paired95% {result['attribution']['ci95_pp']}.\n\n"
            f"Classification: {result['classification']}.\n\n"
            "All288 fresh positions, all three frozen161 fits and both controls retained. "
            "Equal32-world selection per offered action;128 independent evaluation worlds. "
            "Every source game and collected continuation fully replayed.\n\n"
            "Conditional evidence on greedy035 positions without hidden reserves. "
            "No whole-game, equal-wall-time, superhuman or repeated-learning-loop claim. "
            "No automatic extension, tuning, promotion or successor.\n")
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    require(validate_freeze() == binding, "Final dependency mutation")
    old.save_new(OUTPUT / "final_status.json", {"complete": True, "manifest": binding, "time": old.now(),
                 "all_prescribed_audits_pass": True, "classification": result["classification"],
                 "result": artifact(OUTPUT / "result.json"), "report": artifact(OUTPUT / "report.md"),
                 "audit": artifact(OUTPUT / "final_audit.json"), "automatic_successor": False})
    event("complete", classification=result["classification"])


def descriptive(rows):
    def summary(part):
        return {"states": len(part), "primary_pp": 100 * float(np.mean([r["primary"] for r in part])),
                "attribution_pp": 100 * float(np.mean([r["attribution"] for r in part])),
                "arm_scores": {arm: float(np.mean([r["scores"][arm] for r in part])) for arm in ARMS}}
    return {"overall": summary(rows), "cells": {str(c): summary([r for r in rows if r["cell"] == c]) for c in range(6)},
            "choice_change": {arm: float(np.mean([r["choices"][arm] != r["choices"]["reference"] for r in rows])) for arm in ARMS},
            "proposal_change": {arm: float(np.mean([r["arm_indices"][arm] != r["arm_indices"]["reference"] for r in rows])) for arm in ARMS},
            "mean_fourth_raw_rank": {arm: float(np.mean([r["fourth_raw_ranks"][arm] for r in rows])) for arm in ARMS},
            "mean_learned_extra_seconds": {str(seed): float(np.mean([r["proposal_timing"]["learned_extra_wall"][str(seed)] for r in rows])) for seed in MODELS}}


def run():
    log = OUTPUT / "runtime"
    log.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot pipeline; no automatic retry")
        binding = validate_freeze()
        old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                     "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
        try:
            initialize_models()
            build_stage("probe", binding)
            operational_decision(binding)
            build_stage("main", binding)
            finish(binding)
        except BaseException:
            old.save_new(OUTPUT / "failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed_or_held", failure="failure.json")
            raise


def launch():
    binding = validate_freeze()
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder / "launcher.lock"):
        require(not (folder / "process.json").exists() and not (OUTPUT / "launch.json").exists(), "Already launched")
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_learned_proposal_screen.py", "run"],
                                          folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic() + 60
        while not (OUTPUT / "launch.json").exists():
            require(time.monotonic() < deadline, "No launch acknowledgement; do not relaunch")
            time.sleep(.05)
        acknowledgement = old.read_json(OUTPUT / "launch.json")
        require(acknowledgement["pid"] == process["pid"] and acknowledgement["manifest"] == binding, "Wrong supervisor acknowledgement")
        old.save_new(folder / "acknowledged.json", {"time": old.now(), "pid": process["pid"],
                                                   "launch": artifact(OUTPUT / "launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        require(args.checks is not None, "Provide a new check directory")
        return check(args.checks)
    if args.command == "prepare":
        require(args.checks is not None, "Provide passing checks.json")
        prepare(args.checks)
    elif args.command == "launch":
        launch()
    else:
        run()
    return 0
