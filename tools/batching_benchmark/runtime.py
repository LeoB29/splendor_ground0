"""Deferred single-worker benchmark. Importing this module does not load models."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
import traceback

from tools.batching_benchmark import benchmark as b


def compare_traces(reference, observed):
    b.require(len(reference) == len(observed), "Missing/extra continuation jobs")
    for job, (expected, actual) in enumerate(zip(reference, observed)):
        if expected != actual:
            left, right = expected.get("trace", []), actual.get("trace", [])
            step = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                        min(len(left), len(right)))
            raise ValueError(f"Continuation mismatch at job {job}, first differing/ending step {step}")


def timing_summary(rows, schedule):
    expected = {(row["root"], row["repetition"], mode) for row in schedule for mode in row["order"]}
    keys = [(row["root"], row["repetition"], row["mode"]) for row in rows]
    b.require(len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete/duplicate timing blocks")
    for row in rows:
        for field in ("collection_wall", "audit_wall", "collection_cpu", "audit_cpu"):
            b.require(math.isfinite(row[field]) and row[field] > 0, "Invalid timing")
    totals = {mode: {field: sum(row[field] for row in rows if row["mode"] == mode)
                     for field in ("collection_wall", "audit_wall", "collection_cpu", "audit_cpu")}
              for mode in ("serial", "batch16")}
    for value in totals.values():
        value["root_collection_plus_audit_wall"] = value["collection_wall"] + value["audit_wall"]
    per_root = []
    by_key = {key: row for key, row in zip(keys, rows)}
    for key in sorted({row["root"] for row in schedule}):
        ratios = []
        for repetition in range(b.CONFIG["timing_repetitions"]):
            serial, batch = [by_key[key, repetition, mode] for mode in ("serial", "batch16")]
            ratios.append((serial["collection_wall"] + serial["audit_wall"])
                          / (batch["collection_wall"] + batch["audit_wall"]))
        per_root.append({"root": key, "paired_ratios": ratios, "median_ratio": statistics.median(ratios)})
    return {"totals": totals,
            "collection_ratio": totals["serial"]["collection_wall"] / totals["batch16"]["collection_wall"],
            "root_collection_plus_audit_ratio": totals["serial"]["root_collection_plus_audit_wall"]
                    / totals["batch16"]["root_collection_plus_audit_wall"],
            "per_root": per_root, "universal_equivalence_proved": False,
            "full_pipeline_speedup_measured": False, "production_enabled": False}


def collect_custom(p, batch_module, entry, target, mode, include_trace):
    """Same proposal/world/record contract, with isolated continuation scheduling."""
    old, baseline = p.old, p.baseline
    saved = b.verified(entry["root"])
    state = old.deserialize_state_snapshot(saved["snapshot"])
    state_before = old.digest(old.serialize_state_snapshot(state))
    spec = entry["spec"]
    fit = spec["fit"] if state.current_player == spec["candidate_seat"] else None
    started, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": b.SOURCE_BINDING, "time": old.now(), "snapshot": old.serialize_state_snapshot(state),
              "privacy": baseline.privacy(state), "worlds": []}
    traces, metrics = [], {}
    try:
        if fit is None:
            schedule = baseline.root_schedule(state, spec["search_stream"])
            candidates = baseline.candidate_set(state, schedule["candidate_nonce"])
        else:
            proposal = p.learned_proposal(state, spec["search_stream"], fit)
            record["proposal"] = proposal
            schedule, candidates = proposal["schedule"], proposal["candidates"]
        record.update(schedule=schedule, candidates=candidates)
        legal = old.SplendorEnv().legal_actions(state)
        record["scores"] = list(map(float, old._RAW.score_actions(state, legal)))
        actions = [old.decode_action(value) for value in candidates["actions"]]
        work = []
        for seed in schedule["world_seeds"]:
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            legal_world = old.SplendorEnv().legal_actions(world)
            b.require(all(action in legal_world for action in actions), "Illegal forced root")
            record["worlds"].append({"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []})
            work.extend((world, action) for action in actions)
        if mode == "serial":
            values = [old.trajectory(copy.deepcopy(world), action, include_trace=include_trace) for world, action in work]
        else:
            size = int(mode.removeprefix("batch"))
            class MeasuredPolicy(batch_module.Batched035):
                def __init__(self):
                    super().__init__(old._RAW)
                    self.widths = []

                def choose_many(self, rows):
                    self.widths.append(len(rows))
                    return super().choose_many(rows)
            policy = MeasuredPolicy()
            values = batch_module.trajectories(work, policy, size, include_trace=include_trace)
            metrics = {**policy.metrics, "batch_widths": policy.widths,
                       "requested_batch_size": size, "full_batches": policy.widths.count(size)}
        b.require(len(values) == len(work) == entry["jobs"], "Incomplete continuation workload")
        for job, value in enumerate(values):
            if include_trace:
                traces.append(value)
            outcome = {key: value[key] for key in ("reason", "winner", "turns", "trace_sha256", "final_state_sha256")}
            outcome["wdl"] = old.outcome_value(value["winner"], state.current_player) if value["reason"] in old.VALID_NATURAL_TERMINATIONS else None
            b.require(outcome["wdl"] is not None, "Invalid ending; preserve instead of excluding")
            record["worlds"][job // len(actions)]["outcomes"].append(outcome)
        record.update(baseline.select(record["worlds"], len(actions)))
        b.require(state_before == old.digest(old.serialize_state_snapshot(state)), "Source state mutated")
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - started, "cpu": time.process_time() - cpu}
        old.save_new(target, record)
    return record, traces, metrics


def run_root(p, batch_module, verifier, entry, folder, mode, trace=False):
    """Time collection including input/record I/O, then unchanged full root audit."""
    old = p.old
    folder.mkdir(parents=True, exist_ok=False)
    old.save_new(folder / "main_launch.json", {"manifest": b.SOURCE_BINDING, "time": old.now()})
    target = folder / "main" / "roots" / f"{entry['id']}.json"
    started, cpu = time.perf_counter(), time.process_time()
    source = b.verified(entry["root"])
    state = old.deserialize_state_snapshot(source["snapshot"])
    spec = entry["spec"]
    if mode == "serial" and not trace:
        fit = spec["fit"] if state.current_player == spec["candidate_seat"] else None
        _, record = p.choose_root(state, spec["search_stream"], target, b.SOURCE_BINDING, fit)
        traces, metrics = [], {}
    else:
        record, traces, metrics = collect_custom(p, batch_module, entry, target, mode, trace)
    root_sha = old.file_hash(target)
    collection_wall, collection_cpu = time.perf_counter()-started, time.process_time()-cpu
    if trace:
        b.save_new(folder / "traces.json", traces)
    b.save_new(folder / "batch_metrics.json", metrics)
    # Compare saved authoritative scalar outcomes without relying on the new
    # collector's selector. Full action/state traces have a separate exact check.
    for key in ("snapshot", "schedule", "privacy", "candidates", "scores", "worlds", "means", "choice"):
        b.require(record[key] == source[key], f"Changed {key} at {entry['id']} ({mode})")
    audit_start, audit_cpu = time.perf_counter(), time.process_time()
    move = {"root_sha256": root_sha, "action": record["candidates"]["actions"][record["choice"]]}
    receipt = verifier.audit_root("main", spec["index"], state, move, spec, b.SOURCE_BINDING, folder)
    old.save_new(folder / "audit.json", receipt)
    elapsed_audit, elapsed_cpu = time.perf_counter()-audit_start, time.process_time()-audit_cpu
    result = {"root": entry["id"], "mode": mode, "collection_wall": collection_wall,
              "collection_cpu": collection_cpu, "audit_wall": elapsed_audit, "audit_cpu": elapsed_cpu,
              "root_artifact": b.artifact(target), "audit_artifact": b.artifact(folder / "audit.json"),
              "metrics": b.artifact(folder / "batch_metrics.json")}
    if trace:
        result["traces"] = b.artifact(folder / "traces.json")
    return result, traces, metrics


def execute(stage, plan_folder, output, equivalence_folder=None):
    b.require_idle()
    plan, binding = b.load_plan(plan_folder)
    folder = b.path(output)
    b.require(folder.is_relative_to(b.path(b.OUTPUT)) and not folder.exists(), "Choose a new benchmark stage folder")
    equivalence = None
    if stage == "timing":
        seal = b.read(b.path(equivalence_folder) / "complete.json")
        equivalence = b.verified(seal["result"])
        b.require(equivalence.get("passed") is True and equivalence["stage"] == "equivalence"
                  and equivalence["binding"] == binding, "Same-plan completed equivalence required")
        for item in equivalence["artifacts"]:
            b.require(b.file_hash(b.path(item["path"])) == item["sha256"], "Equivalence evidence changed")
    b.require(stage in ("equivalence", "timing"), "Unknown stage")
    os.chdir(b.ROOT)
    sys.path.insert(0, str(b.path("src")))
    from scripts.analysis import learned_proposal_arena as p
    from scripts.analysis import verify_learned_proposal_arena as verifier
    from splendor_ai.alphazero import teacher_batching as batch_module
    p.initialize_models()
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    b.save_new(folder / "launch.json", {"binding": binding, "stage": stage, "time": b.now(), "pid": os.getpid(),
                                       "host": platform.node(), "platform": platform.platform(),
                                       "python": sys.version, "torch": str(p.torch.__version__),
                                       "threads": p.torch.get_num_threads(), "workers": 1, "device": "cpu"})
    artifacts, rows, full_batches = [], [], {str(size): 0 for size in b.CONFIG["batch_sizes"]}
    try:
        def perform(entry, relative, mode, trace=False):
            b.require(time.perf_counter()-started < b.CONFIG["maximum_stage_hours"]*3600,
                      "Six-hour stage ceiling reached; partial evidence retained, no automatic continuation")
            result, traces, metrics = run_root(p, batch_module, verifier, entry, folder / relative, mode, trace)
            for key in ("root_artifact", "audit_artifact", "metrics", "traces"):
                if key in result:
                    artifacts.append(result[key])
            b.save_new(folder / relative / "receipt.json", result)
            artifacts.append(b.artifact(folder / relative / "receipt.json"))
            print({"stage": stage, "root": entry["id"], "mode": mode, "passed": True}, flush=True)
            return result, traces, metrics

        by_id = {entry["id"]: entry for entry in plan["roots"]}
        if stage == "equivalence":
            for index, entry in enumerate(plan["roots"]):
                _, golden, _ = perform(entry, f"root_{index:03d}/serial", "serial", True)
                for size in b.CONFIG["batch_sizes"]:
                    _, observed, metrics = perform(entry, f"root_{index:03d}/batch{size}", f"batch{size}", True)
                    compare_traces(golden, observed)
                    full_batches[str(size)] += metrics["full_batches"]
            b.require(all(full_batches.values()), "Requested batch widths not exercised")
            summary = {"passed": True, "roots": len(plan["roots"]), "full_batches": full_batches,
                       "jobs_per_mode": sum(entry["jobs"] for entry in plan["roots"]),
                       "exact_all_trace_and_final_state_equality": True, "all_full_root_audits_pass": True}
        else:
            warm = by_id[plan["coverage"]["timing_roots"][0]]
            for mode in ("serial", "batch16"):
                perform(warm, f"warmup/{mode}", mode)
            for index, block in enumerate(plan["timing_schedule"]):
                for mode in block["order"]:
                    result, _, _ = perform(by_id[block["root"]], f"block_{index:03d}/{mode}", mode)
                    rows.append({**result, "repetition": block["repetition"]})
            summary = {"passed": True, "rows": rows, **timing_summary(rows, plan["timing_schedule"])}
        b.require(b.active_identity() == plan["active"] and b.code_identity() == plan["code"], "Dependencies changed during run")
        result = {"stage": stage, "binding": binding, "time": b.now(), **summary, "artifacts": artifacts,
                  "seconds": time.perf_counter()-started, "production_enabled": False, "automatic_successor": False}
        b.save_new(folder / "result.json", result)
        b.save_new(folder / "complete.json", {"binding": binding, "result": b.artifact(folder / "result.json"), "complete": True})
    except BaseException:
        b.save_new(folder / "failure.json", {"binding": binding, "stage": stage, "time": b.now(),
                                            "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
