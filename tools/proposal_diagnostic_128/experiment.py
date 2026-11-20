"""One bounded diagnostic, detached from the chat, with no training or successor."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .common import (ROOT, OUTPUT, SOURCE, SOURCE_BINDING, FITS, CONFIG, old,
                     windows, require, artifact, verified, verify_hash,
                     budget_boundary, BudgetExceeded)
from . import data, engine, verify
from tools.proposal_iteration_128 import experiment as source

CHECK_TARGETS = [
    "tools/proposal_diagnostic_128/test_data.py",
    "tools/proposal_diagnostic_128/test_engine.py",
    "tools/proposal_diagnostic_128/test_verify.py",
    "tools/proposal_diagnostic_128/test_experiment.py",
    "tools/proposal_diagnostic_128/test_integration.py",
    "tests/test_windowless_workers_160.py",
]
EXTRAS = ["run_proposal_diagnostic_128.py", "docs/engineering/proposal_diagnostic_128.md",
          "docs/strategy/next_after_proposal_iteration_128.md",
          "docs/engineering/verification_policy.md"]
CONTRASTS = ("treatment_minus_anchor", "treatment_minus_replay", "random_minus_anchor",
             "first_three_minus_anchor", "union_minus_anchor")


def identity():
    # Source imports span mature simulation and model primitives. Retain their
    # exact frozen inventory without adding files to any ancestor inventory.
    parent = old.read_json(SOURCE / "manifest.json")["identity"]
    paths = {ROOT / p for p in parent["dependencies"]}
    paths.update((ROOT / "tools/proposal_diagnostic_128").glob("*.py"))
    paths.update(ROOT / p for p in EXTRAS + CHECK_TARGETS)
    return {"config": CONFIG, "fits": list(FITS), "checks": CHECK_TARGETS,
            "dependencies": {p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in sorted(paths)},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    require(source.validate_freeze() == SOURCE_BINDING, "Parent frozen identity changed")
    source.validate_learning(SOURCE_BINDING)
    final = old.read_json(SOURCE / "final_status.json")
    require(final["manifest"] == SOURCE_BINDING and final["complete"]
            and final["all_prescribed_audits_pass"] and not final["full_inner_replay"],
            "Completed parent audit required")
    result, audit = verified(final["result"]), verified(final["audit"])
    require(audit["passed"] and audit["result"] == final["result"]
            and not result["conditional_stage_b_run"], "Parent closure mismatch")
    files = [artifact(SOURCE / n) for n in (
        "manifest.json", "freeze.json", "final_status.json", "report.md", "result.json",
        "final_audit.json", "stage_a_result.json", "stage_a_final_audit.json",
        "stage_a_complete.json", "probe_a_complete.json", "models_lock.json",
        "learning_complete.json", "schedule.json")]
    stage_a = verified(result["stage_results"]["stage_a"])
    require(stage_a["all_prescribed_audits_pass"] and stage_a["games"] == 402,
            "Wrong completed source arena")
    for item in (final["report"], *audit["stages"].values(), result["learning"]):
        verify_hash(item)
    files.extend(old.read_json(SOURCE / "models_lock.json")["models"].values())
    for item in files:
        verify_hash(item)
    return {"binding": SOURCE_BINDING, "files": files,
            "scope": "Completed sampled parent audit and sealed nine-model lineage"}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    args = ["-q", *CHECK_TARGETS]
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main(" + repr(args) + "))"
    started = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
        value = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT,
                               stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Code changed during preflight")
    old.save_new(folder / "checks.json", {"identity": before, "targets": CHECK_TARGETS,
                 "exit_code": value.returncode, "seconds": time.perf_counter() - started,
                 "logs": [artifact(folder / n) for n in ("stdout.txt", "stderr.txt")]})
    return value.returncode


def prepare(checks_path):
    require(not (OUTPUT / "manifest.json").exists(), "Already frozen; preserve study")
    checks = old.read_json(checks_path)
    require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS
            and checks["identity"] == identity(), "Current passing preflight required")
    for item in checks["logs"]:
        verify_hash(item)
    parent = parent_evidence()
    selection = data.prepare_selection()
    data.validate_selection(selection)
    require(len(selection["main"]) == 201 and len(selection["probe"]) == 12,
            "Wrong fixed panel sizes")
    seeds = [seed for stage in ("probe", "main") for spec in selection[stage]
             for block in ("selection", "assessment") for seed in engine.world_seeds(stage, spec["index"], block)]
    require(len(seeds) == len(set(seeds)) == 213 * 256, "New world namespaces overlap")
    source_seeds = {seed for stage in ("probe", "main") for spec in selection[stage]
                    for seed in verified(spec["source_root"])["schedule"]["world_seeds"]}
    require(set(seeds).isdisjoint(source_seeds), "New/source world namespaces overlap")
    old.save_new(OUTPUT / "selection.json", selection)
    designs = {stage: verify.sample_plan(selection[stage], stage, None) for stage in ("probe", "main")}
    manifest = {"identity": identity(), "parent": parent, "selection": artifact(OUTPUT / "selection.json"),
                "audit_designs": designs, "checks": artifact(checks_path), "owner_authorized": True,
                "authorization": "Owner accepted the complete bounded diagnostic on2026-09-27; no training or successor."}
    old.save_new(OUTPUT / "manifest.json", manifest)
    binding = old.digest(manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": artifact(OUTPUT / "manifest.json"),
                 "binding": binding, "time": old.now(), "owner_authorized": True})
    for stage in ("probe", "main"):
        plan = {**designs[stage], "manifest": binding}
        path = OUTPUT / stage / "audit_sample.json"
        old.save_new(path, plan)
        old.save_new(OUTPUT / stage / "audit_sample_seal.json", {
            "manifest": binding, "stage": stage, "sample": artifact(path),
            "selection": artifact(OUTPUT / "selection.json")})
    require(validate_freeze() == binding, "Post-freeze check failed")
    print({"prepared": True, "binding": binding, "main_states": 201}, flush=True)


def validate_freeze():
    freeze = old.read_json(OUTPUT / "freeze.json")
    manifest = verified(freeze["manifest"])
    require(freeze["owner_authorized"] and old.digest(manifest) == freeze["binding"], "Freeze changed")
    current = identity()
    require(current == manifest["identity"], "Frozen code/runtime changed")
    checks = verified(manifest["checks"])
    require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS
            and checks["identity"] == current, "Frozen preflight changed")
    selection = verified(manifest["selection"])
    require(selection["source_binding"] == SOURCE_BINDING, "Source binding changed")
    for item in [*checks["logs"], *manifest["parent"]["files"], *selection["source_files"]]:
        verify_hash(item)
    for stage in ("probe", "main"):
        seal = old.read_json(OUTPUT / stage / "audit_sample_seal.json")
        require(seal["manifest"] == freeze["binding"] and seal["selection"] == manifest["selection"],
                "Audit selection seal changed")
        plan = verified(seal["sample"])
        require(plan == {**manifest["audit_designs"][stage], "manifest": freeze["binding"]},
                "Frozen audit sampling changed")
    return freeze["binding"]


def event(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    old.atomic_write(OUTPUT / "progress.json", row)
    print(row, flush=True)


def worker_initialize(stage, kind, binding, base):
    folder = Path(base) / "workers" / f"{stage}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt",
                                              folder / f"{os.getpid()}_stderr.txt")
    engine.initialize_models()
    require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(),
            "Worker arithmetic contract")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage,
                 "kind": kind, **health})


def collect_job(stage, spec, binding):
    return engine.collect(stage, spec, binding, OUTPUT)


def audit_job(stage, arg, binding):
    spec, item = arg
    return verify.audit(stage, spec, item, binding, OUTPUT)


def run_pool(stage, kind, arguments, binding, upstream):
    require(validate_freeze() == binding, "Frozen identity changed before stage")
    budget_boundary(OUTPUT)
    start = time.perf_counter()
    launch_path = OUTPUT / f"{stage}_{kind}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "stage": stage, "kind": kind, "time": old.now(),
                 "dependencies": [artifact(p) for p in upstream]})
    runtime = windows.configure_spawn()
    event(f"{stage}_{kind}", completed=0, total=len(arguments))
    task = collect_job if kind == "collection" else audit_job
    files = []
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, kind, binding, OUTPUT)) as pool:
        jobs = [pool.submit(task, stage, arg, binding) for arg in arguments]
        try:
            for job in as_completed(jobs):
                item = job.result()
                files.append(item)
                event(f"{stage}_{kind}", completed=len(files), total=len(arguments))
                budget_boundary(OUTPUT)
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    files.sort(key=lambda item: item["index"])
    require([item["index"] for item in files] == list(range(len(arguments))), "Missing or duplicate state")
    for item in files:
        verify_hash(item)
    old.save_new(OUTPUT / f"{stage}_{kind}_lock.json", {"manifest": binding, "launch": artifact(launch_path),
                 "files": files, "elapsed_seconds": time.perf_counter() - start, "runtime": runtime})
    return files


def stage_lock(stage, kind, binding):
    lock = old.read_json(OUTPUT / f"{stage}_{kind}_lock.json")
    require(lock["manifest"] == binding and [x["index"] for x in lock["files"]]
            == list(range(CONFIG["states"][stage])), "Incomplete stage lock")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["stage"] == stage and launch["kind"] == kind,
            "Stage launch changed")
    for item in [*lock["files"], *launch["dependencies"]]:
        verify_hash(item)
    return lock


def build_stage(stage, binding):
    selection = old.read_json(OUTPUT / "selection.json")
    upstream = [OUTPUT / "freeze.json", OUTPUT / "selection.json", SOURCE / "models_lock.json",
                OUTPUT / stage / "audit_sample.json", OUTPUT / stage / "audit_sample_seal.json"]
    if stage == "main":
        gate = old.read_json(OUTPUT / "operational_decision.json")
        require(gate["manifest"] == binding and gate["reasonable"], "Main requires passing runtime gate")
        upstream += [OUTPUT / "operational_decision.json", OUTPUT / "probe_complete.json"]
    roots = run_pool(stage, "collection", selection[stage], binding, upstream)
    audits = run_pool(stage, "audits", list(zip(selection[stage], roots)), binding,
                      [OUTPUT / f"{stage}_collection_lock.json", *upstream])
    plan_path = OUTPUT / stage / "audit_sample.json"
    plan = old.read_json(plan_path)
    require(plan == verify.sample_plan(selection[stage], stage, binding), "Audit plan reconstruction disagrees")
    coverage = {"states": len(roots), "recorded_worlds": 0, "replayed_worlds": 0,
                "recorded_continuations": 0, "replayed_continuations": 0, "full_stress_roots": 0}
    for spec, root_item, audit_item, sample in zip(selection[stage], roots, audits, plan["selections"], strict=True):
        budget_boundary(OUTPUT)
        receipt = verified(audit_item)
        require(receipt["manifest"] == binding and receipt["passed"] and receipt["spec"] == spec
                and receipt["root"] == {k: root_item[k] for k in ("path", "sha256")}, "Audit linkage mismatch")
        require(receipt["sample_artifact"] == artifact(plan_path)
                and receipt["stress"] == sample["stress"]
                and receipt["ordinary_world_indices"] == sample["ordinary"]
                and receipt["replay_world_indices"] == sample["replay"], "Audit sample coverage mismatch")
        replay_worlds = sum(len(x) for x in sample["replay"].values())
        require(receipt["recorded_worlds"] == 256 and receipt["replayed_worlds"] == replay_worlds
                and receipt["recorded_continuations"] == root_item["continuations"] == 256*receipt["support"]
                and receipt["replayed_continuations"] == replay_worlds*receipt["support"], "Audit counts disagree")
        verify_hash(receipt["selection_artifact"])
        for key in ("recorded_worlds", "replayed_worlds", "recorded_continuations", "replayed_continuations"):
            coverage[key] += receipt[key]
        coverage["full_stress_roots"] += int(receipt["stress"])
    old.save_new(OUTPUT / f"{stage}_complete.json", {"manifest": binding, "time": old.now(),
                 "states": len(roots), "collection": artifact(OUTPUT / f"{stage}_collection_lock.json"),
                 "audits": artifact(OUTPUT / f"{stage}_audits_lock.json"),
                 "coverage": coverage,
                 "all_prescribed_audits_pass": True, "full_inner_replay": stage == "probe"})


def cost_projection(collection_wall, audit_wall, replay_wall, probe_continuations,
                    main_replay_ceiling, size_bytes, free_bytes, elapsed_seconds):
    values = (collection_wall, audit_wall, replay_wall, probe_continuations,
              main_replay_ceiling, size_bytes, free_bytes, elapsed_seconds)
    require(all(type(x) in (int, float) and np.isfinite(x) and x >= 0 for x in values),
            "Invalid cost evidence")
    require(collection_wall > 0 and audit_wall >= replay_wall > 0 and probe_continuations > 0,
            "Complete probe timing required")
    main_collection = collection_wall / probe_continuations * CONFIG["maximum_main_continuations"] / CONFIG["workers"]
    main_audit = ((audit_wall - replay_wall) * 201 / 12
                  + replay_wall / probe_continuations * main_replay_ceiling) / CONFIG["workers"]
    remaining = CONFIG["forecast_allowance"] * (main_collection + main_audit) + CONFIG["finalization_allowance_seconds"]
    projected_bytes = int(size_bytes / probe_continuations * CONFIG["maximum_main_continuations"] * 1.5 + 512*1024**2)
    return {"remaining_seconds": remaining, "remaining_hours": remaining / 3600,
            "collection_seconds_before_allowance": main_collection, "audit_seconds_before_allowance": main_audit,
            "elapsed_seconds": elapsed_seconds, "projected_bytes": projected_bytes, "free_bytes": free_bytes,
            "reasonable": elapsed_seconds + remaining <= CONFIG["maximum_runtime_hours"]*3600
                          and free_bytes - projected_bytes >= CONFIG["disk_reserve_bytes"]}


def operational_decision(binding):
    roots = [verified(x) for x in stage_lock("probe", "collection", binding)["files"]]
    audits = [verified(x) for x in stage_lock("probe", "audits", binding)["files"]]
    plan = old.read_json(OUTPUT / "main/audit_sample.json")
    replay_ceiling = sum(sum(len(v) for v in row["replay"].values()) * 7 for row in plan["selections"])
    value = cost_projection(
        sum(r["timing"]["wall"] for r in roots), sum(r["timing"]["wall"] for r in audits),
        sum(r["timing"]["replay_wall"] for r in audits),
        sum(r["recorded_continuations"] for r in audits), replay_ceiling,
        sum(p.stat().st_size for p in (OUTPUT / "probe").rglob("*") if p.is_file()),
        shutil.disk_usage(ROOT).free, time.time() - old.read_json(OUTPUT / "launch.json")["epoch_seconds"])
    old.save_new(OUTPUT / "operational_decision.json", {"manifest": binding, "time": old.now(), **value})
    if not value["reasonable"]:
        old.save_new(OUTPUT / "operational_hold.json", {"manifest": binding, "time": old.now(),
                     "reason": "Outcome-blind probe forecast exceeds time or disk budget", "forecast": value})
        event("operational_hold", remaining_forecast_hours=value["remaining_hours"])
        return False
    event("probe_passed", remaining_forecast_hours=value["remaining_hours"])
    return True


def independent_statistics(rows):
    """Direct indexed-sum reference, separate from verifier's accumulation."""
    require(len(rows) == 201 and {r["pair"] for r in rows} == set(range(201)), "Complete paired panel required")
    strata = []
    for fit in FITS:
        subset = sorted((r for r in rows if r["fit"] == fit), key=lambda r: r["index"])
        require(len(subset) == 67, "Fixed fit weights changed")
        values = []
        for r in subset:
            s = {a: int(round(v * 256)) for a, v in r["arm_scores"].items()}
            require(all(abs(s[a]/256-r["arm_scores"][a]) < 1e-12 for a in s), "Score grid changed")
            values.append([s["treatment"]-s["anchor"], s["treatment"]-s["replay"],
                           s["random"]-s["anchor"], s["first_three"]-s["anchor"], s["union"]-s["anchor"]])
        strata.append(np.asarray(values, dtype=np.int64))
    point = sum(x.sum(axis=0) for x in strata) / (201*256)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    samples = []
    for _ in range(50):
        totals = np.zeros((1000, len(CONTRASTS)), dtype=np.int64)
        for values in strata:
            indices = rng.integers(0, 67, size=(1000, 67))
            totals += values[indices].sum(axis=1)
        samples.append(totals/(201*256))
    intervals = np.quantile(np.concatenate(samples), [.025, .975], axis=0, method="linear")
    return {name: {"mean": float(point[i]), "ci95": intervals[:, i].tolist()} for i, name in enumerate(CONTRASTS)}


def finish(binding):
    budget_boundary(OUTPUT)
    require(validate_freeze() == binding, "Changed frozen inputs at finalization")
    selection = old.read_json(OUTPUT / "selection.json")
    for stage in ("probe", "main"):
        collected = stage_lock(stage, "collection", binding)
        checked = stage_lock(stage, "audits", binding)
        for root_item, audit_item in zip(collected["files"], checked["files"], strict=True):
            budget_boundary(OUTPUT)
            record, receipt = verified(root_item), verified(audit_item)
            require(receipt["passed"] and receipt["manifest"] == binding
                    and receipt["root"] == {k: root_item[k] for k in ("path", "sha256")},
                    "Final root/audit linkage changed")
            verify_hash(record["selection_artifact"])
    items = stage_lock("main", "audits", binding)["files"]
    summary = verify.summarize(items, selection["main"], binding, OUTPUT)
    budget_boundary(OUTPUT)
    # Raw-record reconstruction and agreement with the verifier are completed
    # below through the engine's immutable assessment summaries.
    raw_rows = []
    for item in stage_lock("main", "collection", binding)["files"]:
        budget_boundary(OUTPUT)
        record = verified(item)
        raw_rows.append(engine.statistics_row(record))
    independent = independent_statistics(raw_rows)
    budget_boundary(OUTPUT)
    for name, row in independent.items():
        checked = summary["statistics"]["contrasts"][name]
        require(abs(row["mean"] - checked["mean_score_difference"]) < 1e-12
                and np.allclose(row["ci95"], checked["ci95"], rtol=0., atol=1e-12),
                "Independent paired statistics disagree")
    primary = independent["treatment_minus_anchor"]
    label = ("conditional_direction_positive" if primary["ci95"][0] > 0 else
             "conditional_direction_negative" if primary["ci95"][1] < 0 else "conditional_direction_inconclusive")
    require(label == summary["statistics"]["classification"], "Final direction classification disagrees")
    result = {"manifest": binding, "time": old.now(), "classification": label,
              **summary, "all_prescribed_audits_pass": True, "full_inner_replay": False,
              "cache": artifact(OUTPUT / "cached_analysis.json"),
              "scope": "Balanced saved-state diagnostic with independent assessment, not full-game strength",
              "automatic_successor": False}
    old.save_new(OUTPUT / "result.json", result)
    old.save_new(OUTPUT / "final_audit.json", {"manifest": binding, "passed": True, "time": old.now(),
                 "result": artifact(OUTPUT / "result.json"), "independent_statistics": independent,
                 "stages": {s: artifact(OUTPUT / f"{s}_complete.json") for s in ("probe", "main")},
                 "full_inner_replay": False})
    lines = ["# Same-state proposal diagnosis", "", f"Classification: {label}.", "",
             "201 positions from201 source pairs; all three frozen fits retained. Gains are terminal score points (win1/draw.5/loss0), not full-game win rates.", "",
             "| Contrast | Gain (percentage points) | Paired95% interval (percentage points) |",
             "|---|---:|---:|"]
    for name, row in independent.items():
        lines.append(f"| {name} | {100*row['mean']:.4f} | [{100*row['ci95'][0]:.4f}, {100*row['ci95'][1]:.4f}] |")
    lines += ["", "Only treatment-minus-anchor is primary; other contrasts and strata are diagnostic. Choices used128 fresh selection worlds and were locked before128 disjoint assessment worlds. The union has more candidates and is not an equal-budget strength claim.", "",
              "All source states, proposals, budgets, world construction and outcome arithmetic checked; prescribed sampled continuation replay plus full stress cases passed. Shared rules/encoding/neural primitives remain trusted. See result.json for coverage and saved-prefix context; saved paired worlds are not independent assessment.", "",
              "No training, automatic extension, promotion, successor or monitoring.", ""]
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    budget_boundary(OUTPUT)
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "classification": label, "all_prescribed_audits_pass": True, "full_inner_replay": False,
                 "result": artifact(OUTPUT / "result.json"), "audit": artifact(OUTPUT / "final_audit.json"),
                 "report": artifact(OUTPUT / "report.md"), "automatic_successor": False})
    event("complete", classification=label)


def run():
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(folder / "stdout.txt", folder / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot run; preserve existing evidence")
        binding = validate_freeze()
        try:
            with source.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(),
                             "epoch_seconds": time.time(), "pid": os.getpid(), "sleep_inhibition": sleep,
                             "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
                event("cached_matched_position_analysis")
                data.cached_analysis(old.read_json(OUTPUT / "selection.json"), binding, OUTPUT)
                budget_boundary(OUTPUT)
                build_stage("probe", binding)
                if not operational_decision(binding):
                    return
                build_stage("main", binding)
                budget_boundary(OUTPUT)
                finish(binding)
        except BudgetExceeded as exc:
            old.save_new(OUTPUT / "operational_hold.json", {"manifest": binding, "time": old.now(), "reason": str(exc)})
            event("operational_hold", reason=str(exc))
        except BaseException:
            old.save_new(OUTPUT / "failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure="failure.json")
            raise


def launch():
    binding = validate_freeze()
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder / "launcher.lock"):
        require(not (folder / "process.json").exists() and not (OUTPUT / "launch.json").exists(), "Already launched")
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_proposal_diagnostic_128.py", "run"],
                                          folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic() + 60
        while not (OUTPUT / "launch.json").exists():
            require(time.monotonic() < deadline, "No acknowledgement; inspect before any retry")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT / "launch.json")
        require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding, "Supervisor identity mismatch")
        old.save_new(folder / "acknowledged.json", {"time": old.now(), "pid": process["pid"], "launch": artifact(OUTPUT / "launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        require(args.checks is not None, "Supply a new check folder")
        return check(args.checks)
    if args.command == "prepare":
        require(args.checks is not None, "Supply current checks.json")
        prepare(args.checks)
    elif args.command == "launch":
        launch()
    else:
        run()
    return 0
