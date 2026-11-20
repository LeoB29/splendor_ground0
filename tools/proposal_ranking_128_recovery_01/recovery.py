"""Resume the interrupted ranking pilot without changing its scientific contract."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.proposal_ranking_128 import common as c, experiment as p
from .reporting import StatusReporter

ROOT, OUTPUT, old = c.ROOT, c.OUTPUT, c.old
RECOVERY = OUTPUT / "recovery_01"
ORIGINAL_BINDING = "e838a6a0415b9e5e4749fe79dc2094cfb38a45665f9957817d37f71225178c82"
ENTRY = ROOT / "run_proposal_ranking_128_recovery_01.py"
PROTOCOL = ROOT / "docs/engineering/proposal_ranking_128_recovery_01.md"
CHECK_TARGETS = [
    "tools/proposal_ranking_128_recovery_01/test_reporting.py",
    "tools/proposal_ranking_128_recovery_01/test_recovery.py",
    "tools/proposal_ranking_128_recovery_01/test_worker.py",
    "tools/proposal_ranking_128/test_experiment.py",
]
SEALED_SOURCE_STAGES = (("source_probe", "collection"),
                        ("source_probe", "audits"), ("source_main", "collection"))
_reporter = None


def event(stage, **values):
    global _reporter
    if _reporter is None:
        _reporter = StatusReporter(RECOVERY, {"manifest": ORIGINAL_BINDING})
    return _reporter.event(stage, **values)


def reconcile_receipt(spec, game_item, audit_item, binding):
    """Validate saved audit linkage and coverage, without repeating its replay."""
    game, receipt = c.verified(game_item), c.verified(audit_item)
    index, stage = spec["index"], spec["stage"]
    c.require(game_item["index"] == audit_item["index"] == index
              and game_item["stage"] == audit_item["stage"] == stage, "Saved receipt index/stage")
    c.require(spec == p.source.independent_spec(stage, index), "Saved independent source specification")
    c.require(game["manifest"] == receipt["manifest"] == binding
              and game["spec"] == receipt["spec"] == spec, "Saved receipt binding/specification")
    c.require("failure" not in game and receipt["passed"] is True
              and receipt["every_move_hidden_input_invariance"] is True, "Saved source audit did not pass")
    c.require(receipt["game"] == {k: game_item[k] for k in ("path", "sha256")}, "Saved game linkage")
    c.require(game["trace"] and game["trace_sha256"] == receipt["trace_sha256"] == old.digest(game["trace"]),
              "Saved trace linkage")
    c.require(game["reason"] == receipt["reason"] and game["reason"] in old.VALID_NATURAL_TERMINATIONS,
              "Saved natural termination")
    coverage = Counter()
    for row in game["trace"]:
        c.require(row["model_id"] == game["model_id"] == f"anchor_{spec['fit']}"
                  and row["model_sha256"] == game["model_sha256"], "Saved source model")
        evidence = row["evidence"]
        coverage.update({"moves": 1, f"seat_{row['actor']}": 1, f"phase_{row['phase']}": 1,
                         f"support_{len(evidence['candidates']['keys'])}": 1,
                         "own_private": int(evidence["privacy"]["own"] > 0),
                         "opponent_private": int(evidence["privacy"]["opponent"] > 0)})
    c.require(receipt["coverage"] == dict(coverage), "Saved source audit coverage")
    return audit_item


def inventory(binding):
    """Reclose the exact failed starting point; no inference or production writes."""
    c.require(binding == ORIGINAL_BINDING and p.validate_freeze() == binding, "Original frozen identity changed")
    c.budget_boundary(OUTPUT)
    for name in ("source_main_audits_lock.json", "selection.json", "selection_seal.json",
                 "operational_decision.json", "operational_hold.json", "result.json", "final_status.json"):
        c.require(not (OUTPUT / name).exists(), f"Unexpected downstream evidence: {name}")
    for stage in ("probe", "main"):
        c.require(not (OUTPUT / stage).exists(), "Evaluation already has partial evidence")
    failure = old.read_json(OUTPUT / "failure.json")
    c.require(failure["manifest"] == binding and "PermissionError: [WinError 5]" in failure["traceback"]
              and "progress.json.tmp" in failure["traceback"], "Different failure requires separate review")
    launch = old.read_json(OUTPUT / "launch.json")
    c.require(launch["manifest"] == binding and launch["owner_authorized"], "Original launch binding")
    complete = old.read_json(OUTPUT / "learning_complete.json")
    c.require(complete["manifest"] == binding, "Learning completion binding")
    for key, name in (("data", "data_lock.json"), ("models", "models_lock.json"),
                      ("fits", "learning_fits_lock.json"), ("fit_audits", "learning_fit_audits_lock.json")):
        c.require(complete[key] == c.artifact(OUTPUT / name), "Learning completion linkage")
    models = p.learning.validate_models(binding, OUTPUT)
    c.require(not models["all_ranking_unchanged"], "Unexpected identity shortcut")
    for kind in ("fits", "fit_audits"):
        p.stage_lock("learning", kind, binding)
    locks = {f"{stage}_{kind}": p.stage_lock(stage, kind, binding) for stage, kind in SEALED_SOURCE_STAGES}
    schedule = old.read_json(OUTPUT / "schedule.json")
    for stage in ("probe", "main"):
        items = locks[f"source_{stage}_collection"]["files"]
        expected_paths = set()
        for spec, item in zip(schedule[stage], items, strict=True):
            c.budget_boundary(OUTPUT)
            path = p.source.game_path(stage, spec["index"], OUTPUT)
            c.require(item == c.artifact(path, index=spec["index"], stage=stage), "Source path/identity changed")
            expected_paths.add(path.resolve())
            game = c.verified(item)
            c.require(game["manifest"] == binding and game["spec"] == spec and "failure" not in game
                      and game["reason"] in old.VALID_NATURAL_TERMINATIONS, "Saved source game incomplete")
            c.require(game["model_sha256"] == models["models"][f"anchor_{spec['fit']}"]["sha256"], "Source model seal")
        c.require({x.resolve() for x in (OUTPUT / "source" / stage / "games").glob("*.json")} == expected_paths,
                  "Unexpected source game files")
    for spec, game, audit in zip(schedule["probe"], locks["source_probe_collection"]["files"],
                                 locks["source_probe_audits"]["files"], strict=True):
        reconcile_receipt(spec, game, audit, binding)
    saved, expected_paths = [], set()
    for spec, game in zip(schedule["main"], locks["source_main_collection"]["files"], strict=True):
        path = p.source.audit_path("main", spec["index"], OUTPUT)
        if path.exists():
            item = c.artifact(path, index=spec["index"], stage="main")
            saved.append(reconcile_receipt(spec, game, item, binding))
            expected_paths.add(path.resolve())
    c.require({x.resolve() for x in (OUTPUT / "source/main/audits").glob("*.json")} == expected_paths,
              "Unexpected main source audit files")
    c.require(len(saved) == 52, "Interrupted audit inventory changed; inspect before freezing")
    names = ["freeze.json", "manifest.json", "launch.json", "failure.json", "progress.json", "schedule.json",
             "learning_complete.json", "data_lock.json", "models_lock.json", "learning_fits_lock.json",
             "learning_fit_audits_lock.json", "source_main_audits_launch.json", "runtime/stdout.txt", "runtime/stderr.txt"]
    names += [f"{stage}_{kind}_lock.json" for stage, kind in SEALED_SOURCE_STAGES]
    interrupted = old.read_json(OUTPUT / "source_main_audits_launch.json")
    c.require(interrupted["manifest"] == binding and interrupted["stage"] == "source_main"
              and interrupted["kind"] == "audits", "Interrupted launch identity")
    for item in interrupted["dependencies"]:
        c.verify_hash(item)
    return {"manifest": binding, "original_launch_epoch": launch["epoch_seconds"],
            "deadline_epoch": launch["epoch_seconds"] + c.CONFIG["maximum_runtime_hours"] * 3600,
            "preserved": [c.artifact(OUTPUT / name) for name in names],
            "sealed_source_stages": {name: lock["files"] for name, lock in locks.items()},
            "retained_main_source_audits": saved,
            "missing_main_source_audit_indices": [s["index"] for s in schedule["main"]
                                                   if s["index"] not in {i["index"] for i in saved}]}


def identity():
    paths = sorted((ROOT / "tools/proposal_ranking_128_recovery_01").glob("*.py"))
    paths += [ENTRY, PROTOCOL]
    return {"dependencies": {p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths},
            "python": sys.version, "checks": CHECK_TARGETS,
            "original_binding": ORIGINAL_BINDING, "amendment": "operational_recovery_only",
            "budget_clock": "original_launch_including_downtime", "automatic_retry": False}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    command = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main(" + repr(["-q", *CHECK_TARGETS]) + "))"
    started = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", "-c", command], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(before == identity(), "Recovery changed during checks")
    old.save_new(folder / "checks.json", {"identity": before, "exit_code": result.returncode,
                 "seconds": time.perf_counter()-started, "logs": [c.artifact(folder / n) for n in ("stdout.txt", "stderr.txt")]})
    print({"checks": str(folder), "exit_code": result.returncode}, flush=True)
    return result.returncode


def prepare(checks_path):
    c.require(not (RECOVERY / "freeze.json").exists(), "Recovery already frozen")
    checks = old.read_json(checks_path)
    c.require(checks["exit_code"] == 0 and checks["identity"] == identity(), "Passing current recovery checks required")
    for item in checks["logs"]:
        c.verify_hash(item)
    with old.process_lock(OUTPUT / "run.lock"):
        reusable = inventory(ORIGINAL_BINDING)
        old.save_new(RECOVERY / "inventory.json", reusable)
        manifest = {"identity": identity(), "original_binding": ORIGINAL_BINDING,
                    "inventory": c.artifact(RECOVERY / "inventory.json"), "checks": c.artifact(checks_path),
                    "owner_authorized": True, "authorization": "Owner explicitly requested resume on 2026-09-27.",
                    "scope": "Reuse completed learning/source work; complete unchanged remaining pipeline."}
        old.save_new(RECOVERY / "manifest.json", manifest)
        old.save_new(RECOVERY / "freeze.json", {"binding": old.digest(manifest),
                     "manifest": c.artifact(RECOVERY / "manifest.json"), "time": old.now()})
    print({"prepared": True, "recovery_binding": validate_recovery(), "retained_audits": 52,
           "remaining_audits": len(reusable["missing_main_source_audit_indices"])}, flush=True)


def validate_recovery(check_original=True):
    freeze = old.read_json(RECOVERY / "freeze.json")
    manifest = c.verified(freeze["manifest"])
    c.require(old.digest(manifest) == freeze["binding"] and manifest["identity"] == identity()
              and manifest["owner_authorized"] and manifest["original_binding"] == ORIGINAL_BINDING,
              "Recovery freeze changed")
    checks = c.verified(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == identity(), "Recovery preflight changed")
    for item in checks["logs"]:
        c.verify_hash(item)
    reuse = c.verified(manifest["inventory"])
    for item in reuse["preserved"]:
        c.verify_hash(item)
    if check_original:
        c.require(p.validate_freeze() == ORIGINAL_BINDING, "Original freeze changed")
    return freeze["binding"]


def worker_initialize(stage, kind, binding, base):
    # Same initialization contract, fresh log namespace to preserve old PID logs.
    validate_recovery(check_original=False)
    folder = RECOVERY / "workers" / f"{stage}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    health = c.windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt", folder / f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    if stage.startswith("source_"):
        p.source.initialize_models(base)
    elif stage in ("probe", "main"):
        p.engine.initialize_models(base)
    c.require(p.torch.get_num_threads() == 1 and p.torch.are_deterministic_algorithms_enabled(), "Worker arithmetic contract")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, "kind": kind, **health})


def run_pool(stage, kind, arguments, binding, upstream):
    recovery_binding = validate_recovery()
    c.require(binding == ORIGINAL_BINDING, "Wrong scientific binding")
    c.budget_boundary(OUTPUT)
    reusable = old.read_json(RECOVERY / "inventory.json")
    key = f"{stage}_{kind}"
    if (stage, kind) in SEALED_SOURCE_STAGES:
        files = p.stage_lock(stage, kind, binding)["files"]
        c.require(files == reusable["sealed_source_stages"][key], "Sealed reuse changed")
        c.require(len(arguments) == len(files), "Reused argument count changed")
        event(f"{key}_reused", completed=len(files), total=len(files))
        return files
    c.require((stage, kind) == ("source_main", "audits")
              or (stage in ("probe", "main") and kind in ("collection", "audits")), "Unapproved recovery work")
    lock_path = OUTPUT / f"{key}_lock.json"
    c.require(not lock_path.exists(), "One-shot recovery stage; preserve completed evidence")
    retained = reusable["retained_main_source_audits"] if stage == "source_main" else []
    files = list(retained)
    for item in retained:
        spec, game = arguments[item["index"]]
        reconcile_receipt(spec, game, item, binding)
    skip = {item["index"] for item in retained}
    pending = [(i, arg) for i, arg in enumerate(arguments) if i not in skip]
    if stage == "source_main":
        c.require([i for i, _ in pending] == reusable["missing_main_source_audit_indices"], "Missing audit schedule changed")
    started = time.perf_counter()
    launch_path = RECOVERY / "stages" / f"{key}_launch.json"
    deps = [*upstream, RECOVERY / "freeze.json", RECOVERY / "inventory.json", RECOVERY / "launch.json"]
    if stage == "source_main":
        deps.append(OUTPUT / "source_main_audits_launch.json")
    old.save_new(launch_path, {"manifest": binding, "recovery_binding": recovery_binding,
                 "stage": stage, "kind": kind, "time": old.now(), "dependencies": [c.artifact(x) for x in deps],
                 "retained": retained, "pending_indices": [i for i, _ in pending]})
    runtime = c.windows.configure_spawn()
    event(key, completed=len(files), total=len(arguments), reused=len(retained))
    with ProcessPoolExecutor(max_workers=c.CONFIG["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, kind, binding, OUTPUT)) as pool:
        jobs = {pool.submit(p.run_job, stage, kind, arg, binding, OUTPUT): i for i, arg in pending}
        try:
            for job in as_completed(jobs):
                item = job.result()
                c.require(item["index"] == jobs[job], "Worker returned another scheduled index")
                c.verify_hash(item)
                files.append(item)
                event(key, completed=len(files), total=len(arguments), reused=len(retained))
                c.budget_boundary(OUTPUT)
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    files.sort(key=lambda item: item["index"])
    c.require([item["index"] for item in files] == list(range(len(arguments))), "Missing or duplicate recovered job")
    for item in files:
        c.verify_hash(item)
    if stage == "source_main":
        for (spec, game), item in zip(arguments, files, strict=True):
            reconcile_receipt(spec, game, item, binding)
    old.save_new(lock_path, {"manifest": binding, "recovery_binding": recovery_binding,
                 "launch": c.artifact(launch_path), "files": files, "elapsed_seconds": time.perf_counter()-started,
                 "elapsed_scope": "recovery invocation only; original launch clock still governs the budget",
                 "reused_count": len(retained), "runtime": runtime})
    p.stage_lock(stage, kind, binding)
    return files


def continue_pipeline(binding):
    # Explicit operational substitution only; all scientific functions stay original.
    previous_event, previous_pool = p.event, p.run_pool
    p.event, p.run_pool = event, run_pool
    try:
        p.build_sources(binding)
        c.budget_boundary(OUTPUT)
        p.build_stage("probe", binding)
        if not p.operational_decision(binding):
            return False
        p.build_stage("main", binding)
        c.budget_boundary(OUTPUT)
        validate_recovery()
        p.finish(binding)
        return True
    finally:
        p.event, p.run_pool = previous_event, previous_pool


def run():
    folder = RECOVERY / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder / "stdout.txt", folder / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        c.require(not (RECOVERY / "launch.json").exists(), "One-shot recovery; no automatic retry")
        recovery_binding = validate_recovery()
        reusable = old.read_json(RECOVERY / "inventory.json")
        c.require(inventory(ORIGINAL_BINDING) == reusable, "Starting evidence changed after recovery freeze")
        try:
            with p.parent.parent.inhibit_system_sleep() as sleep:
                old.save_new(RECOVERY / "launch.json", {"manifest": ORIGINAL_BINDING,
                             "recovery_binding": recovery_binding, "time": old.now(), "epoch_seconds": time.time(),
                             "original_launch_epoch": reusable["original_launch_epoch"], "deadline_epoch": reusable["deadline_epoch"],
                             "pid": os.getpid(), "sleep_inhibition": sleep,
                             "windowless": c.windows.assert_windowless_worker(), "owner_authorized": True})
                event("resuming", reused_models=9, reused_source_games=1134, retained_main_source_audits=52,
                      remaining_main_source_audits=1028, deadline_epoch=reusable["deadline_epoch"])
                complete = continue_pipeline(ORIGINAL_BINDING)
                validate_recovery()
                old.save_new(RECOVERY / "closure.json", {"manifest": ORIGINAL_BINDING,
                             "recovery_binding": recovery_binding, "time": old.now(), "complete": complete,
                             "scientific_status": c.artifact(OUTPUT / ("final_status.json" if complete else "operational_hold.json")),
                             "preserved_original_failure": c.artifact(OUTPUT / "failure.json"), "automatic_successor": False})
        except c.BudgetExceeded as exc:
            receipt = {"manifest": ORIGINAL_BINDING, "recovery_binding": recovery_binding,
                       "time": old.now(), "reason": str(exc)}
            old.save_new(RECOVERY / "operational_hold.json", receipt)
            if not (OUTPUT / "operational_hold.json").exists():
                old.save_new(OUTPUT / "operational_hold.json", receipt)
            event("operational_hold", reason=str(exc))
        except BaseException:
            old.save_new(RECOVERY / "failure.json", {"manifest": ORIGINAL_BINDING,
                         "recovery_binding": recovery_binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure=str(RECOVERY / "failure.json"))
            raise


def launch():
    recovery_binding = validate_recovery()
    c.budget_boundary(OUTPUT)
    folder = RECOVERY / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder / "launcher.lock"):
        c.require(not (folder / "process.json").exists() and not (RECOVERY / "launch.json").exists(), "Recovery already launched")
        with old.process_lock(OUTPUT / "run.lock"):
            c.require(inventory(ORIGINAL_BINDING) == old.read_json(RECOVERY / "inventory.json"), "Evidence changed before launch")
        process = c.windows.detached_launch([sys.executable, "-u", ENTRY, "run"],
                                            folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "recovery_binding": recovery_binding, "time": old.now()})
        deadline = time.monotonic() + 60
        while not (RECOVERY / "launch.json").exists():
            c.require(time.monotonic() < deadline, "No recovery acknowledgement; inspect before any retry")
            time.sleep(.1)
        receipt = old.read_json(RECOVERY / "launch.json")
        c.require(receipt["pid"] == process["pid"] and receipt["recovery_binding"] == recovery_binding,
                  "Recovery supervisor identity mismatch")
        old.save_new(folder / "acknowledged.json", {"time": old.now(), "pid": process["pid"],
                     "launch": c.artifact(RECOVERY / "launch.json")})
        print({"resumed": True, "pid": process["pid"], "recovery_binding": recovery_binding,
               "deadline_epoch": receipt["deadline_epoch"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command in ("check", "prepare"):
        c.require(args.checks is not None, "Supply recovery checks path")
        if args.command == "check":
            return check(args.checks)
        prepare(args.checks)
    elif args.command == "launch":
        launch()
    else:
        run()
    return 0
