"""Freeze and run a single hidden control-variate feasibility study."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from . import common as c
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.proposal_iteration_128 import experiment as ancestor

ROOT, OUTPUT, old = c.ROOT, c.OUTPUT, c.old
CHECK_TARGETS = [f"tools/control_variate_128/test_{name}.py" for name in
                 ("source", "engine", "verify", "statistics", "experiment", "worker")]
EXTRAS = ["run_control_variate_128.py", "docs/engineering/control_variate_128.md",
          "docs/engineering/verification_policy.md"]
RUNTIME_MARKERS = ("abort.json", "failure.json", "operational_hold.json", "launch.json",
                   "runtime/process.json", "timing_launch.json", "timing_lock.json",
                   "train", "validation", "workers", "selection_lock.json", "result.json")
REPORTER = None


def assert_prelaunch_clean(base=None):
    base = Path(OUTPUT if base is None else base)
    present = [name for name in RUNTIME_MARKERS if (base/name).exists()]
    c.require(not present, "Runtime evidence already exists; preserve and inspect: " + ", ".join(present))


def identity():
    c.require(sys.flags.optimize == 0, "Unoptimized Python required")
    freeze = old.read_json(c.CODE_PARENT/"freeze.json")
    parent = c.verified(freeze["manifest"])
    c.require(freeze["binding"] == old.digest(parent) == c.CODE_PARENT_BINDING,
              "Frozen code parent changed")
    dependencies = dict(parent["identity"]["dependencies"])
    for path, sha in dependencies.items():
        c.require(old.file_hash(ROOT/path) == sha, "Frozen parent changed: " + path)
    paths = list((ROOT/"tools/control_variate_128").glob("*.py")) + [ROOT/p for p in EXTRAS]
    dependencies.update({p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths})
    return {"config": c.CONFIG, "fits": list(c.FITS), "checks": CHECK_TARGETS,
            "dependencies": dict(sorted(dependencies.items())), "python": sys.version,
            "optimize": sys.flags.optimize, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": c.windows._identity(c.windows._windowless_executable())}


def check(folder):
    assert_prelaunch_clean()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    before, start = identity(), time.perf_counter()
    fixture = ROOT/"outputs/control_variate_128_fixtures"/folder.name
    c.require(not fixture.exists(), "Preserve previous fixtures")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    args = ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture)]
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main("+repr(args)+"))"
    with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
        proc = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT,
                              stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=1200,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity() == before, "Code changed during official preflight")
    assert_prelaunch_clean()
    old.save_new(folder/"checks.json", {"identity": before, "targets": CHECK_TARGETS,
                 "exit_code": proc.returncode, "seconds": time.perf_counter()-start,
                 "logs": [c.artifact(folder/n) for n in ("stdout.txt", "stderr.txt")]})
    print({"checks_exit": proc.returncode, "seconds": time.perf_counter()-start}, flush=True)
    return proc.returncode


def prepare(checks_path):
    from . import source
    assert_prelaunch_clean()
    c.require(not (OUTPUT/"manifest.json").exists(), "Already frozen; preserve evidence")
    checked, frozen_identity = old.read_json(checks_path), identity()
    c.require(checked["exit_code"] == 0 and checked["targets"] == CHECK_TARGETS
              and checked["identity"] == frozen_identity, "Current passing preflight required")
    for ref in checked["logs"]: c.verify_hash(ref)
    parent = source.closed_parent()
    manifest = {"identity": frozen_identity, "checks": c.artifact(checks_path), "source_parent": parent,
        "owner_authorized": True,
        "authorization": "Owner accepted on2026-10-09: one bounded CPU control-variate feasibility study; frozen143 evaluator, saved336 roots and independent128 terminal outcomes, lawful4/8/16-ply prefixes, TRAIN-only timing and analytic rule calibration, sealed rule before168VAL, independent verification and report. No neural training, new source games, arena, promotion or automatic successor."}
    binding = old.digest(manifest)
    source.prepare(binding, OUTPUT, parent=parent)
    c.require(identity() == frozen_identity, "Code changed during preparation")
    assert_prelaunch_clean()
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"freeze.json", {"binding": binding, "manifest": c.artifact(OUTPUT/"manifest.json"),
                                      "owner_authorized": True, "time": old.now()})
    c.require(validate_freeze() == binding, "Post-freeze identity failed")
    print({"prepared": True, "binding": binding, "states": c.CONFIG["states"]}, flush=True)


def validate_freeze():
    from . import source
    freeze = old.read_json(OUTPUT/"freeze.json")
    manifest = c.verified(freeze["manifest"])
    c.require(freeze["owner_authorized"] and old.digest(manifest) == freeze["binding"]
              and manifest["identity"] == identity(), "Freeze or code changed")
    checked = c.verified(manifest["checks"])
    c.require(checked["exit_code"] == 0 and checked["identity"] == manifest["identity"], "Preflight changed")
    for ref in checked["logs"]: c.verify_hash(ref)
    for ref in manifest["source_parent"]["evidence"]: c.verify_hash(ref)
    source.validate(freeze["binding"], OUTPUT, authenticate_parent=False)
    return freeze["binding"]


def event(stage, **values):
    global REPORTER
    if REPORTER is None: REPORTER = StatusReporter(OUTPUT)
    return REPORTER.event(stage, **values)


def abort(exc, base=None):
    base = Path(OUTPUT if base is None else base)
    try: old.save_new(base/"abort.json", {"time": old.now(), "pid": os.getpid(), "error": str(exc)})
    except FileExistsError: pass


def worker_initialize(kind, binding, base):
    folder = Path(base)/"workers"/kind; folder.mkdir(parents=True, exist_ok=True)
    health = c.windows.assert_windowless_worker(folder/f"{os.getpid()}_stdout.txt", folder/f"{os.getpid()}_stderr.txt")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if kind == "fixture": old.initialize_worker()
    else:
        from . import engine
        engine.worker_initialize(base)
    old.save_new(folder/f"{os.getpid()}_health.json", {"manifest": binding, "kind": kind, **health})


def run_job(kind, arg, binding, base):
    from . import engine, verify
    try:
        c.budget_boundary(base)
        if kind == "timing": return engine.timing(arg, binding, base)
        if kind == "timing_audits": return verify.audit_timing(arg[0], arg[1], binding, base)
        if kind in ("train", "validation"): return engine.collect(arg, binding, base, stage=kind)
        c.require(kind in ("train_audits", "validation_audits"), "Unknown worker job")
        return verify.audit(arg[0], arg[1], binding, base, stage=kind.removesuffix("_audits"))
    except BaseException as exc:
        abort(exc, base); raise


def dispatch(pool, args, job, workers, on_result, boundary):
    iterator, pending, files = iter(args), set(), []
    def submit():
        arg = next(iterator, None)
        if arg is not None: pending.add(pool.submit(job, arg))
    for _ in range(min(workers, len(args))): submit()
    try:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            batch = []
            for future in done:
                pending.remove(future); batch.append(future.result())
            files.extend(batch); boundary(); on_result(len(files))
            for _ in batch: submit()
    except BaseException:
        for future in pending: future.cancel()
        raise
    return files


def run_pool(kind, args, binding):
    c.require(validate_freeze() == binding, "Stage freeze changed")
    c.budget_boundary(OUTPUT)
    start = time.perf_counter()
    old.save_new(OUTPUT/f"{kind}_launch.json", {"manifest": binding, "time": old.now(), "args_sha256": old.digest(args)})
    event(kind, completed=0, total=len(args))
    runtime = c.windows.configure_spawn()
    try:
        with ProcessPoolExecutor(max_workers=c.CONFIG["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_initialize, initargs=(kind, binding, OUTPUT)) as pool:
            try:
                files = dispatch(pool, args, partial(run_job, kind, binding=binding, base=OUTPUT), c.CONFIG["workers"],
                                 lambda n: event(kind, completed=n, total=len(args)), lambda: c.budget_boundary(OUTPUT))
            except BaseException as exc:
                abort(exc, OUTPUT); raise
    except BaseException as exc:
        abort(exc, OUTPUT); raise
    files.sort(key=lambda ref: ref["index"])
    expected = sorted((arg[0] if kind.endswith("_audits") else arg)["index"] for arg in args)
    c.require([r["index"] for r in files] == expected and len(set(expected)) == len(expected), "Missing or duplicate root")
    for ref in files: c.verify_hash(ref)
    old.save_new(OUTPUT/f"{kind}_lock.json", {"manifest": binding, "files": files, "runtime": runtime,
                                           "elapsed_seconds": time.perf_counter()-start})
    return files


def paired_audit_args(specs, refs):
    lookup = {r["index"]: r for r in refs}
    c.require(len(lookup) == len(specs) and set(lookup) == {s["index"] for s in specs}, "Collection inventory differs")
    return [(spec, lookup[spec["index"]]) for spec in specs]


def verify_stage(kind, binding):
    """Require linked, complete independent receipts before consuming a stage."""
    collected = old.read_json(OUTPUT/f"{kind}_lock.json")
    checked = old.read_json(OUTPUT/f"{kind}_audits_lock.json")
    c.require(collected["manifest"] == checked["manifest"] == binding, "Stage manifest changed")
    left = {r["index"]: r for r in collected["files"]}
    right = {r["index"]: r for r in checked["files"]}
    c.require(left.keys() == right.keys() and len(left) == len(collected["files"])
              == len(checked["files"]), "Stage audit coverage differs")
    flags = ["all_endpoint_features_verified", "all_endpoint_values_verified",
             "sampled_prefixes_independently_replayed", "source_world_coupling_verified"]
    flags += (["all_full_outcomes_equal_saved", "matched_timing_arms_verified"] if kind == "timing"
              else ["statistics_projection_verified"])
    for index in left:
        c.verify_hash(left[index])
        audit = c.verified(right[index])
        c.require(audit["manifest"] == binding and audit["index"] == index
                  and audit["passed"] is True and audit["result"] == left[index]
                  and all(audit.get(key) is True for key in flags), "Incomplete stage audit")
    return checked


def verify_selection(binding, selection_ref, audit_ref):
    selected, audited = c.verified(selection_ref), c.verified(audit_ref)
    c.require(selected["manifest"] == audited["manifest"] == binding and audited["passed"] is True
              and audited["selection"] == selection_ref and audited["validation_read"] is False
              and all(audited.get(key) is True for key in ("moments_independently_reconstructed",
                  "allocation_independently_reproduced", "fixed_train_rule_preserved")),
              "Incomplete selection audit")
    return selected


def forecast(timing_refs, binding):
    """Bound the remaining study using current-host timing; no strength reads."""
    from . import engine
    value = engine.execution_forecast(timing_refs, binding, OUTPUT)
    c.require(value["manifest"] == binding and value["outcome_blind"] is True,
              "Invalid operational forecast")
    old.save_new(OUTPUT/"forecast.json", value)
    if not value["disk_reserve_pass"]:
        raise c.BudgetExceeded("Forecast cannot retain the fixed five-GiB disk reserve")
    if not value["within_runtime_ceiling"]:
        raise c.BudgetExceeded("Timing forecast exceeds the fixed two-hour execution ceiling")
    event("forecast_pass", estimated_total_seconds=value["estimated_total_seconds"])
    return value


def finish(binding, result_ref, audit_ref, selection_ref, selection_audit_ref):
    c.require(validate_freeze() == binding, "Final freeze changed")
    result, checked = c.verified(result_ref), c.verified(audit_ref)
    selection = verify_selection(binding, selection_ref, selection_audit_ref)
    c.require(result["manifest"] == checked["manifest"] == binding and checked["passed"] is True,
              "Assessment audit failed")
    c.require(checked["result"] == result_ref, "Assessment audit linkage changed")
    c.require(checked["selection"] == selection_ref and result["selection"] == selection_ref
              and all(checked.get(key) is True for key in ("moments_independently_reconstructed",
                  "bootstrap_independently_reproduced", "fixed_train_rule_preserved")),
              "Incomplete assessment audit")
    for kind in ("timing", "train", "validation"): verify_stage(kind, binding)
    final = {"manifest": binding, "time": old.now(), "complete": True,
             "classification": "control_variate_development_feasibility",
             "all_prescribed_audits_pass": True, "assessment": result_ref, "assessment_audit": audit_ref,
             "selection": selection_ref, "selection_audit": selection_audit_ref,
             "data": c.artifact(OUTPUT/"data_lock.json"), "forecast": c.artifact(OUTPUT/"forecast.json"),
             "new_neural_fits": 0, "new_source_games": 0, "actual_estimator_player_tested": False,
             "automatic_successor": False, "promotion": False,
             "scope": "Predicted variance and CPU efficiency on reused development positions; not game win rates or a deployed planner test",
             "statistics": result}
    old.save_new(OUTPUT/"result.json", final)
    audit = {"manifest": binding, "passed": True, "time": old.now(), "result": c.artifact(OUTPUT/"result.json"),
             "assessment": audit_ref, "selection": selection_audit_ref,
             "prefix_audits": [c.artifact(OUTPUT/f"{s}_audits_lock.json") for s in ("train", "validation")],
             "timing": c.artifact(OUTPUT/"timing_audits_lock.json"), "historical_full_replays": "bounded_timing_only"}
    old.save_new(OUTPUT/"final_audit.json", audit)
    lines = ["# Control-variate feasibility", "",
        "The frozen143 public evaluator estimates short greedy035 continuations. TRAIN selected one horizon, coefficient and finite nested allocation; the rule was sealed and independently checked before VALIDATION prefixes or scientific outcomes were read.", "",
        "The expectation remains the original terminal score. The finite variance and computation of estimating the cheap mean are included. These are predicted variance/cost efficiencies on reused development positions, not game win rates, fresh generalization or a tested deployment speedup. No new learned critic, source games, arena or automatic successor.", "",
        "## Sealed rule", "", "```json", json.dumps(selection, indent=2, sort_keys=True), "```", "",
        "## Assessment", "", "```json", json.dumps(result, indent=2, sort_keys=True), "```", "",
        "Intervals condition on the TRAIN cost calibration and chosen rule. Timing measurements and phase sensitivity must be considered separately. Independent trajectory replay is sampled at every root with fixed full stress roots; every saved endpoint's public value arithmetic is reconstructed. Historical terminal records are reused by hash, with full replay only in the bounded timing panel.", ""]
    with (OUTPUT/"report.md").open("x", encoding="utf-8", newline="\n") as out: out.write("\n".join(lines))
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
        "all_prescribed_audits_pass": True, "classification": final["classification"], "automatic_successor": False,
        "result": c.artifact(OUTPUT/"result.json"), "audit": c.artifact(OUTPUT/"final_audit.json"),
        "report": c.artifact(OUTPUT/"report.md")})
    event("complete", classification=final["classification"])


def calibration_stop(binding, selection_ref, audit_ref):
    selection = verify_selection(binding, selection_ref, audit_ref)
    c.require(selection["status"] in ("no_feasible_allocation", "undefined_zero_training_variance")
              and selection["config"] is None, "Invalid calibration stop")
    for kind in ("timing", "train"): verify_stage(kind, binding)
    c.require(validate_freeze() == binding, "Final freeze changed")
    final = {"manifest": binding, "complete": True, "time": old.now(),
        "classification": selection["status"], "selection": selection_ref, "audit": audit_ref,
        "validation_evaluated": False, "automatic_successor": False, "all_prescribed_audits_pass": True,
        "feasibility_pass": False, "actual_estimator_player_tested": False, "promotion": False,
        "new_neural_fits": 0, "new_source_games": 0}
    old.save_new(OUTPUT/"result.json", final)
    old.save_new(OUTPUT/"final_audit.json", {"manifest": binding, "passed": True, "time": old.now(),
        "result": c.artifact(OUTPUT/"result.json"), "selection": audit_ref,
        "timing": c.artifact(OUTPUT/"timing_audits_lock.json"),
        "train": c.artifact(OUTPUT/"train_audits_lock.json"), "validation_evaluated": False})
    text = ("# Control-variate feasibility\n\nThe prescribed TRAIN calibration stopped with classification `"
        + selection["status"] + "`. No validation outcomes were evaluated and no successor was launched. "
        "This is a nonpassing feasibility result, not a game-strength measurement.\n\n"
        "The source, timing, prefix and selection audits passed. The complete sealed calibration is in "
        "`selection_lock.json`; no alternative evaluator, allocation or population was substituted.\n")
    with (OUTPUT/"report.md").open("x", encoding="utf-8", newline="\n") as out: out.write(text)
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json", {**final, "result": c.artifact(OUTPUT/"result.json"),
        "audit": c.artifact(OUTPUT/"final_audit.json"), "report": c.artifact(OUTPUT/"report.md")})
    event("complete", classification=selection["status"])


def run():
    global REPORTER
    from . import source, statistics
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt", folder/"stderr.txt")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(), "One-shot launch; preserve evidence")
        binding = validate_freeze()
        REPORTER = StatusReporter(OUTPUT, {"manifest": binding})
        try:
            with ancestor.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT/"launch.json", {"manifest": binding, "time": old.now(), "epoch_seconds": time.time(),
                    "pid": os.getpid(), "owner_authorized": True, "sleep_inhibition": sleep,
                    "torch_threads": torch.get_num_threads(), "windowless": c.windows.assert_windowless_worker()})
                timings = run_pool("timing", source.timing_specs(binding, OUTPUT), binding)
                run_pool("timing_audits", paired_audit_args(source.timing_specs(binding, OUTPUT), timings), binding)
                verify_stage("timing", binding)
                forecast(timings, binding)
                train_specs = source.schedule(binding, OUTPUT, "train")
                train = run_pool("train", train_specs, binding)
                run_pool("train_audits", paired_audit_args(train_specs, train), binding)
                verify_stage("train", binding)
                event("calibration")
                selected = statistics.select(train, timings, binding, OUTPUT)
                selected_audit = statistics.audit_selection(train, timings, selected, binding, OUTPUT)
                verify_selection(binding, selected, selected_audit)
                if c.verified(selected)["status"] != "selected":
                    calibration_stop(binding, selected, selected_audit); return
                val_specs = source.schedule(binding, OUTPUT, "validation")
                val = run_pool("validation", val_specs, binding)
                run_pool("validation_audits", paired_audit_args(val_specs, val), binding)
                verify_stage("validation", binding)
                event("assessment")
                result = statistics.assess(val, selected, timings, binding, OUTPUT)
                event("assessment_audit")
                checked = statistics.audit(val, selected, timings, result, binding, OUTPUT)
                finish(binding, result, checked, selected, selected_audit)
        except c.BudgetExceeded as exc:
            old.save_new(OUTPUT/"operational_hold.json", {"manifest": binding, "time": old.now(), "reason": str(exc)})
            event("operational_hold", reason=str(exc))
        except BaseException as exc:
            abort(exc, OUTPUT)
            old.save_new(OUTPUT/"failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure="failure.json"); raise


def launch():
    assert_prelaunch_clean()
    binding = validate_freeze()
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder/"launcher.lock"):
        assert_prelaunch_clean()
        process = c.windows.detached_launch([sys.executable, "-u", ROOT/"run_control_variate_128.py", "run"],
                                           folder/"stdout.txt", folder/"stderr.txt", ROOT)
        old.save_new(folder/"process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic()+600
        while not (OUTPUT/"launch.json").exists():
            c.require(time.monotonic() < deadline, "No acknowledgement; inspect before retry")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT/"launch.json")
        c.require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding, "Supervisor differs")
        old.save_new(folder/"acknowledged.json", {"pid": process["pid"], "time": old.now(), "launch": c.artifact(OUTPUT/"launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command in ("check", "prepare"): c.require(args.checks is not None, "Check destination required")
    if args.command == "check": return check(args.checks)
    if args.command == "prepare": prepare(args.checks)
    elif args.command == "launch": launch()
    else: run()
    return 0
