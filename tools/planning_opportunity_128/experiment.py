"""Hidden one-shot orchestration for planning-opportunity-weighted teaching."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
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
CHECK_TARGETS = [f"tools/planning_opportunity_128/test_{name}.py" for name in
                 ("source", "learning", "assessment", "statistics", "experiment", "worker")]
EXTRAS = ["run_planning_opportunity_128.py", "docs/engineering/planning_opportunity_128.md",
          "docs/engineering/verification_policy.md"]
RUNTIME_MARKERS = ("abort.json", "failure.json", "operational_hold.json", "launch.json",
                   "runtime/process.json", "fits_launch.json", "fit_audits_launch.json",
                   "models_lock.json", "fits", "workers")
REPORTER = None


def assert_prelaunch_clean(base=None):
    base = Path(OUTPUT if base is None else base)
    present = [name for name in RUNTIME_MARKERS if (base/name).exists()]
    c.require(not present, "Runtime evidence already exists; preserve and inspect: " + ", ".join(present))


def identity():
    from . import learning
    c.require(sys.flags.optimize == 0, "Unoptimized Python required")
    dependencies = {}
    parent_freeze = old.read_json(c.CODE_PARENT/"freeze.json")
    parent_manifest = c.verified(parent_freeze["manifest"])
    c.require(parent_freeze["binding"] == old.digest(parent_manifest) == c.CODE_PARENT_BINDING,
              "Frozen code parent identity changed")
    for path, sha in parent_manifest["identity"]["dependencies"].items():
        c.require(old.file_hash(ROOT/path) == sha, "Frozen parent changed: " + path)
        dependencies[path] = sha
    paths = list((ROOT/"tools/planning_opportunity_128").glob("*.py")) + [ROOT/p for p in EXTRAS]
    dependencies.update({p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths})
    return {"config": c.CONFIG, "learning_config": learning.CONFIG, "fits": list(c.FITS),
            "checks": CHECK_TARGETS, "dependencies": dict(sorted(dependencies.items())),
            "python": sys.version, "optimize": sys.flags.optimize,
            "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": c.windows._identity(c.windows._windowless_executable())}


def check(folder):
    assert_prelaunch_clean()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    before, start = identity(), time.perf_counter()
    fixture = ROOT/"outputs/planning_opportunity_128_fixtures"/folder.name
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
    checked = old.read_json(checks_path)
    frozen_identity = identity()
    c.require(checked["exit_code"] == 0 and checked["targets"] == CHECK_TARGETS
              and checked["identity"] == frozen_identity, "Current passing preflight required")
    for ref in checked["logs"]: c.verify_hash(ref)
    parent = source.closed_parent()
    manifest = {"identity": frozen_identity, "checks": c.artifact(checks_path), "source_parent": parent,
                "owner_authorized": True,
                "authorization": "Owner accepted on2026-10-09: one planning-opportunity-weighted full-policy fit, valuehead frozen, three original161 anchors, exactR1 BROAD batches/data/soft.05/lr1e-4/1784updates; unchanged soft targets with positive opportunity loss weights normalized by exact TRAIN exposures within each old/added pool; independent reproduction; original2489VAL and336saved-outcome development screen; hiddenCPU, no new games, rollouts, sweep or automatic successor."}
    binding = old.digest(manifest)
    source.prepare(binding, OUTPUT, parent=parent)
    c.require(identity() == frozen_identity, "Code changed during preparation")
    assert_prelaunch_clean()
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"freeze.json", {"binding": binding, "manifest": c.artifact(OUTPUT/"manifest.json"),
                                      "owner_authorized": True, "time": old.now()})
    c.require(validate_freeze() == binding, "Post-freeze identity failed")
    print({"prepared": True, "binding": binding, "fits": 3, "updates": c.CONFIG["updates"]}, flush=True)


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
    try:
        old.save_new(base/"abort.json", {"time": old.now(), "pid": os.getpid(), "error": str(exc)})
    except FileExistsError:
        pass


def worker_initialize(kind, binding, base):
    folder = Path(base)/"workers"/kind; folder.mkdir(parents=True, exist_ok=True)
    health = c.windows.assert_windowless_worker(folder/f"{os.getpid()}_stdout.txt", folder/f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    old.save_new(folder/f"{os.getpid()}_health.json", {"manifest": binding, "kind": kind, **health})


def run_job(kind, arg, binding, base):
    from . import learning
    try:
        c.budget_boundary(base)
        if kind == "fits": return learning.fit(arg, binding, base)
        c.require(kind == "fit_audits", "Unknown worker job")
        return learning.audit_fit(arg[0], arg[1], binding, base)
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
    files.sort(key=lambda ref: ref["fit"])
    c.require([ref["fit"] for ref in files] == list(c.FITS), "Missing or duplicate fit")
    for ref in files: c.verify_hash(ref)
    old.save_new(OUTPUT/f"{kind}_lock.json", {"manifest": binding, "files": files, "runtime": runtime,
                                           "elapsed_seconds": time.perf_counter()-start})
    return files


def finish(binding, result_ref, audit_ref, fits, fit_audits):
    from . import learning
    c.require(validate_freeze() == binding, "Final freeze changed")
    learning.validate_models(binding, OUTPUT)
    result, checked = c.verified(result_ref), c.verified(audit_ref)
    c.require(result["manifest"] == checked["manifest"] == binding and checked["passed"], "Assessment audit failed")
    c.require(all(checked.get(name) is True for name in (
        "all_predictions_independently_reproduced", "all_private_input_checks_pass",
        "full_validation_old_label_metrics_independently_reproduced",
        "marginal_shortlist_regret_independently_reproduced",
        "acquisition_marginal_regret_independently_reproduced",
        "all_outcome_arithmetic_independently_reproduced", "statistics_independently_reproduced",
        "fixed_controls_reproduced")), "Incomplete assessment audit")
    c.require(checked["result"] == {key: result_ref[key] for key in ("path", "sha256")}, "Assessment linkage differs")
    c.verify_hash(result["validation_old_targets"])
    training = {str(ref["fit"]): c.verified(ref) for ref in fits}
    screen = result["development_screen"]
    final = {"manifest": binding, "time": old.now(), "classification": "planning_opportunity_development_screen",
             "all_prescribed_audits_pass": True, "assessment": result_ref, "assessment_audit": audit_ref,
             "statistics": result["statistics"], "training": training,
             "development_screen_pass": screen["passed"], "development_screen": screen,
             "validation_old_targets": result["validation_old_targets"],
             "models": c.artifact(OUTPUT/"models_lock.json"), "data": c.artifact(OUTPUT/"data_lock.json"),
             "training_intervention": "only_per_position_loss_weight_changes_from_full_BROAD",
             "new_simulations": 0, "new_source_games": 0, "automatic_successor": False,
             "scope": "Reused-development, recorded-support conditional decision utility; not a game win rate or fresh strength test"}
    old.save_new(OUTPUT/"result.json", final)
    old.save_new(OUTPUT/"final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "result": c.artifact(OUTPUT/"result.json"), "assessment": audit_ref, "fits": fit_audits,
                 "training_reproduced": True, "historical_simulations_replayed": False})
    lines = ["# Learning weighted by planning opportunity", "",
             "Three original161 anchors; the full policy trains and the value head remains frozen, matching the saved BROAD controls. The sole recipe change is a per-position loss weight: positive best-outsider score minus best-of-actual-first-three score. Each old/added pool's weights are normalized once using its exact prescribed TRAIN exposures. Soft targets, model, exact batches, optimizer and1,784-update endpoint are unchanged. All three optimizer trajectories and weight calculations independently checked.", "",
             "Zero-weight examples retain their batch slots and all optimizer steps. Scores and weights use finite saved128-world estimates; noisy positive gaps can be amplified. Weight concentration and effective source units are saved in every fit result. This is a test of allocating learning effort, not proof that the teacher's rankings are exact or that representation was the bottleneck.", "",
             "No new games or rollouts. Original2,489 validation positions and336 conditional roots are reused DEVELOPMENT evidence. The168 validation roots come from42 whole pairs; each root uses its original assigned source-fit model. All predictions and selection choices were sealed before independent assessment outcomes were read. Historical simulation audits were authenticated, not replayed again.", "",
             "The test is restricted to recorded candidate support. Results are conditional expected scores, not game win rates, and cannot certify unrestricted deployment. No new target probabilities, features or capacity constraints were introduced.", "",
             "## Validation planning", "", "| Contrast | Gain (pp) | Paired95% interval (pp) |", "|---|---:|---:|"]
    for name, entry in result["statistics"]["groups"]["validation"]["contrasts"].items():
        lines.append(f"| {name} | {100*entry['mean_bounds'][0]:+.6f} | [{100*entry['ci95'][0]:+.6f}, {100*entry['ci95'][1]:+.6f}] |")
    lines += ["", f"Composite development screen: {'PASS' if screen['passed'] else 'NOT ESTABLISHED'}.", "",
              "The screen requires at least50% added-TRAIN marginal shortlist regret capture on every fit, lower pair-weighted validation marginal shortlist regret than both controls for every fit, at least+.5pp planning gain versus anchor with positive paired95% lower bound, positive lower bound versus BROAD, and nonnegative per-fit planning differences. Ordinary outsider regret and label agreement are descriptive. No secondary result rescues a failed requirement.", "",
              "## Gate details", "", "```json", json.dumps(screen, indent=2, sort_keys=True), "```", "",
              "## Full conditional statistics", "", "```json", json.dumps(result["statistics"], indent=2, sort_keys=True), "```", "",
              "No automatic successor, fresh confirmation, parameter sweep or promotion.", ""]
    with (OUTPUT/"report.md").open("x", encoding="utf-8", newline="\n") as handle: handle.write("\n".join(lines))
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "all_prescribed_audits_pass": True, "classification": final["classification"], "automatic_successor": False,
                 "result": c.artifact(OUTPUT/"result.json"), "audit": c.artifact(OUTPUT/"final_audit.json"),
                 "report": c.artifact(OUTPUT/"report.md")})
    event("complete", classification=final["classification"])


def run():
    global REPORTER
    from . import assessment, learning
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt", folder/"stderr.txt")
    torch.set_num_threads(c.CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(), "One-shot launch; preserve evidence")
        binding = validate_freeze()
        REPORTER = StatusReporter(OUTPUT, {"manifest": binding})
        try:
            with ancestor.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT/"launch.json", {"manifest": binding, "time": old.now(), "epoch_seconds": time.time(),
                             "pid": os.getpid(), "owner_authorized": True, "sleep_inhibition": sleep,
                             "torch_threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
                             "windowless": c.windows.assert_windowless_worker()})
                fits = run_pool("fits", list(c.FITS), binding)
                audits = run_pool("fit_audits", list(zip(c.FITS, fits)), binding)
                learning.models_lock(fits, audits, binding, OUTPUT)
                event("assessment")
                result = assessment.assess(binding, OUTPUT)
                event("assessment_audit")
                checked = assessment.audit(binding, result, OUTPUT)
                finish(binding, result, checked, fits, audits)
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
        process = c.windows.detached_launch([sys.executable, "-u", ROOT/"run_planning_opportunity_128.py", "run"],
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
