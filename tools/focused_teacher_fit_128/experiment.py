"""One-shot frozen, hidden focused-fitting experiment and closure."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
import multiprocessing
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
import json
import numpy as np
import torch
from . import common as c
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.proposal_iteration_128 import experiment as ancestor

ROOT, OUTPUT, old = c.ROOT, c.OUTPUT, c.old
CHECK_TARGETS = [f"tools/focused_teacher_fit_128/test_{n}.py" for n in
                 ("source", "learning", "assessment", "experiment", "worker")]
EXTRAS = ["run_focused_teacher_fit_128.py", "docs/engineering/focused_teacher_fit_128.md",
          "docs/engineering/verification_policy.md"]
REPORTER = None


def identity():
    c.require(sys.flags.optimize == 0, "Assertions require unoptimized Python")
    dependencies = {}
    for folder in (c.SOURCE, ROOT/"outputs/proposal_specialist_128"):
        for name, sha in old.read_json(folder/"manifest.json")["identity"]["dependencies"].items():
            c.require(old.file_hash(ROOT/name) == sha, "Frozen predecessor changed: "+name)
            c.require(name not in dependencies or dependencies[name] == sha, "Dependency conflict")
            dependencies[name] = sha
    paths = list((ROOT/"tools/focused_teacher_fit_128").glob("*.py")) + [ROOT/p for p in EXTRAS]
    dependencies.update({p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths})
    return {"config": c.CONFIG, "fits": list(c.FITS), "checks": CHECK_TARGETS,
            "dependencies": dict(sorted(dependencies.items())), "python": sys.version,
            "optimize": sys.flags.optimize, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": c.windows._identity(c.windows._windowless_executable())}


def check(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    before, start = identity(), time.perf_counter()
    fixture = ROOT/"outputs/focused_teacher_fit_128_fixtures"/folder.name
    c.require(not fixture.exists(), "Preserve previous fixtures")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main("+repr(
        ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture)])+"))"
    with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
        proc = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                              stdout=out, stderr=err, timeout=900,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity() == before, "Code changed during official preflight")
    old.save_new(folder/"checks.json", {"identity": before, "targets": CHECK_TARGETS, "exit_code": proc.returncode,
                 "seconds": time.perf_counter()-start, "logs": [c.artifact(folder/n) for n in ("stdout.txt", "stderr.txt")]})
    return proc.returncode


def prepare(checks_path):
    from . import source
    c.require(not (OUTPUT/"manifest.json").exists(), "Already frozen; preserve evidence")
    checked = old.read_json(checks_path)
    c.require(checked["exit_code"] == 0 and checked["targets"] == CHECK_TARGETS and checked["identity"] == identity(),
              "Full current passing preflight required")
    for ref in checked["logs"]: c.verify_hash(ref)
    parent = source.closed_parent()
    manifest = {"identity": identity(), "checks": c.artifact(checks_path), "source_parent": parent,
                "owner_authorized": True,
                "authorization": "Owner accepted focused teacher-fitting test on2026-10-08; three fixed fits, independent reproduction, saved-score diagnostic; no new simulations, arena or successor."}
    old.save_new(OUTPUT/"manifest.json", manifest)
    binding = old.digest(manifest)
    old.save_new(OUTPUT/"freeze.json", {"binding": binding, "manifest": c.artifact(OUTPUT/"manifest.json"),
                                      "owner_authorized": True, "time": old.now()})
    source.prepare(binding, OUTPUT)
    c.require(validate_freeze() == binding, "Post-freeze identity failed")
    print({"prepared": True, "binding": binding, "fits": 3, "updates": c.CONFIG["updates"]}, flush=True)


def validate_freeze():
    from . import source
    freeze = old.read_json(OUTPUT/"freeze.json")
    manifest = c.verified(freeze["manifest"])
    c.require(freeze["owner_authorized"] and old.digest(manifest) == freeze["binding"]
              and manifest["identity"] == identity(), "Freeze or code changed")
    checks = c.verified(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"], "Preflight changed")
    for ref in [*checks["logs"], *manifest["source_parent"]["evidence"]]: c.verify_hash(ref)
    source.validate(freeze["binding"], OUTPUT)
    return freeze["binding"]


def event(stage, **values):
    global REPORTER
    if REPORTER is None: REPORTER = StatusReporter(OUTPUT)
    return REPORTER.event(stage, **values)


def abort(exc, base=OUTPUT):
    path = Path(base)/"abort.json"
    try:
        old.save_new(path, {"time": old.now(), "pid": os.getpid(), "error": str(exc)})
    except FileExistsError:
        pass


def worker_initialize(kind, binding, base):
    folder = Path(base)/"workers"/kind; folder.mkdir(parents=True, exist_ok=True)
    health = c.windows.assert_windowless_worker(folder/f"{os.getpid()}_stdout.txt", folder/f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    torch.set_num_threads(1)
    old.save_new(folder/f"{os.getpid()}_health.json", {"manifest": binding, "kind": kind, **health})


def run_job(kind, arg, binding, base):
    from . import learning
    try:
        c.budget_boundary(base)
        if kind == "fits": return learning.fit(arg, binding, base)
        c.require(kind == "fit_audits", "Unknown job")
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
    from functools import partial
    c.require(validate_freeze() == binding, "Stage freeze changed")
    c.budget_boundary(OUTPUT)
    start = time.perf_counter()
    old.save_new(OUTPUT/f"{kind}_launch.json", {"manifest": binding, "time": old.now(), "args_sha256": old.digest(args)})
    event(kind, completed=0, total=len(args))
    runtime = c.windows.configure_spawn()
    with ProcessPoolExecutor(max_workers=c.CONFIG["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(kind, binding, OUTPUT)) as pool:
        files = dispatch(pool, args, partial(run_job, kind, binding=binding, base=OUTPUT), c.CONFIG["workers"],
                         lambda n: event(kind, completed=n, total=len(args)), lambda: c.budget_boundary(OUTPUT))
    files.sort(key=lambda x: x["fit"])
    c.require([x["fit"] for x in files] == list(c.FITS), "Missing fit")
    for ref in files: c.verify_hash(ref)
    old.save_new(OUTPUT/f"{kind}_lock.json", {"manifest": binding, "files": files, "runtime": runtime,
                                           "elapsed_seconds": time.perf_counter()-start})
    return files


def seal_models(binding, fits, audits):
    models = {}
    for fit, fr, ar in zip(c.FITS, fits, audits, strict=True):
        result, checked = c.verified(fr), c.verified(ar)
        c.require(result["manifest"] == checked["manifest"] == binding and result["fit"] == checked["fit"] == fit,
                  "Fit identity differs")
        c.require(checked["passed"] and checked["complete_fit_bitwise_identical"] and checked["value_head_frozen"]
                  and checked["all_training_metrics_independently_reproduced"]
                  and checked["result"] == {k: fr[k] for k in ("path", "sha256")}, "Fit audit failed")
        c.require(result["selected_update"] == result["updates"] == c.CONFIG["updates"] and result["final_always_selected"],
                  "Final checkpoint rule violated")
        c.require(checked["final_always_selected"] and checked["selected_update"] == result["selected_update"]
                  and checked["selected_training"] == result["selected_training"]
                  and checked["batch_sequence_sha256"] == result["batch_sequence_sha256"]
                  and checked["row_exposures"] == result["row_exposures"]
                  and checked["per_row_exposures"] == result["per_row_exposures"], "Fit audit accounting differs")
        c.require(checked["selected_model"] == result["selected_model"], "Audit selected model differs")
        c.verify_hash(result["selected_model"])
        models[f"focused_{fit}"] = {**result["selected_model"], "source_fit": fit, "selected_update": c.CONFIG["updates"]}
    old.save_new(OUTPUT/"models_lock.json", {"manifest": binding, "models": models, "fits": fits, "audits": audits,
                 "fixed_final": True, "all_fit_audits_pass": True, "time": old.now()})


def finish(binding, result_ref, audit_ref, fits, fit_audits):
    from . import assessment
    c.require(validate_freeze() == binding, "Final freeze changed")
    result, checked = c.verified(result_ref), c.verified(audit_ref)
    c.require(result["manifest"] == checked["manifest"] == binding and checked["passed"], "Assessment audit failed")
    c.require(checked["result"] == {k: result_ref[k] for k in ("path", "sha256")}, "Assessment linkage differs")
    training = {str(ref["fit"]): c.verified(ref) for ref in fits}
    final = {"manifest": binding, "time": old.now(), "classification": "focused_fit_diagnostic_only",
             "all_prescribed_audits_pass": True, "assessment": result_ref, "assessment_audit": audit_ref,
             "statistics": result["statistics"], "training": training,
             "models": c.artifact(OUTPUT/"models_lock.json"), "data": c.artifact(OUTPUT/"data_lock.json"),
             "new_simulations": 0, "automatic_successor": False,
             "scope": "Focused fitting and reused-development conditional utility; no fresh-state or game-strength claim"}
    old.save_new(OUTPUT/"result.json", final)
    old.save_new(OUTPUT/"final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "result": c.artifact(OUTPUT/"result.json"), "assessment": audit_ref, "fits": fit_audits,
                 "training_reproduced": True, "historical_simulations_replayed": False})
    lines = ["# Focused teacher fitting", "", "Three fixed initializers trained on the same168 original TRAIN positions only, for1784 updates each. Final checkpoints selected in advance. All new fits independently reproduced; all saved-score decisions and statistics independently verified.", "",
             "The168 VALIDATION positions were excluded from optimization and checkpoint selection. Both partitions and assessment outcomes are reused development evidence. This is a learning diagnosis, not fresh game strength, model promotion or a unique causal test of historical interference.", "",
             "No new simulations or source games. Historical simulation audits inherited after authentication. Short supports retained; no assessment outcome used for training, stopping or checkpoint selection.", "",
             "## Conditional score results", "", "```json", json.dumps(result["statistics"], indent=2, sort_keys=True), "```", "",
             "## Training result records", ""]
    for fit, record in training.items():
        lines += [f"Fit {fit}: final update {record['selected_update']}; training report {record['selected_training']['path']}.", ""]
    with (OUTPUT/"report.md").open("x", encoding="utf-8", newline="\n") as handle: handle.write("\n".join(lines))
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "all_prescribed_audits_pass": True, "classification": final["classification"], "automatic_successor": False,
                 "result": c.artifact(OUTPUT/"result.json"), "audit": c.artifact(OUTPUT/"final_audit.json"),
                 "report": c.artifact(OUTPUT/"report.md")})
    event("complete", classification=final["classification"])


def run():
    global REPORTER
    from . import assessment
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt", folder/"stderr.txt")
    torch.set_num_threads(c.CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(), "One-shot launch; preserve all evidence")
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
                seal_models(binding, fits, audits)
                event("assessment")
                result = assessment.assess(binding, OUTPUT)
                event("assessment_audit")
                checked = assessment.audit(binding, result, OUTPUT)
                finish(binding, result, checked, fits, audits)
        except c.BudgetExceeded as exc:
            old.save_new(OUTPUT/"operational_hold.json", {"manifest": binding, "time": old.now(), "reason": str(exc)})
            event("operational_hold", reason=str(exc))
        except BaseException as exc:
            abort(exc)
            old.save_new(OUTPUT/"failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure="failure.json"); raise


def launch():
    binding = validate_freeze()
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder/"launcher.lock"):
        c.require(not (folder/"process.json").exists() and not (OUTPUT/"launch.json").exists(), "Already launched")
        process = c.windows.detached_launch([sys.executable, "-u", ROOT/"run_focused_teacher_fit_128.py", "run"],
                                           folder/"stdout.txt", folder/"stderr.txt", ROOT)
        old.save_new(folder/"process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic()+120
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
