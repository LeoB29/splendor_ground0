"""Resume the authorized study with one active-time boundary in every process.

Scientific files keep their original binding. A separate recovery manifest pins
the unchanged stopped run and records every newly dispatched stage. Original
code, labels, launch, abort and stage receipts are never replaced.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from tools.teacher_precision_512 import common as c, experiment as parent, learning, training_source
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from . import diagnostics, inventory

ROOT, old = c.ROOT, c.old
ORIGINAL_STUDY = c.OUTPUT
RECOVERY_OUTPUT = ROOT/"outputs/teacher_precision_512_recovery_01"
ORIGINAL_BINDING = "d2321f90c0d92ec0834a3aa1b4e9ace9834a452428d233c2e41b3f1f0c257b0c"
STUDY, OUTPUT, BINDING = ORIGINAL_STUDY, RECOVERY_OUTPUT, ORIGINAL_BINDING
MAXIMUM_ACTIVE_SECONDS = 48*3600.
_ORIGINAL_VALIDATE = parent.validate_freeze
REPORTER = None
CHECK_TARGETS = [f"tools/teacher_precision_512_recovery_01/test_{name}.py" for name in
                 ("diagnostics", "inventory", "recovery", "worker")]+[
    "tools/teacher_precision_512/test_learning.py", "tools/teacher_precision_512/test_integration.py"]
EXTRAS = ["run_teacher_precision_512_recovery_01.py", "docs/engineering/teacher_precision_512_recovery_01.md"]


def assert_clean(base=None):
    base = Path(OUTPUT if base is None else base)
    names = ("launch.json", "runtime/process.json", "abort.json", "failure.json",
             "operational_hold.json", "final_status.json", "stages")
    present = [name for name in names if (base/name).exists()]
    c.require(not present, "Recovery runtime already exists; preserve and inspect: "+", ".join(present))


def identity():
    original = c.verified(old.read_json(ORIGINAL_STUDY/"freeze.json")["manifest"])
    inherited = parent.identity()
    c.require(inherited == original["identity"], "Original frozen implementation changed")
    dependencies = dict(inherited["dependencies"])
    paths = list((ROOT/"tools/teacher_precision_512_recovery_01").glob("*.py"))+[ROOT/p for p in EXTRAS]
    dependencies.update({p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths})
    return {"dependencies": dict(sorted(dependencies.items())), "original_identity": inherited,
            "original_binding": ORIGINAL_BINDING, "maximum_active_seconds": MAXIMUM_ACTIVE_SECONDS,
            "scientific_payload_unchanged": True, "label_recollection": False,
            "checks": CHECK_TARGETS, "python": sys.version,
            "windowless": c.windows._identity(c.windows._windowless_executable())}


def _launch_receipt():
    c.require((OUTPUT/"launch_seal.json").exists(), "Recovery launch completion seal missing")
    seal = old.read_json(OUTPUT/"launch_seal.json")
    c.require(seal["study_binding"] == BINDING
              and c.verify_hash(seal["launch"]).resolve() == (OUTPUT/"launch.json").resolve(),
              "Recovery launch seal changed")
    launch = c.verified(seal["launch"])
    c.require(launch["study_binding"] == BINDING, "Recovery scientific binding changed")
    return launch


def active_seconds(base=None):
    base = Path(STUDY if base is None else base)
    c.require(base.resolve() == STUDY.resolve(), "Recovery budget received a different study base")
    launch = _launch_receipt()
    prior, epoch = launch["prior_active_seconds"], launch["epoch_seconds"]
    c.require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (prior, epoch)),
              "Invalid recovery active-time accounting")
    return prior+max(0., time.time()-epoch)


def budget_boundary(base=None):
    base = Path(STUDY if base is None else base)
    c.require(base.resolve() == STUDY.resolve(), "Recovery budget received a different study base")
    if (OUTPUT/"abort.json").exists():
        raise c.BudgetExceeded("Recovery cooperative abort; preserve evidence")
    c.require((OUTPUT/"launch.json").exists(), "Recovery launch receipt missing")
    launch = _launch_receipt()
    c.require(launch["study_binding"] == BINDING, "Recovery scientific binding changed")
    # Resolve only the precisely authenticated old abort. A changed old abort
    # or any new recovery abort remains a stop, never a blanket bypass.
    c.require(c.verify_hash(launch["resolved_abort"]).resolve() == (STUDY/"abort.json").resolve(),
              "Resolved original abort changed")
    if active_seconds(base) > MAXIMUM_ACTIVE_SECONDS:
        raise c.BudgetExceeded("Forty-eight-hour cumulative active execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < c.CONFIG["disk_reserve_bytes"]:
        raise c.BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")


def event(stage, **values):
    global REPORTER
    if REPORTER is None or REPORTER.folder != OUTPUT:
        REPORTER = StatusReporter(OUTPUT, {"study_binding": BINDING, "recovery": "teacher_precision_512_recovery_01"})
    return REPORTER.event(stage, **values)


def abort(exc, base=None):
    try:
        old.save_new(OUTPUT/"abort.json", {"study_binding": BINDING, "time": old.now(),
                     "pid": os.getpid(), "error": str(exc), "traceback": traceback.format_exc()})
    except FileExistsError:
        pass


def install_runtime(study_base, recovery_base, original_binding, *, fixture=False):
    """Install the owned adapters in this process, including spawned workers."""
    global STUDY, OUTPUT, BINDING, REPORTER
    study_base, recovery_base = Path(study_base).resolve(), Path(recovery_base).resolve()
    if fixture:
        c.require(not study_base.is_relative_to(ORIGINAL_STUDY.resolve())
                  and not recovery_base.is_relative_to(RECOVERY_OUTPUT.resolve()), "Fixture touches production")
    else:
        c.require(study_base == ORIGINAL_STUDY.resolve() and recovery_base == RECOVERY_OUTPUT.resolve()
                  and original_binding == ORIGINAL_BINDING, "Production recovery destination changed")
    STUDY, OUTPUT, BINDING, REPORTER = study_base, recovery_base, original_binding, None
    c.budget_boundary, c.active_seconds = budget_boundary, active_seconds
    parent.OUTPUT = STUDY
    parent.budget_boundary, parent.validate_freeze = budget_boundary, validate_freeze
    parent.worker_initialize, parent.run_pool = worker_initialize, run_pool
    parent.abort, parent.event = abort, event
    diagnostics.install()
    return {"study_binding": BINDING, "budget_module": budget_boundary.__module__,
            "diagnostics_module": learning.training_report.__module__, "maximum_active_seconds": MAXIMUM_ACTIVE_SECONDS}


def worker_initialize(stage, kind, binding, study_base, recovery_base):
    fixture = Path(study_base).resolve() != ORIGINAL_STUDY.resolve()
    receipt = install_runtime(study_base, recovery_base, binding, fixture=fixture)
    budget_boundary(study_base)
    folder = OUTPUT/"workers"/f"{stage}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    health = c.windows.assert_windowless_worker(folder/f"{os.getpid()}_stdout.txt", folder/f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    if stage.startswith("source_"):
        parent.source.initialize_models(study_base)
    elif stage in ("probe", "main"):
        parent.engine.initialize_models(study_base)
    c.require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "Worker arithmetic contract")
    old.save_new(folder/f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, "kind": kind, **health})
    old.save_new(OUTPUT/"workers"/f"{os.getpid()}_recovery.json", {**receipt, "stage": stage,
                 "kind": kind, "pid": os.getpid(), "active_seconds": active_seconds(study_base),
                 "windowless": c.windows.assert_windowless_worker()})


def validate_freeze(full_sources=False):
    freeze = old.read_json(OUTPUT/"freeze.json")
    manifest = c.verified(freeze["manifest"])
    current = identity()
    c.require(freeze["binding"] == old.digest(manifest) and manifest["identity"] == current
              and manifest["owner_authorized"] and manifest["study_binding"] == ORIGINAL_BINDING,
              "Recovery freeze/code changed")
    checks = c.verified(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == current
              and checks["targets"] == CHECK_TARGETS, "Passing frozen recovery preflight required")
    for ref in checks["logs"]: c.verify_hash(ref)
    c.require(_ORIGINAL_VALIDATE(full_sources=full_sources) == ORIGINAL_BINDING, "Original freeze differs")
    inventory.validate_inventory(manifest["stopped_parent"], deep=full_sources)
    return ORIGINAL_BINDING


def run_pool(stage, kind, arguments, binding, upstream):
    """Unchanged bounded dispatch, new launch receipts, original result paths."""
    c.require(validate_freeze() == binding, "Recovery freeze changed before stage")
    budget_boundary(STUDY)
    c.require(not stage.startswith("labels_"), "Completed labels must never be recollected or replayed")
    started = time.perf_counter()
    launch_path = OUTPUT/"stages"/f"{stage}_{kind}_launch.json"
    expected_indices = list(range(len(arguments)))
    old.save_new(launch_path, {"manifest": binding, "stage": stage, "kind": kind, "time": old.now(),
                 "expected_indices": expected_indices, "arguments_sha256": old.digest(arguments),
                 "dependencies": [c.artifact(p) for p in [*upstream, OUTPUT/"freeze.json"]]})
    runtime = c.windows.configure_spawn()
    event(f"{stage}_{kind}", completed=0, total=len(arguments))
    workers = c.CONFIG["fit_workers"] if stage == "learning" else c.CONFIG["workers"]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                initializer=worker_initialize, initargs=(stage, kind, binding, STUDY, OUTPUT)) as pool:
            try:
                files = parent.dispatch(pool, arguments,
                    partial(parent.run_job, stage, kind, binding=binding, base=STUDY), workers,
                    lambda n:event(f"{stage}_{kind}", completed=n, total=len(arguments)), lambda:budget_boundary(STUDY))
            except BaseException as exc:
                abort(exc); raise
    except BaseException as exc:
        abort(exc); raise
    files.sort(key=lambda item:item["index"])
    c.require([r["index"] for r in files] == expected_indices, "Missing/duplicate recovery job")
    for ref in files: c.verify_hash(ref)
    old.save_new(STUDY/f"{stage}_{kind}_lock.json", {"manifest": binding, "launch": c.artifact(launch_path),
                 "files": files, "elapsed_seconds": time.perf_counter()-started, "runtime": runtime})
    return files


def build_learning(binding):
    event("learning_reuse_verified_targets", label_roots=720, new_label_simulations=0)
    training_source.validate(binding, STUDY, authenticate_parent=False)
    training_source.validate_targets(binding, STUDY)
    targets = c.artifact(STUDY/"target_lock.json")
    upstream = [STUDY/"freeze.json", STUDY/"data_lock.json", STUDY/"target_lock.json", STUDY/"labels_complete.json"]
    fits = run_pool("learning", "fits", list(c.FITS), binding, upstream)
    audits = run_pool("learning", "fit_audits", list(zip(c.FITS, fits)), binding,
                      [*upstream, STUDY/"learning_fits_lock.json"])
    models = learning.models_lock(fits, audits, binding, STUDY)
    c.verify_hash(models); learning.validate_models(binding, STUDY)
    old.save_new(STUDY/"learning_complete.json", {"manifest": binding, "time": old.now(),
        "data": c.artifact(STUDY/"data_lock.json"), "models": models, "targets": targets,
        "labels": c.artifact(STUDY/"labels_complete.json"), "fits": c.artifact(STUDY/"learning_fits_lock.json"),
        "fit_audits": c.artifact(STUDY/"learning_fit_audits_lock.json"), "recovery": c.artifact(OUTPUT/"freeze.json")})


def pipeline(binding):
    build_learning(binding)
    budget_boundary(STUDY)
    parent.build_sources(binding)
    budget_boundary(STUDY)
    parent.build_stage("probe", binding)
    c.require(parent.operational_decision(binding), "Passing fresh evaluation forecast required")
    parent.build_stage("main", binding)
    budget_boundary(STUDY)
    parent.finish(binding)
    old.save_new(OUTPUT/"final_status.json", {"time": old.now(), "complete": True, "study_binding": binding,
        "recovery": c.artifact(OUTPUT/"freeze.json"), "study_final_status": c.artifact(STUDY/"final_status.json"),
        "new_label_simulations": 0, "active_seconds": active_seconds(STUDY), "automatic_successor": False})


def check(folder):
    assert_clean()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    before, began = identity(), time.perf_counter()
    fixture = ROOT/"outputs/teacher_precision_512_recovery_01_fixtures"/folder.name
    c.require(not fixture.exists(), "Preserve prior fixtures")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    args = ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture)]
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main("+repr(args)+"))"
    with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, timeout=1200, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity() == before, "Code changed during recovery preflight"); assert_clean()
    old.save_new(folder/"checks.json", {"identity": before, "targets": CHECK_TARGETS, "exit_code": result.returncode,
                 "seconds": time.perf_counter()-began, "logs": [c.artifact(folder/n) for n in ("stdout.txt", "stderr.txt")]})
    print({"checks_exit": result.returncode, "seconds": time.perf_counter()-began}, flush=True)
    return result.returncode


def prepare(checks_path):
    assert_clean(); c.require(not (OUTPUT/"freeze.json").exists(), "Already frozen; preserve evidence")
    checks, current = old.read_json(checks_path), identity()
    c.require(checks["exit_code"] == 0 and checks["identity"] == current and checks["targets"] == CHECK_TARGETS,
              "Current passing recovery preflight required")
    for ref in checks["logs"]: c.verify_hash(ref)
    evidence = inventory.stopped_parent()
    c.require(evidence["original_binding"] == ORIGINAL_BINDING, "Wrong stopped study")
    manifest = {"identity": current, "checks": c.artifact(checks_path), "stopped_parent": evidence,
        "study_binding": ORIGINAL_BINDING, "owner_authorized": True,
        "authorization": "Owner2026-10-12: fix it, resume, and make sure it doesnt happen again. Repair inherited timer routing, test aged supervisor/hidden workers, reuse all720 labels/targets, continue exact approved training and fresh evaluation. No new science, monitoring or successor."}
    c.require(identity() == current, "Code changed during preparation")
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"freeze.json", {"manifest": c.artifact(OUTPUT/"manifest.json"), "binding": old.digest(manifest),
                 "study_binding": ORIGINAL_BINDING, "time": old.now(), "owner_authorized": True})
    c.require(validate_freeze(full_sources=True) == ORIGINAL_BINDING, "Recovery preparation failed")
    print({"prepared": True, "recovery_binding": old.digest(manifest), "reused_labels": 720,
           "prior_active_hours": evidence["prior_active_seconds"]/3600}, flush=True)


def run():
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt", folder/"stderr.txt")
    with old.process_lock(STUDY/"run.lock"), old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(), "One-shot recovery; preserve existing evidence")
        binding = validate_freeze(full_sources=True)
        manifest = c.verified(old.read_json(OUTPUT/"freeze.json")["manifest"])
        inventory.assert_stopped(manifest["stopped_parent"])
        install_runtime(STUDY, OUTPUT, binding)
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        try:
            with parent.parent.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT/"launch.json", {"study_binding": binding, "time": old.now(),
                    "epoch_seconds": time.time(), "pid": os.getpid(), "sleep_inhibition": sleep,
                    "prior_active_seconds": manifest["stopped_parent"]["prior_active_seconds"],
                    "resolved_abort": c.artifact(STUDY/"abort.json"), "recovery": c.artifact(OUTPUT/"freeze.json"),
                    "windowless": c.windows.assert_windowless_worker(), "owner_authorized": True})
                receipt = old.read_json(OUTPUT/"launch.json")
                c.require(receipt["prior_active_seconds"] == manifest["stopped_parent"]["prior_active_seconds"]
                          and receipt["recovery"] == c.artifact(OUTPUT/"freeze.json"), "Recovery launch accounting differs")
                old.save_new(OUTPUT/"launch_seal.json", {"study_binding": binding,
                             "launch": c.artifact(OUTPUT/"launch.json")})
                pipeline(binding)
        except BaseException as exc:
            abort(exc)
            name = "operational_hold.json" if isinstance(exc, c.BudgetExceeded) else "failure.json"
            old.save_new(OUTPUT/name, {"study_binding": binding, "time": old.now(), "reason": str(exc),
                         "traceback": traceback.format_exc(), "active_seconds": active_seconds(STUDY)})
            event("operational_hold" if isinstance(exc, c.BudgetExceeded) else "failed", reason=str(exc))
            if not isinstance(exc, c.BudgetExceeded): raise


def launch():
    assert_clean(); validate_freeze(full_sources=True)
    manifest = c.verified(old.read_json(OUTPUT/"freeze.json")["manifest"])
    inventory.assert_stopped(manifest["stopped_parent"])
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder/"launcher.lock"):
        assert_clean()
        proc = c.windows.detached_launch([sys.executable, "-u", ROOT/"run_teacher_precision_512_recovery_01.py", "run"],
                                        folder/"stdout.txt", folder/"stderr.txt", ROOT)
        old.save_new(folder/"process.json", {**proc, "study_binding": ORIGINAL_BINDING, "time": old.now()})
        deadline = time.monotonic()+600
        while not (OUTPUT/"launch_seal.json").exists():
            c.require(not (OUTPUT/"failure.json").exists(), "Recovery startup failed; inspect saved evidence")
            c.require(not (folder/"startup_failure.json").exists(), "Recovery startup failed before acknowledgement")
            c.require(time.monotonic() < deadline, "No acknowledgement; inspect before any retry")
            time.sleep(.05)
        receipt = _launch_receipt()
        c.require(receipt["pid"] == proc["pid"] and receipt["study_binding"] == ORIGINAL_BINDING, "Recovery supervisor mismatch")
        old.save_new(folder/"acknowledged.json", {"time": old.now(), "pid": proc["pid"], "launch": c.artifact(OUTPUT/"launch.json")})
        print({"launched": True, "pid": proc["pid"], "study_binding": ORIGINAL_BINDING}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command in ("check", "prepare"): c.require(args.checks is not None, "Supply checks path")
    if args.command == "check": return check(args.checks)
    if args.command == "prepare": prepare(args.checks)
    elif args.command == "launch": launch()
    else:
        try: run()
        except BaseException:
            old.save_new(OUTPUT/"runtime/startup_failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise
    return 0
