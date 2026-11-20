"""Frozen one-shot evaluation of existing focused-training checkpoints."""
from __future__ import annotations
import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from . import common as c
from tools.proposal_iteration_128 import experiment as ancestor
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter

ROOT, OUTPUT, old = c.ROOT, c.OUTPUT, c.old
CHECK_TARGETS = [f"tests/test_saved_teacher_curve_128_{name}.py" for name in
                 ("source", "statistics", "assessment", "experiment")]
EXTRAS = ["run_saved_teacher_curve_128.py", "docs/engineering/saved_teacher_curve_128.md",
          "docs/engineering/verification_policy.md", *CHECK_TARGETS]
REPORTER = None


def identity():
    c.require(sys.flags.optimize == 0, "Unoptimized Python required")
    dependencies = dict(old.read_json(c.PARENT / "manifest.json")["identity"]["dependencies"])
    for path, digest in dependencies.items():
        c.require(old.file_hash(ROOT/path) == digest, "Frozen parent dependency changed: " + path)
    for path in [*(ROOT/"tools/saved_teacher_curve_128").glob("*.py"), *(ROOT/p for p in EXTRAS)]:
        dependencies[path.relative_to(ROOT).as_posix()] = old.file_hash(path)
    return {"config": c.CONFIG, "checks": CHECK_TARGETS, "dependencies": dict(sorted(dependencies.items())),
            "python": sys.version, "optimize": sys.flags.optimize, "numpy": np.__version__,
            "torch": str(torch.__version__), "matplotlib": version("matplotlib"),
            "windowless": c.windows._identity(c.windows._windowless_executable())}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before, start = identity(), time.perf_counter()
    fixture = ROOT / "outputs/saved_teacher_curve_128_fixtures" / folder.name
    c.require(not fixture.exists(), "Preserve previous fixture evidence")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main(" + repr(
        ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture)]) + "))"
    with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
        process = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, timeout=900,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity() == before, "Implementation changed during official checks")
    old.save_new(folder/"checks.json", {"identity": before, "targets": CHECK_TARGETS,
                 "exit_code": process.returncode, "seconds": time.perf_counter()-start,
                 "logs": [c.artifact(folder/name) for name in ("stdout.txt", "stderr.txt")]})
    return process.returncode


def prepare(checks_path):
    from . import source
    c.require(not (OUTPUT/"manifest.json").exists(), "Already frozen; preserve evidence")
    checks = old.read_json(checks_path)
    current = identity()
    c.require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS
              and checks["identity"] == current, "Current full passing preflight required")
    for ref in checks["logs"]: c.verify_hash(ref)
    parent = source.closed_parent()
    c.require(identity() == current, "Implementation changed during preparation")
    manifest = {"identity": current, "checks": c.artifact(checks_path), "source_parent": parent,
                "owner_authorized": True, "authorization":
                "Owner approved all 13 saved-checkpoint learning curves on 2026-10-08; existing positions and outcomes only, no training, games, simulation, checkpoint selection, promotion or successor."}
    old.save_new(OUTPUT/"manifest.json", manifest)
    binding = old.digest(manifest)
    old.save_new(OUTPUT/"freeze.json", {"manifest": c.artifact(OUTPUT/"manifest.json"), "binding": binding,
                                      "owner_authorized": True, "time": old.now()})
    c.require(validate_freeze()[0] == binding, "Post-freeze validation failed")
    print({"prepared": True, "binding": binding, "checkpoints": list(c.CHECKPOINTS)}, flush=True)


def validate_freeze():
    from . import source
    freeze = old.read_json(OUTPUT/"freeze.json")
    manifest = c.verified(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True and manifest["owner_authorized"] is True
              and old.digest(manifest) == freeze["binding"] and manifest["identity"] == identity(),
              "Frozen identities changed")
    checks = c.verified(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"]
              and checks["targets"] == CHECK_TARGETS, "Preflight identity changed")
    for ref in checks["logs"]: c.verify_hash(ref)
    source.validate(manifest["source_parent"])
    return freeze["binding"], manifest["source_parent"]


def event(stage, **values):
    global REPORTER
    if REPORTER is None: REPORTER = StatusReporter(OUTPUT)
    return REPORTER.event(stage, **values)


def abort(exc, base=OUTPUT):
    try:
        old.save_new(Path(base)/"abort.json", {"time": old.now(), "pid": os.getpid(), "error": str(exc)})
    except FileExistsError:
        pass


def finish(binding, parent, result_ref, audit_ref):
    from . import reporting
    c.require(validate_freeze()[0] == binding, "Final identity changed")
    result, audit = c.verified(result_ref), c.verified(audit_ref)
    c.require(result["manifest"] == audit["manifest"] == binding and audit["passed"] is True,
              "All new assessments must pass independent verification")
    c.require(audit["result"] == {k: result_ref[k] for k in ("path", "sha256")}, "Audit result differs")
    c.require(audit["checkpoints"] == list(c.CHECKPOINTS)
              and audit["states_per_checkpoint"] == result["states_per_checkpoint"] == 336
              and set(result["curves"]) == {str(update) for update in c.CHECKPOINTS},
              "Incomplete audited checkpoint population")
    for flag in ("new_inference_independently_reproduced", "lawful_inputs_and_private_invariance_pass",
                 "old_choices_independently_reconstructed", "old_label_metrics_independently_reproduced",
                 "all_new_arithmetic_pass", "endpoint_zero_reproduced", "endpoint_final_reproduced",
                 "historical_training_and_simulation_audits_inherited"):
        c.require(audit[flag] is True, "Required assessment check did not pass: " + flag)
    c.require(audit["new_training"] is False and audit["new_simulations"] == audit["new_trajectory_replays"] == 0,
              "Unexpected new training or simulations")
    c.budget_boundary(OUTPUT)
    final = {"manifest": binding, "time": old.now(), "classification": "saved_curve_diagnostic_only",
             "assessment": result_ref, "assessment_audit": audit_ref, "statistics": result["statistics"],
             "all_prescribed_audits_pass": True, "new_training": False, "new_simulations": 0,
             "new_source_games": 0, "checkpoint_selection": False, "promotion": False,
             "automatic_successor": False, "scope": "Reused-development learning trajectory; no game-strength claim"}
    old.save_new(OUTPUT/"result.json", final)
    old.save_new(OUTPUT/"final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "result": c.artifact(OUTPUT/"result.json"), "assessment": audit_ref,
                 "prior_training_audits_inherited": True, "training_replayed": False,
                 "historical_simulations_replayed": False})
    report_refs = reporting.write_report(binding, result, parent, OUTPUT)
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "all_prescribed_audits_pass": True, "classification": final["classification"],
                 "automatic_successor": False, "result": c.artifact(OUTPUT/"result.json"),
                 "audit": c.artifact(OUTPUT/"final_audit.json"), **report_refs})
    event("complete", classification=final["classification"])


def run():
    global REPORTER
    from . import assessment
    folder = OUTPUT/"runtime"
    folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt", folder/"stderr.txt")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(), "One-shot run; preserve evidence")
        try:
            binding, parent = validate_freeze()
            REPORTER = StatusReporter(OUTPUT, {"manifest": binding})
            with ancestor.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT/"launch.json", {"manifest": binding, "time": old.now(), "epoch_seconds": time.time(),
                             "pid": os.getpid(), "owner_authorized": True, "sleep_inhibition": sleep,
                             "torch_threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
                             "windowless": c.windows.assert_windowless_worker()})
                event("assessment", checkpoints=len(c.CHECKPOINTS), states=c.CONFIG["states"])
                result_ref = assessment.assess(binding, parent, OUTPUT, progress=event)
                event("assessment_audit")
                audit_ref = assessment.audit(binding, parent, result_ref, OUTPUT, progress=event)
                event("finalizing")
                finish(binding, parent, result_ref, audit_ref)
        except c.BudgetExceeded as exc:
            old.save_new(OUTPUT/"operational_hold.json", {"time": old.now(), "reason": str(exc)})
            event("operational_hold", reason=str(exc))
        except BaseException as exc:
            abort(exc)
            old.save_new(OUTPUT/"failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure="failure.json")
            raise


def launch():
    binding, _ = validate_freeze()
    folder = OUTPUT/"runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder/"launcher.lock"):
        c.require(not (folder/"process.json").exists() and not (OUTPUT/"launch.json").exists(), "Already launched")
        process = c.windows.detached_launch([sys.executable, "-u", ROOT/"run_saved_teacher_curve_128.py", "run"],
                                           folder/"stdout.txt", folder/"stderr.txt", ROOT)
        old.save_new(folder/"process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic()+90
        while not (OUTPUT/"launch.json").exists():
            c.require(not (OUTPUT/"failure.json").exists(), "Supervisor failed; inspect preserved failure")
            c.require(time.monotonic() < deadline, "No acknowledgement; inspect before retry")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT/"launch.json")
        c.require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding, "Supervisor differs")
        old.save_new(folder/"acknowledged.json", {"pid": process["pid"], "time": old.now(),
                                                "launch": c.artifact(OUTPUT/"launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command in ("check", "prepare"): c.require(args.checks is not None, "Check path required")
    if args.command == "check": return check(args.checks)
    if args.command == "prepare": prepare(args.checks)
    elif args.command == "launch": launch()
    else: run()
    return 0
