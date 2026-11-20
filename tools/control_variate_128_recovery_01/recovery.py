"""Reuse audited timing and continue unchanged science with a three-hour cap."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
from tools.control_variate_128 import common as c, experiment as parent, source, statistics

ROOT, old = c.ROOT, c.old
PARENT = c.OUTPUT
PARENT_BINDING = "b359bc5c4ee1632fc72188e325db98db7314c4b4a32c42ae375eb5f98464dd3e"
OUTPUT = ROOT/"outputs/control_variate_128_recovery_01"
MAXIMUM_RUNTIME_SECONDS = 10800.
_ORIGINAL_WORKER_INITIALIZE = parent.worker_initialize
CHECK_TARGETS = ["tools/control_variate_128_recovery_01/test_recovery.py",
                 "tools/control_variate_128_recovery_01/test_worker.py"]
EXTRAS = ["run_control_variate_128_recovery_01.py", "docs/engineering/control_variate_128_recovery_01.md"]
RUNTIME_MARKERS = ("abort.json", "failure.json", "operational_hold.json", "launch.json",
    "runtime/process.json", "workers", "train", "validation", "train_launch.json",
    "selection_lock.json", "result.json")
TIMING_FLAGS = ("all_endpoint_features_verified", "all_endpoint_values_verified",
    "sampled_prefixes_independently_replayed", "source_world_coupling_verified",
    "all_full_outcomes_equal_saved", "matched_timing_arms_verified")


def assert_clean(base=None):
    base = Path(OUTPUT if base is None else base)
    found = [name for name in RUNTIME_MARKERS if (base/name).exists()]
    c.require(not found, "Runtime evidence already exists; preserve and inspect: "+", ".join(found))


def identity():
    original = c.verified(old.read_json(PARENT/"freeze.json")["manifest"])
    inherited = parent.identity()
    c.require(inherited == original["identity"], "Original frozen implementation changed")
    dependencies = dict(inherited["dependencies"])
    paths = list((ROOT/"tools/control_variate_128_recovery_01").glob("*.py"))+[ROOT/p for p in EXTRAS]
    dependencies.update({p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths})
    return {"dependencies": dict(sorted(dependencies.items())), "original_identity": inherited,
        "maximum_active_seconds_including_parent": MAXIMUM_RUNTIME_SECONDS,
        "parent_binding": PARENT_BINDING, "reuse_timing_without_resimulation": True,
        "scientific_config": c.CONFIG, "checks": CHECK_TARGETS,
        "python": sys.version, "windowless": c.windows._identity(c.windows._windowless_executable())}


def parent_evidence():
    """Authenticate a completed timing-only operational hold; do not read VAL outcomes."""
    freeze = old.read_json(PARENT/"freeze.json")
    manifest = c.verified(freeze["manifest"])
    c.require(freeze["binding"] == old.digest(manifest) == PARENT_BINDING
              and freeze["owner_authorized"] is True and manifest["identity"] == parent.identity(),
              "Original frozen study changed")
    checked = c.verified(manifest["checks"])
    c.require(checked["exit_code"] == 0 and checked["identity"] == manifest["identity"], "Original preflight changed")
    for ref in checked["logs"]: c.verify_hash(ref)
    for ref in manifest["source_parent"]["evidence"]: c.verify_hash(ref)
    source.validate(PARENT_BINDING, PARENT, authenticate_parent=False)
    for name in ("train", "validation", "train_launch.json", "train_lock.json", "selection_lock.json",
                 "selection_audit.json", "assessment_result.json", "result.json", "failure.json", "abort.json"):
        c.require(not (PARENT/name).exists(), "Original run advanced or failed; inspect before continuation")
    refs = {name: c.artifact(PARENT/(name+".json")) for name in
            ("freeze", "data_lock", "timing_lock", "timing_audits_lock", "forecast", "operational_hold", "launch")}
    hold, forecast, launch = (c.verified(refs[k]) for k in ("operational_hold", "forecast", "launch"))
    c.require(hold["manifest"] == forecast["manifest"] == launch["manifest"] == PARENT_BINDING
              and hold["reason"] == "Timing forecast exceeds the fixed two-hour execution ceiling"
              and forecast["outcome_blind"] is True and forecast["within_runtime_ceiling"] is False
              and forecast["disk_reserve_pass"] is True, "Parent was not the expected time-only hold")
    previous = datetime.fromisoformat(hold["time"]).timestamp()-launch["epoch_seconds"]
    c.require(previous > 0 and previous >= forecast["elapsed_seconds"], "Invalid prior active time")
    timing, audited = c.verified(refs["timing_lock"]), c.verified(refs["timing_audits_lock"])
    expected = [s["index"] for s in source.timing_specs(PARENT_BINDING, PARENT)]
    c.require(len(expected) == 18 and timing["manifest"] == audited["manifest"] == PARENT_BINDING
              and [r["index"] for r in timing["files"]] == [r["index"] for r in audited["files"]] == expected,
              "Timing inventory differs")
    c.require(forecast["timing"] == timing["files"], "Forecast timing linkage differs")
    for record_ref, audit_ref in zip(timing["files"], audited["files"], strict=True):
        record, audit = c.verified(record_ref), c.verified(audit_ref)
        c.require(record["manifest"] == audit["manifest"] == PARENT_BINDING and record["complete"] is True
                  and record["index"] == audit["index"] == record_ref["index"] == audit_ref["index"]
                  and audit["result"] == record_ref and audit["passed"] is True
                  and all(audit.get(flag) is True for flag in TIMING_FLAGS), "Incomplete inherited timing audit")
    return {**refs, "manifest": freeze["manifest"], "checks": manifest["checks"],
        "previous_elapsed_seconds": previous, "timing_roots": 18,
        "main_collection_started": False, "validation_evaluated": False}


def transformed_record(record, binding, original_ref):
    return {**record, "manifest": binding, "reused_from": original_ref}


def transformed_audit(audit, binding, new_result, original_ref):
    return {**audit, "manifest": binding, "result": new_result, "reused_from": original_ref}


def import_timing(base, binding, evidence):
    base = Path(base)
    old_records, old_audits = (c.verified(evidence[k])["files"] for k in ("timing_lock", "timing_audits_lock"))
    records, audits = [], []
    for record_ref, audit_ref in zip(old_records, old_audits, strict=True):
        index = record_ref["index"]
        path = base/"timing"/f"state_{index:04d}.json"
        old.save_new(path, transformed_record(c.verified(record_ref), binding, record_ref))
        new_ref = c.artifact(path, index=index); records.append(new_ref)
        path = base/"timing"/"audits"/f"state_{index:04d}.json"
        old.save_new(path, transformed_audit(c.verified(audit_ref), binding, new_ref, audit_ref))
        audits.append(c.artifact(path, index=index))
    for name, files in (("timing_lock", records), ("timing_audits_lock", audits)):
        old.save_new(base/(name+".json"), {"manifest": binding, "files": files,
            "reused_from": evidence[name], "new_simulations": 0,
            "elapsed_seconds": c.verified(evidence[name])["elapsed_seconds"]})
    old.save_new(base/"timing_reuse.json", {"manifest": binding, "parent": evidence,
        "records": c.artifact(base/"timing_lock.json"), "audits": c.artifact(base/"timing_audits_lock.json"),
        "scientific_payload_unchanged": True, "new_simulations": 0})


def validate_reuse(base, binding, evidence):
    base = Path(base)
    receipt = old.read_json(base/"timing_reuse.json")
    c.require(receipt == {"manifest": binding, "parent": evidence,
        "records": c.artifact(base/"timing_lock.json"), "audits": c.artifact(base/"timing_audits_lock.json"),
        "scientific_payload_unchanged": True, "new_simulations": 0}, "Timing reuse receipt changed")
    original_records, original_audits = (c.verified(evidence[k])["files"] for k in ("timing_lock", "timing_audits_lock"))
    records, audits = (c.verified(receipt[k]) for k in ("records", "audits"))
    for name, lock in (("timing_lock", records), ("timing_audits_lock", audits)):
        c.require(lock["manifest"] == binding and lock["reused_from"] == evidence[name]
                  and lock["new_simulations"] == 0 and len(lock["files"]) == 18
                  and lock["elapsed_seconds"] == c.verified(evidence[name])["elapsed_seconds"], "Timing reuse lock changed")
    for prior, prior_audit, current, audit in zip(original_records, original_audits, records["files"], audits["files"], strict=True):
        c.require(prior["index"] == prior_audit["index"] == current["index"] == audit["index"], "Reuse index changed")
        c.require(c.verified(current) == transformed_record(c.verified(prior), binding, prior), "Reused timing payload changed")
        c.require(c.verified(audit) == transformed_audit(c.verified(prior_audit), binding, current, prior_audit), "Reused timing audit changed")
    return records["files"]


def budget_boundary(base=None):
    base = Path(OUTPUT if base is None else base)
    if (base/"abort.json").exists(): raise c.BudgetExceeded("Cooperative abort requested; preserve evidence")
    if not (base/"launch.json").exists():
        c.require(not base.resolve().is_relative_to(OUTPUT.resolve()), "Production launch receipt missing")
        return
    launch = old.read_json(base/"launch.json")
    previous = launch["previous_elapsed_seconds"]
    c.require(previous >= 0, "Invalid previous active runtime")
    if previous+time.time()-launch["epoch_seconds"] > MAXIMUM_RUNTIME_SECONDS:
        raise c.BudgetExceeded("Three-hour active execution ceiling reached; preserve evidence")
    if shutil.disk_usage(ROOT).free < c.CONFIG["disk_reserve_bytes"]:
        raise c.BudgetExceeded("Five-GiB disk reserve reached; preserve evidence")


def configure(base=None):
    """Install only orchestration/resource adapters, including inside spawned workers."""
    parent.OUTPUT = Path(OUTPUT if base is None else base)
    parent.validate_freeze = validate_freeze
    parent.worker_initialize = worker_initialize
    c.budget_boundary = budget_boundary


def worker_initialize(kind, binding, base):
    configure(base)
    _ORIGINAL_WORKER_INITIALIZE(kind, binding, base)


def check(folder):
    assert_clean()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    before, began = identity(), time.perf_counter()
    fixture = ROOT/"outputs/control_variate_128_recovery_01_fixtures"/folder.name
    c.require(not fixture.exists(), "Preserve prior fixtures")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    args = ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture)]
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main("+repr(args)+"))"
    with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
        proc = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, timeout=600, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity() == before, "Code changed during preflight"); assert_clean()
    old.save_new(folder/"checks.json", {"identity": before, "targets": CHECK_TARGETS, "exit_code": proc.returncode,
        "seconds": time.perf_counter()-began, "logs": [c.artifact(folder/n) for n in ("stdout.txt", "stderr.txt")]})
    print({"checks_exit": proc.returncode, "seconds": time.perf_counter()-began}, flush=True)
    return proc.returncode


def prepare(checks_path):
    assert_clean(); c.require(not (OUTPUT/"manifest.json").exists(), "Already frozen; preserve evidence")
    checked, current = old.read_json(checks_path), identity()
    c.require(checked["exit_code"] == 0 and checked["identity"] == current
              and checked["targets"] == CHECK_TARGETS, "Passing current preflight required")
    for ref in checked["logs"]: c.verify_hash(ref)
    evidence = parent_evidence()
    original = c.verified(evidence["manifest"])
    manifest = {"identity": current, "checks": c.artifact(checks_path), "source_parent": original["source_parent"],
        "held_parent": evidence, "owner_authorized": True,
        "authorization": "Owner2026-10-09 accepted three-hour continuation: sure let's do it, time doesn't matter much. Reuse audited timing; preserve identical science and frozen ancestors. No automatic successor."}
    binding = old.digest(manifest)
    source.prepare(binding, OUTPUT, parent=original["source_parent"])
    import_timing(OUTPUT, binding, evidence)
    c.require(identity() == current, "Code changed during preparation"); assert_clean()
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"freeze.json", {"binding": binding, "manifest": c.artifact(OUTPUT/"manifest.json"),
                                       "owner_authorized": True, "time": old.now()})
    c.require(validate_freeze() == binding, "Post-freeze validation failed")
    print({"prepared": True, "binding": binding, "reused_timing_roots": 18}, flush=True)


def validate_freeze():
    freeze = old.read_json(OUTPUT/"freeze.json"); manifest = c.verified(freeze["manifest"])
    c.require(freeze["owner_authorized"] is True and freeze["binding"] == old.digest(manifest)
              and manifest["identity"] == identity(), "Continuation freeze/code changed")
    checked = c.verified(manifest["checks"])
    c.require(checked["exit_code"] == 0 and checked["identity"] == manifest["identity"], "Continuation preflight changed")
    for ref in checked["logs"]: c.verify_hash(ref)
    c.require(manifest["held_parent"] == parent_evidence(), "Held parent evidence changed")
    original = c.verified(manifest["held_parent"]["manifest"])
    c.require(manifest["source_parent"] == original["source_parent"], "Scientific source changed")
    source.validate(freeze["binding"], OUTPUT, authenticate_parent=False)
    validate_reuse(OUTPUT, freeze["binding"], manifest["held_parent"])
    return freeze["binding"]


def forecast(binding):
    manifest = c.verified(old.read_json(OUTPUT/"freeze.json")["manifest"])
    prior = c.verified(manifest["held_parent"]["forecast"])
    launch = old.read_json(OUTPUT/"launch.json")
    active = launch["previous_elapsed_seconds"]+time.time()-launch["epoch_seconds"]
    estimate = prior["estimated_total_seconds"]-prior["elapsed_seconds"]+active
    free = shutil.disk_usage(ROOT).free
    result = {**prior, "manifest": binding, "reused_from": manifest["held_parent"]["forecast"],
        "timing": old.read_json(OUTPUT/"timing_lock.json")["files"], "elapsed_seconds": active,
        "estimated_total_seconds": estimate, "maximum_active_seconds": MAXIMUM_RUNTIME_SECONDS,
        "within_runtime_ceiling": estimate <= MAXIMUM_RUNTIME_SECONDS,
        "free_disk_bytes": free, "disk_reserve_pass": free-prior["estimated_remaining_bytes"] >= c.CONFIG["disk_reserve_bytes"],
        "cost_estimates_unchanged": True, "idle_pause_excluded": True}
    old.save_new(OUTPUT/"forecast.json", result)
    if not result["disk_reserve_pass"]: raise c.BudgetExceeded("Forecast cannot retain five-GiB reserve")
    if not result["within_runtime_ceiling"]: raise c.BudgetExceeded("Forecast exceeds authorized three-hour active cap")
    parent.event("forecast_pass", estimated_total_seconds=estimate)
    return result


def run():
    configure(); folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt", folder/"stderr.txt")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(), "One-shot continuation; preserve evidence")
        binding = validate_freeze()
        evidence = c.verified(old.read_json(OUTPUT/"freeze.json")["manifest"])["held_parent"]
        parent.REPORTER = parent.StatusReporter(OUTPUT, {"manifest": binding})
        try:
            with parent.ancestor.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT/"launch.json", {"manifest": binding, "time": old.now(), "epoch_seconds": time.time(),
                    "previous_elapsed_seconds": evidence["previous_elapsed_seconds"], "maximum_active_seconds": MAXIMUM_RUNTIME_SECONDS,
                    "pid": os.getpid(), "owner_authorized": True, "sleep_inhibition": sleep,
                    "torch_threads": torch.get_num_threads(), "windowless": c.windows.assert_windowless_worker()})
                parent.verify_stage("timing", binding)
                forecast(binding)
                timings = old.read_json(OUTPUT/"timing_lock.json")["files"]
                train_specs = source.schedule(binding, OUTPUT, "train")
                train = parent.run_pool("train", train_specs, binding)
                parent.run_pool("train_audits", parent.paired_audit_args(train_specs, train), binding)
                parent.verify_stage("train", binding); parent.event("calibration")
                selected = statistics.select(train, timings, binding, OUTPUT)
                selected_audit = statistics.audit_selection(train, timings, selected, binding, OUTPUT)
                parent.verify_selection(binding, selected, selected_audit)
                if c.verified(selected)["status"] != "selected":
                    parent.calibration_stop(binding, selected, selected_audit); return
                val_specs = source.schedule(binding, OUTPUT, "validation")
                val = parent.run_pool("validation", val_specs, binding)
                parent.run_pool("validation_audits", parent.paired_audit_args(val_specs, val), binding)
                parent.verify_stage("validation", binding); parent.event("assessment")
                result = statistics.assess(val, selected, timings, binding, OUTPUT)
                parent.event("assessment_audit")
                checked = statistics.audit(val, selected, timings, result, binding, OUTPUT)
                parent.finish(binding, result, checked, selected, selected_audit)
        except c.BudgetExceeded as exc:
            old.save_new(OUTPUT/"operational_hold.json", {"manifest": binding, "time": old.now(), "reason": str(exc)})
            parent.event("operational_hold", reason=str(exc))
        except BaseException as exc:
            parent.abort(exc, OUTPUT)
            old.save_new(OUTPUT/"failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            parent.event("failed", failure="failure.json"); raise


def launch():
    assert_clean(); binding = validate_freeze()
    folder = OUTPUT/"runtime"; folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder/"launcher.lock"):
        assert_clean()
        process = c.windows.detached_launch([sys.executable, "-u", ROOT/"run_control_variate_128_recovery_01.py", "run"],
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
    parser.add_argument("--checks", type=Path); args = parser.parse_args()
    if args.command in ("check", "prepare"): c.require(args.checks is not None, "Checks destination required")
    if args.command == "check": return check(args.checks)
    if args.command == "prepare": prepare(args.checks)
    elif args.command == "launch": launch()
    else: run()
    return 0
