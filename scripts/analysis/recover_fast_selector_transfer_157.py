"""Reviewable one-shot157 export-I/O recovery; resume needs owner approval.

Frozen scientific code stays byte-identical. Only mutable157 progress writes
receive a bounded Windows sharing-error retry. Immutable record writes retain
their original failure semantics. No fit/game is retried or replaced.
"""
from pathlib import Path
import argparse
import importlib.util
import os
import subprocess
import sys
import time
import traceback

from splendor_ai.alphazero import run_fast_selector_transfer as p

f, old, OUTPUT = p.f, p.old, p.OUTPUT
FOLDER = OUTPUT/"recovery_export_io"
BINDING = "3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122"
ORIGINAL_WRITE = old.atomic_write
verifier_spec = importlib.util.spec_from_file_location("recovery157_audit", str(p.VERIFIER))
audit = importlib.util.module_from_spec(verifier_spec)
verifier_spec.loader.exec_module(audit)


def mutable_progress(path):
    path = Path(path).resolve()
    return path.is_relative_to(OUTPUT.resolve()) and path.name in ("progress.json", "audit_progress.json")


def progress_write(path, payload):
    if not mutable_progress(path):
        return ORIGINAL_WRITE(path, payload)
    for attempt in range(51):
        try:
            return ORIGINAL_WRITE(path, payload)
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 50:
                raise
            time.sleep(.1)


# Spawned workers import this main module too, so only mutable progress I/O is
# patched consistently in fits, game workers and independent verifier workers.
old.atomic_write = progress_write


def verify_precondition():
    assert p.validate_freeze() == BINDING
    failure = old.read_json(OUTPUT/"failure.json")
    assert "PermissionError: [WinError 5]" in failure["traceback"]
    assert "progress.json.tmp" in failure["traceback"] and "run_jobs" in failure["traceback"]
    assert old.read_json(OUTPUT/"preparation"/"exit.json")["exit_code"] == 1
    for name in ("export_lock.json", "export_audit.json", "fits", "fits_lock.json", "probe", "main", "final_status.json"):
        assert not (OUTPUT/name).exists(), name
    return {name: old.file_hash(OUTPUT/name) for name in ("failure.json", "freeze.json", "preparation/exit.json", "preparation/run.txt")}


def inspect_existing(spec, item):
    result = audit.source_game(spec, item)
    return {"path": item["path"], "sha256": item["sha256"], "rows": item["rows"],
            "source_experiment": spec["experiment"], "source_index": spec["index"], "independent_source_audit_pass": True}


def prepare():
    with old.process_lock(OUTPUT/"pipeline.lock"):
        evidence = verify_precondition()
        assert not (FOLDER/"plan.json").exists()
        specs = old.read_json(OUTPUT/"sources.json")
        found = set((OUTPUT/"export").glob("*.pt"))
        assert len(found) == 37 and not list((OUTPUT/"export").glob("*.tmp"))
        jobs = []
        for source in specs:
            path = f.shard_path(source)
            if path in found:
                jobs.append((source, {"path": path.as_posix(), "sha256": old.file_hash(path), "rows": len(source["roots"])}))
        assert len(jobs) == len(found)
        # Do not write original progress during review: use a separate pool.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8, initializer=old.initialize_worker) as pool:
            futures = [pool.submit(inspect_existing, source, item) for source, item in jobs]
            existing = [future.result() for future in futures]
        existing.sort(key=lambda r: r["path"])
        old.save_new(FOLDER/"plan.json", {"manifest": BINDING, "time": old.now(), "script_sha256": old.file_hash(Path(__file__)),
                     "preserved_failure_evidence": evidence, "existing": existing, "remaining_games_to_export": 800-len(existing),
                     "science_change": False, "fit_or_arena_started": False,
                     "io_change": "Only157 mutable progress files: retry Windows5/32 at0.1s intervals up to5s; immutable writes unchanged",
                     "resume_requires_explicit_owner_authorization": True})
        print({"review_prepared": True, "existing_exports_independently_verified": len(existing), "remaining": 800-len(existing)}, flush=True)


def export_remaining(plan):
    specs, splits = old.read_json(OUTPUT/"sources.json"), old.read_json(OUTPUT/"splits.json")
    by_path = {r["path"]: r for r in plan["existing"]}
    assert {path.as_posix() for path in (OUTPUT/"export").glob("*.pt")} == set(by_path)
    existing, pending = [], []
    for source in specs:
        name = f.shard_path(source).as_posix()
        if name in by_path:
            record = by_path[name]
            assert old.file_hash(Path(name)) == record["sha256"] and record["independent_source_audit_pass"]
            existing.append({key: record[key] for key in ("path", "sha256", "rows")})
        else:
            pending.append((source, splits))
    files = existing+p.run_jobs("export_remaining", f.export_game, pending)
    files.sort(key=lambda r: r["path"])
    assert len(files) == 800 and sum(r["rows"] for r in files) == 24342
    old.save_new(OUTPUT/"export_lock.json", {"manifest": BINDING, "files": files})


def checked_auditor(stage):
    p.validate_freeze()
    p.progress(f"{stage}_audit")
    code = subprocess.run([sys.executable, "-u", str(Path(__file__)), "audit", "--stage", stage]).returncode
    if code:
        raise RuntimeError(f"Independent {stage} audit failed: {code}")
    result = old.read_json(OUTPUT/f"{stage}_audit.json")
    assert result["passed"] and result["manifest"] == BINDING
    return result


def resume(authorized):
    assert authorized, "Explicit owner authorization is required for this recovery"
    with old.process_lock(OUTPUT/"pipeline.lock"):
        evidence = verify_precondition()
        plan = old.read_json(FOLDER/"plan.json")
        assert plan["manifest"] == BINDING and plan["preserved_failure_evidence"] == evidence
        assert plan["script_sha256"] == old.file_hash(Path(__file__))
        assert not (FOLDER/"launch.json").exists() and not (FOLDER/"failure.json").exists()
        old.save_new(FOLDER/"launch.json", {"time": old.now(), "pid": os.getpid(), "owner_authorized": True,
                     "plan_sha256": old.file_hash(FOLDER/"plan.json"), "manifest": BINDING})
        try:
            export_remaining(plan)
            checked_auditor("export")
            p.fit_all()
            checked_auditor("fits")
            p.seal_selections()
            p.collect("probe")
            probe = checked_auditor("probe")
            p.progress("probe_passed", measured_seconds=probe["elapsed_seconds"], projected_main_seconds=probe["projected_main_seconds"])
            p.collect("main")
            checked_auditor("main")
            p.close()
            old.save_new(FOLDER/"complete.json", {"manifest": BINDING, "time": old.now(),
                         "final_status_sha256": old.file_hash(OUTPUT/"final_status.json")})
        except BaseException:
            old.save_new(FOLDER/"failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


def run_audit(stage):
    plan = old.read_json(FOLDER/"plan.json")
    launch = old.read_json(FOLDER/"launch.json")
    assert plan["script_sha256"] == old.file_hash(Path(__file__))
    assert launch["owner_authorized"] and launch["plan_sha256"] == old.file_hash(FOLDER/"plan.json")
    with old.process_lock(OUTPUT/"verifier.lock"):
        if stage == "export":
            audit.audit_export()
        elif stage == "fits":
            audit.audit_fits()
        else:
            audit.audit_arena(stage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "resume", "audit"))
    parser.add_argument("--owner-authorized", action="store_true")
    parser.add_argument("--stage", choices=("export", "fits", "probe", "main"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "resume":
        resume(args.owner_authorized)
    else:
        assert args.stage is not None
        run_audit(args.stage)
