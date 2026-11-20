"""Continue the owner-approved157 resume after audit module registration failed.

No scientific file, export, fit or game is edited or replaced. Register the
unchanged dynamic verifier in sys.modules so Windows spawn can pickle its
workers; retain the previously tested progress-only Windows I/O handling.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import importlib.util
import os
import pickle
import subprocess
import sys
import traceback

IO_SCRIPT = Path("scripts/analysis/recover_fast_selector_transfer_157.py")
io_spec = importlib.util.spec_from_file_location("recovery157_io", IO_SCRIPT)
r = importlib.util.module_from_spec(io_spec)
sys.modules[io_spec.name] = r
io_spec.loader.exec_module(r)
# Registration happens in both parent and spawned children. The function's
# module name is preserved; its scientific implementation is byte-identical.
sys.modules[r.verifier_spec.name] = r.audit
p, f, old, audit = r.p, r.f, r.old, r.audit
OUTPUT = p.OUTPUT
FOLDER = OUTPUT/"recovery_audit_import"


def precondition():
    assert p.validate_freeze() == r.BINDING
    assert old.file_hash(IO_SCRIPT) == "4352300d3a5c441ee37e67e55b46ff027e5f33f765f11880cb8b72f8c43e4cfd"
    previous = OUTPUT/"recovery_export_io"
    assert old.read_json(previous/"launch.json")["owner_authorized"]
    assert "Independent export audit failed: 1" in old.read_json(previous/"failure.json")["traceback"]
    assert "import of module 'recovery157_audit' failed" in (previous/"stderr.txt").read_text(encoding="utf-8")
    for name in ("export_audit.json", "fits", "fits_lock.json", "selected_lock.json", "probe", "main", "final_status.json"):
        assert not (OUTPUT/name).exists(), name
    lock = old.read_json(OUTPUT/"export_lock.json")
    assert lock["manifest"] == r.BINDING and len(lock["files"]) == 800
    assert sum(item["rows"] for item in lock["files"]) == 24342
    assert {path.as_posix() for path in (OUTPUT/"export").glob("*.pt")} == {item["path"] for item in lock["files"]}
    for item in lock["files"]:
        assert old.file_hash(Path(item["path"])) == item["sha256"]
    return {"manifest": r.BINDING, "script_sha256": old.file_hash(Path(__file__)),
            "io_script_sha256": old.file_hash(IO_SCRIPT), "export_lock_sha256": old.file_hash(OUTPUT/"export_lock.json"),
            "previous_failure_sha256": old.file_hash(previous/"failure.json"),
            "previous_stderr_sha256": old.file_hash(previous/"stderr.txt")}


def prepare():
    with old.process_lock(OUTPUT/"pipeline.lock"):
        before = precondition()
        assert not (FOLDER/"plan.json").exists()
        sources = old.read_json(OUTPUT/"sources.json")
        files = {Path(item["path"]).name: item for item in old.read_json(OUTPUT/"export_lock.json")["files"]}
        selected = [s for s in sources if s["index"] == 0]
        assert len(selected) == 2
        # Exercise exactly the callable/Windows-spawn path that failed, rather
        # than testing through a wrapper callable in a different module.
        for function in (audit.source_game, audit.replay_fit, audit.replay_game):
            assert pickle.loads(pickle.dumps(function)) is function
        with ProcessPoolExecutor(max_workers=2, initializer=old.initialize_worker) as pool:
            futures = [pool.submit(audit.source_game, s, files[f"{s['experiment']}_0000.pt"]) for s in selected]
            reports = [future.result() for future in futures]
        assert precondition() == before
        old.save_new(FOLDER/"plan.json", {**before, "time": old.now(), "all_800_exports_hash_verified": True,
                     "actual_spawned_verifier_probe_passed": True, "probe_reports": reports,
                     "owner_authorization": "Continue the explicitly authorized resume; fix internal worker loading without changing the experiment",
                     "scientific_change": False})
        print({"ready": True, "exports_verified": 800, "spawned_verifier_games_passed": len(reports)}, flush=True)


def checked_auditor(stage):
    assert p.validate_freeze() == r.BINDING
    p.progress(f"{stage}_audit")
    code = subprocess.run([sys.executable, "-u", str(Path(__file__)), "audit", "--stage", stage]).returncode
    if code:
        raise RuntimeError(f"Independent {stage} audit failed: {code}")
    report = old.read_json(OUTPUT/f"{stage}_audit.json")
    assert report["passed"] and report["manifest"] == r.BINDING
    return report


def resume(authorized):
    assert authorized, "Owner authorization is required"
    with old.process_lock(OUTPUT/"pipeline.lock"):
        expected = precondition()
        plan = old.read_json(FOLDER/"plan.json")
        assert all(plan[key] == value for key, value in expected.items())
        assert plan["actual_spawned_verifier_probe_passed"]
        assert not (FOLDER/"launch.json").exists() and not (FOLDER/"failure.json").exists()
        old.save_new(FOLDER/"launch.json", {"time": old.now(), "pid": os.getpid(), "owner_authorized": True,
                     "plan_sha256": old.file_hash(FOLDER/"plan.json"), "manifest": r.BINDING})
        try:
            # All exports are sealed; resume at the first unfinished audit.
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
            old.save_new(FOLDER/"complete.json", {"manifest": r.BINDING, "time": old.now(),
                         "final_status_sha256": old.file_hash(OUTPUT/"final_status.json")})
        except BaseException:
            old.save_new(FOLDER/"failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


def run_audit(stage):
    plan, launch = [old.read_json(FOLDER/name) for name in ("plan.json", "launch.json")]
    assert plan["script_sha256"] == old.file_hash(Path(__file__))
    assert plan["io_script_sha256"] == old.file_hash(IO_SCRIPT)
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
