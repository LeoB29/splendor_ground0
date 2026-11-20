"""Owner-authorized, separately sealed operational recovery of161 after probe collection.

Frozen scientific functions/files are unchanged. The supervisor installs a
scoped adapter for exactly two identical selected_lock dependency references.
It restores the original verifier on exit. No training or probe collection runs.
"""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
import traceback

from splendor_ai.alphazero import run_teacher_budget as p

old = p.old
RECOVERY = p.OUTPUT / "recovery_01"
BINDING = "f8f1fda4884cdcde518ad079430528b95abaf19e5b47946630dd5c4e5f22eeeb"
FAILURE_SHA = "4634dbdcb2821dde94f2fb25e3dae58a98714c556e72cc77cef3e15aeeb6dbdc"
DIAGNOSTIC = p.OUTPUT / "manual_reviews/status_20260917_failure_01.json"
INVENTORY = [Path(x) for x in (
    "src/splendor_ai/alphazero/recover_teacher_budget.py", "recover_teacher_budget_161.py",
    "tests/test_teacher_budget_recovery_161.py", "docs/engineering/teacher_budget_161_recovery_01.md")]
TESTS = ["tests/test_teacher_budget_recovery_161.py", "tests/test_teacher_budget_161.py",
         "tests/test_teacher_budget_runner_161.py"]
STRICT_VERIFY_FILES = p.verify_files


def verify_files_with_selected_alias(items):
    """Retain strict uniqueness everywhere except this exact two-reference alias."""
    selected = p.OUTPUT / "selected_lock.json"
    if (len(items) == 2 and items[0] == items[1]
            and set(items[0]) == {"path", "sha256"}
            and items[0]["path"] == selected.as_posix()):
        STRICT_VERIFY_FILES([items[0]])  # Still check the supplied hash, never refresh it.
    else:
        STRICT_VERIFY_FILES(items)


@contextmanager
def selected_alias_adapter():
    assert p.verify_files is STRICT_VERIFY_FILES, "No stacked or unexpected runtime repair"
    p.verify_files = verify_files_with_selected_alias
    try:
        yield
    finally:
        p.verify_files = STRICT_VERIFY_FILES


def identity():
    assert p.validate_freeze() == BINDING
    return {"format": "161_operational_recovery_01", "scientific_manifest": BINDING,
            "dependencies": {path.as_posix(): old.file_hash(path) for path in INVENTORY},
            "tests": TESTS, "repair": "exact_duplicate_selected_dependency_alias_only",
            "resume_at": "probe_audit", "retrain": False, "recollect_probe": False,
            "scheduled_monitoring": False, "automatic_retry": False}


def require_restart_boundary(base=None):
    base = p.OUTPUT if base is None else base
    for name in ("probe_games_lock.json", "probe_audit_launch.json", "probe_audit_lock.json",
                 "preflight.json", "main_games_launch.json", "main_games_lock.json",
                 "main_audit_launch.json", "main_audit_lock.json", "result.json", "final_status.json"):
        assert not (base / name).exists(), f"Recovery boundary already crossed: {name}"
    assert not (base / "probe/audit").exists() and not (base / "main").exists()


def saved_probe(binding, diagnostic, base=None):
    """Validate the previously inventoried files without computing game scores."""
    base = p.OUTPUT if base is None else base
    assert diagnostic["manifest"] == binding and diagnostic["probe_full_replay_pending"]
    assert not diagnostic["main_started"]
    launch_path, failure_path = base / "probe_games_launch.json", base / "failure.json"
    launch, failure = old.read_json(launch_path), old.read_json(failure_path)
    assert diagnostic["failure"] == p.h.artifact(failure_path)
    assert launch["manifest"] == binding and launch["stage"] == "probe_games" and launch["jobs"] == 48
    assert launch["upstream"] == launch["selected"] == p.h.artifact(base / "selected_lock.json")
    specs = old.read_json(base / "schedules.json")["probe"]
    assert specs == p.h.schedule("probe") and len(specs) == 48
    files = diagnostic["probe_files"]
    assert [item["index"] for item in files] == list(range(48))
    STRICT_VERIFY_FILES(files)
    assert {path.name for path in (base / "probe/games").glob("*.json")} == {
        f"game_{index:04d}.json" for index in range(48)}
    upper = (datetime.fromisoformat(failure["time"]) - datetime.fromisoformat(launch["time"])).total_seconds()
    assert upper > 0
    for spec, item in zip(specs, files):
        path = p.h.game_path("probe", spec["index"], base)
        assert item == p.h.artifact(path, index=spec["index"])
        assert launch_path.stat().st_mtime_ns <= path.stat().st_mtime_ns <= failure_path.stat().st_mtime_ns
        game = old.read_json(path)
        assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
        assert "test_initial_snapshot" not in game and game["reason"] in old.VALID_NATURAL_TERMINATIONS
        assert 0 <= game["elapsed_seconds"] <= upper
    return files, upper


def verify_completed_work():
    assert p.validate_freeze() == BINDING
    assert old.file_hash(p.OUTPUT / "failure.json") == FAILURE_SHA
    for stage, count in (("export", 400), ("export_audit", 400), ("fits", 6), ("fits_audit", 6)):
        assert len(p.verify_stage(stage, BINDING)["files"]) == count
    selected = old.read_json(p.OUTPUT / "selected_lock.json")
    assert selected["manifest"] == BINDING and len(selected["models"]) == 9
    STRICT_VERIFY_FILES(list(selected["models"].values()) + [selected["fits_lock"], selected["fits_audit_lock"]])
    for seed in p.h.SEEDS:
        paired = []
        assert selected["models"][f"anchor_{seed}"] == p.h.initializer(seed)
        for arm in p.h.ARMS:
            folder = p.OUTPUT / "fits" / f"{arm}_{seed}"
            result, audit = old.read_json(folder / "result.json"), old.read_json(folder / "audit.json")
            assert audit["complete_fit_bitwise_identical"] and audit["result_sha256"] == old.file_hash(folder / "result.json")
            chosen = min(result["curves"], key=lambda r: (r["pair_loss"], r["update"]))
            assert audit["selected"] == result["selected_update"] == chosen["update"]
            assert result["updates"] == 600 and result["train_rows"] == 9577 and result["validation_rows"] == 2408
            assert selected["models"][f"{arm}_{seed}"] == p.h.artifact(
                folder / f"checkpoint_{chosen['update']:04d}.pt", update=chosen["update"])
            for row in result["curves"]:
                assert old.file_hash(folder / f"checkpoint_{row['update']:04d}.pt") == row["checkpoint_sha256"]
                assert old.file_hash(folder / f"validation_{row['update']:04d}.json") == row["validation_sha256"]
            paired.append(result)
        assert paired[0]["batch_sequence_sha256"] == paired[1]["batch_sequence_sha256"]
        assert paired[0]["initializer"] == paired[1]["initializer"] == p.h.initializer(seed)
    return saved_probe(BINDING, old.read_json(DIAGNOSTIC))


def retained_inventory():
    """Pin all completed scientific artifacts/logs; exclude ephemeral OS locks and preparation tests."""
    paths = []
    for path in p.OUTPUT.rglob("*"):
        if not path.is_file() or path.suffix == ".lock":
            continue
        relative = path.relative_to(p.OUTPUT)
        if relative.parts[0] == "recovery_01":
            continue
        if relative.parts[0] == "preparation" and relative.as_posix() not in (
                "preparation/stdout.txt", "preparation/stderr.txt"):
            continue
        assert path.suffix != ".tmp", f"Unexpected partial artifact: {path}"
        paths.append(path)
    return [p.h.artifact(path) for path in sorted(paths)]


def tests(folder):
    folder.mkdir(parents=True, exist_ok=False)
    before, start = identity(), time.perf_counter()
    code = ("import sys; from splendor_ai.alphazero import windowless_workers_160 as w; "
            "w.configure_spawn(); w.assert_windowless_worker(sys.argv[1],sys.argv[2]); "
            "import pytest; raise SystemExit(pytest.main(['-q']+sys.argv[3:]))")
    result = subprocess.run([p.windows.configure_spawn()["executable"], "-u", "-c", code,
                             str((folder / "stdout.txt").resolve()), str((folder / "stderr.txt").resolve()), *TESTS],
                            cwd=p.REPOSITORY, stdin=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert identity() == before
    old.save_new(folder / "tests.json", {"identity": before, "exit_code": result.returncode,
                 "scope": TESTS, "elapsed_seconds": time.perf_counter()-start,
                 "logs": [p.h.artifact(folder / name) for name in ("stdout.txt", "stderr.txt")]})
    return result.returncode


def prepare(evidence_path):
    with old.process_lock(p.OUTPUT / "run.lock"):
        assert not (RECOVERY / "manifest.json").exists()
        require_restart_boundary()
        before = identity()
        evidence = old.read_json(evidence_path)
        assert evidence["identity"] == before and evidence["exit_code"] == 0 and evidence["scope"] == TESTS
        STRICT_VERIFY_FILES(evidence["logs"])
        files, upper = verify_completed_work()
        manifest = {"identity": before, "time": old.now(), "owner_authorization": "repair and restart",
                    "tests": p.h.artifact(evidence_path), "retained": retained_inventory(),
                    "probe_files": files, "probe_elapsed_seconds_upper_bound": upper,
                    "elapsed_source": "original probe launch timestamp through original failure timestamp; excludes recovery downtime"}
        old.save_new(RECOVERY / "manifest.json", manifest)
        p.event("recovery_prepared", recovery_binding=old.digest(manifest), saved_games=len(files))


def validate_recovery():
    manifest = old.read_json(RECOVERY / "manifest.json")
    assert manifest["identity"] == identity() and manifest["owner_authorization"] == "repair and restart"
    STRICT_VERIFY_FILES([manifest["tests"], *manifest["retained"]])
    evidence = old.read_json(Path(manifest["tests"]["path"]))
    assert evidence["identity"] == manifest["identity"] and evidence["exit_code"] == 0 and evidence["scope"] == TESTS
    STRICT_VERIFY_FILES(evidence["logs"])
    return old.digest(manifest), manifest


def seal_saved_probe(manifest, recovery_binding):
    files, upper = saved_probe(BINDING, old.read_json(DIAGNOSTIC))
    assert files == manifest["probe_files"] and upper == manifest["probe_elapsed_seconds_upper_bound"]
    old.save_new(p.OUTPUT / "probe_games_lock.json", {
        "manifest": BINDING, "files": files, "launch": p.h.artifact(p.OUTPUT / "probe_games_launch.json"),
        "runtime": p.windows.configure_spawn(), "elapsed_seconds": upper,
        "elapsed_seconds_kind": "upper_bound_from_original_launch_to_original_failure",
        "recovery_manifest": p.h.artifact(RECOVERY / "manifest.json"), "recovery_binding": recovery_binding})
    p.verify_stage("probe_games", BINDING)
    return files


def continue_pipeline(files):
    specs = old.read_json(p.OUTPUT / "schedules.json")["probe"]
    assert specs == p.h.schedule("probe") and [f["index"] for f in files] == list(range(48))
    p.run_jobs("probe_audit", "game_audit", [("probe", s, item, BINDING) for s, item in zip(specs, files)],
               BINDING, p.OUTPUT / "probe_games_lock.json")
    p.preflight(BINDING)
    p.panel("main", BINDING, p.OUTPUT / "preflight.json")
    p.finish(BINDING)


def run():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    p.windows.assert_windowless_worker(RECOVERY / "stdout.txt", RECOVERY / "stderr.txt")
    with old.process_lock(p.OUTPUT / "run.lock"):
        assert not (RECOVERY / "launch.json").exists(), "One-shot recovery; never automatically retry"
        binding, manifest = validate_recovery()
        require_restart_boundary()
        old.save_new(RECOVERY / "launch.json", {"manifest": binding, "scientific_manifest": BINDING,
                     "pid": os.getpid(), "time": old.now(), "windowless": p.windows.assert_windowless_worker()})
        try:
            with selected_alias_adapter():
                files = seal_saved_probe(manifest, binding)
                p.event("recovery_resuming_probe_audit", games=len(files))
                continue_pipeline(files)
            assert validate_recovery()[0] == binding
            old.save_new(RECOVERY / "complete.json", {"manifest": binding, "time": old.now(),
                         "final_status": p.h.artifact(p.OUTPUT / "final_status.json"),
                         "original_failure": p.h.artifact(p.OUTPUT / "failure.json"),
                         "retained_artifacts_unchanged": True})
        except BaseException:
            old.save_new(RECOVERY / "failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            raise


def launch():
    with old.process_lock(RECOVERY / "launcher.lock"):
        assert not (RECOVERY / "process.json").exists() and not (RECOVERY / "launch.json").exists()
        binding, _ = validate_recovery()
        with old.process_lock(p.OUTPUT / "run.lock"):
            require_restart_boundary()
        process = p.held_launch([sys.executable, "-u", p.REPOSITORY / "recover_teacher_budget_161.py", "run"],
                                RECOVERY, RECOVERY / "launch.json", binding)
        p.event("recovery_launch_acknowledged", pid=process["pid"], recovery_binding=binding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tests", "prepare", "launch", "run"))
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--test-evidence", type=Path)
    args = parser.parse_args()
    if args.command == "tests":
        assert args.test_dir is not None
        return tests(args.test_dir)
    if args.command == "prepare":
        assert args.test_evidence is not None
        return prepare(args.test_evidence)
    return {"launch": launch, "run": run}[args.command]()
