"""One-shot, windowless 161 pipeline with immutable stage evidence."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
import argparse
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import teacher_budget_161 as h
from splendor_ai.alphazero import windowless_workers_160 as windows
from splendor_ai.alphazero import run_world_budget as p160
from splendor_ai.alphazero import run_fast_selector_transfer as p157

old, OUTPUT = h.old, h.OUTPUT
REPOSITORY = Path(__file__).resolve().parents[3]
INVENTORY = [Path(p) for p in (
    "src/splendor_ai/alphazero/teacher_budget_161.py",
    "src/splendor_ai/alphazero/run_teacher_budget.py", "run_teacher_budget_161.py",
    "scripts/analysis/verify_teacher_budget_161.py", "docs/engineering/teacher_budget_161.md",
    "tests/test_teacher_budget_161.py", "tests/test_teacher_budget_runner_161.py",
    "scripts/analysis/pytest_environment_161.py")]
SOURCE_RESULT = "2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5"
SOURCE_FINAL = "df71ac19e846d7869a9ac83676ccb3c348b455129641304cbf68742fe250ba69"
SOURCE_COLLECT = "ac89ccc151af8149e2791b7d6fcd989c7ca9416100ca5cf14e270e72cf0872b9"
PARENT_FILES = {
    "result.json": "aac9f292a92b31ae904b6fe7244f1a8ff57b05fd406d9d035d11bc94da18707a",
    "main_audit.json": "a5057a0aaf0e77c93f340b3046490ad2777b366e9aa8cf9720ad47348a90496c",
    "selected_lock.json": "4942eebb38de81b40d38315ec4abaf6b306f9340f4fd7051113250d4831044d8",
    "fits_audit.json": "77fc545ba9d460731ec881ecb14baf8f11daf5c7158778728fc7eb6da5df31a8"}


def auditor():
    from scripts.analysis import verify_teacher_budget_161
    return verify_teacher_budget_161


def event(stage, **values):
    print({"time": old.now(), "stage": stage, **values}, flush=True)


def identity():
    binding160 = p160.validate_freeze()
    binding157 = p157.validate_freeze()
    assert binding160 == "5e9e9202f44e7b526fff4be97cdcd3186805fbb5e8a184d21a6702de9d922df3"
    assert binding157 == "3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122"
    for name, sha in (("result.json", SOURCE_RESULT), ("final_status.json", SOURCE_FINAL), ("main_collect_lock.json", SOURCE_COLLECT)):
        assert old.file_hash(h.source.OUTPUT / name) == sha
    final = old.read_json(h.source.OUTPUT / "final_status.json")
    assert final["complete"] and final["prespecified_audit_pass"] and final["result_sha256"] == SOURCE_RESULT
    result = old.read_json(h.source.OUTPUT / "result.json")
    assert result["main_audit_lock_sha256"] == old.file_hash(h.source.OUTPUT / "main_audit_lock.json")
    for name, sha in PARENT_FILES.items():
        assert old.file_hash(h.f.OUTPUT / name) == sha
    return {"format": "teacher_budget_161_v1", "config": h.CONFIG,
            "source_binding": binding160, "source_result_sha256": SOURCE_RESULT,
            "initializer_binding": binding157, "initializer_evidence": PARENT_FILES,
            "initializers": {str(seed): h.initializer(seed) for seed in h.SEEDS},
            "dependencies": {p.as_posix(): old.file_hash(p) for p in INVENTORY},
            "windowless_runtime": windows.configure_spawn(), "python": sys.version,
            "torch": str(torch.__version__), "numpy": np.__version__}


def tests(folder):
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    start = time.perf_counter()
    code = ("import sys; from splendor_ai.alphazero import windowless_workers_160 as w; "
            "w.configure_spawn(); w.assert_windowless_worker(sys.argv[1],sys.argv[2]); "
            "import pytest; raise SystemExit(pytest.main(['-q','-p','scripts.analysis.pytest_environment_161']))")
    completed = subprocess.run([windows.configure_spawn()["executable"], "-u", "-c", code,
                                str((folder / "stdout.txt").resolve()), str((folder / "stderr.txt").resolve())],
                               cwd=REPOSITORY, stdin=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert identity() == before
    old.save_new(folder / "tests.json", {"identity": before, "exit_code": completed.returncode,
                 "full_repository": True, "test_environment": "scripts.analysis.pytest_environment_161",
                 "elapsed_seconds": time.perf_counter()-start,
                 "stdout_sha256": old.file_hash(folder / "stdout.txt"), "stderr_sha256": old.file_hash(folder / "stderr.txt")})
    return completed.returncode


def prepare(evidence_path):
    assert not (OUTPUT / "manifest.json").exists()
    manifest = identity()
    evidence = old.read_json(evidence_path)
    assert evidence["identity"] == manifest and evidence["exit_code"] == 0 and evidence["full_repository"]
    for stream in ("stdout", "stderr"):
        assert evidence[f"{stream}_sha256"] == old.file_hash(evidence_path.parent / f"{stream}.txt")
    # Full immutable game and prescribed-audit inventories, without new inference.
    for stage in ("probe", "main"):
        for audit in (False, True):
            p160.verify_lock(stage, manifest["source_binding"], audit)
    source_specs = h.source_inventory()
    assert h.splits() == auditor().independent_split()
    schedules = {stage: h.schedule(stage) for stage in ("probe", "main")}
    for stage, specs in schedules.items():
        assert specs == [auditor().independent_spec(stage, i) for i in range(len(specs))]
    old.save_new(OUTPUT / "sources.json", source_specs)
    old.save_new(OUTPUT / "splits.json", h.splits())
    old.save_new(OUTPUT / "schedules.json", schedules)
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": old.digest(manifest), "time": old.now(), "owner_authorized": True,
                 "files": [h.artifact(OUTPUT / name) for name in ("sources.json", "splits.json", "schedules.json")],
                 "tests": h.artifact(evidence_path)})
    event("frozen", binding=old.digest(manifest), roots=11985, fits=6, main_games=4800)


def verify_files(items):
    assert len({item["path"] for item in items}) == len(items)
    for item in items:
        assert old.file_hash(Path(item["path"])) == item["sha256"]


def validate_freeze():
    manifest = old.read_json(OUTPUT / "manifest.json")
    assert manifest == identity(), "Frozen161 identity changed"
    binding = old.digest(manifest)
    freeze = old.read_json(OUTPUT / "freeze.json")
    assert freeze["manifest"] == binding and freeze["owner_authorized"]
    verify_files([*freeze["files"], freeze["tests"]])
    evidence_path = Path(freeze["tests"]["path"])
    evidence = old.read_json(evidence_path)
    assert evidence["identity"] == manifest and evidence["exit_code"] == 0 and evidence["full_repository"]
    for stream in ("stdout", "stderr"):
        assert evidence[f"{stream}_sha256"] == old.file_hash(evidence_path.parent / f"{stream}.txt")
    return binding


def worker_initialize(stage, binding, base=OUTPUT):
    log = base / "workers" / stage
    log.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(log / f"{os.getpid()}_stdout.txt", log / f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    assert torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled()
    if stage.startswith(("probe_", "main_")):
        h.load_models(base)
    old.save_new(log / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, **health})


def dispatch(kind, args):
    functions = {"export": h.export_game, "export_audit": auditor().source_game,
                 "fits": h.train_fit, "fits_audit": auditor().replay_fit,
                 "games": h.play_game, "game_audit": auditor().replay_game}
    return functions[kind](*args)


def run_jobs(stage, kind, arguments, binding, upstream, workers=8):
    assert validate_freeze() == binding
    launch_path = OUTPUT / f"{stage}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "stage": stage, "time": old.now(),
                 "upstream": h.artifact(upstream), "jobs": len(arguments),
                 "selected": h.artifact(OUTPUT / "selected_lock.json") if stage.startswith(("probe_", "main_")) else None})
    runtime = windows.configure_spawn()  # Every new pool, not only initial startup.
    started = time.perf_counter()
    files = []
    event(stage, jobs=len(arguments), workers=workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, binding)) as pool:
        futures = [pool.submit(dispatch, kind, args) for args in arguments]
        for future in as_completed(futures):
            files.append(future.result())
            if len(files) % 25 == 0 or len(files) == len(arguments):
                event(stage, completed=len(files), total=len(arguments))
    files.sort(key=lambda r: (r.get("index", -1), r.get("arm", ""), r.get("seed", 0)))
    verify_files(files)
    assert len(files) == len(arguments) and validate_freeze() == binding
    launch = old.read_json(launch_path)
    verify_files([launch["upstream"]] + ([launch["selected"]] if launch["selected"] else []))
    old.save_new(OUTPUT / f"{stage}_lock.json", {"manifest": binding, "files": files,
                 "launch": h.artifact(launch_path), "runtime": runtime, "elapsed_seconds": time.perf_counter()-started})
    return files


def verify_stage(stage, binding):
    lock = old.read_json(OUTPUT / f"{stage}_lock.json")
    assert lock["manifest"] == binding
    verify_files([lock["launch"], *lock["files"]])
    launch = old.read_json(Path(lock["launch"]["path"]))
    assert launch["manifest"] == binding and launch["stage"] == stage and launch["jobs"] == len(lock["files"])
    verify_files([launch["upstream"]] + ([launch["selected"]] if launch["selected"] else []))
    for item in lock["files"]:
        assert Path(item["path"]).stat().st_mtime_ns >= Path(lock["launch"]["path"]).stat().st_mtime_ns
    return lock


def seal_selected(binding):
    fits = verify_stage("fits", binding)
    audits = verify_stage("fits_audit", binding)
    assert len(fits["files"]) == len(audits["files"]) == 6
    models = {}
    for seed in h.SEEDS:
        results = []
        models[f"anchor_{seed}"] = h.initializer(seed)
        for arm in h.ARMS:
            folder = OUTPUT / "fits" / f"{arm}_{seed}"
            result = old.read_json(folder / "result.json")
            audit = old.read_json(folder / "audit.json")
            assert audit["complete_fit_bitwise_identical"] and audit["result_sha256"] == old.file_hash(folder / "result.json")
            chosen = min(result["curves"], key=lambda row: (row["pair_loss"], row["update"]))
            assert chosen["update"] == result["selected_update"] == audit["selected"]
            path = folder / f"checkpoint_{chosen['update']:04d}.pt"
            assert old.file_hash(path) == chosen["checkpoint_sha256"]
            models[f"{arm}_{seed}"] = h.artifact(path, update=chosen["update"])
            results.append(result)
        assert results[0]["batch_sequence_sha256"] == results[1]["batch_sequence_sha256"]
        assert results[0]["initializer"] == results[1]["initializer"]
    old.save_new(OUTPUT / "selected_lock.json", {"manifest": binding, "time": old.now(), "models": models,
                 "fits_lock": h.artifact(OUTPUT / "fits_lock.json"), "fits_audit_lock": h.artifact(OUTPUT / "fits_audit_lock.json")})


def panel(stage, binding, upstream):
    specs = old.read_json(OUTPUT / "schedules.json")[stage]
    assert specs == h.schedule(stage)
    files = run_jobs(f"{stage}_games", "games", [(stage, s, binding) for s in specs], binding, upstream)
    assert [r["index"] for r in files] == list(range(len(specs)))
    verify_stage(f"{stage}_games", binding)
    return run_jobs(f"{stage}_audit", "game_audit", [(stage, s, item, binding) for s, item in zip(specs, files)],
                    binding, OUTPUT / f"{stage}_games_lock.json")


def preflight(binding):
    collect = verify_stage("probe_games", binding)
    audit = verify_stage("probe_audit", binding)
    assert len(collect["files"]) == len(audit["files"]) == 48
    size = sum(p.stat().st_size for p in (OUTPUT / "probe").rglob("*") if p.is_file())
    free = shutil.disk_usage(OUTPUT).free
    required = max(10*1024**3, 1.5*100*size+5*1024**3)
    result = {"manifest": binding, "time": old.now(), "integrity_pass": True,
              "outcomes_used_for_gate": False, "estimated_main_seconds": 100*(collect["elapsed_seconds"]+audit["elapsed_seconds"]),
              "estimated_main_bytes": 100*size, "disk_free_bytes": free, "required_free_bytes": required,
              "passed": free >= required, "probe_audit_lock": h.artifact(OUTPUT / "probe_audit_lock.json")}
    old.save_new(OUTPUT / "preflight.json", result)
    assert result["passed"], "Fixed probe storage gate failed"
    event("preflight_passed", estimated_main_hours=result["estimated_main_seconds"]/3600)


def finish(binding):
    for stage in ("export", "export_audit", "fits", "fits_audit", "probe_games", "probe_audit", "main_games", "main_audit"):
        verify_stage(stage, binding)
    selected = old.read_json(OUTPUT / "selected_lock.json")
    verify_files(list(selected["models"].values()))
    games, timing = [], defaultdict(list)
    for item in old.read_json(OUTPUT / "main_audit_lock.json")["files"]:
        report = old.read_json(Path(item["path"]))
        games.append(report["game"])
        for t in report["timings"]:
            timing[t["model"]].append(t)
    stats = h.statistics(games)
    auditor().a.compare(stats, auditor().independent_statistics(games))
    timings = {key: {kind: {"mean": sum(r[kind] for r in rows)/len(rows),
                           "p95": float(np.percentile([r[kind] for r in rows], 95)), "count": len(rows)}
                     for kind in ("wall", "cpu")} for key, rows in sorted(timing.items())}
    assert validate_freeze() == binding
    result = {"manifest": binding, "time": old.now(), "statistics": stats, "timings": timings,
              "games": games, "all_prescribed_audits_pass": True,
              "main_audit_lock": h.artifact(OUTPUT / "main_audit_lock.json"),
              "classification": "positive" if stats["both_pass"] else "clear_success_not_established"}
    old.save_new(OUTPUT / "result.json", result)
    with (OUTPUT / "report.md").open("x", encoding="utf-8") as file:
        file.write("# Experiment161: stronger-teacher incremental learning\n\n")
        for comparison in h.COMPARISONS:
            r = stats[comparison]
            file.write(f"{comparison}: {100*r['score']:.3f}% score; paired 95% interval [{100*r['ci95'][0]:.3f}, {100*r['ci95'][1]:.3f}]%. Gate passed: {r['pass']}.\n\n")
        file.write("Classification: " + result["classification"] + ".\n\n"
                   "4,800 fresh games; 400 whole seed clusters, retaining both comparisons, three fixed fits and both seats. "
                   "All source tensors/labels independently rebuilt, all six fits/checkpoint selections reproduced bitwise, "
                   "all fresh games independently replayed. The saved160 teacher outcomes retain their original sampled-inner audit; "
                   "no new inner rollouts. Intervals condition on the three fixed fits. Validation is reused development data. "
                   "No automatic promotion, extension, successor or monitoring. Failure to meet both gates does not reject distillation generally.\n")
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "all_prescribed_audits_pass": True, "classification": result["classification"],
                 "result": h.artifact(OUTPUT / "result.json"), "report": h.artifact(OUTPUT / "report.md")})
    event("complete", classification=result["classification"])


def run():
    log = OUTPUT / "preparation"
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        assert not (OUTPUT / "launch.json").exists(), "One-shot pipeline; no automatic resume"
        binding = validate_freeze()
        old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                     "owner_authorized": True, "windowless": windows.assert_windowless_worker()})
        try:
            sources = old.read_json(OUTPUT / "sources.json")
            split = old.read_json(OUTPUT / "splits.json")
            files = run_jobs("export", "export", [(s, split[s["pair"]]) for s in sources], binding, OUTPUT / "freeze.json")
            assert [r["index"] for r in files] == list(range(400))
            verify_stage("export", binding)
            audits = run_jobs("export_audit", "export_audit", [(s, item) for s, item in zip(sources, files)], binding, OUTPUT / "export_lock.json")
            coverage = Counter()
            for item in audits:
                coverage.update(old.read_json(Path(item["path"]))["coverage"])
            assert coverage["rows"] == 11985 and coverage["labels_changed"] == 4081
            old.save_new(OUTPUT / "export_summary.json", {"manifest": binding, "coverage": dict(coverage), "time": old.now()})
            fits = [(arm, seed) for arm in h.ARMS for seed in h.SEEDS]
            run_jobs("fits", "fits", fits, binding, OUTPUT / "export_audit_lock.json", workers=3)
            run_jobs("fits_audit", "fits_audit", fits, binding, OUTPUT / "fits_lock.json", workers=3)
            seal_selected(binding)
            panel("probe", binding, OUTPUT / "selected_lock.json")
            preflight(binding)
            panel("main", binding, OUTPUT / "preflight.json")
            finish(binding)
        except BaseException:
            old.save_new(OUTPUT / "failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


def held_launch(command, log, acknowledgement, binding, cwd=REPOSITORY, timeout=60):
    """The direct parent stays alive until both child parent-executable queries finish."""
    log.mkdir(parents=True, exist_ok=True)
    process = windows.detached_launch(command, log / "stdout.txt", log / "stderr.txt", cwd)
    old.save_new(log / "process.json", {**process, "manifest": binding, "time": old.now()})
    deadline = time.monotonic()+timeout
    while not acknowledgement.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Launch acknowledgement absent; preserve evidence, do not automatically relaunch")
        time.sleep(.05)
    record = old.read_json(acknowledgement)
    assert record["pid"] == process["pid"] and record["manifest"] == binding
    old.save_new(log / "acknowledged.json", {"time": old.now(), "pid": process["pid"], "launch": h.artifact(acknowledgement)})
    return process


def launch():
    binding = validate_freeze()
    log = OUTPUT / "preparation"
    with old.process_lock(log / "launcher.lock"):
        assert not (log / "process.json").exists() and not (OUTPUT / "launch.json").exists()
        process = held_launch([sys.executable, "-u", REPOSITORY / "run_teacher_budget_161.py", "run"],
                              log, OUTPUT / "launch.json", binding)
    event("detached_launch_acknowledged", pid=process["pid"])


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
