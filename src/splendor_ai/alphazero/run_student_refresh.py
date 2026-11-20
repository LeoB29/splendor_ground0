"""One-shot, windowless 162 pipeline with immutable stage evidence."""
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

from splendor_ai.alphazero import student_refresh_162 as h
from splendor_ai.alphazero import windowless_workers_160 as windows
from splendor_ai.alphazero import run_teacher_budget as p161
from splendor_ai.alphazero import recover_teacher_budget as recovery161

old, OUTPUT = h.old, h.OUTPUT
REPOSITORY = Path(__file__).resolve().parents[3]
INVENTORY = [Path(p) for p in (
    "src/splendor_ai/alphazero/student_refresh_162.py",
    "src/splendor_ai/alphazero/run_student_refresh.py", "run_student_refresh_162.py",
    "scripts/analysis/verify_student_refresh_162.py", "docs/engineering/student_refresh_162.md",
    "tests/test_student_refresh_162.py", "tests/test_student_refresh_runner_162.py",
    "scripts/analysis/pytest_environment_161.py")]
PARENT_FILES = {'result.json': '813596b6228a2e3530a8b73cd577b557bf211cfd76ed433411ae2313074b2113', 'final_status.json': 'ffea35cbbcc45515119b691bd3f8b4692960c4f6c6385f9e9cf3261ed598d854', 'selected_lock.json': '1d73765ff2aec5b2d94fb057f9d692d5f37198a98b38b64c51480726d8d64441', 'export_lock.json': '721d8fa792a18be7bb99fa82734af6df755eefb2ddd6d13a6c82ac6c1503119a', 'export_audit_lock.json': '8e8ec159e8779f9d0ad6ecd0272704942a5442249a8cf841885dfd9fbdb1db68', 'fits_lock.json': '906589bbc1d81e1ed1163bad3f8ab2f9549793b5ba6b72584a32ad6840e19d86', 'fits_audit_lock.json': '661cd3fafb02a2d1ec81bb1b6dd3ebdeb6c970e918dd1119dcc0a1da720b3317', 'main_audit_lock.json': '91ed5d913516ba7041d347e5ebd8334daeecfc0d68f043e8f2cad28a4d42948a', 'recovery_01/complete.json': '20b2ea15dcb320a0fe112c8f6954eac1cd84cc209d6d5273996d787df65560af'}


def auditor():
    from scripts.analysis import verify_student_refresh_162
    return verify_student_refresh_162


def event(stage, **values):
    print({"time": old.now(), "stage": stage, **values}, flush=True)


def identity():
    parent_binding = p161.validate_freeze()
    assert parent_binding == "f8f1fda4884cdcde518ad079430528b95abaf19e5b47946630dd5c4e5f22eeeb"
    recovery_binding, _ = recovery161.validate_recovery()
    assert recovery_binding == "b3ec4a136483f0c0859edf6cd85e9751573bc26b719467d0b75042b11a43fe0a"
    for name, sha in PARENT_FILES.items():
        assert old.file_hash(h.previous.OUTPUT / name) == sha
    final = old.read_json(h.previous.OUTPUT / "final_status.json")
    assert final["complete"] and final["all_prescribed_audits_pass"]
    assert final["result"]["sha256"] == "813596b6228a2e3530a8b73cd577b557bf211cfd76ed433411ae2313074b2113"
    return {"format": "student_refresh_162_v1", "config": h.CONFIG,
            "parent_binding": parent_binding, "parent_recovery_binding": recovery_binding,
            "parent_evidence": PARENT_FILES,
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
    with recovery161.selected_alias_adapter():
        for stage in ("export", "export_audit", "fits", "fits_audit", "probe_games", "probe_audit", "main_games", "main_audit"):
            p161.verify_stage(stage, manifest["parent_binding"])
    assert h.splits() == auditor().independent_split()
    schedules = {"source": h.source_schedule(), **{stage: h.schedule(stage) for stage in ("probe", "main")}}
    for stage, specs in schedules.items():
        assert specs == [auditor().independent_spec(stage, i) for i in range(len(specs))]
    old.save_new(OUTPUT / "splits.json", h.splits())
    old.save_new(OUTPUT / "schedules.json", schedules)
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": old.digest(manifest), "time": old.now(), "owner_authorized": True,
                 "files": [h.artifact(OUTPUT / name) for name in ("splits.json", "schedules.json")],
                 "tests": h.artifact(evidence_path)})
    event("frozen", binding=old.digest(manifest), source_games=384, roots=2304, fits=6, main_games=4800)


def verify_files(items):
    assert len({item["path"] for item in items}) == len(items)
    for item in items:
        assert old.file_hash(Path(item["path"])) == item["sha256"]


def validate_freeze():
    manifest = old.read_json(OUTPUT / "manifest.json")
    assert manifest == identity(), "Frozen162 identity changed"
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
    if stage.startswith("source_"):
        h.load_source_models()
    elif stage.startswith(("probe_", "main_")):
        h.load_models(base)
    old.save_new(log / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, **health})


def dispatch(kind, args):
    functions = {"teacher": h.query_teacher, "teacher_audit": auditor().audit_teacher,
                 "fits": h.train_fit, "fits_audit": auditor().replay_fit,
                 "games": h.play_game, "game_audit": auditor().replay_game}
    return functions[kind](*args)


def stage_dependencies(stage, upstream, base=OUTPUT):
    items = [h.artifact(upstream)]
    selected = base / "selected_lock.json"
    if stage.startswith(("probe_", "main_")) and Path(upstream).resolve() != selected.resolve():
        items.append(h.artifact(selected))
    return items


def run_jobs(stage, kind, arguments, binding, upstream, workers=8):
    assert validate_freeze() == binding
    launch_path = OUTPUT / f"{stage}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "stage": stage, "time": old.now(),
                 "dependencies": stage_dependencies(stage, upstream, OUTPUT), "jobs": len(arguments)})
    runtime = windows.configure_spawn()  # Every new pool, not only initial startup.
    started = time.perf_counter()
    files = []
    event(stage, jobs=len(arguments), workers=workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, binding)) as pool:
        futures = [pool.submit(dispatch, kind, args) for args in arguments]
        try:
            for future in as_completed(futures):
                files.append(future.result())
                if len(files) % 25 == 0 or len(files) == len(arguments):
                    event(stage, completed=len(files), total=len(arguments))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    files.sort(key=lambda r: (r.get("index", -1), r.get("arm", ""), r.get("seed", 0)))
    verify_files(files)
    assert len(files) == len(arguments) and validate_freeze() == binding
    launch = old.read_json(launch_path)
    verify_files(launch["dependencies"])
    old.save_new(OUTPUT / f"{stage}_lock.json", {"manifest": binding, "files": files,
                 "launch": h.artifact(launch_path), "runtime": runtime, "elapsed_seconds": time.perf_counter()-started})
    return files


def verify_stage(stage, binding):
    lock = old.read_json(OUTPUT / f"{stage}_lock.json")
    assert lock["manifest"] == binding
    verify_files([lock["launch"], *lock["files"]])
    launch = old.read_json(Path(lock["launch"]["path"]))
    assert launch["manifest"] == binding and launch["stage"] == stage and launch["jobs"] == len(lock["files"])
    verify_files(launch["dependencies"])
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
        assert results[0]["slot_sequence_sha256"] == results[1]["slot_sequence_sha256"]
        assert results[0]["initializer"] == results[1]["initializer"]
        assert results[0]["updates"] == results[1]["updates"] == 716
        assert results[0]["train_rows"] == results[1]["train_rows"] == 11413
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
    verify_teacher_lock(binding)
    for stage in STAGES:
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
        file.write("# Experiment162: fresh student-state teacher feedback\n\n")
        for comparison in h.COMPARISONS:
            r = stats[comparison]
            file.write(f"{comparison}: {100*r['score']:.3f}% score; paired 95% interval [{100*r['ci95'][0]:.3f}, {100*r['ci95'][1]:.3f}]%. Gate passed: {r['pass']}.\n\n")
        file.write("Classification: " + result["classification"] + ".\n\n"
                   "4,800 fresh games; 400 whole seed clusters, retaining both comparisons, three fixed fits and both seats. "
                   "All source tensors/labels independently rebuilt, all six fits/checkpoint selections reproduced bitwise, "
                   "all fresh games independently replayed. The old160 labels retain sampled-inner verification; all2304 new teacher roots have complete "
                   "world/label checks and two prospectively chosen worlds with full branch replay each. Intervals condition on the three fixed fits. Validation combines reused old and held-out fresh source clusters. "
                   "No automatic promotion, extension, successor or monitoring. Failure to meet both gates does not reject distillation generally.\n")
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "all_prescribed_audits_pass": True, "classification": result["classification"],
                 "result": h.artifact(OUTPUT / "result.json"), "report": h.artifact(OUTPUT / "report.md")})
    event("complete", classification=result["classification"])


STAGES = ("source_games", "source_audit", "teacher_probe", "teacher_probe_audit", "teacher_rest",
          "teacher_rest_audit", "fits", "fits_audit", "probe_games", "probe_audit", "main_games", "main_audit")


def seal_samples(binding):
    sources = verify_stage("source_games", binding)["files"]
    audits = verify_stage("source_audit", binding)["files"]
    assert len(sources) == len(audits) == 384
    samples = h.sample_inventory(sources, binding)
    auditor().audit_samples(samples, sources, binding)
    probe = h.probe_indices(samples)
    assert probe == auditor().independent_probe_indices(samples)
    old.save_new(OUTPUT / "samples.json", {"manifest": binding, "samples": samples, "probe_indices": probe,
                 "source_lock": h.artifact(OUTPUT / "source_games_lock.json"),
                 "source_audit_lock": h.artifact(OUTPUT / "source_audit_lock.json")})
    event("samples_sealed", roots=len(samples), train=sum(s["split"] == "train" for s in samples),
          validation=sum(s["split"] == "validation" for s in samples))
    return samples, set(probe)


def teacher_preflight(binding):
    collection = verify_stage("teacher_probe", binding)
    audit = verify_stage("teacher_probe_audit", binding)
    assert len(collection["files"]) == len(audit["files"]) == 18
    size = sum(p.stat().st_size for p in (OUTPUT / "teacher").rglob("*") if p.is_file())
    free = shutil.disk_usage(OUTPUT).free
    required = max(10*1024**3, size*128*1.5+5*1024**3)
    estimate = (collection["elapsed_seconds"]+audit["elapsed_seconds"])*127
    result = {"manifest": binding, "time": old.now(), "integrity_pass": True, "outcomes_used_for_gate": False,
              "estimated_remaining_teacher_seconds": estimate, "estimated_teacher_bytes": size*128,
              "disk_free_bytes": free, "required_free_bytes": required, "passed": free >= required,
              "teacher_probe_audit_lock": h.artifact(OUTPUT / "teacher_probe_audit_lock.json")}
    old.save_new(OUTPUT / "teacher_preflight.json", result)
    assert result["passed"], "Fixed teacher probe storage gate failed"
    event("teacher_preflight_passed", estimated_remaining_teacher_hours=estimate/3600)


def seal_teacher(binding):
    items = []
    coverage = Counter()
    for stage in ("teacher_probe", "teacher_rest"):
        files = verify_stage(stage, binding)["files"]
        audits = verify_stage(stage+"_audit", binding)["files"]
        assert [r["index"] for r in files] == [r["index"] for r in audits]
        for item, check in zip(files, audits):
            receipt = old.read_json(Path(item["path"]))
            verify_files([receipt["root"], receipt["tensor"]])
            report = old.read_json(Path(check["path"]))
            assert report["manifest"] == binding and report["passed"] and report["receipt"] == item
            coverage["roots"] += 1
            coverage["continuations"] += report["continuations"]
            coverage["replayed_branches"] += report["replayed_branches"]
            coverage[report["coverage"]["split"]] += 1
        items.extend(files)
    items.sort(key=lambda r: r["index"])
    assert [r["index"] for r in items] == list(range(2304))
    assert coverage["train"] == 1836 and coverage["validation"] == 468
    old.save_new(OUTPUT / "teacher_lock.json", {"manifest": binding, "files": items, "coverage": dict(coverage),
                 "samples": h.artifact(OUTPUT / "samples.json"),
                 "audits": [h.artifact(OUTPUT / (s+"_audit_lock.json")) for s in ("teacher_probe", "teacher_rest")]})


def verify_teacher_lock(binding):
    data = old.read_json(OUTPUT / "teacher_lock.json")
    assert data["manifest"] == binding
    verify_files([data["samples"], *data["audits"], *data["files"]])
    for item in data["files"]:
        receipt = old.read_json(Path(item["path"]))
        verify_files([receipt["root"], receipt["tensor"]])
    return data


def run():
    log = OUTPUT / "preparation"
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        assert not (OUTPUT / "launch.json").exists(), "One-shot pipeline; no automatic resume"
        binding = validate_freeze()
        old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                     "owner_authorized": True, "windowless": windows.assert_windowless_worker()})
        try:
            specs = old.read_json(OUTPUT / "schedules.json")["source"]
            files = run_jobs("source_games", "games", [("source", s, binding) for s in specs], binding, OUTPUT / "freeze.json")
            verify_stage("source_games", binding)
            run_jobs("source_audit", "game_audit", [("source", s, item, binding) for s, item in zip(specs, files)],
                     binding, OUTPUT / "source_games_lock.json")
            samples, probe = seal_samples(binding)
            for stage, group, upstream in (("teacher_probe", [s for s in samples if s["index"] in probe], OUTPUT / "samples.json"),
                                           ("teacher_rest", [s for s in samples if s["index"] not in probe], OUTPUT / "teacher_preflight.json")):
                files = run_jobs(stage, "teacher", [(s, binding) for s in group], binding, upstream)
                verify_stage(stage, binding)
                run_jobs(stage+"_audit", "teacher_audit", [(s, item, binding) for s, item in zip(group, files)],
                         binding, OUTPUT / (stage+"_lock.json"))
                if stage == "teacher_probe":
                    teacher_preflight(binding)
            seal_teacher(binding)
            fits = [(arm, seed) for arm in h.ARMS for seed in h.SEEDS]
            run_jobs("fits", "fits", fits, binding, OUTPUT / "teacher_lock.json", workers=3)
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
        process = held_launch([sys.executable, "-u", REPOSITORY / "run_student_refresh_162.py", "run"],
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
