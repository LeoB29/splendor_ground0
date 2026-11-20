"""160: one frozen, windowless, paired gameplay test of 128 versus 32 worlds."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
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

from splendor_ai.alphazero import world_budget_160 as h
from splendor_ai.alphazero import windowless_workers_160 as windows

old = h.old
OUTPUT = h.OUTPUT
REPOSITORY = Path(__file__).resolve().parents[3]
PROTOCOL = Path("docs/engineering/world_budget_160.md")
VERIFIER = Path("scripts/analysis/verify_world_budget_160.py")
INVENTORY = [Path(p) for p in (
    "src/splendor_ai/alphazero/run_world_budget.py",
    "src/splendor_ai/alphazero/world_budget_160.py",
    "src/splendor_ai/alphazero/windowless_workers_160.py",
    "run_world_budget_160.py", "tests/test_world_budget_160.py",
    "tests/test_windowless_workers_160.py", "tests/test_world_budget_runner_160.py",
)] + [PROTOCOL, VERIFIER]
BASELINE_BINDING = "b287ea26424623b3825e49404467329162fcefb3e114b0064c49bc11b867e8a0"
BASELINE_RESULT = "1fbe33e2582c64f251eb80b25d5bc619f89425f95a0164ca12dceebe65097f98"
BASELINE_AUDIT = "c4e943e32678501e4952233db8a480155182104e1c42343bc87d5198d3469f7d"
CONFIG = {
    "games": 400, "probe_games": 8, "workers": 8, "torch_threads": 1,
    "candidate_worlds": 128, "reference_worlds": 32,
    "environment_seed": {"main": 1600000000, "probe": 1609000000},
    "stream_tag": "160-root-stream-v1", "audit_tag": "160-audit-world-v1",
    "replay_worlds_per_root": 2, "bootstrap_seed": 1603000000,
    "bootstrap_resamples": 50000, "bootstrap_chunk": 1000,
    "minimum_share": .55, "minimum_ci_lower_exclusive": .50,
    "minimum_free_bytes": 10 * 1024**3, "disk_reserve_bytes": 5 * 1024**3,
    "disk_projection_multiplier": 1.5,
    "training": False, "arena": True, "promotion": False,
    "automatic_successor": False, "automatic_resume": False,
    "scheduled_monitoring": False, "full_inner_replay": False,
}


def event(stage, **values):
    print({"time": old.now(), "stage": stage, **values}, flush=True)


def auditor():
    name = "audit160_runtime"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPOSITORY / VERIFIER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def specs(stage):
    if stage not in ("main", "probe"):
        raise ValueError("Only fixed probe and main panels are production schedules")
    count = CONFIG["games" if stage == "main" else "probe_games"]
    return [{"index": i, "pair": i // 2, "candidate_seat": i % 2,
             "environment_seed": CONFIG["environment_seed"][stage] + i // 2,
             "search_stream": old.digest([CONFIG["stream_tag"], stage, i // 2])}
            for i in range(count)]


def identity():
    assert h.baseline.check_identity() == BASELINE_BINDING
    folder = h.baseline.OUTPUT
    assert old.file_hash(folder / "result.json") == BASELINE_RESULT
    assert old.file_hash(folder / "main_audit.json") == BASELINE_AUDIT
    final = old.read_json(folder / "final_status.json")
    assert final["complete"] and final["integrity_pass"]
    assert final["manifest"] == BASELINE_BINDING
    assert final["result_sha256"] == BASELINE_RESULT and final["audit_sha256"] == BASELINE_AUDIT
    # Reuse the independently implemented 158 replay infrastructure. Its code
    # dependency freeze is checked; its provisional gameplay score is no label.
    audit_binding = auditor().a158.p.validate_freeze()
    return {"format": "world_budget_160_v1", "config": CONFIG,
            "baseline_binding": BASELINE_BINDING, "baseline_result_sha256": BASELINE_RESULT,
            "baseline_audit_sha256": BASELINE_AUDIT,
            "audit_infrastructure_158_binding": audit_binding,
            "dependencies": {p.as_posix(): old.file_hash(p) for p in INVENTORY},
            "windowless_runtime": windows.configure_spawn(),
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def statistics(games):
    """One independent resampling unit is the complete seat-swapped seed pair."""
    if not games or len(games) % 2:
        raise ValueError("A nonempty complete paired panel is required")
    rows = sorted(games, key=lambda row: row["index"])
    n, pairs = len(rows), len(rows) // 2
    assert [g["index"] for g in rows] == list(range(n))
    assert all(g["pair"] == i // 2 and g["seat"] == i % 2
               and g["score"] in (0., .5, 1.) for i, g in enumerate(rows))
    halfpoints = np.array([int(2 * g["score"]) for g in rows], dtype=np.int64)
    pair_units = halfpoints.reshape(pairs, 2).sum(axis=1)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    batches = []
    for start in range(0, CONFIG["bootstrap_resamples"], CONFIG["bootstrap_chunk"]):
        count = min(CONFIG["bootstrap_chunk"], CONFIG["bootstrap_resamples"] - start)
        draw = rng.integers(pairs, size=(count, pairs))
        batches.append(pair_units[draw].sum(axis=1) / (4 * pairs))
    ci = np.quantile(np.concatenate(batches), [.025, .975]).tolist()
    share = int(halfpoints.sum()) / (2 * n)
    classification = ("positive" if int(halfpoints.sum()) * 10 >= 11 * n and ci[0] > .50
                      else "below_material_target" if ci[1] < .55 else "inconclusive")
    return {"games": n, "pairs": pairs, "wins": sum(g["score"] == 1 for g in rows),
            "draws": sum(g["score"] == .5 for g in rows),
            "losses": sum(g["score"] == 0 for g in rows), "score_share": share, "ci95": ci,
            "bootstrap_resamples": CONFIG["bootstrap_resamples"],
            "bootstrap_seed": CONFIG["bootstrap_seed"], "classification": classification,
            "mean_score_margin": float(np.mean([g["margin"] for g in rows])),
            "seats": {str(seat): {"games": pairs,
                      "score_share": sum(g["score"] for g in rows if g["seat"] == seat) / pairs,
                      "mean_score_margin": float(np.mean([g["margin"] for g in rows if g["seat"] == seat]))}
                      for seat in (0, 1)}}


def worker_initialize(stage, binding):
    folder = OUTPUT / "runtime" / stage
    folder.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    info = windows.assert_windowless_worker(folder / f"{pid}.stdout.txt", folder / f"{pid}.stderr.txt")
    h.initialize_worker()
    assert torch.get_num_threads() == 1
    old.save_new(folder / f"{pid}.json", {**info, "time": old.now(), "manifest": binding,
                 "torch_threads": torch.get_num_threads(), "model_device": "cpu"})


def collect_task(stage, spec, binding):
    return h.play_game(stage, spec, binding)


def audit_path(stage, index):
    return OUTPUT / stage / "audits" / f"game_{index:04d}.json"


def audit_task(stage, spec, expected_sha, binding):
    path = audit_path(stage, spec["index"])
    assert not path.exists(), "No automatic audit reuse or retry"
    checked = auditor().audit_game(stage, spec["index"], expected_sha, binding)
    old.save_new(path, {"manifest": binding, "game_sha256": expected_sha, "audit": checked})
    return {"index": spec["index"], "sha256": old.file_hash(path)}


def run_stage(stage, binding, audit=False):
    phase = f"{stage}_{'audit' if audit else 'collect'}"
    assert validate_freeze() == binding
    source = None
    if audit:
        source = verify_lock(stage, binding, audit=False)
    else:
        upstream = OUTPUT / ("preflight.json" if stage == "main" else "freeze.json")
        if stage == "main":
            gate = old.read_json(upstream)
            assert gate["manifest"] == binding and gate["passed"]
        old.save_new(OUTPUT / f"{stage}_launch.json", {
            "manifest": binding, "time": old.now(), "stage": stage,
            "specs_sha256": old.digest(specs(stage)), "upstream_path": upstream.as_posix(),
            "upstream_sha256": old.file_hash(upstream)})
    runtime = windows.configure_spawn()
    start = time.perf_counter()
    event(phase + "_started", games=len(specs(stage)), workers=CONFIG["workers"])
    files = {}
    with ProcessPoolExecutor(max_workers=CONFIG["workers"],
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(phase, binding)) as pool:
        futures = [pool.submit(audit_task, stage, spec, source["files"][str(spec["index"])], binding)
                   if audit else pool.submit(collect_task, stage, spec, binding)
                   for spec in specs(stage)]
        try:
            for future in as_completed(futures):
                row = future.result()
                assert str(row["index"]) not in files
                files[str(row["index"])] = row["sha256"]
                event(phase + "_game_completed", index=row["index"], completed=len(files), total=len(futures))
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    assert set(files) == {str(s["index"]) for s in specs(stage)}
    old.save_new(OUTPUT / f"{phase}_lock.json", {"manifest": binding, "time": old.now(),
                 "files": files, "elapsed_seconds": time.perf_counter() - start,
                 "runtime": runtime, "stage_launch_sha256": old.file_hash(OUTPUT / f"{stage}_launch.json"),
                 "source_lock_sha256":
                 old.file_hash(OUTPUT / f"{stage}_collect_lock.json") if audit else None})
    event(phase + "_complete", games=len(files))


def verify_lock(stage, binding, audit):
    phase = f"{stage}_{'audit' if audit else 'collect'}"
    lock = old.read_json(OUTPUT / f"{phase}_lock.json")
    assert lock["manifest"] == binding
    stage_launch = OUTPUT / f"{stage}_launch.json"
    assert lock["stage_launch_sha256"] == old.file_hash(stage_launch)
    launch_record = old.read_json(stage_launch)
    assert launch_record["manifest"] == binding and launch_record["stage"] == stage
    assert launch_record["specs_sha256"] == old.digest(specs(stage))
    assert launch_record["upstream_sha256"] == old.file_hash(Path(launch_record["upstream_path"]))
    assert set(lock["files"]) == {str(s["index"]) for s in specs(stage)}
    for key, sha in lock["files"].items():
        path = audit_path(stage, int(key)) if audit else h.game_path(stage, int(key))
        assert old.file_hash(path) == sha
    if audit:
        assert lock["source_lock_sha256"] == old.file_hash(OUTPUT / f"{stage}_collect_lock.json")
    return lock


def stage_evidence(stage, binding):
    collection = verify_lock(stage, binding, False)
    audit = verify_lock(stage, binding, True)
    games, roots = [], []
    for spec in specs(stage):
        index = spec["index"]
        record = old.read_json(audit_path(stage, index))
        assert record["manifest"] == binding
        assert record["game_sha256"] == collection["files"][str(index)]
        checked = record["audit"]
        assert checked["game"]["index"] == index
        # Check root artifacts again after collection and replay, including their
        # binding into the immutable actual-game trace.
        game = old.read_json(h.game_path(stage, index))
        for move in game["trace"]:
            assert old.file_hash(h.root_path(stage, index, move["turn"])) == move["root_sha256"]
        games.append(checked["game"])
        roots.extend(checked["roots"])
    counts = {key: sum(r[key] for r in roots) for key in ("continuations", "moves", "replayed_branches")}
    counts.update(games=len(games), outer_moves=len(roots),
                  collect_seconds=collection["elapsed_seconds"], audit_seconds=audit["elapsed_seconds"])
    return games, roots, counts


def preflight(binding):
    # This gate does not compute or inspect candidate win shares.
    _, roots, counts = stage_evidence("probe", binding)
    stage_bytes = sum(p.stat().st_size for p in (OUTPUT / "probe").rglob("*") if p.is_file())
    scale = CONFIG["games"] / CONFIG["probe_games"]
    forecast = scale * (counts["collect_seconds"] + counts["audit_seconds"])
    projected_bytes = stage_bytes * scale
    free = shutil.disk_usage(OUTPUT).free
    required = max(CONFIG["minimum_free_bytes"],
                   CONFIG["disk_projection_multiplier"] * projected_bytes + CONFIG["disk_reserve_bytes"])
    result = {"manifest": binding, "time": old.now(), "integrity_pass": True,
              "candidate_score_inspected_for_gate": False, "operational": counts,
              "estimated_main_seconds": forecast, "estimated_main_bytes": projected_bytes,
              "disk_free_bytes": free, "required_free_bytes": required, "disk_pass": free >= required,
              "root_seconds": {str(w): h.baseline.quantiles([r["timing"]["wall"] for r in roots if r["worlds"] == w])
                               for w in h.ROOT_BUDGETS},
              "probe_collect_lock_sha256": old.file_hash(OUTPUT / "probe_collect_lock.json"),
              "probe_audit_lock_sha256": old.file_hash(OUTPUT / "probe_audit_lock.json")}
    result["passed"] = result["integrity_pass"] and result["disk_pass"]
    old.save_new(OUTPUT / "preflight.json", result)
    assert result["passed"], "Timing/integrity preflight failed its fixed storage requirement"
    event("preflight_passed", estimated_main_hours=forecast / 3600, free_gib=free / 1024**3)


def finish(binding):
    games, roots, counts = stage_evidence("main", binding)
    assert len(games) == CONFIG["games"]
    stats = statistics(games)
    auditor().a158.compare(stats, auditor().independent_statistics(games))
    assert validate_freeze() == binding
    result = {"manifest": binding, "time": old.now(), **stats, "rows": games,
              "operational": counts, "root_seconds": {
                  str(w): h.baseline.quantiles([r["timing"]["wall"] for r in roots if r["worlds"] == w])
                  for w in h.ROOT_BUDGETS},
              "audit_scope": "Every outer game, root proposal/score/privacy, world hash, ending, mean and choice; two hash-chosen worlds per root replayed for every proposed action. Not full inner replay.",
              "units": "Candidate128 win=1, draw=0.5, loss=0; seed pair is resampling unit",
              "comparison": "Exact155 with128 worlds versus exact155 with32; unequal compute",
              "main_audit_lock_sha256": old.file_hash(OUTPUT / "main_audit_lock.json")}
    old.save_new(OUTPUT / "result.json", result)
    with (OUTPUT / "report.md").open("x", encoding="utf-8") as report:
        report.write("# Experiment160: 128 versus32 worlds\n\n"
                     f"400 games from200 seat-swapped seed pairs. Candidate score: {100*stats['score_share']:.3f}%, "
                     f"paired95% interval [{100*stats['ci95'][0]:.3f}, {100*stats['ci95'][1]:.3f}]%.\n\n"
                     f"Wins/draws/losses: {stats['wins']}/{stats['draws']}/{stats['losses']}. "
                     f"Classification: **{stats['classification']}**.\n\n"
                     "Positive means observed score>=55% and evidence of superiority over50%; it does not establish "
                     "a true advantage above55%. Below_material_target means the interval upper bound is below55%, "
                     "not necessarily that128 worlds is weaker. Otherwise the result is inconclusive.\n\n"
                     "Only world count changed. This compares playing strength at unequal compute; it cannot establish "
                     "equal-time superiority or superhuman play. Seat and score-margin summaries in result.json are descriptive. "
                     "Every prespecified integrity check completed, including all outer games and two hash-selected worlds "
                     "per root for every proposal. This was sampled inner replay, not full replay. "
                     "No automatic training, promotion, extension, successor or scheduled monitoring.\n")
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "prespecified_audit_pass": True, "full_inner_replay": False,
                 "result_sha256": old.file_hash(OUTPUT / "result.json"),
                 "report_sha256": old.file_hash(OUTPUT / "report.md"),
                 "classification": stats["classification"], "automatic_successor": False})
    event("complete", classification=stats["classification"])


def tests(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    start = time.perf_counter()
    # A GUI-subsystem test supervisor also keeps legacy spawn fixtures windowless.
    code = ("import sys; from splendor_ai.alphazero import windowless_workers_160 as w; "
            "w.configure_spawn(); w.assert_windowless_worker(sys.argv[1],sys.argv[2]); "
            "import pytest; raise SystemExit(pytest.main(['-q']))")
    exe = windows.configure_spawn()["executable"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run([exe, "-u", "-c", code,
                                str((folder / "stdout.txt").resolve()), str((folder / "stderr.txt").resolve())],
                               cwd=REPOSITORY, stdin=subprocess.DEVNULL, creationflags=flags)
    assert identity() == before
    old.save_new(folder / "tests.json", {"identity": before, "exit_code": completed.returncode,
                 "full_repository": True, "elapsed_seconds": time.perf_counter() - start,
                 "stdout_sha256": old.file_hash(folder / "stdout.txt"),
                 "stderr_sha256": old.file_hash(folder / "stderr.txt")})
    event("tests_finished", exit_code=completed.returncode, evidence=str(folder / "tests.json"))
    return completed.returncode


def prepare(test_evidence):
    assert not (OUTPUT / "manifest.json").exists()
    manifest = identity()
    evidence = old.read_json(test_evidence)
    assert evidence["identity"] == manifest and evidence["exit_code"] == 0 and evidence["full_repository"]
    for stream in ("stdout", "stderr"):
        assert evidence[f"{stream}_sha256"] == old.file_hash(Path(test_evidence).parent / f"{stream}.txt")
    schedule = {stage: specs(stage) for stage in ("probe", "main")}
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "schedule.json", schedule)
    old.save_new(OUTPUT / "freeze.json", {"manifest": old.digest(manifest), "time": old.now(),
                 "owner_authorized": True, "schedule_sha256": old.file_hash(OUTPUT / "schedule.json"),
                 "tests_path": Path(test_evidence).as_posix(), "tests_sha256": old.file_hash(test_evidence)})
    event("prepared", binding=old.digest(manifest), games=CONFIG["games"])


def validate_freeze():
    manifest = old.read_json(OUTPUT / "manifest.json")
    assert manifest == identity(), "160 frozen scientific identity changed"
    binding = old.digest(manifest)
    frozen = old.read_json(OUTPUT / "freeze.json")
    assert frozen["manifest"] == binding and frozen["owner_authorized"]
    assert frozen["schedule_sha256"] == old.file_hash(OUTPUT / "schedule.json")
    assert old.read_json(OUTPUT / "schedule.json") == {stage: specs(stage) for stage in ("probe", "main")}
    assert frozen["tests_sha256"] == old.file_hash(Path(frozen["tests_path"]))
    evidence = old.read_json(Path(frozen["tests_path"]))
    assert evidence["identity"] == manifest and evidence["exit_code"] == 0 and evidence["full_repository"]
    for stream in ("stdout", "stderr"):
        assert evidence[f"{stream}_sha256"] == old.file_hash(Path(frozen["tests_path"]).parent / f"{stream}.txt")
    return binding


def run():
    log = OUTPUT / "preparation"
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        assert not (OUTPUT / "launch.json").exists(), "One-shot pipeline; no automatic resume"
        binding = validate_freeze()
        old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                     "owner_authorized": True, "windowless": windows.assert_windowless_worker()})
        try:
            run_stage("probe", binding)
            run_stage("probe", binding, audit=True)
            preflight(binding)
            run_stage("main", binding)
            run_stage("main", binding, audit=True)
            finish(binding)
        except BaseException:
            old.save_new(OUTPUT / "failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


def launch():
    binding = validate_freeze()
    folder = OUTPUT / "preparation"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder / "launcher.lock"):
        assert not (folder / "process.json").exists() and not (OUTPUT / "launch.json").exists()
        command = [sys.executable, "-u", str(REPOSITORY / "run_world_budget_160.py"), "run"]
        process = windows.detached_launch(command, folder / "stdout.txt", folder / "stderr.txt", REPOSITORY)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        event("detached_launch", pid=process["pid"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("tests", "prepare", "launch", "run"))
    parser.add_argument("--test-evidence", type=Path)
    parser.add_argument("--test-dir", type=Path)
    args = parser.parse_args()
    if args.command == "tests":
        assert args.test_dir is not None
        return tests(args.test_dir)
    if args.command == "prepare":
        assert args.test_evidence is not None
        return prepare(args.test_evidence)
    return {"launch": launch, "run": run}[args.command]()
