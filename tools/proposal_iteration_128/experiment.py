"""One detached, frozen learning generation with conditional second arena."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from tools.proposal_iteration_128.common import (
    ROOT, OUTPUT, FITS, CONFIG, old, windows, require, artifact, verified, verify_hash,
)
from tools.proposal_iteration_128 import arena, learning, verify
from tools.learned_proposal_128 import experiment as parent

STAGES = ("probe_a", "stage_a", "probe_b", "stage_b")
CHECK_TARGETS = [
    "tools/proposal_iteration_128/test_learning.py",
    "tools/proposal_iteration_128/test_arena.py",
    "tools/proposal_iteration_128/test_experiment.py",
    "tests/test_windowless_workers_160.py",
    "tools/learned_proposal_128/test_experiment.py",
    "tools/learned_proposal_128/test_verify.py",
]
EXTRAS = (
    "run_proposal_iteration_128.py", "docs/engineering/proposal_iteration_128.md",
    "docs/strategy/next_after_learned128.md", "docs/engineering/verification_policy.md",
)
PARENT_BINDING = "a9a541cb44599e4ed6c6076f05c5b8b29692859ad9c667979b17cf16dd754773"


def identity():
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += sorted((ROOT / "scripts/analysis").glob("*.py"))
    paths += sorted((ROOT / "scripts/experiments").glob("*.py"))
    for folder in ("tools/proposal_iteration_128", "tools/learned_proposal_128"):
        paths += sorted((ROOT / folder).glob("*.py"))
    paths += [ROOT / name for name in sorted(set(EXTRAS) | set(CHECK_TARGETS))]
    return {"config": CONFIG, "fits": list(FITS), "checks": CHECK_TARGETS,
            "dependencies": {p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    require(parent.validate_freeze() == PARENT_BINDING, "Completed parent identity changed")
    final = old.read_json(parent.OUTPUT / "final_status.json")
    require(final["manifest"] == PARENT_BINDING and final["complete"]
            and final["all_prescribed_audits_pass"] and not final["full_inner_replay"]
            and final["classification"] == "positive_practical_progress", "Parent not accepted")
    manifest = verified(old.read_json(parent.OUTPUT / "freeze.json")["manifest"])
    files = [*manifest["parent"]["files"], manifest["checks"], manifest["schedule"],
             manifest["engineering_decision"], *verified(manifest["checks"])["logs"],
             *(artifact(parent.OUTPUT / n) for n in
               ("manifest.json", "freeze.json", "final_status.json", "result.json",
                "final_audit.json", "report.md", "operational_decision.json")),
             *learning.source_dependencies()]
    files.extend({"path": p, "sha256": h} for p, h in manifest["identity"]["dependencies"].items())
    for key in ("result", "audit", "report"):
        verify_hash(final[key])
    audit = verified(final["audit"])
    require(audit["passed"] and audit["result_sha256"] == final["result"]["sha256"]
            == "35769af8ecc525851f2b3da1f4e7aa8ab1699e4db6b5590206d06b212d329d71",
            "Wrong final parent result")
    for stage in ("probe", "main"):
        for kind in ("games", "audits"):
            lock = parent.stage_lock(stage, kind, PARENT_BINDING)
            files += [artifact(parent.OUTPUT / f"{stage}_{kind}_lock.json"), lock["launch"], *lock["files"]]
        files += [artifact(parent.OUTPUT / n) for n in
                  (f"{stage}_complete.json", f"{stage}/audit_sample.json", f"{stage}/audit_sample_seal.json")]
    unique = {}
    for item in files:
        verify_hash(item)
        if item["path"] in unique:
            require(unique[item["path"]]["sha256"] == item["sha256"], "Contradictory ancestor hash")
        unique[item["path"]] = {k: item[k] for k in ("path", "sha256")}
    return {"binding": PARENT_BINDING, "files": list(unique.values()),
            "scope": "Completed learned128 prescribed sampled audit; old161 export/fit lineage retained."}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    args = ["-q", *CHECK_TARGETS]
    code = "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(" + repr(args) + "))"
    start = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Source changed during checks")
    old.save_new(folder / "checks.json", {"identity": before, "targets": CHECK_TARGETS,
                 "exit_code": result.returncode, "seconds": time.perf_counter()-start,
                 "logs": [artifact(folder / n) for n in ("stdout.txt", "stderr.txt")]})
    return result.returncode


def prepare(checks_path):
    require(not (OUTPUT / "manifest.json").exists(), "Already prepared; preserve frozen study")
    checks = old.read_json(checks_path)
    require(checks["identity"] == identity() and checks["targets"] == CHECK_TARGETS
            and checks["exit_code"] == 0, "Current complete passing checks required")
    for item in checks["logs"]:
        verify_hash(item)
    parents = parent_evidence()
    schedule = {s: arena.specs(s) for s in STAGES}
    require(all(rows == verify.independent_specs(s) for s, rows in schedule.items()), "Schedule mismatch")
    old.save_new(OUTPUT / "schedule.json", schedule)
    manifest = {"identity": identity(), "parent": parents, "checks": artifact(checks_path),
                "schedule": artifact(OUTPUT / "schedule.json"), "owner_authorized": True,
                "authorization": "User requested starting the proposed complete staged experiment on2026-09-24."}
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": artifact(OUTPUT / "manifest.json"),
                 "binding": old.digest(manifest), "time": old.now(), "owner_authorized": True})
    print({"prepared": True, "binding": old.digest(manifest)}, flush=True)


def validate_freeze():
    freeze = old.read_json(OUTPUT / "freeze.json")
    manifest = verified(freeze["manifest"])
    require(freeze["owner_authorized"] and old.digest(manifest) == freeze["binding"], "Freeze mismatch")
    require(old.digest(manifest["identity"]) == old.digest(identity()), "Frozen code/config/runtime changed")
    checks = verified(manifest["checks"])
    require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS
            and old.digest(checks["identity"]) == old.digest(identity()), "Frozen checks changed")
    require(verified(manifest["schedule"]) == {s: arena.specs(s) for s in STAGES}, "Schedule changed")
    for item in [*checks["logs"], *manifest["parent"]["files"]]:
        verify_hash(item)
    return freeze["binding"]


def event(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    old.atomic_write(OUTPUT / "progress.json", row)
    print(row, flush=True)


def budget_boundary():
    launch = old.read_json(OUTPUT / "launch.json")
    require(time.time()-launch["epoch_seconds"] <= CONFIG["maximum_runtime_hours"]*3600,
            "Frozen216-hour ceiling crossed; preserve partial evidence and stop")
    require(shutil.disk_usage(ROOT).free >= CONFIG["disk_reserve_bytes"], "Frozen5GiB reserve crossed")


def worker_initialize(stage, kind, binding, base):
    folder = Path(base) / "workers" / f"{stage}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt", folder / f"{os.getpid()}_stderr.txt")
    if kind in ("games", "audits"):
        require(old.read_json(Path(base) / "models_lock.json")["manifest"] == binding, "Worker model binding mismatch")
        if stage != "test":
            learned = old.read_json(Path(base) / "learning_complete.json")
            require(learned["manifest"] == binding, "Learning/worker binding mismatch")
            require(verify_hash(learned["models"]) == (Path(base) / "models_lock.json").resolve(), "Worker registry path")
        arena.initialize_models(base=base)
    else:
        old.initialize_worker()
    require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "Worker arithmetic")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, "kind": kind, **health})


def collect_job(stage, spec, binding):
    arena.play_game(stage, spec, binding)
    return artifact(arena.game_path(stage, spec["index"]), index=spec["index"])


def audit_job(stage, item, binding):
    value = verify.audit_game(stage, item["index"], item["sha256"], binding)
    path = OUTPUT / stage / "audits" / f"game_{item['index']:04d}.json"
    old.save_new(path, {"manifest": binding, "game_artifact": item, "passed": True,
                        "full_inner_replay": False, **value})
    return artifact(path, index=item["index"], continuations=sum(r["continuations"] for r in value["roots"]),
                    replayed_branches=sum(r["replayed_branches"] for r in value["roots"]))


def fit_job(stage, spec, binding):
    item = learning.fit_job(spec["fit"], spec["arm"], binding, OUTPUT)
    return {**item, "index": spec["index"]}


def fit_audit_job(stage, spec, binding):
    item = learning.audit_fit_job(spec["fit"], spec["arm"], binding, OUTPUT)
    return {**item, "index": spec["index"]}


def stage_lock(stage, kind, binding):
    lock = old.read_json(OUTPUT / f"{stage}_{kind}_lock.json")
    count = 6 if stage == "learning" else CONFIG["games"][stage]
    require(lock["manifest"] == binding and len(lock["files"]) == count
            and [r["index"] for r in lock["files"]] == list(range(count)), "Stage completeness")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["stage"] == stage and launch["kind"] == kind, "Launch identity")
    for item in [*lock["files"], *launch["dependencies"]]:
        verify_hash(item)
    return lock


def run_pool(stage, kind, arguments, binding, upstream):
    require(validate_freeze() == binding, "Stage freeze mismatch")
    budget_boundary()
    started = time.perf_counter()
    launch_path = OUTPUT / f"{stage}_{kind}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "time": old.now(), "stage": stage, "kind": kind,
                              "dependencies": [artifact(p) for p in upstream]})
    if kind == "games":
        old.save_new(OUTPUT / f"{stage}_launch.json", {"manifest": binding, "time": old.now(), "schedule": arena.specs(stage)})
    runtime = windows.configure_spawn()
    event(f"{stage}_{kind}", completed=0, total=len(arguments))
    task = {"games": collect_job, "audits": audit_job, "fits": fit_job, "fit_audits": fit_audit_job}[kind]
    workers = CONFIG["fit_workers"] if stage == "learning" else CONFIG["workers"]
    files = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, kind, binding, OUTPUT)) as pool:
        jobs = [pool.submit(task, stage, arg, binding) for arg in arguments]
        try:
            for job in as_completed(jobs):
                files.append(job.result())
                event(f"{stage}_{kind}", completed=len(files), total=len(arguments))
                budget_boundary()
        except BaseException:
            for job in jobs:
                job.cancel()
            # Running single-game/single-fit jobs finish; their evidence remains.
            raise
    files.sort(key=lambda r: r["index"])
    require(len(files) == len(arguments) and len({r["index"] for r in files}) == len(files), "Incomplete jobs")
    for item in files:
        verify_hash(item)
    require(validate_freeze() == binding, "Frozen stage dependencies changed")
    old.save_new(OUTPUT / f"{stage}_{kind}_lock.json", {"manifest": binding, "launch": artifact(launch_path),
                 "files": files, "runtime": runtime, "elapsed_seconds": time.perf_counter()-started})
    return files


def prepare_learning(binding):
    event("extracting_and_checking_training_data", expected_new_roots=11951)
    old.initialize_worker()
    learning.prepare_data(binding, OUTPUT)
    require(validate_freeze() == binding, "Source changed during data export")
    specs = [{"index": i, "fit": fit, "arm": arm}
             for i, (fit, arm) in enumerate((f, a) for f in FITS for a in ("treatment", "replay"))]
    run_pool("learning", "fits", specs, binding, [OUTPUT / "freeze.json", OUTPUT / "data_lock.json"])
    run_pool("learning", "fit_audits", specs, binding,
             [OUTPUT / "data_lock.json", OUTPUT / "learning_fits_lock.json"])
    models_item = learning.seal_models(binding, OUTPUT)
    models = verified(models_item)
    old.save_new(OUTPUT / "learning_complete.json", {"manifest": binding, "time": old.now(),
                 "data": artifact(OUTPUT / "data_lock.json"), "models": models_item,
                 "fits": artifact(OUTPUT / "learning_fits_lock.json"),
                 "fit_audits": artifact(OUTPUT / "learning_fit_audits_lock.json")})
    unchanged = all(models["models"][f"treatment_{fit}"]["unchanged_from_anchor"] for fit in FITS)
    event("learning_complete", all_treatments_unchanged=unchanged)
    return unchanged


def checked_audit_coverage(receipts, sample, sha):
    return parent.checked_audit_coverage(receipts, sample, sha)


def validate_learning(binding):
    learned = old.read_json(OUTPUT / "learning_complete.json")
    require(learned["manifest"] == binding, "Learning binding changed")
    for key in ("data", "models", "fits", "fit_audits"):
        verify_hash(learned[key])
    require(verify_hash(learned["models"]) == (OUTPUT / "models_lock.json").resolve(), "Sealed registry path")
    models = verified(learned["models"])
    require(models["manifest"] == binding and models["equal_dose_verified"]
            and models["data_lock"] == learned["data"], "Model seal lineage")
    require(set(models["models"]) == {f"{a}_{f}" for a in ("anchor", "treatment", "replay") for f in FITS}, "Model registry completeness")
    for item in [*models["models"].values(), *models["fit_results"], *models["fit_audits"]]:
        verify_hash(item)
    for kind in ("fits", "fit_audits"):
        stage_lock("learning", kind, binding)
    return learned


def stage_b_authorized(binding):
    audit = old.read_json(OUTPUT / "stage_a_final_audit.json")
    require(audit["manifest"] == binding and audit["passed"] and not audit["full_inner_replay"], "Stage A audit not accepted")
    require(verify_hash(audit["result"]) == (OUTPUT / "stage_a_result.json").resolve(), "Stage A result not sealed")
    result = verified(audit["result"])
    require(result["manifest"] == binding and result["stage"] == "stage_a"
            and result["all_prescribed_audits_pass"] and result["stage_pass"]
            and result["share"] >= .55 and result["ci95"][0] > .5, "Stage B requires verified practical progress")
    verify.compare({k: result[k] for k in audit["independent_statistics"]}, audit["independent_statistics"])
    require(verify_hash(audit["complete"]) == (OUTPUT / "stage_a_complete.json").resolve(), "Stage A completion changed")
    close_stage("stage_a", binding)
    return True


def build_stage(stage, binding):
    validate_learning(binding)
    upstream = [OUTPUT / "freeze.json", OUTPUT / "schedule.json", OUTPUT / "learning_complete.json",
                OUTPUT / "models_lock.json"]
    if stage in ("probe_b", "stage_b"):
        stage_b_authorized(binding)
        upstream += [OUTPUT / "stage_a_result.json", OUTPUT / "stage_a_final_audit.json"]
    if stage.startswith("stage_"):
        gate = OUTPUT / f"{stage}_operational_decision.json"
        require(old.read_json(gate)["reasonable"], "Operational gate has not passed")
        upstream.append(gate)
    games = run_pool(stage, "games", arena.specs(stage), binding, upstream)
    sample_item = verify.seal_sample(stage, games, binding)
    sample = verify.validate_sample(stage, binding)
    audits = run_pool(stage, "audits", games, binding,
                      [OUTPUT / f"{stage}_games_lock.json", verify_hash(sample_item),
                       OUTPUT / stage / "audit_sample_seal.json", OUTPUT / "models_lock.json"])
    for item, game in zip(audits, games):
        receipt = verified(item)
        require(receipt["manifest"] == binding and receipt["passed"] and not receipt["full_inner_replay"]
                and receipt["game_artifact"] == game, "Audit receipt linkage")
    coverage = checked_audit_coverage([verified(item) for item in audits], sample, sample_item["sha256"])
    old.save_new(OUTPUT / f"{stage}_complete.json", {"manifest": binding, "time": old.now(),
                 "games": len(games), "full_inner_replay": False, "sample": sample_item,
                 "coverage": coverage, "game_lock": artifact(OUTPUT / f"{stage}_games_lock.json"),
                 "audit_lock": artifact(OUTPUT / f"{stage}_audits_lock.json")})


def cost_projection(game_worker_wall, audit_worker_wall, stress_extra_wall, stress_roots,
                    size_bytes, free_bytes, elapsed_seconds, remaining_stages, probe_seconds):
    values = (game_worker_wall, audit_worker_wall, stress_extra_wall, size_bytes, free_bytes,
              elapsed_seconds, probe_seconds)
    require(all(type(x) in (int, float) and np.isfinite(x) and x >= 0 for x in values), "Invalid cost evidence")
    require(game_worker_wall > 0 and audit_worker_wall >= stress_extra_wall > 0
            and 0 < stress_roots <= 162 and remaining_stages in (1, 2), "Incomplete probe costs")
    ratio = 402/12
    game_seconds = ratio*game_worker_wall/8
    audit_seconds = ratio*(audit_worker_wall-stress_extra_wall)/8 + 162*stress_extra_wall/stress_roots/8
    forecast = 1.25*(game_seconds+audit_seconds)
    projected_bytes = int(1.5*ratio*size_bytes + 512*1024**2)
    remaining_forecast = remaining_stages*forecast + (remaining_stages-1)*1.25*probe_seconds
    remaining_bytes = remaining_stages*projected_bytes + (remaining_stages-1)*size_bytes
    reasonable = (forecast <= CONFIG["maximum_forecast_hours_per_stage"]*3600
                  and elapsed_seconds+remaining_forecast <= CONFIG["maximum_runtime_hours"]*3600
                  and free_bytes-remaining_bytes >= CONFIG["disk_reserve_bytes"])
    return {"forecast_seconds": forecast, "forecast_hours": forecast/3600,
            "projected_main_bytes": projected_bytes, "remaining_stages_reserved": remaining_stages,
            "remaining_forecast_hours": remaining_forecast/3600, "projected_remaining_bytes": remaining_bytes,
            "free_bytes": free_bytes, "elapsed_seconds": elapsed_seconds, "reasonable": bool(reasonable),
            "game_worker_wall": game_worker_wall, "audit_worker_wall": audit_worker_wall,
            "stress_extra_wall": stress_extra_wall, "probe_stress_roots": stress_roots,
            "probe_seconds": probe_seconds}


def operational_decision(stage, binding):
    require(stage in ("stage_a", "stage_b"), "Invalid operational stage")
    probe = "probe_a" if stage == "stage_a" else "probe_b"
    games, audits = (stage_lock(probe, kind, binding) for kind in ("games", "audits"))
    receipts = [verified(item) for item in audits["files"]]
    stress = [r for a in receipts for r in a["roots"] if r["stress"]]
    size = sum(p.stat().st_size for p in (OUTPUT / probe).rglob("*") if p.is_file())
    value = {"manifest": binding, "time": old.now(), "outcome_blind": True, "stage": stage,
             "probe": artifact(OUTPUT / f"{probe}_complete.json"),
             **cost_projection(sum(verified(i)["timing"]["wall"] for i in games["files"]),
                               sum(a["audit_timing"]["wall"] for a in receipts),
                               sum(r["audit_timing"]["stress_extra_replay_wall"] for r in stress),
                               len(stress), size, shutil.disk_usage(ROOT).free,
                               time.time()-old.read_json(OUTPUT / "launch.json")["epoch_seconds"],
                               2 if stage == "stage_a" else 1,
                               games["elapsed_seconds"]+audits["elapsed_seconds"]),
             "caveat": "Outcome-blind probe;25%time/50%storage allowance. Stage A reserves both potential arenas; Stage B rechecks remaining budget."}
    old.save_new(OUTPUT / f"{stage}_operational_decision.json", value)
    if not value["reasonable"]:
        old.save_new(OUTPUT / "operational_hold.json", value)
        event("operational_hold", arena_stage=stage, forecast_hours=value["forecast_hours"])
        return False
    event("probe_passed", arena_stage=stage, forecast_hours=value["forecast_hours"])
    return True


def close_stage(stage, binding):
    """Read-only final linkage/coverage closure, without simulations or statistics."""
    sample = verify.validate_sample(stage, binding)
    games, audits = (stage_lock(stage, kind, binding) for kind in ("games", "audits"))
    receipts = [verified(item) for item in audits["files"]]
    coverage = checked_audit_coverage(receipts, sample, old.file_hash(OUTPUT / stage / "audit_sample.json"))
    for game_item, receipt in zip(games["files"], receipts):
        require(receipt["manifest"] == binding and receipt["game_artifact"] == game_item
                and receipt["passed"] and not receipt["full_inner_replay"], "Final audit linkage")
        game = verified(game_item)
        require(len(game["trace"]) == len(receipt["roots"]), "Final root count")
        for move, root in zip(game["trace"], receipt["roots"]):
            require(move["root_sha256"] == root["sha256"] == old.file_hash(arena.root_path(stage, game_item["index"], move["turn"])),
                    "Final root mutated")
    return games, audits, receipts, coverage


def finish_stage(stage, binding):
    games, audits, receipts, coverage = close_stage(stage, binding)
    rows = [r["game"] for r in receipts]
    summary = arena.statistics(rows, stage)
    independent = verify.independent_statistics(rows, stage)
    verify.compare(summary, independent)
    require(validate_freeze() == binding, "Changed source before final stage statistics")
    roots = [r for a in receipts for r in a["roots"]]
    result = {"manifest": binding, "time": old.now(), **summary, "rows": rows,
              "all_prescribed_audits_pass": True, "full_inner_replay": False, "coverage": coverage,
              "conditional_on_stage_a_pass": stage == "stage_b",
              "fits": {str(f): {"games": sum(r["fit"] == f for r in rows),
                                  "share": sum(r["score"] for r in rows if r["fit"] == f)/134} for f in FITS},
              "collection_seconds": games["elapsed_seconds"], "audit_seconds": audits["elapsed_seconds"],
              "decision_seconds": {role: {"count": sum(r["planner"] == role for r in roots),
                                          "mean": float(np.mean([r["timing"]["wall"] for r in roots if r["planner"] == role]))}
                                   for role in ("candidate", "reference")},
              "scope": "Fixed-fit fourth-proposal update in128-world planner; no equal-wall-time or broader-opponent claim."}
    old.save_new(OUTPUT / f"{stage}_result.json", result)
    old.save_new(OUTPUT / f"{stage}_final_audit.json", {"manifest": binding, "passed": True, "time": old.now(),
                 "result": artifact(OUTPUT / f"{stage}_result.json"), "independent_statistics": independent,
                 "complete": artifact(OUTPUT / f"{stage}_complete.json"), "coverage": coverage,
                 "full_inner_replay": False})
    event(f"{stage}_verified", score=summary["share"], stage_pass=summary["stage_pass"])
    return result


def finish(binding, classification, stages):
    require(validate_freeze() == binding, "Frozen source changed before finalization")
    validate_learning(binding)
    results = {s: artifact(OUTPUT / f"{s}_result.json") for s in stages}
    audits = {s: artifact(OUTPUT / f"{s}_final_audit.json") for s in stages}
    for s in stages:
        for part in (("probe_a", "stage_a") if s == "stage_a" else ("probe_b", "stage_b")):
            close_stage(part, binding)
        require(verified(audits[s])["result"] == results[s], "Final result/audit mismatch")
    result = {"manifest": binding, "time": old.now(), "classification": classification,
              "stage_results": results, "all_prescribed_audits_pass": True,
              "full_inner_replay": False, "conditional_stage_b_run": "stage_b" in stages,
              "learning": artifact(OUTPUT / "learning_complete.json"),
              "scope": "One controlled new-corpus learning generation; controls extra old-data practice, not same-state teacher quality alone."}
    old.save_new(OUTPUT / "result.json", result)
    old.save_new(OUTPUT / "final_audit.json", {"manifest": binding, "passed": True,
                 "time": old.now(), "result": artifact(OUTPUT / "result.json"), "stages": audits,
                 "learning": artifact(OUTPUT / "learning_complete.json"), "full_inner_replay": False})
    lines = ["# Controlled proposal learning at128 worlds", "", f"Classification: {classification}.", ""]
    for stage in stages:
        r = verified(results[stage])
        lines += [f"{stage}: {r['wins']}W/{r['draws']}D/{r['losses']}L in402 games. Score {r['share']:.4%}; paired95% interval {r['ci95']}. Stage pass: {r['stage_pass']}.", ""]
    if not stages:
        lines += ["All three selected treatment tensors equal their unchanged anchors. No identical-player arena was run.", ""]
    lines += ["Stage B runs only after verified practical progress in Stage A. All actual games checked; eight of128 worlds at each root plus selected full stress roots replayed. Shared rules/encoding/neural primitives remain trusted dependencies. See stage results for recorded and replayed counts. No automatic promotion, further generation or monitoring.", ""]
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "classification": classification, "all_prescribed_audits_pass": True,
                 "full_inner_replay": False, "result": artifact(OUTPUT / "result.json"),
                 "audit": artifact(OUTPUT / "final_audit.json"), "report": artifact(OUTPUT / "report.md"),
                 "automatic_successor": False})
    event("complete", classification=classification)


def run():
    log = OUTPUT / "runtime"
    log.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot run; preserve partial work")
        binding = validate_freeze()
        try:
            with parent.inhibit_system_sleep() as sleep_receipt:
                old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                             "epoch_seconds": time.time(), "sleep_inhibition": sleep_receipt,
                             "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
                if prepare_learning(binding):
                    finish(binding, "no_selected_model_update", [])
                    return
                build_stage("probe_a", binding)
                if not operational_decision("stage_a", binding):
                    return
                build_stage("stage_a", binding)
                result_a = finish_stage("stage_a", binding)
                if not result_a["stage_pass"]:
                    finish(binding, "practical_progress_not_established", ["stage_a"])
                    return
                build_stage("probe_b", binding)
                if not operational_decision("stage_b", binding):
                    return
                build_stage("stage_b", binding)
                result_b = finish_stage("stage_b", binding)
                label = "positive_controlled_iteration" if result_b["stage_pass"] else "progress_new_corpus_advantage_not_established"
                finish(binding, label, ["stage_a", "stage_b"])
        except BaseException:
            old.save_new(OUTPUT / "failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure="failure.json")
            raise


def launch():
    binding = validate_freeze()
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder / "launcher.lock"):
        require(not (folder / "process.json").exists() and not (OUTPUT / "launch.json").exists(), "Already launched")
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_proposal_iteration_128.py", "run"],
                                          folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic()+60
        while not (OUTPUT / "launch.json").exists():
            require(time.monotonic() < deadline, "No acknowledgement; inspect evidence before any retry")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT / "launch.json")
        require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding, "Supervisor acknowledgement")
        old.save_new(folder / "acknowledged.json", {"time": old.now(), "pid": process["pid"], "launch": artifact(OUTPUT / "launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        require(args.checks is not None, "Provide a new check directory")
        return check(args.checks)
    if args.command == "prepare":
        require(args.checks is not None, "Provide current checks.json")
        prepare(args.checks)
    elif args.command == "launch":
        launch()
    else:
        run()
    return 0
