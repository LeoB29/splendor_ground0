"""One fixed full-game breadth confirmation, with hidden bounded execution."""
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

from .common import (
    ROOT, OUTPUT, SOURCE, SOURCE_BINDING, FITS, CONFIG, old, windows, require, artifact, verified, verify_hash,
    budget_boundary, BudgetExceeded,
)
from . import arena, models, verify
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.learned_proposal_128 import experiment as parent

STAGES = ("probe", "main")
CHECK_TARGETS = [
    "tools/proposal_breadth_arena_128/test_models.py",
    "tools/proposal_breadth_arena_128/test_arena.py",
    "tools/proposal_breadth_arena_128/test_verify.py",
    "tools/proposal_breadth_arena_128/test_experiment.py",
    "tools/proposal_breadth_arena_128/test_integration.py",
    "tools/proposal_breadth_arena_128/test_worker.py",
    "tests/test_windowless_workers_160.py",
]
EXTRAS = ("run_proposal_breadth_arena_128.py", "docs/engineering/proposal_breadth_arena_128.md",
          "docs/strategy/next_after_proposal_breadth_128.md", "docs/engineering/verification_policy.md")
REPORTER = None


def identity():
    previous = old.read_json(SOURCE / "manifest.json")["identity"]
    paths = {ROOT / path for path in previous["dependencies"]}
    for path, expected in previous["dependencies"].items():
        require(old.file_hash(ROOT / path) == expected, "Frozen ancestor dependency changed: "+path)
    paths.update((ROOT / "tools/proposal_breadth_arena_128").glob("*.py"))
    paths.update(ROOT / name for name in (*EXTRAS, *CHECK_TARGETS))
    return {"config": CONFIG, "fits": list(FITS), "checks": CHECK_TARGETS,
            "dependencies": {p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in sorted(paths)},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    files = models.source_evidence()
    for item in files:
        verify_hash(item)
    return {"binding": SOURCE_BINDING, "files": files,
            "scope": "Positive completed breadth screen and unchanged original161/035 identities"}


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
    seeds = [{row["environment_seed"] for row in rows} for rows in schedule.values()]
    require(len(seeds[0]) == 6 and len(seeds[1]) == 201 and seeds[0].isdisjoint(seeds[1]), "Overlapping source namespaces")
    old.save_new(OUTPUT / "schedule.json", schedule)
    manifest = {"identity": identity(), "parent": parents, "checks": artifact(checks_path),
                "schedule": artifact(OUTPUT / "schedule.json"), "owner_authorized": True,
                "authorization": "User requested starting the single proposed breadth gameplay experiment on2026-09-28; no training or automatic successor."}
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": artifact(OUTPUT / "manifest.json"),
                 "binding": old.digest(manifest), "time": old.now(), "owner_authorized": True})
    require(validate_freeze() == old.digest(manifest), "Prepared freeze did not validate")
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
    global REPORTER
    if REPORTER is None:
        REPORTER = StatusReporter(OUTPUT)
    return REPORTER.event(stage, **values)


def worker_initialize(stage, kind, binding, base):
    require(stage in CONFIG["games"] and kind in ("games", "audits"), "Unknown worker role")
    folder = Path(base) / "workers" / f"{stage}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt", folder / f"{os.getpid()}_stderr.txt")
    require(old.read_json(Path(base) / "models_lock.json")["manifest"] == binding, "Worker model binding mismatch")
    arena.initialize_models(base=base)
    require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "Worker arithmetic")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, "kind": kind, **health})


def collect_job(stage, spec, binding, base=OUTPUT, test_snapshot=None):
    arena.play_game(stage, spec, binding, base, test_snapshot)
    return artifact(arena.game_path(stage, spec["index"], base), index=spec["index"])


def audit_job(stage, item, binding, base=OUTPUT, test_snapshot=None):
    value = verify.audit_game(stage, item["index"], item["sha256"], binding, base, test_snapshot)
    path = Path(base) / stage / "audits" / f"game_{item['index']:04d}.json"
    old.save_new(path, {"manifest": binding, "game_artifact": item, "passed": True,
                        "full_inner_replay": False, **value})
    return artifact(path, index=item["index"], continuations=sum(r["continuations"] for r in value["roots"]),
                    replayed_branches=sum(r["replayed_branches"] for r in value["roots"]))


def stage_lock(stage, kind, binding):
    lock = old.read_json(OUTPUT / f"{stage}_{kind}_lock.json")
    count = CONFIG["games"][stage]
    require(lock["manifest"] == binding and len(lock["files"]) == count
            and [r["index"] for r in lock["files"]] == list(range(count)), "Stage completeness")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["stage"] == stage and launch["kind"] == kind, "Launch identity")
    for item in [*lock["files"], *launch["dependencies"]]:
        verify_hash(item)
    return lock


def run_pool(stage, kind, arguments, binding, upstream):
    require(validate_freeze() == binding, "Stage freeze mismatch")
    budget_boundary(OUTPUT)
    started = time.perf_counter()
    launch_path = OUTPUT / f"{stage}_{kind}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "time": old.now(), "stage": stage, "kind": kind,
                              "dependencies": [artifact(p) for p in upstream]})
    if kind == "games":
        old.save_new(OUTPUT / f"{stage}_launch.json", {"manifest": binding, "time": old.now(), "schedule": arena.specs(stage)})
    runtime = windows.configure_spawn()
    event(f"{stage}_{kind}", completed=0, total=len(arguments))
    task = {"games": collect_job, "audits": audit_job}[kind]
    workers = CONFIG["workers"]
    files = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, kind, binding, OUTPUT)) as pool:
        jobs = [pool.submit(task, stage, arg, binding, OUTPUT) for arg in arguments]
        try:
            for job in as_completed(jobs):
                files.append(job.result())
                event(f"{stage}_{kind}", completed=len(files), total=len(arguments))
                budget_boundary(OUTPUT)
        except BaseException:
            for job in jobs:
                job.cancel()
            # Active workers check the same budget between worlds; partial evidence remains.
            raise
    files.sort(key=lambda r: r["index"])
    require(len(files) == len(arguments) and len({r["index"] for r in files}) == len(files), "Incomplete jobs")
    for item in files:
        verify_hash(item)
    require(validate_freeze() == binding, "Frozen stage dependencies changed")
    old.save_new(OUTPUT / f"{stage}_{kind}_lock.json", {"manifest": binding, "launch": artifact(launch_path),
                 "files": files, "runtime": runtime, "elapsed_seconds": time.perf_counter()-started})
    return files


def build_stage(stage, binding):
    models.validate(binding, OUTPUT)
    upstream = [OUTPUT / "freeze.json", OUTPUT / "schedule.json",
                OUTPUT / "models_lock.json"]
    if stage == "main":
        gate = OUTPUT / "operational_decision.json"
        value = old.read_json(gate)
        require(value["manifest"] == binding and value["reasonable"], "Passing operational gate required")
        upstream += [gate, OUTPUT / "probe_complete.json"]
    games = run_pool(stage, "games", arena.specs(stage), binding, upstream)
    sample_item = verify.seal_sample(stage, games, binding, OUTPUT)
    sample = verify.validate_sample(stage, binding, OUTPUT)
    audits = run_pool(stage, "audits", games, binding,
                      [OUTPUT / f"{stage}_games_lock.json", verify_hash(sample_item),
                       OUTPUT / stage / "audit_sample_seal.json", OUTPUT / "models_lock.json"])
    for item, game in zip(audits, games):
        receipt = verified(item)
        require(receipt["manifest"] == binding and receipt["passed"] and not receipt["full_inner_replay"]
                and receipt["game_artifact"] == game, "Audit receipt linkage")
    coverage = checked_audit_coverage([verified(item) for item in audits], sample, sample_item["sha256"])
    old.save_new(OUTPUT / f"{stage}_complete.json", {"manifest": binding, "time": old.now(),
                 "games": len(games), "all_prescribed_audits_pass": True, "full_inner_replay": False, "sample": sample_item,
                 "coverage": coverage, "game_lock": artifact(OUTPUT / f"{stage}_games_lock.json"),
                 "audit_lock": artifact(OUTPUT / f"{stage}_audits_lock.json")})


def checked_audit_coverage(receipts, sample, sample_sha256):
    """Independent accounting against the sealed, outcome-blind inventory."""
    require(all(r["sample_sha256"] == sample_sha256 and not r["full_inner_replay"] for r in receipts), "Audit sample linkage")
    roots = [root for receipt in receipts for root in receipt["roots"]]
    identities = [(r["index"], r["turn"]) for r in roots]
    require(identities == [(r["index"], r["turn"]) for r in sample["inventory"]]
            and len(set(identities)) == len(identities), "Audit root inventory mismatch")
    selections = {r["root_id"]: r for r in sample["selections"]}
    require(len(selections) == len(roots), "Duplicate or missing sample selection")
    for root, metadata in zip(roots, sample["inventory"], strict=True):
        selected = selections[metadata["root_id"]]
        require(root["planner"] == metadata["arm"] and root["fit"] == metadata["fit"], "Audit role/fit mismatch")
        limit = 8 if root["planner"] == "candidate" else 4
        require(type(root["support"]) is int and 1 <= root["support"] <= limit, "Invalid deployed support")
        require(root["sha256"] == metadata["sha256"] and root["support"] == metadata["support"]
                and root["worlds"] == 128 and not root["full_inner_replay"] and not root["full_fixture_replay"]
                and root["ordinary_world_indices"] == selected["ordinary_world_indices"]
                and root["replay_world_indices"] == selected["replay_world_indices"]
                and root["stress"] == selected["stress"], "Audit/sample root mismatch")
        require(root["continuations"] == 128*root["support"]
                and root["replayed_branches"] == len(root["replay_world_indices"])*root["support"], "Delivered continuation mismatch")
    coverage = {"games": len(receipts), "roots": len(roots),
                "recorded_worlds": sum(r["worlds"] for r in roots),
                "recorded_continuations": sum(r["continuations"] for r in roots),
                "replayed_worlds": sum(len(r["replay_world_indices"]) for r in roots),
                "replayed_continuations": sum(r["replayed_branches"] for r in roots),
                "stress_roots": sum(r["stress"] for r in roots)}
    require(all(sample["coverage"][k] == v for k, v in coverage.items()), "Coverage totals mismatch")
    return coverage


def cost_projection(game_worker_wall, audit_worker_wall, stress_extra_wall_by_role,
                    stress_roots_by_role, size_bytes, free_bytes, elapsed_seconds):
    """Use measured work, not probe strength or its oversized stress fraction."""
    require(set(stress_extra_wall_by_role) == set(stress_roots_by_role) == {"candidate", "reference"}, "Missing role timing")
    values = [game_worker_wall, audit_worker_wall, size_bytes, free_bytes, elapsed_seconds,
              *stress_extra_wall_by_role.values(), *stress_roots_by_role.values()]
    require(all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in values), "Invalid probe costs")
    limits = {"candidate": 90, "reference": 81}
    for role in limits:
        require(type(stress_roots_by_role[role]) is int and 0 < stress_roots_by_role[role] <= limits[role]
                and stress_extra_wall_by_role[role] > 0, "Unmeasured required stress role")
    extra = sum(stress_extra_wall_by_role.values())
    require(game_worker_wall > 0 and audit_worker_wall >= extra > 0, "Invalid measured probe work")
    ratio = CONFIG["games"]["main"]/CONFIG["games"]["probe"]
    game_seconds = ratio*game_worker_wall/CONFIG["workers"]
    ordinary_seconds = ratio*(audit_worker_wall-extra)/CONFIG["workers"]
    stress_seconds = sum(limits[role]*stress_extra_wall_by_role[role]/stress_roots_by_role[role]
                         for role in limits)/CONFIG["workers"]
    forecast = CONFIG["forecast_allowance"]*(game_seconds+ordinary_seconds+stress_seconds)
    remaining = forecast+CONFIG["finalization_allowance_seconds"]
    projected = int(1.5*ratio*size_bytes+512*1024**2)
    return {"forecast_seconds": forecast, "forecast_hours": forecast/3600,
            "remaining_forecast_seconds": remaining, "remaining_forecast_hours": remaining/3600,
            "game_worker_wall": game_worker_wall, "audit_worker_wall": audit_worker_wall,
            "stress_extra_wall_by_role": stress_extra_wall_by_role, "probe_stress_roots_by_role": stress_roots_by_role,
            "main_game_seconds_before_allowance": game_seconds, "main_ordinary_audit_seconds_before_allowance": ordinary_seconds,
            "main_stress_seconds_before_allowance": stress_seconds, "maximum_stress_roots_by_role": limits,
            "projected_main_bytes": projected, "free_bytes": free_bytes, "elapsed_seconds": elapsed_seconds,
            "reasonable": forecast <= CONFIG["maximum_forecast_hours"]*3600
                          and elapsed_seconds+remaining <= CONFIG["maximum_runtime_hours"]*3600
                          and free_bytes-projected >= CONFIG["disk_reserve_bytes"]}


def operational_decision(binding):
    budget_boundary(OUTPUT)
    games, audits = (stage_lock("probe", kind, binding) for kind in ("games", "audits"))
    complete = old.read_json(OUTPUT / "probe_complete.json")
    require(complete["manifest"] == binding and complete["all_prescribed_audits_pass"]
            and complete["game_lock"] == artifact(OUTPUT / "probe_games_lock.json")
            and complete["audit_lock"] == artifact(OUTPUT / "probe_audits_lock.json"), "Accepted probe required")
    receipts = [verified(item) for item in audits["files"]]
    stress = [r for receipt in receipts for r in receipt["roots"] if r["stress"]]
    value = {"manifest": binding, "time": old.now(), "outcome_blind": True,
             "probe": artifact(OUTPUT / "probe_complete.json"),
             **cost_projection(sum(verified(item)["timing"]["wall"] for item in games["files"]),
                sum(receipt["audit_timing"]["wall"] for receipt in receipts),
                {role: sum(r["audit_timing"]["stress_extra_replay_wall"] for r in stress if r["planner"] == role)
                 for role in ("candidate", "reference")},
                {role: sum(r["planner"] == role for r in stress) for role in ("candidate", "reference")},
                sum(p.stat().st_size for p in (OUTPUT / "probe").rglob("*") if p.is_file()),
                shutil.disk_usage(ROOT).free, time.time()-old.read_json(OUTPUT / "launch.json")["epoch_seconds"]),
             "caveat": "Twelve-game worker-time forecast; ordinary work scales402/12, stress by90 candidate/81 reference maxima;25%time/50%storage allowance; no outcome used."}
    old.save_new(OUTPUT / "operational_decision.json", value)
    if not value["reasonable"]:
        old.save_new(OUTPUT / "operational_hold.json", {**value, "reason": "Outcome-blind forecast exceeds fixed time or storage budget"})
        event("operational_hold", forecast_hours=value["forecast_hours"])
        return False
    event("probe_passed", forecast_hours=value["forecast_hours"])
    return True


def close_stage(stage, binding):
    """Read-only final linkage/coverage closure, without simulations or statistics."""
    sample = verify.validate_sample(stage, binding, OUTPUT)
    games, audits = (stage_lock(stage, kind, binding) for kind in ("games", "audits"))
    receipts = [verified(item) for item in audits["files"]]
    coverage = checked_audit_coverage(receipts, sample, old.file_hash(OUTPUT / stage / "audit_sample.json"))
    for game_item, receipt in zip(games["files"], receipts, strict=True):
        budget_boundary(OUTPUT)
        require(receipt["manifest"] == binding and receipt["game_artifact"] == game_item
                and receipt["passed"] and not receipt["full_inner_replay"], "Final audit linkage")
        game = verified(game_item)
        require(len(game["trace"]) == len(receipt["roots"]), "Final root count")
        for move, root in zip(game["trace"], receipt["roots"], strict=True):
            budget_boundary(OUTPUT)
            require(move["root_sha256"] == root["sha256"] == old.file_hash(arena.root_path(stage, game_item["index"], move["turn"], OUTPUT)),
                    "Final root mutated")
    complete = old.read_json(OUTPUT / f"{stage}_complete.json")
    require(complete["manifest"] == binding and complete["all_prescribed_audits_pass"]
            and complete["coverage"] == coverage
            and complete["game_lock"] == artifact(OUTPUT / f"{stage}_games_lock.json")
            and complete["audit_lock"] == artifact(OUTPUT / f"{stage}_audits_lock.json")
            and complete["sample"] == artifact(OUTPUT / stage / "audit_sample.json"), "Stage completion linkage changed")
    return games, audits, receipts, coverage


def raw_game_rows(files):
    """Derive final scores independently from saved games, not audit summaries."""
    rows = []
    for item in files:
        budget_boundary(OUTPUT)
        game = verified(item)
        spec = game["spec"]
        require("failure" not in game and game["reason"] in old.VALID_NATURAL_TERMINATIONS, "Invalid saved game")
        require(spec == arena.specs("main")[item["index"]] and game["winner"] in (None, 0, 1), "Saved game identity/outcome")
        winner, seat = game["winner"], spec["candidate_seat"]
        score = .5 if winner is None else int(winner == seat)
        rows.append({"index": spec["index"], "pair": spec["pair"], "fit": spec["fit"], "seat": seat, "score": score})
    return rows


def finish(binding):
    budget_boundary(OUTPUT)
    require(validate_freeze() == binding, "Changed final frozen identity")
    models.validate(binding, OUTPUT)
    close_stage("probe", binding)
    games, audits, receipts, coverage = close_stage("main", binding)
    rows = raw_game_rows(games["files"])
    audited_rows = [receipt["game"] for receipt in receipts]
    for raw, audited in zip(rows, audited_rows, strict=True):
        require(raw == {key: audited[key] for key in raw}, "Saved game/receipt endpoint disagreement")
    summary = arena.statistics(rows, "main")
    independent = verify.independent_statistics(audited_rows, "main")
    verify.compare(summary, independent)
    roots = [r for receipt in receipts for r in receipt["roots"]]
    budget_boundary(OUTPUT)
    result = {"manifest": binding, "time": old.now(), **summary, "rows": audited_rows,
              "all_prescribed_audits_pass": True, "full_inner_replay": False, "coverage": coverage,
              "models": artifact(OUTPUT / "models_lock.json"), "sample": artifact(OUTPUT / "main/audit_sample.json"),
              "fits": {str(fit): {"games": 134, "share": sum(r["score"] for r in rows if r["fit"] == fit)/134} for fit in FITS},
              "collection_seconds": games["elapsed_seconds"], "audit_seconds": audits["elapsed_seconds"],
              "decision_seconds": {role: {"count": sum(r["planner"] == role for r in roots),
                  "mean": float(np.mean([r["timing"]["wall"] for r in roots if r["planner"] == role]))} for role in ("candidate", "reference")},
              "continuations_by_role": {role: sum(r["continuations"] for r in roots if r["planner"] == role) for role in ("candidate", "reference")},
              "scope": "Fixed broader proposals versus learned128 anchor in fresh full games, with additional compute; no equal-budget/time, broader-opponent or superhuman claim.",
              "automatic_successor": False}
    old.save_new(OUTPUT / "result.json", result)
    old.save_new(OUTPUT / "final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "result": artifact(OUTPUT / "result.json"), "independent_statistics": independent,
                 "raw_game_endpoints_agree": True, "models": artifact(OUTPUT / "models_lock.json"),
                 "stages": {stage: artifact(OUTPUT / f"{stage}_complete.json") for stage in STAGES},
                 "coverage": coverage, "full_inner_replay": False})
    lines = ["# Full-game breadth confirmation", "", f"Classification: {summary['classification']}.", "",
        f"{summary['wins']} wins, {summary['draws']} draws and {summary['losses']} losses in 402 fresh games/201 seat pairs; all three fixed fits retained.",
        f"Breadth score: {summary['share']:.4%}; paired 95% interval [{summary['ci95'][0]:.4%}, {summary['ci95'][1]:.4%}]. Draws count half.", "",
        "The fixed practical gate requires observed score at least 55% and 95% lower bound above 50%. This does not prove the true score exceeds 55%.", "",
        f"Every actual game and decision was checked. Prescribed sampled replay passed: {coverage['replayed_continuations']} of {coverage['recorded_continuations']} continuations, with {coverage['stress_roots']} full stress roots. Unreplayed outcomes received structural checks. Mature simulation, encoding and neural primitives remain shared trusted dependencies.", "",
        f"Main collection took {games['elapsed_seconds']/3600:.3f} hours and sampled audit {audits['elapsed_seconds']/3600:.3f} hours.", "",
        "Breadth retains the anchor four and adds up to two ranked plus two identity-uniform moves, with 128 worlds per move. The reference is the unchanged learned128 anchor. Breadth uses additional compute; this is not equal-budget or equal-time gameplay evidence, broad dominance or superhuman performance.", "",
        "No automatic training, promotion, extension, second arena, successor or monitoring.", ""]
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    budget_boundary(OUTPUT)
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "classification": summary["classification"], "all_prescribed_audits_pass": True, "full_inner_replay": False,
                 "result": artifact(OUTPUT / "result.json"), "audit": artifact(OUTPUT / "final_audit.json"),
                 "report": artifact(OUTPUT / "report.md"), "automatic_successor": False})
    event("complete", classification=summary["classification"], score=summary["share"])


def run():
    global REPORTER
    log = OUTPUT / "runtime"
    log.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot run; preserve partial work")
        binding = validate_freeze()
        REPORTER = StatusReporter(OUTPUT, {"manifest": binding})
        try:
            with parent.inhibit_system_sleep() as sleep_receipt:
                old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                             "epoch_seconds": time.time(), "sleep_inhibition": sleep_receipt,
                             "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
                models.prepare(binding, OUTPUT)
                build_stage("probe", binding)
                if not operational_decision(binding):
                    return
                build_stage("main", binding)
                finish(binding)
        except BudgetExceeded as exc:
            if not (OUTPUT / "operational_hold.json").exists():
                old.save_new(OUTPUT / "operational_hold.json", {"manifest": binding, "time": old.now(), "reason": str(exc)})
            event("operational_hold", reason=str(exc))
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
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_proposal_breadth_arena_128.py", "run"],
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
