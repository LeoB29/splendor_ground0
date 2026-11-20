"""Learned fourth proposal versus exact original160 teacher,128 worlds."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import ctypes
from functools import lru_cache
import hashlib
import importlib
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from scripts.analysis import learned_proposal_screen as screen
from scripts.analysis import learned_proposal_arena as previous
from scripts.experiments import learned_proposal_attribution as attribution
from splendor_ai.alphazero import world_budget_160 as world128

old, baseline, windows = screen.old, screen.baseline, screen.windows
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/learned_proposal_128"
PRODUCTION_OUTPUT = OUTPUT.resolve()
MODELS = dict(screen.MODELS)
CONFIG = {
    "games": {"probe": 12, "main": 402, "test": 6},
    "environment_seed": {"probe": 2069000000, "main": 2060000000, "test": 2069800000},
    "assignment_tag": "learned-proposal-128-fit-v1", "stream_tag": "learned-proposal-128-stream-v1",
    "bootstrap_seed": 2063000000, "bootstrap_samples": 50000, "bootstrap_chunk": 1000,
    "worlds": 128, "max_candidates": 4, "workers": 8, "torch_threads": 1, "device": "cpu",
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_forecast_hours": 120., "maximum_runtime_hours": 168.,
    "disk_reserve_bytes": 5 * 1024**3, "maximum_stress_roots": 162,
    "full_inner_replay": False, "replay_worlds_per_root": 8,
    "audit_seed": 2064000000, "audit_world_tag": "learned-proposal-128-audit-world-v1",
    "audit_stress_tag": "learned-proposal-128-audit-stress-v1",
    "backend": "serial", "batch_size": 1, "training": False, "automatic_retry": False,
    "promotion": False, "automatic_successor": False, "scheduled_monitoring": False,
}
EXTRAS = (
    "run_learned_proposal_128.py", "docs/engineering/learned_proposal_128.md",
    "docs/strategy/next_after_proposal_attribution.md", "docs/engineering/verification_policy.md",
)
require = screen.require


def artifact(path, **metadata):
    path = Path(path)
    return {"path": path.resolve().relative_to(ROOT).as_posix(), "sha256": old.file_hash(path), **metadata}


def verify_hash(item):
    path = ROOT / item["path"]
    require(path.resolve().is_relative_to(ROOT), "Artifact outside repository")
    require(old.file_hash(path) == item["sha256"], f"Changed artifact: {path}")
    return path


def verified(item):
    return old.read_json(verify_hash(item))


def initialize_models():
    screen.initialize_models()


@lru_cache(maxsize=3)
def specs(stage):
    require(stage in CONFIG["games"], "Unknown stage")
    count = CONFIG["games"][stage] // 2
    order = sorted(range(count), key=lambda pair: old.digest([CONFIG["assignment_tag"], stage, pair]))
    fits = {pair: list(MODELS)[rank % 3] for rank, pair in enumerate(order)}
    return [{"index": i, "pair": i // 2, "candidate_seat": i % 2, "fit": fits[i // 2],
             "environment_seed": CONFIG["environment_seed"][stage] + i // 2,
             "search_stream": old.digest([CONFIG["stream_tag"], stage, i // 2])}
            for i in range(CONFIG["games"][stage])]


def check_stage(stage, base):
    require(stage in CONFIG["games"], "Unknown stage")
    resolved = Path(base).resolve()
    if stage == "test":
        require(resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents,
                "Synthetic fixture cannot target production evidence")


def game_path(stage, index, base=OUTPUT):
    check_stage(stage, base)
    require(type(index) is int and 0 <= index < CONFIG["games"][stage], "Game outside fixed schedule")
    return Path(base) / stage / "games" / f"game_{index:04d}.json"


def root_path(stage, index, turn, base=OUTPUT):
    game_path(stage, index, base)
    require(type(turn) is int and 0 <= turn < 400, "Invalid root turn")
    return Path(base) / stage / "roots" / f"game_{index:04d}" / f"turn_{turn:03d}.json"


fourth_slot = previous.fourth_slot


def learned_proposal(state, stream, fit):
    """Successful learned candidate list with exact160's128-world schedule."""
    value = previous.learned_proposal(state, stream, fit)
    value["schedule"] = world128.root_schedule(state, stream, 128)
    return value


def choose_root(state, stream, path, binding, fit=None):
    path = Path(path)
    require(not path.exists(), "One-shot root; no overwrite or retry")
    if fit is None:
        # This is the retained teacher, including its second random alternative.
        return world128.choose_root(state, stream, path, binding, 128)
    wall, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "time": old.now(), "snapshot": old.serialize_state_snapshot(state),
              "privacy": baseline.privacy(state), "worlds": []}
    try:
        proposal = learned_proposal(state, stream, fit)
        record.update(schedule=proposal["schedule"], candidates=proposal["candidates"], proposal=proposal)
        legal = old.SplendorEnv().legal_actions(state)
        record["scores"] = list(map(float, old._RAW.score_actions(state, legal)))
        actions = [old.decode_action(a) for a in proposal["candidates"]["actions"]]
        for seed in proposal["schedule"]["world_seeds"]:
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []}
            record["worlds"].append(row)
            for action in actions:
                require(action in old.SplendorEnv().legal_actions(world), "Illegal forced proposal")
                value = old.trajectory(world, action)
                value["wdl"] = old.outcome_value(value["winner"], state.current_player) if value["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                row["outcomes"].append(value)
                require(value["wdl"] is not None, "Invalid continuation ending")
        record.update(world128.select(record["worlds"], len(actions), 128))
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
        old.save_new(path, record)
    return actions[record["choice"]], record


def auditor():
    return importlib.import_module("tools.learned_proposal_128.verify")


def decision(share, interval):
    superiority = interval[0] > .5
    practical = share >= .55 and superiority
    return {"superiority_established": superiority, "practical_progress": practical,
            "material_target_excluded": interval[1] < .55,
            "classification": "positive_practical_progress" if practical else
            "positive_below_observed_target" if superiority else
            "below_material_target" if interval[1] < .55 else "inconclusive"}


def statistics(rows):
    require(len(rows) == 402 and [r["index"] for r in rows] == list(range(402)), "Incomplete or unordered main")
    strata = {fit: [] for fit in MODELS}
    for pair in range(201):
        a, b = rows[2*pair:2*pair+2]
        spec = specs("main")[2*pair]
        require(a["pair"] == b["pair"] == pair and a["seat"] == 0 and b["seat"] == 1
                and a["fit"] == b["fit"] == spec["fit"], "Pair/fit alignment")
        require(a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.), "Invalid score")
        strata[spec["fit"]].append(int(2*a["score"] + 2*b["score"]))
    require(all(len(v) == 67 for v in strata.values()), "Unbalanced fits")
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = []
    for _ in range(50):
        total = np.zeros(1000, dtype=np.int64)
        for values in strata.values():
            indices = rng.integers(0, 67, size=(1000, 67))
            total += np.asarray(values, dtype=np.int64)[indices].sum(axis=1)
        boot.extend((total / 804).tolist())
    share = sum(sum(values) for values in strata.values()) / 804
    interval = np.quantile(boot, [.025, .975], method="linear").tolist()
    return {"games": 402, "pairs": 201, "share": share, "ci95": interval,
            "wins": sum(r["score"] == 1 for r in rows), "draws": sum(r["score"] == .5 for r in rows),
            "losses": sum(r["score"] == 0 for r in rows), **decision(share, interval)}


def play_game(stage, spec, binding, base=OUTPUT, test_initial_snapshot=None):
    """Play the fixed proposal intervention once, retaining every real decision."""
    require(spec == specs(stage)[spec["index"]], "Fixed schedule mismatch")
    path = game_path(stage, spec["index"], base)
    if path.exists():
        raise FileExistsError(f"Proposal arena does not automatically reuse or retry games: {path}")
    if type(spec["candidate_seat"]) is not int or spec["candidate_seat"] not in (0, 1):
        raise ValueError("Invalid candidate seat")
    if test_initial_snapshot is not None and stage != "test":
        raise ValueError("Initial snapshot overrides are restricted to isolated test games")
    wall, cpu = time.perf_counter(), time.process_time()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = (env.initial_state() if test_initial_snapshot is None
             else old.deserialize_state_snapshot(test_initial_snapshot))
    record = {"manifest": binding, "spec": spec, "time": old.now(), "trace": [],
              "initial_state_sha256": old.digest(old.serialize_state_snapshot(state))}
    if test_initial_snapshot is not None:
        record["test_initial_snapshot"] = old.serialize_state_snapshot(state)
    visits, forced, idle, hasher = Counter(), 0, 0, hashlib.sha256()
    reason, winner = "completed", None
    try:
        while not state.terminal:
            signature = old.state_signature(state)
            visits[signature] += 1
            if visits[signature] >= 4:
                reason = "repetition_cutoff"
                break
            if state.turn_index >= 400:
                reason = "max_turns"
                break
            legal = env.legal_actions(state)
            if not legal:
                reason = "stalled"
                break
            row = {"turn": state.turn_index, "actor": state.current_player,
                   "state_sha256": old.digest(old.serialize_state_snapshot(state))}
            fit = spec["fit"] if state.current_player == spec["candidate_seat"] else None
            rp = root_path(stage, spec["index"], state.turn_index, base)
            try:
                action, _ = choose_root(state, spec["search_stream"], rp, binding, fit)
            except Exception:
                if rp.exists():
                    row["root_sha256"] = old.file_hash(rp)
                record["failed_decision"] = row
                raise
            row["root_sha256"] = old.file_hash(rp)
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                record["failed_decision"] = {
                    **row, "action": None if action is None else old.serialize_action(action)}
                break
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            hasher.update(old.digest(row["action"]).encode())
            successor = env.step(state, action)
            forced = forced + 1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, successor, action) else idle + 1
            state = successor
            if forced >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            if idle >= 60:
                reason = "no_progress_cutoff"
                break
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason, winner=winner)
        if reason not in old.VALID_NATURAL_TERMINATIONS:
            raise ValueError(f"Invalid outer ending: {reason}")
    except Exception:
        record["failure"] = traceback.format_exc()
        if "reason" not in record:
            record.update(reason="exception", winner=None)
        raise
    finally:
        final = old.serialize_state_snapshot(state)
        record.update(final_state=final, final_state_sha256=old.digest(final),
                      turns=len(record["trace"]), trace_sha256=hasher.hexdigest(),
                      timing={"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu})
        old.save_new(path, record)
    return {"index": spec["index"], "sha256": old.file_hash(path)}


CHECK_TARGETS = ["tools/learned_proposal_128/test_experiment.py", "tools/learned_proposal_128/test_verify.py",
                 "tests/test_learned_proposal_attribution.py", "tests/test_world_budget_160.py",
                 "tests/test_windowless_workers_160.py"]
CHECK_DESELECTIONS = [{
    "nodeid": "tests/test_world_budget_160.py::test_test_snapshot_and_test_output_cannot_enter_production",
    "reason": "Frozen legacy test assumes the original160 production game does not exist and expects the later snapshot ValueError. Completed160 correctly refuses that existing game first with FileExistsError. Preserve the old test and all production evidence.",
    "replacement": "tools/learned_proposal_128/test_experiment.py::test_completed160_production_refusal_and_isolated_snapshot_guards",
}]


def identity():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts/analysis").glob("*.py"))
    paths += sorted((ROOT / "scripts/experiments").glob("*.py"))
    paths += sorted((ROOT / "tools/learned_proposal_128").glob("*.py"))
    paths += [ROOT / name for name in sorted(set(EXTRAS) | set(CHECK_TARGETS))]
    return {"config": CONFIG, "models": {str(k): list(v) for k, v in MODELS.items()},
            "check_plan": {"targets": CHECK_TARGETS, "deselections": CHECK_DESELECTIONS},
            "dependencies": {path.relative_to(ROOT).as_posix(): old.file_hash(path) for path in paths},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    binding = attribution.validate_freeze()
    require(binding == "0f2772aebd99e2fef939dc1a9464790daa1f6c3b1a8e9fe1326a2361a85e44e2", "Wrong learned candidate parent")
    final = old.read_json(attribution.OUTPUT / "final_status.json")
    require(final["manifest"] == binding and final["complete"] and final["all_prescribed_audits_pass"]
            and final["full_inner_replay"] and final["classification"] == "positive_practical_progress", "Candidate arena not complete/audited")
    for name in ("result", "report", "audit"):
        verify_hash(final[name])
    audit = verified(final["audit"])
    require(audit["passed"] and audit["result_sha256"] == final["result"]["sha256"]
            == "7a1aa0d4fb2da8b7f6159f8bc830efaefa384aaf3816bea6e52af024dca87461", "Candidate result/audit identity")
    manifest = verified(old.read_json(attribution.OUTPUT / "freeze.json")["manifest"])
    files = [*manifest["parent"]["files"], manifest["checks"], manifest["schedule"],
             *verified(manifest["checks"])["logs"],
             *(artifact(attribution.OUTPUT / name) for name in ("manifest.json", "freeze.json", "final_status.json")),
             *(final[name] for name in ("result", "report", "audit"))]
    files.extend({"path": name, "sha256": sha} for name, sha in manifest["identity"]["dependencies"].items())
    for stage in ("probe", "main"):
        for kind in ("games", "audits"):
            attribution.stage_lock(stage, kind, binding)
            files.append(artifact(attribution.OUTPUT / f"{stage}_{kind}_lock.json"))
        complete_item = audit[stage]
        require(verified(complete_item)["manifest"] == binding, "Parent completion identity")
        files.append(complete_item)
    teacher = original_teacher_evidence()
    files.extend(teacher["files"])
    unique = {item["path"]: {"path": item["path"], "sha256": item["sha256"]} for item in files}
    for item in unique.values():
        verify_hash(item)
    return {"learned_attribution": binding, "screen": manifest["parent"]["screen"],
            "original160": teacher["binding"], "original160_result": teacher["result"],
            "original160_audit_scope": "All original actual games checked; two hash-selected worlds per root replayed, not exhaustive inner replay.",
            "files": list(unique.values())}


def original_teacher_evidence():
    """Explicitly bind the retained teacher's completed evidence, without replay."""
    from splendor_ai.alphazero import run_world_budget as teacher
    binding = teacher.validate_freeze()
    require(binding == "5e9e9202f44e7b526fff4be97cdcd3186805fbb5e8a184d21a6702de9d922df3",
            "Wrong original160 teacher identity")
    folder = ROOT / "outputs/world_budget_160"
    final = old.read_json(folder / "final_status.json")
    require(final["manifest"] == binding and final["complete"] and final["prespecified_audit_pass"]
            and not final["full_inner_replay"] and final["classification"] == "positive",
            "Original160 teacher has no completed prescribed audit")
    result = artifact(folder / "result.json")
    require(result["sha256"] == final["result_sha256"]
            == "2e3efa2c524f8d8c68284fad0f3400365bcb08f4b598aecf121039a4c45b19e5",
            "Original160 result identity changed")
    require(old.file_hash(folder / "report.md") == final["report_sha256"], "Original160 report changed")
    files = [artifact(folder / name) for name in
             ("manifest.json", "freeze.json", "schedule.json", "final_status.json", "result.json", "report.md", "preflight.json")]
    teacher_manifest = old.read_json(folder / "manifest.json")
    files.extend({"path": name, "sha256": sha} for name, sha in teacher_manifest["dependencies"].items())
    freeze = old.read_json(folder / "freeze.json")
    test_path = ROOT / freeze["tests_path"]
    files.extend(artifact(path) for path in (test_path, test_path.parent / "stdout.txt", test_path.parent / "stderr.txt"))
    for stage in ("probe", "main"):
        files.append(artifact(folder / f"{stage}_launch.json"))
        for is_audit in (False, True):
            lock = teacher.verify_lock(stage, binding, is_audit)
            files.append(artifact(folder / f"{stage}_{'audit' if is_audit else 'collect'}_lock.json"))
            for key, sha in lock["files"].items():
                path = teacher.audit_path(stage, int(key)) if is_audit else world128.game_path(stage, int(key))
                files.append({"path": Path(path).as_posix(), "sha256": sha})
    return {"binding": binding, "result": result, "files": files}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    arguments = ["-q", *CHECK_TARGETS, *("--deselect=" + row["nodeid"] for row in CHECK_DESELECTIONS)]
    code = "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(" + repr(arguments) + "))"
    started = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Identity changed during checks")
    old.save_new(folder / "checks.json", {"identity": before, "exit_code": result.returncode,
                 "targets": CHECK_TARGETS, "deselections": CHECK_DESELECTIONS,
                 "pytest_arguments": arguments, "seconds": time.perf_counter() - started,
                 "logs": [artifact(folder / name) for name in ("stdout.txt", "stderr.txt")]})
    return result.returncode


def prepare(checks_path):
    require(not (OUTPUT / "manifest.json").exists(), "Already prepared")
    evidence = old.read_json(checks_path)
    require(evidence["identity"] == identity() and evidence["exit_code"] == 0, "Current passing checks required")
    require(evidence["targets"] == CHECK_TARGETS and evidence["deselections"] == CHECK_DESELECTIONS,
            "The complete prescribed check plan and explicit legacy replacement are required")
    for item in evidence["logs"]:
        verify_hash(item)
    parents = parent_evidence()
    engineering_path = ROOT / "outputs/batching_benchmark_v2/adoption_decision.json"
    engineering = old.read_json(engineering_path)
    require(engineering["passed"] and engineering["selected_backend"] == CONFIG["backend"],
            "The recorded benchmark decision must match the frozen backend")
    for item in engineering.get("files", []):
        verify_hash(item)
    schedule = {stage: specs(stage) for stage in ("probe", "main")}
    for stage, rows in schedule.items():
        require(rows == auditor().independent_specs(stage), "Independent schedule mismatch")
    old.save_new(OUTPUT / "schedule.json", schedule)
    manifest = {"identity": identity(), "parent": parents, "checks": artifact(checks_path),
                "engineering_decision": artifact(engineering_path),
                "schedule": artifact(OUTPUT / "schedule.json"), "owner_authorized": True}
    old.save_new(OUTPUT / "manifest.json", manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": artifact(OUTPUT / "manifest.json"),
                 "binding": old.digest(manifest), "time": old.now(), "owner_authorized": True})
    print({"prepared": True, "binding": old.digest(manifest)}, flush=True)


def validate_freeze():
    freeze = old.read_json(OUTPUT / "freeze.json")
    manifest = verified(freeze["manifest"])
    require(old.digest(manifest) == freeze["binding"] and freeze["owner_authorized"], "Freeze mismatch")
    require(old.digest(manifest["identity"]) == old.digest(identity()), "Frozen code/config/runtime changed")
    checks = verified(manifest["checks"])
    require(checks["exit_code"] == 0 and old.digest(checks["identity"]) == old.digest(identity()), "Check identity mismatch")
    require(checks["targets"] == CHECK_TARGETS and checks["deselections"] == CHECK_DESELECTIONS,
            "Frozen check plan or explicit legacy replacement changed")
    require(verified(manifest["schedule"]) == {s: specs(s) for s in ("probe", "main")}, "Frozen schedule changed")
    for item in [*checks["logs"], *manifest["parent"]["files"]]:
        verify_hash(item)
    engineering = verified(manifest["engineering_decision"])
    require(engineering["passed"] and engineering["selected_backend"] == CONFIG["backend"],
            "Engineering/backend decision mismatch")
    for item in engineering.get("files", []):
        verify_hash(item)
    return freeze["binding"]


def event(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    old.atomic_write(OUTPUT / "progress.json", row)
    print(row, flush=True)


def worker_initialize(stage, binding, base):
    folder = Path(base) / "workers" / stage
    folder.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt", folder / f"{os.getpid()}_stderr.txt")
    initialize_models()
    require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "Worker arithmetic config")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, **health})


def collect_job(stage, spec, binding, base=OUTPUT, test_snapshot=None):
    item = play_game(stage, spec, binding, base, test_snapshot)
    path = game_path(stage, spec["index"], base)
    return {**item, "path": path.resolve().relative_to(ROOT).as_posix()}


def audit_job(stage, item, binding, base=OUTPUT, test_snapshot=None):
    result = auditor().audit_game(stage, item["index"], item["sha256"], binding, base, test_snapshot)
    path = Path(base) / stage / "audits" / f"game_{item['index']:04d}.json"
    old.save_new(path, {"manifest": binding, "game_artifact": item, "passed": True,
                        "full_inner_replay": False, **result})
    return artifact(path, index=item["index"], continuations=sum(r["continuations"] for r in result["roots"]),
                    replayed_branches=sum(r["replayed_branches"] for r in result["roots"]))


def stage_lock(stage, kind, binding):
    lock = old.read_json(OUTPUT / f"{stage}_{kind}_lock.json")
    require(lock["manifest"] == binding and len(lock["files"]) == CONFIG["games"][stage]
            and [x["index"] for x in lock["files"]] == list(range(CONFIG["games"][stage])), "Stage completeness")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["stage"] == stage and launch["kind"] == kind, "Stage launch identity")
    for item in [*lock["files"], *launch["dependencies"]]:
        verify_hash(item)
    return lock


def run_pool(stage, kind, arguments, binding, upstream):
    require(validate_freeze() == binding, "Stage freeze mismatch")
    started = time.perf_counter()
    launch_path = OUTPUT / f"{stage}_{kind}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "time": old.now(), "stage": stage, "kind": kind,
                              "dependencies": [artifact(p) for p in upstream]})
    if kind == "games":
        old.save_new(OUTPUT / f"{stage}_launch.json", {"manifest": binding, "time": old.now(), "schedule": specs(stage)})
    runtime = windows.configure_spawn()
    event(f"{stage}_{kind}", completed=0, total=len(arguments))
    task = collect_job if kind == "games" else audit_job
    files = []
    with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(f"{stage}_{kind}", binding, OUTPUT)) as pool:
        jobs = [pool.submit(task, stage, argument, binding) for argument in arguments]
        try:
            for job in as_completed(jobs):
                files.append(job.result())
                event(f"{stage}_{kind}", completed=len(files), total=len(arguments))
                launch = old.read_json(OUTPUT / "launch.json")
                require(time.time() - launch["epoch_seconds"] <= CONFIG["maximum_runtime_hours"] * 3600,
                        "Frozen168-hour runtime ceiling exceeded at a job boundary; preserve evidence and stop")
                require(shutil.disk_usage(ROOT).free >= CONFIG["disk_reserve_bytes"],
                        "Frozen5GiB disk reserve crossed at a job boundary; preserve evidence and stop")
        except BaseException:
            for job in jobs:
                job.cancel()
            # Already-running one-game tasks finish and preserve their evidence.
            raise
    files.sort(key=lambda row: row["index"])
    require(len(files) == len(arguments) and len({r["index"] for r in files}) == len(files), "Incomplete/duplicate jobs")
    for item in files:
        verify_hash(item)
    require(validate_freeze() == binding, "Stage changed frozen evidence")
    old.save_new(OUTPUT / f"{stage}_{kind}_lock.json", {"manifest": binding, "launch": artifact(launch_path),
                 "files": files, "runtime": runtime, "elapsed_seconds": time.perf_counter() - started})
    return files


def build_stage(stage, binding):
    games = run_pool(stage, "games", specs(stage), binding, [OUTPUT / "freeze.json", OUTPUT / "schedule.json"])
    sample = auditor().seal_sample(stage, games, binding)
    saved_sample = auditor().validate_sample(stage, binding)
    audits = run_pool(stage, "audits", games, binding,
                      [OUTPUT / f"{stage}_games_lock.json", verify_hash(sample),
                       OUTPUT / stage / "audit_sample_seal.json"])
    for item, game in zip(audits, games):
        receipt = verified(item)
        require(receipt["passed"] and not receipt["full_inner_replay"] and receipt["game_artifact"] == game, "Audit accounting")
    coverage = checked_audit_coverage([verified(item) for item in audits], saved_sample, sample["sha256"])
    old.save_new(OUTPUT / f"{stage}_complete.json", {"manifest": binding, "time": old.now(),
                 "games": len(games), "full_inner_replay": False,
                 "sample": sample, "coverage": coverage,
                 "game_lock": artifact(OUTPUT / f"{stage}_games_lock.json"),
                 "audit_lock": artifact(OUTPUT / f"{stage}_audits_lock.json"),
                 "continuations": sum(item["continuations"] for item in audits),
                 "replayed_branches": sum(item["replayed_branches"] for item in audits)})


def checked_audit_coverage(receipts, sample, sample_sha256):
    """Reconcile independent audit receipts against the pre-replay sample inventory."""
    require(all(receipt["sample_sha256"] == sample_sha256 and not receipt["full_inner_replay"]
                for receipt in receipts), "Audit receipt is not linked to the prescribed sampled manifest")
    roots = [root for receipt in receipts for root in receipt["roots"]]
    ids = [(root["index"], root["turn"]) for root in roots]
    expected = [(root["index"], root["turn"]) for root in sample["inventory"]]
    require(ids == expected and len(set(ids)) == len(ids), "Audit root inventory differs from sealed sample")
    selections = {row["root_id"]: row for row in sample["selections"]}
    for root, metadata in zip(roots, sample["inventory"]):
        selection = selections[metadata["root_id"]]
        require(root["sha256"] == metadata["sha256"] and root["support"] == metadata["support"]
                and root["worlds"] == 128 and not root["full_inner_replay"] and not root["full_fixture_replay"]
                and root["ordinary_world_indices"] == selection["ordinary_world_indices"]
                and root["replay_world_indices"] == selection["replay_world_indices"]
                and root["stress"] == selection["stress"], "Audit/sample root coverage mismatch")
        require(root["continuations"] == 128 * root["support"]
                and root["replayed_branches"] == len(root["replay_world_indices"]) * root["support"],
                "Audit branch coverage is inconsistent with selected whole worlds")
    actual = {"games": len(receipts), "roots": len(roots),
              "recorded_worlds": sum(root["worlds"] for root in roots),
              "recorded_continuations": sum(root["continuations"] for root in roots),
              "replayed_worlds": sum(len(root["replay_world_indices"]) for root in roots),
              "replayed_continuations": sum(root["replayed_branches"] for root in roots),
              "stress_roots": sum(root["stress"] for root in roots)}
    require(all(actual[key] == sample["coverage"][key] for key in actual), "Stage audit coverage totals differ from sealed sample")
    return actual


def cost_projection(game_worker_wall, audit_worker_wall, stress_extra_wall,
                    stress_roots, size_bytes, free_bytes):
    """Outcome-blind worker-time forecast; stress coverage is fixed, not per game."""
    values = (game_worker_wall, audit_worker_wall, stress_extra_wall)
    require(all(type(v) in (float, int) and np.isfinite(v) and v >= 0 for v in values),
            "Missing/nonfinite probe cost evidence")
    require(game_worker_wall > 0 and audit_worker_wall > 0 and stress_extra_wall > 0
            and audit_worker_wall >= stress_extra_wall,
            "Invalid measured probe replay costs")
    require(type(stress_roots) is int and 0 < stress_roots <= CONFIG["maximum_stress_roots"],
            "Missing/invalid stress-root timing sample")
    ratio = 402 / 12
    game_seconds = ratio * game_worker_wall / CONFIG["workers"]
    ordinary_audit_seconds = ratio * (audit_worker_wall - stress_extra_wall) / CONFIG["workers"]
    stress_seconds = (CONFIG["maximum_stress_roots"] * stress_extra_wall / stress_roots
                      / CONFIG["workers"])
    forecast = 1.25 * (game_seconds + ordinary_audit_seconds + stress_seconds)
    projected = int(1.5 * ratio * size_bytes + 512 * 1024**2)
    return {"forecast_seconds": forecast, "forecast_hours": forecast / 3600,
            "game_worker_wall": game_worker_wall, "audit_worker_wall": audit_worker_wall,
            "stress_extra_wall": stress_extra_wall, "probe_stress_roots": stress_roots,
            "main_game_seconds_before_allowance": game_seconds,
            "main_ordinary_audit_seconds_before_allowance": ordinary_audit_seconds,
            "main_stress_seconds_before_allowance": stress_seconds,
            "projected_main_bytes": projected, "free_bytes": free_bytes,
            "reasonable": forecast <= CONFIG["maximum_forecast_hours"] * 3600
            and free_bytes - projected >= CONFIG["disk_reserve_bytes"]}


def operational_decision(binding):
    games, audits = (stage_lock("probe", kind, binding) for kind in ("games", "audits"))
    game_worker_wall = sum(verified(item)["timing"]["wall"] for item in games["files"])
    receipts = [verified(item) for item in audits["files"]]
    audit_worker_wall = sum(receipt["audit_timing"]["wall"] for receipt in receipts)
    stress = [root for receipt in receipts for root in receipt["roots"] if root["stress"]]
    stress_extra_wall = sum(root["audit_timing"]["stress_extra_replay_wall"] for root in stress)
    size = sum(path.stat().st_size for path in (OUTPUT / "probe").rglob("*") if path.is_file())
    value = {"manifest": binding, "time": old.now(), "outcome_blind": True,
             "probe": artifact(OUTPUT / "probe_complete.json"),
             **cost_projection(game_worker_wall, audit_worker_wall, stress_extra_wall,
                               len(stress), size, shutil.disk_usage(ROOT).free),
             "caveat": "Twelve-game worker-wall estimate/8; ordinary work scales402/12, stress to162 roots;25%time/50%storage allowances; no strength used."}
    old.save_new(OUTPUT / "operational_decision.json", value)
    if not value["reasonable"]:
        old.save_new(OUTPUT / "operational_hold.json", value)
        event("operational_hold", forecast_hours=value["forecast_hours"])
        return False
    event("probe_passed_main_authorized", forecast_hours=value["forecast_hours"])
    return True


def finish(binding):
    for stage in ("probe", "main"):
        saved_sample = auditor().validate_sample(stage, binding)
        games, audits = (stage_lock(stage, kind, binding) for kind in ("games", "audits"))
        checked_audit_coverage([verified(item) for item in audits["files"]], saved_sample,
                               old.file_hash(OUTPUT / stage / "audit_sample.json"))
        for game, item in zip(games["files"], audits["files"]):
            receipt = verified(item)
            require(receipt["manifest"] == binding and receipt["passed"] and not receipt["full_inner_replay"]
                    and receipt["game_artifact"] == game, "Final receipt linkage")
            saved = verified(game)
            require(len(saved["trace"]) == len(receipt["roots"]), "Outer/root accounting")
            for move, root in zip(saved["trace"], receipt["roots"]):
                require(move["root_sha256"] == root["sha256"] == old.file_hash(root_path(stage, game["index"], move["turn"])), "Final root mutation")
    receipts = [verified(item) for item in stage_lock("main", "audits", binding)["files"]]
    require(validate_freeze() == binding, "Frozen dependencies changed before final statistics publication")
    rows = [receipt["game"] for receipt in receipts]
    summary = statistics(rows)
    independent = auditor().independent_statistics(rows)
    auditor().compare(summary, independent)
    roots = [root for receipt in receipts for root in receipt["roots"]]
    coverage = {"roots": len(roots), "recorded_worlds": sum(r["worlds"] for r in roots),
                "replayed_worlds": sum(len(r["replay_world_indices"]) for r in roots),
                "continuations": sum(r["continuations"] for r in roots),
                "replayed_branches": sum(r["replayed_branches"] for r in roots),
                "stress_roots": sum(r["stress"] for r in roots),
                "ordinary_replay_worlds_per_root": 8,
                "ordinary_replayed_branches": sum(len(r["ordinary_world_indices"]) * r["support"] for r in roots),
                "stress_extra_replayed_branches": sum((len(r["replay_world_indices"]) - len(r["ordinary_world_indices"])) * r["support"] for r in roots),
                "sample": artifact(OUTPUT / "main/audit_sample.json"),
                "sample_seal": artifact(OUTPUT / "main/audit_sample_seal.json")}
    def part_summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part) / len(part),
                "mean_margin": sum(r["margin"] for r in part) / len(part)}
    def latency(role):
        values = [r["timing"]["wall"] for r in roots if r["planner"] == role]
        return {"count": len(values), "mean": float(np.mean(values)), "p95": float(np.quantile(values, .95))}
    result = {"manifest": binding, "time": old.now(), **summary, "rows": rows,
              "all_prescribed_audits_pass": True, "full_inner_replay": False,
              "fits": {str(fit): part_summary([r for r in rows if r["fit"] == fit]) for fit in MODELS},
              "seats": {str(seat): part_summary([r for r in rows if r["seat"] == seat]) for seat in (0, 1)},
              "decision_seconds": {role: latency(role) for role in ("candidate", "reference")},
              **coverage,
              "audit_worker_seconds": sum(r["audit_timing"]["wall"] for r in receipts),
              "main_collection_seconds": stage_lock("main", "games", binding)["elapsed_seconds"],
              "main_audit_seconds": stage_lock("main", "audits", binding)["elapsed_seconds"],
              "shared_trusted_dependencies": ["frozen Splendor rules engine", "canonical grouping and encoders",
                                              "Torch model arithmetic", "greedy035 continuation policy"],
              "verification_scope": "All actual games and root records checked; eight selected worlds out of128 at each root plus fixed full-root stress coverage. Unreplayed branch records are structurally checked, not independently executed.",
              "private_roots": sum(bool(r["privacy"]["own"] or r["privacy"]["opponent"]) for r in roots),
              "scope": "Frozen161 learned fourth proposal versus exact original160 teacher, both128 worlds; no equal-wall-time, broader-strength or repeated-learning claim."}
    old.save_new(OUTPUT / "result.json", result)
    old.save_new(OUTPUT / "final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "games": 402, "full_inner_replay": False, "independent_statistics": independent,
                 **coverage,
                 "result_sha256": old.file_hash(OUTPUT / "result.json"),
                 "probe": artifact(OUTPUT / "probe_complete.json"), "main": artifact(OUTPUT / "main_complete.json")})
    report = (f"# Learned128 versus original128 teacher result\n\n{summary['games']} fresh games, all three fixed 161 fits, both seats. "
              f"Score {summary['share']:.4%}; paired 95% interval {summary['ci95']}.\n\n"
              f"Classification: {summary['classification']}. Superiority: {summary['superiority_established']}; "
              f"practical progress: {summary['practical_progress']}.\n\n"
              "Every actual game checked; prespecified sampled continuation replay passed; independent statistics agree. "
              f"Checked {coverage['roots']} actual roots; replayed {coverage['replayed_branches']} of {coverage['continuations']} recorded continuations, "
              f"including {coverage['stress_roots']} full stress roots and eight worlds at every other root. "
              "Unreplayed branch records received structural checks. The rules engine, encoders and neural arithmetic remain shared trusted dependencies. "
              f"Main collection took {result['main_collection_seconds'] / 3600:.3f} hours; sampled verification took {result['main_audit_seconds'] / 3600:.3f} hours.\n\n"
              "Both planners use 128 worlds and the same first three candidates; the fourth is learned 161 versus the original second random proposal. "
              "This tests whether learned guidance improves the retained 128-world teacher. "
              "The interval conditions on all three fixed models. No broader-opponent, "
              "equal-wall-time, strongest-agent, superhuman or repeated-learning-loop claim. "
              "No automatic extension, training, promotion, successor or monitoring.\n")
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report)
    require(validate_freeze() == binding, "Final dependency mutation")
    old.save_new(OUTPUT / "final_status.json", {"complete": True, "manifest": binding, "time": old.now(),
                 "classification": summary["classification"], "all_prescribed_audits_pass": True,
                 "full_inner_replay": False, "result": artifact(OUTPUT / "result.json"),
                 "report": artifact(OUTPUT / "report.md"), "audit": artifact(OUTPUT / "final_audit.json"),
                 "automatic_successor": False})
    event("complete", classification=summary["classification"])


def _set_sleep_state(flags):
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint]
    function.restype = ctypes.c_uint
    previous = function(flags)
    require(previous != 0, "Windows refused the process-scoped system-sleep inhibition")
    return int(previous)


@contextmanager
def inhibit_system_sleep():
    """Keep this supervisor's system awake; leave display and power settings alone."""
    require(os.name == "nt", "Production experiment requires its frozen Windows runtime")
    previous = _set_sleep_state(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    try:
        yield {"system_required": True, "display_required": False, "previous_flags": previous}
    finally:
        _set_sleep_state(0x80000000)  # Release only this thread's continuing request.


def run():
    log = OUTPUT / "runtime"
    log.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot pipeline; no automatic retry")
        binding = validate_freeze()
        try:
            with inhibit_system_sleep() as sleep_receipt:
                old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                             "epoch_seconds": time.time(), "sleep_inhibition": sleep_receipt,
                             "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
                build_stage("probe", binding)
                if operational_decision(binding):
                    build_stage("main", binding)
                    finish(binding)
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
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_learned_proposal_128.py", "run"],
                                          folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic() + 60
        while not (OUTPUT / "launch.json").exists():
            require(time.monotonic() < deadline, "No acknowledgement; inspect evidence, do not relaunch")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT / "launch.json")
        require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding, "Wrong supervisor acknowledgement")
        old.save_new(folder / "acknowledged.json", {"time": old.now(), "pid": process["pid"], "launch": artifact(OUTPUT / "launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        require(args.checks is not None, "Provide new check directory")
        return check(args.checks)
    if args.command == "prepare":
        require(args.checks is not None, "Provide passing checks.json")
        prepare(args.checks)
    elif args.command == "launch":
        launch()
    else:
        run()
    return 0
