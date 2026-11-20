"""One bounded full-game learned fourth-proposal comparison against exact155."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from splendor_ai.alphazero.strong_policy import encode_position

old, baseline, windows = screen.old, screen.baseline, screen.windows
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/learned_proposal_arena"
PRODUCTION_OUTPUT = OUTPUT.resolve()
MODELS = dict(screen.MODELS)
CONFIG = {
    "games": {"probe": 12, "main": 402, "test": 6},
    "environment_seed": {"probe": 2049000000, "main": 2040000000, "test": 2049800000},
    "assignment_tag": "learned-proposal-arena-fit-v1", "stream_tag": "learned-proposal-arena-stream-v1",
    "bootstrap_seed": 2043000000, "bootstrap_samples": 50000, "bootstrap_chunk": 1000,
    "worlds": 32, "max_candidates": 4, "workers": 8, "torch_threads": 1, "device": "cpu",
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "maximum_forecast_hours": 60., "disk_reserve_bytes": 5 * 1024**3,
    "full_inner_replay": True, "training": False, "automatic_retry": False,
    "promotion": False, "automatic_successor": False, "scheduled_monitoring": False,
}
EXTRAS = (
    "run_learned_proposal_arena.py", "tests/test_learned_proposal_arena.py",
    "docs/engineering/learned_proposal_arena.md", "docs/strategy/next_after_learned_proposals.md",
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


def fourth_slot(keys, reference, values):
    require(len(set(keys)) == len(keys) == len(values) and np.isfinite(values).all(), "Group scores")
    require(len(reference) == min(4, len(keys)) and len(set(reference)) == len(reference)
            and set(reference).issubset(keys), "Original support")
    if len(reference) < 4:
        return list(reference)
    allowed = [j for j, key in enumerate(keys) if key not in reference[:3]]
    return reference[:3] + [keys[max(allowed, key=lambda j: values[j])]]


def learned_proposal(state, stream, fit):
    require(fit in MODELS, "Unknown frozen fit")
    started = time.perf_counter()
    legal = old.SplendorEnv().legal_actions(state)
    schedule = baseline.root_schedule(state, stream)
    reference = baseline.candidate_set(state, schedule["candidate_nonce"])
    groups = old.actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    representatives = {repr(key): old.serialize_action(action) for key, action in groups.items()}
    representatives.update(zip(reference["keys"], reference["actions"], strict=True))
    model = screen._MODELS[fit]
    inference_start = time.perf_counter()
    with torch.no_grad():
        row, encoded_keys, _ = encode_position(state, model.config, legal)
        logits, _ = model(row["observation"][None], row["features"][None],
                          torch.ones((1, len(legal)), dtype=torch.bool))
    require([repr(key) for key in encoded_keys] == keys and torch.isfinite(logits).all().item(), "Invalid model input/output")
    values = [float(torch.amax(logits[0][row["groups"] == j])) for j in range(len(keys))]
    extra = time.perf_counter() - inference_start
    offered = fourth_slot(keys, reference["keys"], values)
    candidates = {"keys": offered, "actions": [representatives[key] for key in offered],
                  "canonical_support": len(keys), "legal_support": len(legal)}
    return {"schedule": schedule, "reference": reference, "candidates": candidates,
            "fit": fit, "group_keys": keys, "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": logits[0].tolist(), "learned_group_scores": values,
            "timing": {"wall": time.perf_counter() - started, "learned_extra_wall": extra}}


def choose_root(state, stream, path, binding, fit=None):
    path = Path(path)
    require(not path.exists(), "One-shot root; no overwrite or retry")
    if fit is None:
        return baseline.choose_root(state, stream, path, binding)
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
        record.update(baseline.select(record["worlds"], len(actions)))
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
        old.save_new(path, record)
    return actions[record["choice"]], record


def auditor():
    return importlib.import_module("scripts.analysis.verify_learned_proposal_arena")


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


CHECK_TARGETS = ["tests/test_learned_proposal_arena.py", "tests/test_learned_proposal_screen.py",
                 "tests/test_rollout_selector_gameplay_155.py", "tests/test_windowless_workers_160.py"]


def identity():
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts/analysis").glob("*.py"))
    paths += [ROOT / name for name in sorted(set(EXTRAS) | set(CHECK_TARGETS))]
    return {"config": CONFIG, "models": {str(k): list(v) for k, v in MODELS.items()},
            "dependencies": {path.relative_to(ROOT).as_posix(): old.file_hash(path) for path in paths},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    binding = screen.validate_freeze()
    final = old.read_json(screen.OUTPUT / "final_status.json")
    require(final["manifest"] == binding and final["complete"] and final["all_prescribed_audits_pass"], "Screen not audited")
    for name in ("result", "report", "audit"):
        verify_hash(final[name])
    require(verified(final["audit"])["passed"], "Screen audit failed")
    for stage in ("probe", "main"):
        for kind in ("sources", "sources_audit", "proposals", "proposals_audit", "selection", "selection_audit", "evaluation", "evaluation_audit"):
            screen.verify_stage(f"{stage}_{kind}", binding)
    retained = verified(old.read_json(screen.OUTPUT / "freeze.json")["manifest"])["parent"]
    files = [*retained["files"], *retained["models"].values(),
             *(artifact(screen.OUTPUT / name) for name in ("manifest.json", "freeze.json", "final_status.json", "state_results.json")),
             *(artifact(path) for path in sorted(screen.OUTPUT.glob("*_lock.json"))),
             *(final[name] for name in ("result", "report", "audit"))]
    base = ROOT / "outputs/rollout_selector_gameplay_155"
    parent_final, manifest = old.read_json(base / "final_status.json"), old.read_json(base / "manifest.json")
    require(parent_final["complete"] and parent_final["integrity_pass"] and parent_final["statistical_pass"]
            and old.digest(manifest) == parent_final["manifest"] == "b287ea26424623b3825e49404467329162fcefb3e114b0064c49bc11b867e8a0", "155 identity")
    for name, expected in {**manifest["dependencies"], **manifest["parent"]["files"]}.items():
        require(old.file_hash(ROOT / name) == expected, f"Changed155 evidence: {name}")
        files.append({"path": name, "sha256": expected})
    for name in ("result", "main_audit"):
        expected = parent_final["result_sha256" if name == "result" else "audit_sha256"]
        require(old.file_hash(base / f"{name}.json") == expected, "155 final evidence changed")
    files.extend(artifact(base / f"{name}.json") for name in ("manifest", "final_status", "result", "main_audit"))
    unique = {item["path"]: {"path": item["path"], "sha256": item["sha256"]} for item in files}
    for item in unique.values():
        verify_hash(item)
    return {"screen": binding, "reference": parent_final["manifest"], "files": list(unique.values())}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    code = "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(" + repr(["-q", *CHECK_TARGETS]) + "))"
    started = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Identity changed during checks")
    old.save_new(folder / "checks.json", {"identity": before, "exit_code": result.returncode,
                 "targets": CHECK_TARGETS, "seconds": time.perf_counter() - started,
                 "logs": [artifact(folder / name) for name in ("stdout.txt", "stderr.txt")]})
    return result.returncode


def prepare(checks_path):
    require(not (OUTPUT / "manifest.json").exists(), "Already prepared")
    evidence = old.read_json(checks_path)
    require(evidence["identity"] == identity() and evidence["exit_code"] == 0, "Current passing checks required")
    for item in evidence["logs"]:
        verify_hash(item)
    parents = parent_evidence()
    schedule = {stage: specs(stage) for stage in ("probe", "main")}
    for stage, rows in schedule.items():
        require(rows == auditor().independent_specs(stage), "Independent schedule mismatch")
    old.save_new(OUTPUT / "schedule.json", schedule)
    manifest = {"identity": identity(), "parent": parents, "checks": artifact(checks_path),
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
    require(verified(manifest["schedule"]) == {s: specs(s) for s in ("probe", "main")}, "Frozen schedule changed")
    for item in [*checks["logs"], *manifest["parent"]["files"]]:
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
                        "full_inner_replay": True, **result})
    return artifact(path, index=item["index"], continuations=sum(r["continuations"] for r in result["roots"]))


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
    audits = run_pool(stage, "audits", games, binding, [OUTPUT / f"{stage}_games_lock.json"])
    for item, game in zip(audits, games):
        receipt = verified(item)
        require(receipt["passed"] and receipt["full_inner_replay"] and receipt["game_artifact"] == game, "Audit accounting")
    old.save_new(OUTPUT / f"{stage}_complete.json", {"manifest": binding, "time": old.now(),
                 "games": len(games), "full_inner_replay": True,
                 "game_lock": artifact(OUTPUT / f"{stage}_games_lock.json"),
                 "audit_lock": artifact(OUTPUT / f"{stage}_audits_lock.json"),
                 "continuations": sum(item["continuations"] for item in audits)})


def cost_projection(game_lock, audit_lock, size_bytes, free_bytes):
    # Whole-stage elapsed times include parallel scheduling and startup overhead.
    ratio = 402 / 12
    forecast = 1.25 * ratio * (game_lock["elapsed_seconds"] + audit_lock["elapsed_seconds"])
    projected = int(1.5 * ratio * size_bytes + 512 * 1024**2)
    return {"forecast_seconds": forecast, "forecast_hours": forecast / 3600,
            "projected_main_bytes": projected, "free_bytes": free_bytes,
            "reasonable": forecast <= 60 * 3600 and free_bytes - projected >= 5 * 1024**3}


def operational_decision(binding):
    games, audits = (stage_lock("probe", kind, binding) for kind in ("games", "audits"))
    size = sum(path.stat().st_size for path in (OUTPUT / "probe").rglob("*") if path.is_file())
    value = {"manifest": binding, "time": old.now(), "outcome_blind": True,
             "probe": artifact(OUTPUT / "probe_complete.json"),
             **cost_projection(games, audits, size, shutil.disk_usage(ROOT).free),
             "caveat": "Twelve-game timing estimate with25%time/50%storage allowances; no strength used."}
    old.save_new(OUTPUT / "operational_decision.json", value)
    if not value["reasonable"]:
        old.save_new(OUTPUT / "operational_hold.json", value)
        event("operational_hold", forecast_hours=value["forecast_hours"])
        return False
    event("probe_passed_main_authorized", forecast_hours=value["forecast_hours"])
    return True


def finish(binding):
    for stage in ("probe", "main"):
        games, audits = (stage_lock(stage, kind, binding) for kind in ("games", "audits"))
        for game, item in zip(games["files"], audits["files"]):
            receipt = verified(item)
            require(receipt["manifest"] == binding and receipt["passed"] and receipt["full_inner_replay"]
                    and receipt["game_artifact"] == game, "Final receipt linkage")
            saved = verified(game)
            require(len(saved["trace"]) == len(receipt["roots"]), "Outer/root accounting")
            for move, root in zip(saved["trace"], receipt["roots"]):
                require(move["root_sha256"] == root["sha256"] == old.file_hash(root_path(stage, game["index"], move["turn"])), "Final root mutation")
    receipts = [verified(item) for item in stage_lock("main", "audits", binding)["files"]]
    rows = [receipt["game"] for receipt in receipts]
    summary = statistics(rows)
    independent = auditor().independent_statistics(rows)
    auditor().compare(summary, independent)
    roots = [root for receipt in receipts for root in receipt["roots"]]
    def part_summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part) / len(part),
                "mean_margin": sum(r["margin"] for r in part) / len(part)}
    def latency(role):
        values = [r["timing"]["wall"] for r in roots if r["planner"] == role]
        return {"count": len(values), "mean": float(np.mean(values)), "p95": float(np.quantile(values, .95))}
    result = {"manifest": binding, "time": old.now(), **summary, "rows": rows,
              "all_prescribed_audits_pass": True, "full_inner_replay": True,
              "fits": {str(fit): part_summary([r for r in rows if r["fit"] == fit]) for fit in MODELS},
              "seats": {str(seat): part_summary([r for r in rows if r["seat"] == seat]) for seat in (0, 1)},
              "decision_seconds": {role: latency(role) for role in ("candidate", "reference")},
              "continuations": sum(r["continuations"] for r in roots),
              "private_roots": sum(bool(r["privacy"]["own"] or r["privacy"]["opponent"]) for r in roots),
              "scope": "Learned-proposal package versus exact155 at32 worlds; no full-game raw-ranked control, equal-wall-time or strongest-agent claim."}
    old.save_new(OUTPUT / "result.json", result)
    old.save_new(OUTPUT / "final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "games": 402, "full_inner_replay": True, "independent_statistics": independent,
                 "roots": len(roots), "continuations": result["continuations"],
                 "result_sha256": old.file_hash(OUTPUT / "result.json"),
                 "probe": artifact(OUTPUT / "probe_complete.json"), "main": artifact(OUTPUT / "main_complete.json")})
    report = (f"# Learned fourth-proposal full-game result\n\n{summary['games']} fresh games, all three fixed161 fits, both seats. "
              f"Score {summary['share']:.4%}; paired95% interval {summary['ci95']}.\n\n"
              f"Classification: {summary['classification']}. Superiority: {summary['superiority_established']}; "
              f"practical progress: {summary['practical_progress']}.\n\n"
              "Every outer game and collected continuation fully replayed; independent statistics agree. "
              "Both planners use32 worlds; only the candidate's fourth proposal changes. "
              "The interval conditions on all three fixed models. No full-game raw-ranked-control, "
              "equal-wall-time, strongest-agent, superhuman or repeated-learning-loop claim. "
              "No automatic extension, training, promotion, successor or monitoring.\n")
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report)
    require(validate_freeze() == binding, "Final dependency mutation")
    old.save_new(OUTPUT / "final_status.json", {"complete": True, "manifest": binding, "time": old.now(),
                 "classification": summary["classification"], "all_prescribed_audits_pass": True,
                 "full_inner_replay": True, "result": artifact(OUTPUT / "result.json"),
                 "report": artifact(OUTPUT / "report.md"), "audit": artifact(OUTPUT / "final_audit.json"),
                 "automatic_successor": False})
    event("complete", classification=summary["classification"])


def run():
    log = OUTPUT / "runtime"
    log.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(log / "stdout.txt", log / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot pipeline; no automatic retry")
        binding = validate_freeze()
        old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(), "pid": os.getpid(),
                     "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
        try:
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
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_learned_proposal_arena.py", "run"],
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
