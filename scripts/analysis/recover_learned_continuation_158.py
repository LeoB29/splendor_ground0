"""Owner-authorized recovery of158's interrupted main replay, without new games.

The frozen verifier is imported unchanged. Only missing whole-game audit tasks
are dispatched; all19 previously completed certificates remain byte-identical.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import argparse
import importlib.util
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
import traceback

from splendor_ai.alphazero import run_learned_continuation as p

SCRIPT = Path(__file__).resolve()
REPOSITORY = SCRIPT.parents[2]
SPEC = importlib.util.spec_from_file_location("recovery158_frozen_audit", REPOSITORY/p.VERIFIER)
a = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = a  # Also executed in every Windows spawn child.
SPEC.loader.exec_module(a)
old, OUTPUT = p.old, p.OUTPUT
FOLDER = OUTPUT/"recovery_main_replay"
PLAN_DOC = Path("docs/engineering/learned_continuation_158_recovery.md")
TEST_FILE = Path("tests/test_learned_continuation_158_recovery.py")
BINDING = "06fcd7d298a3099b11206ff3cc5c4343b5c9c1fb01b3ef975c87fafa015d40ab"
RESULT_SHA = "2abae0ec098780c23dcdfd9d82e78044e99d5d2e36ca0021f8e7ee373d3c4fb4"
GAMES_LOCK_SHA = "b535ff50e868020d8e1392f9ae95526e53f7cf1e866cdf6dec50299708b0b06a"
PRIOR_REPLAYS_DIGEST = "da283a5cd8b66e3992d4f0e24af9c693bd3f7c0bd32cd3d8cdc7b51470c60609"
PRIOR_INDICES = set(range(18)) | {22}
PRIOR_AUDIT_SECONDS = 9145.344842699997


def event(stage, **values):
    print(json.dumps({"time": old.now(), "stage": stage, **values}), flush=True)


def recovery_identity():
    return {path.as_posix(): old.file_hash(path) for path in (SCRIPT, PLAN_DOC, TEST_FILE)}


def certificates(base, stage, files, binding, expected_hashes):
    """Admit exactly the externally sealed existing certificates, never a fallback."""
    folder = base/stage/"replay"
    expected_paths = {folder/f"game_{int(i):04d}.json" for i in expected_hashes}
    assert set(folder.glob("*.json")) == expected_paths, "Unexpected/missing prior replay"
    rows = {}
    for key, sha in expected_hashes.items():
        index = int(key)
        path = folder/f"game_{index:04d}.json"
        assert old.file_hash(path) == sha, "Prior replay changed"
        record = old.read_json(path)
        assert set(record) == {"manifest", "game_sha256", "reconstruction"}
        assert record["manifest"] == binding and record["game_sha256"] == files[key]
        row = record["reconstruction"]
        assert row["game"]["index"] == index
        assert all(math.isfinite(row["audit_timing"][k]) and row["audit_timing"][k] >= 0
                   for k in ("wall", "cpu"))
        rows[index] = row
    return rows


def snapshot(prior_hashes):
    """Read-only whole-evidence check; called before preparation, launch and closure."""
    assert p.validate_freeze() == BINDING
    assert old.file_hash(OUTPUT/"result.json") == RESULT_SHA
    assert old.file_hash(OUTPUT/"main_games_lock.json") == GAMES_LOCK_SHA
    assert old.read_json(OUTPUT/"schedule.json") == {s: list(a.independent_specs(s)) for s in ("probe", "main")}
    frozen, launch, preflight = [old.read_json(OUTPUT/name) for name in
                                 ("freeze.json", "main_launch.json", "preflight.json")]
    assert launch["manifest"] == preflight["manifest"] == BINDING
    assert launch["schedule_sha256"] == frozen["schedule_sha256"]
    assert launch["upstream"] == {"preflight_sha256": old.file_hash(OUTPUT/"preflight.json")}
    assert preflight["passed"] and preflight["audit_sha256"] == old.file_hash(OUTPUT/"probe_audit.json")
    assert preflight["timing_sha256"] == old.file_hash(OUTPUT/"timing_probe.json")
    lock = old.read_json(OUTPUT/"main_games_lock.json")
    assert lock["manifest"] == BINDING and lock["launch_sha256"] == old.file_hash(OUTPUT/"main_launch.json")
    assert set(lock["files"]) == {str(i) for i in range(402)}
    assert set((OUTPUT/"main/games").glob("*.json")) == {a.game_path(OUTPUT, "main", i) for i in range(402)}
    root_hashes = {}
    for i in range(402):
        game_path = a.game_path(OUTPUT, "main", i)
        assert old.file_hash(game_path) == lock["files"][str(i)]
        game = old.read_json(game_path)
        assert game["manifest"] == BINDING and game["spec"] == a.independent_spec("main", i)
        assert "failure" not in game and game["reason"] in ("completed", "forced_pass_deadlock")
        for move in game["trace"]:
            path = a.root_path(OUTPUT, "main", i, move["turn"])
            assert old.file_hash(path) == move["root_sha256"], (i, move["turn"], "root hash")
            assert path.as_posix() not in root_hashes
            root_hashes[path.as_posix()] = move["root_sha256"]
    assert set((OUTPUT/"main/roots").rglob("*.json")) == {Path(path) for path in root_hashes}
    for index, sha in prior_hashes.items():
        assert old.file_hash(OUTPUT/"main/replay"/f"game_{int(index):04d}.json") == sha
    names = ("freeze.json", "manifest.json", "schedule.json", "main_launch.json", "main_games_lock.json",
             "preflight.json", "probe_audit.json", "timing_probe.json", "observations.json", "result.json",
             "preparation/process.json", "preparation/run.txt", "preparation/launcher_stdout.txt",
             "preparation/launcher_stderr.txt")
    return {"manifest": BINDING, "files": {name: old.file_hash(OUTPUT/name) for name in names},
            "root_count": len(root_hashes), "root_hashes_digest": old.digest(root_hashes),
            "prior_replay_hashes": prior_hashes, "recovery_identity": recovery_identity()}


def guard_incomplete():
    for name in ("main_audit.json", "final_status.json", "report.md", "preparation/exit.json"):
        assert not (OUTPUT/name).exists(), f"Existing {name} requires review"
    assert not list(OUTPUT.glob("*failure*.json")), "Recorded failure requires review"


def locks():
    stack = ExitStack()
    try:
        for name in ("preparation/launcher.lock", "run.lock", "main_collection.lock", "audit.lock"):
            stack.enter_context(old.process_lock(OUTPUT/name))
        return stack
    except BaseException:
        stack.close()
        raise


def prepare():
    with locks():
        guard_incomplete()
        assert not (FOLDER/"plan.json").exists()
        prior = {str(int(path.stem.split("_")[1])): old.file_hash(path)
                 for path in (OUTPUT/"main/replay").glob("*.json")}
        assert set(map(int, prior)) == PRIOR_INDICES and old.digest(prior) == PRIOR_REPLAYS_DIGEST
        before = snapshot(prior)
        files = old.read_json(OUTPUT/"main_games_lock.json")["files"]
        rows = certificates(OUTPUT, "main", files, BINDING, prior)
        observations = old.read_json(OUTPUT/"observations.json")
        for index, row in rows.items():
            a.compare(row["game"], observations["games"][index])
            a.compare(row["roots"], [root for root in observations["roots"] if root["index"] == index])
            game = old.read_json(a.game_path(OUTPUT, "main", index))
            assert row["initial_sha256"] == game["initial_state_sha256"]
        tests = old.read_json(FOLDER/"tests.json")
        assert tests["passed"] and tests["identity"] == recovery_identity()
        assert tests["log_sha256"] == old.file_hash(FOLDER/"tests.txt")
        old.save_new(FOLDER/"plan.json", {**before, "time": old.now(), "owner_authorized": True,
                     "authorization": "yes please recover", "completed": sorted(rows),
                     "missing": [i for i in range(402) if i not in rows],
                     "tests_sha256": old.file_hash(FOLDER/"tests.json"), "scientific_change": False,
                     "prior_audit_seconds": PRIOR_AUDIT_SECONDS})
        event("recovery_prepared", existing=len(rows), remaining=402-len(rows), roots=before["root_count"])


def replay_missing(base, stage, files, binding, audited, *, workers=8, context=None):
    missing = sorted(set(map(int, files)) - set(audited))
    if not missing:
        return audited
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=p.h.initialize_worker) as pool:
        futures = {}
        for i in missing:
            fixture = None if context is None else {**context, "spec": a.test_spec(i)}
            futures[pool.submit(a.audit_game, stage, i, files[str(i)], binding, fixture)] = i
        event("replay_workers_started", pid=os.getpid(), existing=len(audited), remaining=len(missing), workers=workers)
        try:
            for future in as_completed(futures):
                i = futures[future]
                reconstruction = future.result()
                old.save_new(base/stage/"replay"/f"game_{i:04d}.json",
                             {"manifest": binding, "game_sha256": files[str(i)], "reconstruction": reconstruction})
                audited[i] = reconstruction
                event("main_replay" if stage == "main" else "test_replay", games_complete=len(audited), games_total=len(files))
        except BaseException:
            for future in futures:
                future.cancel()
            if stage == "main":
                old.save_new(FOLDER/"audit_failure.json", {"time": old.now(), "completed": sorted(audited),
                             "traceback": traceback.format_exc()})
            raise
    return audited


def combine(audited):
    assert set(audited) == set(range(402)), "Full402 replay required on either strength outcome"
    for i in range(0, 402, 2):
        assert audited[i]["initial_sha256"] == audited[i+1]["initial_sha256"]
    return {"games": [audited[i]["game"] for i in range(402)],
            "roots": [root for i in range(402) for root in audited[i]["roots"]]}


def close(plan, audited, wall_seconds):
    data = combine(audited)
    expected = {a.root_path(OUTPUT, "main", row["index"], row["turn"]) for row in data["roots"]}
    assert set((OUTPUT/"main/roots").rglob("*.json")) == expected
    a.compare(data, p.observations("main"))
    a.compare(data, old.read_json(OUTPUT/"observations.json"))
    result = old.read_json(OUTPUT/"result.json")
    stats = a.independent_statistics(data["games"])
    a.compare(stats, {key: result[key] for key in stats})
    a.compare(a.independent_operational(data), result["operational"])
    current = snapshot(plan["prior_replay_hashes"])
    assert all(current[key] == plan[key] for key in current), "Frozen/recovery input changed"
    replay_paths = {OUTPUT/"main/replay"/f"game_{i:04d}.json" for i in range(402)}
    assert set((OUTPUT/"main/replay").glob("*.json")) == replay_paths
    files = old.read_json(OUTPUT/"main_games_lock.json")["files"]
    replay_hashes = {}
    for i in range(402):
        path = OUTPUT/"main/replay"/f"game_{i:04d}.json"
        record = old.read_json(path)
        assert record == {"manifest": BINDING, "game_sha256": files[str(i)], "reconstruction": audited[i]}
        replay_hashes[str(i)] = old.file_hash(path)
    audit = {"passed": True, "manifest": BINDING, "games": 402,
             "games_lock_sha256": GAMES_LOCK_SHA, "roots": len(data["roots"]),
             "outer_moves": sum(game["turns"] for game in data["games"]),
             "continuations": sum(root["continuations"] for root in data["roots"]),
             "continuation_moves": sum(root["moves"] for root in data["roots"]),
             "wall_seconds": wall_seconds + PRIOR_AUDIT_SECONDS,
             "worker_cpu_seconds": sum(row["audit_timing"]["cpu"] for row in audited.values()),
             "shared_frozen_engine": True, "every_root_information_invariance": True,
             "every_continuation_information_invariance": True, "every_learned_decision_evidence_replayed": True,
             "replay_files": replay_hashes, "result_sha256": RESULT_SHA,
             "observations_sha256": old.file_hash(OUTPUT/"observations.json"),
             "recovery": {"plan_sha256": old.file_hash(FOLDER/"plan.json"), "reused_games": plan["completed"],
                          "newly_replayed_games": plan["missing"], "recovery_wall_seconds": wall_seconds,
                          "timing_caveat": "Wall sums logged prior audit segment and recovery; excludes downtime. CPU omits lost unfinished work."}}
    old.save_new(OUTPUT/"main_audit.json", audit)
    final = {"complete": True, "finished": old.now(), "manifest": BINDING,
             "statistical_pass": result["statistical_pass"], "integrity_pass": True,
             "advance_gate_passed": result["statistical_pass"], "result_sha256": RESULT_SHA,
             "audit_sha256": old.file_hash(OUTPUT/"main_audit.json"),
             "elapsed_seconds": (datetime.fromisoformat(old.now())-datetime.fromisoformat(
                 old.read_json(OUTPUT/"probe_launch.json")["time"])).total_seconds(),
             **{k: p.CONFIG[k] for k in ("training", "arena", "promotion", "automatic_successor")},
             "recovery_plan_sha256": old.file_hash(FOLDER/"plan.json")}
    report = (f"# Experiment158 result\n\nCandidate score {result['share']:.4%}; stratified paired95% CI "
              f"[{result['ci95'][0]:.4%}, {result['ci95'][1]:.4%}]. W/D/L "
              f"{result['wdl']['wins']}/{result['wdl']['draws']}/{result['wdl']['losses']}. "
              f"Mean margin {result['mean_margin']:+.4f}.\n\nStatistical pass: {result['statistical_pass']}. "
              f"Integrity pass: True. All402 games, {audit['roots']} roots and {audit['continuations']} "
              "continuations independently reconstructed. The19 preserved certificates and383 recovered "
              "replays jointly cover the full fixed panel. No collected game was rerun or changed.\n\n"
              "Both planners use32 worlds. This tests exact157 continuations for both simulated players "
              "against greedy035 continuations, conditional on three fixed models. The complete substitution "
              "includes proposal randomness; no unique cause, equal-time efficiency, repeated improvement "
              "or superhuman performance is established. Full statistics, gates and operational summaries "
              "were independently checked. No training, promotion or successor ran.\n")
    with (OUTPUT/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    old.save_new(OUTPUT/"final_status.json", final)
    old.save_new(FOLDER/"complete.json", {"time": old.now(), "manifest": BINDING,
                 "final_status_sha256": old.file_hash(OUTPUT/"final_status.json"),
                 "main_audit_sha256": final["audit_sha256"], "result_sha256": RESULT_SHA})
    event("complete", **final)


def resume(authorized):
    assert authorized, "Owner authorization is required"
    wall = time.perf_counter()
    with locks():
        guard_incomplete()
        assert not any((FOLDER/name).exists() for name in ("launch.json", "failure.json", "audit_failure.json"))
        plan = old.read_json(FOLDER/"plan.json")
        assert plan["owner_authorized"] and plan["tests_sha256"] == old.file_hash(FOLDER/"tests.json")
        current = snapshot(plan["prior_replay_hashes"])
        assert all(current[key] == plan[key] for key in current)
        files = old.read_json(OUTPUT/"main_games_lock.json")["files"]
        audited = certificates(OUTPUT, "main", files, BINDING, plan["prior_replay_hashes"])
        assert sorted(audited) == plan["completed"]
        assert sorted(set(range(402))-set(audited)) == plan["missing"]
        old.save_new(FOLDER/"launch.json", {"time": old.now(), "pid": os.getpid(), "owner_authorized": True,
                     "plan_sha256": old.file_hash(FOLDER/"plan.json"), "manifest": BINDING,
                     "reused": len(audited), "remaining": len(plan["missing"])})
        try:
            audited = replay_missing(OUTPUT, "main", files, BINDING, audited)
            close(plan, audited, time.perf_counter()-wall)
        except BaseException:
            old.save_new(FOLDER/"failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


def spawn_probe(fixture_path):
    fixture = old.read_json(fixture_path)
    base = Path(fixture["base_dir"]).resolve()
    assert base != OUTPUT.resolve() and OUTPUT.resolve() not in base.parents
    assert fixture["kind"] == "158-spawn-test-fixture-v1"
    files = {str(row["index"]): row["sha256"] for row in fixture["games"]}
    assert set(files) == {"0", "1"}
    prior = fixture["prior_replay_hashes"]
    assert set(prior) == {"0"}
    audited = certificates(base, "test", files, fixture["binding"], prior)
    rows = replay_missing(base, "test", files, fixture["binding"], audited, workers=2,
                          context={"base_dir": str(base), "initial_snapshot": fixture["initial_snapshot"]})
    assert old.file_hash(base/"test/replay/game_0000.json") == prior["0"]
    assert set(rows) == {0, 1} and rows[0]["initial_sha256"] == rows[1]["initial_sha256"]
    event("spawn_probe_passed", passed=True, reused=1, newly_replayed=1,
          roots=sum(len(row["roots"]) for row in rows.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "resume", "spawn-probe"))
    parser.add_argument("--owner-authorized", action="store_true")
    parser.add_argument("--fixture", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "resume":
        resume(args.owner_authorized)
    else:
        assert args.fixture is not None
        spawn_probe(args.fixture)
