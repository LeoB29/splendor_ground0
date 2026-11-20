"""155: fixed complete-game deployment of the 154 terminal-rollout selector."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import run_rollout_selector as parent
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

old = parent.old
OUTPUT = Path("outputs/rollout_selector_gameplay_155")
PROTOCOL = Path("docs/engineering/rollout_selector_gameplay_155.md")
VERIFIER = Path("scripts/analysis/verify_rollout_selector_gameplay_155.py")
PARENT_BINDING = "8ee50225d1e73b5ea778192ce5b32cb5fdad36d29ae2ab139c62042e6dd4350f"
PARENT_AUDIT = "7e12fbe45201e41cb35930bb0e5a9553686d80572a8ad95b73a4b907e75d8ddf"
CONFIG = {
    "games": 400, "probe_games": 8, "workers": 8, "torch_threads": 1,
    "worlds": 32, "max_candidates": 4, "max_turns": 400,
    "repetition_limit": 4, "no_progress_limit": 60,
    "environment_seed": {"main": 1_550_000_000, "probe": 1_559_000_000},
    "stream_tag": "155-independent-search-stream-v1",
    "candidate_tag": "155-candidate-v1", "candidate_split": "155-live-v1",
    "world_tag": "155-world-v1", "bootstrap_seed": 1_553_000_000,
    "bootstrap_replicates": 50000, "bootstrap_chunk": 1000,
    "minimum_share": .55, "minimum_ci_lower_exclusive": .50,
    "main_continuation_ceiling": 10_240_000,
    "training": False, "arena": True, "promotion": False, "automatic_successor": False,
}


def parent_inputs():
    assert parent.check_identity() == PARENT_BINDING
    assert old.file_hash(parent.OUTPUT/"final_audit.json") == PARENT_AUDIT
    final = old.read_json(parent.OUTPUT/"final_status.json")
    audit = old.read_json(parent.OUTPUT/"final_audit.json")
    assert final["complete"] and audit["passed"] and final["audit_sha256"] == PARENT_AUDIT
    assert final["result_sha256"] == audit["result_sha256"] == old.file_hash(parent.OUTPUT/"result.json")
    paths = [parent.OUTPUT/n for n in ("manifest.json", "final_audit.json", "final_status.json", "result.json")]
    paths += [parent.VERIFIER]
    return {"binding": PARENT_BINDING, "files": {p.as_posix(): old.file_hash(p) for p in paths}}


def identity():
    files = [Path(__file__).relative_to(Path.cwd()), PROTOCOL, VERIFIER,
             Path("run_rollout_selector_gameplay_155.py"), Path("tests/test_rollout_selector_gameplay_155.py")]
    return {"format": "rollout_selector_gameplay_155_v1", "config": CONFIG,
            "parent": parent_inputs(), "dependencies": {p.as_posix(): old.file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def check_identity():
    saved = old.read_json(OUTPUT/"manifest.json")
    if saved != identity():
        raise ValueError("155 frozen identity changed")
    return old.digest(saved)


def progress(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    old.atomic_write(OUTPUT/"progress.json", row)
    print(row, flush=True)


def schedule(stage):
    if stage not in ("probe", "main"):
        raise ValueError("Unknown stage")
    n = CONFIG["probe_games" if stage == "probe" else "games"]
    return [{"index": i, "pair": i//2, "candidate_seat": i%2,
             "environment_seed": CONFIG["environment_seed"][stage]+i//2,
             "search_stream": old.digest([CONFIG["stream_tag"], stage, i])} for i in range(n)]


def root_schedule(state, stream):
    """Only the player's private RNG stream and observer-legal information."""
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    candidate = old.digest([CONFIG["candidate_tag"], stream, belief])
    seeds = [f"{CONFIG['world_tag']}|{stream}|{belief}|{j:02d}" for j in range(CONFIG["worlds"])]
    return {"belief_sha256": belief, "candidate_nonce": candidate, "world_seeds": seeds}


def candidate_set(state, nonce):
    return old.candidates(state, CONFIG["candidate_split"], nonce)


def privacy(state):
    counts = [sum(player.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE
                  for i in range(len(player.reserved_cards))) for player in state.players]
    return {"own": counts[state.current_player], "opponent": counts[1-state.current_player]}


def select(worlds, support):
    assert 1 <= support <= CONFIG["max_candidates"] and len(worlds) == CONFIG["worlds"]
    assert all(len(w["outcomes"]) == support for w in worlds)
    if any(o["reason"] not in old.VALID_NATURAL_TERMINATIONS or o["wdl"] not in (-1., 0., 1.)
           for w in worlds for o in w["outcomes"]):
        raise ValueError("Invalid inner ending: no exclusion or replacement")
    means = [sum(w["outcomes"][j]["wdl"] for w in worlds)/len(worlds) for j in range(support)]
    return {"choice": max(range(support), key=lambda j: means[j]), "means": means}


def root_path(stage, index, turn):
    return OUTPUT/stage/"roots"/f"game_{index:04d}"/f"turn_{turn:03d}.json"


def game_path(stage, index):
    return OUTPUT/stage/"games"/f"game_{index:04d}.json"


def choose_root(state, stream, path, binding):
    spec = root_schedule(state, stream)
    snapshot = old.serialize_state_snapshot(state)
    if path.exists():
        record = old.read_json(path)
        assert record["manifest"] == binding and old.digest(record["snapshot"]) == old.digest(snapshot)
        assert record["schedule"] == spec and "failure" not in record
        assert {k: record[k] for k in ("choice", "means")} == select(record["worlds"], len(record["candidates"]["actions"]))
        return old.decode_action(record["candidates"]["actions"][record["choice"]]), record
    wall, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "time": old.now(), "snapshot": snapshot,
              "schedule": spec, "privacy": privacy(state), "worlds": []}
    try:
        legal = old.SplendorEnv().legal_actions(state)
        scores = old._RAW.score_actions(state, legal)
        assert len(scores) == len(legal) and np.isfinite(scores).all()
        record["scores"] = [float(v) for v in scores]
        record["candidates"] = candidate_set(state, spec["candidate_nonce"])
        assert record["candidates"]["actions"][0] == old.serialize_action(old._RAW.choose_action(old.SplendorEnv(), state, legal))
        actions = [old.decode_action(a) for a in record["candidates"]["actions"]]
        for seed in spec["world_seeds"]:
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []}
            record["worlds"].append(row)
            for action in actions:
                assert action in old.SplendorEnv().legal_actions(world)
                result = old.trajectory(world, action)
                result["wdl"] = old.outcome_value(result["winner"], state.current_player) if result["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                row["outcomes"].append(result)
                if result["wdl"] is None:
                    raise ValueError("Invalid rollout ending")
        record.update(select(record["worlds"], len(actions)))
    except Exception:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}
        old.save_new(path, record)
    return actions[record["choice"]], record


def play_game(stage, spec, binding):
    path = game_path(stage, spec["index"])
    if path.exists():
        record = old.read_json(path)
        assert record["spec"] == spec and record["manifest"] == binding and "failure" not in record
        assert record["reason"] in old.VALID_NATURAL_TERMINATIONS
        return {"index": spec["index"], "sha256": old.file_hash(path)}
    wall, cpu = time.perf_counter(), time.process_time()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    record = {"manifest": binding, "spec": spec, "time": old.now(), "trace": [],
              "initial_state_sha256": old.digest(old.serialize_state_snapshot(state))}
    visits, forced, idle, hasher = Counter(), 0, 0, hashlib.sha256()
    reason, winner = "completed", None
    try:
        while not state.terminal:
            visits[old.state_signature(state)] += 1
            if visits[old.state_signature(state)] >= CONFIG["repetition_limit"]:
                reason = "repetition_cutoff"
                break
            if state.turn_index >= CONFIG["max_turns"]:
                reason = "max_turns"
                break
            legal = env.legal_actions(state)
            if not legal:
                reason = "stalled"
                break
            row = {"turn": state.turn_index, "actor": state.current_player,
                   "state_sha256": old.digest(old.serialize_state_snapshot(state))}
            if state.current_player == spec["candidate_seat"]:
                rp = root_path(stage, spec["index"], state.turn_index)
                action, _ = choose_root(state, spec["search_stream"], rp, binding)
                row["root_sha256"] = old.file_hash(rp)
            else:
                start_wall, start_cpu = time.perf_counter(), time.process_time()
                action = old._RAW.choose_action(env, state, legal)
                row["timing"] = {"wall": time.perf_counter()-start_wall, "cpu": time.process_time()-start_cpu}
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                break
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            hasher.update(old.digest(row["action"]).encode())
            nxt = env.step(state, action)
            forced = forced+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, nxt, action) else idle+1
            state = nxt
            old.atomic_write(OUTPUT/stage/"telemetry"/f"game_{spec['index']:04d}.json",
                             {"time": old.now(), "index": spec["index"], "turns": len(record["trace"]),
                              "elapsed_seconds": time.perf_counter()-wall})
            if forced >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            if idle >= CONFIG["no_progress_limit"]:
                reason = "no_progress_cutoff"
                break
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason, winner=winner)
        if reason not in old.VALID_NATURAL_TERMINATIONS:
            raise ValueError(f"Invalid outer ending: {reason}")
    except Exception:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        final = old.serialize_state_snapshot(state)
        record.update(final_state=final, final_state_sha256=old.digest(final),
                      turns=len(record["trace"]), trace_sha256=hasher.hexdigest(),
                      timing={"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu})
        old.save_new(path, record)
    return {"index": spec["index"], "sha256": old.file_hash(path)}


def collect(stage):
    binding = check_identity()
    launch = OUTPUT/f"{stage}_launch.json"
    assert old.read_json(launch)["manifest"] == binding
    lock = OUTPUT/f"{stage}_games_lock.json"
    if lock.exists():
        saved = old.read_json(lock)
        assert saved["manifest"] == binding and saved["launch_sha256"] == old.file_hash(launch)
        assert sorted(int(k) for k in saved["files"]) == list(range(len(schedule(stage))))
        for i, sha in saved["files"].items():
            assert old.file_hash(game_path(stage, int(i))) == sha
        return saved
    start = time.perf_counter()
    files = {}
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=old.initialize_worker) as pool:
        pending = [pool.submit(play_game, stage, spec, binding) for spec in schedule(stage)]
        try:
            for future in as_completed(pending):
                info = future.result()
                files[str(info["index"])] = info["sha256"]
                progress(f"{stage}_collection", games_complete=len(files), games_total=len(pending), elapsed_seconds=time.perf_counter()-start)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    saved = {"manifest": binding, "launch_sha256": old.file_hash(launch), "time": old.now(),
             "files": files, "collection_wall_seconds": time.perf_counter()-start}
    old.save_new(lock, saved)
    return saved


def quantiles(values):
    assert values and all(np.isfinite(v) and v >= 0 for v in values)
    return {"count": len(values), "mean": sum(values)/len(values),
            "p95": float(np.quantile(values, .95)), "max": max(values), "sum": sum(values)}


def observations(stage):
    rows, roots, raw_times = [], [], []
    lock = old.read_json(OUTPUT/f"{stage}_games_lock.json")
    for spec in schedule(stage):
        path = game_path(stage, spec["index"])
        assert old.file_hash(path) == lock["files"][str(spec["index"])]
        game = old.read_json(path)
        assert game["reason"] in old.VALID_NATURAL_TERMINATIONS
        score = (old.outcome_value(game["winner"], spec["candidate_seat"])+1)/2
        scores = [p["score"] for p in game["final_state"]["players"]]
        rows.append({"index": spec["index"], "pair": spec["pair"], "seat": spec["candidate_seat"],
                     "score": score, "margin": scores[spec["candidate_seat"]]-scores[1-spec["candidate_seat"]],
                     "turns": game["turns"], "reason": game["reason"], "timing": game["timing"]})
        for move in game["trace"]:
            if "root_sha256" not in move:
                raw_times.append(move["timing"])
                continue
            rp = root_path(stage, spec["index"], move["turn"])
            assert old.file_hash(rp) == move["root_sha256"]
            r = old.read_json(rp)
            outcomes = [o for w in r["worlds"] for o in w["outcomes"]]
            roots.append({"index": spec["index"], "seat": spec["candidate_seat"], "turn": move["turn"],
                          "choice": r["choice"], "support": len(r["candidates"]["actions"]),
                          "canonical_support": r["candidates"]["canonical_support"],
                          "legal_support": r["candidates"]["legal_support"],
                          "privacy": r["privacy"], "timing": r["timing"], "continuations": len(outcomes),
                          "moves": sum(o["turns"] for o in outcomes),
                          "reasons": dict(Counter(o["reason"] for o in outcomes))})
    return {"games": rows, "roots": roots, "raw_timings": raw_times}


def statistics(rows):
    assert len(rows) == CONFIG["games"] and [r["index"] for r in rows] == list(range(CONFIG["games"]))
    pairs = []
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        assert a["pair"] == b["pair"] == i//2 and (a["seat"], b["seat"]) == (0, 1)
        assert a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.)
        pairs.append((a["score"]+b["score"])/2)
    values = np.asarray(pairs, dtype=np.float64)
    rng, boot = np.random.default_rng(CONFIG["bootstrap_seed"]), []
    for start in range(0, CONFIG["bootstrap_replicates"], CONFIG["bootstrap_chunk"]):
        count = min(CONFIG["bootstrap_chunk"], CONFIG["bootstrap_replicates"]-start)
        indices = rng.integers(0, len(pairs), size=(count, len(pairs)))
        boot.extend(values[indices].mean(axis=1).tolist())
    ci = np.quantile(boot, [.025, .975]).tolist()
    share = sum(pairs)/len(pairs)
    gates = {"material_share": share >= CONFIG["minimum_share"],
             "positive_lower_bound": ci[0] > CONFIG["minimum_ci_lower_exclusive"]}
    def summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part)/len(part),
                "wdl": {k: sum(r["score"] == v for r in part) for k, v in (("wins", 1.), ("draws", .5), ("losses", 0.))},
                "mean_margin": sum(r["margin"] for r in part)/len(part),
                "mean_turns": sum(r["turns"] for r in part)/len(part)}
    return {**summary(rows), "pairs": len(pairs), "ci95": ci, "statistical_gates": gates,
            "statistical_pass": all(gates.values()), "seats": {str(s): summary(rows[s::2]) for s in (0, 1)}}


def operational(data):
    roots = data["roots"]
    reasons = Counter()
    for root in roots:
        reasons.update(root["reasons"])
    def summary(part):
        return {"roots": len(part), "changes": sum(r["choice"] != 0 for r in part),
                "change_fraction": sum(r["choice"] != 0 for r in part)/len(part) if part else None,
                "support_counts": dict(Counter(str(r["support"]) for r in part)),
                "own_private_roots": sum(r["privacy"]["own"] > 0 for r in part),
                "opponent_private_roots": sum(r["privacy"]["opponent"] > 0 for r in part)}
    return {**summary(roots), "continuations": sum(r["continuations"] for r in roots),
            "continuation_moves": sum(r["moves"] for r in roots), "inner_reasons": dict(reasons),
            "outer_reasons": dict(Counter(g["reason"] for g in data["games"])),
            "decision_time": {k: quantiles([r["timing"][k] for r in roots]) for k in ("wall", "cpu")},
            "raw_decision_time": {k: quantiles([r[k] for r in data["raw_timings"]]) for k in ("wall", "cpu")},
            "game_time": {k: quantiles([r["timing"][k] for r in data["games"]]) for k in ("wall", "cpu")},
            "turns": quantiles([g["turns"] for g in data["games"]]),
            "seats": {str(s): summary([r for r in roots if r["seat"] == s]) for s in (0, 1)},
            "phases": {str(s): summary([r for r in roots if old.phase(r["turn"]) == s]) for s in range(3)},
            "privacy_strata": {str(s): summary([r for r in roots if bool(r["privacy"]["own"] or r["privacy"]["opponent"]) == s]) for s in (False, True)}}


def audit_stage(stage):
    path = OUTPUT/f"{stage}_audit.json"
    if not path.exists():
        subprocess.run([sys.executable, str(VERIFIER), stage], check=True)
    result = old.read_json(path)
    assert result["passed"] and result["manifest"] == check_identity()
    assert result["games_lock_sha256"] == old.file_hash(OUTPUT/f"{stage}_games_lock.json")
    if stage == "main":
        assert result["result_sha256"] == old.file_hash(OUTPUT/"result.json")
        assert result["observations_sha256"] == old.file_hash(OUTPUT/"observations.json")
    return result


def verify_test_evidence():
    evidence = old.read_json(OUTPUT/"preparation/tests.json")
    assert evidence["exit_code"] == 0 and evidence["full_repository"]
    assert evidence["identity"] == identity()
    assert evidence["log_sha256"] == old.file_hash(OUTPUT/"preparation/tests.txt")
    return old.file_hash(OUTPUT/"preparation/tests.json")


def launch_stage(stage, binding, upstream):
    path = OUTPUT/f"{stage}_launch.json"
    value = {"manifest": binding, "schedule_sha256": old.file_hash(OUTPUT/"schedule.json"), "upstream": upstream}
    if path.exists():
        saved = old.read_json(path)
        assert all(saved[k] == v for k, v in value.items())
    else:
        old.save_new(path, {**value, "time": old.now()})


def prepare(resume=False):
    if not (OUTPUT/"manifest.json").exists():
        test_sha = verify_test_evidence()
        old.save_new(OUTPUT/"manifest.json", identity())
        old.save_new(OUTPUT/"schedule.json", {s: schedule(s) for s in ("probe", "main")})
        old.save_new(OUTPUT/"freeze.json", {"time": old.now(), "manifest": check_identity(), "tests_sha256": test_sha,
                     "schedule_sha256": old.file_hash(OUTPUT/"schedule.json")})
    elif not resume:
        raise ValueError("Existing freeze: only explicit identical resume")
    binding = check_identity()
    frozen = old.read_json(OUTPUT/"freeze.json")
    assert frozen["tests_sha256"] == verify_test_evidence() and frozen["manifest"] == binding
    assert frozen["schedule_sha256"] == old.file_hash(OUTPUT/"schedule.json")
    assert old.read_json(OUTPUT/"schedule.json") == {s: schedule(s) for s in ("probe", "main")}
    if (OUTPUT/"preflight.json").exists():
        preflight = old.read_json(OUTPUT/"preflight.json")
        assert preflight["passed"] and preflight["manifest"] == binding
        assert preflight["audit_sha256"] == old.file_hash(OUTPUT/"probe_audit.json")
        assert preflight["timing_sha256"] == old.file_hash(OUTPUT/"timing_probe.json")
        return
    launch_stage("probe", binding, {"freeze_sha256": old.file_hash(OUTPUT/"freeze.json")})
    collect("probe")
    audit = audit_stage("probe")
    lock = old.read_json(OUTPUT/"probe_games_lock.json")
    ratio = CONFIG["games"]/CONFIG["probe_games"]
    timing = {"outcome_blind": True, "probe_games": CONFIG["probe_games"],
              "collection_seconds": lock["collection_wall_seconds"], "audit_seconds": audit["wall_seconds"],
              "main_projection_seconds": ratio*(lock["collection_wall_seconds"]+audit["wall_seconds"]),
              "probe_roots": audit["roots"], "probe_continuations": audit["continuations"],
              "caveat": "Eight games; scheduling, longer games and contention can change cost. Fixed budgets remain unchanged."}
    parent.save_identical_or_new(OUTPUT/"timing_probe.json", timing)
    old.save_new(OUTPUT/"preflight.json", {"passed": True, "time": old.now(), "manifest": binding,
                 "audit_sha256": old.file_hash(OUTPUT/"probe_audit.json"), "timing_sha256": old.file_hash(OUTPUT/"timing_probe.json")})
    progress("preflight_passed", **timing)


def run_main():
    binding = check_identity()
    preflight = old.read_json(OUTPUT/"preflight.json")
    assert preflight["passed"] and preflight["manifest"] == binding
    assert preflight["audit_sha256"] == old.file_hash(OUTPUT/"probe_audit.json")
    assert preflight["timing_sha256"] == old.file_hash(OUTPUT/"timing_probe.json")
    launch_stage("main", binding, {"preflight_sha256": old.file_hash(OUTPUT/"preflight.json")})
    collect("main")
    data = observations("main")
    parent.save_identical_or_new(OUTPUT/"observations.json", data)
    result = {**statistics(data["games"]), "operational": operational(data),
              "interpretation": "Complete system against greedy035 with additional compute; no 084 or superhuman claim."}
    assert result["operational"]["continuations"] <= CONFIG["main_continuation_ceiling"]
    parent.save_identical_or_new(OUTPUT/"result.json", result)
    progress("main_collected_audit_pending", games=CONFIG["games"])
    # Deliberately unconditional: replay runs even when the strength gates fail.
    audit = audit_stage("main")
    assert verify_test_evidence() == old.read_json(OUTPUT/"freeze.json")["tests_sha256"]
    check_identity()
    final = {"complete": True, "finished": old.now(), "manifest": binding,
             "statistical_pass": result["statistical_pass"], "integrity_pass": audit["passed"],
             "advance_gate_passed": result["statistical_pass"] and audit["passed"],
             "result_sha256": old.file_hash(OUTPUT/"result.json"), "audit_sha256": old.file_hash(OUTPUT/"main_audit.json"),
             "elapsed_seconds": (datetime.fromisoformat(old.now())-datetime.fromisoformat(old.read_json(OUTPUT/"probe_launch.json")["time"])).total_seconds(),
             **{k: CONFIG[k] for k in ("training", "arena", "promotion", "automatic_successor")}}
    report = (f"# Experiment 155 result\n\nComplete and independently replayed. Candidate score share {result['share']:.4%}; "
              f"95% paired CI [{result['ci95'][0]:.4%}, {result['ci95'][1]:.4%}]. "
              f"W/D/L: {result['wdl']['wins']}/{result['wdl']['draws']}/{result['wdl']['losses']}.\n\n"
              f"Statistical pass: {result['statistical_pass']}. Integrity pass: {audit['passed']}. "
              f"All {audit['games']} games, {audit['roots']} roots and {audit['continuations']} continuations replayed.\n\n"
              "The JSON result contains seat/phase/privacy coverage, margins, turns and measured CPU/wall latency. "
              "Uniform observer-legal worlds; no learned opponent posterior. The auditor shares the frozen engine. "
              "Extra-compute comparison against 035 only. No training, promotion or automatic successor.\n")
    report_path = OUTPUT/"report.md"
    if report_path.exists():
        assert report_path.read_text(encoding="utf-8") == report
    else:
        report_path.write_text(report, encoding="utf-8")
    old.save_new(OUTPUT/"final_status.json", final)
    progress("complete", **final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "start", "run", "resume"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        if list(OUTPUT.glob("failure_*.json")):
            raise ValueError("Recorded failure requires review; no automatic retry")
        if (OUTPUT/"final_status.json").exists():
            raise ValueError("Experiment already closed")
        if args.mode != "resume" and (OUTPUT/"main_launch.json").exists():
            raise ValueError("Existing launch requires explicit identical resume")
        try:
            progress("starting", mode=args.mode)
            if args.mode in ("prepare", "start", "resume"):
                prepare(resume=args.mode == "resume")
            if args.mode in ("start", "run", "resume"):
                run_main()
        except Exception:
            old.save_new(OUTPUT/f"failure_{args.mode}_{time.time_ns()}.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
