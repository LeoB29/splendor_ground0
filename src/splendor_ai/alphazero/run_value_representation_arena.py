"""Experiment 146 conditional arena: trainable residual versus public value."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
import subprocess
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero.network import NetworkPrediction
from splendor_ai.alphazero.search import SearchConfig
from splendor_ai.alphazero.strong_policy import BudgetSearch
from splendor_ai.alphazero.run_public_value_search import (
    PublicValueInference, CHECKPOINT, CHECKPOINT_SHA, SIMPLE, SIMPLE_SHA,
    replay_summary, search_record, describe,
)
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.training.replay import collect_game_replay, serialize_state_snapshot, _serialize_action

from splendor_ai.alphazero import run_value_representation as pilot

OUTPUT = pilot.OUTPUT / "arena"
ARMS = ("learned", "public")
CONFIG = {
    "games": 800, "seed_pairs": 400, "seed_start": 1_461_000_000,
    "probe_seed": 1_469_100_000, "search_seed_offset": 146013,
    "decision_seconds": .4, "workers": 8, "torch_threads": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "bootstrap_samples": 50000, "bootstrap_seed": 146998,
    "material_score_share": .55, "maximum_mean_seconds": .44,
    "maximum_p95_seconds": .50, "mean_time_ratio_limits": [.95, 1.05],
    "search": asdict(SearchConfig(simulations=64, c_puct=1.5, max_depth=96,
                                  mode="real", root_noise_fraction=0.,
                                  temperature_turns=0, temperature=0., seed=0)),
    "audit_search_moves": "first_and_last_decision_of_each_arm_in_every_game",
}

_MODELS = None


def initialize_worker(probe=False):
    global _MODELS
    torch.set_num_threads(1)
    base = PublicValueInference()
    model = pilot.ResidualValue(True).eval() if probe else pilot.load_value("trainable")
    _MODELS = {"learned": pilot.ResidualInference(model, base), "public": base}


def schedule(index):
    if not 0 <= index < CONFIG["games"]:
        raise ValueError("Game outside frozen schedule")
    return CONFIG["seed_start"] + index // 2, index % 2


def move_seed(seed, player, turn):
    # Independent of treatment: each seat/turn receives the same RNG across a pair.
    return seed * 10000 + player * 1000 + turn + CONFIG["search_seed_offset"]


class ArenaBot:
    def __init__(self, arm, seed, trace):
        self.arm, self.seed, self.trace = arm, seed, trace
        self.search = BudgetSearch(_MODELS[arm], SearchConfig(**CONFIG["search"]))

    def choose_action(self, env, state, legal_actions=None):
        legal = env.legal_actions(state) if legal_actions is None else legal_actions
        state_hash = digest(serialize_state_snapshot(state))
        seed = move_seed(self.seed, state.current_player, state.turn_index)
        started, cpu = time.perf_counter(), time.process_time()
        result = self.search.run(env, state, legal, self_play=False, seed=seed,
                                 seconds=CONFIG["decision_seconds"])
        elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu
        action = result.action
        self.trace.append({"turn": state.turn_index, "player": state.current_player,
                           "arm": self.arm, "state_hash": state_hash,
                           "action": _serialize_action(action),
                           "action_key": repr(canonical_successor_action_key(state, action)),
                           "legal_count": len(legal), "decision_seconds": elapsed,
                           "cpu_seconds": cpu_elapsed, "search_seed": seed,
                           "search_seconds": result.elapsed_seconds, **search_record(result)})
        return action


def play_one(index, binding, root=None, probe=False):
    root = OUTPUT if root is None else Path(root)
    path = root / "games" / f"game_{index:04d}.json"
    seed, learned_seat = (CONFIG["probe_seed"] + index // 2, index % 2) if probe else schedule(index)
    if path.exists():
        raise FileExistsError(path)
    trace = []
    bots = [ArenaBot("learned" if seat == learned_seat else "public", seed, trace) for seat in range(2)]
    started = time.perf_counter()
    replay = collect_game_replay(*bots, seed=seed, max_turns=CONFIG["max_turns"],
                                 repetition_limit=CONFIG["repetition_limit"], no_progress_limit=CONFIG["no_progress_limit"])
    record = {"index": index, "seed": seed, "learned_seat": learned_seat,
              "manifest": binding, "probe": probe, **replay_summary(replay),
              "trace": trace, "elapsed_seconds": time.perf_counter() - started}
    record["learned_points"] = .5 if record["winner"] is None else float(record["winner"] == learned_seat)
    # Save invalid/cutoff evidence before statistical gating; do not replace games.
    atomic_write(path, record)
    return {"index": index, "valid_natural": record["valid_natural"], "sha256": file_hash(path)}


def paired_statistics(points):
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (CONFIG["games"],) or not np.isfinite(values).all() or not np.isin(values, [0., .5, 1.]).all():
        raise ValueError("Incomplete or invalid outcome schedule")
    paired = values.reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    means = np.concatenate([paired[rng.integers(0, len(paired), (1000, len(paired)))].mean(axis=1)
                            for _ in range(CONFIG["bootstrap_samples"] // 1000)])
    return float(values.mean()), [float(x) for x in np.quantile(means, [.025, .975])]


def decision_label(share, interval, integrity):
    if not integrity:
        return "invalid_integrity_or_time_match"
    if interval[0] > .5 and share >= CONFIG["material_score_share"]:
        return "material_learned_value_benefit"
    if interval[1] < .5:
        return "learned_value_worse_than_public"
    if interval[0] > .5:
        return "positive_value_benefit_below_material_threshold"
    return "value_benefit_not_established"


def timing_summary(records):
    traces = {arm: [t for r in records for t in r["trace"] if t["arm"] == arm] for arm in ARMS}
    timings = {arm: describe([t["decision_seconds"] for t in traces[arm]]) for arm in ARMS}
    budgets = {arm: bool(timings[arm]["mean"] <= CONFIG["maximum_mean_seconds"]
                         and timings[arm]["p95"] <= CONFIG["maximum_p95_seconds"]
                         and all(t["search_seconds"] >= CONFIG["decision_seconds"] for t in traces[arm])) for arm in ARMS}
    ratio = timings["learned"]["mean"] / timings["public"]["mean"]
    lo, hi = CONFIG["mean_time_ratio_limits"]
    return {"decision_seconds": timings, "budget_passed_by_arm": budgets,
            "mean_decision_time_ratio": ratio, "time_match_passed": bool(lo <= ratio <= hi),
            "cpu_seconds": {arm: describe([t["cpu_seconds"] for t in traces[arm]]) for arm in ARMS},
            "simulations": {arm: describe([t["simulations"] for t in traces[arm]]) for arm in ARMS},
            "max_depth_reached": {arm: describe([t["max_depth"] for t in traces[arm]]) for arm in ARMS}}


def summarize(records):
    records = sorted(records, key=lambda r: r["index"])
    if [r["index"] for r in records] != list(range(CONFIG["games"])):
        raise ValueError("Missing or duplicate game")
    for record in records:
        if (record["seed"], record["learned_seat"]) != schedule(record["index"]) or record["probe"]:
            raise ValueError("Seat pair or seed mismatch")
        winner = record["winner"]
        if winner not in (None, 0, 1) or (not record["valid_natural"] and winner is not None):
            raise ValueError("Invalid winner")
        expected = .5 if winner is None else float(winner == record["learned_seat"])
        if record["learned_points"] != expected:
            raise ValueError("Outcome accounting mismatch")
        if any(t["arm"] != ("learned" if t["player"] == record["learned_seat"] else "public") for t in record["trace"]):
            raise ValueError("Treatment/seat mismatch")
    share, interval = paired_statistics([r["learned_points"] for r in records])
    timing = timing_summary(records)
    valid = all(r["valid_natural"] for r in records)
    integrity = valid and all(timing["budget_passed_by_arm"].values()) and timing["time_match_passed"]
    wins = sum(r["winner"] == r["learned_seat"] for r in records if r["winner"] is not None)
    losses = sum(r["winner"] != r["learned_seat"] for r in records if r["winner"] is not None)
    return {"games": len(records), "seed_pairs": len(records)//2, "learned_wins": wins,
            "learned_losses": losses, "draws_or_invalid": len(records)-wins-losses,
            "learned_score_share": share, "paired_ci95": interval,
            "practical_55_percent_ruled_out": bool(integrity and interval[1] < CONFIG["material_score_share"]),
            "all_natural": valid, "integrity_and_time_passed": bool(integrity),
            "termination_counts": dict(Counter(r["termination_reason"] for r in records)),
            "decision": decision_label(share, interval, integrity),
            "learned_seat_score_share": {str(seat): float(np.mean([r["learned_points"] for r in records if r["learned_seat"] == seat])) for seat in (0, 1)},
            "mean_score_margin": float(np.mean([r["final_scores"][r["learned_seat"]] - r["final_scores"][1-r["learned_seat"]] for r in records])),
            "turns": describe([r["turns"] for r in records]), **timing}



def preflight(binding):
    import runpy
    root = OUTPUT / "preflight"
    initialize_worker(probe=True)
    for i in range(2):
        play_one(i, binding, root, probe=True)
    verifier = runpy.run_path("scripts/analysis/verify_value_representation_arena_146.py")
    verifier["initialize"](probe=True)
    checks = [verifier["audit_game"](root / "games" / f"game_{i:04d}.json", binding, probe=True) for i in range(2)]
    timing = timing_summary([read_json(root / "games" / f"game_{i:04d}.json") for i in range(2)])
    passed = all(c["valid_natural"] for c in checks) and all(timing["budget_passed_by_arm"].values()) and timing["time_match_passed"]
    atomic_write(OUTPUT / "preflight.json", {"manifest": binding, "passed": bool(passed), "checks": checks, "timing": timing})
    if not passed:
        raise ValueError("Arena probe failed")


def run(manifest):
    import runpy
    pilot.check_lock(manifest)
    gate = read_json(pilot.OUTPUT / "result.json")
    audit = read_json(pilot.OUTPUT / "prediction_verification.json")
    if not gate["gate"]["passed"] or not audit["passed"] or audit["result_sha256"] != file_hash(pilot.OUTPUT / "result.json"):
        raise ValueError("A verified successful prediction gate is required")
    if pilot.identity() != manifest or (OUTPUT / "manifest.json").exists() or (OUTPUT / "games").exists():
        raise ValueError("Changed identity or arena already launched")
    binding = digest(manifest)
    atomic_write(OUTPUT / "manifest.json", manifest)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker) as pool:
        jobs = [pool.submit(play_one, i, binding) for i in range(CONFIG["games"])]
        for completed, job in enumerate(as_completed(jobs), 1):
            job.result()
            if completed % 16 == 0:
                print(json.dumps({"stage": "conditional_arena", "completed": completed, "total": CONFIG["games"]}), flush=True)
    paths = [OUTPUT / "games" / f"game_{i:04d}.json" for i in range(CONFIG["games"])]
    atomic_write(OUTPUT / "result.json", {"manifest": binding, "summary": summarize([read_json(p) for p in paths]),
        "game_hashes": {p.name: file_hash(p) for p in paths}, "elapsed_seconds": time.perf_counter()-started})
    subprocess.run([sys.executable, "scripts/analysis/verify_value_representation_arena_146.py"], check=True)
