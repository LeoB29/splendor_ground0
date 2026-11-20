"""Experiment 145: public versus neutral leaf value at the same search allowance."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
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

OUTPUT = Path("outputs/public_value_ablation_145")
PROTOCOL = Path("docs/engineering/public_value_ablation_145.md")
PRIOR = Path("outputs/public_value_search_144/manifest.json")
PRIOR_SHA = "80d7a963a0057f08ab31336425c720863bc3ad91d04b25083f99364b9f23f332"
ARMS = ("public", "neutral")
CONFIG = {
    "games": 800, "seed_pairs": 400, "seed_start": 1_450_000_000,
    "probe_seed": 1_459_000_000, "search_seed_offset": 145013,
    "decision_seconds": .4, "workers": 8, "torch_threads": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "bootstrap_samples": 50000, "bootstrap_seed": 145999,
    "material_score_share": .55, "maximum_mean_seconds": .44,
    "maximum_p95_seconds": .50, "mean_time_ratio_limits": [.95, 1.05],
    "search": asdict(SearchConfig(simulations=64, c_puct=1.5, max_depth=96,
                                  mode="real", root_noise_fraction=0.,
                                  temperature_turns=0, temperature=0., seed=0)),
    "audit_search_moves": "first_and_last_decision_of_each_arm_in_every_game",
}
_MODELS = None


class LeafModeInference:
    """Run identical frozen inference work, replacing only the neutral leaf scalar."""
    def __init__(self, base, mode):
        if mode not in ARMS:
            raise ValueError("Unknown value treatment")
        self.base, self.mode = base, mode

    def predict_one(self, state, actions):
        prediction = self.base.predict_one(state, actions)
        return NetworkPrediction(prediction.logits, prediction.value if self.mode == "public" else 0.)


def initialize_worker():
    global _MODELS
    torch.set_num_threads(CONFIG["torch_threads"])
    base = PublicValueInference()
    _MODELS = {mode: LeafModeInference(base, mode) for mode in ARMS}


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
    seed, public_seat = (CONFIG["probe_seed"] + index // 2, index % 2) if probe else schedule(index)
    if path.exists():
        raise FileExistsError(path)
    trace = []
    bots = [ArenaBot("public" if seat == public_seat else "neutral", seed, trace) for seat in range(2)]
    started = time.perf_counter()
    replay = collect_game_replay(*bots, seed=seed, max_turns=CONFIG["max_turns"],
                                 repetition_limit=CONFIG["repetition_limit"], no_progress_limit=CONFIG["no_progress_limit"])
    record = {"index": index, "seed": seed, "public_seat": public_seat,
              "manifest": binding, "probe": probe, **replay_summary(replay),
              "trace": trace, "elapsed_seconds": time.perf_counter() - started}
    record["public_points"] = .5 if record["winner"] is None else float(record["winner"] == public_seat)
    # Save invalid/cutoff evidence before statistical gating; do not replace games.
    atomic_write(path, record)
    return {"index": index, "valid_natural": record["valid_natural"], "sha256": file_hash(path)}


def identity():
    expected = {CHECKPOINT: CHECKPOINT_SHA, SIMPLE: SIMPLE_SHA, PRIOR: PRIOR_SHA}
    for path, sha in expected.items():
        if file_hash(path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    for name, sha in read_json(PRIOR)["dependencies"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Prior frozen dependency changed: {name}")
    files = sorted(Path("src/splendor_ai").rglob("*.py")) + [
        Path("run_public_value_ablation_145.py"), Path("tests/test_public_value_ablation_145.py"),
        Path("scripts/analysis/verify_public_value_ablation_145.py"), PROTOCOL,
    ]
    return {"format": "public_value_ablation_145_v1", "config": CONFIG,
            "inputs": {p.as_posix(): sha for p, sha in expected.items()},
            "dependencies": {p.as_posix(): file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


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
        return "material_public_value_benefit"
    if interval[1] < .5:
        return "public_value_worse_than_neutral"
    if interval[0] > .5:
        return "positive_value_benefit_below_material_threshold"
    return "value_benefit_not_established"


def timing_summary(records):
    traces = {arm: [t for r in records for t in r["trace"] if t["arm"] == arm] for arm in ARMS}
    timings = {arm: describe([t["decision_seconds"] for t in traces[arm]]) for arm in ARMS}
    budgets = {arm: bool(timings[arm]["mean"] <= CONFIG["maximum_mean_seconds"]
                         and timings[arm]["p95"] <= CONFIG["maximum_p95_seconds"]
                         and all(t["search_seconds"] >= CONFIG["decision_seconds"] for t in traces[arm])) for arm in ARMS}
    ratio = timings["public"]["mean"] / timings["neutral"]["mean"]
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
        if (record["seed"], record["public_seat"]) != schedule(record["index"]) or record["probe"]:
            raise ValueError("Seat pair or seed mismatch")
        winner = record["winner"]
        if winner not in (None, 0, 1) or (not record["valid_natural"] and winner is not None):
            raise ValueError("Invalid winner")
        expected = .5 if winner is None else float(winner == record["public_seat"])
        if record["public_points"] != expected:
            raise ValueError("Outcome accounting mismatch")
        if any(t["arm"] != ("public" if t["player"] == record["public_seat"] else "neutral") for t in record["trace"]):
            raise ValueError("Treatment/seat mismatch")
    share, interval = paired_statistics([r["public_points"] for r in records])
    timing = timing_summary(records)
    valid = all(r["valid_natural"] for r in records)
    integrity = valid and all(timing["budget_passed_by_arm"].values()) and timing["time_match_passed"]
    wins = sum(r["winner"] == r["public_seat"] for r in records if r["winner"] is not None)
    losses = sum(r["winner"] != r["public_seat"] for r in records if r["winner"] is not None)
    return {"games": len(records), "seed_pairs": len(records)//2, "public_wins": wins,
            "public_losses": losses, "draws_or_invalid": len(records)-wins-losses,
            "public_score_share": share, "paired_ci95": interval,
            "practical_55_percent_ruled_out": bool(integrity and interval[1] < CONFIG["material_score_share"]),
            "all_natural": valid, "integrity_and_time_passed": bool(integrity),
            "termination_counts": dict(Counter(r["termination_reason"] for r in records)),
            "decision": decision_label(share, interval, integrity),
            "public_seat_score_share": {str(seat): float(np.mean([r["public_points"] for r in records if r["public_seat"] == seat])) for seat in (0, 1)},
            "mean_score_margin": float(np.mean([r["final_scores"][r["public_seat"]] - r["final_scores"][1-r["public_seat"]] for r in records])),
            "turns": describe([r["turns"] for r in records]), **timing}


def preflight():
    import runpy
    binding = digest(identity())
    root = OUTPUT / "preflight"
    if (OUTPUT / "manifest.json").exists() or (root / "games").exists():
        raise FileExistsError("Preparation already exists; do not overwrite evidence")
    initialize_worker()
    for index in range(2):
        play_one(index, binding, root, probe=True)
    audit = runpy.run_path("scripts/analysis/verify_public_value_ablation_145.py")["audit_game"]
    checks = [audit(root / "games" / f"game_{i:04d}.json", binding, probe=True) for i in range(2)]
    records = [read_json(root / "games" / f"game_{i:04d}.json") for i in range(2)]
    timing = timing_summary(records)
    result = {"manifest": binding, "passed": bool(all(r["valid_natural"] for r in records)
              and all(timing["budget_passed_by_arm"].values()) and timing["time_match_passed"]),
              "audits": checks, "timing": timing,
              "projected_parallel_arena_minutes": sum(r["elapsed_seconds"] for r in records)/2*CONFIG["games"]/CONFIG["workers"]/60}
    atomic_write(OUTPUT / "preflight.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


def run():
    with process_lock(OUTPUT / "process.lock"):
        current = identity()
        binding = digest(current)
        prepared = read_json(OUTPUT / "preflight.json")
        if prepared["manifest"] != binding or not prepared["passed"]:
            raise ValueError("Preflight failed or source changed")
        if (OUTPUT / "manifest.json").exists():
            raise FileExistsError("145 already launched; no automatic restart")
        atomic_write(OUTPUT / "manifest.json", current)
        started = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker) as pool:
                futures = [pool.submit(play_one, i, binding) for i in range(CONFIG["games"])]
                for completed, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if completed % 8 == 0 or completed == CONFIG["games"]:
                        print(json.dumps({"completed": completed, "total": CONFIG["games"],
                                          "elapsed_seconds": time.perf_counter()-started}), flush=True)
            if identity() != current:
                raise ValueError("Frozen dependencies changed during arena")
            paths = [OUTPUT / "games" / f"game_{i:04d}.json" for i in range(CONFIG["games"])]
            result = {"manifest": binding, "summary": summarize([read_json(p) for p in paths]),
                      "game_hashes": {p.name: file_hash(p) for p in paths},
                      "elapsed_seconds": time.perf_counter()-started}
            atomic_write(OUTPUT / "result.json", result)
            print(json.dumps({"status": "arena_complete_pending_audit", "summary": result["summary"]}, allow_nan=False), flush=True)
        except BaseException:
            atomic_write(OUTPUT / "failure.json", {"manifest": binding, "traceback": traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    args = parser.parse_args()
    (preflight if args.stage == "preflight" else run)()


if __name__ == "__main__":
    main()
