"""Experiment 144: frozen public value plus 035 PUCT versus exact raw 035."""
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
from splendor_ai.alphazero.strong_policy import BudgetSearch, StrongPolicyInference
from splendor_ai.alphazero.run_strong_policy_pilot import CHECKPOINT, CHECKPOINT_SHA
from splendor_ai.alphazero.run_value_generalization_scale import simple_features
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS
from splendor_ai.training.replay import collect_game_replay, serialize_state_snapshot, _serialize_action

OUTPUT = Path("outputs/public_value_search_144")
PROTOCOL = Path("docs/engineering/public_value_search_144.md")
SIMPLE = Path("outputs/value_generalization_scale_143/simple.json")
SIMPLE_SHA = "a5c87ab3fb0f3f22a0e0e25a5adbfcad1bb3422c7f40cdc060d33a0d3a9b720c"
PRIOR = Path("outputs/value_generalization_scale_143/manifest.json")
PRIOR_SHA = "050fb3a771df9817665ebd744fe3a476ddf2e6dec8f4d7ae8852569275b1c4f6"
CONFIG = {
    "games": 400, "seed_pairs": 200, "seed_start": 1_440_000_000,
    "probe_seed": 1_449_000_000, "search_seed_offset": 144013,
    "decision_seconds": .4, "workers": 8, "torch_threads": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "bootstrap_samples": 50000, "bootstrap_seed": 144999,
    "material_score_share": .55, "maximum_mean_seconds": .44,
    "maximum_p95_seconds": .50,
    "search": asdict(SearchConfig(simulations=64, c_puct=1.5, max_depth=96,
                                  mode="real", root_noise_fraction=0.,
                                  temperature_turns=0, temperature=0., seed=0)),
    "audit_search_moves": "first_and_last_search_decision_in_every_game",
}
_VALUE = _RAW = None


class PublicValueInference(StrongPolicyInference):
    def __init__(self):
        if file_hash(CHECKPOINT) != CHECKPOINT_SHA or file_hash(SIMPLE) != SIMPLE_SHA:
            raise ValueError("Frozen model identity changed")
        super().__init__(CHECKPOINT)
        data = read_json(SIMPLE)
        if data["model"] != "public_score_resource_ridge_15_v1":
            raise ValueError("Wrong public evaluator")
        self.coefficients = torch.tensor(data["coefficients"], dtype=torch.float64)
        if self.coefficients.shape != (15,) or not torch.isfinite(self.coefficients).all():
            raise ValueError("Invalid public evaluator coefficients")

    def public_value(self, state):
        # Exact 143 arithmetic: round public features to float32, dot in float64.
        return float((simple_features(state).double() @ self.coefficients).clamp(-1, 1))

    def predict_one(self, state, actions):
        prediction = super().predict_one(state, actions)
        return NetworkPrediction(prediction.logits, self.public_value(state))


def initialize_worker():
    global _VALUE, _RAW
    torch.set_num_threads(CONFIG["torch_threads"])
    _VALUE = PublicValueInference()
    _RAW = LegalActionAttentionScorerBot(CHECKPOINT)


def schedule(index):
    if not 0 <= index < CONFIG["games"]:
        raise ValueError("Game outside frozen schedule")
    return CONFIG["seed_start"] + index // 2, index % 2


def move_seed(seed, candidate_seat, turn):
    return seed * 10000 + candidate_seat * 1000 + turn + CONFIG["search_seed_offset"]


def search_record(result):
    return {
        "simulations": result.simulations, "expansions": result.expansions,
        "inference_calls": result.inference_calls, "max_depth": result.max_depth_reached,
        "raw_network_value": result.raw_network_value, "root_value": result.search_root_value,
        "selected_row": result.selected_action_row, "mode": result.search_mode,
        "edges": [{"key": repr(e.action_key), "visits": e.visits, "probability": e.probability,
                   "network_prior": e.network_prior, "prior": e.search_prior, "q": e.q}
                  for e in result.root_edges],
    }


class ArenaBot:
    def __init__(self, candidate, seed, candidate_seat, trace):
        self.candidate, self.seed, self.candidate_seat, self.trace = candidate, seed, candidate_seat, trace
        self.search = BudgetSearch(_VALUE, SearchConfig(**CONFIG["search"])) if candidate else None

    def choose_action(self, env, state, legal_actions=None):
        legal = env.legal_actions(state) if legal_actions is None else legal_actions
        state_hash = digest(serialize_state_snapshot(state))
        seed = move_seed(self.seed, self.candidate_seat, state.turn_index)
        started, cpu = time.perf_counter(), time.process_time()
        if self.candidate:
            result = self.search.run(env, state, legal, self_play=False, seed=seed,
                                     seconds=CONFIG["decision_seconds"])
            action = result.action
        else:
            action = _RAW.choose_action(env, state, legal)
        elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu
        record = {"turn": state.turn_index, "player": state.current_player,
                  "candidate": self.candidate, "state_hash": state_hash,
                  "action": _serialize_action(action), "action_key": repr(canonical_successor_action_key(state, action)),
                  "legal_count": len(legal), "decision_seconds": elapsed, "cpu_seconds": cpu_elapsed}
        if self.candidate:
            record.update(search_record(result))
            record["search_seed"] = seed
            record["search_seconds"] = result.elapsed_seconds
        self.trace.append(record)
        return action


def replay_summary(replay):
    valid = replay.termination_reason in VALID_NATURAL_TERMINATIONS
    return {"winner": replay.winner if valid else None, "valid_natural": valid,
            "diagnostic_adjudicated_winner": replay.winner,
            "termination_reason": replay.termination_reason,
            "final_scores": list(replay.final_scores), "turns": replay.turns,
            "stalled": replay.stalled, "timed_out": replay.timed_out,
            "final_state": replay.final_state_snapshot}


def play_one(index, binding, root=None, probe=False):
    root = OUTPUT if root is None else Path(root)
    path = root / "games" / f"game_{index:04d}.json"
    seed, candidate_seat = (CONFIG["probe_seed"] + index // 2, index % 2) if probe else schedule(index)
    if path.exists():
        raise FileExistsError(path)
    trace = []
    bots = [ArenaBot(seat == candidate_seat, seed, candidate_seat, trace) for seat in range(2)]
    started = time.perf_counter()
    replay = collect_game_replay(*bots, seed=seed, max_turns=CONFIG["max_turns"],
                                 repetition_limit=CONFIG["repetition_limit"], no_progress_limit=CONFIG["no_progress_limit"])
    record = {"index": index, "seed": seed, "candidate_seat": candidate_seat,
              "manifest": binding, "probe": probe, **replay_summary(replay),
              "trace": trace, "elapsed_seconds": time.perf_counter() - started}
    record["candidate_points"] = (.5 if record["winner"] is None else float(record["winner"] == candidate_seat))
    # Persist even invalid games before applying scientific gates; never replace them.
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
        Path("run_public_value_search_144.py"), Path("tests/test_public_value_search_144.py"),
        Path("scripts/analysis/verify_public_value_search_144.py"), PROTOCOL,
    ]
    return {"format": "public_value_search_144_v1", "config": CONFIG,
            "inputs": {p.as_posix(): sha for p, sha in expected.items()},
            "dependencies": {p.as_posix(): file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def paired_statistics(points):
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (CONFIG["games"],) or not np.isfinite(values).all():
        raise ValueError("Incomplete or invalid outcome schedule")
    paired = values.reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    means = np.concatenate([paired[rng.integers(0, len(paired), (1000, len(paired)))].mean(axis=1)
                            for _ in range(CONFIG["bootstrap_samples"] // 1000)])
    return float(values.mean()), [float(x) for x in np.quantile(means, [.025, .975])]


def decision_label(share, interval, integrity):
    if not integrity:
        return "invalid_integrity_or_budget"
    if interval[0] > .5 and share >= CONFIG["material_score_share"]:
        return "material_search_system_improvement"
    if interval[1] < .5:
        return "search_system_worse_than_raw_035"
    if interval[0] > .5:
        return "positive_improvement_below_material_threshold"
    return "improvement_not_established"


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "min": float(a.min()), "max": float(a.max())}


def summarize(records):
    records = sorted(records, key=lambda r: r["index"])
    if [r["index"] for r in records] != list(range(CONFIG["games"])):
        raise ValueError("Missing or duplicate game")
    for record in records:
        if (record["seed"], record["candidate_seat"]) != schedule(record["index"]) or record["probe"]:
            raise ValueError("Seat pair or seed mismatch")
        winner = record["winner"]
        expected = .5 if winner is None else float(winner == record["candidate_seat"])
        if record["candidate_points"] != expected:
            raise ValueError("Outcome accounting mismatch")
    share, interval = paired_statistics([r["candidate_points"] for r in records])
    candidate = [t for r in records for t in r["trace"] if t["candidate"]]
    raw = [t for r in records for t in r["trace"] if not t["candidate"]]
    timing = {"candidate": describe([t["decision_seconds"] for t in candidate]),
              "raw_035": describe([t["decision_seconds"] for t in raw])}
    budget = bool(timing["candidate"]["mean"] <= CONFIG["maximum_mean_seconds"] and
                  timing["candidate"]["p95"] <= CONFIG["maximum_p95_seconds"] and
                  all(t["search_seconds"] >= CONFIG["decision_seconds"] for t in candidate))
    valid = all(r["valid_natural"] for r in records)
    wins = sum(r["winner"] == r["candidate_seat"] for r in records if r["winner"] is not None)
    losses = sum(r["winner"] != r["candidate_seat"] for r in records if r["winner"] is not None)
    return {
        "games": len(records), "seed_pairs": len(records)//2, "candidate_wins": wins,
        "candidate_losses": losses, "draws_or_invalid": len(records)-wins-losses,
        "candidate_score_share": share, "paired_ci95": interval,
        "practical_55_percent_ruled_out": bool(interval[1] < CONFIG["material_score_share"]),
        "all_natural": valid, "budget_passed": budget,
        "termination_counts": dict(Counter(r["termination_reason"] for r in records)),
        "decision": decision_label(share, interval, valid and budget),
        "candidate_seat_score_share": {str(seat): float(np.mean([r["candidate_points"] for r in records if r["candidate_seat"] == seat])) for seat in (0, 1)},
        "mean_score_margin": float(np.mean([r["final_scores"][r["candidate_seat"]] - r["final_scores"][1-r["candidate_seat"]] for r in records])),
        "decision_seconds": timing,
        "mean_decision_time_ratio": timing["candidate"]["mean"] / timing["raw_035"]["mean"],
        "candidate_cpu_seconds": describe([t["cpu_seconds"] for t in candidate]),
        "simulations": describe([t["simulations"] for t in candidate]),
        "max_depth_reached": describe([t["max_depth"] for t in candidate]),
        "turns": describe([r["turns"] for r in records]),
    }


def preflight():
    import runpy
    current = identity()
    binding = digest(current)
    root = OUTPUT / "preflight"
    if (OUTPUT / "manifest.json").exists() or (root / "games").exists():
        raise FileExistsError("Preparation already exists; do not overwrite evidence")
    initialize_worker()
    for index in range(2):
        play_one(index, binding, root, probe=True)
    audit = runpy.run_path("scripts/analysis/verify_public_value_search_144.py")["audit_game"]
    checks = [audit(root / "games" / f"game_{i:04d}.json", binding, probe=True) for i in range(2)]
    records = [read_json(root / "games" / f"game_{i:04d}.json") for i in range(2)]
    seconds = [t["decision_seconds"] for r in records for t in r["trace"] if t["candidate"]]
    result = {"manifest": binding, "passed": bool(all(r["valid_natural"] for r in records)
              and np.mean(seconds) <= CONFIG["maximum_mean_seconds"]
              and np.quantile(seconds, .95) <= CONFIG["maximum_p95_seconds"]),
              "audits": checks, "decision_seconds": describe(seconds),
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
            raise FileExistsError("144 already launched; no automatic restart")
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
