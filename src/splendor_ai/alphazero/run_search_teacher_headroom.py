"""Experiment 148: fixed public-value search compute and conditional competence."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero.search import SearchConfig
from splendor_ai.alphazero.strong_policy import BudgetSearch
from splendor_ai.alphazero.run_public_value_search import (
    PublicValueInference, CHECKPOINT, CHECKPOINT_SHA, SIMPLE, SIMPLE_SHA,
    replay_summary, search_record, describe,
)
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.training.replay import collect_game_replay, serialize_state_snapshot, _serialize_action

OUTPUT = Path("outputs/search_teacher_headroom_148")
PROTOCOL = Path("docs/engineering/search_teacher_headroom_148.md")
VERIFIER = Path("scripts/analysis/verify_search_teacher_headroom_148.py")
PRIOR = Path("outputs/value_training_exposure_147/manifest.json")
PRIOR_SHA = "4320fffe2bea63c4d85675b44e73cf2dd28978b4d47b2e60b55536d5b4f63a3e"
PANELS = ("compute", "competence")
CONFIG = {
    "games_per_panel": 400, "seed_pairs_per_panel": 200,
    "panels": {
        "compute": {"seed_start": 1_480_000_000, "probe_seed": 1_489_000_000,
                    "bootstrap_seed": 148999, "reference": "short"},
        "competence": {"seed_start": 1_481_000_000, "probe_seed": 1_489_001_000,
                       "bootstrap_seed": 148998, "reference": "raw"},
    },
    "seconds_by_arm": {"long": 1.6, "short": .4},
    "search_seed_offset": 148013, "workers": 8, "torch_threads": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "bootstrap_samples": 50000, "material_score_share": .55,
    "maximum_mean_budget_ratio": 1.10, "maximum_p95_budget_ratio": 1.25,
    "search": asdict(SearchConfig(simulations=64, c_puct=1.5, max_depth=96,
                                  mode="real", root_noise_fraction=0.,
                                  temperature_turns=0, temperature=0., seed=0)),
    "audit_search_moves": "first_and_last_decision_of_each_timed_arm_in_every_game",
    "conditional_stage": "complete_compute_panel_and_matching_independent_audit_and_material_superiority",
}
_VALUE = _RAW = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def initialize_worker():
    global _VALUE, _RAW
    torch.set_num_threads(CONFIG["torch_threads"])
    _VALUE = PublicValueInference()
    _RAW = LegalActionAttentionScorerBot(CHECKPOINT)


def arms(panel):
    return ("long", CONFIG["panels"][panel]["reference"])


def schedule(panel, index, probe=False):
    limit = 2 if probe else CONFIG["games_per_panel"]
    if panel not in PANELS or not 0 <= index < limit:
        raise ValueError("Game outside frozen schedule")
    namespace = "probe_seed" if probe else "seed_start"
    return CONFIG["panels"][panel][namespace] + index // 2, index % 2


def move_seed(seed, player, turn):
    # Independent of treatment: same seat and turn get the same RNG across a pair.
    return seed * 10000 + player * 1000 + turn + CONFIG["search_seed_offset"]


class ArenaBot:
    def __init__(self, arm, seed, trace):
        if arm not in ("long", "short", "raw"):
            raise ValueError(arm)
        self.arm, self.seed, self.trace = arm, seed, trace
        self.search = None if arm == "raw" else BudgetSearch(_VALUE, SearchConfig(**CONFIG["search"]))

    def choose_action(self, env, state, legal_actions=None):
        legal = env.legal_actions(state) if legal_actions is None else legal_actions
        state_hash = digest(serialize_state_snapshot(state))
        seed = move_seed(self.seed, state.current_player, state.turn_index)
        started, cpu = time.perf_counter(), time.process_time()
        if self.search is None:
            action = _RAW.choose_action(env, state, legal)
        else:
            result = self.search.run(env, state, legal, self_play=False, seed=seed,
                                     seconds=CONFIG["seconds_by_arm"][self.arm])
            action = result.action
        elapsed, cpu_elapsed = time.perf_counter() - started, time.process_time() - cpu
        row = {"turn": state.turn_index, "player": state.current_player,
               "arm": self.arm, "state_hash": state_hash, "action": _serialize_action(action),
               "action_key": repr(canonical_successor_action_key(state, action)),
               "legal_count": len(legal), "decision_seconds": elapsed, "cpu_seconds": cpu_elapsed}
        if self.search is not None:
            row.update(search_record(result))
            row.update(search_seed=seed, search_seconds=result.elapsed_seconds,
                       allowance_seconds=CONFIG["seconds_by_arm"][self.arm])
        self.trace.append(row)
        return action


def play_one(panel, index, binding, root=None, probe=False):
    root = OUTPUT if root is None else Path(root)
    path = root / panel / "games" / f"game_{index:04d}.json"
    seed, candidate_seat = schedule(panel, index, probe)
    if path.exists():
        raise FileExistsError(path)
    trace = []
    bots = [ArenaBot("long" if seat == candidate_seat else arms(panel)[1], seed, trace) for seat in range(2)]
    started = time.perf_counter()
    replay = collect_game_replay(*bots, seed=seed, max_turns=CONFIG["max_turns"],
                                 repetition_limit=CONFIG["repetition_limit"], no_progress_limit=CONFIG["no_progress_limit"])
    record = {"panel": panel, "index": index, "seed": seed, "candidate_seat": candidate_seat,
              "manifest": binding, "probe": probe, **replay_summary(replay),
              "trace": trace, "elapsed_seconds": time.perf_counter() - started}
    record["candidate_points"] = .5 if record["winner"] is None else float(record["winner"] == candidate_seat)
    atomic_write(path, record)
    return {"panel": panel, "index": index, "valid_natural": record["valid_natural"], "sha256": file_hash(path)}


def identity():
    expected = {CHECKPOINT: CHECKPOINT_SHA, SIMPLE: SIMPLE_SHA, PRIOR: PRIOR_SHA,
                Path("outputs/public_value_ablation_145/manifest.json"):
                    "29fc3c24c916bee6b9b62ab18ab36c11a4390696466edb7d7f75af29dc680450",
                Path("outputs/public_value_search_144/manifest.json"):
                    "80d7a963a0057f08ab31336425c720863bc3ad91d04b25083f99364b9f23f332"}
    for path, sha in expected.items():
        if file_hash(path) != sha:
            raise ValueError(f"Frozen input changed: {path}")
    # Use explicit old inventories; never regenerate an older dynamic identity.
    prior = read_json(PRIOR)
    for field in ("dependencies", "inputs"):
        for name, sha in prior[field].items():
            if file_hash(Path(name)) != sha:
                raise ValueError(f"Prior frozen {field} changed: {name}")
    files = sorted(Path("src/splendor_ai").rglob("*.py")) + [PROTOCOL, VERIFIER,
        Path("run_search_teacher_headroom_148.py"), Path("tests/test_search_teacher_headroom_148.py"),
        Path("scripts/analysis/verify_public_value_search_144.py")]
    return {"format": "search_teacher_headroom_148_v1", "config": CONFIG,
            "inputs": {p.as_posix(): sha for p, sha in expected.items()},
            "dependencies": {p.as_posix(): file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def paired_statistics(panel, points):
    values = np.asarray(points, dtype=np.float64)
    if (values.shape != (CONFIG["games_per_panel"],) or not np.isfinite(values).all()
            or not np.isin(values, [0., .5, 1.]).all()):
        raise ValueError("Incomplete or invalid outcome schedule")
    paired = values.reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(CONFIG["panels"][panel]["bootstrap_seed"])
    means = np.concatenate([paired[rng.integers(0, len(paired), (1000, len(paired)))].mean(axis=1)
                            for _ in range(CONFIG["bootstrap_samples"] // 1000)])
    return float(values.mean()), [float(x) for x in np.quantile(means, [.025, .975])]


def decision_label(share, interval, integrity):
    if not integrity:
        return "invalid_integrity_or_budget"
    if interval[0] > .5 and share >= CONFIG["material_score_share"]:
        return "material_superiority"
    if interval[1] < .5:
        return "candidate_weaker"
    if interval[0] > .5:
        return "positive_below_material_threshold"
    return "improvement_not_established"


def timing_summary(panel, records):
    traces = {arm: [t for r in records for t in r["trace"] if t["arm"] == arm] for arm in arms(panel)}
    timings = {arm: describe([t["decision_seconds"] for t in traces[arm]]) for arm in arms(panel)}
    timed = [arm for arm in arms(panel) if arm != "raw"]
    budgets = {}
    for arm in timed:
        budget = CONFIG["seconds_by_arm"][arm]
        budgets[arm] = bool(traces[arm] and timings[arm]["mean"] <= budget * CONFIG["maximum_mean_budget_ratio"]
                           and timings[arm]["p95"] <= budget * CONFIG["maximum_p95_budget_ratio"]
                           and all(t["allowance_seconds"] == budget
                                   and t["decision_seconds"] >= t["search_seconds"] >= budget for t in traces[arm]))
    return {"decision_seconds": timings, "budget_passed_by_arm": budgets,
            "mean_decision_time_ratio": timings["long"]["mean"] / timings[arms(panel)[1]]["mean"],
            "cpu_seconds": {arm: describe([t["cpu_seconds"] for t in traces[arm]]) for arm in arms(panel)},
            "simulations": {arm: describe([t["simulations"] for t in traces[arm]]) for arm in timed},
            "max_depth_reached": {arm: describe([t["max_depth"] for t in traces[arm]]) for arm in timed}}


def summarize(panel, records):
    records = sorted(records, key=lambda r: r["index"])
    if [r["index"] for r in records] != list(range(CONFIG["games_per_panel"])):
        raise ValueError("Missing or duplicate game")
    for r in records:
        if (r["panel"] != panel or r["probe"] or
                (r["seed"], r["candidate_seat"]) != schedule(panel, r["index"])):
            raise ValueError("Panel, seat pair or seed mismatch")
        if r["winner"] not in (None, 0, 1) or (not r["valid_natural"] and r["winner"] is not None):
            raise ValueError("Invalid winner")
        points = .5 if r["winner"] is None else float(r["winner"] == r["candidate_seat"])
        if r["candidate_points"] != points:
            raise ValueError("Outcome accounting mismatch")
        if any(t["arm"] != ("long" if t["player"] == r["candidate_seat"] else arms(panel)[1]) for t in r["trace"]):
            raise ValueError("Treatment/seat mismatch")
    share, interval = paired_statistics(panel, [r["candidate_points"] for r in records])
    timing = timing_summary(panel, records)
    valid = all(r["valid_natural"] for r in records)
    integrity = valid and all(timing["budget_passed_by_arm"].values())
    wins = sum(r["winner"] == r["candidate_seat"] for r in records if r["winner"] is not None)
    losses = sum(r["winner"] != r["candidate_seat"] for r in records if r["winner"] is not None)
    return {"panel": panel, "games": len(records), "seed_pairs": len(records)//2,
            "candidate_wins": wins, "candidate_losses": losses, "draws_or_invalid": len(records)-wins-losses,
            "candidate_score_share": share, "paired_ci95": interval,
            "practical_55_percent_ruled_out": bool(integrity and interval[1] < CONFIG["material_score_share"]),
            "all_natural": valid, "integrity_and_budget_passed": bool(integrity),
            "termination_counts": dict(Counter(r["termination_reason"] for r in records)),
            "decision": decision_label(share, interval, integrity),
            "candidate_seat_score_share": {str(s): float(np.mean([r["candidate_points"] for r in records if r["candidate_seat"] == s])) for s in (0, 1)},
            "mean_score_margin": float(np.mean([r["final_scores"][r["candidate_seat"]] - r["final_scores"][1-r["candidate_seat"]] for r in records])),
            "turns": describe([r["turns"] for r in records]), **timing}


def audited_panel_passed(panel, binding):
    root = OUTPUT / panel
    result = read_json(root / "result.json")
    audit = read_json(root / "verification.json")
    if (result["manifest"] != binding or audit["manifest"] != binding or audit["panel"] != panel
            or not audit["passed"] or audit["result_sha256"] != file_hash(root / "result.json")
            or audit["games"] != CONFIG["games_per_panel"]):
        raise ValueError("Complete matching independent audit required")
    summary = result["summary"]
    return bool(summary["integrity_and_budget_passed"] and summary["candidate_score_share"] >= CONFIG["material_score_share"]
                and summary["paired_ci95"][0] > .5 and summary["decision"] == "material_superiority")


def preflight():
    binding = digest(identity())
    root = OUTPUT / "preflight"
    if (OUTPUT / "manifest.json").exists() or root.exists():
        raise FileExistsError("Preparation already exists; preserve evidence")
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker) as pool:
        futures = [pool.submit(play_one, panel, index, binding, root, True) for panel in PANELS for index in range(2)]
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps({"probe_complete": row["panel"], "index": row["index"], "natural": row["valid_natural"]}), flush=True)
    audit_game = runpy.run_path(str(VERIFIER))["audit_game"]
    checks = [audit_game(root / panel / "games" / f"game_{i:04d}.json", binding, panel, probe=True)
              for panel in PANELS for i in range(2)]
    panels = {}
    for panel in PANELS:
        records = [read_json(root / panel / "games" / f"game_{i:04d}.json") for i in range(2)]
        timing = timing_summary(panel, records)
        panels[panel] = {"passed": bool(all(r["valid_natural"] for r in records) and all(timing["budget_passed_by_arm"].values())),
                         "timing": timing, "projected_parallel_arena_minutes":
                             sum(r["elapsed_seconds"] for r in records)/2*CONFIG["games_per_panel"]/CONFIG["workers"]/60}
    if digest(identity()) != binding:
        raise ValueError("Source changed during preflight")
    report = {"manifest": binding, "passed": all(p["passed"] for p in panels.values()), "panels": panels, "audits": checks}
    atomic_write(OUTPUT / "preflight.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


def run_panel(panel, manifest):
    binding = digest(manifest)
    if panel == "competence" and not audited_panel_passed("compute", binding):
        raise ValueError("Conditional competence blocked: compute gate did not pass")
    if identity() != manifest:
        raise ValueError("Frozen identity changed before panel")
    root = OUTPUT / panel
    if root.exists():
        raise FileExistsError("Panel already exists; no restart or overwrite")
    started = time.perf_counter()
    atomic_write(root / "stage_started.json", {"manifest": binding, "panel": panel, "started_at": utc_now()})
    print(json.dumps({"stage_started": panel, "games": CONFIG["games_per_panel"]}), flush=True)
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker) as pool:
        futures = [pool.submit(play_one, panel, i, binding) for i in range(CONFIG["games_per_panel"])]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == CONFIG["games_per_panel"]:
                progress = {"panel": panel, "completed": completed, "total": CONFIG["games_per_panel"],
                            "elapsed_seconds": time.perf_counter()-started, "updated_at": utc_now()}
                atomic_write(OUTPUT / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    if identity() != manifest:
        raise ValueError("Frozen identity changed during panel")
    paths = [root / "games" / f"game_{i:04d}.json" for i in range(CONFIG["games_per_panel"])]
    result = {"manifest": binding, "summary": summarize(panel, [read_json(p) for p in paths]),
              "game_hashes": {p.name: file_hash(p) for p in paths}, "elapsed_seconds": time.perf_counter()-started,
              "completed_at": utc_now()}
    atomic_write(root / "result.json", result)
    print(json.dumps({"panel": panel, "status": "arena_complete_pending_audit", "summary": result["summary"]}, allow_nan=False), flush=True)
    with (root / "verification.log").open("x", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", str(VERIFIER), panel], stdout=log, stderr=subprocess.STDOUT, check=True)
    return audited_panel_passed(panel, binding)


def run():
    with process_lock(OUTPUT / "process.lock"):
        manifest = identity()
        binding = digest(manifest)
        prepared = read_json(OUTPUT / "preflight.json")
        if prepared["manifest"] != binding or not prepared["passed"]:
            raise ValueError("Matching successful preflight required")
        if (OUTPUT / "manifest.json").exists():
            raise FileExistsError("148 already launched; no automatic restart")
        started = time.perf_counter()
        atomic_write(OUTPUT / "manifest.json", manifest)
        atomic_write(OUTPUT / "started.json", {"manifest": binding, "started_at": utc_now()})
        try:
            compute_passed = run_panel("compute", manifest)
            competence_passed = run_panel("competence", manifest) if compute_passed else False
            if identity() != manifest:
                raise ValueError("Frozen identity changed before completion")
            completed = PANELS if compute_passed else ("compute",)
            artifacts = {panel: {"result_sha256": file_hash(OUTPUT / panel / "result.json"),
                                 "verification_sha256": file_hash(OUTPUT / panel / "verification.json")}
                         for panel in completed}
            final = {"status": "complete", "manifest": binding, "promotion": False, "training_ran": False,
                     "conditional_competence_ran": compute_passed, "panels": artifacts,
                     "decision": "teacher_headroom_confirmed" if competence_passed else
                         ("competence_gate_failed_stop" if compute_passed else "compute_gate_failed_stop"),
                     "elapsed_seconds": time.perf_counter()-started, "completed_at": utc_now()}
            atomic_write(OUTPUT / "final_status.json", final)
            print(json.dumps(final, allow_nan=False), flush=True)
        except BaseException:
            atomic_write(OUTPUT / "failure.json", {"manifest": binding, "traceback": traceback.format_exc(), "failed_at": utc_now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    args = parser.parse_args()
    (preflight if args.stage == "preflight" else run)()


if __name__ == "__main__":
    main()
