"""158: a frozen, paired test of learned continuations in the unchanged155 root operator."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import learned_continuation_158 as h
from splendor_ai.alphazero import run_fast_selector_transfer as parent

old, OUTPUT, MODELS = h.old, h.OUTPUT, h.MODELS
PROTOCOL = Path("docs/engineering/learned_continuation_158.md")
VERIFIER = Path("scripts/analysis/verify_learned_continuation_158.py")
LAUNCHER = Path("scripts/analysis/launch_learned_continuation_158.py")
INVENTORY = [Path("src/splendor_ai/alphazero/learned_continuation_158.py"),
             Path("src/splendor_ai/alphazero/run_learned_continuation.py"),
             PROTOCOL, VERIFIER, LAUNCHER, Path("run_learned_continuation_158.py"),
             Path("tests/test_learned_continuation_158.py")]
CONFIG = {
    "games": 402, "probe_games": 12, "main_pairs": 201, "probe_pairs": 6,
    "workers": 8, "torch_threads": 1, "worlds": 32, "max_candidates": 4,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "models": list(MODELS), "assignment_tag": "158-model-assignment-v1",
    "environment_seed": {"main": 1580000000, "probe": 1589000000},
    "test_environment_seed": 1589800000,
    "stream_tag": "158-root-stream-v1", "continuation_tag": "158-continuation-v1",
    "root_operator": "unchanged155", "candidate_controller": "exact157_two_network_restricted",
    "bootstrap_seed": 1583000000, "bootstrap_replicates": 50000, "bootstrap_chunk": 1000,
    "minimum_share": .55, "minimum_ci_lower_exclusive": .50,
    "main_continuation_ceiling": 402*400*32*4,
    "training": False, "arena": True, "promotion": False, "automatic_successor": False,
}
PARENT_BINDING = "3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122"
PARENT_HASHES = {
    "result.json": "aac9f292a92b31ae904b6fe7244f1a8ff57b05fd406d9d035d11bc94da18707a",
    "main_audit.json": "a5057a0aaf0e77c93f340b3046490ad2777b366e9aa8cf9720ad47348a90496c",
    "final_status.json": "3da68af5ee0a3752b78129d03848d85ba80758497758de9a267f4cd63b0dbae4",
    "fits_audit.json": "77fc545ba9d460731ec881ecb14baf8f11daf5c7158778728fc7eb6da5df31a8",
    "selected_lock.json": "4942eebb38de81b40d38315ec4abaf6b306f9340f4fd7051113250d4831044d8",
}


def parent_inputs():
    assert parent.validate_freeze() == PARENT_BINDING
    for name, sha in PARENT_HASHES.items():
        assert old.file_hash(parent.OUTPUT/name) == sha
    final = old.read_json(parent.OUTPUT/"final_status.json")
    audit = old.read_json(parent.OUTPUT/"main_audit.json")
    assert final["complete"] and final["integrity_pass"] and audit["passed"]
    assert final["result_sha256"] == PARENT_HASHES["result.json"]
    assert final["audit_sha256"] == PARENT_HASHES["main_audit.json"]
    result = old.read_json(parent.OUTPUT/"result.json")
    assert audit["recomputed_result_digest"] == old.digest({k: result[k] for k in ("statistics", "timings")})
    return {"binding": PARENT_BINDING, "files": PARENT_HASHES,
            "tests_sha256": parent.verify_test_evidence(), "selected_models": h.selected_models()}


def identity():
    return {"format": "learned_continuation_158_v1", "config": CONFIG,
            "parent": parent_inputs(),
            "dependencies": {p.as_posix(): old.file_hash(p) for p in INVENTORY},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def check_identity():
    saved = old.read_json(OUTPUT/"manifest.json")
    assert saved == identity(), "158 frozen identity changed"
    return old.digest(saved)


def verify_test_evidence():
    folder = OUTPUT/"preparation"
    record = old.read_json(folder/"tests.json")
    assert record["full_repository"] and record["exit_code"] == 0
    assert record["identity"] == identity()
    assert record["log_sha256"] == old.file_hash(folder/"tests.txt")
    assert record["wrapper_sha256"] == old.file_hash(LAUNCHER)
    return old.file_hash(folder/"tests.json")


def schedule(stage):
    assert stage in ("probe", "main")
    n = CONFIG[f"{stage}_pairs"]
    assert n % len(MODELS) == 0
    ordered = sorted(range(n), key=lambda pair: old.digest([CONFIG["assignment_tag"], stage, pair]))
    assigned = {pair: MODELS[rank % len(MODELS)] for rank, pair in enumerate(ordered)}
    return [{"index": 2*pair+seat, "pair": pair, "candidate_seat": seat,
             "model_id": assigned[pair], "environment_seed": CONFIG["environment_seed"][stage]+pair,
             "search_stream": old.digest([CONFIG["stream_tag"], stage, pair])}
            for pair in range(n) for seat in (0, 1)]


def freeze():
    assert not (OUTPUT/"manifest.json").exists(), "Existing freeze needs review; no automatic resume"
    tests = verify_test_evidence()
    manifest = identity()
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"schedule.json", {s: schedule(s) for s in ("probe", "main")})
    old.save_new(OUTPUT/"freeze.json", {"manifest": old.digest(manifest), "time": old.now(),
                 "tests_sha256": tests, "schedule_sha256": old.file_hash(OUTPUT/"schedule.json"),
                 "selected_lock_sha256": PARENT_HASHES["selected_lock.json"]})
    return validate_freeze()


def validate_freeze():
    binding = check_identity()
    saved = old.read_json(OUTPUT/"freeze.json")
    assert saved["manifest"] == binding and saved["tests_sha256"] == verify_test_evidence()
    assert saved["schedule_sha256"] == old.file_hash(OUTPUT/"schedule.json")
    assert saved["selected_lock_sha256"] == PARENT_HASHES["selected_lock.json"]
    assert old.read_json(OUTPUT/"schedule.json") == {s: schedule(s) for s in ("probe", "main")}
    return binding


def progress(stage, **values):
    row = {"time": old.now(), "stage": stage, **values}
    h.progress_write(OUTPUT/"progress.json", row)
    print(row, flush=True)


def launch_stage(stage, binding, upstream):
    old.save_new(OUTPUT/f"{stage}_launch.json", {"manifest": binding, "time": old.now(),
                 "schedule_sha256": old.file_hash(OUTPUT/"schedule.json"), "upstream": upstream})


def collect(stage):
    binding = validate_freeze()
    launch = OUTPUT/f"{stage}_launch.json"
    assert old.read_json(launch)["manifest"] == binding
    assert not (OUTPUT/f"{stage}_games_lock.json").exists()
    specs = schedule(stage)
    assert not list((OUTPUT/stage/"games").glob("*.json")), "No automatic replay/replacement of games"
    start, files = time.perf_counter(), {}
    progress(f"{stage}_collection", games_complete=0, games_total=len(specs))
    with old.process_lock(OUTPUT/f"{stage}_collection.lock"):
        with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=h.initialize_worker) as pool:
            pending = [pool.submit(h.play_game, stage, spec, binding) for spec in specs]
            try:
                for future in as_completed(pending):
                    item = future.result()
                    assert str(item["index"]) not in files
                    files[str(item["index"])] = item["sha256"]
                    progress(f"{stage}_collection", games_complete=len(files), games_total=len(specs),
                             elapsed_seconds=time.perf_counter()-start)
            except BaseException:
                old.save_new(OUTPUT/f"failure_{stage}_collection.json",
                             {"time": old.now(), "traceback": traceback.format_exc(), "completed_files": files})
                for future in pending:
                    future.cancel()
                raise
    assert set(files) == {str(s["index"]) for s in specs}
    saved = {"manifest": binding, "launch_sha256": old.file_hash(launch), "time": old.now(),
             "files": files, "collection_wall_seconds": time.perf_counter()-start}
    old.save_new(OUTPUT/f"{stage}_games_lock.json", saved)
    return saved


def observations(stage):
    games, roots = [], []
    lock = old.read_json(OUTPUT/f"{stage}_games_lock.json")
    for spec in schedule(stage):
        path = h.game_path(stage, spec["index"])
        assert old.file_hash(path) == lock["files"][str(spec["index"])]
        game = old.read_json(path)
        assert game["spec"] == spec and "failure" not in game
        assert game["reason"] in old.VALID_NATURAL_TERMINATIONS
        scores = [p["score"] for p in game["final_state"]["players"]]
        seat = spec["candidate_seat"]
        games.append({"index": spec["index"], "pair": spec["pair"], "model_id": spec["model_id"],
                      "seat": seat, "score": (old.outcome_value(game["winner"], seat)+1)/2,
                      "margin": scores[seat]-scores[1-seat], "turns": game["turns"],
                      "reason": game["reason"], "timing": game["timing"]})
        for move in game["trace"]:
            rp = h.root_path(stage, spec["index"], move["turn"])
            assert old.file_hash(rp) == move["root_sha256"]
            root = old.read_json(rp)
            outcomes = [o for w in root["worlds"] for o in w["outcomes"]]
            roots.append({"index": spec["index"], "model_id": spec["model_id"],
                          "planner": "candidate" if move["actor"] == seat else "reference",
                          "seat": move["actor"], "turn": move["turn"], "choice": root["choice"],
                          "support": len(root["candidates"]["actions"]),
                          "canonical_support": root["candidates"]["canonical_support"],
                          "legal_support": root["candidates"]["legal_support"],
                          "privacy": root["privacy"], "timing": root["timing"],
                          "continuations": len(outcomes), "moves": sum(o["turns"] for o in outcomes),
                          "reasons": dict(Counter(o["reason"] for o in outcomes))})
    return {"games": games, "roots": roots}


def statistics(rows):
    assert len(rows) == CONFIG["games"]
    expected = schedule("main")
    for row, spec in zip(rows, expected):
        assert (row["index"], row["pair"], row["seat"], row["model_id"]) == (
            spec["index"], spec["pair"], spec["candidate_seat"], spec["model_id"])
        assert row["score"] in (0., .5, 1.) and np.isfinite(row["margin"])
    pairs = {model: [] for model in MODELS}
    for a, b in zip(rows[::2], rows[1::2]):
        assert a["pair"] == b["pair"] and a["model_id"] == b["model_id"]
        pairs[a["model_id"]].append(int(2*a["score"]+2*b["score"]))
    assert all(len(v) == CONFIG["main_pairs"]//3 for v in pairs.values())
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = []
    arrays = {m: np.asarray(pairs[m], dtype=np.int64) for m in MODELS}
    for start in range(0, CONFIG["bootstrap_replicates"], CONFIG["bootstrap_chunk"]):
        count = min(CONFIG["bootstrap_chunk"], CONFIG["bootstrap_replicates"]-start)
        sample = np.zeros(count, dtype=np.int64)
        for model in MODELS:
            values = arrays[model]
            sample += values[rng.integers(0, len(values), size=(count, len(values)))].sum(axis=1)
        boot.extend((sample/(4*CONFIG["main_pairs"])).tolist())
    ci = np.quantile(boot, [.025, .975]).tolist()

    def summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part)/len(part),
                "wdl": {key: sum(r["score"] == val for r in part)
                        for key, val in (("wins", 1.), ("draws", .5), ("losses", 0.))},
                "mean_margin": sum(r["margin"] for r in part)/len(part),
                "mean_turns": sum(r["turns"] for r in part)/len(part)}
    seats = {str(s): summary(rows[s::2]) for s in (0, 1)}
    # Exact half-point accounting avoids a floating-point gate at a rational boundary.
    half_points = sum(int(2*r["score"]) for r in rows)
    gates = {"material_share": half_points*10 >= 11*len(rows),
             "positive_lower_bound": ci[0] > .5,
             "positive_margin": sum(r["margin"] for r in rows) > 0,
             "both_seats": all(sum(int(2*r["score"]) for r in rows[s::2]) >= len(rows[s::2])
                               for s in (0, 1))}
    return {**summary(rows), "pairs": CONFIG["main_pairs"], "ci95": ci,
            "statistical_gates": gates, "statistical_pass": all(gates.values()),
            "seats": seats, "models": {m: summary([r for r in rows if r["model_id"] == m]) for m in MODELS},
            "estimand": "uniform mixture of three fixed157 treatment planners"}


def operational(data):
    quantiles = h.baseline.quantiles
    roots, games = data["roots"], data["games"]

    def counts(part):
        endings = Counter()
        for row in part:
            endings.update(row["reasons"])
        return {"roots": len(part), "continuations": sum(r["continuations"] for r in part),
                "continuation_moves": sum(r["moves"] for r in part), "inner_reasons": dict(endings)}

    def planner(part):
        return {**counts(part), "changes": sum(r["choice"] != 0 for r in part),
                "change_fraction": sum(r["choice"] != 0 for r in part)/len(part),
                "support_counts": dict(Counter(str(r["support"]) for r in part)),
                "own_private_roots": sum(r["privacy"]["own"] > 0 for r in part),
                "opponent_private_roots": sum(r["privacy"]["opponent"] > 0 for r in part),
                "decision_time": {k: quantiles([r["timing"][k] for r in part]) for k in ("wall", "cpu")}}
    return {**counts(roots), "outer_reasons": dict(Counter(g["reason"] for g in games)),
            "game_time": {k: quantiles([g["timing"][k] for g in games]) for k in ("wall", "cpu")},
            "turns": quantiles([g["turns"] for g in games]),
            "planners": {name: planner([r for r in roots if r["planner"] == name])
                         for name in ("candidate", "reference")}}


def audit_stage(stage):
    validate_freeze()
    assert not (OUTPUT/f"{stage}_audit.json").exists()
    progress(f"{stage}_audit_starting")
    subprocess.run([sys.executable, "-u", str(VERIFIER), stage], check=True)
    saved = old.read_json(OUTPUT/f"{stage}_audit.json")
    assert saved["passed"] and saved["manifest"] == check_identity()
    assert saved["games_lock_sha256"] == old.file_hash(OUTPUT/f"{stage}_games_lock.json")
    if stage == "main":
        assert saved["result_sha256"] == old.file_hash(OUTPUT/"result.json")
        assert saved["observations_sha256"] == old.file_hash(OUTPUT/"observations.json")
    return saved


def preflight(binding):
    launch_stage("probe", binding, {"freeze_sha256": old.file_hash(OUTPUT/"freeze.json")})
    lock = collect("probe")
    audit = audit_stage("probe")
    data = observations("probe")
    operation = operational(data)
    ratio = CONFIG["games"]/CONFIG["probe_games"]
    timing = {"outcome_blind": True, "probe_games": CONFIG["probe_games"],
              "collection_seconds": lock["collection_wall_seconds"], "audit_seconds": audit["wall_seconds"],
              "main_projection_seconds": ratio*(lock["collection_wall_seconds"]+audit["wall_seconds"]),
              "operational": operation,
              "caveat": "Twelve off-panel games, eight workers; scheduling, game lengths and contention affect projection. Fixed budgets unchanged."}
    old.save_new(OUTPUT/"timing_probe.json", timing)
    old.save_new(OUTPUT/"preflight.json", {"passed": True, "manifest": binding, "time": old.now(),
                 "audit_sha256": old.file_hash(OUTPUT/"probe_audit.json"),
                 "timing_sha256": old.file_hash(OUTPUT/"timing_probe.json")})
    progress("preflight_passed", main_projection_hours=timing["main_projection_seconds"]/3600)


def run_main(binding):
    probe = old.read_json(OUTPUT/"preflight.json")
    assert probe["passed"] and probe["manifest"] == validate_freeze() == binding
    assert probe["audit_sha256"] == old.file_hash(OUTPUT/"probe_audit.json")
    assert probe["timing_sha256"] == old.file_hash(OUTPUT/"timing_probe.json")
    launch_stage("main", binding, {"preflight_sha256": old.file_hash(OUTPUT/"preflight.json")})
    collect("main")
    data = observations("main")
    result = {**statistics(data["games"]), "operational": operational(data),
              "interpretation": "Learned157 continuations versus greedy035 continuations in155 at32 worlds; conditional on three fixed models. No equal-time or superhuman claim."}
    assert result["operational"]["continuations"] <= CONFIG["main_continuation_ceiling"]
    old.save_new(OUTPUT/"observations.json", data)
    old.save_new(OUTPUT/"result.json", result)
    progress("main_collected_audit_pending", games=CONFIG["games"])
    audit = audit_stage("main")  # Full replay on either statistical outcome.
    assert validate_freeze() == binding
    final = {"complete": True, "finished": old.now(), "manifest": binding,
             "statistical_pass": result["statistical_pass"], "integrity_pass": audit["passed"],
             "advance_gate_passed": result["statistical_pass"] and audit["passed"],
             "result_sha256": old.file_hash(OUTPUT/"result.json"),
             "audit_sha256": old.file_hash(OUTPUT/"main_audit.json"),
             "elapsed_seconds": (datetime.fromisoformat(old.now())-datetime.fromisoformat(
                 old.read_json(OUTPUT/"probe_launch.json")["time"])).total_seconds(),
             **{k: CONFIG[k] for k in ("training", "arena", "promotion", "automatic_successor")}}
    report = (f"# Experiment158 result\n\nCandidate score {result['share']:.4%}; stratified paired95% CI "
              f"[{result['ci95'][0]:.4%}, {result['ci95'][1]:.4%}]. W/D/L "
              f"{result['wdl']['wins']}/{result['wdl']['draws']}/{result['wdl']['losses']}. "
              f"Mean score margin {result['mean_margin']:+.4f}.\n\n"
              f"Statistical pass: {result['statistical_pass']}. Integrity pass: {audit['passed']}. "
              f"All {audit['games']} games, {audit['roots']} roots and {audit['continuations']} "
              "continuations independently reconstructed. Three fixed learned models received134 games each; "
              "the bootstrap resampled67 complete initial-seed pairs within each model.\n\n"
              "Both planners use32 worlds. The candidate substitutes the exact157 controller for both "
              "simulated players. The result concerns this complete substitution, including its proposal "
              "randomness; it does not isolate a unique cause or establish equal-time efficiency, "
              "repeated improvement or superhuman play. Engine, encoders and neural scorers are shared "
              "frozen primitives; world/candidate construction, action and game loops, input invariance "
              "and statistical reconstruction are independently checked. See result.json for per-model "
              "and seat estimates, timing and coverage. No promotion, training or successor ran.\n")
    with (OUTPUT/"report.md").open("x", encoding="utf-8") as handle:
        handle.write(report)
    old.save_new(OUTPUT/"final_status.json", final)
    progress("complete", **final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("start",))
    parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with old.process_lock(OUTPUT/"run.lock"):
        assert not (OUTPUT/"manifest.json").exists(), "Existing experiment needs review; no automatic resume"
        assert not list(OUTPUT.glob("failure*.json")), "Prior failure requires review"
        try:
            binding = freeze()
            progress("frozen", manifest=binding)
            preflight(binding)
            run_main(binding)
        except BaseException:
            old.save_new(OUTPUT/f"failure_start_{time.time_ns()}.json",
                         {"time": old.now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
