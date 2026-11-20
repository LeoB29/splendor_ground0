"""157: sealed transfer pilot, complete audit on either strength outcome."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero import fast_selector_157 as f

old, OUTPUT, CONFIG = f.old, f.OUTPUT, f.CONFIG
PROTOCOL = Path("docs/engineering/fast_selector_transfer_157.md")
VERIFIER = Path("scripts/analysis/verify_fast_selector_transfer_157.py")
INVENTORY = [Path("src/splendor_ai/alphazero/fast_selector_157.py"),
             Path("src/splendor_ai/alphazero/run_fast_selector_transfer.py"), PROTOCOL, VERIFIER,
             Path("run_fast_selector_transfer_157.py"), Path("tests/test_fast_selector_transfer_157.py"),
             Path("scripts/analysis/launch_fast_selector_transfer_157.py")]
_MODELS = {}


def identity():
    assert f.parent.check_identity() == "dab0e5502a4eca5005c04c02d2095acc601a7cadee196c2173725d85bce69774"
    assert old.file_hash(f.parent.OUTPUT/"main_audit.json") == "20e079af9fcb851f36c6c2fa460912325a2fd0b75a4564d111f0c709dc4dd2ab"
    assert old.file_hash(f.parent.OUTPUT/"final_status.json") == "7ca4c20902e57b3ad491618889c75dfa767af3e1e424d0f96c9d5dac949329aa"
    parents = {}
    for exp, module in f.SOURCES.items():
        final, audit = [old.read_json(module.OUTPUT/name) for name in ("final_status.json", "main_audit.json")]
        assert final["complete"] and audit["passed"]
        assert final["audit_sha256"] == old.file_hash(module.OUTPUT/"main_audit.json")
        assert final["result_sha256"] == audit["result_sha256"] == old.file_hash(module.OUTPUT/"result.json")
        parents[str(exp)] = {name: old.file_hash(module.OUTPUT/name) for name in
                             ("manifest.json", "final_status.json", "main_audit.json", "main_games_lock.json", "result.json")}
    return {"format": "fast_selector_transfer_157_v1", "config": CONFIG, "parents": parents,
            "dependencies": {p.as_posix(): old.file_hash(p) for p in INVENTORY},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def check_identity():
    saved = old.read_json(OUTPUT/"manifest.json")
    assert saved == identity(), "157 identity changed"
    return old.digest(saved)


def verify_test_evidence():
    path = OUTPUT/"preparation"/"tests.json"
    report = old.read_json(path)
    assert report["identity"] == identity() and report["full_repository"] and report["exit_code"] == 0
    assert report["log_sha256"] == old.file_hash(path.with_name("tests.txt"))
    assert report["wrapper_sha256"] == old.file_hash(Path("scripts/analysis/launch_fast_selector_transfer_157.py"))
    return old.file_hash(path)


def progress(stage, **data):
    record = {"time": old.now(), "stage": stage, **data}
    old.atomic_write(OUTPUT/"progress.json", record)
    print(record, flush=True)


def schedule(stage):
    assert stage in ("probe", "main")
    result = []
    for pair in range(CONFIG[f"{stage}_pairs"]):
        for arm in f.ARMS:
            for seed in f.SEEDS:
                for seat in (0, 1):
                    result.append({"index": len(result), "pair": pair, "seat": seat, "arm": arm, "seed": seed,
                                   "environment_seed": CONFIG["environment_start"][stage]+pair,
                                   "stream": old.digest([CONFIG["stream_tag"], stage, pair, seat])})
    return result


def freeze():
    assert not (OUTPUT/"manifest.json").exists(), "Existing run needs explicit review; no automatic resume"
    tests = verify_test_evidence()
    manifest = identity()
    sources = f.source_inventory()
    old.save_new(OUTPUT/"manifest.json", manifest)
    old.save_new(OUTPUT/"sources.json", sources)
    old.save_new(OUTPUT/"splits.json", f.split_membership())
    old.save_new(OUTPUT/"schedules.json", {stage: schedule(stage) for stage in ("probe", "main")})
    old.save_new(OUTPUT/"freeze.json", {"manifest": old.digest(manifest), "time": old.now(), "tests_sha256": tests,
                 **{name: old.file_hash(OUTPUT/name) for name in ("sources.json", "splits.json", "schedules.json")}})
    return old.digest(manifest)


def validate_freeze():
    binding = check_identity()
    lock = old.read_json(OUTPUT/"freeze.json")
    assert lock["manifest"] == binding and lock["tests_sha256"] == verify_test_evidence()
    for name in ("sources.json", "splits.json", "schedules.json"):
        assert old.file_hash(OUTPUT/name) == lock[name]
    assert old.read_json(OUTPUT/"splits.json") == f.split_membership()
    assert old.read_json(OUTPUT/"schedules.json") == {s: schedule(s) for s in ("probe", "main")}
    return binding


def run_jobs(stage, function, arguments, workers=8, initializer=old.initialize_worker):
    start = time.perf_counter()
    results = []
    progress(stage, complete=0, total=len(arguments))
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
        futures = [pool.submit(function, *args) for args in arguments]
        try:
            for future in as_completed(futures):
                results.append(future.result())
                progress(stage, complete=len(results), total=len(futures), elapsed_seconds=time.perf_counter()-start)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results


def export():
    binding = validate_freeze()
    specs, splits = old.read_json(OUTPUT/"sources.json"), old.read_json(OUTPUT/"splits.json")
    files = run_jobs("export", f.export_game, [(spec, splits) for spec in specs])
    files.sort(key=lambda r: r["path"])
    assert sum(r["rows"] for r in files) == 24342
    old.save_new(OUTPUT/"export_lock.json", {"manifest": binding, "files": files})


def auditor(stage):
    validate_freeze()
    progress(f"{stage}_audit")
    completed = subprocess.run([sys.executable, "-u", str(VERIFIER), stage])
    if completed.returncode:
        raise RuntimeError(f"Independent {stage} audit failed: {completed.returncode}")
    report = old.read_json(OUTPUT/f"{stage}_audit.json")
    assert report["passed"] and report["manifest"] == check_identity()
    return report


def fit_all():
    assert old.read_json(OUTPUT/"export_audit.json")["passed"]
    files = run_jobs("fitting", f.fit_one, [(arm, seed) for arm in f.ARMS for seed in f.SEEDS], workers=3)
    files.sort(key=lambda r: (r["arm"], r["seed"]))
    old.save_new(OUTPUT/"fits_lock.json", {"manifest": check_identity(), "files": files,
                 "export_lock_sha256": old.file_hash(OUTPUT/"export_lock.json"),
                 "export_audit_sha256": old.file_hash(OUTPUT/"export_audit.json")})


def seal_selections():
    assert old.read_json(OUTPUT/"fits_audit.json")["passed"]
    selected = {}
    for arm in f.ARMS:
        for seed in f.SEEDS:
            folder = OUTPUT/"fits"/f"{arm}_{seed}"
            report = old.read_json(folder/"result.json")
            path = folder/f"checkpoint_{report['selected_update']:04d}.pt"
            selected[f"{arm}_{seed}"] = {"path": path.as_posix(), "sha256": old.file_hash(path), "update": report["selected_update"]}
    old.save_new(OUTPUT/"selected_lock.json", {"manifest": check_identity(), "time": old.now(), "selected": selected,
                 "fits_lock_sha256": old.file_hash(OUTPUT/"fits_lock.json"),
                 "fits_audit_sha256": old.file_hash(OUTPUT/"fits_audit.json")})


def initialize_games():
    old.initialize_worker()
    _MODELS.clear()
    lock = old.read_json(OUTPUT/"selected_lock.json")
    for key, item in lock["selected"].items():
        assert old.file_hash(Path(item["path"])) == item["sha256"]
        _MODELS[key] = f.StrongPolicyInference(item["path"]).model


def game_path(stage, index):
    return OUTPUT/stage/"games"/f"game_{index:04d}.json"


def play_game(stage, spec, binding):
    path = game_path(stage, spec["index"])
    assert not path.exists(), "No silent game reuse/replacement"
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    model = _MODELS[f"{spec['arm']}_{spec['seed']}"]
    record = {"manifest": binding, "spec": spec, "time": old.now(), "trace": []}
    wall = time.perf_counter()
    seen, forced, idle = Counter(), 0, 0
    reason, winner = "completed", None
    try:
        while not state.terminal:
            signature = old.state_signature(state)
            seen[signature] += 1
            assert seen[signature] < 4, "repetition_cutoff"
            assert state.turn_index < 400, "max_turns"
            legal = env.legal_actions(state)
            assert legal, "stalled"
            row = {"turn": state.turn_index, "actor": state.current_player,
                   "state_sha256": old.digest(old.serialize_state_snapshot(state)),
                   "legal_support": len(legal), "phase": old.phase(state.turn_index)}
            sw, sc = time.perf_counter(), time.process_time()
            if state.current_player == spec["seat"]:
                action, evidence = f.choose(state, spec["stream"], model)
                row["evidence"] = evidence
            else:
                action = old._RAW.choose_action(env, state, legal)
            row["timing"] = {"wall": time.perf_counter()-sw, "cpu": time.process_time()-sc}
            assert action is not None and action in legal, "illegal_or_missing_action"
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            nxt = env.step(state, action)
            forced = forced+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, nxt, action) else idle+1
            state = nxt
            if forced >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            assert idle < 60, "no_progress_cutoff"
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason, winner=winner)
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record.update(final_state=old.serialize_state_snapshot(state), elapsed_seconds=time.perf_counter()-wall)
        old.save_new(path, record)
    return {"index": spec["index"], "sha256": old.file_hash(path)}


def collect(stage):
    binding = validate_freeze()
    old.save_new(OUTPUT/f"{stage}_launch.json", {"manifest": binding, "time": old.now(),
                 "selected_lock_sha256": old.file_hash(OUTPUT/"selected_lock.json"),
                 "schedules_sha256": old.file_hash(OUTPUT/"schedules.json")})
    started = time.perf_counter()
    files = run_jobs(f"{stage}_collection", play_game, [(stage, s, binding) for s in schedule(stage)], initializer=initialize_games)
    files.sort(key=lambda r: r["index"])
    old.save_new(OUTPUT/f"{stage}_games_lock.json", {"manifest": binding, "files": files,
                 "launch_sha256": old.file_hash(OUTPUT/f"{stage}_launch.json"),
                 "elapsed_seconds": time.perf_counter()-started})


def statistics(rows):
    assert len(rows) == 4800 and {r["index"] for r in rows} == set(range(4800))
    values = np.full((400, 2, 3, 2), np.nan)
    margins = values.copy()
    for row in rows:
        coord = (row["pair"], f.ARMS.index(row["arm"]), f.SEEDS.index(row["seed"]), row["seat"])
        assert np.isnan(values[coord])
        values[coord], margins[coord] = row["score"], row["margin"]
    assert np.isfinite(values).all() and np.isfinite(margins).all()
    paired = values.mean(axis=3).mean(axis=2)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    samples = []
    for _ in range(50):
        idx = rng.integers(0, 400, size=(1000, 400))
        samples.append(paired[idx].mean(axis=1))
    boot = np.concatenate(samples)
    treatment_ci = np.quantile(boot[:, 0], [.025, .975]).tolist()
    diff_ci = np.quantile(boot[:, 0]-boot[:, 1], [.025, .975]).tolist()
    treatment, control = [float(values[:, a].sum()/2400) for a in (0, 1)]
    difference = float((values[:, 0].sum()-values[:, 1].sum())/2400)
    seat_shares = values[:, 0].mean(axis=(0, 1)).tolist()
    margin = float(margins[:, 0].mean())
    practical = treatment >= .55 and treatment_ci[0] > .5 and margin > 0 and min(seat_shares) >= .5
    attribution = difference >= .05 and diff_ci[0] > 0
    per_fit = {f"{arm}_{seed}": {"share": float(values[:, a, s].mean()),
               "seat_shares": values[:, a, s].mean(axis=0).tolist(), "margin": float(margins[:, a, s].mean())}
               for a, arm in enumerate(f.ARMS) for s, seed in enumerate(f.SEEDS)}
    return {"treatment_share": treatment, "control_share": control, "treatment_ci": treatment_ci,
            "difference": difference, "difference_ci": diff_ci, "treatment_seats": seat_shares,
            "treatment_margin": margin, "per_fit": per_fit, "practical_pass": practical, "attribution_pass": attribution}


def timings(rows):
    def summary(group):
        return {kind: f.parent.parent.quantiles([r["timing"][kind] for r in group]) for kind in ("wall", "cpu")}
    result = {}
    for arm in f.ARMS:
        for seed in f.SEEDS:
            group = [r for r in rows if r["arm"] == arm and r["seed"] == seed]
            result[f"{arm}_{seed}"] = {}
            for role in ("controller", "raw"):
                moves = [r for r in group if r["role"] == role]
                record = {"all": summary(moves), "phase": {}, "support": {}}
                for phase in range(3):
                    cell = [r for r in moves if r["phase"] == phase]
                    if cell:
                        record["phase"][str(phase)] = summary(cell)
                if role == "controller":
                    for support in range(1, 5):
                        cell = [r for r in moves if r["support"] == support]
                        if cell:
                            record["support"][str(support)] = summary(cell)
                result[f"{arm}_{seed}"][role] = record
    treatment = sum(result[f"treatment_{s}"]["controller"]["all"]["wall"]["mean"] for s in f.SEEDS)/3
    raw = sum(result[f"treatment_{s}"]["raw"]["all"]["wall"]["mean"] for s in f.SEEDS)/3
    return {"per_fit": result, "treatment_mean_wall": treatment, "matched_raw_mean_wall": raw,
            "ratio": treatment/raw, "passed": treatment < .1 and treatment <= 10*raw}


def close():
    audit = old.read_json(OUTPUT/"main_audit.json")
    assert audit["passed"]
    assert audit["observations_sha256"] == old.file_hash(OUTPUT/"main_observations.json")
    assert audit["games_lock_sha256"] == old.file_hash(OUTPUT/"main_games_lock.json")
    assert old.read_json(OUTPUT/"main_launch.json")["selected_lock_sha256"] == old.file_hash(OUTPUT/"selected_lock.json")
    observations = old.read_json(OUTPUT/"main_observations.json")
    result = {"statistics": statistics(observations["games"]), "timings": timings(observations["moves"])}
    assert old.digest(result) == audit["recomputed_result_digest"]
    result["passed"] = all((result["statistics"]["practical_pass"], result["statistics"]["attribution_pass"], result["timings"]["passed"]))
    old.save_new(OUTPUT/"result.json", result)
    binding = validate_freeze()
    old.save_new(OUTPUT/"final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "integrity_pass": True, "experiment_pass": result["passed"], "promotion": False,
                 "result_sha256": old.file_hash(OUTPUT/"result.json"), "audit_sha256": old.file_hash(OUTPUT/"main_audit.json")})
    report = Path("docs/engineering/fast_selector_transfer_157_result.md")
    with report.open("x", encoding="utf-8") as out:
        s = result["statistics"]
        out.write(f"# Experiment157 complete\n\nAll six fits and 4,800 fresh games independently replayed.\n\n"
                  f"Treatment share {s['treatment_share']:.4%}, 95% pair CI {s['treatment_ci']}; "
                  f"control {s['control_share']:.4%}. Difference {s['difference']:.4%}, CI {s['difference_ci']}.\n\n"
                  f"Practical gate: {s['practical_pass']}; attribution: {s['attribution_pass']}; latency: {result['timings']['passed']}.\n\n"
                  "Full curves and subgroup predictions: fits/*/validation_*.json; every fit result includes selected/final updates. "
                  "Per-fit outcomes and CPU/wall timing distributions: outputs/fast_selector_transfer_157/result.json. "
                  "Development data were reused. Intervals condition on these three fixed training seeds. "
                  "Both outcomes stop; no promotion or successor.\n")
    progress("complete", passed=result["passed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "status"])
    args = parser.parse_args()
    if args.command == "status":
        print(old.read_json(OUTPUT/"progress.json"))
        return
    with old.process_lock(OUTPUT/"pipeline.lock"):
        assert not (OUTPUT/"failure.json").exists()
        try:
            freeze()
            export()
            auditor("export")
            fit_all()
            auditor("fits")
            seal_selections()
            collect("probe")
            probe = auditor("probe")
            progress("probe_passed", measured_seconds=probe["elapsed_seconds"], projected_main_seconds=probe["projected_main_seconds"])
            collect("main")
            auditor("main")
            close()
        except BaseException:
            old.save_new(OUTPUT/"failure.json", {"time": old.now(), "traceback": traceback.format_exc()})
            raise
