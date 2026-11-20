"""Authorized breadth versus precision: equal continuation budgets and fixed gates."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from .common import (ROOT, OUTPUT, SOURCE, SOURCE_BINDING, FITS, CONFIG, old,
                     windows, require, artifact, verified, verify_hash,
                     budget_boundary, BudgetExceeded)
from . import models, source, engine, verify
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.proposal_iteration_128 import experiment as parent

CHECK_TARGETS = [
    "tools/proposal_breadth_128/test_models.py",
    "tools/proposal_breadth_128/test_source.py",
    "tools/proposal_breadth_128/test_engine.py",
    "tools/proposal_breadth_128/test_verify.py",
    "tools/proposal_breadth_128/test_experiment.py",
    "tools/proposal_breadth_128/test_integration.py",
    "tools/proposal_breadth_128/test_worker.py",
    "tests/test_windowless_workers_160.py",
]
EXTRAS = ["run_proposal_breadth_128.py", "docs/engineering/proposal_breadth_128.md",
          "docs/strategy/next_after_proposal_ranking_128.md",
          "docs/engineering/verification_policy.md"]
CONTRASTS = ("breadth_minus_precision", "breadth_minus_anchor", "precision_minus_anchor")
BLOCKS = ("selection", "extension", "assessment")
REPORTER = None


def identity():
    previous = old.read_json(SOURCE / "manifest.json")["identity"]
    paths = {ROOT / p for p in previous["dependencies"]}
    recovery = old.read_json(SOURCE / "recovery_01/manifest.json")["identity"]
    paths.update(ROOT / p for p in recovery["dependencies"])
    for ancestor in (previous, recovery):
        for path, expected in ancestor["dependencies"].items():
            require(old.file_hash(ROOT / path) == expected, "Frozen ancestor dependency changed: "+path)
    paths.update((ROOT / "tools/proposal_breadth_128").glob("*.py"))
    paths.update(ROOT / p for p in EXTRAS + CHECK_TARGETS)
    return {"config": CONFIG,
            "fits": list(FITS), "checks": CHECK_TARGETS,
            "dependencies": {p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in sorted(paths)},
            "python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable())}


def parent_evidence():
    files = models.source_evidence()
    for item in files:
        verify_hash(item)
    return {"binding": SOURCE_BINDING, "files": files,
            "scope": "Closed ranking/recovery provenance and exactly three unchanged original161 anchors"}


def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    before = identity()
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main(" + repr(["-q", *CHECK_TARGETS]) + "))"
    started = time.perf_counter()
    with (folder / "stdout.txt").open("x", encoding="utf-8") as out, (folder / "stderr.txt").open("x", encoding="utf-8") as err:
        completed = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Code changed during preflight")
    old.save_new(folder / "checks.json", {"identity": before, "targets": CHECK_TARGETS,
                 "exit_code": completed.returncode, "seconds": time.perf_counter()-started,
                 "logs": [artifact(folder / n) for n in ("stdout.txt", "stderr.txt")]})
    return completed.returncode


def prepare(checks_path):
    require(not (OUTPUT / "manifest.json").exists(), "Already frozen; preserve evidence")
    checks = old.read_json(checks_path)
    require(checks["exit_code"] == 0 and checks["identity"] == identity()
            and checks["targets"] == CHECK_TARGETS, "Current complete passing preflight required")
    for item in checks["logs"]:
        verify_hash(item)
    evidence = parent_evidence()
    schedule = {stage: source.schedule(stage) for stage in ("probe", "main")}
    seeds = [s["environment_seed"] for rows in schedule.values() for s in rows]
    require(len(seeds) == len(set(seeds)) == 1134, "Source namespaces overlap")
    worlds = [seed for stage in ("probe", "main") for i in range(CONFIG["states"][stage])
              for block in BLOCKS for seed in engine.world_seeds(stage, i, block)]
    require(len(worlds) == len(set(worlds)) == 378*384 and set(worlds).isdisjoint(seeds), "World namespaces overlap")
    old.save_new(OUTPUT / "schedule.json", schedule)
    manifest = {"identity": identity(), "parent": evidence, "schedule": artifact(OUTPUT / "schedule.json"),
                "checks": artifact(checks_path), "owner_authorized": True,
                "authorization": "Owner accepted the fixed breadth-versus-precision experiment on2026-09-27; no training, arena, extension or successor."}
    old.save_new(OUTPUT / "manifest.json", manifest)
    binding = old.digest(manifest)
    old.save_new(OUTPUT / "freeze.json", {"manifest": artifact(OUTPUT / "manifest.json"),
                 "binding": binding, "time": old.now(), "owner_authorized": True})
    require(validate_freeze() == binding, "Post-freeze mismatch")
    print({"prepared": True, "binding": binding, "main_states": 360}, flush=True)


def validate_freeze():
    freeze = old.read_json(OUTPUT / "freeze.json")
    manifest = verified(freeze["manifest"])
    require(freeze["owner_authorized"] and old.digest(manifest) == freeze["binding"], "Freeze changed")
    current = identity()
    require(current == manifest["identity"], "Frozen code/runtime changed")
    checks = verified(manifest["checks"])
    require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS
            and checks["identity"] == current, "Frozen preflight changed")
    require(verified(manifest["schedule"]) == {s: source.schedule(s) for s in ("probe", "main")}, "Schedule changed")
    for item in [*checks["logs"], *manifest["parent"]["files"]]:
        verify_hash(item)
    return freeze["binding"]


def event(stage, **values):
    global REPORTER
    if REPORTER is None:
        REPORTER = StatusReporter(OUTPUT)
    return REPORTER.event(stage, **values)


def worker_initialize(stage, kind, binding, base):
    folder = Path(base) / "workers" / f"{stage}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    health = windows.assert_windowless_worker(folder / f"{os.getpid()}_stdout.txt", folder / f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    if stage.startswith("source_"):
        source.initialize_models(base)
    elif stage in ("probe", "main"):
        engine.initialize_models(base)
    require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), "Worker arithmetic contract")
    old.save_new(folder / f"{os.getpid()}_health.json", {"manifest": binding, "stage": stage, "kind": kind, **health})


def run_job(stage, kind, argument, binding, base):
    if stage.startswith("source_"):
        if kind == "collection":
            return source.collect(argument, binding, base)
        return source.audit(argument[0], argument[1], binding, base)
    if kind == "collection":
        return engine.collect(stage, argument, binding, base)
    return verify.audit(stage, argument[0], argument[1], binding, base)


def run_pool(stage, kind, arguments, binding, upstream):
    require(validate_freeze() == binding, "Frozen identity changed before stage")
    budget_boundary(OUTPUT)
    started = time.perf_counter()
    launch_path = OUTPUT / f"{stage}_{kind}_launch.json"
    old.save_new(launch_path, {"manifest": binding, "stage": stage, "kind": kind, "time": old.now(),
                 "dependencies": [artifact(p) for p in upstream]})
    runtime = windows.configure_spawn()
    event(f"{stage}_{kind}", completed=0, total=len(arguments))
    files = []
    workers = CONFIG["workers"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_initialize, initargs=(stage, kind, binding, OUTPUT)) as pool:
        jobs = [pool.submit(run_job, stage, kind, arg, binding, OUTPUT) for arg in arguments]
        try:
            for job in as_completed(jobs):
                files.append(job.result())
                event(f"{stage}_{kind}", completed=len(files), total=len(arguments))
                budget_boundary(OUTPUT)
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    files.sort(key=lambda item: item["index"])
    require([x["index"] for x in files] == list(range(len(arguments))), "Missing or duplicate job")
    for item in files:
        verify_hash(item)
    old.save_new(OUTPUT / f"{stage}_{kind}_lock.json", {"manifest": binding, "launch": artifact(launch_path),
                 "files": files, "elapsed_seconds": time.perf_counter()-started, "runtime": runtime})
    return files


def stage_lock(stage, kind, binding):
    lock = old.read_json(OUTPUT / f"{stage}_{kind}_lock.json")
    count = (CONFIG["source_games"][stage[7:]]
             if stage.startswith("source_") else CONFIG["states"][stage])
    require(lock["manifest"] == binding and [x["index"] for x in lock["files"]] == list(range(count)), "Incomplete stage lock")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["stage"] == stage and launch["kind"] == kind, "Stage launch changed")
    for item in [*lock["files"], *launch["dependencies"]]:
        verify_hash(item)
    return lock


def build_sources(binding):
    models.validate(binding, OUTPUT)
    schedule = old.read_json(OUTPUT / "schedule.json")
    collected, audited = {}, {}
    for stage in ("probe", "main"):
        upstream = [OUTPUT / "freeze.json", OUTPUT / "schedule.json", OUTPUT / "models_lock.json"]
        collected[stage] = run_pool(f"source_{stage}", "collection", schedule[stage], binding, upstream)
        audited[stage] = run_pool(f"source_{stage}", "audits", list(zip(schedule[stage], collected[stage])), binding,
                                  [*upstream, OUTPUT / f"source_{stage}_collection_lock.json"])
    item = source.prepare_selection(collected, audited, binding, OUTPUT)
    selection = verified(item)
    require(len(selection["main"]) == 360 and len(selection["probe"]) == 18, "Wrong panel sizes")
    old.save_new(OUTPUT / "selection_seal.json", {"manifest": binding, "selection": item,
                 "source_locks": [artifact(OUTPUT / f"source_{s}_{k}_lock.json")
                                  for s in ("probe", "main") for k in ("collection", "audits")]})
    for stage in ("probe", "main"):
        plan = verify.sample_plan(selection[stage], stage, binding)
        path = OUTPUT / stage / "audit_sample.json"
        old.save_new(path, plan)
        old.save_new(OUTPUT / stage / "audit_sample_seal.json", {"manifest": binding,
                     "stage": stage, "sample": artifact(path), "selection": item})
    budget_boundary(OUTPUT)
    return item


def validate_selection(binding):
    seal = old.read_json(OUTPUT / "selection_seal.json")
    require(seal["manifest"] == binding and seal["selection"] == artifact(OUTPUT / "selection.json"), "Selection seal changed")
    selection = verified(seal["selection"])
    require(selection["manifest"] == binding, "Selection binding changed")
    for item in seal["source_locks"]:
        verify_hash(item)
    for stage in ("probe", "main"):
        sample_seal = old.read_json(OUTPUT / stage / "audit_sample_seal.json")
        require(sample_seal["manifest"] == binding and sample_seal["stage"] == stage
                and sample_seal["selection"] == seal["selection"], "Sample seal changed")
        require(verified(sample_seal["sample"]) == verify.sample_plan(selection[stage], stage, binding), "Sample changed")
    return selection


def build_stage(stage, binding):
    selection = validate_selection(binding)
    models.validate(binding, OUTPUT)
    upstream = [OUTPUT / "freeze.json", OUTPUT / "selection_seal.json", OUTPUT / "selection.json",
                OUTPUT / "models_lock.json",
                OUTPUT / stage / "audit_sample.json", OUTPUT / stage / "audit_sample_seal.json"]
    if stage == "main":
        gate = old.read_json(OUTPUT / "operational_decision.json")
        require(gate["manifest"] == binding and gate["reasonable"], "Passing operational forecast required")
        upstream += [OUTPUT / "operational_decision.json", OUTPUT / "probe_complete.json"]
    roots = run_pool(stage, "collection", selection[stage], binding, upstream)
    audits = run_pool(stage, "audits", list(zip(selection[stage], roots)), binding,
                      [*upstream, OUTPUT / f"{stage}_collection_lock.json"])
    plan_path = OUTPUT / stage / "audit_sample.json"
    plan = old.read_json(plan_path)
    require(plan == verify.sample_plan(selection[stage], stage, binding), "Sampling reconstruction disagrees")
    keys = [f"{kind}_{unit}" for kind in ("recorded", "replayed") for unit in ("worlds", "continuations")]
    keys += [f"{kind}_{block}_{unit}" for kind in ("recorded", "replayed") for block in BLOCKS for unit in ("worlds", "continuations")]
    coverage = {"states": len(roots), "full_stress_roots": 0, **dict.fromkeys(keys, 0)}
    for spec, root_item, audit_item, sample in zip(selection[stage], roots, audits, plan["selections"], strict=True):
        budget_boundary(OUTPUT)
        receipt = verified(audit_item)
        validate_coverage(receipt, spec, root_item, sample, artifact(plan_path), binding)
        verify_hash(receipt["selection_artifact"])
        for key in keys:
            coverage[key] += receipt[key]
        coverage["full_stress_roots"] += int(receipt["stress"])
    old.save_new(OUTPUT / f"{stage}_complete.json", {"manifest": binding, "time": old.now(),
                 "states": len(roots), "collection": artifact(OUTPUT / f"{stage}_collection_lock.json"),
                 "audits": artifact(OUTPUT / f"{stage}_audits_lock.json"), "coverage": coverage,
                 "all_prescribed_audits_pass": True, "full_inner_replay": stage == "probe"})


def geometry(support):
    """Orchestrator's independent continuation arithmetic."""
    require(type(support) is int and support > 0, "Invalid support")
    m, n = min(support, 4), min(support, 8)
    require(128*n % m == 0, "Fractional precision budget")
    return {"selection": 128, "extension": 128*n//m-128, "assessment": 128}, {
        "selection": n, "extension": m, "assessment": min(n, 3)}


def validate_coverage(receipt, spec, root, sample, sample_artifact, binding):
    require(receipt["manifest"] == binding and receipt["passed"] and receipt["spec"] == spec
            and receipt["root"] == {k: root[k] for k in ("path", "sha256")}, "Audit linkage mismatch")
    require(receipt["sample_artifact"] == sample_artifact and receipt["stress"] == sample["stress"]
            and receipt["ordinary_world_indices"] == sample["ordinary"]
            and receipt["replay_world_indices"] == sample["replay"], "Audit sample mismatch")
    lengths, supports = geometry(spec["canonical_support"])
    require(1 <= receipt["support_assessment"] <= supports["assessment"], "Assessment support mismatch")
    supports["assessment"] = receipt["support_assessment"]
    for block in BLOCKS:
        require(receipt[f"support_{block}"] == supports[block], "Candidate support mismatch")
        for kind, count in (("recorded", lengths[block]), ("replayed", len(sample["replay"][block]))):
            require(receipt[f"{kind}_{block}_worlds"] == count
                    and receipt[f"{kind}_{block}_continuations"] == count*supports[block], "Block coverage mismatch")
    for kind in ("recorded", "replayed"):
        for unit in ("worlds", "continuations"):
            require(receipt[f"{kind}_{unit}"] == sum(receipt[f"{kind}_{block}_{unit}"] for block in BLOCKS), "Total coverage mismatch")
    require(receipt["recorded_continuations"] == root["continuations"], "Root work mismatch")
    require(receipt["full_inner_replay"] == (receipt["recorded_worlds"] == receipt["replayed_worlds"]), "Replay disclosure mismatch")


def cost_projection(collection, replay, probe_counts, main_counts, main_replays,
                    collection_structural, audit_structural, size_bytes, free_bytes, elapsed_seconds):
    """Outcome-blind, separately measured rates for all three simulation blocks."""
    for value in (collection, replay, probe_counts, main_counts, main_replays):
        require(set(value) == set(BLOCKS), "Incomplete timing blocks")
    numbers = [x for value in (collection, replay, probe_counts, main_counts, main_replays) for x in value.values()]
    numbers += [collection_structural, audit_structural, size_bytes, free_bytes, elapsed_seconds]
    require(all(type(x) in (int, float) and np.isfinite(x) and x >= 0 for x in numbers), "Invalid cost evidence")
    missing = [b for b in BLOCKS if (main_counts[b] or main_replays[b])
               and min(probe_counts[b], collection[b], replay[b]) <= 0]
    if missing:
        return {"reasonable": False, "reason": "Unmeasured required probe blocks", "missing_blocks": missing,
                "remaining_seconds": None, "remaining_hours": None, "elapsed_seconds": elapsed_seconds}
    collection_rates = {b: collection[b]/probe_counts[b] if probe_counts[b] else 0. for b in BLOCKS}
    replay_rates = {b: replay[b]/probe_counts[b] if probe_counts[b] else 0. for b in BLOCKS}
    main_collection = (sum(collection_rates[b]*main_counts[b] for b in BLOCKS)+20*collection_structural)/CONFIG["workers"]
    main_audit = (sum(replay_rates[b]*main_replays[b] for b in BLOCKS)+20*audit_structural)/CONFIG["workers"]
    remaining = CONFIG["forecast_allowance"]*(main_collection+main_audit)+CONFIG["finalization_allowance_seconds"]
    require(sum(probe_counts.values()) > 0, "No probe work")
    projected = int(size_bytes/sum(probe_counts.values())*sum(main_counts.values())*1.5+512*1024**2)
    return {"remaining_seconds": remaining, "remaining_hours": remaining/3600,
            "collection_seconds_before_allowance": main_collection, "audit_seconds_before_allowance": main_audit,
            "collection_seconds_per_continuation": collection_rates, "replay_seconds_per_continuation": replay_rates,
            "probe_continuations": probe_counts, "main_continuations_upper_bound": main_counts,
            "main_replay_continuations_upper_bound": main_replays,
            "elapsed_seconds": elapsed_seconds, "projected_bytes": projected, "free_bytes": free_bytes,
            "reasonable": elapsed_seconds+remaining <= CONFIG["maximum_runtime_hours"]*3600
                          and free_bytes-projected >= CONFIG["disk_reserve_bytes"]}


def operational_decision(binding):
    budget_boundary(OUTPUT)
    roots = [verified(x) for x in stage_lock("probe", "collection", binding)["files"]]
    audits = [verified(x) for x in stage_lock("probe", "audits", binding)["files"]]
    complete = old.read_json(OUTPUT / "probe_complete.json")
    require(complete["manifest"] == binding and complete["all_prescribed_audits_pass"]
            and complete["full_inner_replay"], "Fully audited probe required")
    selection = validate_selection(binding)
    plan = old.read_json(OUTPUT / "main/audit_sample.json")
    collection = {b: sum(r[b]["timing"]["wall"] for r in roots) for b in BLOCKS}
    structural = sum(r["timing"]["wall"] for r in roots)-sum(collection.values())
    require(structural >= -1e-6, "Invalid collection timing accounting")
    replay = {b: sum(r["timing"][f"{b}_replay_wall"] for r in audits) for b in BLOCKS}
    counts = {b: sum(r[f"recorded_{b}_continuations"] for r in audits) for b in BLOCKS}
    require(all(r["recorded_continuations"] == r["replayed_continuations"] for r in audits), "Incomplete probe replay")
    main_counts, main_replays = dict.fromkeys(BLOCKS, 0), dict.fromkeys(BLOCKS, 0)
    for spec, sample in zip(selection["main"], plan["selections"], strict=True):
        lengths, supports = geometry(spec["canonical_support"])
        for block in BLOCKS:
            main_counts[block] += lengths[block]*supports[block]
            main_replays[block] += len(sample["replay"][block])*supports[block]
    require(sum(main_counts.values()) <= CONFIG["maximum_main_continuations"], "Main work ceiling exceeded")
    value = cost_projection(collection, replay, counts, main_counts, main_replays,
        max(0., structural), sum(r["timing"]["structural_wall"] for r in audits),
        sum(p.stat().st_size for p in (OUTPUT / "probe").rglob("*") if p.is_file()),
        shutil.disk_usage(ROOT).free, time.time()-old.read_json(OUTPUT / "launch.json")["epoch_seconds"])
    old.save_new(OUTPUT / "operational_decision.json", {"manifest": binding, "time": old.now(), **value})
    if not value["reasonable"]:
        old.save_new(OUTPUT / "operational_hold.json", {"manifest": binding, "time": old.now(),
                     "reason": value.get("reason", "Outcome-blind forecast exceeds runtime or storage budget"), "forecast": value})
        event("operational_hold", remaining_forecast_hours=value["remaining_hours"])
        return False
    event("probe_passed", remaining_forecast_hours=value["remaining_hours"])
    return True


def independent_statistics(rows):
    require(len(rows) == 360 and {r["index"] for r in rows} == set(range(360)), "Complete fixed panel required")
    require(len({r["game_index"] for r in rows}) == 360, "Repeated source games")
    strata = []
    for fit in FITS:
        for seat in (0, 1):
            for phase in (0, 1, 2):
                subset = sorted((r for r in rows if (r["fit"], r["seat"], r["phase"]) == (fit, seat, phase)), key=lambda r: r["index"])
                require(len(subset) == 20, "Fixed source-cell weights changed")
                values = []
                for row in subset:
                    scores = {a: int(round(v*256)) for a, v in row["arm_scores"].items()}
                    require(set(scores) == {"anchor", "breadth", "precision"}
                            and all(0 <= n <= 256 and abs(n/256-row["arm_scores"][a]) < 1e-12 for a, n in scores.items()), "Score scale changed")
                    values.append([scores["breadth"]-scores["precision"], scores["breadth"]-scores["anchor"], scores["precision"]-scores["anchor"]])
                strata.append(np.asarray(values, dtype=np.int64))
    totals = sum(x.sum(axis=0) for x in strata)
    point = totals/(360*256)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    samples = []
    for _ in range(50):
        batch = np.zeros((1000, 3), dtype=np.int64)
        for values in strata:
            batch += values[rng.integers(0, 20, size=(1000, 20))].sum(axis=1)
        samples.append(batch/(360*256))
    intervals = np.quantile(np.concatenate(samples), [.025, .975], axis=0, method="linear")
    contrasts = {name: {"mean": float(point[i]), "ci95": intervals[:, i].tolist()} for i, name in enumerate(CONTRASTS)}
    gates = {"anchor_practical_gain": bool(totals[1]*200 >= 360*256),
             "anchor_superiority": bool(intervals[0, 1] > 0), "precision_superiority": bool(intervals[0, 0] > 0)}
    gates["gate_pass"] = all(gates.values())
    return {"contrasts": contrasts, **gates,
            "classification": "positive_conditional_breadth_contribution" if gates["gate_pass"] else "practical_progress_not_established"}


def finish(binding):
    budget_boundary(OUTPUT)
    require(validate_freeze() == binding, "Frozen inputs changed at finalization")
    models.validate(binding, OUTPUT)
    selection = validate_selection(binding)
    for stage in ("probe", "main"):
        sources = stage_lock(f"source_{stage}", "collection", binding)
        source_audits = stage_lock(f"source_{stage}", "audits", binding)
        require(len(sources["files"]) == len(source_audits["files"]), "Source closure changed")
        for root_item, audit_item in zip(stage_lock(stage, "collection", binding)["files"],
                                         stage_lock(stage, "audits", binding)["files"], strict=True):
            budget_boundary(OUTPUT)
            record, receipt = verified(root_item), verified(audit_item)
            require(receipt["manifest"] == binding and receipt["passed"]
                    and receipt["root"] == {k: root_item[k] for k in ("path", "sha256")}, "Final receipt linkage changed")
            verify_hash(record["selection_artifact"])
    summary = verify.summarize(stage_lock("main", "audits", binding)["files"], selection["main"], binding, OUTPUT)
    rows = []
    for item in stage_lock("main", "collection", binding)["files"]:
        budget_boundary(OUTPUT)
        rows.append(engine.statistics_row(verified(item)))
    independent = independent_statistics(rows)
    checked = summary["statistics"]
    for name, row in independent["contrasts"].items():
        comparison = checked["contrasts"][name]
        require(abs(row["mean"]-comparison["mean_score_difference"]) < 1e-12
                and np.allclose(row["ci95"], comparison["ci95"], rtol=0., atol=1e-12), "Independent statistics disagree")
    for key in ("anchor_practical_gain", "anchor_superiority", "precision_superiority", "gate_pass", "classification"):
        require(independent[key] == checked[key], "Independent gate disagrees")
    label = independent["classification"]
    budget_boundary(OUTPUT)
    old.save_new(OUTPUT / "result.json", {"manifest": binding, "time": old.now(), **summary,
                 "classification": label, "all_prescribed_audits_pass": True, "full_inner_replay": False,
                 "models": artifact(OUTPUT / "models_lock.json"), "selection": artifact(OUTPUT / "selection_seal.json"),
                 "scope": "Fresh fast161-source positions, independent assessment under greedy035 continuations; not full-game strength",
                 "automatic_successor": False})
    old.save_new(OUTPUT / "final_audit.json", {"manifest": binding, "time": old.now(), "passed": True,
                 "result": artifact(OUTPUT / "result.json"), "independent_statistics": independent,
                 "stages": {s: artifact(OUTPUT / f"{s}_complete.json") for s in ("probe", "main")},
                 "models": artifact(OUTPUT / "models_lock.json"), "full_inner_replay": False})
    lines = ["# Breadth versus matched precision", "", f"Classification: {label}.", "",
             "360 fresh source positions, all three fits retained; score win1/draw.5/loss0. These are conditional decision scores, not arena win rates.", "",
             "| Contrast | Gain (percentage points) | Paired95% interval (percentage points) |", "|---|---:|---:|"]
    for name, row in independent["contrasts"].items():
        lines.append(f"| {name} | {100*row['mean']:.4f} | [{100*row['ci95'][0]:.4f}, {100*row['ci95'][1]:.4f}] |")
    lines += ["", "The practical gate requires >=+0.5pp versus anchor and a positive95% lower endpoint versus both anchor and equal-budget precision. This is a compute-allocation hurdle, not a conversion to whole-game strength.", "",
              "Three unchanged original161 anchors were hash-verified. Breadth adds up to two ranked and two identity-uniform proposals. Precision spends exactly the same continuation count on the original candidates. Shared execution does not measure separate arm wall times; equal branch counts do not imply equal time.", "",
              "Breadth uses128 shared selection worlds. Precision extends the original candidates to128/160/192/224/256 worlds according to support. All choices are locked before128 independent assessment worlds. Every source game and proposal was checked; prescribed sampled continuation replay plus full stress cases passed. Shared simulation/encoding/neural primitives remain trusted.", "",
              "No automatic arena, extension, promotion, successor or monitoring.", ""]
    with (OUTPUT / "report.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    budget_boundary(OUTPUT)
    old.save_new(OUTPUT / "final_status.json", {"manifest": binding, "time": old.now(), "complete": True,
                 "classification": label, "all_prescribed_audits_pass": True, "full_inner_replay": False,
                 "result": artifact(OUTPUT / "result.json"), "audit": artifact(OUTPUT / "final_audit.json"),
                 "report": artifact(OUTPUT / "report.md"), "automatic_successor": False})
    event("complete", classification=label)


def run():
    global REPORTER
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(folder / "stdout.txt", folder / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot run; preserve existing evidence")
        binding = validate_freeze()
        REPORTER = StatusReporter(OUTPUT, {"manifest": binding})
        try:
            with parent.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(),
                             "epoch_seconds": time.time(), "pid": os.getpid(), "sleep_inhibition": sleep,
                             "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
                models.prepare(binding, OUTPUT)
                budget_boundary(OUTPUT)
                build_sources(binding)
                budget_boundary(OUTPUT)
                build_stage("probe", binding)
                if not operational_decision(binding):
                    return
                build_stage("main", binding)
                budget_boundary(OUTPUT)
                finish(binding)
        except BudgetExceeded as exc:
            if not (OUTPUT / "operational_hold.json").exists():
                old.save_new(OUTPUT / "operational_hold.json", {"manifest": binding, "time": old.now(), "reason": str(exc)})
            event("operational_hold", reason=str(exc))
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
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_proposal_breadth_128.py", "run"],
                                          folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic()+60
        while not (OUTPUT / "launch.json").exists():
            require(time.monotonic() < deadline, "No acknowledgement; inspect before any retry")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT / "launch.json")
        require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding, "Supervisor identity mismatch")
        old.save_new(folder / "acknowledged.json", {"time": old.now(), "pid": process["pid"], "launch": artifact(OUTPUT / "launch.json")})
        print({"launched": True, "pid": process["pid"], "binding": binding}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "prepare", "launch", "run"))
    parser.add_argument("--checks", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        require(args.checks is not None, "Supply a new check folder")
        return check(args.checks)
    if args.command == "prepare":
        require(args.checks is not None, "Supply current checks.json")
        prepare(args.checks)
    elif args.command == "launch":
        launch()
    else:
        run()
    return 0
