"""Authorized matched action-card residuals and fresh conditional assessment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from functools import partial
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

from .common import (ROOT, OUTPUT, FITS, ARMS, CONTRASTS, CONFIG, old, windows,
                     require, artifact, verified, verify_hash, budget_boundary, BudgetExceeded)
from . import learning, training_source, source, engine, verify, statistics, common as c, model
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.proposal_iteration_128 import experiment as parent

CHECK_TARGETS = [f"tools/action_card_relations_128/test_{name}.py" for name in
                 ("model", "training_source", "learning", "source", "engine", "verify", "statistics", "experiment", "worker")]
EXTRAS = ["run_action_card_relations_128.py", "docs/engineering/action_card_relations_128.md",
          "docs/engineering/verification_policy.md"]
ANCESTORS = ("soft_teacher_targets_128", "proposal_breadth_128", "equal_budget_breadth_128", "planner_state_transfer_128",
             "independent_game_coverage_128", "independent_game_coverage_128_recovery_01", "frozen_representation_128")
BLOCKS = ("selection", "assessment")
REPORTER = None
RUNTIME_MARKERS = ("abort.json", "failure.json", "operational_hold.json", "launch.json",
                   "runtime/process.json", "learning_fits_launch.json", "models_lock.json")


def assert_prelaunch_clean(base=None):
    base = Path(OUTPUT if base is None else base)
    present = [name for name in RUNTIME_MARKERS if (base/name).exists()]
    require(not present, "Runtime evidence already exists; preserve and inspect: "+", ".join(present))


def identity():
    require(sys.flags.optimize == 0, "Unoptimized Python required for audit assertions")
    dependencies = {}
    for name in ANCESTORS:
        for path, expected in old.read_json(ROOT/"outputs"/name/"manifest.json")["identity"]["dependencies"].items():
            require(old.file_hash(ROOT/path) == expected, "Frozen ancestor changed: "+path)
            require(path not in dependencies or dependencies[path] == expected, "Conflicting dependency")
            dependencies[path] = expected
    paths = list((ROOT/"tools/action_card_relations_128").glob("*.py")) + [ROOT/p for p in EXTRAS]
    dependencies.update({p.relative_to(ROOT).as_posix(): old.file_hash(p) for p in paths})
    return {"config": CONFIG, "learning_config": learning.CONFIG, "fits": list(FITS),
            "checks": CHECK_TARGETS, "dependencies": dict(sorted(dependencies.items())),
            "python": sys.version, "optimize": sys.flags.optimize,
            "numpy": np.__version__, "torch": str(torch.__version__),
            "windowless": windows._identity(windows._windowless_executable()),
            "model_config": model.MODEL_CONFIG}

def parent_evidence():
    return training_source.closed_parent()


def forbidden_source_seeds(evidence):
    proof = evidence["training_disjointness"]
    values = set(proof["historical_environment_seeds"]) | set(proof["added_environment_seeds"])
    parent_manifest = old.read_json(c.PARENT / "manifest.json")
    values.update(parent_manifest["source_parent"]["forbidden_environment_seeds"])
    for rows in old.read_json(c.PARENT / "schedule.json").values():
        values.update(spec["environment_seed"] for spec in rows)
    return values

def check(folder):
    assert_prelaunch_clean()
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    before, started = identity(), time.perf_counter()
    fixture = ROOT/"outputs/action_card_relations_128_fixtures"/folder.name
    require(not fixture.exists(), "Preserve previous fixtures")
    fixture.parent.mkdir(parents=True, exist_ok=True)
    args = ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture)]
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main("+repr(args)+"))"
    with (folder/"stdout.txt").open("x", encoding="utf-8") as out, (folder/"stderr.txt").open("x", encoding="utf-8") as err:
        completed = subprocess.run([sys.executable,"-u","-c",code], cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err, timeout=1200,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    require(identity() == before, "Code changed during official preflight")
    assert_prelaunch_clean()
    old.save_new(folder/"checks.json", {"identity":before,"targets":CHECK_TARGETS,
                 "exit_code":completed.returncode,"seconds":time.perf_counter()-started,
                 "logs":[artifact(folder/n) for n in ("stdout.txt","stderr.txt")]})
    return completed.returncode

def prepare(checks_path):
    assert_prelaunch_clean()
    require(not (OUTPUT/"manifest.json").exists(), "Already frozen; preserve evidence")
    checks = old.read_json(checks_path)
    require(checks["exit_code"] == 0 and checks["identity"] == identity()
            and checks["targets"] == CHECK_TARGETS, "Current full passing preflight required")
    for ref in checks["logs"]: verify_hash(ref)
    evidence = parent_evidence()
    schedule = {stage:source.schedule(stage) for stage in ("probe","main")}
    seeds = [s["environment_seed"] for rows in schedule.values() for s in rows]
    worlds = [seed for stage in ("probe","main") for index in range(CONFIG["states"][stage])
              for block in BLOCKS for seed in engine.world_seeds(stage,index,block)]
    require(len(seeds) == len(set(seeds)) == 1134, "Source namespaces overlap")
    require(len(worlds) == len(set(worlds)) == 378*256 and set(worlds).isdisjoint(seeds), "World namespaces overlap")
    forbidden = forbidden_source_seeds(evidence)
    require((set(seeds)|set(worlds)).isdisjoint(forbidden), "Fresh source/world seeds overlap previous games")
    old.save_new(OUTPUT/"schedule.json",schedule)
    manifest = {"identity":identity(),"source_parent":evidence,"schedule":artifact(OUTPUT/"schedule.json"),
                "checks":artifact(checks_path),"owner_authorized":True,
                "authorization":"Owner accepted action-card relations recommendation on2026-10-11: six matched residual fits on frozen161 proposers, exact R1 BROAD recipe, fresh360-position four-arm evaluation; hidden CPU process, no automatic arena or successor."}
    binding = old.digest(manifest)
    data = training_source.prepare(binding,OUTPUT,parent=evidence)
    verify_hash(data)
    require(identity() == manifest["identity"], "Code changed during data preparation")
    old.save_new(OUTPUT/"manifest.json",manifest)
    old.save_new(OUTPUT/"freeze.json",{"manifest":artifact(OUTPUT/"manifest.json"),"binding":binding,
                 "data":data,"time":old.now(),"owner_authorized":True})
    require(validate_freeze(full_sources=True) == binding, "Post-freeze identity mismatch")
    print({"prepared":True,"binding":binding,"fits":6,"main_states":360},flush=True)

def validate_freeze(full_sources=False):
    freeze = old.read_json(OUTPUT/"freeze.json")
    manifest = verified(freeze["manifest"])
    require(freeze["owner_authorized"] and manifest["owner_authorized"]
            and old.digest(manifest) == freeze["binding"], "Freeze changed")
    current = identity()
    require(current == manifest["identity"], "Frozen code/runtime changed")
    checks = verified(manifest["checks"])
    require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS
            and checks["identity"] == current, "Frozen preflight changed")
    require(verified(manifest["schedule"]) == {s:source.schedule(s) for s in ("probe","main")}, "Schedule changed")
    for ref in checks["logs"]: verify_hash(ref)
    if full_sources:
        for ref in manifest["source_parent"]["evidence"]: verify_hash(ref)
    verify_hash(freeze["data"])
    training_source.validate(freeze["binding"],OUTPUT,authenticate_parent=False)
    return freeze["binding"]

def event(stage, **values):
    global REPORTER
    if REPORTER is None or REPORTER.folder != OUTPUT:
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


def abort(exc,base=None):
    base = OUTPUT if base is None else base
    try: old.save_new(Path(base)/"abort.json",{"time":old.now(),"pid":os.getpid(),"error":str(exc)})
    except FileExistsError: pass


def run_job(stage,kind,argument,binding,base):
    try:
        budget_boundary(base)
        if stage == "learning":
            if kind == "fits":
                result = learning.fit(argument,binding,base)
            else:
                require(kind == "fit_audits", "Unknown learning stage")
                result = learning.audit_fit(argument[0],argument[1],binding,base)
        elif stage.startswith("source_"):
            require(kind in ("collection", "audits"), "Unknown source stage")
            result = (source.collect(argument,binding,base) if kind == "collection"
                      else source.audit(argument[0],argument[1],binding,base))
        else:
            require(kind in ("collection", "audits"), "Unknown evaluation stage")
            result = (engine.collect(stage,argument,binding,base) if kind == "collection"
                      else verify.audit(stage,argument[0],argument[1],binding,base))
        verify_hash(result)
        old.save_new(Path(base)/"completed_jobs"/f"{stage}_{kind}"/f"{result['index']:06d}.json",
                     {"manifest":binding,"stage":stage,"kind":kind,"time":old.now(),
                      "argument_sha256":old.digest(argument),"artifact":result})
        return result
    except BaseException as exc:
        abort(exc,base)
        raise


def dispatch(pool,arguments,job,workers,on_result,boundary):
    iterator,pending,files = iter(arguments),set(),[]
    def submit():
        arg = next(iterator,None)
        if arg is not None: pending.add(pool.submit(job,arg))
    for _ in range(min(workers,len(arguments))): submit()
    try:
        while pending:
            done,_ = wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            boundary()
            batch=[]
            for future in done:
                pending.remove(future); batch.append(future.result())
            files.extend(batch)
            if batch: on_result(len(files))
            for _ in batch: submit()
    except BaseException:
        for future in pending: future.cancel()
        raise
    return files

def run_pool(stage,kind,arguments,binding,upstream):
    require(validate_freeze() == binding, "Frozen identity changed before stage")
    budget_boundary(OUTPUT)
    started=time.perf_counter()
    launch_path=OUTPUT/f"{stage}_{kind}_launch.json"
    old.save_new(launch_path,{"manifest":binding,"stage":stage,"kind":kind,"time":old.now(),
                 "arguments_sha256":old.digest(arguments),"dependencies":[artifact(p) for p in upstream]})
    runtime=windows.configure_spawn()
    event(f"{stage}_{kind}",completed=0,total=len(arguments))
    workers=CONFIG["fit_workers"] if stage == "learning" else CONFIG["workers"]
    try:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_initialize,initargs=(stage,kind,binding,OUTPUT)) as pool:
            try:
                files=dispatch(pool,arguments,partial(run_job,stage,kind,binding=binding,base=OUTPUT),workers,
                               lambda n:event(f"{stage}_{kind}",completed=n,total=len(arguments)),lambda:budget_boundary(OUTPUT))
            except BaseException as exc:
                abort(exc,OUTPUT)
                raise
    except BaseException as exc:
        abort(exc,OUTPUT); raise
    files.sort(key=lambda item:item["index"])
    require([x["index"] for x in files] == list(range(len(arguments))), "Missing/duplicate job")
    for item in files: verify_hash(item)
    old.save_new(OUTPUT/f"{stage}_{kind}_lock.json",{"manifest":binding,"launch":artifact(launch_path),
                 "files":files,"elapsed_seconds":time.perf_counter()-started,"runtime":runtime})
    return files

def stage_lock(stage, kind, binding):
    lock = old.read_json(OUTPUT / f"{stage}_{kind}_lock.json")
    count = (len(FITS) if stage == "learning" else CONFIG["source_games"][stage[7:]]
             if stage.startswith("source_") else CONFIG["states"][stage])
    require(lock["manifest"] == binding and [x["index"] for x in lock["files"]] == list(range(count)), "Incomplete stage lock")
    launch = verified(lock["launch"])
    require(launch["manifest"] == binding and launch["stage"] == stage and launch["kind"] == kind, "Stage launch changed")
    for item in [*lock["files"], *launch["dependencies"]]:
        verify_hash(item)
    return lock


def build_learning(binding):
    event("learning_data_validation")
    training_source.validate(binding,OUTPUT,authenticate_parent=False)
    data=artifact(OUTPUT/"data_lock.json")
    event("training_feature_audit", completed=0,total=CONFIG["train_rows"]+CONFIG["added_rows"])
    features=training_source.audit_features(binding,OUTPUT)
    verify_hash(features)
    upstream=[OUTPUT/"freeze.json",OUTPUT/"data_lock.json",ROOT/features["path"]]
    fits=run_pool("learning","fits",list(FITS),binding,upstream)
    audits=run_pool("learning","fit_audits",list(zip(FITS,fits)),binding,
                    [*upstream,OUTPUT/"learning_fits_lock.json"])
    models=learning.models_lock(fits,audits,binding,OUTPUT)
    verify_hash(models); learning.validate_models(binding,OUTPUT)
    old.save_new(OUTPUT/"learning_complete.json",{"manifest":binding,"time":old.now(),"data":data,"models":models,
                 "features":features,"fits":artifact(OUTPUT/"learning_fits_lock.json"),"fit_audits":artifact(OUTPUT/"learning_fit_audits_lock.json")})

def build_sources(binding):
    learning.validate_models(binding, OUTPUT)
    schedule = old.read_json(OUTPUT / "schedule.json")
    collected, audited = {}, {}
    for stage in ("probe", "main"):
        upstream = [OUTPUT / "freeze.json", OUTPUT / "schedule.json", OUTPUT / "learning_complete.json", OUTPUT / "models_lock.json"]
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


def build_stage(stage,binding):
    selection=validate_selection(binding)
    learning.validate_models(binding,OUTPUT)
    upstream=[OUTPUT/"freeze.json",OUTPUT/"selection_seal.json",OUTPUT/"selection.json",OUTPUT/"models_lock.json",
              OUTPUT/"learning_complete.json",OUTPUT/stage/"audit_sample.json",OUTPUT/stage/"audit_sample_seal.json"]
    if stage == "main":
        gate=old.read_json(OUTPUT/"operational_decision.json")
        require(gate["manifest"] == binding and gate["reasonable"], "Passing operational forecast required")
        upstream += [OUTPUT/"operational_decision.json",OUTPUT/"probe_complete.json"]
    roots=run_pool(stage,"collection",selection[stage],binding,upstream)
    audits=run_pool(stage,"audits",list(zip(selection[stage],roots)),binding,[*upstream,OUTPUT/f"{stage}_collection_lock.json"])
    summary=verify.summarize(audits,selection[stage],binding,OUTPUT)
    for spec,ref,auditref in zip(selection[stage],roots,audits,strict=True):
        receipt=verified(auditref)
        require(receipt["root"] == {k:ref[k] for k in ("path","sha256")}
                and receipt["spec"] == spec and receipt["recorded_continuations"] == ref["continuations"], "Stage root/audit linkage")
        verify.compare(receipt["row"],engine.statistics_row(verified(ref)))
    old.save_new(OUTPUT/f"{stage}_complete.json",{"manifest":binding,"time":old.now(),"states":len(roots),
                 "all_prescribed_audits_pass":True,"coverage":summary["coverage"],"full_inner_replay":summary["full_inner_replay"],
                 "collection":artifact(OUTPUT/f"{stage}_collection_lock.json"),"audits":artifact(OUTPUT/f"{stage}_audits_lock.json")})

def cost_projection(collection,replay,counts,main_counts,main_replays,structural,audit_structural,size,free,elapsed):
    values=[*collection.values(),*replay.values(),*counts.values(),*main_counts.values(),*main_replays.values(),
            structural,audit_structural,size,free,elapsed]
    require(all(type(v) in (int,float) and np.isfinite(v) and v >= 0 for v in values), "Invalid timing/storage evidence")
    require(all(counts[b] > 0 and collection[b] > 0 and replay[b] > 0 for b in BLOCKS), "Unmeasured probe blocks")
    rates={b:collection[b]/counts[b] for b in BLOCKS}
    audit_rates={b:replay[b]/counts[b] for b in BLOCKS}
    work=(sum(rates[b]*main_counts[b]+audit_rates[b]*main_replays[b] for b in BLOCKS)
          +20*(structural+audit_structural))/CONFIG["workers"]
    remaining=CONFIG["forecast_allowance"]*work+CONFIG["finalization_allowance_seconds"]
    projected=int(size/sum(counts.values())*sum(main_counts.values())*1.5+512*1024**2)
    return {"reasonable":elapsed+remaining <= CONFIG["maximum_runtime_hours"]*3600 and free-projected >= CONFIG["disk_reserve_bytes"],
            "remaining_seconds":remaining,"remaining_hours":remaining/3600,"elapsed_seconds":elapsed,
            "projected_bytes":projected,"free_bytes":free,"collection_seconds_per_continuation":rates,
            "replay_seconds_per_continuation":audit_rates,"main_continuations_upper_bound":main_counts,
            "main_prescribed_replay_continuations_upper_bound":main_replays,
            "extra_unresolved_replay_not_in_projection":True,"probe_continuations":counts}

def operational_decision(binding):
    complete=old.read_json(OUTPUT/"probe_complete.json")
    require(complete["all_prescribed_audits_pass"] and complete["full_inner_replay"], "Fully checked probe required")
    roots=[verified(r) for r in stage_lock("probe","collection",binding)["files"]]
    audits=[verified(r) for r in stage_lock("probe","audits",binding)["files"]]
    collection={b:sum(r[b]["timing"]["wall"] for r in roots) for b in BLOCKS}
    replay={b:sum(r["timing"][b+"_replay_wall"] for r in audits) for b in BLOCKS}
    counts={b:sum(r[f"recorded_{b}_continuations"] for r in audits) for b in BLOCKS}
    specs=old.read_json(OUTPUT/"selection.json")["main"]
    plan=old.read_json(OUTPUT/"main/audit_sample.json")
    main_counts,main_replays=dict.fromkeys(BLOCKS,0),dict.fromkeys(BLOCKS,0)
    for spec,sample in zip(specs,plan["selections"],strict=True):
        for block,limit in (("selection",CONFIG["maximum_candidates"]),("assessment",CONFIG["maximum_assessed_choices"])):
            support=min(spec["canonical_support"],limit)
            main_counts[block]+=128*support
            main_replays[block]+=support*len(sample["replay"][block])
    value=cost_projection(collection,replay,counts,main_counts,main_replays,
        max(0.,sum(r["timing"]["wall"] for r in roots)-sum(collection.values())),
        sum(r["timing"]["structural_wall"] for r in audits),
        sum(p.stat().st_size for p in (OUTPUT/"probe").rglob("*") if p.is_file()),
        shutil.disk_usage(ROOT).free,c.active_seconds(OUTPUT))
    old.save_new(OUTPUT/"operational_decision.json",{"manifest":binding,"time":old.now(),**value})
    if not value["reasonable"]: raise BudgetExceeded("Outcome-blind forecast exceeds twenty-four-hour active execution or storage budget")
    event("probe_passed",remaining_forecast_hours=value["remaining_hours"])
    return True

def independent_statistics(rows):
    return statistics.summarize(rows)

def finish(binding):
    budget_boundary(OUTPUT)
    require(validate_freeze(full_sources=True) == binding, "Frozen sources changed at finalization")
    learning.validate_models(binding,OUTPUT)
    specs=validate_selection(binding)["main"]
    summary=verify.summarize(stage_lock("main","audits",binding)["files"],specs,binding,OUTPUT)
    rawrows=[engine.statistics_row(verified(r)) for r in stage_lock("main","collection",binding)["files"]]
    verify.compare(rawrows,summary["rows"])
    actual=verify.statistics(summary["rows"])
    independent=independent_statistics(rawrows)
    for name,row in independent["contrasts"].items():
        for key in ("mean_score_difference","mean_bounds","ci95"):
            verify.compare(row[key],actual["contrasts"][name][key])
    for key in ("anchor_practical_gain","anchor_superiority","raw_superiority","relations_attribution","gate_pass","classification","unknown_outcomes_imputed"):
        require(independent[key] == actual[key], "Independent decision differs: "+key)
    verify.compare(independent["per_fit"], actual["per_fit"])
    label=actual["classification"]
    old.save_new(OUTPUT/"result.json",{"manifest":binding,"time":old.now(),**summary,"statistics":actual,
                 "classification":label,"all_prescribed_audits_pass":True,"full_inner_replay":False,
                 "learning":artifact(OUTPUT/"learning_complete.json"),"selection":artifact(OUTPUT/"selection_seal.json"),
                 "scope":"Fresh fast161-source positions; greedy035 continuation; conditional decisions, not whole-game strength",
                 "automatic_successor":False})
    old.save_new(OUTPUT/"final_audit.json",{"manifest":binding,"time":old.now(),"passed":True,
                 "result":artifact(OUTPUT/"result.json"),"independent_statistics":independent,
                 "stages":{s:artifact(OUTPUT/f"{s}_complete.json") for s in ("probe","main")},
                 "learning":artifact(OUTPUT/"learning_complete.json"),"coverage":summary["coverage"],"full_inner_replay":False})
    lines=["# Matched action-card relations experiment","",f"Classification: {label}.","",
           "360 fresh positions from distinct games; six new fits, all three paired initializers retained. Conditional score win1/draw.5/loss0, not arena win rates.","",
           "| Contrast | Mean bounds (percentage points) | Paired95% interval (percentage points) |","|---|---:|---:|"]
    for name,row in actual["contrasts"].items():
        lines.append(f"| {name} | [{100*row['mean_bounds'][0]:.4f}, {100*row['mean_bounds'][1]:.4f}] | [{100*row['ci95'][0]:.4f}, {100*row['ci95'][1]:.4f}] |")
    lines += ["", "Joint gate: relations-anchor mean lower >=+0.5pp and paired95% lower>0, plus relations-raw paired95% lower>0.",
              f"Explicit-relations attribution (relations-raw lower>0): {actual['relations_attribution']}. Shared gains without this contrast support the common residual package only.",
              "", "Training trajectories were independently reproduced. Both arms receive the exact same R1 BROAD examples, targets, batches and1784 updates. All base parameters stay frozen; only the matched correction modules train. The720 previous test roots are permanently retired from evaluation.",
              "", "All fresh source games and proposals checked; full18-root probe, prespecified8/128 world samples, metadata-selected main stress roots and every unknown branch replayed. Mature bounded repetition handling is inherited; unknowns are not imputed. Main continuation replay is not exhaustive.",
              "", "No checkpoint/subset/fit selection, automatic arena, promotion, extension, successor or monitor.",""]
    with (OUTPUT/"report.md").open("x",encoding="utf-8",newline="\n") as handle: handle.write("\n".join(lines))
    budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json",{"manifest":binding,"time":old.now(),"complete":True,
                 "classification":label,"relations_attribution":actual["relations_attribution"],
                 "all_prescribed_audits_pass":True,"full_inner_replay":False,
                 "result":artifact(OUTPUT/"result.json"),"audit":artifact(OUTPUT/"final_audit.json"),
                 "report":artifact(OUTPUT/"report.md"),"automatic_successor":False})
    event("complete",classification=label,relations_attribution=actual["relations_attribution"])

def run():
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    windows.assert_windowless_worker(folder / "stdout.txt", folder / "stderr.txt")
    with old.process_lock(OUTPUT / "run.lock"):
        require(not (OUTPUT / "launch.json").exists(), "One-shot run; preserve existing evidence")
        binding = validate_freeze(full_sources=True)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        try:
            with parent.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT / "launch.json", {"manifest": binding, "time": old.now(),
                             "epoch_seconds": time.time(), "pid": os.getpid(), "sleep_inhibition": sleep,
                             "windowless": windows.assert_windowless_worker(), "owner_authorized": True})
                build_learning(binding)
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
        except BaseException as exc:
            abort(exc,OUTPUT)
            old.save_new(OUTPUT / "failure.json", {"manifest": binding, "time": old.now(), "traceback": traceback.format_exc()})
            event("failed", failure="failure.json")
            raise


def launch():
    assert_prelaunch_clean()
    binding = validate_freeze(full_sources=True)
    folder = OUTPUT / "runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with old.process_lock(folder / "launcher.lock"):
        require(not (folder / "process.json").exists() and not (OUTPUT / "launch.json").exists(), "Already launched")
        process = windows.detached_launch([sys.executable, "-u", ROOT / "run_action_card_relations_128.py", "run"],
                                          folder / "stdout.txt", folder / "stderr.txt", ROOT)
        old.save_new(folder / "process.json", {**process, "manifest": binding, "time": old.now()})
        deadline = time.monotonic()+600
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
