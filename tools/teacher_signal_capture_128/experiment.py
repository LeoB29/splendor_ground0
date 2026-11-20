"""Freeze, run and verify the fixed no-training teacher-signal capture study."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
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
from . import common as c, models, source, engine, verify, statistics
from tools.proposal_ranking_128_recovery_01.reporting import StatusReporter
from tools.proposal_iteration_128 import experiment as parent

ROOT, OUTPUT, old = c.ROOT, c.OUTPUT, c.old
CHECK_TARGETS = [f"tools/teacher_signal_capture_128/test_{name}.py" for name in
                 ("models","source","engine","verify","integration","experiment","worker")]
EXTRAS = ["run_teacher_signal_capture_128.py","docs/engineering/teacher_signal_capture_128.md",
          "docs/engineering/verification_policy.md"]
ANCESTORS = [c.LEARNING/"manifest.json",c.SOURCE/"manifest.json",
             ROOT/"outputs/planner_state_transfer_128/manifest.json"]
REPORTER = None
BLOCKS = ("assessment",)

def identity():
    c.require(sys.flags.optimize == 0,"Independent assertions require unoptimized Python")
    dependencies = {}
    for path in ANCESTORS:
        ancestor = old.read_json(path)["identity"]
        for name, expected in ancestor["dependencies"].items():
            c.require(old.file_hash(ROOT/name) == expected,"Frozen ancestor dependency changed: "+name)
            c.require(name not in dependencies or dependencies[name] == expected,"Conflicting ancestor dependency")
            dependencies[name] = expected
    paths = list((ROOT/"tools/teacher_signal_capture_128").glob("*.py")) + [ROOT/p for p in EXTRAS+CHECK_TARGETS]
    dependencies.update({p.relative_to(ROOT).as_posix():old.file_hash(p) for p in paths})
    return {"config":c.CONFIG,"fits":list(c.FITS),"checks":CHECK_TARGETS,"dependencies":dict(sorted(dependencies.items())),
            "python":sys.version,"optimize":sys.flags.optimize,"numpy":np.__version__,"torch":str(torch.__version__),
            "windowless":c.windows._identity(c.windows._windowless_executable())}

def check(folder):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    before,start = identity(),time.perf_counter()
    fixture_folder = ROOT/"outputs/teacher_signal_capture_128_fixtures"/folder.name
    c.require(not fixture_folder.exists(), "Preflight fixture directory already exists")
    fixture_folder.parent.mkdir(parents=True, exist_ok=True)
    code = "import torch,pytest;torch.set_num_threads(1);raise SystemExit(pytest.main("+repr(
        ["-q", *CHECK_TARGETS, "-p", "no:cacheprovider", "--basetemp", str(fixture_folder)])+"))"
    with (folder/"stdout.txt").open("x",encoding="utf-8") as out,(folder/"stderr.txt").open("x",encoding="utf-8") as err:
        proc = subprocess.run([sys.executable,"-u","-c",code],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                              timeout=600, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    c.require(identity() == before,"Code changed during preflight")
    old.save_new(folder/"checks.json",{"identity":before,"targets":CHECK_TARGETS,"exit_code":proc.returncode,
                 "seconds":time.perf_counter()-start,"logs":[c.artifact(folder/n) for n in ("stdout.txt","stderr.txt")]})
    return proc.returncode

def prepare(checks_path):
    c.require(not (OUTPUT/"manifest.json").exists(),"Already frozen; preserve evidence")
    checks = old.read_json(checks_path)
    c.require(checks["exit_code"] == 0 and checks["targets"] == CHECK_TARGETS and checks["identity"] == identity(),
              "Current full passing preflight required")
    for ref in checks["logs"]: c.verify_hash(ref)
    model_parent = models.closed_parent()
    selected = source.build()
    schedule = selected["schedule"]
    worldset = [seed for stage in ("probe","main") for spec in schedule[stage] for block in BLOCKS
                for seed in engine.world_seeds(stage,spec["index"],block)]
    c.require(len(worldset) == len(set(worldset)) == 354*128,"Fresh world namespaces overlap")
    c.require(set(worldset).isdisjoint(s["environment_seed"] for specs in schedule.values() for s in specs),"Source/continuation namespace overlap")
    old_seeds = {seed for specs in schedule.values() for spec in specs
                 for seed in c.verified(spec["source_root"])["schedule"]["world_seeds"]}
    c.require(set(worldset).isdisjoint(old_seeds), "New worlds overlap original label worlds")
    c.require(selected["training_data"] == model_parent["training_data"], "Source/model training corpus differs")
    old.save_new(OUTPUT/"source_seal.json",selected)
    old.save_new(OUTPUT/"schedule.json",schedule)
    manifest = {"identity":identity(),"source":c.artifact(OUTPUT/"source_seal.json"),
                "schedule":c.artifact(OUTPUT/"schedule.json"),"model_parent":model_parent,
                "checks":c.artifact(checks_path),"owner_authorized":True,
                "authorization":"Owner accepted teacher-signal and frozen-learner capture diagnosis on2026-10-08. Fresh assessment only; no training, source games, promotion, arena or successor."}
    old.save_new(OUTPUT/"manifest.json",manifest)
    binding = old.digest(manifest)
    old.save_new(OUTPUT/"freeze.json",{"binding":binding,"manifest":c.artifact(OUTPUT/"manifest.json"),"time":old.now(),"owner_authorized":True})
    models.prepare(binding,OUTPUT)
    for stage in ("probe","main"):
        old.save_new(OUTPUT/stage/"audit_sample.json",verify.sample_plan(schedule[stage],stage,binding))
    c.require(validate_freeze() == binding,"Post-freeze verification failed")
    print({"prepared":True,"binding":binding,"main_states":336,"pairs":84},flush=True)

def validate_freeze():
    freeze = old.read_json(OUTPUT/"freeze.json")
    manifest = c.verified(freeze["manifest"])
    c.require(freeze["owner_authorized"] and old.digest(manifest) == freeze["binding"] and manifest["identity"] == identity(),"Freeze/code identity changed")
    checks = c.verified(manifest["checks"])
    c.require(checks["exit_code"] == 0 and checks["identity"] == manifest["identity"],"Preflight changed")
    selected,schedule = c.verified(manifest["source"]),c.verified(manifest["schedule"])
    c.require(selected["schedule"] == schedule,"Source/schedule seal changed")
    source.validate_schedule(schedule)
    for ref in [*checks["logs"],*manifest["model_parent"]["evidence"],*selected["source_evidence"]]: c.verify_hash(ref)
    models.validate(freeze["binding"],OUTPUT)
    for stage in ("probe","main"):
        c.require(old.read_json(OUTPUT/stage/"audit_sample.json") == verify.sample_plan(schedule[stage],stage,freeze["binding"]),"Audit sample changed")
    return freeze["binding"]

def event(stage,**values):
    global REPORTER
    if REPORTER is None: REPORTER = StatusReporter(OUTPUT)
    return REPORTER.event(stage,**values)

def worker_initialize(stage,kind,binding,base):
    folder = Path(base)/"workers"/f"{stage}_{kind}"
    folder.mkdir(parents=True,exist_ok=True)
    health = c.windows.assert_windowless_worker(folder/f"{os.getpid()}_stdout.txt",folder/f"{os.getpid()}_stderr.txt")
    old.initialize_worker()
    engine.initialize_models(base)
    c.require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(),"Worker arithmetic differs")
    old.save_new(folder/f"{os.getpid()}_health.json",{"manifest":binding,"stage":stage,"kind":kind,**health})

def abort(exc,base=OUTPUT):
    path = Path(base)/"abort.json"
    if not path.exists(): old.save_new(path,{"time":old.now(),"type":type(exc).__name__,"reason":str(exc)})

def run_job(stage,kind,arg,binding,base):
    try:
        c.budget_boundary(base)
        return engine.collect(stage,arg,binding,base) if kind == "collection" else verify.audit(stage,arg[0],arg[1],binding,base)
    except BaseException as exc:
        abort(exc,base)
        raise

def dispatch(pool,arguments,job,workers,on_result,boundary):
    """At most workers tasks in flight; no unbounded queued work after failure."""
    iterator,pending,files = iter(arguments),set(),[]
    def submit():
        arg = next(iterator,None)
        if arg is not None: pending.add(pool.submit(job,arg))
    for _ in range(min(workers,len(arguments))): submit()
    try:
        while pending:
            done,_ = wait(pending,return_when=FIRST_COMPLETED)
            batch = []
            for future in done:
                pending.remove(future)
                batch.append(future.result())
            files.extend(batch)
            boundary()
            on_result(len(files))
            for _ in batch: submit()
    except BaseException:
        for future in pending: future.cancel()
        raise
    return files

def run_pool(stage,kind,args,binding):
    from functools import partial
    c.require(validate_freeze() == binding,"Stage freeze changed")
    c.budget_boundary(OUTPUT)
    launch = OUTPUT/f"{stage}_{kind}_launch.json"
    old.save_new(launch,{"manifest":binding,"stage":stage,"kind":kind,"time":old.now(),"arguments_sha256":old.digest(args)})
    event(f"{stage}_{kind}",completed=0,total=len(args))
    runtime,start = c.windows.configure_spawn(),time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=min(c.CONFIG["workers"],len(args)),mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_initialize,initargs=(stage,kind,binding,OUTPUT)) as pool:
            job = partial(run_job,stage,kind,binding=binding,base=OUTPUT)
            files = dispatch(pool,args,job,c.CONFIG["workers"],
                             lambda n:event(f"{stage}_{kind}",completed=n,total=len(args)),lambda:c.budget_boundary(OUTPUT))
    except BaseException as exc:
        abort(exc)
        raise
    files.sort(key=lambda r:r["index"])
    c.require([r["index"] for r in files] == list(range(len(args))),"Stage incomplete")
    for ref in files: c.verify_hash(ref)
    old.save_new(OUTPUT/f"{stage}_{kind}_lock.json",{"manifest":binding,"files":files,"runtime":runtime,
                 "launch":c.artifact(launch),"elapsed_seconds":time.perf_counter()-start})
    return files

def stage_lock(stage,kind,binding):
    value = old.read_json(OUTPUT/f"{stage}_{kind}_lock.json")
    c.require(value["manifest"] == binding and [r["index"] for r in value["files"]] == list(range(c.CONFIG["states"][stage])),"Stage seal changed")
    c.verify_hash(value["launch"])
    for ref in value["files"]: c.verify_hash(ref)
    return value

def build_stage(stage,binding):
    specs = old.read_json(OUTPUT/"schedule.json")[stage]
    if stage == "main":
        gate = old.read_json(OUTPUT/"operational_decision.json")
        c.require(gate["manifest"] == binding and gate["reasonable"],"Operational probe must pass")
    roots = run_pool(stage,"collection",specs,binding)
    audits = run_pool(stage,"audits",list(zip(specs,roots)),binding)
    summary = verify.summarize(audits,specs,binding,OUTPUT)
    for spec,ref,auditref in zip(specs,roots,audits,strict=True):
        receipt = c.verified(auditref)
        c.require(receipt["root"] == {k:ref[k] for k in ("path","sha256")}
                  and receipt["spec"] == spec and receipt["recorded_continuations"] == ref["continuations"],"Stage audit/root linkage")
        verify.compare(receipt["row"],engine.statistics_row(c.verified(ref)))
    old.save_new(OUTPUT/f"{stage}_complete.json",{"manifest":binding,"time":old.now(),
                 "all_prescribed_audits_pass":True,"coverage":summary["coverage"],"full_inner_replay":summary["full_inner_replay"],
                 "collection":c.artifact(OUTPUT/f"{stage}_collection_lock.json"),"audits":c.artifact(OUTPUT/f"{stage}_audits_lock.json")})

def cost_projection(collection,replay,counts,main_counts,main_replays,structural,audit_structural,size,free,elapsed):
    values = [*collection.values(),*replay.values(),*counts.values(),*main_counts.values(),*main_replays.values(),structural,audit_structural,size,free,elapsed]
    c.require(all(np.isfinite(v) and v >= 0 for v in values),"Invalid timing or storage evidence")
    c.require(all(counts[b] > 0 and collection[b] > 0 and replay[b] > 0 for b in BLOCKS),"Unmeasured required blocks")
    rates = {b:collection[b]/counts[b] for b in BLOCKS}
    audit_rates = {b:replay[b]/counts[b] for b in BLOCKS}
    work = (sum(rates[b]*main_counts[b]+audit_rates[b]*main_replays[b] for b in BLOCKS)+(336/18)*(structural+audit_structural))/c.CONFIG["workers"]
    remaining = c.CONFIG["forecast_allowance"]*work+c.CONFIG["finalization_allowance_seconds"]
    projected = int(size/sum(counts.values())*sum(main_counts.values())*1.5+512*1024**2)
    return {"reasonable":elapsed+remaining <= c.CONFIG["maximum_runtime_hours"]*3600 and free-projected >= c.CONFIG["disk_reserve_bytes"],
            "remaining_seconds":remaining,"remaining_hours":remaining/3600,"elapsed_seconds":elapsed,
            "projected_bytes":projected,"free_bytes":free,"collection_seconds_per_continuation":rates,
            "replay_seconds_per_continuation":audit_rates,"main_continuations_upper_bound":main_counts,
            "main_replay_continuations_upper_bound":main_replays,"probe_continuations":counts}

def operational_decision(binding):
    complete = old.read_json(OUTPUT/"probe_complete.json")
    c.require(complete["all_prescribed_audits_pass"] and complete["full_inner_replay"],"Fully replayed probe required")
    roots = [c.verified(r) for r in stage_lock("probe","collection",binding)["files"]]
    audits = [c.verified(r) for r in stage_lock("probe","audits",binding)["files"]]
    collection = {b:sum(r[b]["timing"]["wall"] for r in roots) for b in BLOCKS}
    replay = {b:sum(r["timing"][b+"_replay_wall"] for r in audits) for b in BLOCKS}
    counts = {b:sum(r[f"recorded_{b}_continuations"] for r in audits) for b in BLOCKS}
    specs = old.read_json(OUTPUT/"schedule.json")["main"]
    plan = old.read_json(OUTPUT/"main/audit_sample.json")
    main_counts,main_replays = dict.fromkeys(BLOCKS,0),dict.fromkeys(BLOCKS,0)
    for spec,sample in zip(specs,plan["selections"],strict=True):
        for b,limit in (("assessment",8),):
            support = min(spec["canonical_support"],limit)
            main_counts[b] += support*128
            main_replays[b] += support*len(sample["replay"][b])
    value = cost_projection(collection,replay,counts,main_counts,main_replays,
        max(0,sum(r["timing"]["wall"] for r in roots)-sum(collection.values())),
        sum(r["timing"]["structural_wall"] for r in audits),
        sum(p.stat().st_size for p in (OUTPUT/"probe").rglob("*") if p.is_file()),
        shutil.disk_usage(ROOT).free,time.time()-old.read_json(OUTPUT/"launch.json")["epoch_seconds"])
    old.save_new(OUTPUT/"operational_decision.json",{"manifest":binding,"time":old.now(),**value})
    if not value["reasonable"]: raise c.BudgetExceeded("Outcome-blind forecast exceeds eight-hour execution or storage budget")
    event("probe_passed",remaining_forecast_hours=value["remaining_hours"])

def finish(binding):
    c.require(validate_freeze() == binding,"Final freeze changed")
    specs = old.read_json(OUTPUT/"schedule.json")["main"]
    summary = verify.summarize(stage_lock("main","audits",binding)["files"],specs,binding,OUTPUT)
    rawrows = [engine.statistics_row(c.verified(r)) for r in stage_lock("main","collection",binding)["files"]]
    verify.compare(rawrows,summary["rows"])
    result = statistics.summarize(rawrows)
    independent = verify.independent_statistics(summary["rows"])
    verify.compare(result,independent)
    old.save_new(OUTPUT/"result.json",{"manifest":binding,"time":old.now(),**summary,"statistics":result,
                 "classification":result["classification"],"all_prescribed_audits_pass":True,
                 "models":c.artifact(OUTPUT/"models_lock.json"),"source":c.artifact(OUTPUT/"source_seal.json"),
                 "scope":"Fresh assessment of fixed choices on reused training/validation states; greedy035 continuation; diagnostic evidence, not whole-game strength or a unique causal mechanism",
                 "automatic_successor":False})
    old.save_new(OUTPUT/"final_audit.json",{"manifest":binding,"passed":True,"time":old.now(),
                 "result":c.artifact(OUTPUT/"result.json"),"independent_statistics":independent,
                 "coverage":summary["coverage"],"stages":{s:c.artifact(OUTPUT/f"{s}_complete.json") for s in ("probe","main")}})
    lines = ["# Teacher signal and frozen learner capture", "",
             f"Classification: {result['classification']}.", "",
             "336 positions / 84 whole source pairs: 42 training and 42 validation pairs, four roots per pair. All models and choices were fixed before independent128-world assessment. These are conditional decision scores, not game win rates.", "",
             "Training and validation are original development partitions; validation helped select checkpoints. Fresh assessment provides new outcome evidence without making these fresh states or proving a unique learning mechanism.", "",
             "Independent statistics, including train/validation contrasts and diagnostic labels:", "", "```json",
             __import__("json").dumps(result, indent=2, sort_keys=True), "```", "",
             "Only original128-world selection or frozen neural rankings choose scored actions. A new assessment argmax is diagnostic repeatability only and is never graded on the same worlds as a claimed improvement.", "",
             "Unknown outcomes remain bounded; none are imputed. Original source audits were inherited. All new proposal/math/world checks, ordinary8/128world replay samples, full probe and prespecified stress roots passed. Full main inner replay was not performed.", "",
             "No training, model selection, source games, promotion, extension, arena or automatic successor.", ""]
    with (OUTPUT/"report.md").open("x",encoding="utf-8",newline="\n") as handle: handle.write("\n".join(lines))
    c.budget_boundary(OUTPUT)
    old.save_new(OUTPUT/"final_status.json",{"manifest":binding,"time":old.now(),"complete":True,
                 "all_prescribed_audits_pass":True,"classification":result["classification"],
                 "result":c.artifact(OUTPUT/"result.json"),"audit":c.artifact(OUTPUT/"final_audit.json"),
                 "report":c.artifact(OUTPUT/"report.md"),"automatic_successor":False})
    event("complete",classification=result["classification"])

def run():
    global REPORTER
    folder = OUTPUT/"runtime"
    folder.mkdir(parents=True,exist_ok=True)
    c.windows.assert_windowless_worker(folder/"stdout.txt",folder/"stderr.txt")
    with old.process_lock(OUTPUT/"run.lock"):
        c.require(not (OUTPUT/"launch.json").exists(),"One-shot run; preserve previous evidence")
        binding = validate_freeze()
        REPORTER = StatusReporter(OUTPUT,{"manifest":binding})
        try:
            with parent.parent.inhibit_system_sleep() as sleep:
                old.save_new(OUTPUT/"launch.json",{"manifest":binding,"time":old.now(),"epoch_seconds":time.time(),
                             "pid":os.getpid(),"sleep_inhibition":sleep,"windowless":c.windows.assert_windowless_worker(),"owner_authorized":True})
                c.budget_boundary(OUTPUT)
                build_stage("probe",binding)
                operational_decision(binding)
                build_stage("main",binding)
                finish(binding)
        except (c.BudgetExceeded,engine.AmbiguousDecision) as exc:
            old.save_new(OUTPUT/"operational_hold.json",{"manifest":binding,"time":old.now(),"reason":str(exc)})
            event("operational_hold",reason=str(exc))
        except BaseException:
            old.save_new(OUTPUT/"failure.json",{"manifest":binding,"time":old.now(),"traceback":traceback.format_exc()})
            event("failed",failure="failure.json")
            raise

def launch():
    binding = validate_freeze()
    folder = OUTPUT/"runtime"
    folder.mkdir(parents=True,exist_ok=True)
    with old.process_lock(folder/"launcher.lock"):
        c.require(not (folder/"process.json").exists() and not (OUTPUT/"launch.json").exists(),"Already launched")
        process = c.windows.detached_launch([sys.executable,"-u",ROOT/"run_teacher_signal_capture_128.py","run"],folder/"stdout.txt",folder/"stderr.txt",ROOT)
        old.save_new(folder/"process.json",{**process,"manifest":binding,"time":old.now()})
        deadline = time.monotonic()+60
        while not (OUTPUT/"launch.json").exists():
            c.require(time.monotonic() < deadline,"No acknowledgement; inspect before retry")
            time.sleep(.05)
        receipt = old.read_json(OUTPUT/"launch.json")
        c.require(receipt["pid"] == process["pid"] and receipt["manifest"] == binding,"Supervisor mismatch")
        old.save_new(folder/"acknowledged.json",{"time":old.now(),"pid":process["pid"],"launch":c.artifact(OUTPUT/"launch.json")})
        print({"launched":True,"pid":process["pid"],"binding":binding},flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("check","prepare","launch","run"))
    parser.add_argument("--checks",type=Path)
    args = parser.parse_args()
    if args.command == "check":
        c.require(args.checks is not None,"New check folder required")
        return check(args.checks)
    if args.command == "prepare":
        c.require(args.checks is not None,"checks.json required")
        prepare(args.checks)
    elif args.command == "launch": launch()
    else: run()
    return 0
