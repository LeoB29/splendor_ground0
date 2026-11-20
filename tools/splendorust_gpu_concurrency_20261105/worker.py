"""Fresh evaluator process for qualification or one predetermined concurrency cell."""
from pathlib import Path
import argparse
import os
import time
import traceback

from . import common as c


def validate_contract(plan, mode, cell_id):
    runtime = plan["runtime"]
    c.require(plan["schema"] == "splendorust_independent_concurrency_v1" and mode in ("qualification", "cell") and
              runtime["gpu_physical_batch"] == 8 and runtime["live_callback_batch"] == 1 and
              runtime["maximum_batch_wait_seconds"] == 0. and runtime["internal_seconds"] == 9. and
              runtime["outer_seconds"] == 10. and plan["group_gap_seconds"] == 0., "unchanged immediate-dispatch scientific contract")
    c.require(plan["worker_limits"] == dict(qualification_seconds=1200,cell_seconds=300)
              and plan["worker_affinity"] == list(range(19)), "prospective worker clock and affinity")
    c.require(plan["warmup"] == dict(calls_per_cell=5,case_index=0,client_id="warmup",slot=0,
        actual_service=True,inside_measured_panel=False,exact_within_route=True,
        internal_seconds=9.,outer_seconds=10.,failure_stops_study=True), "fixed predeclared service warmup")
    if mode == "qualification":
        c.require(cell_id is None, "qualification has no diagnostic cell")
        return None
    matches = [row for row in plan["cells"] if row["cell_id"] == cell_id]
    c.require(len(matches) == 1, "exact prospectively frozen independent cell")
    return matches[0]


def run(plan, folder, mode, cell_id=None):
    spec = validate_contract(plan,mode,cell_id)
    folder = Path(folder).resolve(); folder.mkdir(parents=True,exist_ok=True)
    c.require(not (folder/"runtime_sources.json").exists(), "fresh evaluator payload; no retries")
    began, wall = time.monotonic(), time.time()
    limit = plan["worker_limits"][mode+"_seconds"]
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()  # Exactly once before any model in this process.
    runtime_plan = plan["runtime"]
    c.require(sources == runtime_plan["source_guard"], "frozen corrected runtime source receipt")
    c.save(folder/"runtime_sources.json",dict(sources=sources,ruleset_id=runtime_plan["ruleset_id"],
        catalogue_sha256=runtime_plan["catalogue_sha256"],turn_contract=runtime_plan["turn_contract"]))
    import psutil
    process = psutil.Process(); process.cpu_affinity(plan["worker_affinity"])
    c.save(folder/"worker_identity.json",dict(pid=os.getpid(),created=process.create_time(),mode=mode,
        cell_id=cell_id,worker_affinity=process.cpu_affinity(),started_monotonic=began,started_unix=wall))
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from tools.splendorust_gpu_live_20261105_03 import backend, qualification
    from . import cell, stability
    runtimes = []; cleanup_errors = []; error = None; result = None

    def boundary():
        c.require(time.monotonic()-began < limit and time.time()-wall < limit, "Actual evaluator dual-clock bound")

    def stage(name):
        boundary(); c.atomic_json(folder/"stage.json",dict(stage=name,time=c.now(),pid=os.getpid(),mode=mode,cell_id=cell_id))

    def build(route, destination):
        stage("initialize_"+route)
        runtime = backend.build_runtime(runtime_plan,route,destination,runtime_sources=sources)
        runtimes.append(runtime)
        return runtime

    try:
        if mode == "qualification":
            cpu = build("cpu",folder/"cpu"); gpu = build("gpu",folder/"gpu")
            stage("packing_and_concurrency_qualification")
            packing = stability.qualify(runtime_plan,cpu,gpu,folder/"stability",boundary)
            stage("cpu_actual_entrypoint_and_fixed_roots")
            left = qualification.run_qualification(runtime_plan,cpu,folder/"qualification_cpu",boundary)
            stage("gpu_actual_entrypoint_and_fixed_roots")
            right = qualification.run_qualification(runtime_plan,gpu,folder/"qualification_gpu",boundary)
            stage("compare_fixed_root_decisions")
            comparison = qualification.compare_qualification(left,right,runtime_plan)
            c.save(folder/"decision_comparison.json",comparison)
            result = dict(status="qualification_passed",packing=packing,cpu=left["summary"],gpu=right["summary"],
                          decision_comparison=comparison,diagnostic_decisions=0,game_continuations=0)
        else:
            cell.validate_cases(plan)
            runtime = build(spec["route"],folder/"runtime")
            stage("five_fixed_service_warmup_calls")
            cell.warmup_runtime(plan,runtime,folder)
            stage("fixed_eight_case_cell")
            result = cell.run_cell(plan,spec,runtime,folder/"cell",boundary)
    except BaseException:
        error = traceback.format_exc()
        c.save(folder/"worker_failure.json",dict(error=error,time=c.now(),mode=mode,cell_id=cell_id,no_retry=True))
        for runtime in runtimes:
            try:runtime.abort()
            except BaseException as caught:cleanup_errors.append(repr(caught))
    finally:
        for runtime in runtimes:
            try:
                runtime.close()
                c.require(not runtime.active and not runtime.service.pending and not runtime.service.clients,
                          "all owned operations and callbacks closed")
                c.save(runtime.folder/"final_identity.json",dict(time=c.now(),identity=runtime.identity,
                    weights_verified_unchanged=True,service_closed=True,active_operations=runtime.active,
                    completed_requests=runtime.service.completed_requests,physical_batches=runtime.service.batch_count))
            except BaseException as caught:
                cleanup_errors.append(repr(caught))
        c.save(folder/"cleanup.json",dict(time=c.now(),success=not cleanup_errors,errors=cleanup_errors))
    if error or cleanup_errors:
        return 1
    try:
        boundary(); c.verify(plan["inputs"])
        c.require(result is not None, "explicit completed evaluator outcome")
        if mode == "qualification":
            exit_code = 0
        else:
            exit_code = {"passed":0,"deadline_failed":20,"integrity_failed":1}[result["status"]]
        c.save(folder/"worker_result.json",dict(mode=mode,cell_id=cell_id,status=result["status"],result=result,
            exit_code=exit_code,no_retry=True,reference_weights_unchanged=True,training_updates=0,game_continuations=0))
        stage("completed_pending_independent_acceptance" if exit_code == 0 else "preserved_failed_cell")
        return exit_code
    except BaseException:
        c.save(folder/"postrun_failure.json",dict(error=traceback.format_exc(),time=c.now(),no_retry=True))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",required=True,choices=("qualification","cell"))
    parser.add_argument("--cell-id")
    parser.add_argument("--plan",required=True)
    parser.add_argument("--plan-sha256",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    c.require(c.sha(args.plan) == args.plan_sha256, "exact frozen plan before model imports")
    plan = c.read(args.plan); c.verify(plan["inputs"])
    validate_contract(plan,args.mode,args.cell_id)
    return run(plan,Path(args.output),args.mode,args.cell_id)


if __name__ == "__main__":
    raise SystemExit(main())
