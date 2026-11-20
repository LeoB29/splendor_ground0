"""Fresh bounded collector workers; qualified route-specific target capture."""
from pathlib import Path
import argparse
import os
import time
import traceback

from . import common as c


def initialize_runtime(operation, destination, route, implementation):
    """Complete runtime preparation, including GPU compilation; outside collection."""
    destination = Path(destination).resolve()
    started, wall = time.perf_counter(), time.time(); error = None
    try:
        return operation()
    except BaseException:
        error = traceback.format_exc(); raise
    finally:
        ended, endwall = time.perf_counter(), time.time()
        c.save(destination.parent/"initialization"/(destination.name+".json"), dict(
            schema="collector_runtime_initialization_v1", route=route, implementation=implementation,
            runtime_folder=str(destination), status="completed" if error is None else "failed", error=error,
            started_perf_counter=started, ended_perf_counter=ended, elapsed_seconds=ended-started,
            started_unix=wall, ended_unix=endwall, no_retry=True, inside_collection_timing=False,
            scope="Complete runtime preparation including model load and compiled graph checks; not isolated compilation"))


def validate_contract(plan, mode, cell_id):
    from . import design, collector, qualification
    runtime = plan["runtime"]
    c.require(plan["schema"] == "splendorust_compiled_gpu_collector_v1" and mode in ("qualification", "games"),
              "explicit frozen collector worker mode")
    c.require(runtime["gpu_physical_batch"] == 8 and runtime["live_callback_batch"] == 1
              and runtime["maximum_batch_wait_seconds"] == 0.
              and runtime["internal_seconds"] == plan["internal_seconds"] == c.INTERNAL_SECONDS
              and runtime["outer_seconds"] == plan["outer_seconds"] == c.OUTER_SECONDS
              and runtime["execution_policy_id"] == plan["execution_policy_id"] == c.EXECUTION_POLICY_ID,
              "unchanged physical graph, callback and original complete response clocks")
    c.require(plan["worker_limits"] == dict(qualification_seconds=3600, games_seconds=14400)
              and plan["worker_affinity"] == list(range(19)) and plan["supervisor_affinity"] == [19]
              and plan["model_seconds"] == c.MODEL_SECONDS and plan["total_seconds"] == c.TOTAL_SECONDS,
              "frozen actual worker/global allowances and CPU separation")
    c.require(plan["qualification"] == qualification.build_plan() and plan["games"] == collector.build_plan()
              and plan["cells"] == design.cells(plan["games"]), "fixed actual loader and matched game plans")
    c.require(plan["qualification_routes"] == ["cpu", "gpu"] and plan["qualification_decisions_per_route"] == 116
              and plan["qualification_fixture_trajectories"] == plan["qualification_extra_continuations"] == 2,
              "both actual collector/loader routes and official fixture continuations")
    c.require(plan["maximum_new_games"] == 64 and plan["independent_setup_seeds"] == 32
              and plan["maximum_model_replays"] == 192
              and plan["retries"] == plan["replacements"] == plan["training_updates"] == 0
              and plan["timing_observer"] is False and plan["available_ram_reserve_bytes"] == 0
              and plan["forecast_admission_gate"] is False, "fixed finite collection; no forecast/RAM admission")
    c.require(runtime["native_capture_binary"] == plan["native_capture"]["binary"], "same authenticated target capture bridge")
    if mode == "qualification":
        c.require(cell_id is None, "one independent both-route qualification worker")
        return None
    matches = [row for row in plan["cells"] if row["cell_id"] == cell_id]
    c.require(len(matches) == 1, "one frozen game cohort")
    return matches[0]


def verify_admission(plan, mode, qualification_sha):
    """Recompute saved admission before any model construction or game request."""
    if mode == "qualification":
        c.require(qualification_sha is None, "qualification does not self-admit")
        return None
    from . import audit
    path = c.OUT/"qualification_gate.json"
    c.require(type(qualification_sha) is str and len(qualification_sha) == 64
              and c.sha(path) == qualification_sha, "controller-bound qualification admission")
    gate = c.read(path); bindings = gate["source_audits"]
    c.require(gate["plan_sha256"] == c.sha(c.PLAN) and gate["passed"] is True
              and len(bindings) == 1 and c.resolve(bindings[0]).resolve() == c.OUT/"qualification/saved_audit.json",
              "one passed original-plan independent collector audit")
    c.verify(bindings); proof = c.read(c.resolve(bindings[0])); recomputed = audit.qualification_gate(proof)
    c.require(set(gate) == set(recomputed) | {"plan_sha256", "source_audits"}
              and all(gate[key] == value for key, value in recomputed.items())
              and recomputed["passed"] is True, "exact independently recomputed collector admission")
    return dict(admission=c.artifact(path), source_audits=bindings, passed=True,
                independently_recomputed=True, verified_before_model_creation=True)


def run(plan, folder, mode, cell_id=None, claim_sha=None, qualification_sha=None):
    spec = validate_contract(plan, mode, cell_id)
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    c.require(folder.is_relative_to(c.OUT) and not (folder/"runtime_sources.json").exists(), "fresh separate worker; no retry")
    began, wall = time.monotonic(), time.time(); limit = plan["worker_limits"][mode+"_seconds"]
    c.require(c.sha(c.OUT/"claim.json") == claim_sha, "original global clock receipt")
    claim = c.read(c.OUT/"claim.json")
    c.require(claim["plan_sha256"] == c.sha(c.PLAN) and claim["model_seconds"] == plan["model_seconds"]
              and claim["total_seconds"] == plan["total_seconds"], "original unextended global clocks")
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()  # Exactly once at every new collector/evaluator startup.
    runtime_plan = plan["runtime"]
    c.require(sources == runtime_plan["source_guard"] == plan["source_guard"], "actual corrected runtime sources")
    c.save(folder/"runtime_sources.json", dict(sources=sources, ruleset_id=runtime_plan["ruleset_id"],
        catalogue_sha256=runtime_plan["catalogue_sha256"], turn_contract=runtime_plan["turn_contract"]))
    import psutil
    process = psutil.Process(); process.cpu_affinity(plan["worker_affinity"])
    c.save(folder/"worker_identity.json", dict(pid=os.getpid(), created=process.create_time(), mode=mode,
        cell_id=cell_id, worker_affinity=process.cpu_affinity(), started_monotonic=began, started_unix=wall,
        global_claim=c.artifact(c.OUT/"claim.json"), execution_policy_id=c.EXECUTION_POLICY_ID))
    from . import backend, collector, qualification
    runtimes = []; closed = set(); cleanup_errors = []; error = None; result = None

    def boundary():
        c.require(time.monotonic()-began < limit and time.time()-wall < limit, "actual worker dual-clock cap")
        c.require(time.monotonic()-claim["monotonic_origin"] < plan["model_seconds"]
                  and time.time()-claim["unix_origin"] < plan["model_seconds"], "original global model cutoff")

    def stage(name):
        boundary(); c.atomic_json(folder/"stage.json", dict(stage=name, time=c.now(), pid=os.getpid(), mode=mode, cell_id=cell_id))

    def build(route, destination):
        implementation = "eager" if route == "cpu" else "compiled"
        scheduler = "original8" if route == "cpu" else "row_fifo32"
        stage("initialize_"+route)
        runtime = initialize_runtime(lambda: backend.build_runtime(runtime_plan, route, destination,
            runtime_sources=sources, scheduler=scheduler, implementation=implementation), destination, route, implementation)
        runtimes.append(runtime)
        if route == "gpu":
            c.require(runtime.identity["compilation_fingerprint"] == plan["compilation_fingerprint"],
                      "same accepted compiled GPU graph before requests")
            c.verify([plan["compiled_identity"]])
            previous = c.read(c.resolve(plan["compiled_identity"]))
            c.require(previous["compilation_fingerprint"] == plan["compilation_fingerprint"], "accepted prior graph binding")
            c.save(runtime.folder/"compilation_admission.json", dict(actual=c.artifact(runtime.folder/"runtime_identity.json"),
                qualified=plan["compiled_identity"], compilation_fingerprint=plan["compilation_fingerprint"], checked_before_requests=True))
        return runtime

    def close_runtime(runtime):
        if id(runtime) in closed: return
        runtime.close()
        c.require(not runtime.active and not runtime.service.pending and not runtime.service.clients,
                  "all owned operations and callbacks closed")
        c.save(runtime.folder/"final_identity.json", dict(time=c.now(), identity=runtime.identity,
            weights_verified_unchanged=True, service_closed=True, active_operations=runtime.active,
            completed_requests=runtime.service.completed_requests, physical_batches=runtime.service.batch_count))
        closed.add(id(runtime))

    try:
        admission = verify_admission(plan, mode, qualification_sha)
        if admission is not None: c.save(folder/"qualification_gate_binding.json", admission)
        c.require(backend.source_equivalence() == plan["backend_source_equivalence"], "frozen unchanged backend source proof")
        c.require(backend.capture_provenance() == plan["native_capture"], "frozen target capture provenance")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if mode == "qualification":
            summaries = {}
            for route in plan["qualification_routes"]:
                runtime = build(route, folder/(route+"_runtime"))
                stage("actual_"+route+"_collector_loader_qualification")
                summaries[route] = qualification.run_qualification(plan, route, runtime, folder/route, boundary)
                c.require(summaries[route]["status"] == "qualified" and summaries[route]["complete_decisions"] == 116,
                          "complete actual collector qualification before next route")
                close_runtime(runtime)
            result = dict(status="qualified", routes=summaries, complete_decisions=232,
                          new_games=0, engineering_fixture_trajectories=2, training_updates=0)
        else:
            runtime = build(spec["route"], folder/"runtime")
            stage("fixed_complete_game_cohort_"+spec["cell_id"])
            result = collector.run_cell(plan, spec, runtime, folder/"cell", boundary)
    except BaseException:
        error = traceback.format_exc()
        c.save(folder/"worker_failure.json", dict(error=error, time=c.now(), mode=mode, cell_id=cell_id, no_retry=True))
        for runtime in runtimes:
            if id(runtime) not in closed:
                try: runtime.abort()
                except BaseException as caught: cleanup_errors.append(repr(caught))
    finally:
        for runtime in runtimes:
            try: close_runtime(runtime)
            except BaseException as caught: cleanup_errors.append(repr(caught))
        c.save(folder/"cleanup.json", dict(time=c.now(), success=not cleanup_errors, errors=cleanup_errors))
    if error or cleanup_errors: return 1
    try:
        boundary(); c.verify(plan["inputs"])
        c.require(result is not None, "explicit worker outcome")
        exit_code = 0 if mode == "qualification" else {"completed":0, "deadline_failed":20, "invalid_game":1}[result["status"]]
        status = "passed" if exit_code == 0 else result["status"]
        c.save(folder/"worker_result.json", dict(mode=mode, cell_id=cell_id, status=status, result=result,
            exit_code=exit_code, no_retry=True, reference_weights_unchanged=True, training_updates=0,
            execution_policy_id=c.EXECUTION_POLICY_ID))
        stage("completed_pending_independent_acceptance" if exit_code == 0 else "preserved_failed_cohort")
        return exit_code
    except BaseException:
        c.save(folder/"postrun_failure.json", dict(error=traceback.format_exc(), time=c.now(), no_retry=True))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("qualification", "games"))
    parser.add_argument("--cell-id"); parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--qualification-sha256")
    parser.add_argument("--plan", required=True); parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    c.require(Path(args.plan).resolve() == c.PLAN and c.sha(args.plan) == args.plan_sha256, "exact frozen plan before models")
    plan = c.read(args.plan); c.verify(plan["inputs"])
    return run(plan, Path(args.output), args.mode, args.cell_id, args.claim_sha256, args.qualification_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
