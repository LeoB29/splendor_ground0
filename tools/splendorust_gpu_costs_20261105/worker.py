"""Fresh GPU16 worker for observer qualification and bounded cost diagnosis."""
from pathlib import Path
import argparse
import os
import time
import traceback

from . import common as c


def validate_contract(plan, mode, cell_id):
    from . import design, cell
    runtime = plan["runtime"]
    c.require(plan["schema"] == "splendorust_gpu_costs_v1" and mode in ("qualification", "cell", "isolated")
              and runtime["gpu_physical_batch"] == 8 and runtime["live_callback_batch"] == 1
              and runtime["maximum_batch_wait_seconds"] == 0.
              and runtime["internal_seconds"] == c.INTERNAL_SECONDS and runtime["outer_seconds"] == c.OUTER_SECONDS
              and runtime["execution_policy_id"] == plan["execution_policy_id"] == c.EXECUTION_POLICY_ID,
              "unchanged GPU network, search and complete-response clocks")
    c.require(plan["worker_limits"] == dict(qualification_seconds=2700, cell_seconds=900, isolated_seconds=900)
              and plan["worker_affinity"] == list(range(19)), "fixed worker cap and affinity")
    c.require(plan["cells"] == design.cells() and plan["isolated"] == design.isolated()
              and plan["new_games"] == plan["training_updates"] == plan["retries"] == 0
              and plan["maximum_diagnostic_decisions"] == 192, "fixed diagnostic sequence and no collection")
    c.require(plan["qualification_routes"] == ["gpu"] and plan["qualification_decisions_per_route"] == 67,
              "fresh actual GPU entrypoint qualification")
    c.require(plan["warmup"] == dict(calls_per_cell=5, case_index=0, client_id="warmup", slot=0,
        actual_service=True, inside_measured_panel=False, exact_within_route=True,
        internal_seconds=c.INTERNAL_SECONDS, outer_seconds=c.OUTER_SECONDS, failure_stops_study=True),
        "fixed service warmup")
    if mode == "qualification":
        c.require(cell_id is None, "qualification has no timing cell")
        return None
    specs = plan["cells"] if mode == "cell" else plan["isolated"]
    matches = [row for row in specs if row["cell_id"] == cell_id]
    c.require(len(matches) == 1, "one exact frozen worker stage")
    if mode == "cell": cell.validate_spec(plan, matches[0])
    return matches[0]


def verify_full_stage_gate(plan, mode, spec, gate_sha):
    """A full diagnostic worker independently verifies the original overhead admission."""
    full = mode == "isolated" or (mode == "cell" and spec["phase"] == "diagnostic")
    if not full:
        c.require(gate_sha is None, "no premature overhead-admission binding")
        return None
    from . import audit
    path = c.OUT/"overhead_gate.json"
    c.require(type(gate_sha) is str and len(gate_sha) == 64 and c.sha(path) == gate_sha,
              "controller-bound exact overhead gate before models")
    gate = c.read(path)
    c.require(gate["plan_sha256"] == c.sha(c.PLAN) and gate["passed"] is True,
              "passed original-plan overhead gate")
    bindings = gate["source_cell_audits"]
    names = plan["overhead"]["order"]
    c.require(len(bindings) == 4 and [c.resolve(item).resolve() for item in bindings] ==
              [(c.OUT/"cells"/name/"saved_audit.json").resolve() for name in names],
              "exact four overhead source audits")
    c.verify(bindings)
    proofs = [c.read(c.resolve(item)) for item in bindings]
    recomputed = audit.overhead_gate(proofs)
    c.require(set(gate) == set(recomputed) | {"plan_sha256", "source_cell_audits"}
              and all(gate[key] == value for key, value in recomputed.items()) and recomputed["passed"] is True,
              "independently recomputed complete overhead admission")
    return dict(gate=c.artifact(path), source_cell_audits=bindings, passed=True,
                independently_recomputed=True, verified_before_model_creation=True)


def run(plan, folder, mode, cell_id=None, claim_sha=None, gate_sha=None):
    spec = validate_contract(plan, mode, cell_id)
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    c.require(folder.is_relative_to(c.OUT), "new worker evidence only")
    c.require(not (folder/"runtime_sources.json").exists(), "fresh worker, no retry")
    began, wall = time.monotonic(), time.time()
    limit = plan["worker_limits"][mode+"_seconds"]
    c.require(c.sha(c.OUT/"claim.json") == claim_sha, "original global clock receipt")
    claim = c.read(c.OUT/"claim.json")
    c.require(claim["plan_sha256"] == c.sha(c.PLAN) and claim["model_seconds"] == plan["model_seconds"],
              "unextended model allowance")
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()  # Exactly once before models in every new evaluator.
    runtime_plan = plan["runtime"]
    c.require(sources == runtime_plan["source_guard"], "corrected runtime source receipt")
    c.save(folder/"runtime_sources.json", dict(sources=sources, ruleset_id=runtime_plan["ruleset_id"],
        catalogue_sha256=runtime_plan["catalogue_sha256"], turn_contract=runtime_plan["turn_contract"]))
    import psutil
    process = psutil.Process(); process.cpu_affinity(plan["worker_affinity"])
    c.save(folder/"worker_identity.json", dict(pid=os.getpid(), created=process.create_time(), mode=mode,
        cell_id=cell_id, worker_affinity=process.cpu_affinity(), started_monotonic=began, started_unix=wall,
        global_claim=c.artifact(c.OUT/"claim.json"), execution_policy_id=c.EXECUTION_POLICY_ID))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from . import backend, cell
    from tools.splendorust_gpu_throughput_20261105 import qualification as q
    runtimes = []; cleanup_errors = []; exports = []; error = None; result = None

    def boundary():
        c.require(time.monotonic()-began < limit and time.time()-wall < limit, "actual worker dual-clock cap")
        c.require(time.monotonic()-claim["monotonic_origin"] < plan["model_seconds"]
                  and time.time()-claim["unix_origin"] < plan["model_seconds"], "original global model cutoff")

    def stage(name):
        boundary()
        c.atomic_json(folder/"stage.json", dict(stage=name, time=c.now(), pid=os.getpid(), mode=mode, cell_id=cell_id))

    def build(destination, observed):
        stage("initialize_gpu_observed" if observed else "initialize_gpu_control")
        runtime = backend.build_runtime(runtime_plan, "gpu", destination, runtime_sources=sources,
                                        scheduler="row_fifo32", observed=observed)
        runtimes.append(runtime)
        return runtime

    try:
        gate_binding = verify_full_stage_gate(plan, mode, spec, gate_sha)
        if gate_binding is not None:
            c.save(folder/"overhead_gate_binding.json", gate_binding)
        cell.validate_cases(plan)
        c.require(backend.source_equivalence() == plan["observer_source_equivalence"],
                  "frozen observer/source equivalence proof")
        if mode == "qualification":
            runtime = build(folder/"gpu", True)
            stage("actual_gpu_corrected_entrypoint")
            runtime.observer.mark("qualification_start")
            qualified = q.run_qualification(runtime_plan, runtime, folder/"qualification_gpu", boundary)
            runtime.observer.mark("qualification_end")
            result = dict(status="qualification_passed", gpu=qualified["summary"],
                reused_scheduler_qualification=plan["reused_scheduler_qualification"],
                fresh_decisions=67, observed=True, diagnostic_decisions=0, new_games=0)
        else:
            runtime = build(folder/"runtime", spec["observed"])
            if mode == "cell":
                stage("five_fixed_service_warmup_calls")
                cell.warmup_runtime(plan, runtime, folder)
                stage("fixed_thirty_two_case_panel")
                result = cell.run_cell(plan, spec, runtime, folder/"cell", boundary)
            else:
                stage("isolated_full_eight_row_batches")
                result = cell.run_isolated(plan, spec, runtime, folder/"isolated", boundary)
    except BaseException:
        error = traceback.format_exc()
        c.save(folder/"worker_failure.json", dict(error=error, time=c.now(), mode=mode, cell_id=cell_id, no_retry=True))
        for runtime in runtimes:
            try: runtime.abort()
            except BaseException as caught: cleanup_errors.append(repr(caught))
    finally:
        for runtime in runtimes:
            try:
                runtime.close()
                c.require(not runtime.active and not runtime.service.pending and not runtime.service.clients,
                          "all owned operations and callbacks closed")
                c.save(runtime.folder/"final_identity.json", dict(time=c.now(), identity=runtime.identity,
                    weights_verified_unchanged=True, service_closed=True, active_operations=runtime.active,
                    completed_requests=runtime.service.completed_requests, physical_batches=runtime.service.batch_count))
            except BaseException as caught:
                cleanup_errors.append(repr(caught))
            try:
                exports.append(runtime.observer.export(runtime.folder/"observer_timing.json",
                    scope=dict(mode=mode, cell_id=cell_id, observed=runtime.observer.enabled,
                        phase="qualification" if spec is None else spec.get("phase", "isolated"),
                        exported_after_runtime_close=True, outside_measured_intervals=True)))
            except BaseException as caught:
                cleanup_errors.append(repr(caught))
        c.save(folder/"cleanup.json", dict(time=c.now(), success=not cleanup_errors, errors=cleanup_errors))
    if error or cleanup_errors: return 1
    try:
        boundary(); c.verify(plan["inputs"])
        c.require(result is not None, "explicit worker outcome")
        exit_code = 0 if mode in ("qualification", "isolated") else {
            "passed": 0, "deadline_failed": 20, "integrity_failed": 1}[result["status"]]
        c.save(folder/"worker_result.json", dict(mode=mode, cell_id=cell_id, status=result["status"], result=result,
            exit_code=exit_code, no_retry=True, reference_weights_unchanged=True, training_updates=0,
            new_games=0, observer_exports=exports, execution_policy_id=c.EXECUTION_POLICY_ID))
        stage("completed_pending_independent_acceptance" if exit_code == 0 else "preserved_failed_cell")
        return exit_code
    except BaseException:
        c.save(folder/"postrun_failure.json", dict(error=traceback.format_exc(), time=c.now(), no_retry=True))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("qualification", "cell", "isolated"))
    parser.add_argument("--cell-id")
    parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--gate-sha256")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    c.require(Path(args.plan).resolve() == c.PLAN and c.sha(args.plan) == args.plan_sha256, "exact frozen plan before models")
    plan = c.read(args.plan); c.verify(plan["inputs"])
    validate_contract(plan, args.mode, args.cell_id)
    return run(plan, Path(args.output), args.mode, args.cell_id, args.claim_sha256, args.gate_sha256)


if __name__ == "__main__": raise SystemExit(main())
