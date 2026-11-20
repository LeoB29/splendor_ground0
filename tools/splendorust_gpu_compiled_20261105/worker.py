"""Fresh support/entrypoint qualification and admitted exact saved-position cells."""
from pathlib import Path
import argparse
import os
import time
import traceback

from . import common as c


def initialize_runtime(operation, destination, implementation):
    """Durable complete preparation interval; never a pure compiler timing claim."""
    destination = Path(destination).resolve()
    started, started_unix = time.perf_counter(), time.time()
    error = None
    try:
        return operation()
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        ended, ended_unix = time.perf_counter(), time.time()
        c.save(destination.parent/"initialization"/(destination.name+".json"), dict(
            schema="compiled_runtime_initialization_v1", implementation=implementation,
            runtime_folder=str(destination), status="completed" if error is None else "failed", error=error,
            started_perf_counter=started, ended_perf_counter=ended, elapsed_seconds=ended-started,
            started_unix=started_unix, ended_unix=ended_unix, no_retry=True, inside_panel_timing=False,
            scope="Complete runtime preparation, including compilation and graph checks for candidate; not isolated compiler duration"))


def validate_contract(plan, mode, cell_id):
    from . import design, cell, qualification
    runtime = plan["runtime"]
    c.require(plan["schema"] == "splendorust_gpu_compiled_v1" and mode in ("qualification", "cell")
              and runtime["gpu_physical_batch"] == 8 and runtime["live_callback_batch"] == 1
              and runtime["maximum_batch_wait_seconds"] == 0.
              and runtime["internal_seconds"] == c.INTERNAL_SECONDS and runtime["outer_seconds"] == c.OUTER_SECONDS
              and runtime["execution_policy_id"] == plan["execution_policy_id"] == c.EXECUTION_POLICY_ID,
              "unchanged GPU network, search and original complete-response clocks")
    c.require(plan["worker_limits"] == dict(qualification_seconds=2700, cell_seconds=900)
              and plan["worker_affinity"] == list(range(19)), "fixed worker cap and affinity")
    c.require(plan["cells"] == design.cells() and plan["qualification_cell"] == design.qualification_cell()
              and plan["support"] == qualification.specification() == design.support()
              and plan["new_games"] == plan["training_updates"] == plan["retries"] == 0
              and plan["maximum_panel_decisions"] == 160 and plan["timing_decisions"] == 128
              and plan["timing_observer"] is False, "one fixed support/correctness/timing sequence")
    c.require(plan["qualification_routes"] == ["compiled_gpu"] and plan["qualification_decisions_per_route"] == 67,
              "fresh actual candidate GPU entrypoint qualification")
    c.require(plan["warmup"] == dict(calls_per_cell=5, case_index=0, client_id="warmup", slot=0,
        actual_service=True, inside_measured_panel=False, exact_within_route=True,
        internal_seconds=c.INTERNAL_SECONDS, outer_seconds=c.OUTER_SECONDS, failure_stops_study=True),
        "unchanged actual service warmup")
    if mode == "qualification":
        c.require(cell_id is None, "support worker is not a timing cell")
        return None
    matches = [row for row in [plan["qualification_cell"]]+plan["cells"] if row["cell_id"] == cell_id]
    c.require(len(matches) == 1, "exactly one frozen cell")
    cell.validate_spec(plan, matches[0])
    return matches[0]


def verify_admission(plan, mode, spec, support_sha, qualification_sha):
    """Recompute the exact prior admission before loading any cell model."""
    if mode == "qualification":
        c.require(support_sha is None and qualification_sha is None, "no premature admission")
        return None, None
    from . import audit
    support_only = spec["phase"] == "decision_qualification"
    c.require((qualification_sha is None) if support_only else (support_sha is None), "only the correct admission kind")
    supplied = support_sha if support_only else qualification_sha
    path = c.OUT/("support_admission.json" if support_only else "qualification_gate.json")
    c.require(type(supplied) is str and len(supplied) == 64 and c.sha(path) == supplied, "controller-bound prior admission")
    gate = c.read(path)
    c.require(gate["plan_sha256"] == c.sha(c.PLAN) and gate["passed"] is True, "passed original-plan admission")
    wanted = [c.OUT/"qualification/saved_audit.json"]
    if not support_only:
        wanted.append(c.OUT/"cells/qual_compiled/saved_audit.json")
    bindings = gate["source_audits"]
    c.require([c.resolve(item).resolve() for item in bindings] == [item.resolve() for item in wanted],
              "exact prescribed source audit artifacts")
    c.verify(bindings); proofs = [c.read(c.resolve(item)) for item in bindings]
    recomputed = audit.support_admission(proofs[0]) if support_only else audit.qualification_gate(*proofs)
    c.require(set(gate) == set(recomputed) | {"plan_sha256", "source_audits"}
              and all(gate[key] == value for key, value in recomputed.items()) and recomputed["passed"] is True,
              "independently recomputed complete admission")
    c.verify([proofs[0]["compiled_identity"]])
    identity = c.read(c.resolve(proofs[0]["compiled_identity"]))
    c.require(identity["implementation"] == "compiled"
              and identity["compilation_fingerprint"] == proofs[0]["compilation_fingerprint"], "accepted compilation identity")
    return dict(admission=c.artifact(path), source_audits=bindings, passed=True,
                independently_recomputed=True, verified_before_model_creation=True,
                compiled_identity=proofs[0]["compiled_identity"], compilation_fingerprint=identity["compilation_fingerprint"]), identity


def run(plan, folder, mode, cell_id=None, claim_sha=None, support_sha=None, qualification_sha=None):
    spec = validate_contract(plan, mode, cell_id)
    folder = Path(folder).resolve(); folder.mkdir(parents=True, exist_ok=True)
    c.require(folder.is_relative_to(c.OUT) and not (folder/"runtime_sources.json").exists(), "fresh separate worker; no retry")
    began, wall = time.monotonic(), time.time(); limit = plan["worker_limits"][mode+"_seconds"]
    c.require(c.sha(c.OUT/"claim.json") == claim_sha, "original global clock receipt")
    claim = c.read(c.OUT/"claim.json")
    c.require(claim["plan_sha256"] == c.sha(c.PLAN) and claim["model_seconds"] == plan["model_seconds"], "unextended model allowance")
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()  # Exactly once at every new evaluator startup, before models.
    runtime_plan = plan["runtime"]
    c.require(sources == runtime_plan["source_guard"], "corrected runtime source receipt")
    c.save(folder/"runtime_sources.json", dict(sources=sources, ruleset_id=runtime_plan["ruleset_id"],
        catalogue_sha256=runtime_plan["catalogue_sha256"], turn_contract=runtime_plan["turn_contract"]))
    import psutil
    process = psutil.Process(); process.cpu_affinity(plan["worker_affinity"])
    c.save(folder/"worker_identity.json", dict(pid=os.getpid(), created=process.create_time(), mode=mode,
        cell_id=cell_id, worker_affinity=process.cpu_affinity(), started_monotonic=began, started_unix=wall,
        global_claim=c.artifact(c.OUT/"claim.json"), execution_policy_id=c.EXECUTION_POLICY_ID))
    from . import backend, cell, qualification, compilation
    from tools.splendorust_gpu_throughput_20261105 import qualification as q
    runtimes = []; closed = set(); cleanup_errors = []; error = None; result = None
    support_results = {}; build_destination = None; phase = "startup"

    def boundary():
        c.require(time.monotonic()-began < limit and time.time()-wall < limit, "actual worker dual-clock cap")
        c.require(time.monotonic()-claim["monotonic_origin"] < plan["model_seconds"]
                  and time.time()-claim["unix_origin"] < plan["model_seconds"], "original global model cutoff")

    def stage(name):
        nonlocal phase
        boundary(); phase = name
        c.atomic_json(folder/"stage.json", dict(stage=name, time=c.now(), pid=os.getpid(), mode=mode, cell_id=cell_id))

    def build(destination, implementation):
        nonlocal build_destination
        build_destination = Path(destination); stage("initialize_"+implementation)
        runtime = initialize_runtime(lambda: backend.build_runtime(runtime_plan, "gpu", destination,
            runtime_sources=sources, scheduler="row_fifo32", implementation=implementation), destination, implementation)
        runtimes.append(runtime)
        return runtime

    def close_runtime(runtime):
        if id(runtime) in closed:
            return
        runtime.close()
        c.require(not runtime.active and not runtime.service.pending and not runtime.service.clients,
                  "all owned operations and callbacks closed")
        c.save(runtime.folder/"final_identity.json", dict(time=c.now(), identity=runtime.identity,
            weights_verified_unchanged=True, service_closed=True, active_operations=runtime.active,
            completed_requests=runtime.service.completed_requests, physical_batches=runtime.service.batch_count))
        closed.add(id(runtime))

    try:
        binding, qualified_identity = verify_admission(plan, mode, spec, support_sha, qualification_sha)
        if binding is not None:
            c.save(folder/("support_admission_binding.json" if spec["phase"] == "decision_qualification"
                           else "qualification_gate_binding.json"), binding)
        cell.validate_cases(plan)
        c.require(backend.source_equivalence() == plan["compilation_source_equivalence"], "frozen compilation/source proof")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "1"
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        if mode == "qualification":
            fixtures = qualification.build_fixtures(plan)
            c.save(folder/"support/fixtures.json", fixtures)
            eager = build(folder/"eager", "eager")
            stage("eager_fixed_physical_support")
            support_results["eager"] = qualification.run_implementation(plan, eager, folder/"support/eager", fixtures,
                                                                        boundary=boundary)
            close_runtime(eager)
            compiled = build(folder/"compiled", "compiled")
            stage("compiled_fixed_physical_support")
            support_results["compiled"] = qualification.run_implementation(plan, compiled, folder/"support/compiled", fixtures,
                eager_folder=folder/"support/eager", boundary=boundary)
            stage("actual_compiled_gpu_corrected_entrypoint")
            qualified = q.run_qualification(runtime_plan, compiled, folder/"qualification_gpu", boundary)
            result = dict(status="qualification_passed", physical_support=support_results,
                gpu=qualified["summary"], fresh_decisions=67, reused_qualification=plan["reused_qualification"],
                compilation_fingerprint=compiled.identity["compilation_fingerprint"], timing_cells_started=0,
                panel_decisions=0, new_games=0, training_updates=0)
        else:
            runtime = build(folder/"runtime", spec["implementation"])
            if spec["implementation"] == "compiled":
                c.require(runtime.identity["compilation_fingerprint"] == qualified_identity["compilation_fingerprint"],
                          "fresh compiled graph equals support-qualified graph before requests")
                c.save(folder/"compilation_admission.json", dict(actual=c.artifact(runtime.folder/"runtime_identity.json"),
                    qualified=binding["compiled_identity"], compilation_fingerprint=runtime.identity["compilation_fingerprint"],
                    checked_before_requests=True))
            stage("five_fixed_service_warmup_calls")
            cell.warmup_runtime(plan, runtime, folder)
            stage("fixed_thirty_two_case_"+spec["phase"])
            result = cell.run_cell(plan, spec, runtime, folder/"cell", boundary)
    except (qualification.SupportMismatch, qualification.SupportUnsupported, compilation.CompilationRejected) as caught:
        if mode != "qualification":
            error = traceback.format_exc()
            c.save(folder/"worker_failure.json", dict(error=error, time=c.now(), mode=mode, cell_id=cell_id, no_retry=True))
        else:
            if isinstance(caught, qualification.SupportMismatch):
                preserved = caught.receipt; reason = "exact_output_mismatch"
            elif isinstance(caught, qualification.SupportUnsupported):
                preserved = caught.receipt; reason = "runtime_unsupported"
            else:
                refusal = build_destination.with_name(build_destination.name+"_compilation")/"compilation_failure.json"
                payload = c.read(refusal)
                c.require(compilation.classify_saved_compile_failure(payload) == "candidate_rejected", "independently classified compile refusal")
                preserved = c.artifact(refusal); reason = "compile_rejected"
            result = dict(status="qualification_rejected", reason_kind=reason, phase=phase,
                rejection=preserved, exception_type=type(caught).__name__, error_chain=cell.error_chain(caught),
                traceback=traceback.format_exc(), completed_physical_support=support_results,
                fresh_decisions=0, timing_cells_started=0, panel_decisions=0, new_games=0, training_updates=0, no_retry=True)
            c.save(folder/"qualification_rejection.json", result)
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
    if error or cleanup_errors:
        return 1
    try:
        boundary(); c.verify(plan["inputs"])
        c.require(result is not None, "explicit worker outcome")
        exit_code = 0 if mode == "qualification" else {
            "passed": 0, "deadline_failed": 20, "exact_mismatch": 21, "integrity_failed": 1}[result["status"]]
        c.save(folder/"worker_result.json", dict(mode=mode, cell_id=cell_id, status=result["status"], result=result,
            exit_code=exit_code, no_retry=True, reference_weights_unchanged=True, training_updates=0,
            new_games=0, execution_policy_id=c.EXECUTION_POLICY_ID))
        stage("completed_pending_independent_acceptance" if exit_code == 0 else "preserved_failed_cell")
        return exit_code
    except BaseException:
        c.save(folder/"postrun_failure.json", dict(error=traceback.format_exc(), time=c.now(), no_retry=True))
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("qualification", "cell"))
    parser.add_argument("--cell-id"); parser.add_argument("--claim-sha256", required=True)
    parser.add_argument("--support-sha256"); parser.add_argument("--qualification-sha256")
    parser.add_argument("--plan", required=True); parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    c.require(Path(args.plan).resolve() == c.PLAN and c.sha(args.plan) == args.plan_sha256, "exact frozen plan before models")
    plan = c.read(args.plan); c.verify(plan["inputs"])
    return run(plan, Path(args.output), args.mode, args.cell_id, args.claim_sha256,
               args.support_sha256, args.qualification_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
