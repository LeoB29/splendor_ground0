"""Saved-only independent audit of the bounded CPU/GPU concurrency diagnostic.

No helper in this module constructs a model, starts native search or plays games.
The command-line entry point checks corrected sources once before saved replay.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import sys

from . import common as c

LIBRARY = c.ROOT / "outputs/splendorust_gpu_live_report_20261105_03/audit_saved.py"
LIBRARY_SHA = "680782c698a135d0661afc15e972e9168649eb09d606891e88bf709727300e55"
WEIGHTS = "32cdd9466befc527d265a356be65200c3666a4fa8a82cfefdc40b327b2f529db"
CELL_ORDER = ("cpu_c1", "gpu_c1", "gpu_c2", "cpu_c2", "cpu_c4", "gpu_c4", "gpu_c8", "cpu_c8")


def require(value, message):
    if not value:
        raise ValueError("CONCURRENCY_AUDIT: " + message)


def read(path):
    path = Path(path)
    if path.name in ("active_decisions.json", "stage.json"):
        from tools.splendorust_gpu_live_20261105_03.status_io import read_status_json
        return read_status_json(path)
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bits(values):
    return b"".join(struct.pack("<f", x) for x in values)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def timing(values):
    require(values and all(math.isfinite(x) and x >= 0 for x in values), "finite raw times")
    return dict(count=len(values), mean=statistics.fmean(values), p50=statistics.median(values),
                p95=percentile(values, .95), minimum=min(values), maximum=max(values))


def load_saved_library(container):
    require(c.sha(LIBRARY) == LIBRARY_SHA, "authenticated independent03 saved arithmetic/forensic source")
    spec = importlib.util.spec_from_file_location("concurrency_prior_saved_auditor", LIBRARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = Path(container)
    return module


def validate_case(case):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import rules
    from tools.splendorust_policy_head_20261104.capture_adapter import public_root_tokens, main_support
    c.verify([case["source_initial"]])
    require(read(c.resolve(case["source_initial"])) == case["snapshot"], "unchanged saved opening")
    require(type(case["index"]) is int and 0 <= case["index"] < 8, "permanent opening index")
    state = guard.decode_training_snapshot(case["snapshot"])
    observer = rules.observation_payload(state)
    require(case["observer"] == observer and case["request"] == dict(player="search", observer=observer)
            and case["request_sha256"] == digest(case["request"]), "actual lawful corrected request")
    require(case["expected_root_tokens"] == public_root_tokens(observer), "actual corrected root encoding")
    mask = main_support(observer)[2]
    require(case["legal_policy_indices"] == [i for i, value in enumerate(mask) if value], "actual legal model support")
    return state


def validate_clock(clock):
    require(clock["internal_seconds"] == 9. and clock["outer_seconds"] == 10. and clock["no_retry"] is True,
            "unchanged complete-response guards and no retry")
    require(math.isfinite(clock["elapsed_seconds"]) and clock["elapsed_seconds"] >= 0,
            "finite actual complete-response time")
    if "ended_perf_counter" in clock:
        require(clock["elapsed_seconds"] == clock["ended_perf_counter"]-clock["started_perf_counter"],
                "raw clock arithmetic")
    if clock["status"] == "ok":
        require(clock["completed"] is True and clock["elapsed_seconds"] < 10., "completed response before outer cutoff")
    else:
        require(clock["status"] in ("error", "timeout") and clock["completed"] is False, "preserved failed response")
    return clock["status"] == "ok"


def callback_hashes(result):
    return [value for phase in result["diagnostics"]["phases"] for value in phase["neural_callbacks"]]


def root_output(case, result):
    from tools.splendorust_gpu_live_20261105_03 import qualification as q
    from tools.splendorust_search512_20261101 import scientific
    scientific.validate_result(result, case["observer"], "search", expected_quota=512)
    phases = result["diagnostics"]["phases"]
    require(phases and phases[0]["neural_callback_count"] > 0, "opening reaches learned Main inference")
    first = phases[0]["inference"][0]
    require(first["batch"] == 1 and first["tokens"] == case["expected_root_tokens"] and
            first["input_sha256"] == digest(first["tokens"]) and first["outputs"] == [first["output83"]] and
            first["output_sha256"] == digest(first["outputs"]), "actual root input/output binding")
    q.corpus._actual_encoded_costs(q.guard.decode_training_snapshot(case["snapshot"]), result["diagnostics"])
    return first["output83"]


def exact_baseline(case, result, baseline):
    if baseline is None:
        return False
    left, right = root_output(case, result), root_output(case, baseline)
    require(left == right and bits(left) == bits(right), "same-route root float32 bits equal C1")
    require(canonical(result) == canonical(baseline), "same-route full scientific result equals C1")
    return True


def read_journal(folder, route):
    events, requests = [], {}
    path = Path(folder)/"inference.jsonl"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                require(event["kind"] == "inference_batch" and event["route"] == route, "backend batch identity")
                rows = event["requests"]
                physical = 8 if route == "gpu" else 1
                require(event["physical_batch"] == physical and 1 <= len(rows) <= physical, "fixed physical batch")
                occupied = sorted(row["slot"] for row in rows)
                require(len(set(occupied)) == len(occupied) and all(0 <= x < 8 for x in occupied), "unique permanent slots")
                for row in rows:
                    require(row["route"] == route and row["batch_id"] == event["batch_id"] and row["native_batch"] == 1
                            and row["physical_batch"] == physical and row["occupied_slots"] == occupied,
                            "actual batch-one callback and packing")
                    require(row["submitted_perf"] <= row["dispatch_perf"] < row["complete_readback_perf"] < row["deadline_perf"]
                            and row["queue_seconds"] == row["dispatch_perf"]-row["submitted_perf"]
                            and row["forward_roundtrip_seconds"] == row["complete_readback_perf"]-row["dispatch_perf"],
                            "every completed callback timing/deadline")
                    require(row["request_id"] not in requests, "no repeated backend request")
                    requests[row["request_id"]] = row
                events.append(event)
    require(sorted(event["batch_id"] for event in events) == list(range(1, len(events)+1)), "complete physical batch inventory")
    return events, requests


def client_rows(case, requests):
    client = "case_"+case["case_id"]
    rows = sorted((row for row in requests.values() if row["client_id"] == client),
                  key=lambda row: int(row["request_id"].rsplit(":", 1)[1]))
    require([row["request_id"] for row in rows] == [client+":"+str(i+1) for i in range(len(rows))],
            "every returned callback in original order")
    require(all(row["slot"] == case["index"] for row in rows), "case keeps original slot at every concurrency")
    require(all(left["complete_readback_perf"] <= right["submitted_perf"] for left, right in zip(rows, rows[1:])),
            "one outstanding native callback per case")
    return rows


def audit_case(case, folder, requests, baseline=None, *, is_c1=False):
    from tools import future_training_rules_20261021 as guard
    from tools.external_strength_screen_20261018 import audit as ledger
    folder = Path(folder)
    require(read(folder/"case.json") == case and read(folder/"request.json") == case["request"], "prospective saved case/request")
    rows = client_rows(case, requests)
    if not (folder/"clock.json").exists():
        require(not rows and not (folder/"result.json").exists() and not (folder/"attempt.json").exists(), "unstarted case has no attempt, result or inference")
        return dict(case_id=case["case_id"], attempted=False, completed=False, callbacks=0, baseline_exact=False)
    clock = read(folder/"clock.json")
    success = validate_clock(clock)
    attempt = read(folder/"attempt.json")
    require(attempt["case_id"] == case["case_id"] and attempt["request_sha256"] == case["request_sha256"] and
            attempt["started_perf_counter"] == clock["started_perf_counter"], "original durable attempt and whole-response clock")
    for row in rows:
        require(row["deadline_perf"] == clock["started_perf_counter"]+9. and
                row["submitted_perf"] >= clock["started_perf_counter"], "original whole-decision deadline on every callback")
    if rows:
        require(rows[0]["input_sha256"] == digest(case["expected_root_tokens"]), "first actual callback equals corrected opening input")
    returned = read(folder/"service_calls.json")
    require(len(returned) <= len(rows), "returned service receipts are a completed journal prefix")
    for receipt, row in zip(returned, rows):
        require({k: v for k, v in receipt.items() if k != "client_received_perf"} == row and
                row["complete_readback_perf"] <= receipt["client_received_perf"] < row["deadline_perf"],
                "every returned receipt binds exact journal response")
    result_path = folder/"result.json"
    if result_path.exists():
        require(success, "completed scientific result has a successful whole-response clock")
        result = read(result_path)
        output = root_output(case, result)
        from tools.splendorust_gpu_live_20261105_03 import qualification as q
        expected_root = q.validate_fixed_result(dict(case, observer_sha256=digest(case["observer"])), result)
        require(read(folder/"root.json") == expected_root, "saved root metadata bound to actual corrected output")
        raw = read(folder/"raw_result.json")
        require(raw["result"] == result and raw["observer"] == case["observer"] and raw["action"] == result["action"]
                and raw["diagnostics"] == result["diagnostics"], "actual prevalidation result retained")
        after = read(folder/"after.json")
        require(after == ledger.transition(case["snapshot"], result["action"]), "independent legal corrected afterstate")
        guard.check_training_transition(case["snapshot"], result["action"], after)
        require(len(returned) == len(rows) and callback_hashes(result) ==
                [digest([row["input_sha256"], row["output_sha256"], 1]) for row in rows], "every scientific callback binds real service order")
        matched = True if is_c1 else exact_baseline(case, result, baseline)
        return dict(case_id=case["case_id"], attempted=True, completed=True, callbacks=len(rows), baseline_exact=matched,
                    elapsed_seconds=clock["elapsed_seconds"], root_output83=output, result=result)
    require(not success or (folder/"failure.json").exists(), "failed validation retains actual error")
    require((folder/"failure.json").exists(), "failed attempted case retained")
    if (folder/"raw_result.json").exists():
        raw = read(folder/"raw_result.json")
        require(raw["observer"] == case["observer"] and raw["action"] == raw["result"]["action"] and
                raw["diagnostics"] == raw["result"]["diagnostics"], "failed-call prevalidation result binding")
        root_output(case, raw["result"])
        require(callback_hashes(raw["result"]) == [digest([row["input_sha256"], row["output_sha256"], 1]) for row in rows],
                "late unaccepted full result still binds every saved callback")
        if not is_c1 and baseline is not None:
            exact_baseline(case, raw["result"], baseline)
    return dict(case_id=case["case_id"], attempted=True, completed=False, callbacks=len(rows), baseline_exact=False,
                clock=clock, failure=read(folder/"failure.json"))


def qualified_rate(status, rows, wall_seconds):
    """Incomplete or missing-baseline cells never receive a partial rate."""
    eligible = status == "passed" and len(rows) == 8 and all(row["completed"] and row["baseline_exact"] for row in rows)
    if not eligible:
        return dict(eligible=False, completed_decisions_per_second=None)
    require(math.isfinite(wall_seconds) and wall_seconds > 0, "positive complete cell wall time")
    return dict(eligible=True, completed_decisions_per_second=8/wall_seconds)


def admitted_baseline_results(cell, status, closure, completed):
    if cell["concurrency"] == 1 and (status != "passed" or not closure["final_weights_verified"]):
        return {}
    return completed


def audit_qualification(plan, container):
    """Reverify the actual new worker's complete qualification, never old results."""
    container = Path(container)
    lib = load_saved_library(container)
    runtime = plan["runtime"]
    compare = lib.independent_compare()
    stability, service_rows = lib.audit_stability(runtime, compare, complete=True)
    backends = {route: lib.read_backend(route, complete=True) for route in ("cpu", "gpu")}
    owner = read(container/"owner.json")
    closure = lib.verify_runtime_receipts(runtime, owner, backends, complete=True)
    for service in service_rows:
        receipt = service["receipt"]
        row = backends[service["route"]][1][receipt["request_id"]]
        require({k: v for k, v in receipt.items() if k != "client_received_perf"} == row and
                row["complete_readback_perf"] <= receipt["client_received_perf"] < row["deadline_perf"],
                "all80 qualification service calls bind real backend responses")
    proof, qualified, callbacks = {}, {}, {}
    for route in ("cpu", "gpu"):
        qualified[route], proof[route] = lib.audit_qualification(runtime, route, compare)
        callbacks[route] = lib.bind_qualification_callbacks(route, qualified[route], backends[route][1])
        require(sum(callbacks[route].values())+40 == len(backends[route][1]), "all qualification worker requests accounted")
    from tools.splendorust_gpu_live_20261105_03 import qualification as q
    comparison = q.compare_qualification(qualified["cpu"], qualified["gpu"], runtime)
    require(comparison == read(container/"worker/decision_comparison.json"), "full35 control comparison rederived")
    for case, left, right, row in zip(runtime["qualification"]["decision_panel"], qualified["cpu"]["fixed_roots"],
                                      qualified["gpu"]["fixed_roots"], comparison["rows"], strict=True):
        computed = compare([left["root"]["output83"]], [right["root"]["output83"]],
                           [case["legal_policy_indices"]], runtime["numerical"])
        require(computed == row["numerical"] and computed["passed"], "independent35 numerical arithmetic")
    require(read(container/"worker/cleanup.json")["success"] is True, "qualification cleanup")
    worker = read(container/"worker/worker_result.json")
    require(worker["mode"] == "qualification" and worker["cell_id"] is None and worker["exit_code"] == 0 and
            worker["reference_weights_unchanged"] is True and worker["result"]["decision_comparison"] == comparison and
            worker["result"]["packing"] == read(container/"worker/stability/summary.json"), "qualified worker result binds all new saved checks")
    return dict(status="saved_fresh_qualification_verified", stability=stability, routes=proof,
                callbacks=callbacks, runtime_closure=closure, comparison=comparison, model_calls=0)


def audit_cross_baselines(plan, baselines):
    """Only root numerical tolerance is gated across routes; actions descriptive."""
    complete = all(set(baselines.get(route, {})) == {case["case_id"] for case in plan["cases"]} for route in ("cpu", "gpu"))
    if not complete:
        return dict(complete=False, passed=False, rows=[], reason="Both complete C1 baselines are required")
    compare = load_saved_library(c.OUT).independent_compare()
    rows = []
    for case in plan["cases"]:
        left, right = (baselines[route][case["case_id"]] for route in ("cpu", "gpu"))
        value = compare([root_output(case, left)], [root_output(case, right)],
                        [case["legal_policy_indices"]], plan["runtime"]["numerical"])
        require(value["passed"], "C1 CPU/GPU corrected opening root numerical tolerance")
        rows.append(dict(case_id=case["case_id"], numerical=value, full_action_same=left["action"] == right["action"]))
    return dict(complete=True, passed=True, rows=rows, action_agreement_is_gate=False)


def select_eligible(cells, cross_baselines):
    result = {}
    for route in ("cpu", "gpu"):
        choices = [cell for cell in cells if cell["route"] == route and cell["eligible"] and cross_baselines["passed"]]
        if choices:
            choice = min(choices, key=lambda cell: (-cell["completed_decisions_per_second"], cell["concurrency"]))
            result[route] = dict(cell_id=choice["cell_id"], concurrency=choice["concurrency"],
                                 completed_decisions_per_second=choice["completed_decisions_per_second"])
        else:
            result[route] = None
    return result


def audit_runtime(plan, folder, route, events, requests, *, incomplete=False):
    folder = Path(folder)
    identity = read(folder/"runtime_identity.json")
    runtime = plan["runtime"]
    require(identity["route"] == route and identity["weights_sha256"] == WEIGHTS and
            identity["checkpoint_sha256"] == runtime["checkpoint"]["sha256"] and
            identity["runtime_sources"] == runtime["source_guard"] and identity["threads"] == identity["interop_threads"] == 1,
            "actual immutable original runtime identity")
    if route == "gpu":
        require("AMD Radeon RX 7800 XT" in identity["device"]["selected"]["name"] and
                identity["device"]["torch_device"] == "privateuseone:"+str(identity["device"]["selected"]["index"]), "selected physical GPU")
    if incomplete:
        return dict(initial_identity_verified=True, final_weights_verified=False,
                    reason="Supervisor terminated an expired ordinary request; final runtime receipt unavailable")
    final, closed = read(folder/"runtime_final.json"), read(folder/"final_identity.json")
    require(final["unchanged_weights_sha256"] == WEIGHTS and closed["weights_verified_unchanged"] is True and
            closed["service_closed"] is True and closed["active_operations"] == {} and
            read(folder/"active_decisions.json")["active"] == [], "closed runtime and unchanged final weights")
    require({k: v for k, v in closed["identity"].items() if k != "warnings"} ==
            {k: v for k, v in identity.items() if k != "warnings"}, "initial/final full runtime identity")
    require(final["service_batches"] == closed["physical_batches"] == len(events) and
            final["service_requests"] == closed["completed_requests"] == len(requests), "all backend counters reconstructed")
    lib = load_saved_library(c.OUT)
    require(not any(lib.fallback(row["message"]) for row in final["warnings"]), "no unsupported backend/fallback warning")
    return dict(initial_identity_verified=True, final_weights_verified=True, warnings=final["warnings"], service_failure=final["service_failure"])


def audit_warmup(plan, folder, requests):
    folder = Path(folder)
    value = read(folder/"warmup.json")
    require(value["status"] == "passed" and value["planned_calls"] == len(value["calls"]) == 5 and
            value["inside_cell_timing"] is False and value["search_decisions"] == 0 and value["no_retry"] is True and
            value["error"] is None, "all five bounded real-service warmup calls passed")
    baseline = value["calls"][0]["output"]
    require(len(baseline) == 1 and len(baseline[0]) == 83 and all(math.isfinite(x) for x in baseline[0]), "warmup finite full network output")
    for index, row in enumerate(value["calls"]):
        require(row == read(folder/"warmup_raw"/f"{index:02d}.json") and row["index"] == index and
                row["clock"] == read(folder/"warmup_clocks"/f"{index:02d}.json") and validate_clock(row["clock"]), "actual raw warmup/clock binding")
        require(row["output"] == baseline and bits(row["output"][0]) == bits(baseline[0]), "five exact same-route warmup outputs")
        receipt = row["receipt"]
        journal = requests["warmup:"+str(index+1)]
        require({k: v for k, v in receipt.items() if k != "client_received_perf"} == journal and
                journal["slot"] == 0 and journal["client_id"] == "warmup" and
                journal["deadline_perf"] == row["clock"]["started_perf_counter"]+9. and
                journal["input_sha256"] == digest(plan["cases"][0]["expected_root_tokens"]) and
                journal["output_sha256"] == digest(row["output"]) and
                journal["complete_readback_perf"] <= receipt["client_received_perf"] < journal["deadline_perf"], "warmup actual source/slot/receipt")
    return value


def deadline_classification(chain):
    allowed = {
        "GPU_LIVE_INTERNAL_DEADLINE: original complete9s budget", "Internal fixed-work deadline exceeded",
        "GPU inference callback exceeded original 9s decision deadline", "Queued inference expired before dispatch",
        "Completed inference exceeded original callback deadline", "Inference complete readback arrived after decision deadline",
        "Full ordinary response exceeded fixed9s internal guard", "Complete player response exceeded fixed outer guard",
        "Complete response observed after fixed outer guard", "Whole completed operation exceeded dual-clock10s bound"}
    found = False
    for row in chain:
        if row["type"] == "TimeoutError" and row["message"] in allowed:
            found = True
        elif row["type"] == "RuntimeError" and (row["message"] == "Inference service failed; no retry" or
                row["message"].startswith("Inference request failed: ")):
            pass
        else:
            return "integrity_failed"
    if not found:
        return "integrity_failed"
    return "deadline_failed" if chain[0]["type"] == "TimeoutError" else "cancelled"


def wrapped_deadline_proof(initial):
    if initial["status"] != "cancelled" or deadline_classification(initial["error_chain"]) != "cancelled":
        return None
    clock = initial["clock"]
    messages = [row["message"] for row in initial["error_chain"] if row["type"] == "TimeoutError"]
    outer = {"Complete player response exceeded fixed outer guard", "Complete response observed after fixed outer guard",
             "Whole completed operation exceeded dual-clock10s bound"}
    required = 10. if any(message in outer for message in messages) else 9.
    values = [clock.get(key) for key in ("elapsed_seconds", "started_perf_counter", "ended_perf_counter")]
    if (not all(type(value) in (int, float) and math.isfinite(value) for value in values) or clock["elapsed_seconds"] < required or
            clock["elapsed_seconds"] != clock["ended_perf_counter"]-clock["started_perf_counter"]):
        return None
    return dict(kind="wrapped_known_deadline_and_own_original_clock_expired", required_elapsed_seconds=required,
                observed_elapsed_seconds=clock["elapsed_seconds"], started_perf_counter=clock["started_perf_counter"],
                ended_perf_counter=clock["ended_perf_counter"], known_timeout_messages=messages)


def validate_group_failures(outcomes, initial_failures):
    """Reconstruct peer cancellation from the immutable pre-resolution errors."""
    by_id = {row["case_id"]: row for row in outcomes}
    require(len(by_id) == len(outcomes), "unique group outcomes")
    triggers = set()
    for case_id, initial in initial_failures.items():
        require(case_id in by_id and initial["clock"]["status"] in ("error", "timeout") and
                initial["clock"]["completed"] is False and initial["clock"]["internal_seconds"] == 9. and
                initial["clock"]["outer_seconds"] == 10., "failed original clock required for any group cancellation")
        require(initial["status"] == deadline_classification(initial["error_chain"]), "raw initial failure classification preserved")
        if initial["status"] == "deadline_failed" or wrapped_deadline_proof(initial) is not None:
            triggers.add(case_id)
    for case_id, initial in initial_failures.items():
        final = by_id[case_id]
        chain = initial["error_chain"]
        proof = wrapped_deadline_proof(initial)
        if proof is not None:
            require(final == dict(initial, initial_status="cancelled", status="deadline_failed", deadline_proof=proof),
                    "wrapped known timeout additionally proves this request's own unchanged deadline expired")
            continue
        poison = bool(chain) and any(row["message"] in ("Evaluator aborted; no replacement or retry", "Service closed with outstanding inference") for row in chain)
        poison = poison and all(row["type"] == "RuntimeError" and (row["message"] in (
            "Inference service failed; no retry", "Evaluator aborted; no replacement or retry", "Service closed with outstanding inference") or
            row["message"].startswith("Inference request failed: ")) for row in chain)
        if initial["status"] == "cancelled" or poison:
            require(bool(triggers) and final["status"] == "cancelled" and final["initial_status"] == initial["status"],
                    "peer cancellation needs a same-group direct deadline")
            actual = final["cancellation_trigger_case_ids"]
            require(len(actual) == len(set(actual)) and set(actual) == triggers, "all and only actual same-group timeout trigger IDs")
            expected = dict(initial, initial_status=initial["status"], status="cancelled", cancellation_trigger_case_ids=actual)
            require(final == expected, "final peer outcome changes only authenticated cancellation metadata")
        else:
            require(initial["status"] == "deadline_failed" and final == initial, "direct deadline retains exact original failure")
    return sorted(triggers)


def require_processes_absent(identities):
    import psutil
    unique = {(row["pid"], row["created"]): row for row in identities if row is not None}
    for row in unique.values():
        try:
            require(psutil.Process(row["pid"]).create_time() != row["created"], "exact owned process still alive")
        except psutil.NoSuchProcess:
            pass
    return list(unique.values())


def audit_stage_execution(plan, container, *, mode, cell=None):
    container = Path(container)
    owner, execution = read(container/"owner.json"), read(container/"execution.json")
    stage_id = "qualification" if mode == "qualification" else cell["cell_id"]
    seconds = plan["worker_limits"]["qualification_seconds" if mode == "qualification" else "cell_seconds"]
    require(owner["identity"] == execution["identity"] and owner["stage_id"] == execution["stage_id"] == stage_id and
            owner["mode"] == execution["mode"] == mode and owner["plan_sha256"] == execution["plan_sha256"] == c.sha(c.PLAN), "actual exact frozen stage ownership")
    require(owner["maximum_seconds"] == execution["maximum_seconds"] == seconds and
            owner["monotonic_origin"] == execution["monotonic_origin"] and owner["unix_origin"] == execution["unix_origin"] and
            0 <= execution["monotonic_end"]-execution["monotonic_origin"] < seconds and
            0 <= execution["unix_end"]-execution["unix_origin"] < seconds, "actual unchanged stage dual-clock cap")
    require(execution["retries"] == 0 and not execution["cleanup_errors"] and not execution["native_fallback_lines"], "no retry, clean exact cleanup and no native fallback")
    absent = require_processes_absent(execution["observed_owned_identities"]+[execution["identity"]])
    worker = read(container/"worker/worker_identity.json")
    require(dict(pid=worker["pid"], created=worker["created"]) == execution["identity"] and
            worker["mode"] == mode and worker["cell_id"] == (None if cell is None else cell["cell_id"]) and
            worker["worker_affinity"] == plan["worker_affinity"], "actual fresh stage worker identity/affinity")
    lib = load_saved_library(c.OUT)
    stderr = (container/"worker/stderr.log").read_text(encoding="utf-8", errors="replace")
    require(not any(lib.fallback(line) for line in stderr.splitlines()), "independent native fallback scan")
    return execution, absent


def audit_cell(plan, cell, worker_folder, baselines=None):
    folder = Path(worker_folder)
    baselines = {} if baselines is None else baselines
    execution, absent = audit_stage_execution(plan, folder.parent, mode="cell", cell=cell)
    require(execution["status"] in ("completed", "deadline_failed"), "global integrity failures cannot qualify as isolated cells")
    require(read(folder/"runtime_sources.json")["sources"] == plan["source_guard"], "cell actual startup corrected-source check")
    for case in plan["cases"]:
        validate_case(case)
    events, requests = read_journal(folder/"runtime", cell["route"])
    warmup = audit_warmup(plan, folder, requests)
    outer = execution["outer_deadline_observed"]
    if outer is not None:
        return audit_killed_cell(plan, cell, folder, execution, absent, events, requests, warmup, baselines)
    require(execution["worker_exit_code"] == (0 if execution["status"] == "completed" else 20), "exact returned worker exit convention")
    closure = audit_runtime(plan, folder/"runtime", cell["route"], events, requests)
    require(read(folder/"cleanup.json")["success"] is True, "worker cleanup complete before next independent cell")
    summary = read(folder/"cell/cell_summary.json")
    require(summary["status"] != "passed" or closure["service_failure"] is None, "passed cell service has no latent backend failure")
    worker_result = read(folder/"worker_result.json")
    require(worker_result["mode"] == "cell" and worker_result["cell_id"] == cell["cell_id"] and
            worker_result["result"] == summary and worker_result["exit_code"] == execution["worker_exit_code"] and
            worker_result["reference_weights_unchanged"] is True and worker_result["no_retry"] is True,
            "returned worker binds exact cell outcome and exit")
    require(summary["cell_id"] == cell["cell_id"] and summary["route"] == cell["route"] and summary["concurrency"] == cell["concurrency"] and
            summary["status"] == ("passed" if execution["status"] == "completed" else "deadline_failed") and
            summary["planned_case_count"] == 8 and summary["no_retry"] is True and summary["game_continuations"] == 0, "fixed isolated cell identity/status")
    rows, outcomes, completed = [], {}, {}
    for case in plan["cases"]:
        case_folder = folder/"cell"/case["case_id"]
        row = audit_case(case, case_folder, requests, baselines.get(case["case_id"]), is_c1=cell["concurrency"] == 1)
        if row["attempted"]:
            outcome = read(case_folder/"outcome.json")
            require(outcome["case_id"] == case["case_id"] and outcome["index"] == case["index"] and outcome["client_id"] == "case_"+case["case_id"] and
                    outcome["decision_started"] is True and
                    outcome["clock"] == read(case_folder/"clock.json") and outcome["request_sha256"] == case["request_sha256"] and outcome["no_retry"] is True,
                    "every terminal outcome bound to original case/request/clock")
            if row["completed"]:
                require(outcome["status"] == "completed" and outcome["result_sha256"] == digest(row["result"]) and
                        outcome["after_sha256"] == digest(read(case_folder/"after.json")), "completed outcome result/transition hashes")
                completed[case["case_id"]] = row.pop("result")
            else:
                require(outcome["status"] in ("deadline_failed", "cancelled"), "only direct deadline or resolved peer failure")
                row["failure_kind"] = outcome["status"]
            outcomes[case["index"]] = outcome
        rows.append(row)
    groups = summary["groups"]
    require(groups and len(groups) <= len(cell["groups"]), "bounded planned group prefix")
    expected_attempts = []
    for index, group in enumerate(groups):
        indexes = cell["groups"][index]
        expected_attempts += indexes
        path = folder/"cell/groups"/f"{index:03d}"
        require(group == read(path/"summary.json") and group["group_index"] == index and group["case_indexes"] == indexes and
                group["case_ids"] == [plan["cases"][i]["case_id"] for i in indexes] and
                group["outcomes"] == [outcomes[i] for i in indexes], "all scheduled group cases and durable outcomes")
        failures = {plan["cases"][i]["case_id"]: read(folder/"cell"/plan["cases"][i]["case_id"]/"failure.json")
                    for i in indexes if not rows[i]["completed"]}
        validate_group_failures(group["outcomes"], failures)
        require(read(path/"start.json") == dict(group_index=index, case_indexes=indexes,
                started_perf_counter=group["started_perf_counter"], started_unix=group["started_unix"]), "original barrier start receipt")
        require(group["wall_seconds"] == group["ended_perf_counter"]-group["started_perf_counter"] > 0 and
                group["attempted"] == len(indexes) and group["completed"] == sum(rows[i]["completed"] for i in indexes), "raw group duration/counts")
        for i in indexes:
            clock = outcomes[i]["clock"]
            require(group["started_perf_counter"] <= clock["started_perf_counter"] and
                    clock.get("ended_perf_counter", clock["started_perf_counter"]+clock["elapsed_seconds"]) <= group["ended_perf_counter"], "case clock wholly inside group")
        if index:
            require(groups[index-1]["ended_perf_counter"] <= group["started_perf_counter"], "successive groups never overlap")
        require(group["status"] == ("passed" if all(rows[i]["completed"] for i in indexes) else "deadline_failed") and
                (index == len(groups)-1 or group["status"] == "passed"), "stop cell after first failed group")
    require(set(outcomes) == set(expected_attempts) and summary["attempted_case_ids"] == [row["case_id"] for row in rows if row["attempted"]] and
            summary["completed_case_ids"] == list(completed) and summary["never_started_case_ids"] == [row["case_id"] for row in rows if not row["attempted"]], "exact attempted/completed/unstarted partition")
    require(summary["started_perf_counter"] == groups[0]["started_perf_counter"] and summary["started_unix"] == groups[0]["started_unix"] and
            summary["ended_perf_counter"] >= groups[-1]["ended_perf_counter"] and
            summary["wall_seconds"] == summary["ended_perf_counter"]-summary["started_perf_counter"] > 0 and
            warmup["calls"][-1]["clock"]["ended_perf_counter"] < summary["started_perf_counter"], "complete panel wall includes groups and excludes warmup")
    require(len(requests) == 5+sum(row["callbacks"] for row in rows), "all cell backend requests accounted as warmup or original cases")
    require(summary["intentional_intergroup_gap_seconds"] == 0. and summary["full_cell_completed"] is
            (summary["status"] == "passed" and len(outcomes) == 8), "all8 criterion and no added wait")
    if plan["cases"][0]["case_id"] in completed:
        value = root_output(plan["cases"][0], completed[plan["cases"][0]["case_id"]])
        require(value == warmup["calls"][0]["output"][0] and bits(value) == bits(warmup["calls"][0]["output"][0]), "native root equals warmed same model exactly")
    rate = qualified_rate(summary["status"], rows, summary["wall_seconds"])
    panel_events = [event for event in events if event["requests"][0]["client_id"] != "warmup"]
    panel_requests = [row for event in panel_events for row in event["requests"]]
    require(len(panel_requests) == len(requests)-5 and all(row["client_id"].startswith("case_") and
            summary["started_perf_counter"] <= row["submitted_perf"] and row["complete_readback_perf"] <= summary["ended_perf_counter"]
            for row in panel_requests), "panel-only backend timing excludes startup and warmup")
    backend_metrics = dict(completed_callback_prefixes=len(panel_requests), physical_batches=len(panel_events),
        occupied_rows_histogram={str(key): value for key, value in sorted(Counter(len(event["requests"]) for event in panel_events).items())},
        mean_occupied_rows=len(panel_requests)/len(panel_events) if panel_events else None,
        queue_seconds=timing([row["queue_seconds"] for row in panel_requests]) if panel_requests else None,
        physical_forward_readback_seconds=timing([event["requests"][0]["forward_roundtrip_seconds"] for event in panel_events]) if panel_events else None,
        scope="Actual panel callback prefixes only; excludes warmup; failed-cell samples do not define a completed-panel rate")
    return dict(status=summary["status"], cell_id=cell["cell_id"], route=cell["route"], concurrency=cell["concurrency"], **rate,
                case_rows=rows, completed_results=admitted_baseline_results(cell, summary["status"], closure, completed),
                retained_completed_results=completed, wall_seconds=summary["wall_seconds"], runtime_closure=closure,
                callback_count=len(requests)-5, warmup_calls=5, physical_batches=len(events), backend_metrics=backend_metrics, owned_processes_absent=absent,
                complete_response_seconds=timing([row["elapsed_seconds"] for row in rows if row["completed"]]) if completed else None,
                model_calls=0, new_games=0)


def audit_killed_cell(plan, cell, folder, execution, absent, events, requests, warmup, baselines):
    """Parent-authenticated outer expiry is an incomplete cell, never a rate."""
    outer = execution["outer_deadline_observed"]
    clients = {"case_"+case["case_id"]: case for case in plan["cases"]}
    require(execution["status"] == "deadline_failed" and execution["worker_exit_code"] != 0 and
            outer["kind"] == "ordinary" and outer["client_id"] in clients and
            abs(outer["deadline_perf"]-outer["started_perf"]-10.) < 1e-5 and
            abs(outer["deadline_unix"]-outer["started_unix"]-10.) < 1e-5 and
            (outer["observed_perf"] >= outer["deadline_perf"] or outer["observed_unix"] >= outer["deadline_unix"]),
            "only authenticated original ordinary10s expiry permits isolated supervisor termination")
    closure = audit_runtime(plan, folder/"runtime", cell["route"], events, requests, incomplete=True)
    rows, completed = [], {}
    for case in plan["cases"]:
        path = folder/"cell"/case["case_id"]
        require(read(path/"case.json") == case and read(path/"request.json") == case["request"], "killed cell preserves prospective request")
        journal = client_rows(case, requests)
        if outer["client_id"] == "case_"+case["case_id"]:
            trigger_attempt = read(path/"attempt.json")
            require(trigger_attempt["started_perf_counter"] == outer["started_perf"] and
                    trigger_attempt["request_sha256"] == case["request_sha256"], "parent expired operation is exactly this original case attempt")
        if (path/"result.json").exists() and (path/"after.json").exists() and (path/"service_calls.json").exists():
            row = audit_case(case, path, requests, baselines.get(case["case_id"]), is_c1=cell["concurrency"] == 1)
            if row["completed"]:
                completed[case["case_id"]] = row.pop("result")
            rows.append(row)
            continue
        attempted = (path/"attempt.json").exists()
        require(attempted or not journal, "callbacks require a durable actual attempt")
        if attempted:
            attempt = read(path/"attempt.json")
            require(attempt["case_id"] == case["case_id"] and attempt["request_sha256"] == case["request_sha256"] and
                    case["index"] in cell["groups"][attempt["group_index"]], "killed original group member/request")
            group_start = read(folder/"cell/groups"/f"{attempt['group_index']:03d}"/"start.json")
            require(group_start["case_indexes"] == cell["groups"][attempt["group_index"]] and
                    group_start["started_perf_counter"] <= attempt["started_perf_counter"], "killed attempt follows its original group barrier")
            if journal:
                started = attempt["started_perf_counter"]
                require(all(row["deadline_perf"] == started+9. and row["submitted_perf"] >= started for row in journal) and
                        journal[0]["input_sha256"] == digest(case["expected_root_tokens"]), "killed callback prefix original deadline/input")
            if (path/"clock.json").exists():
                clock = read(path/"clock.json")
                require(clock["started_perf_counter"] == attempt["started_perf_counter"], "killed saved clock retains original attempt origin")
                validate_clock(clock)
        rows.append(dict(case_id=case["case_id"], attempted=attempted, completed=False, callbacks=len(journal), baseline_exact=False,
                         failure_kind="supervisor_ordinary_deadline" if outer["client_id"] == "case_"+case["case_id"] else
                                      "supervisor_interrupted_peer" if attempted else "never_started",
                         complete_response_clock_available=(path/"clock.json").exists()))
    require(len(requests) == 5+sum(row["callbacks"] for row in rows), "all saved killed-cell callback prefixes accounted")
    return dict(status="deadline_failed", cell_id=cell["cell_id"], route=cell["route"], concurrency=cell["concurrency"],
                eligible=False, completed_decisions_per_second=None, wall_seconds=None, case_rows=rows,
                completed_results=admitted_baseline_results(cell, "deadline_failed", closure, completed), retained_completed_results=completed,
                supervisor_expiry=outer, runtime_closure=closure, owned_processes_absent=absent,
                callback_count=len(requests)-5, warmup_calls=5, model_calls=0, new_games=0,
                unfinished_callback_and_final_weight_receipts_unavailable=True)


def load_baselines(plan, cell, folder):
    require(cell["concurrency"] == 1, "only C1 establishes the route baseline")
    result = {}
    for case in plan["cases"]:
        path = Path(folder)/"cell"/case["case_id"]
        if (path/"result.json").exists() and (path/"after.json").exists() and (path/"outcome.json").exists():
            if read(path/"outcome.json")["status"] == "completed":
                value = read(path/"result.json")
                root_output(case, value)
                result[case["case_id"]] = value
    return result


check_cross_route_baselines = audit_cross_baselines


def audit_study(plan, output, *, expected_plan_sha, controller_session, controller_exit_code, controller_chunk, allow_failure=False):
    """Final independent saved evidence; called only after actual controller exit."""
    from . import design
    output = Path(output)
    require(c.sha(output/"plan.json") == expected_plan_sha, "explicit observed frozen plan SHA")
    c.verify(plan["inputs"])
    previous = read(c.resolve(plan["prior"]["plan"]))
    require(plan["runtime"] == {key: previous[key] for key in design.RUNTIME_KEYS} and plan["cells"] == design.cells() and
            [row["cell_id"] for row in plan["cells"]] == list(CELL_ORDER), "unchanged original scientific runtime and prospective cell order")
    require(plan["total_seconds"] == 3600 and plan["worker_limits"] == dict(qualification_seconds=1200, cell_seconds=300) and
            plan["maximum_diagnostic_decisions"] == 64 and plan["group_gap_seconds"] == 0. and
            plan["retries"] == plan["new_games"] == plan["training_updates"] == 0, "bounded independent diagnostic authorization")
    for case in plan["cases"]:
        validate_case(case)
    for key, tree in (("preserved_evidence", design.PRIOR), ("preserved_reports", design.PRIOR_REPORT)):
        items = plan["prior"][key]
        c.verify(items)
        require(sorted(items, key=lambda x: x["path"]) == sorted((c.artifact(path) for path in tree.rglob("*") if path.is_file()),
                key=lambda x: x["path"]), "closed predecessor evidence/report inventory untouched")
    claim, execution = read(output/"claim.json"), read(output/"execution.json")
    require(execution["plan_sha256"] == claim["plan_sha256"] == expected_plan_sha and execution["retries"] == claim["retries"] == 0 and
            execution["monotonic_origin"] == claim["monotonic_origin"] and execution["unix_origin"] == claim["unix_origin"] and
            0 <= execution["monotonic_end"]-claim["monotonic_origin"] and
            0 <= execution["unix_end"]-claim["unix_origin"], "global original actual clock identity")
    within_global_bound = execution["monotonic_end"]-claim["monotonic_origin"] < 3600 and execution["unix_end"]-claim["unix_origin"] < 3600
    absent = require_processes_absent(execution["observed_owned_identities"]+[dict(pid=claim["pid"], created=claim["created"])])
    success = execution["status"] == "completed_cells_pending_independent_acceptance"
    require(controller_exit_code == (0 if success else 1), "root-observed retained controller exit")
    stage_rows = execution["stages"]
    require([row["stage_id"] for row in stage_rows] == (["qualification"]+list(CELL_ORDER))[:len(stage_rows)], "no missing/reordered/repeated stages")
    if not success:
        require(allow_failure and execution["status"] == "failed" and read(output/"failure.json") == execution and
                not (output/"completion.json").exists(), "explicit preserved fatal-study audit only")
        return audit_failed_study(plan, output, execution, absent, expected_plan_sha, controller_session,
                                  controller_exit_code, controller_chunk, within_global_bound)
    require(within_global_bound and not execution["cleanup_errors"], "successful study actual total bound and cleanup")
    require(success and len(stage_rows) == 9 and (output/"completion.json").exists() and not (output/"failure.json").exists(),
            "all independent cells executed; fatal study is preserved but cannot receive final completion acceptance")
    qualification_execution, _ = audit_stage_execution(plan, output/"qualification", mode="qualification")
    require(qualification_execution["status"] == "completed" and qualification_execution["worker_exit_code"] == 0, "fresh qualification worker actually passed")
    qualification = audit_qualification(plan, output/"qualification")
    require(qualification == read(output/"qualification/saved_audit.json"), "qualification saved proof rederived")
    baselines, proofs, previous_end = {"cpu": {}, "gpu": {}}, [], qualification_execution["unix_end"]
    for index, cell in enumerate(plan["cells"]):
        container = output/"cells"/cell["cell_id"]
        stage_execution = read(container/"execution.json")
        require(stage_execution["unix_origin"] >= previous_end and stage_rows[index+1] ==
                {key: stage_execution[key] for key in ("stage_id", "status", "worker_exit_code")}, "independent cells never overlap and global receipt binds each exit")
        previous_end = stage_execution["unix_end"]
        proof = audit_cell(plan, cell, container/"worker", baselines[cell["route"]])
        require(proof == read(container/"saved_audit.json"), "every per-cell admission proof independently rederived")
        proofs.append(proof)
        if cell["concurrency"] == 1:
            baselines[cell["route"]] = proof["completed_results"]
    cross = audit_cross_baselines(plan, baselines)
    calls = sum(sum(row["attempted"] for row in proof["case_rows"]) for proof in proofs)
    require(calls <= 64, "at most64 actual diagnostic search calls including failures")
    selection = select_eligible(proofs, cross)
    summaries = [{key: value for key, value in proof.items() if key not in ("completed_results", "retained_completed_results")} for proof in proofs]
    return dict(status="saved_concurrency_study_verified", plan_sha256=expected_plan_sha, auditor=c.artifact(Path(__file__)), frozen_inputs_unchanged=len(plan["inputs"]),
                qualification=qualification, cells=summaries, cross_route_baselines=cross, selection=selection,
                diagnostic_decisions_attempted=calls, auditor_models_launched=0, auditor_games_launched=0, study_new_games=0,
                exact_owned_processes_absent=absent,
                retained_controller_exit=dict(session_id=controller_session, exit_code=controller_exit_code, tool_chunk=controller_chunk,
                    source="Root-provided retained execution; process absence independently checked"),
                selection_requires_both_complete_C1_numeric_baselines=True,
                scope="Eight fixed saved opening requests per cell; no whole-game throughput or playing-strength conclusion")


def audit_failed_study(plan, output, execution, absent, expected, session, exit_code, chunk, within_bound):
    """Preserve a stopped study without certifying failed scientific records."""
    stages, baselines = [], {"cpu": {}, "gpu": {}}
    for index, recorded in enumerate(execution["stages"]):
        cell = None if index == 0 else plan["cells"][index-1]
        container = output/"qualification" if cell is None else output/"cells"/cell["cell_id"]
        observed = read(container/"execution.json")
        require(recorded == {key: observed[key] for key in ("stage_id", "status", "worker_exit_code")} and
                observed["plan_sha256"] == expected, "fatal study exact retained stage exit")
        stage_absent = require_processes_absent(observed["observed_owned_identities"]+[observed["identity"]])
        owner_exists = (container/"owner.json").exists()
        if owner_exists:
            owner = read(container/"owner.json")
            require(owner["identity"] == observed["identity"] and owner["plan_sha256"] == expected and
                    owner["monotonic_origin"] == observed["monotonic_origin"] and owner["unix_origin"] == observed["unix_origin"],
                    "fatal stage original owner and clock")
        proof = None
        verification_error = None
        if observed["status"] in ("completed", "deadline_failed"):
            try:
                if cell is None:
                    audit_stage_execution(plan, container, mode="qualification")
                    proof = audit_qualification(plan, container)
                else:
                    proof = audit_cell(plan, cell, container/"worker", baselines[cell["route"]])
                    if cell["concurrency"] == 1:
                        baselines[cell["route"]] = proof["completed_results"]
                if (container/"saved_audit.json").exists():
                    require(proof == read(container/"saved_audit.json"), "previous admission proof still reproduces")
            except Exception as error:
                verification_error = repr(error)
                proof = None
        if proof is not None:
            proof = {key: value for key, value in proof.items() if key not in ("completed_results", "retained_completed_results")}
        stages.append(dict(stage_id=observed["stage_id"], status=observed["status"], worker_exit_code=observed["worker_exit_code"],
                           execution=c.artifact(container/"execution.json"), owner_present=owner_exists,
                           cleanup_errors=observed["cleanup_errors"], exact_processes_absent=stage_absent,
                           completed_evidence_proof=proof, verification_error=verification_error))
    return dict(status="saved_fatal_study_preservation_audited", plan_sha256=expected, auditor=c.artifact(Path(__file__)), frozen_inputs_unchanged=len(plan["inputs"]),
                retained_failure=c.artifact(output/"failure.json"), stages=stages, selection=None,
                full_study_completed=False, completed_panel_throughput_claim=False, within_global_bound=within_bound,
                controller_cleanup_errors=execution["cleanup_errors"], exact_owned_processes_absent=absent,
                failure_artifacts=[c.artifact(path) for path in sorted(output.rglob("*failure*.json"))],
                retained_controller_exit=dict(session_id=session, exit_code=exit_code, tool_chunk=chunk,
                    source="Root-provided retained execution; exact process absence independently checked"),
                auditor_models_launched=0, auditor_games_launched=0,
                limitation="Preservation audit only. Unverified or failed stage records are retained, not certified correct; no selection is made.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-plan-sha", required=True)
    parser.add_argument("--controller-session", required=True, type=int)
    parser.add_argument("--controller-exit-code", required=True, type=int)
    parser.add_argument("--controller-chunk", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-failure", action="store_true")
    args = parser.parse_args()
    from tools.future_training_rules_20261021 import verify_runtime_sources
    plan = read(c.PLAN)
    require(verify_runtime_sources() == plan["source_guard"], "saved auditor startup corrected source receipt")
    result = audit_study(plan, c.OUT, expected_plan_sha=args.expected_plan_sha, controller_session=args.controller_session,
                         controller_exit_code=args.controller_exit_code, controller_chunk=args.controller_chunk, allow_failure=args.allow_failure)
    require("torch" not in sys.modules and "torch_directml" not in sys.modules, "saved audit loads no model runtime")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(result)+"\n")
    print(canonical(dict(status=result["status"], selection=result["selection"], diagnostic_decisions_attempted=result.get("diagnostic_decisions_attempted"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
