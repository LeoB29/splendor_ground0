"""One isolated, fixed-state concurrency cell; no game continuation or retries."""
from __future__ import annotations

from pathlib import Path
import math
import struct
import threading
import time
import traceback

from . import common as c
from tools.splendorust_gpu_live_20261105_03 import qualification as q, games


DEADLINE_MESSAGES = frozenset((
    "GPU_LIVE_INTERNAL_DEADLINE: original complete9s budget",
    "Internal fixed-work deadline exceeded",
    "GPU inference callback exceeded original 9s decision deadline",
    "Queued inference expired before dispatch",
    "Completed inference exceeded original callback deadline",
    "Inference complete readback arrived after decision deadline",
    "Full ordinary response exceeded fixed9s internal guard",
    "Complete player response exceeded fixed outer guard",
    "Complete response observed after fixed outer guard",
    "Whole completed operation exceeded dual-clock10s bound",
))


def error_chain(error):
    result = []; seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error)); result.append(dict(type=type(error).__name__, message=str(error)))
        error = error.__cause__ if error.__cause__ is not None else (
            None if error.__suppress_context__ else error.__context__)
    return result


def classify_decision_error(error):
    """Only known decision-clock failures may become an isolated negative cell."""
    chain = error_chain(error)
    found = False
    for row in chain:
        if row["type"] == "TimeoutError" and row["message"] in DEADLINE_MESSAGES:
            found = True
        elif row["type"] == "RuntimeError" and (
                row["message"] == "Inference service failed; no retry" or
                row["message"].startswith("Inference request failed: ")):
            continue
        else:
            return "integrity_failed"
    if not found:
        return "integrity_failed"
    return "deadline_failed" if chain[0]["type"] == "TimeoutError" else "cancelled"


def resolve_group_outcomes(rows, errors):
    """Authorize peer cancellation only against a proven same-group deadline."""
    def failed_clock(row):
        clock = row.get("clock")
        return (clock is not None and clock.get("status") in ("error","timeout") and
                clock.get("completed") is False and clock.get("internal_seconds") == 9. and
                clock.get("outer_seconds") == 10.)
    triggers = [index for index,row in rows.items() if row["status"] == "deadline_failed"
                and failed_clock(row) and index in errors and type(errors[index]) is TimeoutError
                and str(errors[index]) in DEADLINE_MESSAGES]
    for index,row in rows.items():
        if index not in errors or not failed_clock(row) or classify_decision_error(errors[index]) != "cancelled":
            continue
        messages = [item["message"] for item in error_chain(errors[index]) if item["type"] == "TimeoutError"]
        required = 10. if any(message in (
            "Complete player response exceeded fixed outer guard",
            "Complete response observed after fixed outer guard",
            "Whole completed operation exceeded dual-clock10s bound") for message in messages) else 9.
        clock = row["clock"]
        elapsed = clock.get("elapsed_seconds")
        started, ended = clock.get("started_perf_counter"), clock.get("ended_perf_counter")
        if (all(type(value) in (int,float) and math.isfinite(value) for value in (elapsed,started,ended))
                and elapsed == ended-started and elapsed >= required):
            row["initial_status"] = row["status"]
            row["status"] = "deadline_failed"
            row["deadline_proof"] = dict(kind="wrapped_known_deadline_and_own_original_clock_expired",
                required_elapsed_seconds=required,observed_elapsed_seconds=elapsed,
                started_perf_counter=started,ended_perf_counter=ended,known_timeout_messages=messages)
            triggers.append(index)
    for index,row in rows.items():
        if index not in errors:
            continue
        chain = error_chain(errors[index])
        known_poison = (bool(chain) and any(item["message"] in (
            "Evaluator aborted; no replacement or retry","Service closed with outstanding inference") for item in chain)
            and all(item["type"] == "RuntimeError" and (item["message"] in (
                "Inference service failed; no retry","Evaluator aborted; no replacement or retry",
                "Service closed with outstanding inference") or item["message"].startswith("Inference request failed: "))
                for item in chain))
        if row["status"] == "cancelled" or known_poison:
            row["initial_status"] = row["status"]
            row["status"] = "cancelled" if triggers and failed_clock(row) else "integrity_failed"
            row["cancellation_trigger_case_ids"] = [rows[i]["case_id"] for i in triggers]
        elif row["status"] == "deadline_failed" and index not in triggers:
            row["initial_status"] = row["status"]; row["status"] = "integrity_failed"
    if any(row["status"] == "integrity_failed" for row in rows.values()):
        return "integrity_failed"
    return "deadline_failed" if errors else "passed"


def validate_cases(plan):
    cases = plan["cases"]
    c.require(len(cases) == 8 and [row["index"] for row in cases] == list(range(8))
              and [row["case_id"] for row in cases] == [f"g{i:03d}" for i in range(8)], "fixed eight case identities")
    for case in cases:
        c.verify([case["source_initial"]])
        c.require(c.read(c.resolve(case["source_initial"])) == case["snapshot"], "preserved original opening bytes")
        state = q.guard.decode_training_snapshot(case["snapshot"])
        observer = q.rules.observation_payload(state)
        c.require(observer == case["observer"] and not state.terminal
                  and case["request"] == dict(player="search",observer=observer)
                  and c.digest(case["request"]) == case["request_sha256"], "corrected opening request identity")
    return cases


def warmup_runtime(plan, runtime, folder):
    """Five prospective service calls, outside cell timing and without search."""
    q.validate_runtime(runtime)
    folder = Path(folder)
    records = []; error = None; cleanup_error = None
    runtime.service.register("warmup", 0)
    try:
        baseline = None
        for index in range(5):
            def operation(started):
                with runtime.operation("warmup", started=started, kind="service_warmup"):
                    return runtime.service.infer("warmup", plan["cases"][0]["expected_root_tokens"],
                                                 1, deadline=started+9.)
            (output, receipt), clock = q.timed_call(operation, abort=runtime.abort,
                on_receipt=lambda value, i=index:c.save(folder/"warmup_clocks"/f"{i:02d}.json",value))
            row = dict(index=index,output=output,receipt=receipt,clock=clock)
            c.save(folder/"warmup_raw"/f"{index:02d}.json",row); records.append(row)
            packed = b"".join(struct.pack("<83f", *values) for values in output)
            if baseline is None:
                baseline = (output, packed)
            c.require(output == baseline[0] and packed == baseline[1], "exact fixed same-route service warmup outputs")
    except BaseException:
        error = traceback.format_exc()
        raise
    finally:
        try:
            if "warmup" in runtime.service.clients:
                runtime.service.unregister("warmup")
        except BaseException:
            cleanup_error = traceback.format_exc()
            raise
        finally:
            c.save(folder/"warmup.json",dict(status="passed" if error is None and cleanup_error is None else "failed",calls=records,
                planned_calls=5,error=error,cleanup_error=cleanup_error,inside_cell_timing=False,search_decisions=0,no_retry=True))


def _one_case(case, runtime, adapter, folder, group_index):
    client = "case_"+case["case_id"]
    error = None; outcome = None
    try:
        def operation(started):
            c.save(folder/"attempt.json",dict(group_index=group_index,case_id=case["case_id"],
                started_perf_counter=started,observed_unix=time.time(),request_sha256=case["request_sha256"]))
            return q.ordinary(runtime,{"search":adapter},case["request"],started=started,
                client_ids={"search":client}, evidence_callback=lambda raw:c.save(folder/"raw_result.json",raw))
        result, clock = q.timed_call(
            operation,
            abort=runtime.abort, on_receipt=lambda value:c.save(folder/"clock.json",value))
        # Preserve the unchanged full scientific response before downstream gates.
        c.save(folder/"result.json",result)
        fixed = dict(case, observer_sha256=c.digest(case["observer"]))
        root = q.validate_fixed_result(fixed,result); c.save(folder/"root.json",root)
        after = games.actual_after(case["snapshot"],result["action"])
        c.save(folder/"after.json",after)
        q.scientific.audit_decision(case["observer"],result["action"],result["diagnostics"],"search",
            before=case["snapshot"],after=after,expected_quota=512)
        outcome = dict(case_id=case["case_id"],index=case["index"],client_id=client,status="completed",
                       decision_started=True,
                       result_sha256=c.digest(result),clock=clock,request_sha256=case["request_sha256"],
                       after_sha256=c.digest(after),no_retry=True)
    except BaseException as caught:
        error = caught
        clock = c.read(folder/"clock.json") if (folder/"clock.json").exists() else None
        classification = classify_decision_error(caught)
        if clock is None or clock["status"] == "ok" or clock["completed"] is True:
            classification = "integrity_failed"
        outcome = dict(case_id=case["case_id"],index=case["index"],client_id=client,
                       decision_started=True,
                       status=classification,error_chain=error_chain(caught),clock=clock,
                       error=traceback.format_exc(),request_sha256=case["request_sha256"],no_retry=True)
        c.save(folder/"failure.json",outcome)
    finally:
        c.save(folder/"service_calls.json",list(adapter.last_service_calls))
    return outcome, error


def run_cell(plan, spec, runtime, folder, boundary=lambda:None):
    """Run all predetermined groups or stop after the first failed group."""
    q.validate_runtime(runtime)
    cases = validate_cases(plan)
    c.require(spec in plan["cells"] and runtime.route == spec["route"]
              and spec["concurrency"] in (1,2,4,8) and plan["group_gap_seconds"] == 0., "fixed cell dispatch contract")
    groups = spec["groups"]
    c.require(groups == [list(range(i,i+spec["concurrency"])) for i in range(0,8,spec["concurrency"])],
              "fixed disjoint case groups and permanent slots")
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    adapters = {}; outcomes = {}; group_rows = []; status = "passed"
    start = None; end = None
    try:
        for case in cases:
            boundary(); case_folder = folder/case["case_id"]
            c.save(case_folder/"case.json",case); c.save(case_folder/"request.json",case["request"])
            adapters[case["index"]] = runtime.make_adapter("case_"+case["case_id"],case["index"],512)
        for group_index, indexes in enumerate(groups):
            boundary(); group_folder = folder/"groups"/f"{group_index:03d}"
            origin = {}; lock = threading.Lock(); threads = []; rows = {}; errors = {}

            def begin_group():
                origin.update(started_perf_counter=time.perf_counter(),started_unix=time.time())
                c.save(group_folder/"start.json",dict(group_index=group_index,case_indexes=indexes,**origin))

            barrier = threading.Barrier(len(indexes)+1,action=begin_group)

            def execute(index):
                case = cases[index]; case_folder = folder/case["case_id"]
                try:
                    barrier.wait(timeout=10.)
                    boundary()
                    row, error = _one_case(case,runtime,adapters[index],case_folder,group_index)
                    with lock:
                        rows[index] = row
                        if error is not None:errors[index] = error
                except BaseException as error:
                    # Barrier, evidence or guard errors never count as a normal
                    # decision timeout and never authorize a subsequent cell.
                    row = dict(case_id=case["case_id"],index=index,status="integrity_failed",
                               decision_started=(case_folder/"attempt.json").exists(),
                               error_chain=error_chain(error),error=traceback.format_exc(),no_retry=True)
                    with lock:rows[index] = row; errors[index] = error
                    try:c.save(case_folder/"worker_case_failure.json",row)
                    except BaseException:pass  # The outer worker still fails closed.

            for index in indexes:
                thread = threading.Thread(target=execute,args=(index,),name="fixed-case-"+cases[index]["case_id"],daemon=True)
                thread.start(); threads.append(thread)
            barrier.wait(timeout=10.)
            if start is None:start = dict(origin)
            # This is a bounded return/cleanup wait, not a new decision budget.
            # The original timed_call and controller registry still enforce10s.
            join_until = origin["started_perf_counter"]+11.
            for thread in threads:
                thread.join(timeout=max(0.,join_until-time.perf_counter()))
            c.require(not any(thread.is_alive() for thread in threads), "Decision group did not return after its original outer guards")
            c.require(set(rows) == set(indexes), "every barrier-released case has an explicit terminal outcome")
            outcomes.update(rows)
            group_status = resolve_group_outcomes(rows,errors)
            for index in indexes:
                c.save(folder/cases[index]["case_id"]/"outcome.json",rows[index])
            group_row = dict(group_index=group_index,case_indexes=indexes,case_ids=[cases[i]["case_id"] for i in indexes],
                status=group_status,**origin,ended_perf_counter=time.perf_counter(),ended_unix=time.time(),
                attempted=sum(row["decision_started"] for row in rows.values()),completed=sum(row["status"]=="completed" for row in rows.values()),
                outcomes=[rows[i] for i in indexes],no_retry=True,
                wall_scope="barrier origin through all durable case writes; excludes this group summary serialization")
            group_row["wall_seconds"] = group_row["ended_perf_counter"]-origin["started_perf_counter"]
            c.save(group_folder/"summary.json",group_row); group_rows.append(group_row)
            if group_status != "passed":
                status = group_status
                break
        end = dict(ended_perf_counter=time.perf_counter(),ended_unix=time.time())
        c.require(start is not None, "at least one planned diagnostic group")
        summary = dict(status=status,cell_id=spec["cell_id"],route=spec["route"],concurrency=spec["concurrency"],
            planned_case_count=8,attempted_case_ids=[cases[i]["case_id"] for i in sorted(outcomes) if outcomes[i]["decision_started"]],
            completed_case_ids=[cases[i]["case_id"] for i in sorted(outcomes) if outcomes[i]["status"]=="completed"],
            never_started_case_ids=[case["case_id"] for case in cases if not outcomes.get(case["index"],{}).get("decision_started")],
            groups=group_rows,**start,**end,wall_seconds=end["ended_perf_counter"]-start["started_perf_counter"],
            includes="group barrier publication, all decision observations, validation, durable case/group writes, intergroup bookkeeping",
            excludes="model and native startup, fixed service warmup, final cell-summary serialization and cleanup",
            intentional_intergroup_gap_seconds=0.,full_cell_completed=status=="passed" and len(outcomes)==8,
            exact_C1_reproduction_pending=True,eligible_for_selection=False,no_retry=True,game_continuations=0)
        c.require(status != "passed" or summary["full_cell_completed"], "successful cell completes all8 original cases")
        c.save(folder/"cell_summary.json",summary)
        return summary
    finally:
        # A failed service can have already poisoned the pending siblings. All
        # group results are retained; runtime ownership never crosses a cell.
        for adapter in adapters.values():
            adapter.close()
