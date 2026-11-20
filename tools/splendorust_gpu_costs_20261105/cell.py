"""Saved-position cost diagnostic and isolated physical batches; no new games.

Native clients are prepared once outside the panel timer, then reused across
fixed waves. Every search keeps its own original complete-decision clock.
"""
from __future__ import annotations

from pathlib import Path
import struct
import threading
import time
import traceback

from . import common as c
from tools.splendorust_gpu_throughput_20261105 import qualification as q, games
from tools.splendorust_gpu_throughput_20261105.cell import (
    DEADLINE_MESSAGES, error_chain, classify_decision_error, resolve_group_outcomes,
)


def record_panel(runtime, path, value, owner):
    """Time unchanged durable evidence writes; control calls save directly."""
    if runtime.observer.enabled:
        with runtime.observer.span("case_recording", owner+"/"+Path(path).name):
            return c.save(path, value)
    return c.save(path, value)


def expected_move(case, route):
    item = case["expected_results"][route]
    c.verify([item])
    row = c.read(c.resolve(item))
    c.require(row["before"] == case["snapshot"] and row["observer"] == case["observer"],
              "accepted same-state route response")
    return row


def validate_cases(plan):
    cases = plan["cases"]
    c.require(len(cases) == 32 and [row["index"] for row in cases] == list(range(32))
              and [row["case_id"] for row in cases] == [f"p{i:03d}" for i in range(32)],
              "fixed thirty-two case identities")
    for case in cases:
        c.require(case["slot"] == case["index"] % 8, "permanent source-game physical row")
        c.verify([case["source_before"]])
        source = c.read(c.resolve(case["source_before"]))
        c.require(source["before"] == case["snapshot"], "preserved original beforestate")
        state = q.guard.decode_training_snapshot(case["snapshot"])
        observer = q.rules.observation_payload(state)
        c.require(observer == case["observer"] and not state.terminal
                  and case["request"] == dict(player="search", observer=observer)
                  and c.digest(case["request"]) == case["request_sha256"],
                  "corrected saved request identity")
        for route in ("cpu", "gpu"):
            row = expected_move(case, route)
            q.validate_fixed_result(dict(case, observer_sha256=c.digest(observer)), row["result"])
            c.require(games.actual_after(case["snapshot"], row["result"]["action"]) == row["after"],
                      "accepted route independent afterstate")
    return cases


def warmup_runtime(plan, runtime, folder):
    """Five prospective service calls; no search and outside panel timing."""
    q.validate_runtime(runtime)
    folder = Path(folder)
    records = []; error = None; cleanup_error = None
    runtime.observer.mark("warmup_start")
    runtime.service.register("warmup", 0)
    try:
        baseline = None
        for index in range(5):
            def operation(started):
                with runtime.operation("warmup", started=started, kind="service_warmup"):
                    return runtime.service.infer("warmup", plan["cases"][0]["expected_root_tokens"],
                                                 1, deadline=started+c.INTERNAL_SECONDS)
            (output, receipt), clock = q.timed_call(operation, abort=runtime.abort,
                on_receipt=lambda value, i=index: c.save(folder/"warmup_clocks"/f"{i:02d}.json", value))
            row = dict(index=index, output=output, receipt=receipt, clock=clock)
            c.save(folder/"warmup_raw"/f"{index:02d}.json", row); records.append(row)
            packed = b"".join(struct.pack("<83f", *values) for values in output)
            if baseline is None:
                baseline = (output, packed)
            c.require(output == baseline[0] and packed == baseline[1], "exact same-route service warmups")
    except BaseException:
        error = traceback.format_exc(); raise
    finally:
        try:
            if "warmup" in runtime.service.clients:
                runtime.service.unregister("warmup")
        except BaseException:
            cleanup_error = traceback.format_exc(); raise
        finally:
            runtime.observer.mark("warmup_end")
            c.save(folder/"warmup.json", dict(status="passed" if error is None and cleanup_error is None else "failed",
                calls=records, planned_calls=5, error=error, cleanup_error=cleanup_error,
                inside_cell_timing=False, search_decisions=0, no_retry=True))


def _one_case(case, runtime, adapter, folder, group_index):
    client = adapter.client_id
    error = None; outcome = None
    try:
        def operation(started):
            record_panel(runtime, folder/"attempt.json", dict(group_index=group_index, case_id=case["case_id"],
                client_id=client, slot=adapter.slot, started_perf_counter=started,
                observed_unix=time.time(), request_sha256=case["request_sha256"]), case["case_id"])
            return q.ordinary(runtime, {"search": adapter}, case["request"], started=started,
                client_ids={"search": client}, evidence_callback=lambda raw: record_panel(runtime, folder/"raw_result.json", raw, case["case_id"]))
        result, clock = q.timed_call(operation, abort=runtime.abort,
            on_receipt=lambda value: record_panel(runtime, folder/"clock.json", value, case["case_id"]))
        record_panel(runtime, folder/"result.json", result, case["case_id"])
        root = q.validate_fixed_result(dict(case, observer_sha256=c.digest(case["observer"])), result)
        record_panel(runtime, folder/"root.json", root, case["case_id"])
        after = games.actual_after(case["snapshot"], result["action"])
        record_panel(runtime, folder/"after.json", after, case["case_id"])
        q.scientific.audit_decision(case["observer"], result["action"], result["diagnostics"], "search",
            before=case["snapshot"], after=after, expected_quota=512)
        reference = expected_move(case, runtime.route)
        c.require(result == reference["result"] and after == reference["after"],
                  "exact accepted same-route full decision and afterstate")
        outcome = dict(case_id=case["case_id"], index=case["index"], client_id=client, slot=adapter.slot,
            status="completed", decision_started=True, result_sha256=c.digest(result), clock=clock,
            request_sha256=case["request_sha256"], after_sha256=c.digest(after),
            exact_accepted_source=True, expected_result=case["expected_results"][runtime.route], no_retry=True)
    except BaseException as caught:
        error = caught
        clock = c.read(folder/"clock.json") if (folder/"clock.json").exists() else None
        classification = classify_decision_error(caught)
        if clock is None or clock["status"] == "ok" or clock["completed"] is True:
            classification = "integrity_failed"
        outcome = dict(case_id=case["case_id"], index=case["index"], client_id=client, slot=adapter.slot,
            decision_started=True, status=classification, error_chain=error_chain(caught), clock=clock,
            error=traceback.format_exc(), request_sha256=case["request_sha256"], no_retry=True)
        record_panel(runtime, folder/"failure.json", outcome, case["case_id"])
    finally:
        # ProxySearchOpponent resets this list at each choose; it is one decision.
        record_panel(runtime, folder/"service_calls.json", list(adapter.last_service_calls), case["case_id"])
    return outcome, error


def validate_spec(plan, spec):
    c.require(spec in plan["cells"] and spec["concurrency"] == 16
              and plan["group_gap_seconds"] == 0., "fixed concurrency dispatch")
    c.require((spec["route"], spec["scheduler"], spec["concurrency"]) == ("gpu", "row_fifo32", 16)
              and type(spec["observed"]) is bool and spec["phase"] in ("overhead", "diagnostic"),
              "unchanged GPU16 with explicit observer setting")
    groups = [list(range(i, i+spec["concurrency"])) for i in range(0, 32, spec["concurrency"])]
    c.require(spec["groups"] == groups, "fixed complete disjoint case waves")
    return groups


def run_cell(plan, spec, runtime, folder, boundary=lambda: None):
    q.validate_runtime(runtime)
    cases = validate_cases(plan); groups = validate_spec(plan, spec)
    c.require(runtime.route == spec["route"] and runtime.identity["scheduler"] == spec["scheduler"],
              "actual route and scheduler")
    c.require(runtime.observer.enabled is spec["observed"], "actual observer matches frozen cell")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    adapters = {}; outcomes = {}; group_rows = []; status = "passed"; start = None
    try:
        for case in cases:
            c.save(folder/case["case_id"]/"case.json", case)
            c.save(folder/case["case_id"]/"request.json", case["request"])
        for index in range(spec["concurrency"]):
            boundary()
            adapters[index] = runtime.make_adapter(f"worker_{index}", index % 8, 512)
        c.save(folder/"adapters_ready.json", dict(count=len(adapters), prepared_before_panel=True,
            clients=[dict(index=i, client_id=a.client_id, slot=a.slot) for i, a in adapters.items()],
            time=c.now(), observed_perf_counter=time.perf_counter()))
        for group_index, indexes in enumerate(groups):
            boundary(); group_folder = folder/"groups"/f"{group_index:03d}"
            origin = {}; lock = threading.Lock(); threads = []; rows = {}; errors = {}

            def begin_group():
                if group_index == 0:
                    runtime.observer.mark("panel_start")
                origin.update(started_perf_counter=time.perf_counter(), started_unix=time.time())
                record_panel(runtime, group_folder/"start.json", dict(group_index=group_index, case_indexes=indexes, **origin), f"group:{group_index}")

            barrier = threading.Barrier(len(indexes)+1, action=begin_group)

            def execute(index):
                case = cases[index]; case_folder = folder/case["case_id"]
                try:
                    barrier.wait(timeout=10.); boundary()
                    row, error = _one_case(case, runtime, adapters[index % spec["concurrency"]], case_folder, group_index)
                    with lock:
                        rows[index] = row
                        if error is not None: errors[index] = error
                except BaseException as error:
                    row = dict(case_id=case["case_id"], index=index, status="integrity_failed",
                        decision_started=(case_folder/"attempt.json").exists(), error_chain=error_chain(error),
                        error=traceback.format_exc(), no_retry=True)
                    with lock: rows[index] = row; errors[index] = error
                    try: record_panel(runtime, case_folder/"worker_case_failure.json", row, case["case_id"])
                    except BaseException: pass

            for index in indexes:
                thread = threading.Thread(target=execute, args=(index,), name="saved-case-"+cases[index]["case_id"], daemon=True)
                thread.start(); threads.append(thread)
            barrier.wait(timeout=10.)
            if start is None: start = dict(origin)
            join_until = origin["started_perf_counter"]+c.OUTER_SECONDS+1.
            for thread in threads:
                thread.join(timeout=max(0., join_until-time.perf_counter()))
            c.require(not any(thread.is_alive() for thread in threads), "group returned under original outer safeguards")
            c.require(set(rows) == set(indexes), "every released case has one outcome")
            outcomes.update(rows); group_status = resolve_group_outcomes(rows, errors)
            for index in indexes:
                record_panel(runtime, folder/cases[index]["case_id"]/"outcome.json", rows[index], cases[index]["case_id"])
            group_row = dict(group_index=group_index, case_indexes=indexes, case_ids=[cases[i]["case_id"] for i in indexes],
                status=group_status, **origin, ended_perf_counter=time.perf_counter(), ended_unix=time.time(),
                attempted=sum(row["decision_started"] for row in rows.values()),
                completed=sum(row["status"] == "completed" for row in rows.values()),
                outcomes=[rows[i] for i in indexes], no_retry=True,
                wall_scope="barrier origin through durable case writes; excludes group summary serialization")
            group_row["wall_seconds"] = group_row["ended_perf_counter"]-origin["started_perf_counter"]
            record_panel(runtime, group_folder/"summary.json", group_row, f"group:{group_index}"); group_rows.append(group_row)
            if group_status != "passed": status = group_status; break
        end = dict(ended_perf_counter=time.perf_counter(), ended_unix=time.time())
        runtime.observer.mark("panel_end")
        c.require(start is not None, "at least one prospective group")
        summary = dict(status=status, cell_id=spec["cell_id"], route=spec["route"], scheduler=spec["scheduler"],
            concurrency=spec["concurrency"], observed=spec["observed"], phase=spec["phase"], planned_case_count=32,
            attempted_case_ids=[cases[i]["case_id"] for i in sorted(outcomes) if outcomes[i]["decision_started"]],
            completed_case_ids=[cases[i]["case_id"] for i in sorted(outcomes) if outcomes[i]["status"] == "completed"],
            never_started_case_ids=[case["case_id"] for case in cases if not outcomes.get(case["index"], {}).get("decision_started")],
            groups=group_rows, **start, **end, wall_seconds=end["ended_perf_counter"]-start["started_perf_counter"],
            includes="barrier publication, decision observations, validation, durable writes and intergroup bookkeeping",
            excludes="model and native startup, fixed warmup, final cell summary serialization and cleanup",
            intentional_intergroup_gap_seconds=0., full_cell_completed=status == "passed" and len(outcomes) == 32,
            exact_accepted_source_pending_independent_audit=True, eligible_for_selection=False,
            no_retry=True, game_continuations=0)
        c.require(status != "passed" or summary["full_cell_completed"], "passed cell completes all32 cases")
        c.save(folder/"cell_summary.json", summary)
        return summary
    finally:
        for adapter in adapters.values(): adapter.close()


def isolated_batches(plan):
    """Four unchanged B8 input groups, with exact accepted GPU row outputs."""
    cases = validate_cases(plan)
    batches = []
    for batch_index, start in enumerate(range(0, 32, 8)):
        indexes = list(range(start, start+8))
        tokens = [value for index in indexes for value in cases[index]["expected_root_tokens"]]
        outputs = [expected_move(cases[index], "gpu")["result"]["diagnostics"]["phases"][0]["inference"][0]["output83"]
                   for index in indexes]
        c.require(len(tokens) == 8*31*48 and all(cases[index]["slot"] == index % 8 for index in indexes),
                  "same saved physical B8 row inputs")
        batches.append(dict(batch_index=batch_index, case_indexes=indexes,
            case_ids=[cases[index]["case_id"] for index in indexes], input_tokens=tokens,
            input_sha256=c.digest(tokens), expected_outputs=outputs,
            expected_source_results=[cases[index]["expected_results"]["gpu"] for index in indexes]))
    return batches


def run_isolated(plan, spec, runtime, folder, boundary=lambda: None):
    """Fresh loaded-runtime calls; journal/export and raw result writes stay outside each call clock."""
    from . import design
    c.require(spec in plan["isolated"] and plan["isolated"] == design.isolated()
              and runtime.route == "gpu" and runtime.identity["scheduler"] == "row_fifo32"
              and runtime.observer.enabled is True, "frozen isolated original GPU16 backend")
    c.require(not runtime.adapters and not runtime.service.clients and not runtime.service.pending,
              "isolated physical calls have no native or concurrent search clients")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    batches = isolated_batches(plan); c.save(folder/"batches.json", batches)
    rows = []; phase_rows = {}
    for phase, cycles in (("warmup", spec["warmup_cycles"]), ("measured", spec["measured_cycles"])):
        runtime.observer.mark("isolated_"+phase+"_start")
        started, wall = time.perf_counter(), time.time()
        selected = []
        for cycle in range(cycles):
            for batch in batches:
                boundary()
                index = cycle*4+batch["batch_index"]
                path = folder/phase/f"{index:04d}"
                observer_start = runtime.observer.mark(f"isolated_{phase}_{index:04d}_start")
                def operation(origin):
                    with runtime.operation("isolated_physical", started=origin, kind="isolated_physical"):
                        return runtime.service.physical_forward(batch["input_tokens"],
                            deadline=origin+c.INTERNAL_SECONDS)
                output, clock = q.timed_call(operation, abort=runtime.abort,
                    on_receipt=lambda receipt, target=path: c.save(target/"clock.json", receipt))
                observer_end = runtime.observer.mark(f"isolated_{phase}_{index:04d}_end")
                row = dict(phase=phase, index=index, cycle=cycle, batch_index=batch["batch_index"],
                    case_indexes=batch["case_indexes"], physical_batch=8, input_sha256=batch["input_sha256"],
                    outputs=output, clock=clock, observer_start=observer_start, observer_end=observer_end, no_retry=True)
                # The complete raw result is preserved before the exact-output gate.
                c.save(path/"result.json", row)
                c.require(output == batch["expected_outputs"] and
                    b"".join(struct.pack("<83f", *values) for values in output) ==
                    b"".join(struct.pack("<83f", *values) for values in batch["expected_outputs"]),
                    "exact isolated saved physical rows and float32 bytes")
                selected.append(row); rows.append(row)
        ended, ended_wall = time.perf_counter(), time.time()
        runtime.observer.mark("isolated_"+phase+"_end")
        phase_row = dict(phase=phase, cycles=cycles, calls=len(selected), physical_rows=len(selected)*8,
            started_perf_counter=started, ended_perf_counter=ended, started_unix=wall, ended_unix=ended_wall,
            loop_wall_seconds=ended-started, elapsed_complete_call_seconds=[row["clock"]["elapsed_seconds"] for row in selected],
            loop_scope="Includes between-call correctness checks and durable result/clock writes; not pure inference timing",
            per_call_scope="Original timed call through complete output return; result and clock files written afterward",
            exact_saved_outputs=True, no_retry=True)
        c.save(folder/phase/"summary.json", phase_row); phase_rows[phase] = phase_row
    c.require(len(rows) == 148 and len(phase_rows["measured"]["elapsed_complete_call_seconds"]) == 128,
              "fixed twenty warmup plus128 measured physical calls")
    result = dict(status="isolated_batches_passed", cell_id=spec["cell_id"], route="gpu", scheduler="row_fifo32",
        observed=True, specification=spec, warmup_calls=20, measured_calls=128, physical_batch=8,
        phases=phase_rows, batches_sha256=c.digest(batches), all_outputs_exact=True,
        ordinary_decisions=0, new_games=0, training_updates=0, no_retry=True)
    c.save(folder/"summary.json", result)
    return result

