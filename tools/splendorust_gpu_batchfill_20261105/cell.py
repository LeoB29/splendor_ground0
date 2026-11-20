"""Saved-position searches and actual collision qualification; no new games.

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
            c.save(folder/"warmup.json", dict(status="passed" if error is None and cleanup_error is None else "failed",
                calls=records, planned_calls=5, error=error, cleanup_error=cleanup_error,
                inside_cell_timing=False, search_decisions=0, no_retry=True))


def _one_case(case, runtime, adapter, folder, group_index):
    client = adapter.client_id
    error = None; outcome = None
    try:
        def operation(started):
            c.save(folder/"attempt.json", dict(group_index=group_index, case_id=case["case_id"],
                client_id=client, slot=adapter.slot, started_perf_counter=started,
                observed_unix=time.time(), request_sha256=case["request_sha256"]))
            return q.ordinary(runtime, {"search": adapter}, case["request"], started=started,
                client_ids={"search": client}, evidence_callback=lambda raw: c.save(folder/"raw_result.json", raw))
        result, clock = q.timed_call(operation, abort=runtime.abort,
            on_receipt=lambda value: c.save(folder/"clock.json", value))
        c.save(folder/"result.json", result)
        root = q.validate_fixed_result(dict(case, observer_sha256=c.digest(case["observer"])), result)
        c.save(folder/"root.json", root)
        after = games.actual_after(case["snapshot"], result["action"])
        c.save(folder/"after.json", after)
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
        c.save(folder/"failure.json", outcome)
    finally:
        # ProxySearchOpponent resets this list at each choose; it is one decision.
        c.save(folder/"service_calls.json", list(adapter.last_service_calls))
    return outcome, error


def validate_spec(plan, spec):
    c.require(spec in plan["cells"] and spec["concurrency"] in (8, 16, 32)
              and plan["group_gap_seconds"] == 0., "fixed concurrency dispatch")
    c.require((spec["route"], spec["scheduler"], spec["concurrency"]) in (
        ("cpu", "original8", 8), ("gpu", "original8", 8),
        ("gpu", "row_fifo32", 16), ("gpu", "row_fifo32", 32)), "only four frozen configurations")
    groups = [list(range(i, i+spec["concurrency"])) for i in range(0, 32, spec["concurrency"])]
    c.require(spec["groups"] == groups, "fixed complete disjoint case waves")
    return groups


def run_cell(plan, spec, runtime, folder, boundary=lambda: None):
    q.validate_runtime(runtime)
    cases = validate_cases(plan); groups = validate_spec(plan, spec)
    c.require(runtime.route == spec["route"] and runtime.identity["scheduler"] == spec["scheduler"],
              "actual route and scheduler")
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
                origin.update(started_perf_counter=time.perf_counter(), started_unix=time.time())
                c.save(group_folder/"start.json", dict(group_index=group_index, case_indexes=indexes, **origin))

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
                    try: c.save(case_folder/"worker_case_failure.json", row)
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
                c.save(folder/cases[index]["case_id"]/"outcome.json", rows[index])
            group_row = dict(group_index=group_index, case_indexes=indexes, case_ids=[cases[i]["case_id"] for i in indexes],
                status=group_status, **origin, ended_perf_counter=time.perf_counter(), ended_unix=time.time(),
                attempted=sum(row["decision_started"] for row in rows.values()),
                completed=sum(row["status"] == "completed" for row in rows.values()),
                outcomes=[rows[i] for i in indexes], no_retry=True,
                wall_scope="barrier origin through durable case writes; excludes group summary serialization")
            group_row["wall_seconds"] = group_row["ended_perf_counter"]-origin["started_perf_counter"]
            c.save(group_folder/"summary.json", group_row); group_rows.append(group_row)
            if group_status != "passed": status = group_status; break
        end = dict(ended_perf_counter=time.perf_counter(), ended_unix=time.time())
        c.require(start is not None, "at least one prospective group")
        summary = dict(status=status, cell_id=spec["cell_id"], route=spec["route"], scheduler=spec["scheduler"],
            concurrency=spec["concurrency"], planned_case_count=32,
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


def collision_qualification(plan, runtime, folder, boundary=lambda: None):
    """Actual GPU service: 16/32 logical clients, two original-clock calls each."""
    c.require(runtime.route == "gpu" and runtime.identity["scheduler"] == "row_fifo32",
              "candidate scheduler collision qualification")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    all_rows = []; group_records = []
    for count in (16, 32):
        boundary(); clients = {}; errors = {}; rows = {}; lock = threading.Lock()
        origin = {}; group_folder = folder/f"clients_{count}"
        for index in range(count):
            client = f"collision{count}_{index:02d}"; clients[index] = client
            runtime.service.register(client, index % 8)
        def start_group():
            origin.update(started_perf_counter=time.perf_counter(), started_unix=time.time())
            c.save(group_folder/"start.json", dict(clients=count, **origin))
        barrier = threading.Barrier(count+1, action=start_group)
        def execute(index):
            client = clients[index]; case = plan["cases"][index]
            try:
                barrier.wait(timeout=10.)
                expected = expected_move(case, "gpu")["result"]["diagnostics"]["phases"][0]["inference"][0]["outputs"]
                for repetition in range(2):
                    boundary()
                    def operation(started):
                        with runtime.operation(client, started=started, kind="collision_qualification"):
                            return runtime.service.infer(client, case["expected_root_tokens"], 1,
                                deadline=started+c.INTERNAL_SECONDS)
                    (output, receipt), clock = q.timed_call(operation, abort=runtime.abort,
                        on_receipt=lambda value, rep=repetition: c.save(group_folder/client/f"clock_{rep}.json", value))
                    row = dict(client_id=client, client_index=index, slot=index % 8, repetition=repetition,
                        case_id=case["case_id"], inputs=case["expected_root_tokens"], outputs=output,
                        expected_result=case["expected_results"]["gpu"], receipt=receipt, clock=clock)
                    c.save(group_folder/client/f"result_{repetition}.json", row)
                    c.require(output == expected and struct.pack("<83f", *output[0]) == struct.pack("<83f", *expected[0]),
                              "exact accepted collision service output")
                    with lock: rows[(index, repetition)] = row
            except BaseException as error:
                with lock: errors[index] = traceback.format_exc()
                try: c.save(group_folder/client/"failure.json", dict(error=traceback.format_exc(), no_retry=True))
                finally: runtime.abort()
        threads = [threading.Thread(target=execute, args=(index,), name=clients[index], daemon=True) for index in range(count)]
        try:
            for thread in threads: thread.start()
            barrier.wait(timeout=10.)
            end = origin["started_perf_counter"]+2*c.OUTER_SECONDS+1.
            for thread in threads: thread.join(timeout=max(0., end-time.perf_counter()))
            c.require(not any(thread.is_alive() for thread in threads), "bounded collision qualification returned")
            c.require(not errors and len(rows) == count*2, "all collision requests completed exactly")
            values = [rows[(i, r)] for i in range(count) for r in range(2)]
            all_rows.extend(values)
            group = dict(clients=count, callbacks=len(values), status="passed", **origin,
                ended_perf_counter=time.perf_counter(), source_case_ids=[plan["cases"][i]["case_id"] for i in range(count)],
                same_rows_collide=True, repetitions=2, no_retry=True)
            c.save(group_folder/"summary.json", group); group_records.append(group)
        finally:
            for client in clients.values():
                if client in runtime.service.clients: runtime.service.unregister(client)
    summary = dict(status="actual_collision_service_qualified", route="gpu", scheduler="row_fifo32",
        groups=group_records, callbacks=len(all_rows), exact_source_outputs=True, search_decisions=0,
        games=0, training_updates=0, no_retry=True)
    c.save(folder/"summary.json", summary)
    return summary
