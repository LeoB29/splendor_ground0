"""Authenticate separate timing sidecars without changing deadline receipts.

Overlapping wall/CPU scopes remain separate. Failed or killed requests can have
partial/missing sidecars; they are described, never promoted to completed data.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics

from tools.splendorust_component_timing_20261025.analyze import interval_union_ns

PARENT_SCHEMA = "splendorust_graph_timing_parent_v1"
CHILD_SCHEMA = "splendorust_graph_timing_child_v1"
GRAPH_IDENTITY = "c4d572908ddb3c216689c702bb00367dc263c9f1374be8bd397250e0b65e6ba2"
PARENT_LABELS = {"initial_sample", "periodic_sample", "aggregate_guard", "resource_guard_check",
                 "resource_guard", "stdin_write", "stdin_flush", "stdout_raw_read"}
SUPERVISION_LABELS = PARENT_LABELS - {"stdin_write", "stdin_flush", "stdout_raw_read"}
CHILD_LABELS = {"inference", "native_write", "native_flush", "native_read_wait", "callback_hash",
                "observer_encode", "decode_observer"}
CHILD_FIELDS = ("calls", "inclusive_wall_ns", "inclusive_cpu_ns", "max_wall_ns", "failures",
                "exclusive_wall_ns", "exclusive_cpu_ns")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def require(value, message):
    if not value:
        raise ValueError("Graph timing analysis: " + message)


def _int(value, name, *, positive=False):
    require(type(value) is int and (1 if positive else 0) <= value < 2**63, name + " integer bound")
    return value


def _finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), name + " finite numeric")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(value, name, *, nullable=False):
    require(nullable and value is None or type(value) is str and SHA256.fullmatch(value) is not None,
            name + " SHA256")


def _validate_parent(parent, clock, *, completed):
    require(type(parent) is dict and parent.get("schema") == PARENT_SCHEMA, "parent schema")
    require(parent.get("clock") == "perf_counter_ns" and parent.get("process_cpu_clock") == "process_time_ns"
            and parent.get("thread_cpu_clock") == "thread_time_ns", "parent clocks")
    require(parent.get("request_id") == clock["request_id"] and
            parent.get("times_are_diagnostic_not_trusted_deadline_receipts") is True and
            parent.get("measurement_units") == "nanoseconds", "parent request/scope")
    for field in ("pid", "child_pid"):
        _int(parent.get(field), field, positive=True)
    start = _int(parent.get("request_start_ns"), "trusted origin")
    wrapper_start = _int(parent.get("wrapper_start_ns"), "wrapper start")
    wrapper_end = _int(parent.get("wrapper_end_ns"), "wrapper end")
    require(wrapper_start <= start <= wrapper_end, "parent wrapper order")
    snapshot = _int(parent.get("profile_snapshot_ns"), "immutable profile snapshot")
    require(wrapper_end <= snapshot, "profile snapshot precedes wrapper end")
    require(parent.get("request_capture_maximum_bytes") == 1024*1024 and
            type(parent.get("request_raw_complete")) is bool and type(parent.get("request_raw_overflow")) is bool,
            "bounded raw request capture")
    require(_int(parent.get("request_raw_bytes"), "raw request byte count") <= 1024*1024, "raw request capture bound")
    _sha(parent.get("request_raw_sha256"), "captured raw request")
    require(not completed or parent["request_raw_complete"] and not parent["request_raw_overflow"],
            "successful request capture incomplete")
    for field in ("wrapper_process_cpu_ns", "wrapper_thread_cpu_ns"):
        _int(parent.get(field), field)
    elapsed, arrival = clock.get("receipt_elapsed_seconds"), parent.get("receipt_arrival_ns")
    reconstructed = None if elapsed is None else start+round(_finite(elapsed, "receipt elapsed")*1e9)
    require(parent.get("receipt_reconstructed_arrival_ns") == reconstructed, "reconstructed receipt differs")
    if arrival is not None:
        _int(arrival, "arrival")
        require(arrival == reconstructed and start <= arrival <= wrapper_end, "raw timestamp/receipt coupling")
    elif completed:
        raise ValueError("Graph timing analysis: successful raw arrival missing")
    raw_arrivals = parent.get("raw_arrivals_ns")
    require(type(raw_arrivals) is list and len(raw_arrivals) <= 256, "bounded raw timestamps")
    for stamp in raw_arrivals:
        _int(stamp, "raw timestamp")
    require(arrival is None or arrival in raw_arrivals, "raw arrival missing from observations")
    require(parent.get("maximum_spans") == 256, "parent span bound")
    dropped = _int(parent.get("dropped_spans"), "dropped span count")
    pending = parent.get("pending_spans_at_snapshot")
    require(type(pending) is dict and set(pending) <= PARENT_LABELS, "bounded pending span labels")
    for count in pending.values():
        _int(count, "pending count", positive=True)
    require(not completed or not dropped and not pending, "successful attribution requires complete parent spans")
    spans, totals = parent.get("spans"), parent.get("totals")
    require(type(spans) is list and len(spans) <= 256 and type(totals) is dict and set(totals) <= PARENT_LABELS,
            "bounded fixed parent labels")
    calculated = {}
    for span in spans:
        require(type(span) is dict and set(span) == {"label", "start_ns", "end_ns", "process_cpu_ns", "thread_cpu_ns"},
                "parent span fields")
        label = span["label"]
        require(label in PARENT_LABELS, "unknown parent label")
        for field in ("start_ns", "end_ns", "process_cpu_ns", "thread_cpu_ns"):
            _int(span[field], "span " + field)
        require(span["start_ns"] <= span["end_ns"] <= snapshot, "span order")
        value = calculated.setdefault(label, {"calls": 0, "wall_ns": 0, "process_cpu_ns": 0, "thread_cpu_ns": 0})
        value["calls"] += 1
        value["wall_ns"] += span["end_ns"]-span["start_ns"]
        value["process_cpu_ns"] += span["process_cpu_ns"]
        value["thread_cpu_ns"] += span["thread_cpu_ns"]
    if not dropped:
        require(calculated == totals, "parent totals differ from preserved spans")
    else:
        require(sum(item.get("calls", 0) for item in totals.values())-len(spans) == dropped,
                "dropped span total count")
        require(all(label in totals and all(totals[label].get(key, -1) >= value for key, value in item.items())
                    for label, item in calculated.items()), "retained spans exceed full aggregates")
    for total in totals.values():
        require(type(total) is dict and set(total) == {"calls", "wall_ns", "process_cpu_ns", "thread_cpu_ns"},
                "parent total fields")
        for field, value in total.items():
            _int(value, "parent total " + field)
    if arrival is not None and not dropped:
        require(any(s["label"] == "stdout_raw_read" and s["end_ns"] == arrival for s in spans),
                "raw timestamp does not end a recorded read")
    for label in ("stdin_write", "stdin_flush"):
        selected = [s for s in spans if s["label"] == label]
        for suffix, function, key in (("start", min, "start_ns"), ("end", max, "end_ns")):
            expected = function((s[key] for s in selected), default=None)
            require(parent.get(label+"_"+suffix+"_ns") == expected, "dispatch marker differs from raw spans")
    dispatch = parent.get("dispatch_complete_ns")
    require(dispatch == parent.get("stdin_flush_end_ns"), "dispatch/flush binding")
    if completed:
        require(type(dispatch) is int and start <= parent["stdin_write_start_ns"] <= parent["stdin_write_end_ns"]
                <= parent["stdin_flush_start_ns"] <= dispatch <= snapshot, "complete serial dispatch markers")
    metrics = {"wrapper_process_cpu_ns": parent["wrapper_process_cpu_ns"],
               "wrapper_thread_cpu_ns": parent["wrapper_thread_cpu_ns"], "labels": {},
               "dropped_spans": dropped, "pending_spans": sum(pending.values()),
               "wall_unions_are_partial": bool(dropped or pending)}
    if dispatch is not None:
        metrics["start_to_dispatch_complete_ns"] = dispatch-start
    for label, total in totals.items():
        selected = [s for s in spans if s["label"] == label]
        item = {"calls": total["calls"], "unclipped_inclusive_wall_ns": total["wall_ns"],
                "unclipped_process_cpu_ns": total["process_cpu_ns"], "unclipped_thread_cpu_ns": total["thread_cpu_ns"]}
        if arrival is not None:
            contained = [s for s in selected if start <= s["start_ns"] <= s["end_ns"] <= arrival]
            item.update(clipped_wall_union_ns=interval_union_ns([(s["start_ns"], s["end_ns"]) for s in selected], start, arrival),
                        fully_contained_span_thread_cpu_ns=sum(s["thread_cpu_ns"] for s in contained),
                        boundary_crossing_spans=len(selected)-len(contained))
        metrics["labels"][label] = item
    if arrival is not None:
        metrics["supervision_wall_union_within_trusted_ns"] = interval_union_ns(
            [(s["start_ns"], s["end_ns"]) for s in spans if s["label"] in SUPERVISION_LABELS], start, arrival)
    return start, arrival, metrics


def _validate_handle(profile, child, *, completed, startup=False):
    require(type(profile) is dict and profile.get("schema") == CHILD_SCHEMA, "child handle schema")
    require(profile.get("pid") == child["pid"] and profile.get("clock") == "perf_counter_ns" and
            profile.get("cpu_clock") == "process_time_ns" and profile.get("measurement_units") == "nanoseconds",
            "child handle identity/clocks")
    require(profile.get("request_read_complete_ns") == (None if startup else child["request_read_complete_ns"]),
            "child read marker binding")
    for field in ("times_are_diagnostic_not_trusted_deadline_receipts", "inclusive_labels_overlap",
                  "exclusive_totals_exclude_nested_components"):
        require(profile.get(field) is True, "child measurement scope " + field)
    require(profile.get("cpu_scope") == "current_python_process_only_excludes_native_rust_and_parent",
            "child CPU scope")
    require(type(profile.get("handle_failed")) is bool and (not completed or not profile["handle_failed"]),
            "handle outcome mismatch")
    begin = _int(profile.get("handle_start_ns"), "handle start")
    end = _int(profile.get("handle_end_ns"), "handle end")
    require(child["request_read_complete_ns"] <= begin <= end, "child handle order")
    wall, cpu = end-begin, _int(profile.get("handle_cpu_ns"), "handle CPU")
    _int(profile.get("component_nested_calls"), "nested calls")
    require(profile.get("total_fields") == list(CHILD_FIELDS), "fixed component fields")
    totals = profile.get("totals")
    require(type(totals) is dict and set(totals) == CHILD_LABELS, "fixed component labels")
    labels = {}
    for label, values in totals.items():
        require(type(values) is list and len(values) == len(CHILD_FIELDS), "component array bound")
        for field, value in zip(CHILD_FIELDS, values):
            _int(value, "component " + field)
        item = dict(zip(CHILD_FIELDS, values))
        require((item["calls"] > 0 or not any(values)) and item["failures"] <= item["calls"]
                and (not completed or item["failures"] == 0)
                and item["max_wall_ns"] <= item["inclusive_wall_ns"]
                and item["exclusive_wall_ns"] <= item["inclusive_wall_ns"]
                and item["exclusive_cpu_ns"] <= item["inclusive_cpu_ns"], "component consistency")
        labels[label] = item
    exclusive_wall = sum(item["exclusive_wall_ns"] for item in labels.values())
    exclusive_cpu = sum(item["exclusive_cpu_ns"] for item in labels.values())
    require(exclusive_wall <= wall and exclusive_cpu <= cpu, "exclusive components exceed handle")
    require(profile.get("handle_unattributed_wall_ns") == wall-exclusive_wall and
            profile.get("handle_unattributed_cpu_ns") == cpu-exclusive_cpu, "child exclusive accounting")
    partial = profile.get("partial_counters")
    require(type(partial) is dict, "partial counters")
    for field in ("inference_calls_returned", "inference_calls_failed", "last_exchange_callback_hash_count", "last_exchange_callback_rows"):
        _int(partial.get(field), field)
    inference = labels["inference"]
    require(partial["inference_calls_returned"] == inference["calls"]-inference["failures"] and
            partial["inference_calls_failed"] == inference["failures"], "partial inference counters differ")
    for field in ("last_inference_end_ns", "last_inference_wall_ns", "last_inference_cpu_ns"):
        if inference["calls"]:
            _int(partial.get(field), field)
        else:
            require(partial.get(field) is None, "no-call last-inference marker")
    if inference["calls"]:
        require(begin <= partial["last_inference_end_ns"]-partial["last_inference_wall_ns"] <=
                partial["last_inference_end_ns"] <= end and
                partial["last_inference_wall_ns"] <= inference["max_wall_ns"] and
                partial["last_inference_cpu_ns"] <= inference["inclusive_cpu_ns"], "last inference bounds")
    require(partial.get("callback_scope") == "last_native_select_exchange_only_not_all_decision_phases",
            "partial callback scope")
    return {"handle_wall_ns": wall, "handle_cpu_ns": cpu, "labels": labels,
            "handle_unattributed_wall_ns": wall-exclusive_wall, "handle_unattributed_cpu_ns": cpu-exclusive_cpu,
            "partial_counters": {key: value for key, value in partial.items() if type(value) is int}}, begin, end


def _validate_child(child, parent, row, start, arrival, *, completed):
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA, "child sidecar schema")
    require(len(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 8192,
            "bounded ordinary child sidecar")
    clock, quota = row["clock"], row["quota"]
    require(child.get("request_id") == clock["request_id"] and child.get("quota") == quota, "child request/quota")
    require(child.get("backend") == ("qualified_graph64" if quota == 64 else "unchanged_eager32") and
            child.get("graph_identity_sha256") == (GRAPH_IDENTITY if quota == 64 else None), "child backend identity")
    require(child.get("pid") == parent["child_pid"], "parent/child PID binding")
    python_pids = {p["pid"] for p in clock.get("telemetry", {}).get("process_tree", []) if p.get("role") == "python"}
    require(python_pids == {child["pid"]}, "supervised Python identity differs")
    require(child.get("clock") == "perf_counter_ns" and child.get("clock_implementation") == "QueryPerformanceCounter()",
            "shared cross-process QPC runtime")
    for field in ("times_are_diagnostic_not_trusted_deadline_receipts", "success_sidecar_written_after_response_flush",
                  "post_flush_may_follow_parent_receipt", "child_read_may_precede_parent_dispatch_flush_completion",
                  "killed_worker_sidecar_may_be_missing"):
        require(child.get(field) is True, "child sidecar scope " + field)
    read = _int(child.get("request_read_complete_ns"), "child input complete")
    require(start <= read <= parent["wrapper_end_ns"], "child input outside parent request")
    origin = _finite(child.get("request_started_seconds"), "internal clock origin")
    require(0 <= read-origin*1e9, "internal origin/sample order")
    require(child.get("outcome") in ("response_flushed", "failed_before_successful_emission"), "child outcome")
    require(not completed or child["outcome"] == "response_flushed", "successful request has failed sidecar")
    payload = {"player": "search", "observer": row["observer"]}
    require(child.get("payload_sha256") == _digest(payload), "child input content binding")
    _sha(child.get("request_raw_sha256"), "raw input")
    require(parent["request_raw_complete"] and not parent["request_raw_overflow"] and
            child["request_raw_sha256"] == parent["request_raw_sha256"], "actual transported request hash differs")
    _sha(child.get("result_sha256"), "scientific result", nullable=not completed)
    if completed:
        require(child["result_sha256"] == _digest(clock["result"]), "child scientific result binding")
        require(child.get("error") is None, "successful child error")
    else:
        require(child.get("error") is None or type(child["error"]) is str and len(child["error"]) <= 4096,
                "bounded child error")
    profiles = child.get("profiles")
    require(type(profiles) is list and len(profiles) <= 1 and (not completed or len(profiles) == 1),
            "ordinary request has exactly one handle profile")
    metrics = {"parent_start_to_child_read_ns": read-start}
    if profiles:
        handle, begin, end = _validate_handle(profiles[0], child, completed=completed)
        metrics.update(handle)
        metrics["child_read_to_handle_start_ns"] = begin-read
    else:
        begin = end = None
    pre, post = child.get("pre_write_ns"), child.get("post_flush_ns")
    if child["outcome"] == "response_flushed":
        _int(pre, "pre-write"); _int(post, "post-flush")
        require(read <= pre <= post and (end is None or end <= pre), "child emission order")
        # The trusted read can precede the writer's post-flush sample.
        require(arrival is None or pre <= arrival, "pre-write occurs after trusted complete response")
        # The final cooperative .9s check precedes this sample. A scheduling
        # pause between the two cannot be mistaken for a measured gate time.
        _sha(child.get("serialized_packet_utf8_sha256"), "serialized response")
        _int(child.get("serialized_packet_utf8_bytes"), "serialized response bytes", positive=True)
        require(child["serialized_packet_utf8_bytes"] <= 256*1024, "serialized response size bound")
        metrics["write_and_flush_wall_ns"] = post-pre
        if end is not None:
            metrics["handle_end_to_pre_write_ns"] = pre-end
        if arrival is not None:
            metrics["pre_write_to_trusted_arrival_ns"] = arrival-pre
            metrics["signed_post_flush_to_trusted_arrival_ns"] = arrival-post
            if end is not None:
                partition = {"parent_start_to_child_read_ns": read-start, "child_read_to_handle_start_ns": begin-read,
                             "child_handle_ns": end-begin, "handle_end_to_pre_write_ns": pre-end,
                             "pre_write_to_trusted_arrival_ns": arrival-pre}
                require(sum(partition.values()) == arrival-start, "nonoverlapping boundary partition")
                metrics["ordered_boundary_segments_ns"] = partition
    else:
        require(pre is None and post is None and child.get("serialized_packet_utf8_sha256") is None
                and child.get("serialized_packet_utf8_bytes") is None, "failed-before-emission marker contradiction")
    dispatch = parent.get("dispatch_complete_ns")
    if dispatch is not None:
        metrics["signed_dispatch_complete_to_child_read_ns"] = read-dispatch
    return metrics


def validate_profile_row(row):
    require(row.get("mode") in ("baseline", "profiled") and row.get("quota") in (64, 32), "profile mode/quota")
    require(row.get("player_role") == ("challenger" if row["quota"] == 64 else "opponent"), "original player role")
    clock = row.get("clock")
    require(type(clock) is dict and type(row.get("completed")) is bool and clock.get("deadline_seconds") == 1.,
            "unchanged hard-clock receipt")
    completed = row["completed"]
    require(completed == (clock.get("status") == "ok"), "completion/status mismatch")
    if completed:
        require(clock.get("accepted") is True and clock.get("received_before_deadline") is True and
                0 <= _finite(clock.get("receipt_elapsed_seconds"), "trusted arrival") <= 1. and
                type(clock.get("result")) is dict, "successful unchanged clock required")
    else:
        require(clock.get("accepted") is False, "failed request cannot be accepted")
    require(not isinstance(clock.get("result"), dict) or "timing_profile" not in clock["result"],
            "scientific response contains profiling fields")
    parent, child = row.get("parent_profile"), row.get("child_profile")
    if row["mode"] == "baseline":
        require(parent is None and child is None, "baseline unexpectedly instrumented")
        return {"complete_decision": completed, "instrumented": False}
    require(parent is not None or not completed, "successful parent profile missing")
    require(child is not None or not completed, "successful child profile missing")
    result = {"complete_decision": completed, "instrumented": True,
              "missing_parent_profile": parent is None, "missing_child_profile": child is None,
              "causal_component_attribution_complete": False}
    if parent is None:
        require(child is None, "unbound child profile without parent")
        return result
    start, arrival, metrics = _validate_parent(parent, clock, completed=completed)
    result["parent"] = metrics
    if arrival is not None:
        result["trusted_arrival_ns"] = arrival-start
    if child is not None:
        result["child"] = _validate_child(child, parent, row, start, arrival, completed=completed)
    return result


def validate_startup_profiles(parent, child, receipt, payload, quota, *, worker_pid=None):
    """Validate evidence for the actual admission suite, not move attribution.

    The frozen suite has52 completed decisions:37 through ordinary handle and
    15 direct reference reproductions. Its26 expected malformed-input rejections
    also invoke handle, producing exactly63 profiles (37success,26failure).
    """
    require(quota in (32, 64) and type(receipt) is dict and receipt.get("status") == "ok"
            and receipt.get("accepted") is True and receipt.get("received_before_deadline") is True,
            "successful startup receipt")
    require(type(receipt.get("result")) is dict, "startup scientific result")
    start, arrival, parent_metrics = _validate_parent(parent, receipt, completed=False)
    require(arrival is not None and parent["request_raw_complete"] and not parent["request_raw_overflow"],
            "complete startup transport")
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA and
            child.get("request_id") == receipt["request_id"] and child.get("quota") == quota,
            "startup child request/quota")
    require(len(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <= 512*1024,
            "startup profile byte bound")
    require(child.get("pid") == parent["child_pid"] and (worker_pid is None or child["pid"] == worker_pid),
            "startup supervised PID")
    python_pids = {p["pid"] for p in receipt.get("telemetry", {}).get("process_tree", []) if p.get("role") == "python"}
    require(python_pids == {child["pid"]}, "startup sampled PID")
    require(child.get("backend") == ("qualified_graph64" if quota == 64 else "unchanged_eager32") and
            child.get("graph_identity_sha256") == (GRAPH_IDENTITY if quota == 64 else None), "startup backend")
    require(child.get("clock") == "perf_counter_ns" and child.get("clock_implementation") == "QueryPerformanceCounter()"
            and child.get("times_are_diagnostic_not_trusted_deadline_receipts") is True, "startup clock/scope")
    require(child.get("outcome") == "response_flushed" and child.get("error") is None, "startup emission outcome")
    require(child.get("payload_sha256") == _digest(payload) and child.get("result_sha256") == _digest(receipt["result"]),
            "startup payload/result binding")
    require(child.get("request_raw_sha256") == parent["request_raw_sha256"], "startup raw request binding")
    _sha(child.get("serialized_packet_utf8_sha256"), "startup serialized response")
    require(0 < _int(child.get("serialized_packet_utf8_bytes"), "startup response bytes") <= 4*1024*1024,
            "startup response size bound")
    read = _int(child.get("request_read_complete_ns"), "startup input complete")
    pre, post = _int(child.get("pre_write_ns"), "startup prewrite"), _int(child.get("post_flush_ns"), "startup postflush")
    require(start <= read <= pre <= arrival and pre <= post, "startup boundary order")
    require(_finite(child.get("request_started_seconds"), "startup internal origin")*1e9 <= read,
            "startup internal origin order")
    profiles = child.get("profiles")
    require(type(profiles) is list and len(profiles) <= 64, "bounded startup handles")
    if payload == {"operation": "entrypoint_checks"}:
        result = receipt["result"]
        require(result.get("status") == "entrypoint_counterexamples_passed" and result.get("challenger_quota") == quota
                and result.get("counts", {}).get("complete_decisions") == 52, "actual52 decision suite")
        require(len(profiles) == 63, "exact63 actual wrapped admission handles")
        previous_end = read
        for index, profile in enumerate(profiles):
            require(profile.get("handle_failed") is (index >= 37), "admission successful/rejected handle order")
            _, begin, end = _validate_handle(profile, child, completed=index < 37, startup=True)
            require(previous_end <= begin <= end <= pre, "sequential admission handle bounds")
            previous_end = end
        counts = {"wrapped_successes": 37, "intentional_rejections": 26, "unprofiled_direct_reference_reproductions": 15}
    else:
        require(type(payload) is dict and set(payload) == {"operation", "fixtures"} and payload["operation"] == "graph_parity"
                and quota == 64 and type(payload["fixtures"]) is list and len(payload["fixtures"]) == 11,
                "only approved startup operations")
        require(not profiles, "graph parity must not fabricate ordinary handle profiles")
        counts = {"wrapped_successes": 0, "intentional_rejections": 0, "unprofiled_graph_parity_cases": 11}
    return {"status": "startup_sidecars_bound", "profiles": len(profiles), **counts,
            "parent_dropped_spans": parent_metrics["dropped_spans"],
            "parent_pending_spans": parent_metrics["pending_spans"],
            "parent_component_attribution_not_performed": True, "deadline_adjustment_seconds": 0.}


def _distribution(values):
    require(bool(values) and all(type(v) in (int, float) and math.isfinite(v) for v in values), "finite scalar summary")
    ordered = sorted(values)
    index = .95*(len(values)-1)
    lo, hi = math.floor(index), math.ceil(index)
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": ordered[0], "maximum": ordered[-1],
            "p95": ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)}


def summarize(rows, *, boundary=lambda: None):
    groups, failures = {}, []
    for row in rows:
        boundary()
        require(row.get("stage") in ("overhead", "diagnostic"), "analysis stage")
        key = (row["stage"], row["quota"])
        group = groups.setdefault(key, {"timings": {mode: [] for mode in ("baseline", "profiled")},
                                       "metrics": {}, "profiled_identities": [], "failed_rows": 0})
        if not row["completed"]:
            failure = {key: row.get(key) for key in ("index", "stage", "fixture_id", "quota", "player_role", "mode", "pass_index")}
            failure["clock_status"] = row["clock"]["status"]
            failure["receipt_elapsed_seconds"] = row["clock"].get("receipt_elapsed_seconds")
            try:
                failure["profile_evidence"] = validate_profile_row(row)
                failure["profile_validation"] = "partial_evidence_checked"
            except (ValueError, TypeError, KeyError) as error:
                failure["profile_validation"] = "invalid_or_unbound_preserved_evidence"
                failure["profile_validation_error"] = str(error)
            failures.append(failure); group["failed_rows"] += 1
            continue
        metrics = validate_profile_row(row)
        group["timings"][row["mode"]].append(row["clock"]["receipt_elapsed_seconds"])
        if row["mode"] == "baseline":
            continue
        group["profiled_identities"].append({key: row[key] for key in ("index", "fixture_id", "pass_index")})
        def append(prefix, value):
            for name, item in value.items():
                field = prefix+"."+name if prefix else name
                if type(item) is dict:
                    append(field, item)
                elif type(item) is int:
                    group["metrics"].setdefault(field, []).append(item)
        append("", metrics)
    summaries = []
    for (stage, quota), group in sorted(groups.items()):
        summaries.append({"stage": stage, "quota": quota, "failed_rows": group["failed_rows"],
                          "trusted_completed_timing_seconds": {mode: _distribution(values) if values else {"count": 0}
                                                                for mode, values in group["timings"].items()},
                          "profiled_metrics": {name: {"unit": "nanoseconds" if name.endswith("_ns") else "count",
                                                     "distribution": _distribution(values)}
                                                for name, values in sorted(group["metrics"].items())},
                          "profiled_identities": group["profiled_identities"]})
    return {"schema": "splendorust_graph_timing_analysis_v1", "rows": len(rows), "groups": summaries,
            "failures": failures, "all_collected_decisions_completed": not failures,
            "complete_planned_collection_inferred": False, "deadline_adjustment_seconds": 0.,
            "causal_bottleneck_proven": False,
            "interpretation": [
                "Only original raw receipts determine acceptance. Sidecar I/O and profiling do not adjust deadlines.",
                "Missing/invalid failure sidecars remain explicit; no failed or truncated panel is reported complete.",
                "Parent dispatch completion can follow child input-read completion because pipe operations overlap.",
                "Child post-flush can follow parent trusted arrival; signed differences are retained, not clamped into invented latency.",
                "Ordered start/read/handle/pre-write/arrival segments partition elapsed wall time, not causal resource costs.",
                "Parent supervision wall intervals overlap child execution; union only within each declared scope.",
                "CPU measurements retain their recorded boundaries; clipped wall intervals never receive proportionally allocated CPU.",
                "Inference wraps conversion and neural execution; native-read waiting also includes Rust work and scheduling.",
                "Child process CPU excludes native Rust and parent; wall-minus-CPU does not establish scheduling delay.",
                "Payload/result hashes bind content. Raw serialized packet hashes require separate raw-log binding by the runner.",
                "These selected saved positions and two passes do not establish a general tail probability or reproduce historical host conditions."]}


analyze_rows = summarize
analyze = summarize
