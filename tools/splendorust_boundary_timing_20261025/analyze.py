"""Authenticate minimal boundary timestamps without revising trusted clocks.

Segments partition elapsed wall time; they do not identify CPU, inference or
scheduling costs. Pipe reads and writes overlap across processes legitimately.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics

PARENT_SCHEMA = "splendorust_boundary_timing_parent_v1"
CHILD_SCHEMA = "splendorust_boundary_timing_child_v1"
GRAPH_IDENTITY = "c4d572908ddb3c216689c702bb00367dc263c9f1374be8bd397250e0b65e6ba2"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HANDLE_FIELDS = {"schema", "pid", "clock", "measurement_units", "request_read_complete_ns",
                 "handle_start_ns", "handle_end_ns", "handle_failed",
                 "times_are_diagnostic_not_trusted_deadline_receipts"}


def require(value, message):
    if not value:
        raise ValueError("Boundary timing analysis: " + message)


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
    require(parent.get("clock") == "perf_counter_ns" and
            parent.get("clock_implementation") == "QueryPerformanceCounter()", "parent shared QPC clock")
    require(parent.get("request_id") == clock["request_id"] and
            parent.get("times_are_diagnostic_not_trusted_deadline_receipts") is True and
            parent.get("measurement_units") == "nanoseconds", "parent request/scope")
    require(not ({"spans", "totals", "wrapper_process_cpu_ns", "wrapper_thread_cpu_ns",
                  "process_cpu_clock", "thread_cpu_clock"} & set(parent)), "minimal parent observer only")
    for field in ("pid", "child_pid"):
        _int(parent.get(field), field, positive=True)
    start = _int(parent.get("request_start_ns"), "trusted origin")
    snapshot = _int(parent.get("capture_snapshot_ns"), "capture snapshot")
    require(start <= snapshot, "parent snapshot order")
    require(parent.get("request_capture_maximum_bytes") == 1024*1024 and
            type(parent.get("request_raw_complete")) is bool and type(parent.get("request_raw_overflow")) is bool,
            "bounded raw request capture")
    require(_int(parent.get("request_raw_bytes"), "raw request byte count") <= 1024*1024, "raw request capture bound")
    _sha(parent.get("request_raw_sha256"), "captured raw request")
    require(not completed or parent["request_raw_complete"] and not parent["request_raw_overflow"],
            "successful request capture incomplete")
    elapsed, arrival = clock.get("receipt_elapsed_seconds"), parent.get("receipt_arrival_ns")
    reconstructed = None if elapsed is None else start+round(_finite(elapsed, "receipt elapsed")*1e9)
    require(parent.get("receipt_reconstructed_arrival_ns") == reconstructed, "reconstructed receipt differs")
    if arrival is not None:
        _int(arrival, "arrival")
        require(arrival == reconstructed and start <= arrival <= snapshot, "raw timestamp/receipt coupling")
    elif completed:
        raise ValueError("Boundary timing analysis: successful raw arrival missing")
    arrivals = parent.get("raw_arrivals_ns")
    require(parent.get("maximum_arrival_events") == 16 and type(arrivals) is list and len(arrivals) <= 16,
            "bounded raw timestamps")
    dropped = _int(parent.get("dropped_arrival_events"), "dropped raw arrivals")
    require(not completed or dropped == 0, "successful arrival inventory truncated")
    for stamp in arrivals:
        _int(stamp, "raw timestamp")
        require(start <= stamp <= snapshot, "raw timestamp outside request")
    require(arrival is None or arrival in arrivals, "raw arrival missing from observations")
    for field in ("dispatch_write_calls", "dispatch_flush_calls", "dispatch_write_failures", "dispatch_flush_failures",
                  "pending_writes", "pending_flushes"):
        _int(parent.get(field), field)
    require(parent.get("response_capture_maximum_bytes") == 4*1024*1024, "response capture bound")
    if arrival is not None:
        require(type(parent.get("response_capture_overflow")) is bool, "response overflow flag")
        _sha(parent.get("response_raw_sha256"), "actual raw response")
        require(0 < _int(parent.get("response_raw_bytes"), "actual response bytes") <= 4*1024*1024,
                "bounded raw response capture")
    else:
        require(parent.get("response_raw_sha256") is None and parent.get("response_raw_bytes") is None and
                parent.get("response_capture_overflow") is None,
                "absent arrival cannot have captured response")
    require(not completed or not parent["response_capture_overflow"], "successful raw response capture overflow")
    markers = ("stdin_write_start_ns", "stdin_write_end_ns", "stdin_flush_start_ns", "stdin_flush_end_ns")
    previous = start
    for field in markers:
        stamp = parent.get(field)
        if stamp is not None:
            _int(stamp, field)
            require(previous <= stamp <= snapshot, "serial dispatch marker order")
            previous = stamp
    dispatch = parent.get("dispatch_complete_ns")
    expected_dispatch = (parent.get("stdin_flush_end_ns") if parent["dispatch_flush_calls"] and
                         not parent["pending_flushes"] and not parent["dispatch_flush_failures"] else None)
    require(dispatch == expected_dispatch, "dispatch/flush binding")
    if completed:
        require(all(type(parent.get(key)) is int for key in markers) and
                parent["dispatch_write_calls"] >= 1 and parent["dispatch_flush_calls"] == 1 and
                parent["pending_writes"] == parent["pending_flushes"] == 0 and
                parent["dispatch_write_failures"] == parent["dispatch_flush_failures"] == 0,
                "complete single dispatch required")
    require(parent.get("raw_reader_semantics") == "unchanged raw FileIO.readline then immediate trusted perf_counter_ns",
            "trusted raw reader semantics")
    for field in ("child_read_may_precede_dispatch_completion", "child_post_flush_may_follow_trusted_arrival",
                  "hashes_calculated_after_original_request_returns", "dispatch_markers_may_be_partial_if_pending"):
        require(parent.get(field) is True, "parent observer scope " + field)
    metrics = {"pending_writes": parent["pending_writes"], "pending_flushes": parent["pending_flushes"]}
    if dispatch is not None:
        metrics["start_to_dispatch_complete_ns"] = dispatch-start
    if parent.get("stdin_write_start_ns") is not None:
        metrics["start_to_dispatch_start_ns"] = parent["stdin_write_start_ns"]-start
    return start, arrival, metrics


def _validate_handle(profile, child, *, completed, startup=False):
    require(type(profile) is dict and set(profile) == HANDLE_FIELDS and profile.get("schema") == CHILD_SCHEMA,
            "minimal child handle fields/schema")
    require(profile.get("pid") == child["pid"] and profile.get("clock") == "perf_counter_ns" and
            profile.get("measurement_units") == "nanoseconds", "child handle identity/clock")
    require(profile.get("request_read_complete_ns") == (None if startup else child["request_read_complete_ns"]),
            "child read marker binding")
    require(profile.get("times_are_diagnostic_not_trusted_deadline_receipts") is True, "child handle scope")
    require(type(profile.get("handle_failed")) is bool and (not completed or not profile["handle_failed"]),
            "handle outcome mismatch")
    begin = _int(profile.get("handle_start_ns"), "handle start")
    end = _int(profile.get("handle_end_ns"), "handle end")
    require(child["request_read_complete_ns"] <= begin <= end, "child handle order")
    return {"handle_wall_ns": end-begin}, begin, end


def _child_identity(child, parent, receipt, payload, quota, *, completed, startup=False, worker_pid=None):
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA, "child sidecar schema")
    require(len(json.dumps(child, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) <=
            (512*1024 if startup else 8192), "bounded child sidecar")
    require(child.get("request_id") == receipt["request_id"] and child.get("quota") == quota, "child request/quota")
    require(child.get("backend") == ("qualified_graph64" if quota == 64 else "unchanged_eager32") and
            child.get("graph_identity_sha256") == (GRAPH_IDENTITY if quota == 64 else None), "child backend identity")
    require(child.get("pid") == parent["child_pid"] and (worker_pid is None or child["pid"] == worker_pid),
            "parent/child supervised PID binding")
    python_pids = {p["pid"] for p in receipt.get("telemetry", {}).get("process_tree", []) if p.get("role") == "python"}
    require(python_pids == {child["pid"]}, "sampled Python PID identity differs")
    require(child.get("clock") == "perf_counter_ns" and child.get("clock_implementation") == "QueryPerformanceCounter()",
            "shared cross-process QPC runtime")
    for field in ("times_are_diagnostic_not_trusted_deadline_receipts", "success_sidecar_written_after_response_flush",
                  "post_flush_may_follow_parent_receipt", "child_read_may_precede_parent_dispatch_flush_completion",
                  "killed_worker_sidecar_may_be_missing", "text_transport_may_expand_newlines"):
        require(child.get(field) is True, "child sidecar scope " + field)
    require(child.get("outcome") in ("response_flushed", "failed_before_successful_emission"), "child outcome")
    require(not completed or child["outcome"] == "response_flushed", "successful request has failed sidecar")
    require(child.get("payload_sha256") == _digest(payload), "child input content binding")
    require(parent["request_raw_complete"] and not parent["request_raw_overflow"] and
            child.get("request_raw_sha256") == parent["request_raw_sha256"], "actual transported request hash differs")
    _sha(child.get("result_sha256"), "scientific result", nullable=not completed)
    if completed:
        require(child["result_sha256"] == _digest(receipt["result"]), "child scientific result binding")
        require(child.get("error") is None, "successful child error")
    else:
        require(child.get("error") is None or type(child.get("error")) is str and len(child["error"]) <= 4096,
                "bounded child error")


def _boundaries(child, parent, start, arrival, *, begin=None, end=None, startup=False):
    read = _int(child.get("request_read_complete_ns"), "child input complete")
    require(start <= read <= parent["capture_snapshot_ns"], "child input outside parent request")
    origin = _finite(child.get("request_started_seconds"), "internal clock origin")
    require(0 <= read-origin*1e9, "internal origin/sample order")
    metrics = {"parent_start_to_child_read_ns": read-start}
    if begin is not None:
        metrics.update(child_read_to_handle_start_ns=begin-read, handle_wall_ns=end-begin)
    pre, post = child.get("pre_write_ns"), child.get("post_flush_ns")
    if child["outcome"] == "response_flushed":
        _int(pre, "pre-write"); _int(post, "post-flush")
        require(read <= pre <= post and (end is None or end <= pre), "child emission order")
        require(arrival is None or pre <= arrival, "pre-write occurs after trusted complete response")
        _sha(child.get("serialized_packet_utf8_sha256"), "serialized response")
        size = _int(child.get("serialized_packet_utf8_bytes"), "serialized response bytes", positive=True)
        require(size <= (4*1024*1024 if startup else 256*1024), "serialized response size bound")
        if arrival is not None:
            raw_size = parent["response_raw_bytes"]
            require(raw_size-size in (0, 1), "only terminal LF/CRLF response size difference permitted")
            if raw_size == size:
                require(parent["response_raw_sha256"] == child["serialized_packet_utf8_sha256"],
                        "equal-length raw response hash differs")
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
        require(post is None and child.get("serialized_packet_utf8_sha256") is None and
                child.get("serialized_packet_utf8_bytes") is None, "failed-before-emission marker contradiction")
        # A write/flush exception can follow the marker but cannot certify a
        # complete response. Keep the attempted-write boundary without making
        # a completed delivery segment or packet binding out of partial I/O.
        if pre is not None:
            _int(pre, "partial pre-write")
            require(read <= pre and (end is None or end <= pre), "partial child emission order")
            metrics["partial_emission_attempt_recorded"] = True
            if end is not None:
                metrics["handle_end_to_partial_pre_write_ns"] = pre-end
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
    require(not isinstance(clock.get("result"), dict) or not ({"timing_profile", "boundary_profile"} & set(clock["result"])),
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
        _child_identity(child, parent, clock, {"player": "search", "observer": row["observer"]}, row["quota"],
                        completed=completed)
        profiles = child.get("profiles")
        require(type(profiles) is list and len(profiles) <= 1 and (not completed or len(profiles) == 1),
                "ordinary request has exactly one handle profile")
        begin = end = None
        if profiles:
            _, begin, end = _validate_handle(profiles[0], child, completed=completed)
        result["child"] = _boundaries(child, parent, start, arrival, begin=begin, end=end)
    return result


def validate_startup_profiles(parent, child, receipt, payload, quota, *, worker_pid=None):
    """Bind 37 successful +26 rejected wrapped handles and15 direct checks."""
    require(quota in (32, 64) and type(receipt) is dict and receipt.get("status") == "ok" and
            receipt.get("accepted") is True and receipt.get("received_before_deadline") is True and
            type(receipt.get("result")) is dict, "successful startup receipt")
    start, arrival, _ = _validate_parent(parent, receipt, completed=True)
    _child_identity(child, parent, receipt, payload, quota, completed=True, startup=True, worker_pid=worker_pid)
    _boundaries(child, parent, start, arrival, startup=True)
    profiles = child.get("profiles")
    require(type(profiles) is list and len(profiles) <= 64, "bounded startup handles")
    if payload == {"operation": "entrypoint_checks"}:
        result = receipt["result"]
        require(result.get("status") == "entrypoint_counterexamples_passed" and result.get("challenger_quota") == quota and
                result.get("counts", {}).get("complete_decisions") == 52, "actual52 decision suite")
        require(len(profiles) == 63, "exact63 actual wrapped admission handles")
        previous_end = child["request_read_complete_ns"]
        for index, profile in enumerate(profiles):
            require(profile.get("handle_failed") is (index >= 37), "admission successful/rejected handle order")
            _, begin, end = _validate_handle(profile, child, completed=index < 37, startup=True)
            require(previous_end <= begin <= end <= child["pre_write_ns"], "sequential admission handle bounds")
            previous_end = end
        counts = {"wrapped_successes": 37, "intentional_rejections": 26, "unprofiled_direct_reference_reproductions": 15}
    else:
        require(type(payload) is dict and set(payload) == {"operation", "fixtures"} and payload["operation"] == "graph_parity" and
                quota == 64 and type(payload["fixtures"]) is list and len(payload["fixtures"]) == 11,
                "only approved startup operations")
        require(not profiles, "graph parity must not fabricate ordinary handle profiles")
        counts = {"wrapped_successes": 0, "intentional_rejections": 0, "unprofiled_graph_parity_cases": 11}
    return {"status": "startup_sidecars_bound", "profiles": len(profiles), **counts,
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
    return {"schema": "splendorust_boundary_timing_analysis_v1", "rows": len(rows), "groups": summaries,
            "failures": failures, "all_collected_decisions_completed": not failures,
            "complete_planned_collection_inferred": False, "deadline_adjustment_seconds": 0.,
            "causal_bottleneck_proven": False,
            "interpretation": [
                "Only original raw receipts determine acceptance; observer I/O never adjusts deadlines.",
                "Missing or invalid failure sidecars remain explicit; incomplete panels are never reported complete.",
                "Parent dispatch completion can follow child input-read completion because pipe operations overlap.",
                "Child post-flush can follow parent trusted arrival; signed differences remain unclamped.",
                "Start/read/handle/pre-write/arrival segments partition elapsed wall time, not causal resource costs.",
                "This observer does not measure inference, native work, process CPU or supervision components.",
                "The pre-write marker follows the original final guard and may include an unobserved scheduling pause.",
                "Packet hashes and raw reader timestamps require the separate exact raw-log verifier.",
                "Saved selected roots and two passes cannot establish general tail probabilities or past host conditions."]}


analyze_rows = summarize
analyze = summarize
