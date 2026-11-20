"""Validate diagnostic provenance and summarize overlapping component timings.

Only the original whole-response clock decides deadlines. Component spans are
descriptive observations; inclusive totals across labels/processes must not be
added or subtracted from that clock. No model or process is invoked here.
"""
from __future__ import annotations

import math

PARENT_SCHEMA = "splendorust_component_parent_v1"
CHILD_SCHEMA = "splendorust_component_child_v1"
PARENT_LABELS = {"initial_sample", "periodic_sample", "aggregate_guard", "resource_guard_check",
                 "resource_guard", "stdin_write", "stdin_flush", "stdout_raw_read"}
SUPERVISION_LABELS = {"initial_sample", "periodic_sample", "aggregate_guard", "resource_guard_check", "resource_guard"}
CHILD_LABELS = {"inference", "native_write", "native_flush", "native_read_wait", "callback_hash",
                "observer_encode", "decode_observer"}
CHILD_FIELDS = ("calls", "inclusive_wall_ns", "inclusive_cpu_ns", "max_wall_ns", "failures",
                "exclusive_wall_ns", "exclusive_cpu_ns")
GAP_FIELDS = ("unattributed_dispatch_to_child_read_ns", "unattributed_read_to_handle_ns",
              "child_handle_wall_ns", "unattributed_handle_to_pre_emit_ns", "unattributed_pre_emit_to_receipt_ns")


def require(value, message):
    if not value:
        raise ValueError("Component timing analysis: " + message)


def _int(value, name, *, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), name + " must be a bounded nonnegative integer")
    require(value < 2**63, name + " exceeds diagnostic integer bound")
    return value


def interval_union_ns(intervals, lower, upper):
    """Exact union of clipped wall intervals; never scale associated CPU time."""
    require(type(lower) is int and type(upper) is int and lower <= upper, "ordered clipping window")
    clipped = []
    for start, end in intervals:
        require(type(start) is int and type(end) is int and start <= end, "ordered interval")
        left, right = max(start, lower), min(end, upper)
        if left < right:
            clipped.append((left, right))
    total, rightmost = 0, lower
    for left, right in sorted(clipped):
        if right > rightmost:
            total += right-max(left, rightmost)
            rightmost = right
    return total


def _validate_parent(parent, clock):
    require(type(parent) is dict and parent.get("schema") == PARENT_SCHEMA, "parent profile schema")
    require(parent.get("clock") == "perf_counter_ns" and parent.get("process_cpu_clock") == "process_time_ns"
            and parent.get("thread_cpu_clock") == "thread_time_ns", "parent diagnostic clocks")
    require(parent.get("request_id") == clock["request_id"], "parent request identity")
    for field in ("pid", "child_pid"):
        _int(parent.get(field), field, positive=True)
    start = _int(parent.get("request_start_ns"), "trusted request origin")
    arrival = _int(parent.get("receipt_arrival_ns"), "trusted raw line timestamp")
    require(start <= arrival and arrival-start <= 1_000_000_000, "trusted one-second window")
    elapsed = clock.get("receipt_elapsed_seconds")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed <= 1,
            "finite trusted arrival")
    require(round(elapsed*1e9) == arrival-start and parent.get("receipt_reconstructed_arrival_ns") == arrival,
            "profile timestamps do not match original receipt")
    raw_arrivals = parent.get("raw_arrivals_ns")
    require(type(raw_arrivals) is list and len(raw_arrivals) <= 256 and arrival in raw_arrivals,
            "trusted line stamp missing from bounded raw reader observations")
    for stamp in raw_arrivals:
        _int(stamp, "raw line timestamp")
    require(parent.get("maximum_spans") == 256 and type(parent.get("dropped_spans")) is int
            and parent["dropped_spans"] == 0, "complete bounded spans required for attribution")
    require(type(parent.get("pending_spans_at_snapshot")) is dict and not parent["pending_spans_at_snapshot"],
            "unfinished parent spans cannot support complete attribution")
    wrapper_start = _int(parent.get("wrapper_start_ns"), "wrapper start")
    wrapper_end = _int(parent.get("wrapper_end_ns"), "wrapper end")
    require(wrapper_start <= start <= arrival <= wrapper_end, "wrapper/trusted bounds")
    for field in ("wrapper_process_cpu_ns", "wrapper_thread_cpu_ns"):
        _int(parent.get(field), field)
    spans, totals = parent.get("spans"), parent.get("totals")
    require(type(spans) is list and 0 < len(spans) <= 256 and type(totals) is dict
            and bool(totals) and set(totals) <= PARENT_LABELS, "bounded fixed-label parent measurements")
    calculated = {}
    for span in spans:
        require(type(span) is dict and set(span) == {"label", "start_ns", "end_ns", "process_cpu_ns", "thread_cpu_ns"},
                "parent span fields")
        label = span["label"]
        require(label in PARENT_LABELS, "unknown parent component")
        for field in ("start_ns", "end_ns", "process_cpu_ns", "thread_cpu_ns"):
            _int(span[field], "parent span " + field)
        require(span["start_ns"] <= span["end_ns"] <= wrapper_end, "ordered completed parent span")
        value = calculated.setdefault(label, {"calls": 0, "wall_ns": 0, "process_cpu_ns": 0, "thread_cpu_ns": 0})
        value["calls"] += 1
        value["wall_ns"] += span["end_ns"]-span["start_ns"]
        value["process_cpu_ns"] += span["process_cpu_ns"]
        value["thread_cpu_ns"] += span["thread_cpu_ns"]
    require(calculated == totals, "parent aggregates do not match complete raw intervals")
    for total in totals.values():
        require(type(total) is dict and set(total) == {"calls", "wall_ns", "process_cpu_ns", "thread_cpu_ns"},
                "parent total fields")
        for field, value in total.items():
            _int(value, "parent total " + field)
    require(any(s["label"] == "stdout_raw_read" and s["end_ns"] == arrival for s in spans),
            "trusted timestamp not coupled to a completed raw read")
    by_label = {}
    for label, total in totals.items():
        relevant = [s for s in spans if s["label"] == label]
        entirely_inside = [s for s in relevant if start <= s["start_ns"] <= s["end_ns"] <= arrival]
        by_label[label] = {
            "calls": total["calls"],
            "clipped_wall_union_ns": interval_union_ns([(s["start_ns"], s["end_ns"]) for s in relevant], start, arrival),
            "unclipped_inclusive_wall_ns": total["wall_ns"],
            "unclipped_process_cpu_ns": total["process_cpu_ns"],
            "unclipped_thread_cpu_ns": total["thread_cpu_ns"],
            "fully_contained_span_thread_cpu_ns": sum(s["thread_cpu_ns"] for s in entirely_inside),
            "boundary_crossing_spans": len(relevant)-len(entirely_inside),
        }
    supervision = interval_union_ns([(s["start_ns"], s["end_ns"]) for s in spans
                                     if s["label"] in SUPERVISION_LABELS], start, arrival)
    return start, arrival, {"supervision_wall_union_within_trusted_ns": supervision, "labels": by_label,
                            "wrapper_process_cpu_ns": parent["wrapper_process_cpu_ns"],
                            "wrapper_thread_cpu_ns": parent["wrapper_thread_cpu_ns"]}


def _validate_child(child, parent, clock, start, arrival):
    require(type(child) is dict and child.get("schema") == CHILD_SCHEMA, "child profile schema")
    require(child.get("clock") == "perf_counter_ns" and child.get("cpu_clock") == "process_time_ns"
            and child.get("measurement_units") == "nanoseconds", "child diagnostic clocks/units")
    require(child.get("times_are_diagnostic_not_trusted_deadline_receipts") is True and
            child.get("inclusive_labels_overlap") is True and
            child.get("exclusive_totals_exclude_nested_components") is True and child.get("handle_failed") is False and
            child.get("cpu_scope") == "current_python_process_only_excludes_native_rust_and_parent",
            "child measurement scope")
    require(child.get("total_fields") == list(CHILD_FIELDS), "child total field semantics")
    require(type(child.get("pid")) is int and child["pid"] == parent["child_pid"], "parent/child process binding")
    python_pids = {p["pid"] for p in clock.get("telemetry", {}).get("process_tree", []) if p.get("role") == "python"}
    require(python_pids == {child["pid"]}, "child profile differs from supervised Python identity")
    fields = ("request_read_complete_ns", "handle_start_ns", "handle_end_ns", "before_emit_ns")
    points = [_int(child.get(field), field) for field in fields]
    require(start <= points[0] <= points[1] <= points[2] <= points[3] <= arrival,
            "child diagnostic boundaries outside trusted request window")
    handle_wall = points[2]-points[1]
    handle_cpu = _int(child.get("handle_cpu_ns"), "full handle CPU")
    _int(child.get("component_nested_calls"), "nested-call count")
    totals = child.get("totals")
    require(type(totals) is dict and set(totals) == CHILD_LABELS, "fixed child components")
    labels = {}
    for label, values in totals.items():
        require(type(values) is list and len(values) == len(CHILD_FIELDS), "child component total fields")
        for field, value in zip(CHILD_FIELDS, values):
            _int(value, "child " + field)
        item = dict(zip(CHILD_FIELDS, values))
        require((item["calls"] > 0 or not any(values)) and item["failures"] == 0 and item["max_wall_ns"] <= item["inclusive_wall_ns"]
                and item["exclusive_wall_ns"] <= item["inclusive_wall_ns"]
                and item["exclusive_cpu_ns"] <= item["inclusive_cpu_ns"], "completed child component consistency")
        labels[label] = item
    exclusive_wall = sum(value["exclusive_wall_ns"] for value in labels.values())
    exclusive_cpu = sum(value["exclusive_cpu_ns"] for value in labels.values())
    require(exclusive_wall <= handle_wall and exclusive_cpu <= handle_cpu, "child exclusive components exceed full handle")
    unattributed_wall = _int(child.get("handle_unattributed_wall_ns"), "unattributed handle wall")
    unattributed_cpu = _int(child.get("handle_unattributed_cpu_ns"), "unattributed handle CPU")
    require(unattributed_wall == handle_wall-exclusive_wall and unattributed_cpu == handle_cpu-exclusive_cpu,
            "child local exclusive accounting differs")
    boundaries = [start, *points, arrival]
    gaps = dict(zip(GAP_FIELDS, [right-left for left, right in zip(boundaries, boundaries[1:])]))
    require(sum(gaps.values()) == arrival-start, "ordered wall boundary partition differs")
    return {"handle_wall_ns": handle_wall, "handle_cpu_ns": handle_cpu, "labels": labels,
            "handle_unattributed_wall_ns": unattributed_wall, "handle_unattributed_cpu_ns": unattributed_cpu,
            "ordered_boundary_segments_ns": gaps}


def validate_profile_row(row):
    """Authenticate profile/window coupling; never alter a trusted receipt."""
    require(row.get("mode") in ("baseline", "profiled"), "profiling mode")
    clock = row.get("clock")
    require(type(clock) is dict and row.get("completed") is True and clock.get("status") == "ok"
            and clock.get("accepted") is True and clock.get("received_before_deadline") is True
            and clock.get("deadline_seconds") == 1., "completed unchanged hard-clock decision required")
    result = clock.get("result")
    require(type(result) is dict, "scientific response required")
    parent = row.get("parent_profile")
    if row["mode"] == "baseline":
        require(parent is None and "timing_profile" not in result, "baseline unexpectedly instrumented")
        return None
    start, arrival, parent_metrics = _validate_parent(parent, clock)
    child_metrics = _validate_child(result.get("timing_profile"), parent, clock, start, arrival)
    return {"trusted_arrival_ns": arrival-start, "parent": parent_metrics, "child": child_metrics}


def summarize(rows, *, boundary=lambda: None):
    """Report component distributions by stage/quota, with causal limits explicit."""
    from .qualification import distribution
    def metric_distribution(values):
        return {key: value for key, value in distribution(values).items() if key != "above_diagnostic_margin"}
    groups = {}
    for row in rows:
        boundary()
        require(row.get("stage") in ("overhead", "diagnostic") and row.get("quota") in (32, 64), "diagnostic row group")
        metrics = validate_profile_row(row)
        group = groups.setdefault((row["stage"], row["quota"]), {"timings": {mode: [] for mode in ("baseline", "profiled")},
                                                                   "metrics": {}, "profiled_identities": []})
        group["timings"][row["mode"]].append(row["clock"]["receipt_elapsed_seconds"])
        if metrics is None:
            continue
        group["profiled_identities"].append({key: row[key] for key in ("index", "fixture_id", "pass_index")})
        def append(prefix, value):
            for key, item in value.items():
                name = prefix+"."+key if prefix else key
                if type(item) is dict:
                    append(name, item)
                elif type(item) is int:
                    group["metrics"].setdefault(name, []).append(item)
        append("", metrics)
    result = []
    for (stage, quota), group in sorted(groups.items()):
        require(all(group["timings"].values()), "both modes required for component summary")
        result.append({"stage": stage, "quota": quota,
                       "trusted_timing_seconds": {mode: distribution(values) for mode, values in group["timings"].items()},
                       "profiled_metrics": {name: {"unit": "nanoseconds" if name.endswith("_ns") else "count",
                                                   "distribution": metric_distribution(values)}
                                             for name, values in sorted(group["metrics"].items())},
                       "profiled_identities": group["profiled_identities"]})
    return {"schema": "splendorust_component_timing_analysis_v1", "rows": len(rows), "groups": result,
            "deadline_adjustment_seconds": 0., "causal_bottleneck_proven": False,
            "interpretation": [
                "Trusted arrival is unchanged; no component duration is subtracted from a deadline.",
                "Parent supervision is a union clipped to the trusted window. Its labels overlap each other and concurrent child work.",
                "Unclipped CPU spans retain their original bounds; CPU is never proportionally allocated to clipped wall intervals.",
                "Child exclusive totals partition only measured nested child spans; unlabelled handle time includes work and waiting.",
                "Raw-read wall time includes waiting for the entire response. Reader thread CPU is measured active thread time, not pure transport cost.",
                "Before-emission-to-receipt and dispatch gaps remain unattributed serialization, communication, supervision and scheduling time.",
                "Parent/child CPU scopes differ and native Rust CPU is absent from child process timers. Do not add inclusive wall totals.",
                "These selected development roots and instrumentation observations do not establish an unconditional tail rate or a causal host-load diagnosis."]}


analyze_rows = summarize
analyze = summarize
