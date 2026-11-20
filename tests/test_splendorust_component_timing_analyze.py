"""Counterexamples for clipped, overlapping and process-scoped timing evidence."""
from copy import deepcopy
import pytest

from tools.splendorust_component_timing_20261025 import analyze as a

MS = 1_000_000


def profile_row():
    start, arrival = 1000*MS, 1500*MS
    spans = []
    for label, left, right, cpu in [
        ("initial_sample", 0, 50, 8), ("periodic_sample", 40, 80, 6),
        ("aggregate_guard", 50, 60, 2), ("periodic_sample", 490, 520, 4),
        ("stdin_write", 50, 55, 1), ("stdout_raw_read", -100, 500, 20),
    ]:
        spans.append({"label": label, "start_ns": start+left*MS, "end_ns": start+right*MS,
                      "process_cpu_ns": (cpu+1)*MS, "thread_cpu_ns": cpu*MS})
    totals = {}
    for span in spans:
        total = totals.setdefault(span["label"], {"calls": 0, "wall_ns": 0, "process_cpu_ns": 0, "thread_cpu_ns": 0})
        total["calls"] += 1
        total["wall_ns"] += span["end_ns"]-span["start_ns"]
        for field in ("process_cpu_ns", "thread_cpu_ns"): total[field] += span[field]
    parent = {"schema": a.PARENT_SCHEMA, "request_id": "request-1", "clock": "perf_counter_ns",
              "process_cpu_clock": "process_time_ns", "thread_cpu_clock": "thread_time_ns",
              "pid": 321, "child_pid": 123, "request_start_ns": start, "receipt_arrival_ns": arrival,
              "receipt_reconstructed_arrival_ns": arrival, "raw_arrivals_ns": [arrival],
              "maximum_spans": 256, "dropped_spans": 0,
              "pending_spans_at_snapshot": {},
              "wrapper_start_ns": start-1, "wrapper_end_ns": arrival+30*MS,
              "wrapper_process_cpu_ns": 100*MS, "wrapper_thread_cpu_ns": 60*MS,
              "spans": spans, "totals": totals}
    child = {"schema": a.CHILD_SCHEMA, "clock": "perf_counter_ns", "cpu_clock": "process_time_ns",
             "measurement_units": "nanoseconds", "pid": 123,
             "times_are_diagnostic_not_trusted_deadline_receipts": True, "inclusive_labels_overlap": True,
             "exclusive_totals_exclude_nested_components": True, "handle_failed": False,
             "total_fields": list(a.CHILD_FIELDS),
             "cpu_scope": "current_python_process_only_excludes_native_rust_and_parent",
             "request_read_complete_ns": start+60*MS, "handle_start_ns": start+65*MS,
             "handle_end_ns": start+450*MS, "before_emit_ns": start+460*MS, "handle_cpu_ns": 180*MS,
             "component_nested_calls": 65, "handle_unattributed_wall_ns": 25*MS,
             "handle_unattributed_cpu_ns": 23*MS,
             "totals": {"inference": [65, 300*MS, 150*MS, 10*MS, 0, 280*MS, 140*MS],
                        "callback_hash": [65, 20*MS, 10*MS, MS, 0, 20*MS, 10*MS],
                        "native_read_wait": [66, 50*MS, 2*MS, 2*MS, 0, 50*MS, 2*MS],
                        "native_write": [66, 10*MS, 5*MS, MS, 0, 10*MS, 5*MS]}}
    for label in a.CHILD_LABELS:
        child["totals"].setdefault(label, [0]*7)
    return {"index": 0, "stage": "overhead", "fixture_id": "development/0", "pass_index": 0,
            "quota": 64, "mode": "profiled", "completed": True, "parent_profile": parent,
            "clock": {"request_id": "request-1", "status": "ok", "accepted": True,
                      "received_before_deadline": True, "deadline_seconds": 1., "receipt_elapsed_seconds": .5,
                      "result": {"diagnostics": {"untouched": True}, "timing_profile": child},
                      "telemetry": {"process_tree": [{"pid": 123, "role": "python"}]}}}


def baseline_row():
    row = profile_row()
    row["mode"] = "baseline"; row["parent_profile"] = None
    row["clock"]["result"].pop("timing_profile")
    return row


def test_union_clips_and_never_double_counts_nested_or_concurrent_spans():
    assert a.interval_union_ns([(0, 50), (40, 80), (50, 60), (490, 520)], 0, 500) == 90
    assert a.interval_union_ns([(-100, 600)], 0, 500) == 500
    assert a.interval_union_ns([(500, 600), (-100, 0)], 0, 500) == 0


def test_parent_postarrival_is_clipped_but_cpu_not_proportionally_allocated():
    row = profile_row(); original = deepcopy(row)
    result = a.validate_profile_row(row)
    parent = result["parent"]
    assert parent["supervision_wall_union_within_trusted_ns"] == 90*MS
    periodic = parent["labels"]["periodic_sample"]
    assert periodic["clipped_wall_union_ns"] == 50*MS
    assert periodic["unclipped_inclusive_wall_ns"] == 70*MS
    assert periodic["unclipped_thread_cpu_ns"] == 10*MS
    assert periodic["fully_contained_span_thread_cpu_ns"] == 6*MS
    assert periodic["boundary_crossing_spans"] == 1
    raw = parent["labels"]["stdout_raw_read"]
    assert raw["clipped_wall_union_ns"] == 500*MS and raw["unclipped_inclusive_wall_ns"] == 600*MS
    assert raw["fully_contained_span_thread_cpu_ns"] == 0
    assert row == original  # The original trusted clock/profile remains intact.


def test_child_nested_exclusive_and_unattributed_gaps_have_explicit_scope():
    result = a.validate_profile_row(profile_row())["child"]
    assert result["handle_wall_ns"] == 385*MS
    assert result["labels"]["inference"]["inclusive_wall_ns"] == 300*MS
    assert result["labels"]["inference"]["exclusive_wall_ns"] == 280*MS
    assert result["handle_unattributed_wall_ns"] == 25*MS
    segments = result["ordered_boundary_segments_ns"]
    assert sum(segments.values()) == 500*MS
    assert segments["unattributed_pre_emit_to_receipt_ns"] == 40*MS


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(request_id="different"),
    lambda p: p.update(receipt_arrival_ns=p["receipt_arrival_ns"]+1),
    lambda p: p.update(receipt_reconstructed_arrival_ns=0),
    lambda p: p.update(raw_arrivals_ns=[]),
    lambda p: p.update(dropped_spans=1),
    lambda p: p.update(pending_spans_at_snapshot={"stdin_flush": 1}),
    lambda p: p["totals"]["initial_sample"].update(wall_ns=0),
    lambda p: p["spans"][0].update(thread_cpu_ns=-1),
    lambda p: p["spans"][0].update(label="invented"),
    lambda p: p.update(child_pid=999),
])
def test_uncoupled_or_incomplete_parent_profile_is_rejected(mutation):
    row = profile_row(); mutation(row["parent_profile"])
    with pytest.raises(ValueError): a.validate_profile_row(row)


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(pid=999),
    lambda p: p.update(handle_end_ns=p["handle_start_ns"]-1),
    lambda p: p.update(before_emit_ns=None),
    lambda p: p.update(before_emit_ns=2_000*MS),
    lambda p: p.update(cpu_scope="all_processes"),
    lambda p: p.update(inclusive_labels_overlap=False),
    lambda p: p.update(handle_unattributed_wall_ns=0),
    lambda p: p["totals"]["inference"].__setitem__(0, True),
    lambda p: p["totals"]["inference"].__setitem__(4, 1),
    lambda p: p["totals"]["inference"].__setitem__(5, 500*MS),
    lambda p: p["totals"]["native_flush"].__setitem__(1, 1),
    lambda p: p.update(total_fields=["unknown"]),
])
def test_child_bounds_identity_units_and_nested_accounting_are_checked(mutation):
    row = profile_row(); mutation(row["clock"]["result"]["timing_profile"])
    with pytest.raises(ValueError): a.validate_profile_row(row)


def test_baseline_must_remain_uninstrumented():
    assert a.validate_profile_row(baseline_row()) is None
    row = profile_row(); row["mode"] = "baseline"
    with pytest.raises(ValueError, match="baseline unexpectedly"): a.validate_profile_row(row)


def test_actual_child_profiler_zero_label_table_is_accepted():
    from tools.splendorust_component_timing_20261025.child_profile import ChildProfiler
    row = profile_row(); start = row["parent_profile"]["request_start_ns"]
    times = iter([start+65*MS, start+450*MS]); cpus = iter([0, 180*MS])
    profiler = ChildProfiler(request_read_complete_ns=start+60*MS, pid=123,
                             wall_clock=lambda: next(times), cpu_clock=lambda: next(cpus))
    profiler.finish()
    child = profiler.snapshot(); child["before_emit_ns"] = start+460*MS
    row["clock"]["result"]["timing_profile"] = child
    result = a.validate_profile_row(row)
    assert result["child"]["handle_unattributed_wall_ns"] == 385*MS
    assert all(value["calls"] == 0 for value in result["child"]["labels"].values())


def test_invalid_success_or_incorrect_clock_process_tree_cannot_be_analyzed():
    row = profile_row(); row["clock"]["telemetry"]["process_tree"][0]["pid"] = 999
    with pytest.raises(ValueError, match="supervised Python"): a.validate_profile_row(row)
    row = profile_row(); row["clock"]["status"] = "timeout"
    with pytest.raises(ValueError, match="completed unchanged"): a.validate_profile_row(row)


def test_report_keeps_scopes_separate_and_never_adjusts_deadline():
    report = a.summarize([baseline_row(), profile_row()])
    assert report["rows"] == 2 and report["deadline_adjustment_seconds"] == 0.
    assert not report["causal_bottleneck_proven"]
    group = report["groups"][0]
    assert group["trusted_timing_seconds"]["profiled"]["mean"] == .5
    metric = group["profiled_metrics"]["parent.supervision_wall_union_within_trusted_ns"]
    assert metric["unit"] == "nanoseconds" and metric["distribution"]["mean"] == 90*MS
    assert "above_diagnostic_margin" not in metric["distribution"]
    assert not any("adjusted" in name for name in group["profiled_metrics"])


def test_missing_mode_cannot_be_presented_as_paired_component_summary():
    with pytest.raises(ValueError, match="both modes required"):
        a.summarize([profile_row()])
