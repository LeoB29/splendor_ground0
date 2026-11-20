"""Timing provenance and concurrency tests; no models or worker processes."""
from copy import deepcopy

import pytest

from tools.splendorust_graph_timing_20261025 import analyze as a


def refresh_parent(parent):
    totals = {}
    for span in parent["spans"]:
        total = totals.setdefault(span["label"], {"calls": 0, "wall_ns": 0, "process_cpu_ns": 0, "thread_cpu_ns": 0})
        total["calls"] += 1
        total["wall_ns"] += span["end_ns"]-span["start_ns"]
        for key in ("process_cpu_ns", "thread_cpu_ns"):
            total[key] += span[key]
    parent["totals"] = totals


def profile_row(quota=64):
    start = 1_000_000_000
    def stamp(ms):
        return start + ms*1_000_000
    spans = [dict(label=label, start_ns=stamp(lo), end_ns=stamp(hi),
                  process_cpu_ns=cpu*1_000_000, thread_cpu_ns=thread*1_000_000)
             for label, lo, hi, cpu, thread in (("initial_sample", 0, 10, 2, 2),
                 ("stdin_write", 10, 18, 1, 1), ("stdin_flush", 18, 30, 1, 1),
                 ("periodic_sample", 30, 70, 10, 8), ("aggregate_guard", 35, 55, 4, 3),
                 ("stdout_raw_read", -100, 400, 280, 2))]
    parent = dict(schema=a.PARENT_SCHEMA, request_id="request-00000001", pid=10, child_pid=42,
                  clock="perf_counter_ns", process_cpu_clock="process_time_ns", thread_cpu_clock="thread_time_ns",
                  request_start_ns=start, receipt_arrival_ns=stamp(400), receipt_reconstructed_arrival_ns=stamp(400),
                  wrapper_start_ns=stamp(-1), wrapper_end_ns=stamp(450), wrapper_process_cpu_ns=300_000_000,
                  wrapper_thread_cpu_ns=20_000_000, raw_arrivals_ns=[stamp(400)], spans=spans,
                  profile_snapshot_ns=stamp(460), request_capture_maximum_bytes=1024*1024,
                  request_raw_complete=True, request_raw_overflow=False, request_raw_bytes=100, request_raw_sha256="0"*64,
                  maximum_spans=256, dropped_spans=0, pending_spans_at_snapshot={},
                  stdin_write_start_ns=stamp(10), stdin_write_end_ns=stamp(18),
                  stdin_flush_start_ns=stamp(18), stdin_flush_end_ns=stamp(30), dispatch_complete_ns=stamp(30),
                  times_are_diagnostic_not_trusted_deadline_receipts=True, measurement_units="nanoseconds")
    refresh_parent(parent)
    totals = {label: [0]*7 for label in a.CHILD_LABELS}
    totals["inference"] = [1, 100_000_000, 80_000_000, 100_000_000, 0, 100_000_000, 80_000_000]
    totals["native_read_wait"] = [1, 40_000_000, 10_000_000, 40_000_000, 0, 40_000_000, 10_000_000]
    inner = dict(schema=a.CHILD_SCHEMA, pid=42, clock="perf_counter_ns", cpu_clock="process_time_ns",
                 measurement_units="nanoseconds", request_read_complete_ns=stamp(20), handle_start_ns=stamp(25),
                 handle_end_ns=stamp(225), handle_cpu_ns=160_000_000, handle_failed=False, totals=totals,
                 total_fields=list(a.CHILD_FIELDS), component_nested_calls=0, handle_unattributed_wall_ns=60_000_000,
                 handle_unattributed_cpu_ns=70_000_000, cpu_scope="current_python_process_only_excludes_native_rust_and_parent",
                 times_are_diagnostic_not_trusted_deadline_receipts=True, inclusive_labels_overlap=True,
                 exclusive_totals_exclude_nested_components=True,
                 partial_counters=dict(inference_calls_returned=1, inference_calls_failed=0,
                     last_inference_end_ns=stamp(125), last_inference_wall_ns=100_000_000,
                     last_inference_cpu_ns=80_000_000, last_exchange_callback_hash_count=1, last_exchange_callback_rows=1,
                     callback_scope="last_native_select_exchange_only_not_all_decision_phases"))
    observer = {"example": "corrected"}
    result = {"fixed": True}
    child = dict(schema=a.CHILD_SCHEMA, request_id="request-00000001", pid=42, quota=quota,
                 backend="qualified_graph64" if quota == 64 else "unchanged_eager32",
                 graph_identity_sha256=a.GRAPH_IDENTITY if quota == 64 else None,
                 clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()",
                 request_read_complete_ns=stamp(20), request_started_seconds=1.019,
                 pre_write_ns=stamp(230), post_flush_ns=stamp(410), outcome="response_flushed", error=None,
                 profiles=[inner], request_raw_sha256="0"*64,
                 payload_sha256=a._digest({"player": "search", "observer": observer}), result_sha256=a._digest(result),
                 serialized_packet_utf8_sha256="1"*64, serialized_packet_utf8_bytes=100,
                 times_are_diagnostic_not_trusted_deadline_receipts=True, success_sidecar_written_after_response_flush=True,
                 post_flush_may_follow_parent_receipt=True, child_read_may_precede_parent_dispatch_flush_completion=True,
                 killed_worker_sidecar_may_be_missing=True)
    clock = dict(request_id="request-00000001", status="ok", accepted=True, received_before_deadline=True,
                 deadline_seconds=1., receipt_elapsed_seconds=.4, result=result,
                 telemetry={"process_tree": [{"role": "python", "pid": 42}]})
    return dict(index=0, stage="overhead", fixture_id="saved/one.json", pass_index=0, mode="profiled", quota=quota,
                player_role="challenger" if quota == 64 else "opponent", completed=True, clock=clock, observer=observer,
                child_profile=child, parent_profile=parent)


def test_valid_overlaps_are_not_false_causal_ordering_failures():
    result = a.validate_profile_row(profile_row())
    child = result["child"]
    assert child["signed_dispatch_complete_to_child_read_ns"] == -10_000_000
    assert child["signed_post_flush_to_trusted_arrival_ns"] == -10_000_000
    assert sum(child["ordered_boundary_segments_ns"].values()) == result["trusted_arrival_ns"] == 400_000_000
    # Nested 20ms aggregate guard is already inside a 40ms sample.
    assert result["parent"]["supervision_wall_union_within_trusted_ns"] == 50_000_000


def test_boundary_crossing_cpu_is_never_fractionally_clipped():
    result = a.validate_profile_row(profile_row())
    raw = result["parent"]["labels"]["stdout_raw_read"]
    assert raw["clipped_wall_union_ns"] == 400_000_000
    assert raw["unclipped_inclusive_wall_ns"] == 500_000_000
    assert raw["unclipped_thread_cpu_ns"] == 2_000_000
    assert raw["fully_contained_span_thread_cpu_ns"] == 0
    assert raw["boundary_crossing_spans"] == 1


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["clock"].update(receipt_elapsed_seconds=.401), "receipt"),
    (lambda r: r["child_profile"].update(pid=43), "PID"),
    (lambda r: r["child_profile"].update(graph_identity_sha256="2"*64), "backend"),
    (lambda r: r["child_profile"].update(payload_sha256="2"*64), "input"),
    (lambda r: r["child_profile"].update(result_sha256="2"*64), "result"),
    (lambda r: r["child_profile"].update(quota=32), "quota"),
    (lambda r: r["parent_profile"].update(dispatch_complete_ns=1), "dispatch"),
    (lambda r: r["parent_profile"].update(pending_spans_at_snapshot={"stdin_flush": 1}), "complete parent"),
    (lambda r: r["parent_profile"].update(dropped_spans=1), "complete parent"),
    (lambda r: r["child_profile"]["profiles"][0]["totals"]["inference"].__setitem__(5, 300_000_000), "consistency"),
])
def test_false_binding_or_incomplete_success_is_rejected(mutation, match):
    row = profile_row()
    mutation(row)
    with pytest.raises(ValueError, match=match):
        a.validate_profile_row(row)


def test_fixed_zero_component_and_no_neural_call_profiles_are_valid():
    row = profile_row(32)
    inner = row["child_profile"]["profiles"][0]
    inner["totals"]["inference"] = [0]*7
    inner.update(handle_unattributed_wall_ns=160_000_000, handle_unattributed_cpu_ns=150_000_000)
    inner["partial_counters"].update(inference_calls_returned=0, last_inference_end_ns=None,
                                    last_inference_wall_ns=None, last_inference_cpu_ns=None)
    assert a.validate_profile_row(row)["child"]["labels"]["inference"]["calls"] == 0


def test_failure_missing_child_is_disclosed_and_not_success():
    row = profile_row()
    row["completed"] = False
    row["clock"].update(status="worker_error", accepted=False, result=None)
    row["child_profile"] = None
    result = a.summarize([row])
    assert not result["all_collected_decisions_completed"]
    assert result["failures"][0]["profile_evidence"]["missing_child_profile"]
    assert result["groups"][0]["trusted_completed_timing_seconds"]["profiled"]["count"] == 0
    row["completed"] = True
    row["clock"].update(status="ok", accepted=True, result={})
    with pytest.raises(ValueError, match="child profile missing"):
        a.validate_profile_row(row)


def test_late_success_packet_stays_failed_but_retains_timing():
    row = profile_row()
    row["completed"] = False
    row["clock"].update(status="timeout", accepted=False, received_before_deadline=False,
                         receipt_elapsed_seconds=1.1, result=None)
    parent = row["parent_profile"]
    parent.update(receipt_arrival_ns=2_100_000_000, receipt_reconstructed_arrival_ns=2_100_000_000,
                  wrapper_end_ns=2_200_000_000, profile_snapshot_ns=2_210_000_000, raw_arrivals_ns=[2_100_000_000])
    next(s for s in parent["spans"] if s["label"] == "stdout_raw_read")["end_ns"] = 2_100_000_000
    refresh_parent(parent)
    result = a.summarize([row])
    assert result["failures"][0]["profile_evidence"]["trusted_arrival_ns"] == 1_100_000_000
    assert not result["all_collected_decisions_completed"]
    assert result["deadline_adjustment_seconds"] == 0


def test_bad_failure_evidence_is_preserved_not_silently_passed():
    row = profile_row()
    row["completed"] = False
    row["clock"].update(status="worker_error", accepted=False)
    row["child_profile"]["pid"] = 99
    result = a.summarize([row])
    assert result["failures"][0]["profile_validation"] == "invalid_or_unbound_preserved_evidence"
    assert "PID" in result["failures"][0]["profile_validation_error"]
    assert not result["complete_planned_collection_inferred"]


def test_no_arbitrary_short_marker_gap_gate_hides_scheduling_stall():
    row = profile_row()
    # A pause between the internal clock read and the ns marker is possible;
    # this is diagnostic evidence, not an alternative deadline calculation.
    row["child_profile"]["request_started_seconds"] = .8
    assert a.validate_profile_row(row)["complete_decision"]


def test_partial_failures_can_preserve_pending_parent_spans():
    row = profile_row()
    row["completed"] = False
    row["clock"].update(status="worker_error", accepted=False)
    row["parent_profile"].update(pending_spans_at_snapshot={"stdin_flush": 1})
    assert a.validate_profile_row(row)["parent"]["pending_spans"] == 1


def test_baseline_may_not_contain_instrumentation():
    row = profile_row()
    row["mode"] = "baseline"
    with pytest.raises(ValueError, match="baseline unexpectedly"):
        a.validate_profile_row(row)
    row.update(parent_profile=None, child_profile=None)
    assert not a.validate_profile_row(row)["instrumented"]


def test_signed_gap_summaries_and_child_bounds():
    row = profile_row()
    result = a.summarize([row])
    metrics = result["groups"][0]["profiled_metrics"]
    assert metrics["child.signed_post_flush_to_trusted_arrival_ns"]["distribution"]["mean"] == -10_000_000
    bad = deepcopy(row)
    bad["child_profile"]["profiles"][0]["partial_counters"]["last_inference_end_ns"] = 9_000_000_000
    with pytest.raises(ValueError, match="last inference"):
        a.validate_profile_row(bad)


def startup_record(operation="entrypoint_checks"):
    row = profile_row()
    child = row["child_profile"]
    payload = {"operation": operation}
    if operation == "entrypoint_checks":
        result = {"status": "entrypoint_counterexamples_passed", "challenger_quota": 64,
                  "counts": {"complete_decisions": 52}}
        original = child["profiles"][0]
        profiles = []
        for index in range(63):
            inner = deepcopy(original)
            start = child["request_read_complete_ns"]+index*1_000_000
            inner.update(request_read_complete_ns=None, handle_start_ns=start, handle_end_ns=start+1_000_000,
                         handle_cpu_ns=0, handle_failed=index >= 37,
                         totals={label: [0]*7 for label in a.CHILD_LABELS},
                         handle_unattributed_wall_ns=1_000_000, handle_unattributed_cpu_ns=0)
            inner["partial_counters"].update(inference_calls_returned=0, inference_calls_failed=0,
                last_inference_end_ns=None, last_inference_wall_ns=None, last_inference_cpu_ns=None,
                last_exchange_callback_hash_count=0, last_exchange_callback_rows=0)
            profiles.append(inner)
        child["profiles"] = profiles
    else:
        payload["fixtures"] = [{}]*11
        result = {"status": "passed"}
        child["profiles"] = []
    row["clock"]["result"] = result
    row["clock"]["deadline_seconds"] = 300.
    child["payload_sha256"] = a._digest(payload)
    child["result_sha256"] = a._digest(result)
    return row["parent_profile"], child, row["clock"], payload


def test_startup63_handles37_success26_expected_rejections():
    parent, child, receipt, payload = startup_record()
    result = a.validate_startup_profiles(parent, child, receipt, payload, 64, worker_pid=42)
    assert result["profiles"] == 63 and result["wrapped_successes"] == 37
    assert result["intentional_rejections"] == 26
    assert result["unprofiled_direct_reference_reproductions"] == 15
    assert result["parent_component_attribution_not_performed"]
    child["profiles"] = child["profiles"][:52]
    with pytest.raises(ValueError, match="exact63"):
        a.validate_startup_profiles(parent, child, receipt, payload, 64)


def test_startup_rejection_order_and_inner_null_marker_are_checked():
    parent, child, receipt, payload = startup_record()
    child["profiles"][37]["handle_failed"] = False
    with pytest.raises(ValueError, match="order"):
        a.validate_startup_profiles(parent, child, receipt, payload, 64)
    child["profiles"][37]["handle_failed"] = True
    child["profiles"][0]["request_read_complete_ns"] = child["request_read_complete_ns"]
    with pytest.raises(ValueError, match="read marker"):
        a.validate_startup_profiles(parent, child, receipt, payload, 64)


def test_startup_dropped_parent_spans_remain_bounded_unattributed():
    parent, child, receipt, payload = startup_record()
    raw = parent["spans"].pop()
    parent["dropped_spans"] = 1
    assert raw["label"] == "stdout_raw_read"
    result = a.validate_startup_profiles(parent, child, receipt, payload, 64)
    assert result["parent_dropped_spans"] == 1
    assert result["parent_component_attribution_not_performed"]


def test_graph_parity_zero_handles_and_backend_are_bound():
    parent, child, receipt, payload = startup_record("graph_parity")
    assert a.validate_startup_profiles(parent, child, receipt, payload, 64)["profiles"] == 0
    child["quota"] = 32
    child["backend"] = "unchanged_eager32"
    child["graph_identity_sha256"] = None
    with pytest.raises(ValueError, match="approved startup"):
        a.validate_startup_profiles(parent, child, receipt, payload, 32)


def test_actual_request_bytes_bind_parent_and_child():
    row = profile_row()
    row["child_profile"]["request_raw_sha256"] = "9"*64
    with pytest.raises(ValueError, match="transported request"):
        a.validate_profile_row(row)


def test_writer_span_can_finish_after_wrapper_before_snapshot():
    row = profile_row()
    parent = row["parent_profile"]
    flush = next(s for s in parent["spans"] if s["label"] == "stdin_flush")
    flush["end_ns"] = 1_455_000_000
    parent["stdin_flush_end_ns"] = parent["dispatch_complete_ns"] = flush["end_ns"]
    refresh_parent(parent)
    result = a.validate_profile_row(row)
    assert result["child"]["signed_dispatch_complete_to_child_read_ns"] == -435_000_000
