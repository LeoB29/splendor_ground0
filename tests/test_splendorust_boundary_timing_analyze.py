"""Minimal boundary provenance, concurrent transport and preserved failure checks."""
from copy import deepcopy
import pytest
from tools.splendorust_boundary_timing_20261025 import analyze as a


def profile_row(quota=64):
    start = 1_000_000_000
    def stamp(ms):
        return start+ms*1_000_000
    parent = dict(schema=a.PARENT_SCHEMA, request_id="request-00000001", pid=10, child_pid=42,
        clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()", request_start_ns=start,
        receipt_arrival_ns=stamp(400), receipt_reconstructed_arrival_ns=stamp(400), capture_snapshot_ns=stamp(460),
        raw_arrivals_ns=[stamp(400)], maximum_arrival_events=16, dropped_arrival_events=0,
        request_capture_maximum_bytes=1024*1024, request_raw_complete=True, request_raw_overflow=False,
        request_raw_bytes=100, request_raw_sha256="0"*64,
        response_raw_sha256="1"*64, response_raw_bytes=100, response_capture_overflow=False,
        response_capture_maximum_bytes=4*1024*1024,
        stdin_write_start_ns=stamp(10), stdin_write_end_ns=stamp(18), stdin_flush_start_ns=stamp(18),
        stdin_flush_end_ns=stamp(30), dispatch_complete_ns=stamp(30), dispatch_write_calls=1, dispatch_flush_calls=1,
        dispatch_write_failures=0, dispatch_flush_failures=0, pending_writes=0, pending_flushes=0,
        times_are_diagnostic_not_trusted_deadline_receipts=True, measurement_units="nanoseconds",
        raw_reader_semantics="unchanged raw FileIO.readline then immediate trusted perf_counter_ns",
        child_read_may_precede_dispatch_completion=True, child_post_flush_may_follow_trusted_arrival=True,
        hashes_calculated_after_original_request_returns=True, dispatch_markers_may_be_partial_if_pending=True)
    profile = dict(schema=a.CHILD_SCHEMA, pid=42, clock="perf_counter_ns", measurement_units="nanoseconds",
        request_read_complete_ns=stamp(20), handle_start_ns=stamp(25), handle_end_ns=stamp(225), handle_failed=False,
        times_are_diagnostic_not_trusted_deadline_receipts=True)
    observer, result = {"example": "corrected"}, {"fixed": True}
    child = dict(schema=a.CHILD_SCHEMA, request_id="request-00000001", pid=42, quota=quota,
        backend="qualified_graph64" if quota == 64 else "unchanged_eager32",
        graph_identity_sha256=a.GRAPH_IDENTITY if quota == 64 else None,
        clock="perf_counter_ns", clock_implementation="QueryPerformanceCounter()",
        request_read_complete_ns=stamp(20), request_started_seconds=1.019,
        pre_write_ns=stamp(230), post_flush_ns=stamp(410), outcome="response_flushed", error=None,
        profiles=[profile], request_raw_sha256="0"*64,
        payload_sha256=a._digest({"player": "search", "observer": observer}), result_sha256=a._digest(result),
        serialized_packet_utf8_sha256="1"*64, serialized_packet_utf8_bytes=100,
        times_are_diagnostic_not_trusted_deadline_receipts=True, success_sidecar_written_after_response_flush=True,
        post_flush_may_follow_parent_receipt=True, child_read_may_precede_parent_dispatch_flush_completion=True,
        killed_worker_sidecar_may_be_missing=True, text_transport_may_expand_newlines=True)
    clock = dict(request_id="request-00000001", status="ok", accepted=True, received_before_deadline=True,
        deadline_seconds=1., receipt_elapsed_seconds=.4, result=result,
        telemetry={"process_tree": [{"role": "python", "pid": 42}]})
    return dict(index=0, stage="overhead", fixture_id="saved/one.json", pass_index=0, mode="profiled", quota=quota,
        player_role="challenger" if quota == 64 else "opponent", completed=True, clock=clock, observer=observer,
        child_profile=child, parent_profile=parent)


def test_pipe_overlap_is_valid_and_ordered_segments_partition_time():
    result = a.validate_profile_row(profile_row())
    child = result["child"]
    assert child["signed_dispatch_complete_to_child_read_ns"] == -10_000_000
    assert child["signed_post_flush_to_trusted_arrival_ns"] == -10_000_000
    assert sum(child["ordered_boundary_segments_ns"].values()) == result["trusted_arrival_ns"] == 400_000_000
    assert child["handle_wall_ns"] == 200_000_000
    assert not result["causal_component_attribution_complete"]


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r["clock"].update(receipt_elapsed_seconds=.401), "receipt"),
    (lambda r: r["child_profile"].update(pid=43), "PID"),
    (lambda r: r["child_profile"].update(graph_identity_sha256="2"*64), "backend"),
    (lambda r: r["child_profile"].update(payload_sha256="2"*64), "input"),
    (lambda r: r["child_profile"].update(result_sha256="2"*64), "result"),
    (lambda r: r["child_profile"].update(quota=32), "quota"),
    (lambda r: r["parent_profile"].update(dispatch_complete_ns=1), "dispatch"),
    (lambda r: r["parent_profile"].update(pending_writes=1), "single dispatch"),
    (lambda r: r["parent_profile"].update(dropped_arrival_events=1), "truncated"),
    (lambda r: r["parent_profile"].update(response_raw_sha256="2"*64), "response hash"),
    (lambda r: r["parent_profile"].update(response_raw_bytes=102), "CRLF"),
    (lambda r: r["parent_profile"].update(spans=[]), "minimal parent"),
    (lambda r: r["child_profile"]["profiles"][0].update(handle_cpu_ns=5), "minimal child"),
    (lambda r: r["parent_profile"].update(request_raw_complete=False), "capture incomplete"),
    (lambda r: r["child_profile"].update(request_raw_sha256="3"*64), "transported request"),
    (lambda r: r["clock"].update(deadline_seconds=1.1), "unchanged hard-clock"),
    (lambda r: r["child_profile"].update(clock_implementation="different"), "QPC"),
    (lambda r: r["child_profile"]["profiles"][0].update(handle_end_ns=2_000_000_000), "emission"),
])
def test_mutated_identity_boundaries_or_added_detail_rejected(mutation, match):
    row = profile_row(); mutation(row)
    with pytest.raises(ValueError, match=match):
        a.validate_profile_row(row)


def test_crlf_size_difference_retained_for_exact_raw_log_checker():
    row = profile_row()
    row["parent_profile"].update(response_raw_bytes=101, response_raw_sha256="4"*64)
    assert a.validate_profile_row(row)["complete_decision"]


def test_final_guard_marker_gap_has_no_arbitrary_latency_gate():
    row = profile_row(32)
    row["child_profile"]["request_started_seconds"] = .8
    assert a.validate_profile_row(row)["complete_decision"]


def failed_row():
    row = profile_row()
    row["completed"] = False
    row["clock"].update(status="worker_error", accepted=False, result=None)
    return row


def test_missing_external_kill_sidecar_is_disclosed_without_success():
    row = failed_row(); row["child_profile"] = None
    result = a.summarize([row])
    assert not result["all_collected_decisions_completed"]
    assert result["failures"][0]["profile_evidence"]["missing_child_profile"]
    assert result["groups"][0]["trusted_completed_timing_seconds"]["profiled"]["count"] == 0
    row["completed"] = True; row["clock"].update(status="ok", accepted=True, result={})
    with pytest.raises(ValueError, match="child profile missing"):
        a.validate_profile_row(row)


def test_no_parent_no_child_failure_is_explicitly_partial():
    row = failed_row(); row.update(parent_profile=None, child_profile=None)
    result = a.validate_profile_row(row)
    assert result["missing_parent_profile"] and result["missing_child_profile"]


def test_late_completed_packet_is_not_promoted():
    row = failed_row()
    row["clock"].update(status="timeout", received_before_deadline=False, receipt_elapsed_seconds=1.1)
    row["parent_profile"].update(receipt_arrival_ns=2_100_000_000, receipt_reconstructed_arrival_ns=2_100_000_000,
        capture_snapshot_ns=2_200_000_000, raw_arrivals_ns=[2_100_000_000])
    result = a.summarize([row])
    assert result["failures"][0]["profile_evidence"]["trusted_arrival_ns"] == 1_100_000_000
    assert not result["all_collected_decisions_completed"]
    assert result["deadline_adjustment_seconds"] == 0


def test_bad_failure_evidence_preserved_not_silently_accepted():
    row = failed_row(); row["child_profile"]["pid"] = 99
    result = a.summarize([row])
    assert result["failures"][0]["profile_validation"] == "invalid_or_unbound_preserved_evidence"
    assert not result["complete_planned_collection_inferred"]


def test_partial_failure_before_emission_has_no_fabricated_delivery():
    row = failed_row(); child = row["child_profile"]
    child.update(outcome="failed_before_successful_emission", error="original deadline",
        pre_write_ns=None, post_flush_ns=None, serialized_packet_utf8_sha256=None,
        serialized_packet_utf8_bytes=None, result_sha256=None)
    child["profiles"][0]["handle_failed"] = True
    result = a.validate_profile_row(row)
    assert result["child"]["handle_wall_ns"] == 200_000_000
    assert "pre_write_to_trusted_arrival_ns" not in result["child"]


def test_no_arrival_pending_flush_failure_remains_partial():
    row = failed_row(); row["child_profile"] = None
    row["clock"]["receipt_elapsed_seconds"] = None
    parent = row["parent_profile"]
    parent.update(receipt_arrival_ns=None, receipt_reconstructed_arrival_ns=None, raw_arrivals_ns=[],
        response_raw_sha256=None, response_raw_bytes=None, response_capture_overflow=None,
        stdin_flush_end_ns=None, dispatch_complete_ns=None, pending_flushes=1)
    assert a.validate_profile_row(row)["parent"]["pending_flushes"] == 1


def test_baseline_may_not_have_observer_and_summary_retains_signed_gaps():
    row = profile_row()
    result = a.summarize([row])
    assert result["groups"][0]["profiled_metrics"]["child.signed_post_flush_to_trusted_arrival_ns"]["distribution"]["mean"] == -10_000_000
    assert not result["causal_bottleneck_proven"]
    row["mode"] = "baseline"
    with pytest.raises(ValueError, match="baseline unexpectedly"):
        a.validate_profile_row(row)
    row.update(parent_profile=None, child_profile=None)
    assert not a.validate_profile_row(row)["instrumented"]


def startup_record(operation="entrypoint_checks"):
    row = profile_row(); child = row["child_profile"]
    payload = {"operation": operation}
    if operation == "entrypoint_checks":
        result = {"status": "entrypoint_counterexamples_passed", "challenger_quota": 64,
                  "counts": {"complete_decisions": 52}}
        profiles = []
        for index in range(63):
            inner = deepcopy(child["profiles"][0]); start = child["request_read_complete_ns"]+index*1_000_000
            inner.update(request_read_complete_ns=None, handle_start_ns=start, handle_end_ns=start+1_000_000,
                         handle_failed=index >= 37)
            profiles.append(inner)
        child["profiles"] = profiles
    else:
        payload["fixtures"] = [{}]*11; result = {"status": "passed"}; child["profiles"] = []
    row["clock"].update(result=result, deadline_seconds=300.)
    child.update(payload_sha256=a._digest(payload), result_sha256=a._digest(result))
    return row["parent_profile"], child, row["clock"], payload


def test_actual_startup_63_handles_plus_15_direct_reference_checks():
    parent, child, receipt, payload = startup_record()
    result = a.validate_startup_profiles(parent, child, receipt, payload, 64, worker_pid=42)
    assert result["profiles"] == 63 and result["wrapped_successes"] == 37 and result["intentional_rejections"] == 26
    assert result["unprofiled_direct_reference_reproductions"] == 15
    child["profiles"] = child["profiles"][:52]
    with pytest.raises(ValueError, match="exact63"):
        a.validate_startup_profiles(parent, child, receipt, payload, 64)


def test_startup_rejection_order_and_no_transport_marker_on_inner_call():
    parent, child, receipt, payload = startup_record()
    child["profiles"][37]["handle_failed"] = False
    with pytest.raises(ValueError, match="order"):
        a.validate_startup_profiles(parent, child, receipt, payload, 64)
    child["profiles"][37]["handle_failed"] = True
    child["profiles"][0]["request_read_complete_ns"] = child["request_read_complete_ns"]
    with pytest.raises(ValueError, match="read marker"):
        a.validate_startup_profiles(parent, child, receipt, payload, 64)


def test_graph_parity_no_fabricated_handle_and_graph32_rejected():
    parent, child, receipt, payload = startup_record("graph_parity")
    assert a.validate_startup_profiles(parent, child, receipt, payload, 64)["profiles"] == 0
    child.update(quota=32, backend="unchanged_eager32", graph_identity_sha256=None)
    with pytest.raises(ValueError, match="approved startup"):
        a.validate_startup_profiles(parent, child, receipt, payload, 32)


def test_writer_flush_may_finish_after_trusted_response_arrival():
    row = profile_row(); row["parent_profile"].update(stdin_flush_end_ns=1_455_000_000, dispatch_complete_ns=1_455_000_000)
    assert a.validate_profile_row(row)["child"]["signed_dispatch_complete_to_child_read_ns"] == -435_000_000


def test_original_partial_write_loop_is_a_complete_single_dispatch():
    row = profile_row()
    row["parent_profile"]["dispatch_write_calls"] = 4
    assert a.validate_profile_row(row)["complete_decision"]


def test_failed_stdout_flush_retains_attempted_write_without_claiming_delivery():
    row = failed_row(); child = row["child_profile"]
    child.update(outcome="failed_before_successful_emission", error="stdout flush failed", post_flush_ns=None,
        serialized_packet_utf8_sha256=None, serialized_packet_utf8_bytes=None)
    result = a.validate_profile_row(row)
    assert result["child"]["partial_emission_attempt_recorded"]
    assert result["child"]["handle_end_to_partial_pre_write_ns"] == 5_000_000
    assert "ordered_boundary_segments_ns" not in result["child"]
    assert "pre_write_to_trusted_arrival_ns" not in result["child"]
    assert not result["complete_decision"]
    child["post_flush_ns"] = 1_410_000_000
    with pytest.raises(ValueError, match="marker contradiction"):
        a.validate_profile_row(row)


def test_partial_pre_write_must_follow_handler_end():
    row = failed_row(); child = row["child_profile"]
    child.update(outcome="failed_before_successful_emission", error="stdout write failed", pre_write_ns=1_200_000_000,
        post_flush_ns=None, serialized_packet_utf8_sha256=None, serialized_packet_utf8_bytes=None)
    with pytest.raises(ValueError, match="partial child emission order"):
        a.validate_profile_row(row)
