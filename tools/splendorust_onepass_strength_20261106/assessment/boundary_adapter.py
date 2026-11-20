"""Prospectively expanded raw capture bound; all original timestamp checks retained."""
from tools.splendorust_boundary_timing_20261025.analyze import PARENT_SCHEMA,require,_int,_finite,_sha,_validate_handle

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
    require(parent.get("response_capture_maximum_bytes") == 8*1024*1024, "response capture bound")
    if arrival is not None:
        require(type(parent.get("response_capture_overflow")) is bool, "response overflow flag")
        _sha(parent.get("response_raw_sha256"), "actual raw response")
        require(0 < _int(parent.get("response_raw_bytes"), "actual response bytes") <= 8*1024*1024,
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
