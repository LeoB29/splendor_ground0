"""Model-free transparent delegation, nesting and observer evidence checks."""
import io
import json
import threading

import pytest

from tools.splendorust_gpu_costs_20261105.observer import Recorder, TimedPipe, LABELS


def events(recorder):
    value = recorder.snapshot()
    return [dict(zip(value["columns"], row)) for row in value["events"]]


def test_nested_wall_spans_keep_explicit_parent_order_and_owner():
    values = iter((10., 11., 12., 13.))
    recorder = Recorder(True, clock=lambda: next(values))
    with recorder.span("callback_total", "client7"):
        with recorder.span("callback_bookkeeping"):
            pass
    child, parent = events(recorder)
    assert (parent["span_id"], child["span_id"], child["parent_id"]) == (1, 2, 1)
    assert parent["parent_id"] is None and child["owner"] == parent["owner"] == "client7"
    assert (parent["start"], child["start"], child["end"], parent["end"]) == (10., 11., 12., 13.)
    assert not child["failed"] and not parent["failed"]


def test_disabled_recorder_uses_no_span_clock_and_keeps_zero_events():
    def forbidden():
        raise AssertionError("disabled span must not read a clock")
    recorder = Recorder(False, clock=forbidden)
    with recorder.span("callback_total"):
        with recorder.span("pipe_readline"):
            pass
    assert recorder.snapshot()["events"] == []


def test_span_preserves_original_exception_identity():
    recorder = Recorder(True)
    expected = OSError("original I/O error")
    with pytest.raises(OSError) as caught:
        with recorder.span("pipe_write", "client"):
            raise expected
    assert caught.value is expected
    assert events(recorder)[0]["failed"] is True


def test_parallel_nesting_never_uses_another_threads_parent():
    recorder = Recorder(True)
    barrier = threading.Barrier(3)
    def run(name):
        with recorder.span("callback_total", name):
            barrier.wait(2)
            with recorder.span("callback_bookkeeping"):
                pass
    threads = [threading.Thread(target=run, args=(f"client{i}",)) for i in range(2)]
    for thread in threads: thread.start()
    barrier.wait(2)
    for thread in threads: thread.join(2)
    rows = events(recorder); by_id = {row["span_id"]: row for row in rows}
    assert len(rows) == 4 and len(by_id) == 4
    for row in rows:
        if row["parent_id"] is not None:
            parent = by_id[row["parent_id"]]
            assert row["owner"] == parent["owner"] and row["thread_id"] == parent["thread_id"]
            assert parent["start"] <= row["start"] <= row["end"] <= parent["end"]


def test_markers_and_exclusive_durable_export_preserve_scope(tmp_path):
    recorder = Recorder(True)
    first = recorder.mark("panel_start")
    with recorder.span("pipe_readline", "c1"):
        pass
    last = recorder.mark("panel_end")
    path = tmp_path/"observer.json"
    result = recorder.export(path, scope={"mode": "panel", "observed": True})
    saved = json.loads(path.read_text())
    assert result["bytes"] == path.stat().st_size
    assert saved["scope"] == {"mode": "panel", "observed": True}
    assert saved["markers"] == [first, last]
    assert first["event_index"] == 0 and last["event_index"] == 1
    assert first["perf_counter"] <= saved["events"][0][-3] <= last["perf_counter"]
    with pytest.raises(RuntimeError, match="only once"):
        recorder.export(path, scope={})
    with pytest.raises(RuntimeError, match="after export"):
        with recorder.span("pipe_write"): pass


def test_export_cannot_replace_existing_evidence(tmp_path):
    path = tmp_path/"existing.json"; path.write_text("original evidence")
    recorder = Recorder(False)
    with pytest.raises(FileExistsError): recorder.export(path, scope={})
    assert path.read_text() == "original evidence" and not recorder.exported


def test_actual_event_capacity_failure_is_explicit_and_preserves_prefix():
    recorder = Recorder(True, maximum_events=1)
    with recorder.span("pipe_read"): pass
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        with recorder.span("pipe_write"): pass
    assert len(recorder.events) == 1


def test_pipe_proxy_preserves_bytes_text_reads_writes_flush_and_attributes():
    recorder = Recorder(True)
    stream = io.StringIO("first\nsecond\n")
    proxy = TimedPipe(stream, recorder, "native0")
    assert proxy.readline(3) == "fir"
    assert proxy.read(2) == "st"
    assert proxy.readline() == "\n"
    assert next(iter(proxy)) == "second\n"
    assert proxy.write("new") == 3
    assert proxy.flush() is None and proxy.tell() == stream.tell()
    assert not proxy.closed
    labels = [LABELS[row["kind"]] for row in events(recorder)]
    assert labels == ["pipe_readline", "pipe_read", "pipe_readline", "pipe_iteration_read", "pipe_write", "pipe_flush"]
    assert {row["owner"] for row in events(recorder)} == {"native0"}


@pytest.mark.parametrize("method", ["readline", "read", "read1", "write", "flush"])
def test_pipe_proxy_passes_arguments_return_values_and_exceptions_exactly(method):
    marker = object(); expected = BrokenPipeError("saved original failure")
    class Stream:
        failing = False
        calls = []
        def operation(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            if self.failing: raise expected
            return marker
    setattr(Stream, method, Stream.operation)
    stream = Stream(); recorder = Recorder(True); proxy = TimedPipe(stream, recorder, "native3")
    assert getattr(proxy, method)("payload", size=7) is marker
    stream.failing = True
    with pytest.raises(BrokenPipeError) as caught:
        getattr(proxy, method)("payload2", size=8)
    assert caught.value is expected
    assert stream.calls == [(("payload",), {"size": 7}), (("payload2",), {"size": 8})]
    assert [row["failed"] for row in events(recorder)] == [False, True]


def test_pipe_context_manager_delegates_close_without_owning_extra_resources():
    recorder = Recorder(True); stream = io.StringIO()
    proxy = TimedPipe(stream, recorder, "native")
    with proxy as returned:
        assert returned is proxy
        returned.write("unchanged")
    assert stream.closed and proxy.closed
