"""Small in-memory wall-time spans; never a GPU-kernel timing claim.

The regular scientific/audit journal is retained unchanged. These tuples are
additional evidence and are exported only after the measured workload ends.
Concurrent client durations overlap and cannot be added as elapsed wall time.
"""
from contextlib import contextmanager
import hashlib
import itertools
import json
import os
from pathlib import Path
import threading
import time
import uuid


LABELS = (
    "raw_forward_total", "raw_input_validation", "tensor_construction",
    "gpu_roundtrip_synchronized", "output_conversion", "raw_output_validation",
    "service_dispatch_total", "service_input_packing", "service_output_validation",
    "service_receipts", "service_signal", "journal_total", "journal_serialization",
    "journal_write_flush", "callback_total", "callback_entry_checks",
    "callback_service_wait", "callback_bookkeeping", "pipe_readline", "pipe_read",
    "pipe_read1", "pipe_write", "pipe_flush", "pipe_iteration_read",
    "case_recording",
)
KIND = {label: index for index, label in enumerate(LABELS)}
COLUMNS = ("span_id", "parent_id", "kind", "thread_id", "owner", "start", "end", "failed")
MAX_EVENTS = 2_000_000


class Recorder:
    def __init__(self, enabled, *, clock=time.perf_counter, maximum_events=MAX_EVENTS):
        if type(enabled) is not bool or type(maximum_events) is not int or maximum_events <= 0:
            raise ValueError("Explicit observer mode and positive bounded tuple capacity required")
        self.enabled = enabled
        self.clock = clock
        self.maximum_events = maximum_events
        self.session_id = uuid.uuid4().hex
        self.events = []
        self.markers = []
        self.ids = itertools.count(1)
        self.local = threading.local()
        self.exported = False

    @contextmanager
    def span(self, label, owner=None):
        if not self.enabled:
            yield
            return
        if label not in KIND:
            raise ValueError("Unknown observer span label: " + str(label))
        if self.exported:
            raise RuntimeError("Observer cannot accept spans after export")
        stack = getattr(self.local, "stack", None)
        if stack is None:
            stack = self.local.stack = []
        span_id = next(self.ids)
        parent = stack[-1][0] if stack else None
        resolved_owner = str(owner) if owner is not None else (stack[-1][1] if stack else "unscoped")
        thread_id = threading.get_ident()
        started = self.clock()
        stack.append((span_id, resolved_owner))
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            ended = self.clock()
            stack.pop()
            if len(self.events) >= self.maximum_events:
                raise RuntimeError("Actual observer tuple capacity exceeded")
            if ended < started:
                raise RuntimeError("Observer monotonic clock moved backwards")
            self.events.append((span_id, parent, KIND[label], thread_id, resolved_owner,
                                started, ended, failed))

    def mark(self, label):
        if self.exported or type(label) is not str or not label:
            raise ValueError("Explicit marker before export required")
        marker = dict(label=label, event_index=len(self.events), perf_counter=self.clock())
        self.markers.append(marker)
        return dict(marker)

    def snapshot(self):
        return dict(schema="gpu_cost_wall_spans_v1", enabled=self.enabled,
                    session_id=self.session_id, clock="time.perf_counter wall time",
                    columns=list(COLUMNS), labels=list(LABELS), events=list(self.events),
                    markers=[dict(marker) for marker in self.markers],
                    maximum_events=self.maximum_events,
                    warnings=["GPU roundtrip includes host work, transfers, device execution and synchronized readback; not GPU kernel time",
                              "Pipe read waits include native CPU work and operating-system scheduling",
                              "Nested spans and concurrent-client spans overlap; do not sum them as elapsed wall time"])

    def export(self, path, *, scope):
        if self.exported:
            raise RuntimeError("Observer evidence may be exported only once")
        path = Path(path)
        payload = self.snapshot()
        payload["scope"] = scope
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.exported = True
        content = path.read_bytes()
        root = Path(__file__).resolve().parents[2]
        resolved = path.resolve()
        return dict(path=resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(resolved),
                    bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


class TimedPipe:
    """Instance-local delegation for the exact pipe operations used by native I/O.

    This wrapper neither buffers, encodes, decodes, retries nor alters exceptions.
    The parent's original clock remains responsible for blocking operations.
    """
    def __init__(self, stream, recorder, owner):
        self._stream, self._recorder, self._owner = stream, recorder, str(owner)

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def readline(self, *args, **kwargs):
        with self._recorder.span("pipe_readline", self._owner):
            return self._stream.readline(*args, **kwargs)

    def read(self, *args, **kwargs):
        with self._recorder.span("pipe_read", self._owner):
            return self._stream.read(*args, **kwargs)

    def read1(self, *args, **kwargs):
        with self._recorder.span("pipe_read1", self._owner):
            return self._stream.read1(*args, **kwargs)

    def write(self, *args, **kwargs):
        with self._recorder.span("pipe_write", self._owner):
            return self._stream.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        with self._recorder.span("pipe_flush", self._owner):
            return self._stream.flush(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        with self._recorder.span("pipe_iteration_read", self._owner):
            return next(self._stream)

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *args):
        return self._stream.__exit__(*args)
