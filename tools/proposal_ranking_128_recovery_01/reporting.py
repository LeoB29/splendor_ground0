"""Best-effort status only; scientific artifacts must use their strict writers.

Each event first gets a unique, exclusive-create file. progress.json is merely
an advisory convenience: failure to replace it never invalidates or interrupts
scientific work. Failed temporary/partial files are retained as evidence.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

_FALLBACK_SESSIONS = itertools.count()
REPLACE_RETRY_DELAYS = (.025, .050)


class StatusReporter:
    """Use ``event = StatusReporter(folder, metadata).event`` at the call site.

    Metadata and event values must be JSON-serializable. Programming errors in
    those values are not silently hidden. Status filesystem/stream errors are
    suppressed and exposed through ``last_errors``. Initialization performs no
    filesystem writes, and a new instance never reuses old event/temp names.
    """

    def __init__(self, folder, metadata=None):
        self.folder = Path(folder)
        self.metadata = copy.deepcopy({} if metadata is None else metadata)
        self.pid = os.getpid()
        try:
            self.session = uuid.uuid4().hex
        except OSError:
            # Even unavailable OS entropy must not stop a scientific run merely
            # to create a status identifier. The process-local counter prevents
            # same-time reinitializations from colliding in this fallback.
            self.session = f"fallback-{time.time_ns()}-{next(_FALLBACK_SESSIONS)}"
        self.sequence = 0
        self.last_errors = []
        self._lock = threading.Lock()

    def _error(self, operation, error, **detail):
        self.last_errors.append({"operation": operation, "error": type(error).__name__,
                                 "message": str(error), **detail})

    def _immutable_event(self, path, payload):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload+"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, UnicodeError) as error:
            self._error("immutable_event", error, path=str(path))

    def _advisory_progress(self, path, temporary, payload):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(payload+"\n")
                handle.flush()
        except (OSError, UnicodeError) as error:
            self._error("progress_temporary", error, path=str(temporary))
            return
        # No lock acquisition loop or unbounded retry. At most three replace
        # calls and75ms of total sleeps; other OSErrors are not retried.
        for attempt in range(len(REPLACE_RETRY_DELAYS)+1):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as error:
                if attempt == len(REPLACE_RETRY_DELAYS):
                    self._error("progress_replace", error, path=str(path),
                                temporary=str(temporary), attempts=attempt+1)
                    return
                try:
                    time.sleep(REPLACE_RETRY_DELAYS[attempt])
                except OSError as sleep_error:
                    self._error("progress_retry_wait", sleep_error, temporary=str(temporary))
                    return
            except OSError as error:
                self._error("progress_replace", error, path=str(path),
                            temporary=str(temporary), attempts=attempt+1)
                return

    @staticmethod
    def _write_stream(stream, payload):
        if stream is None:
            raise OSError("Status stream is unavailable")
        stream.write(payload+"\n")
        stream.flush()

    def _stdout(self, row):
        def payload():
            value = {**row, "status_io_errors": list(self.last_errors)} if self.last_errors else row
            return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
        try:
            self._write_stream(sys.stdout, payload())
        except (OSError, ValueError) as error:
            self._error("stdout", error)
            try:
                self._write_stream(sys.stderr, payload())
            except (OSError, ValueError) as stderr_error:
                self._error("stderr", stderr_error)

    def event(self, stage, **values):
        with self._lock:
            self.sequence += 1
            self.last_errors = []
            stamp = time.time_ns()
            identity = f"{stamp:020d}_{self.pid}_{self.session}_{self.sequence:08d}"
            row = {**self.metadata, **values, "stage": stage,
                   "time": datetime.now(timezone.utc).isoformat(),
                   "status_event_id": identity, "status_session": self.session,
                   "status_sequence": self.sequence, "status_pid": self.pid}
            payload = json.dumps(row, ensure_ascii=True, sort_keys=True, allow_nan=False)
            self._immutable_event(self.folder/"status_events"/f"{identity}.json", payload)
            self._advisory_progress(self.folder/"progress.json",
                                    self.folder/f".progress_{identity}.tmp", payload)
            self._stdout(row)
            return row
