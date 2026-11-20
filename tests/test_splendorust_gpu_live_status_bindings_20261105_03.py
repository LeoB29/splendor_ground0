"""Small actual-Windows checks for cached functions and fresh status handles."""
from collections import Counter
import ctypes
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="actual Windows API bindings")
SOURCE = Path(__file__).resolve().parents[1] / "tools/splendorust_gpu_live_20261105_03/status_io.py"


def tracked_module(monkeypatch):
    real_loader = ctypes.WinDLL
    observed = SimpleNamespace(loads=[], lookups=[], prototypes=[], calls=[])

    class Function:
        def __init__(self, name, target):
            object.__setattr__(self, "name", name)
            object.__setattr__(self, "target", target)

        def __setattr__(self, key, value):
            observed.prototypes.append((self.name, key, value))
            setattr(self.target, key, value)

        def __call__(self, *args):
            result = self.target(*args)
            observed.calls.append((self.name, args, result))
            return result

    class Library:
        def __init__(self, target):
            self.target = target

        def __getattr__(self, name):
            observed.lookups.append(name)
            return Function(name, getattr(self.target, name))

    def load(*args, **kwargs):
        observed.loads.append((args, kwargs))
        return Library(real_loader(*args, **kwargs))

    monkeypatch.setattr(ctypes, "WinDLL", load)
    spec = importlib.util.spec_from_file_location("focused_status_bindings", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, observed


def test_bindings_initialized_once_and_reused_while_each_read_opens_fresh_handles(tmp_path, monkeypatch):
    status, observed = tracked_module(monkeypatch)
    assert observed.loads == [(("kernel32",), {"use_last_error": True})]
    assert observed.lookups == ["CreateMutexW", "WaitForSingleObject", "ReleaseMutex", "CloseHandle", "CreateFileW"]
    assert len(observed.prototypes) == 10
    bindings = status._WIN32_API
    assert type(bindings) is tuple and len(bindings) == 5
    prototypes = list(observed.prototypes)
    destination = tmp_path / "active_decisions.json"

    for sequence in range(5):
        destination.write_text(json.dumps({"sequence": sequence}), encoding="utf-8")
        assert status.read_status_json(destination) == {"sequence": sequence}
        assert status._WIN32_API is bindings

    assert len(observed.loads) == 1 and observed.prototypes == prototypes
    counts = Counter(call[0] for call in observed.calls)
    assert counts == {"CreateMutexW": 5, "WaitForSingleObject": 5, "ReleaseMutex": 5,
                      "CloseHandle": 5, "CreateFileW": 5}
    assert all(call[1][1] == 250 for call in observed.calls if call[0] == "WaitForSingleObject")
    # HANDLE numbers may be reused by Windows; a new API call proves each open.
    assert all(call[1][1] is False for call in observed.calls if call[0] == "CreateMutexW")


def test_shared_file_handle_is_closed_after_every_read_and_not_retained(tmp_path, monkeypatch):
    status, observed = tracked_module(monkeypatch)
    destination = tmp_path / "stage.json"
    streams = []
    for sequence in range(3):
        payload = json.dumps({"sequence": sequence}).encode()
        destination.write_bytes(payload)
        with status.open_shared_read(destination) as stream:
            streams.append(stream)
            assert not stream.closed and stream.read() == payload
        assert stream.closed
    assert len(observed.loads) == 1 and len(observed.prototypes) == 10
    assert [call[0] for call in observed.calls] == ["CreateFileW"]*3
    assert len({id(stream) for stream in streams}) == 3 and all(stream.closed for stream in streams)


def test_cached_functions_do_not_cache_missing_path_or_previous_status(tmp_path, monkeypatch):
    status, observed = tracked_module(monkeypatch)
    destination = tmp_path / "active_decisions.json"
    with pytest.raises(OSError) as error:
        status.read_status_json(destination)
    assert error.value.winerror == 2
    destination.write_text('{"sequence":7}', encoding="utf-8")
    assert status.read_status_json(destination) == {"sequence": 7}
    assert len(observed.loads) == 1
    counts = Counter(call[0] for call in observed.calls)
    assert counts == {"CreateMutexW": 2, "WaitForSingleObject": 2, "ReleaseMutex": 2,
                      "CloseHandle": 2, "CreateFileW": 2}
