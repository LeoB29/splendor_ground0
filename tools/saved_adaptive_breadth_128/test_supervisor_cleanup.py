"""A failed launch receipt must not leave an unsupervised arithmetic worker."""
from types import SimpleNamespace

import pytest

import run_saved_adaptive_breadth_128 as runner
from tools.saved_adaptive_breadth_128 import common as c


def test_failed_worker_receipt_stops_owned_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OUTPUT", tmp_path / "study")
    (c.OUTPUT / "runtime").mkdir(parents=True)
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    monkeypatch.setattr(runner, "frozen", lambda **kwargs: ("bound", {}))
    c.save(c.OUTPUT / "launch.json", {"manifest": "bound", "hidden": True, "automatic_successor": False,
                                     "epoch": 90.0, "deadline_epoch": 3690.0, "maximum_seconds": 3600})
    events = []

    class Child:
        pid = 1234

        def poll(self):
            return None

        def kill(self):
            events.append("killed")

        def wait(self, timeout):
            events.append(("reaped", timeout))
            return -9

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: Child())

    def denied(*args, **kwargs):
        raise PermissionError("Synthetic receipt write failure")

    monkeypatch.setattr(c, "save", denied)
    with pytest.raises(PermissionError, match="receipt write"):
        runner.supervise()
    assert events == ["killed", ("reaped", 10)]
    assert not (c.OUTPUT / "final_status.json").exists()
