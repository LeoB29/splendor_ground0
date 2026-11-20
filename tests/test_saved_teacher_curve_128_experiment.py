from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import time
import pytest
from tools.saved_teacher_curve_128 import common as c, experiment as exp


def test_production_receipt_budget_and_preserved_abort(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError): c.budget_boundary(tmp_path)
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda _: SimpleNamespace(free=10*1024**3))
    c.budget_boundary(tmp_path)
    monkeypatch.setattr(c.time, "time", lambda: c.old.read_json(tmp_path/"launch.json")["epoch_seconds"]+1801)
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)
    exp.abort(ValueError("first"), tmp_path)
    ref = c.artifact(tmp_path/"abort.json")
    exp.abort(ValueError("second"), tmp_path)
    c.verify_hash(ref)
    assert c.old.read_json(tmp_path/"abort.json")["error"] == "first"


def test_storage_reserve(tmp_path, monkeypatch):
    c.old.save_new(tmp_path/"launch.json", {"epoch_seconds": time.time()})
    monkeypatch.setattr(c.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    with pytest.raises(c.BudgetExceeded): c.budget_boundary(tmp_path)


def test_failed_audit_cannot_finish(tmp_path, monkeypatch):
    monkeypatch.setattr(exp, "OUTPUT", tmp_path)
    monkeypatch.setattr(exp, "validate_freeze", lambda: ("fixture", {}))
    c.old.save_new(tmp_path/"assessment.json", {"manifest": "fixture"})
    c.old.save_new(tmp_path/"audit.json", {"manifest": "fixture", "passed": False})
    with pytest.raises(ValueError):
        exp.finish("fixture", {}, c.artifact(tmp_path/"assessment.json"), c.artifact(tmp_path/"audit.json"))
    assert not (tmp_path/"final_status.json").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hidden-process integration")
def test_actual_hidden_model_loading_process(tmp_path):
    registry = c.old.read_json(c.PARENT/"models_lock.json")
    ref = registry["models"]["focused_161001"]
    c.old.save_new(tmp_path/"fixture_model.json", ref)
    launch = c.windows.detached_launch([sys.executable, "-u", "-m", "tools.saved_teacher_curve_128.worker_fixture", tmp_path],
                                       tmp_path/"stdout.txt", tmp_path/"stderr.txt", c.ROOT)
    assert Path(launch["command"][0]).name.lower() == "pythonw.exe"
    assert launch["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+40
    while not (tmp_path/"complete.json").exists():
        if (tmp_path/"failure.json").exists(): pytest.fail(c.old.read_json(tmp_path/"failure.json")["traceback"])
        if time.monotonic() > deadline: pytest.fail(f"Hidden integration timeout pid={launch['pid']}")
        time.sleep(.05)
    result = c.old.read_json(tmp_path/"complete.json")
    assert result["health"]["pid"] == launch["pid"]
    assert result["health"]["console_window"] == result["health"]["console_process_count"] == 0
    assert result["health"]["console_process_ids"] == []
    assert result["model"] == ref and result["parameters"] > 0
    assert result["eval"] and result["deterministic"] and result["threads"] == 1
    assert (tmp_path/"stderr.txt").stat().st_size == 0
