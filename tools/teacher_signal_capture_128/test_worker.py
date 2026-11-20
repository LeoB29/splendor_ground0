from pathlib import Path
import subprocess
import sys
import time
import pytest
from . import common as c

@pytest.mark.skipif(sys.platform != "win32",reason="Actual Windows hidden worker")
def test_actual_hidden_supervisor_and_nine_model_worker(tmp_path):
    registry = c.old.read_json(c.LEARNING/"models_lock.json")
    c.old.save_new(tmp_path/"models_lock.json",{"manifest":"fixture","models":registry["models"]})
    seal = c.artifact(tmp_path/"models_lock.json")
    launched = c.windows.detached_launch([sys.executable,"-u","-m","tools.teacher_signal_capture_128.worker_fixture",tmp_path],
                                        tmp_path/"stdout.txt",tmp_path/"stderr.txt",c.ROOT)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+60
    while not (tmp_path/"worker_fixture_complete.json").exists():
        if (tmp_path/"worker_fixture_failure.json").exists(): pytest.fail(c.old.read_json(tmp_path/"worker_fixture_failure.json")["traceback"])
        if time.monotonic() > deadline: pytest.fail(f"Hidden worker timed out; pid={launched['pid']}")
        time.sleep(.05)
    value = c.old.read_json(tmp_path/"worker_fixture_complete.json")
    assert value["supervisor"]["pid"] == launched["pid"]
    worker = value["worker"]
    assert worker["models"] == registry["models"] and len(worker["parameter_counts"]) == 9
    assert all(n > 0 for n in worker["parameter_counts"].values())
    assert worker["all_eval"] and worker["deterministic"] and worker["threads"] == 1
    assert worker["health"]["parent_pid"] == launched["pid"]
    for health in (value["supervisor"],worker["health"]):
        assert health["console_window"] == health["console_process_count"] == 0
        assert health["console_process_ids"] == []
        assert Path(health["executable"]).name.lower() == "pythonw.exe"
    assert all(p.stat().st_size == 0 for p in tmp_path.rglob("*stderr.txt"))
    c.verify_hash(seal)
