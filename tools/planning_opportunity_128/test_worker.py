from pathlib import Path
import subprocess
import sys
import time
import pytest
from . import common as c


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hidden-process integration")
def test_actual_hidden_supervisor_and_model_worker(tmp_path):
    registry = c.old.read_json(c.SOURCE/"models_lock.json")["models"]
    refs = {key: value for key, value in registry.items() if key.startswith("anchor_")}
    c.old.save_new(tmp_path/"fixture_models.json", refs)
    seal = c.artifact(tmp_path/"fixture_models.json")
    launch = c.windows.detached_launch([sys.executable, "-u", "-m", "tools.planning_opportunity_128.worker_fixture", tmp_path],
                                       tmp_path/"stdout.txt", tmp_path/"stderr.txt", c.ROOT)
    assert Path(launch["command"][0]).name.lower() == "pythonw.exe"
    assert launch["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+60
    while not (tmp_path/"worker_fixture_complete.json").exists():
        if (tmp_path/"worker_fixture_failure.json").exists(): pytest.fail(c.old.read_json(tmp_path/"worker_fixture_failure.json")["traceback"])
        if time.monotonic() > deadline: pytest.fail(f"Hidden fixture timeout, pid={launch['pid']}")
        time.sleep(.05)
    record = c.old.read_json(tmp_path/"worker_fixture_complete.json")
    assert record["supervisor"]["pid"] == launch["pid"]
    worker = record["worker"]
    assert worker["models"] == refs and len(worker["parameter_counts"]) == 3
    assert all(value > 0 for value in worker["parameter_counts"].values())
    assert worker["all_eval"] and worker["deterministic"] and worker["threads"] == 1
    assert worker["health"]["parent_pid"] == launch["pid"]
    for health in (record["supervisor"], worker["health"]):
        assert health["console_window"] == health["console_process_count"] == 0
        assert health["console_process_ids"] == []
        assert Path(health["executable"]).name.lower() == "pythonw.exe"
    assert all(path.stat().st_size == 0 for path in tmp_path.rglob("*stderr.txt"))
    c.verify_hash(seal)
