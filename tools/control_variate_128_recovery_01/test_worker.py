"""Hidden spawn installs the recovery's own active-time boundary."""
from pathlib import Path
import subprocess
import sys
import time
import pytest
from tools.control_variate_128 import common as c
from . import recovery as r


@pytest.mark.skipif(sys.platform != "win32",reason="Windows hidden-process integration")
def test_hidden_recovery_worker_uses_three_hour_boundary_and_original_model(tmp_path):
    before=c.old.digest(c.CONFIG)
    launch=c.windows.detached_launch([sys.executable,"-u","-m",
        "tools.control_variate_128_recovery_01.worker_fixture",tmp_path],
        tmp_path/"stdout.txt",tmp_path/"stderr.txt",c.ROOT)
    assert Path(launch["command"][0]).name.lower() == "pythonw.exe"
    assert launch["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline=time.monotonic()+60
    while not (tmp_path/"worker_fixture_complete.json").exists():
        if (tmp_path/"worker_fixture_failure.json").exists():
            pytest.fail(c.old.read_json(tmp_path/"worker_fixture_failure.json")["traceback"])
        if time.monotonic() > deadline:pytest.fail(f"Hidden recovery fixture timeout, pid={launch['pid']}")
        time.sleep(.05)
    record=c.old.read_json(tmp_path/"worker_fixture_complete.json");worker=record["worker"]
    assert record["supervisor"]["pid"] == launch["pid"]
    assert worker["raw_initialized"] and worker["threads"] == 1 and worker["deterministic"]
    assert worker["budget_module"] == "tools.control_variate_128_recovery_01.recovery"
    assert worker["budget_name"] == "budget_boundary"
    assert worker["maximum_runtime_seconds"] == 10800
    assert worker["passes_after_two_hours"] and worker["stops_after_three_active_hours"]
    assert worker["scientific_config"] == c.CONFIG and c.old.digest(c.CONFIG) == before
    assert worker["health"]["parent_pid"] == launch["pid"]
    for health in (record["supervisor"],worker["health"]):
        assert health["console_window"] == health["console_process_count"] == 0
        assert health["console_process_ids"] == []
        assert Path(health["executable"]).name.lower() == "pythonw.exe"
    assert all(p.stat().st_size == 0 for p in tmp_path.rglob("*stderr.txt"))
