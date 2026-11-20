"""Actual Windows pythonw workers load only arena anchors without gameplay."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools.equal_budget_breadth_arena_128 import common as c


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows pythonw spawn integration")
def test_game_and_audit_workers_hidden_with_isolated_three_anchor_registry(tmp_path):
    prior = c.old.read_json(c.SOURCE / "models_lock.json")
    registry = {"manifest": "fixture", "models": {
        f"anchor_{fit}": dict(prior["models"][f"anchor_{fit}"]) for fit in c.FITS},
        "continuation_model": prior["continuation_model"]}
    c.old.save_new(tmp_path / "models_lock.json", registry)
    seal = c.artifact(tmp_path / "models_lock.json")
    stdout, stderr = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable, "-u", "-m", "tools.equal_budget_breadth_arena_128.worker_fixture", tmp_path],
        stdout, stderr, c.ROOT)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic() + 90
    complete = tmp_path / "worker_fixture_complete.json"
    failure = tmp_path / "worker_fixture_failure.json"
    while not complete.exists():
        if failure.exists():
            pytest.fail(c.old.read_json(failure)["traceback"])
        if stderr.exists() and stderr.stat().st_size:
            pytest.fail(stderr.read_text(encoding="utf-8"))
        if time.monotonic() > deadline:
            pytest.fail(f"Hidden initialization timed out; pid={launched['pid']}; {stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert [s["result"]["kind"] for s in record["workers"]] == ["games", "audits"]
    checks, pids = [record["supervisor"]], []
    for worker in record["workers"]:
        value = worker["result"]
        assert value["models"] == registry["models"] and value["model_count"] == 3
        assert set(value["parameter_counts"]) == set(registry["models"])
        assert all(n > 0 for n in value["parameter_counts"].values())
        assert value["all_eval"] and value["deterministic"] and value["threads"] == 1
        assert value["ancestor_registries_unchanged"]
        check = value["health"]
        assert check["parent_pid"] == launched["pid"]
        assert Path(check["parent_executable"]).name.lower() == "pythonw.exe"
        assert Path(worker["runtime"]["executable"]).name.lower() == "pythonw.exe"
        pids.append(check["pid"])
        checks.append(check)
        folder = tmp_path / "workers" / f"main_{value['kind']}"
        health_files = list(folder.glob("*_health.json"))
        assert len(health_files) == 1
        saved = c.old.read_json(health_files[0])
        assert saved["manifest"] == "fixture" and saved["stage"] == "main" and saved["kind"] == value["kind"]
        assert saved["pid"] == check["pid"]
        checks.append(saved)
        assert all(p.stat().st_size == 0 for p in folder.glob("*_stderr.txt"))
        assert f"kind={value['kind']}" in (folder / f"{check['pid']}_stdout.txt").read_text(encoding="utf-8")
    assert len(set(pids)) == 2
    for check in checks:
        assert Path(check["executable"]).name.lower() == "pythonw.exe"
        assert check["console_window"] == check["console_process_count"] == 0
        assert check["console_process_ids"] == []
    assert stderr.stat().st_size == (tmp_path / "supervisor_stderr.txt").stat().st_size == 0
    c.verify_hash(seal)
    assert not failure.exists()
