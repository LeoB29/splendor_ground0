"""Actual game/audit pythonw workers load original anchors without simulation."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools.equal_budget_breadth_arena_128_recovery_02 import common as c

FIXTURE_BINDING = "isolated-recovery-two-hidden-worker-fixture"


@pytest.mark.skipif(sys.platform != "win32",reason="Actual Windows pythonw spawn integration")
def test_recovery_game_and_audit_workers_hidden_with_original_anchor_registry(tmp_path):
    assert not tmp_path.resolve().is_relative_to(c.SOURCE.resolve())
    prior = c.old.read_json(c.SOURCE/"models_lock.json")
    registry = {"manifest":"fixture","models":{
        f"anchor_{fit}":dict(prior["models"][f"anchor_{fit}"]) for fit in c.FITS},
        "continuation_model":prior["continuation_model"]}
    c.old.save_new(tmp_path/"models_lock.json",registry)
    c.old.save_new(tmp_path/"inventory.json",{"original_binding":c.ORIGINAL_BINDING,
        "previous_binding":c.PREVIOUS_BINDING,"sources":{},"origins":{},
        "models":c.artifact(tmp_path/"models_lock.json"),"games":{},"roots":{},"original_stage_launches":{}})
    c.old.save_new(tmp_path/"freeze.json",{"binding":FIXTURE_BINDING})
    seals = [c.artifact(tmp_path/name) for name in ("models_lock.json","inventory.json","freeze.json")]
    stdout,stderr = tmp_path/"stdout.txt",tmp_path/"stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable,"-u","-m","tools.equal_budget_breadth_arena_128_recovery_02.worker_fixture",tmp_path],
        stdout,stderr,c.ROOT)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+90
    complete,failure = tmp_path/"worker_fixture_complete.json",tmp_path/"worker_fixture_failure.json"
    while not complete.exists():
        if failure.exists():pytest.fail(c.old.read_json(failure)["traceback"])
        if stderr.exists() and stderr.stat().st_size:pytest.fail(stderr.read_text(encoding="utf-8"))
        if time.monotonic() > deadline:
            pytest.fail(f"Hidden recovery initialization timed out; pid={launched['pid']}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert [worker["result"]["kind"] for worker in record["workers"]] == ["games","audits"]
    checks,pids = [record["supervisor"]],[]
    for worker in record["workers"]:
        result = worker["result"]
        assert result["models"] == registry["models"] and result["model_count"] == 3
        assert set(result["parameter_counts"]) == set(registry["models"])
        assert all(n > 0 for n in result["parameter_counts"].values())
        assert result["all_eval"] and result["all_cpu"] and result["deterministic"] and result["threads"] == 1
        assert result["ancestor_registries_unchanged"] and result["frozen_arena_registry_shared"]
        assert result["recovery_binding"] == FIXTURE_BINDING
        health = result["health"]
        assert health["parent_pid"] == launched["pid"]
        assert Path(health["parent_executable"]).name.lower() == "pythonw.exe"
        assert Path(worker["runtime"]["executable"]).name.lower() == "pythonw.exe"
        pids.append(health["pid"])
        checks.append(health)
        folder = tmp_path/"workers"/result["kind"]
        health_files = list(folder.glob("*_health.json"))
        assert len(health_files) == 1
        saved = c.old.read_json(health_files[0])
        assert saved["manifest"] == c.ORIGINAL_BINDING and saved["recovery_binding"] == FIXTURE_BINDING
        assert saved["kind"] == result["kind"] and saved["pid"] == health["pid"]
        checks.append(saved)
        assert all(path.stat().st_size == 0 for path in folder.glob("*_stderr.txt"))
        assert f"kind={result['kind']}" in (folder/f"{health['pid']}_stdout.txt").read_text(encoding="utf-8")
    assert len(set(pids)) == 2
    for health in checks:
        assert Path(health["executable"]).name.lower() == "pythonw.exe"
        assert health["console_window"] == health["console_process_count"] == 0
        assert health["console_process_ids"] == []
    assert stderr.stat().st_size == (tmp_path/"supervisor_stderr.txt").stat().st_size == 0
    for seal in seals:c.verify_hash(seal)
    assert not failure.exists()
    assert not (tmp_path/"main").exists() and not (tmp_path/"test").exists()
