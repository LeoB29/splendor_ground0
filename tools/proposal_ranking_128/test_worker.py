"""Actual hidden workers initialize new source/engine registries without play."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools.proposal_ranking_128 import common as c


@pytest.mark.skipif(sys.platform != "win32",reason="Actual Windows pythonw spawn integration")
def test_new_source_and_engine_worker_initializers_are_hidden_and_load_sealed_models(tmp_path):
    prior = c.old.read_json(c.SOURCE/"models_lock.json")["models"]
    registry = {"manifest":"fixture","models":{
        f"{arm}_{fit}":dict(prior[f"{old_arm}_{fit}"])
        for arm,old_arm in (("anchor","anchor"),("ranking","treatment"),("ce","replay"))
        for fit in c.FITS}}
    c.old.save_new(tmp_path/"models_lock.json",registry)
    seal = c.artifact(tmp_path/"models_lock.json")
    stdout,stderr = tmp_path/"stdout.txt",tmp_path/"stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable,"-u","-m","tools.proposal_ranking_128.worker_fixture",tmp_path],stdout,stderr,c.ROOT)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+90
    complete,failure = tmp_path/"worker_fixture_complete.json",tmp_path/"worker_fixture_failure.json"
    while not complete.exists():
        if failure.exists():
            pytest.fail(c.old.read_json(failure)["traceback"])
        if time.monotonic() > deadline:
            pytest.fail(f"Hidden initialization fixture timed out; pid={launched['pid']}; {stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert [s["result"]["stage"] for s in record["stages"]] == ["source_main","main"]
    checks = [record["supervisor"]]
    pids = []
    for stage in record["stages"]:
        value = stage["result"]
        expected = registry["models"] if value["stage"] == "main" else {
            k:v for k,v in registry["models"].items() if k.startswith("anchor_")}
        assert value["models"] == expected
        assert value["model_count"] == len(expected)
        assert set(value["parameter_counts"]) == set(expected) and all(n > 0 for n in value["parameter_counts"].values())
        assert value["all_eval"] and value["deterministic"] and value["threads"] == 1
        assert value["ancestor_registry_unchanged"]
        check = value["health"]
        assert check["parent_pid"] == launched["pid"]
        assert Path(check["parent_executable"]).name.lower() == "pythonw.exe"
        assert Path(stage["runtime"]["executable"]).name.lower() == "pythonw.exe"
        pids.append(check["pid"])
        checks.append(check)
        folder = tmp_path/"workers"/f"{value['stage']}_collection"
        health_files = list(folder.glob("*_health.json"))
        assert len(health_files) == 1
        saved = c.old.read_json(health_files[0])
        assert saved["manifest"] == "fixture" and saved["stage"] == value["stage"] and saved["kind"] == "collection"
        assert saved["pid"] == check["pid"]
        checks.append(saved)
        assert all(p.stat().st_size == 0 for p in folder.glob("*_stderr.txt"))
        assert f"stage={value['stage']}" in (folder/f"{check['pid']}_stdout.txt").read_text(encoding="utf-8")
    assert len(set(pids)) == 2
    for check in checks:
        assert Path(check["executable"]).name.lower() == "pythonw.exe"
        assert check["console_window"] == check["console_process_count"] == 0
        assert check["console_process_ids"] == []
    assert stderr.stat().st_size == (tmp_path/"supervisor_stderr.txt").stat().st_size == 0
    c.verify_hash(seal)
    assert not failure.exists()
