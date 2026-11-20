"""Real pythonw recovery pools preserve the original logs and model behavior."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools.proposal_ranking_128 import common as c


@pytest.mark.skipif(sys.platform != "win32",reason="Actual Windows recovery spawn integration")
def test_hidden_recovery_initializers_use_fresh_logs_and_original_model_loading(tmp_path):
    prior = c.old.read_json(c.SOURCE/"models_lock.json")["models"]
    registry = {"manifest":"fixture","models":{
        f"{arm}_{fit}":dict(prior[f"{old_arm}_{fit}"])
        for arm,old_arm in (("anchor","anchor"),("ranking","treatment"),("ce","replay")) for fit in c.FITS}}
    c.old.save_new(tmp_path/"models_lock.json",registry)
    model_seal = c.artifact(tmp_path/"models_lock.json")
    c.old.save_new(tmp_path/"worker_fixture_guard.json",{"isolated_recovery_worker_fixture":True})
    preserved = {}
    for name in ("workers/source_main_audits/123_stdout.txt","workers/source_main_audits/123_stderr.txt",
                 "workers/main_collection/456_stdout.txt","workers/main_collection/456_stderr.txt",
                 "runtime/stdout.txt","runtime/stderr.txt","failure.json","progress.json"):
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(f"preserved original evidence: {name}\n",encoding="utf-8")
        preserved[name] = path.read_bytes()
    original_workers = {str(path.relative_to(tmp_path)) for path in (tmp_path/"workers").rglob("*") if path.is_file()}
    folder = tmp_path/"recovery_01"
    folder.mkdir()
    stdout,stderr = folder/"launcher_stdout.txt",folder/"launcher_stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable,"-u","-m","tools.proposal_ranking_128_recovery_01.worker_fixture",tmp_path],stdout,stderr,c.ROOT)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    complete,failure = folder/"worker_fixture_complete.json",folder/"worker_fixture_failure.json"
    deadline = time.monotonic()+90
    while not complete.exists():
        if failure.exists():
            pytest.fail(c.old.read_json(failure)["traceback"])
        if time.monotonic() > deadline:
            pytest.fail(f"Recovery worker fixture timed out; pid={launched['pid']}; {stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert [(s["stage"],s["kind"]) for s in record["stages"]] == [("source_main","audits"),("main","collection")]
    checks,pids = [record["supervisor"]],[]
    for stage in record["stages"]:
        value = stage["result"]
        expected = registry["models"] if stage["stage"] == "main" else {k:v for k,v in registry["models"].items() if k.startswith("anchor_")}
        assert value["models"] == expected and value["model_count"] == len(expected)
        assert set(value["parameter_counts"]) == set(expected) and all(n > 0 for n in value["parameter_counts"].values())
        assert value["all_eval"] and value["deterministic"] and value["threads"] == 1
        assert value["ancestor_registry_unchanged"]
        assert value["recovery_validation_calls"] == [{"check_original":False}]
        assert Path(value["recovery_folder"]) == folder.resolve()
        check = value["health"]
        assert check["parent_pid"] == launched["pid"]
        assert Path(check["parent_executable"]).name.lower() == "pythonw.exe"
        assert Path(stage["runtime"]["executable"]).name.lower() == "pythonw.exe"
        pids.append(check["pid"]);checks.append(check)
        worker_folder = folder/"workers"/f"{stage['stage']}_{stage['kind']}"
        health_files = list(worker_folder.glob("*_health.json"))
        assert len(health_files) == 1
        saved = c.old.read_json(health_files[0])
        assert saved["manifest"] == "fixture" and saved["stage"] == stage["stage"] and saved["kind"] == stage["kind"]
        assert saved["pid"] == check["pid"]
        assert Path(saved["stdout_path"]).parent == Path(saved["stderr_path"]).parent == worker_folder.resolve()
        checks.append(saved)
        assert (worker_folder/f"{check['pid']}_stderr.txt").stat().st_size == 0
        assert f"stage={stage['stage']}" in (worker_folder/f"{check['pid']}_stdout.txt").read_text(encoding="utf-8")
    assert len(set(pids)) == 2
    for check in checks:
        assert Path(check["executable"]).name.lower() == "pythonw.exe"
        assert check["console_window"] == check["console_process_count"] == 0
        assert check["console_process_ids"] == []
    assert stderr.stat().st_size == (folder/"supervisor_stderr.txt").stat().st_size == 0
    c.verify_hash(model_seal)
    assert not failure.exists()
    assert {str(path.relative_to(tmp_path)) for path in (tmp_path/"workers").rglob("*") if path.is_file()} == original_workers
    assert all((tmp_path/name).read_bytes() == content for name,content in preserved.items())
