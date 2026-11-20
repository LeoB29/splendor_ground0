"""Actual Windows hidden children retain the new96h guard in all stage paths."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from . import common as c


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows pythonw spawn integration")
@pytest.mark.parametrize("prior_hours", [24, 48])
def test_aged_hidden_workers_cover_diagnostics_source_evaluation_and_audit(tmp_path, prior_hours):
    stdout, stderr = tmp_path/"stdout.txt", tmp_path/"stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable,"-u","-m","tools.structured_proposer_128.worker_fixture",tmp_path,str(prior_hours)],
        stdout,stderr,c.ROOT)
    c.old.save_new(tmp_path/"hidden_launch.json", launched)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+420
    complete,failure = tmp_path/"worker_fixture_complete.json",tmp_path/"worker_fixture_failure.json"
    while not complete.exists():
        if failure.exists(): pytest.fail(c.old.read_json(failure)["traceback"])
        if time.monotonic() > deadline:
            pytest.fail(f"Hidden aged fixture timed out; pid={launched['pid']}; {stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert record["original_receipts_unchanged"] and record["prior_active_seconds"] == prior_hours*3600
    assert [v["result"]["task"] for v in record["stages"]] == ["normal","audit","source","evaluation"]
    checks,pids = [record["supervisor"]],[]
    for stage in record["stages"]:
        value = stage["result"]
        assert value["active_seconds"] >= record["prior_active_seconds"]
        assert value["threads"] == 1 and value["deterministic"]
        health = value["health"]
        checks.append(health);pids.append(health["pid"])
        assert health["parent_pid"] == launched["pid"]
        assert Path(health["parent_executable"]).name.lower() == "pythonw.exe"
        assert Path(stage["runtime"]["executable"]).name.lower() == "pythonw.exe"
        folder = tmp_path/"study/workers"/f"{stage['stage']}_{stage['kind']}"
        saved = c.old.read_json(folder/f"{health['pid']}_health.json")
        assert saved["manifest"] == "fixture" and saved["stage"] == stage["stage"] and saved["kind"] == stage["kind"]
        checks.append(saved)
        assert all(path.stat().st_size == 0 for path in folder.glob("*_stderr.txt"))
        assert f"task={value['task']}" in (folder/f"{health['pid']}_stdout.txt").read_text(encoding="utf-8")
    assert len(set(pids)) == 4
    for health in checks:
        assert Path(health["executable"]).name.lower() == "pythonw.exe"
        assert health["console_window"] == health["console_process_count"] == 0
        assert health["console_process_ids"] == []
    normal,audit,source,evaluation = [item["result"]["result"] for item in record["stages"]]
    assert normal["rows"] == audit["rows"] == 8
    assert normal["parameters_unchanged"] and audit["parameters_unchanged"] and audit["independent_matches_normal"]
    assert source["moves"] > 0 and source["all_move_audit"]
    for name in ("source","source_audit"):c.verify_hash(source[name])
    assert evaluation["recorded_continuations"] > 0 and evaluation["full_inner_replay"] and evaluation["identical_statistics_row"]
    for name in ("evaluation","evaluation_audit"):c.verify_hash(evaluation[name])
    controls = evaluation["guard_controls"]
    assert controls["ceiling_checked"] and controls["storage_checked"] and controls["abort_checked"]
    assert len(controls["exceptions"]) == 12
    assert {value["mode"] for value in controls["exceptions"]} == {"96h","storage","abort"}
    assert {value["exception"] for value in controls["exceptions"]} == {"tools.structured_proposer_128.common.BudgetExceeded"}
    assert c.verified(controls["abort"])["error"] == "Isolated aged hidden-worker cooperative hold"
    for seal in record["original_seals"]:c.verify_hash(seal)
    assert stderr.stat().st_size == (tmp_path/"supervisor_stderr.txt").stat().st_size == 0
    assert not failure.exists()
