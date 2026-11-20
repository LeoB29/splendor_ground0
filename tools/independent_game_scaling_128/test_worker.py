"""Actual Windows hidden children retain the new168h guard in all stage paths."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from . import common as c


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows pythonw spawn integration")
def test_96h_hidden_workers_cover_diagnostics_sources_labels_evaluation_and_audit(tmp_path):
    stdout, stderr = tmp_path/"stdout.txt", tmp_path/"stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable,"-u","-m","tools.independent_game_scaling_128.worker_fixture",tmp_path],
        stdout,stderr,c.ROOT)
    c.old.save_new(tmp_path/"hidden_launch.json", launched)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic()+420
    complete,failure = tmp_path/"worker_fixture_complete.json",tmp_path/"worker_fixture_failure.json"
    while not complete.exists():
        if failure.exists(): pytest.fail(c.old.read_json(failure)["traceback"])
        if time.monotonic() > deadline:
            pytest.fail(f"Hidden96h fixture timed out; pid={launched['pid']}; {stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert record["original_receipts_unchanged"] and record["prior_active_seconds"] == 96*3600
    assert [v["result"]["task"] for v in record["stages"]] == ["normal","audit","source_train","source_eval","labels","evaluation"]
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
    assert len(set(pids)) == 6
    for health in checks:
        assert Path(health["executable"]).name.lower() == "pythonw.exe"
        assert health["console_window"] == health["console_process_count"] == 0
        assert health["console_process_ids"] == []
    normal,audit,train,source,labels,evaluation = [item["result"]["result"] for item in record["stages"]]
    assert normal["rows"] == audit["rows"] == 5
    assert normal["parameters_unchanged"] and audit["parameters_unchanged"] and audit["independent_matches_normal"]
    assert train["training_registry_only"] and not source["training_registry_only"]
    for value in (train,source):
        assert value["moves"] > 0 and value["all_move_audit"]
        for name in ("source","source_audit"):c.verify_hash(value[name])
    assert labels["full_inner_replay"] and labels["replayed_continuations"] == 128
    for name in ("label","label_audit"):c.verify_hash(labels[name])
    assert evaluation["recorded_continuations"] > 0 and evaluation["full_inner_replay"] and evaluation["identical_statistics_row"]
    for name in ("evaluation","evaluation_audit"):c.verify_hash(evaluation[name])
    controls = evaluation["guard_controls"]
    assert controls["ceiling_checked"] and controls["abort_checked"]
    assert len(controls["exceptions"]) == 8
    assert {value["mode"] for value in controls["exceptions"]} == {"168h","abort"}
    assert {value["exception"] for value in controls["exceptions"]} == {"tools.independent_game_scaling_128.common.BudgetExceeded"}
    assert c.verified(controls["abort"])["error"] == "Isolated96h hidden-worker cooperative hold"
    for seal in record["original_seals"]:c.verify_hash(seal)
    assert stderr.stat().st_size == (tmp_path/"supervisor_stderr.txt").stat().st_size == 0
    assert not failure.exists()
