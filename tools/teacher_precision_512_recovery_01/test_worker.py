"""Real hidden workers must survive inherited guard ages in downstream stages."""
from pathlib import Path
import subprocess
import sys
import time

import pytest

from tools.teacher_precision_512 import common as c


@pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows pythonw spawn integration")
def test_aged_recovery_hidden_workers_cover_diagnostics_source_evaluation_and_audit(tmp_path):
    stdout, stderr = tmp_path/"stdout.txt", tmp_path/"stderr.txt"
    launched = c.windows.detached_launch(
        [sys.executable, "-u", "-m", "tools.teacher_precision_512_recovery_01.worker_fixture", tmp_path],
        stdout, stderr, c.ROOT)
    assert Path(launched["command"][0]).name.lower() == "pythonw.exe"
    assert launched["creationflags"] & subprocess.CREATE_NO_WINDOW
    deadline = time.monotonic() + 300
    complete, failure = tmp_path/"worker_fixture_complete.json", tmp_path/"worker_fixture_failure.json"
    while not complete.exists():
        if failure.exists():
            pytest.fail(c.old.read_json(failure)["traceback"])
        if time.monotonic() > deadline:
            pytest.fail(f"Hidden aged fixture timed out; pid={launched['pid']}; {stderr.read_text(encoding='utf-8')}")
        time.sleep(.05)
    record = c.old.read_json(complete)
    assert record["supervisor"]["pid"] == launched["pid"]
    assert record["original_runtime_untouched"]
    assert record["prior_active_seconds"] > 13*3600
    assert [item["result"]["task"] for item in record["stages"]] == ["normal", "audit", "source", "evaluation"]
    checks, pids = [record["supervisor"]], []
    for stage in record["stages"]:
        value = stage["result"]
        assert value["active_seconds"] >= record["prior_active_seconds"]
        assert value["threads"] == 1 and value["deterministic"]
        assert value["report_module"] == "tools.teacher_precision_512_recovery_01.diagnostics"
        health = value["health"]
        checks.append(health); pids.append(health["pid"])
        assert health["parent_pid"] == launched["pid"]
        assert Path(health["parent_executable"]).name.lower() == "pythonw.exe"
        assert Path(stage["runtime"]["executable"]).name.lower() == "pythonw.exe"
        folder = tmp_path/"recovery/workers"/f"{stage['stage']}_{stage['kind']}"
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
    normal, audit, source, evaluation = [item["result"]["result"] for item in record["stages"]]
    assert normal["old_rows"] == normal["precision_rows"] == audit["old_rows"] == audit["precision_rows"] == 5
    assert normal["precision_worlds"] == audit["precision_worlds"] == 512
    assert normal["parameters_unchanged"] and audit["parameters_unchanged"] and audit["independent_matches_normal"]
    assert source["moves"] > 0 and source["all_move_audit"]
    assert evaluation["recorded_continuations"] > 0
    assert evaluation["full_inner_replay"] and evaluation["identical_statistics_row"]
    controls = evaluation["guard_controls"]
    assert controls["ceiling_checked"] and controls["new_abort_checked"] and controls["old_abort_unchanged"]
    assert controls["exception_classes"] == ["tools.teacher_precision_512.common.BudgetExceeded"]*4
    assert c.verified(controls["new_abort"])["error"] == "Isolated hidden fixture cooperative hold"
    for name in ("source", "source_audit"):
        c.verify_hash(source[name])
    for name in ("evaluation", "evaluation_audit"):
        c.verify_hash(evaluation[name])
    for seal in record["original_seals"]:
        c.verify_hash(seal)
    assert not (tmp_path/"study/workers").exists()
    assert (tmp_path/"recovery/abort.json").exists()
    assert stderr.stat().st_size == (tmp_path/"supervisor_stderr.txt").stat().st_size == 0
    assert not failure.exists()
