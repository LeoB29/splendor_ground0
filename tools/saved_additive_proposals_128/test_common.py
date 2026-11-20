"""Provenance, separation, and bounded hidden-launch regressions."""
import copy
import json
import os
from pathlib import Path
import sys
import time

import pytest

from . import common as c
import run_saved_additive_proposals_128 as runner


@pytest.fixture
def original():
    return c.read(c.SOURCE / "main/roots/state_0000.json")


def test_source_contract_and_all_closed_receipts():
    inputs = c.collect_source()
    assert inputs["root_count"] == 360 and inputs["pair_count"] == 180
    assert inputs["matrix_totals"]["recorded_selection_continuations"] == 344832
    assert inputs["matrix_totals"]["recorded_assessment_continuations"] == 63872
    assert inputs["matrix_totals"]["unresolved_selection_continuations"] == 1
    assert inputs["matrix_totals"]["unresolved_assessment_continuations"] == 1


def test_metadata_validates_without_computing_new_choice(original):
    assert c.validate_record(original, original["spec"])["recorded_selection_continuations"] == 1024


@pytest.mark.parametrize("change", ["selection", "assessment", "snapshot", "choice_time", "candidate", "proof"])
def test_sealed_record_tampering_rejected(original, change):
    changed = copy.deepcopy(original)
    if change == "selection":
        changed["selection"]["worlds"][0]["outcomes"][0]["wdl"] = -1
    elif change == "assessment":
        changed["assessment"]["world_seeds"][0] = changed["selection"]["world_seeds"][0]
    elif change == "snapshot":
        changed["snapshot"]["unbound_addition"] = 1
    elif change == "choice_time":
        changed["assessment_started_at_ns"] = changed["selection_lock"]["created_at_ns"]
    elif change == "candidate":
        changed["candidates"]["anchor"]["indices"][0] = 1
    elif change == "proof":
        changed["selection_lock"]["proofs_sha256"] = c.digest([{"unsupported": True}])
    with pytest.raises(ValueError):
        c.validate_record(changed, original["spec"])


def test_pair_is_unit_not_seat_or_root():
    schedule = c.read(c.SOURCE / "schedule.json")["main"]
    c.validate_panel(schedule)
    bad = copy.deepcopy(schedule)
    bad[0]["pair"] = bad[2]["pair"]
    with pytest.raises(ValueError):
        c.validate_panel(bad)
    bad = copy.deepcopy(schedule)
    bad[0]["source_arm"] = "reference"
    with pytest.raises(ValueError):
        c.validate_panel(bad)


def test_authentication_and_exclusive_write(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    target = tmp_path / "record.json"
    c.save(target, {"state": 1})
    ref = c.reference(target)
    assert c.authenticate(ref) == target
    with pytest.raises(FileExistsError):
        c.save(target, {"state": 2})
    target.write_text('{"state":3}', encoding="utf-8")
    with pytest.raises(ValueError, match="Changed artifact"):
        c.authenticate(ref)
    with pytest.raises(ValueError, match="outside workspace"):
        c.local("../escape")


def test_budget_enforces_deadline_and_disk(monkeypatch):
    monkeypatch.setattr(runner.time, "time", lambda: 100)
    with pytest.raises(ValueError, match="ceiling"):
        runner.budget({"deadline_epoch": 99})
    class Usage:
        free = 0
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: Usage())
    with pytest.raises(ValueError, match="Disk reserve"):
        runner.budget({"deadline_epoch": 101})


@pytest.mark.skipif(os.name != "nt", reason="Windows hidden launch contract")
def test_actual_hidden_launch_fixture(tmp_path, monkeypatch):
    # Pytest may import this package under a shorter name; patch the runner's
    # module explicitly so its child can only use this synthetic workspace.
    monkeypatch.setattr(runner, "c", c)
    monkeypatch.setattr(c, "ROOT", tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.setattr(c, "OUTPUT", output)
    monkeypatch.setattr(runner, "frozen", lambda: ("fixture-binding", {}))
    script = tmp_path / "run_saved_additive_proposals_128.py"
    script.write_text(
        'import ctypes,json,os,pathlib\n'
        'p=pathlib.Path(__file__).parent/"hidden_done.json"\n'
        'p.write_text(json.dumps({"console":ctypes.windll.kernel32.GetConsoleWindow(),'
        '"threads":os.environ.get("OPENBLAS_NUM_THREADS")}))\n', encoding="utf-8")
    runner.launch()
    deadline = time.monotonic() + 20
    done = tmp_path / "hidden_done.json"
    while not done.exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert done.exists(), "Detached fixture did not finish"
    assert json.loads(done.read_text()) == {"console": 0, "threads": "1"}
    assert c.read(output / "launch_ack.json")["manifest"] == "fixture-binding"
    with pytest.raises(ValueError, match="Already launched"):
        runner.launch()


def test_production_modules_do_not_import_simulation_or_model_runtime():
    import ast
    paths = [Path(c.__file__), c.ROOT / "run_saved_additive_proposals_128.py",
             Path(c.__file__).with_name("analysis.py"), Path(c.__file__).with_name("verify.py")]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.startswith(("torch", "splendor", "tools.planner_state_transfer_128", "tools.proposal_")) for n in names), (path, names)
