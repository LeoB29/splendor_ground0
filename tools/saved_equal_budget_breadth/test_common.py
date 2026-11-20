import json
from pathlib import Path

import pytest

from tools.saved_equal_budget_breadth import common as c


def test_source_authentication_rejects_mutation_and_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    path = tmp_path / "record.json"
    c.save(path, {"scientific": [1, 2, 3]})
    ref = c.reference(path)
    inventory = {}
    assert c.authenticate(ref, inventory) == path
    c.check_inventory(inventory)
    path.write_text('{"scientific":[1,2,4]}')
    with pytest.raises(ValueError, match="Changed artifact"):
        c.authenticate(ref)
    with pytest.raises(FileNotFoundError):
        c.authenticate({"path": "missing", "sha256": "0" * 64})


def test_input_path_cannot_escape_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside workspace"):
        c.local("../outside.json")


def test_immutable_scientific_artifact_not_overwritten(tmp_path):
    path = tmp_path / "result.json"
    c.save(path, {"value": 1})
    with pytest.raises(FileExistsError):
        c.save(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_advisory_status_contention_does_not_fail_science(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    def denied(*args):
        raise PermissionError("Windows reader holds advisory file")
    monkeypatch.setattr(c.os, "replace", denied)
    assert c.status({"stage": "complete"}) is False


def test_status_is_replaceable_and_scientific_config_is_fixed(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    assert c.status({"stage": "analysis"}) is True
    assert c.status({"stage": "complete"}) is True
    assert c.read(tmp_path / "status.json")["stage"] == "complete"
    assert c.CONFIG["partitions"] == 64
    assert c.CONFIG["bootstrap_resamples"] == 50000
    assert c.CONFIG["execution_cap_seconds"] == 3600
    assert c.CONFIG["automatic_successor"] is False


def test_deadline_and_disk_stop_execution(tmp_path, monkeypatch):
    import run_saved_equal_budget_breadth as runner
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="ceiling"):
        runner.budget({"deadline_epoch": 0})
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda p: type("Usage", (), {"free": 0})())
    with pytest.raises(ValueError, match="Disk reserve"):
        runner.budget({"deadline_epoch": 1e20})
