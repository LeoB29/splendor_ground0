"""Temporary-file orchestration contracts; every subprocess launch is mocked."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_saved_adaptive_breadth_128 as runner
from tools.saved_adaptive_breadth_128 import common as c


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OUTPUT", tmp_path / "study")
    c.OUTPUT.mkdir()
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    return tmp_path


def launch_info():
    return {"manifest": "bound", "hidden": True, "automatic_successor": False,
            "epoch": 90.0, "deadline_epoch": 3690.0, "maximum_seconds": 3600}


@pytest.mark.parametrize("field,value", [
    ("manifest", "wrong"), ("hidden", False), ("automatic_successor", True),
    ("epoch", float("nan")), ("deadline_epoch", float("inf")),
    ("epoch", True), ("epoch", 101.0), ("maximum_seconds", 7200),
    ("deadline_epoch", 3691.0),
])
def test_changed_launch_contract_is_rejected(sandbox, field, value):
    info = launch_info()
    assert runner.validate_launch(info, "bound") is True
    info[field] = value
    with pytest.raises(ValueError):
        runner.validate_launch(info, "bound")


def test_launch_is_detached_hidden_and_has_no_successor(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "frozen", lambda **kwargs: ("bound", {}))
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(pid=1234))
    runner.launch()
    assert len(calls) == 1
    command, args = calls[0]
    assert command[1:] == [str(c.ROOT / "run_saved_adaptive_breadth_128.py"), "supervise"]
    assert args["stdin"] == runner.subprocess.DEVNULL and args["close_fds"] is True
    assert all(args["env"][key] == "1" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    if runner.os.name == "nt":
        assert Path(command[0]).name == "pythonw.exe"
        assert args["creationflags"] & runner.subprocess.CREATE_NO_WINDOW
        assert args["creationflags"] & runner.subprocess.DETACHED_PROCESS
    else:
        assert args["start_new_session"] is True
    info = c.read(c.OUTPUT / "launch.json")
    assert runner.validate_launch(info, "bound") is True
    assert info["deadline_epoch"] == 3700.0
    ack = c.read(c.OUTPUT / "launch_ack.json")
    assert ack["pid"] == 1234 and ack["launch"] == c.reference(c.OUTPUT / "launch.json")
    with pytest.raises(ValueError, match="Already launched"):
        runner.launch()
    assert len(calls) == 1


def test_supervisor_kills_only_its_worker_at_original_deadline(sandbox, monkeypatch):
    info = launch_info()
    c.save(c.OUTPUT / "launch.json", info)
    (c.OUTPUT / "runtime").mkdir()
    monkeypatch.setattr(runner, "frozen", lambda **kwargs: ("bound", {}))
    calls = []

    class Child:
        pid = 1234
        killed = False

        def wait(self, timeout):
            calls.append(("wait", timeout))
            if not self.killed:
                raise runner.subprocess.TimeoutExpired("synthetic worker", timeout)
            return -9

        def kill(self):
            self.killed = True
            calls.append(("kill", self.pid))

    child = Child()

    def spawn(command, **kwargs):
        assert command[-1] == "execute"
        assert kwargs["stdin"] == runner.subprocess.DEVNULL
        if runner.os.name == "nt":
            assert kwargs["creationflags"] & runner.subprocess.CREATE_NO_WINDOW
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    with pytest.raises(TimeoutError, match="ceiling"):
        runner.supervise()
    assert calls == [("wait", 3590.0), ("kill", 1234), ("wait", 10)]
    assert not (c.OUTPUT / "final_status.json").exists()


def test_supervisor_rejects_false_completion_even_if_child_exits_zero(sandbox, monkeypatch):
    c.save(c.OUTPUT / "launch.json", launch_info())
    (c.OUTPUT / "runtime").mkdir()
    c.save(c.OUTPUT / "final_status.json", {"manifest": "bound", "complete": True,
                                           "all_prescribed_checks_pass": False})
    monkeypatch.setattr(runner, "frozen", lambda **kwargs: ("bound", {}))
    monkeypatch.setattr(runner.subprocess, "Popen",
                        lambda *args, **kwargs: SimpleNamespace(pid=1234, wait=lambda timeout: 0))
    with pytest.raises(ValueError, match="did not close"):
        runner.supervise()


def test_preflight_timeout_leaves_failed_receipt_and_logs(sandbox, monkeypatch):
    monkeypatch.setattr(c, "code_identity", lambda: {"version": "synthetic"})

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 300
        if runner.os.name == "nt":
            assert kwargs["creationflags"] & runner.subprocess.CREATE_NO_WINDOW
        kwargs["stdout"].write(b"synthetic progress\n")
        raise runner.subprocess.TimeoutExpired(command, 300)

    monkeypatch.setattr(runner.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="Preflight failed"):
        runner.check()
    receipt = c.read(c.OUTPUT / "preflight_01/checks.json")
    assert receipt["passed"] is False and receipt["timeout"] is True and receipt["returncode"] != 0
    assert c.authenticate(receipt["stdout"]).read_text() == "synthetic progress\n"
    assert c.authenticate(receipt["stderr"]).is_file()
    assert not (c.OUTPUT / "freeze.json").exists()


def test_stale_preflight_is_rejected_before_source_collection(sandbox, monkeypatch):
    path = c.OUTPUT / "checks.json"
    c.save(path, {"passed": True, "returncode": 0, "identity": {"version": "old"}})
    monkeypatch.setattr(c, "code_identity", lambda: {"version": "new"})

    def forbidden():
        raise AssertionError("Stale preflight reached source collection")

    monkeypatch.setattr(c, "collect_source", forbidden)
    with pytest.raises(ValueError, match="stale"):
        runner.freeze(c.relative(path))
    assert not (c.OUTPUT / "manifest.json").exists()


def test_frozen_rechecks_returncode_despite_pass_boolean(sandbox, monkeypatch):
    identity = {"version": "synthetic"}
    monkeypatch.setattr(c, "code_identity", lambda: identity)
    checks = c.OUTPUT / "checks.json"
    c.save(checks, {"passed": True, "returncode": 1, "identity": identity})
    manifest = {"identity": identity, "checks": c.reference(checks)}
    c.save(c.OUTPUT / "manifest.json", manifest)
    c.save(c.OUTPUT / "freeze.json", {"owner_authorized": True, "binding": c.digest(manifest),
                                      "manifest": c.reference(c.OUTPUT / "manifest.json")})
    with pytest.raises(ValueError, match="Invalid frozen preflight"):
        runner.frozen(check_sources=False)


def receipt_envelope():
    refs = [{"path": name, "sha256": str(j)*64}
            for j, name in enumerate(("analysis.json", "source.json", "prior.json"), 1)]
    envelope = {"manifest": "bound", "analysis": refs[0], "source": refs[1], "prior_analysis": refs[2],
                "receipt": {"passed": True, "index": 0, "partitions": 64,
                    "independent_selection_and_assessment_reconstruction": True,
                    "whole_world_vectors_preserved": True, "exact_rational_comparisons": True,
                    "equal_per_arm_selection_budget": True},
                "prior_choices": {"passed": True, "prior_choices_preserved": True,
                                  "prior_scores_reused": False, "partitions": 64}}
    return envelope, refs


@pytest.mark.parametrize("group,field,value", [
    ("receipt", "passed", False), ("receipt", "passed", 1),
    ("receipt", "index", 1), ("receipt", "index", False),
    ("receipt", "partitions", 63), ("receipt", "partitions", 64.0),
    ("receipt", "independent_selection_and_assessment_reconstruction", False),
    ("receipt", "whole_world_vectors_preserved", False),
    ("receipt", "exact_rational_comparisons", False),
    ("receipt", "equal_per_arm_selection_budget", False),
    ("prior_choices", "passed", False), ("prior_choices", "prior_choices_preserved", False),
    ("prior_choices", "prior_scores_reused", True), ("prior_choices", "partitions", 63),
])
def test_semantically_invalid_receipt_cannot_close(group, field, value):
    envelope, refs = receipt_envelope()
    assert runner.validate_receipt_envelope(envelope, *refs, 0, c.CONFIG) == 64
    altered = deepcopy(envelope)
    altered[group][field] = value
    with pytest.raises(ValueError):
        runner.validate_receipt_envelope(altered, *refs, 0, c.CONFIG)


@pytest.mark.parametrize("field", ["analysis", "source", "prior_analysis"])
def test_valid_receipt_for_another_artifact_cannot_close(field):
    envelope, refs = receipt_envelope()
    envelope = deepcopy(envelope)
    envelope[field]["sha256"] = "f"*64
    with pytest.raises(ValueError, match="links differ"):
        runner.validate_receipt_envelope(envelope, *refs, 0, c.CONFIG)
