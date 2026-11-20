"""Operational recovery regressions; all writes use temporary fixture outputs."""
from concurrent.futures import Future
from pathlib import Path
import copy

import pytest

from splendor_ai.alphazero import recover_teacher_budget as r

p, old = r.p, r.old


def put(path, value):
    old.save_new(path, value)
    return p.h.artifact(path)


@pytest.fixture
def alias_files(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    selected = put(tmp_path / "selected_lock.json", {"fixture": True})
    other = put(tmp_path / "other.json", {"fixture": True})
    return selected, other


def test_exact_alias_accepted_with_original_hash_checks(alias_files):
    selected, other = alias_files
    with pytest.raises(AssertionError):
        r.STRICT_VERIFY_FILES([selected, selected])
    r.verify_files_with_selected_alias([selected, selected])
    r.verify_files_with_selected_alias([selected, other])
    r.verify_files_with_selected_alias([selected])
    Path(selected["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(AssertionError):
        r.verify_files_with_selected_alias([selected, selected])


@pytest.mark.parametrize("kind", ["unrelated", "three", "conflicting", "extra_fields", "alias_plus_other"])
def test_adapter_does_not_relax_other_duplicate_rules(alias_files, kind):
    selected, other = alias_files
    cases = {"unrelated": [other, other], "three": [selected]*3,
             "conflicting": [selected, {**selected, "sha256": "0"*64}],
             "extra_fields": [{**selected, "index": 0}]*2,
             "alias_plus_other": [selected, selected, other]}
    with pytest.raises(AssertionError):
        r.verify_files_with_selected_alias(cases[kind])


def test_adapter_scope_restores_on_failure_and_refuses_stacking(alias_files):
    selected, _ = alias_files
    assert p.verify_files is r.STRICT_VERIFY_FILES
    with pytest.raises(RuntimeError, match="fixture failure"):
        with r.selected_alias_adapter():
            p.verify_files([selected, selected])
            with pytest.raises(AssertionError):
                with r.selected_alias_adapter():
                    pass
            raise RuntimeError("fixture failure")
    assert p.verify_files is r.STRICT_VERIFY_FILES
    with pytest.raises(AssertionError):
        p.verify_files([selected, selected])


class InlinePool:
    """Execute only synthetic file jobs; exercise original supervisor bookkeeping."""
    def __init__(self, **kwargs):
        assert kwargs["initializer"] is p.worker_initialize
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


@pytest.fixture
def stage_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(p, "validate_freeze", lambda: "fixture")
    monkeypatch.setattr(p, "ProcessPoolExecutor", InlinePool)
    selected = put(tmp_path / "selected_lock.json", {"fixture": True})
    def synthetic_job(kind, args):
        stage, index = args
        return {**put(tmp_path / stage / f"game_{index:04d}.json", {"fixture": True}), "index": index}
    monkeypatch.setattr(p, "dispatch", synthetic_job)
    return tmp_path, selected


def test_original_post_collection_duplicate_failure_reproduced(stage_runner):
    base, selected = stage_runner
    with pytest.raises(AssertionError):
        p.run_jobs("probe_games", "games", [("probe_games", 0)], "fixture", Path(selected["path"]), workers=1)
    assert (base / "probe_games/game_0000.json").exists()
    assert not (base / "probe_games_lock.json").exists()


def test_original_runner_all_four_remaining_stage_bookkeeping_paths(stage_runner, monkeypatch):
    base, selected = stage_runner
    with r.selected_alias_adapter():
        probe = p.run_jobs("probe_games", "games", [("probe_games", i) for i in range(48)],
                           "fixture", Path(selected["path"]), workers=1)
        assert len(p.verify_stage("probe_games", "fixture")["files"]) == 48
        p.run_jobs("probe_audit", "game_audit", [("probe_audit", i) for i in range(48)],
                   "fixture", base / "probe_games_lock.json", workers=1)
        # Real preflight verifier sees the same repeated dependency and must pass it.
        monkeypatch.setattr(p.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 100*1024**3})())
        p.preflight("fixture")
        assert old.read_json(base / "preflight.json")["passed"]
        for stage, upstream in (("main_games", "preflight.json"), ("main_audit", "main_games_lock.json")):
            p.run_jobs(stage, "fixture", [(stage, i) for i in range(2)], "fixture", base / upstream, workers=1)
            assert len(p.verify_stage(stage, "fixture")["files"]) == 2
        assert len(probe) == 48
    assert p.verify_files is r.STRICT_VERIFY_FILES


@pytest.fixture
def saved_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUTPUT", tmp_path)
    monkeypatch.setattr(r, "RECOVERY", tmp_path / "recovery_01")
    monkeypatch.setattr(r, "DIAGNOSTIC", tmp_path / "diagnostic.json")
    selected = put(tmp_path / "selected_lock.json", {"fixture": True})
    specs = p.h.schedule("probe")
    put(tmp_path / "schedules.json", {"probe": specs})
    put(tmp_path / "probe_games_launch.json", {"manifest": r.BINDING, "stage": "probe_games", "jobs": 48,
        "upstream": selected, "selected": selected, "time": "2026-09-16T23:09:34.160697+00:00"})
    files = []
    for spec in specs:
        path = p.h.game_path("probe", spec["index"], tmp_path)
        put(path, {"manifest": r.BINDING, "spec": spec, "reason": "completed", "elapsed_seconds": 1.})
        files.append(p.h.artifact(path, index=spec["index"]))
    failure = put(tmp_path / "failure.json", {"time": "2026-09-16T23:09:50.135120+00:00", "fixture": True})
    diagnostic = {"manifest": r.BINDING, "probe_full_replay_pending": True, "main_started": False,
                  "failure": failure, "probe_files": files}
    put(r.DIAGNOSTIC, diagnostic)
    return tmp_path, diagnostic


def test_reconstruction_binds_existing_games_and_excludes_downtime(saved_boundary):
    base, diagnostic = saved_boundary
    r.require_restart_boundary()
    files, upper = r.saved_probe(r.BINDING, diagnostic)
    assert upper == 15.974423 and files == diagnostic["probe_files"]
    before = copy.deepcopy(files)
    manifest = {"probe_files": files, "probe_elapsed_seconds_upper_bound": upper}
    put(r.RECOVERY / "manifest.json", manifest)
    with r.selected_alias_adapter():
        assert r.seal_saved_probe(manifest, "recovery-fixture") == files
        lock = p.verify_stage("probe_games", r.BINDING)
        assert lock["elapsed_seconds"] == 15.974423
    assert files == before
    r.STRICT_VERIFY_FILES(files)
    with pytest.raises(AssertionError, match="boundary"):
        r.require_restart_boundary()
    with r.selected_alias_adapter(), pytest.raises(FileExistsError):
        r.seal_saved_probe(manifest, "recovery-fixture")


@pytest.mark.parametrize("kind", ["game_hash", "missing_game", "duplicate_index", "changed_schedule", "extra_game", "wrong_failure"])
def test_saved_probe_rejects_missing_changed_or_replaced_evidence(saved_boundary, kind):
    base, diagnostic = saved_boundary
    if kind == "game_hash":
        Path(diagnostic["probe_files"][0]["path"]).write_text("corrupt", encoding="utf-8")
    elif kind == "missing_game":
        # Alter only this test's inventory; no deletion necessary.
        diagnostic["probe_files"] = diagnostic["probe_files"][:-1]
    elif kind == "duplicate_index":
        diagnostic["probe_files"][1] = diagnostic["probe_files"][0]
    elif kind == "changed_schedule":
        schedule = old.read_json(base / "schedules.json")
        schedule["probe"][0]["seed"] = 1
        old.atomic_write(base / "schedules.json", schedule)
    elif kind == "extra_game":
        put(base / "probe/games/game_9999.json", {})
    else:
        diagnostic["failure"]["sha256"] = "0"*64
    with pytest.raises(AssertionError):
        r.saved_probe(r.BINDING, diagnostic)


@pytest.mark.parametrize("artifact", ["probe_audit_launch.json", "main_games_launch.json", "result.json"])
def test_existing_downstream_launch_or_result_prevents_restart(saved_boundary, artifact):
    base, _ = saved_boundary
    put(base / artifact, {})
    with pytest.raises(AssertionError, match="boundary"):
        r.require_restart_boundary()


@pytest.mark.parametrize("stop", [False, True])
def test_continue_order_skips_training_and_probe_collection_and_stops_on_error(saved_boundary, monkeypatch, stop):
    base, diagnostic = saved_boundary
    calls = []
    def jobs(stage, kind, args, binding, upstream):
        assert stage == "probe_audit" and kind == "game_audit" and len(args) == 48
        assert binding == r.BINDING and upstream == base / "probe_games_lock.json"
        assert [entry[2] for entry in args] == diagnostic["probe_files"]
        calls.append("probe_audit")
        if stop:
            raise RuntimeError("fixture audit failed")
    monkeypatch.setattr(p, "run_jobs", jobs)
    monkeypatch.setattr(p, "preflight", lambda binding: calls.append("preflight"))
    monkeypatch.setattr(p, "panel", lambda stage, binding, upstream: calls.append(stage))
    monkeypatch.setattr(p, "finish", lambda binding: calls.append("finish"))
    if stop:
        with pytest.raises(RuntimeError, match="audit failed"):
            r.continue_pipeline(diagnostic["probe_files"])
        assert calls == ["probe_audit"]
    else:
        r.continue_pipeline(diagnostic["probe_files"])
        assert calls == ["probe_audit", "preflight", "main", "finish"]


def test_retained_inventory_is_fixed_to_original_artifacts(saved_boundary):
    base, _ = saved_boundary
    put(base / "recovery_01/transient.json", {})
    (base / "run.lock").write_bytes(b"0")
    put(base / "preparation/old_test.json", {})
    put(base / "preparation/stderr.txt", {"original": "failure"})
    retained = r.retained_inventory()
    paths = {Path(item["path"]).relative_to(base).as_posix() for item in retained}
    assert "preparation/stderr.txt" in paths and "failure.json" in paths
    assert not any(path.startswith("recovery_01/") for path in paths)
    assert "run.lock" not in paths and "preparation/old_test.json" not in paths
    r.STRICT_VERIFY_FILES(retained)
