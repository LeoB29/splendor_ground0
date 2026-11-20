"""Operational recovery fixtures; no model loading, training, or source replay."""
import copy
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.proposal_ranking_128_recovery_01 import recovery as h, reporting


ORIGINAL_OUTPUT = h.OUTPUT


def saved_pair(folder, index=0, *, write_audit=True):
    """Copy one already audited record; only its artifact path changes."""
    game = h.old.read_json(h.p.source.game_path("main", index, ORIGINAL_OUTPUT))
    receipt = h.old.read_json(h.p.source.audit_path("main", index, ORIGINAL_OUTPUT))
    game_path = h.p.source.game_path("main", index, folder)
    h.old.save_new(game_path, game)
    game_item = h.c.artifact(game_path, index=index, stage="main")
    receipt["game"] = {key: game_item[key] for key in ("path", "sha256")}
    if write_audit:
        audit = write_receipt(folder, index, receipt)
    else:
        audit = None
    return game["spec"], game_item, receipt, audit


def write_receipt(folder, index, receipt):
    path = h.p.source.audit_path("main", index, folder)
    h.old.save_new(path, receipt)
    return h.c.artifact(path, index=index, stage="main")


def test_saved_receipt_accepts_real_schema_without_replaying(tmp_path):
    spec, game, _, audit = saved_pair(tmp_path)
    assert h.reconcile_receipt(spec, game, audit, h.ORIGINAL_BINDING) == audit


@pytest.mark.parametrize("field", ["passed", "invariance", "spec", "manifest", "trace", "game", "coverage", "reason"])
def test_rehashed_receipt_corruption_is_rejected(tmp_path, field):
    spec, game, receipt, _ = saved_pair(tmp_path / "original")
    broken = copy.deepcopy(receipt)
    if field == "passed":
        broken["passed"] = False
    elif field == "invariance":
        broken["every_move_hidden_input_invariance"] = False
    elif field == "spec":
        broken["spec"]["index"] += 1
    elif field == "manifest":
        broken["manifest"] = "another-binding"
    elif field == "trace":
        broken["trace_sha256"] = "bad"
    elif field == "game":
        broken["game"]["sha256"] = "bad"
    elif field == "coverage":
        broken["coverage"]["moves"] -= 1
    else:
        broken["reason"] = "unresolved"
    audit = write_receipt(tmp_path / "mutated", 0, broken)
    with pytest.raises(ValueError):
        h.reconcile_receipt(spec, game, audit, h.ORIGINAL_BINDING)


def test_saved_artifact_identity_and_rehashed_model_linkage_fail_closed(tmp_path):
    spec, game_item, receipt, audit = saved_pair(tmp_path / "original")
    for changes in ({"index": 1}, {"stage": "probe"}, {"sha256": "bad"}):
        with pytest.raises(ValueError):
            h.reconcile_receipt(spec, game_item, {**audit, **changes}, h.ORIGINAL_BINDING)
    game = h.c.verified(game_item)
    game["trace"][0]["model_sha256"] = "wrong-turn-model"
    game["trace_sha256"] = h.old.digest(game["trace"])
    path = tmp_path / "changed-game.json"
    h.old.save_new(path, game)
    changed = h.c.artifact(path, index=0, stage="main")
    receipt["trace_sha256"] = game["trace_sha256"]
    receipt["game"] = {key: changed[key] for key in ("path", "sha256")}
    new_audit = write_receipt(tmp_path / "changed-audit", 0, receipt)
    with pytest.raises(ValueError, match="model"):
        h.reconcile_receipt(spec, changed, new_audit, h.ORIGINAL_BINDING)


def pool_fixture(tmp_path, monkeypatch, *, failing=False, wrong_index=False):
    base = tmp_path / "original-output"
    recovery = base / "recovery_01"
    monkeypatch.setattr(h, "OUTPUT", base)
    monkeypatch.setattr(h, "RECOVERY", recovery)
    monkeypatch.setattr(h, "_reporter", None)
    monkeypatch.setattr(h, "validate_recovery", lambda **kw: "recovery-binding")
    monkeypatch.setattr(h.c.windows, "configure_spawn", lambda: {"fixture": "no process"})
    monkeypatch.setattr(h.c.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))
    now = h.time.time()
    h.old.save_new(base / "launch.json", {"epoch_seconds": now-60, "manifest": h.ORIGINAL_BINDING})
    for path, value in ((base / "freeze.json", {"frozen": True}),
                        (base / "failure.json", {"traceback": "preserved reporting failure"}),
                        (base / "progress.json", {"stage": "original failed progress"}),
                        (base / "source_main_audits_launch.json", {"original_failed_stage": True}),
                        (recovery / "freeze.json", {"recovery": True}),
                        (recovery / "launch.json", {"epoch_seconds": now})):
        h.old.save_new(path, value)
    records = [saved_pair(base, index, write_audit=index != 1) for index in range(3)]
    arguments = [(spec, game) for spec, game, _, _ in records]
    retained = [records[index][3] for index in (0, 2)]
    sealed = {f"{stage}_{kind}": [game for _, game in arguments]
              for stage, kind in h.SEALED_SOURCE_STAGES}
    h.old.save_new(recovery / "inventory.json", {"retained_main_source_audits": retained,
                 "missing_main_source_audit_indices": [1], "sealed_source_stages": sealed})
    checked = []

    def stage_lock(stage, kind, binding):
        checked.append((stage, kind, binding))
        key = f"{stage}_{kind}"
        if (stage, kind) in h.SEALED_SOURCE_STAGES:
            return {"files": sealed[key]}
        lock = h.old.read_json(base / f"{key}_lock.json")
        assert lock["manifest"] == binding and [item["index"] for item in lock["files"]] == [0, 1, 2]
        launch = h.c.verified(lock["launch"])
        assert launch["manifest"] == binding and launch["recovery_binding"] == "recovery-binding"
        assert launch["pending_indices"] == [1] and launch["retained"] == retained
        for item in lock["files"] + launch["dependencies"]:
            h.c.verify_hash(item)
        return lock

    monkeypatch.setattr(h.p, "stage_lock", stage_lock)
    submissions = []

    class ImmediatePool:
        def __init__(self, **kwargs):
            assert kwargs["initializer"] is h.worker_initialize
            assert kwargs["initargs"] == ("source_main", "audits", h.ORIGINAL_BINDING, base)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, stage, kind, argument, binding, output):
            assert function is h.p.run_job and output == base and binding == h.ORIGINAL_BINDING
            index = argument[0]["index"]
            submissions.append(index)
            result = Future()
            if failing:
                result.set_exception(RuntimeError("scientific worker discrepancy"))
            else:
                item = write_receipt(base, index, records[index][2])
                result.set_result({**item, "index": 0} if wrong_index else item)
            return result

    monkeypatch.setattr(h, "ProcessPoolExecutor", ImmediatePool)
    preserved = [base / name for name in ("failure.json", "progress.json", "source_main_audits_launch.json")]
    preserved += [h.c.verify_hash(item) for item in retained]
    hashes = {path: h.old.file_hash(path) for path in preserved}
    return SimpleNamespace(base=base, recovery=recovery, arguments=arguments, retained=retained,
                           sealed=sealed, submissions=submissions, hashes=hashes, checked=checked)


def test_recovery_skips_retained_indices_even_when_status_progress_is_locked(tmp_path, monkeypatch):
    fixture = pool_fixture(tmp_path, monkeypatch)
    original_replace = reporting.os.replace

    def locked(source, target):
        if Path(target) == fixture.recovery / "progress.json":
            raise PermissionError(13, "persistent status reader lock")
        return original_replace(source, target)

    monkeypatch.setattr(reporting.os, "replace", locked)
    monkeypatch.setattr(reporting.time, "sleep", lambda seconds: None)
    files = h.run_pool("source_main", "audits", fixture.arguments, h.ORIGINAL_BINDING,
                       [fixture.base / "freeze.json"])
    assert fixture.submissions == [1] and [item["index"] for item in files] == [0, 1, 2]
    assert [files[0], files[2]] == fixture.retained
    assert all(h.old.file_hash(path) == digest for path, digest in fixture.hashes.items())
    lock = h.old.read_json(fixture.base / "source_main_audits_lock.json")
    assert lock["reused_count"] == 2 and h.c.verify_hash(lock["launch"]).parent == fixture.recovery / "stages"
    assert len(list((fixture.recovery / "status_events").glob("*.json"))) == 2
    assert h._reporter.last_errors[0]["operation"] == "progress_replace"
    assert not (fixture.recovery / "failure.json").exists()
    with pytest.raises(ValueError, match="One-shot"):
        h.run_pool("source_main", "audits", fixture.arguments, h.ORIGINAL_BINDING, [])


@pytest.mark.parametrize("stage,kind", h.SEALED_SOURCE_STAGES)
def test_completed_source_pools_return_existing_seals_without_submitting(tmp_path, monkeypatch, stage, kind):
    fixture = pool_fixture(tmp_path, monkeypatch)
    files = h.run_pool(stage, kind, fixture.arguments, h.ORIGINAL_BINDING, [])
    assert files == fixture.sealed[f"{stage}_{kind}"] and fixture.submissions == []
    assert not (fixture.recovery / "stages").exists()


@pytest.mark.parametrize("wrong_index", [False, True])
def test_worker_failure_or_wrong_identity_cannot_become_success(tmp_path, monkeypatch, wrong_index):
    fixture = pool_fixture(tmp_path, monkeypatch, failing=not wrong_index, wrong_index=wrong_index)
    with pytest.raises((RuntimeError, ValueError), match="scientific worker discrepancy|another scheduled index"):
        h.run_pool("source_main", "audits", fixture.arguments, h.ORIGINAL_BINDING, [fixture.base / "freeze.json"])
    assert fixture.submissions == [1]
    assert not (fixture.base / "source_main_audits_lock.json").exists()
    assert not (fixture.base / "final_status.json").exists()
    assert all(h.old.file_hash(path) == digest for path, digest in fixture.hashes.items())


def test_new_recovery_launch_does_not_reset_original_deadline(tmp_path, monkeypatch):
    fixture = pool_fixture(tmp_path, monkeypatch)
    original = h.old.read_json(fixture.base / "launch.json")["epoch_seconds"]
    recovery = h.old.read_json(fixture.recovery / "launch.json")["epoch_seconds"]
    assert recovery > original
    monkeypatch.setattr(h.c.time, "time", lambda: original + 8*3600 + 1)
    with pytest.raises(h.c.BudgetExceeded, match="Eight-hour"):
        h.run_pool("source_main", "audits", fixture.arguments, h.ORIGINAL_BINDING, [])
    assert fixture.submissions == [] and not (fixture.recovery / "stages").exists()
    assert all(h.old.file_hash(path) == digest for path, digest in fixture.hashes.items())


def test_continuation_calls_unchanged_science_and_restores_operational_hooks(monkeypatch):
    calls = []
    old_event, old_pool = h.p.event, h.p.run_pool
    monkeypatch.setattr(h.c, "budget_boundary", lambda output: None)
    monkeypatch.setattr(h, "validate_recovery", lambda: calls.append("freeze"))
    monkeypatch.setattr(h.p, "build_learning", lambda binding: pytest.fail("Recovery must not repeat learning"))
    monkeypatch.setattr(h.p, "build_sources", lambda binding: calls.append(("sources", h.p.event is h.event, h.p.run_pool is h.run_pool)))
    monkeypatch.setattr(h.p, "build_stage", lambda stage, binding: calls.append(stage))
    monkeypatch.setattr(h.p, "operational_decision", lambda binding: calls.append("gate") or True)
    monkeypatch.setattr(h.p, "finish", lambda binding: calls.append("finish"))
    assert h.continue_pipeline(h.ORIGINAL_BINDING)
    assert calls == [("sources", True, True), "probe", "gate", "main", "freeze", "finish"]
    assert h.p.event is old_event and h.p.run_pool is old_pool


def test_continuation_honors_cost_hold_and_restores_hooks_after_failure(monkeypatch):
    calls = []
    old_event, old_pool = h.p.event, h.p.run_pool
    monkeypatch.setattr(h.c, "budget_boundary", lambda output: None)
    monkeypatch.setattr(h.p, "build_learning", lambda binding: pytest.fail("Recovery must not repeat learning"))
    monkeypatch.setattr(h.p, "build_sources", lambda binding: calls.append("sources"))
    monkeypatch.setattr(h.p, "build_stage", lambda stage, binding: calls.append(stage))
    monkeypatch.setattr(h.p, "operational_decision", lambda binding: calls.append("gate") or False)
    assert not h.continue_pipeline(h.ORIGINAL_BINDING)
    assert calls == ["sources", "probe", "gate"] and h.p.event is old_event and h.p.run_pool is old_pool
    def broken(binding):
        raise RuntimeError("strict source failure")
    monkeypatch.setattr(h.p, "build_sources", broken)
    with pytest.raises(RuntimeError, match="strict source failure"):
        h.continue_pipeline(h.ORIGINAL_BINDING)
    assert h.p.event is old_event and h.p.run_pool is old_pool


@pytest.mark.parametrize("name", ["selection.json", "source_main_audits_lock.json", "operational_decision.json",
                                  "final_status.json", "probe", "main"])
def test_inventory_rejects_unexpected_downstream_evidence(tmp_path, monkeypatch, name):
    monkeypatch.setattr(h, "OUTPUT", tmp_path)
    monkeypatch.setattr(h.p, "validate_freeze", lambda: h.ORIGINAL_BINDING)
    monkeypatch.setattr(h.c, "budget_boundary", lambda output: None)
    path = tmp_path / name
    if name.endswith(".json"):
        h.old.save_new(path, {"unexpected": True})
    else:
        path.mkdir()
    with pytest.raises(ValueError, match="Unexpected downstream|partial evidence"):
        h.inventory(h.ORIGINAL_BINDING)
    assert path.exists()
