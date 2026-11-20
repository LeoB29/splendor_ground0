"""Synthetic provenance/overlay tests; no policy, game rollout or production write."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import common as c, inventory as inv


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    source, recovery = tmp_path/"original", tmp_path/"original/recovery_01"
    monkeypatch.setattr(c, "SOURCE", source)
    monkeypatch.setattr(c, "OUTPUT", recovery)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(c.time, "time", lambda: c.ORIGINAL_LAUNCH_EPOCH+100)
    monkeypatch.setattr(inv.original, "validate_freeze", lambda: c.ORIGINAL_BINDING)
    monkeypatch.setattr(inv.original.models, "validate", lambda *args: None)
    c._CACHE.clear()

    def write(name, value, **metadata):
        path = source/name
        c.old.save_new(path, value)
        return c.artifact(path, **metadata)

    schedule = {stage: inv.old_arena.specs(stage) for stage in ("probe", "main")}
    write("launch.json", {"manifest": c.ORIGINAL_BINDING, "owner_authorized": True,
                           "epoch_seconds": c.ORIGINAL_LAUNCH_EPOCH})
    write("schedule.json", schedule)
    write("failure.json", {"manifest": c.ORIGINAL_BINDING, "traceback": "Invalid continuation ending"})
    for stage in ("probe", "main"):
        write(f"{stage}_launch.json", {"manifest": c.ORIGINAL_BINDING, "schedule": schedule[stage],
                                      "time": "synthetic timestamp"})
    for name in ("freeze.json", "manifest.json", "models_lock.json", "main_games_launch.json",
                 "probe_games_launch.json", "probe_audits_launch.json"):
        write(name, {"fixture": True})
    write("operational_decision.json", {"manifest": c.ORIGINAL_BINDING, "reasonable": True, "outcome_blind": True})
    all_games = {"probe": [], "main": []}
    for stage, count in (("probe", 12), ("main", 26)):
        for index in range(count):
            failed = stage == "main" and index == 13
            moves = []
            for turn in range(18 if failed else 1):
                snapshot = {"synthetic": [stage, index, turn]}
                record = {"manifest": c.ORIGINAL_BINDING, "root_identity": [stage, index, turn], "snapshot": snapshot}
                if failed and turn == 17:
                    record["failure"] = "unresolved synthetic trajectory"
                    outcome = {"reason": "repetition_cutoff", "winner": None, "wdl": None,
                               "turns": 18, "trace_sha256": "a"*64, "final_state_sha256": "b"*64}
                    record["worlds"] = [{} for _ in range(114)] + [
                        {"candidate_indices": [3], "outcomes": [outcome]}]
                ref = write(f"{stage}/roots/game_{index:04d}/turn_{turn:03d}.json", record)
                if failed and turn == 17:
                    write("recovery_diagnosis_01/reproduction.json", {
                        "source": ref, "world_index": 114, "candidate_index": 3,
                        "same_recorded_fields": True, "recorded": outcome,
                        "reproduced": {key: value for key, value in outcome.items() if key != "wdl"}})
                moves.append({"turn": turn, "actor": turn % 2, "state_sha256": c.old.digest(snapshot),
                              "root_sha256": ref["sha256"]})
            game = {"manifest": c.ORIGINAL_BINDING, "spec": schedule[stage][index],
                    "trace": moves[:-1] if failed else moves, "reason": "exception" if failed else "completed"}
            if failed:
                game.update(failure="synthetic", failed_decision=moves[-1])
            all_games[stage].append(write(f"{stage}/games/game_{index:04d}.json", game, index=index))
    audits = []
    for item in all_games["probe"]:
        audits.append(write(f"probe/audits/game_{item['index']:04d}.json", {
            "manifest": c.ORIGINAL_BINDING, "passed": True, "game_artifact": item}, index=item["index"]))
    locks = {"games": {"files": all_games["probe"]}, "audits": {"files": audits}}
    lock_refs = {kind: write(f"probe_{kind}_lock.json", lock) for kind, lock in locks.items()}
    monkeypatch.setattr(inv.original, "stage_lock", lambda stage, kind, binding: locks[kind])
    coverage = {"games": 12, "roots": 12, "recorded_worlds": 12, "recorded_continuations": 12,
                "replayed_worlds": 12, "replayed_continuations": 12, "stress_roots": 12}
    monkeypatch.setattr(inv.original, "checked_audit_coverage", lambda *args: deepcopy(coverage))
    sample = write("probe/audit_sample.json", {"manifest": c.ORIGINAL_BINDING,
                                              "games_lock_sha256": lock_refs["games"]["sha256"]})
    write("probe/audit_sample_seal.json", {"manifest": c.ORIGINAL_BINDING,
                                           "sample_sha256": sample["sha256"],
                                           "games_lock_sha256": lock_refs["games"]["sha256"]})
    write("probe_complete.json", {"manifest": c.ORIGINAL_BINDING, "games": 12,
                                   "all_prescribed_audits_pass": True, "full_inner_replay": False,
                                   "game_lock": lock_refs["games"], "audit_lock": lock_refs["audits"],
                                   "sample": sample, "coverage": coverage})
    return source, recovery


def freeze_fixture(recovery, value):
    c.old.save_new(recovery/"inventory.json", value)
    manifest = {"inventory": c.artifact(recovery/"inventory.json")}
    c.old.save_new(recovery/"manifest.json", manifest)
    binding = c.old.digest(manifest)
    c.old.save_new(recovery/"freeze.json", {"binding": binding, "manifest": c.artifact(recovery/"manifest.json")})
    c.old.save_new(recovery/"main_launch.json", {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding})
    return binding


def test_inventory_retains_exact_completed_panel_failure_and_original_clock(source_fixture):
    _, recovery = source_fixture
    value = inv.build()
    assert inv.validate(value) is True
    assert [item["index"] for item in value["retained_main_games"]] == inv.RETAINED
    assert len(value["pending_main_indices"]) == 377 and value["pending_main_indices"][:2] == [13, 26]
    assert value["deadline_epoch"] == c.ORIGINAL_LAUNCH_EPOCH+120*3600
    assert c.verified(value["diagnosis"])["same_recorded_fields"] is True
    binding = freeze_fixture(recovery, value)
    assert c.recovery_binding(recovery) == binding
    assert c.load_inventory(recovery) == value
    assert c.resolve_game("main", 13, recovery) is None
    assert c.resolve_partial_game("main", 13, recovery) == c.verify_hash(value["source_game"])
    assert c.resolve_partial_game("main", 26, recovery) is None
    assert c.resolve_root("main", 13, 17, recovery) == c.verify_hash(value["failed_root"])
    path = c.resolve_game("main", 0, recovery)
    assert c.validate_origin(c.old.read_json(path), path, c.ORIGINAL_BINDING, recovery)


@pytest.mark.parametrize("damage", ["extra_game", "extra_root", "wrong_clock", "incomplete_game", "wrong_prefix", "root_link"])
def test_changed_starting_state_is_not_silently_recovered(source_fixture, damage):
    source, _ = source_fixture
    if damage == "extra_game":
        c.old.save_new(source/"main/games/game_0026.json", {"extra": True})
    elif damage == "extra_root":
        c.old.save_new(source/"main/roots/game_0000/turn_001.json", {"extra": True})
    else:
        name = "launch.json" if damage == "wrong_clock" else f"main/games/game_{13 if damage == 'wrong_prefix' else 0:04d}.json"
        path = source/name
        value = c.old.read_json(path)
        if damage == "wrong_clock":
            value["epoch_seconds"] += 1
        elif damage == "incomplete_game":
            value["failure"] = "cannot reuse"
        elif damage == "wrong_prefix":
            value["trace"].pop()
        else:
            value["trace"][0]["root_sha256"] = "f"*64
        c.old.atomic_write(path, value)
    with pytest.raises(ValueError):
        inv.build()


def test_original_mutation_is_rejected_on_individual_resolution_and_full_validation(source_fixture):
    _, recovery = source_fixture
    value = inv.build()
    freeze_fixture(recovery, value)
    path = c.resolve_root("main", 0, 0, recovery)
    path.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Changed artifact"):
        c.resolve_root("main", 0, 0, recovery)
    with pytest.raises(ValueError, match="Changed artifact"):
        inv.validate(value)


@pytest.mark.parametrize("damage", ["source", "coordinate", "recorded", "reproduced", "flag"])
def test_diagnosis_must_authenticate_exact_failed_branch(source_fixture, damage):
    source, _ = source_fixture
    path = source/"recovery_diagnosis_01/reproduction.json"
    value = c.old.read_json(path)
    if damage == "source":
        value["source"]["sha256"] = "f"*64
    elif damage == "coordinate":
        value["candidate_index"] = 2
    elif damage in ("recorded", "reproduced"):
        value[damage]["turns"] += 1
    else:
        value["same_recorded_fields"] = False
    c.old.atomic_write(path, value)
    with pytest.raises(ValueError, match="Diagnosis"):
        inv.build()


def test_only_named_failed_root_may_be_replaced_and_new_origin_is_checked(source_fixture):
    _, recovery = source_fixture
    value = inv.build()
    binding = freeze_fixture(recovery, value)
    path = recovery/"main/roots/game_0013/turn_017.json"
    record = {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding, "root_identity": ["main", 13, 17]}
    c.old.save_new(path, record)
    assert c.resolve_root("main", 13, 17, recovery) == path
    record["recovery_binding"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        c.validate_origin(record, path, c.ORIGINAL_BINDING, recovery)
    duplicate = recovery/"main/roots/game_0000/turn_000.json"
    c.old.save_new(duplicate, {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding})
    with pytest.raises(ValueError, match="Duplicate successful"):
        c.resolve_root("main", 0, 0, recovery)


def test_exact_union_excludes_failed_original_but_detects_extra_overlay(source_fixture):
    _, recovery = source_fixture
    value = inv.build()
    binding = freeze_fixture(recovery, value)
    games = {c.verify_hash(ref) for ref in value["games"]["main"].values()}
    roots = {c.verify_hash(ref) for ref in value["roots"]["main"].values() if ref != value["failed_root"]}
    assert c.validate_resolved_inventory("main", games, roots, recovery)
    path = recovery/"main/roots/game_0026/turn_000.json"
    c.old.save_new(path, {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding,
                          "root_identity": ["main", 26, 0]})
    with pytest.raises(ValueError, match="Merged root inventory"):
        c.validate_resolved_inventory("main", games, roots, recovery)


def test_abort_disk_and_original_deadline_are_independent_of_relaunch(source_fixture, monkeypatch, tmp_path):
    _, recovery = source_fixture
    recovery.mkdir()
    c.old.save_new(recovery/"launch.json", {"epoch_seconds": c.ORIGINAL_LAUNCH_EPOCH+100000})
    monkeypatch.setattr(c.time, "time", lambda: c.DEADLINE_EPOCH+1)
    with pytest.raises(c.BudgetExceeded, match="Original"):
        c.budget_boundary(recovery)
    isolated = tmp_path/"isolated"
    isolated.mkdir()
    c.budget_boundary(isolated)
    c.old.save_new(isolated/"abort.json", {"reason": "fixture"})
    with pytest.raises(c.RecoveryAborted):
        c.budget_boundary(isolated)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(c.BudgetExceeded, match="disk reserve"):
        c.budget_boundary(tmp_path/"another_fixture")
