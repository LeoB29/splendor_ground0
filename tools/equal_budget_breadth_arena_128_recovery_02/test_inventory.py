"""Bounded synthetic provenance tests; no production writes or gameplay."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.equal_budget_breadth_arena_128_recovery_01 import inventory as prior_inventory
from tools.equal_budget_breadth_arena_128_recovery_01.test_inventory import source_fixture as original_fixture
from . import common as c, inventory as inv


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    source, previous = original_fixture.__wrapped__(tmp_path, monkeypatch)
    output = source/"recovery_02"
    monkeypatch.setattr(c, "SOURCE", source)
    monkeypatch.setattr(c, "PREVIOUS", previous)
    monkeypatch.setattr(c, "OUTPUT", output)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda path: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(c.time, "time", lambda: c.ORIGINAL_LAUNCH_EPOCH+100)
    monkeypatch.setattr(inv.previous, "validate_freeze", lambda full_inventory=False: c.PREVIOUS_BINDING)
    c._CACHE.clear()
    prior = prior_inventory.build()

    def write(name, value, **metadata):
        path = previous/name
        c.old.save_new(path, value)
        return c.artifact(path, **metadata)

    write("inventory.json", prior)
    log = write("fixture_check_log.txt", {"exit_code": 0})
    checks = write("fixture_checks.json", {"exit_code": 0, "logs": [log]})
    manifest = write("manifest.json", {"checks": checks})
    write("freeze.json", {"binding": c.PREVIOUS_BINDING, "manifest": manifest})
    write("resource_gate.json", {"reasonable": True, "outcome_blind": True})
    write("launch.json", {"manifest": c.ORIGINAL_BINDING, "recovery_binding": c.PREVIOUS_BINDING,
                          "original_launch_epoch": c.ORIGINAL_LAUNCH_EPOCH})
    schedule = inv.old_arena.specs("main")
    write("main_launch.json", {"manifest": c.ORIGINAL_BINDING, "recovery_binding": c.PREVIOUS_BINDING,
                               "schedule": schedule, "time": "fixture"})
    write("main_games_launch.json", {"manifest": c.ORIGINAL_BINDING, "recovery_binding": c.PREVIOUS_BINDING,
                                     "stage": "main", "kind": "games"})
    stop = {"reason": "Unresolved repetition can change the original-order winner", "type": "AmbiguousDecision"}
    write("abort.json", stop)
    write("operational_hold.json", {**stop, "manifest": c.PREVIOUS_BINDING, "original_binding": c.ORIGINAL_BINDING})
    write("progress.json", {"stage": "operational_hold"})
    for index in range(318):
        if str(index) in prior["games"]["main"]:
            continue
        partial_turn = inv.PARTIAL_TURNS.get(index)
        if index == 13:
            inherited_game = c.verified(prior["source_game"])
            moves = deepcopy(inherited_game["trace"])
            turns = [17]
        else:
            moves = []
            turns = range(partial_turn+1 if partial_turn is not None else 1)
        for turn in turns:
            snapshot = {"synthetic": ["main", index, turn]}
            record = {"manifest": c.ORIGINAL_BINDING, "recovery_binding": c.PREVIOUS_BINDING,
                      "root_identity": ["main", index, turn], "snapshot": snapshot, "worlds": [],
                      "timing": {"wall": 1, "cpu": 1}}
            if index == 13:
                record["recovery_origin"] = inv._minimal(prior["failed_root"])
            if partial_turn == turn:
                record["failure"] = "AmbiguousDecision" if index == 310 else "RecoveryAborted"
                if index == 310:
                    record.update(hold="ambiguous_repetition_interval", choice=None, certificate={"choice": None})
                    outcomes = [{"reason": "repetition_cutoff", "winner": None, "wdl": None,
                                 "turns": 29, "trace_sha256": str(j)*64, "final_state_sha256": str(j+2)*64}
                                for j in range(2)]
                    record["worlds"] = [{} for _ in range(42)] + [{"candidate_indices": [0, 1], "outcomes": outcomes}]
            ref = write(f"main/roots/game_{index:04d}/turn_{turn:03d}.json", record, index=index, turn=turn)
            if index == 310 and turn == 58:
                diagnosis_source = inv._minimal(ref)
            moves.append({"turn": turn, "state_sha256": c.old.digest(snapshot), "root_sha256": ref["sha256"]})
        game = {"manifest": c.ORIGINAL_BINDING, "recovery_binding": c.PREVIOUS_BINDING, "spec": schedule[index],
                "trace": moves[:-1] if partial_turn is not None else moves,
                "reason": "exception" if partial_turn is not None else "completed"}
        if index == 13:
            game["recovery_origin"] = inv._minimal(prior["source_game"])
        if partial_turn is not None:
            game.update(failure="AmbiguousDecision" if index == 310 else "RecoveryAborted", failed_decision=moves[-1])
        write(f"main/games/game_{index:04d}.json", game, index=index)
    branches = []
    for candidate, outcome in enumerate(outcomes):
        path = source/f"recovery_diagnosis_02/candidate_{candidate}_reproduction.json"
        c.old.save_new(path, {"source": diagnosis_source, "world_index": 42, "candidate_index": candidate,
                             "same_recorded_fields": True, "recorded": outcome,
                             "reproduced": {k: v for k, v in outcome.items() if k != "wdl"}})
        branches.append(c.artifact(path))
    c.old.save_new(source/inv.DIAGNOSIS_RELATIVE, {"source": diagnosis_source, "branches": branches,
        "same_recorded_fields": True, "equal_after_sorting_purchased_cards_only": True,
        "equal_final_termination_context": False, "extended_continuation_calls": 0})
    return source, previous, output


def frozen_inventory(output, value):
    c.old.save_new(output/"inventory.json", value)
    manifest = {"inventory": c.artifact(output/"inventory.json")}
    c.old.save_new(output/"manifest.json", manifest)
    binding = c.old.digest(manifest)
    c.old.save_new(output/"freeze.json", {"binding": binding, "manifest": c.artifact(output/"manifest.json")})
    c.old.save_new(output/"main_launch.json", {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding})
    return binding


def test_flat_union_population_histories_and_original_clock(source_fixture):
    _, _, output = source_fixture
    value = inv.build()
    assert inv.validate(value)
    assert [r["index"] for r in value["retained_main_games"]] == list(range(310))
    assert value["pending_main_indices"] == list(range(310, 402))
    assert len(value["partial_games"]["main"]) == len(value["replaceable_roots"]["main"]) == 8
    assert len(value["game_histories"]["main"]["13"]) == len(value["root_histories"]["main"]["13:17"]) == 2
    assert value["deadline_epoch"] == c.ORIGINAL_LAUNCH_EPOCH+120*3600
    frozen_inventory(output, value)
    for index in (0, 13, 309):
        path = c.resolve_game("main", index, output)
        assert c.validate_origin(c.old.read_json(path), path, c.ORIGINAL_BINDING, output)
    assert c.resolve_game("main", 310, output) is None
    assert c.resolve_partial_game("main", 310, output) == c.verify_hash(value["partial_games"]["main"]["310"])
    assert c.resolve_partial_game("main", 318, output) is None


def test_inherited_r1_lineage_reaches_original_failed_ancestors(source_fixture):
    _, _, output = source_fixture
    value = inv.build()
    frozen_inventory(output, value)
    for ref in (value["games"]["main"]["13"], value["roots"]["main"]["13:17"]):
        record = c.verified(ref)
        ancestor = c.lineage_ancestor(record, output)
        assert "failure" in c.old.read_json(ancestor)
        assert c.source_by_path(ancestor, output)["path"] == record["recovery_origin"]["path"]


def test_new_lineage_uses_immediate_failed_r1_record(source_fixture):
    _, _, output = source_fixture
    value = inv.build()
    binding = frozen_inventory(output, value)
    ref = value["replaceable_roots"]["main"]["310:58"]
    record = {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding, "root_identity": ["main", 310, 58],
              "recovery_origin": inv._minimal(ref)}
    assert c.lineage_ancestor(record, output) == c.verify_hash(ref)
    record["root_identity"] = ["main", 311, 35]
    with pytest.raises(ValueError, match="Ancestor root identity"):
        c.lineage_ancestor(record, output)


@pytest.mark.parametrize("damage", ["extra_game", "extra_root", "partial_turn", "root_link", "wrong_generation"])
def test_changed_starting_population_or_links_rejected(source_fixture, damage):
    _, previous, _ = source_fixture
    if damage.startswith("extra"):
        path = previous/("main/games/game_0318.json" if damage == "extra_game" else "main/roots/game_0309/turn_001.json")
        c.old.save_new(path, {"extra": True})
    else:
        path = previous/"main/games/game_0310.json"
        game = c.old.read_json(path)
        if damage == "partial_turn":
            game["trace"].pop()
        elif damage == "root_link":
            game["failed_decision"]["root_sha256"] = "f"*64
        else:
            game["recovery_binding"] = "wrong"
        c.old.atomic_write(path, game)
    with pytest.raises(ValueError):
        inv.build()


def test_only_failed_roots_can_be_overlaid_and_generation_is_authenticated(source_fixture):
    _, _, output = source_fixture
    value = inv.build()
    binding = frozen_inventory(output, value)
    path = output/"main/roots/game_0310/turn_058.json"
    record = {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding, "root_identity": ["main", 310, 58]}
    c.old.save_new(path, record)
    assert c.resolve_root("main", 310, 58, output) == path
    record["recovery_binding"] = c.PREVIOUS_BINDING
    with pytest.raises(ValueError, match="provenance"):
        c.validate_origin(record, path, c.ORIGINAL_BINDING, output)
    duplicate = output/"main/roots/game_0310/turn_057.json"
    c.old.save_new(duplicate, {"manifest": c.ORIGINAL_BINDING, "recovery_binding": binding})
    with pytest.raises(ValueError, match="Duplicate successful"):
        c.resolve_root("main", 310, 57, output)


def test_exact_effective_union_excludes_all_historical_failures(source_fixture):
    _, _, output = source_fixture
    value = inv.build()
    binding = frozen_inventory(output, value)
    games = {c.verify_hash(ref) for ref in value["games"]["main"].values()}
    roots = {c.verify_hash(ref) for key, ref in value["roots"]["main"].items()
             if key not in value["replaceable_roots"]["main"]}
    assert c.validate_resolved_inventory("main", games, roots, output)
    c.old.save_new(output/"main/roots/game_0318/turn_000.json", {
        "manifest": c.ORIGINAL_BINDING, "recovery_binding": binding, "root_identity": ["main", 318, 0]})
    with pytest.raises(ValueError, match="Merged root inventory"):
        c.validate_resolved_inventory("main", games, roots, output)


def test_source_mutation_rejected_on_resolution_and_full_validation(source_fixture):
    _, _, output = source_fixture
    value = inv.build()
    frozen_inventory(output, value)
    path = c.resolve_root("main", 310, 58, output)
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="Changed artifact"):
        c.resolve_root("main", 310, 58, output)
    with pytest.raises(ValueError, match="Changed artifact"):
        inv.validate(value)


@pytest.mark.parametrize("damage", ["source", "reproduced"])
def test_diagnosis_links_and_reproduction_are_checked(source_fixture, damage):
    source, _, _ = source_fixture
    comparison_path = source/inv.DIAGNOSIS_RELATIVE
    comparison = c.old.read_json(comparison_path)
    if damage == "source":
        comparison["source"]["sha256"] = "f"*64
    else:
        path = source/"recovery_diagnosis_02/candidate_1_reproduction.json"
        branch = c.old.read_json(path)
        branch["reproduced"]["turns"] += 1
        c.old.atomic_write(path, branch)
        comparison["branches"][1] = c.artifact(path)
    c.old.atomic_write(comparison_path, comparison)
    with pytest.raises(ValueError, match="Diagnosis"):
        inv.build()


def test_deadline_abort_and_disk_do_not_reset(source_fixture, monkeypatch, tmp_path):
    _, _, output = source_fixture
    monkeypatch.setattr(c.time, "time", lambda: c.DEADLINE_EPOCH+1)
    with pytest.raises(c.BudgetExceeded, match="Original"):
        c.budget_boundary(output)
    isolated = tmp_path/"isolated"
    isolated.mkdir()
    c.budget_boundary(isolated)
    c.old.save_new(isolated/"abort.json", {"reason": "fixture"})
    with pytest.raises(c.RecoveryAborted):
        c.budget_boundary(isolated)
    monkeypatch.setattr(c.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(c.BudgetExceeded, match="disk reserve"):
        c.budget_boundary(tmp_path/"another_fixture")
