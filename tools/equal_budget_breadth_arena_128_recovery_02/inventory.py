"""Pin the complete original/R1 evidence and flatten its effective records."""
from copy import deepcopy
from pathlib import Path

from tools.equal_budget_breadth_arena_128 import experiment as original
from tools.equal_budget_breadth_arena_128 import arena as old_arena
from tools.equal_budget_breadth_arena_128_recovery_01 import experiment as previous
from . import common as c

RETAINED = list(range(310))
PARTIAL_TURNS = {310: 58, 311: 35, 312: 36, 313: 34, 314: 20, 315: 5, 316: 2, 317: 0}
DIAGNOSIS_RELATIVE = "recovery_diagnosis_02/comparison.json"


def _minimal(ref):
    return {key: ref[key] for key in ("path", "sha256")}


def _validate_diagnosis(ref, ambiguous):
    comparison, root = c.verified(ref), c.verified(ambiguous)
    c.require(comparison["source"] == _minimal(ambiguous)
              and comparison["same_recorded_fields"] is True
              and comparison["equal_after_sorting_purchased_cards_only"] is True
              and comparison["equal_final_termination_context"] is False
              and comparison["extended_continuation_calls"] == 0,
              "Diagnosis source or comparison differs")
    c.require(len(comparison["branches"]) == 2, "Diagnosis branch inventory differs")
    world = root["worlds"][42]
    for candidate, branch_ref in enumerate(comparison["branches"]):
        path = c.verify_hash(branch_ref)
        c.require(path.resolve() == (c.SOURCE/f"recovery_diagnosis_02/candidate_{candidate}_reproduction.json").resolve(),
                  "Diagnosis branch path differs")
        branch = c.old.read_json(path)
        c.require(branch["source"] == comparison["source"] and branch["world_index"] == 42
                  and branch["candidate_index"] == candidate and branch["same_recorded_fields"] is True,
                  "Diagnosis branch source/coordinates differ")
        c.require(candidate in world["candidate_indices"], "Diagnosis candidate absent")
        recorded = world["outcomes"][world["candidate_indices"].index(candidate)]
        c.require(branch["recorded"] == recorded and recorded["reason"] == "repetition_cutoff"
                  and recorded["winner"] is None and recorded["wdl"] is None,
                  "Diagnosis recorded outcome differs")
        c.require(all(branch["reproduced"][key] == recorded[key] for key in
                      ("reason", "winner", "turns", "trace_sha256", "final_state_sha256")),
                  "Diagnosis reproduction differs")
    return comparison["branches"]


def build():
    c.budget_boundary(c.OUTPUT)
    c.require(original.validate_freeze() == c.ORIGINAL_BINDING, "Original frozen identity differs")
    c.require(previous.validate_freeze(full_inventory=True) == c.PREVIOUS_BINDING, "R1 frozen identity differs")
    original.models.validate(c.ORIGINAL_BINDING, c.SOURCE)
    prior = c.old.read_json(c.PREVIOUS/"inventory.json")
    for name in ("main_games_lock.json", "main_audits_lock.json", "main_complete.json", "result.json", "final_status.json"):
        c.require(not (c.PREVIOUS/name).exists(), f"Unexpected R1 downstream evidence: {name}")
    c.require(not (c.PREVIOUS/"main/audit_sample.json").exists(), "R1 main sample already exists")
    preserved = {ref["path"]: deepcopy(ref) for ref in prior["preserved"]}
    sources, origins = {}, {}
    histories = {"games": {"probe": {}, "main": {}}, "roots": {"probe": {}, "main": {}}}

    def retain(path, **metadata):
        ref = c.artifact(path, **metadata)
        c.require(ref["path"] not in preserved or preserved[ref["path"]] == _minimal(ref),
                  "Changed preserved source")
        preserved[ref["path"]] = _minimal(ref)
        return ref

    def add_source(ref, generation, stage, kind, index, launch, turn=None):
        c.require(ref["path"] not in sources or sources[ref["path"]] == ref, "Conflicting source record")
        sources[ref["path"]] = deepcopy(ref)
        origin = {"generation": generation, "stage": stage, "kind": kind, "index": index, "launch": launch}
        if turn is not None:
            origin["turn"] = turn
        origins[ref["path"]] = origin
        key = str(index) if kind == "game" else f"{index}:{turn}"
        history = histories["games" if kind == "game" else "roots"][stage].setdefault(key, [])
        if ref not in history:
            history.append(deepcopy(ref))

    for stage in ("probe", "main"):
        launch = prior["original_stage_launches"][stage]
        for ref in prior["games"][stage].values():
            add_source(ref, "original", stage, "game", ref["index"], launch)
        for ref in prior["roots"][stage].values():
            add_source(ref, "original", stage, "root", ref["index"], launch, ref["turn"])
    add_source(prior["source_game"], "original", "main", "game", 13, prior["original_stage_launches"]["main"])
    parent_refs = {}
    for name in ("freeze.json", "manifest.json", "inventory.json", "resource_gate.json", "launch.json",
                 "main_launch.json", "main_games_launch.json", "abort.json", "operational_hold.json", "progress.json"):
        parent_refs[name.removesuffix(".json")] = retain(c.PREVIOUS/name)
    manifest = c.verified(parent_refs["manifest"])
    retain(c.verify_hash(manifest["checks"]))
    checks = c.verified(manifest["checks"])
    for ref in checks["logs"]:
        retain(c.verify_hash(ref))
    for relative in ("runtime", "workers", "status_events"):
        for path in (c.PREVIOUS/relative).rglob("*"):
            if path.is_file():
                retain(path)
    launch = c.verified(parent_refs["launch"])
    main_launch = c.verified(parent_refs["main_launch"])
    schedule = old_arena.specs("main")
    c.require(launch["manifest"] == c.ORIGINAL_BINDING and launch["recovery_binding"] == c.PREVIOUS_BINDING
              and launch["original_launch_epoch"] == c.ORIGINAL_LAUNCH_EPOCH, "R1 launch binding/clock differs")
    c.require(main_launch["manifest"] == c.ORIGINAL_BINDING
              and main_launch["recovery_binding"] == c.PREVIOUS_BINDING
              and main_launch["schedule"] == schedule, "R1 main launch differs")
    main_launch_mtime = c.verify_hash(parent_refs["main_launch"]).stat().st_mtime_ns
    hold, abort = c.verified(parent_refs["operational_hold"]), c.verified(parent_refs["abort"])
    c.require(hold["manifest"] == c.PREVIOUS_BINDING and hold["original_binding"] == c.ORIGINAL_BINDING
              and hold["type"] == abort["type"] == "AmbiguousDecision"
              and hold["reason"] == abort["reason"] == "Unresolved repetition can change the original-order winner",
              "R1 stop reason differs")
    games, roots = deepcopy(prior["games"]), deepcopy(prior["roots"])
    partial_games, replaceable_roots = {"probe": {}, "main": {}}, {"probe": {}, "main": {}}
    expected_games, expected_roots = set(), set()
    existing_originals = set(map(int, prior["games"]["main"]))
    for index in range(318):
        if index in existing_originals:
            continue
        c.budget_boundary(c.OUTPUT)
        path = c.PREVIOUS/"main/games"/f"game_{index:04d}.json"
        expected_games.add(path.resolve())
        ref = retain(path, index=index)
        game = c.verified(ref)
        c.require(game["manifest"] == c.ORIGINAL_BINDING and game["recovery_binding"] == c.PREVIOUS_BINDING
                  and game["spec"] == schedule[index], "R1 game binding/spec differs")
        c.require(path.stat().st_mtime_ns >= main_launch_mtime,
                  "R1 game predates launch")
        add_source(ref, "recovery_01", "main", "game", index, parent_refs["main_launch"])
        if index in PARTIAL_TURNS:
            turn = PARTIAL_TURNS[index]
            failure_type = "AmbiguousDecision" if index == 310 else "RecoveryAborted"
            c.require(failure_type in game.get("failure", "") and game["reason"] == "exception"
                      and len(game["trace"]) == turn and game["failed_decision"]["turn"] == turn,
                      "R1 partial game prefix differs")
            partial_games["main"][str(index)] = ref
            moves = [*game["trace"], game["failed_decision"]]
        else:
            c.require("failure" not in game and game["reason"] in c.old.VALID_NATURAL_TERMINATIONS,
                      "Incomplete game in retained R1 population")
            games["main"][str(index)] = ref
            moves = game["trace"]
        c.require([move["turn"] for move in moves] == list(range(len(moves))), "R1 game prefix has gaps")
        for move in moves:
            turn, key = move["turn"], f"{index}:{move['turn']}"
            root_path = c.PREVIOUS/"main/roots"/f"game_{index:04d}"/f"turn_{turn:03d}.json"
            if root_path.exists():
                inherited = prior["roots"]["main"].get(key)
                c.require(inherited is None or inherited == prior["failed_root"], "R1 overwrote completed original root")
                expected_roots.add(root_path.resolve())
                root_ref = retain(root_path, index=index, turn=turn)
                root = c.verified(root_ref)
                c.require(root["manifest"] == c.ORIGINAL_BINDING and root["recovery_binding"] == c.PREVIOUS_BINDING,
                          "R1 root binding differs")
                c.require(root_path.stat().st_mtime_ns >= main_launch_mtime,
                          "R1 root predates launch")
                add_source(root_ref, "recovery_01", "main", "root", index, parent_refs["main_launch"], turn)
            else:
                root_ref = prior["roots"]["main"].get(key)
                c.require(root_ref is not None, "Missing inherited original root")
                root = c.verified(root_ref)
            c.require(root_ref["sha256"] == move["root_sha256"]
                      and root["root_identity"] == ["main", index, turn]
                      and c.old.digest(root["snapshot"]) == move["state_sha256"], "R1 root/trace linkage differs")
            failed = index in PARTIAL_TURNS and turn == PARTIAL_TURNS[index]
            c.require(("failure" in root) == failed, "Unexpected R1 root failure")
            roots["main"][key] = root_ref
            if failed:
                replaceable_roots["main"][key] = root_ref
    c.require({p.resolve() for p in (c.PREVIOUS/"main/games").glob("*.json")} == expected_games,
              "Unexpected R1 game files")
    c.require({p.resolve() for p in (c.PREVIOUS/"main/roots").rglob("*.json")} == expected_roots,
              "Unexpected R1 root files")
    c.require(sorted(map(int, games["main"])) == RETAINED, "Wrong completed merged population")
    ambiguous = replaceable_roots["main"]["310:58"]
    failed = c.verified(ambiguous)
    c.require(failed.get("hold") == "ambiguous_repetition_interval" and failed["choice"] is None
              and failed["certificate"]["choice"] is None, "Ambiguous root hold differs")
    diagnosis = retain(c.SOURCE/DIAGNOSIS_RELATIVE)
    for branch in _validate_diagnosis(diagnosis, ambiguous):
        retain(c.verify_hash(branch))
    value = {"original_binding": c.ORIGINAL_BINDING, "previous_binding": c.PREVIOUS_BINDING,
             "original_launch_epoch": c.ORIGINAL_LAUNCH_EPOCH, "deadline_epoch": c.DEADLINE_EPOCH,
             "original_launch": prior["original_launch"], "original_main_launch": prior["original_main_launch"],
             "original_stage_launches": prior["original_stage_launches"], "previous": parent_refs,
             "models": prior["models"], "probe": prior["probe"], "probe_coverage": prior["probe_coverage"],
             "games": games, "roots": roots, "partial_games": partial_games, "replaceable_roots": replaceable_roots,
             "ambiguous_root": ambiguous, "diagnosis": diagnosis, "sources": sources, "origins": origins,
             "game_histories": histories["games"], "root_histories": histories["roots"],
             "retained_main_games": [games["main"][str(index)] for index in RETAINED],
             "pending_main_indices": list(range(310, 402)), "partial_turns": {str(i): t for i, t in PARTIAL_TURNS.items()},
             "preserved": [preserved[path] for path in sorted(preserved)]}
    _validate_structure(value)
    return value


def _validate_structure(value):
    c.require(value["original_binding"] == c.ORIGINAL_BINDING and value["previous_binding"] == c.PREVIOUS_BINDING
              and value["original_launch_epoch"] == c.ORIGINAL_LAUNCH_EPOCH
              and value["deadline_epoch"] == c.DEADLINE_EPOCH, "Inventory binding/clock differs")
    c.require(sorted(map(int, value["games"]["main"])) == RETAINED
              and value["retained_main_games"] == [value["games"]["main"][str(i)] for i in RETAINED]
              and value["pending_main_indices"] == list(range(310, 402)), "Recovery population differs")
    c.require(value["partial_turns"] == {str(i): t for i, t in PARTIAL_TURNS.items()}
              and set(value["partial_games"]["main"]) == set(map(str, PARTIAL_TURNS))
              and set(value["replaceable_roots"]["main"]) == {f"{i}:{t}" for i, t in PARTIAL_TURNS.items()},
              "Partial lineage population differs")
    c.require(value["ambiguous_root"] == value["replaceable_roots"]["main"]["310:58"], "Ambiguous root differs")
    preserved = value["preserved"]
    lookup = {ref["path"]: ref["sha256"] for ref in preserved}
    c.require(len(lookup) == len(preserved), "Duplicate preserved artifacts")
    c.require(set(value["sources"]) == set(value["origins"]), "Origin membership differs")
    for key, ref in value["sources"].items():
        c.require(ref["path"] == key and lookup.get(key) == ref["sha256"], "Unsealed source record")
        origin = value["origins"][key]
        c.require(origin["generation"] in ("original", "recovery_01") and origin["stage"] in ("probe", "main"),
                  "Unknown source generation")
        launch = origin["launch"]
        c.require(lookup.get(launch["path"]) == launch["sha256"], "Unsealed source launch")
    for maps in (value["games"], value["roots"], value["partial_games"], value["replaceable_roots"]):
        for rows in maps.values():
            for ref in rows.values():
                c.require(value["sources"].get(ref["path"]) == ref, "Unsealed effective record")
    for name, kind in (("game_histories", "game"), ("root_histories", "root")):
        for stage, rows in value[name].items():
            for key, history in rows.items():
                c.require(history and len({r["path"] for r in history}) == len(history), "Duplicate or empty source history")
                for ref in history:
                    c.require(value["sources"].get(ref["path"]) == ref, "Unsealed history record")
                    origin = value["origins"][ref["path"]]
                    expected_key = str(origin["index"]) if kind == "game" else f"{origin['index']}:{origin['turn']}"
                    c.require(origin["kind"] == kind and origin["stage"] == stage and key == expected_key,
                              "History identity differs")
                effective = (value["games"][stage].get(key) or value["partial_games"][stage].get(key)
                             if kind == "game" else value["roots"][stage].get(key))
                c.require(effective == history[-1], "History does not end at effective record")
    for stage, rows in value["replaceable_roots"].items():
        for key, ref in rows.items():
            c.require(value["roots"][stage][key] == ref, "Replaceable root is not current")
    for name in ("original_launch", "original_main_launch", "models", "diagnosis"):
        ref = value[name]
        c.require(lookup.get(ref["path"]) == ref["sha256"], "Unsealed lineage artifact")


def validate(value):
    """Reauthenticate all frozen evidence without any policy evaluation or rollout."""
    _validate_structure(value)
    for ref in value["preserved"]:
        c.verify_hash(ref)
    branches = _validate_diagnosis(value["diagnosis"], value["ambiguous_root"])
    lookup = {ref["path"]: ref["sha256"] for ref in value["preserved"]}
    c.require(all(lookup.get(ref["path"]) == ref["sha256"] for ref in branches), "Unsealed diagnosis branch")
    return True
