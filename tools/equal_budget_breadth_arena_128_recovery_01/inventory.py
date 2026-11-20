"""Seal completed games and explicit failure lineage without running a policy."""
from pathlib import Path

from tools.equal_budget_breadth_arena_128 import experiment as original
from tools.equal_budget_breadth_arena_128 import arena as old_arena
from . import common as c

RETAINED = list(range(13)) + list(range(14, 26))


def _validate_diagnosis(ref, failed_root):
    diagnosis, root = c.verified(ref), c.verified(failed_root)
    c.require(diagnosis["source"] == {key: failed_root[key] for key in ("path", "sha256")}
              and diagnosis["world_index"] == 114 and diagnosis["candidate_index"] == 3
              and diagnosis["same_recorded_fields"] is True, "Diagnosis source/coordinate linkage differs")
    world = root["worlds"][114]
    c.require(3 in world["candidate_indices"], "Diagnosis candidate absent from failed world")
    recorded = world["outcomes"][world["candidate_indices"].index(3)]
    c.require(diagnosis["recorded"] == recorded and recorded["reason"] == "repetition_cutoff"
              and recorded["winner"] is None and recorded["wdl"] is None,
              "Diagnosis recorded failure differs")
    c.require(all(diagnosis["reproduced"][key] == recorded[key] for key in
                  ("reason", "winner", "turns", "trace_sha256", "final_state_sha256")),
              "Diagnosis exact reproduction differs")


def build():
    c.budget_boundary(c.OUTPUT)
    c.require(original.validate_freeze() == c.ORIGINAL_BINDING, "Original frozen identity differs")
    original.models.validate(c.ORIGINAL_BINDING, c.SOURCE)
    for name in ("main_games_lock.json", "main_audits_lock.json", "main_complete.json", "result.json", "final_status.json"):
        c.require(not (c.SOURCE/name).exists(), f"Unexpected downstream original evidence: {name}")
    c.require(not (c.SOURCE/"main/audit_sample.json").exists(), "Original main sample already exists")
    preserved = {}

    def retain(path, **metadata):
        item = c.artifact(path, **metadata)
        minimal = {key: item[key] for key in ("path", "sha256")}
        c.require(item["path"] not in preserved or preserved[item["path"]] == minimal,
                  "Conflicting original evidence hash")
        preserved[item["path"]] = minimal
        return item

    def read_named(name):
        item = retain(c.SOURCE/name)
        return c.verified(item), item

    launch, launch_ref = read_named("launch.json")
    c.require(launch["manifest"] == c.ORIGINAL_BINDING and launch["owner_authorized"] is True
              and launch["epoch_seconds"] == c.ORIGINAL_LAUNCH_EPOCH, "Original launch clock differs")
    schedule, _ = read_named("schedule.json")
    c.require(schedule == {stage: old_arena.specs(stage) for stage in ("probe", "main")}, "Original schedule differs")
    failure, _ = read_named("failure.json")
    c.require(failure["manifest"] == c.ORIGINAL_BINDING and "Invalid continuation ending" in failure["traceback"],
              "Different original failure requires review")
    main_launch, main_launch_ref = read_named("main_launch.json")
    c.require(main_launch["manifest"] == c.ORIGINAL_BINDING
              and main_launch["schedule"] == schedule["main"], "Original main launch differs")
    stage_launches = {"main": main_launch_ref, "probe": retain(c.SOURCE/"probe_launch.json")}
    probe_launch = c.verified(stage_launches["probe"])
    c.require(probe_launch["manifest"] == c.ORIGINAL_BINDING
              and probe_launch["schedule"] == schedule["probe"], "Original probe launch differs")
    for name in ("freeze.json", "manifest.json", "models_lock.json", "operational_decision.json",
                 "main_games_launch.json", "probe_games_launch.json", "probe_audits_launch.json"):
        retain(c.SOURCE/name)
    operational = c.old.read_json(c.SOURCE/"operational_decision.json")
    c.require(operational["manifest"] == c.ORIGINAL_BINDING and operational["reasonable"] is True
              and operational["outcome_blind"] is True, "Original probe resource gate did not pass")
    for relative in ("runtime", "workers", "status_events"):
        for path in (c.SOURCE/relative).rglob("*"):
            if path.is_file():
                retain(path)
    for name in ("progress.json",):
        path = c.SOURCE/name
        if path.is_file():
            retain(path)

    complete, complete_ref = read_named("probe_complete.json")
    c.require(complete["manifest"] == c.ORIGINAL_BINDING and complete["games"] == 12
              and complete["all_prescribed_audits_pass"] is True and complete["full_inner_replay"] is False,
              "Original probe incomplete")
    probe = {"complete": complete_ref}
    locks = {}
    for kind, key in (("games", "game_lock"), ("audits", "audit_lock")):
        ref = retain(c.SOURCE/f"probe_{kind}_lock.json")
        c.require(complete[key] == ref, "Probe completion lock link differs")
        locks[kind] = original.stage_lock("probe", kind, c.ORIGINAL_BINDING)
        probe[key] = ref
        for item in locks[kind]["files"]:
            retain(c.verify_hash(item))
    probe["sample"] = retain(c.SOURCE/"probe/audit_sample.json")
    probe["sample_seal"] = retain(c.SOURCE/"probe/audit_sample_seal.json")
    c.require(complete["sample"] == probe["sample"], "Probe sample link differs")
    sample, seal = c.verified(probe["sample"]), c.verified(probe["sample_seal"])
    c.require(seal["manifest"] == sample["manifest"] == c.ORIGINAL_BINDING
              and seal["sample_sha256"] == probe["sample"]["sha256"]
              and seal["games_lock_sha256"] == sample["games_lock_sha256"] == probe["game_lock"]["sha256"],
              "Probe sample seal differs")
    receipts = [c.verified(ref) for ref in locks["audits"]["files"]]
    for game_ref, receipt in zip(locks["games"]["files"], receipts, strict=True):
        c.require(receipt["manifest"] == c.ORIGINAL_BINDING and receipt["passed"] is True
                  and receipt["game_artifact"] == game_ref, "Probe audit linkage differs")
    coverage = original.checked_audit_coverage(receipts, sample, probe["sample"]["sha256"])
    c.require(coverage == complete["coverage"], "Probe verified coverage differs")

    games, roots = {"probe": {}, "main": {}}, {"probe": {}, "main": {}}
    source_game = failed_root = None
    for stage, expected_indices in (("probe", list(range(12))), ("main", list(range(26)))):
        expected_paths, expected_root_paths = set(), set()
        for index in expected_indices:
            c.budget_boundary(c.OUTPUT)
            path = c.SOURCE/stage/"games"/f"game_{index:04d}.json"
            expected_paths.add(path.resolve())
            ref = retain(path, index=index)
            game = c.verified(ref)
            c.require(game["manifest"] == c.ORIGINAL_BINDING and game["spec"] == schedule[stage][index],
                      "Original game binding/spec differs")
            failed = stage == "main" and index == 13
            if failed:
                c.require("failure" in game and game["reason"] == "exception"
                          and len(game["trace"]) == 17 and game["failed_decision"]["turn"] == 17,
                          "Original failed game prefix differs")
                source_game = ref
                moves = [*game["trace"], game["failed_decision"]]
            else:
                c.require("failure" not in game and game["reason"] in c.old.VALID_NATURAL_TERMINATIONS,
                          "Original completed game is incomplete")
                games[stage][str(index)] = ref
                moves = game["trace"]
            c.require([move["turn"] for move in moves] == list(range(len(moves))), "Original game prefix has gaps")
            for move in moves:
                turn = move["turn"]
                root_path = c.SOURCE/stage/"roots"/f"game_{index:04d}"/f"turn_{turn:03d}.json"
                expected_root_paths.add(root_path.resolve())
                root_ref = retain(root_path, index=index, turn=turn)
                c.require(root_ref["sha256"] == move["root_sha256"], "Original root/trace hash differs")
                root = c.verified(root_ref)
                c.require(root["manifest"] == c.ORIGINAL_BINDING and root["root_identity"] == [stage, index, turn]
                          and c.old.digest(root["snapshot"]) == move["state_sha256"], "Original root state/identity differs")
                is_failed_root = failed and turn == 17
                c.require(("failure" in root) == is_failed_root, "Unexpected original root failure")
                c.require(root_path.stat().st_mtime_ns >= c.verify_hash(stage_launches[stage]).stat().st_mtime_ns,
                          "Original root predates stage launch")
                roots[stage][f"{index}:{turn}"] = root_ref
                if is_failed_root:
                    failed_root = root_ref
            c.require(path.stat().st_mtime_ns >= c.verify_hash(stage_launches[stage]).stat().st_mtime_ns,
                      "Original game predates stage launch")
        c.require({path.resolve() for path in (c.SOURCE/stage/"games").glob("*.json")} == expected_paths,
                  "Unexpected original game files")
        c.require({path.resolve() for path in (c.SOURCE/stage/"roots").rglob("*.json")} == expected_root_paths,
                  "Unexpected original root files")
    c.require(source_game is not None and failed_root is not None
              and sorted(map(int, games["main"])) == RETAINED, "Wrong retained game population")
    diagnosis = retain(c.SOURCE/"recovery_diagnosis_01/reproduction.json")
    _validate_diagnosis(diagnosis, failed_root)
    return {"original_binding": c.ORIGINAL_BINDING, "original_launch_epoch": c.ORIGINAL_LAUNCH_EPOCH,
            "deadline_epoch": c.DEADLINE_EPOCH, "original_launch": launch_ref,
            "original_main_launch": main_launch_ref, "original_stage_launches": stage_launches,
            "models": c.artifact(c.SOURCE/"models_lock.json"), "games": games, "roots": roots,
            "source_game": source_game, "failed_root": failed_root, "diagnosis": diagnosis,
            "main_prefix": {"index": 13, "turns": 17, "failed_turn": 17},
            "retained_main_games": [games["main"][str(index)] for index in RETAINED],
            "pending_main_indices": [index for index in range(402) if index not in RETAINED],
            "probe": probe, "probe_coverage": coverage,
            "preserved": [preserved[path] for path in sorted(preserved)]}


def validate(value):
    """Reauthenticate the sealed starting evidence without rebuilding gameplay."""
    c.require(value["original_binding"] == c.ORIGINAL_BINDING
              and value["original_launch_epoch"] == c.ORIGINAL_LAUNCH_EPOCH
              and value["deadline_epoch"] == c.DEADLINE_EPOCH, "Inventory binding/clock differs")
    c.require(value["main_prefix"] == {"index": 13, "turns": 17, "failed_turn": 17}, "Failure lineage differs")
    c.require(sorted(map(int, value["games"]["main"])) == RETAINED
              and [ref["index"] for ref in value["retained_main_games"]] == RETAINED
              and value["pending_main_indices"] == [index for index in range(402) if index not in RETAINED],
              "Recovery retained/pending population differs")
    refs = value["preserved"]
    c.require(len({ref["path"] for ref in refs}) == len(refs), "Duplicate preserved artifact")
    lookup = {ref["path"]: ref["sha256"] for ref in refs}
    for stage in ("probe", "main"):
        for ref in [*value["games"][stage].values(), *value["roots"][stage].values()]:
            c.require(lookup.get(ref["path"]) == ref["sha256"], "Unsealed original game/root")
    for name in ("original_launch", "original_main_launch", "models", "source_game", "failed_root", "diagnosis"):
        ref = value[name]
        c.require(lookup.get(ref["path"]) == ref["sha256"], "Unsealed original lineage")
    c.require(value["failed_root"] == value["roots"]["main"]["13:17"], "Failed root mapping differs")
    for ref in refs:
        c.verify_hash(ref)
    _validate_diagnosis(value["diagnosis"], value["failed_root"])
    return True
