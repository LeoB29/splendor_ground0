"""Recovery provenance, overlay resolution and the original resource clock."""
from pathlib import Path
import shutil
import time

from tools.equal_budget_breadth_arena_128 import common as original

ROOT = original.ROOT
SOURCE = original.OUTPUT
OUTPUT = SOURCE / "recovery_01"
ORIGINAL_BINDING = "4afa5eb8b9d29cedae07c0c8eb00cb4cfbc34cf03fa1c11e97a7ae9bb6d30f7b"
ORIGINAL_LAUNCH_EPOCH = 1791028925.3389308
DEADLINE_EPOCH = ORIGINAL_LAUNCH_EPOCH + 120*3600
CONFIG, FITS = original.CONFIG, original.FITS
old, baseline, windows = original.old, original.baseline, original.windows
require, artifact, verified, verify_hash = original.require, original.artifact, original.verified, original.verify_hash
BudgetExceeded = original.BudgetExceeded
_CACHE = {}


class RecoveryAborted(RuntimeError):
    """An explicit cooperative abort retains every saved artifact."""


def _production(base):
    return Path(base).resolve().is_relative_to(SOURCE.resolve())


def budget_boundary(base=OUTPUT):
    base = Path(base)
    if (base / "abort.json").exists():
        raise RecoveryAborted("Recovery abort requested; preserve partial evidence")
    if _production(base) and time.time() > DEADLINE_EPOCH:
        raise BudgetExceeded("Original five-day execution ceiling reached; downtime is included")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve partial evidence")


def _stamp(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _load(base):
    base = Path(base).resolve()
    path, seal_path = base / "inventory.json", base / "freeze.json"
    key = str(base), _stamp(path), _stamp(seal_path)
    if key in _CACHE:
        return _CACHE[key]
    require(path.is_file(), "Missing frozen recovery inventory")
    if seal_path.exists():
        seal = old.read_json(seal_path)
        if "manifest" in seal:
            manifest = verified(seal["manifest"])
            require(old.digest(manifest) == seal["binding"], "Recovery manifest binding changed")
            require(verify_hash(manifest["inventory"]).resolve() == path, "Recovery inventory seal path changed")
        else:
            require(not _production(base), "Production recovery freeze has no manifest")
    else:
        require(not _production(base), "Production recovery inventory is not frozen")
    value = old.read_json(path)
    require(value["original_binding"] == ORIGINAL_BINDING, "Wrong original inventory binding")
    by_path = {}
    for stage in value["games"].values():
        for ref in stage.values():
            by_path[(ROOT / ref["path"]).resolve()] = ref
    for stage in value["roots"].values():
        for ref in stage.values():
            by_path[(ROOT / ref["path"]).resolve()] = ref
    if value.get("source_game"):
        ref = value["source_game"]
        by_path[(ROOT / ref["path"]).resolve()] = ref
    _CACHE[key] = value, by_path
    return _CACHE[key]


def load_inventory(base=OUTPUT):
    return _load(base)[0]


def recovery_binding(base=OUTPUT):
    seal = old.read_json(Path(base) / "freeze.json")
    require(type(seal["binding"]) is str and bool(seal["binding"]), "Missing recovery binding")
    if "manifest" in seal:
        require(old.digest(verified(seal["manifest"])) == seal["binding"], "Recovery binding changed")
    else:
        require(not _production(base), "Production recovery freeze has no manifest")
    return seal["binding"]


def _coordinates(stage, index, turn=None):
    require(stage in CONFIG["games"] and type(index) is int and 0 <= index < CONFIG["games"][stage],
            "Invalid recovery game identity")
    if turn is not None:
        require(type(turn) is int and 0 <= turn < CONFIG["max_turns"], "Invalid recovery turn")


def validate_origin(record, path, binding, base=OUTPUT):
    """Authenticate source membership or the new owned record and launch time."""
    path, base = Path(path).resolve(), Path(base).resolve()
    require(binding == ORIGINAL_BINDING and record["manifest"] == binding, "Scientific binding differs")
    inventory, originals = _load(base)
    if path in originals:
        require(verify_hash(originals[path]).resolve() == path, "Original artifact path differs")
        stage = record.get("root_identity", [None])[0]
        if stage is None:
            stage = next((s for s, rows in inventory["games"].items()
                          if any((ROOT/ref["path"]).resolve() == path for ref in rows.values())), "main")
        launch_ref = inventory["original_stage_launches"][stage]
        launch = verify_hash(launch_ref)
    else:
        require(path.is_relative_to(base), "Record outside owned recovery namespace")
        relative = path.relative_to(base)
        require(len(relative.parts) >= 3 and relative.parts[0] in CONFIG["games"]
                and relative.parts[1] in ("games", "roots"), "Not an owned game/root artifact")
        require(record.get("recovery_binding") == recovery_binding(base), "Recovery record provenance differs")
        launch = base / f"{relative.parts[0]}_launch.json"
        require(launch.is_file(), "Missing recovery stage launch")
        launched = old.read_json(launch)
        require(launched["manifest"] == binding and launched["recovery_binding"] == recovery_binding(base),
                "Recovery stage launch provenance differs")
    require(path.stat().st_mtime_ns >= launch.stat().st_mtime_ns, "Record predates its originating launch")
    return True


def resolve_game(stage, index, base=OUTPUT):
    _coordinates(stage, index)
    base = Path(base)
    inventory = load_inventory(base)
    original_ref = inventory["games"].get(stage, {}).get(str(index))
    target = base / stage / "games" / f"game_{index:04d}.json"
    if target.exists():
        require(original_ref is None, "Duplicate completed original game in recovery overlay")
        validate_origin(old.read_json(target), target, ORIGINAL_BINDING, base)
        return target
    return verify_hash(original_ref) if original_ref is not None else None


def resolve_partial_game(stage, index, base=OUTPUT):
    _coordinates(stage, index)
    inventory = load_inventory(base)
    if stage == "main" and index == inventory["main_prefix"]["index"]:
        return verify_hash(inventory["source_game"])
    return None


def resolve_root(stage, index, turn, base=OUTPUT):
    _coordinates(stage, index, turn)
    base = Path(base)
    inventory = load_inventory(base)
    original_ref = inventory["roots"].get(stage, {}).get(f"{index}:{turn}")
    target = base / stage / "roots" / f"game_{index:04d}" / f"turn_{turn:03d}.json"
    if target.exists():
        require(original_ref is None or original_ref == inventory.get("failed_root"),
                "Duplicate successful original root in recovery overlay")
        validate_origin(old.read_json(target), target, ORIGINAL_BINDING, base)
        return target
    return verify_hash(original_ref) if original_ref is not None else None


def validate_resolved_inventory(stage, expected_games, expected_roots, base=OUTPUT):
    """Require exact original-plus-overlay sets, with only named failure replaced."""
    require(stage in CONFIG["games"], "Unknown stage inventory")
    base = Path(base)
    inventory = load_inventory(base)
    actual_games = {verify_hash(ref).resolve() for ref in inventory["games"].get(stage, {}).values()}
    actual_roots = {verify_hash(ref).resolve() for ref in inventory["roots"].get(stage, {}).values()
                    if ref != inventory.get("failed_root")}
    for path in (base / stage / "games").glob("*.json"):
        record = old.read_json(path)
        index = record["spec"]["index"]
        require(resolve_game(stage, index, base).resolve() == path.resolve(), "Unexpected overlay game path")
        require("failure" not in record, "Failed overlay game in accepted inventory")
        actual_games.add(path.resolve())
    for path in (base / stage / "roots").rglob("*.json"):
        record = old.read_json(path)
        identity = record["root_identity"]
        require(identity[0] == stage and resolve_root(*identity, base=base).resolve() == path.resolve(),
                "Unexpected overlay root path")
        require("failure" not in record, "Failed overlay root in accepted inventory")
        actual_roots.add(path.resolve())
    require(actual_games == {Path(path).resolve() for path in expected_games}, "Merged game inventory differs")
    require(actual_roots == {Path(path).resolve() for path in expected_roots}, "Merged root inventory differs")
    return True
