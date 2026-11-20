"""Three-generation provenance, immutable overlay resolution and original clock."""
from pathlib import Path
import shutil
import time

from tools.equal_budget_breadth_arena_128 import common as original

ROOT, SOURCE = original.ROOT, original.OUTPUT
PREVIOUS, OUTPUT = SOURCE/"recovery_01", SOURCE/"recovery_02"
ORIGINAL_BINDING = "4afa5eb8b9d29cedae07c0c8eb00cb4cfbc34cf03fa1c11e97a7ae9bb6d30f7b"
PREVIOUS_BINDING = "3d4e16b42cd755369238ecd56766f5d0ac340f58fad61c6037bae127e33713fa"
ORIGINAL_LAUNCH_EPOCH = 1791028925.3389308
DEADLINE_EPOCH = ORIGINAL_LAUNCH_EPOCH+120*3600
CONFIG, FITS = original.CONFIG, original.FITS
old, baseline, windows = original.old, original.baseline, original.windows
require, artifact, verified, verify_hash = original.require, original.artifact, original.verified, original.verify_hash
BudgetExceeded = original.BudgetExceeded
_CACHE = {}


class RecoveryAborted(RuntimeError):
    """A cooperative abort preserves every saved artifact."""


def _production(base):
    return Path(base).resolve().is_relative_to(SOURCE.resolve())


def budget_boundary(base=OUTPUT):
    base = Path(base)
    if (base/"abort.json").exists():
        raise RecoveryAborted("Recovery abort requested; preserve partial evidence")
    if _production(base) and time.time() > DEADLINE_EPOCH:
        raise BudgetExceeded("Original five-day execution ceiling reached; downtime is included")
    if shutil.disk_usage(ROOT).free < CONFIG["disk_reserve_bytes"]:
        raise BudgetExceeded("Five-GiB disk reserve reached; preserve partial evidence")


def _stamp(path):
    return (path.stat().st_size, path.stat().st_mtime_ns) if path.exists() else None


def _load(base):
    base = Path(base).resolve()
    path, seal_path = base/"inventory.json", base/"freeze.json"
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
    require(value["original_binding"] == ORIGINAL_BINDING and value["previous_binding"] == PREVIOUS_BINDING,
            "Wrong source inventory binding")
    by_path = {(ROOT/ref["path"]).resolve(): ref for ref in value["sources"].values()}
    require(len(by_path) == len(value["sources"]), "Duplicate canonical source paths")
    _CACHE[key] = value, by_path
    return _CACHE[key]


def load_inventory(base=OUTPUT):
    return _load(base)[0]


def recovery_binding(base=OUTPUT):
    seal = old.read_json(Path(base)/"freeze.json")
    require(type(seal["binding"]) is str and bool(seal["binding"]), "Missing recovery binding")
    if "manifest" in seal:
        require(old.digest(verified(seal["manifest"])) == seal["binding"], "Recovery binding changed")
    else:
        require(not _production(base), "Production recovery freeze has no manifest")
    return seal["binding"]


def source_by_path(path, base=OUTPUT):
    """Return a hash-authenticated original/R1 game or root artifact, or None."""
    _, sources = _load(base)
    path = (ROOT/Path(path)).resolve()
    ref = sources.get(path)
    if ref is not None:
        require(verify_hash(ref).resolve() == path, "Source path differs")
    return ref


def _coordinates(stage, index, turn=None):
    require(stage in CONFIG["games"] and type(index) is int and 0 <= index < CONFIG["games"][stage],
            "Invalid recovery game identity")
    if turn is not None:
        require(type(turn) is int and 0 <= turn < CONFIG["max_turns"], "Invalid recovery turn")


def validate_origin(record, path, binding, base=OUTPUT):
    """Validate the binding and launch of the generation that wrote the record."""
    path, base = Path(path).resolve(), Path(base).resolve()
    require(binding == ORIGINAL_BINDING and record["manifest"] == binding, "Scientific binding differs")
    inventory, sources = _load(base)
    if path in sources:
        ref = sources[path]
        require(verify_hash(ref).resolve() == path, "Source artifact path differs")
        origin = inventory["origins"][ref["path"]]
        expected = None if origin["generation"] == "original" else PREVIOUS_BINDING
        require(record.get("recovery_binding") == expected, "Inherited generation binding differs")
        launch = verify_hash(origin["launch"])
        if origin["kind"] == "root":
            require(record["root_identity"] == [origin["stage"], origin["index"], origin["turn"]],
                    "Inherited root identity differs")
        else:
            require(record["spec"]["index"] == origin["index"], "Inherited game identity differs")
    else:
        require(path.is_relative_to(base), "Record outside owned recovery namespace")
        relative = path.relative_to(base)
        require(len(relative.parts) >= 3 and relative.parts[0] in CONFIG["games"]
                and relative.parts[1] in ("games", "roots"), "Not an owned game/root artifact")
        current = recovery_binding(base)
        require(record.get("recovery_binding") == current, "Recovery record provenance differs")
        launch = base/f"{relative.parts[0]}_launch.json"
        require(launch.is_file(), "Missing recovery stage launch")
        launched = old.read_json(launch)
        require(launched["manifest"] == binding and launched["recovery_binding"] == current,
                "Recovery stage launch provenance differs")
    require(path.stat().st_mtime_ns >= launch.stat().st_mtime_ns, "Record predates its originating launch")
    return True


def lineage_ancestor(record, base=OUTPUT):
    """Authenticate an immediate failed predecessor across both recovery generations."""
    if "recovery_origin" not in record:
        return None
    claimed = record["recovery_origin"]
    path = (ROOT/claimed["path"]).resolve()
    ref = source_by_path(path, base)
    require(ref is not None and claimed == {key: ref[key] for key in ("path", "sha256")},
            "Unsealed recovery ancestor")
    inherited = old.read_json(path)
    validate_origin(inherited, path, ORIGINAL_BINDING, base)
    require("failure" in inherited, "Recovery ancestor was already complete")
    if "root_identity" in record:
        require(record["root_identity"] == inherited["root_identity"], "Ancestor root identity differs")
        stage, index, turn = record["root_identity"]
        expected = load_inventory(base)["replaceable_roots"].get(stage, {}).get(f"{index}:{turn}")
    else:
        require(record["spec"] == inherited["spec"], "Ancestor game identity differs")
        expected = load_inventory(base)["partial_games"].get("main", {}).get(str(record["spec"]["index"]))
    if record.get("recovery_binding") == PREVIOUS_BINDING:
        require(inherited.get("recovery_binding") is None, "R1 ancestor must belong to original generation")
    else:
        require(record.get("recovery_binding") == recovery_binding(base) and ref == expected,
                "Recovery ancestor is not the pinned immediate predecessor")
    return path


def resolve_game(stage, index, base=OUTPUT):
    _coordinates(stage, index)
    inventory, base = load_inventory(base), Path(base)
    inherited = inventory["games"].get(stage, {}).get(str(index))
    target = base/stage/"games"/f"game_{index:04d}.json"
    if target.exists():
        require(inherited is None, "Duplicate completed inherited game in recovery overlay")
        validate_origin(old.read_json(target), target, ORIGINAL_BINDING, base)
        return target
    return verify_hash(inherited) if inherited is not None else None


def resolve_partial_game(stage, index, base=OUTPUT):
    _coordinates(stage, index)
    ref = load_inventory(base)["partial_games"].get(stage, {}).get(str(index))
    return verify_hash(ref) if ref is not None else None


def resolve_root(stage, index, turn, base=OUTPUT):
    _coordinates(stage, index, turn)
    inventory, base = load_inventory(base), Path(base)
    key = f"{index}:{turn}"
    inherited = inventory["roots"].get(stage, {}).get(key)
    target = base/stage/"roots"/f"game_{index:04d}"/f"turn_{turn:03d}.json"
    if target.exists():
        require(inherited is None or inventory["replaceable_roots"].get(stage, {}).get(key) == inherited,
                "Duplicate successful inherited root in recovery overlay")
        validate_origin(old.read_json(target), target, ORIGINAL_BINDING, base)
        return target
    return verify_hash(inherited) if inherited is not None else None


def validate_resolved_inventory(stage, expected_games, expected_roots, base=OUTPUT):
    require(stage in CONFIG["games"], "Unknown stage inventory")
    inventory, base = load_inventory(base), Path(base)
    actual_games = {verify_hash(ref).resolve() for ref in inventory["games"].get(stage, {}).values()}
    replaced = inventory["replaceable_roots"].get(stage, {})
    actual_roots = {verify_hash(ref).resolve() for key, ref in inventory["roots"].get(stage, {}).items()
                    if key not in replaced}
    for path in (base/stage/"games").glob("*.json"):
        record = old.read_json(path)
        require(resolve_game(stage, record["spec"]["index"], base).resolve() == path.resolve(),
                "Unexpected overlay game path")
        require("failure" not in record, "Failed overlay game in accepted inventory")
        actual_games.add(path.resolve())
    for path in (base/stage/"roots").rglob("*.json"):
        record = old.read_json(path)
        identity = record["root_identity"]
        require(identity[0] == stage and resolve_root(*identity, base=base).resolve() == path.resolve(),
                "Unexpected overlay root path")
        require("failure" not in record, "Failed overlay root in accepted inventory")
        actual_roots.add(path.resolve())
    require(actual_games == {Path(path).resolve() for path in expected_games}, "Merged game inventory differs")
    require(actual_roots == {Path(path).resolve() for path in expected_roots}, "Merged root inventory differs")
    return True
