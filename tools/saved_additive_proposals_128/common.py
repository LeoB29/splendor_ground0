"""Frozen provenance and contracts; this module never loads a model or simulator."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/saved_additive_proposals_128"
SOURCE = ROOT / "outputs/planner_state_transfer_128"
SOURCE_BINDING = "6e97b233924572d03afcda8a49b497509976e6fa69ef8b0df4ae9c05dd3dd1f4"
SOURCE_RESULT = "a82cf71e5910c49ca4eb0efbb66c1011981c61a3f4c98ebade0cbc8f8e6ef8d1"
SOURCE_AUDIT = "2de4f5b3266b11ebd69a46fef2881d56a67b97ef094198cccf1afb2130645c99"
CONFIG = {
    "fits": [161001, 161002, 161003], "roots": 360,
    "bootstrap_seed": 2203000000, "bootstrap_resamples": 50000,
    "bootstrap_chunk": 1000, "practical_gain": 0.005,
    "execution_cap_seconds": 3600, "disk_reserve_bytes": 5368709120,
    "selection_worlds": 128, "assessment_worlds": 128,
    "automatic_successor": False, "new_simulations": 0, "new_fits": 0,
    "full_inner_replay": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(path):
    p = (ROOT / path).resolve()
    require(p.is_relative_to(ROOT), f"Path outside workspace: {path}")
    return p


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def status(value):
    path = OUTPUT / "status.json"
    tmp = OUTPUT / f"status.{os.getpid()}.tmp"
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump({"time": now(), **value}, handle, sort_keys=True, allow_nan=False)
        os.replace(tmp, path)
    except OSError:
        return False
    return True


def reference(path):
    return {"path": relative(path), "sha256": sha(path)}


def authenticate(ref, inventory=None):
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
            and isinstance(ref.get("sha256"), str), "Malformed artifact reference")
    p = local(ref["path"])
    actual = sha(p)
    require(actual == ref["sha256"], f"Changed artifact: {ref['path']}")
    if inventory is not None:
        key = relative(p)
        require(key not in inventory or inventory[key] == actual, "Conflicting input hashes")
        inventory[key] = actual
    return p


def check_inventory(inventory):
    require(bool(inventory), "Empty frozen inventory")
    for path, expected in inventory.items():
        authenticate({"path": path, "sha256": expected})


def code_identity():
    require(sys.flags.optimize == 0, "Independent assertions require unoptimized Python")
    paths = sorted((ROOT / "tools/saved_additive_proposals_128").glob("*.py"))
    paths += [ROOT / "run_saved_additive_proposals_128.py",
              ROOT / "docs/engineering/saved_additive_proposals_128.md"]
    require(all(p.is_file() for p in paths), "Missing study dependency")
    return {"files": {relative(p): sha(p) for p in paths},
            "python": sys.version, "optimize": sys.flags.optimize, "numpy": np.__version__, "config": CONFIG}


def validate_record(record, spec):
    """Validate sealed identities and separation before computing new choices."""
    require(record["manifest"] == SOURCE_BINDING and record["stage"] == "main", "Wrong source binding/stage")
    require(record["spec"] == spec, "Changed source specification")
    require(digest(record["snapshot"]) == spec["snapshot_sha256"], "Changed snapshot")
    require(record["privacy"] == spec["privacy"], "Changed privacy metadata")
    keys, actions = record["union"]["keys"], record["union"]["actions"]
    require(len(keys) == len(actions) == len(set(keys)) and 1 <= len(keys) <= 10, "Invalid original union")
    g = spec["canonical_support"]
    require(type(g) is int and g > 0, "Invalid canonical support")
    require(set(record["candidates"]) == {"anchor", "broad", "narrow", "teacher"}, "Missing source arm")
    for name, arm in record["candidates"].items():
        ids = arm["indices"]
        require(len(ids) == len(set(ids)) == min(g, 8 if name == "teacher" else 4), "Invalid arm support")
        require(all(type(i) is int and 0 <= i < len(keys) for i in ids), "Candidate outside union")
        require([keys[i] for i in ids] == arm["keys"] and [actions[i] for i in ids] == arm["actions"], "Candidate identity mismatch")
    anchor = record["candidates"]["anchor"]["indices"]
    require(all(record["candidates"][a]["indices"][:3] == anchor[:3] for a in ("broad", "narrow", "teacher")), "Changed common prefix")
    require(record["candidates"]["teacher"]["indices"][:len(anchor)] == anchor, "Teacher lost anchor prefix")
    identity = {k: record[k] for k in ("spec", "snapshot", "privacy", "candidates", "union", "proposed_indices", "teacher_added_indices")}
    identity["proposals"] = {a: {k: v for k, v in p.items() if k != "timing"} for a, p in record["proposals"].items()}
    lock = record["selection_lock"]
    require(lock["manifest"] == SOURCE_BINDING and lock["stage"] == "main" and lock["index"] == spec["index"], "Changed selection lock identity")
    require(lock["identity_digest"] == digest(identity), "Changed proposal identity seal")
    require(lock["selection_sha256"] == digest(record["selection"]), "Changed selection seal")
    # This closed source has no coupling proofs; unsupported proofs hold rather than being ignored.
    require(lock["proofs_sha256"] == digest([]), "Unexpected coupling proof; fail closed")
    require(record["assessment_started_at_ns"] > lock["created_at_ns"], "Assessment preceded choice lock")
    require(lock["value_scale"] == "terminal_wdl", "Changed outcome scale")
    require(set(lock["choices"]) == set(record["candidates"]), "Incomplete original choices")
    for name, choice in lock["choices"].items():
        require(choice in record["candidates"][name]["indices"], "Original choice outside arm")
    require(lock["assessment_indices"] == sorted(set(lock["choices"].values())), "Sparse assessment differs from locked choices")
    totals = Counter()
    for name, offset in (("selection", 0), ("assessment", 256)):
        block = record[name]
        ids = list(range(len(keys))) if name == "selection" else lock["assessment_indices"]
        seeds = list(range(2181000000 + spec["index"] * 512 + offset, 2181000000 + spec["index"] * 512 + offset + 128))
        require(block["candidate_indices"] == ids and block["world_seeds"] == seeds, "Changed world namespace/support")
        require(len(block["worlds"]) == 128, "Incomplete world block")
        for seed, world in zip(seeds, block["worlds"]):
            require(world["seed"] == seed and len(world["outcomes"]) == len(ids), "World identity mismatch")
            for out in world["outcomes"]:
                value = out["wdl"]
                if value is None:
                    require(out["reason"] == "repetition_cutoff" and out["winner"] is None, "Invalid unresolved outcome")
                    totals[f"unresolved_{name}_continuations"] += 1
                else:
                    require(type(value) in (int, float) and value in (-1, 0, 1)
                            and out["reason"] in ("completed", "forced_pass_deadlock"), "Invalid terminal outcome")
                    expected_wdl = 0 if out["winner"] is None else (1 if out["winner"] == spec["seat"] else -1)
                    require(out["winner"] in (None, 0, 1) and value == expected_wdl, "Winner/WDL mismatch")
                totals[f"recorded_{name}_continuations"] += 1
    return dict(totals)


def validate_panel(specs):
    require(len(specs) == 360 and [s["index"] for s in specs] == list(range(360)), "Incomplete panel")
    require(len({s["game_index"] for s in specs}) == 360, "Repeated source game")
    pairs = defaultdict(list)
    for s in specs:
        pairs[s["pair"]].append(s)
    require(len(pairs) == 180, "Expected 180 source pairs")
    counts = Counter()
    for pair, rows in pairs.items():
        require(len(rows) == 2 and {r["seat"] for r in rows} == {0, 1}
                and {r["game_index"] for r in rows} == {pair * 2, pair * 2 + 1}, "Broken source pair")
        cells = {(r["fit"], r["phase"], r["source_arm"]) for r in rows}
        require(len(cells) == 1, "Pair crosses strata")
        counts[next(iter(cells))] += 1
    expected = {(f, p, a) for f in CONFIG["fits"] for p in range(3) for a in ("candidate", "reference")}
    require(set(counts) == expected and set(counts.values()) == {10}, "Unbalanced 18-cell panel")


def collect_source():
    """Authenticate closed source audits; do not reconstruct historical trajectories."""
    inventory = {}
    def get(ref):
        return read(authenticate(ref, inventory))
    def fixed(name):
        return get(reference(SOURCE / name))
    seal, state = fixed("freeze.json"), fixed("final_status.json")
    manifest = get(seal["manifest"])
    require(seal["binding"] == digest(manifest) == SOURCE_BINDING, "Unexpected source manifest")
    require(state["manifest"] == SOURCE_BINDING and state["complete"] is True
            and state["all_prescribed_audits_pass"] is True, "Source is not complete")
    require(state["result"]["sha256"] == SOURCE_RESULT and state["audit"]["sha256"] == SOURCE_AUDIT, "Unexpected source result/audit")
    result, audit = get(state["result"]), get(state["audit"])
    authenticate(state["report"], inventory)
    require(audit["passed"] is True and audit["result"] == state["result"]
            and audit["manifest"] == result["manifest"] == SOURCE_BINDING, "Source audit closure failed")
    require(result["all_prescribed_audits_pass"] is True, "Source checks incomplete")
    source = get(manifest["source"])
    schedule = get(manifest["schedule"])
    require(result["source"] == manifest["source"] and schedule == source["schedule"], "Source schedule seal mismatch")
    require(source["selection_uses_outcomes"] is False and source["new_source_games"] == 0, "Source selection changed")
    models = get(result["models"])
    require(models["manifest"] == SOURCE_BINDING and models["training"] is False
            and models["specialist_reselection"] is False and len(models["models"]) == 9, "Model registry changed")
    for ref in [*models["models"].values(), models["continuation_model"], *models["evidence"]]:
        authenticate(ref, inventory)
    main = get(audit["stages"]["main"])
    require(main["all_prescribed_audits_pass"] is True and main["manifest"] == SOURCE_BINDING
            and main["coverage"]["states"] == 360, "Incomplete main closure")
    require(main["coverage"] == result["coverage"] == audit["coverage"], "Coverage mismatch")
    collection, receipts = get(main["collection"]), get(main["audits"])
    require(collection["manifest"] == receipts["manifest"] == SOURCE_BINDING, "Main lock binding changed")
    roots, audits, specs = collection["files"], receipts["files"], schedule["main"]
    validate_panel(specs)
    require([r["index"] for r in roots] == [r["index"] for r in audits] == list(range(360)), "Changed root ordering")
    require(len(result["rows"]) == 360, "Missing closed result rows")
    totals = Counter()
    for ref, auditref, spec, closed_row in zip(roots, audits, specs, result["rows"]):
        record, receipt = get(ref), get(auditref)
        totals.update(validate_record(record, spec))
        require(receipt["passed"] is True and receipt["manifest"] == SOURCE_BINDING
                and receipt["spec"] == spec and receipt["stage"] == "main"
                and receipt["root"] == {k: ref[k] for k in ("path", "sha256")}, "Root audit mismatch")
        require(receipt["row"] == closed_row, "Closed source row mismatch")
        require(receipt["selection_artifact"] == record["selection_artifact"], "Selection audit link changed")
        selected = get(record["selection_artifact"])
        for key in ("selection", "selection_lock"):
            require(selected[key] == record[key], "Selection artifact mismatch")
        require(selected["equivalence_proofs"] == [] and digest(selected["identity"]) == record["selection_lock"]["identity_digest"], "Selection identity/proof mismatch")
        authenticate(receipt["sample_artifact"], inventory)
        for key in ("source_game", "source_root", "source_audit"):
            authenticate(spec[key], inventory)
    for key, value in totals.items():
        require(main["coverage"][key] == value, f"Matrix coverage differs: {key}")
    for path, value in manifest["identity"]["dependencies"].items():
        if path.startswith("tools/planner_state_transfer_128/") or path == "run_planner_state_transfer_128.py":
            authenticate({"path": path, "sha256": value}, inventory)
    return {"source_binding": SOURCE_BINDING, "inventory": inventory, "roots": roots,
            "audits": audits, "specs": specs, "root_count": 360, "pair_count": 180,
            "matrix_totals": dict(totals), "inherited_coverage": main["coverage"],
            "source_result": state["result"], "source_audit": state["audit"],
            "source_statistics": result["statistics"], "new_trajectory_replays": 0}
