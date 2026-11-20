"""Immutable inputs and provenance for an arithmetic-only development screen."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/saved_equal_budget_breadth"
SOURCE = ROOT / "outputs/proposal_breadth_128"
SOURCE_BINDING = "231bdd640ac88af2470c24af38a04e28248a7edeba8e376173df5b356fd27a74"
SOURCE_RESULT = "2592d17b5ed5f1ca2a7f463ffa27b9e803db3600b0b3f62ea177aab26e2a2570"
SOURCE_AUDIT = "f9273a1612636b4c13be6e47768a8ba05a0159f3b176aeddea5aca076faabf55"
CONFIG = {
    "partitions": 64, "partition_seed": 2140000000,
    "bootstrap_seed": 2143000000, "bootstrap_resamples": 50000,
    "bootstrap_chunk": 500, "practical_gain": 0.005,
    "expected_cells": 18, "cell_size": 20,
    "fits": [161001, 161002, 161003], "roots": 360,
    "execution_cap_seconds": 3600, "disk_reserve_bytes": 5368709120,
    "score": "(saved_wdl+1)/2", "automatic_successor": False,
    "full_inner_replay": False, "new_simulations": 0, "new_fits": 0,
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
            handle.write("\n")
        os.replace(tmp, path)
    except OSError:
        # Status is advisory; Windows readers must not invalidate scientific artifacts.
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
    paths = sorted((ROOT / "tools/saved_equal_budget_breadth").glob("*.py"))
    paths += [ROOT / "run_saved_equal_budget_breadth.py",
              ROOT / "docs/engineering/saved_equal_budget_breadth.md",
              ROOT / "docs/strategy/next_after_proposal_specialist_128.md"]
    require(all(p.is_file() for p in paths), "Missing study dependency")
    return {"files": {relative(p): sha(p) for p in paths},
            "python": sys.version, "numpy": np.__version__, "config": CONFIG}


def validate_record(record, spec):
    """Check saved matrix identities without choosing any new experimental move."""
    require(record["manifest"] == SOURCE_BINDING and record["stage"] == "main", "Wrong source study/stage")
    require(record["spec"] == spec, "Changed source selection")
    require(digest(record["snapshot"]) == spec["snapshot_sha256"], "Snapshot digest mismatch")
    g = spec["canonical_support"]
    require(type(g) is int and g >= 1, "Invalid canonical support")
    m, n = min(g, 4), min(g, 8)
    e = 128 * n // m - 128
    expected_budget = {"canonical_support": g, "anchor_candidates": m,
                       "breadth_candidates": n, "anchor_worlds": 128,
                       "breadth_worlds": 128, "precision_worlds": 128 + e,
                       "extension_worlds": e, "arm_continuations": {
                           "anchor": 128 * m, "breadth": 128 * n, "precision": 128 * n}}
    require(record["budgets"] == expected_budget, "Changed original budget")
    keys, actions = record["union"]["keys"], record["union"]["actions"]
    require(len(keys) == len(actions) == len(set(keys)) == n, "Changed candidate union")
    for arm in ("anchor", "breadth", "precision"):
        k = n if arm == "breadth" else m
        c = record["candidates"][arm]
        require(c["indices"] == list(range(k)) and c["keys"] == keys[:k]
                and c["actions"] == actions[:k], "Original candidate prefix changed")
    seen = set()
    for block, count, support, offset in (("selection", 128, n, 0), ("extension", e, m, 128)):
        data = record[block]
        seeds = [2101000000 + spec["index"] * 1024 + offset + j for j in range(count)]
        require(data["candidate_indices"] == list(range(support)), "Matrix candidate order changed")
        require(data["world_seeds"] == seeds and len(data["worlds"]) == count, "Changed world namespace/count")
        for world, seed in zip(data["worlds"], seeds):
            require(world["seed"] == seed and seed not in seen, "Duplicate/misaligned world")
            seen.add(seed)
            require(len(world["outcomes"]) == support, "Incomplete saved world vector")
            for item in world["outcomes"]:
                require(type(item["wdl"]) in (int, float) and item["wdl"] in (-1, 0, 1), "Invalid saved WDL")
                require(type(item["turns"]) is int and item["turns"] >= 0, "Invalid trajectory length")
    require(digest(record["selection"]) == record["selection_lock"]["selection_sha256"]
            and digest(record["extension"]) == record["selection_lock"]["extension_sha256"], "Broken original selection seal")
    return {"common_worlds": 128, "extension_worlds": e, "selection_cells": 128 * n + e * m}


def collect_source():
    """Authenticate the selected evidence, not a recursive historical re-audit."""
    inventory = {}

    def get(ref):
        return read(authenticate(ref, inventory))

    def fixed(name, expected=None):
        p = SOURCE / name
        return get({"path": relative(p), "sha256": expected or sha(p)})

    frozen = fixed("freeze.json")
    require(frozen["binding"] == SOURCE_BINDING and frozen["owner_authorized"] is True, "Source not frozen/authorized")
    manifest = get(frozen["manifest"])
    require(digest(manifest) == SOURCE_BINDING, "Source manifest binding changed")
    state = fixed("final_status.json")
    require(state["complete"] is True and state["all_prescribed_audits_pass"] is True
            and state["manifest"] == SOURCE_BINDING, "Source incomplete")
    require(state["result"]["sha256"] == SOURCE_RESULT and state["audit"]["sha256"] == SOURCE_AUDIT,
            "Published source result changed")
    result, audit = get(state["result"]), get(state["audit"])
    authenticate(state["report"], inventory)
    require(result["manifest"] == audit["manifest"] == SOURCE_BINDING and audit["passed"] is True
            and audit["result"] == state["result"] and result["all_prescribed_audits_pass"] is True,
            "Source result/audit linkage failed")
    require(result["models"] == audit["models"], "Source model receipt changed")
    models = get(result["models"])
    require(set(models["models"]) == {f"anchor_{f}" for f in CONFIG["fits"]}, "Changed fit set")
    for ref in [*models["models"].values(), models["continuation_model"]]:
        authenticate(ref, inventory)
    seal = get(result["selection"])
    selection = get(seal["selection"])
    require(seal["manifest"] == selection["manifest"] == SOURCE_BINDING, "Source selection binding failed")
    main = get(audit["stages"]["main"])
    require(main["manifest"] == SOURCE_BINDING and main["states"] == 360
            and main["all_prescribed_audits_pass"] is True, "Main closure incomplete")
    collection, receipts = get(main["collection"]), get(main["audits"])
    require(collection["manifest"] == receipts["manifest"] == SOURCE_BINDING, "Main lock binding failed")
    rootrefs, auditrefs = collection["files"], receipts["files"]
    specs = selection["main"]
    require(len(rootrefs) == len(auditrefs) == len(specs) == 360, "Incomplete root panel")
    expected = list(range(360))
    require([x["index"] for x in specs] == [x["index"] for x in rootrefs]
            == [x["index"] for x in auditrefs] == expected, "Wrong root ordering/duplicates")
    cells = Counter((s["fit"], s["seat"], s["phase"]) for s in specs)
    expected_cells = {(f, s, p) for f in CONFIG["fits"] for s in (0, 1) for p in (0, 1, 2)}
    require(set(cells) == expected_cells and set(cells.values()) == {20}, "Changed 18-cell panel")
    require(len({s["game_index"] for s in specs}) == 360, "Repeated source game")
    sample_seal = fixed("main/audit_sample_seal.json")
    require(sample_seal["selection"] == seal["selection"] and sample_seal["manifest"] == SOURCE_BINDING,
            "Audit sample selection changed")
    sample = get(sample_seal["sample"])
    totals = Counter()
    for ref, auditref, spec in zip(rootrefs, auditrefs, specs):
        record, receipt = get(ref), get(auditref)
        counts = validate_record(record, spec)
        totals.update(counts)
        require(receipt["passed"] is True and receipt["manifest"] == SOURCE_BINDING
                and receipt["stage"] == "main" and receipt["spec"] == spec
                and receipt["root"] == {k: ref[k] for k in ("path", "sha256")}, "Root audit mismatch")
        require(receipt["sample_artifact"] == sample_seal["sample"]
                and receipt["selection_artifact"] == record["selection_artifact"], "Original audit links changed")
        selected = get(record["selection_artifact"])
        for k in ("selection", "extension", "selection_lock"):
            require(selected[k] == record[k], f"Original selection record differs: {k}")
        require(digest(selected["identity"]) == record["selection_lock"]["identity_digest"],
                "Original proposal identity seal differs")
        game, game_audit = get(spec["source_game"]), get(spec["source_audit"])
        require(game_audit["passed"] is True and game_audit["game"] == spec["source_game"]
                and game_audit["manifest"] == game["manifest"] == SOURCE_BINDING
                and game["spec"] == game_audit["spec"], "Source game receipt invalid")
        require(game["spec"]["index"] == spec["game_index"] and game["spec"]["fit"] == spec["fit"]
                and game["model_sha256"] == models["models"][f"anchor_{spec['fit']}"]["sha256"], "Source game identity changed")
        moves = [row for row in game["trace"] if row["turn"] == spec["turn"]]
        require(len(moves) == 1 and moves[0]["snapshot"] == record["snapshot"]
                and moves[0]["state_sha256"] == spec["snapshot_sha256"]
                and moves[0]["actor"] == spec["seat"] and moves[0]["phase"] == spec["phase"], "Selected source position changed")
    # Only direct simulation provenance is rechecked; historical learning chains remain sealed.
    for path in ("tools/proposal_breadth_128/engine.py", "tools/proposal_breadth_128/common.py",
                 "tools/proposal_breadth_128/source.py", "tools/proposal_breadth_128/verify.py"):
        authenticate({"path": path, "sha256": manifest["identity"]["dependencies"][path]}, inventory)
    return {"source_binding": SOURCE_BINDING, "inventory": inventory, "roots": rootrefs,
            "specs": specs, "root_count": 360, "cell_count": 18,
            "matrix_totals": dict(totals), "inherited_coverage": main["coverage"],
            "inherited_full_inner_replay": False, "new_trajectory_replays": 0,
            "source_audit": state["audit"], "source_result": state["result"]}
