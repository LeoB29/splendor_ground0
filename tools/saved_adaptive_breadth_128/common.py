"""Immutable inputs and provenance for one adaptive allocation development screen."""
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
OUTPUT = ROOT / "outputs/saved_adaptive_breadth_128"
SOURCE = ROOT / "outputs/proposal_breadth_128"
SOURCE_BINDING = "231bdd640ac88af2470c24af38a04e28248a7edeba8e376173df5b356fd27a74"
SOURCE_RESULT = "2592d17b5ed5f1ca2a7f463ffa27b9e803db3600b0b3f62ea177aab26e2a2570"
SOURCE_AUDIT = "f9273a1612636b4c13be6e47768a8ba05a0159f3b176aeddea5aca076faabf55"
PRIOR = ROOT / "outputs/saved_equal_budget_breadth"
PRIOR_BINDING = "8782aa6b1728438251682220d6944bcaded3f1b8df9d79a0cd3f428f6709ab7b"
PRIOR_RESULT = "28434252712f2b16df5fa5651a36f1a3e5837d2b42a67e2ccd583c6860287422"
PRIOR_AUDIT = "444af0cf17b9d1dc6ed824294e29957c1161c3e73dd194b38d8e304f46540ee0"
PRIOR_REPORT = "607931a4685e519c182384a06626d04e0c4d204271a34cc841e8b01180f72dbc"
CONFIG = {
    "partitions": 64, "partition_seed": 2140000000,
    "bootstrap_seed": 2163000000, "bootstrap_resamples": 50000,
    "bootstrap_chunk": 500, "incremental_gain": 0.0025, "anchor_gain": 0.005,
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
    paths = sorted((ROOT / "tools/saved_adaptive_breadth_128").glob("*.py"))
    paths += [ROOT / "run_saved_adaptive_breadth_128.py",
              ROOT / "docs/engineering/saved_adaptive_breadth_128.md",
              ROOT / "docs/strategy/next_after_equal_budget_breadth_128.md",
              ROOT / "docs/engineering/verification_policy.md"]
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


def collect_original_source():
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


def validate_preserved_choices(previous, current):
    """Keep the prior partition identities and U/A choices, not its old scores."""
    require(previous["spec"] == current["spec"], "Prior source identity differs")
    require(previous["canonical_support"] == current["canonical_support"], "Prior support differs")
    old_parts, new_parts = previous["partitions"], current["partitions"]
    require(len(old_parts) == len(new_parts) == CONFIG["partitions"], "Prior partition count differs")
    for index, (old, new) in enumerate(zip(old_parts, new_parts, strict=True)):
        require(old["partition"] == new["partition"] == index, "Prior partition identity differs")
        for name in ("common_order", "extension_order", "remainder_order"):
            require(old[name] == new[name], f"Prior ordering differs: {name}")
        require(old["choices"]["breadth"] == new["choices"]["uniform"]
                and old["choices"]["reference"] == new["choices"]["anchor"],
                "Prior uniform/anchor choice differs")
    return {"passed": True, "partitions": len(new_parts), "prior_choices_preserved": True,
            "prior_scores_reused": False}


def collect_source():
    """Authenticate the original closed panel and its frozen saved-screen choices."""
    source = collect_original_source()
    # Compare the original inventory before adding the saved-screen evidence.
    original_inventory = dict(source["inventory"])
    inventory = source["inventory"]

    def get(ref):
        return read(authenticate(ref, inventory))

    def fixed(name):
        p = PRIOR / name
        return get(reference(p))

    seal = fixed("freeze.json")
    require(seal["owner_authorized"] is True and seal["binding"] == PRIOR_BINDING,
            "Prior saved screen freeze differs")
    manifest = get(seal["manifest"])
    require(digest(manifest) == PRIOR_BINDING, "Prior saved screen binding differs")
    require(manifest["inputs"] == {**source, "inventory": original_inventory},
            "Prior and original source inventories differ")
    prior_config = manifest["identity"]["config"]
    for name in ("partitions", "partition_seed", "roots", "fits", "expected_cells", "cell_size"):
        require(prior_config[name] == CONFIG[name], f"Prior partition/panel configuration differs: {name}")
    for path, expected in manifest["identity"]["files"].items():
        authenticate({"path": path, "sha256": expected}, inventory)
    checks = get(manifest["checks"])
    require(checks["passed"] is True and checks["returncode"] == 0
            and checks["identity"] == manifest["identity"], "Prior preflight differs")
    for name in ("stdout", "stderr"):
        authenticate(checks[name], inventory)
    status = fixed("final_status.json")
    require(status["complete"] is True and status["all_prescribed_checks_pass"] is True
            and status["manifest"] == PRIOR_BINDING, "Prior saved screen incomplete")
    for name, expected in (("result", PRIOR_RESULT), ("audit", PRIOR_AUDIT), ("report", PRIOR_REPORT)):
        require(status[name]["sha256"] == expected, f"Prior published {name} differs")
        authenticate(status[name], inventory)
    result, audit = get(status["result"]), get(status["audit"])
    require(result["manifest"] == audit["manifest"] == PRIOR_BINDING
            and result["all_prescribed_checks_pass"] is True and audit["passed"] is True
            and audit["result"] == status["result"] and result["source_binding"] == SOURCE_BINDING,
            "Prior saved screen closure differs")
    require(audit["inherited_coverage"] == source["inherited_coverage"], "Inherited coverage differs")
    require(type(audit["source_inventory_count"]) is int
            and audit["source_inventory_count"] == len(original_inventory), "Prior source inventory count differs")
    independent = audit["independent_statistics"]
    require(independent["passed"] is True and independent["independent_bootstrap"] is True
            and independent["resampling_unit"] == "root", "Prior independent statistics incomplete")
    for field in ("mean_difference_fraction", "ci95", "gate_pass"):
        require(independent[field] == result["statistics"][field], f"Prior independent statistics differ: {field}")
    analysis = get(result["analysis"])
    require(analysis["manifest"] == PRIOR_BINDING and analysis["statistics"] == result["statistics"],
            "Prior saved statistics linkage differs")
    prior_roots, prior_receipts = analysis["roots"], audit["root_receipts"]
    require(len(prior_roots) == len(prior_receipts) == source["root_count"], "Prior panel incomplete")
    for spec, root_ref, row_ref, receipt_ref in zip(source["specs"], source["roots"],
                                                   prior_roots, prior_receipts, strict=True):
        row, receipt = get(row_ref), get(receipt_ref)
        expected_root = {k: root_ref[k] for k in ("path", "sha256")}
        require(row["manifest"] == receipt["manifest"] == PRIOR_BINDING
                and row["source"] == receipt["source"] == expected_root
                and receipt["analysis"] == row_ref and receipt["receipt"]["passed"] is True,
                "Prior per-root closure differs")
        require(row["result"]["spec"] == {k: spec[k] for k in
                ("index", "pair", "game_index", "fit", "seat", "phase")}, "Prior row identity differs")
        require(len(row["result"]["partitions"]) == CONFIG["partitions"], "Prior permutations incomplete")
        evidence = receipt["receipt"]
        require(type(evidence["index"]) is int and evidence["index"] == spec["index"]
                and type(evidence["partitions"]) is int and evidence["partitions"] == CONFIG["partitions"],
                "Prior verified panel identity differs")
        for field in ("independent_selection_and_assessment_reconstruction", "whole_world_vectors_preserved",
                      "exact_rational_comparisons", "equal_per_arm_selection_budget"):
            require(evidence[field] is True, f"Prior verification assurance missing: {field}")
        require(evidence["mean_difference_fraction"] == row["result"]["mean_difference_fraction"]
                and evidence["mean_score_difference"] == row["result"]["mean_score_difference"],
                "Prior verified statistics differ")
    source.update({"prior_binding": PRIOR_BINDING, "prior_result": status["result"],
                   "prior_audit": status["audit"], "prior_roots": prior_roots,
                   "prior_receipts": prior_receipts, "prior_choices_preserved": True,
                   "prior_scores_reused": False})
    return source
