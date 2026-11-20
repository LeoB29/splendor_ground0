"""Outcome-blind, source-bound selection of saved breadth-arena positions.

This module reads the existing authenticated metadata inventory, never the saved
decision outcomes or timings. It neither writes artifacts nor starts a player.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "outputs/proposal_breadth_arena_128"
BINDING = "a8395f535998379a2062ebce8fed79cc6a26e5cd1deb699c96bd4b9dca06deb5"
FITS = (161001, 161002, 161003)
CELLS = tuple((fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in (0, 1, 2))
SELECTION_TAG = "local-player-qualification-128-root-selection-v1"
SOURCE_HASHES = {
    "main/audit_sample.json": "b6b278e5e2fcfe1c8d1f33fd1e1d0acef85ae8b8d3fd86cc8691f512116b81ba",
    "main_complete.json": "534d7f91135e400799cc2d41686a4ab6dfb114a36d92879f84e304d029612650",
    "main_audits_lock.json": "8878e0e32492c6ff0fb8f38d6a43cc1e47ab0357884ccfb0a0b7e34c6e4538aa",
    "main_games_lock.json": "c7684cbceb0d0c5f4b17ab2ffc0ccefa797bb4e3c09028bdd5c830176093bd71",
    "models_lock.json": "210b3e0667df528b909d4410000b98a86c452b9ac8e616ad153b71f2291c049c",
    "final_audit.json": "e82b556f009fc6a70a01d101eeb98d2f09924414c2aea205269e1694b03fc591",
}
METADATA_FIELDS = {
    "root_id", "index", "pair", "fit", "turn", "arm", "actor", "phase",
    "privacy", "privacy_category", "support", "canonical_support", "worlds", "sha256",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _metadata(row):
    require(isinstance(row, dict) and set(row) == METADATA_FIELDS, "Unexpected selection metadata fields")
    require(all(type(row[key]) is int for key in ("index", "pair", "fit", "turn", "actor", "phase",
                                                "privacy_category", "support", "canonical_support", "worlds")),
            "Invalid metadata integer")
    require(0 <= row["index"] < 402 and row["pair"] == row["index"] // 2 and row["fit"] in FITS
            and 0 <= row["turn"] < 400 and row["actor"] in (0, 1), "Invalid saved root identity")
    require(row["root_id"] == f'{row["index"]:04d}:{row["turn"]:03d}'
            and row["phase"] == (0 if row["turn"] < 20 else 1 if row["turn"] < 40 else 2),
            "Inconsistent metadata identity/phase")
    require(row["arm"] in ("candidate", "reference")
            and (row["arm"] == "candidate") == (row["actor"] == row["index"] % 2), "Invalid candidate seat")
    require(isinstance(row["privacy"], dict) and set(row["privacy"]) == {"own", "opponent"}
            and all(type(v) is int and 0 <= v <= 3 for v in row["privacy"].values()), "Invalid privacy counts")
    require(row["privacy_category"] == int(bool(row["privacy"]["own"])) + 2 * int(bool(row["privacy"]["opponent"])),
            "Inconsistent privacy category")
    require(row["worlds"] == 128 and row["canonical_support"] >= 1
            and row["support"] == min(row["canonical_support"], 8 if row["arm"] == "candidate" else 4),
            "Invalid support metadata")
    require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["sha256"]), "Invalid root hash")
    return deepcopy(row)


def _priority(row, stratum):
    return digest([SELECTION_TAG, stratum, row["root_id"]]), row["root_id"]


def _selected(row, reason):
    return {
        **deepcopy(row), "reason": reason,
        "path": f'{SOURCE}/main/roots/game_{row["index"]:04d}/turn_{row["turn"]:03d}.json',
        "selection_sha256": _priority(row, reason)[0],
        "context": {"fit": row["fit"], "identity": ["main", row["index"], row["turn"]],
                    "stream": digest(["proposal-breadth-arena-128-stream-v1", "main", row["pair"]])},
    }


def select_inventory(inventory):
    """Pure selection from metadata only; canonical order does not depend on input order."""
    require(isinstance(inventory, list) and inventory, "Empty metadata inventory")
    rows = [_metadata(row) for row in inventory]
    require(len({r["root_id"] for r in rows}) == len(rows), "Duplicate metadata root")
    candidates = [r for r in rows if r["arm"] == "candidate"]
    core, used, core_counts = [], set(), []
    for cell in CELLS:
        options = [r for r in candidates if (r["fit"], r["actor"], r["phase"]) == cell]
        require(bool(options), f"No candidate-side saved root for cell {cell}")
        core_counts.append({"cell": list(cell), "eligible": len(options)})
        reason = ["core", *cell]
        chosen = min(options, key=lambda r: _priority(r, reason))
        core.append(_selected(chosen, reason))
        used.add(chosen["root_id"])
    predicates = (
        ("own_private", lambda r: r["privacy"]["own"] > 0),
        ("opponent_private", lambda r: r["privacy"]["opponent"] > 0),
        ("both_private", lambda r: min(r["privacy"].values()) > 0),
        ("canonical_support_1_to_3", lambda r: r["canonical_support"] <= 3),
        ("canonical_support_4_to_7", lambda r: 4 <= r["canonical_support"] <= 7),
    )
    edges, absent, edge_counts = [], [], {}
    for name, predicate in predicates:
        options = [r for r in candidates if r["root_id"] not in used and predicate(r)]
        edge_counts[name] = {"eligible": sum(bool(predicate(r)) for r in candidates), "available_unselected": len(options)}
        if not options:
            absent.append(name)
            continue
        reason = ["edge", name]
        chosen = min(options, key=lambda r: _priority(r, reason))
        edges.append(_selected(chosen, reason))
        used.add(chosen["root_id"])
    return {"selection_tag": SELECTION_TAG, "cells": [list(c) for c in CELLS],
            "core": core, "edges": edges, "absent_edge_strata": absent,
            "core_eligible_counts": core_counts, "edge_eligible_counts": edge_counts,
            "synthetic_edges": ["ordered_first_maximum_tie"],
            "inventory_roots": len(rows), "candidate_roots": len(candidates),
            "selection_uses": "authenticated identity, seat, phase, privacy and support metadata only",
            "selection_reads_outcomes_or_timings": False}


def select_saved_roots(repo_root=ROOT):
    """Return a compact manifest; root records remain unopened and are not copied."""
    repo_root = Path(repo_root)
    sources = []
    for relative, expected in SOURCE_HASHES.items():
        path = repo_root / SOURCE / relative
        require(file_hash(path) == expected, f"Changed source artifact: {relative}")
        sources.append({"path": f"{SOURCE}/{relative}", "sha256": expected})
    complete = json.loads((repo_root / SOURCE / "main_complete.json").read_text(encoding="utf-8"))
    require(complete["manifest"] == BINDING and complete["all_prescribed_audits_pass"] is True
            and complete["games"] == 402, "Source completion binding mismatch")
    for field, relative in (("sample", "main/audit_sample.json"), ("game_lock", "main_games_lock.json"),
                            ("audit_lock", "main_audits_lock.json")):
        require(complete[field] == {"path": f"{SOURCE}/{relative}", "sha256": SOURCE_HASHES[relative]},
                "Source completion reference mismatch")
    sample = json.loads((repo_root / SOURCE / "main/audit_sample.json").read_text(encoding="utf-8"))
    require(sample["manifest"] == BINDING, "Source inventory binding mismatch")
    plan = select_inventory(sample["inventory"])
    require(plan["inventory_roots"] == 23900, "Unexpected source inventory size")
    return {"schema": "local_player_qualification_root_manifest_v1", "source_binding": BINDING,
            "source_artifacts": sources, **plan}
