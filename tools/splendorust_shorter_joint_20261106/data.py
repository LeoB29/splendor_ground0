"""Authenticated, outcome-blind subsets of the accepted corrected corpus.

No model is constructed here. The owning worker calls the corrected-rules
startup guard once before using this module. Whole games are decoded with the
unchanged qualified loader, one at a time; only selected numeric examples stay
resident. The original corpus and every historical file remain read-only.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from tools.splendorust_policy_head_20261104 import data as accepted
from tools.splendorust_policy_head_20261104 import training as accepted_training

ROOT = accepted.ROOT
DOMAIN = "splendorust_learning_diagnostic_20261106_v1"
SCHEMA = "splendorust_learning_diagnostic_selection_v1"
STUDY = "splendorust_shorter_joint_20261106_v1"
SHORT_SCHEMA = "splendorust_shorter_joint_selection_v1"
STEPS = 96
ORIGINAL_SELECTION = {
    "path": "outputs/splendorust_learning_diagnostic_20261106_01/selection.json",
    "bytes": 4190920,
    "sha256": "757199b51e66cde02b56128e07cbc3fb31fc741e76073adc6425eec25f3d8e56",
}
ORIGINAL_BUILDER = {
    "path": "tools/splendorust_learning_diagnostic_20261106/data.py",
    "bytes": 19814,
    "sha256": "7b8e62c6feab3b8a9bf8081e84514b9b357308c1dcf4bf761bf0d277e37da9fd",
}
POLICY_FREEZE = "outputs/splendorust_policy_head_20261104_01/freeze.json"
POLICY_FREEZE_SHA = "52641e03dda1a14cb7f4675f97a3c1b1412226679b000db7a78354a6071b81b2"
REUSE_PROOF_SHA = "20e40d20713056f7f32db752a12caddfa61756d08cb343c72f2cd0b3dd493f2a"
SOURCE_CODE = {
    "tools/splendorust_policy_head_20261104/data.py": "996d840138e510bd5c2294873149b069731d79742ceaa1e821d95e449a5c2ab5",
    "tools/splendorust_policy_head_20261104/training.py": "cd822a5bb6642c438ed24224956b8cc9cc5e9557bb4063ac8e7cb6626f007e48",
}
COUNTS = {"train": 10660, "validation": 1802, "offline": 1755}
GAME_COUNTS = {"train": 192, "validation": 32, "offline": 32}
IDENTITY = deepcopy(accepted.IDENTITY)
require, canonical, digest = accepted.require, accepted.canonical, accepted.digest


def artifact(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "diagnostic artifact stays in workspace")
    before = path.stat()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), "stable diagnostic input")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": after.st_size, "sha256": sha}


def _unique_inputs(items):
    unique = {}
    for item in items:
        require(type(item) is dict and set(item) == {"path", "bytes", "sha256"}, "complete input descriptor")
        require(unique.setdefault(item["path"], item) == item, "consistent input identity")
    return [deepcopy(unique[key]) for key in sorted(unique)]


def _source_context(boundary=lambda: None):
    """Authenticate existing acceptance and descriptors, without rereading games."""
    code = []
    for path, sha in SOURCE_CODE.items():
        boundary(); item = artifact(ROOT / path)
        require(item["sha256"] == sha, "unchanged accepted loader/collator source")
        code.append(item)
    item = artifact(ROOT / POLICY_FREEZE)
    require(item["sha256"] == POLICY_FREEZE_SHA, "exact accepted policy-study freeze")
    freeze = accepted.read_artifact(item, boundary=boundary)
    proof_item = freeze["source_reuse_proof"]
    require(proof_item["sha256"] == REUSE_PROOF_SHA, "exact accepted corpus reuse proof")
    proof = accepted.read_artifact(proof_item, boundary=boundary)
    require(proof["status"] == "accepted_corpus_reuse_verified" and
            proof["source_games_reused"] == 256 and proof["source_games_generated"] == 0 and
            proof["previous_candidate_not_used"] is True and
            proof["declaration"] == freeze["source_reuse"] and
            proof["game_counts"] == GAME_COUNTS and proof["example_counts"] == COUNTS,
            "accepted original corrected corpus and original split counts")
    documents = accepted.accepted_source_documents(proof["declaration"], boundary=boundary)
    manifest, specs = documents["corpus_manifest"], documents["source_specs"]
    accepted.validate_accepted_corpus_reuse(manifest, specs, proof["declaration"], boundary=boundary)
    accepted.validate_accepted_input_qualification(proof["input_qualification"], proof["declaration"], boundary=boundary)
    indexed = {x["path"]: x for x in proof["inputs"]}
    for descriptor in manifest["games"]:
        require(indexed.get(descriptor["path"]) == descriptor, "every original game descriptor retained")
    binding = proof["input_qualification"]
    require(indexed.get(binding["plan"]["path"]) == binding["plan"] and
            all(path in indexed for path in [binding["proof_path"], *binding["row_paths"]]),
            "original actual-loader counterexamples are authenticated")
    inputs = _unique_inputs([item, proof_item, *proof["declaration"].values(), *proof["inputs"], *code])
    receipt = {"policy_freeze": item, "source_reuse_proof": proof_item,
               "declaration": deepcopy(proof["declaration"]), "metadata": deepcopy(manifest["metadata"]),
               "input_qualification": deepcopy(binding), "inputs": inputs,
               "source_games": 256, "new_games": 0, "opened_heldout_is_exploratory": True}
    return manifest, specs, receipt


def _sha(value):
    require(type(value) is str and len(value) == 64 and set(value) <= set("0123456789abcdef"), "full lowercase SHA256")
    return value


def root_identity(example):
    """Target contents and outputs never enter the sampling key."""
    require(type(example["game_index"]) is int and type(example["ply"]) is int and
            example["game_index"] >= 0 and example["ply"] >= 0, "canonical integer game/ply identity")
    require(example["split"] in GAME_COUNTS, "original source split")
    return {"game_index": example["game_index"], "ply": example["ply"], "split": example["split"],
            "row_sha256": _sha(example["row_sha256"]),
            "root_input_sha256": _sha(example["root_input_sha256"])}


def _catalogue_from_decoded(spec, descriptor, decoded):
    return {"spec": deepcopy(spec), "artifact": deepcopy(descriptor),
            "roots": [root_identity(example) for example in decoded["examples"]]}


def _build_original_selection(games, receipt):
    """Pure deterministic selector: only identity/order fields affect choice."""
    require(type(games) is list and len(games) == 256, "exact original 256-game inventory")
    ids = [g["spec"]["game_index"] for g in games]
    require(ids == list(range(256)), "original canonical game IDs in order")
    game_counts, counts = Counter(), Counter()
    roots_by_game = {}
    for game in games:
        spec, roots = game["spec"], game["roots"]
        require(type(spec["game_index"]) is int and spec["split"] in GAME_COUNTS and
                spec["teacher_simulations"] == 512 and spec["purpose"] == "single_generation_learning_pilot",
                "original scientific source scope")
        require(type(roots) is list and roots and all(root_identity(x) == x for x in roots), "complete minimal root identities")
        require(all(x["game_index"] == spec["game_index"] and x["split"] == spec["split"] for x in roots), "root split and game binding")
        plies = [x["ply"] for x in roots]
        require(plies == sorted(set(plies)), "unique roots in original ply order")
        game_counts[spec["split"]] += 1; counts[spec["split"]] += len(roots)
        roots_by_game[spec["game_index"]] = roots
    require(dict(game_counts) == GAME_COUNTS and dict(counts) == COUNTS, "exact accepted game/root counts")
    train_ids = [g["spec"]["game_index"] for g in games if g["spec"]["split"] == "train"]
    heldout_ids = [g["spec"]["game_index"] for g in games if g["spec"]["split"] != "train"]
    ranked = sorted(train_ids, key=lambda g: (digest([DOMAIN, "game", g]), g))
    cells, repetitions = [], []
    for repetition in (0, 1):
        dense_ids = sorted(ranked[64 * repetition:64 * (repetition + 1)])
        pools, selected = [], {"dense": [], "broad": []}
        phase_counts = {"dense": [0, 0, 0], "broad": [0, 0, 0]}
        for rank, game_id in enumerate(ranked):
            roots = roots_by_game[game_id]; n = len(roots)
            thirds = [roots[:n // 3], roots[n // 3:2 * n // 3], roots[2 * n // 3:]]
            require(all(len(part) >= 16 for part in thirds), "every original game supports 48 unique roots")
            chosen = [sorted(part, key=lambda row: (digest([DOMAIN, "root", repetition, game_id, row["ply"]]), row["ply"]))[:16]
                      for part in thirds]
            pools.append({"game_index": game_id, "third_plies": [[x["ply"] for x in part] for part in chosen]})
            six_third = (rank + repetition) % 3
            for third, part in enumerate(chosen):
                if game_id in dense_ids:
                    selected["dense"].extend([game_id, x["ply"]] for x in part)
                    phase_counts["dense"][third] += len(part)
                quota = 6 if third == six_third else 5
                selected["broad"].extend([game_id, x["ply"]] for x in part[:quota])
                phase_counts["broad"][third] += quota
        repetitions.append({"repetition": repetition, "dense_game_ids": dense_ids, "dense_pools": pools})
        for breadth in ("dense", "broad"):
            references = sorted(selected[breadth])
            require(len(references) == len({tuple(x) for x in references}) == 3072 and
                    phase_counts[breadth] == [1024, 1024, 1024], "fixed unique records and matched trajectory thirds")
            game_ids = sorted({x[0] for x in references})
            require(len(game_ids) == (64 if breadth == "dense" else 192), "fixed independent game count")
            permutations = [sorted(range(3072), key=lambda i: (digest([DOMAIN, "batch_order", repetition, breadth, epoch, i]), i))
                            for epoch in range(7)]
            batches = [order[offset:offset + 32] for order in permutations for offset in range(0, 3072, 32)][:668]
            cells.append({"cell_id": f"r{repetition}_{breadth}", "repetition": repetition, "breadth": breadth,
                          "game_ids": game_ids, "root_ids": references, "phase_counts": phase_counts[breadth],
                          "permutations": permutations, "batches": batches, "probe_batches": deepcopy(batches[:4]),
                          "roots": 3072, "optimizer_steps": 668, "presentations": 21376,
                          "training_weights": "one_per_record_equal_game_quotas"})
    return {"schema": SCHEMA, "domain": DOMAIN, "canonical_encoding": "UTF8_json_sort_keys_compact_ascii_no_nan",
            "game_rank": ranked, "source_receipt": deepcopy(receipt), "inputs": deepcopy(receipt["inputs"]),
            "games": deepcopy(games), "heldout_game_ids": heldout_ids,
            "repetitions": repetitions, "cells": cells, "game_counts": dict(game_counts), "root_counts": dict(counts),
            "new_games": 0, "heldout_previously_opened": True, **IDENTITY}


def _build_selection(games, receipt):
    """Keep the original identity ranking; prospectively truncate broad fits."""
    original = _build_original_selection(games, receipt)
    result = deepcopy(original)
    cells = []
    for source in original["cells"]:
        if source["breadth"] != "broad":
            continue
        cell = deepcopy(source)
        cell.update(permutations=deepcopy(source["permutations"][:1]),
                    batches=deepcopy(source["batches"][:STEPS]), probe_batches=[],
                    optimizer_steps=STEPS, presentations=3072,
                    original_cell_sha256=digest(source),
                    schedule_relation="exact_first_96_batches_of_original_668")
        require(sorted(i for batch in cell["batches"] for i in batch) == list(range(3072)),
                "one complete pass, every selected root once")
        cells.append(cell)
    result.update(schema=SHORT_SCHEMA, study_id=STUDY, cells=cells,
                  original_selection=deepcopy(ORIGINAL_SELECTION),
                  original_builder=deepcopy(ORIGINAL_BUILDER),
                  original_selection_canonical_sha256=digest(original),
                  historical_data_relabelled=False, diagnostic_gradient_probes=0,
                  inputs=_unique_inputs([*receipt["inputs"], ORIGINAL_SELECTION, ORIGINAL_BUILDER]))
    return result


def _read_original_selection(boundary=lambda: None):
    boundary()
    require(artifact(ROOT / ORIGINAL_BUILDER["path"]) == ORIGINAL_BUILDER,
            "unchanged original sampling source")
    return accepted.read_artifact(ORIGINAL_SELECTION, boundary=boundary)


def prepare_selection(*, boundary=lambda: None):
    """Derive from authenticated saved identities; actual loader rechecks rows later."""
    manifest, specs, receipt = _source_context(boundary)
    original = _read_original_selection(boundary)
    require(original["source_receipt"] == receipt and
            [x["spec"] for x in original["games"]] == specs and
            [x["artifact"] for x in original["games"]] == manifest["games"],
            "same accepted corrected corpus and source descriptors")
    require(_build_original_selection(original["games"], receipt) == original,
            "identical original outcome-blind selector and complete batch orders")
    result = _build_selection(original["games"], receipt)
    boundary()
    return result


def validate_selection(selection, *, authenticate=True, boundary=lambda: None):
    require(type(selection) is dict, "selection manifest object")
    boundary()
    if authenticate:
        manifest, specs, receipt = _source_context(boundary)
        require(selection["source_receipt"] == receipt, "unchanged accepted input bindings")
        require([x["spec"] for x in selection["games"]] == specs and
                [x["artifact"] for x in selection["games"]] == manifest["games"], "original source descriptors and splits")
        original = _read_original_selection(boundary)
        require(_build_original_selection(selection["games"], receipt) == original,
                "exact authenticated original root inventory and schedule")
    expected = _build_selection(selection["games"], selection["source_receipt"])
    require(selection == expected, "exact prospectively frozen outcome-blind selection and orders")
    boundary()
    return selection


def source_bindings(selection=None, *, boundary=lambda: None):
    if selection is None:
        return deepcopy(_source_context(boundary)[2]["inputs"])
    validate_selection(selection, boundary=boundary)
    return deepcopy(selection["inputs"])


def _load_selected(selection, repetition, breadth, *, boundary=lambda: None):
    require(type(repetition) is int and repetition in (0, 1) and breadth == "broad", "two fixed broad cells only")
    cell = next(x for x in selection["cells"] if x["repetition"] == repetition and x["breadth"] == breadth)
    wanted = {tuple(x) for x in cell["root_ids"]}
    train, heldout = [], []
    wanted_games = set(cell["game_ids"]) | set(selection["heldout_game_ids"])
    for game in selection["games"]:
        spec = game["spec"]
        if spec["game_index"] not in wanted_games:
            continue
        boundary()
        decoded = accepted.load_game(accepted.read_artifact(game["artifact"], boundary=boundary), spec, boundary=boundary)
        require([root_identity(x) for x in decoded["examples"]] == game["roots"], "decoded roots match the frozen complete source inventory")
        for example in decoded["examples"]:
            if spec["split"] == "train":
                if (example["game_index"], example["ply"]) in wanted:
                    train.append(example)
            else:
                heldout.append(example)
        del decoded
    train.sort(key=lambda x: (x["game_index"], x["ply"]))
    heldout.sort(key=lambda x: (x["game_index"], x["ply"]))
    require([[x["game_index"], x["ply"]] for x in train] == cell["root_ids"], "actual selected loader exact ordering and no missing roots")
    counts = Counter(x["game_index"] for x in train)
    held_counts = Counter(x["game_index"] for x in heldout)
    require(len(train) == 3072 and len(heldout) == 3557 and sorted(held_counts) == selection["heldout_game_ids"] and
            set(counts.values()) == {16}, "actual fixed selected and held-out coverage")
    return {"train_examples": train, "train_weights": [1.] * len(train),
            "heldout_examples": heldout,
            "heldout_weights": [len(heldout) / (64 * held_counts[x["game_index"]]) for x in heldout],
            "train_game_ids": cell["game_ids"], "heldout_game_ids": selection["heldout_game_ids"],
            "train_counts": dict(counts), "heldout_counts": dict(held_counts),
            "batches": deepcopy(cell["batches"]), "probe_batches": deepcopy(cell["probe_batches"]),
            "selection_cell": deepcopy(cell), "source_receipt": deepcopy(selection["source_receipt"]),
            "metadata": deepcopy(selection["source_receipt"]["metadata"]),
            "source_descriptors": deepcopy(selection["inputs"])}


def load_selected(selection, repetition, breadth, *, boundary=lambda: None):
    validate_selection(selection, boundary=boundary)
    return _load_selected(selection, repetition, breadth, boundary=boundary)


def collate_inputs(examples, weights):
    """Exactly the accepted materialized Torch input collation, no model call."""
    return accepted_training.collate_inputs(examples, weights)


def collate(examples, weights):
    """Exactly the accepted official signed-value and input collation."""
    return accepted_training.collate(examples, weights)


def _write_exclusive(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n"); stream.flush(); os.fsync(stream.fileno())


def qualify_data(selection, folder, *, boundary=lambda: None):
    """Actual corrected fixture decoding, actual subset loading and collation.

The structural fixtures have no invented terminal labels. The two engineering
updates receive a distinct batch of 32 authentic, selected training examples.
"""
    validate_selection(selection, boundary=boundary)
    receipt = selection["source_receipt"]; binding = receipt["input_qualification"]
    indexed = {x["path"]: x for x in receipt["inputs"]}
    cases = accepted.read_artifact(binding["plan"], boundary=boundary)["loader_cases"]
    rows = [accepted.read_artifact(indexed[path], boundary=boundary) for path in binding["row_paths"]]
    fixture_proof = accepted.qualify_captured_rows(cases, rows, boundary=boundary)
    original_proof = accepted.read_artifact(indexed[binding["proof_path"]], boundary=boundary)
    require(original_proof["actual_loader"] == fixture_proof, "unchanged saved actual-loader fixture proof")
    decoded = [accepted.decode_row(row) for row in rows]
    eligible = [x for x in decoded if x["eligible"]]
    arrays = accepted.collate_arrays(eligible, [1.] * len(eligible))
    actual = collate_inputs(eligible, [1.] * len(eligible))
    require(all(actual[key].tolist() == value.tolist() for key, value in arrays.items()), "actual diagnostic collator preserves corrected fixture tensors/masks")
    loaded = _load_selected(selection, 0, "broad", boundary=boundary)
    indices = loaded["batches"][0]
    examples = [loaded["train_examples"][index] for index in indices]
    weights = [loaded["train_weights"][index] for index in indices]
    batch = collate(examples, weights)
    require(batch["value"].tolist() == [x["value"] for x in examples], "actual official canonical value labels in selected batch")
    rejection = []
    for name, mutate in [
        ("missing_identity", lambda x: x.pop("ruleset_id")),
        ("wrong_root_identity", lambda x: x["games"][0]["roots"][0].__setitem__("row_sha256", "0" * 64)),
        ("changed_batch_order", lambda x: x["cells"][0]["batches"][0].reverse()),
        ("heldout_in_training", lambda x: x["cells"][0]["root_ids"][0].__setitem__(0, 192)),
    ]:
        bad = deepcopy(selection); mutate(bad)
        try:
            validate_selection(bad, authenticate=False, boundary=boundary)
            # Root hashes are authenticated against actual source rows, not
            # inferred from a self-consistent selection alone.
            if name == "wrong_root_identity":
                game = bad["games"][0]
                observed = accepted.load_game(accepted.read_artifact(game["artifact"], boundary=boundary), game["spec"], boundary=boundary)
                require([root_identity(x) for x in observed["examples"]] == game["roots"], "actual source rejects changed root identity")
        except (ValueError, KeyError, TypeError):
            rejection.append({"case": name, "rejected": True})
        else:
            raise ValueError("Diagnostic selected loader accepted " + name)
    manifest = accepted.read_artifact(receipt["declaration"]["corpus_manifest"], boundary=boundary)
    specs = accepted.read_artifact(receipt["declaration"]["source_specs"], boundary=boundary)
    manifest_rejections = accepted.qualify_manifest_rejections(manifest, specs, boundary=boundary)
    proof = {"status": "actual_selected_loader_and_collator_qualified", "selection_sha256": digest(selection),
             "inputs": deepcopy(selection["inputs"]), "fixture_proof": fixture_proof,
             "fixture_rows": len(rows), "fixture_torch_arrays_sha256": digest({k: v.tolist() for k, v in actual.items()}),
             "selected_training_roots": 3072, "heldout_roots": 3557, "heldout_games": 64,
             "engineering_root_ids": [[x["game_index"], x["ply"]] for x in examples],
             "official_value_batch_sha256": digest(batch["value"].tolist()),
             "selected_rejections": rejection, "manifest_rejections": manifest_rejections,
             "network_forwards": 0, "optimizer_updates": 0, "new_games": 0,
             "terminal_labels_fabricated": False, **IDENTITY}
    _write_exclusive(Path(folder) / "data_qualification.json", proof)
    return {"proof": proof, "qualification_examples": examples, "qualification_weights": weights}
