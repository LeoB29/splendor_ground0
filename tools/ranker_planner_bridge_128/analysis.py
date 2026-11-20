"""Pure saved-world calculations for the post hoc 64-world planner bridge.

This module performs no I/O, inference, training or simulation.  Its caller must
authenticate the frozen prediction, source and audit references before supplying
records.  Only the recorded selection worlds are used; independent assessment
records are deliberately never accessed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np


MODELS = ("ridge", "hgb")
PLANNERS = ("anchor", *MODELS)
SOURCE_ARMS = ("anchor", "raw", "relations", "teacher")
FITS = (161001, 161002, 161003)
CELLS = tuple((fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in (0, 1, 2))
STATES = 720
PER_CELL = 40
WORLDS = 128
HALF_WORLDS = 64
SCORE_DENOMINATOR = 256
BOOTSTRAP_SAMPLES = 50000
BOOTSTRAP_SEED = 2813000000
BOOTSTRAP_CHUNK = 1000
IDENTITY_FIELDS = ("index", "game_index", "fit", "seat", "phase", "snapshot_sha256")
NATURAL_TERMINATIONS = ("completed", "forced_pass_deadlock")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _mapping(value, message):
    require(isinstance(value, dict), message)
    return value


def _keys(value, message, *, empty=False):
    require(isinstance(value, list) and (empty or bool(value))
            and all(isinstance(key, str) and bool(key) for key in value), message)
    require(len(value) == len(set(value)), message)
    return value


def _identity(row):
    _mapping(row, "Missing root identity")
    require(all(field in row for field in IDENTITY_FIELDS), "Incomplete root identity")
    require(all(type(row[field]) is int for field in IDENTITY_FIELDS[:-1]),
            "Root identity fields must be integers")
    require(0 <= row["index"] < STATES and row["game_index"] >= 0
            and (row["fit"], row["seat"], row["phase"]) in CELLS,
            "Root identity outside the fixed panel")
    digest = row["snapshot_sha256"]
    require(isinstance(digest, str) and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "Invalid snapshot digest")
    return tuple(row[field] for field in IDENTITY_FIELDS)


def _sealed_choices(prediction):
    _identity(prediction)
    outsiders = _keys(prediction.get("outsider_keys"), "Invalid sealed outsider support", empty=True)
    require(len(outsiders) <= 5, "Sealed outsider support exceeds five")
    scores = _mapping(prediction.get("scores"), "Missing sealed model scores")
    choices = _mapping(prediction.get("choices"), "Missing sealed model choices")
    require(set(scores) == set(choices) == set(MODELS), "Both frozen rankers are required")
    for name in MODELS:
        values = scores[name]
        require(isinstance(values, list) and len(values) == len(outsiders)
                and all(type(value) in (int, float) and math.isfinite(value) for value in values),
                "Invalid sealed score vector")
        expected = outsiders[max(range(len(outsiders)), key=values.__getitem__)] if outsiders else None
        require(choices[name] == expected, "Sealed choice is not the ordered first maximum")
    require("anchor_key" in prediction
            and (prediction["anchor_key"] in outsiders if outsiders else prediction["anchor_key"] is None),
            "Invalid sealed anchor choice")
    return outsiders


def _source_candidates(root):
    union = _mapping(root.get("union"), "Missing recorded union")
    keys = _keys(union.get("keys"), "Invalid union keys")
    require(1 <= len(keys) <= 10, "Union support outside the original limit")
    actions = union.get("actions")
    require(isinstance(actions, list) and len(actions) == len(keys), "Incomplete union representatives")
    recorded = _mapping(root.get("candidates"), "Missing recorded candidate sets")
    require(set(recorded) == set(SOURCE_ARMS), "Incomplete original candidate arms")
    for name in SOURCE_ARMS:
        candidate = _mapping(recorded[name], "Invalid original candidate set")
        offered = _keys(candidate.get("keys"), "Invalid candidate keys")
        indices, concrete = candidate.get("indices"), candidate.get("actions")
        require(isinstance(indices, list) and len(indices) == len(offered)
                and all(type(index) is int and 0 <= index < len(keys) for index in indices)
                and len(indices) == len(set(indices)), "Invalid candidate indices")
        require([keys[index] for index in indices] == offered, "Candidate keys/indices mismatch")
        require(isinstance(concrete, list) and concrete == [actions[index] for index in indices],
                "Candidate concrete representatives mismatch")
    proposals = _mapping(root.get("proposals"), "Missing original proposals")
    anchor = _mapping(proposals.get("anchor"), "Missing original anchor proposal")
    legal_keys = _keys(anchor.get("group_keys"), "Invalid original anchor canonical support")
    values = anchor.get("learned_group_scores")
    require(isinstance(values, list) and len(values) == len(legal_keys)
            and all(type(value) in (int, float) and math.isfinite(value) for value in values),
            "Invalid original anchor scores")
    require(set(keys) <= set(legal_keys), "Union action absent from canonical legal support")
    first = recorded["anchor"]["keys"][:3]
    for name in SOURCE_ARMS:
        offered = recorded[name]["keys"]
        limit = 8 if name == "teacher" else 4
        require(len(offered) == min(limit, len(legal_keys)) and offered[:3] == first,
                "Original candidate support/common prefix changed")
    expected_union = list(first)
    for name in SOURCE_ARMS:
        expected_union.extend(key for key in recorded[name]["keys"] if key not in expected_union)
    require(expected_union == keys, "Recorded union ordering changed")
    return keys, recorded, legal_keys, values, first


def _half_totals(root, support):
    """Validate and sum only the two saved selection-world halves."""
    block = _mapping(root.get("selection"), "Missing selection-world block")
    require(block.get("candidate_indices") == list(range(support)), "Selection omitted/reordered candidates")
    seeds, worlds = block.get("world_seeds"), block.get("worlds")
    require(isinstance(seeds, list) and len(seeds) == WORLDS
            and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(set(seeds)) == WORLDS, "Invalid selection-world seed schedule")
    require(isinstance(worlds, list) and len(worlds) == WORLDS, "Exactly 128 selection worlds are required")
    totals = [[0] * support for _ in range(2)]
    for index, world in enumerate(worlds):
        _mapping(world, "Malformed selection world")
        require(type(world.get("seed")) is int and world["seed"] == seeds[index],
                "Selection-world ordering/seed mismatch")
        outcomes = world.get("outcomes")
        require(isinstance(outcomes, list) and len(outcomes) == support, "Incomplete candidate outcome coverage")
        for candidate, outcome in enumerate(outcomes):
            _mapping(outcome, "Malformed candidate outcome")
            value = outcome.get("wdl")
            require(type(value) in (int, float) and value in (-1, 0, 1)
                    and outcome.get("reason") in NATURAL_TERMINATIONS,
                    "Unknown or malformed selection outcome")
            totals[index // HALF_WORLDS][candidate] += int(value)
    return totals


def row_from_root(prediction, root):
    """Reconstruct both cross-block planner choices from a single authenticated root."""
    outsiders = _sealed_choices(prediction)
    _mapping(root, "Missing original root")
    require(root.get("stage") == "main", "Only main roots enter the diagnostic")
    require(_identity(prediction) == _identity(root.get("spec")), "Prediction/root identity mismatch")
    keys, recorded, legal_keys, anchor_scores, first = _source_candidates(root)
    require(outsiders == [key for key in recorded["teacher"]["keys"] if key not in first],
            "Sealed teacher outsider support differs from the root")
    full_scores = dict(zip(legal_keys, anchor_scores, strict=True))
    remaining = [key for key in legal_keys if key not in first]
    full_anchor = max(remaining, key=full_scores.__getitem__) if remaining else None
    restricted_anchor = max(outsiders, key=full_scores.__getitem__) if outsiders else None
    require(prediction["anchor_key"] == full_anchor == restricted_anchor,
            "Sealed anchor differs from the original ordered maximum")
    proposals = {"anchor": prediction["anchor_key"], **prediction["choices"]}
    candidates = {name: list(first) + ([] if proposals[name] is None else [proposals[name]])
                  for name in PLANNERS}
    require(candidates["anchor"] == recorded["anchor"]["keys"], "Anchor candidate reconstruction changed")
    for offered in candidates.values():
        require(len(offered) == len(set(offered)) and set(offered) <= set(keys),
                "Duplicate or unrecorded diagnostic candidate")
    half_totals = _half_totals(root, len(keys))
    positions = {key: index for index, key in enumerate(keys)}
    directions = []
    for selecting in range(2):
        assessing = 1 - selecting
        choices = {name: max(candidates[name], key=lambda key: half_totals[selecting][positions[key]])
                   for name in PLANNERS}
        comparator = half_totals[assessing][positions[choices["anchor"]]]
        delta = {name: half_totals[assessing][positions[choices[name]]] - comparator for name in MODELS}
        directions.append({"selection_range": [selecting * HALF_WORLDS, (selecting + 1) * HALF_WORLDS],
                           "assessment_range": [assessing * HALF_WORLDS, (assessing + 1) * HALF_WORLDS],
                           "choices": choices, "delta_wdl_sums": delta})
    return {field: prediction[field] for field in IDENTITY_FIELDS} | {
        "outsider_count": len(outsiders), "proposals": proposals, "candidates": candidates,
        "directions": directions,
        "delta_numerators": {name: sum(direction["delta_wdl_sums"][name] for direction in directions)
                             for name in MODELS},
        "unknown_selection_outcomes": 0,
    }


def _validated_rows(rows, boundary):
    require(isinstance(rows, (list, tuple)) and len(rows) == STATES, "Exactly 720 diagnostic roots are required")
    for row in rows:
        boundary()
        _identity(row)
        require(type(row.get("outsider_count")) is int and 0 <= row["outsider_count"] <= 5,
                "Invalid outsider support count")
        require(type(row.get("unknown_selection_outcomes")) is int and row["unknown_selection_outcomes"] == 0,
                "Unknown outcomes are forbidden")
        proposals = _mapping(row.get("proposals"), "Missing diagnostic proposals")
        candidates = _mapping(row.get("candidates"), "Missing diagnostic candidates")
        require(set(proposals) == set(candidates) == set(PLANNERS), "Incomplete diagnostic planners")
        first = _keys(candidates["anchor"], "Invalid anchor candidate list")[:3]
        for name in PLANNERS:
            offered = _keys(candidates[name], "Invalid diagnostic candidate list")
            proposal = proposals[name]
            require((proposal is None) == (row["outsider_count"] == 0), "Outsider/proposal absence mismatch")
            require(proposal is None or isinstance(proposal, str) and proposal not in first,
                    "Invalid fourth proposal")
            require(offered == first + ([] if proposal is None else [proposal]), "Diagnostic candidates changed")
            if row["outsider_count"] == 1:
                require(proposal == proposals["anchor"], "Singleton outsider mismatch")
        directions = row.get("directions")
        require(isinstance(directions, list) and len(directions) == 2, "Both fixed directions are required")
        for index, direction in enumerate(directions):
            _mapping(direction, "Malformed directional record")
            require(direction.get("selection_range") == [index * 64, (index + 1) * 64]
                    and direction.get("assessment_range") == [(1 - index) * 64, (2 - index) * 64],
                    "Changed world split or direction order")
            choices = _mapping(direction.get("choices"), "Missing directional choices")
            delta = _mapping(direction.get("delta_wdl_sums"), "Missing directional differences")
            require(set(choices) == set(PLANNERS) and set(delta) == set(MODELS), "Incomplete directional planners")
            require(all(choices[name] in candidates[name] for name in PLANNERS), "Directional choice outside support")
            for name in MODELS:
                require(type(delta[name]) is int and -128 <= delta[name] <= 128, "Invalid directional integer difference")
                if choices[name] == choices["anchor"]:
                    require(delta[name] == 0, "Identical chosen actions must cancel exactly")
                if candidates[name] == candidates["anchor"]:
                    require(choices[name] == choices["anchor"], "Identical planners must choose identically")
        numerators = _mapping(row.get("delta_numerators"), "Missing root integer differences")
        require(set(numerators) == set(MODELS), "Incomplete root contrasts")
        for name in MODELS:
            require(type(numerators[name]) is int and -256 <= numerators[name] <= 256
                    and numerators[name] == sum(direction["delta_wdl_sums"][name] for direction in directions),
                    "Root integer difference does not match its two directions")
    require([row["index"] for row in rows] == list(range(STATES)),
            "Missing, duplicated or out-of-order root indices")
    require(len({row["game_index"] for row in rows}) == STATES, "Repeated source games are not independent roots")
    require(Counter((row["fit"], row["seat"], row["phase"]) for row in rows)
            == Counter({cell: PER_CELL for cell in CELLS}), "The fixed balanced strata changed")
    return list(rows)


def bootstrap_totals(rows, boundary=lambda: None):
    """Return the registered joint bootstrap's integer totals in fixed model order."""
    ordered = _validated_rows(rows, boundary)
    matrices = [np.asarray([[row["delta_numerators"][name] for name in MODELS] for row in ordered
                            if (row["fit"], row["seat"], row["phase"]) == cell], dtype=np.int64)
                for cell in CELLS]
    random = np.random.default_rng(BOOTSTRAP_SEED)
    totals = np.zeros((BOOTSTRAP_SAMPLES, len(MODELS)), dtype=np.int64)
    for start in range(0, BOOTSTRAP_SAMPLES, BOOTSTRAP_CHUNK):
        boundary()
        size = min(BOOTSTRAP_CHUNK, BOOTSTRAP_SAMPLES - start)
        for matrix in matrices:
            sampled = random.integers(0, PER_CELL, size=(size, PER_CELL))
            totals[start:start + size] += matrix[sampled].sum(axis=1)
    boundary()
    return totals


def statistics(rows, boundary=lambda: None):
    """Summarize the development diagnostic without a gate or classification."""
    ordered = _validated_rows(rows, boundary)
    totals = bootstrap_totals(ordered, boundary)
    denominator = STATES * SCORE_DENOMINATOR
    quantiles = np.quantile(totals, [.0125, .9875], axis=0, method="linear") / denominator
    contrasts = {}
    for index, name in enumerate(MODELS):
        numerator = sum(row["delta_numerators"][name] for row in ordered)
        mean = numerator / denominator
        interval = [float(quantiles[0, index]), float(quantiles[1, index])]
        contrasts[name + "_minus_anchor"] = {
            "integer_numerator_total": numerator, "mean_score_difference": mean,
            "mean_percentage_points": 100 * mean, "ci975": interval,
            "ci975_percentage_points": [100 * value for value in interval],
        }
    proposal_changes, final_changes, fourth = {}, {}, {}
    for name in MODELS:
        proposals = sum(row["proposals"][name] != row["proposals"]["anchor"] for row in ordered)
        changed = sum(direction["choices"][name] != direction["choices"]["anchor"]
                      for row in ordered for direction in row["directions"])
        proposal_changes[name] = {"roots": proposals, "fraction": proposals / STATES}
        final_changes[name] = {"direction_choices": changed, "fraction": changed / (2 * STATES)}
    for name in PLANNERS:
        chosen = sum(row["proposals"][name] is not None and direction["choices"][name] == row["proposals"][name]
                     for row in ordered for direction in row["directions"])
        fourth[name] = {"direction_choices": chosen, "fraction": chosen / (2 * STATES)}
    support_counts = Counter(row["outsider_count"] for row in ordered)
    boundary()
    return {
        "states": STATES, "source_games": STATES, "states_per_cell": PER_CELL,
        "cells": [list(cell) for cell in CELLS], "cell_counts": [PER_CELL] * len(CELLS),
        "models": list(MODELS), "selector_worlds": HALF_WORLDS, "directions_per_root": 2,
        "root_score_denominator": SCORE_DENOMINATOR, "contrasts": contrasts,
        "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_chunk": BOOTSTRAP_CHUNK,
        "bootstrap_totals_sha256": hashlib.sha256(np.asarray(totals, dtype="<i8", order="C").tobytes(order="C")).hexdigest(),
        "bootstrap_totals_encoding": "little_endian_int64_C_order",
        "bootstrap_totals_shape": [BOOTSTRAP_SAMPLES, len(MODELS)],
        "outsider_support_counts": {str(count): support_counts[count] for count in range(6)},
        "degenerate_support_roots_retained": sum(row["outsider_count"] <= 1 for row in ordered),
        "unknown_selection_outcomes": 0,
        "proposal_changes": proposal_changes, "final_action_changes": final_changes,
        "fourth_selected": fourth,
    }
