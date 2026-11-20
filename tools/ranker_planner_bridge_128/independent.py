"""Independent saved-world arithmetic, with no inference or simulation imports.

The caller authenticates the frozen sources and compares ``reconstruct`` with
the primary row.  This implementation tallies wins and losses separately and
uses resampling multiplicities for its independent bootstrap calculation.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math

import numpy as np


MODELS = ("ridge", "hgb")
PLAYERS = ("anchor", "ridge", "hgb")
IDENTITY = ("index", "game_index", "fit", "seat", "phase", "snapshot_sha256")
CELLS = tuple((fit, seat, phase) for fit in (161001, 161002, 161003)
              for seat in (0, 1) for phase in (0, 1, 2))
STATES = 720
PER_CELL = 40
DENOMINATOR = 256
REPLICATES = 50000
SEED = 2813000000
CHUNK = 1000


def _require(condition, message):
    if not condition:
        raise ValueError("Independent bridge check: " + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _check_identity(value):
    _require(all(type(value[key]) is int for key in IDENTITY[:-1]),
             "identity indices must be integers rather than boolean aliases")
    _require(value["index"] >= 0 and value["game_index"] >= 0
             and (value["fit"], value["seat"], value["phase"]) in CELLS,
             "identity is outside the fixed panel")
    snapshot = value["snapshot_sha256"]
    _require(isinstance(snapshot, str) and len(snapshot) == 64
             and all(character in "0123456789abcdef" for character in snapshot),
             "invalid source snapshot digest")


def _candidate_record(record, union, representatives):
    keys, positions, actions = record["keys"], record["indices"], record["actions"]
    _require(isinstance(keys, list) and keys and len(keys) == len(set(keys)),
             "candidate list is empty or duplicated")
    _require(isinstance(positions, list) and len(positions) == len(keys)
             and all(type(i) is int and 0 <= i < len(union) for i in positions),
             "invalid candidate positions")
    _require(keys == [union[i] for i in positions], "candidate keys disagree with positions")
    _require(actions == [representatives[i] for i in positions],
             "candidate representatives disagree with union")
    return keys


def reconstruct(prediction, root):
    """Reconstruct one row without reading any original assessment outcomes."""
    spec = root["spec"]
    _check_identity(prediction)
    _check_identity(spec)
    _require(all(prediction[key] == spec[key] for key in IDENTITY),
             "prediction and root identities differ")
    _require(root.get("stage") == "main", "root is not a main-panel root")
    union = root["union"]["keys"]
    representatives = root["union"]["actions"]
    _require(isinstance(union, list) and 1 <= len(union) <= 10
             and all(isinstance(key, str) for key in union)
             and len(union) == len(set(union)) and len(representatives) == len(union),
             "invalid union keys or representatives")
    for recorded in root["candidates"].values():
        _candidate_record(recorded, union, representatives)
    anchor = root["candidates"]["anchor"]["keys"]
    teacher = root["candidates"]["teacher"]["keys"]
    _require(len(anchor) == min(4, len(union)), "anchor support has the wrong length")
    first = anchor[:3]
    outsiders = [key for key in teacher if key not in first]
    _require(len(outsiders) <= 5 and prediction["outsider_keys"] == outsiders,
             "sealed outsider support differs from recorded support")
    proposals = {"anchor": prediction["anchor_key"], **prediction["choices"]}
    _require(set(prediction["scores"]) == set(prediction["choices"]) == set(MODELS),
             "both fixed alternative recipes are required")
    _require((proposals["anchor"] in outsiders) if outsiders else proposals["anchor"] is None,
             "anchor outsider is invalid")
    _require(first + ([] if proposals["anchor"] is None else [proposals["anchor"]]) == anchor,
             "sealed anchor does not reproduce the recorded candidate list")
    for model in MODELS:
        scores = prediction["scores"][model]
        _require(isinstance(scores, list) and len(scores) == len(outsiders)
                 and all(_number(score) for score in scores), "invalid sealed model scores")
        ranking = sorted(range(len(scores)), key=lambda j: (-scores[j], j))
        expected = outsiders[ranking[0]] if ranking else None
        _require(proposals[model] == expected, "sealed choice is not its first score maximum")
    candidates = {
        player: list(first) + ([] if proposals[player] is None else [proposals[player]])
        for player in PLAYERS
    }
    for offered in candidates.values():
        _require(offered and len(offered) == len(set(offered))
                 and all(key in union for key in offered), "invalid reconstructed planner support")

    block = root["selection"]
    _require(block["candidate_indices"] == list(range(len(union))),
             "selection candidate order differs from union")
    worlds, seeds = block["worlds"], block["world_seeds"]
    _require(len(worlds) == len(seeds) == 128 and len(set(seeds)) == 128,
             "selection must contain 128 distinct declared worlds")
    # Separate counts provide an alternative to summing the primary WDL matrix.
    wins = [[0] * len(union) for _ in range(2)]
    losses = [[0] * len(union) for _ in range(2)]
    for world_index, world in enumerate(worlds):
        _require(world["seed"] == seeds[world_index], "world order or seed identity differs")
        outcomes = world["outcomes"]
        _require(len(outcomes) == len(union), "selection outcome coverage is incomplete")
        half = world_index // 64
        for column, outcome in enumerate(outcomes):
            value = outcome["wdl"]
            _require(_number(value) and value in (-1, 0, 1)
                     and outcome.get("reason") != "repetition_cutoff",
                     "selection contains an unresolved or malformed outcome")
            wins[half][column] += int(value == 1)
            losses[half][column] += int(value == -1)
    totals = [[wins[h][j] - losses[h][j] for j in range(len(union))] for h in range(2)]
    columns = {key: j for j, key in enumerate(union)}
    directions = []
    for selection_half in (0, 1):
        assessment_half = 1 - selection_half
        choices = {}
        for player in PLAYERS:
            offered = candidates[player]
            ordered = sorted(enumerate(offered),
                             key=lambda item: (-totals[selection_half][columns[item[1]]], item[0]))
            choices[player] = ordered[0][1]
        anchor_total = totals[assessment_half][columns[choices["anchor"]]]
        differences = {model: totals[assessment_half][columns[choices[model]]] - anchor_total
                       for model in MODELS}
        directions.append({
            "selection_range": [64 * selection_half, 64 * (selection_half + 1)],
            "assessment_range": [64 * assessment_half, 64 * (assessment_half + 1)],
            "choices": choices,
            "delta_wdl_sums": differences,
        })
    return {**{key: prediction[key] for key in IDENTITY},
            "outsider_count": len(outsiders), "proposals": proposals,
            "candidates": candidates, "directions": directions,
            "delta_numerators": {model: sum(direction["delta_wdl_sums"][model]
                                             for direction in directions) for model in MODELS},
            "unknown_selection_outcomes": 0}


def _check_rows(rows, boundary):
    _require(isinstance(rows, list) and len(rows) == STATES, "exactly 720 rows are required")
    _require([row["index"] for row in rows] == list(range(STATES)),
             "root identities are missing, duplicated or reordered")
    _require(all(type(row["game_index"]) is int and row["game_index"] >= 0 for row in rows)
             and len({row["game_index"] for row in rows}) == STATES,
             "source-game identities are invalid or duplicated")
    cells = Counter((row["fit"], row["seat"], row["phase"]) for row in rows)
    _require(cells == Counter({cell: PER_CELL for cell in CELLS}), "fixed strata changed")
    for row in rows:
        boundary()
        _check_identity(row)
        _require(type(row["outsider_count"]) is int and 0 <= row["outsider_count"] <= 5,
                 "invalid outsider support count")
        _require(row["unknown_selection_outcomes"] == 0, "unknown outcomes entered the statistic")
        _require(set(row["proposals"]) == set(row["candidates"]) == set(PLAYERS),
                 "missing planner in saved row")
        _require(set(row["delta_numerators"]) == set(MODELS), "missing model numerator")
        _require(len(row["directions"]) == 2, "both directions are required")
        anchor_candidates = row["candidates"]["anchor"]
        first = anchor_candidates[:3]
        for player in PLAYERS:
            proposal = row["proposals"][player]
            offered = row["candidates"][player]
            _require(offered == first + ([] if proposal is None else [proposal])
                     and offered and len(offered) == len(set(offered)),
                     "saved offered support is inconsistent")
            _require((proposal is None) == (row["outsider_count"] == 0),
                     "saved outsider presence is inconsistent")
        if row["outsider_count"] <= 1:
            _require(all(row["proposals"][model] == row["proposals"]["anchor"] for model in MODELS),
                     "degenerate support changed a proposal")
        for h, direction in enumerate(row["directions"]):
            _require(direction["selection_range"] == [64 * h, 64 * (h + 1)]
                     and direction["assessment_range"] == [64 * (1 - h), 64 * (2 - h)],
                     "direction ranges changed")
            _require(set(direction["choices"]) == set(PLAYERS)
                     and set(direction["delta_wdl_sums"]) == set(MODELS),
                     "saved directional comparison is incomplete")
            _require(all(direction["choices"][player] in row["candidates"][player] for player in PLAYERS),
                     "saved choice is outside offered support")
            for model in MODELS:
                value = direction["delta_wdl_sums"][model]
                _require(type(value) is int and -128 <= value <= 128, "invalid directional integer gain")
                if direction["choices"][model] == direction["choices"]["anchor"]:
                    _require(value == 0, "identical final actions did not cancel")
                if row["proposals"][model] == row["proposals"]["anchor"]:
                    _require(direction["choices"][model] == direction["choices"]["anchor"],
                             "identical planners chose different actions")
        for model in MODELS:
            numerator = row["delta_numerators"][model]
            _require(type(numerator) is int and numerator == sum(
                direction["delta_wdl_sums"][model] for direction in row["directions"]),
                "root numerator does not average both directions")
    return cells


def verify_statistics(rows, summary, boundary=lambda: None):
    """Check every numerical summary through resampling-count matrix products."""
    cells = _check_rows(rows, boundary)
    support_counts = Counter(row["outsider_count"] for row in rows)
    expected = {
        "states": STATES, "source_games": STATES, "states_per_cell": PER_CELL,
        "cells": [list(cell) for cell in CELLS], "cell_counts": [cells[cell] for cell in CELLS],
        "models": list(MODELS), "selector_worlds": 64, "directions_per_root": 2,
        "root_score_denominator": DENOMINATOR,
        "bootstrap_samples": REPLICATES, "bootstrap_seed": SEED, "bootstrap_chunk": CHUNK,
        "bootstrap_totals_encoding": "little_endian_int64_C_order",
        "bootstrap_totals_shape": [REPLICATES, len(MODELS)],
        "outsider_support_counts": {str(k): support_counts[k] for k in range(6)},
        "degenerate_support_roots_retained": support_counts[0] + support_counts[1],
        "unknown_selection_outcomes": 0,
    }
    for field, value in expected.items():
        _require(summary.get(field) == value, "summary field mismatch: " + field)
    proposal_changes, final_changes, fourth_selected = {}, {}, {}
    for model in MODELS:
        changed = sum(row["proposals"][model] != row["proposals"]["anchor"] for row in rows)
        final_changed = sum(direction["choices"][model] != direction["choices"]["anchor"]
                            for row in rows for direction in row["directions"])
        proposal_changes[model] = {"roots": changed, "fraction": changed / STATES}
        final_changes[model] = {"direction_choices": final_changed, "fraction": final_changed / (2 * STATES)}
    for player in PLAYERS:
        fourth = sum(row["proposals"][player] is not None
                     and direction["choices"][player] == row["proposals"][player]
                     for row in rows for direction in row["directions"])
        fourth_selected[player] = {"direction_choices": fourth, "fraction": fourth / (2 * STATES)}
    for field, value in (("proposal_changes", proposal_changes), ("final_action_changes", final_changes),
                         ("fourth_selected", fourth_selected)):
        _require(summary.get(field) == value, "summary field mismatch: " + field)

    matrices = []
    for cell in CELLS:
        ordered = sorted((row for row in rows if (row["fit"], row["seat"], row["phase"]) == cell),
                         key=lambda row: row["index"])
        matrices.append(np.array([[row["delta_numerators"][model] for model in MODELS]
                                  for row in ordered], dtype=np.int64))
    rng = np.random.default_rng(SEED)
    resampled_totals = np.empty((REPLICATES, len(MODELS)), dtype=np.int64)
    offsets = np.arange(CHUNK, dtype=np.int64)[:, None] * PER_CELL
    for start in range(0, REPLICATES, CHUNK):
        boundary()
        chunk_totals = np.zeros((CHUNK, len(MODELS)), dtype=np.int64)
        for matrix in matrices:
            draws = rng.integers(0, PER_CELL, size=(CHUNK, PER_CELL))
            # Each row's 40 indices become counts; no direct sampled-value indexing.
            multiplicities = np.bincount((draws + offsets).ravel(), minlength=CHUNK * PER_CELL)
            multiplicities = multiplicities.reshape(CHUNK, PER_CELL)
            chunk_totals += multiplicities @ matrix
        resampled_totals[start:start + CHUNK] = chunk_totals
        boundary()
    digest = hashlib.sha256(resampled_totals.astype("<i8", copy=False).tobytes(order="C")).hexdigest()
    _require(summary.get("bootstrap_totals_sha256") == digest, "bootstrap integer totals differ")
    # Interpolate integer totals first, then scale, matching the registered
    # exact interval representation without an extra floating-point rounding.
    intervals = np.percentile(resampled_totals, [1.25, 98.75], axis=0, method="linear")
    intervals = intervals / (STATES * DENOMINATOR)
    contrasts = {}
    for column, model in enumerate(MODELS):
        numerator = sum(row["delta_numerators"][model] for row in rows)
        mean = numerator / (STATES * DENOMINATOR)
        interval = intervals[:, column].tolist()
        contrasts[model + "_minus_anchor"] = {
            "integer_numerator_total": numerator,
            "mean_score_difference": mean, "mean_percentage_points": 100 * mean,
            "ci975": interval, "ci975_percentage_points": [100 * value for value in interval],
        }
    _require(summary.get("contrasts") == contrasts, "means or percentile intervals differ")
    boundary()
    return {"passed": True, "statistics_match": True, "states": STATES,
            "bootstrap_samples": REPLICATES, "bootstrap_totals_sha256": digest,
            "method": "independent_win_loss_tallies_and_bootstrap_multiplicity_matrices"}
