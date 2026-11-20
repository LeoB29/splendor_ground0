"""Prospective fixed-ranker assessment on an authenticated, completed panel.

No function in this module trains a model, chooses a model from outcomes, runs a
simulation, or edits the original study.  ``evaluate_sealed`` checks the separate
prediction seal before opening any outcome artifact.  The prediction-producing
caller must use source snapshots and frozen models, not partially read root files.

The two primary comparisons use nominal 97.5% two-sided stratified percentile
bootstrap intervals each (Bonferroni family coverage nominally 95%).  This is an
approximate bootstrap procedure, not an exact finite-sample coverage guarantee.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MODELS = ("ridge", "hgb")
FITS = (161001, 161002, 161003)
CELLS = tuple((fit, seat, phase) for fit in FITS for seat in (0, 1) for phase in (0, 1, 2))
STATES = 720
PER_CELL = 40
WORLDS = 128
SCORE_DENOMINATOR = 2 * WORLDS
BOOTSTRAP_SAMPLES = 50000
BOOTSTRAP_SEED = 2713000000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(reference, repo_root, expected=None):
    require(isinstance(reference, dict) and {"path", "sha256"} <= reference.keys(),
            "Missing authenticated artifact reference")
    path = (Path(repo_root) / reference["path"]).resolve()
    require(path.is_relative_to(Path(repo_root).resolve()), "Artifact escapes repository")
    if expected is not None:
        require(path == Path(expected).resolve(), "Artifact path does not match its role")
    require(path.is_file() and _sha(path) == reference["sha256"], "Artifact hash mismatch")
    return path


def _verified(reference, repo_root, expected=None):
    return _read(_verified_path(reference, repo_root, expected))


def _reference(path, repo_root):
    path = Path(path).resolve()
    return {"path": path.relative_to(Path(repo_root).resolve()).as_posix(), "sha256": _sha(path)}


def _same_reference(left, right, repo_root):
    return ((Path(repo_root) / left["path"]).resolve() == (Path(repo_root) / right["path"]).resolve()
            and left["sha256"] == right["sha256"])


def _identity(row):
    return tuple(row[key] for key in ("index", "game_index", "fit", "seat", "phase", "snapshot_sha256"))


def _validate_panel(rows):
    require(len(rows) == STATES and sorted(row["index"] for row in rows) == list(range(STATES)),
            "Exactly the fixed 720 indexed roots are required")
    require(all(type(row["index"]) is int and type(row["game_index"]) is int
                and row["game_index"] >= 0 for row in rows), "Invalid root/source-game identity")
    require(len({row["game_index"] for row in rows}) == STATES,
            "Repeated source games are not independent sampling units")
    require(Counter((row["fit"], row["seat"], row["phase"]) for row in rows)
            == Counter({cell: PER_CELL for cell in CELLS}), "The fixed 18 balanced strata changed")


def _prediction_choices(prediction):
    keys = prediction["outsider_keys"]
    require(isinstance(keys, list) and all(isinstance(key, str) for key in keys)
            and len(keys) == len(set(keys)), "Invalid or duplicated outsider support")
    require(set(prediction["scores"]) == set(prediction["choices"]) == set(MODELS),
            "Both and only the two frozen recipes are required")
    for name in MODELS:
        scores = prediction["scores"][name]
        require(isinstance(scores, list) and len(scores) == len(keys)
                and all(type(value) in (int, float) and math.isfinite(value) for value in scores),
                "Nonfinite or incomplete sealed predictions")
        chosen = keys[max(range(len(keys)), key=lambda j: scores[j])] if keys else None
        require(prediction["choices"][name] == chosen, "Choice is not the sealed first maximum")
    require(prediction["anchor_key"] in keys if keys else prediction["anchor_key"] is None,
            "Anchor choice is outside the recorded teacher outsiders")


def validate_prediction_seal(predictions_path, seal_path, *, repo_root, manifest, study_binding):
    """Only prediction/model/source metadata are accessed; no evaluation artifact."""
    seal = _read(seal_path)
    require(seal.get("manifest") == manifest and seal.get("study_binding") == study_binding,
            "Prediction seal belongs to another study")
    require(seal.get("predictions_locked_before_outcome_read") is True,
            "Predictions must be sealed before outcome access")
    document = _verified(seal["predictions"], repo_root, predictions_path)
    require(document.get("schema") == "alternative_ranker_predictions_v1"
            and document.get("manifest") == manifest and document.get("study_binding") == study_binding,
            "Wrong prediction schema or identity")
    for key in ("models_lock", "selection"):
        require(_same_reference(document[key], seal[key], repo_root), "Prediction dependencies changed")
        _verified_path(document[key], repo_root)
    _validate_panel(document["rows"])
    for row in document["rows"]:
        _prediction_choices(row)
    return document


def _authenticated_panel(study_root, repo_root, study_binding, selection_reference):
    """Check existing completion/audits; never repair, resume, or replay the study."""
    study_root = Path(study_root).resolve()
    status = _read(study_root / "final_status.json")
    require(status.get("manifest") == study_binding and status.get("complete") is True
            and status.get("all_prescribed_audits_pass") is True,
            "Current study must be complete with every prescribed audit passed")
    require(status.get("classification") != "acquisition_not_established"
            and status.get("fresh_evaluation_started") is not False,
            "No complete fresh panel exists after an acquisition stop")
    audit = _verified(status["audit"], repo_root, study_root / "final_audit.json")
    require(audit.get("manifest") == study_binding and audit.get("passed") is True,
            "Final study audit failed or belongs to another manifest")
    require(_same_reference(status["result"], audit["result"], repo_root), "Final result link changed")
    _verified_path(status["result"], repo_root, study_root / "result.json")
    freeze = _read(study_root / "freeze.json")
    require(freeze.get("binding") == study_binding, "Original study freeze changed")
    selection_seal = _read(study_root / "selection_seal.json")
    require(selection_seal.get("manifest") == study_binding
            and _same_reference(selection_seal["selection"], selection_reference, repo_root),
            "Predictions target another selected panel")
    selection = _verified(selection_reference, repo_root, study_root / "selection.json")
    require(selection.get("manifest") == study_binding, "Selected panel identity changed")
    specs = selection["main"]
    _validate_panel(specs)
    require(len(selection_seal["source_locks"]) == 4, "Incomplete source-game audit seals")
    for reference in selection_seal["source_locks"]:
        _verified_path(reference, repo_root)
    stages = {}
    for stage, count in (("probe", 18), ("main", STATES)):
        complete = _verified(audit["stages"][stage], repo_root, study_root / f"{stage}_complete.json")
        require(complete.get("manifest") == study_binding and complete.get("states") == count
                and complete.get("all_prescribed_audits_pass") is True,
                "Incomplete collection or audit coverage")
        if stage == "probe":
            require(complete.get("full_inner_replay") is True, "Probe full replay did not pass")
        for kind in ("collection", "audits"):
            lock = _verified(complete[kind], repo_root, study_root / f"{stage}_{kind}_lock.json")
            require(lock.get("manifest") == study_binding and len(lock["files"]) == count
                    and [ref["index"] for ref in lock["files"]] == list(range(count)),
                    "Stage lock does not cover its complete fixed schedule")
            stages[(stage, kind)] = lock["files"]
    return specs, stages[("main", "collection")], stages[("main", "audits")]


def _outcome_bounds(root):
    keys = root["union"]["keys"]
    require(1 <= len(keys) <= 10 and len(set(keys)) == len(keys), "Invalid recorded union")
    worlds = root["selection"]["worlds"]
    require(len(worlds) == WORLDS, "Exactly 128 selection worlds are required")
    lower, upper, unknown = [0] * len(keys), [0] * len(keys), [0] * len(keys)
    for world in worlds:
        require(len(world["outcomes"]) == len(keys), "Missing candidate outcome coverage")
        for j, outcome in enumerate(world["outcomes"]):
            value = outcome["wdl"]
            if value is None:
                require(outcome.get("reason") == "repetition_cutoff" and outcome.get("winner") is None,
                        "Unresolved WDL lacks the recorded repetition-cutoff semantics")
                upper[j] += 2
                unknown[j] += 1
            else:
                require(type(value) in (int, float) and value in (-1, 0, 1)
                        and outcome.get("reason") != "repetition_cutoff", "Invalid recorded WDL")
                lower[j] += int(value) + 1
                upper[j] += int(value) + 1
    return {key: [lower[j], upper[j]] for j, key in enumerate(keys)}, sum(unknown)


def row_from_root(prediction, root):
    """Evaluate fixed predictions, without selecting anything using outcomes."""
    _prediction_choices(prediction)
    spec = root["spec"]
    require(root.get("stage") == "main" and _identity(prediction) == _identity(spec),
            "Prediction/root identity mismatch")
    union = root["union"]["keys"]
    first = root["candidates"]["anchor"]["keys"][:3]
    teacher = root["candidates"]["teacher"]
    require(teacher["keys"] == [union[j] for j in teacher["indices"]], "Teacher keys/indices disagree")
    outsiders = [key for key in teacher["keys"] if key not in first]
    require(prediction["outsider_keys"] == outsiders, "Recorded teacher outsider support changed")
    proposal = root["proposals"]["anchor"]
    group_keys, anchor_scores = proposal["group_keys"], proposal["learned_group_scores"]
    require(len(group_keys) == len(anchor_scores) and len(set(group_keys)) == len(group_keys)
            and all(type(value) in (int, float) and math.isfinite(value) for value in anchor_scores),
            "Invalid recorded anchor scores")
    require(all(key in group_keys for key in outsiders), "Recorded outsider lacks anchor score")
    scores = dict(zip(group_keys, anchor_scores))
    anchor = max(outsiders, key=lambda key: scores[key]) if outsiders else None
    require(anchor == prediction["anchor_key"], "Assigned anchor differs from recorded restricted maximum")
    values, unknown = _outcome_bounds(root)
    deltas, regrets = {}, {}
    anchor_regret = [0, 0]
    if outsiders:
        best_lower = max(values[key][0] for key in outsiders)
        best_upper = max(values[key][1] for key in outsiders)
        anchor_regret = [max(0, best_lower - values[anchor][1]), best_upper - values[anchor][0]]
    for name in MODELS:
        chosen = prediction["choices"][name]
        if len(outsiders) <= 1 or chosen == anchor:
            deltas[name] = [0, 0]
        else:
            deltas[name] = [values[chosen][0] - values[anchor][1], values[chosen][1] - values[anchor][0]]
        if len(outsiders) <= 1:
            regrets[name] = [0, 0]
        else:
            regrets[name] = [max(0, best_lower - values[chosen][1]), best_upper - values[chosen][0]]
    if len(outsiders) <= 1:
        anchor_regret = [0, 0]
    return {key: prediction[key] for key in ("index", "game_index", "fit", "seat", "phase")} | {
        "outsider_count": len(outsiders), "choices": {"anchor": anchor, **prediction["choices"]},
        "delta_numerators": deltas, "anchor_regret_numerators": anchor_regret,
        "model_regret_numerators": regrets, "unknown_selection_outcomes": unknown,
    }


def _ratio_bounds(numerator, denominator):
    if denominator[0] <= 0:
        return None
    values = [n / d for n in numerator for d in denominator]
    return [min(values), max(values)]


def evaluate_rows(rows, *, bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED,
                  bootstrap_chunk=1000):
    """Paired source-game bootstrap; no model/checkpoint/subgroup selection."""
    _validate_panel(rows)
    require(type(bootstrap_samples) is int and bootstrap_samples >= 2
            and type(bootstrap_seed) is int and bootstrap_seed >= 0
            and type(bootstrap_chunk) is int and bootstrap_chunk > 0, "Invalid bootstrap settings")
    for row in rows:
        require(set(row["delta_numerators"]) == set(row["model_regret_numerators"]) == set(MODELS),
                "Incomplete evaluated recipes")
        require(type(row["outsider_count"]) is int and 0 <= row["outsider_count"] <= 5,
                "Invalid outsider count")
        for name in MODELS:
            bound = row["delta_numerators"][name]
            require(len(bound) == 2 and all(type(x) is int for x in bound)
                    and -SCORE_DENOMINATOR <= bound[0] <= bound[1] <= SCORE_DENOMINATOR,
                    "Invalid conservative score difference")
            if row["outsider_count"] <= 1 or row["choices"][name] == row["choices"]["anchor"]:
                require(bound == [0, 0], "Same action or degenerate support must cancel exactly")
        for bound in [row["anchor_regret_numerators"], *row["model_regret_numerators"].values()]:
            require(len(bound) == 2 and all(type(x) is int for x in bound)
                    and 0 <= bound[0] <= bound[1] <= SCORE_DENOMINATOR, "Invalid regret bounds")
    matrices = []
    for cell in CELLS:
        subset = sorted((row for row in rows if (row["fit"], row["seat"], row["phase"]) == cell),
                        key=lambda row: row["index"])
        matrices.append(np.asarray([[number for name in MODELS for number in row["delta_numerators"][name]]
                                    for row in subset], dtype=np.int64))
    totals = sum((matrix.sum(axis=0) for matrix in matrices), np.zeros(4, dtype=np.int64))
    independent_totals = [sum(row["delta_numerators"][name][endpoint] for row in rows)
                          for name in MODELS for endpoint in (0, 1)]
    require(totals.tolist() == independent_totals, "Independent integer mean check failed")
    denominator = STATES * SCORE_DENOMINATOR
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.empty((bootstrap_samples, 4), dtype=np.int64)
    for start in range(0, bootstrap_samples, bootstrap_chunk):
        size = min(bootstrap_chunk, bootstrap_samples - start)
        block = np.zeros((size, 4), dtype=np.int64)
        for matrix in matrices:
            block += matrix[rng.integers(0, PER_CELL, size=(size, PER_CELL))].sum(axis=1)
        draws[start:start + size] = block
    quantiles = np.quantile(draws, [.0125, .9875], axis=0, method="linear") / denominator
    anchor_regret = [sum(row["anchor_regret_numerators"][j] for row in rows) / denominator for j in (0, 1)]
    contrasts = {}
    for j, name in enumerate(MODELS):
        bounds = [int(totals[2*j]) / denominator, int(totals[2*j+1]) / denominator]
        interval = [float(quantiles[0, 2*j]), float(quantiles[1, 2*j+1])]
        capture = _ratio_bounds(bounds, anchor_regret)
        contrasts[name + "_minus_anchor"] = {
            "mean_score_difference": bounds[0] if bounds[0] == bounds[1] else None,
            "mean_bounds": bounds, "mean_percentage_point_bounds": [100*x for x in bounds],
            "ci975": interval, "ci975_percentage_points": [100*x for x in interval],
            "superiority_established": interval[0] > 0,
            "classification": ("positive_restricted_ranking_gain" if interval[0] > 0 else
                               "restricted_ranking_deterioration" if interval[1] < 0 else
                               "restricted_ranking_gain_not_established"),
            "supplementary_capture_bounds": capture,
            "supplementary_model_regret_bounds": [sum(row["model_regret_numerators"][name][k] for row in rows) / denominator for k in (0, 1)],
        }
    return {"states": STATES, "source_games": STATES, "states_per_cell": PER_CELL,
            "cells": [list(cell) for cell in CELLS], "models": list(MODELS), "worlds": WORLDS,
            "primary_endpoint": "fixed_model_minus_assigned_anchor_restricted_outsider_score",
            "contrasts": contrasts, "bootstrap_samples": bootstrap_samples, "bootstrap_seed": bootstrap_seed,
            "interval_method": "paired_stratified_percentile_bootstrap_Bonferroni_two_97.5_percent_intervals",
            "nominal_family_coverage": .95, "finite_sample_coverage_guaranteed": False,
            "independent_integer_mean_check": True,
            "supplementary_anchor_regret_bounds": anchor_regret,
            "supplementary_capture_qualification": "Gain divided by empirical noisy-maximum anchor regret; not unbiased capture of latent true regret; no capture-based success gate.",
            "degenerate_support_roots_retained": sum(row["outsider_count"] <= 1 for row in rows),
            "unknown_selection_outcomes": sum(row["unknown_selection_outcomes"] for row in rows),
            "unknown_outcomes_imputed": False, "automatic_successor": False,
            "scope": "Fresh recorded teacher-outsider ranking under greedy035 continuations; not full-legal proposal, planner, arena, or superhuman strength.",
            "rows": sorted(rows, key=lambda row: row["index"])}


def evaluate_sealed(predictions_path, seal_path, *, repo_root, study_root, manifest, study_binding,
                    bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED):
    """Read outcomes only after predictions are immutable and study audits complete."""
    document = validate_prediction_seal(predictions_path, seal_path, repo_root=repo_root,
                                        manifest=manifest, study_binding=study_binding)
    specs, roots, audits = _authenticated_panel(study_root, repo_root, study_binding, document["selection"])
    predictions = {row["index"]: row for row in document["rows"]}
    rows = []
    for spec, root_ref, audit_ref in zip(specs, roots, audits, strict=True):
        prediction = predictions[spec["index"]]
        require(_identity(prediction) == _identity(spec), "Sealed predictions target a different source state")
        receipt = _verified(audit_ref, repo_root)
        require(receipt.get("manifest") == study_binding and receipt.get("stage") == "main"
                and receipt.get("passed") is True and receipt["spec"] == spec
                and _same_reference(receipt["root"], root_ref, repo_root), "Root audit coverage/linkage failed")
        root = _verified(root_ref, repo_root)
        require(root.get("manifest") == study_binding and root["spec"] == spec,
                "Root manifest/specification changed")
        rows.append(row_from_root(prediction, root))
    result = evaluate_rows(rows, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed)
    result.update(manifest=manifest, study_binding=study_binding,
                  predictions=_reference(predictions_path, repo_root), prediction_seal=_reference(seal_path, repo_root),
                  original_study_final_status=_reference(Path(study_root) / "final_status.json", repo_root),
                  all_prescribed_original_audits_pass=True)
    return result
