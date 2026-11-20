"""Seal outcome-blind predictions from audited source snapshots only.

This command is explicit and deferred until the original study has finished.
It never opens a root, selection-world, assessment-world, or result artifact.
Source-game JSON includes terminal metadata; only its selected audited snapshot
is used.  No claim is made that those source JSON bytes exclude game outcomes.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from . import evaluation as e

PREDICTION_CONFIG = {
    "schema": "alternative_ranker_predictions_v1",
    "source": "audited_original_selected_source_snapshot",
    "world_budget_for_candidate_nonce": 128,
    "teacher_random_tag": "structured-proposer-128-teacher-random-v1",
    "teacher_stage": "main",
    "maximum_teacher_candidates": 8,
    "anchor_choice": "first_maximum_over_teacher_outsiders",
    "ranker_choice": "first_maximum_over_teacher_outsiders",
    "fresh_root_files_read": False,
    "automatic_prediction": False,
}


def teacher_support(group_keys, anchor_scores, reference_keys, index, digest):
    """Reconstruct exact ordered teacher support using no payoff information."""
    e.require(len(group_keys) == len(anchor_scores) and len(group_keys) == len(set(group_keys))
              and all(type(v) in (int, float) and math.isfinite(v) for v in anchor_scores),
              "Invalid full-legal original-anchor scores")
    e.require(len(reference_keys) == min(4, len(group_keys))
              and len(set(reference_keys)) == len(reference_keys)
              and set(reference_keys) <= set(group_keys), "Invalid original first-three support")
    first = reference_keys[:3]
    remaining = [j for j, key in enumerate(group_keys) if key not in first]
    anchor_index = max(remaining, key=lambda j: anchor_scores[j]) if remaining else None
    anchor = first + ([] if anchor_index is None else [group_keys[anchor_index]])
    remaining = [j for j, key in enumerate(group_keys) if key not in anchor]
    ranked = sorted(remaining, key=lambda j: (-anchor_scores[j], j))[:min(2, min(8, len(group_keys))-len(anchor))]
    available = [j for j in remaining if j not in ranked]
    uniform = sorted(available, key=lambda j: (
        digest([PREDICTION_CONFIG["teacher_random_tag"], "main", index, group_keys[j]]), j))
    uniform = uniform[:min(8, len(group_keys))-len(anchor)-len(ranked)]
    teacher = anchor + [group_keys[j] for j in ranked+uniform]
    outsiders = [key for key in teacher if key not in first]
    return {"first_three": list(first), "teacher_keys": teacher, "outsider_keys": outsiders,
            "anchor_key": None if anchor_index is None else group_keys[anchor_index]}


def source_snapshot(spec, *, repo_root, study_binding, digest):
    """Authenticate the same selected snapshot used by the original engine."""
    game = e._verified(spec["source_game"], repo_root)
    receipt = e._verified(spec["source_audit"], repo_root)
    e.require(game.get("manifest") == receipt.get("manifest") == study_binding
              and receipt.get("passed") is True and "failure" not in game, "Invalid audited source game")
    e.require(e._same_reference(receipt["game"], spec["source_game"], repo_root), "Source audit link changed")
    source_spec = game["spec"]
    e.require(receipt["spec"] == source_spec and spec["game_index"] == spec["pair"] == source_spec["index"]
              and spec["source_stage"] == source_spec["stage"] == "main"
              and all(spec[key] == source_spec[key] for key in ("fit", "seat", "phase", "search_stream")),
              "Wrong selected source game or assigned anchor")
    moves = [move for move in game["trace"] if move["turn"] == spec["turn"]]
    e.require(len(moves) == 1, "Selected source turn missing or duplicated")
    snapshot = moves[0]["snapshot"]
    e.require(digest(snapshot) == spec["snapshot_sha256"], "Selected source snapshot changed")
    return snapshot


def prediction_row(spec, snapshot, anchor_model, rankers):
    """Model inference only, over a reconstructed lawful source observation."""
    import torch
    from splendor_ai.alphazero import run_afterstate_advantage as old
    from splendor_ai.alphazero import run_rollout_selector_gameplay as baseline
    from splendor_ai.alphazero import world_budget_160 as world128
    from splendor_ai.alphazero.strong_policy import encode_position
    from .features import group_features
    from .learning import predict_scores

    state = old.deserialize_state_snapshot(snapshot)
    e.require(not state.terminal and state.turn_index == spec["turn"] and state.current_player == spec["seat"]
              and old.phase(state.turn_index) == spec["phase"], "Wrong source decision")
    legal = old.SplendorEnv().legal_actions(state)
    schedule = world128.root_schedule(state, spec["search_stream"], 128)
    reference = baseline.candidate_set(state, schedule["candidate_nonce"])
    with torch.no_grad():
        encoded, encoded_keys, _ = encode_position(state, anchor_model.config, legal)
        logits, _ = anchor_model(encoded["observation"][None], encoded["features"][None],
                                 torch.ones((1, len(legal)), dtype=torch.bool))
    keys = [repr(key) for key in encoded_keys]
    e.require(torch.isfinite(logits).all().item(), "Nonfinite original anchor logits")
    scores = [float(torch.amax(logits[0][encoded["groups"] == j])) for j in range(len(keys))]
    support = teacher_support(keys, scores, reference["keys"], spec["index"], old.digest)
    outsiders = support["outsider_keys"]
    members = [torch.where(encoded["groups"] == keys.index(key))[0].tolist() for key in outsiders]
    vectors = group_features(encoded["observation"], encoded["features"], members)
    ranker_scores, choices = {}, {}
    for name in e.MODELS:
        values = predict_scores(rankers[name], vectors).tolist() if outsiders else []
        ranker_scores[name] = [float(value) for value in values]
        choices[name] = outsiders[max(range(len(outsiders)), key=lambda j: values[j])] if outsiders else None
    row = {key: spec[key] for key in ("index", "game_index", "fit", "seat", "phase", "snapshot_sha256")}
    row.update(outsider_keys=outsiders, anchor_key=support["anchor_key"], scores=ranker_scores, choices=choices)
    e._prediction_choices(row)
    return row


def _load_anchors(registry, verify_hash):
    from splendor_ai.alphazero.strong_policy import StrongPolicyInference
    anchors = {}
    for fit in e.FITS:
        reference = registry["models"][f"anchor_{fit}"]
        e.require(reference["source_fit"] == fit, "Original anchor lineage changed")
        anchors[fit] = StrongPolicyInference(verify_hash(reference)).model
    return anchors


def _initialize_reference():
    from splendor_ai.alphazero import run_afterstate_advantage as old
    old.initialize_worker()


def _resource_boundary(started, common):
    """No source progress parsing: this explicit command requires completion."""
    import psutil
    e.require(time.monotonic() - started <= common.RESOURCE_CONFIG["maximum_seconds"],
              "Two-hour prediction ceiling reached; preserve fitted models")
    value = common.resource_snapshot()
    for field, minimum in (("available_bytes", "ongoing_minimum_available_bytes"),
                           ("commit_headroom_bytes", "ongoing_minimum_commit_headroom_bytes"),
                           ("disk_free_bytes", "disk_reserve_bytes")):
        e.require(value[field] >= common.RESOURCE_CONFIG[minimum], "Prediction resource reserve reached")
    e.require(psutil.Process().memory_info().rss <= common.RESOURCE_CONFIG["maximum_process_rss_bytes"],
              "Prediction process memory ceiling reached")


def create_predictions(*, repo_root, base, study_root, manifest):
    """Explicit one-shot command; no automatic wait, retries, or evaluation."""
    from . import common as c
    from .learning import CONFIG as LEARNING_CONFIG, load_model
    from .features import FEATURE_CONFIG

    repo_root, base, study_root = map(lambda path: Path(path).resolve(), (repo_root, base, study_root))
    e.require(repo_root == c.ROOT.resolve() and base == c.OUTPUT.resolve()
              and study_root == c.STUDY_ROOT.resolve(), "Production prediction paths changed")
    frozen = c.validate_freeze()
    registration = c.verified(frozen["manifest"])
    e.require(frozen["binding"] == manifest, "Wrong alternative-ranker manifest")
    study_binding = registration["study_binding"]
    # Existence only: do not read the original result or classification before sealing predictions.
    e.require((study_root / "final_status.json").is_file() and (study_root / "main_complete.json").is_file(),
              "Wait for the completed original panel; no automatic monitoring")
    prediction_path, seal_path = base / "predictions.json", base / "predictions_seal.json"
    e.require(not prediction_path.exists() and not seal_path.exists(), "One-shot predictions already exist; preserve them")
    models_ref = c.artifact(base / "models_lock.json")
    models_lock = c.verified(models_ref)
    e.require(models_lock.get("manifest") == manifest and set(models_lock["models"]) == set(e.MODELS),
              "Both fixed rankers must be sealed before source inference")
    fitted = c.verified(models_lock["learning"])
    e.require(fitted.get("config") == LEARNING_CONFIG and fitted.get("feature_config") == FEATURE_CONFIG
              and fitted.get("fresh_outcomes_read") is False and fitted.get("validation_read") is False
              and fitted.get("new_simulations") == 0 and fitted.get("hyperparameter_selection") is False
              and set(fitted["models"]) == set(e.MODELS)
              and e._same_reference(fitted["dataset"], models_lock["dataset"], repo_root),
              "Fitted recipes or TRAIN contract differ from their registration")
    c.verify_hash(models_lock["dataset"])
    for name in e.MODELS:
        fitted_model = fitted["models"][name]
        e.require(fitted_model.get("identical_second_fit") is True and fitted_model.get("selected_fit") == "first"
                  and e._same_reference(fitted_model["model"], models_lock["models"][name], repo_root),
                  "Model seal lacks exact deterministic fitting reproduction")
        e.require(all(isinstance(fitted_model.get(key), str) and len(fitted_model[key]) == 64
                      and all(char in "0123456789abcdef" for char in fitted_model[key])
                      for key in ("parameter_sha256", "all_train_prediction_sha256")),
                  "Missing parameter or full-TRAIN prediction reproduction digest")
    c.configure_compute()
    c.resource_gate()
    started = time.monotonic()
    rankers = {name: load_model(models_lock["models"][name], expected_recipe=name,
                               expected_dataset=models_lock["dataset"]) for name in e.MODELS}
    selection_seal = c.read_json(study_root / "selection_seal.json")
    e.require(selection_seal.get("manifest") == study_binding, "Original selection identity changed")
    selection_ref = selection_seal["selection"]
    selection = e._verified(selection_ref, repo_root, study_root / "selection.json")
    e.require(selection.get("manifest") == study_binding, "Selected source manifest changed")
    specs = selection["main"]
    e._validate_panel(specs)
    registry_ref = c.artifact(study_root / "models_lock.json")
    registry = c.verified(registry_ref)
    e.require(registry.get("manifest") == study_binding, "Original anchor registry changed")
    original_data = c.verified(registration["source"]["data"])
    for fit in e.FITS:
        key = f"anchor_{fit}"
        e.require(e._same_reference(registry["models"][key], original_data["parent"]["models"][key], repo_root),
                  "Anchor checkpoint differs from the preregistered original initializer")
    anchors = _load_anchors(registry, c.verify_hash)
    _initialize_reference()
    rows = []
    for spec in specs:
        _resource_boundary(started, c)
        snapshot = source_snapshot(spec, repo_root=repo_root, study_binding=study_binding, digest=c.digest)
        rows.append(prediction_row(spec, snapshot, anchors[spec["fit"]], rankers))
    _resource_boundary(started, c)
    document = {"schema": PREDICTION_CONFIG["schema"], "manifest": manifest, "study_binding": study_binding,
                "models_lock": models_ref, "selection": selection_ref, "original_anchors": registry_ref,
                "time": c.now(), "rows": rows, "teacher_outcomes_accessed": False,
                "new_simulations": 0, "config": PREDICTION_CONFIG}
    c.new_json(prediction_path, document)
    seal = {"manifest": manifest, "study_binding": study_binding, "time": c.now(),
            "predictions": c.artifact(prediction_path), "models_lock": models_ref,
            "selection": selection_ref, "predictions_locked_before_outcome_read": True}
    c.new_json(seal_path, seal)
    e.validate_prediction_seal(prediction_path, seal_path, repo_root=repo_root,
                               manifest=manifest, study_binding=study_binding)
    return {"predictions": c.artifact(prediction_path), "seal": c.artifact(seal_path),
            "states": len(rows), "teacher_outcomes_accessed": False, "new_simulations": 0}
