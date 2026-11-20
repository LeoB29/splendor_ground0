"""Lock old-world decisions, then independently assess frozen signal and capture."""
from __future__ import annotations

from pathlib import Path
import math
import random
import time
import traceback

import torch

from . import common as c
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

ROOT, OUTPUT, CONFIG = c.ROOT, c.OUTPUT, c.CONFIG
old, baseline, require = c.old, c.baseline, c.require
PROPOSERS = ("anchor", "broad", "narrow")
ARMS = ("anchor", "teacher", "teacher_outsider", "original_fourth") + tuple(
    f"{arm}_{role}" for arm in PROPOSERS for role in ("direct", "outsider", "supported"))
CONTRASTS = {name: tuple(name.split("_minus_")) for name in (
    "teacher_minus_anchor", "teacher_outsider_minus_anchor_outsider",
    "broad_direct_minus_anchor_direct", "narrow_direct_minus_anchor_direct",
    "broad_direct_minus_narrow_direct", "teacher_minus_anchor_direct",
    "teacher_minus_broad_direct", "teacher_minus_narrow_direct",
    "broad_outsider_minus_anchor_outsider", "narrow_outsider_minus_anchor_outsider",
    "broad_outsider_minus_narrow_outsider", "teacher_outsider_minus_broad_outsider",
    "teacher_outsider_minus_narrow_outsider", "broad_supported_minus_anchor",
    "narrow_supported_minus_anchor", "broad_supported_minus_narrow_supported",
    "teacher_minus_broad_supported", "teacher_minus_narrow_supported")}
_MODELS, _MODEL_ARTIFACTS = {}, {}


class AmbiguousDecision(ValueError):
    def __init__(self, selection):
        super().__init__("Unresolved old outcome can change a locked decision")
        self.selection = selection


def initialize_models(base=OUTPUT, registry=None):
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture registry targets production")
    registry = old.read_json(Path(base)/"models_lock.json") if registry is None else registry
    require(set(registry["models"]) == {f"{a}_{f}" for a in PROPOSERS for f in c.FITS}, "Incomplete model seal")
    models, identities = {}, {}
    for key, item in registry["models"].items():
        require(item["source_fit"] == int(key.split("_")[1]), "Wrong model fit")
        models[key] = StrongPolicyInference(c.verify_hash(item)).model
        identities[key] = dict(item)
    old.initialize_worker()
    _MODELS.clear(); _MODELS.update(models)
    _MODEL_ARTIFACTS.clear(); _MODEL_ARTIFACTS.update(identities)


def check_stage(stage, index, base=OUTPUT):
    require(stage in ("probe", "main", "test") and type(index) is int and index >= 0, "Invalid root identity")
    if stage == "test":
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
    require(index < (18 if stage == "test" else CONFIG["states"][stage]), "Root outside frozen schedule")


def root_path(stage, index, base=OUTPUT):
    check_stage(stage, index, base)
    return Path(base)/stage/"roots"/f"state_{index:04d}.json"


def selection_path(stage, index, base=OUTPUT):
    check_stage(stage, index, base)
    return Path(base)/stage/"selections"/f"state_{index:04d}_selection.json"


def world_seeds(stage, index, block="assessment"):
    require(stage in ("probe", "main", "test") and block == "assessment", "World namespace")
    require(type(index) is int and 0 <= index < (18 if stage == "test" else CONFIG["states"][stage]), "World index")
    offset = {"probe": 0, "main": 1000000, "test": 2000000}[stage] + index*512
    return [CONFIG["world_seed_base"] + offset + i for i in range(128)]


def validate_candidates(state, candidates):
    keys, actions = candidates["keys"], candidates["actions"]
    require(1 <= len(keys) <= 8 and len(keys) == len(actions) == len(set(keys)), "Invalid recorded support")
    legal = old.SplendorEnv().legal_actions(state)
    decoded = [old.decode_action(a) for a in actions]
    require(all(a in legal for a in decoded), "Illegal recorded candidate")
    require([repr(old.canonical_successor_action_key(state, a)) for a in decoded] == keys, "Changed candidate identity")
    return legal


def frozen_proposals(state, candidates, fit):
    require(fit in c.FITS, "Unknown source fit")
    legal = validate_candidates(state, candidates)
    expected_keys = [repr(k) for k in old.actions_by_canonical_successor_key(state, legal)]
    result = {}
    for arm in PROPOSERS:
        started = time.perf_counter()
        key = f"{arm}_{fit}"
        require(key in _MODELS and key in _MODEL_ARTIFACTS, "Unsealed model")
        model, identity = _MODELS[key], _MODEL_ARTIFACTS[key]
        with torch.no_grad():
            row, keys, _ = encode_position(state, model.config, legal)
            logits, _ = model(row["observation"][None], row["features"][None], torch.ones((1, len(legal)), dtype=torch.bool))
        require([repr(k) for k in keys] == expected_keys and torch.isfinite(logits).all().item(), "Invalid model inputs or outputs")
        values = [float(torch.amax(logits[0][row["groups"] == i])) for i in range(len(keys))]
        result[arm] = {"model_id": key, "model_sha256": identity["sha256"], "fit": fit,
            "group_keys": expected_keys, "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": logits[0].tolist(), "learned_group_scores": values,
            "offered_group_scores": [values[expected_keys.index(k)] for k in candidates["keys"]],
            "timing": {"wall": time.perf_counter()-started}}
    return result


def outcome_value(outcome):
    if outcome["reason"] == "repetition_cutoff":
        require(outcome["winner"] is None and outcome["wdl"] is None, "Invented censored result")
        return None
    value = outcome["wdl"]
    require(outcome["reason"] in old.VALID_NATURAL_TERMINATIONS and type(value) in (int, float)
            and value in (-1., 0., 1.), "Invalid continuation outcome")
    return int(value)


def bounded_means(worlds, support, count=128):
    require(count in (64, 128) and len(worlds) == count and type(support) is int and 1 <= support <= 8, "Incorrect world budget")
    require(all(len(w["outcomes"]) == support for w in worlds), "Missing or extra outcomes")
    sums, missing, cells = [0]*support, [0]*support, []
    for wi, world in enumerate(worlds):
        for ci, outcome in enumerate(world["outcomes"]):
            value = outcome_value(outcome)
            if value is None:
                missing[ci] += 1
                cells.append({"world": wi, "candidate": ci})
            else:
                sums[ci] += value
    return {"means": [sums[i]/count if missing[i] == 0 else None for i in range(support)],
            "means_bounds": [[(sums[i]-missing[i])/count, (sums[i]+missing[i])/count] for i in range(support)],
            "sums_wdl": sums, "unknown_counts": missing, "unresolved_cells": cells}


def certificate(value, indices):
    require(len(indices) == len(set(indices)) and all(type(i) is int and 0 <= i < len(value["sums_wdl"]) for i in indices), "Invalid decision support")
    winners = []
    margins = {}
    for position, candidate in enumerate(indices):
        row = [0 if candidate == other else value["sums_wdl"][candidate]-value["sums_wdl"][other]
               -value["unknown_counts"][candidate]-value["unknown_counts"][other] for other in indices]
        if all(margin > 0 if j < position else margin >= 0 for j, margin in enumerate(row) if j != position):
            winners.append(candidate)
            margins[candidate] = row
    require(len(winners) <= 1, "Nonunique certified winner")
    choice = winners[0] if winners else None
    return {"indices": list(indices), "choice": choice, "pairwise_lower_numerators": margins.get(choice)}


def rebuild_choices(old_worlds, proposals, candidates):
    require(set(proposals) == set(PROPOSERS), "Incomplete model proposals")
    support = len(candidates["keys"])
    require(len(candidates["actions"]) == support == len(set(candidates["keys"])), "Invalid offered support")
    value = bounded_means(old_worlds, support)
    full, common, outside = list(range(support)), list(range(min(support, 3))), list(range(3, support))
    supports = {"anchor": full[:4], "teacher": full, "teacher_outsider": outside}
    choices, certificates = {"original_fourth": 3 if outside else None}, {}
    supports["original_fourth"] = [3] if outside else []
    for arm in PROPOSERS:
        scores = proposals[arm]["offered_group_scores"]
        require(len(scores) == support and all(type(v) in (int, float) and math.isfinite(v) for v in scores), "Invalid offered logits")
        choices[f"{arm}_direct"] = max(full, key=lambda i: scores[i])
        choices[f"{arm}_outsider"] = max(outside, key=lambda i: scores[i]) if outside else None
        supports[f"{arm}_direct"], supports[f"{arm}_outsider"] = full, outside
        supports[f"{arm}_supported"] = common + ([choices[f"{arm}_outsider"]] if outside else [])
    require(choices["anchor_outsider"] == (3 if outside else None), "Original fourth is not the anchor's best recorded outsider")
    for arm in ("anchor", "teacher", "teacher_outsider") + tuple(f"{a}_supported" for a in PROPOSERS):
        certificates[arm] = certificate(value, supports[arm])
        choices[arm] = certificates[arm]["choice"]
    value.update(choices={arm: choices[arm] for arm in ARMS}, supports={arm: supports[arm] for arm in ARMS},
                 rollout_certificates=certificates)
    if any(choices[arm] is None and supports[arm] for arm in choices):
        raise AmbiguousDecision(value)
    require(choices["anchor_supported"] == choices["anchor"], "Anchor identity failed")
    return value


def scored_choices(value, choices):
    require(set(choices) == set(ARMS), "Incomplete frozen decisions")
    n = len(value["means"])
    require(all(i is None or type(i) is int and 0 <= i < n for i in choices.values()), "Choice outside support")
    require(all((i is None) == (n <= 3 and (arm.endswith("_outsider") or arm == "original_fourth")) for arm, i in choices.items()), "Invalid absent decision")
    bounds = {arm: None if i is None else [(1+x)/2 for x in value["means_bounds"][i]] for arm, i in choices.items()}
    scores = {arm: None if i is None or value["means"][i] is None else (1+value["means"][i])/2 for arm, i in choices.items()}
    deltas = {}
    for name, (left, right) in CONTRASTS.items():
        require((choices[left] is None) == (choices[right] is None), "One-sided absent contrast")
        deltas[name] = [0., 0.] if choices[left] == choices[right] else [bounds[left][0]-bounds[right][1], bounds[left][1]-bounds[right][0]]
    return {"arm_scores": scores, "arm_score_bounds": bounds, "delta_bounds": deltas}


def summarize(worlds, choices):
    support = len(worlds[0]["outcomes"])
    value = bounded_means(worlds, support)
    teacher = certificate(value, list(range(support)))
    outsider = certificate(value, list(range(3, support)))
    agreements = {}
    for name, cert in (("teacher", teacher), ("outsider", outsider)):
        old_choice = choices["teacher" if name == "teacher" else "teacher_outsider"]
        agreements[f"{name}_agreement"] = None if cert["choice"] is None else cert["choice"] == old_choice
    return {"outsider_available": support > 3, "assessment_indices": list(range(support)), "assessment_means_wdl": value["means"],
            "assessment_wdl_bounds": value["means_bounds"], "assessment_unknown_counts": value["unknown_counts"],
            "choices": choices, **scored_choices(value, choices),
            "fresh_selection": {"teacher_certificate": teacher, "outsider_certificate": outsider, **agreements}}


def proposal_identity(record):
    return {"spec": record["spec"], "snapshot": record["snapshot"], "privacy": record["privacy"],
            "candidates": record["candidates"], "old_teacher_choice": record["old_teacher_choice"], "old_means": record["old_means"],
            "proposals": {arm: {k: v for k, v in proposal.items() if k != "timing"} for arm, proposal in record["proposals"].items()}}


def statistics_row(record):
    require("failure" not in record, "Failed root in statistics")
    n = len(record["candidates"]["keys"])
    old_block, assessment, lock = record["old_selection"], record["assessment"], record["selection_lock"]
    require(old_block["candidate_indices"] == assessment["candidate_indices"] == list(range(n)), "Missing recorded candidates")
    expected = world_seeds(record["stage"], record["spec"]["index"])
    require(assessment["world_seeds"] == [w["seed"] for w in assessment["worlds"]] == expected, "Changed assessment namespace")
    require(old_block["world_seeds"] == [w["seed"] for w in old_block["worlds"]]
            and len(set(old_block["world_seeds"])) == 128 and set(old_block["world_seeds"]).isdisjoint(expected), "Old/new world overlap or corruption")
    selection = rebuild_choices(old_block["worlds"], record["proposals"], record["candidates"])
    require(selection["choices"]["teacher"] == record["old_teacher_choice"] and selection["means"] == record["old_means"], "Changed original teacher")
    require(selection == lock["selection_statistics"] and selection["choices"] == lock["choices"], "Changed locked choices")
    require(lock["identity_digest"] == old.digest(proposal_identity(record))
            and lock["old_selection_sha256"] == old.digest(old_block)
            and lock["assessment_indices"] == list(range(n)), "Changed lock evidence")
    require(lock["created_at_ns"] <= record["assessment_started_at_ns"], "Assessment preceded choice lock")
    summary = summarize(assessment["worlds"], selection["choices"])
    require(summary == record["summary"], "Summary differs from saved outcomes")
    return {**{key: record["spec"][key] for key in ("index", "pair", "game_index", "fit", "seat", "phase", "split")},
            "privacy": record["privacy"], "support": n,
            **{key: summary[key] for key in ("outsider_available", "choices", "arm_scores", "arm_score_bounds", "delta_bounds", "fresh_selection")}}


def run_block(state, actions, seeds, block, base):
    started, cpu = time.perf_counter(), time.process_time()
    try:
        require(len(actions) == len(block["candidate_indices"]), "Candidate accounting")
        for seed in seeds:
            c.budget_boundary(base)
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            saved = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []}
            block["worlds"].append(saved)
            legal = old.SplendorEnv().legal_actions(world)
            for action in actions:
                require(action in legal, "Illegal forced candidate")
                outcome = old.trajectory(world, action)
                outcome["wdl"] = old.outcome_value(outcome["winner"], state.current_player) if outcome["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                saved["outcomes"].append(outcome)
                outcome_value(outcome)
    finally:
        block["timing"] = {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu}


def collect(stage, spec, binding, base=OUTPUT):
    index = spec["index"]
    path, seal_path = root_path(stage, index, base), selection_path(stage, index, base)
    require(not path.exists() and not seal_path.exists(), "One-shot root; preserve all evidence")
    started, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "stage": stage, "spec": spec, "time": old.now()}
    try:
        c.budget_boundary(base)
        if "fixture_root" in spec:
            require(stage == "test", "Synthetic source outside isolated fixture")
            source_root = spec["fixture_root"]
        else:
            from . import source
            source_root = source.load_root(spec, base)
        require("failure" not in source_root and not source_root.get("equivalence_proofs"), "Unsupported old root evidence")
        snapshot = source_root["snapshot"]
        require(old.digest(snapshot) == spec["snapshot_sha256"], "Source snapshot hash")
        state = old.deserialize_state_snapshot(snapshot)
        require(not state.terminal and state.current_player == spec["seat"] and state.turn_index == spec["turn"]
                and old.phase(state.turn_index) == spec["phase"], "Wrong source decision")
        record.update(snapshot=snapshot, privacy=baseline.privacy(state), candidates=source_root["candidates"],
                      old_teacher_choice=source_root["choice"], old_means=source_root["means"])
        require(record["privacy"] == source_root["privacy"], "Source privacy changed")
        n = len(record["candidates"]["keys"])
        record["old_selection"] = {"candidate_indices": list(range(n)), "world_seeds": source_root["schedule"]["world_seeds"], "worlds": source_root["worlds"]}
        require(record["old_selection"]["world_seeds"] == [w["seed"] for w in source_root["worlds"]]
                and len(set(record["old_selection"]["world_seeds"])) == 128, "Invalid source world schedule")
        record["proposals"] = frozen_proposals(state, record["candidates"], spec["fit"])
        selection = rebuild_choices(source_root["worlds"], record["proposals"], record["candidates"])
        require(selection["choices"]["teacher"] == source_root["choice"] and selection["means"] == source_root["means"], "Original choice or mean changed")
        record["selection_lock"] = {"manifest": binding, "stage": stage, "index": index,
            "identity_digest": old.digest(proposal_identity(record)), "old_selection_sha256": old.digest(record["old_selection"]),
            "selection_statistics": selection, "choices": selection["choices"], "assessment_indices": list(range(n)),
            "time": old.now(), "created_at_ns": time.time_ns()}
        old.save_new(seal_path, {"manifest": binding, "stage": stage, "index": index, "identity": proposal_identity(record),
                                 "old_selection": record["old_selection"], "selection_lock": record["selection_lock"]})
        record["selection_artifact"] = c.artifact(seal_path)
        record["assessment_started_at_ns"] = time.time_ns()
        seeds = world_seeds(stage, index)
        require(set(seeds).isdisjoint(record["old_selection"]["world_seeds"]), "Old/new world overlap")
        record["assessment"] = {"candidate_indices": list(range(n)), "world_seeds": seeds, "worlds": []}
        actions = [old.decode_action(a) for a in record["candidates"]["actions"]]
        run_block(state, actions, seeds, record["assessment"], base)
        record["summary"] = summarize(record["assessment"]["worlds"], selection["choices"])
    except BaseException as exc:
        if isinstance(exc, AmbiguousDecision):
            record["ambiguous_selection"] = exc.selection
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu}
        old.save_new(path, record)
    return c.artifact(path, index=index, continuations=128*n)
