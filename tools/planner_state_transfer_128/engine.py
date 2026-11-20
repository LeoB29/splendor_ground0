"""Frozen learner transfer on audited planner states, with bounded unknown outcomes."""
from __future__ import annotations

from pathlib import Path
import math
import random
import time
import traceback

import torch

from tools.planner_state_transfer_128 import common as c
from tools.equal_budget_breadth_arena_128_recovery_02 import proof as repetition_proof
from tools.proposal_breadth_128.common import CONFIG as BREADTH_CONFIG
from scripts.analysis.learned_proposal_arena import fourth_slot
from splendor_ai.alphazero import world_budget_160 as world128
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

ROOT, OUTPUT, CONFIG = c.ROOT, c.OUTPUT, c.CONFIG
old, baseline, require = c.old, c.baseline, c.require
ARMS = ("anchor", "broad", "narrow", "teacher")
PROPOSERS = ARMS[:3]
CONTRASTS = (("broad_minus_anchor", "broad", "anchor"),
             ("broad_minus_narrow", "broad", "narrow"),
             ("narrow_minus_anchor", "narrow", "anchor"),
             ("teacher_minus_anchor", "teacher", "anchor"))
_MODELS, _MODEL_ARTIFACTS = {}, {}


class AmbiguousDecision(ValueError):
    """No single original-order choice is guaranteed by the saved bounded outcomes."""
    def __init__(self, selection):
        super().__init__("Unresolved repetition can change an original-order choice")
        self.selection = selection


def initialize_models(base=OUTPUT, registry=None):
    """A private registry; explicit overrides are restricted to isolated fixtures."""
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture registry targets production")
    registry = old.read_json(Path(base)/"models_lock.json") if registry is None else registry
    require(set(registry["models"]) == {f"{arm}_{fit}" for arm in PROPOSERS for fit in c.FITS}, "Incomplete model seal")
    models, artifacts = {}, {}
    for key, item in registry["models"].items():
        require(item["source_fit"] == int(key.split("_")[1]), "Wrong model fit")
        path = c.verify_hash(item)
        models[key], artifacts[key] = StrongPolicyInference(path).model, dict(item)
    old.initialize_worker()
    _MODELS.clear(); _MODELS.update(models)
    _MODEL_ARTIFACTS.clear(); _MODEL_ARTIFACTS.update(artifacts)


def check_stage(stage, index, base=OUTPUT):
    require(stage in ("probe", "main", "test") and type(index) is int and index >= 0, "Invalid root identity")
    if stage == "test":
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture cannot target production")
        require(index < 18, "Fixture outside schedule")
    else:
        require(index < CONFIG["states"][stage], "Root outside frozen schedule")


def root_path(stage, index, base=OUTPUT):
    check_stage(stage, index, base)
    return Path(base)/stage/"roots"/f"state_{index:04d}.json"


def selection_path(stage, index, base=OUTPUT):
    check_stage(stage, index, base)
    return Path(base)/stage/"selections"/f"state_{index:04d}_selection.json"


def world_seeds(stage, index, block):
    require(stage in ("probe", "main", "test") and block in ("selection", "assessment"), "World namespace")
    require(type(index) is int and 0 <= index < (18 if stage == "test" else CONFIG["states"][stage]), "World index")
    offset = {"probe": 0, "main": 1000000, "test": 2000000}[stage]+index*512
    offset += 256 if block == "assessment" else 0
    return [CONFIG["world_seed_base"]+offset+i for i in range(128)]


def learned_proposal(state, stream, source_model):
    require(source_model in _MODELS and source_model in _MODEL_ARTIFACTS, "Unsealed proposer")
    started = time.perf_counter()
    legal = old.SplendorEnv().legal_actions(state)
    schedule = world128.root_schedule(state, stream, 128)
    reference = baseline.candidate_set(state, schedule["candidate_nonce"])
    groups = old.actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    reps = {repr(key): old.serialize_action(action) for key, action in groups.items()}
    reps.update(zip(reference["keys"], reference["actions"], strict=True))
    model = _MODELS[source_model]
    inference_start = time.perf_counter()
    with torch.no_grad():
        row, encoded_keys, _ = encode_position(state, model.config, legal)
        logits, _ = model(row["observation"][None], row["features"][None], torch.ones((1,len(legal)), dtype=torch.bool))
    require([repr(key) for key in encoded_keys] == keys and torch.isfinite(logits).all().item(), "Invalid proposer inputs")
    values = [float(torch.amax(logits[0][row["groups"] == j])) for j in range(len(keys))]
    extra = time.perf_counter()-inference_start
    offered = fourth_slot(keys, reference["keys"], values)
    identity = _MODEL_ARTIFACTS[source_model]
    return {"schedule": schedule, "reference": reference,
            "candidates": {"keys": offered, "actions": [reps[key] for key in offered],
                           "canonical_support": len(keys), "legal_support": len(legal)},
            "model_id": source_model, "model_sha256": identity["sha256"], "fit": identity["source_fit"],
            "group_keys": keys, "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": logits[0].tolist(), "learned_group_scores": values,
            "timing": {"wall": time.perf_counter()-started, "learned_extra_wall": extra}}


def assemble_union(reference, proposals, group_keys, representatives, identity):
    require(set(proposals) == set(PROPOSERS), "Incomplete proposer arms")
    require(isinstance(identity, (tuple, list)) and len(identity) == 2, "Missing proposal identity")
    stage, index = identity
    require(stage in ("probe", "main", "test") and type(index) is int and index >= 0, "Invalid proposal identity")
    require(len(set(group_keys)) == len(group_keys) and set(representatives) == set(group_keys), "Canonical support")
    require(len(reference["keys"]) == min(4,len(group_keys)), "Reference support")
    first, by_arm = list(reference["keys"][:3]), {}
    require(first and len(first) == len(set(first)), "Common proposal support")
    for arm in PROPOSERS:
        value = proposals[arm]["candidates"]
        keys = value["keys"]
        require(len(keys) == min(4,len(group_keys)) and keys[:3] == first and len(set(keys)) == len(keys), "Changed common candidates")
        require(all(representatives[k] == a for k,a in zip(keys,value["actions"],strict=True)), "Wrong representative")
        by_arm[arm] = list(keys)
    scores = proposals["anchor"]["learned_group_scores"]
    require(len(scores) == len(group_keys) and all(type(v) in (int, float) and math.isfinite(v) for v in scores),
            "Invalid anchor group scores")
    anchor = by_arm["anchor"]
    remaining = [j for j, key in enumerate(group_keys) if key not in anchor]
    ranked = sorted(remaining, key=lambda j: (-scores[j], j))[:min(2, min(8, len(group_keys))-len(anchor))]
    available = [j for j in remaining if j not in ranked]
    uniform = sorted(available, key=lambda j: (old.digest([BREADTH_CONFIG["random_tag"], stage, index, group_keys[j]]), j))
    uniform = uniform[:min(8, len(group_keys))-len(anchor)-len(ranked)]
    by_arm["teacher"] = anchor + [group_keys[j] for j in ranked+uniform]
    union = list(first)
    for arm in ARMS:
        for key in by_arm[arm]:
            if key not in union:
                union.append(key)
    require(1 <= len(union) <= CONFIG["maximum_candidates"], "Union exceeds ten candidates")
    candidates = {arm: {"keys": keys, "actions": [representatives[k] for k in keys],
                        "indices": [union.index(k) for k in keys]} for arm,keys in by_arm.items()}
    proposed = {arm: union.index(keys[3]) if len(keys) >= 4 else None for arm,keys in by_arm.items()}
    return {"candidates": candidates, "union": {"keys": union, "actions": [representatives[k] for k in union]},
            "proposed_indices": proposed,
            "teacher_added_indices": {"ranked": [union.index(group_keys[j]) for j in ranked],
                                      "random": [union.index(group_keys[j]) for j in uniform]}}


def proposal_union(state, stream, fit, identity=None):
    require(fit in c.FITS, "Unknown fit")
    proposals = {arm: learned_proposal(state,stream,f"{arm}_{fit}") for arm in PROPOSERS}
    reference = proposals["anchor"]["reference"]
    require(all(p["reference"] == reference for p in proposals.values()), "Mismatched reference")
    groups = old.actions_by_canonical_successor_key(state,old.SplendorEnv().legal_actions(state))
    keys = [repr(k) for k in groups]
    reps = {repr(k):old.serialize_action(a) for k,a in groups.items()}
    reps.update(zip(reference["keys"],reference["actions"],strict=True))
    return {"proposals": proposals, **assemble_union(reference,proposals,keys,reps,identity)}


def outcome_value(outcome):
    if outcome["reason"] == "repetition_cutoff":
        require(outcome["winner"] is None and outcome["wdl"] is None, "Censored repetition has invented result")
        return None
    value = outcome["wdl"]
    require(outcome["reason"] in old.VALID_NATURAL_TERMINATIONS and type(value) in (int, float)
            and value in (-1., 0., 1.), "Invalid continuation ending")
    return int(value)


def bounded_means(worlds, support):
    require(len(worlds) == 128 and type(support) is int and 1 <= support <= CONFIG["maximum_candidates"],
            "Incorrect world budget")
    require(all(len(w["outcomes"]) == support for w in worlds), "Missing or extra candidate outcome")
    sums, missing, cells = [0]*support, [0]*support, []
    for wi, world in enumerate(worlds):
        for ci, outcome in enumerate(world["outcomes"]):
            value = outcome_value(outcome)
            if value is None:
                missing[ci] += 1
                cells.append({"world": wi, "candidate": ci})
            else:
                sums[ci] += value
    return {"means": [sums[j]/128 if missing[j] == 0 else None for j in range(support)],
            "means_bounds": [[(sums[j]-missing[j])/128, (sums[j]+missing[j])/128] for j in range(support)],
            "sums_wdl": sums, "unknown_counts": missing, "unresolved_cells": cells}


def choose(worlds, candidates, support, equivalence_proofs=None):
    """Certify one original-order maximum per arm for all admissible unknown WDLs."""
    require(set(candidates) == set(ARMS), "Incomplete selection arms")
    value = bounded_means(worlds, support)
    unresolved = value["unresolved_cells"]
    if equivalence_proofs is None:
        groups = [{"context_sha256": None, "cells": [cell]} for cell in unresolved]
    else:
        require([{"world": p["world"], "candidate": p["candidate"]} for p in equivalence_proofs] == unresolved,
                "Missing or reordered equivalence proofs")
        by_context = {}
        for item in equivalence_proofs:
            require(item["context_sha256"] == old.digest(item["context"]), "Changed equivalence context hash")
            by_context.setdefault(item["context_sha256"], []).append({"world": item["world"], "candidate": item["candidate"]})
        groups = [{"context_sha256": key, "cells": by_context[key]} for key in sorted(by_context)]
    coefficients = [[sum(cell["candidate"] == j for cell in group["cells"]) for j in range(support)] for group in groups]
    choices, certificates = {}, {}
    for arm in ARMS:
        indices = candidates[arm]["indices"]
        require(indices and len(set(indices)) == len(indices)
                and all(type(i) is int and 0 <= i < support for i in indices), "Invalid arm support")
        winners = []
        margins_by_candidate = {}
        for order, candidate in enumerate(indices):
            margins = [0 if candidate == other else value["sums_wdl"][candidate]-value["sums_wdl"][other]
                       -sum(abs(a[candidate]-a[other]) for a in coefficients) for other in indices]
            if all(margin > 0 if j < order else margin >= 0 for j, margin in enumerate(margins) if j != order):
                winners.append(candidate)
                margins_by_candidate[candidate] = margins
        require(len(winners) <= 1, "Nonunique bounded selection")
        choices[arm] = winners[0] if winners else None
        certificates[arm] = {"indices": indices, "choice": choices[arm],
                             "pairwise_lower_numerators": margins_by_candidate.get(choices[arm])}
    value.update(choices=choices, certificate={
        "rule": "coupled_repetition_original_order_v1" if equivalence_proofs is not None else "bounded_repetition_original_order_v1",
        "unknown_groups": groups, "arm_certificates": certificates})
    if any(choice is None for choice in choices.values()):
        raise AmbiguousDecision(value)
    return value


def proof_record(record):
    """Adapt this common-world block to the previously proved exact replay schema."""
    block = record["selection"]
    return {"snapshot": record["snapshot"], "schedule": {"world_seeds": block["world_seeds"]},
            "worlds": [{**world, "candidate_indices": block["candidate_indices"]} for world in block["worlds"]],
            "candidates": record["union"]}


def selection_for_record(record, base=OUTPUT, build_proofs=False):
    support = len(record["union"]["keys"])
    proofs = record.get("equivalence_proofs")
    if proofs is not None:
        repetition_proof.validate(proof_record(record), proofs)
        return choose(record["selection"]["worlds"], record["candidates"], support, proofs)
    try:
        return choose(record["selection"]["worlds"], record["candidates"], support)
    except AmbiguousDecision as exc:
        if not build_proofs:
            raise
        unresolved = exc.selection["unresolved_cells"]
    started, cpu = time.perf_counter(), time.process_time()
    record["equivalence_proofs"] = proofs = []
    adapted = proof_record(record)
    try:
        for cell in unresolved:
            c.budget_boundary(base)
            proofs.append(repetition_proof.build(adapted, cell["world"], cell["candidate"]))
        repetition_proof.validate(adapted, proofs)
        return choose(record["selection"]["worlds"], record["candidates"], support, proofs)
    finally:
        record["proof_timing"] = {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu,
                                  "replayed_continuations": len(proofs)}


def summarize(worlds, assessment_indices, choices, proposed_indices, support):
    require(set(choices) == set(proposed_indices) == set(ARMS), "Incomplete summary arms")
    require(all(type(i) is int and 0 <= i < support for i in choices.values()), "Choice outside union")
    require(all(i is None or type(i) is int and 0 <= i < support for i in proposed_indices.values()), "Fourth outside union")
    require(assessment_indices == sorted(set(choices.values()))
            and len(assessment_indices) <= CONFIG["maximum_assessed_choices"], "Assessment differs from locked choices")
    value = bounded_means(worlds, len(assessment_indices))
    by_union = dict(zip(assessment_indices, value["means"], strict=True))
    bounds = dict(zip(assessment_indices, value["means_bounds"], strict=True))
    score_bounds = {arm: [(1+v)/2 for v in bounds[i]] for arm, i in choices.items()}
    deltas = {name: [0., 0.] if choices[left] == choices[right] else
              [score_bounds[left][0]-score_bounds[right][1], score_bounds[left][1]-score_bounds[right][0]]
              for name, left, right in CONTRASTS}
    return {"assessment_indices": assessment_indices, "assessment_means_wdl": value["means"],
            "assessment_wdl_bounds": value["means_bounds"], "assessment_unknown_counts": value["unknown_counts"],
            "arm_wdl": {arm: by_union[i] for arm, i in choices.items()},
            "arm_scores": {arm: (1+by_union[i])/2 if by_union[i] is not None else None for arm, i in choices.items()},
            "arm_score_bounds": score_bounds, "delta_bounds": deltas,
            "choices": choices, "proposed_indices": proposed_indices}


def statistics_row(record):
    require("failure" not in record, "Failed root in statistics")
    support = len(record["union"]["keys"])
    for name in ("selection", "assessment"):
        expected = world_seeds(record["stage"], record["spec"]["index"], name)
        require(record[name]["world_seeds"] == expected and [w["seed"] for w in record[name]["worlds"]] == expected,
                "Changed world schedule")
    require(record["selection"]["candidate_indices"] == list(range(support)), "Selection omitted candidate")
    selection = selection_for_record(record)
    lock = record["selection_lock"]
    require(selection == lock["selection_statistics"] and selection["choices"] == lock["choices"], "Changed locked selection")
    require(lock["identity_digest"] == old.digest(proposal_identity(record))
            and lock["selection_sha256"] == old.digest(record["selection"])
            and lock["proofs_sha256"] == old.digest(record.get("equivalence_proofs", [])), "Changed selection evidence")
    indices = record["assessment"]["candidate_indices"]
    require(indices == lock["assessment_indices"], "Changed locked assessment support")
    result = summarize(record["assessment"]["worlds"], indices, selection["choices"], record["proposed_indices"], support)
    require(result == record["summary"], "Stored summary differs from raw assessment")
    return {**{k: record["spec"][k] for k in ("index", "pair", "game_index", "fit", "seat", "phase", "source_arm")},
            "privacy": record["privacy"],
            **{k: result[k] for k in ("arm_scores", "arm_score_bounds", "delta_bounds", "choices", "proposed_indices")}}


def proposal_identity(record):
    return {"spec":record["spec"],"snapshot":record["snapshot"],"privacy":record["privacy"],
            "proposals":{a:{k:v for k,v in p.items() if k != "timing"} for a,p in record["proposals"].items()},
            "candidates":record["candidates"],"union":record["union"],"proposed_indices":record["proposed_indices"],
            "teacher_added_indices": record["teacher_added_indices"]}


def run_block(state, actions, seeds, block, base):
    start,cpu = time.perf_counter(),time.process_time()
    c.budget_boundary(base)
    require(len(actions) == len(block["candidate_indices"]),"Sparse candidate accounting")
    for seed in seeds:
        c.budget_boundary(base)
        world = old.determinize_for_observer(state,state.current_player,random.Random(seed))
        saved = {"seed":seed,"state_sha256":old.digest(old.serialize_state_snapshot(world)),"outcomes":[]}
        block["worlds"].append(saved)
        legal = old.SplendorEnv().legal_actions(world)
        for action in actions:
            require(action in legal,"Illegal forced candidate")
            outcome = old.trajectory(world,action)
            outcome["wdl"] = old.outcome_value(outcome["winner"],state.current_player) if outcome["reason"] in old.VALID_NATURAL_TERMINATIONS else None
            saved["outcomes"].append(outcome)
            outcome_value(outcome)
    block["timing"] = {"wall":time.perf_counter()-start,"cpu":time.process_time()-cpu}


def collect(stage, spec, binding, base=OUTPUT):
    index = spec["index"]
    path,seal_path = root_path(stage,index,base),selection_path(stage,index,base)
    require(not path.exists() and not seal_path.exists(),"One-shot root; preserve existing evidence")
    started,cpu = time.perf_counter(),time.process_time()
    record = {"manifest":binding,"stage":stage,"spec":spec,"time":old.now()}
    try:
        c.budget_boundary(base)
        if "fixture_snapshot" in spec:
            require(stage == "test","Synthetic source outside isolated test")
            snapshot = spec["fixture_snapshot"]
        else:
            from tools.planner_state_transfer_128 import source
            snapshot = source.load_snapshot(spec, base)
        require(old.digest(snapshot) == spec["snapshot_sha256"],"Source snapshot hash")
        state = old.deserialize_state_snapshot(snapshot)
        require(not state.terminal and state.turn_index == spec["turn"] and state.current_player == spec["seat"]
                and old.phase(state.turn_index) == spec["phase"],"Wrong source decision")
        record.update(snapshot=snapshot,privacy=baseline.privacy(state))
        record.update(proposal_union(state,spec["search_stream"],spec["fit"],(stage,index)))
        actions = [old.decode_action(a) for a in record["union"]["actions"]]
        record["selection"] = {"candidate_indices":list(range(len(actions))),"world_seeds":world_seeds(stage,index,"selection"),"worlds":[]}
        run_block(state,actions,record["selection"]["world_seeds"],record["selection"],base)
        selection = selection_for_record(record, base, build_proofs=True)
        choices = selection["choices"]
        indices = sorted(set(choices.values()))
        record["selection_lock"] = {"manifest":binding,"stage":stage,"index":index,
             "identity_digest":old.digest(proposal_identity(record)),"selection_sha256":old.digest(record["selection"]),
             "selection_statistics":selection,"proofs_sha256":old.digest(record.get("equivalence_proofs", [])),
             "value_scale":"terminal_wdl","choices":choices,"assessment_indices":indices,
             "time":old.now(),"created_at_ns":time.time_ns()}
        old.save_new(seal_path,{"manifest":binding,"stage":stage,"index":index,"identity":proposal_identity(record),
                               "selection":record["selection"],"equivalence_proofs":record.get("equivalence_proofs", []),
                               "selection_lock":record["selection_lock"]})
        record["selection_artifact"] = c.artifact(seal_path)
        record["assessment_started_at_ns"] = time.time_ns()
        record["assessment"] = {"candidate_indices":indices,"world_seeds":world_seeds(stage,index,"assessment"),"worlds":[]}
        require(set(record["selection"]["world_seeds"]).isdisjoint(record["assessment"]["world_seeds"]),"World blocks overlap")
        run_block(state,[actions[i] for i in indices],record["assessment"]["world_seeds"],record["assessment"],base)
        record["summary"] = summarize(record["assessment"]["worlds"],indices,choices,record["proposed_indices"],len(actions))
    except BaseException as exc:
        if isinstance(exc, AmbiguousDecision):
            record["ambiguous_selection"] = exc.selection
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall":time.perf_counter()-started,"cpu":time.process_time()-cpu}
        old.save_new(path,record)
    return c.artifact(path,index=index,continuations=128*(len(actions)+len(indices)),
                      proof_continuations=len(record.get("equivalence_proofs", [])))
