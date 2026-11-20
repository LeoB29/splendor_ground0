"""Three fixed proposers; full selection and sparse, independent assessment."""
from __future__ import annotations

from pathlib import Path
import random
import time
import traceback

import torch

from tools.proposal_specialist_128 import common as c
from scripts.analysis.learned_proposal_arena import fourth_slot
from splendor_ai.alphazero import world_budget_160 as world128
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

ROOT, OUTPUT, CONFIG = c.ROOT, c.OUTPUT, c.CONFIG
old, baseline, require = c.old, c.baseline, c.require
ARMS = ("anchor", "specialist", "control")
PROPOSERS = ARMS
_MODELS, _MODEL_ARTIFACTS = {}, {}


def initialize_models(base=OUTPUT, registry=None):
    """A private registry; explicit overrides are restricted to isolated fixtures."""
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture registry targets production")
    registry = old.read_json(Path(base)/"models_lock.json") if registry is None else registry
    require(set(registry["models"]) == {f"{arm}_{fit}" for arm in ARMS for fit in c.FITS}, "Incomplete model seal")
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


def assemble_union(reference, proposals, group_keys, representatives):
    require(set(proposals) == set(ARMS), "Incomplete proposer arms")
    require(len(set(group_keys)) == len(group_keys) and set(representatives) == set(group_keys), "Canonical support")
    require(len(reference["keys"]) == min(4,len(group_keys)), "Reference support")
    first, by_arm = list(reference["keys"][:3]), {}
    require(first and len(first) == len(set(first)), "Common proposal support")
    for arm in ARMS:
        value = proposals[arm]["candidates"]
        keys = value["keys"]
        require(len(keys) == min(4,len(group_keys)) and keys[:3] == first and len(set(keys)) == len(keys), "Changed common candidates")
        require(all(representatives[k] == a for k,a in zip(keys,value["actions"],strict=True)), "Wrong representative")
        by_arm[arm] = list(keys)
    union = list(first)
    for arm in ARMS:
        for key in by_arm[arm]:
            if key not in union:
                union.append(key)
    require(1 <= len(union) <= 6, "Union exceeds six candidates")
    candidates = {arm: {"keys": keys, "actions": [representatives[k] for k in keys],
                        "indices": [union.index(k) for k in keys]} for arm,keys in by_arm.items()}
    proposed = {arm: union.index(keys[3]) if len(keys) == 4 else None for arm,keys in by_arm.items()}
    return {"candidates": candidates, "union": {"keys": union, "actions": [representatives[k] for k in union]},
            "proposed_indices": proposed}


def proposal_union(state, stream, fit, identity=None):
    require(fit in c.FITS, "Unknown fit")
    proposals = {arm: learned_proposal(state,stream,f"{arm}_{fit}") for arm in ARMS}
    reference = proposals["anchor"]["reference"]
    require(all(p["reference"] == reference for p in proposals.values()), "Mismatched reference")
    groups = old.actions_by_canonical_successor_key(state,old.SplendorEnv().legal_actions(state))
    keys = [repr(k) for k in groups]
    reps = {repr(k):old.serialize_action(a) for k,a in groups.items()}
    reps.update(zip(reference["keys"],reference["actions"],strict=True))
    return {"proposals": proposals, **assemble_union(reference,proposals,keys,reps)}


def means(worlds, support):
    require(len(worlds) == 128 and type(support) is int and 1 <= support <= 6, "Incorrect world budget")
    require(all(len(w["outcomes"]) == support for w in worlds), "Missing or extra candidate outcome")
    columns = [[w["outcomes"][j]["wdl"] for w in worlds] for j in range(support)]
    require(all(type(v) in (int,float) and v in (-1.,0.,1.) for col in columns for v in col), "Invalid terminal value")
    return [sum(col)/128 for col in columns]


def choose(worlds, candidates, support):
    require(set(candidates) == set(ARMS), "Incomplete selection arms")
    values, choices = means(worlds,support), {}
    for arm in ARMS:
        indices = candidates[arm]["indices"]
        require(indices and len(set(indices)) == len(indices) and all(type(i) is int and 0 <= i < support for i in indices), "Invalid arm support")
        choices[arm] = max(indices,key=lambda i:values[i])
    return values, choices


def summarize(worlds, assessment_indices, choices, proposed_indices, support):
    require(set(choices) == set(proposed_indices) == set(ARMS), "Incomplete summary arms")
    require(all(type(i) is int and 0 <= i < support for i in choices.values()), "Choice outside union")
    require(all(i is None or type(i) is int and 0 <= i < support for i in proposed_indices.values()), "Fourth outside union")
    require(assessment_indices == sorted(set(choices.values())) and len(assessment_indices) <= 3, "Assessment differs from locked choices")
    values = means(worlds,len(assessment_indices))
    by_union = dict(zip(assessment_indices,values,strict=True))
    return {"assessment_indices":assessment_indices, "assessment_means_wdl":values,
            "arm_wdl":{arm:by_union[i] for arm,i in choices.items()},
            "arm_scores":{arm:(1+by_union[i])/2 for arm,i in choices.items()},
            "choices":choices, "proposed_indices":proposed_indices}


def statistics_row(record):
    require("failure" not in record,"Failed root in statistics")
    support = len(record["union"]["keys"])
    require(record["selection"]["candidate_indices"] == list(range(support)),"Selection omitted candidate")
    values, choices = choose(record["selection"]["worlds"],record["candidates"],support)
    require(values == record["selection_lock"]["means"] and choices == record["selection_lock"]["choices"],"Changed locked selection")
    result = summarize(record["assessment"]["worlds"],record["assessment"]["candidate_indices"],choices,record["proposed_indices"],support)
    require(result == record["summary"],"Stored summary differs from raw assessment")
    return {**{k:record["spec"][k] for k in ("index","pair","game_index","fit","seat","phase")},
            "privacy":record["privacy"],
            **{k:result[k] for k in ("arm_scores","choices","proposed_indices")}}


def proposal_identity(record):
    return {"spec":record["spec"],"snapshot":record["snapshot"],"privacy":record["privacy"],
            "proposals":{a:{k:v for k,v in p.items() if k != "timing"} for a,p in record["proposals"].items()},
            "candidates":record["candidates"],"union":record["union"],"proposed_indices":record["proposed_indices"]}


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
            require(outcome["wdl"] is not None,"Invalid continuation ending")
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
            game,receipt = c.verified(spec["source_game"]),c.verified(spec["source_audit"])
            require(game["manifest"] == receipt["manifest"] == binding and receipt["passed"] and "failure" not in game,"Invalid audited source")
            require(receipt["game"] == {k:spec["source_game"][k] for k in ("path","sha256")},"Source receipt link")
            source_spec = game["spec"]
            require(receipt["spec"] == source_spec and spec["pair"] == spec["game_index"] == source_spec["index"],"Wrong source game identity")
            require(spec["source_stage"] == source_spec["stage"] and all(spec[k] == source_spec[k] for k in ("fit","seat","phase","search_stream")),"Wrong source cell")
            require(stage == "test" or spec["source_stage"] == stage,"Cross-stage source")
            moves = [m for m in game["trace"] if m["turn"] == spec["turn"]]
            require(len(moves) == 1,"Missing source turn")
            snapshot = moves[0]["snapshot"]
        require(old.digest(snapshot) == spec["snapshot_sha256"],"Source snapshot hash")
        state = old.deserialize_state_snapshot(snapshot)
        require(not state.terminal and state.turn_index == spec["turn"] and state.current_player == spec["seat"]
                and old.phase(state.turn_index) == spec["phase"],"Wrong source decision")
        record.update(snapshot=snapshot,privacy=baseline.privacy(state))
        record.update(proposal_union(state,spec["search_stream"],spec["fit"]))
        actions = [old.decode_action(a) for a in record["union"]["actions"]]
        record["selection"] = {"candidate_indices":list(range(len(actions))),"world_seeds":world_seeds(stage,index,"selection"),"worlds":[]}
        run_block(state,actions,record["selection"]["world_seeds"],record["selection"],base)
        values,choices = choose(record["selection"]["worlds"],record["candidates"],len(actions))
        indices = sorted(set(choices.values()))
        record["selection_lock"] = {"manifest":binding,"stage":stage,"index":index,
             "identity_digest":old.digest(proposal_identity(record)),"selection_sha256":old.digest(record["selection"]),
             "means":values,"value_scale":"terminal_wdl","choices":choices,"assessment_indices":indices,
             "time":old.now(),"created_at_ns":time.time_ns()}
        old.save_new(seal_path,{"manifest":binding,"stage":stage,"index":index,"identity":proposal_identity(record),
                               "selection":record["selection"],"selection_lock":record["selection_lock"]})
        record["selection_artifact"] = c.artifact(seal_path)
        record["assessment_started_at_ns"] = time.time_ns()
        record["assessment"] = {"candidate_indices":indices,"world_seeds":world_seeds(stage,index,"assessment"),"worlds":[]}
        require(set(record["selection"]["world_seeds"]).isdisjoint(record["assessment"]["world_seeds"]),"World blocks overlap")
        run_block(state,[actions[i] for i in indices],record["assessment"]["world_seeds"],record["assessment"],base)
        record["summary"] = summarize(record["assessment"]["worlds"],indices,choices,record["proposed_indices"],len(actions))
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall":time.perf_counter()-started,"cpu":time.process_time()-cpu}
        old.save_new(path,record)
    return c.artifact(path,index=index,continuations=128*(len(actions)+len(indices)))
