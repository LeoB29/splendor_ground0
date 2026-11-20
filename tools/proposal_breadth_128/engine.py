"""Fixed broader proposals versus an exactly matched sampling budget."""
from __future__ import annotations

import math
from pathlib import Path
import random
import time
import traceback

import torch

from tools.proposal_breadth_128 import common as c
from scripts.analysis.learned_proposal_arena import fourth_slot
from splendor_ai.alphazero import world_budget_160 as world128
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

ROOT, OUTPUT, CONFIG = c.ROOT, c.OUTPUT, c.CONFIG
old, baseline, require = c.old, c.baseline, c.require
ARMS = ("anchor", "breadth", "precision")
_MODELS, _MODEL_ARTIFACTS = {}, {}


def initialize_models(base=OUTPUT, registry=None):
    """Load three frozen anchors without mutating any ancestor model registry."""
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture registry targets production")
    registry = old.read_json(Path(base)/"models_lock.json") if registry is None else registry
    require(set(registry["models"]) == {f"anchor_{fit}" for fit in c.FITS}, "Incomplete anchor seal")
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
    """The extension reserves128 draws; each root uses only its required prefix."""
    require(stage in ("probe", "main", "test") and block in ("selection", "extension", "assessment"), "World namespace")
    require(type(index) is int and 0 <= index < (18 if stage == "test" else CONFIG["states"][stage]), "World index")
    offset = {"probe": 0, "main": 1000000, "test": 2000000}[stage]+index*1024
    offset += {"selection":0, "extension":128, "assessment":512}[block]
    return [CONFIG["world_seed_base"]+offset+i for i in range(128)]


def budgets(canonical_support):
    require(type(canonical_support) is int and canonical_support > 0, "Invalid canonical support")
    m, n = min(canonical_support,4), min(canonical_support,8)
    require((128*n) % m == 0, "Nonintegral precision budget")
    precision = 128*n//m
    return {"canonical_support":canonical_support, "anchor_candidates":m, "breadth_candidates":n,
            "anchor_worlds":128, "breadth_worlds":128, "precision_worlds":precision,
            "extension_worlds":precision-128,
            "arm_continuations":{"anchor":128*m, "breadth":128*n, "precision":precision*m}}


def learned_proposal(state, stream, source_model):
    require(source_model in _MODELS and source_model in _MODEL_ARTIFACTS, "Unsealed proposer")
    started = time.perf_counter()
    legal = old.SplendorEnv().legal_actions(state)
    schedule = world128.root_schedule(state, stream, 128)
    reference = baseline.candidate_set(state, schedule["candidate_nonce"])
    groups = old.actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    reps = {repr(key):old.serialize_action(action) for key,action in groups.items()}
    reps.update(zip(reference["keys"],reference["actions"],strict=True))
    model = _MODELS[source_model]
    inference_start = time.perf_counter()
    with torch.no_grad():
        row, encoded_keys, _ = encode_position(state,model.config,legal)
        logits,_ = model(row["observation"][None],row["features"][None],torch.ones((1,len(legal)),dtype=torch.bool))
    require([repr(key) for key in encoded_keys] == keys and torch.isfinite(logits).all().item(), "Invalid proposer inputs")
    values = [float(torch.amax(logits[0][row["groups"] == j])) for j in range(len(keys))]
    extra = time.perf_counter()-inference_start
    offered = fourth_slot(keys,reference["keys"],values)
    identity = _MODEL_ARTIFACTS[source_model]
    return {"schedule":schedule, "reference":reference,
            "candidates":{"keys":offered, "actions":[reps[key] for key in offered],
                          "canonical_support":len(keys), "legal_support":len(legal)},
            "model_id":source_model, "model_sha256":identity["sha256"], "fit":identity["source_fit"],
            "group_keys":keys, "legal_actions":[old.serialize_action(a) for a in legal],
            "learned_logits":logits[0].tolist(), "learned_group_scores":values,
            "timing":{"wall":time.perf_counter()-started, "learned_extra_wall":extra}}


def assemble_union(proposal, representatives, identity):
    """Keep the anchor intact; append ranked then uniform distinct challengers."""
    require(isinstance(identity,(tuple,list)) and len(identity) == 2, "Missing proposal identity")
    stage,index = identity
    require(stage in ("probe","main","test") and type(index) is int and index >= 0, "Invalid proposal identity")
    keys, scores = proposal["group_keys"], proposal["learned_group_scores"]
    require(keys and len(set(keys)) == len(keys) == len(scores) and set(representatives) == set(keys), "Canonical support")
    require(all(isinstance(v,(float,int)) and math.isfinite(v) for v in scores), "Invalid group score")
    budget = budgets(len(keys))
    anchor = proposal["candidates"]
    base_keys, base_actions = anchor["keys"], anchor["actions"]
    require(len(base_keys) == len(base_actions) == len(set(base_keys)) == budget["anchor_candidates"]
            and set(base_keys).issubset(keys), "Invalid original candidates")
    require(anchor["canonical_support"] == len(keys), "Original support mismatch")
    remaining = [j for j,key in enumerate(keys) if key not in base_keys]
    ranked = sorted(remaining,key=lambda j:(-scores[j],j))[:min(2,budget["breadth_candidates"]-len(base_keys))]
    available = [j for j in remaining if j not in ranked]
    random_order = sorted(available,key=lambda j:(old.digest([CONFIG["random_tag"],stage,index,keys[j]]),j))
    uniform = random_order[:budget["breadth_candidates"]-len(base_keys)-len(ranked)]
    added = [keys[j] for j in ranked+uniform]
    union_keys = list(base_keys)+added
    union_actions = list(base_actions)+[representatives[key] for key in added]
    n,m = len(union_keys),len(base_keys)
    require(n == budget["breadth_candidates"] and len(set(union_keys)) == n, "Incorrect broader support")
    candidates = {arm:{"keys":list(union_keys[:(n if arm == "breadth" else m)]),
                      "actions":list(union_actions[:(n if arm == "breadth" else m)]),
                      "indices":list(range(n if arm == "breadth" else m))} for arm in ARMS}
    return {"candidates":candidates, "union":{"keys":union_keys,"actions":union_actions},
            "proposed_indices":{arm:3 if m == 4 else None for arm in ARMS},
            "added_indices":{"ranked":list(range(m,m+len(ranked))),"random":list(range(m+len(ranked),n))},
            "budgets":budget}


def proposal_union(state, stream, fit, identity):
    require(fit in c.FITS, "Unknown fit")
    proposal = learned_proposal(state,stream,f"anchor_{fit}")
    groups = old.actions_by_canonical_successor_key(state,old.SplendorEnv().legal_actions(state))
    reps = {repr(key):old.serialize_action(action) for key,action in groups.items()}
    return {"proposals":{"anchor":proposal}, **assemble_union(proposal,reps,identity)}


def means(worlds, support, world_count=128):
    require(type(world_count) is int and world_count > 0 and len(worlds) == world_count
            and type(support) is int and 1 <= support <= 8, "Incorrect world budget")
    require(all(len(w["outcomes"]) == support for w in worlds), "Missing or extra candidate outcome")
    columns = [[w["outcomes"][j]["wdl"] for w in worlds] for j in range(support)]
    require(all(type(v) in (int,float) and v in (-1.,0.,1.) for col in columns for v in col), "Invalid terminal value")
    return [sum(col)/world_count for col in columns]


def choose(selection, extension, candidates, budget):
    require(budget == budgets(budget["canonical_support"]), "Changed compute budget")
    require(set(candidates) == set(ARMS), "Incomplete selection arms")
    n,m,e = budget["breadth_candidates"],budget["anchor_candidates"],budget["extension_worlds"]
    require(selection["candidate_indices"] == list(range(n)), "Selection omitted candidate")
    require(extension["candidate_indices"] == list(range(m)), "Extension omitted original candidate")
    for arm in ARMS:
        require(candidates[arm]["indices"] == list(range(n if arm == "breadth" else m)), "Changed arm support/order")
    common = means(selection["worlds"],n)
    require(len(extension["worlds"]) == e, "Incorrect extension world budget")
    if e:
        extra = means(extension["worlds"],m,e)
        # Sum integer WDL directly; averaging two differently sized blocks is wrong.
        precision = [(sum(w["outcomes"][j]["wdl"] for w in selection["worlds"])
                     + sum(w["outcomes"][j]["wdl"] for w in extension["worlds"]))
                     / budget["precision_worlds"] for j in range(m)]
        require(len(extra) == m, "Invalid extension support")
    else:
        precision = list(common[:m])
    values = {"anchor":common[:m],"breadth":common,"precision":precision}
    choices = {arm:max(candidates[arm]["indices"],key=lambda i:values[arm][i]) for arm in ARMS}
    return values,choices


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
            "choices":choices,"proposed_indices":proposed_indices}


def statistics_row(record):
    """Reconstruct all arm choices and scores from the stored block outcomes."""
    require("failure" not in record, "Failed root in statistics")
    for block in ("selection","extension","assessment"):
        expected = world_seeds(record["stage"],record["spec"]["index"],block)
        if block == "extension":
            expected = expected[:record["budgets"]["extension_worlds"]]
        value = record[block]
        require(value["world_seeds"] == expected and [w["seed"] for w in value["worlds"]] == expected, "Changed world schedule")
    values,choices = choose(record["selection"],record["extension"],record["candidates"],record["budgets"])
    lock = record["selection_lock"]
    require(values == lock["means"] and choices == lock["choices"], "Changed locked selection")
    require(lock["identity_digest"] == old.digest(proposal_identity(record))
            and lock["selection_sha256"] == old.digest(record["selection"])
            and lock["extension_sha256"] == old.digest(record["extension"]), "Changed selection evidence")
    indices = record["assessment"]["candidate_indices"]
    require(indices == lock["assessment_indices"], "Changed locked assessment support")
    result = summarize(record["assessment"]["worlds"],indices,choices,record["proposed_indices"],len(record["union"]["keys"]))
    require(result == record["summary"], "Stored summary differs from raw assessment")
    return {**{k:record["spec"][k] for k in ("index","pair","game_index","fit","seat","phase")},
            "privacy":record["privacy"],
            **{k:result[k] for k in ("arm_scores","choices","proposed_indices")},
            "added_indices":record["added_indices"],"budgets":record["budgets"]}


def proposal_identity(record):
    return {"spec":record["spec"],"snapshot":record["snapshot"],"privacy":record["privacy"],
            "proposals":{a:{k:v for k,v in p.items() if k != "timing"} for a,p in record["proposals"].items()},
            **{k:record[k] for k in ("candidates","union","proposed_indices","added_indices","budgets")}}


def run_block(state, actions, seeds, block, base):
    start,cpu = time.perf_counter(),time.process_time()
    c.budget_boundary(base)
    require(len(actions) == len(block["candidate_indices"]), "Sparse candidate accounting")
    require(seeds == block["world_seeds"] and not block["worlds"], "Changed block schedule")
    for seed in seeds:
        c.budget_boundary(base)
        world = old.determinize_for_observer(state,state.current_player,random.Random(seed))
        saved = {"seed":seed,"state_sha256":old.digest(old.serialize_state_snapshot(world)),"outcomes":[]}
        block["worlds"].append(saved)
        legal = old.SplendorEnv().legal_actions(world)
        for action in actions:
            require(action in legal, "Illegal forced candidate")
            outcome = old.trajectory(world,action)
            outcome["wdl"] = old.outcome_value(outcome["winner"],state.current_player) if outcome["reason"] in old.VALID_NATURAL_TERMINATIONS else None
            saved["outcomes"].append(outcome)
            require(outcome["wdl"] is not None, "Invalid continuation ending")
    block["timing"] = {"wall":time.perf_counter()-start,"cpu":time.process_time()-cpu}


def collect(stage, spec, binding, base=OUTPUT):
    index = spec["index"]
    path,seal_path = root_path(stage,index,base),selection_path(stage,index,base)
    require(not path.exists() and not seal_path.exists(), "One-shot root; preserve existing evidence")
    started,cpu = time.perf_counter(),time.process_time()
    record = {"manifest":binding,"stage":stage,"spec":spec,"time":old.now()}
    try:
        c.budget_boundary(base)
        if "fixture_snapshot" in spec:
            require(stage == "test", "Synthetic source outside isolated test")
            snapshot = spec["fixture_snapshot"]
        else:
            game,receipt = c.verified(spec["source_game"]),c.verified(spec["source_audit"])
            require(game["manifest"] == receipt["manifest"] == binding and receipt["passed"] and "failure" not in game, "Invalid audited source")
            require(receipt["game"] == {k:spec["source_game"][k] for k in ("path","sha256")}, "Source receipt link")
            source_spec = game["spec"]
            require(receipt["spec"] == source_spec and spec["pair"] == spec["game_index"] == source_spec["index"], "Wrong source game identity")
            require(spec["source_stage"] == source_spec["stage"] and all(spec[k] == source_spec[k] for k in ("fit","seat","phase","search_stream")), "Wrong source cell")
            require(stage == "test" or spec["source_stage"] == stage, "Cross-stage source")
            moves = [m for m in game["trace"] if m["turn"] == spec["turn"]]
            require(len(moves) == 1, "Missing source turn")
            snapshot = moves[0]["snapshot"]
        require(old.digest(snapshot) == spec["snapshot_sha256"], "Source snapshot hash")
        state = old.deserialize_state_snapshot(snapshot)
        require(not state.terminal and state.turn_index == spec["turn"] and state.current_player == spec["seat"]
                and old.phase(state.turn_index) == spec["phase"], "Wrong source decision")
        record.update(snapshot=snapshot,privacy=baseline.privacy(state))
        record.update(proposal_union(state,spec["search_stream"],spec["fit"],(stage,index)))
        actions = [old.decode_action(a) for a in record["union"]["actions"]]
        m,e = record["budgets"]["anchor_candidates"],record["budgets"]["extension_worlds"]
        record["selection"] = {"candidate_indices":list(range(len(actions))),"world_seeds":world_seeds(stage,index,"selection"),"worlds":[]}
        record["extension"] = {"candidate_indices":list(range(m)),"world_seeds":world_seeds(stage,index,"extension")[:e],"worlds":[]}
        run_block(state,actions,record["selection"]["world_seeds"],record["selection"],base)
        run_block(state,actions[:m],record["extension"]["world_seeds"],record["extension"],base)
        values,choices = choose(record["selection"],record["extension"],record["candidates"],record["budgets"])
        indices = sorted(set(choices.values()))
        record["selection_lock"] = {"manifest":binding,"stage":stage,"index":index,
             "identity_digest":old.digest(proposal_identity(record)),"selection_sha256":old.digest(record["selection"]),
             "extension_sha256":old.digest(record["extension"]),"means":values,"value_scale":"terminal_wdl",
             "choices":choices,"assessment_indices":indices,"time":old.now(),"created_at_ns":time.time_ns()}
        old.save_new(seal_path,{"manifest":binding,"stage":stage,"index":index,"identity":proposal_identity(record),
                               "selection":record["selection"],"extension":record["extension"],"selection_lock":record["selection_lock"]})
        record["selection_artifact"] = c.artifact(seal_path)
        record["assessment_started_at_ns"] = time.time_ns()
        record["assessment"] = {"candidate_indices":indices,"world_seeds":world_seeds(stage,index,"assessment"),"worlds":[]}
        all_seeds = [seed for block in ("selection","extension","assessment") for seed in record[block]["world_seeds"]]
        require(len(all_seeds) == len(set(all_seeds)), "World blocks overlap")
        run_block(state,[actions[i] for i in indices],record["assessment"]["world_seeds"],record["assessment"],base)
        record["summary"] = summarize(record["assessment"]["worlds"],indices,choices,record["proposed_indices"],len(actions))
        record["work"] = {block:{"continuations":sum(len(w["outcomes"]) for w in record[block]["worlds"]),
                                  "turns":sum(o["turns"] for w in record[block]["worlds"] for o in w["outcomes"])}
                          for block in ("selection","extension","assessment")}
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall":time.perf_counter()-started,"cpu":time.process_time()-cpu}
        old.save_new(path,record)
    return c.artifact(path,index=index,continuations=sum(v["continuations"] for v in record["work"].values()))
