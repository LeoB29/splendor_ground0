"""Fresh equal-budget uniform breadth with an expensive teacher context."""
from __future__ import annotations

import math
from pathlib import Path
import random
import time
import traceback

import torch

from tools.equal_budget_breadth_128 import common as c
from scripts.analysis.learned_proposal_arena import fourth_slot
from splendor_ai.alphazero import world_budget_160 as world128
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

ROOT, OUTPUT, CONFIG = c.ROOT, c.OUTPUT, c.CONFIG
old, baseline, require = c.old, c.baseline, c.require
ARMS = ("anchor", "uniform", "teacher")
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
    """Shared full selection matrix, then a disjoint assessment block."""
    require(stage in ("probe", "main", "test") and block in ("selection", "assessment"), "World namespace")
    require(type(index) is int and 0 <= index < (18 if stage == "test" else CONFIG["states"][stage]), "World index")
    offset = {"probe": 0, "main": 1000000, "test": 2000000}[stage]+index*1024
    offset += {"selection":0, "assessment":512}[block]
    return [CONFIG["world_seed_base"]+offset+i for i in range(128)]


def budgets(canonical_support):
    require(type(canonical_support) is int and canonical_support > 0, "Invalid canonical support")
    m, n = min(canonical_support,4), min(canonical_support,8)
    q, r = divmod(128*m, n)
    return {"canonical_support":canonical_support, "anchor_candidates":m, "breadth_candidates":n,
            "anchor_worlds":128, "teacher_worlds":128, "uniform_base_worlds":q,
            "uniform_remainder_count":r,
            "arm_continuations":{"anchor":128*m, "uniform":128*m, "teacher":128*n}}


def uniform_allocation(union_keys, budget, identity):
    """Allocate extra cells by source identity, before reading any outcomes."""
    stage, index = identity
    require(stage in ("probe", "main", "test") and type(index) is int and index >= 0, "Allocation identity")
    require(budget == budgets(budget["canonical_support"]), "Changed allocation budget")
    n = budget["breadth_candidates"]
    require(len(union_keys) == len(set(union_keys)) == n, "Allocation support")
    order = sorted(range(n), key=lambda j:(old.digest([CONFIG["remainder_tag"], stage, index, union_keys[j]]), j))
    selected = order[:budget["uniform_remainder_count"]]
    counts = [budget["uniform_base_worlds"]+int(j in selected) for j in range(n)]
    require(sum(counts) == budget["arm_continuations"]["uniform"], "Allocation total")
    return {"remainder_order":order, "remainder_indices":selected, "world_counts":counts}


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
    candidates = {arm:{"keys":list(union_keys[:(m if arm == "anchor" else n)]),
                      "actions":list(union_actions[:(m if arm == "anchor" else n)]),
                      "indices":list(range(m if arm == "anchor" else n))} for arm in ARMS}
    return {"candidates":candidates, "union":{"keys":union_keys,"actions":union_actions},
            "proposed_indices":{arm:3 if m == 4 else None for arm in ARMS},
            "added_indices":{"ranked":list(range(m,m+len(ranked))),"random":list(range(m+len(ranked),n))},
            "budgets":budget, "uniform_allocation":uniform_allocation(union_keys, budget, identity)}


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


def choose(selection, candidates, budget, allocation):
    """Read each arm's allocated cells only; preserve exact rational ties."""
    require(budget == budgets(budget["canonical_support"]), "Changed compute budget")
    require(set(candidates) == set(ARMS), "Incomplete selection arms")
    n, m = budget["breadth_candidates"], budget["anchor_candidates"]
    require(selection["candidate_indices"] == list(range(n)), "Selection omitted candidate")
    # Structural validation covers the full saved matrix; arm means read only
    # their permitted prefixes below, regardless of contextual teacher outcomes.
    means(selection["worlds"], n)
    order, remainder = allocation["remainder_order"], allocation["remainder_indices"]
    require(sorted(order) == list(range(n)) and all(type(j) is int for j in order), "Invalid remainder order")
    require(remainder == order[:budget["uniform_remainder_count"]], "Changed remainder allocation")
    counts_uniform = [budget["uniform_base_worlds"]+int(j in remainder) for j in range(n)]
    require(allocation["world_counts"] == counts_uniform, "Changed uniform world counts")
    counts = {"anchor":[128]*m, "uniform":counts_uniform, "teacher":[128]*n}
    sums, accesses, values, choices = {}, {}, {}, {}
    for arm in ARMS:
        indices = candidates[arm]["indices"]
        require(indices == list(range(m if arm == "anchor" else n)), "Changed arm support/order")
        accesses[arm] = [list(range(count)) for count in counts[arm]]
        sums[arm] = [sum(int(selection["worlds"][wi]["outcomes"][j]["wdl"]) for wi in accesses[arm][j])
                     for j in indices]
        require(sum(counts[arm]) == budget["arm_continuations"][arm], "Wrong arm cell budget")
        values[arm] = [total/count for total, count in zip(sums[arm], counts[arm], strict=True)]
        best = indices[0]
        for j in indices[1:]:
            if sums[arm][j]*counts[arm][best] > sums[arm][best]*counts[arm][j]:
                best = j
        choices[arm] = best
    return {"means":values, "sums_wdl":sums, "counts":counts, "world_indices":accesses}, choices


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
    for block in ("selection","assessment"):
        expected = world_seeds(record["stage"],record["spec"]["index"],block)
        value = record[block]
        require(value["world_seeds"] == expected and [w["seed"] for w in value["worlds"]] == expected, "Changed world schedule")
    allocation = uniform_allocation(record["union"]["keys"], record["budgets"], (record["stage"], record["spec"]["index"]))
    require(allocation == record["uniform_allocation"], "Changed identity-based allocation")
    arithmetic,choices = choose(record["selection"],record["candidates"],record["budgets"],allocation)
    lock = record["selection_lock"]
    require(arithmetic == lock["arithmetic"] and arithmetic["means"] == lock["means"] and choices == lock["choices"], "Changed locked selection")
    require(lock["identity_digest"] == old.digest(proposal_identity(record))
            and lock["selection_sha256"] == old.digest(record["selection"]), "Changed selection evidence")
    indices = record["assessment"]["candidate_indices"]
    require(indices == lock["assessment_indices"], "Changed locked assessment support")
    result = summarize(record["assessment"]["worlds"],indices,choices,record["proposed_indices"],len(record["union"]["keys"]))
    require(result == record["summary"], "Stored summary differs from raw assessment")
    return {**{k:record["spec"][k] for k in ("index","pair","game_index","fit","seat","phase")},
            "privacy":record["privacy"],
            **{k:result[k] for k in ("arm_scores","choices","proposed_indices")},
            "added_indices":record["added_indices"],"budgets":record["budgets"],"uniform_allocation":record["uniform_allocation"]}


def proposal_identity(record):
    return {"spec":record["spec"],"snapshot":record["snapshot"],"privacy":record["privacy"],
            "proposals":{a:{k:v for k,v in p.items() if k != "timing"} for a,p in record["proposals"].items()},
            **{k:record[k] for k in ("candidates","union","proposed_indices","added_indices","budgets","uniform_allocation")}}


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
        record["selection"] = {"candidate_indices":list(range(len(actions))),"world_seeds":world_seeds(stage,index,"selection"),"worlds":[]}
        run_block(state,actions,record["selection"]["world_seeds"],record["selection"],base)
        arithmetic,choices = choose(record["selection"],record["candidates"],record["budgets"],record["uniform_allocation"])
        indices = sorted(set(choices.values()))
        record["selection_lock"] = {"manifest":binding,"stage":stage,"index":index,
             "identity_digest":old.digest(proposal_identity(record)),"selection_sha256":old.digest(record["selection"]),
             "means":arithmetic["means"],"arithmetic":arithmetic,"value_scale":"terminal_wdl",
             "choices":choices,"assessment_indices":indices,"time":old.now(),"created_at_ns":time.time_ns()}
        old.save_new(seal_path,{"manifest":binding,"stage":stage,"index":index,"identity":proposal_identity(record),
                               "selection":record["selection"],"selection_lock":record["selection_lock"]})
        record["selection_artifact"] = c.artifact(seal_path)
        record["assessment_started_at_ns"] = time.time_ns()
        record["assessment"] = {"candidate_indices":indices,"world_seeds":world_seeds(stage,index,"assessment"),"worlds":[]}
        all_seeds = [seed for block in ("selection","assessment") for seed in record[block]["world_seeds"]]
        require(len(all_seeds) == len(set(all_seeds)), "World blocks overlap")
        run_block(state,[actions[i] for i in indices],record["assessment"]["world_seeds"],record["assessment"],base)
        record["summary"] = summarize(record["assessment"]["worlds"],indices,choices,record["proposed_indices"],len(actions))
        record["work"] = {block:{"continuations":sum(len(w["outcomes"]) for w in record[block]["worlds"]),
                                  "turns":sum(o["turns"] for w in record[block]["worlds"] for o in w["outcomes"])}
                          for block in ("selection","assessment")}
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall":time.perf_counter()-started,"cpu":time.process_time()-cpu}
        old.save_new(path,record)
    return c.artifact(path,index=index,continuations=sum(v["continuations"] for v in record["work"].values()))
