"""Uniform breadth versus learned128 at exactly matched continuation counts."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from tools.equal_budget_breadth_arena_128 import common as c
from scripts.analysis.learned_proposal_arena import fourth_slot
from splendor_ai.alphazero import world_budget_160 as world128
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

old, baseline, windows = c.old, c.baseline, c.windows
ROOT, OUTPUT, CONFIG, require = c.ROOT, c.OUTPUT, c.CONFIG, c.require
PRODUCTION_OUTPUT = OUTPUT.resolve()
FITS = c.FITS
_MODELS, _MODEL_ARTIFACTS = {}, {}


def initialize_models(base=OUTPUT, registry=None):
    """Both actors use the same assigned anchor; ancestor registries stay intact."""
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(PRODUCTION_OUTPUT), "Fixture registry cannot target production")
    registry = old.read_json(Path(base)/"models_lock.json") if registry is None else registry
    require(set(registry["models"]) == {f"anchor_{fit}" for fit in FITS}, "Incomplete model registry")
    loaded,artifacts = {},{}
    for key,item in registry["models"].items():
        require(item["source_fit"] == int(key.split("_")[1]), "Registry source fit mismatch")
        path = c.verify_hash(item)
        loaded[key],artifacts[key] = StrongPolicyInference(path).model,dict(item)
    old.initialize_worker()
    _MODELS.clear(); _MODELS.update(loaded)
    _MODEL_ARTIFACTS.clear(); _MODEL_ARTIFACTS.update(artifacts)


def model_id(stage, fit, candidate):
    require(stage in CONFIG["games"] and fit in FITS and type(candidate) is bool, "Unknown proposal role")
    return f"anchor_{fit}"


@lru_cache(maxsize=3)
def specs(stage):
    require(stage in CONFIG["games"], "Unknown stage")
    count = CONFIG["games"][stage]//2
    order = sorted(range(count),key=lambda pair:old.digest([CONFIG["assignment_tag"],stage,pair]))
    fits = {pair:FITS[rank % 3] for rank,pair in enumerate(order)}
    return [{"index":i,"pair":i//2,"candidate_seat":i%2,"fit":fits[i//2],
             "environment_seed":CONFIG["environment_seed"][stage]+i//2,
             "search_stream":old.digest([CONFIG["stream_tag"],stage,i//2])}
            for i in range(CONFIG["games"][stage])]


def check_stage(stage, base):
    require(stage in CONFIG["games"], "Unknown stage")
    if stage == "test":
        require(not Path(base).resolve().is_relative_to(PRODUCTION_OUTPUT), "Synthetic fixture cannot target production evidence")


def game_path(stage, index, base=OUTPUT):
    check_stage(stage,base)
    require(type(index) is int and 0 <= index < CONFIG["games"][stage], "Game outside fixed schedule")
    return Path(base)/stage/"games"/f"game_{index:04d}.json"


def root_path(stage, index, turn, base=OUTPUT):
    game_path(stage,index,base)
    require(type(turn) is int and 0 <= turn < 400, "Invalid root turn")
    return Path(base)/stage/"roots"/f"game_{index:04d}"/f"turn_{turn:03d}.json"


def root_identity(identity):
    require(isinstance(identity,(list,tuple)) and len(identity) == 3, "Invalid public root identity")
    stage,index,turn = identity
    require(stage in CONFIG["games"] and type(index) is int and 0 <= index < CONFIG["games"][stage]
            and type(turn) is int and 0 <= turn < 400, "Invalid public root identity")
    return [stage,index,turn]


def learned_proposal(state, stream, source_model):
    """Exact incumbent proposal semantics, including concrete representatives."""
    require(source_model in _MODELS and source_model in _MODEL_ARTIFACTS, "Unsealed proposal model")
    started = time.perf_counter()
    legal = old.SplendorEnv().legal_actions(state)
    schedule = world128.root_schedule(state,stream,128)
    reference = baseline.candidate_set(state,schedule["candidate_nonce"])
    groups = old.actions_by_canonical_successor_key(state,legal)
    keys = [repr(key) for key in groups]
    representatives = {repr(key):old.serialize_action(action) for key,action in groups.items()}
    representatives.update(zip(reference["keys"],reference["actions"],strict=True))
    model = _MODELS[source_model]
    inference_start = time.perf_counter()
    with torch.no_grad():
        row,encoded_keys,_ = encode_position(state,model.config,legal)
        logits,_ = model(row["observation"][None],row["features"][None],torch.ones((1,len(legal)),dtype=torch.bool))
    require([repr(key) for key in encoded_keys] == keys and torch.isfinite(logits).all().item(), "Invalid model input/output")
    values = [float(torch.amax(logits[0][row["groups"] == j])) for j in range(len(keys))]
    extra = time.perf_counter()-inference_start
    offered = fourth_slot(keys,reference["keys"],values)
    return {"schedule":schedule,"reference":reference,
            "candidates":{"keys":offered,"actions":[representatives[key] for key in offered],
                          "canonical_support":len(keys),"legal_support":len(legal)},
            "model_id":source_model,"model_sha256":_MODEL_ARTIFACTS[source_model]["sha256"],
            "fit":_MODEL_ARTIFACTS[source_model]["source_fit"],"group_keys":keys,
            "legal_actions":[old.serialize_action(a) for a in legal],
            "learned_logits":logits[0].tolist(),"learned_group_scores":values,
            "timing":{"wall":time.perf_counter()-started,"learned_extra_wall":extra}}


def expanded_candidates(anchor, representatives, identity, breadth):
    identity = root_identity(identity)
    require(type(breadth) is bool, "Invalid breadth role")
    keys,values = anchor["group_keys"],anchor["learned_group_scores"]
    require(keys and len(set(keys)) == len(keys) == len(values) and set(keys) == set(representatives), "Canonical support mismatch")
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values), "Invalid group scores")
    original = anchor["candidates"]
    offered = list(original["keys"])
    actions = list(original["actions"])
    m = min(len(keys),4)
    require(len(offered) == len(actions) == len(set(offered)) == m and set(offered).issubset(keys), "Invalid original candidates")
    require(original["canonical_support"] == len(keys), "Original support mismatch")
    ranked,uniform = [],[]
    if breadth:
        remaining = [i for i,key in enumerate(keys) if key not in offered]
        ranked = sorted(remaining,key=lambda i:(-values[i],i))[:min(2,min(len(keys),8)-m)]
        remaining = [i for i in remaining if i not in ranked]
        uniform = sorted(remaining,key=lambda i:(old.digest([CONFIG["random_tag"],*identity,keys[i]]),i))[:min(len(keys),8)-m-len(ranked)]
        added = [keys[i] for i in ranked+uniform]
        offered += added
        actions += [representatives[key] for key in added]
    return {"candidates":{**original,"keys":offered,"actions":actions},
            "anchor_candidates":original,"root_identity":identity,"breadth":breadth,
            "added_indices":{"ranked":list(range(m,m+len(ranked))),
                             "random":list(range(m+len(ranked),m+len(ranked)+len(uniform)))}}


def proposal_for_root(state, stream, source_model, identity, breadth):
    identity = root_identity(identity)
    require(identity[2] == state.turn_index, "Root identity turn mismatch")
    anchor = learned_proposal(state,stream,source_model)
    groups = old.actions_by_canonical_successor_key(state,old.SplendorEnv().legal_actions(state))
    representatives = {repr(key):old.serialize_action(action) for key,action in groups.items()}
    expanded = expanded_candidates(anchor,representatives,identity,breadth)
    allocation = allocation_for_candidates(expanded["candidates"],identity,breadth)
    return {**anchor,**expanded,"allocation":allocation}


def allocation_for_candidates(candidates,identity,breadth):
    identity = root_identity(identity)
    require(type(breadth) is bool,"Invalid allocation role")
    support = candidates["canonical_support"]
    require(type(support) is int and support >= 1,"Invalid canonical support")
    keys = candidates["keys"]
    n,m = len(keys),min(support,4)
    require(n == min(support,8 if breadth else 4) and len(set(keys)) == n,"Invalid allocation candidates")
    require(len(candidates["actions"]) == n,"Invalid allocation actions")
    budget = 128*m
    q,r = divmod(budget,n)
    order = sorted(range(n),key=lambda j:(old.digest([CONFIG["remainder_tag"],*identity,keys[j]]),j))
    remainder = order[:r]
    counts = [q+int(j in remainder) for j in range(n)]
    require(sum(counts) == budget and all(1 <= count <= 128 for count in counts),"Incorrect matched allocation")
    return {"world_counts":counts,"remainder_order":order,"remainder_indices":remainder,
            "world_count":max(counts),"continuations":sum(counts)}


def select(worlds, world_counts):
    require(isinstance(world_counts,list) and 1 <= len(world_counts) <= 8
            and all(type(count) is int and 1 <= count <= 128 for count in world_counts),"Invalid support or world counts")
    require(len(worlds) == max(world_counts),"Wrong actual world count")
    sums,seen = [0]*len(world_counts),[0]*len(world_counts)
    for wi,world in enumerate(worlds):
        active = [j for j,count in enumerate(world_counts) if wi < count]
        require(world["candidate_indices"] == active and all(type(j) is int for j in world["candidate_indices"]),"Changed active candidate support")
        require(len(world["outcomes"]) == len(active),"Incomplete forced-action support")
        for j,outcome in zip(active,world["outcomes"],strict=True):
            require(outcome["reason"] in old.VALID_NATURAL_TERMINATIONS and type(outcome["wdl"]) in (float,int)
                    and outcome["wdl"] in (-1.,0.,1.),"Invalid inner ending")
            sums[j] += int(outcome["wdl"])
            seen[j] += 1
    require(seen == world_counts,"Missing allocated candidate cells")
    best = 0
    for j in range(1,len(sums)):
        if sums[j]*world_counts[best] > sums[best]*world_counts[j]:
            best = j
    return {"choice":best,"means":[total/count for total,count in zip(sums,world_counts,strict=True)],
            "sums_wdl":sums,"counts":list(world_counts)}


def choose_root(state, stream, path, binding, source_model, identity, breadth, base=OUTPUT):
    path = Path(path)
    identity = root_identity(identity)
    require(path.resolve() == root_path(*identity,base=base).resolve(), "Root path identity mismatch")
    require(not path.exists(), "One-shot root; no overwrite or retry")
    wall,cpu = time.perf_counter(),time.process_time()
    record = {"manifest":binding,"time":old.now(),"snapshot":old.serialize_state_snapshot(state),
              "privacy":baseline.privacy(state),"worlds":[],"root_identity":identity,
              "planner":"candidate" if breadth else "reference"}
    try:
        c.budget_boundary(base)
        proposal = proposal_for_root(state,stream,source_model,identity,breadth)
        allocation = proposal["allocation"]
        record.update(schedule=proposal["schedule"],candidates=proposal["candidates"],proposal=proposal,allocation=allocation)
        legal = old.SplendorEnv().legal_actions(state)
        record["scores"] = list(map(float,old._RAW.score_actions(state,legal)))
        actions = [old.decode_action(a) for a in proposal["candidates"]["actions"]]
        for wi,seed in enumerate(proposal["schedule"]["world_seeds"][:allocation["world_count"]]):
            c.budget_boundary(base)
            world = old.determinize_for_observer(state,state.current_player,random.Random(seed))
            active = [j for j,count in enumerate(allocation["world_counts"]) if wi < count]
            row = {"seed":seed,"state_sha256":old.digest(old.serialize_state_snapshot(world)),"candidate_indices":active,"outcomes":[]}
            record["worlds"].append(row)
            legal = old.SplendorEnv().legal_actions(world)
            for j in active:
                action = actions[j]
                require(action in legal, "Illegal forced proposal")
                value = old.trajectory(world,action)
                value["wdl"] = old.outcome_value(value["winner"],state.current_player) if value["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                row["outcomes"].append(value)
                require(value["wdl"] is not None, "Invalid continuation ending")
        record.update(select(record["worlds"],allocation["world_counts"]))
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall":time.perf_counter()-wall,"cpu":time.process_time()-cpu}
        old.save_new(path,record)
    return actions[record["choice"]],record


def decision(share, interval, stage="main"):
    require(stage == "main", "Statistics stage")
    require(type(share) in (int,float) and 0 <= share <= 1 and len(interval) == 2
            and 0 <= interval[0] <= interval[1] <= 1, "Invalid score interval")
    superiority = interval[0] > .5
    practical = share >= .55 and superiority
    return {"stage":stage,"stage_pass":practical,"superiority_established":superiority,
            "inferiority_established":interval[1] < .5,
            "practical_progress":practical,"material_target_excluded":interval[1] < .55,
            "classification":"positive_practical_progress" if practical else
            "positive_below_observed_target" if superiority else
            "demonstrated_harm" if interval[1] < .5 else
            "below_material_target" if interval[1] < .55 else "inconclusive"}


def statistics(rows, stage="main"):
    require(stage == "main", "Statistics stage")
    require(len(rows) == 402 and [r["index"] for r in rows] == list(range(402)), "Incomplete or unordered stage")
    strata = {fit:[] for fit in FITS}
    for pair in range(201):
        a,b = rows[2*pair:2*pair+2]
        spec = specs(stage)[2*pair]
        require(a["pair"] == b["pair"] == pair and a["seat"] == 0 and b["seat"] == 1
                and a["fit"] == b["fit"] == spec["fit"], "Pair/fit alignment")
        require(all(type(r["score"]) in (int,float) and r["score"] in (0.,.5,1.) for r in (a,b)), "Invalid score")
        strata[spec["fit"]].append(int(2*a["score"]+2*b["score"]))
    require(all(len(v) == 67 for v in strata.values()), "Unbalanced fits")
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = []
    for _ in range(50):
        total = np.zeros(1000,dtype=np.int64)
        for values in strata.values():
            indices = rng.integers(0,67,size=(1000,67))
            total += np.asarray(values,dtype=np.int64)[indices].sum(axis=1)
        boot.extend((total/804).tolist())
    share = sum(sum(values) for values in strata.values())/804
    interval = np.quantile(boot,[.025,.975],method="linear").tolist()
    return {"games":402,"pairs":201,"share":share,"ci95":interval,
            "wins":sum(r["score"] == 1 for r in rows),"draws":sum(r["score"] == .5 for r in rows),
            "losses":sum(r["score"] == 0 for r in rows),**decision(share,interval,stage)}


def play_game(stage, spec, binding, base=OUTPUT, test_initial_snapshot=None):
    """Play once; save every decision and retain all partial failure evidence."""
    path = game_path(stage,spec["index"],base)
    require(spec == specs(stage)[spec["index"]], "Fixed schedule mismatch")
    if path.exists():
        raise FileExistsError(f"Breadth arena does not automatically reuse or retry games: {path}")
    require(type(spec["candidate_seat"]) is int and spec["candidate_seat"] in (0,1), "Invalid candidate seat")
    require(test_initial_snapshot is None or stage == "test", "Initial snapshot overrides are restricted to isolated test games")
    wall,cpu = time.perf_counter(),time.process_time()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state() if test_initial_snapshot is None else old.deserialize_state_snapshot(test_initial_snapshot)
    record = {"manifest":binding,"spec":spec,"time":old.now(),"trace":[],
              "initial_state_sha256":old.digest(old.serialize_state_snapshot(state))}
    if test_initial_snapshot is not None:
        record["test_initial_snapshot"] = old.serialize_state_snapshot(state)
    visits,forced,idle,hasher = Counter(),0,0,hashlib.sha256()
    reason,winner = "completed",None
    try:
        while not state.terminal:
            c.budget_boundary(base)
            signature = old.state_signature(state)
            visits[signature] += 1
            if visits[signature] >= 4:
                reason = "repetition_cutoff"
                break
            if state.turn_index >= 400:
                reason = "max_turns"
                break
            legal = env.legal_actions(state)
            if not legal:
                reason = "stalled"
                break
            row = {"turn":state.turn_index,"actor":state.current_player,
                   "state_sha256":old.digest(old.serialize_state_snapshot(state))}
            breadth = state.current_player == spec["candidate_seat"]
            source_model = model_id(stage,spec["fit"],breadth)
            identity = [stage,spec["index"],state.turn_index]
            rp = root_path(*identity,base=base)
            try:
                action,_ = choose_root(state,spec["search_stream"],rp,binding,source_model,identity,breadth,base)
            except BaseException:
                if rp.exists():
                    row["root_sha256"] = old.file_hash(rp)
                record["failed_decision"] = row
                raise
            row["root_sha256"] = old.file_hash(rp)
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                record["failed_decision"] = {**row,"action":None if action is None else old.serialize_action(action)}
                break
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            hasher.update(old.digest(row["action"]).encode())
            successor = env.step(state,action)
            forced = forced+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state,successor,action) else idle+1
            state = successor
            if forced >= 2:
                reason,winner = "forced_pass_deadlock",old.adjudicate_scores(state)
                break
            if idle >= 60:
                reason = "no_progress_cutoff"
                break
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason,winner=winner)
        require(reason in old.VALID_NATURAL_TERMINATIONS, f"Invalid outer ending: {reason}")
    except BaseException:
        record["failure"] = traceback.format_exc()
        if "reason" not in record:
            record.update(reason="exception",winner=None)
        raise
    finally:
        final = old.serialize_state_snapshot(state)
        record.update(final_state=final,final_state_sha256=old.digest(final),turns=len(record["trace"]),
                      trace_sha256=hasher.hexdigest(),timing={"wall":time.perf_counter()-wall,"cpu":time.process_time()-cpu})
        old.save_new(path,record)
    return {"index":spec["index"],"sha256":old.file_hash(path)}
