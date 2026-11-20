"""Frozen161 restricted self-play, complete replay, and identity-only panels."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import math
import time
import traceback

from tools.proposal_specialist_128 import common as c
from splendor_ai.alphazero import fast_selector_157 as fast
from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from scripts.analysis import verify_fast_selector_transfer_157 as independent

ROOT, OUTPUT, CONFIG = c.ROOT,c.OUTPUT,c.CONFIG
old,require = c.old,c.require
CELLS = tuple((fit,seat,phase) for fit in c.FITS for seat in (0,1) for phase in (0,1,2))
_MODELS,_MODEL_ARTIFACTS = {},{}


def initialize_models(base=OUTPUT, registry=None):
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()),"Fixture registry targets production")
    registry = old.read_json(Path(base)/"models_lock.json") if registry is None else registry
    require(set(registry["models"]) == {f"{arm}_{fit}" for arm in ("anchor","specialist","control") for fit in c.FITS},
            "Incomplete model seal")
    models,artifacts = {},{}
    for fit in c.FITS:
        key = f"anchor_{fit}"
        item = registry["models"][key]
        require(item["source_fit"] == fit,"Wrong source model fit")
        models[key],artifacts[key] = StrongPolicyInference(c.verify_hash(item)).model,dict(item)
    old.initialize_worker()
    _MODELS.clear(); _MODELS.update(models)
    _MODEL_ARTIFACTS.clear(); _MODEL_ARTIFACTS.update(artifacts)


def schedule(stage):
    require(stage in ("main","probe","test"),"Unknown source stage")
    count = 18 if stage == "test" else CONFIG["source_games"][stage]
    order = sorted(range(count),key=lambda i:(old.digest([CONFIG["source_assignment_tag"],stage,i]),i))
    assignment = {index:CELLS[rank%18] for rank,index in enumerate(order)}
    result = []
    for index in range(count):
        fit,seat,phase = assignment[index]
        result.append({"stage":stage,"index":index,"fit":fit,"seat":seat,"phase":phase,
                       "environment_seed":CONFIG["source_seed"][stage]+index,
                       "search_stream":old.digest([CONFIG["source_stream_tag"],stage,index])})
    return result


@lru_cache(maxsize=3)
def _independent_order(stage):
    count = {"main":1080,"probe":54,"test":18}[stage]
    return sorted(range(count),key=lambda i:(old.digest(["proposal-specialist-128-source-cell-v1",stage,i]),i))


def independent_spec(stage,index):
    count = {"main":1080,"probe":54,"test":18}[stage]
    require(type(index) is int and 0 <= index < count,"Source outside schedule")
    ranked = _independent_order(stage)
    cell = ranked.index(index)%18
    fit,seat,phase = c.FITS[cell//6],(cell//3)%2,cell%3
    return {"stage":stage,"index":index,"fit":fit,"seat":seat,"phase":phase,
            "environment_seed":{"main":2135000000,"probe":2136000000,"test":2138000000}[stage]+index,
            "search_stream":old.digest(["proposal-specialist-128-source-stream-v1",stage,index])}


def game_path(stage,index,base=OUTPUT):
    require(stage in ("main","probe","test") and type(index) is int,"Invalid source identity")
    require(0 <= index < (18 if stage == "test" else CONFIG["source_games"][stage]),"Source outside schedule")
    if stage == "test":
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()),"Synthetic game targets production")
    return Path(base)/"source"/stage/"games"/f"game_{index:04d}.json"


def audit_path(stage,index,base=OUTPUT):
    game_path(stage,index,base)
    return Path(base)/"source"/stage/"audits"/f"game_{index:04d}.json"


def _initial(spec,game=None):
    env = old.SplendorEnv(seed=spec["environment_seed"])
    if game is not None and "fixture_snapshot" in game:
        require(spec["stage"] == "test","Synthetic production source")
        state = old.deserialize_state_snapshot(game["fixture_snapshot"])
    else:
        state = env.initial_state()
    return env,state


def collect(spec,binding,base=OUTPUT,fixture_snapshot=None):
    stage,index = spec["stage"],spec["index"]
    path = game_path(stage,index,base)
    require(spec == schedule(stage)[index],"Changed source specification")
    require(not path.exists(),"One-shot source game; preserve existing evidence")
    require(fixture_snapshot is None or stage == "test","Synthetic production override")
    key = f"anchor_{spec['fit']}"
    require(key in _MODELS and key in _MODEL_ARTIFACTS,"Unsealed source model")
    start,cpu = time.perf_counter(),time.process_time()
    game = {"manifest":binding,"spec":spec,"time":old.now(),"model_id":key,
            "model_sha256":_MODEL_ARTIFACTS[key]["sha256"],"trace":[]}
    if fixture_snapshot is not None:
        game["fixture_snapshot"] = fixture_snapshot
    env,state = _initial(spec,game)
    seen,forced,idle = Counter(),0,0
    reason,winner = "completed",None
    try:
        c.budget_boundary(base)
        while not state.terminal:
            c.budget_boundary(base)
            seen[old.state_signature(state)] += 1
            require(seen[old.state_signature(state)] < CONFIG["repetition_limit"],"Source repetition cutoff")
            require(state.turn_index < CONFIG["max_turns"],"Source turn cutoff")
            legal = env.legal_actions(state)
            require(bool(legal),"Source stalled")
            snapshot = old.serialize_state_snapshot(state)
            sw,sc = time.perf_counter(),time.process_time()
            action,evidence = fast.choose(state,spec["search_stream"],_MODELS[key])
            timing = {"wall":time.perf_counter()-sw,"cpu":time.process_time()-sc}
            require(action in legal,"Illegal source action")
            game["trace"].append({"turn":state.turn_index,"actor":state.current_player,"phase":old.phase(state.turn_index),
                 "snapshot":snapshot,"state_sha256":old.digest(snapshot),"legal_support":len(legal),
                 "model_id":key,"model_sha256":game["model_sha256"],"evidence":evidence,
                 "action":old.serialize_action(action),"timing":timing})
            nxt = env.step(state,action)
            forced = forced+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state,nxt,action) else idle+1
            state = nxt
            if forced >= 2:
                reason,winner = "forced_pass_deadlock",old.adjudicate_scores(state)
                break
            require(idle < CONFIG["no_progress_limit"],"Source no-progress cutoff")
        if reason == "completed":
            winner = state.winner
        require(game["trace"] and reason in old.VALID_NATURAL_TERMINATIONS,"Invalid source termination")
        game.update(reason=reason,winner=winner)
    except BaseException:
        game["failure"] = traceback.format_exc()
        raise
    finally:
        game.update(final_state=old.serialize_state_snapshot(state),trace_sha256=old.digest(game["trace"]),
                    timing={"wall":time.perf_counter()-start,"cpu":time.process_time()-cpu})
        old.save_new(path,game)
    return c.artifact(path,index=index,stage=stage)


def _timing(value):
    require(set(value) == {"wall","cpu"} and all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in value.values()),"Invalid source timing")


def audit(spec,item,binding,base=OUTPUT):
    stage,index = spec["stage"],spec["index"]
    target = audit_path(stage,index,base)
    require(not target.exists(),"One-shot source audit")
    require(spec == independent_spec(stage,index),"Source specification differs from independent schedule")
    path = game_path(stage,index,base)
    require(c.verify_hash(item).resolve() == path.resolve(),"Source artifact path")
    game = c.verified(item)
    require(game["manifest"] == binding and game["spec"] == spec and "failure" not in game,"Invalid source record")
    require(game["trace"] and old.digest(game["trace"]) == game["trace_sha256"],"Source trace hash")
    key = f"anchor_{spec['fit']}"
    require(game["model_id"] == key and game["model_sha256"] == _MODEL_ARTIFACTS[key]["sha256"],"Source model identity")
    start,cpu = time.perf_counter(),time.process_time()
    env,state = _initial(spec,game)
    seen,forced,idle,coverage = Counter(),0,0,Counter()
    for row in game["trace"]:
        c.budget_boundary(base)
        require(not state.terminal and state.turn_index < 400 and forced < 2,"Post-terminal source move")
        seen[old.state_signature(state)] += 1
        require(seen[old.state_signature(state)] < 4,"Source repetition")
        snapshot = old.serialize_state_snapshot(state)
        require(old.digest(row["snapshot"]) == row["state_sha256"] == old.digest(snapshot),"Changed source state")
        require(row["turn"] == state.turn_index and row["actor"] == state.current_player,"Source turn identity")
        require(row["phase"] == (0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2),"Source phase")
        require(row["model_id"] == key and row["model_sha256"] == game["model_sha256"],"Source turn model")
        legal = env.legal_actions(state)
        require(legal and len(legal) == row["legal_support"],"Source legal support")
        action,evidence = independent.independent_decision(state,spec["search_stream"],_MODELS[key])
        independent.compare(evidence,row["evidence"])
        require(action in legal and old.serialize_action(action) == row["action"],"Source decision mismatch")
        _timing(row["timing"])
        coverage.update({"moves":1,f"seat_{state.current_player}":1,f"phase_{row['phase']}":1,
                         f"support_{len(evidence['candidates']['keys'])}":1,
                         "own_private":int(evidence["privacy"]["own"] > 0),
                         "opponent_private":int(evidence["privacy"]["opponent"] > 0)})
        nxt = env.step(state,action)
        forced = forced+1 if len(legal) == 1 and old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(state,nxt,action) else idle+1
        require(forced >= 2 or idle < 60,"Source idle cutoff")
        state = nxt
    require(old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"]),"Source final state")
    reason = "forced_pass_deadlock" if forced >= 2 else "completed"
    require(state.terminal or forced >= 2,"Unresolved source game")
    ranks = [(p.score,-len(p.purchased_cards)) for p in state.players]
    winner = (None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)) if forced >= 2 else state.winner
    require(reason == game["reason"] and winner == game["winner"],"Source final outcome")
    _timing(game["timing"])
    require(game["timing"]["wall"] >= sum(r["timing"]["wall"] for r in game["trace"]),"Source elapsed accounting")
    receipt = {"manifest":binding,"spec":spec,"passed":True,"game":c.artifact(path),
               "trace_sha256":game["trace_sha256"],"coverage":dict(coverage),
               "every_move_hidden_input_invariance":True,"reason":reason,
               "timing":{"wall":time.perf_counter()-start,"cpu":time.process_time()-cpu}}
    old.save_new(target,receipt)
    return c.artifact(target,index=index,stage=stage)


def _inventory(stage,items,audits,binding,base):
    expected = schedule(stage)
    require(len(items) == len(audits) == len(expected),"Incomplete source pool")
    entries = []
    for item,checked,spec in zip(items,audits,expected,strict=True):
        c.budget_boundary(base)
        require(item["index"] == checked["index"] == spec["index"],"Unordered source pool")
        game,receipt = c.verified(item),c.verified(checked)
        require(game["manifest"] == receipt["manifest"] == binding and game["spec"] == receipt["spec"] == spec,"Source binding/specification")
        require("failure" not in game and receipt["passed"] and receipt["game"] == {k:item[k] for k in ("path","sha256")},"Source audit linkage")
        require(game["trace_sha256"] == receipt["trace_sha256"] == old.digest(game["trace"]),"Source audited trace")
        require(game["reason"] in old.VALID_NATURAL_TERMINATIONS,"Invalid source ending")
        # Keep only outcome-independent selection metadata in memory. Full traces
        # include every hidden snapshot and need not all remain resident at once.
        metadata = {"trace":[{k:row[k] for k in ("turn","actor","phase","state_sha256")} |
                    {"evidence":{"privacy":row["evidence"]["privacy"],
                     "candidates":{"canonical_support":row["evidence"]["candidates"]["canonical_support"]}}}
                    for row in game["trace"]]}
        entries.append((spec,metadata,item,checked))
    return entries


def _game_priority(stage,index):
    return (old.digest([CONFIG["selection_tag"],"game",stage,index]),index)


def _turn_priority(stage,index,turn):
    return (old.digest([CONFIG["selection_tag"],"turn",stage,index,turn]),turn)


def _selected_spec(index,stage,entry,move):
    spec,_,item,checked = entry
    return {"index":index,"pair":spec["index"],"fit":spec["fit"],"seat":spec["seat"],"phase":spec["phase"],
            "game_index":spec["index"],"turn":move["turn"],"source_stage":stage,
            "source_game":{k:item[k] for k in ("path","sha256")},
            "source_audit":{k:checked[k] for k in ("path","sha256")},
            "search_stream":spec["search_stream"],"snapshot_sha256":move["state_sha256"],
            "privacy":move["evidence"]["privacy"],"canonical_support":move["evidence"]["candidates"]["canonical_support"]}


def prepare_selection(items,audits,binding,base=OUTPUT):
    require(set(items) == set(audits) == {"main","probe"},"Source stages missing")
    path = Path(base)/"selection.json"
    require(not path.exists(),"Source selection already sealed")
    output = {"manifest":binding,"sources":{},"coverage":{}}
    for stage in ("probe","main"):
        entries = _inventory(stage,items[stage],audits[stage],binding,base)
        chosen,eligible_counts = [],{}
        needed = 20 if stage == "main" else 1
        for cell in CELLS:
            eligible = []
            for entry in entries:
                spec,game,_,_ = entry
                if (spec["fit"],spec["seat"],spec["phase"]) == cell:
                    moves = [r for r in game["trace"] if r["actor"] == cell[1] and r["phase"] == cell[2]]
                    if moves:
                        eligible.append((entry,moves))
            eligible_counts[":".join(map(str,cell))] = len(eligible)
            if len(eligible) < needed:
                raise c.BudgetExceeded(f"Fixed {stage} source pool short in cell {cell}: {len(eligible)} < {needed}; no extension")
            eligible.sort(key=lambda x:_game_priority(stage,x[0][0]["index"]))
            for entry,moves in eligible[:needed]:
                move = min(moves,key=lambda r:_turn_priority(stage,entry[0]["index"],r["turn"]))
                chosen.append((entry,move))
        chosen.sort(key=lambda x:x[0][0]["index"])
        output[stage] = [_selected_spec(i,stage,entry,move) for i,(entry,move) in enumerate(chosen)]
        require(len(output[stage]) == CONFIG["states"][stage] and len({r["pair"] for r in output[stage]}) == len(output[stage]),"Selected panel coverage")
        output["sources"][stage] = {"games":items[stage],"audits":audits[stage]}
        output["coverage"][stage] = {"source_games":len(entries),"selected_states":len(chosen),"eligible_by_cell":eligible_counts}
    validate_selection(output,binding,base)
    old.save_new(path,output)
    return c.artifact(path)


def validate_selection(selection,binding,base=OUTPUT):
    """Second implementation: eligibility/ranks determine every retained game/turn."""
    require(selection["manifest"] == binding,"Panel manifest")
    for stage,needed in (("main",20),("probe",1)):
        source = selection["sources"][stage]
        entries = _inventory(stage,source["games"],source["audits"],binding,base)
        selected,counts = [],{}
        for fit,seat,phase in CELLS:
            candidates = []
            for entry in entries:
                spec,game,_,_ = entry
                require(spec == independent_spec(stage,spec["index"]),"Independent source schedule")
                if (spec["fit"],spec["seat"],spec["phase"]) != (fit,seat,phase):
                    continue
                turns = [r for r in game["trace"] if r["actor"] == seat and r["phase"] == phase]
                if turns:
                    rank = (old.digest(["proposal-specialist-128-panel-v1","game",stage,spec["index"]]),spec["index"])
                    candidates.append((rank,entry,turns))
            counts[f"{fit}:{seat}:{phase}"] = len(candidates)
            require(len(candidates) >= needed,"Independent panel source shortage")
            for _,entry,turns in sorted(candidates,key=lambda v:v[0])[:needed]:
                move = sorted(turns,key=lambda r:(old.digest(["proposal-specialist-128-panel-v1","turn",stage,entry[0]["index"],r["turn"]]),r["turn"]))[0]
                selected.append((entry,move))
        selected.sort(key=lambda x:x[0][0]["index"])
        rebuilt = [_selected_spec(i,stage,e,m) for i,(e,m) in enumerate(selected)]
        require(selection[stage] == rebuilt,"Identity-only panel selection mismatch")
        require(selection["coverage"][stage] == {"source_games":len(entries),"selected_states":len(selected),"eligible_by_cell":counts},"Panel coverage mismatch")
    return True


def replay_selected_state(spec,base=OUTPUT):
    """Replay actual source actions; model decisions were audited for every game."""
    game,receipt = c.verified(spec["source_game"]),c.verified(spec["source_audit"])
    source_spec = game["spec"]
    game_path(source_spec["stage"],source_spec["index"],base)
    require(receipt["passed"] and receipt["manifest"] == game["manifest"] and receipt["spec"] == source_spec,"Selected source audit")
    require(receipt["game"] == spec["source_game"] and "failure" not in game,"Selected source link")
    require(spec["source_stage"] == source_spec["stage"] and spec["game_index"] == spec["pair"] == source_spec["index"],"Selected source identity")
    require(all(spec[k] == source_spec[k] for k in ("fit","seat","phase","search_stream")),"Selected source cell")
    require(old.digest(game["trace"]) == game["trace_sha256"] == receipt["trace_sha256"],"Selected source trace seal")
    env,state = _initial(source_spec,game)
    for move in game["trace"]:
        c.budget_boundary(base)
        require(not state.terminal and state.turn_index == move["turn"] and state.current_player == move["actor"],"Selected source replay identity")
        snapshot = old.serialize_state_snapshot(state)
        require(old.digest(move["snapshot"]) == move["state_sha256"] == old.digest(snapshot),"Selected source replay state")
        if state.turn_index == spec["turn"]:
            require(spec["snapshot_sha256"] == old.digest(snapshot) and state.current_player == spec["seat"] and old.phase(state.turn_index) == spec["phase"],"Selected source turn/cell")
            require(spec["privacy"] == c.baseline.privacy(state) and spec["canonical_support"] == len(old.actions_by_canonical_successor_key(state,env.legal_actions(state))),"Selected lawful metadata")
            return snapshot
        action = old.decode_action(move["action"])
        require(action in env.legal_actions(state),"Selected replay illegal action")
        state = env.step(state,action)
    raise ValueError("Selected turn absent from audited source")
