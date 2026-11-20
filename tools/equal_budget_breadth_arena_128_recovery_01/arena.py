"""Conservative repetition-interval extension of the frozen equal-budget arena."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from pathlib import Path
import random
import time
import traceback

from tools.equal_budget_breadth_arena_128 import arena as original
from tools.equal_budget_breadth_arena_128_recovery_01 import common as c

old, baseline, windows = original.old, original.baseline, original.windows
ROOT, OUTPUT, CONFIG, FITS = original.ROOT, c.OUTPUT, original.CONFIG, original.FITS
require = original.require
world128 = original.world128
_MODELS, _MODEL_ARTIFACTS = original._MODELS, original._MODEL_ARTIFACTS
model_id, specs = original.model_id, original.specs
root_identity = original.root_identity
learned_proposal, expanded_candidates = original.learned_proposal, original.expanded_candidates
proposal_for_root, allocation_for_candidates = original.proposal_for_root, original.allocation_for_candidates
decision, statistics = original.decision, original.statistics


class AmbiguousDecision(ValueError):
    """The prescribed continuation budget does not determine one robust move."""
    def __init__(self, selection):
        super().__init__("Unresolved repetition can change the original-order winner")
        self.selection = selection


def initialize_models(base=OUTPUT, registry=None):
    if registry is not None:
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Fixture registry cannot target recovery production")
    if registry is None and not (Path(base)/"models_lock.json").exists():
        model_lock = c.verify_hash(c.load_inventory(base)["models"])
        require(model_lock.resolve() == (c.SOURCE/"models_lock.json").resolve(),"Changed original model registry")
        return original.initialize_models(c.SOURCE)
    return original.initialize_models(base,registry)


def check_stage(stage,base):
    original.check_stage(stage,base)
    if stage == "test":
        require(not Path(base).resolve().is_relative_to(OUTPUT.resolve()), "Synthetic fixture cannot target recovery production")


def game_path(stage,index,base=OUTPUT):
    check_stage(stage,base)
    return original.game_path(stage,index,base)


def root_path(stage,index,turn,base=OUTPUT):
    check_stage(stage,base)
    return original.root_path(stage,index,turn,base)


def _bindings(binding,recovery_binding,base):
    require(isinstance(recovery_binding,str) and bool(recovery_binding), "Missing recovery binding")
    if Path(base).resolve().is_relative_to(OUTPUT.resolve()):
        require(binding == c.ORIGINAL_BINDING, "Changed original manifest")


def _outcome(outcome):
    reason,value = outcome["reason"],outcome["wdl"]
    if reason == "repetition_cutoff":
        require(value is None and outcome["winner"] is None,"Censored repetition has an invented result")
        return None
    require(reason in old.VALID_NATURAL_TERMINATIONS and type(value) in (int,float)
            and value in (-1.,0.,1.),"Invalid inner ending")
    return int(value)


def select(worlds,world_counts):
    """Require one first-index maximum for every bounded unresolved assignment."""
    require(isinstance(world_counts,list) and 1 <= len(world_counts) <= 8
            and all(type(n) is int and 1 <= n <= 128 for n in world_counts),"Invalid support or world counts")
    require(len(worlds) == max(world_counts),"Wrong actual world count")
    sums,known,unknown = ([0]*len(world_counts) for _ in range(3))
    cells = []
    for wi,world in enumerate(worlds):
        active = [j for j,n in enumerate(world_counts) if wi < n]
        require(world["candidate_indices"] == active and all(type(j) is int for j in world["candidate_indices"]),"Changed active candidate support")
        require(len(world["outcomes"]) == len(active),"Incomplete forced-action support")
        for j,outcome in zip(active,world["outcomes"],strict=True):
            value = _outcome(outcome)
            if value is None:
                unknown[j] += 1
                cells.append({"world":wi,"candidate":j})
            else:
                sums[j] += value
                known[j] += 1
    require([a+b for a,b in zip(known,unknown,strict=True)] == world_counts,"Missing allocated candidate cells")
    if not cells:
        return original.select(worlds,world_counts)
    lower = [s-u for s,u in zip(sums,unknown,strict=True)]
    upper = [s+u for s,u in zip(sums,unknown,strict=True)]
    winners = [j for j in range(len(sums)) if all(
        lower[j]*world_counts[k] > upper[k]*world_counts[j] if k < j else
        lower[j]*world_counts[k] >= upper[k]*world_counts[j]
        for k in range(len(sums)) if k != j)]
    require(len(winners) <= 1,"Nonunique interval certificate")
    choice = winners[0] if winners else None
    value = {"choice":choice,"means":[None if u else s/n for s,u,n in zip(sums,unknown,world_counts,strict=True)],
             "sums_wdl":sums,"counts":list(world_counts),"certificate":{
                 "rule":"repetition_interval_original_order_v1","known_counts":known,
                 "unresolved_counts":unknown,"lower_sums":lower,"upper_sums":upper,
                 "unresolved_cells":cells,"choice":choice}}
    if choice is None:
        raise AmbiguousDecision(value)
    return value


def _check_reused_root(record,state,stream,source_model,identity,breadth,binding,recovery_binding):
    require(record["manifest"] == binding and record["root_identity"] == identity,"Inherited root identity mismatch")
    require(record.get("recovery_binding",recovery_binding) == recovery_binding,"Inherited recovery binding mismatch")
    require(old.digest(record["snapshot"]) == old.digest(old.serialize_state_snapshot(state)),"Inherited root state mismatch")
    require(record["planner"] == ("candidate" if breadth else "reference"),"Inherited planner mismatch")
    proposal = record["proposal"]
    require(proposal["model_id"] == source_model and proposal["model_sha256"] == _MODEL_ARTIFACTS[source_model]["sha256"],"Inherited model mismatch")
    require(proposal["root_identity"] == identity and proposal["breadth"] is breadth,"Inherited proposal identity mismatch")
    require(record["schedule"] == proposal["schedule"] == world128.root_schedule(state,stream,128),"Inherited schedule mismatch")
    require(record["candidates"] == proposal["candidates"],"Inherited candidates mismatch")
    require(record["allocation"] == proposal["allocation"] == allocation_for_candidates(record["candidates"],identity,breadth),"Inherited allocation mismatch")


def choose_root(state,stream,path,binding,source_model,identity,breadth,base=OUTPUT,recovery_binding=None):
    path,identity = Path(path),root_identity(identity)
    _bindings(binding,recovery_binding,base)
    require(path.resolve() == root_path(*identity,base=base).resolve(),"Root path identity mismatch")
    inherited_path = c.resolve_root(*identity,base=base)
    inherited = None if inherited_path is None else old.read_json(inherited_path)
    if inherited is not None:
        c.validate_origin(inherited,inherited_path,binding,base)
        _check_reused_root(inherited,state,stream,source_model,identity,breadth,binding,recovery_binding)
        if "failure" not in inherited:
            arithmetic = select(inherited["worlds"],inherited["allocation"]["world_counts"])
            require(all(inherited[key] == value for key,value in arithmetic.items()),"Inherited selection mismatch")
            return old.decode_action(inherited["candidates"]["actions"][inherited["choice"]]),inherited
        require(Path(inherited_path).resolve() != path.resolve(),"Recovery failures cannot be retried")
    require(not path.exists(),"One-shot recovery root; no overwrite or retry")
    wall,cpu = time.perf_counter(),time.process_time()
    record = {"manifest":binding,"recovery_binding":recovery_binding,"time":old.now(),
              "snapshot":old.serialize_state_snapshot(state),"privacy":baseline.privacy(state),
              "worlds":[],"root_identity":identity,"planner":"candidate" if breadth else "reference"}
    try:
        c.budget_boundary(base)
        if inherited is None:
            proposal = proposal_for_root(state,stream,source_model,identity,breadth)
            record.update(schedule=proposal["schedule"],candidates=proposal["candidates"],proposal=proposal,allocation=proposal["allocation"])
            record["scores"] = list(map(float,old._RAW.score_actions(state,old.SplendorEnv().legal_actions(state))))
        else:
            for key in ("schedule","candidates","proposal","allocation","scores","worlds"):
                record[key] = copy.deepcopy(inherited[key])
            record["recovery_origin"] = c.artifact(inherited_path)
            record["inherited_timing"] = inherited["timing"]
            record["inherited_cells"] = sum(len(w["outcomes"]) for w in record["worlds"])
        allocation = record["allocation"]
        actions = [old.decode_action(a) for a in record["candidates"]["actions"]]
        require(len(record["worlds"]) <= allocation["world_count"],"Excess inherited worlds")
        for wi,seed in enumerate(record["schedule"]["world_seeds"][:allocation["world_count"]]):
            c.budget_boundary(base)
            active = [j for j,n in enumerate(allocation["world_counts"]) if wi < n]
            if wi < len(record["worlds"]):
                row = record["worlds"][wi]
                require(row["seed"] == seed and row["candidate_indices"] == active and len(row["outcomes"]) <= len(active),"Invalid inherited world prefix")
                require(wi == len(record["worlds"])-1 or len(row["outcomes"]) == len(active),"Noncontiguous inherited cells")
                for outcome in row["outcomes"]:
                    _outcome(outcome)
                if len(row["outcomes"]) == len(active):
                    continue
            else:
                row = {"seed":seed,"candidate_indices":active,"outcomes":[]}
                record["worlds"].append(row)
            world = old.determinize_for_observer(state,state.current_player,random.Random(seed))
            state_hash = old.digest(old.serialize_state_snapshot(world))
            require(row.get("state_sha256",state_hash) == state_hash,"Inherited world state mismatch")
            row["state_sha256"] = state_hash
            legal = old.SplendorEnv().legal_actions(world)
            for j in active[len(row["outcomes"]):]:
                require(actions[j] in legal,"Illegal forced proposal")
                value = old.trajectory(world,actions[j])
                value["wdl"] = old.outcome_value(value["winner"],state.current_player) if value["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                row["outcomes"].append(value)
                _outcome(value)
        record.update(select(record["worlds"],allocation["world_counts"]))
    except AmbiguousDecision as exc:
        record.update(exc.selection)
        record["hold"] = "ambiguous_repetition_interval"
        record["failure"] = traceback.format_exc()
        raise
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["new_cells"] = sum(len(w["outcomes"]) for w in record["worlds"])-record.get("inherited_cells",0)
        record["timing"] = {"wall":time.perf_counter()-wall,"cpu":time.process_time()-cpu}
        old.save_new(path,record)
    return actions[record["choice"]],record


def play_game(stage,spec,binding=None,base=OUTPUT,test_initial_snapshot=None,recovery_binding=None):
    binding = c.ORIGINAL_BINDING if binding is None else binding
    _bindings(binding,recovery_binding,base)
    path = game_path(stage,spec["index"],base)
    require(spec == specs(stage)[spec["index"]],"Fixed schedule mismatch")
    require(test_initial_snapshot is None or stage == "test","Initial snapshot overrides are restricted to isolated test games")
    inherited_path = c.resolve_game(stage,spec["index"],base=base)
    if inherited_path is None:
        inherited_path = c.resolve_partial_game(stage,spec["index"],base=base)
    inherited = None if inherited_path is None else old.read_json(inherited_path)
    if inherited is not None:
        c.validate_origin(inherited,inherited_path,binding,base)
        require(inherited["manifest"] == binding and inherited["spec"] == spec,"Inherited game identity mismatch")
        require(inherited.get("recovery_binding",recovery_binding) == recovery_binding,"Inherited game recovery binding mismatch")
        if "failure" not in inherited:
            require(inherited["reason"] in old.VALID_NATURAL_TERMINATIONS,"Invalid inherited outer ending")
            return {"index":spec["index"],"sha256":old.file_hash(inherited_path)}
        require(Path(inherited_path).resolve() != path.resolve(),"Recovery games cannot be retried")
    require(not path.exists(),"One-shot recovery game; no overwrite or retry")
    wall,cpu = time.perf_counter(),time.process_time()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state() if test_initial_snapshot is None else old.deserialize_state_snapshot(test_initial_snapshot)
    record = {"manifest":binding,"recovery_binding":recovery_binding,"spec":spec,"time":old.now(),"trace":[],
              "initial_state_sha256":old.digest(old.serialize_state_snapshot(state))}
    if test_initial_snapshot is not None:
        record["test_initial_snapshot"] = old.serialize_state_snapshot(state)
    if inherited is not None:
        require(inherited["initial_state_sha256"] == record["initial_state_sha256"],"Inherited initial state mismatch")
        record["recovery_origin"] = c.artifact(inherited_path)
        record["inherited_timing"] = inherited["timing"]
        record["inherited_turns"] = len(inherited["trace"])
    visits,forced,idle,hasher = Counter(),0,0,hashlib.sha256()
    reason,winner = "completed",None
    try:
        while not state.terminal:
            c.budget_boundary(base)
            signature = old.state_signature(state)
            visits[signature] += 1
            if visits[signature] >= 4:
                reason = "repetition_cutoff"; break
            if state.turn_index >= 400:
                reason = "max_turns"; break
            legal = env.legal_actions(state)
            if not legal:
                reason = "stalled"; break
            row = {"turn":state.turn_index,"actor":state.current_player,"state_sha256":old.digest(old.serialize_state_snapshot(state))}
            breadth = state.current_player == spec["candidate_seat"]
            source_model = model_id(stage,spec["fit"],breadth)
            identity = [stage,spec["index"],state.turn_index]
            rp = root_path(*identity,base=base)
            prefix_index = len(record["trace"])
            if inherited is not None and prefix_index == len(inherited["trace"]):
                require(old.digest(old.serialize_state_snapshot(state)) == old.digest(inherited["final_state"]),"Inherited outer prefix state mismatch")
            try:
                action,_ = choose_root(state,spec["search_stream"],rp,binding,source_model,identity,breadth,base,recovery_binding)
            except BaseException:
                failure_path = rp if rp.exists() else c.resolve_root(*identity,base=base)
                if failure_path is not None:
                    row["root_sha256"] = old.file_hash(failure_path)
                record["failed_decision"] = row
                raise
            resolved = c.resolve_root(*identity,base=base)
            require(resolved is not None,"Missing completed root")
            row["root_sha256"] = old.file_hash(resolved)
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                record["failed_decision"] = {**row,"action":None if action is None else old.serialize_action(action)}
                break
            row["action"] = old.serialize_action(action)
            if inherited is not None and prefix_index < len(inherited["trace"]):
                require(row == inherited["trace"][prefix_index],"Changed inherited outer action or root hash")
            record["trace"].append(row)
            hasher.update(old.digest(row["action"]).encode())
            successor = env.step(state,action)
            forced = forced+1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state,successor,action) else idle+1
            state = successor
            if forced >= 2:
                reason,winner = "forced_pass_deadlock",old.adjudicate_scores(state); break
            if idle >= 60:
                reason = "no_progress_cutoff"; break
        if reason == "completed":winner = state.winner
        record.update(reason=reason,winner=winner)
        require(reason in old.VALID_NATURAL_TERMINATIONS,f"Invalid outer ending: {reason}")
    except BaseException:
        record["failure"] = traceback.format_exc()
        if "reason" not in record:record.update(reason="exception",winner=None)
        raise
    finally:
        final = old.serialize_state_snapshot(state)
        record.update(final_state=final,final_state_sha256=old.digest(final),turns=len(record["trace"]),
                      trace_sha256=hasher.hexdigest(),timing={"wall":time.perf_counter()-wall,"cpu":time.process_time()-cpu})
        old.save_new(path,record)
    return {"index":spec["index"],"sha256":old.file_hash(path)}
