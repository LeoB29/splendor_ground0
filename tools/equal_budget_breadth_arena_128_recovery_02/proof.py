"""Exact saved-continuation replay and conservative future-context equivalence."""
from collections import Counter
import copy
import hashlib
import random

from tools.equal_budget_breadth_arena_128_recovery_02 import common as c

old,require = c.old,c.require
FIELDS = ("reason","winner","turns","trace_sha256","final_state_sha256")


def canonical_snapshot(snapshot):
    value = copy.deepcopy(snapshot)
    for player in value["players"]:
        player["purchased_card_ids"] = sorted(player["purchased_card_ids"])
    return value


def _inventory(snapshot):
    return tuple(tuple(sorted(player["purchased_card_ids"])) for player in snapshot["players"])


def _world(record,wi):
    require(type(wi) is int and 0 <= wi < len(record["worlds"]),"Invalid proof world")
    state = old.deserialize_state_snapshot(record["snapshot"])
    seed = record["schedule"]["world_seeds"][wi]
    require(record["worlds"][wi]["seed"] == seed,"Changed proof seed")
    world = old.determinize_for_observer(state,state.current_player,random.Random(seed))
    require(old.digest(old.serialize_state_snapshot(world)) == record["worlds"][wi]["state_sha256"],"Changed proof world")
    return world,state.current_player


def context_from_replay(record,wi,ci,replay):
    world,root_actor = _world(record,wi)
    row = record["worlds"][wi]
    require(type(ci) is int and ci in row["candidate_indices"],"Invalid proof candidate")
    saved = row["outcomes"][row["candidate_indices"].index(ci)]
    require(saved["reason"] == "repetition_cutoff" and saved["winner"] is None and saved["wdl"] is None,"Proof is not an unresolved repetition")
    require(all(replay[key] == saved[key] for key in FIELDS),"Proof replay changed recorded result")
    require(set(replay) == set(FIELDS)|{"trace","final_state"},"Unexpected proof replay schema")
    require(type(replay["turns"]) is int and replay["turns"] == len(replay["trace"]),"Incomplete proof trace")
    env,seen,history = old.SplendorEnv(),Counter(),[]
    forced,idle = 0,0
    hasher = hashlib.sha256()
    for i,move in enumerate(replay["trace"]):
        require(not world.terminal,"Proof continues after natural ending")
        snapshot = old.serialize_state_snapshot(world)
        require(old.digest(snapshot) == move["state_hash"] and type(move["turn"]) is int
                and move["turn"] == world.turn_index and type(move["actor"]) is int
                and move["actor"] == world.current_player,"Proof trace state or actor differs")
        signature = old.state_signature(world)
        seen[signature] += 1
        require(seen[signature] < old.CONFIG["repetition_limit"] and world.turn_index < old.CONFIG["max_turns"],"Proof continues past an original guard")
        canonical = canonical_snapshot(snapshot)
        history.append((_inventory(snapshot),old.digest(old.state_signature(old.deserialize_state_snapshot(canonical)))))
        legal = env.legal_actions(world)
        action = old.decode_action(move["action"])
        require(legal and action in legal,"Illegal proof action")
        if i == 0:
            require(old.digest(move["action"]) == old.digest(record["candidates"]["actions"][ci]),"Changed proof root action")
        hasher.update(old.digest(move["action"]).encode())
        successor = env.step(world,action)
        forced = forced+1 if old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(world,successor,action) else idle+1
        require(forced < 2 and idle < old.CONFIG["no_progress_limit"],"Proof bypasses another original ending")
        world = successor
    final = old.serialize_state_snapshot(world)
    require(not world.terminal and old.digest(final) == replay["final_state_sha256"]
            and old.digest(replay["final_state"]) == old.digest(final),"Proof final state differs")
    require(hasher.hexdigest() == replay["trace_sha256"],"Proof action trace hash differs")
    final_signature = old.state_signature(world)
    seen[final_signature] += 1
    require(seen[final_signature] == old.CONFIG["repetition_limit"],"Proof final cutoff is not the prescribed visit")
    canonical = canonical_snapshot(final)
    history.append((_inventory(final),old.digest(old.state_signature(old.deserialize_state_snapshot(canonical)))))
    live = Counter(signature for inventory,signature in history if inventory == _inventory(final))
    return {"canonical_final_state_sha256":old.digest(canonical),
            "live_repetition_counts":[{"signature_sha256":signature,"count":live[signature]} for signature in sorted(live)],
            "forced_pass_count":forced,"no_progress_count":idle,"root_actor":root_actor,
            "policy_sha256":old.CHECKPOINT_SHA}


def build(record,wi,ci):
    require(old.file_hash(old.CHECKPOINT) == old.CHECKPOINT_SHA,"Changed continuation checkpoint")
    world,_ = _world(record,wi)
    action = old.decode_action(record["candidates"]["actions"][ci])
    replay = old.trajectory(world,action,include_trace=True)
    context = context_from_replay(record,wi,ci,replay)
    return {"world":wi,"candidate":ci,"replay":replay,"context":context,"context_sha256":old.digest(context)}


def validate(record,equivalence_proofs):
    expected = [(wi,ci) for wi,row in enumerate(record["worlds"])
                for ci,outcome in zip(row["candidate_indices"],row["outcomes"],strict=True)
                if outcome["reason"] == "repetition_cutoff"]
    require([(item["world"],item["candidate"]) for item in equivalence_proofs] == expected,"Missing, reordered or extra equivalence proofs")
    for item in equivalence_proofs:
        context = context_from_replay(record,item["world"],item["candidate"],item["replay"])
        require(item["context"] == context and item["context_sha256"] == old.digest(context),"Changed proof context")
    return True
