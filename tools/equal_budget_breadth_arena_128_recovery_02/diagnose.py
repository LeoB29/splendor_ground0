"""One exact reproduction of each of the two ambiguous saved continuations."""
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random

from tools.equal_budget_breadth_arena_128 import common as c
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector

DEST = c.OUTPUT/"recovery_diagnosis_02"
SOURCE = c.OUTPUT/"recovery_01/main/roots/game_0310/turn_058.json"
EXPECTED = "a612be2ff084af6f89f83389c49b2f78058406d8e1e5faaba988f2fdfed65ae7"
FIELDS = ("reason","winner","turns","trace_sha256","final_state_sha256")


def plain(value):
    return json.loads(json.dumps(value,sort_keys=True))


def differences(a,b,path=""):
    if a == b:return []
    if isinstance(a,dict) and isinstance(b,dict) and a.keys() == b.keys():
        return [item for key in sorted(a) for item in differences(a[key],b[key],f"{path}/{key}")]
    if isinstance(a,list) and isinstance(b,list):
        if Counter(json.dumps(v,sort_keys=True) for v in a) == Counter(json.dumps(v,sort_keys=True) for v in b):
            return [{"path":path,"kind":"order_only","candidate_0":a,"candidate_1":b}]
        if len(a) == len(b):
            return [item for i,(x,y) in enumerate(zip(a,b)) for item in differences(x,y,f"{path}/{i}")]
    return [{"path":path,"kind":"value","candidate_0":a,"candidate_1":b}]


def final_facts(state):
    legal = c.old.SplendorEnv().legal_actions(state)
    config = c.old._RAW._model.config
    observations = [list(encode_public_observation_tensor(state,seat).vector) for seat in (0,1)]
    vectors = [list(fit_feature_vector(feature.vector,config.action_feature_size))
               for feature in encode_legal_action_features(state=state,legal_actions=legal,player_id=state.current_player)]
    scores = list(c.old._RAW.score_actions(state,legal))
    return {"state_signature":plain(c.old.state_signature(state)),
            "observations_by_seat":observations,"model_config":asdict(config),
            "model_observation":list(fit_feature_vector(observations[state.current_player],config.observation_size)),
            "legal_actions":[c.old.serialize_action(a) for a in legal],"model_action_features":vectors,
            "model_scores":scores,"greedy_index":max(range(len(scores)),key=lambda j:scores[j]) if scores else None,
            "adjudicated_scores_winner":c.old.adjudicate_scores(state)}


def reproduce(record,wi,ci):
    root = c.old.deserialize_state_snapshot(record["snapshot"])
    world = c.old.determinize_for_observer(root,root.current_player,random.Random(record["schedule"]["world_seeds"][wi]))
    saved_world = record["worlds"][wi]
    c.require(c.old.digest(c.old.serialize_state_snapshot(world)) == saved_world["state_sha256"],"World did not reproduce")
    action = c.old.decode_action(record["candidates"]["actions"][ci])
    result = c.old.trajectory(world,action,include_trace=True)
    recorded = saved_world["outcomes"][saved_world["candidate_indices"].index(ci)]
    c.require(all(result[key] == recorded[key] for key in FIELDS),"Exact saved continuation did not reproduce")
    c.require(recorded["wdl"] is None and result["reason"] == "repetition_cutoff" and result["winner"] is None,"Unexpected ambiguity kind")
    env = c.old.SplendorEnv()
    positions,visits = [],defaultdict(list)
    forced,idle = 0,0
    for i,move in enumerate(result["trace"]):
        snapshot = c.old.serialize_state_snapshot(world)
        c.require(c.old.digest(snapshot) == move["state_hash"],"Saved trace replay state differs")
        signature = c.old.digest(c.old.state_signature(world))
        visits[signature].append(i)
        positions.append({"step":i,"signature":signature,"snapshot":snapshot,"action":move["action"]})
        legal = env.legal_actions(world)
        action = c.old.decode_action(move["action"])
        c.require(action in legal,"Saved action is illegal")
        successor = env.step(world,action)
        forced = forced+1 if c.old.is_forced_pass_turn(legal) else 0
        idle = 0 if c.old.is_progress_transition(world,successor,action) else idle+1
        world = successor
    snapshot = c.old.serialize_state_snapshot(world)
    c.require(c.old.digest(snapshot) == result["final_state_sha256"],"Final replay state differs")
    signature = c.old.digest(c.old.state_signature(world))
    visits[signature].append(len(result["trace"]))
    positions.append({"step":len(result["trace"]),"signature":signature,"snapshot":snapshot})
    return {"world_index":wi,"candidate_index":ci,"recorded":recorded,"reproduced":result,
            "same_recorded_fields":True,"positions":positions,
            "repeated_signatures":{key:steps for key,steps in visits.items() if len(steps)>1},
            "termination_context":{"signature_visits":{key:len(steps) for key,steps in visits.items()},
                                   "forced_passes":forced,"no_progress":idle},
            "final_facts":final_facts(world)}


def main():
    c.require(not DEST.exists(),"Preserve existing second diagnosis")
    c.require(c.old.file_hash(SOURCE) == EXPECTED,"Saved ambiguous root changed")
    record = c.old.read_json(SOURCE)
    c.old.initialize_worker()
    c.require(c.old.file_hash(c.old.CHECKPOINT) == c.old.CHECKPOINT_SHA,"Changed continuation checkpoint")
    branches,branch_refs = [],[]
    for ci in (0,1):
        branch = reproduce(record,42,ci)
        path = DEST/f"candidate_{ci}_reproduction.json"
        c.old.save_new(path,{"source":c.artifact(SOURCE),**branch})
        branches.append(branch)
        branch_refs.append(c.artifact(path))
    states = [plain(branch["reproduced"]["final_state"]) for branch in branches]
    sorted_purchases = copy.deepcopy(states)
    for state in sorted_purchases:
        for player in state["players"]:
            player["purchased_card_ids"] = sorted(player["purchased_card_ids"])
    diff = differences(*states)
    facts_diff = differences(*[plain(branch["final_facts"]) for branch in branches])
    summary = {"source":c.artifact(SOURCE),"time":c.old.now(),"module":c.artifact(Path(__file__)),
               "same_recorded_fields":True,"continuation_calls_this_run":2,"extended_continuation_calls":0,
               "prior_attempt":{"verified_but_unsaved_continuation_calls":2,
                                "failure":"Postprocessing KeyError: purchased_cards; actual snapshot key purchased_card_ids",
                                "authorized_bounded_repeat":True},"total_known_continuation_calls":4,
               "state_differences":diff,"final_facts_differences":facts_diff,
               "equal_after_sorting_purchased_cards_only":sorted_purchases[0] == sorted_purchases[1],
               "equal_final_termination_context":branches[0]["termination_context"] == branches[1]["termination_context"],
               "branches":branch_refs,"scope":"Only the two saved failed branches; one authorized exact repeat after a postprocessing error. No extended rollout, changed rule, full game, or training."}
    c.old.save_new(DEST/"comparison.json",summary)
    c.require(c.old.file_hash(SOURCE) == EXPECTED,"Source changed during diagnosis")
    print({"same_recorded_fields":True,"state_differences":diff,
           "fact_difference_paths":[item["path"] for item in facts_diff],
           "equal_after_sorting_purchased_cards_only":summary["equal_after_sorting_purchased_cards_only"],
           "saved":str(DEST/"comparison.json")},flush=True)


if __name__ == "__main__":main()
