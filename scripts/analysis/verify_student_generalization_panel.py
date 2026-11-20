"""Full trajectory replay with independent student decision reconstruction."""
from collections import Counter
from pathlib import Path
import math

from splendor_ai.eval import student_generalization_panel as h
from scripts.analysis import verify_teacher_budget_161 as independent


def replay_game(stage, spec, item, binding, base=h.OUTPUT):
    old = h.old
    h.game_path(stage, spec["index"], base)  # enforce isolated fixtures
    path = Path(item["path"])
    h.require(old.file_hash(path) == item["sha256"], "Game hash changed")
    game = old.read_json(path)
    h.require(game["manifest"] == binding and game["spec"] == spec and "failure" not in game, "Invalid game")
    if stage != "test":
        h.require("test_initial_snapshot" not in game and h.schedule(stage)[spec["index"]] == spec, "Invalid production record")
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = old.deserialize_state_snapshot(game["test_initial_snapshot"]) if "test_initial_snapshot" in game else env.initial_state()
    bot = h.opponent_bot(spec["opponent"])
    seen, passes, idle = Counter(), 0, 0
    timings = []
    for row in game["trace"]:
        h.require(not state.terminal and state.turn_index < 400 and passes < 2, "Trace continues after ending")
        seen[old.state_signature(state)] += 1
        h.require(seen[old.state_signature(state)] < 4, "Repetition")
        h.require(row["turn"] == state.turn_index and row["actor"] == state.current_player, "Turn mismatch")
        h.require(row["state_sha256"] == old.digest(old.serialize_state_snapshot(state)), "State mismatch")
        legal = env.legal_actions(state)
        h.require(legal and len(legal) == row["legal_support"], "Legal support mismatch")
        candidate = state.current_player == spec["seat"]
        key = f"{spec['arm']}_{spec['seed']}" if candidate else spec["opponent"]
        h.require(row["model"] == key, "Wrong actor model")
        if candidate:
            action, evidence = independent.a.independent_decision(state, spec["stream"], h.MODELS[key])
            independent.a.compare(evidence, row["evidence"])
        else:
            action = bot.choose_action(env, state, legal)
            h.require("evidence" not in row, "Unexpected opponent evidence")
        h.require(action in legal and old.serialize_action(action) == row["action"], "Action mismatch")
        h.require(set(row["timing"]) == {"wall", "cpu"} and
                  all(math.isfinite(v) and v >= 0 for v in row["timing"].values()), "Invalid timing")
        timings.append({"model": key, **row["timing"]})
        successor = env.step(state, action)
        passes = passes+1 if old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(state, successor, action) else idle+1
        h.require(passes >= 2 or idle < 60, "No progress")
        state = successor
    h.require(state.terminal or passes >= 2, "Incomplete trajectory")
    reason = "forced_pass_deadlock" if passes >= 2 else "completed"
    winner = old.adjudicate_scores(state) if passes >= 2 else state.winner
    h.require(game["reason"] == reason and game["winner"] == winner, "Ending mismatch")
    h.require(old.digest(old.serialize_state_snapshot(state)) == old.digest(game["final_state"]), "Final state mismatch")
    h.require(math.isfinite(game["elapsed_seconds"]) and
              game["elapsed_seconds"] >= sum(t["wall"] for t in timings), "Invalid elapsed time")
    score = .5 if winner is None else float(winner == spec["seat"])
    report = {"manifest": binding, "game_artifact": item, "full_replay": True,
              "game": {**spec, "score": score, "reason": reason,
                       "margin": state.players[spec["seat"]].score-state.players[1-spec["seat"]].score},
              "timings": timings}
    destination = Path(base) / stage / "audit" / f"game_{spec['index']:04d}.json"
    old.save_new(destination, report)
    return h.parent.artifact(destination, index=spec["index"])
