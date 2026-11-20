"""160 gameplay: exact 155 with 128 versus 32 sampled hidden worlds.

The reference calls the frozen 155 root chooser directly. The candidate only
extends its world schedule; proposals, continuations, objective and ties are
unchanged. All evidence is one-shot, including failed roots and partial games.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import random
import time
import traceback

import numpy as np

from splendor_ai.alphazero import run_rollout_selector_gameplay as baseline

old = baseline.old
OUTPUT = Path("outputs/world_budget_160")
_PRODUCTION_OUTPUT = OUTPUT.resolve()
ROOT_BUDGETS = (32, 128)


def initialize_worker():
    """Load only the exact frozen greedy035 policy used throughout 155."""
    old.initialize_worker()


def _check_budget(worlds):
    if type(worlds) is not int or worlds not in ROOT_BUDGETS:
        raise ValueError("160 permits exactly 32 or 128 worlds")


def _check_stage(stage, base=OUTPUT):
    if stage in ("probe", "main"):
        return
    target = Path(base).resolve()
    if (stage == "test" and target != _PRODUCTION_OUTPUT
            and _PRODUCTION_OUTPUT not in target.parents):
        return
    raise ValueError("Unknown stage or test stage targeting production evidence")


def root_path(stage, index, turn, base=OUTPUT):
    _check_stage(stage, base)
    if type(index) is not int or index < 0 or type(turn) is not int or turn < 0:
        raise ValueError("Game index and turn must be nonnegative integers")
    return Path(base) / stage / "roots" / f"game_{index:04d}" / f"turn_{turn:03d}.json"


def game_path(stage, index, base=OUTPUT):
    _check_stage(stage, base)
    if type(index) is not int or index < 0:
        raise ValueError("Game index must be a nonnegative integer")
    return Path(base) / stage / "games" / f"game_{index:04d}.json"


def root_schedule(state, stream, worlds):
    """Preserve 155's lawful proposal seed and first 32 common world draws."""
    _check_budget(worlds)
    schedule = baseline.root_schedule(state, stream)
    if len(schedule["world_seeds"]) != 32:
        raise ValueError("Frozen 155 world count changed")
    if worlds == 32:
        return schedule
    belief = schedule["belief_sha256"]
    return {**schedule, "world_seeds": [
        *schedule["world_seeds"],
        *(f"{baseline.CONFIG['world_tag']}|{stream}|{belief}|{j:02d}"
          for j in range(32, worlds)),
    ]}


def candidate_set(state, nonce):
    return baseline.candidate_set(state, nonce)


def select(rows, support, worlds):
    """Average exact terminal WDL; retain the first proposal in exact ties."""
    _check_budget(worlds)
    if worlds == 32:
        return baseline.select(rows, support)
    if not 1 <= support <= 4 or len(rows) != worlds:
        raise ValueError("Invalid support or world count")
    if any(len(row["outcomes"]) != support for row in rows):
        raise ValueError("Incomplete forced-action support")
    if any(outcome["reason"] not in old.VALID_NATURAL_TERMINATIONS
           or outcome["wdl"] not in (-1., 0., 1.)
           for row in rows for outcome in row["outcomes"]):
        raise ValueError("Invalid inner ending: no exclusion or replacement")
    means = [sum(row["outcomes"][j]["wdl"] for row in rows) / worlds
             for j in range(support)]
    return {"choice": max(range(support), key=lambda j: means[j]), "means": means}


def choose_root(state, stream, path, binding, worlds):
    """Change only 155's world count; never reuse or overwrite existing roots."""
    _check_budget(worlds)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"160 does not automatically reuse or retry roots: {path}")
    if worlds == 32:
        return baseline.choose_root(state, stream, path, binding)
    schedule = root_schedule(state, stream, worlds)
    wall, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "time": old.now(),
              "snapshot": old.serialize_state_snapshot(state), "schedule": schedule,
              "privacy": baseline.privacy(state), "worlds": []}
    try:
        legal = old.SplendorEnv().legal_actions(state)
        scores = old._RAW.score_actions(state, legal)
        assert len(scores) == len(legal) and np.isfinite(scores).all()
        record["scores"] = [float(value) for value in scores]
        record["candidates"] = candidate_set(state, schedule["candidate_nonce"])
        assert record["candidates"]["actions"][0] == old.serialize_action(
            old._RAW.choose_action(old.SplendorEnv(), state, legal))
        actions = [old.decode_action(payload) for payload in record["candidates"]["actions"]]
        for seed in schedule["world_seeds"]:
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)),
                   "outcomes": []}
            record["worlds"].append(row)
            for action in actions:
                assert action in old.SplendorEnv().legal_actions(world)
                result = old.trajectory(world, action)
                result["wdl"] = (old.outcome_value(result["winner"], state.current_player)
                                 if result["reason"] in old.VALID_NATURAL_TERMINATIONS else None)
                row["outcomes"].append(result)
                if result["wdl"] is None:
                    raise ValueError("Invalid rollout ending: no exclusion or retry")
        record.update(select(record["worlds"], len(actions), worlds))
    except Exception:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
        old.save_new(path, record)
    return actions[record["choice"]], record


def play_game(stage, spec, binding, base=OUTPUT, test_initial_snapshot=None):
    """Play both fixed world budgets once, retaining every actual decision."""
    path = game_path(stage, spec["index"], base)
    if path.exists():
        raise FileExistsError(f"160 does not automatically reuse or retry games: {path}")
    if type(spec["candidate_seat"]) is not int or spec["candidate_seat"] not in (0, 1):
        raise ValueError("Invalid candidate seat")
    if test_initial_snapshot is not None and stage != "test":
        raise ValueError("Initial snapshot overrides are restricted to isolated test games")
    wall, cpu = time.perf_counter(), time.process_time()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = (env.initial_state() if test_initial_snapshot is None
             else old.deserialize_state_snapshot(test_initial_snapshot))
    record = {"manifest": binding, "spec": spec, "time": old.now(), "trace": [],
              "initial_state_sha256": old.digest(old.serialize_state_snapshot(state))}
    if test_initial_snapshot is not None:
        record["test_initial_snapshot"] = old.serialize_state_snapshot(state)
    visits, forced, idle, hasher = Counter(), 0, 0, hashlib.sha256()
    reason, winner = "completed", None
    try:
        while not state.terminal:
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
            row = {"turn": state.turn_index, "actor": state.current_player,
                   "state_sha256": old.digest(old.serialize_state_snapshot(state))}
            worlds = 128 if state.current_player == spec["candidate_seat"] else 32
            rp = root_path(stage, spec["index"], state.turn_index, base)
            try:
                action, _ = choose_root(state, spec["search_stream"], rp, binding, worlds)
            except Exception:
                if rp.exists():
                    row["root_sha256"] = old.file_hash(rp)
                record["failed_decision"] = row
                raise
            row["root_sha256"] = old.file_hash(rp)
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                record["failed_decision"] = {
                    **row, "action": None if action is None else old.serialize_action(action)}
                break
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            hasher.update(old.digest(row["action"]).encode())
            successor = env.step(state, action)
            forced = forced + 1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, successor, action) else idle + 1
            state = successor
            if forced >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            if idle >= 60:
                reason = "no_progress_cutoff"
                break
        if reason == "completed":
            winner = state.winner
        record.update(reason=reason, winner=winner)
        if reason not in old.VALID_NATURAL_TERMINATIONS:
            raise ValueError(f"Invalid outer ending: {reason}")
    except Exception:
        record["failure"] = traceback.format_exc()
        if "reason" not in record:
            record.update(reason="exception", winner=None)
        raise
    finally:
        final = old.serialize_state_snapshot(state)
        record.update(final_state=final, final_state_sha256=old.digest(final),
                      turns=len(record["trace"]), trace_sha256=hasher.hexdigest(),
                      timing={"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu})
        old.save_new(path, record)
    return {"index": spec["index"], "sha256": old.file_hash(path)}
