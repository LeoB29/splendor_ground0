"""158 gameplay: the unchanged 155 root operator with exact 157 continuations.

Only the candidate's simulated future action selector changes. Both players in
each candidate future use the same sealed treatment model. Reference roots call
the frozen 155 implementation directly. No fitting or nested search occurs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import random
import re
import time
import traceback

import numpy as np

from splendor_ai.alphazero import fast_selector_157 as f
from splendor_ai.alphazero import run_fast_selector_transfer as parent157
from splendor_ai.alphazero import run_rollout_selector_gameplay as baseline
from splendor_ai.alphazero.strong_policy import StrongPolicyInference

old = baseline.old
OUTPUT = Path("outputs/learned_continuation_158")
_PRODUCTION_OUTPUT = OUTPUT.resolve()
MODELS = ("treatment_157001", "treatment_157002", "treatment_157003")
MODELS_CACHE = {}
SELECTED_LOCK_SHA256 = "4942eebb38de81b40d38315ec4abaf6b306f9340f4fd7051113250d4831044d8"
PARENT_BINDING = "3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122"


def selected_models():
    """Resolve all three previously sealed treatment selections without choice."""
    path = parent157.OUTPUT / "selected_lock.json"
    assert old.file_hash(path) == SELECTED_LOCK_SHA256, "157 selection seal changed"
    lock = old.read_json(path)
    assert lock["manifest"] == PARENT_BINDING
    selected = {name: dict(lock["selected"][name]) for name in MODELS}
    for name, update in zip(MODELS, (1220, 1220, 1144), strict=True):
        item = selected[name]
        expected = parent157.OUTPUT / "fits" / name / f"checkpoint_{update:04d}.pt"
        assert item["path"] == expected.as_posix() and item["update"] == update
        assert old.file_hash(expected) == item["sha256"], f"Changed checkpoint: {name}"
    return selected


def initialize_worker():
    """Load the fixed proposer's weights and all three fixed treatment scorers."""
    old.initialize_worker()
    selected = selected_models()
    loaded = {name: StrongPolicyInference(item["path"]).model.eval()
              for name, item in selected.items()}
    MODELS_CACHE.clear()
    MODELS_CACHE.update(loaded)


def _check_stage(stage):
    if stage in ("probe", "main"):
        return
    redirected = OUTPUT.resolve()
    if (stage == "test" and redirected != _PRODUCTION_OUTPUT
            and _PRODUCTION_OUTPUT not in redirected.parents):
        return
    raise ValueError("Unknown stage or test stage targeting production evidence")


def root_path(stage, index, turn):
    _check_stage(stage)
    return OUTPUT / stage / "roots" / f"game_{index:04d}" / f"turn_{turn:03d}.json"


def game_path(stage, index):
    _check_stage(stage)
    return OUTPUT / stage / "games" / f"game_{index:04d}.json"


def progress_write(path, row):
    """Retry Windows reader contention only for explicitly mutable telemetry.

    Immutable experiment evidence always uses save_new and receives no retry.
    A brief status reader can temporarily block replacement on Windows; bounded
    retries here cannot rerun a game, model decision, or immutable record write.
    """
    path = Path(path)
    try:
        relative = path.resolve().relative_to(OUTPUT.resolve())
    except ValueError as exc:
        raise ValueError("Progress target is outside the 158 output directory") from exc
    parts = relative.parts
    allowed = relative.as_posix() in ("progress.json", "audit_progress.json")
    redirected = OUTPUT.resolve()
    test_directory = redirected != _PRODUCTION_OUTPUT and _PRODUCTION_OUTPUT not in redirected.parents
    mutable_stages = ("probe", "main", "test") if test_directory else ("probe", "main")
    allowed |= (len(parts) == 3 and parts[0] in mutable_stages
                and parts[1] == "telemetry"
                and re.fullmatch(r"game_\d{4}\.json", parts[2]) is not None)
    if not allowed:
        raise ValueError("Retry writer is reserved for mutable progress/telemetry")
    for attempt in range(51):
        try:
            old.atomic_write(path, row)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 50:
                raise
            time.sleep(.1)


def continuation_stream(root_stream, world_index, step, actor):
    """Corresponding future slots share RNG across forced root alternatives.

    The root stream is the exogenous pair stream. The exact 157 chooser then
    incorporates the acting player's lawful inner belief hash. In particular,
    the original root observer's belief does not seed the opponent's decisions:
    it may contain private facts that the opposing simulated actor cannot know.
    No hidden snapshot identity or candidate ordinal enters this stream, and no
    mutable RNG is consumed by earlier alternatives.
    """
    return old.digest(["158-continuation-v1", root_stream, world_index, step, actor])


def trajectory(state, root_action, stream, world_index, model_id):
    """Terminal continuation with streamed hashes and retained failure evidence.

    Step zero is the forced root action. Every subsequent simulated action is
    chosen by the complete frozen 157 controller, for either acting seat. The
    transient action trace is retained only on invalid termination or exception;
    normal results bind all choices and full logits through decision_sha256.
    """
    if model_id not in MODELS_CACHE:
        raise ValueError(f"Uninitialized continuation model: {model_id}")
    env = old.SplendorEnv()
    seen, forced, no_progress, turns = {}, 0, 0, 0
    action_hash, decision_hash = hashlib.sha256(), hashlib.sha256()
    trace, decisions = [], 0
    reason, winner, failure = "completed", None, None
    attempted = None
    try:
        while not state.terminal:
            signature = old.state_signature(state)
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] >= 4:
                reason = "repetition_cutoff"
                break
            if state.turn_index >= 400:
                reason = "max_turns"
                break
            legal = env.legal_actions(state)
            if not legal:
                reason = "stalled"
                break
            actor = state.current_player
            if turns == 0:
                action = root_action
            else:
                proposal_stream = continuation_stream(stream, world_index, turns, actor)
                action, decision = f.choose(state, proposal_stream, MODELS_CACHE[model_id])
                decision_hash.update(old.digest({"step": turns, "actor": actor,
                                                "stream": proposal_stream,
                                                "decision": decision}).encode())
                decisions += 1
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                attempted = None if action is None else old.serialize_action(action)
                break
            payload = old.serialize_action(action)
            action_hash.update(old.digest(payload).encode())
            trace.append({"step": turns, "turn": state.turn_index, "actor": actor,
                          "action": payload})
            successor = env.step(state, action)
            turns += 1
            forced = forced + 1 if old.is_forced_pass_turn(legal) else 0
            no_progress = 0 if old.is_progress_transition(state, successor, action) else no_progress + 1
            state = successor
            if forced >= 2:
                reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
                break
            if no_progress >= 60:
                reason = "no_progress_cutoff"
                break
        if reason == "completed":
            winner = state.winner
    except Exception:
        reason, winner, failure = "exception", None, traceback.format_exc()
    final = old.serialize_state_snapshot(state)
    result = {"reason": reason, "winner": winner, "turns": turns,
              "trace_sha256": action_hash.hexdigest(), "final_state_sha256": old.digest(final),
              "decision_sha256": decision_hash.hexdigest(), "learned_decisions": decisions}
    if reason not in old.VALID_NATURAL_TERMINATIONS:
        result.update(trace=trace, final_state=final)
        if reason == "illegal_or_missing_action":
            result["attempted_action"] = attempted
        if failure is not None:
            result["failure"] = failure
    return result


def choose_root(state, stream, path, binding, model_id):
    """Keep every 155 root operation except the candidate's continuation policy."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"158 does not automatically reuse or retry roots: {path}")
    if model_id is None:
        return baseline.choose_root(state, stream, path, binding)
    if model_id not in MODELS or model_id not in MODELS_CACHE:
        raise ValueError(f"Unknown or uninitialized treatment model: {model_id}")
    schedule = baseline.root_schedule(state, stream)
    wall, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "time": old.now(),
              "snapshot": old.serialize_state_snapshot(state), "schedule": schedule,
              "privacy": baseline.privacy(state), "continuation_model": model_id,
              "worlds": []}
    try:
        legal = old.SplendorEnv().legal_actions(state)
        scores = old._RAW.score_actions(state, legal)
        assert len(scores) == len(legal) and np.isfinite(scores).all()
        record["scores"] = [float(value) for value in scores]
        record["candidates"] = baseline.candidate_set(state, schedule["candidate_nonce"])
        assert record["candidates"]["actions"][0] == old.serialize_action(
            old._RAW.choose_action(old.SplendorEnv(), state, legal))
        actions = [old.decode_action(payload) for payload in record["candidates"]["actions"]]
        for world_index, seed in enumerate(schedule["world_seeds"]):
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)),
                   "outcomes": []}
            record["worlds"].append(row)
            for action in actions:
                assert action in old.SplendorEnv().legal_actions(world)
                result = trajectory(world, action, stream, world_index, model_id)
                result["wdl"] = (old.outcome_value(result["winner"], state.current_player)
                                 if result["reason"] in old.VALID_NATURAL_TERMINATIONS else None)
                row["outcomes"].append(result)
                if result["wdl"] is None:
                    raise ValueError("Invalid learned rollout ending: no exclusion or retry")
        record.update(baseline.select(record["worlds"], len(actions)))
    except Exception:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
        old.save_new(path, record)
    return actions[record["choice"]], record


def play_game(stage, spec, binding):
    """Play both frozen planners once, preserving every root and actual action."""
    path = game_path(stage, spec["index"])
    if path.exists():
        raise FileExistsError(f"158 does not automatically reuse or retry games: {path}")
    if spec["model_id"] not in MODELS or spec["candidate_seat"] not in (0, 1):
        raise ValueError("Invalid fixed treatment/seat assignment")
    if stage == "test" and spec["environment_seed"] < 1_589_800_000:
        raise ValueError("Test games require a separate test-only environment seed")
    wall, cpu = time.perf_counter(), time.process_time()
    env = old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    record = {"manifest": binding, "spec": spec, "time": old.now(), "trace": [],
              "initial_state_sha256": old.digest(old.serialize_state_snapshot(state))}
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
            model_id = spec["model_id"] if state.current_player == spec["candidate_seat"] else None
            rp = root_path(stage, spec["index"], state.turn_index)
            try:
                action, _ = choose_root(state, spec["search_stream"], rp, binding, model_id)
            except Exception:
                if rp.exists():
                    row["root_sha256"] = old.file_hash(rp)
                record["failed_decision"] = row
                raise
            row["root_sha256"] = old.file_hash(rp)
            if action is None or action not in legal:
                reason = "illegal_or_missing_action"
                record["failed_decision"] = {**row, "action": None if action is None else old.serialize_action(action)}
                break
            row["action"] = old.serialize_action(action)
            record["trace"].append(row)
            hasher.update(old.digest(row["action"]).encode())
            successor = env.step(state, action)
            forced = forced + 1 if old.is_forced_pass_turn(legal) else 0
            idle = 0 if old.is_progress_transition(state, successor, action) else idle + 1
            state = successor
            progress_write(OUTPUT / stage / "telemetry" / f"game_{spec['index']:04d}.json",
                           {"time": old.now(), "index": spec["index"],
                            "turns": len(record["trace"]),
                            "elapsed_seconds": time.perf_counter() - wall})
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
