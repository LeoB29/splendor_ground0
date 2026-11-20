"""Second-generation learned proposals, with sealed dynamic model identities.

Both sides use learned fourth proposals. The fixed greedy035 controller remains
the continuation policy; new models never mutate an ancestor module's registry.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from tools.proposal_iteration_128 import common as c
from scripts.analysis.learned_proposal_arena import fourth_slot
from splendor_ai.alphazero import world_budget_160 as world128
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position

old, baseline, windows = c.old, c.baseline, c.windows
ROOT, OUTPUT, CONFIG, require = c.ROOT, c.OUTPUT, c.CONFIG, c.require
PRODUCTION_OUTPUT = OUTPUT.resolve()
FITS = (161001, 161002, 161003)
_MODELS, _MODEL_ARTIFACTS = {}, {}


def initialize_models(base=OUTPUT, registry=None):
    """Load only the post-fit audited registry; optional explicit isolated fixture."""
    if registry is not None:
        resolved = Path(base).resolve()
        require(resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents,
                "Fixture registry cannot target production")
    registry = old.read_json(Path(base) / "models_lock.json") if registry is None else registry
    expected = {f"{arm}_{fit}" for arm in ("anchor", "treatment", "replay") for fit in FITS}
    require(set(registry["models"]) == expected, "Incomplete model registry")
    loaded, artifacts = {}, {}
    for key, item in registry["models"].items():
        arm, fit = key.split("_")
        require(item["source_fit"] == int(fit), "Registry source fit mismatch")
        path = (ROOT / item["path"]).resolve()
        require(path.is_relative_to(ROOT) and old.file_hash(path) == item["sha256"], "Changed proposal model")
        loaded[key] = StrongPolicyInference(path).model
        artifacts[key] = dict(item)
    old.initialize_worker()
    _MODELS.clear()
    _MODELS.update(loaded)
    _MODEL_ARTIFACTS.clear()
    _MODEL_ARTIFACTS.update(artifacts)


def model_id(stage, fit, candidate):
    require(stage in CONFIG["games"] and fit in FITS and type(candidate) is bool, "Unknown proposal role")
    arm = "treatment" if candidate else "replay" if stage in ("probe_b", "stage_b") else "anchor"
    return f"{arm}_{fit}"


@lru_cache(maxsize=5)
def specs(stage):
    require(stage in CONFIG["games"], "Unknown stage")
    count = CONFIG["games"][stage] // 2
    order = sorted(range(count), key=lambda pair: old.digest([CONFIG["assignment_tag"], stage, pair]))
    fits = {pair: FITS[rank % 3] for rank, pair in enumerate(order)}
    return [{"index": i, "pair": i // 2, "candidate_seat": i % 2, "fit": fits[i // 2],
             "environment_seed": CONFIG["environment_seed"][stage] + i // 2,
             "search_stream": old.digest([CONFIG["stream_tag"], stage, i // 2])}
            for i in range(CONFIG["games"][stage])]


def check_stage(stage, base):
    require(stage in CONFIG["games"], "Unknown stage")
    resolved = Path(base).resolve()
    if stage == "test":
        require(resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents,
                "Synthetic fixture cannot target production evidence")


def game_path(stage, index, base=OUTPUT):
    check_stage(stage, base)
    require(type(index) is int and 0 <= index < CONFIG["games"][stage], "Game outside fixed schedule")
    return Path(base) / stage / "games" / f"game_{index:04d}.json"


def root_path(stage, index, turn, base=OUTPUT):
    game_path(stage, index, base)
    require(type(turn) is int and 0 <= turn < 400, "Invalid root turn")
    return Path(base) / stage / "roots" / f"game_{index:04d}" / f"turn_{turn:03d}.json"


def learned_proposal(state, stream, source_model):
    require(source_model in _MODELS and source_model in _MODEL_ARTIFACTS, "Unsealed proposal model")
    started = time.perf_counter()
    legal = old.SplendorEnv().legal_actions(state)
    schedule = world128.root_schedule(state, stream, 128)
    reference = baseline.candidate_set(state, schedule["candidate_nonce"])
    groups = old.actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    representatives = {repr(key): old.serialize_action(action) for key, action in groups.items()}
    representatives.update(zip(reference["keys"], reference["actions"], strict=True))
    model = _MODELS[source_model]
    inference_start = time.perf_counter()
    with torch.no_grad():
        row, encoded_keys, _ = encode_position(state, model.config, legal)
        logits, _ = model(row["observation"][None], row["features"][None],
                          torch.ones((1, len(legal)), dtype=torch.bool))
    require([repr(key) for key in encoded_keys] == keys and torch.isfinite(logits).all().item(), "Invalid model input/output")
    values = [float(torch.amax(logits[0][row["groups"] == j])) for j in range(len(keys))]
    extra = time.perf_counter() - inference_start
    offered = fourth_slot(keys, reference["keys"], values)
    return {"schedule": schedule, "reference": reference,
            "candidates": {"keys": offered, "actions": [representatives[key] for key in offered],
                           "canonical_support": len(keys), "legal_support": len(legal)},
            "model_id": source_model, "model_sha256": _MODEL_ARTIFACTS[source_model]["sha256"],
            "fit": _MODEL_ARTIFACTS[source_model]["source_fit"], "group_keys": keys,
            "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": logits[0].tolist(), "learned_group_scores": values,
            "timing": {"wall": time.perf_counter() - started, "learned_extra_wall": extra}}


def choose_root(state, stream, path, binding, source_model):
    path = Path(path)
    require(not path.exists(), "One-shot root; no overwrite or retry")
    wall, cpu = time.perf_counter(), time.process_time()
    record = {"manifest": binding, "time": old.now(), "snapshot": old.serialize_state_snapshot(state),
              "privacy": baseline.privacy(state), "worlds": []}
    try:
        proposal = learned_proposal(state, stream, source_model)
        record.update(schedule=proposal["schedule"], candidates=proposal["candidates"], proposal=proposal)
        legal = old.SplendorEnv().legal_actions(state)
        record["scores"] = list(map(float, old._RAW.score_actions(state, legal)))
        actions = [old.decode_action(a) for a in proposal["candidates"]["actions"]]
        for seed in proposal["schedule"]["world_seeds"]:
            world = old.determinize_for_observer(state, state.current_player, random.Random(seed))
            row = {"seed": seed, "state_sha256": old.digest(old.serialize_state_snapshot(world)), "outcomes": []}
            record["worlds"].append(row)
            for action in actions:
                require(action in old.SplendorEnv().legal_actions(world), "Illegal forced proposal")
                value = old.trajectory(world, action)
                value["wdl"] = old.outcome_value(value["winner"], state.current_player) if value["reason"] in old.VALID_NATURAL_TERMINATIONS else None
                row["outcomes"].append(value)
                require(value["wdl"] is not None, "Invalid continuation ending")
        record.update(world128.select(record["worlds"], len(actions), 128))
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["timing"] = {"wall": time.perf_counter() - wall, "cpu": time.process_time() - cpu}
        old.save_new(path, record)
    return actions[record["choice"]], record


def decision(share, interval, stage):
    require(stage in ("stage_a", "stage_b"), "Statistics stage")
    superiority = interval[0] > .5
    practical = share >= .55 and superiority
    return {"stage": stage, "stage_pass": practical if stage == "stage_a" else superiority,
            "superiority_established": superiority, "practical_progress": practical,
            "material_target_excluded": interval[1] < .55,
            "classification": "positive_practical_progress" if practical else
            "positive_below_observed_target" if superiority else
            "below_material_target" if interval[1] < .55 else "inconclusive"}


def statistics(rows, stage):
    require(stage in ("stage_a", "stage_b"), "Statistics stage")
    require(len(rows) == 402 and [r["index"] for r in rows] == list(range(402)), "Incomplete or unordered stage")
    strata = {fit: [] for fit in FITS}
    for pair in range(201):
        a, b = rows[2*pair:2*pair+2]
        spec = specs(stage)[2*pair]
        require(a["pair"] == b["pair"] == pair and a["seat"] == 0 and b["seat"] == 1
                and a["fit"] == b["fit"] == spec["fit"], "Pair/fit alignment")
        require(a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.), "Invalid score")
        strata[spec["fit"]].append(int(2*a["score"] + 2*b["score"]))
    require(all(len(v) == 67 for v in strata.values()), "Unbalanced fits")
    rng = np.random.default_rng(CONFIG["bootstrap_seed"][stage])
    boot = []
    for _ in range(50):
        total = np.zeros(1000, dtype=np.int64)
        for values in strata.values():
            indices = rng.integers(0, 67, size=(1000, 67))
            total += np.asarray(values, dtype=np.int64)[indices].sum(axis=1)
        boot.extend((total / 804).tolist())
    share = sum(sum(values) for values in strata.values()) / 804
    interval = np.quantile(boot, [.025, .975], method="linear").tolist()
    return {"games": 402, "pairs": 201, "share": share, "ci95": interval,
            "wins": sum(r["score"] == 1 for r in rows), "draws": sum(r["score"] == .5 for r in rows),
            "losses": sum(r["score"] == 0 for r in rows), **decision(share, interval, stage)}


def play_game(stage, spec, binding, base=OUTPUT, test_initial_snapshot=None):
    """Play the fixed proposal intervention once, retaining every real decision."""
    require(spec == specs(stage)[spec["index"]], "Fixed schedule mismatch")
    path = game_path(stage, spec["index"], base)
    if path.exists():
        raise FileExistsError(f"Proposal arena does not automatically reuse or retry games: {path}")
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
            source_model = model_id(stage, spec["fit"], state.current_player == spec["candidate_seat"])
            rp = root_path(stage, spec["index"], state.turn_index, base)
            try:
                action, _ = choose_root(state, spec["search_stream"], rp, binding, source_model)
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

