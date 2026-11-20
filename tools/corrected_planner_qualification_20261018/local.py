"""Explicit corrected-rule orchestration; no historical globals are rebound.

Only this module's privately loaded models are mutable. Every legality and
transition operation uses CorrectedEnv. The reused encoders receive corrected
Card objects and the complete explicit legal list; no historical catalogue or
afterstate cache is consulted on this execution path.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import torch

from tools.external_strength_screen_20261018 import rules
from tools.local_player_qualification_128 import local as historical
from tools.local_player_qualification_128.selection import (
    BINDING, FITS, ROOT, SOURCE, SOURCE_HASHES, file_hash, require,
)
from tools.proposal_breadth_arena_128 import arena
from splendor_ai.alphazero.strong_policy import StrongPolicyInference, encode_position
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import actions_by_canonical_successor_key, canonical_successor_action_key
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS, adjudicate_scores, is_forced_pass_turn, outcome_value
from splendor_ai.training.replay import serialize_state_snapshot, serialize_action

old = arena.old
SCHEMA = "corrected_planner_observer_v1"
PROFILE = "four_candidate128_corrected_rules_20261018_v1"
_MODELS, _ARTIFACTS = {}, {}
_RAW = None
_INITIALIZED = False


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def make_observer_payload(snapshot, context):
    """Convert a labelled corrected snapshot in the trusted parent only."""
    state = rules.deserialize_snapshot(snapshot)
    view = rules.observation_payload(state)
    result = {**view, "schema": SCHEMA, "context": historical._context(context)}
    state_from_observer_payload(result)
    return _copy(result)


def state_from_observer_payload(payload):
    require(isinstance(payload, dict) and set(payload) ==
            {"schema", "context", "ruleset_id", "catalogue_sha256", "observation"}
            and payload["schema"] == SCHEMA, "Invalid corrected planner envelope")
    context = historical._context(payload["context"])
    source = {k: deepcopy(v) for k, v in payload.items() if k != "context"}
    source["schema"] = rules.OBSERVATION_SCHEMA
    state = rules.state_from_observation(source)
    require(not state.terminal and state.winner is None and context["identity"][2] == state.turn_index,
            "Expected a live matching corrected decision")
    assert_corrected_cards(state)
    return state


def assert_corrected_cards(state):
    """Catch accidental historical reconstruction, including hidden world cards."""
    expected = rules.corrected_cards_by_id()
    actual = [c for values in state.visible_tier_cards.values() for c in values]
    actual += [c for values in state.hidden_tier_decks.values() for c in values]
    actual += [c for p in state.players for c in p.reserved_cards + p.purchased_cards]
    require(len(actual) == 90 and {c.card_id for c in actual} == set(expected), "Incomplete corrected catalogue")
    require(all(c == expected[c.card_id] for c in actual), "Historical card face reached corrected planner")


def initialize(registry=None, repo_root=ROOT):
    """Load all three unchanged proposal fits and035 privately, once per worker."""
    global _INITIALIZED, _RAW
    require(not _INITIALIZED, "Corrected worker already initialized")
    repo_root = Path(repo_root)
    path = repo_root / SOURCE / "models_lock.json"
    require(file_hash(path) == SOURCE_HASHES["models_lock.json"], "Changed model registry")
    sealed = json.loads(path.read_text(encoding="utf-8"))
    require(registry is None or registry == sealed, "Model registry mismatch")
    require(sealed["manifest"] == BINDING and set(sealed["models"]) == {f"anchor_{fit}" for fit in FITS},
            "Incomplete frozen proposal fits")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    loaded, artifacts = {}, {}
    for model_id, item in sealed["models"].items():
        require(item["source_fit"] == int(model_id.split("_")[1]) and item["unchanged_from_anchor"] is True,
                "Unexpected proposal model")
        checkpoint = repo_root / item["path"]
        require(file_hash(checkpoint) == item["sha256"], "Changed proposal checkpoint")
        loaded[model_id] = StrongPolicyInference(checkpoint).model
        artifacts[model_id] = deepcopy(item)
    continuation = sealed["continuation_model"]
    checkpoint = repo_root / continuation["path"]
    require(file_hash(checkpoint) == continuation["sha256"], "Changed continuation checkpoint")
    _RAW = LegalActionAttentionScorerBot(checkpoint)
    _MODELS.update(loaded)
    _ARTIFACTS.update(artifacts)
    _INITIALIZED = True
    return {"models": deepcopy(artifacts), "continuation_model": deepcopy(continuation),
            "backend": "unchanged_serial", "worlds": 128, "candidates": 4,
            "torch_threads": torch.get_num_threads(), "profile": PROFILE,
            "ruleset_id": rules.RULESET_ID, "catalogue_sha256": rules.CATALOGUE_SHA256,
            "binding_strategy": "explicit_corrected_functions_no_global_rebinding",
            "deadlock_compatibility": "sole PASS counts only when it claims no noble"}


def registered_model(model_id):
    require(_INITIALIZED and model_id in _MODELS and model_id in _ARTIFACTS, "Uninitialized proposal model")
    return _MODELS[model_id], deepcopy(_ARTIFACTS[model_id])


def score_actions(state, legal):
    require(_INITIALIZED and _RAW is not None, "Uninitialized continuation model")
    scores = list(map(float, _RAW.score_actions(state, legal)))
    require(len(scores) == len(legal) and all(math.isfinite(v) for v in scores), "Invalid complete035 scores")
    return scores


def candidates(state, nonce, legal=None, scores=None):
    """Unchanged two ranked plus two seeded proposals, under corrected legality."""
    legal = rules.CorrectedEnv().legal_actions(state) if legal is None else legal
    scores = score_actions(state, legal) if scores is None else scores
    require(bool(legal) and len(scores) == len(legal) and all(math.isfinite(v) for v in scores), "Invalid proposal scores")
    ranked = sorted(range(len(legal)), key=lambda i: (-scores[i], i))
    groups = actions_by_canonical_successor_key(state, legal)
    chosen, keys = [], []
    for i in ranked:
        key = canonical_successor_action_key(state, legal[i])
        if key not in keys:
            chosen.append(legal[i]); keys.append(key)
        if len(chosen) == min(2, len(groups)):
            break
    rest = [key for key in groups if key not in keys]
    rng = random.Random(f"149-candidates-v1|{arena.baseline.CONFIG['candidate_split']}|{nonce}")
    for key in rng.sample(rest, min(2, len(rest))):
        chosen.append(groups[key]); keys.append(key)
    return {"actions": [serialize_action(a) for a in chosen], "keys": [repr(k) for k in keys],
            "canonical_support": len(groups), "legal_support": len(legal)}


def proposal_for_root(state, context, legal, scores):
    model_id = f"anchor_{context['fit']}"
    model, artifact = registered_model(model_id)
    schedule = arena.world128.root_schedule(state, context["stream"], 128)
    reference = candidates(state, schedule["candidate_nonce"], legal, scores)
    groups = actions_by_canonical_successor_key(state, legal)
    keys = [repr(key) for key in groups]
    representatives = {repr(key): serialize_action(action) for key, action in groups.items()}
    representatives.update(zip(reference["keys"], reference["actions"], strict=True))
    with torch.no_grad():
        row, encoded_keys, _ = encode_position(state, model.config, legal)
        logits, _ = model(row["observation"][None], row["features"][None], torch.ones((1, len(legal)), dtype=torch.bool))
    require([repr(k) for k in encoded_keys] == keys and torch.isfinite(logits).all().item(), "Invalid learned proposal")
    values = [float(torch.amax(logits[0][row["groups"] == j])) for j in range(len(keys))]
    offered = arena.fourth_slot(keys, reference["keys"], values)
    candidate = {"keys": offered, "actions": [representatives[k] for k in offered],
                 "canonical_support": len(keys), "legal_support": len(legal)}
    return {"schedule": schedule, "reference": reference, "candidates": candidate,
            "model_id": model_id, "model_sha256": artifact["sha256"], "fit": artifact["source_fit"],
            "group_keys": keys, "legal_actions": [serialize_action(a) for a in legal],
            "learned_logits": logits[0].tolist(), "learned_group_scores": values,
            "anchor_candidates": deepcopy(candidate), "root_identity": list(context["identity"]),
            "breadth": False, "added_indices": {"ranked": [], "random": []}}


def trajectory(state, root_action=None, *, boundary=lambda: None, capture=False):
    """Exact serial greedy035 work, with corrected transitions and auditable scores."""
    env = rules.CorrectedEnv()
    seen, forced, no_progress, turns = {}, 0, 0, 0
    hasher, trace = hashlib.sha256(), []
    reason, winner = "completed", None
    while not state.terminal:
        boundary()
        signature = state_signature(state)
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] >= old.CONFIG["repetition_limit"]:
            reason = "repetition_cutoff"; break
        if state.turn_index >= old.CONFIG["max_turns"]:
            reason = "max_turns"; break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"; break
        first = turns == 0 and root_action is not None
        scores = None if first else score_actions(state, legal)
        selected = legal.index(root_action) if first else max(range(len(legal)), key=lambda i: scores[i])
        action = legal[selected]
        payload = serialize_action(action)
        hasher.update(old.digest(payload).encode())
        if capture:
            trace.append({"state_hash": old.digest(serialize_state_snapshot(state)), "action": payload,
                          "turn": state.turn_index, "actor": state.current_player,
                          "legal_count": len(legal), "selected_index": selected, "forced": first,
                          "scores_sha256": None if first else old.digest(scores)})
        successor = env.step(state, action)
        turns += 1
        forced = forced + 1 if is_forced_pass_turn(legal) and action.noble_id is None else 0
        no_progress = 0 if is_progress_transition(state, successor, action) else no_progress + 1
        state = successor
        if forced >= 2:
            reason, winner = "forced_pass_deadlock", adjudicate_scores(state); break
        if no_progress >= old.CONFIG["no_progress_limit"]:
            reason = "no_progress_cutoff"; break
    if reason == "completed":
        winner = state.winner
    result = {"reason": reason, "winner": winner, "turns": turns, "trace_sha256": hasher.hexdigest(),
              "final_state_sha256": old.digest(serialize_state_snapshot(state))}
    return result, trace


def choose(payload, breadth=False, boundary=lambda: None, trace_sink=None):
    """One full4x128 corrected decision; no incomplete-world choice or fallback."""
    require(_INITIALIZED, "Corrected worker not initialized")
    require(breadth is False, "Only the authorized four-candidate policy is available")
    require(trace_sink is None or callable(trace_sink), "Invalid continuation trace sink")
    boundary()
    state = state_from_observer_payload(payload)
    context = historical._context(payload["context"])
    legal = rules.CorrectedEnv().legal_actions(state)
    proposal_scores = score_actions(state, legal)
    proposal = proposal_for_root(state, context, legal, proposal_scores)
    # Historical root records make a second035 scoring call after proposal
    # inference. Retain that numerical work instead of reusing the first call.
    scores = score_actions(state, legal)
    require(scores == proposal_scores, "Repeated deterministic root scores changed")
    record = {"root_identity": list(context["identity"]), "planner": "reference",
              "privacy": arena.baseline.privacy(state), "schedule": proposal["schedule"],
              "candidates": proposal["candidates"], "proposal": proposal, "scores": scores, "worlds": [],
              "profile": PROFILE, "ruleset_id": rules.RULESET_ID, "catalogue_sha256": rules.CATALOGUE_SHA256}
    actions = [old.decode_action(a) for a in proposal["candidates"]["actions"]]
    require(1 <= len(actions) <= 4, "Unexpected four-candidate support")
    for world_index, seed in enumerate(proposal["schedule"]["world_seeds"]):
        boundary()
        world = determinize_for_observer(state, state.current_player, random.Random(seed))
        assert_corrected_cards(world)
        row = {"seed": seed, "state_sha256": old.digest(serialize_state_snapshot(world)), "outcomes": []}
        legal_world = rules.CorrectedEnv().legal_actions(world)
        for candidate_index, action in enumerate(actions):
            require(action in legal_world, "Illegal corrected forced proposal")
            value, trace = trajectory(world, action, boundary=boundary, capture=trace_sink is not None)
            if trace_sink is not None:
                trace_sink(_copy({"world_index": world_index, "candidate_index": candidate_index,
                    "seed": seed, "initial_state_sha256": row["state_sha256"], "steps": trace, **value}))
            require(value["reason"] in VALID_NATURAL_TERMINATIONS, "Invalid corrected continuation ending")
            value["wdl"] = outcome_value(value["winner"], state.current_player)
            row["outcomes"].append(value)
        record["worlds"].append(row)
    require(len(record["worlds"]) == 128, "Incomplete corrected world schedule")
    record.update(arena.select(record["worlds"], len(actions)))
    return _copy({"action": serialize_action(actions[record["choice"]]), "record": record})
