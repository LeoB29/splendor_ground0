"""Independent recovery audit with conservative unresolved-result certification.

Sampling uses identities and lawful metadata only. Every declared world and
outcome field is checked; only selected trajectories are independently replayed.
The same frozen engine/model primitives are shared with the previous auditors.
"""
from collections import Counter
import copy
from fractions import Fraction
import hashlib
import math
import random
import re
from pathlib import Path
import time

import numpy as np
import torch

from tools.equal_budget_breadth_arena_128_recovery_02 import arena as p
from tools.equal_budget_breadth_arena_128_recovery_03 import common as c
from scripts.analysis import verify_learned_proposal_screen as a_screen
from scripts.analysis import verify_rollout_selector_154 as a154
from scripts.analysis import verify_afterstate_advantage_149 as a149
from scripts.analysis import verify_fast_selector_transfer_157 as a157
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

old, OUTPUT, PRODUCTION_OUTPUT = p.old, p.OUTPUT, c.OUTPUT.resolve()
compare = a_screen.compare
FITS = (161001, 161002, 161003)
AUDIT_SEED = 2174000000
WORLD_TAG = "equal-budget-breadth-arena-128-audit-world-v1"
STRESS_TAG = "equal-budget-breadth-arena-128-audit-stress-v1"
SAMPLE_CONFIG = {"seed": AUDIT_SEED, "world_tag": WORLD_TAG, "stress_tag": STRESS_TAG,
                 "maximum_worlds_per_root": 128, "ordinary_world_divisor": 16,
                 "ordinary_world_minimum": 2,
                 "stress_fit_is_assigned_fit_for_both_arms": True,
                 "maximum_stress_roots": 171,
                 "candidate_canonical_support_stress": [5, 6, 7],
                 "all_unresolved_repetition_cells_replayed": True,
                 "full_inner_replay": False}


def independent_specs(stage):
    assert stage in ("probe", "main", "test")
    count = {"probe": 6, "main": 201, "test": 3}[stage]
    ranked = sorted(range(count), key=lambda pair: old.digest(["equal-budget-breadth-arena-128-fit-v1", stage, pair]))
    result = []
    for pair in range(count):
        fit = (161001, 161002, 161003)[ranked.index(pair) % 3]
        for seat in (0, 1):
            result.append({"index": 2*pair+seat, "pair": pair, "candidate_seat": seat, "fit": fit,
                           "environment_seed": {"probe": 2179000000, "main": 2170000000, "test": 2179800000}[stage]+pair,
                           "search_stream": old.digest(["equal-budget-breadth-arena-128-stream-v1", stage, pair])})
    return result


def independent_spec(stage, index):
    rows = independent_specs(stage)
    assert type(index) is int and 0 <= index < len(rows)
    return rows[index]


def independent_schedule(state, stream):
    belief = old.digest(canonical_public_belief_key(state, state.current_player))
    return {"belief_sha256": belief, "candidate_nonce": old.digest(["155-candidate-v1", stream, belief]),
            "world_seeds": [f"155-world-v1|{stream}|{belief}|{j:02d}" for j in range(128)]}


def independent_model_id(stage, fit, candidate):
    assert stage in ("probe", "main", "test")
    assert fit in FITS and type(candidate) is bool
    return f"anchor_{fit}"


def independent_allocation(candidates, identity, breadth):
    """Rebuild exact per-candidate budget without collection allocation helpers."""
    assert type(breadth) is bool
    assert isinstance(identity, (list, tuple)) and len(identity) == 3
    stage, index, turn = identity
    independent_spec(stage, index)
    assert type(turn) is int and 0 <= turn < 400
    support = candidates["canonical_support"]
    assert type(support) is int and support >= 1
    keys = candidates["keys"]
    m, n = min(support, 4), min(support, 8 if breadth else 4)
    assert len(keys) == len(set(keys)) == n and all(isinstance(key, str) for key in keys)
    q, remainder = divmod(128*m, n)
    order = sorted(range(n), key=lambda j: (
        old.digest(["equal-budget-breadth-arena-128-remainder-v1", stage, index, turn, keys[j]]), j))
    extra = order[:remainder]
    counts = [q+int(j in extra) for j in range(n)]
    assert sum(counts) == 128*m and 1 <= min(counts) <= max(counts) <= 128
    return {"world_counts": counts, "remainder_order": order, "remainder_indices": extra,
            "world_count": max(counts), "continuations": sum(counts)}


def independent_selection(worlds, world_counts):
    """Reconstruct bounds without calling collection selection/certificate code."""
    assert isinstance(world_counts, list) and 1 <= len(world_counts) <= 8
    assert all(type(count) is int and 1 <= count <= 128 for count in world_counts)
    assert len(worlds) == max(world_counts)
    totals, delivered, unknown = ([0]*len(world_counts) for _ in range(3))
    unresolved = []
    for wi, world in enumerate(worlds):
        active = [j for j, count in enumerate(world_counts) if wi < count]
        assert world["candidate_indices"] == active and all(type(j) is int for j in world["candidate_indices"])
        assert len(world["outcomes"]) == len(active)
        for j, outcome in zip(active, world["outcomes"], strict=True):
            value = outcome["wdl"]
            if outcome["reason"] == "repetition_cutoff":
                assert value is None and outcome["winner"] is None
                unknown[j] += 1
                unresolved.append({"world": wi, "candidate": j})
            else:
                assert outcome["reason"] in ("completed", "forced_pass_deadlock")
                assert type(value) in (int, float) and value in (-1, 0, 1)
                totals[j] += int(value)
            delivered[j] += 1
    assert delivered == world_counts
    lows = [Fraction(s-u, n) for s, u, n in zip(totals, unknown, delivered, strict=True)]
    highs = [Fraction(s+u, n) for s, u, n in zip(totals, unknown, delivered, strict=True)]
    winners = [j for j, lower in enumerate(lows)
               if all(lower > upper if k < j else lower >= upper
                      for k, upper in enumerate(highs) if k != j)]
    assert len(winners) <= 1
    choice = winners[0] if winners else None
    result = {"choice": choice,
              "means": [None if u else float(Fraction(s, n))
                        for s, u, n in zip(totals, unknown, delivered, strict=True)],
              "sums_wdl": totals, "counts": delivered}
    if unresolved:
        result["certificate"] = {
            "rule": "repetition_interval_original_order_v1",
            "known_counts": [n-u for n, u in zip(delivered, unknown, strict=True)],
            "unresolved_counts": unknown,
            "lower_sums": [s-u for s, u in zip(totals, unknown, strict=True)],
            "upper_sums": [s+u for s, u in zip(totals, unknown, strict=True)],
            "unresolved_cells": unresolved, "choice": choice}
    return result


def independent_replay_cells(worlds, sampled_worlds):
    """Keep outcome-blind sampling fixed, then add every unresolved cell once."""
    assert sampled_worlds == sorted(set(sampled_worlds))
    assert all(type(wi) is int and 0 <= wi < len(worlds) for wi in sampled_worlds)
    sampled, unresolved = set(), set()
    for wi, world in enumerate(worlds):
        for candidate, outcome in zip(world["candidate_indices"], world["outcomes"], strict=True):
            cell = (wi, candidate)
            if wi in sampled_worlds:
                sampled.add(cell)
            if outcome["reason"] == "repetition_cutoff":
                unresolved.add(cell)
    return {"sampled": sampled, "unresolved": unresolved,
            "extra": unresolved-sampled, "all": sampled | unresolved}


def independent_coupled_selection(worlds, counts, proofs):
    """Bound each pairwise difference with one variable per proven context."""
    interval = independent_selection(worlds, counts)
    assert interval["choice"] is None, "Coupling is only an extension of an ambiguous R1 decision"
    unknown = interval["certificate"]["unresolved_cells"]
    assert [(proof["world"], proof["candidate"]) for proof in proofs] == [
        (cell["world"], cell["candidate"]) for cell in unknown]
    groups, contexts = {}, {}
    for proof in proofs:
        key = old.digest(proof["context"])
        assert proof["context_sha256"] == key
        assert key not in contexts or contexts[key] == proof["context"]
        contexts[key] = proof["context"]
        groups.setdefault(key, []).append({"world": proof["world"], "candidate": proof["candidate"]})
    grouped = [{"context_sha256": key, "cells": groups[key]} for key in sorted(groups)]
    coefficients = [Counter(cell["candidate"] for cell in row["cells"]) for row in grouped]
    sums = interval["sums_wdl"]
    pairwise, winners = [], []
    for candidate in range(len(counts)):
        lower = []
        for competitor in range(len(counts)):
            # Independent rational coefficients; restore exact integer numerator.
            known = Fraction(sums[candidate], counts[candidate])-Fraction(sums[competitor], counts[competitor])
            radius = sum(abs(Fraction(group[candidate], counts[candidate])-
                             Fraction(group[competitor], counts[competitor])) for group in coefficients)
            numerator = (known-radius)*counts[candidate]*counts[competitor]
            assert numerator.denominator == 1
            lower.append(numerator.numerator)
        pairwise.append(lower)
        if all(value > 0 if other < candidate else value >= 0
               for other, value in enumerate(lower) if other != candidate):
            winners.append(candidate)
    assert len(winners) <= 1
    choice = winners[0] if winners else None
    certificate = {**interval["certificate"], "rule": "coupled_repetition_original_order_v1",
                   "choice": choice, "groups": grouped,
                   "pairwise_lower_numerators": pairwise[choice] if choice is not None else None}
    return {**interval, "choice": choice, "certificate": certificate}


def certified_selection(root, counts, independently_replayed_proofs=None):
    arithmetic = independent_selection(root["worlds"], counts)
    if arithmetic["choice"] is None:
        if independently_replayed_proofs is None:
            independently_replayed_proofs = [independent_saved_proof(root, proof)
                                            for proof in root["equivalence_proofs"]]
        assert old.digest(root["equivalence_proofs"]) == old.digest(independently_replayed_proofs)
        arithmetic = independent_coupled_selection(root["worlds"], counts, independently_replayed_proofs)
    else:
        assert "equivalence_proofs" not in root, "Unnecessary coupling can change the inherited schema"
    choice = arithmetic["choice"]
    assert choice is not None, "Unresolved continuation can change the selected move"
    assert root["means"] == arithmetic["means"]
    assert root["sums_wdl"] == arithmetic["sums_wdl"] and all(type(value) is int for value in root["sums_wdl"])
    assert root["counts"] == arithmetic["counts"] and all(type(value) is int for value in root["counts"])
    assert type(root["choice"]) is int and root["choice"] == choice
    assert ("certificate" in root) == ("certificate" in arithmetic)
    if "certificate" in arithmetic:
        # Digest comparison also distinguishes bool/float impostors from integer bounds.
        assert old.digest(root["certificate"]) == old.digest(arithmetic["certificate"])
    return arithmetic


def _canonical_purchased_order(snapshot):
    canonical = copy.deepcopy(snapshot)
    for player in canonical["players"]:
        player["purchased_card_ids"] = sorted(player["purchased_card_ids"])
    return canonical


def _context_from_history(history, forced, idle, actor):
    final = _canonical_purchased_order(history[-1])
    final_purchases = [player["purchased_card_ids"] for player in final["players"]]
    counts = Counter()
    for snapshot in history:
        canonical = _canonical_purchased_order(snapshot)
        if [player["purchased_card_ids"] for player in canonical["players"]] != final_purchases:
            continue
        state = old.deserialize_state_snapshot(canonical)
        counts[old.digest(old.state_signature(state))] += 1
    assert old.CHECKPOINT_SHA == "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
    return {"canonical_final_state_sha256": old.digest(final),
            "live_repetition_counts": [{"signature_sha256": key, "count": counts[key]} for key in sorted(counts)],
            "forced_pass_count": forced, "no_progress_count": idle,
            "root_actor": actor, "policy_sha256": old.CHECKPOINT_SHA}


def independent_equivalence_proof(world, forced_action, actor, wi, candidate):
    """Replay once independently, retaining the sufficient state of all guards."""
    env, state = old.SplendorEnv(), world
    visits, history, trace, hashed = Counter(), [], [], hashlib.sha256()
    forced, idle, moves, reason, winner = 0, 0, 0, "completed", None
    while not state.terminal:
        history.append(old.serialize_state_snapshot(state))
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
        if moves == 0:
            action = forced_action
            assert action in legal
        else:
            scores = old._RAW.score_actions(state, legal)
            assert all(math.isfinite(value) for value in scores)
            action = legal[max(range(len(legal)), key=lambda j: scores[j])]
        payload = old.serialize_action(action)
        hashed.update(old.digest(payload).encode())
        trace.append({"state_hash": old.digest(history[-1]), "action": payload,
                      "turn": state.turn_index, "actor": state.current_player})
        successor = env.step(state, action)
        forced = forced+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(state, successor, action) else idle+1
        moves += 1
        state = successor
        if forced == 2:
            reason, winner = "forced_pass_deadlock", adjudicate(state)
            break
        if idle >= 60:
            reason = "no_progress_cutoff"
            break
    assert reason == "repetition_cutoff", "Claimed coalescence cell does not reproduce a repetition cutoff"
    final = old.serialize_state_snapshot(state)
    assert history[-1] == final  # Its fourth visit belongs to the guard state.
    replay = {"reason": reason, "winner": winner, "turns": moves,
              "trace_sha256": hashed.hexdigest(), "final_state_sha256": old.digest(final),
              "trace": trace, "final_state": final}
    context = _context_from_history(history, forced, idle, actor)
    return {"world": wi, "candidate": candidate, "replay": replay,
            "context": context, "context_sha256": old.digest(context)}


def independent_saved_proof(root, proof):
    """Authenticate committed traces and derive context without neural inference."""
    wi, candidate = proof["world"], proof["candidate"]
    assert type(wi) is int and 0 <= wi < len(root["worlds"])
    assert type(candidate) is int and candidate in root["worlds"][wi]["candidate_indices"]
    source = old.deserialize_state_snapshot(root["snapshot"])
    actor = source.current_player
    saved = root["worlds"][wi]
    outcome = saved["outcomes"][saved["candidate_indices"].index(candidate)]
    _check_outcome(outcome, actor, source.turn_index)
    assert outcome["reason"] == "repetition_cutoff"
    schedule = root["schedule"]["world_seeds"]
    assert saved["seed"] == schedule[wi]
    state = a154.independent_world(source, saved["seed"])
    assert old.digest(old.serialize_state_snapshot(state)) == saved["state_sha256"]
    replay = proof["replay"]
    assert set(replay) == {"reason", "winner", "turns", "trace_sha256", "final_state_sha256", "trace", "final_state"}
    assert {key: replay[key] for key in outcome if key != "wdl"} == {
        key: value for key, value in outcome.items() if key != "wdl"}
    assert len(replay["trace"]) == replay["turns"] and replay["turns"] > 0
    env, visits, history, hashed = old.SplendorEnv(), Counter(), [], hashlib.sha256()
    forced, idle = 0, 0
    for i, move in enumerate(replay["trace"]):
        assert not state.terminal and state.turn_index < 400 and forced < 2 and idle < 60
        snapshot = old.serialize_state_snapshot(state)
        history.append(snapshot)
        visits[old.state_signature(state)] += 1
        assert visits[old.state_signature(state)] < 4
        assert move["state_hash"] == old.digest(snapshot)
        assert move["turn"] == state.turn_index and move["actor"] == state.current_player
        assert set(move) == {"state_hash", "action", "turn", "actor"}
        legal = env.legal_actions(state)
        action = old.decode_action(move["action"])
        assert action in legal
        if i == 0:
            assert move["action"] == root["candidates"]["actions"][candidate]
        hashed.update(old.digest(move["action"]).encode())
        successor = env.step(state, action)
        forced = forced+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(state, successor, action) else idle+1
        state = successor
    assert not state.terminal and forced < 2 and idle < 60
    history.append(old.serialize_state_snapshot(state))
    visits[old.state_signature(state)] += 1
    assert visits[old.state_signature(state)] == 4
    assert old.digest(replay["final_state"]) == old.digest(history[-1])
    assert replay["final_state_sha256"] == old.digest(history[-1])
    assert replay["trace_sha256"] == hashed.hexdigest()
    context = _context_from_history(history, forced, idle, actor)
    rebuilt = {"world": wi, "candidate": candidate, "replay": replay,
               "context": context, "context_sha256": old.digest(context)}
    assert old.digest(proof) == old.digest(rebuilt)
    return rebuilt


def independent_proposal(state, stream, source_model):
    """Rebuild grouping, raw candidates and fourth-slot ranking independently."""
    model = p._MODELS[source_model]
    identity = p._MODEL_ARTIFACTS[source_model]
    legal = old.SplendorEnv().legal_actions(state)
    raw = list(map(float, old._RAW.score_actions(state, legal)))
    group_keys, members, reps = [], {}, {}
    for index, action in enumerate(legal):
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in members:
            group_keys.append(key)
            members[key] = []
            reps[key] = old.serialize_action(action)
        members[key].append(index)
    order, top_reps = [], {}
    for index in sorted(range(len(legal)), key=lambda j: (-raw[j], j)):
        key = repr(old.canonical_successor_action_key(state, legal[index]))
        if key not in order:
            order.append(key)
            top_reps[key] = old.serialize_action(legal[index])
    schedule = independent_schedule(state, stream)
    chosen = order[:2]
    reps.update({key: top_reps[key] for key in chosen})
    remaining = [key for key in group_keys if key not in chosen]
    rng = random.Random(f"149-candidates-v1|155-live-v1|{schedule['candidate_nonce']}")
    chosen += rng.sample(remaining, min(2, len(remaining)))
    reference = {"keys": chosen, "actions": [reps[key] for key in chosen],
                 "canonical_support": len(group_keys), "legal_support": len(legal)}
    obs = encode_public_observation_tensor(state, state.current_player)
    features = encode_legal_action_features(state=state, legal_actions=legal, player_id=state.current_player)
    observation = torch.tensor(fit_feature_vector(obs.vector, model.config.observation_size), dtype=torch.float32)
    inputs = torch.tensor([fit_feature_vector(row.vector, model.config.action_feature_size) for row in features], dtype=torch.float32)
    with torch.no_grad():
        logits, _ = model(observation[None], inputs[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all().item()
    values = logits[0].tolist()
    grouped = [max(values[j] for j in members[key]) for key in group_keys]
    offered = list(chosen)
    if len(chosen) == 4:
        allowed = [j for j, key in enumerate(group_keys) if key not in chosen[:3]]
        selected = min(allowed, key=lambda j: (-grouped[j], j))
        offered = chosen[:3] + [group_keys[selected]]
    return {"schedule": schedule, "reference": reference,
            "candidates": {"keys": offered, "actions": [reps[key] for key in offered],
                           "canonical_support": len(group_keys), "legal_support": len(legal)},
            "model_id": source_model, "model_sha256": identity["sha256"],
            "fit": identity["source_fit"], "group_keys": group_keys,
            "legal_actions": [old.serialize_action(a) for a in legal],
            "learned_logits": values, "learned_group_scores": grouped}


def independent_proposal_for_root(state, stream, source_model, identity, breadth):
    """Independent canonical expansion; original representatives stay untouched."""
    assert isinstance(identity, (list, tuple)) and len(identity) == 3
    stage, index, turn = identity
    independent_spec(stage, index)
    assert type(turn) is int and turn == state.turn_index and 0 <= turn < 400
    assert type(breadth) is bool
    anchor = independent_proposal(state, stream, source_model)
    original = anchor["candidates"]
    keys, values = anchor["group_keys"], anchor["learned_group_scores"]
    representatives = {}
    # Canonical first representatives must be rebuilt from the legal actions.
    # In particular, the reference random fourth can have a different payload.
    for action in old.SplendorEnv().legal_actions(state):
        key = repr(old.canonical_successor_action_key(state, action))
        if key not in representatives:
            representatives[key] = old.serialize_action(action)
    offered, actions = list(original["keys"]), list(original["actions"])
    ranked, uniform = [], []
    if breadth:
        target = min(len(keys), 8)
        order = sorted(range(len(keys)), key=lambda i: (-values[i], i))
        ranked = [keys[i] for i in order if keys[i] not in offered][:min(2, target-len(offered))]
        remaining = [i for i, key in enumerate(keys) if key not in offered and key not in ranked]
        hashed = sorted(remaining, key=lambda i: (
            old.digest(["equal-budget-breadth-arena-128-random-v1", stage, index, turn, keys[i]]), i))
        uniform = [keys[i] for i in hashed[:target-len(offered)-len(ranked)]]
        offered += ranked+uniform
        actions += [representatives[key] for key in ranked+uniform]
    m = len(original["keys"])
    candidates = {**original, "keys": offered, "actions": actions}
    return {**anchor, "anchor_candidates": original,
            "candidates": candidates,
            "added_indices": {"ranked": list(range(m, m+len(ranked))),
                              "random": list(range(m+len(ranked), m+len(ranked)+len(uniform)))},
            "root_identity": list(identity), "breadth": breadth,
            "allocation": independent_allocation(candidates, identity, breadth)}


def adjudicate(state):
    ranks = [(player.score, -len(player.purchased_cards)) for player in state.players]
    return None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)


def _timing(values):
    assert set(values) == {"wall", "cpu"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())


def _timing_fields(record):
    inherited = record.get("inherited_timing", {"wall": 0., "cpu": 0.})
    _timing(record["timing"])
    _timing(inherited)
    return {"timing": record["timing"], "inherited_timing": inherited}


def _artifact(path, **extra):
    path = Path(path).resolve()
    try:
        name = path.relative_to(p.ROOT).as_posix()
    except ValueError:
        name = str(path)
    return {"path": name, "sha256": old.file_hash(path), **extra}


def _artifact_path(item):
    path = Path(item["path"])
    return path if path.is_absolute() else p.ROOT / path


def root_id(index, turn):
    return f"{index:04d}:{turn:03d}"


def ordinary_indices(stage, index, turn, world_count):
    assert stage in ("probe", "main", "test")
    assert type(index) is int and index >= 0 and type(turn) is int and 0 <= turn < 400
    assert type(world_count) is int and 1 <= world_count <= 128
    sample_size = min(world_count, max(2, math.ceil(world_count/16)))
    return sorted(sorted(range(world_count), key=lambda wi: (
        old.digest([WORLD_TAG, AUDIT_SEED, stage, index, turn, wi]), wi))[:sample_size])


def stress_priority(stage, row):
    return old.digest([STRESS_TAG, AUDIT_SEED, stage, row["index"], row["turn"]]), row["root_id"]


def _root_metadata(stage, spec, move, root):
    state = old.deserialize_state_snapshot(root["snapshot"])
    assert move["turn"] == state.turn_index and move["actor"] == state.current_player
    assert move["state_sha256"] == old.digest(root["snapshot"])
    assert not state.terminal and 0 <= state.turn_index < 400
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert root["privacy"] == privacy
    candidate = state.current_player == spec["candidate_seat"]
    assert root["planner"] == ("candidate" if candidate else "reference")
    assert root["root_identity"] == [stage, spec["index"], state.turn_index]
    support = len(root["candidates"]["actions"])
    canonical_support = len({repr(old.canonical_successor_action_key(state, action))
                             for action in old.SplendorEnv().legal_actions(state)})
    assert canonical_support == root["candidates"]["canonical_support"]
    assert support == min(canonical_support, 8 if candidate else 4)
    assert 1 <= support <= 8 and len(root["candidates"]["keys"]) == support
    allocation = independent_allocation(root["candidates"], root["root_identity"], candidate)
    assert root["allocation"] == root["proposal"]["allocation"] == allocation
    assert len(root["worlds"]) == allocation["world_count"] and len(root["schedule"]["world_seeds"]) == 128
    return {"root_id": root_id(spec["index"], state.turn_index), "index": spec["index"],
            "pair": spec["pair"], "fit": spec["fit"], "turn": state.turn_index,
            "arm": "candidate" if state.current_player == spec["candidate_seat"] else "reference",
            "actor": state.current_player, "phase": 0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2,
            "privacy": privacy, "privacy_category": int(bool(privacy["own"]))+2*int(bool(privacy["opponent"])),
            "support": support, "canonical_support": canonical_support,
            "worlds": allocation["world_count"], "world_counts": allocation["world_counts"],
            "continuations": allocation["continuations"], "sha256": move["root_sha256"]}


def metadata_inventory(stage, game_items, binding, base=OUTPUT):
    """Read only identity/metadata fields; scores/outcomes never enter sampling."""
    base = Path(base)
    specs = independent_specs(stage)
    assert [item["index"] for item in game_items] == list(range(len(specs)))
    expected_games, expected_roots, inventory = set(), set(), []
    for spec, item in zip(specs, game_items):
        c.budget_boundary(base)
        path = c.resolve_game(stage, spec["index"], base)
        expected_games.add(path.resolve())
        if "path" in item:
            assert _artifact_path(item).resolve() == path.resolve()
        assert old.file_hash(path) == item["sha256"]
        game = old.read_json(path)
        c.validate_origin(game, path, binding, base)
        assert game["spec"] == spec and "failure" not in game
        seen = set()
        for move in game["trace"]:
            c.budget_boundary(base)
            assert type(move["turn"]) is int and move["turn"] not in seen
            seen.add(move["turn"])
            root_path = c.resolve_root(stage, spec["index"], move["turn"], base)
            expected_roots.add(root_path.resolve())
            assert old.file_hash(root_path) == move["root_sha256"]
            root = old.read_json(root_path)
            c.validate_origin(root, root_path, binding, base)
            assert "failure" not in root
            inventory.append(_root_metadata(stage, spec, move, root))
    c.validate_resolved_inventory(stage, expected_games, expected_roots, base)
    assert len({row["root_id"] for row in inventory}) == len(inventory)
    return sorted(inventory, key=lambda row: (row["index"], row["turn"]))


def sample_plan(stage, inventory, binding, game_lock_sha256):
    """Pure deterministic construction, including absent cells and overlap counts."""
    assert inventory and len({row["root_id"] for row in inventory}) == len(inventory)
    assert inventory == sorted(inventory, key=lambda row: (row["index"], row["turn"]))
    for row in inventory:
        assert row["root_id"] == root_id(row["index"], row["turn"])
        assert row["fit"] in FITS and row["arm"] in ("candidate", "reference")
        assert row["actor"] in (0, 1) and row["phase"] in (0, 1, 2) and row["privacy_category"] in range(4)
        assert 1 <= row["support"] <= 8
        assert row["support"] == min(row["canonical_support"], 8 if row["arm"] == "candidate" else 4)
        counts = row["world_counts"]
        assert len(counts) == row["support"] and all(type(count) is int and count > 0 for count in counts)
        q, remainder = divmod(128*min(row["canonical_support"], 4), row["support"])
        assert sorted(counts) == [q]*(row["support"]-remainder)+[q+1]*remainder
        assert row["worlds"] == max(counts) and row["continuations"] == sum(counts)
    cells, full = [], set()
    for arm in ("candidate", "reference"):
        for fit in FITS:
            group = [row for row in inventory if row["arm"] == arm and row["fit"] == fit]
            for phase in range(3):
                for actor in (0, 1):
                    for privacy in range(4):
                        part = [row for row in group if (row["phase"], row["actor"], row["privacy_category"]) == (phase, actor, privacy)]
                        key = f"phase_privacy|{arm}|{fit}|{phase}|{actor}|{privacy}"
                        selected = min(part, key=lambda row: stress_priority(stage, row))["root_id"] if part else None
                        cells.append({"cell": key, "eligible_count": len(part), "selected_root": selected})
                        if selected is not None:
                            full.add(selected)
            for support in (1, 2, 3):
                part = [row for row in group if row["support"] == support]
                selected = min(part, key=lambda row: stress_priority(stage, row))["root_id"] if part else None
                cells.append({"cell": f"short_support|{arm}|{fit}|{support}", "eligible_count": len(part), "selected_root": selected})
                if selected is not None:
                    full.add(selected)
    for fit in FITS:
        for support in (5, 6, 7):
            part = [row for row in inventory if row["arm"] == "candidate"
                    and row["fit"] == fit and row["canonical_support"] == support]
            selected = min(part, key=lambda row: stress_priority(stage, row))["root_id"] if part else None
            cells.append({"cell": f"candidate_canonical_support|{fit}|{support}",
                          "eligible_count": len(part), "selected_root": selected})
            if selected is not None:
                full.add(selected)
    assert len(cells) == 171 and len(full) <= 171
    selections = []
    for row in inventory:
        ordinary = ordinary_indices(stage, row["index"], row["turn"], row["worlds"])
        stress = row["root_id"] in full
        selections.append({"root_id": row["root_id"], "ordinary_world_indices": ordinary,
                           "replay_world_indices": list(range(row["worlds"])) if stress else ordinary,
                           "stress": stress})
    selected_cells = sum(cell["selected_root"] is not None for cell in cells)
    return {"format": "equal-budget-breadth-arena-128-recovery-02-audit-sample-v1", "manifest": binding, "stage": stage,
            "config": SAMPLE_CONFIG, "games_lock_sha256": game_lock_sha256,
            "inventory": inventory, "selections": selections, "stress_cells": cells,
            "full_stress_roots": sorted(full), "coverage": {
                "games": len({row["index"] for row in inventory}), "roots": len(inventory),
                "recorded_worlds": sum(row["worlds"] for row in inventory),
                "recorded_continuations": sum(row["continuations"] for row in inventory),
                "replayed_worlds": sum(len(row["replay_world_indices"]) for row in selections),
                "replayed_continuations": sum(sum(wi < count for count in root["world_counts"])
                                              for root, selected in zip(inventory, selections, strict=True)
                                              for wi in selected["replay_world_indices"]),
                "stress_cells": len(cells), "selected_stress_cells": selected_cells,
                "stress_roots": len(full), "stress_selection_overlaps": selected_cells-len(full),
                "missing_stress_cells": [row["cell"] for row in cells if row["selected_root"] is None],
                "full_inner_replay": False}}


def seal_sample(stage, game_items, binding, base=OUTPUT):
    base = Path(base)
    lock_path = base / f"{stage}_games_lock.json"
    lock = old.read_json(lock_path)
    assert lock["manifest"] == binding and lock["files"] == game_items
    assert lock["recovery_binding"] == c.recovery_binding(base)
    inventory = metadata_inventory(stage, game_items, binding, base)
    sample = sample_plan(stage, inventory, binding, old.file_hash(lock_path))
    sample["recovery_binding"] = c.recovery_binding(base)
    path = base / stage / "audit_sample.json"
    old.save_new(path, sample)
    old.save_new(base / stage / "audit_sample_seal.json", {
        "manifest": binding, "recovery_binding": c.recovery_binding(base),
        "stage": stage, "sample_sha256": old.file_hash(path),
        "games_lock_sha256": old.file_hash(lock_path)})
    return _artifact(path)


def load_sample(stage, binding, base=OUTPUT):
    """Cheap per-game seal check; global reconstruction occurs before/after audit."""
    base = Path(base)
    path = base / stage / "audit_sample.json"
    seal = old.read_json(base / stage / "audit_sample_seal.json")
    assert seal["manifest"] == binding and seal["stage"] == stage
    assert seal["recovery_binding"] == c.recovery_binding(base)
    assert seal["sample_sha256"] == old.file_hash(path)
    assert seal["games_lock_sha256"] == old.file_hash(base / f"{stage}_games_lock.json")
    sample = old.read_json(path)
    assert sample["manifest"] == binding and sample["stage"] == stage and sample["config"] == SAMPLE_CONFIG
    assert sample["recovery_binding"] == c.recovery_binding(base)
    assert sample["games_lock_sha256"] == seal["games_lock_sha256"]
    return sample


def validate_sample(stage, binding, base=OUTPUT):
    sample = load_sample(stage, binding, base)
    lock_path = Path(base) / f"{stage}_games_lock.json"
    lock = old.read_json(lock_path)
    assert lock["manifest"] == binding and lock["recovery_binding"] == c.recovery_binding(base)
    inventory = metadata_inventory(stage, lock["files"], binding, base)
    expected = sample_plan(stage, inventory, binding, old.file_hash(lock_path))
    expected["recovery_binding"] = c.recovery_binding(base)
    compare(sample, expected)
    return sample


def _check_outcome(outcome, actor, turn):
    assert set(outcome) == {"reason", "winner", "turns", "trace_sha256", "final_state_sha256", "wdl"}
    assert outcome["reason"] in ("completed", "forced_pass_deadlock", "repetition_cutoff")
    assert outcome["winner"] is None or type(outcome["winner"]) is int and outcome["winner"] in (0, 1)
    assert type(outcome["turns"]) is int and 1 <= outcome["turns"] <= 400-turn
    if outcome["reason"] == "repetition_cutoff":
        assert outcome["winner"] is None and outcome["wdl"] is None
    else:
        assert type(outcome["wdl"]) in (int, float) and math.isfinite(outcome["wdl"])
        assert outcome["wdl"] == (0. if outcome["winner"] is None else 1. if outcome["winner"] == actor else -1.)
    for key in ("trace_sha256", "final_state_sha256"):
        assert isinstance(outcome[key], str) and re.fullmatch(r"[0-9a-f]{64}", outcome[key])


def _check_root_prefix(root, stage, index, turn, binding, base):
    if "recovery_binding" in root:
        delivered = sum(len(row["outcomes"]) for row in root["worlds"])
        assert type(root["new_cells"]) is int and root["new_cells"] >= 0
        assert root["new_cells"] == delivered-root.get("inherited_cells", 0)
    inventory = c.load_inventory(base)
    required = inventory["replaceable_roots"].get(stage, {}).get(f"{index}:{turn}")
    if root.get("recovery_binding") == c.recovery_binding(base) and required is not None:
        assert "recovery_origin" in root
    inherited_path = c.lineage_ancestor(root, base)
    if inherited_path is None:
        assert not any(key in root for key in ("recovery_origin", "inherited_cells", "inherited_timing"))
        return
    inherited = old.read_json(inherited_path)
    c.validate_origin(inherited, inherited_path, binding, base)
    assert "failure" in inherited and "failure" not in root
    assert root["recovery_origin"] == c.artifact(inherited_path)
    for key in ("snapshot", "privacy", "root_identity", "planner", "schedule", "candidates",
                "proposal", "allocation", "scores"):
        if key in inherited:
            assert root[key] == inherited[key]
    assert root["inherited_timing"] == inherited["timing"]
    assert root["inherited_cells"] == sum(len(row["outcomes"]) for row in inherited["worlds"])
    assert len(root["worlds"]) >= len(inherited["worlds"])
    for row, original in zip(root["worlds"], inherited["worlds"]):
        assert {k: v for k, v in row.items() if k != "outcomes"} == {
            k: v for k, v in original.items() if k != "outcomes"}
        assert row["outcomes"][:len(original["outcomes"])] == original["outcomes"]


def _check_game_prefix(game, stage, index, binding, base):
    required = c.load_inventory(base)["partial_games"].get(stage, {}).get(str(index))
    if game.get("recovery_binding") == c.recovery_binding(base) and required is not None:
        assert "recovery_origin" in game
    inherited_path = c.lineage_ancestor(game, base)
    if inherited_path is None:
        assert not any(key in game for key in ("recovery_origin", "inherited_turns", "inherited_timing"))
        return None
    inherited = old.read_json(inherited_path)
    c.validate_origin(inherited, inherited_path, binding, base)
    assert "failure" in inherited and "failure" not in game
    assert game["recovery_origin"] == c.artifact(inherited_path)
    assert game["spec"] == inherited["spec"]
    assert game["initial_state_sha256"] == inherited["initial_state_sha256"]
    assert game["inherited_timing"] == inherited["timing"]
    assert game["inherited_turns"] == len(inherited["trace"])
    assert game["trace"][:len(inherited["trace"])] == inherited["trace"]
    return inherited


def audit_root(stage, index, state, move, spec, binding, base=OUTPUT, selection=None, full_fixture=False):
    c.budget_boundary(base)
    assert spec == independent_spec(stage, index)
    start, cpu = time.perf_counter(), time.process_time()
    if full_fixture:
        resolved = Path(base).resolve()
        assert stage == "test" and resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents
        assert not resolved.is_relative_to(c.SOURCE)
    path = c.resolve_root(stage, index, state.turn_index, base)
    assert old.file_hash(path) == move["root_sha256"]
    root = old.read_json(path)
    c.validate_origin(root, path, binding, base)
    assert "failure" not in root
    _check_root_prefix(root, stage, index, state.turn_index, binding, base)
    assert old.digest(root["snapshot"]) == old.digest(old.serialize_state_snapshot(state))
    schedule = independent_schedule(state, spec["search_stream"])
    assert root["schedule"] == schedule
    obs, features, _, _, legal = a157.input_invariance(state, old._RAW._model.config)
    with torch.no_grad():
        logits, _ = old._RAW._model(obs[None], features[None], torch.ones((1, len(legal)), dtype=torch.bool))
    assert torch.isfinite(logits).all()
    compare(root["scores"], logits[0].tolist())
    altered = a157.hidden_alternative(state)
    assert independent_schedule(altered, spec["search_stream"]) == schedule
    candidate = state.current_player == spec["candidate_seat"]
    source_model = independent_model_id(stage, spec["fit"], candidate)
    identity = [stage, index, state.turn_index]
    assert root["root_identity"] == identity
    assert root["planner"] == ("candidate" if candidate else "reference")
    expected = independent_proposal_for_root(state, spec["search_stream"], source_model, identity, candidate)
    compare({key: value for key, value in root["proposal"].items() if key != "timing"}, expected)
    proposal_timing = root["proposal"]["timing"]
    assert set(proposal_timing) == {"wall", "learned_extra_wall"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in proposal_timing.values())
    assert proposal_timing["learned_extra_wall"] <= proposal_timing["wall"]
    a157.input_invariance(state, p._MODELS[source_model].config)
    compare(expected, independent_proposal_for_root(altered, spec["search_stream"], source_model, identity, candidate))
    offered = expected["candidates"]
    assert offered["keys"][:3] == expected["reference"]["keys"][:3]
    assert root["candidates"] == offered and len(offered["actions"]) == min(8 if candidate else 4, offered["canonical_support"])
    assert len(set(offered["keys"])) == len(offered["actions"])
    allocation = expected["allocation"]
    assert root["allocation"] == allocation
    counts, world_count = allocation["world_counts"], allocation["world_count"]
    ordinary = ordinary_indices(stage, index, state.turn_index, world_count)
    assert selection is not None and selection["root_id"] == root_id(index, state.turn_index)
    assert selection["ordinary_world_indices"] == ordinary and type(selection["stress"]) is bool
    expected_indices = list(range(world_count)) if selection["stress"] else ordinary
    assert selection["replay_world_indices"] == expected_indices
    replay_indices = list(range(world_count)) if full_fixture else expected_indices
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert root["privacy"] == privacy and len(root["worlds"]) == world_count
    replay_cells = independent_replay_cells(root["worlds"], replay_indices)
    proof_by_cell = {}
    if "equivalence_proofs" in root:
        proofs = root["equivalence_proofs"]
        keys = [(proof["world"], proof["candidate"]) for proof in proofs]
        assert keys == sorted(replay_cells["unresolved"]), "Every and only unresolved cell needs one proof"
        proof_by_cell = dict(zip(keys, proofs, strict=True))
    independent_proofs = []
    moves, reasons = 0, Counter()
    ordinary_wall, extra_wall, censored_wall = 0., 0., 0.
    for wi, (seed, saved) in enumerate(zip(schedule["world_seeds"][:world_count], root["worlds"], strict=True)):
        c.budget_boundary(base)
        active = [j for j, count in enumerate(counts) if wi < count]
        assert set(saved) == {"seed", "state_sha256", "candidate_indices", "outcomes"}
        assert saved["seed"] == seed and len(saved["outcomes"]) == len(active)
        assert saved["candidate_indices"] == active and all(type(j) is int for j in saved["candidate_indices"])
        world = a154.independent_world(state, seed)
        world_sha = old.digest(old.serialize_state_snapshot(world))
        assert saved["state_sha256"] == world_sha
        assert old.digest(old.serialize_state_snapshot(a154.independent_world(altered, seed))) == world_sha
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        legal_world = old.SplendorEnv().legal_actions(world)
        for j, outcome in zip(active, saved["outcomes"], strict=True):
            payload = offered["actions"][j]
            action = old.decode_action(payload)
            assert action in legal_world
            _check_outcome(outcome, state.current_player, state.turn_index)
            if (wi, j) in replay_cells["all"]:
                replay_start = time.perf_counter()
                if (wi, j) in proof_by_cell:
                    proof = independent_equivalence_proof(world, action, state.current_player, wi, j)
                    assert old.digest(proof_by_cell[(wi, j)]) == old.digest(proof)
                    independent_proofs.append(proof)
                    rebuilt = {key: proof["replay"][key] for key in outcome if key != "wdl"}
                    rebuilt["wdl"] = None
                else:
                    rebuilt = a149.replay_branch(world, action, state.current_player)
                compare(outcome, rebuilt)
                elapsed = time.perf_counter()-replay_start
                if wi in ordinary:
                    ordinary_wall += elapsed
                elif wi in replay_indices:
                    extra_wall += elapsed
                else:
                    censored_wall += elapsed
                moves += rebuilt["turns"]
                reasons[rebuilt["reason"]] += 1
    arithmetic = certified_selection(root, counts, independent_proofs if proof_by_cell else None)
    choice = arithmetic["choice"]
    assert move["action"] == offered["actions"][choice]
    _timing(root["timing"])
    assert sum(reasons.values()) == len(replay_cells["all"])
    assert reasons.get("repetition_cutoff", 0) == len(replay_cells["unresolved"])
    extra_cells = sorted(replay_cells["extra"])
    return {"manifest": binding, "recovery_binding": c.recovery_binding(base),
            "index": index, "turn": state.turn_index, "seat": state.current_player,
            "planner": "candidate" if candidate else "reference", "fit": spec["fit"],
            "model_id": source_model, "model_sha256": expected["model_sha256"],
            "sha256": move["root_sha256"], "choice": choice, "support": len(counts),
            "canonical_support": offered["canonical_support"], "added_indices": expected["added_indices"],
            "assigned_fit": spec["fit"], "privacy": privacy, **_timing_fields(root),
            "worlds": world_count, "recorded_worlds": world_count,
            "replayed_worlds": len({wi for wi, j in replay_cells["all"]}),
            "world_counts": counts, "allocation": allocation,
            "continuations": allocation["continuations"], "replayed_branches": sum(reasons.values()),
            "terminal_continuations": allocation["continuations"]-len(replay_cells["unresolved"]),
            "unresolved_continuations": len(replay_cells["unresolved"]),
            "certified_roots": int(bool(replay_cells["unresolved"])),
            "coupled_certified_roots": int(bool(proof_by_cell)),
            "equivalence_proof_cells": len(proof_by_cell),
            "coupled_unknown_groups": len(arithmetic.get("certificate", {}).get("groups", [])),
            "censored_replay_cells": len(replay_cells["unresolved"]),
            "extra_censored_replay_cells": len(extra_cells),
            "extra_censored_cells": [list(cell) for cell in extra_cells],
            "extra_censored_replay_world_indices": sorted({wi for wi, j in extra_cells}),
            "ordinary_world_indices": ordinary, "replay_world_indices": replay_indices,
            "stress": selection["stress"], "full_fixture_replay": bool(full_fixture),
            "full_inner_replay": bool(full_fixture), "moves": moves, "reasons": dict(reasons),
            "audit_timing": {"wall": time.perf_counter()-start, "cpu": time.process_time()-cpu,
                             "ordinary_replay_wall": ordinary_wall, "stress_extra_replay_wall": extra_wall,
                             "extra_censored_replay_wall": censored_wall}}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def independent_statistics(rows, stage="main"):
    assert stage == "main"
    assert len(rows) == 402 and [r["index"] for r in rows] == list(range(402))
    specs = independent_specs(stage)
    strata = {seed: [] for seed in (161001, 161002, 161003)}
    for i in range(0, 402, 2):
        a, b = rows[i:i+2]
        assert (a["seat"], b["seat"]) == (0, 1) and a["pair"] == b["pair"] == i // 2
        assert a["fit"] == b["fit"] == specs[i]["fit"]
        assert all(type(row["score"]) in (int, float) and row["score"] in (0., .5, 1.) for row in (a, b))
        strata[a["fit"]].append(int(2*a["score"]+2*b["score"]))
    assert all(len(v) == 67 for v in strata.values())
    rng = np.random.default_rng(2173000000)
    boot = []
    for _ in range(50):
        draws = [rng.integers(0, 67, size=(1000, 67)) for _ in range(3)]
        for i in range(1000):
            total = 0
            for values, indices in zip(strata.values(), draws):
                counts = np.bincount(indices[i], minlength=67)
                total += sum(int(n)*value for n, value in zip(counts, values))
            boot.append(total / 804)
    share = sum(r["score"] for r in rows) / 402
    interval = [float(linear_quantile(boot, q)) for q in (.025, .975)]
    superiority = interval[0] > .5
    practical = 10 * sum(int(2*r["score"]) for r in rows) >= 11 * 402 and superiority
    label = ("positive_practical_progress" if practical else
             "positive_below_observed_target" if superiority else
             "demonstrated_harm" if interval[1] < .5 else
             "below_material_target" if interval[1] < .55 else "inconclusive")
    return {"games": 402, "pairs": 201, "share": share, "ci95": interval,
            "wins": sum(r["score"] == 1 for r in rows), "draws": sum(r["score"] == .5 for r in rows),
            "losses": sum(r["score"] == 0 for r in rows), "stage": stage,
            "stage_pass": practical, "superiority_established": superiority,
            "inferiority_established": interval[1] < .5,
            "practical_progress": practical, "material_target_excluded": interval[1] < .55, "classification": label}


def _assert_inherited_snapshot(snapshot, inherited_snapshot, index, turn):
    """Compare exact serialized states while accepting JSON object-key encoding."""
    actual_sha, expected_sha = old.digest(snapshot), old.digest(inherited_snapshot)
    assert actual_sha == expected_sha, (
        f"Game {index}, turn {turn}: inherited snapshot canonical mismatch; "
        f"reconstructed={actual_sha}, saved={expected_sha}"
    )


def audit_game(stage, index, expected_sha, binding, base=OUTPUT, test_initial_snapshot=None, full_fixture=False):
    """Reconstruct a complete outer game; source policies are independently checked."""
    wall, cpu = time.perf_counter(), time.process_time()
    base = Path(base)
    spec = independent_spec(stage, index)
    if stage == "test":
        resolved = base.resolve()
        assert resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents
        assert not resolved.is_relative_to(c.SOURCE)
        assert test_initial_snapshot is not None
        state = old.deserialize_state_snapshot(test_initial_snapshot)
        env = old.SplendorEnv()
    else:
        assert test_initial_snapshot is None and not full_fixture
        env = old.SplendorEnv(seed=spec["environment_seed"])
        state = env.initial_state()
    sample = load_sample(stage, binding, base)
    inventory = {row["root_id"]: row for row in sample["inventory"] if row["index"] == index}
    selected = {row["root_id"]: row for row in sample["selections"] if row["root_id"] in inventory}
    assert len(selected) == len(inventory)
    lock = old.read_json(base / f"{stage}_games_lock.json")
    assert lock["manifest"] == binding and lock["recovery_binding"] == c.recovery_binding(base)
    assert lock["files"][index]["index"] == index and lock["files"][index]["sha256"] == expected_sha
    path = c.resolve_game(stage, index, base)
    assert old.file_hash(path) == expected_sha
    game = old.read_json(path)
    c.validate_origin(game, path, binding, base)
    assert game["spec"] == spec and "failure" not in game
    inherited = _check_game_prefix(game, stage, index, binding, base)
    ids = [root_id(index, move["turn"]) for move in game["trace"]]
    assert len(set(ids)) == len(ids) and set(ids) == set(inventory)
    if stage == "test":
        assert old.digest(game["test_initial_snapshot"]) == old.digest(test_initial_snapshot)
    else:
        assert "test_initial_snapshot" not in game
    initial_sha = old.digest(old.serialize_state_snapshot(state))
    assert game["initial_state_sha256"] == initial_sha
    start_turn = state.turn_index
    visits, passes, idle, hasher, roots = Counter(), 0, 0, hashlib.sha256(), []
    for n, move in enumerate(game["trace"]):
        c.budget_boundary(base)
        assert not state.terminal and state.turn_index < 400
        visits[old.state_signature(state)] += 1
        assert visits[old.state_signature(state)] < 4
        assert move["turn"] == state.turn_index == start_turn+n and move["actor"] == state.current_player
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        if inherited is not None and n == len(inherited["trace"]):
            _assert_inherited_snapshot(old.serialize_state_snapshot(state), inherited["final_state"], index, state.turn_index)
        legal = env.legal_actions(state)
        assert legal
        identity = root_id(index, state.turn_index)
        root_record = old.read_json(c.resolve_root(stage, index, state.turn_index, base))
        compare(_root_metadata(stage, spec, move, root_record), inventory[identity])
        roots.append(audit_root(stage, index, state, move, spec, binding, base, selected[identity], full_fixture))
        action = old.decode_action(move["action"])
        assert action in legal
        hasher.update(old.digest(move["action"]).encode())
        successor = env.step(state, action)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(state, successor, action) else idle+1
        state = successor
        if n < len(game["trace"])-1:
            assert passes < 2 and idle < 60
    assert game["turns"] == len(game["trace"]) and game["trace_sha256"] == hasher.hexdigest()
    assert old.digest(game["final_state"]) == old.digest(old.serialize_state_snapshot(state))
    assert game["final_state_sha256"] == old.digest(game["final_state"])
    if game["reason"] == "completed":
        assert state.terminal and passes < 2 and idle < 60
        winner = state.winner
    else:
        assert game["reason"] == "forced_pass_deadlock" and passes == 2
        winner = adjudicate(state)
    assert game["winner"] == winner
    seat = spec["candidate_seat"]
    score = .5 if winner is None else float(winner == seat)
    _timing(game["timing"])
    row = {"index": index, "pair": spec["pair"], "seat": seat, "fit": spec["fit"], "score": score,
           "margin": state.players[seat].score-state.players[1-seat].score,
           "turns": game["turns"], "reason": game["reason"], **_timing_fields(game)}
    return {"manifest": binding, "recovery_binding": c.recovery_binding(base),
            "game": row, "roots": roots, "initial_sha256": initial_sha,
            "sample_sha256": old.file_hash(base / stage / "audit_sample.json"),
            "full_inner_replay": bool(full_fixture),
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}

