"""156 independent full-game, information-set, continuation and paired audit.

Shares the frozen engine/scorer and separately implemented 149 branch replay,
154 catalog-minus-known world builder and canonical candidate reconstruction.
Separately reconstructs both players' decisions and all156 statistics. Shared
engine, scorers, serialization, canonical keys and parent149/154 audit helpers.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import math
import time
import random
import copy
import statistics

from splendor_ai.alphazero.resolver_champion_156 import r084

import numpy as np
import torch

from splendor_ai.alphazero import run_rollout_selector_champion as p
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

spec = importlib.util.spec_from_file_location("audit154_for156", "scripts/analysis/verify_rollout_selector_154.py")
a154 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a154)
compare = a154.compare


def independent_spec(stage, index):
    assert stage in ("probe", "main") and 0 <= index < (8 if stage == "probe" else 400)
    return {"index": index, "pair": index//2, "candidate_seat": index%2,
            "environment_seed": (1569000000 if stage == "probe" else 1560000000)+index//2,
            "search_stream": p.old.digest(["156-independent-search-stream-v1", stage, index]), "champion_stream_index": index}


def independent_schedule(state, stream):
    key = p.old.digest(canonical_public_belief_key(state, state.current_player))
    return {"belief_sha256": key, "candidate_nonce": p.old.digest(["155-candidate-v1", stream, key]),
            "world_seeds": ["155-world-v1|%s|%s|%02d" % (stream, key, i) for i in range(32)]}


def hidden_alternative(state):
    changed = state.copy_shallow()
    other = changed.players[1-state.current_player]
    for cards in changed.hidden_tier_decks.values():
        cards.reverse()
    for i, card in enumerate(other.reserved_cards):
        deck = changed.hidden_tier_decks[card.tier]
        if other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE and deck:
            other.reserved_cards[i], deck[0] = deck[0], other.reserved_cards[i]
    return changed


def audit_root(stage, index, state, move, stream, binding):
    path = p.OUTPUT/stage/"roots"/f"game_{index:04d}"/f"turn_{state.turn_index:03d}.json"
    assert p.old.file_hash(path) == move["root_sha256"]
    assert path.stat().st_mtime_ns >= (p.OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    record = p.old.read_json(path)
    assert record["manifest"] == binding and "failure" not in record
    assert p.old.digest(record["snapshot"]) == p.old.digest(p.old.serialize_state_snapshot(state))
    schedule = independent_schedule(state, stream)
    assert record["schedule"] == schedule
    candidates = a154.independent_candidates(state, "155-live-v1", schedule["candidate_nonce"])
    assert record["candidates"] == candidates
    legal = p.old.SplendorEnv().legal_actions(state)
    scores = [float(v) for v in p.old._RAW.score_actions(state, legal)]
    assert all(math.isfinite(v) for v in scores)
    compare(record["scores"], scores)
    assert candidates["actions"][0] == p.old.serialize_action(legal[max(range(len(legal)), key=lambda i: scores[i])])
    # Audit each actual root's information invariance, not just hand-picked cases.
    changed = hidden_alternative(state)
    assert independent_schedule(changed, stream) == schedule
    changed_legal = p.old.SplendorEnv().legal_actions(changed)
    assert changed_legal == legal
    compare([float(v) for v in p.old._RAW.score_actions(changed, changed_legal)], scores)
    assert a154.independent_candidates(changed, "155-live-v1", schedule["candidate_nonce"]) == candidates
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert record["privacy"] == privacy and len(record["worlds"]) == 32
    totals, reasons, moves = [0.]*len(candidates["actions"]), Counter(), 0
    for seed, row in zip(schedule["world_seeds"], record["worlds"]):
        assert row["seed"] == seed and len(row["outcomes"]) == len(totals)
        world = a154.independent_world(state, seed)
        world_snapshot = p.old.serialize_state_snapshot(world)
        assert p.old.digest(world_snapshot) == row["state_sha256"]
        assert p.old.serialize_state_snapshot(a154.independent_world(changed, seed)) == world_snapshot
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        for j, (action, outcome) in enumerate(zip(candidates["actions"], row["outcomes"])):
            decoded = p.old.decode_action(action)
            assert decoded in p.old.SplendorEnv().legal_actions(world)
            replayed = a154.a149.replay_branch(world, decoded, state.current_player)
            assert replayed == outcome and replayed["reason"] in ("completed", "forced_pass_deadlock")
            totals[j] += replayed["wdl"]
            reasons[replayed["reason"]] += 1
            moves += replayed["turns"]
    means = [v/32 for v in totals]
    choice = max(range(len(means)), key=lambda i: means[i])
    compare(record["means"], means)
    assert record["choice"] == choice and move["action"] == candidates["actions"][choice]
    for key in ("wall", "cpu"):
        assert math.isfinite(record["timing"][key]) and record["timing"][key] >= 0
    return {"index": index, "seat": state.current_player, "turn": state.turn_index,
            "choice": choice, "support": len(totals), "canonical_support": candidates["canonical_support"],
            "legal_support": len(legal), "privacy": privacy, "timing": record["timing"],
            "continuations": sum(reasons.values()), "moves": moves, "reasons": dict(reasons)}


def distinct(state, scores):
    unique = {}
    for row in scores:
        unique.setdefault(p.old.canonical_successor_action_key(state, row.action), row)
    return list(unique.values())


def champion_plan(stage, index, decision, state, scorer):
    env, seat = p.old.SplendorEnv(), state.current_player
    legal = env.legal_actions(state)
    scores = scorer.score_actions(state, legal)
    assert scores and all(math.isfinite(r.score) for r in scores)
    ranked = distinct(state, scores)
    plan = {"scores": [{"action": p.old.serialize_action(r.action), "score": r.score,
                        "action_index": r.action_index} for r in scores],
            "parent": p.old.serialize_action(ranked[0].action), "worlds": [], "leaves": [], "reason": None}
    if len(ranked) < 4:
        plan["reason"] = "fewer_than_four_root_actions"
        return plan
    keys = tuple(p.old.canonical_successor_action_key(state, r.action) for r in ranked[:4])
    plan["keys"] = keys
    start = 1569400000 if stage == "probe" else 1564000000
    root_index = start+index*1000+seat*400+decision
    worlds = [a154.independent_world(state, 49000000+root_index*10003+i) for i in range(3)]
    plan["worlds"] = [{"seed": 49000000+root_index*10003+i,
                       "state_sha256": p.old.digest(p.old.serialize_state_snapshot(world))} for i, world in enumerate(worlds)]
    information_actions = {}
    for det, world in enumerate(worlds):
        mapping = p.old.actions_by_canonical_successor_key(world, env.legal_actions(world))
        if any(k not in mapping for k in keys):
            plan["reason"] = "root_mapping_changed"
            return plan
        for key in keys:
            after = env.step(world, mapping[key])
            if after.terminal:
                plan["reason"] = "root_action_terminal"
                return plan
            responses = distinct(after, scorer.score_actions(after, env.legal_actions(after)))
            if len(responses) < 4:
                plan["reason"] = "fewer_than_four_opponent_actions"
                return plan
            responses = responses[:4]
            response_keys = tuple(p.old.canonical_successor_action_key(after, r.action) for r in responses)
            info = (key, canonical_public_belief_key(after, after.current_player))
            if info in information_actions and information_actions[info] != response_keys:
                plan["reason"] = "opponent_action_set_unstable"
                return plan
            information_actions[info] = response_keys
            for row, response_key in zip(responses, response_keys):
                plan["leaves"].append((det, key, info, response_key, after, row.action))
    return plan


def fallback_action(state, scores, counts):
    reps = {}
    for row in scores:
        key = p.old.canonical_successor_action_key(state, row.action)
        previous = reps.get(key)
        if previous is None or (row.score, -row.action_index) > (previous.score, -previous.action_index):
            reps[key] = row
    buys = [r for r in reps.values() if r.action.action_type.name in ("BUY_VISIBLE", "BUY_RESERVED")]
    if buys:
        return max(buys, key=lambda r: (r.score, -r.action_index)).action
    minimum = min(counts.get(k, 0) for k in reps)
    return max((r for k, r in reps.items() if counts.get(k, 0) == minimum),
               key=lambda r: (r.score, -r.action_index)).action


def replay_champion_branch(state, action, actor, seed, scorer):
    """Independent084 stochastic rollout loop, preserving its historical rules."""
    env, rng = p.old.SplendorEnv(), random.Random(seed)
    hasher = hashlib.sha256()
    hasher.update(p.old.digest(p.old.serialize_action(action)).encode())
    current = env.step(state, action)
    start_turn, moves = current.turn_index, 1
    visits, counts, idle, fallbacks, adjudicated = Counter(), {}, 0, 0, False
    while not current.terminal and current.turn_index < start_turn+400:
        key = p.old.state_signature(current)
        visits[key] += 1
        scores = scorer.score_actions(current, env.legal_actions(current))
        assert scores and all(math.isfinite(r.score) for r in scores)
        if visits[key] >= 4 or idle >= 60:
            chosen = fallback_action(current, scores, counts.setdefault(key, {}))
            fallbacks += 1
        elif len(scores) == 1 or rng.random() >= .1:
            chosen = scores[0].action
        else:
            support = scores[:min(3, len(scores))]
            maximum = max(r.score for r in support)
            weights = [math.exp((r.score-maximum)/.5) for r in support]
            chosen = rng.choices([r.action for r in support], weights=weights, k=1)[0]
        successor_key = p.old.canonical_successor_action_key(current, chosen)
        uses = counts.setdefault(key, {})
        uses[successor_key] = uses.get(successor_key, 0)+1
        hasher.update(p.old.digest(p.old.serialize_action(chosen)).encode())
        nxt = env.step(current, chosen)
        idle = 0 if p.old.is_progress_transition(current, nxt, chosen) else idle+1
        current = nxt
        moves += 1
        if fallbacks >= 64:
            adjudicated = True
            break
    margin = float(current.players[actor].score-current.players[1-actor].score)
    outcome = {"value": max(-1., min(1., margin/20.)), "terminal": current.terminal,
               "resolved": current.terminal or adjudicated, "turns": current.turn_index-start_turn,
               "fallback_triggers": fallbacks, "loop_adjudicated": adjudicated,
               "hard_cap_reached": not current.terminal and current.turn_index >= start_turn+400}
    return {"seed": seed, "state_sha256": p.old.digest(p.old.serialize_state_snapshot(state)),
            "action": p.old.serialize_action(action), "actor": actor, "outcome": outcome,
            "moves": moves, "trace_sha256": hasher.hexdigest(),
            "final_state_sha256": p.old.digest(p.old.serialize_state_snapshot(current))}


def independent_resolution(leaves, keys):
    values, choices = [], []
    for key in keys:
        groups = {}
        for leaf in leaves:
            if leaf[1] == key:
                groups.setdefault(leaf[2], []).append(leaf)
        value = 0.
        for information, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
            dets = sorted({r[0] for r in rows})
            actions = sorted({r[3] for r in rows}, key=repr)
            means = []
            for action in actions:
                part = [r for r in rows if r[3] == action]
                assert len(part) == len(dets)
                means.append(sum(r[4] for r in part)/len(part))
            selected = min(range(len(actions)), key=lambda j: (means[j], repr(actions[j])))
            value += sum(r[4] for r in rows if r[3] == actions[selected])/3
            choices.append({"root_action": repr(key), "opponent_information_set": repr(information),
                            "determinizations": dets, "opponent_actions": [repr(a) for a in actions],
                            "action_means": means, "selected_action": repr(actions[selected])})
        values.append(value)
    winner = max(range(len(values)), key=lambda j: (values[j], -j))
    return {"root_actions": [repr(k) for k in keys], "action_values": values,
            "regrets_from_best": [v-values[winner] for v in values], "best_index": winner,
            "best_action": repr(keys[winner]), "opponent_choices": choices}


def audit_champion(stage, index, state, move, binding, scorer, telemetry, reference=None):
    path = p.champion_root_path(stage, index, state.turn_index)
    assert p.old.file_hash(path) == move["champion_root_sha256"]
    assert path.stat().st_mtime_ns >= (p.OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    record = p.old.read_json(path)
    assert record["manifest"] == binding and "failure" not in record
    assert p.old.digest(record["snapshot"]) == p.old.digest(p.old.serialize_state_snapshot(state))
    assert record["seat"] == state.current_player and record["stream_index"] == index
    assert record["decision_index"] == telemetry["decisions"] and record["before"] == telemetry
    decision = telemetry["decisions"]
    plan = champion_plan(stage, index, decision, state, scorer)
    alternate = champion_plan(stage, index, decision, hidden_alternative(state), scorer)
    for key in ("scores", "parent", "worlds", "reason"):
        compare(plan[key], alternate[key])
    compare(record["scores"], plan["scores"])
    assert record["parent"] == plan["parent"] and record["worlds"] == plan["worlds"]
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert record["privacy"] == privacy
    after = copy.deepcopy(telemetry)
    after["decisions"] += 1
    phase = "early" if state.turn_index < 20 else "mid" if state.turn_index < 50 else "late"
    phase_row = after["by_phase"].setdefault(phase, {"decisions": 0, "completed_searches": 0, "switches": 0, "fallbacks": 0})
    phase_row["decisions"] += 1
    loops, triggers, moves = 0, 0, 0
    if plan["reason"] is not None:
        assert record["kind"] == "fallback" and record["reason"] == plan["reason"] and not record["branches"]
        selected = plan["parent"]
        after["fallbacks"] += 1
        after["fallback_reasons"][plan["reason"]] = after["fallback_reasons"].get(plan["reason"], 0)+1
        phase_row["fallbacks"] += 1
    else:
        assert record["kind"] == "search" and len(record["branches"]) == 192 and len(plan["leaves"]) == 48
        leaves = []
        for i, (det, root, information, response, start, action) in enumerate(plan["leaves"]):
            # Equivalent hidden allocations must produce exactly the same tree.
            alternate_leaf = alternate["leaves"][i]
            assert (det, root, information, response, action) == (*alternate_leaf[:4], alternate_leaf[5])
            assert p.old.digest(p.old.serialize_state_snapshot(start)) == p.old.digest(p.old.serialize_state_snapshot(alternate_leaf[4]))
            outcomes = []
            for j in range(4):
                seed_start = 1569000000000000000 if stage == "probe" else 1560000000000000000
                seed = seed_start+index*1000000000+state.current_player*400000000+decision*1000000+det*10000+j
                replayed = replay_champion_branch(start, action, state.current_player, seed, scorer)
                compare(replayed, record["branches"][4*i+j])
                outcome = replayed["outcome"]
                assert outcome["resolved"] and not outcome["hard_cap_reached"]
                loops += int(outcome["loop_adjudicated"])
                triggers += outcome["fallback_triggers"]
                moves += replayed["moves"]
                outcomes.append(outcome["value"])
            leaves.append((det, root, information, response, statistics.fmean(outcomes)))
        resolution = independent_resolution(leaves, plan["keys"])
        compare(resolution, record["resolution"])
        mapping = p.old.actions_by_canonical_successor_key(state, p.old.SplendorEnv().legal_actions(state))
        selected = p.old.serialize_action(mapping[plan["keys"][resolution["best_index"]]])
        after["completed_searches"] += 1
        phase_row["completed_searches"] += 1
        after["continuations"] += 192
        after["resolved_continuations"] += 192
        after["loop_adjudications"] += loops
        after["rollout_fallback_triggers"] += triggers
        changed = p.old.canonical_successor_action_key(state, p.old.decode_action(selected)) != p.old.canonical_successor_action_key(state, p.old.decode_action(plan["parent"]))
        after["switches"] += int(changed)
        phase_row["switches"] += int(changed)
    assert record["action"] == selected == move["action"] and record["after"] == after
    if reference is not None:
        assert reference.telemetry == telemetry
        assert p.old.serialize_action(reference.choose_action(p.old.SplendorEnv(), state)) == selected
        assert reference.telemetry == after
    assert all(math.isfinite(record["timing"][k]) and record["timing"][k] >= 0 for k in ("wall", "cpu"))
    return {"index": index, "seat": state.current_player, "turn": state.turn_index,
            "kind": record["kind"], "reason": plan["reason"], "changed": after["switches"] > telemetry["switches"],
            "privacy": privacy, "timing": record["timing"], "continuations": len(record["branches"]),
            "moves": moves, "loop_adjudications": loops, "rollout_fallback_triggers": triggers,
            "hard_caps": 0, "resolved": len(record["branches"])}, after


def independent_champion_operational(rows):
    def part(part):
        return {"decisions": len(part), "searches": sum(r["kind"] == "search" for r in part),
                "changes": sum(r["changed"] for r in part),
                "fallbacks": dict(Counter(r["reason"] for r in part if r["kind"] == "fallback")),
                "own_private_roots": sum(r["privacy"]["own"] > 0 for r in part),
                "opponent_private_roots": sum(r["privacy"]["opponent"] > 0 for r in part)}
    def dist(values):
        return {"count": len(values), "mean": sum(values)/len(values), "p95": linear_quantile(values, .95),
                "max": max(values), "sum": sum(values)}
    paths = sum(r["continuations"] for r in rows)
    loops = sum(r["loop_adjudications"] for r in rows)
    return {**part(rows), "continuations": paths, "moves": sum(r["moves"] for r in rows),
            "loop_adjudications": loops, "loop_fraction": loops/paths if paths else 0.,
            "hard_caps": sum(r["hard_caps"] for r in rows), "resolved": sum(r["resolved"] for r in rows),
            "rollout_fallback_triggers": sum(r["rollout_fallback_triggers"] for r in rows),
            "decision_time": {k: dist([r["timing"][k] for r in rows]) for k in ("wall", "cpu")},
            "seats": {str(s): part([r for r in rows if r["seat"] == s]) for s in (0, 1)},
            "phases": {str(s): part([r for r in rows if (0 if r["turn"] < 20 else 1 if r["turn"] < 40 else 2) == s]) for s in range(3)}}


def audit_game(stage, index, sha, binding):
    wall, cpu = time.perf_counter(), time.process_time()
    path = p.OUTPUT/stage/"games"/f"game_{index:04d}.json"
    assert p.old.file_hash(path) == sha and path.stat().st_mtime_ns >= (p.OUTPUT/f"{stage}_launch.json").stat().st_mtime_ns
    record = p.old.read_json(path)
    spec = independent_spec(stage, index)
    assert record["spec"] == spec and record["manifest"] == binding and "failure" not in record
    env = p.old.SplendorEnv(seed=spec["environment_seed"])
    state = env.initial_state()
    initial_sha = p.old.digest(p.old.serialize_state_snapshot(state))
    assert record["initial_state_sha256"] == initial_sha
    visits, hasher, passes, idle = Counter(), hashlib.sha256(), 0, 0
    roots, champion_rows = [], []
    scorer = r084._LegalActionScorer(r084.CHECKPOINT, "cpu", "legal_action_attention_scorer")
    reference = r084.FrozenPublicBeliefResolverBot(p.champion.config(stage), index, 1-spec["candidate_seat"], scorer=scorer)
    telemetry = reference.telemetry
    for n, move in enumerate(record["trace"]):
        assert not state.terminal and state.turn_index < 400
        visits[p.old.state_signature(state)] += 1
        assert visits[p.old.state_signature(state)] < 4
        assert move["state_sha256"] == p.old.digest(p.old.serialize_state_snapshot(state))
        assert move["turn"] == state.turn_index == n and move["actor"] == state.current_player
        legal = env.legal_actions(state)
        if state.current_player == spec["candidate_seat"]:
            roots.append(audit_root(stage, index, state, move, spec["search_stream"], binding))
        else:
            assert "root_sha256" not in move
            cr, telemetry = audit_champion(stage, index, state, move, binding, scorer, telemetry,
                                           reference if stage == "probe" else None)
            champion_rows.append(cr)
        action = p.old.decode_action(move["action"])
        assert action in legal
        hasher.update(p.old.digest(move["action"]).encode())
        nxt = env.step(state, action)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if p.old.is_progress_transition(state, nxt, action) else idle+1
        state = nxt
        if n < len(record["trace"])-1:
            assert passes < 2 and idle < 60
    assert record["turns"] == len(record["trace"]) and record["trace_sha256"] == hasher.hexdigest()
    assert p.old.digest(record["final_state"]) == p.old.digest(p.old.serialize_state_snapshot(state))
    assert record["final_state_sha256"] == p.old.digest(record["final_state"])
    if record["reason"] == "completed":
        assert state.terminal and idle < 60 and passes < 2
        winner = state.winner
    else:
        assert record["reason"] == "forced_pass_deadlock" and passes == 2
        # Independently apply score then fewest bought-card tiebreak.
        ranks = [(player.score, -len(player.purchased_cards)) for player in state.players]
        winner = None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)
    assert record["champion_telemetry"] == telemetry
    reference.close()
    assert record["winner"] == winner
    seat = spec["candidate_seat"]
    score = .5 if winner is None else float(winner == seat)
    row = {"index": index, "pair": index//2, "seat": seat, "score": score,
           "margin": state.players[seat].score-state.players[1-seat].score,
           "turns": record["turns"], "reason": record["reason"], "timing": record["timing"]}
    assert all(math.isfinite(row["timing"][k]) and row["timing"][k] >= 0 for k in ("wall", "cpu"))
    return {"game": row, "roots": roots, "champion_roots": champion_rows, "initial_sha256": initial_sha,
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered)-1)*q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(point-lo)


def independent_statistics(rows):
    assert len(rows) == 400 and [r["index"] for r in rows] == list(range(400))
    values = []
    for pair in range(200):
        a, b = rows[pair*2:pair*2+2]
        assert a["pair"] == b["pair"] == pair and a["seat"] == 0 and b["seat"] == 1
        assert a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.)
        values.append((a["score"]+b["score"])/2)
    rng, boot = np.random.default_rng(1563000000), []
    # Explicit resample occurrence counts; no matrix product / NumPy BLAS.
    for _ in range(50):
        indices = rng.integers(0, 200, size=(1000, 200))
        for sample in indices:
            counts = np.bincount(sample, minlength=200)
            boot.append(float((counts*np.asarray(values)).sum())/200)
    ci = [linear_quantile(boot, q) for q in (.025, .975)]
    def summary(part):
        return {"games": len(part), "share": sum(r["score"] for r in part)/len(part),
                "wdl": {"wins": sum(r["score"] == 1 for r in part), "draws": sum(r["score"] == .5 for r in part),
                        "losses": sum(r["score"] == 0 for r in part)},
                "mean_margin": sum(r["margin"] for r in part)/len(part),
                "mean_turns": sum(r["turns"] for r in part)/len(part)}
    result = summary(rows)
    gates = {"material_share": result["share"] >= .55, "positive_lower_bound": ci[0] > .50,
             "positive_margin": sum(r["margin"] for r in rows) > 0,
             "both_seats": all(sum(r["score"] for r in rows[s::2])/200 >= .5 for s in (0, 1))}
    return {**result, "pairs": 200, "ci95": ci, "statistical_gates": gates,
            "statistical_pass": all(gates.values()), "seats": {str(s): summary(rows[s::2]) for s in (0, 1)}}


def independent_operational(data):
    roots, games = data["roots"], data["games"]
    def counts(part):
        n, changes = len(part), sum(r["choice"] > 0 for r in part)
        return {"roots": n, "changes": changes, "change_fraction": changes/n if n else None,
                "support_counts": dict(Counter(str(r["support"]) for r in part)),
                "own_private_roots": sum(r["privacy"]["own"] > 0 for r in part),
                "opponent_private_roots": sum(r["privacy"]["opponent"] > 0 for r in part)}
    def distribution(values):
        assert values and all(math.isfinite(v) and v >= 0 for v in values)
        return {"count": len(values), "mean": sum(values)/len(values), "p95": linear_quantile(values, .95),
                "max": max(values), "sum": sum(values)}
    inner = Counter()
    for r in roots:
        inner.update(r["reasons"])
    return {**counts(roots), "continuations": sum(r["continuations"] for r in roots),
            "continuation_moves": sum(r["moves"] for r in roots), "inner_reasons": dict(inner),
            "outer_reasons": dict(Counter(g["reason"] for g in games)),
            "decision_time": {k: distribution([r["timing"][k] for r in roots]) for k in ("wall", "cpu")},
            "champion": independent_champion_operational(data["champion_roots"]),
            "game_time": {k: distribution([r["timing"][k] for r in games]) for k in ("wall", "cpu")},
            "turns": distribution([g["turns"] for g in games]),
            "seats": {str(s): counts([r for r in roots if r["seat"] == s]) for s in (0, 1)},
            "phases": {str(s): counts([r for r in roots if (0 if r["turn"] < 20 else 1 if r["turn"] < 40 else 2) == s]) for s in range(3)},
            "privacy_strata": {str(s): counts([r for r in roots if bool(r["privacy"]["own"] or r["privacy"]["opponent"]) == s]) for s in (False, True)}}


def verify(stage):
    wall = time.perf_counter()
    binding = p.check_identity()
    schedule = {s: [independent_spec(s, i) for i in range(8 if s == "probe" else 400)] for s in ("probe", "main")}
    assert p.old.read_json(p.OUTPUT/"schedule.json") == schedule
    frozen = p.old.read_json(p.OUTPUT/"freeze.json")
    assert frozen["manifest"] == binding and frozen["schedule_sha256"] == p.old.file_hash(p.OUTPUT/"schedule.json")
    assert frozen["tests_sha256"] == p.verify_test_evidence()
    launch = p.old.read_json(p.OUTPUT/f"{stage}_launch.json")
    assert launch["manifest"] == binding and launch["schedule_sha256"] == frozen["schedule_sha256"]
    if stage == "probe":
        assert launch["upstream"] == {"freeze_sha256": p.old.file_hash(p.OUTPUT/"freeze.json")}
    else:
        assert launch["upstream"] == {"preflight_sha256": p.old.file_hash(p.OUTPUT/"preflight.json")}
        preflight = p.old.read_json(p.OUTPUT/"preflight.json")
        assert preflight["passed"] and preflight["manifest"] == binding
        assert preflight["audit_sha256"] == p.old.file_hash(p.OUTPUT/"probe_audit.json")
    lock_path = p.OUTPUT/f"{stage}_games_lock.json"
    lock = p.old.read_json(lock_path)
    n = len(schedule[stage])
    assert lock["manifest"] == binding and lock["launch_sha256"] == p.old.file_hash(p.OUTPUT/f"{stage}_launch.json")
    assert set(lock["files"]) == {str(i) for i in range(n)}
    assert set((p.OUTPUT/stage/"games").glob("*.json")) == {p.game_path(stage, i) for i in range(n)}
    audited = {}
    # Individual audit certificates allow explicit crash-resume without rerunning
    # already verified complete games; certificate hashes bind their entire tree.
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        pending = {}
        for i in range(n):
            certificate = p.OUTPUT/stage/"replay"/f"game_{i:04d}.json"
            if certificate.exists():
                saved = p.old.read_json(certificate)
                assert saved["manifest"] == binding and saved["game_sha256"] == lock["files"][str(i)]
                assert p.old.file_hash(p.game_path(stage, i)) == saved["game_sha256"]
                for row in p.old.read_json(p.game_path(stage, i))["trace"]:
                    if "root_sha256" in row:
                        assert p.old.file_hash(p.root_path(stage, i, row["turn"])) == row["root_sha256"]
                    else:
                        assert p.old.file_hash(p.champion_root_path(stage, i, row["turn"])) == row["champion_root_sha256"]
                audited[i] = saved["reconstruction"]
            else:
                pending[pool.submit(audit_game, stage, i, lock["files"][str(i)], binding)] = i
        for future in as_completed(pending):
            i = pending[future]
            audited[i] = future.result()
            p.old.save_new(p.OUTPUT/stage/"replay"/f"game_{i:04d}.json",
                           {"manifest": binding, "game_sha256": lock["files"][str(i)], "reconstruction": audited[i]})
            p.progress(f"{stage}_replay", games_complete=len(audited), games_total=n, elapsed_seconds=time.perf_counter()-wall)
    for i in range(0, n, 2):
        assert audited[i]["initial_sha256"] == audited[i+1]["initial_sha256"]
    data = {"games": [audited[i]["game"] for i in range(n)],
            "roots": [r for i in range(n) for r in audited[i]["roots"]],
            "champion_roots": [r for i in range(n) for r in audited[i]["champion_roots"]]}
    expected_roots = {p.root_path(stage, r["index"], r["turn"]) for r in data["roots"]}
    assert set((p.OUTPUT/stage/"roots").rglob("*.json")) == expected_roots
    assert set((p.OUTPUT/stage/"champion_roots").rglob("*.json")) == {p.champion_root_path(stage, r["index"], r["turn"]) for r in data["champion_roots"]}
    champion_counts = independent_champion_operational(data["champion_roots"])
    assert champion_counts["continuations"] == 192*champion_counts["searches"] == champion_counts["resolved"]
    assert champion_counts["hard_caps"] == 0 and champion_counts["loop_fraction"] <= .001
    output = {"passed": True, "manifest": binding, "games_lock_sha256": p.old.file_hash(lock_path),
              "games": n, "outer_moves": sum(g["turns"] for g in data["games"]), "roots": len(data["roots"]),
              "continuations": sum(r["continuations"] for r in data["roots"])+champion_counts["continuations"],
              "candidate_continuations": sum(r["continuations"] for r in data["roots"]),
              "champion_continuations": champion_counts["continuations"], "champion_roots": len(data["champion_roots"]),
              "continuation_moves": sum(r["moves"] for r in data["roots"])+champion_counts["moves"],
              "wall_seconds": time.perf_counter()-wall,
              "worker_cpu_seconds": sum(v["audit_timing"]["cpu"] for v in audited.values()),
              "shared_frozen_engine": True, "every_root_information_invariance": True,
              "replay_files": {str(i): p.old.file_hash(p.OUTPUT/stage/"replay"/f"game_{i:04d}.json") for i in range(n)}}
    if stage == "main":
        compare(data, p.old.read_json(p.OUTPUT/"observations.json"))
        result = p.old.read_json(p.OUTPUT/"result.json")
        stats = independent_statistics(data["games"])
        compare(stats, {k: result[k] for k in stats})
        compare(independent_operational(data), result["operational"])
        assert output["continuations"] <= 25600000
        output.update(result_sha256=p.old.file_hash(p.OUTPUT/"result.json"), observations_sha256=p.old.file_hash(p.OUTPUT/"observations.json"))
    p.check_identity()
    p.old.save_new(p.OUTPUT/f"{stage}_audit.json", output)
    p.progress(f"{stage}_audit_passed", games=n, roots=output["roots"], continuations=output["continuations"], wall_seconds=output["wall_seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("probe", "main"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify(args.stage)
