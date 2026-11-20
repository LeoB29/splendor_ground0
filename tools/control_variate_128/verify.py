"""Independent prefix traversal, all-endpoint143 arithmetic and timing receipts."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import random
import struct
import time

from splendor_ai.engine.constants import TOKEN_COLORS
from . import common as c

old = c.old


def _base(base):
    return Path(c.OUTPUT if base is None else base)


def _same(actual, expected):
    if isinstance(expected, dict):
        c.require(isinstance(actual, dict) and actual.keys() == expected.keys(), "Audit dictionary structure differs")
        for key in expected: _same(actual[key], expected[key])
    elif isinstance(expected, list):
        c.require(isinstance(actual, list) and len(actual) == len(expected), "Audit list structure differs")
        for a, b in zip(actual, expected, strict=True): _same(a, b)
    elif type(expected) is float:
        c.require(type(actual) in (int, float) and math.isfinite(actual)
                  and math.isclose(actual, expected, rel_tol=0., abs_tol=1e-12), "Audit numeric value differs")
    else:
        c.require(actual == expected, "Audit value differs")


def facts_from_state(state):
    players = []
    for index in range(2):
        p = state.players[index]
        players.append({"score": p.score, "bonuses": [p.bonuses[k] for k in TOKEN_COLORS],
            "tokens": [p.tokens[k] for k in TOKEN_COLORS]+[p.tokens["gold"]],
            "purchased": len(p.purchased_cards), "reserved": len(p.reserved_cards), "nobles": len(p.nobles)})
    return {"players": players, "current_player": state.current_player}


def independent_features(facts, actor):
    c.require(actor in (0, 1) and facts["current_player"] in (0, 1) and len(facts["players"]) == 2,
              "Invalid public endpoint players")
    for player in facts["players"]:
        c.require(set(player) == {"score", "bonuses", "tokens", "purchased", "reserved", "nobles"}
                  and len(player["bonuses"]) == 5 and len(player["tokens"]) == 6, "Invalid143 public facts")
        c.require(all(type(v) is int and v >= 0 for v in [player["score"], player["purchased"], player["reserved"],
                  player["nobles"], *player["tokens"], *player["bonuses"]]), "Non-public or invalid resource count")
    a, b = facts["players"][actor], facts["players"][actor ^ 1]
    differences = [(a["score"]-b["score"], 15)]
    differences.extend((a["bonuses"][j]-b["bonuses"][j], 7) for j in range(5))
    differences.extend((a["tokens"][j]-b["tokens"][j], 7) for j in range(5))
    differences.extend(((a["tokens"][5]-b["tokens"][5], 5), (a["purchased"]-b["purchased"], 20),
                        (a["reserved"]-b["reserved"], 3), (a["nobles"]-b["nobles"], 3)))
    return [struct.unpack("f", struct.pack("f", numerator/denominator))[0] for numerator, denominator in differences]


def verify_endpoint(point, actor, horizon, coefficients, saved=None):
    c.require(point["horizon"] == horizon and type(point["plies"]) is int and 0 <= point["plies"] <= horizon,
              "Invalid endpoint horizon/ply count")
    # Validate every public component even where exact terminal score overrides143.
    features = independent_features(point["public"], actor)
    if point["reason"] in ("completed", "forced_pass_deadlock"):
        c.require(point["winner"] in (None, 0, 1) and point["features"] is None and point["predicted_wdl"] is None,
                  "Terminal must bypass143")
        value = .5 if point["winner"] is None else float(point["winner"] == actor)
        if saved is not None:
            c.require(point["reason"] == saved["reason"] and point["winner"] == saved["winner"]
                      and point["plies"] == saved["turns"] and point["state_sha256"] == saved["final_state_sha256"]
                      and point["trace_sha256"] == saved["trace_sha256"] and value == (1+saved["wdl"])/2,
                      "Terminal prefix differs from historical outcome")
    else:
        c.require(point["reason"] == "horizon" and point["plies"] == horizon and point["winner"] is None,
                  "Invalid prefix censoring")
        _same(point["features"], features)
        prediction = min(1., max(-1., math.fsum(coefficients[j]*features[j] for j in range(15))))
        _same(point["predicted_wdl"], prediction)
        value = (1+prediction)/2
    _same(point["z"], value)
    return value


def replay_prefix(world, action, actor, horizons, coefficients):
    """Independent state machine and greedy score/first-maximum choice path."""
    env, current = old.SplendorEnv(), world
    visits, passes, idle = Counter(), 0, 0
    digest, points, count = hashlib.sha256(), {}, 0
    winner, reason = None, "horizon"
    while count < max(horizons):
        if current.terminal:
            reason, winner = "completed", current.winner
            break
        visits[old.state_signature(current)] += 1
        c.require(visits[old.state_signature(current)] < 4 and current.turn_index < 400, "Independent invalid prefix cutoff")
        legal = env.legal_actions(current)
        c.require(legal, "Independent stalled prefix")
        if count:
            scores = old._RAW.score_actions(current, legal)
            chosen = legal[min(range(len(legal)), key=lambda i: (-scores[i], i))]
        else:
            chosen = action
        c.require(chosen in legal, "Independent illegal candidate")
        digest.update(old.digest(old.serialize_action(chosen)).encode())
        nxt = env.step(current, chosen)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if old.is_progress_transition(current, nxt, chosen) else idle+1
        count += 1
        current = nxt
        if passes == 2:
            reason, winner = "forced_pass_deadlock", old.adjudicate_scores(current)
        else:
            c.require(idle < 60, "Independent no-progress prefix cutoff")
            if current.terminal:
                reason, winner = "completed", current.winner
        natural = reason in ("completed", "forced_pass_deadlock")
        requested = [h for h in horizons if h >= count] if natural else ([count] if count in horizons else [])
        for horizon in requested:
            facts = facts_from_state(current)
            features = None if natural else independent_features(facts, actor)
            prediction = None if natural else min(1., max(-1., math.fsum(x*y for x, y in zip(features, coefficients, strict=True))))
            z = (.5 if winner is None else float(winner == actor)) if natural else (1+prediction)/2
            points[str(horizon)] = {"horizon": horizon, "plies": count, "reason": reason if natural else "horizon",
                "winner": winner if natural else None, "public": facts, "features": features, "predicted_wdl": prediction,
                "z": z, "state_sha256": old.digest(old.serialize_state_snapshot(current)), "trace_sha256": digest.hexdigest()}
        if natural: break
    if count == 0 and current.terminal:
        for horizon in horizons:
            points[str(horizon)] = {"horizon": horizon, "plies": 0, "reason": "completed", "winner": current.winner,
                "public": facts_from_state(current), "features": None, "predicted_wdl": None,
                "z": .5 if current.winner is None else float(current.winner == actor),
                "state_sha256": old.digest(old.serialize_state_snapshot(current)), "trace_sha256": digest.hexdigest()}
    c.require(set(points) == set(map(str, horizons)), "Independent missing horizons")
    return points


def _load(spec, binding, base, stage):
    from . import source
    root, ref = source.load_root(spec, binding, base, stage=stage)
    evaluator = source.evaluator(base)
    return root, ref, evaluator, c.verified(evaluator)["coefficients"]


def audit(spec, ref, binding, base=None, stage="train"):
    from . import engine, source
    base = _base(base)
    c.budget_boundary(base)
    started = time.perf_counter()
    record = c.verified(ref)
    root, source_ref, evaluator, coefficients = _load(spec, binding, base, stage)
    horizons = engine._horizons(stage, binding, base)
    c.require(record["manifest"] == binding and record["stage"] == stage and record["spec"] == spec
              and record["source"] == source_ref and record["evaluator"] == evaluator and record["complete"]
              and record["horizons"] == list(horizons) and record["new_terminal_simulations"] == 0
              and record["forced_root_counts_as_ply"], "Changed prefix identity")
    expected_identity = {key: root["spec"][key] for key in ("index", "pair", "fit", "split", "game_index", "phase", "seat")}
    c.require(all(record[key] == value for key, value in expected_identity.items()), "Changed statistical root identity")
    state = old.deserialize_state_snapshot(root["snapshot"])
    actions = [old.decode_action(payload) for payload in root["candidates"]["actions"]]
    c.require(record["support"] == len(actions) and record["reference"] == root["old_teacher_choice"], "Changed candidate/reference mapping")
    worlds = root["assessment"]["worlds"]
    c.require(len(record["worlds"]) == len(worlds) == 128, "Changed world population")
    selected = source.audit_indices(spec)
    c.require(len(set(selected)) == len(selected) and all(0 <= i < 128 for i in selected), "Invalid prefix replay sample")
    z = {str(h): [] for h in horizons}
    cells, endpoints = 0, 0
    for wi, (saved, produced) in enumerate(zip(worlds, record["worlds"], strict=True)):
        c.budget_boundary(base)
        c.require(produced["index"] == wi and produced["seed"] == saved["seed"]
                  and produced["state_sha256"] == saved["state_sha256"] and len(produced["candidates"]) == len(actions),
                  "Changed coupled world identity")
        if wi in selected:
            world = old.determinize_for_observer(state, state.current_player, random.Random(saved["seed"]))
            c.require(old.digest(old.serialize_state_snapshot(world)) == saved["state_sha256"], "Independent world reconstruction differs")
        row = {str(h): [] for h in horizons}
        for ci, candidate in enumerate(produced["candidates"]):
            c.require(set(candidate) == set(z), "Wrong endpoint horizon set")
            for h in horizons:
                row[str(h)].append(verify_endpoint(candidate[str(h)], state.current_player, h, coefficients, saved["outcomes"][ci]))
                endpoints += 1
            if wi in selected:
                _same(candidate, replay_prefix(world, actions[ci], state.current_player, horizons, coefficients))
                cells += 1
        for key in z: z[key].append(row[key])
    y = [[.5+.5*value["wdl"] for value in w["outcomes"]] for w in worlds]
    _same(record["y"], y); _same(record["z"], z)
    _same(record["statistics"], {**expected_identity, "support": len(actions), "reference": root["old_teacher_choice"], "y": y, "z": z})
    result = {"manifest": binding, "stage": stage, "index": spec["index"], "result": ref, "passed": True,
        "all_endpoint_features_verified": True, "all_endpoint_values_verified": True,
        "sampled_prefixes_independently_replayed": True, "source_world_coupling_verified": True,
        "statistics_projection_verified": True, "audited_world_indices": selected, "replayed_candidates": cells,
        "checked_endpoints": endpoints, "full_prefix_replay": len(selected) == 128,
        "new_terminal_simulations": 0, "elapsed_seconds": time.perf_counter()-started}
    path = base/stage/"audits"/f"state_{spec['index']:04d}.json"
    old.save_new(path, result)
    return c.artifact(path, index=spec["index"])


def audit_timing(spec, ref, binding, base=None):
    from . import source
    base = _base(base)
    c.budget_boundary(base)
    record = c.verified(ref)
    root, source_ref, evaluator, coefficients = _load(spec, binding, base, "train")
    c.require(record["manifest"] == binding and record["spec"] == spec and record["source"] == source_ref
              and record["evaluator"] == evaluator and record["complete"] and record["matched_world_candidate_costs"]
              and record["full_outcomes_equal_saved"] and record["backend"] == "serial" and record["torch_threads"] == 1,
              "Changed timing identity")
    c.require(all(record[key] == root["spec"][key] for key in ("index", "pair", "fit", "split", "game_index", "phase", "seat"))
              and record["split"] == "train", "Changed timing strata")
    state = old.deserialize_state_snapshot(root["snapshot"])
    actions = [old.decode_action(payload) for payload in root["candidates"]["actions"]]
    indices = source.timing_world_indices(spec)
    expected_cells = [(wi, ci) for wi in indices for ci in range(len(actions))]
    c.require(record["world_indices"] == indices and [(r["world"], r["candidate"]) for r in record["cells"]] == expected_cells,
              "Changed timing sample")
    labels = [("full_only", None)]+[("full_plus", h) for h in c.HORIZONS]+[("prefix_only", h) for h in c.HORIZONS]
    totals = {label: {"cpu": 0., "wall": 0., "cells": 0} for label in labels}
    for cell_index, cell in enumerate(record["cells"]):
        c.budget_boundary(base)
        wi, ci = cell["world"], cell["candidate"]
        saved = root["assessment"]["worlds"][wi]
        world = old.determinize_for_observer(state, state.current_player, random.Random(saved["seed"]))
        c.require(old.digest(old.serialize_state_snapshot(world)) == saved["state_sha256"], "Timing world differs")
        independent = replay_prefix(world, actions[ci], state.current_player, c.HORIZONS, coefficients)
        offset = (spec["index"]+cell_index) % 7
        c.require([(r["arm"], r["horizon"]) for r in cell["samples"]] == labels[offset:]+labels[:offset], "Timing order is not fixed balanced rotation")
        for sample in cell["samples"]:
            key = sample["arm"], sample["horizon"]
            for unit in ("cpu", "wall"):
                c.require(type(sample[unit]) in (int, float) and math.isfinite(sample[unit]) and sample[unit] >= 0,
                          "Invalid timing measurement")
                totals[key][unit] += sample[unit]
            totals[key]["cells"] += 1
            cost = sample["result"]["endpoint_timing"]
            c.require(cost["points"] == len(sample["result"]["endpoints"])
                      and all(type(cost[k]) in (int, float) and math.isfinite(cost[k]) and cost[k] >= 0 for k in ("cpu", "wall")),
                      "Invalid endpoint overhead timing")
            if sample["arm"] != "prefix_only":
                _same(sample["result"]["outcome"], saved["outcomes"][ci])
            h = sample["horizon"]
            expected = {} if h is None else {str(h): independent[str(h)]}
            _same(sample["result"]["endpoints"], expected)
            if h is not None:
                verify_endpoint(sample["result"]["endpoints"][str(h)], state.current_player, h, coefficients, saved["outcomes"][ci])
    expected = {"full_only": totals[("full_only", None)],
        "full_plus": {str(h): totals[("full_plus", h)] for h in c.HORIZONS},
        "prefix_only": {str(h): totals[("prefix_only", h)] for h in c.HORIZONS}}
    _same(record["timing"], expected)
    _same(record["statistics"], {**{key: root["spec"][key] for key in ("index", "pair", "fit", "split", "game_index", "phase", "seat")},
                                  "support": len(actions), "timing": expected})
    c.require(record["full_terminal_replays"] == 4*len(expected_cells), "Changed bounded full replay count")
    result = {"manifest": binding, "index": spec["index"], "result": ref, "passed": True,
        "all_endpoint_features_verified": True, "all_endpoint_values_verified": True,
        "sampled_prefixes_independently_replayed": True, "source_world_coupling_verified": True,
        "all_full_outcomes_equal_saved": True, "matched_timing_arms_verified": True,
        "replayed_candidates": len(expected_cells), "new_terminal_simulations": 0}
    path = base/"timing"/"audits"/f"state_{spec['index']:04d}.json"
    old.save_new(path, result)
    return c.artifact(path, index=spec["index"])
