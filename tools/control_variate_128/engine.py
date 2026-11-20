"""Exact greedy035 prefixes and matched timing for a saved-outcome feasibility test."""
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch

from splendor_ai.engine.constants import TOKEN_COLORS
from . import common as c

old = c.old
_COEFFICIENTS = None
_EVALUATOR = None


def _base(base):
    return Path(c.OUTPUT if base is None else base)


def worker_initialize(base=None):
    """One immutable policy and one small evaluator per worker, not per root."""
    global _COEFFICIENTS, _EVALUATOR
    from . import source
    base = _base(base)
    old.initialize_worker()
    reference = source.evaluator(base)
    data = c.verified(reference)
    coefficients = data["coefficients"]
    c.require(len(coefficients) == 15 and all(math.isfinite(v) for v in coefficients), "Invalid frozen143 evaluator")
    _COEFFICIENTS = torch.tensor(coefficients, dtype=torch.float64)
    _EVALUATOR = reference


def public_facts(state):
    """Only public counts/resources needed by143; never private card identities."""
    return {"current_player": state.current_player, "players": [{
        "score": player.score, "bonuses": [player.bonuses[color] for color in TOKEN_COLORS],
        "tokens": [player.tokens[color] for color in (*TOKEN_COLORS, "gold")],
        "purchased": len(player.purchased_cards), "reserved": len(player.reserved_cards),
        "nobles": len(player.nobles)} for player in state.players]}


def public_features(facts, actor):
    c.require(actor in (0, 1), "Invalid root actor")
    own, other = facts["players"][actor], facts["players"][1-actor]
    values = [(own["score"]-other["score"])/15]
    values += [(a-b)/7 for a, b in zip(own["bonuses"], other["bonuses"], strict=True)]
    values += [(own["tokens"][i]-other["tokens"][i])/7 for i in range(5)]
    values += [(own["tokens"][5]-other["tokens"][5])/5,
               (own["purchased"]-other["purchased"])/20,
               (own["reserved"]-other["reserved"])/3,
               (own["nobles"]-other["nobles"])/3]
    c.require(len(values) == 15, "Wrong143 feature shape")
    return torch.tensor(values, dtype=torch.float32)


def endpoint(state, actor, horizon, plies, reason, winner, trace_hash):
    facts = public_facts(state)
    natural = reason in old.VALID_NATURAL_TERMINATIONS
    c.require(natural or reason == "horizon", "Invalid prefix cutoff")
    if natural:
        features, prediction = None, None
        z = (1+old.outcome_value(winner, actor))/2
    else:
        c.require(_COEFFICIENTS is not None, "Initialize143 evaluator before prefixes")
        features = public_features(facts, actor)
        prediction = float((features.double() @ _COEFFICIENTS).clamp(-1, 1))
        z = (1+prediction)/2
        features = features.tolist()
    return {"horizon": horizon, "plies": plies, "reason": reason, "winner": winner,
        "state_sha256": old.digest(old.serialize_state_snapshot(state)), "trace_sha256": trace_hash,
        "public": facts, "features": features, "predicted_wdl": prediction, "z": z}


def traverse(world, action, actor, horizons, *, full=False):
    """Forced root is ply1. Apply the unchanged149 termination order exactly."""
    c.require(actor in (0, 1) and all(type(h) is int and h > 0 for h in horizons), "Invalid prefix contract")
    c.require(len(set(horizons)) == len(horizons) and (full or horizons), "Duplicate or empty horizon")
    env, state = old.SplendorEnv(), world
    visits, forced, idle, turns = {}, 0, 0, 0
    hasher, points = hashlib.sha256(), {}
    endpoint_cost = {"cpu": 0., "wall": 0., "points": 0}
    def capture(horizon, reason, winner):
        started, cpu = time.perf_counter(), time.process_time()
        points[str(horizon)] = endpoint(state, actor, horizon, turns, reason, winner, hasher.hexdigest())
        endpoint_cost["wall"] += time.perf_counter()-started
        endpoint_cost["cpu"] += time.process_time()-cpu
        endpoint_cost["points"] += 1
    reason, winner = "completed", None
    limit = max(horizons) if horizons else None
    while not state.terminal:
        signature = old.state_signature(state)
        visits[signature] = visits.get(signature, 0)+1
        if visits[signature] >= 4:
            reason = "repetition_cutoff"; break
        if state.turn_index >= 400:
            reason = "max_turns"; break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"; break
        selected = action if turns == 0 else old._RAW.choose_action(env, state, legal)
        c.require(selected in legal, "Illegal continuation action")
        hasher.update(old.digest(old.serialize_action(selected)).encode())
        nxt = env.step(state, selected)
        turns += 1
        forced = forced+1 if old.is_forced_pass_turn(legal) else 0
        idle = 0 if old.is_progress_transition(state, nxt, selected) else idle+1
        state = nxt
        if forced >= 2:
            reason, winner = "forced_pass_deadlock", old.adjudicate_scores(state)
            break
        if idle >= 60:
            reason = "no_progress_cutoff"; break
        if state.terminal:
            break
        if turns in horizons:
            capture(turns, "horizon", None)
        if not full and turns == limit:
            reason = "horizon"; break
    if reason == "completed":
        winner = state.winner
    c.require(reason in old.VALID_NATURAL_TERMINATIONS or reason == "horizon", "Non-natural prefix/full termination: "+reason)
    if reason in old.VALID_NATURAL_TERMINATIONS:
        for horizon in horizons:
            if str(horizon) not in points:
                capture(horizon, reason, winner)
    c.require(set(points) == set(map(str, horizons)), "Incomplete endpoint accounting")
    outcome = {"reason": reason, "winner": winner, "turns": turns,
        "trace_sha256": hasher.hexdigest(), "final_state_sha256": old.digest(old.serialize_state_snapshot(state))}
    if reason in old.VALID_NATURAL_TERMINATIONS:
        outcome["wdl"] = old.outcome_value(winner, actor)
    return {"endpoints": points, "outcome": outcome, "endpoint_timing": endpoint_cost}


def _horizons(stage, binding, base):
    c.require(stage in ("train", "validation", "test"), "Invalid prefix stage")
    if stage == "test":
        c.require(not base.resolve().is_relative_to(c.OUTPUT.resolve()), "Fixture cannot target production")
    if stage != "validation":
        return tuple(c.HORIZONS)
    selection = old.read_json(base/"selection_lock.json")
    audit = old.read_json(base/"selection_audit.json")
    c.require(selection["manifest"] == audit["manifest"] == binding and audit["passed"]
              and selection["status"] == "selected" and selection["selected_before_validation"]
              and not selection["validation_read"] and selection["config"]["horizon"] in c.HORIZONS,
              "Validation requires sealed TRAIN-selected horizon")
    c.require(c.verified(audit["selection"]) == selection, "Selected horizon audit linkage differs")
    return (selection["config"]["horizon"],)


def _source(spec, binding, base, stage):
    from . import source
    return source.load_root(spec, binding, base, stage=stage)


def _world(state, saved):
    world = old.determinize_for_observer(state, state.current_player, random.Random(saved["seed"]))
    c.require(old.digest(old.serialize_state_snapshot(world)) == saved["state_sha256"], "Saved world reconstruction changed")
    return world


def _identity(root):
    spec = root["spec"]
    return {key: spec[key] for key in ("index", "pair", "fit", "split", "game_index", "phase", "seat")}


def _check_saved_outcome(outcome, saved):
    c.require(outcome == saved and saved["reason"] in old.VALID_NATURAL_TERMINATIONS, "Full continuation differs from saved teacher world")


def statistics_row(record):
    return {key: record[key] for key in ("index", "pair", "fit", "split", "game_index", "phase", "seat",
                                       "support", "reference", "y", "z")}


def collect(spec, binding, base=None, stage="train"):
    base = _base(base)
    c.budget_boundary(base)
    started, cpu = time.perf_counter(), time.process_time()
    horizons = _horizons(stage, binding, base)
    root, reference = _source(spec, binding, base, stage)
    state = old.deserialize_state_snapshot(root["snapshot"])
    actions = [old.decode_action(item) for item in root["candidates"]["actions"]]
    worlds, results = root["assessment"]["worlds"], []
    for wi, saved in enumerate(worlds):
        c.budget_boundary(base)
        world = _world(state, saved)
        candidates = []
        for ci, action in enumerate(actions):
            prefix = traverse(world, action, state.current_player, horizons)
            for point in prefix["endpoints"].values():
                if point["reason"] in old.VALID_NATURAL_TERMINATIONS:
                    expected = saved["outcomes"][ci]
                    c.require(point["plies"] == expected["turns"] and point["state_sha256"] == expected["final_state_sha256"]
                              and point["trace_sha256"] == expected["trace_sha256"]
                              and point["z"] == (1+expected["wdl"])/2, "Terminal prefix differs from saved outcome")
            candidates.append(prefix["endpoints"])
        results.append({"index": wi, "seed": saved["seed"], "state_sha256": saved["state_sha256"], "candidates": candidates})
    record = {"manifest": binding, "stage": stage, "spec": spec, "source": reference, "evaluator": _EVALUATOR,
        **_identity(root), "support": len(actions), "reference": root["old_teacher_choice"], "horizons": list(horizons),
        "y": [[(1+item["wdl"])/2 for item in saved["outcomes"]] for saved in worlds],
        "z": {str(h): [[item[str(h)]["z"] for item in saved["candidates"]] for saved in results] for h in horizons},
        "worlds": results, "forced_root_counts_as_ply": True, "new_terminal_simulations": 0,
        "timing": {"wall": time.perf_counter()-started, "cpu": time.process_time()-cpu}, "complete": True}
    record["statistics"] = statistics_row(record)
    path = base/stage/"roots"/f"state_{spec['index']:04d}.json"
    old.save_new(path, record)
    return c.artifact(path, index=spec["index"])


def timing(spec, binding, base=None):
    """Seven cost arms, deterministic balanced order, exactly matched states."""
    from . import source
    base = _base(base)
    c.budget_boundary(base)
    began_all = time.perf_counter()
    root, reference = _source(spec, binding, base, "train")
    c.require(root["spec"]["split"] == "train", "Timing may use TRAIN only")
    state = old.deserialize_state_snapshot(root["snapshot"])
    actions = [old.decode_action(item) for item in root["candidates"]["actions"]]
    indices = source.timing_world_indices(spec)
    labels = [("full_only", None)]+[("full_plus", h) for h in c.HORIZONS]+[("prefix_only", h) for h in c.HORIZONS]
    totals = {"full_only": {"cpu": 0., "wall": 0., "cells": 0},
              "full_plus": {str(h): {"cpu": 0., "wall": 0., "cells": 0} for h in c.HORIZONS},
              "prefix_only": {str(h): {"cpu": 0., "wall": 0., "cells": 0} for h in c.HORIZONS}}
    cells = []
    for position, wi in enumerate(indices):
        saved = root["assessment"]["worlds"][wi]
        for ci, action in enumerate(actions):
            c.budget_boundary(base)
            shift = (spec["index"]+position*len(actions)+ci) % len(labels)
            order = labels[shift:]+labels[:shift]
            samples = []
            for name, h in order:
                began, cpu = time.perf_counter(), time.process_time()
                world = _world(state, saved)
                value = traverse(world, action, state.current_player, () if h is None else (h,), full=name != "prefix_only")
                cost = {"wall": time.perf_counter()-began, "cpu": time.process_time()-cpu}
                if name != "prefix_only":
                    _check_saved_outcome(value["outcome"], saved["outcomes"][ci])
                target = totals[name] if h is None else totals[name][str(h)]
                for key in ("wall", "cpu"): target[key] += cost[key]
                target["cells"] += 1
                samples.append({"arm": name, "horizon": h, **cost, "result": value})
            for horizon in c.HORIZONS:
                points = [sample["result"]["endpoints"][str(horizon)] for sample in samples
                          if sample["horizon"] == horizon]
                c.require(points[0] == points[1], "Integrated and standalone prefixes differ")
            cells.append({"world": wi, "candidate": ci, "samples": samples})
    record = {"manifest": binding, "spec": spec, "source": reference, "evaluator": _EVALUATOR,
        **_identity(root), "support": len(actions),
        "world_indices": indices, "cells": cells, "timing": totals, "backend": "serial", "torch_threads": 1,
        "matched_world_candidate_costs": True, "full_outcomes_equal_saved": True,
        "full_terminal_replays": 4*len(cells),
        "elapsed_seconds": time.perf_counter()-began_all, "complete": True}
    record["statistics"] = {**_identity(root), "support": len(actions), "timing": totals}
    path = base/"timing"/f"state_{spec['index']:04d}.json"
    old.save_new(path, record)
    return c.artifact(path, index=spec["index"])


def execution_forecast(timing_refs, binding, base=None):
    """Outcome-blind conservative cost/storage screen from actual prefix timings."""
    base = _base(base)
    records = [c.verified(ref) for ref in timing_refs]
    c.require(len(records) == 18 and len({r["index"] for r in records}) == 18
              and all(r["manifest"] == binding and r["split"] == "train" and r["complete"] for r in records),
              "Complete TRAIN timing panel required")
    c.require({(r["fit"], r["seat"], r["phase"]) for r in records}
              == {(fit, seat, phase) for fit in c.FITS for seat in (0, 1) for phase in range(3)},
              "Unbalanced timing panel")
    per_cell, endpoint_costs, point_bytes = [], [], []
    for record in records:
        measure = record["timing"]["prefix_only"]["16"]
        c.require(measure["cells"] > 0, "Empty timing measurement")
        per_cell.append(max(measure["cpu"], measure["wall"])/measure["cells"])
        point_cpu = point_wall = 0.
        count = 0
        for cell in record["cells"]:
            for sample in cell["samples"]:
                if sample["horizon"] is not None:
                    cost = sample["result"]["endpoint_timing"]
                    point_cpu += cost["cpu"]; point_wall += cost["wall"]; count += cost["points"]
                    point_bytes.extend(len(json.dumps(point, indent=2).encode()) for point in sample["result"]["endpoints"].values())
        c.require(count > 0, "Missing timed endpoint overhead")
        endpoint_costs.append(max(point_cpu, point_wall)/count)
    prefix_cost, extra_point_cost = max(per_cell), max(endpoint_costs)
    support, worlds = 8, 128
    main_cells = 336*worlds*support
    # The frozen39-cell stress recipe selects at most39 full roots per split.
    audit_worlds = 336*8+2*39*(128-8)
    audit_cells = audit_worlds*support
    # TRAIN has three recorded endpoints, validation only the selected one.
    main_points = 168*worlds*support*4
    audit_extra_points = (168*8+39*(128-8))*support*2
    prefix_work = prefix_cost*(main_cells+audit_cells)
    # Measured prefix16 includes one endpoint. Account two more forTRAIN and
    # all-endpoint independent arithmetic; conservative extra work uses full
    # measured extraction/hash/evaluation cost even though audit needs no hash.
    endpoint_work = extra_point_cost*(168*worlds*support*2+audit_extra_points+main_points)
    elapsed = time.time()-old.read_json(base/"launch.json")["epoch_seconds"] if (base/"launch.json").exists() else 0.
    estimate = elapsed+c.CONFIG["forecast_allowance"]*(prefix_work+endpoint_work)/c.CONFIG["workers"]+c.CONFIG["finalization_allowance_seconds"]
    disk_estimate = math.ceil((main_points*max(point_bytes)+main_cells*350)*1.5)
    free = shutil.disk_usage(c.ROOT).free
    return {"manifest": binding, "outcome_blind": True, "estimated_total_seconds": estimate,
        "within_runtime_ceiling": estimate <= c.CONFIG["maximum_runtime_hours"]*3600,
        "disk_reserve_pass": free-disk_estimate >= c.CONFIG["disk_reserve_bytes"],
        "elapsed_seconds": elapsed, "workers": c.CONFIG["workers"], "maximum_horizon": 16,
        "main_candidate_prefixes": main_cells, "maximum_audited_candidate_prefixes": audit_cells,
        "maximum_stress_roots": 78, "all_endpoint_checks": main_points,
        "slowest_timing_root_prefix_seconds_per_cell": prefix_cost,
        "slowest_timing_root_endpoint_seconds_per_point": extra_point_cost,
        "forecast_allowance": c.CONFIG["forecast_allowance"], "estimated_remaining_bytes": disk_estimate,
        "free_disk_bytes": free, "timing": timing_refs}
