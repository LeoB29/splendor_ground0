"""154: separate-process source, world, trajectory, choice and bootstrap audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import math
import random
import time

import numpy as np
import torch

from splendor_ai.alphazero import run_rollout_selector as p
from splendor_ai.engine.data import build_base_deck_by_tier
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

spec = importlib.util.spec_from_file_location("audit149_for154", "scripts/analysis/verify_afterstate_advantage_149.py")
a149 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a149)


def compare(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            compare(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif isinstance(a, float):
        assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12, (a, b)
    else:
        assert a == b, (a, b)


def independent_world(state, seed):
    """Reconstruct unknown pools from catalog minus legally known identities."""
    world = state.copy_shallow()
    observer = state.current_player
    known = {c.card_id for cards in state.visible_tier_cards.values() for c in cards}
    for seat, player in enumerate(state.players):
        known.update(c.card_id for c in player.purchased_cards)
        known.update(c.card_id for i, c in enumerate(player.reserved_cards)
                     if seat == observer or player.reserved_card_source(i) == RESERVE_SOURCE_VISIBLE)
    rng = random.Random(seed)
    for tier, catalog in sorted(build_base_deck_by_tier().items()):
        unknown = sorted((c for c in catalog if c.card_id not in known), key=lambda c: c.card_id)
        other = world.players[1-observer]
        private = [i for i, c in enumerate(other.reserved_cards) if c.tier == tier and other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE]
        assert len(unknown) == state.deck_counts[tier]+len(private)
        rng.shuffle(unknown)
        for i in private:
            other.reserved_cards[i] = unknown.pop()
        world.hidden_tier_decks[tier] = unknown
        world.deck_counts[tier] = len(unknown)
    return world


def independent_candidates(state, split, source_index):
    legal = p.old.SplendorEnv().legal_actions(state)
    values = p.old._RAW.score_actions(state, legal)
    assert all(math.isfinite(float(v)) for v in values)
    groups = p.old.actions_by_canonical_successor_key(state, legal)
    keys, actions = [], []
    for j in sorted(range(len(legal)), key=lambda j: (-values[j], j)):
        key = p.old.canonical_successor_action_key(state, legal[j])
        if key not in keys:
            keys.append(key)
            actions.append(legal[j])
        if len(keys) == min(2, len(groups)):
            break
    remaining = [k for k in groups if k not in keys]
    rng = random.Random(f"149-candidates-v1|{split}|{source_index}")
    for key in rng.sample(remaining, min(2, len(remaining))):
        keys.append(key)
        actions.append(groups[key])
    return {"actions": [p.old.serialize_action(a) for a in actions], "keys": [repr(k) for k in keys],
            "canonical_support": len(groups), "legal_support": len(legal)}


def source_check(index, info, probe=False):
    path = p.source_path(index, probe)
    if not probe:
        assert path.stat().st_mtime_ns > (p.OUTPUT/"launch.json").stat().st_mtime_ns
    assert p.old.file_hash(path) == info["sha256"]
    record = p.old.read_json(path)
    assert record["index"] == index and record["seed"] == (1549000000 if probe else 1540000000)+index
    assert record["reason"] == info["reason"] and record["turns"] == info["turns"]
    assert record["timing"] == info["timing"]
    env = p.old.SplendorEnv(seed=record["seed"])
    state = env.initial_state()
    seen, hasher, passes, idle = Counter(), hashlib.sha256(), 0, 0
    eligible, best = [0]*6, {}
    for i, row in enumerate(record["trace"]):
        assert not state.terminal and state.turn_index < 400
        seen[p.old.state_signature(state)] += 1
        assert seen[p.old.state_signature(state)] < 4
        snapshot = p.old.serialize_state_snapshot(state)
        assert row["state_hash"] == p.old.digest(snapshot) and row["turn"] == state.turn_index and row["actor"] == state.current_player
        if all(player.reserved_card_source(j) == RESERVE_SOURCE_VISIBLE for player in state.players for j in range(len(player.reserved_cards))):
            phase = 0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2
            c = phase*2+state.current_player
            eligible[c] += 1
            key = p.old.digest(["154-position-v1", index, state.turn_index])
            if c not in best or key < best[c]["key"]:
                best[c] = {"key": key, "turn": state.turn_index, "snapshot_sha256": p.old.digest(snapshot)}
        legal = env.legal_actions(state)
        scores = p.old._RAW.score_actions(state, legal)
        chosen = legal[max(range(len(legal)), key=lambda j: scores[j])]
        assert p.old.serialize_action(chosen) == row["action"]
        hasher.update(p.old.digest(row["action"]).encode())
        successor = env.step(state, chosen)
        passes = passes+1 if len(legal) == 1 and legal[0].action_type.name == "PASS" else 0
        idle = 0 if p.old.is_progress_transition(state, successor, chosen) else idle+1
        state = successor
        if i < len(record["trace"])-1:
            assert passes < 2 and idle < 60
    assert len(record["trace"]) == record["turns"] and hasher.hexdigest() == record["trace_sha256"]
    assert p.old.digest(p.old.serialize_state_snapshot(state)) == p.old.digest(record["final_state"]) == record["final_state_sha256"]
    if record["reason"] == "completed":
        assert state.terminal and idle < 60
        winner = state.winner
    else:
        assert record["reason"] == "forced_pass_deadlock" and passes == 2
        winner = p.old.adjudicate_scores(state)
    assert winner == record["winner"]
    return {"index": index, "moves": record["turns"], "eligible": eligible, "best": best, "sha256": info["sha256"]}


def independent_matching(sources, quota):
    buckets, choices, prefix = [[] for _ in range(6)], {}, []
    def assign(game, touched):
        for cell in choices[game]:
            if cell in touched:
                continue
            touched.add(cell)
            if len(buckets[cell]) < quota:
                buckets[cell].append(game)
                return True
            for i, previous in enumerate(buckets[cell]):
                if assign(previous, touched):
                    buckets[cell][i] = game
                    return True
        return False
    for row in sources:
        i = row["index"]
        assert i == len(prefix)
        choices[i] = [c for c in range(6) if row["eligible"][c]]
        prefix.append({"source_index": i, "eligible_by_cell": row["eligible"], "sha256": row["sha256"]})
        assign(i, set())
        if all(len(b) == quota for b in buckets):
            break
    assert all(len(b) == quota for b in buckets)
    return {i: c for c, b in enumerate(buckets) for i in b}, prefix


def panel_integrity():
    lock = p.old.read_json(p.OUTPUT/"panel_lock.json")
    assert lock["manifest"] == p.check_identity()
    assert lock["panel_sha256"] == p.old.file_hash(p.OUTPUT/"panel.json")
    assert lock["sources_sha256"] == p.old.file_hash(p.OUTPUT/"sources_lock.json")
    sources = p.old.read_json(p.OUTPUT/"sources_lock.json")
    assert set(sources["files"]) == {str(i) for i in range(len(sources["files"]))}
    assert len(sources["files"]) <= 4096 and 0 <= sources["unused_prefetched"] < 8
    assert sources["selected_prefix"]+sources["unused_prefetched"] == len(sources["files"])
    for i, info in sources["files"].items():
        assert p.old.file_hash(p.source_path(int(i))) == info["sha256"]
    return p.old.read_json(p.OUTPUT/"panel.json"), sources


def verify_panel():
    panel, sources = panel_integrity()
    assert not (p.OUTPUT/"targets/discovery").exists()
    checked = []
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        jobs = [pool.submit(source_check, int(i), info) for i, info in sources["files"].items()]
        for i, job in enumerate(as_completed(jobs), 1):
            checked.append(job.result())
            if i % 128 == 0 or i == len(jobs):
                p.progress("audit_sources", games=i, total=len(jobs))
    checked.sort(key=lambda row: row["index"])
    assignments, filtering = independent_matching(checked, 128)
    assert filtering == panel["filtering"] and len(filtering) == sources["selected_prefix"]
    assert panel["counts"] == [128]*6 and len(panel["samples"]) == len(assignments) == 768
    assert len({s["source_seed"] for s in panel["samples"]}) == 768
    p.old.initialize_worker()
    for i, (row, (source_index, cell)) in enumerate(zip(panel["samples"], sorted(assignments.items()))):
        assert row["index"] == i and row["source_index"] == source_index
        assert row["cell"] == cell and row["source_seed"] == 1540000000+source_index
        best = checked[source_index]["best"][cell]
        assert row["turn"] == best["turn"] and p.old.digest(row["snapshot"]) == best["snapshot_sha256"]
        state = p.old.deserialize_state_snapshot(row["snapshot"])
        assert row["actor"] == state.current_player and p.old.eligible(state, cell)
        expected = independent_candidates(state, "154-main", source_index)
        assert all(row[k] == v for k, v in expected.items())
        world = independent_world(state, 1549300000+row["index"])
        assert torch.equal(torch.as_tensor(p.old._RAW.score_actions(state, p.old.SplendorEnv().legal_actions(state))),
                           torch.as_tensor(p.old._RAW.score_actions(world, p.old.SplendorEnv().legal_actions(world))))
    return {"source_games": len(checked), "source_moves": sum(v["moves"] for v in checked), "states": 768,
            "panel_lock_sha256": p.old.file_hash(p.OUTPUT/"panel_lock.json")}


def target_check(stage, sample, choice, upstream, info):
    started = time.process_time()
    path = p.OUTPUT/"targets"/stage/f"state_{sample['index']:04d}.json"
    assert p.old.file_hash(path) == info["sha256"]
    record = p.old.read_json(path)
    probe = stage.startswith("probe_")
    discovery = stage.endswith("discovery")
    count = 4 if probe else 32 if discovery else 128
    seed_start = {"discovery": 1541000000, "evaluation": 1542000000,
                  "probe_discovery": 1549100000, "probe_evaluation": 1549200000}[stage]
    expected = list(range(len(sample["actions"]))) if discovery else [0] if choice == 0 else [0, choice]
    assert record["stage"] == stage and record["index"] == sample["index"]
    assert record["sample_sha256"] == p.old.digest(sample) and record["upstream_sha256"] == upstream
    assert record["action_indices"] == expected and len(record["worlds"]) == count
    source = p.old.deserialize_state_snapshot(sample["snapshot"])
    moves, reasons = 0, Counter()
    for i, saved in enumerate(record["worlds"]):
        seed = seed_start+1000*sample["index"]+i
        assert saved["seed"] == seed
        world = independent_world(source, seed)
        assert saved["world_sha256"] == p.old.digest(p.old.serialize_state_snapshot(world))
        assert len(saved["outcomes"]) == len(expected)
        for j, action_index in enumerate(expected):
            rebuilt = a149.replay_branch(world, p.old.decode_action(sample["actions"][action_index]), source.current_player)
            assert rebuilt == saved["outcomes"][j], (stage, sample["index"], i, action_index)
            assert rebuilt["reason"] in ("completed", "forced_pass_deadlock")
            moves += rebuilt["turns"]
            reasons[rebuilt["reason"]] += 1
    assert info["natural"] and info["continuations"] == count*len(expected) and info["moves"] == moves
    assert info["reasons"] == dict(reasons) and info["timing"] == record["timing"]
    if not probe:
        predecessor = "discovery_launch.json" if discovery else "evaluation_launch.json"
        assert path.stat().st_mtime_ns > (p.OUTPUT/predecessor).stat().st_mtime_ns
        if not discovery:
            assert path.stat().st_mtime_ns > (p.OUTPUT/"selection_lock.json").stat().st_mtime_ns
        else:
            assert path.stat().st_mtime_ns < (p.OUTPUT/"selection_lock.json").stat().st_mtime_ns
    elif not discovery:
        assert path.stat().st_mtime_ns > (p.OUTPUT/"probe_selection_lock.json").stat().st_mtime_ns
    return {"continuations": count*len(expected), "moves": moves, "reasons": dict(reasons), "cpu_seconds": time.process_time()-started}


def replay_targets(stage, samples, choices, upstream):
    lock = p.old.read_json(p.OUTPUT/f"{stage}_targets_lock.json")
    assert lock["all_natural"] and lock["upstream_sha256"] == upstream
    assert set(lock["files"]) == {str(s["index"]) for s in samples}
    total, moves, cpu, reasons = 0, 0, 0., Counter()
    with ProcessPoolExecutor(max_workers=8, initializer=p.old.initialize_worker) as pool:
        jobs = [pool.submit(target_check, stage, sample, choices[i] if choices else None, upstream, lock["files"][str(sample["index"])]) for i, sample in enumerate(samples)]
        for i, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            total += row["continuations"]
            moves += row["moves"]
            cpu += row["cpu_seconds"]
            reasons.update(row["reasons"])
            if i % 16 == 0 or i == len(jobs):
                p.progress("audit_"+stage, states=i, total=len(jobs))
    return {"continuations": total, "moves": moves, "reasons": dict(reasons), "cpu_seconds": cpu,
            "targets_lock_sha256": p.old.file_hash(p.OUTPUT/f"{stage}_targets_lock.json")}


def independent_selection(samples, records):
    selected = []
    for sample, record in zip(samples, records):
        n = len(sample["actions"])
        assert record["action_indices"] == list(range(n))
        values = [sum(world["outcomes"][j]["wdl"] for world in record["worlds"])/len(record["worlds"]) for j in range(n)]
        chosen = 0
        for j in range(1, n):
            if values[j] > values[chosen]:
                chosen = j
        selected.append({"index": sample["index"], "sample_sha256": p.old.digest(sample), "choice": chosen,
                         "action": sample["actions"][chosen], "key": sample["keys"][chosen], "discovery_q": values})
    return selected


def choices_integrity(panel):
    selection = p.old.read_json(p.OUTPUT/"selection_lock.json")
    assert selection["manifest"] == p.check_identity()
    assert selection["panel_lock_sha256"] == p.old.file_hash(p.OUTPUT/"panel_lock.json")
    assert selection["targets_sha256"] == p.old.file_hash(p.OUTPUT/"discovery_targets_lock.json")
    records = p.read_records("discovery", panel["samples"], p.old.file_hash(p.OUTPUT/"panel_lock.json"))
    rebuilt = independent_selection(panel["samples"], records)
    assert rebuilt == selection["rows"]
    return rebuilt


def verify_probe():
    assert not (p.OUTPUT/"launch.json").exists()
    p.old.initialize_worker()
    source_lock = p.old.read_json(p.OUTPUT/"probe_sources_lock.json")
    assert set(source_lock["files"]) == {"0", "1"}
    sources = [source_check(int(i), info, True) for i, info in source_lock["files"].items()]
    panel = p.old.read_json(p.OUTPUT/"probe_panel.json")
    assert panel == p.old.read_json(p.old.OUTPUT/"probe_panel.json") and len(panel["samples"]) == 6
    for s in panel["samples"]:
        state = p.old.deserialize_state_snapshot(s["snapshot"])
        assert p.old.eligible(state, s["cell"])
        assert all(s[k] == v for k, v in independent_candidates(state, "probe", s["source_index"]).items())
    discovery = replay_targets("probe_discovery", panel["samples"], None, p.old.file_hash(p.OUTPUT/"probe_panel.json"))
    records = p.read_records("probe_discovery", panel["samples"], p.old.file_hash(p.OUTPUT/"probe_panel.json"))
    selection = p.old.read_json(p.OUTPUT/"probe_selection_lock.json")
    assert selection["targets_sha256"] == discovery["targets_lock_sha256"]
    assert independent_selection(panel["samples"], records) == selection["rows"]
    evaluation = replay_targets("probe_evaluation", panel["samples"], [r["choice"] for r in selection["rows"]], p.old.file_hash(p.OUTPUT/"probe_selection_lock.json"))
    return {"source_games": 2, "source_moves": sum(r["moves"] for r in sources),
            "discovery": discovery, "evaluation": evaluation,
            "probe_panel_sha256": p.old.file_hash(p.OUTPUT/"probe_panel.json"),
            "probe_selection_sha256": p.old.file_hash(p.OUTPUT/"probe_selection_lock.json"),
            "probe_sources_sha256": p.old.file_hash(p.OUTPUT/"probe_sources_lock.json"),
            "continuation_cpu_seconds": discovery["cpu_seconds"]+evaluation["cpu_seconds"]}


def prior_locks():
    launch = p.old.read_json(p.OUTPUT/"launch.json")
    assert launch["manifest"] == p.check_identity() and launch["preflight_sha256"] == p.old.file_hash(p.OUTPUT/"preflight.json")
    preflight = p.old.read_json(p.OUTPUT/"preflight.json")
    assert preflight["passed"] and preflight["manifest"] == launch["manifest"]
    assert preflight["probe_audit_sha256"] == p.old.file_hash(p.OUTPUT/"probe_audit.json")
    assert preflight["timing_sha256"] == p.old.file_hash(p.OUTPUT/"timing_probe.json")
    panel_audit = p.old.read_json(p.OUTPUT/"panel_audit.json")
    assert panel_audit["passed"] and panel_audit["manifest"] == launch["manifest"]
    assert panel_audit["panel_lock_sha256"] == p.old.file_hash(p.OUTPUT/"panel_lock.json")
    assert p.old.read_json(p.OUTPUT/"discovery_launch.json")["panel_audit_sha256"] == p.old.file_hash(p.OUTPUT/"panel_audit.json")


def verify_discovery():
    prior_locks()
    panel, _ = panel_integrity()
    assert not (p.OUTPUT/"targets/evaluation").exists()
    choices_integrity(panel)
    replay = replay_targets("discovery", panel["samples"], None, p.old.file_hash(p.OUTPUT/"panel_lock.json"))
    assert replay["continuations"] <= 98304
    return {**replay, "selection_sha256": p.old.file_hash(p.OUTPUT/"selection_lock.json")}


def independent_state_results(samples, choices, evaluation):
    results = []
    for sample, row, record in zip(samples, choices, evaluation):
        indices = record["action_indices"]
        assert indices == ([0] if row["choice"] == 0 else [0, row["choice"]])
        baseline = sum(w["outcomes"][0]["wdl"] for w in record["worlds"])/128
        selected = sum(w["outcomes"][-1]["wdl"] for w in record["worlds"])/128
        results.append({"index": sample["index"], "source_seed": sample["source_seed"], "cell": sample["cell"],
                        "choice": row["choice"], "candidate_count": len(sample["actions"]), "baseline_q": baseline, "selected_q": selected,
                        "evaluation_gain": selected-baseline, "discovery_gain": row["discovery_q"][row["choice"]]-row["discovery_q"][0]})
    return results


def independent_statistics(rows):
    groups = [[r for r in rows if r["cell"] == c] for c in range(6)]
    assert len({len(g) for g in groups}) == 1
    rng, resamples = np.random.default_rng(1543000000), []
    for _ in range(50):
        sums = np.zeros(1000)
        for group in groups:
            values = np.asarray([r["evaluation_gain"] for r in group])
            indices = rng.integers(len(group), size=(1000, len(group)))
            for j, draw in enumerate(indices):
                frequencies = np.bincount(draw, minlength=len(group))
                sums[j] += float((frequencies*values).sum())/len(group)
        resamples.extend((sums/6).tolist())
    ordered = sorted(resamples)
    def quantile(q):
        position = (len(ordered)-1)*q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)
    interval = [quantile(.025), quantile(.975)]
    gain = sum(r["evaluation_gain"] for r in rows)/len(rows)
    discovery = sum(r["discovery_gain"] for r in rows)/len(rows)
    half = (interval[1]-interval[0])/2
    checks = {"material_gain": gain >= .02, "positive_lower_bound": interval[0] > 0, "precision": half <= .02}
    def summary(part):
        return {"states": len(part), "gain": sum(r["evaluation_gain"] for r in part)/len(part),
                "discovery_gain": sum(r["discovery_gain"] for r in part)/len(part),
                "switch_fraction": sum(r["choice"] != 0 for r in part)/len(part)}
    return {"states": len(rows), "gain": gain, "ci95": interval, "half_width": half,
            "apparent_discovery_gain": discovery, "discovery_minus_evaluation": discovery-gain,
            "switch_fraction": sum(r["choice"] != 0 for r in rows)/len(rows),
            "baseline_only_states": sum(r["candidate_count"] == 1 for r in rows), "coincident_choices": sum(r["choice"] == 0 for r in rows),
            "cells": {str(c): summary(groups[c]) for c in range(6)},
            "phases": {str(c): summary([r for r in rows if r["cell"]//2 == c]) for c in range(3)},
            "seats": {str(c): summary([r for r in rows if r["cell"]%2 == c]) for c in range(2)},
            "statistical_gates": checks, "statistical_pass": all(checks.values()),
            "conditional_move_evidence_only": True, "arena": False, "training": False}


def verify_final():
    prior_locks()
    panel, _ = panel_integrity()
    choices = choices_integrity(panel)
    discovery = p.old.read_json(p.OUTPUT/"discovery_audit.json")
    assert discovery["passed"] and discovery["manifest"] == p.check_identity()
    assert discovery["selection_sha256"] == p.old.file_hash(p.OUTPUT/"selection_lock.json")
    assert discovery["targets_lock_sha256"] == p.old.file_hash(p.OUTPUT/"discovery_targets_lock.json")
    launch = p.old.read_json(p.OUTPUT/"evaluation_launch.json")
    assert launch["discovery_audit_sha256"] == p.old.file_hash(p.OUTPUT/"discovery_audit.json")
    assert launch["selection_sha256"] == discovery["selection_sha256"]
    evaluation = p.read_records("evaluation", panel["samples"], discovery["selection_sha256"])
    rows = independent_state_results(panel["samples"], choices, evaluation)
    compare(rows, p.old.read_json(p.OUTPUT/"state_results.json"))
    compare(independent_statistics(rows), p.old.read_json(p.OUTPUT/"result.json"))
    p.progress("audit_statistics_passed")
    replay = replay_targets("evaluation", panel["samples"], [r["choice"] for r in choices], discovery["selection_sha256"])
    assert replay["continuations"] <= 196608 and replay["continuations"]+discovery["continuations"] <= 294912
    return {"evaluation": replay, "main_continuations": replay["continuations"]+discovery["continuations"],
            "main_moves": replay["moves"]+discovery["moves"], "discovery_audit_sha256": p.old.file_hash(p.OUTPUT/"discovery_audit.json"),
            "panel_audit_sha256": p.old.file_hash(p.OUTPUT/"panel_audit.json"), "result_sha256": p.old.file_hash(p.OUTPUT/"result.json"),
            "state_results_sha256": p.old.file_hash(p.OUTPUT/"state_results.json"),
            "evaluation_targets_sha256": p.old.file_hash(p.OUTPUT/"evaluation_targets_lock.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("probe", "panel", "discovery", "final"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started, binding = time.perf_counter(), p.check_identity()
    result = {"probe": verify_probe, "panel": verify_panel, "discovery": verify_discovery, "final": verify_final}[args.stage]()
    assert p.check_identity() == binding
    p.old.save_new(p.OUTPUT/f"{args.stage}_audit.json", {"passed": True, "manifest": binding, "seconds": time.perf_counter()-started, **result})


if __name__ == "__main__":
    main()
