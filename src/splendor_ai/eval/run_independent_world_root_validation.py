"""Frozen-choice, crossed original/fresh-world diagnostic for experiment 138."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import torch

from splendor_ai.cfr import ResolvedLeaf, deterministic_belief_range, resolve_two_ply
from splendor_ai.diagnostics import state_signature
from splendor_ai.encoding import canonical_public_belief_key, canonical_successor_action_key
from splendor_ai.encoding.public_belief import actions_by_canonical_successor_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE
from splendor_ai.eval import run_expanded_root_resolver_headroom as audit085
from splendor_ai.eval.run_expanded_root_regret_stability_replication import ExpandedRootReplicationConfig
from splendor_ai.eval.run_bounded_public_belief_resolving import CHECKPOINT, CHECKPOINT_SHA256
from splendor_ai.termination import adjudicate_scores, is_forced_pass_turn, outcome_value
from splendor_ai.training.replay import deserialize_state_snapshot, serialize_state_snapshot
from splendor_ai.training.run_action_preference_repair import rollout_score_margin_loop_safe
from splendor_ai.training.run_policy_search_disagreement import _LegalActionScorer


PROTOCOL = Path("docs/engineering/independent_world_root_validation_138.md")
OUTPUT = Path("outputs/independent_world_root_validation_138")
SELECTION = Path("outputs/expanded_root_regret_stability_replication_086/selection.json")
RESULT086 = SELECTION.with_name("result.json")
SOURCE_HASHES = {
    SELECTION.as_posix(): "7bebc07fede835be023c986701fd813e4134374f1a4559fdcaf9439910d7e46f",
    RESULT086.as_posix(): "8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5",
    CHECKPOINT.as_posix(): CHECKPOINT_SHA256,
}
CONDITIONS = ("original_worlds", "fresh_worlds")
CELLS = tuple(f"seat{seat}|{phase}" for seat in (0, 1) for phase in ("early", "mid", "late"))
BLOCKS, WORLDS, REPLIES, REPLICATES, WORKERS = 4, 3, 4, 4, 10
MAX_CONTINUATIONS = 46_080
_SCORER: Any = None


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def input_identity() -> dict[str, Any]:
    hashes = {name: file_hash(Path(name)) for name in SOURCE_HASHES}
    if hashes != SOURCE_HASHES:
        raise ValueError("138 frozen source/checkpoint hash mismatch")
    dependencies = sorted(Path("src/splendor_ai").rglob("*.py")) + [
        Path("run_independent_world_root_validation_138.py"),
        Path("tests/test_independent_world_root_validation_138.py"),
    ]
    return {"format": "independent_world_root_validation_138_v1",
            "sources": hashes, "protocol_sha256": file_hash(PROTOCOL),
            "dependencies": {p.as_posix(): file_hash(p) for p in dependencies},
            "workers": WORKERS, "torch_threads": 1}


def select_states(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Select on IDs only; the results/decisions are not arguments to selection."""
    selected = []
    for cell in CELLS:
        rows = [s for s in source["selected"] if s["selection_cell"] == cell]
        def rank(s):
            key = f"138-selection-v1|{s['source_seed']}|{s['turn_index']}|{s['actor_seat']}|{s['phase']}"
            return hashlib.sha256(key.encode("utf-8")).hexdigest(), s["source_seed"], s["turn_index"]
        if len(rows) < 10:
            raise ValueError("138 insufficient source cell")
        selected.extend(sorted(rows, key=rank)[:10])
    if len({s["source_seed"] for s in selected}) != 60:
        raise ValueError("138 must have 60 distinct source games")
    return selected


def build_manifest(source: dict[str, Any], result: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    selected = select_states(source)
    decisions = {r["sample_index"]: r for r in result["rows"]}
    rows = []
    for schedule_index, sample in enumerate(selected):
        old = decisions[sample["sample_index"]]
        for key in ("source_seed", "turn_index", "actor_seat", "phase", "expanded_root_keys", "champion_root_keys"):
            if old[key] != sample[key]:
                raise ValueError(f"138 source/decision mismatch: {key}")
        champion = sample["champion_root_keys"][old["champion_selected_index"]]
        expanded = sample["expanded_root_keys"][old["expanded_selected_index"]]
        keep = ("sample_index", "source_seed", "turn_index", "actor_seat", "phase", "selection_cell", "state_snapshot")
        rows.append({**{k: sample[k] for k in keep}, "schedule_index": schedule_index,
                     "champion_action": champion, "expanded_action": expanded,
                     "source_root_count": len(sample["expanded_root_keys"])})
    return {"identity": identity, "states": rows, "selection_outcome_blind": True,
            "state_count": 60, "distinct_games": 60,
            "cells": dict(Counter(s["selection_cell"] for s in rows))}


def worlds_for(sample: dict[str, Any], condition: str, block: int):
    if condition not in CONDITIONS or block not in range(BLOCKS):
        raise ValueError("138 invalid condition/block")
    state = deserialize_state_snapshot(sample["state_snapshot"])
    if condition == "original_worlds":
        cfg = ExpandedRootReplicationConfig()
        index = audit085.belief_root_index(cfg, sample["source_seed"], sample["turn_index"])
    else:
        index = 1_380_000_000 + sample["sample_index"] * 100 + block * 10
    return deterministic_belief_range(state, sample["actor_seat"], root_index=index, count=WORLDS)


def continuation_seed(sample_index: int, block: int, world: int, replicate: int) -> int:
    if not (0 <= sample_index < 300 and 0 <= block < BLOCKS and 0 <= world < WORLDS and 0 <= replicate < REPLICATES):
        raise ValueError("138 invalid continuation seed slot")
    return 1_380_000_000_000_000_000 + sample_index * 100_000 + block * 1_000 + world * 10 + replicate


def root_keys(sample: dict[str, Any]):
    # ast parses only literal canonical tuples; never evaluate executable text.
    return tuple(dict.fromkeys(ast.literal_eval(sample[k]) for k in ("champion_action", "expanded_action")))


def prepare_branches(worlds, keys, scorer):
    """No selection/filtering of worlds, including terminal/fewer-reply cases."""
    env = SplendorEnv()
    branches = []
    action_sets = {}
    for key in keys:
        for index, world in enumerate(worlds):
            legal = actions_by_canonical_successor_key(world, env.legal_actions(world))
            if key not in legal:
                raise ValueError("138 fixed root not legal in sampled world")
            after = env.step(world, legal[key])
            if after.terminal:
                replies = []
                info = (key, ("terminal",))
            else:
                replies = list(audit085._distinct_canonical_scores(
                    after, scorer.score_actions(after, env.legal_actions(after))))[:REPLIES]
                if not replies:
                    raise ValueError("138 nonterminal state has no legal reply")
                info = (key, canonical_public_belief_key(after, after.current_player))
            reply_keys = tuple(canonical_successor_action_key(after, r.action) for r in replies)
            if action_sets.setdefault(info, reply_keys) != reply_keys:
                raise ValueError("138 unstable opponent information-set replies")
            branches.append({"world": index, "root": key, "after": after,
                             "information_set": info, "replies": replies})
    return branches


class TrackingEnv:
    """Observes exactly the transitions made by the unchanged rollout routine."""
    def __init__(self):
        self.env = SplendorEnv()
        self.last_state = None

    def step(self, state, action):
        self.last_state = self.env.step(state, action)
        return self.last_state

    def legal_actions(self, state):
        return self.env.legal_actions(state)


def terminal_record(state, player: int) -> dict[str, Any]:
    winner = adjudicate_scores(state)
    return {"wdl": outcome_value(winner, player), "winner": winner,
            "final_scores": [p.score for p in state.players],
            "final_card_counts": [len(p.purchased_cards) for p in state.players]}


def observe_rollout(state, action, player: int, seed: int, scorer):
    env = TrackingEnv()
    result = rollout_score_margin_loop_safe(env, scorer, state, action, player, seed,
                                          .1, .5, 3, 400, 20.0, 64)
    end = env.last_state
    if end is None:
        raise ValueError("138 rollout did not produce an endpoint")
    if not result.resolved or result.hard_cap_reached or not math.isfinite(result.value):
        raise ValueError("138 unresolved/hard-capped/nonfinite continuation")
    reason = "completed" if result.terminal else "loop_adjudicated"
    if result.loop_adjudicated:
        legal = env.legal_actions(end)
        if is_forced_pass_turn(legal):
            other = env.env.step(end, legal[0])
            if is_forced_pass_turn(env.legal_actions(other)):
                reason = "forced_pass_deadlock_adjudicated"
    return {**asdict(result), **terminal_record(end, player), "termination_reason": reason}


def resolve_records(records: list[dict[str, Any]], keys):
    """Resolve margin and WDL separately, and evaluate WDL under margin replies."""
    leaves = [ResolvedLeaf(r["world"], r["root"], r["info"], r["reply"], r["margin"], r["wdl"]) for r in records]
    margin = resolve_two_ply(leaves, keys, WORLDS, payoff="discovery_payoff")
    wdl = resolve_two_ply(leaves, keys, WORLDS, payoff="evaluation_payoff")
    under_margin = {repr(key): 0.0 for key in keys}
    for choice in margin.opponent_choices:
        matching = [r for r in records if repr(r["root"]) == choice["root_action"]
                    and repr(r["info"]) == choice["opponent_information_set"]
                    and repr(r["reply"]) == choice["selected_action"]]
        if len(matching) != len(choice["determinizations"]):
            raise ValueError("138 WDL margin-response accounting mismatch")
        under_margin[choice["root_action"]] += sum(r["wdl"] for r in matching) / WORLDS
    values = {repr(key): {"margin": margin.action_values[i], "wdl_min": wdl.action_values[i],
                         "wdl_under_margin": under_margin[repr(key)]} for i, key in enumerate(keys)}
    changes = sum(a["selected_action"] != b["selected_action"]
                  for a, b in zip(margin.opponent_choices, wdl.opponent_choices, strict=True))
    return values, changes, len(margin.opponent_choices)


def evaluate_block(sample, condition, block, scorer):
    worlds = worlds_for(sample, condition, block)
    keys = root_keys(sample)
    branches = prepare_branches(worlds, keys, scorer)
    records = []
    telemetry = Counter()
    for branch in branches:
        after, info = branch["after"], branch["information_set"]
        if after.terminal:
            endpoint = terminal_record(after, sample["actor_seat"])
            margin = (endpoint["final_scores"][sample["actor_seat"]] - endpoint["final_scores"][1-sample["actor_seat"]]) / 20.0
            records.append({"world": branch["world"], "root": branch["root"], "info": info,
                            "reply": ("terminal",), "margin": max(-1., min(1., margin)),
                            "wdl": endpoint["wdl"], "returns": [endpoint], "terminal_root": True})
            telemetry["terminal_roots"] += 1
            continue
        telemetry["expected_continuations"] += len(branch["replies"]) * REPLICATES
        telemetry["fewer_than_four_reply_branches"] += int(len(branch["replies"]) < REPLIES)
        for reply in branch["replies"]:
            returns = []
            for rep in range(REPLICATES):
                seed = continuation_seed(sample["sample_index"], block, branch["world"], rep)
                outcome = observe_rollout(after, reply.action, sample["actor_seat"], seed, scorer)
                returns.append(outcome)
                telemetry["continuations"] += 1
                telemetry["resolved_continuations"] += int(outcome["resolved"])
                telemetry["loop_adjudications"] += int(outcome["loop_adjudicated"])
                telemetry["fallback_triggers"] += outcome["fallback_triggers"]
                telemetry["hard_caps"] += int(outcome["hard_cap_reached"])
                telemetry[outcome["termination_reason"]] += 1
            records.append({"world": branch["world"], "root": branch["root"], "info": info,
                            "reply": canonical_successor_action_key(after, reply.action),
                            "margin": statistics.fmean(r["value"] for r in returns),
                            "wdl": statistics.fmean(r["wdl"] for r in returns), "returns": returns})
    values, changes, choices = resolve_records(records, keys)
    gains = {metric: values[sample["expanded_action"]][metric] - values[sample["champion_action"]][metric]
             for metric in ("margin", "wdl_min", "wdl_under_margin")}
    saved_records = [{**{k: v for k, v in r.items() if k not in ("root", "reply", "info")},
                      "root": repr(r["root"]), "reply": repr(r["reply"]),
                      "information_set_sha256": digest(r["info"])} for r in records]
    return {"condition": condition, "block": block, "values": values, "gains": gains,
            "world_hashes": [digest(serialize_state_snapshot(w)) for w in worlds],
            "telemetry": dict(telemetry), "wdl_reply_changes": changes, "opponent_choices": choices,
            "leaves": saved_records}


def initialize_worker():
    global _SCORER
    torch.set_num_threads(1)
    _SCORER = _LegalActionScorer(CHECKPOINT, "cpu", "legal_action_attention_scorer")


def evaluate_state(sample):
    started = time.perf_counter()
    blocks = []
    # Interleave conditions and alternate first condition to reduce order effects.
    for block in range(BLOCKS):
        order = CONDITIONS if (sample["schedule_index"] + block) % 2 == 0 else CONDITIONS[::-1]
        blocks.extend(evaluate_block(sample, condition, block, _SCORER) for condition in order)
    gains = {condition: {metric: statistics.fmean(b["gains"][metric] for b in blocks if b["condition"] == condition)
                         for metric in ("margin", "wdl_min", "wdl_under_margin")} for condition in CONDITIONS}
    state = deserialize_state_snapshot(sample["state_snapshot"])
    private_count = sum(p.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE
                        for p in state.players for i in range(len(p.reserved_cards)))
    keep = ("sample_index", "schedule_index", "source_seed", "phase", "actor_seat", "selection_cell",
            "champion_action", "expanded_action", "source_root_count")
    return {**{k: sample[k] for k in keep}, "sample_sha256": digest(sample), "blocks": blocks,
            "gains": gains, "difference": gains["fresh_worlds"]["margin"] - gains["original_worlds"]["margin"],
            "existing_blind_reserves": private_count, "elapsed_seconds": time.perf_counter() - started}


def paired_summary(rows, *, samples=50_000):
    ordered = sorted(rows, key=lambda r: (r["selection_cell"], r["source_seed"]))
    values = np.array([[r["gains"]["original_worlds"]["margin"],
                        r["gains"]["fresh_worlds"]["margin"], r["difference"]] for r in ordered])
    rng = np.random.default_rng(138300000)
    boot = np.zeros((samples, 3))
    for cell in sorted({r["selection_cell"] for r in ordered}):
        indices = np.array([i for i, r in enumerate(ordered) if r["selection_cell"] == cell])
        draws = rng.choice(indices, size=(samples, len(indices)), replace=True)
        boot += values[draws].sum(axis=1) / len(ordered)
    intervals = np.quantile(boot, [.025, .975], axis=0)
    return {name: {"mean": float(values[:, i].mean()), "ci95": intervals[:, i].tolist()}
            for i, name in enumerate(("original_gain", "fresh_gain", "fresh_minus_original"))}


def classify(summary, valid=True):
    if not valid:
        return "independent_world_validation_integrity_failed"
    old, fresh, delta = (summary[k] for k in ("original_gain", "fresh_gain", "fresh_minus_original"))
    if old["mean"] <= 0:
        return "conditional_gain_not_reproduced"
    if old["ci95"][0] <= 0:
        return "original_world_gain_uncertain"
    if delta["mean"] <= -.025 and delta["ci95"][1] < 0:
        return "material_world_attenuation_supported"
    if fresh["ci95"][0] > 0 and delta["ci95"][0] > -.025:
        return "conditional_gain_survives_fresh_worlds"
    return "independent_world_validation_inconclusive"


def summarize(rows):
    summary = paired_summary(rows)
    telemetry = {}
    for condition in CONDITIONS:
        count = Counter()
        for row in rows:
            for b in row["blocks"]:
                if b["condition"] == condition:
                    count.update(b["telemetry"])
        telemetry[condition] = dict(count)
    gates = {"sixty_states": len(rows) == 60,
             "sixty_source_games": len({r["source_seed"] for r in rows}) == 60,
             "balanced_cells": Counter(r["selection_cell"] for r in rows) == Counter({c: 10 for c in CELLS}),
             "maximum_budget": sum(t.get("continuations", 0) for t in telemetry.values()) <= MAX_CONTINUATIONS}
    for condition, t in telemetry.items():
        n = t.get("continuations", 0)
        gates[f"{condition}_accounting"] = n == t.get("expected_continuations", 0) == t.get("resolved_continuations", 0)
        gates[f"{condition}_no_hard_caps"] = t.get("hard_caps", 0) == 0
        gates[f"{condition}_bounded_adjudication"] = t.get("loop_adjudications", 0) / max(1, n) <= .001
    breakdowns = {}
    for field in ("phase", "actor_seat", "existing_blind_reserves"):
        breakdowns[field] = {}
        for value in sorted({r[field] for r in rows}):
            group = [r for r in rows if r[field] == value]
            breakdowns[field][str(value)] = {"states": len(group),
                "original_gain": statistics.fmean(r["gains"]["original_worlds"]["margin"] for r in group),
                "fresh_gain": statistics.fmean(r["gains"]["fresh_worlds"]["margin"] for r in group),
                "difference": statistics.fmean(r["difference"] for r in group)}
    secondary = {condition: {m: statistics.fmean(r["gains"][condition][m] for r in rows)
                             for m in ("wdl_min", "wdl_under_margin")} for condition in CONDITIONS}
    return {"summary": summary, "telemetry": telemetry, "integrity_gates": gates,
            "valid": all(gates.values()), "decision": classify(summary, all(gates.values())),
            "breakdowns": breakdowns, "secondary_wdl_gains": secondary,
            "changed_fixed_choices": sum(r["champion_action"] != r["expanded_action"] for r in rows),
            "action_families": {name: dict(Counter(ast.literal_eval(r[name])[0] for r in rows))
                                for name in ("champion_action", "expanded_action")},
            "secondary_reply_changes": sum(b["wdl_reply_changes"] for r in rows for b in r["blocks"]),
            "secondary_reply_choices": sum(b["opponent_choices"] for r in rows for b in r["blocks"]),
            "bootstrap": {"samples": 50_000, "seed": 138300000, "units": "distinct_source_games_within_six_cells"}}


def verify_sample(sample, scorer):
    state = deserialize_state_snapshot(sample["state_snapshot"])
    if serialize_state_snapshot(deserialize_state_snapshot(serialize_state_snapshot(state))) != serialize_state_snapshot(state):
        raise ValueError("138 snapshot round-trip mismatch")
    if state.current_player != sample["actor_seat"]:
        raise ValueError("138 acting seat mismatch")
    key = canonical_public_belief_key(state, state.current_player)
    # Full inventory must be a flat sorted multiset: private/deck membership may change.
    def inventory(s):
        ids = [c.card_id for tier in (1, 2, 3) for c in s.hidden_tier_decks[tier] + s.visible_tier_cards[tier]]
        ids += [c.card_id for p in s.players for c in p.reserved_cards + p.purchased_cards]
        return sorted(ids)
    altered = state.copy_shallow()
    for tier in (1, 2, 3):
        altered.hidden_tier_decks[tier] = list(reversed(altered.hidden_tier_decks[tier]))
    altered_sample = {**sample, "state_snapshot": serialize_state_snapshot(altered)}
    record = {"sample_index": sample["sample_index"], "blocks": []}
    original = worlds_for(sample, "original_worlds", 0)
    for condition in CONDITIONS:
        for block in range(BLOCKS):
            worlds = worlds_for(sample, condition, block)
            other = worlds_for(altered_sample, condition, block)
            for world, reordered in zip(worlds, other, strict=True):
                if canonical_public_belief_key(world, state.current_player) != key or inventory(world) != inventory(state):
                    raise ValueError("138 world changes observer information/card inventory")
                if serialize_state_snapshot(world) != serialize_state_snapshot(reordered):
                    raise ValueError("138 world schedule depends on actual hidden deck order")
            branches = prepare_branches(worlds, root_keys(sample), scorer)
            if condition == "original_worlds":
                for root in root_keys(sample):
                    legacy = audit085._prepare_single_root(original, root, ExpandedRootReplicationConfig(), scorer)
                    current = [b for b in branches if b["root"] == root]
                    for a, b in zip(legacy, current, strict=True):
                        if state_signature(a["after_root"]) != state_signature(b["after"]):
                            raise ValueError("138 original branch does not reproduce 086")
                        if a["information_set"] != b["information_set"] or [r.action for r in a["responses"]] != [r.action for r in b["replies"]]:
                            raise ValueError("138 original response structure mismatch")
            record["blocks"].append({"condition": condition, "block": block,
                "world_hashes": [digest(serialize_state_snapshot(w)) for w in worlds],
                "expected_continuations": sum(len(b["replies"]) * REPLICATES for b in branches)})
    return record


def preflight(identity, manifest):
    cached_path = OUTPUT / "preflight.json"
    launch_identity = {"inputs": identity, "manifest_sha256": digest(manifest)}
    if cached_path.exists():
        cached = read_json(cached_path)
        if cached["identity"] != launch_identity:
            raise ValueError("138 cached preflight identity mismatch")
        return cached
    initialize_worker()
    start = time.perf_counter()
    structures = []
    for i, sample in enumerate(manifest["states"]):
        structures.append(verify_sample(sample, _SCORER))
        if (i+1) % 10 == 0:
            print(f"[138:preflight] structural_states={i+1}/60", flush=True)
    source = read_json(SELECTION)
    used = {s["sample_index"] for s in manifest["states"]}
    probe = min((s for s in source["selected"] if s["sample_index"] not in used), key=lambda s: s["sample_index"])
    probe_state = deserialize_state_snapshot(probe["state_snapshot"])
    probe_actions = _SCORER.score_actions(probe_state, SplendorEnv().legal_actions(probe_state))
    probe_started = time.perf_counter()
    for index in range(8):
        # Runtime-only probe is outside the panel; no outcome is serialized.
        observe_rollout(probe_state, probe_actions[0].action, probe["actor_seat"],
                        1_389_000_000_000_000_000 + index, _SCORER)
    seconds = time.perf_counter() - probe_started
    projection = seconds / 8 * MAX_CONTINUATIONS * 2
    expected = sum(b["expected_continuations"] for s in structures for b in s["blocks"])
    payload = {"identity": launch_identity, "status": "prepared",
               "structures": structures, "expected_primary_continuations": expected,
               "runtime_probe": {"source_sample_index": probe["sample_index"], "continuations": 8,
                                 "seconds": seconds, "conservative_projected_cpu_seconds": projection},
               "passed": projection <= 43_200 and expected <= MAX_CONTINUATIONS,
               "precision_scenarios": [{"state_difference_sd": sd, "normal_ci95_half_width": 1.96*sd/math.sqrt(60)} for sd in (.05, .10, .20)],
               "elapsed_seconds": time.perf_counter()-start, "training": False,
               "protected_continuations": 0}
    atomic_write(cached_path, payload)
    return payload


def validate_saved_state(payload, sample, identity, structure=None):
    if payload.get("identity") != identity or payload["row"]["sample_sha256"] != digest(sample):
        raise ValueError("138 saved state identity mismatch")
    row = payload["row"]
    expected = {(c, b) for c in CONDITIONS for b in range(BLOCKS)}
    if {(r["condition"], r["block"]) for r in row["blocks"]} != expected or len(row["blocks"]) != 8:
        raise ValueError("138 saved state has incomplete/duplicate blocks")
    if structure is not None:
        reference = {(b["condition"], b["block"]): b for b in structure["blocks"]}
        for block in row["blocks"]:
            frozen = reference[block["condition"], block["block"]]
            count = block["telemetry"].get("continuations", 0)
            if (block["world_hashes"] != frozen["world_hashes"]
                    or count != frozen["expected_continuations"]
                    or count != block["telemetry"].get("resolved_continuations", 0)):
                raise ValueError("138 saved block differs from prepared worlds/accounting")
            if set(block["values"]) != set((sample["champion_action"], sample["expanded_action"])):
                raise ValueError("138 fixed actions changed during evaluation")
            for metric in ("margin", "wdl_min", "wdl_under_margin"):
                gain = block["values"][sample["expanded_action"]][metric] - block["values"][sample["champion_action"]][metric]
                if abs(gain - block["gains"][metric]) > 1e-12:
                    raise ValueError("138 block gain accounting mismatch")
        for condition in CONDITIONS:
            for metric in ("margin", "wdl_min", "wdl_under_margin"):
                mean = statistics.fmean(b["gains"][metric] for b in row["blocks"] if b["condition"] == condition)
                if abs(mean - row["gains"][condition][metric]) > 1e-12:
                    raise ValueError("138 state gain accounting mismatch")
        difference = row["gains"]["fresh_worlds"]["margin"] - row["gains"]["original_worlds"]["margin"]
        if abs(difference - row["difference"]) > 1e-12:
            raise ValueError("138 paired difference accounting mismatch")
    return row


@contextmanager
def process_lock(path):
    """Kernel lock releases on crash, so the file can safely remain on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def run(*, launch=False):
    with process_lock(OUTPUT / "process.lock"):
        identity = input_identity()
        manifest = build_manifest(read_json(SELECTION), read_json(RESULT086), identity)
        manifest_path = OUTPUT / "state_manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise ValueError("138 existing manifest identity mismatch")
        else:
            atomic_write(manifest_path, manifest)
        if not launch:
            result = preflight(identity, manifest)
            print(json.dumps({k: result[k] for k in ("status", "passed", "runtime_probe", "expected_primary_continuations")}), flush=True)
            return result
        if not (OUTPUT / "preflight.json").exists():
            raise ValueError("138 --launch requires completed preflight")
        prepared = read_json(OUTPUT / "preflight.json")
        expected_identity = {"inputs": identity, "manifest_sha256": digest(manifest)}
        if prepared["identity"] != expected_identity or not prepared["passed"]:
            raise ValueError("138 preparation failed or changed")
        launch_identity = {**expected_identity, "preflight_sha256": file_hash(OUTPUT / "preflight.json")}
        structures = {s["sample_index"]: s for s in prepared["structures"]}
        final_path = OUTPUT / "result.json"
        if final_path.exists():
            final = read_json(final_path)
            if final["identity"] != launch_identity:
                raise ValueError("138 result identity mismatch")
            return final
        pending, rows = [], []
        for sample in manifest["states"]:
            path = OUTPUT / "states" / f"state_{sample['schedule_index']:02d}.json"
            if path.exists():
                rows.append(validate_saved_state(read_json(path), sample, launch_identity, structures[sample["sample_index"]]))
            else:
                pending.append(sample)
        started = time.perf_counter()
        print(f"[138:run] completed={len(rows)}/60 pending={len(pending)} workers={WORKERS}", flush=True)
        atomic_write(OUTPUT / "run_identity.json", {"identity": launch_identity, "pid": os.getpid(), "started_unix": time.time()})
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=initialize_worker) as executor:
            futures = {executor.submit(evaluate_state, s): s for s in pending}
            for done, future in enumerate(as_completed(futures), 1):
                sample = futures[future]
                row = future.result()
                payload = {"identity": launch_identity, "row": row}
                validate_saved_state(payload, sample, launch_identity, structures[sample["sample_index"]])
                atomic_write(OUTPUT / "states" / f"state_{sample['schedule_index']:02d}.json", payload)
                rows.append(row)
                elapsed = time.perf_counter()-started
                remaining = len(pending)-done
                print(f"[138:run] completed={len(rows)}/60 elapsed={elapsed:.1f}s eta={elapsed/done*remaining:.1f}s", flush=True)
        # Guard dependencies/checkpoint/source files again before reporting evidence.
        if input_identity() != identity:
            raise ValueError("138 inputs changed during evaluation")
        rows.sort(key=lambda r: r["schedule_index"])
        final = {"identity": launch_identity, "status": "complete", **summarize(rows),
                 "elapsed_this_session_seconds": time.perf_counter()-started,
                 "aggregate_state_wall_seconds": sum(r["elapsed_seconds"] for r in rows),
                 "state_results": [f"states/state_{r['schedule_index']:02d}.json" for r in rows],
                 "automatic_continuation": False, "training": False, "promotion": False,
                 "full_game_arena": False, "research_champion": "084", "fast_champion": "035"}
        atomic_write(final_path, final)
        print(f"[138:complete] decision={final['decision']}", flush=True)
        return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true", help="Run protected evaluation after verified preflight")
    args = parser.parse_args()
    run(launch=args.launch)


if __name__ == "__main__":
    main()
