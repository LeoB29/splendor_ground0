"""All-game reconstruction with a sealed8/128-world sample and full stress roots.

Sampling uses identities and lawful metadata only. Every declared world and
outcome field is checked; only selected trajectories are independently replayed.
The same frozen engine/model primitives are shared with the previous auditors.
"""
from collections import Counter
import hashlib
import math
import random
import re
from pathlib import Path
import time

import numpy as np
import torch

from tools.proposal_iteration_128 import arena as p
from scripts.analysis import verify_learned_proposal_screen as a_screen
from scripts.analysis import verify_rollout_selector_154 as a154
from scripts.analysis import verify_afterstate_advantage_149 as a149
from scripts.analysis import verify_fast_selector_transfer_157 as a157
from splendor_ai.encoding import encode_public_observation_tensor, encode_legal_action_features, fit_feature_vector
from splendor_ai.encoding.public_belief import canonical_public_belief_key
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE

old, OUTPUT, PRODUCTION_OUTPUT = p.old, p.OUTPUT, p.PRODUCTION_OUTPUT
compare = a_screen.compare
FITS = (161001, 161002, 161003)
AUDIT_SEED = 2074000000
WORLD_TAG = "proposal-iteration-128-audit-world-v1"
STRESS_TAG = "proposal-iteration-128-audit-stress-v1"
SAMPLE_CONFIG = {"seed": AUDIT_SEED, "world_tag": WORLD_TAG, "stress_tag": STRESS_TAG,
                 "worlds_per_root": 128, "ordinary_worlds_per_root": 8,
                 "stress_fit_is_assigned_fit_for_both_arms": True,
                 "full_inner_replay": False}


def independent_specs(stage):
    assert stage in ("probe_a", "stage_a", "probe_b", "stage_b", "test")
    count = {"probe_a": 6, "stage_a": 201, "probe_b": 6, "stage_b": 201, "test": 3}[stage]
    ranked = sorted(range(count), key=lambda pair: old.digest(["proposal-iteration-128-fit-v1", stage, pair]))
    result = []
    for pair in range(count):
        fit = (161001, 161002, 161003)[ranked.index(pair) % 3]
        for seat in (0, 1):
            result.append({"index": 2*pair+seat, "pair": pair, "candidate_seat": seat, "fit": fit,
                           "environment_seed": {"probe_a": 2079000000, "stage_a": 2070000000, "probe_b": 2079100000, "stage_b": 2071000000, "test": 2079800000}[stage]+pair,
                           "search_stream": old.digest(["proposal-iteration-128-stream-v1", stage, pair])})
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
    assert stage in ("probe_a", "stage_a", "probe_b", "stage_b", "test")
    assert fit in FITS and type(candidate) is bool
    if candidate:
        return f"treatment_{fit}"
    if stage == "probe_b" or stage == "stage_b":
        return f"replay_{fit}"
    return f"anchor_{fit}"


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


def adjudicate(state):
    ranks = [(player.score, -len(player.purchased_cards)) for player in state.players]
    return None if ranks[0] == ranks[1] else (0 if ranks[0] > ranks[1] else 1)


def _timing(values):
    assert set(values) == {"wall", "cpu"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values.values())


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


def ordinary_indices(stage, index, turn):
    assert stage in ("probe_a", "stage_a", "probe_b", "stage_b", "test")
    assert type(index) is int and index >= 0 and type(turn) is int and 0 <= turn < 400
    return sorted(sorted(range(128), key=lambda wi: (
        old.digest([WORLD_TAG, AUDIT_SEED, stage, index, turn, wi]), wi))[:8])


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
    support = len(root["candidates"]["actions"])
    assert 1 <= support <= 4 and len(root["candidates"]["keys"]) == support
    assert len(root["worlds"]) == 128 and len(root["schedule"]["world_seeds"]) == 128
    return {"root_id": root_id(spec["index"], state.turn_index), "index": spec["index"],
            "pair": spec["pair"], "fit": spec["fit"], "turn": state.turn_index,
            "arm": "candidate" if state.current_player == spec["candidate_seat"] else "reference",
            "actor": state.current_player, "phase": 0 if state.turn_index < 20 else 1 if state.turn_index < 40 else 2,
            "privacy": privacy, "privacy_category": int(bool(privacy["own"]))+2*int(bool(privacy["opponent"])),
            "support": support, "worlds": 128, "sha256": move["root_sha256"]}


def metadata_inventory(stage, game_items, binding, base=OUTPUT):
    """Read only identity/metadata fields; scores/outcomes never enter sampling."""
    base = Path(base)
    specs = independent_specs(stage)
    assert [item["index"] for item in game_items] == list(range(len(specs)))
    expected_games, expected_roots, inventory = set(), set(), []
    for spec, item in zip(specs, game_items):
        path = base / stage / "games" / f"game_{spec['index']:04d}.json"
        expected_games.add(path.resolve())
        if "path" in item:
            assert _artifact_path(item).resolve() == path.resolve()
        assert old.file_hash(path) == item["sha256"]
        game = old.read_json(path)
        assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
        seen = set()
        for move in game["trace"]:
            assert type(move["turn"]) is int and move["turn"] not in seen
            seen.add(move["turn"])
            root_path = base / stage / "roots" / f"game_{spec['index']:04d}" / f"turn_{move['turn']:03d}.json"
            expected_roots.add(root_path.resolve())
            assert old.file_hash(root_path) == move["root_sha256"]
            root = old.read_json(root_path)
            assert root["manifest"] == binding and "failure" not in root
            inventory.append(_root_metadata(stage, spec, move, root))
    assert {path.resolve() for path in (base / stage / "games").glob("*.json")} == expected_games
    assert {path.resolve() for path in (base / stage / "roots").rglob("*.json")} == expected_roots
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
        assert row["worlds"] == 128 and 1 <= row["support"] <= 4
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
    selections = []
    for row in inventory:
        ordinary = ordinary_indices(stage, row["index"], row["turn"])
        stress = row["root_id"] in full
        selections.append({"root_id": row["root_id"], "ordinary_world_indices": ordinary,
                           "replay_world_indices": list(range(128)) if stress else ordinary,
                           "stress": stress})
    selected_cells = sum(cell["selected_root"] is not None for cell in cells)
    return {"format": "proposal-iteration-128-audit-sample-v1", "manifest": binding, "stage": stage,
            "config": SAMPLE_CONFIG, "games_lock_sha256": game_lock_sha256,
            "inventory": inventory, "selections": selections, "stress_cells": cells,
            "full_stress_roots": sorted(full), "coverage": {
                "games": len({row["index"] for row in inventory}), "roots": len(inventory),
                "recorded_worlds": 128*len(inventory),
                "recorded_continuations": 128*sum(row["support"] for row in inventory),
                "replayed_worlds": sum(len(row["replay_world_indices"]) for row in selections),
                "replayed_continuations": sum(len(selected["replay_world_indices"])*root["support"]
                                              for root, selected in zip(inventory, selections)),
                "stress_cells": len(cells), "selected_stress_cells": selected_cells,
                "stress_roots": len(full), "stress_selection_overlaps": selected_cells-len(full),
                "missing_stress_cells": [row["cell"] for row in cells if row["selected_root"] is None],
                "full_inner_replay": False}}


def seal_sample(stage, game_items, binding, base=OUTPUT):
    base = Path(base)
    lock_path = base / f"{stage}_games_lock.json"
    lock = old.read_json(lock_path)
    assert lock["manifest"] == binding and lock["files"] == game_items
    inventory = metadata_inventory(stage, game_items, binding, base)
    sample = sample_plan(stage, inventory, binding, old.file_hash(lock_path))
    path = base / stage / "audit_sample.json"
    old.save_new(path, sample)
    old.save_new(base / stage / "audit_sample_seal.json", {
        "manifest": binding, "stage": stage, "sample_sha256": old.file_hash(path),
        "games_lock_sha256": old.file_hash(lock_path)})
    return _artifact(path)


def load_sample(stage, binding, base=OUTPUT):
    """Cheap per-game seal check; global reconstruction occurs before/after audit."""
    base = Path(base)
    path = base / stage / "audit_sample.json"
    seal = old.read_json(base / stage / "audit_sample_seal.json")
    assert seal["manifest"] == binding and seal["stage"] == stage
    assert seal["sample_sha256"] == old.file_hash(path)
    assert seal["games_lock_sha256"] == old.file_hash(base / f"{stage}_games_lock.json")
    sample = old.read_json(path)
    assert sample["manifest"] == binding and sample["stage"] == stage and sample["config"] == SAMPLE_CONFIG
    assert sample["games_lock_sha256"] == seal["games_lock_sha256"]
    return sample


def validate_sample(stage, binding, base=OUTPUT):
    sample = load_sample(stage, binding, base)
    lock_path = Path(base) / f"{stage}_games_lock.json"
    lock = old.read_json(lock_path)
    assert lock["manifest"] == binding
    inventory = metadata_inventory(stage, lock["files"], binding, base)
    compare(sample, sample_plan(stage, inventory, binding, old.file_hash(lock_path)))
    return sample


def _check_outcome(outcome, actor, turn):
    assert set(outcome) == {"reason", "winner", "turns", "trace_sha256", "final_state_sha256", "wdl"}
    assert outcome["reason"] in ("completed", "forced_pass_deadlock")
    assert outcome["winner"] is None or type(outcome["winner"]) is int and outcome["winner"] in (0, 1)
    assert type(outcome["turns"]) is int and 1 <= outcome["turns"] <= 400-turn
    assert type(outcome["wdl"]) in (int, float) and math.isfinite(outcome["wdl"])
    assert outcome["wdl"] == (0. if outcome["winner"] is None else 1. if outcome["winner"] == actor else -1.)
    for key in ("trace_sha256", "final_state_sha256"):
        assert isinstance(outcome[key], str) and re.fullmatch(r"[0-9a-f]{64}", outcome[key])


def audit_root(stage, index, state, move, spec, binding, base=OUTPUT, selection=None, full_fixture=False):
    start, cpu = time.perf_counter(), time.process_time()
    ordinary = ordinary_indices(stage, index, state.turn_index)
    assert selection is not None and selection["root_id"] == root_id(index, state.turn_index)
    assert selection["ordinary_world_indices"] == ordinary and type(selection["stress"]) is bool
    expected_indices = list(range(128)) if selection["stress"] else ordinary
    assert selection["replay_world_indices"] == expected_indices
    if full_fixture:
        resolved = Path(base).resolve()
        assert stage == "test" and resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents
    replay_indices = list(range(128)) if full_fixture else expected_indices
    path = Path(base) / stage / "roots" / f"game_{index:04d}" / f"turn_{state.turn_index:03d}.json"
    assert old.file_hash(path) == move["root_sha256"]
    assert path.stat().st_mtime_ns >= (Path(base) / f"{stage}_launch.json").stat().st_mtime_ns
    root = old.read_json(path)
    assert root["manifest"] == binding and "failure" not in root
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
    expected = independent_proposal(state, spec["search_stream"], source_model)
    compare({key: value for key, value in root["proposal"].items() if key != "timing"}, expected)
    proposal_timing = root["proposal"]["timing"]
    assert set(proposal_timing) == {"wall", "learned_extra_wall"}
    assert all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in proposal_timing.values())
    assert proposal_timing["learned_extra_wall"] <= proposal_timing["wall"]
    a157.input_invariance(state, p._MODELS[source_model].config)
    compare(expected, independent_proposal(altered, spec["search_stream"], source_model))
    offered = expected["candidates"]
    assert offered["keys"][:3] == expected["reference"]["keys"][:3]
    assert root["candidates"] == offered and len(offered["actions"]) == min(4, offered["canonical_support"])
    assert len(set(offered["keys"])) == len(offered["actions"])
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    privacy = {"own": sum(own.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(own.reserved_cards))),
               "opponent": sum(other.reserved_card_source(i) != RESERVE_SOURCE_VISIBLE for i in range(len(other.reserved_cards)))}
    assert root["privacy"] == privacy and len(root["worlds"]) == 128
    totals, moves, reasons = [0.] * len(offered["actions"]), 0, Counter()
    ordinary_wall, extra_wall = 0., 0.
    for wi, (seed, saved) in enumerate(zip(schedule["world_seeds"], root["worlds"])):
        assert saved["seed"] == seed and len(saved["outcomes"]) == len(totals)
        world = a154.independent_world(state, seed)
        world_sha = old.digest(old.serialize_state_snapshot(world))
        assert saved["state_sha256"] == world_sha
        assert old.digest(old.serialize_state_snapshot(a154.independent_world(altered, seed))) == world_sha
        assert world.players[state.current_player].reserved_cards == own.reserved_cards
        legal_world = old.SplendorEnv().legal_actions(world)
        for j, (payload, outcome) in enumerate(zip(offered["actions"], saved["outcomes"])):
            action = old.decode_action(payload)
            assert action in legal_world
            _check_outcome(outcome, state.current_player, state.turn_index)
            totals[j] += outcome["wdl"]
            if wi in replay_indices:
                replay_start = time.perf_counter()
                rebuilt = a149.replay_branch(world, action, state.current_player)
                compare(outcome, rebuilt)
                elapsed = time.perf_counter()-replay_start
                if wi in ordinary:
                    ordinary_wall += elapsed
                else:
                    extra_wall += elapsed
                moves += rebuilt["turns"]
                reasons[rebuilt["reason"]] += 1
    means = [v / 128 for v in totals]
    choice = min(range(len(means)), key=lambda j: (-means[j], j))
    compare(root["means"], means)
    assert type(root["choice"]) is int and root["choice"] == choice
    assert move["action"] == offered["actions"][choice]
    _timing(root["timing"])
    return {"index": index, "turn": state.turn_index, "seat": state.current_player,
            "planner": "candidate" if candidate else "reference", "fit": spec["fit"],
            "model_id": source_model, "model_sha256": expected["model_sha256"],
            "sha256": move["root_sha256"], "choice": choice, "support": len(totals),
            "assigned_fit": spec["fit"], "privacy": privacy, "timing": root["timing"],
            "worlds": 128, "recorded_worlds": 128, "replayed_worlds": len(replay_indices),
            "continuations": 128*len(totals), "replayed_branches": sum(reasons.values()),
            "ordinary_world_indices": ordinary, "replay_world_indices": replay_indices,
            "stress": selection["stress"], "full_fixture_replay": bool(full_fixture),
            "full_inner_replay": bool(full_fixture), "moves": moves, "reasons": dict(reasons),
            "audit_timing": {"wall": time.perf_counter()-start, "cpu": time.process_time()-cpu,
                             "ordinary_replay_wall": ordinary_wall, "stress_extra_replay_wall": extra_wall}}


def linear_quantile(values, q):
    ordered = sorted(values)
    point = (len(ordered) - 1) * q
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def independent_statistics(rows, stage):
    assert stage in ("stage_a", "stage_b")
    assert len(rows) == 402 and [r["index"] for r in rows] == list(range(402))
    specs = independent_specs(stage)
    strata = {seed: [] for seed in (161001, 161002, 161003)}
    for i in range(0, 402, 2):
        a, b = rows[i:i+2]
        assert (a["seat"], b["seat"]) == (0, 1) and a["pair"] == b["pair"] == i // 2
        assert a["fit"] == b["fit"] == specs[i]["fit"]
        assert a["score"] in (0., .5, 1.) and b["score"] in (0., .5, 1.)
        strata[a["fit"]].append(int(2*a["score"]+2*b["score"]))
    assert all(len(v) == 67 for v in strata.values())
    rng = np.random.default_rng({"stage_a": 2073000000, "stage_b": 2073100000}[stage])
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
             "below_material_target" if interval[1] < .55 else "inconclusive")
    return {"games": 402, "pairs": 201, "share": share, "ci95": interval,
            "wins": sum(r["score"] == 1 for r in rows), "draws": sum(r["score"] == .5 for r in rows),
            "losses": sum(r["score"] == 0 for r in rows), "stage": stage,
            "stage_pass": practical if stage == "stage_a" else superiority, "superiority_established": superiority,
            "practical_progress": practical, "material_target_excluded": interval[1] < .55, "classification": label}


def audit_game(stage, index, expected_sha, binding, base=OUTPUT, test_initial_snapshot=None, full_fixture=False):
    """Reconstruct a complete outer game; source policies are independently checked."""
    wall, cpu = time.perf_counter(), time.process_time()
    base = Path(base)
    spec = independent_spec(stage, index)
    if stage == "test":
        resolved = base.resolve()
        assert resolved != PRODUCTION_OUTPUT and PRODUCTION_OUTPUT not in resolved.parents
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
    assert lock["files"][index]["index"] == index and lock["files"][index]["sha256"] == expected_sha
    path = base/stage/"games"/f"game_{index:04d}.json"
    assert old.file_hash(path) == expected_sha
    assert path.stat().st_mtime_ns >= (base/f"{stage}_launch.json").stat().st_mtime_ns
    game = old.read_json(path)
    assert game["manifest"] == binding and game["spec"] == spec and "failure" not in game
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
        assert not state.terminal and state.turn_index < 400
        visits[old.state_signature(state)] += 1
        assert visits[old.state_signature(state)] < 4
        assert move["turn"] == state.turn_index == start_turn+n and move["actor"] == state.current_player
        assert move["state_sha256"] == old.digest(old.serialize_state_snapshot(state))
        legal = env.legal_actions(state)
        assert legal
        identity = root_id(index, state.turn_index)
        root_record = old.read_json(base / stage / "roots" / f"game_{index:04d}" / f"turn_{state.turn_index:03d}.json")
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
           "turns": game["turns"], "reason": game["reason"], "timing": game["timing"]}
    return {"game": row, "roots": roots, "initial_sha256": initial_sha,
            "sample_sha256": old.file_hash(base / stage / "audit_sample.json"),
            "full_inner_replay": bool(full_fixture),
            "audit_timing": {"wall": time.perf_counter()-wall, "cpu": time.process_time()-cpu}}


def _fixture_job(kind, fixture, index):
    base = Path(fixture["base"])
    p.check_stage("test", base)
    if kind == "games":
        return p.play_game("test", independent_spec("test", index), fixture["binding"], base, fixture["snapshot"])
    assert kind == "audits"
    item = fixture["games"][index]
    return audit_game("test", index, item["sha256"], fixture["binding"], base, fixture["snapshot"],
                      full_fixture=fixture.get("full_fixture_replay", False))


def fixture_pipeline(path):
    """Isolated synthetic Windows spawn, using the actual new worker initializer."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from tools.proposal_iteration_128.experiment import worker_initialize
    fixture = old.read_json(Path(path))
    assert fixture["kind"] == "proposal-iteration-128-synthetic-v1"
    base = Path(fixture["base"])
    p.check_stage("test", base)
    p.windows.assert_windowless_worker(base / "parent_stdout.txt", base / "parent_stderr.txt")
    assert fixture["registry"]["manifest"] == fixture["binding"]
    old.save_new(base / "models_lock.json", fixture["registry"])
    old.save_new(base / "test_launch.json", {"manifest": fixture["binding"]})
    for kind in ("games", "audits"):
        p.windows.configure_spawn()
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=worker_initialize, initargs=("test", kind, fixture["binding"], base)) as pool:
            values = list(pool.map(_fixture_job, [kind]*6, [fixture]*6, range(6)))
        if kind == "games":
            fixture["games"] = values
            old.save_new(base / "test_games_lock.json", {"manifest": fixture["binding"], "files": values})
            seal_sample("test", values, fixture["binding"], base)
            validate_sample("test", fixture["binding"], base)
    validate_sample("test", fixture["binding"], base)
    old.save_new(base / "fixture_result.json", {"passed": True, "games": 6,
                 "roots": sum(len(row["roots"]) for row in values),
                 "continuations": sum(root["continuations"] for row in values for root in row["roots"]),
                 "replayed_branches": sum(root["replayed_branches"] for row in values for root in row["roots"]),
                 "full_fixture_replay": fixture.get("full_fixture_replay", False), "receipts": values})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explicit isolated synthetic Windows spawn check only")
    parser.add_argument("--fixture", required=True)
    fixture_pipeline(parser.parse_args().fixture)
