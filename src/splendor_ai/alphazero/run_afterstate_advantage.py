"""Experiment 149: matched afterstate advantage-learning pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from splendor_ai.alphazero.afterstate_149 import AdvantageModel, deterministic_part, features
from splendor_ai.alphazero.run_strong_policy_pilot import CHECKPOINT, CHECKPOINT_SHA
from splendor_ai.bots.hidden_information import determinize_for_observer
from splendor_ai.bots.model_bot import LegalActionAttentionScorerBot
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import actions_by_canonical_successor_key, canonical_successor_action_key
from splendor_ai.engine.actions import Action, ActionType
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import RESERVE_SOURCE_VISIBLE
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS, adjudicate_scores, is_forced_pass_turn, outcome_value
from splendor_ai.training.replay import serialize_state_snapshot, deserialize_state_snapshot, serialize_action

OUTPUT = Path("outputs/afterstate_advantage_149")
PROTOCOL = Path("docs/engineering/afterstate_advantage_149.md")
VERIFIER = Path("scripts/analysis/verify_afterstate_advantage_149.py")
PRIOR = Path("outputs/search_teacher_headroom_148/manifest.json")
PRIOR_SHA = "2f7a300fb8d058e05e64ab0f3e5c28d53004524820b13cd01995814215f8f257"
CORPUS = Path("outputs/value_generalization_scale_143")
ARMS = ("control", "afterstate")
CONFIG = {
    "states": {"train": 768, "validation": 192, "test": 192},
    "worlds": {"train": 32, "validation": 64, "test": 128},
    "world_seed_start": {"probe": 1_499_100_000, "train": 1_490_100_000,
                         "validation": 1_491_100_000, "test": 1_492_100_000},
    "fresh_source_seed_start": {"probe": 1_499_000_000, "test": 1_490_000_000},
    "source_limit": {"probe": 60, "test": 1024},
    "candidates": 4, "workers": 8, "torch_threads": 1,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "training_seeds": [149001, 149002, 149003], "updates": 4096,
    "batch_states": 64, "validation_interval": 128,
    "learning_rate": .0003, "weight_decay": .0001, "gradient_norm": 1.,
    "bootstrap_seed": 149999, "bootstrap_samples": 50000,
    "material_wdl_gain": .02, "maximum_ci_half_width": .02,
    "audit": "all_sources_all_features_all_continuations_all_six_fits_all_predictions_and_independent_statistics",
}
_RAW = None


def now():
    return datetime.now(timezone.utc).isoformat()


def save_new(path, value):
    if path.exists():
        raise FileExistsError(path)
    atomic_write(path, value)


def tensor_save(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def initialize_worker():
    global _RAW
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _RAW = LegalActionAttentionScorerBot(CHECKPOINT)


def decode_action(payload):
    data = dict(payload)
    data["action_type"] = ActionType[data["action_type"]]
    for field in ("take_tokens", "return_tokens", "spend_tokens", "metadata"):
        data[field] = tuple(tuple(x) if isinstance(x, list) else x for x in data.get(field, []))
    return Action(**data)


def phase(turn):
    return 0 if turn < 20 else 1 if turn < 40 else 2


def eligible(state, cell):
    return (not state.terminal and phase(state.turn_index) * 2 + state.current_player == cell
            and all(p.reserved_card_source(i) == RESERVE_SOURCE_VISIBLE
                    for p in state.players for i in range(len(p.reserved_cards))))


def trajectory(state, root_action=None, include_trace=False):
    """Exact raw-035 continuation with explicit natural/cutoff accounting."""
    env = SplendorEnv()
    seen, forced, no_progress, turns = {}, 0, 0, 0
    hasher, trace = hashlib.sha256(), []
    reason, winner = "completed", None
    while not state.terminal:
        signature = state_signature(state)
        seen[signature] = seen.get(signature, 0) + 1
        if seen[signature] >= CONFIG["repetition_limit"]:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= CONFIG["max_turns"]:
            reason = "max_turns"
            break
        legal = env.legal_actions(state)
        if not legal:
            reason = "stalled"
            break
        action = root_action if turns == 0 and root_action is not None else _RAW.choose_action(env, state, legal)
        payload = serialize_action(action)
        hasher.update(digest(payload).encode())
        if include_trace:
            trace.append({"state_hash": digest(serialize_state_snapshot(state)), "action": payload,
                          "turn": state.turn_index, "actor": state.current_player})
        successor = env.step(state, action)
        turns += 1
        forced = forced + 1 if is_forced_pass_turn(legal) else 0
        no_progress = 0 if is_progress_transition(state, successor, action) else no_progress + 1
        state = successor
        if forced >= 2:
            reason, winner = "forced_pass_deadlock", adjudicate_scores(state)
            break
        if no_progress >= CONFIG["no_progress_limit"]:
            reason = "no_progress_cutoff"
            break
    if reason == "completed":
        winner = state.winner
    final = serialize_state_snapshot(state)
    result = {"reason": reason, "winner": winner, "turns": turns,
              "trace_sha256": hasher.hexdigest(), "final_state_sha256": digest(final)}
    if include_trace:
        result.update(trace=trace, final_state=final)
    return result


def candidates(state, split, source_index):
    env = SplendorEnv()
    legal = env.legal_actions(state)
    scores = _RAW.score_actions(state, legal)
    if not np.isfinite(scores).all():
        raise ValueError("Invalid 035 scores")
    ranked = sorted(range(len(legal)), key=lambda i: (-scores[i], i))
    groups = actions_by_canonical_successor_key(state, legal)
    chosen, keys = [], []
    for i in ranked:
        key = canonical_successor_action_key(state, legal[i])
        if key not in keys:
            # Preserve exact greedy concrete action and ranked alternative.
            chosen.append(legal[i])
            keys.append(key)
        if len(chosen) == min(2, len(groups)):
            break
    rest = [key for key in groups if key not in keys]
    rng = random.Random(f"149-candidates-v1|{split}|{source_index}")
    for key in rng.sample(rest, min(2, len(rest))):
        chosen.append(groups[key])
        keys.append(key)
    return {"actions": [serialize_action(a) for a in chosen], "keys": [repr(k) for k in keys],
            "canonical_support": len(groups), "legal_support": len(legal)}


def reconstruct_source(record, verify_raw=False):
    env = SplendorEnv(seed=record["seed"])
    state = env.initial_state()
    states = []
    for row in record["trace"]:
        if digest(serialize_state_snapshot(state)) != row["state_hash"]:
            raise ValueError("Source state hash mismatch")
        states.append(state)
        action = decode_action(row["action"])
        if verify_raw and _RAW.choose_action(env, state) != action:
            raise ValueError("Fresh source deviated from exact raw035")
        state = env.step(state, action)
    if verify_raw:
        if digest(serialize_state_snapshot(state)) != digest(record["final_state"]):
            raise ValueError("Fresh source final state mismatch")
        if digest(record["final_state"]) != record["final_state_sha256"]:
            raise ValueError("Fresh source final hash mismatch")
        if record["reason"] == "completed":
            if not state.terminal or state.winner != record["winner"]:
                raise ValueError("Fresh source terminal outcome mismatch")
        elif record["reason"] == "forced_pass_deadlock":
            if len(states) < 2 or not all(is_forced_pass_turn(env.legal_actions(s)) for s in states[-2:]):
                raise ValueError("Fresh source deadlock not forced")
            if adjudicate_scores(state) != record["winner"]:
                raise ValueError("Fresh source deadlock outcome mismatch")
        else:
            raise ValueError("Fresh source did not terminate naturally")
    return states


def place_source(game, cells, assigned, quota, visited=None):
    """Capacity-constrained augmenting path; one source game per chosen state."""
    visited = set() if visited is None else visited
    for cell in cells[game]:
        if cell in visited:
            continue
        visited.add(cell)
        if len(assigned[cell]) < quota:
            assigned[cell].append(game)
            return True
        for k, previous in enumerate(assigned[cell]):
            if place_source(previous, cells, assigned, quota, visited):
                assigned[cell][k] = game
                return True
    return False


def build_panel(split, root=OUTPUT, audit=False):
    """First feasible balanced matching over eligible games; no outcome selection."""
    root = Path(root)
    old = split in ("train", "validation")
    indices = range(0, 4096) if split == "train" else range(4096, 4608) if split == "validation" else range(CONFIG["source_limit"][split])
    quota = 1 if split == "probe" else CONFIG["states"][split] // 6
    assigned, cells, best_states, filtering = [[] for _ in range(6)], {}, {}, []
    sealed = read_json(CORPUS / "development_summary.json")["files"] if old else None
    for index in indices:
        path = CORPUS / "games" / f"game_{index:04d}.json" if old else root / "sources" / split / f"game_{index:04d}.json"
        if old:
            if file_hash(path) != sealed[str(index)]:
                raise ValueError("143 development source changed")
            record = read_json(path)
        elif audit:
            record = read_json(path)
        else:
            seed = CONFIG["fresh_source_seed_start"][split] + index
            game = trajectory(SplendorEnv(seed=seed).initial_state(), include_trace=True)
            record = {"seed": seed, "index": index, **game}
            save_new(path, record)
        if (old and not record["valid_natural"]) or (not old and record["reason"] not in VALID_NATURAL_TERMINATIONS):
            raise RuntimeError("Non-natural source: stop rather than replace it")
        states = reconstruct_source(record, verify_raw=audit and not old)
        choices = [[s for s in states if eligible(s, cell)] for cell in range(6)]
        cells[index] = [c for c in range(6) if choices[c]]
        best_states[index] = {c: min(choices[c], key=lambda s: digest(["149-position-v1", split, index, s.turn_index])) for c in cells[index]}
        filtering.append({"source_index": index, "eligible_by_cell": [len(c) for c in choices],
                          "source_path": path.as_posix(), "source_sha256": file_hash(path)})
        place_source(index, cells, assigned, quota)
        if all(len(group) == quota for group in assigned):
            break
    counts = [len(group) for group in assigned]
    if counts != [quota] * 6:
        raise RuntimeError(f"Fixed source schedule exhausted: {split} {counts}")
    game_cell = {game: cell for cell, group in enumerate(assigned) for game in group}
    if len(game_cell) != quota * 6:
        raise ValueError("Source-game reuse in matching")
    samples = []
    seed_start = 1_430_000_000 if old else CONFIG["fresh_source_seed_start"][split]
    for index, cell in sorted(game_cell.items()):
        state = best_states[index][cell]
        samples.append({"index": len(samples), "source_index": index, "source_seed": seed_start + index,
                        "cell": cell, "turn": state.turn_index, "actor": state.current_player,
                        "snapshot": serialize_state_snapshot(state), **candidates(state, split, index)})
    return {"split": split, "counts": counts, "filtering": filtering, "samples": samples}


def feature_data(panel):
    arrays = {arm: [] for arm in ARMS}
    mask = []
    for sample in panel["samples"]:
        state = deserialize_state_snapshot(sample["snapshot"])
        actions = [decode_action(p) for p in sample["actions"]]
        for arm in ARMS:
            row = [features(state, a, arm) for a in actions]
            arrays[arm].append(torch.stack(row + [row[0]] * (4 - len(row))))
        mask.append([i < len(actions) and i > 0 for i in range(4)])
    return {**{arm: torch.stack(rows) for arm, rows in arrays.items()},
            "mask": torch.tensor(mask, dtype=torch.bool)}


def world_seed(split, sample_index, world):
    return CONFIG["world_seed_start"][split] + sample_index * 1000 + world


def evaluate_sample(split, sample, worlds=None):
    n = CONFIG["worlds"][split] if worlds is None else worlds
    state = deserialize_state_snapshot(sample["snapshot"])
    actions = [decode_action(p) for p in sample["actions"]]
    rows = []
    for world_index in range(n):
        seed = world_seed(split, sample["index"], world_index)
        world = determinize_for_observer(state, state.current_player, random.Random(seed))
        outcomes = []
        for action in actions:
            result = trajectory(world, action)
            valid = result["reason"] in VALID_NATURAL_TERMINATIONS
            result["wdl"] = outcome_value(result["winner"], state.current_player) if valid else None
            outcomes.append(result)
        rows.append({"seed": seed, "world_sha256": digest(serialize_state_snapshot(world)), "outcomes": outcomes})
    return {"split": split, "index": sample["index"], "sample_sha256": digest(sample), "worlds": rows}


def collect_task(split, sample, root):
    record = evaluate_sample(split, sample)
    path = Path(root) / "targets" / split / f"state_{sample['index']:04d}.json"
    save_new(path, record)
    return {"index": sample["index"], "sha256": file_hash(path),
            "natural": all(o["wdl"] is not None for w in record["worlds"] for o in w["outcomes"])}


def collect_panel(panel):
    split = panel["split"]
    files = {}
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker) as pool:
        jobs = [pool.submit(collect_task, split, sample, OUTPUT) for sample in panel["samples"]]
        for done, job in enumerate(as_completed(jobs), 1):
            row = job.result()
            files[str(row["index"])] = row
            if done % 8 == 0 or done == len(jobs):
                print({"stage": split + "_targets", "states": done, "total": len(jobs)}, flush=True)
    summary = {"files": files, "all_natural": all(r["natural"] for r in files.values())}
    save_new(OUTPUT / f"{split}_targets_lock.json", summary)
    if not summary["all_natural"]:
        raise RuntimeError("Non-natural continuations: preserve evidence; do not fit or advance")


def target_data(panel, root=OUTPUT):
    values = []
    split = panel["split"]
    lock = read_json(Path(root) / f"{split}_targets_lock.json")
    for sample in panel["samples"]:
        path = Path(root) / "targets" / split / f"state_{sample['index']:04d}.json"
        if file_hash(path) != lock["files"][str(sample["index"])]["sha256"]:
            raise ValueError("Target changed")
        record = read_json(path)
        if record["sample_sha256"] != digest(sample):
            raise ValueError("Target/sample binding mismatch")
        row = torch.tensor([[o["wdl"] for o in w["outcomes"]] for w in record["worlds"]], dtype=torch.float32).mean(0)
        values.append(torch.cat([row, row[:1].repeat(4-len(row))]))
    q = torch.stack(values)
    return q, q - q[:, :1]


def state_loss(pred, y, mask):
    count = mask.sum(-1)
    return ((pred-y).square() * mask).sum(-1) / count.clamp_min(1)


def fit_model(arm, seed, training, validation, *, updates=None):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = AdvantageModel(training[arm].shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator().manual_seed(seed + 149000)
    best, best_update, weights, history = float("inf"), 0, None, []
    total = CONFIG["updates"] if updates is None else updates
    for update in range(total + 1):
        if update % CONFIG["validation_interval"] == 0 or update == total:
            model.eval()
            with torch.no_grad():
                val = float(state_loss(model(validation[arm]), validation["y"], validation["mask"]).mean())
                train = float(state_loss(model(training[arm]), training["y"], training["mask"]).mean())
            if not np.isfinite(val):
                raise ValueError("Nonfinite fit")
            history.append({"update": update, "validation_mse": val, "training_mse": train})
            if val < best:
                best, best_update, weights = val, update, copy.deepcopy(model.state_dict())
        if update == total:
            break
        model.train()
        indices = torch.randint(len(training["y"]), (CONFIG["batch_states"],), generator=generator)
        pred = model(training[arm][indices])
        loss = state_loss(pred, training["y"][indices], training["mask"][indices]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_norm"], error_if_nonfinite=True)
        optimizer.step()
    return {"arm": arm, "seed": seed, "width": training[arm].shape[-1], "selected_update": best_update,
            "selected_state": weights, "final_state": model.state_dict(), "history": history,
            "parameters": sum(p.numel() for p in model.parameters())}


def fit_task(arm, seed, training, validation):
    started = time.perf_counter()
    result = fit_model(arm, seed, training, validation)
    path = OUTPUT / "models" / f"{arm}_{seed}.pt"
    tensor_save(path, result)
    return {"path": path.as_posix(), "sha256": file_hash(path), "selected_update": result["selected_update"],
            "seconds": time.perf_counter()-started, "parameters": result["parameters"]}


def test_predictions(data, selection):
    result = {}
    for arm in ARMS:
        rows = []
        for seed in CONFIG["training_seeds"]:
            info = selection["models"][f"{arm}_{seed}"]
            path = Path(info["path"])
            if file_hash(path) != info["sha256"]:
                raise ValueError("Selected model changed")
            saved = torch.load(path, weights_only=True)
            model = AdvantageModel(saved["width"])
            model.load_state_dict(saved["selected_state"])
            model.eval()
            with torch.no_grad():
                prediction = model(data[arm])
                legal = data["mask"].clone()
                legal[:, 0] = True
                choice = prediction.masked_fill(~legal, -torch.inf).argmax(-1)
            rows.append({"seed": seed, "advantages": prediction.tolist(), "choices": choice.tolist()})
        result[arm] = rows
    return result


def statistics(q, y, mask, predictions):
    q, y, mask = q.numpy().astype(np.float64), y.numpy().astype(np.float64), mask.numpy()
    picked, errors = {}, {}
    for arm in ARMS:
        picked[arm] = np.stack([q[np.arange(len(q)), row["choices"]] for row in predictions[arm]])
        errors[arm] = np.stack([(((np.asarray(row["advantages"])-y)**2 * mask).sum(-1) / np.maximum(mask.sum(-1), 1))
                               for row in predictions[arm]])
    contrasts = {"afterstate_minus_control": (picked["afterstate"]-picked["control"]).mean(0),
                 "afterstate_minus_baseline": (picked["afterstate"]-q[:, 0]).mean(0),
                 "control_minus_baseline": (picked["control"]-q[:, 0]).mean(0),
                 "mse_afterstate_minus_control": (errors["afterstate"]-errors["control"]).mean(0)}
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    matrix = np.stack(list(contrasts.values()), -1)
    boot = np.concatenate([matrix[rng.integers(len(q), size=(1000, len(q)))].mean(1)
                           for _ in range(CONFIG["bootstrap_samples"] // 1000)])
    out = {}
    for j, name in enumerate(contrasts):
        interval = np.quantile(boot[:, j], [.025, .975]).tolist()
        out[name] = {"mean": float(matrix[:, j].mean()), "ci95": interval,
                     "half_width": (interval[1]-interval[0])/2}
    gates = {name: out[name]["mean"] >= .02 and out[name]["ci95"][0] > 0 and out[name]["half_width"] <= .02
             for name in ("afterstate_minus_control", "afterstate_minus_baseline")}
    gates["mse"] = out["mse_afterstate_minus_control"]["ci95"][1] < 0
    return {"contrasts": out, "statistical_gates": gates, "statistical_pass": all(gates.values()),
            "mse": {a: float(errors[a].mean()) for a in ARMS},
            "per_seed": [{"seed": seed, **{a + "_minus_baseline": float((picked[a][i]-q[:, 0]).mean()) for a in ARMS},
                          "afterstate_minus_control": float((picked["afterstate"][i]-picked["control"][i]).mean())}
                         for i, seed in enumerate(CONFIG["training_seeds"])],
            "selected_baseline_fraction": {a: [float(np.mean(np.asarray(r["choices"]) == 0)) for r in predictions[a]] for a in ARMS}}


def identity():
    if file_hash(PRIOR) != PRIOR_SHA or file_hash(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Prior manifest or 035 changed")
    prior = read_json(PRIOR)
    for field in ("dependencies", "inputs"):
        for name, sha in prior[field].items():
            if file_hash(Path(name)) != sha:
                raise ValueError(f"Prior frozen {field} changed: {name}")
    files = sorted(Path("src/splendor_ai").rglob("*.py")) + [PROTOCOL, VERIFIER,
              Path("run_afterstate_advantage_149.py"), Path("tests/test_afterstate_advantage_149.py")]
    inputs = [PRIOR, CHECKPOINT, CORPUS / "manifest.json", CORPUS / "development_summary.json"]
    return {"format": "afterstate_advantage_149_v1", "config": CONFIG,
            "dependencies": {p.as_posix(): file_hash(p) for p in files},
            "inputs": {p.as_posix(): file_hash(p) for p in inputs},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def check_identity():
    manifest = read_json(OUTPUT / "manifest.json")
    if identity() != manifest:
        raise ValueError("149 frozen identity changed")
    return digest(manifest)


def preflight():
    save_new(OUTPUT / "manifest.json", identity())
    initialize_worker()
    started = time.perf_counter()
    panel = build_panel("probe")
    save_new(OUTPUT / "probe_panel.json", panel)
    if digest(build_panel("probe", audit=True)) != digest(panel):
        raise ValueError("Probe source reconstruction failed")
    development = {}
    for split in ("train", "validation"):
        source_panel = build_panel(split)
        save_new(OUTPUT / f"{split}_panel.json", source_panel)
        tensor_save(OUTPUT / f"{split}_features.pt", feature_data(source_panel))
        development[split] = {"panel": file_hash(OUTPUT / f"{split}_panel.json"),
                              "features": file_hash(OUTPUT / f"{split}_features.pt")}
        print({"preflight": "development_panel_filled", "split": split, "states": len(source_panel["samples"]),
               "cells": source_panel["counts"]}, flush=True)
    save_new(OUTPUT / "development_lock.json", development)
    # Validate features and exact composition before collecting any timing paths.
    comparisons = 0
    for sample in panel["samples"]:
        state = deserialize_state_snapshot(sample["snapshot"])
        for action in SplendorEnv().legal_actions(state):
            for k in range(3):
                world = determinize_for_observer(state, state.current_player, random.Random(149900+k))
                after = deterministic_part(world, action)
                card = None if after.chance_kind is None else world.hidden_tier_decks[after.chance_tier][0].card_id
                if serialize_state_snapshot(after.resolve(card)) != serialize_state_snapshot(SplendorEnv().step(world, action)):
                    raise ValueError("Composition failed")
                for arm in ARMS:
                    if not torch.equal(features(world, action, arm), features(state, action, arm)):
                        raise ValueError("Hidden-world leakage")
                comparisons += 1
    start_targets = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6, initializer=initialize_worker) as pool:
        results = list(pool.map(probe_task, panel["samples"]))
    elapsed = time.perf_counter()-start_targets
    # Do not publish or use timing-path WDL gains or model comparisons.
    count = sum(r["continuations"] for r in results)
    result = {"complete": True, "composition_comparisons": comparisons, "continuations": count,
              "target_seconds": elapsed, "total_seconds": time.perf_counter()-started,
              "all_natural": all(r["natural"] for r in results),
              "maximum_main_continuations": 245760,
              "rough_collection_hours_at_probe_throughput": elapsed / count * 245760 / 3600,
              "note": "Six-worker small-sample throughput includes startup; full audit repeats continuations. No outcome gains inspected."}
    save_new(OUTPUT / "preflight.json", result)
    if not result["all_natural"]:
        raise RuntimeError("Preflight non-natural continuation")
    print(result, flush=True)


def probe_task(sample):
    record = evaluate_sample("probe", sample, worlds=4)
    return {"continuations": sum(len(w["outcomes"]) for w in record["worlds"]),
            "natural": all(o["wdl"] is not None for w in record["worlds"] for o in w["outcomes"])}


def run():
    if (OUTPUT / "launch.json").exists():
        raise RuntimeError("This bounded pilot has already launched; no automatic restart")
    binding = check_identity()
    if not read_json(OUTPUT / "preflight.json")["all_natural"]:
        raise ValueError("Preflight did not pass")
    started = time.perf_counter()
    save_new(OUTPUT / "launch.json", {"manifest": binding, "started": now()})
    initialize_worker()
    development = read_json(OUTPUT / "development_lock.json")
    for split in ("train", "validation"):
        for kind, suffix in (("panel", "json"), ("features", "pt")):
            if file_hash(OUTPUT / f"{split}_{kind}.{suffix}") != development[split][kind]:
                raise ValueError("Preflight development panel or features changed")
    data = {}
    for split in ("train", "validation"):
        panel = read_json(OUTPUT / f"{split}_panel.json")
        collect_panel(panel)
        data[split] = torch.load(OUTPUT / f"{split}_features.pt", weights_only=True)
        _, data[split]["y"] = target_data(panel)
    models = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(fit_task, arm, seed, data["train"], data["validation"]): f"{arm}_{seed}"
                for seed in CONFIG["training_seeds"] for arm in ARMS}
        for job in as_completed(jobs):
            models[jobs[job]] = job.result()
            print({"stage": "fit", "model": jobs[job], **models[jobs[job]]}, flush=True)
    selection = {"manifest": binding, "locked_at": now(), "models": models,
                 "development_lock": file_hash(OUTPUT / "development_lock.json"),
                 "targets": {s: file_hash(OUTPUT / f"{s}_targets_lock.json") for s in ("train", "validation")}}
    save_new(OUTPUT / "selection_lock.json", selection)
    check_identity()
    panel = build_panel("test")
    save_new(OUTPUT / "test_panel.json", panel)
    test = feature_data(panel)
    tensor_save(OUTPUT / "test_features.pt", test)
    predictions = test_predictions(test, selection)
    save_new(OUTPUT / "test_choices_lock.json", {"locked_at": now(), "selection_sha256": file_hash(OUTPUT / "selection_lock.json"),
             "panel_sha256": file_hash(OUTPUT / "test_panel.json"), "features_sha256": file_hash(OUTPUT / "test_features.pt"), "predictions": predictions})
    collect_panel(panel)
    q, y = target_data(panel)
    result = statistics(q, y, test["mask"], predictions)
    save_new(OUTPUT / "result.json", result)
    print({"stage": "collection_and_fits_complete", "seconds": time.perf_counter()-started, "statistics": result}, flush=True)
    subprocess.run([sys.executable, str(VERIFIER)], check=True)
    audit = read_json(OUTPUT / "audit.json")
    if not audit["passed"]:
        raise RuntimeError("Independent audit failed")
    check_identity()
    save_new(OUTPUT / "final_status.json", {"complete": True, "finished": now(), "seconds": time.perf_counter()-started,
             "advance_gate_passed": result["statistical_pass"], "result_sha256": file_hash(OUTPUT / "result.json"),
             "audit_sha256": file_hash(OUTPUT / "audit.json"), "arena": False, "promotion": False, "automatic_successor": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    args = parser.parse_args()
    with process_lock(OUTPUT / "run.lock"):
        try:
            preflight() if args.mode == "preflight" else run()
        except Exception:
            atomic_write(OUTPUT / f"failure_{args.mode}.json", {"time": now(), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
