"""Frozen, resumable, one-generation strong-policy learning experiment 142."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from splendor_ai.alphazero.search import SearchConfig
from splendor_ai.alphazero.strong_policy import (
    BudgetSearch, StrongPolicyInference, encode_position, grouped_logits, save_model,
)
from splendor_ai.diagnostics import is_progress_transition, state_signature
from splendor_ai.encoding import canonical_successor_action_key
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS, adjudicate_scores, is_forced_pass_turn, outcome_value
from splendor_ai.training.replay import serialize_action, serialize_state_snapshot

OUTPUT = Path("outputs/strong_policy_pilot_142")
PROTOCOL = Path("docs/engineering/strong_policy_pilot_142.md")
CHECKPOINT = Path("outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt")
CHECKPOINT_SHA = "4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd"
CONFIG = {
    "bootstrap_games": 256, "selfplay_games": 128, "arena_pairs": 200,
    "workers": 8, "torch_threads": 1, "simulations": 128, "decision_seconds": 0.4,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "warmup_epochs": 8, "warmup_lr": 0.001, "warmup_batch": 128,
    "joint_epochs": 4, "joint_lr": 0.00001, "joint_batch": 64,
    "weight_decay": 0.0001, "validation_fraction": 0.2,
    "bootstrap_seed": 1_420_000_000, "selfplay_seed": 1_421_000_000,
    "arena_seed": 1_422_000_000, "probe_seed": 1_429_000_000,
    "train_seed": 142_010, "bootstrap_resamples": 50_000, "bootstrap_stat_seed": 142_999,
}
SEARCH = SearchConfig(simulations=CONFIG["simulations"], c_puct=1.5, max_depth=96,
                      mode="real", root_noise_fraction=0.25, temperature_turns=20, temperature=1.0, seed=142)
_MODELS = []


def identity():
    if file_hash(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("035 source checkpoint changed")
    paths = sorted(Path("src/splendor_ai").rglob("*.py")) + [
        Path("run_strong_policy_pilot_142.py"), Path("tests/test_strong_policy_pilot_142.py"),
        Path("scripts/analysis/verify_strong_policy_pilot_142.py"), PROTOCOL,
    ]
    return {"format": "strong_policy_pilot_142_v1", "config": CONFIG, "search": asdict(SEARCH),
            "source_checkpoint": CHECKPOINT_SHA,
            "dependencies": {p.as_posix(): file_hash(p) for p in paths},
            "python": __import__("sys").version, "torch": str(torch.__version__), "numpy": np.__version__}


def freeze():
    current = identity()
    path = OUTPUT / "manifest.json"
    if path.exists():
        if read_json(path) != current:
            raise ValueError("Frozen experiment identity changed")
    else:
        atomic_write(path, current)
    return current


def initialize_worker(paths):
    global _MODELS
    torch.set_num_threads(1)
    _MODELS = [StrongPolicyInference(path) for path in paths]


def _tensor_save(path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def play_game(stage, index, input_hash, output_root=None):
    """Save a complete action trace and, for learning, observer-legal tensors."""
    root = OUTPUT if output_root is None else Path(output_root)
    arena = stage == "arena"
    seed = CONFIG[f"{stage}_seed"] + (index // 2 if arena else index)
    candidate_seat = index % 2 if arena else None
    env = SplendorEnv(seed=seed)
    state = env.initial_state()
    rng = random.Random(seed + 142_003)
    visits = Counter()
    forced = progress = 0
    trace, rows = [], []
    reason = "completed"
    searches = [BudgetSearch(model, SEARCH) for model in _MODELS]
    while not state.terminal:
        visits[state_signature(state)] += 1
        if visits[state_signature(state)] >= CONFIG["repetition_limit"]:
            reason = "repetition_cutoff"
            break
        if state.turn_index >= CONFIG["max_turns"]:
            reason = "max_turns"
            break
        actions = env.legal_actions(state)
        if not actions:
            reason = "stalled"
            break
        actor = state.current_player
        model_index = int(actor == candidate_seat) if arena else 0
        model = _MODELS[model_index]
        search_seed = seed * 1_000_003 + state.turn_index * 97_409 + 142
        record = {"turn": state.turn_index, "actor": actor, "model_index": model_index,
                  "state_hash": digest(serialize_state_snapshot(state))}
        cpu_start = time.process_time()
        started = time.perf_counter()
        if stage == "bootstrap":
            logits, _, row, keys, _ = model.full_prediction(state, actions)
            # The existing 035 policy, with a fixed opening-only top-three mixture.
            best = int(logits.argmax())
            chosen = best
            if state.turn_index < 20 and rng.random() < 0.2:
                top = sorted(range(len(actions)), key=lambda j: (-float(logits[j]), j))[:3]
                weights = torch.softmax(logits[top] / 0.5, 0).tolist()
                chosen = rng.choices(top, weights=weights, k=1)[0]
            action = actions[chosen]
            record["raw_policy_row"] = best
            record["selected_row"] = chosen
        else:
            result = searches[model_index].run(env, state, actions, self_play=not arena,
                                              seed=search_seed, seconds=CONFIG["decision_seconds"] if arena else None)
            action = result.action
            record.update({"simulations": result.simulations, "expansions": result.expansions,
                           "max_depth": result.max_depth_reached, "search_seed": search_seed,
                           "raw_value": result.raw_network_value,
                           "root_visits": [edge.visits for edge in result.root_edges],
                           "root_keys": [repr(edge.action_key) for edge in result.root_edges],
                           "selected_key": repr(canonical_successor_action_key(state, action))})
        record["decision_seconds"] = time.perf_counter() - started
        record["cpu_seconds"] = time.process_time() - cpu_start
        record["action"] = serialize_action(action)
        if not arena:
            if stage != "bootstrap":
                row, keys, _ = encode_position(state, model.model.config, actions)
                probs = {edge.action_key: edge.probability for edge in result.root_edges}
                row["policy_target"] = torch.tensor([probs[key] for key in keys], dtype=torch.float32)
                record["group_keys"] = [repr(key) for key in keys]
            row["actor"] = actor
            row["turn"] = state.turn_index
            rows.append(row)
        trace.append(record)
        next_state = env.step(state, action)
        forced = forced + 1 if is_forced_pass_turn(actions) else 0
        progress = 0 if is_progress_transition(state, next_state, action) else progress + 1
        state = next_state
        if forced >= 2:
            reason = "forced_pass_deadlock"
            break
        if progress >= CONFIG["no_progress_limit"]:
            reason = "no_progress_cutoff"
            break
    valid = reason in VALID_NATURAL_TERMINATIONS
    winner = state.winner if state.terminal else adjudicate_scores(state) if valid else None
    for row in rows:
        row["value_target"] = outcome_value(winner, row["actor"]) if valid else None
    result_record = {"stage": stage, "index": index, "seed": seed, "input_hash": input_hash,
                     "candidate_seat": candidate_seat, "valid_natural": valid,
                     "winner": winner, "termination_reason": reason, "positions": len(trace),
                     "final_state": serialize_state_snapshot(state), "trace": trace}
    if arena:
        result_record["candidate_points"] = (0.5 if winner is None else float(winner == candidate_seat))
    else:
        tensor_path = root / stage / f"game_{index:04d}.pt"
        _tensor_save(tensor_path, {"stage": stage, "index": index, "seed": seed,
                                   "input_hash": input_hash, "valid_natural": valid, "rows": rows})
        result_record["tensor_sha256"] = file_hash(tensor_path)
    atomic_write(root / stage / f"game_{index:04d}.json", result_record)
    # No partial winners or losses in progress output.
    return index


def run_games(stage, paths, count, manifest):
    for path in paths:
        if Path(path) != CHECKPOINT:
            report_name = "bootstrap_training.json" if Path(path).name == "parent.pt" else "selfplay_training.json"
            report = read_json(OUTPUT / report_name)
            if report["manifest"] != digest(manifest) or report["output_sha256"] != file_hash(Path(path)):
                raise ValueError("Search input checkpoint does not match its training record")
    binding = {"manifest": digest(manifest), "models": [file_hash(Path(p)) for p in paths], "stage": stage}
    input_hash = digest(binding)
    directory = OUTPUT / stage
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    for index in range(count):
        path = directory / f"game_{index:04d}.json"
        if not path.exists():
            if path.with_suffix(".pt").exists():
                raise RuntimeError("Orphan replay tensor; inspect interrupted atomic game before resuming")
            pending.append(index)
        else:
            record = read_json(path)
            if record["input_hash"] != input_hash or record["index"] != index:
                raise ValueError("Existing game belongs to a different run")
            if stage != "arena" and record["tensor_sha256"] != file_hash(path.with_suffix(".pt")):
                raise ValueError("Replay tensor changed")
    if pending:
        with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker, initargs=(paths,)) as pool:
            futures = [pool.submit(play_game, stage, index, input_hash) for index in pending]
            for completed, future in enumerate(as_completed(futures), start=count - len(pending) + 1):
                future.result()
                if completed % 8 == 0 or completed == count:
                    print(json.dumps({"stage": stage, "completed": completed, "total": count}), flush=True)
    records = [read_json(directory / f"game_{i:04d}.json") for i in range(count)]
    summary = {"binding": binding, "games": count, "valid_games": sum(r["valid_natural"] for r in records),
               "positions": sum(r["positions"] for r in records),
               "termination_counts": dict(Counter(r["termination_reason"] for r in records)),
               "files": {f"game_{i:04d}.json": file_hash(directory / f"game_{i:04d}.json") for i in range(count)}}
    atomic_write(directory / "summary.json", summary)
    if stage != "arena" and summary["valid_games"] < math.ceil(0.98 * count):
        raise RuntimeError(f"{stage}: more than 2% non-natural games; stop before training")
    return summary


def split_games(count, seed):
    ids = list(range(count))
    random.Random(seed).shuffle(ids)
    validation_count = max(1, round(count * CONFIG["validation_fraction"]))
    return sorted(ids[validation_count:]), sorted(ids[:validation_count])


def load_rows(stage, indices):
    rows = []
    for index in indices:
        payload = torch.load(OUTPUT / stage / f"game_{index:04d}.pt", weights_only=True)
        if payload["valid_natural"]:
            rows.extend(payload["rows"])
    return rows


def collate(rows):
    count = max(len(row["features"]) for row in rows)
    obs = torch.stack([row["observation"] for row in rows])
    features = torch.zeros(len(rows), count, rows[0]["features"].shape[-1])
    mask = torch.zeros(len(rows), count, dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row["features"])
        features[i, :n] = row["features"]
        mask[i, :n] = True
    return obs, features, mask


def joint_loss(model, rows):
    logits, values = model(*collate(rows))
    terms = []
    for i, row in enumerate(rows):
        scores = grouped_logits(logits[i, :len(row["groups"])], row["groups"], row["group_count"])
        terms.append(-(row["policy_target"] * torch.log_softmax(scores, 0)).sum())
    policy = torch.stack(terms).mean()
    value = torch.nn.functional.mse_loss(values, torch.tensor([row["value_target"] for row in rows]))
    return policy + value, policy, value


def policy_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("value_head."):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train(stage, source, destination, manifest):
    report_path = OUTPUT / f"{stage}_training.json"
    if destination.exists():
        report = read_json(report_path)
        if report["manifest"] != digest(manifest) or report["output_sha256"] != file_hash(destination):
            raise ValueError("Existing trained checkpoint identity mismatch")
        return report
    warmup = stage == "bootstrap"
    summary = read_json(OUTPUT / stage / "summary.json")
    if summary["binding"]["manifest"] != digest(manifest):
        raise ValueError("Training corpus manifest mismatch")
    if summary["valid_games"] < math.ceil(.98 * CONFIG[f"{stage}_games"]):
        raise ValueError("Training corpus failed completion gate")
    for name, sha in summary["files"].items():
        path = OUTPUT / stage / name
        if file_hash(path) != sha or file_hash(path.with_suffix(".pt")) != read_json(path)["tensor_sha256"]:
            raise ValueError("Training corpus changed")
    torch.set_num_threads(1)
    seed = CONFIG["train_seed"] + (0 if warmup else 1)
    torch.manual_seed(seed)
    random.seed(seed)
    model = StrongPolicyInference(source).model
    before_policy = policy_hash(model)
    count = CONFIG[f"{stage}_games"]
    train_ids, val_ids = split_games(count, seed)
    training, validation = load_rows(stage, train_ids), load_rows(stage, val_ids)
    if not training or not validation or (not warmup and len(training) + len(validation) < 5000):
        raise RuntimeError("Insufficient valid learning dose")
    if warmup:
        # An untrained inherited value head has no privileged status; start at zero.
        torch.nn.init.zeros_(model.value_head[-2].weight)
        torch.nn.init.zeros_(model.value_head[-2].bias)
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("value_head."))
        with torch.no_grad():
            def hidden(rows):
                return torch.cat([model.state_encoder(torch.stack([r["observation"] for r in rows[i:i+256]]))
                                  for i in range(0, len(rows), 256)])
            train_x, val_x = hidden(training), hidden(validation)
        train_y = torch.tensor([r["value_target"] for r in training])
        val_y = torch.tensor([r["value_target"] for r in validation])
    else:
        for param in model.parameters():
            param.requires_grad_(True)
    batch = CONFIG["warmup_batch" if warmup else "joint_batch"]
    epochs = CONFIG["warmup_epochs" if warmup else "joint_epochs"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=CONFIG["warmup_lr" if warmup else "joint_lr"], weight_decay=CONFIG["weight_decay"])
    def evaluate():
        model.eval()
        with torch.no_grad():
            if warmup:
                mse = float(torch.nn.functional.mse_loss(model.value_head(val_x).squeeze(-1), val_y))
                return {"total": mse, "value": mse}
            totals = np.zeros(3)
            for start in range(0, len(validation), batch):
                part = validation[start:start+batch]
                totals += np.array([float(x) for x in joint_loss(model, part)]) * len(part)
            return dict(zip(("total", "policy", "value"), (totals / len(validation)).tolist()))
    history = [{"epoch": 0, "validation": evaluate()}]
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_rng = random.Random(seed + epoch)
        order = list(range(len(training)))
        epoch_rng.shuffle(order)
        if not warmup:
            order.sort(key=lambda i: len(training[i]["features"]))
        batches = [order[start:start+batch] for start in range(0, len(order), batch)]
        epoch_rng.shuffle(batches)
        losses = []
        for ids in batches:
            optimizer.zero_grad(set_to_none=True)
            if warmup:
                loss = torch.nn.functional.mse_loss(model.value_head(train_x[ids]).squeeze(-1), train_y[ids])
            else:
                loss, _, _ = joint_loss(model, [training[i] for i in ids])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradients")
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "training_mean_batch_loss": float(np.mean(losses)), "validation": evaluate()})
        print(json.dumps({"stage": f"{stage}_training", "epoch": epoch, "total_epochs": epochs}), flush=True)
    after_policy = policy_hash(model)
    if warmup and (before_policy != after_policy or history[-1]["validation"]["value"] >= history[0]["validation"]["value"]):
        raise RuntimeError("Value bootstrap failed retention or held-out improvement gate")
    report = {"manifest": digest(manifest), "source_sha256": file_hash(Path(source)),
              "data_summary_sha256": file_hash(OUTPUT / stage / "summary.json"),
              "train_game_indices": train_ids, "validation_game_indices": val_ids,
              "training_positions": len(training), "validation_positions": len(validation),
              "history": history, "before_policy_sha256": before_policy, "after_policy_sha256": after_policy,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "wall_seconds": time.perf_counter() - started, "selection": "fixed_final_epoch"}
    model.eval()
    save_model(destination, model, 0 if warmup else 1, report)
    report["output_sha256"] = file_hash(destination)
    atomic_write(report_path, report)
    return report


def arena_summary(records):
    if len(records) != 2 * CONFIG["arena_pairs"]:
        raise ValueError("Arena incomplete")
    records = sorted(records, key=lambda r: r["index"])
    points = []
    for i in range(CONFIG["arena_pairs"]):
        pair = records[2*i:2*i+2]
        if [r["index"] for r in pair] != [2*i, 2*i+1] or [r["candidate_seat"] for r in pair] != [0, 1] or pair[0]["seed"] != pair[1]["seed"]:
            raise ValueError("Broken arena seat pair")
        points.append(np.mean([r["candidate_points"] for r in pair]))
    arr = np.array(points)
    rng = np.random.default_rng(CONFIG["bootstrap_stat_seed"])
    boot = np.empty(CONFIG["bootstrap_resamples"])
    for i in range(0, len(boot), 1000):
        n = min(1000, len(boot)-i)
        boot[i:i+n] = arr[rng.integers(0, len(arr), (n, len(arr)))].mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()
    timing = {}
    for index, name in enumerate(("parent", "successor")):
        moves = [step for r in records for step in r["trace"] if step["model_index"] == index]
        elapsed = [s["decision_seconds"] for s in moves]
        timing[name] = {"moves": len(moves), "mean_seconds": float(np.mean(elapsed)),
                        "p95_seconds": float(np.quantile(elapsed, .95)), "max_seconds": max(elapsed),
                        "mean_simulations": float(np.mean([s["simulations"] for s in moves])),
                        "mean_depth": float(np.mean([s["max_depth"] for s in moves])),
                        "total_cpu_seconds": sum(s["cpu_seconds"] for s in moves)}
    ratio = timing["successor"]["mean_seconds"] / timing["parent"]["mean_seconds"]
    natural = all(r["valid_natural"] for r in records)
    timing_ok = .95 <= ratio <= 1.05 and all(CONFIG["decision_seconds"] <= v["mean_seconds"] <= .5 for v in timing.values())
    mean = float(arr.mean())
    decision = ("invalid_integrity_or_time_match" if not (natural and timing_ok) else
                "material_improvement" if ci[0] > .5 and mean >= .55 else
                "positive_but_below_material_point_estimate" if ci[0] > .5 else
                "regression" if ci[1] < .5 else "improvement_not_established")
    counts = Counter("win" if r["candidate_points"] == 1 else "loss" if r["candidate_points"] == 0 else "draw" for r in records)
    return {"candidate_score_share": mean, "paired_ci95": ci, "wdl": dict(counts),
            "decision": decision, "natural_termination_gate": natural, "time_match_gate": timing_ok,
            "successor_parent_mean_time_ratio": ratio, "timing": timing,
            "non_natural_games": sum(not r["valid_natural"] for r in records),
            "by_candidate_seat": {str(seat): float(np.mean([r["candidate_points"] for r in records if r["candidate_seat"] == seat])) for seat in (0, 1)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "bootstrap", "warmup", "selfplay", "train", "arena", "summarize", "all"))
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with process_lock(OUTPUT / "process.lock"):
        manifest = freeze()
        if args.stage != "freeze":
            preflight = read_json(OUTPUT / "preflight.json")
            if preflight["manifest"] != digest(manifest) or preflight["status"] != "passed":
                raise ValueError("Matching successful preflight is required before execution")
        parent, successor = OUTPUT / "parent.pt", OUTPUT / "successor.pt"
        def selected(name):
            return args.stage in (name, "all")
        if selected("bootstrap"):
            run_games("bootstrap", [str(CHECKPOINT)], CONFIG["bootstrap_games"], manifest)
        if selected("warmup"):
            train("bootstrap", CHECKPOINT, parent, manifest)
        if selected("selfplay"):
            run_games("selfplay", [str(parent)], CONFIG["selfplay_games"], manifest)
        if selected("train"):
            train("selfplay", parent, successor, manifest)
        if selected("arena"):
            run_games("arena", [str(parent), str(successor)], 2 * CONFIG["arena_pairs"], manifest)
        if selected("summarize"):
            records = [read_json(OUTPUT / "arena" / f"game_{i:04d}.json") for i in range(2 * CONFIG["arena_pairs"])]
            result = {"manifest": digest(manifest), "arena_summary_sha256": file_hash(OUTPUT / "arena" / "summary.json"),
                      "parent_sha256": file_hash(parent), "successor_sha256": file_hash(successor),
                      **arena_summary(records)}
            atomic_write(OUTPUT / "result.json", result)
            print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
