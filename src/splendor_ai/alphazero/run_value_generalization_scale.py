"""Experiment 143: controlled value learning on 256 versus 4,096 fresh games."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from splendor_ai.alphazero.strong_policy import StrongPolicyInference, save_model
from splendor_ai.alphazero.run_strong_policy_pilot import CHECKPOINT, CHECKPOINT_SHA, policy_hash
from splendor_ai.engine.constants import TOKEN_COLORS
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.termination import VALID_NATURAL_TERMINATIONS, outcome_value
from splendor_ai.training.replay import collect_game_replay, serialize_state_snapshot

OUTPUT = Path("outputs/value_generalization_scale_143")
PROTOCOL = Path("docs/engineering/value_generalization_scale_143.md")
PRIOR_MANIFEST = Path("outputs/strong_policy_pilot_142/manifest.json")
CONFIG = {
    "training_games": 4096, "small_games": 256, "validation_games": 512, "test_games": 512,
    "seed_start": 1_430_000_000, "probe_seed": 1_439_000_000, "behavior_seed_offset": 143003,
    "opening_turns": 20, "epsilon": .2, "temperature": .5, "top_k": 3,
    "max_turns": 400, "repetition_limit": 4, "no_progress_limit": 60,
    "workers": 8, "torch_threads": 1, "training_seed": 143010,
    "batch_size": 128, "max_updates": 4096, "validation_interval": 128,
    "patience": 4, "min_improvement": .001, "learning_rate": .001,
    "weight_decay": .0001, "gradient_norm": 1.,
    "ridge_grid": [.001, .01, .1, 1., 10.],
    "bootstrap_samples": 50000, "bootstrap_seed": 143999,
    "material_relative_mse_reduction": .05,
}
MODEL_NAMES = ("neutral", "small", "large", "simple")
CONTRASTS = (("large_minus_neutral", 2, 0), ("large_minus_small", 2, 1),
             ("large_minus_simple", 2, 3), ("simple_minus_neutral", 3, 0))
_INFERENCE = None


def split_for(index):
    if not 0 <= index < 5120:
        raise ValueError("Index outside frozen schedule")
    return "training" if index < 4096 else "validation" if index < 4608 else "test"


def small_indices():
    return sorted(sorted(range(4096), key=lambda i: hashlib.sha256(f"143-small-v1|{i}".encode()).hexdigest())[:256])


def identity():
    if file_hash(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("035 checkpoint changed")
    prior = read_json(PRIOR_MANIFEST)
    for name, sha in prior["dependencies"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Prior frozen dependency changed: {name}")
    files = sorted(Path("src/splendor_ai").rglob("*.py")) + [
        Path("run_value_generalization_scale_143.py"), Path("tests/test_value_generalization_scale_143.py"),
        Path("scripts/analysis/verify_value_generalization_scale_143.py"), PROTOCOL,
    ]
    return {"format": "value_generalization_scale_143_v1", "config": CONFIG,
            "source_checkpoint_sha256": CHECKPOINT_SHA, "prior_manifest_sha256": file_hash(PRIOR_MANIFEST),
            "small_training_indices": small_indices(),
            "dependencies": {p.as_posix(): file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


def freeze():
    current = identity()
    path = OUTPUT / "manifest.json"
    if path.exists():
        if read_json(path) != current:
            raise ValueError("143 frozen identity changed")
    else:
        atomic_write(path, current)
    return current


def tensor_save(path, payload):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def simple_features(state):
    """Fifteen public, actor-minus-opponent score/resource differences."""
    own, other = state.players[state.current_player], state.players[1-state.current_player]
    return torch.tensor(
        [(own.score-other.score)/15]
        + [(own.bonuses[c]-other.bonuses[c])/7 for c in TOKEN_COLORS]
        + [(own.tokens[c]-other.tokens[c])/7 for c in TOKEN_COLORS]
        + [(own.tokens["gold"]-other.tokens["gold"])/5,
           (len(own.purchased_cards)-len(other.purchased_cards))/20,
           (len(own.reserved_cards)-len(other.reserved_cards))/3,
           (len(own.nobles)-len(other.nobles))/3], dtype=torch.float32)


class CollectionBot:
    def __init__(self, inference, seed):
        self.inference = inference
        self.rng = random.Random(seed + CONFIG["behavior_seed_offset"])
        self.observations, self.simple, self.trace = [], [], []

    def choose_action(self, env, state, legal_actions=None):
        legal = env.legal_actions(state) if legal_actions is None else legal_actions
        scores, _, row, _, _ = self.inference.full_prediction(state, legal)
        if not torch.isfinite(scores).all():
            raise RuntimeError("Nonfinite 035 scores")
        best = chosen = int(scores.argmax())
        if state.turn_index < CONFIG["opening_turns"] and self.rng.random() < CONFIG["epsilon"]:
            top = sorted(range(len(legal)), key=lambda j: (-float(scores[j]), j))[:CONFIG["top_k"]]
            weights = torch.softmax(scores[top] / CONFIG["temperature"], 0).tolist()
            chosen = self.rng.choices(top, weights=weights, k=1)[0]
        self.observations.append(row["observation"])
        self.simple.append(simple_features(state))
        self.trace.append({"state_hash": digest(serialize_state_snapshot(state)),
                           "greedy_row": best, "selected_row": chosen})
        return legal[chosen]


def initialize_worker():
    global _INFERENCE
    torch.set_num_threads(1)
    _INFERENCE = StrongPolicyInference(CHECKPOINT)


def collect_one(index, binding, *, root=None, seed_override=None):
    root = OUTPUT if root is None else Path(root)
    seed = CONFIG["seed_start"] + index if seed_override is None else seed_override
    bot = CollectionBot(_INFERENCE, seed)
    replay = collect_game_replay(bot, bot, seed=seed, max_turns=CONFIG["max_turns"],
                                 repetition_limit=CONFIG["repetition_limit"], no_progress_limit=CONFIG["no_progress_limit"])
    valid = replay.termination_reason in VALID_NATURAL_TERMINATIONS
    winner = replay.winner if valid else None
    if not bot.observations:
        raise RuntimeError("Empty game")
    observations = torch.stack(bot.observations)
    with torch.no_grad():
        latent = _INFERENCE.model.state_encoder(observations).detach()
    targets = torch.tensor([outcome_value(winner, step.player_id) for step in replay.steps], dtype=torch.float32) if valid else None
    tensor_path = root / "games" / f"game_{index:04d}.pt"
    tensor_save(tensor_path, {"index": index, "seed": seed, "binding": binding,
                              "observations": observations, "latent": latent,
                              "simple": torch.stack(bot.simple), "targets": targets,
                              "turns": torch.tensor([step.turn_index for step in replay.steps]),
                              "actors": torch.tensor([step.player_id for step in replay.steps])})
    trace = [{**info, "turn": step.turn_index, "actor": step.player_id, "action": step.action_payload}
             for info, step in zip(bot.trace, replay.steps, strict=True)]
    record = {"index": index, "seed": seed, "split": split_for(index), "binding": binding,
              "positions": len(trace), "valid_natural": valid, "termination_reason": replay.termination_reason,
              "winner": winner, "final_state": replay.final_state_snapshot, "trace": trace,
              "tensor_sha256": file_hash(tensor_path)}
    atomic_write(tensor_path.with_suffix(".json"), record)
    return index


def load_record(index, manifest, root=None):
    root = OUTPUT if root is None else Path(root)
    path = root / "games" / f"game_{index:04d}.json"
    record = read_json(path)
    if record["index"] != index or record["seed"] != CONFIG["seed_start"] + index or record["binding"] != digest(manifest):
        raise ValueError("Game identity mismatch")
    if record["tensor_sha256"] != file_hash(path.with_suffix(".pt")):
        raise ValueError("Game tensor changed")
    summary_path = root / ("development_summary.json" if index < 4608 else "test_summary.json")
    if summary_path.exists() and read_json(summary_path)["files"][str(index)] != file_hash(path):
        raise ValueError("Sealed game record changed")
    return record


def collect_range(stage, indices, manifest):
    if stage == "test":
        check_selection_lock(manifest)
    pending = []
    for index in indices:
        path = OUTPUT / "games" / f"game_{index:04d}.json"
        if path.exists():
            load_record(index, manifest)
        elif path.with_suffix(".pt").exists():
            raise RuntimeError("Orphan tensor; inspect interrupted atomic game before resuming")
        else:
            pending.append(index)
    if pending:
        with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=initialize_worker) as pool:
            jobs = [pool.submit(collect_one, index, digest(manifest)) for index in pending]
            for done, job in enumerate(as_completed(jobs), start=len(indices)-len(pending)+1):
                job.result()
                if done % 128 == 0 or done == len(indices):
                    print(json.dumps({"stage": stage, "completed": done, "total": len(indices)}), flush=True)
    records = [load_record(index, manifest) for index in indices]
    summary = {"manifest": digest(manifest), "stage": stage, "games": len(records),
               "positions": sum(r["positions"] for r in records),
               "valid_games": sum(r["valid_natural"] for r in records),
               "termination_counts": dict(Counter(r["termination_reason"] for r in records)),
               "files": {str(r["index"]): file_hash(OUTPUT / "games" / f"game_{r['index']:04d}.json") for r in records}}
    summary_path = OUTPUT / f"{stage}_summary.json"
    if summary_path.exists() and read_json(summary_path) != summary:
        raise ValueError("Sealed collection summary changed")
    atomic_write(summary_path, summary)
    return summary


@dataclass
class ValueData:
    ids: list[int]
    x: torch.Tensor
    simple: torch.Tensor
    y: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor

    @property
    def weights(self):
        return torch.repeat_interleave(1. / self.lengths, self.lengths)


def load_data(indices, manifest):
    ids, xs, simples, ys, lengths = [], [], [], [], []
    for index in indices:
        record = load_record(index, manifest)
        if not record["valid_natural"]:
            continue
        data = torch.load(OUTPUT / "games" / f"game_{index:04d}.pt", weights_only=True)
        if data["binding"] != digest(manifest) or data["index"] != index:
            raise ValueError("Tensor binding mismatch")
        if data["targets"] is None or not torch.isfinite(data["latent"]).all():
            raise ValueError("Invalid learning row")
        ids.append(index)
        xs.append(data["latent"])
        simples.append(data["simple"])
        ys.append(data["targets"])
        lengths.append(len(data["targets"]))
    if len(ids) < math.ceil(.98 * len(indices)):
        raise RuntimeError("More than 2% non-natural games in learning split")
    n = torch.tensor(lengths, dtype=torch.long)
    return ValueData(ids, torch.cat(xs), torch.cat(simples), torch.cat(ys), n,
                     torch.cat((torch.zeros(1, dtype=torch.long), n.cumsum(0)[:-1])))


def game_mse(prediction, data):
    error = (prediction - data.y).square()
    return (error * data.weights).sum() / len(data.ids)


def sample_batch_indices(data, generator, batch_size):
    game = torch.randint(len(data.ids), (batch_size,), generator=generator)
    turn = (torch.rand(batch_size, generator=generator) * data.lengths[game]).long()
    return data.offsets[game] + turn


@torch.no_grad()
def head_predictions(head, data):
    return torch.cat([head(data.x[start:start+4096]).squeeze(-1) for start in range(0, len(data.y), 4096)])


def update_selection(loss, best_loss, stale):
    if loss < best_loss - CONFIG["min_improvement"]:
        return True, loss, 0
    return False, best_loss, stale + 1


def train_head(name, data, validation, manifest):
    report_path = OUTPUT / f"{name}_training.json"
    destination = OUTPUT / f"{name}.pt"
    if report_path.exists():
        report = read_json(report_path)
        if report.get("status") != "complete":
            raise RuntimeError("Incomplete recorded fit; do not silently restart")
        if report["manifest"] != digest(manifest) or report["checkpoint_sha256"] != file_hash(destination):
            raise ValueError("Trained head identity changed")
        return report
    torch.manual_seed(CONFIG["training_seed"])
    model = StrongPolicyInference(CHECKPOINT).model
    initial_policy_hash = policy_hash(model)
    for param_name, parameter in model.named_parameters():
        parameter.requires_grad_(param_name.startswith("value_head."))
    torch.nn.init.zeros_(model.value_head[-2].weight)
    torch.nn.init.zeros_(model.value_head[-2].bias)
    head = model.value_head
    optimizer = torch.optim.AdamW(head.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    rng = torch.Generator().manual_seed(CONFIG["training_seed"])
    best_loss = float(game_mse(head_predictions(head, validation), validation))
    best_state, best_step, stale = copy.deepcopy(head.state_dict()), 0, 0
    history = [{"step": 0, "validation_game_mse": best_loss, "selected": True}]
    report = {"manifest": digest(manifest), "name": name, "status": "running", "history": history,
              "training_ids": data.ids, "validation_ids": validation.ids,
              "training_positions": len(data.y), "validation_positions": len(validation.y),
              "initial_policy_sha256": initial_policy_hash, "selected_step": 0,
              "trainable_parameters": sum(p.numel() for p in head.parameters())}
    atomic_write(report_path, report)
    tensor_save(OUTPUT / "selected_heads" / f"{name}_0000.pt", {"state_dict": best_state, "step": 0})
    started = time.perf_counter()
    loss_sum, batches = 0., 0
    seen_positions = torch.zeros(len(data.y), dtype=torch.bool)
    seen_games = torch.zeros(len(data.ids), dtype=torch.bool)
    selected_dose = {"games": 0, "positions": 0}
    for step in range(1, CONFIG["max_updates"] + 1):
        ids = sample_batch_indices(data, rng, CONFIG["batch_size"])
        seen_positions[ids] = True
        seen_games[torch.bucketize(ids, data.offsets[1:], right=True)] = True
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(head(data.x[ids]).squeeze(-1), data.y[ids])
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite value loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), CONFIG["gradient_norm"])
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite value gradient")
        optimizer.step()
        loss_sum += float(loss.detach())
        batches += 1
        if step % CONFIG["validation_interval"] == 0:
            val_loss = float(game_mse(head_predictions(head, validation), validation))
            improved, best_loss, stale = update_selection(val_loss, best_loss, stale)
            if improved:
                best_state, best_step = copy.deepcopy(head.state_dict()), step
                selected_dose = {"games": int(seen_games.sum()), "positions": int(seen_positions.sum())}
                tensor_save(OUTPUT / "selected_heads" / f"{name}_{step:04d}.pt", {"state_dict": best_state, "step": step})
            history.append({"step": step, "validation_game_mse": val_loss,
                            "mean_recent_training_batch_loss": loss_sum / batches, "selected": improved,
                            "distinct_training_games_seen": int(seen_games.sum()),
                            "distinct_training_positions_seen": int(seen_positions.sum())})
            report["selected_step"] = best_step
            atomic_write(report_path, report)
            print(json.dumps({"stage": name + "_training", "step": step, "max_steps": CONFIG["max_updates"]}), flush=True)
            loss_sum, batches = 0., 0
            if stale >= CONFIG["patience"]:
                break
    head.load_state_dict(best_state)
    model.eval()
    final_hash = policy_hash(model)
    report.update({"status": "complete", "completed_updates": step, "selected_step": best_step,
                   "selected_validation_game_mse": float(game_mse(head_predictions(head, validation), validation)),
                   "selected_training_game_mse": float(game_mse(head_predictions(head, data), data)),
                   "final_policy_sha256": final_hash, "policy_retention_passed": final_hash == initial_policy_hash,
                   "selected_training_dose": selected_dose,
                   "wall_seconds": time.perf_counter() - started})
    if final_hash != initial_policy_hash:
        report["status"] = "failed_policy_retention"
        atomic_write(report_path, report)
        raise RuntimeError("035 policy changed")
    save_model(destination, model, 143, report)
    report["checkpoint_sha256"] = file_hash(destination)
    atomic_write(report_path, report)
    return report


def fit_simple(data, validation, manifest):
    path = OUTPUT / "simple.json"
    if path.exists():
        result = read_json(path)
        if result["manifest"] != digest(manifest):
            raise ValueError("Simple baseline identity mismatch")
        return result
    # Keep dense linear algebra in Torch: this host's NumPy MKL and Torch
    # OpenMP runtimes conflict when both BLAS implementations are initialized.
    x, y = data.simple.double(), data.y.double()
    weights = data.weights.double() / len(data.ids)
    gram, rhs = x.T @ (x * weights[:, None]), x.T @ (y * weights)
    candidates = []
    for ridge in CONFIG["ridge_grid"]:
        coefficients = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1], dtype=torch.float64), rhs)
        predicted = (validation.simple.double() @ coefficients).clamp(-1, 1).float()
        candidates.append({"ridge": ridge, "coefficients": coefficients.tolist(),
                           "validation_game_mse": float(game_mse(predicted, validation))})
    selected = min(range(len(candidates)), key=lambda i: (candidates[i]["validation_game_mse"], i))
    result = {"manifest": digest(manifest), "model": "public_score_resource_ridge_15_v1",
              "training_ids": data.ids, "validation_ids": validation.ids,
              "candidates": candidates, "selected_index": selected, **candidates[selected]}
    atomic_write(path, result)
    return result


def seal_selection(manifest):
    if identity() != manifest:
        raise ValueError("Frozen dependencies changed before model selection")
    files = ["small.pt", "large.pt", "small_training.json", "large_training.json", "simple.json", "development_summary.json"]
    value = {"manifest": digest(manifest), "files": {name: file_hash(OUTPUT / name) for name in files},
             "test_indices": list(range(4608, 5120)), "primary_candidate": "large"}
    path = OUTPUT / "selection_lock.json"
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Model selection lock changed")
    else:
        if any((OUTPUT / "games" / f"game_{i:04d}.json").exists() for i in range(4608, 5120)):
            raise RuntimeError("Protected games exist before model selection lock")
        atomic_write(path, value)
    return value


def check_selection_lock(manifest):
    lock = read_json(OUTPUT / "selection_lock.json")
    if lock["manifest"] != digest(manifest):
        raise ValueError("Selection manifest mismatch")
    if any(file_hash(OUTPUT / name) != sha for name, sha in lock["files"].items()):
        raise ValueError("Selected model or development corpus changed")
    return lock


def summarize_game_errors(errors):
    """Whole-game means and paired whole-game bootstrap; lower is better."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 4 or not np.isfinite(errors).all():
        raise ValueError("Invalid game errors")
    means = errors.mean(0)
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    boot = np.empty((CONFIG["bootstrap_samples"], 4))
    for start in range(0, len(boot), 1000):
        count = min(1000, len(boot)-start)
        ids = rng.integers(0, len(errors), (count, len(errors)))
        boot[start:start+count] = errors[ids].mean(1)
    contrasts = {name: {"mean": float(means[a]-means[b]),
                        "ci95": np.quantile(boot[:, a]-boot[:, b], [.025, .975]).tolist()}
                 for name, a, b in CONTRASTS}
    return {"mse": dict(zip(MODEL_NAMES, means.tolist())), "contrasts": contrasts,
            "large_beats_neutral": contrasts["large_minus_neutral"]["ci95"][1] < 0,
            "large_material_improvement": contrasts["large_minus_neutral"]["ci95"][1] < 0 and means[2] <= .95*means[0],
            "scale_benefit_established": contrasts["large_minus_small"]["ci95"][1] < 0,
            "large_beats_simple": contrasts["large_minus_simple"]["ci95"][1] < 0,
            "simple_beats_large": contrasts["large_minus_simple"]["ci95"][0] > 0,
            "simple_beats_neutral": contrasts["simple_minus_neutral"]["ci95"][1] < 0}


def classify_result(statistics, valid_test_games):
    if valid_test_games != 512:
        return "invalid_test_completion"
    if statistics["large_material_improvement"]:
        return ("material_value_improvement_and_scale_benefit" if statistics["scale_benefit_established"]
                else "material_value_improvement_without_established_scale_benefit")
    if statistics["large_beats_neutral"]:
        return "positive_value_improvement_below_material_threshold"
    if statistics["contrasts"]["large_minus_neutral"]["ci95"][0] > 0:
        return "large_value_regression"
    return "value_generalization_not_established"


def evaluate_test(manifest):
    if identity() != manifest:
        raise ValueError("Frozen dependencies changed before test evaluation")
    lock = check_selection_lock(manifest)
    path = OUTPUT / "result.json"
    if path.exists():
        raise RuntimeError("Protected evaluation already sealed; use the verifier, not another evaluation")
    heads = [StrongPolicyInference(OUTPUT / f"{name}.pt").model.value_head for name in ("small", "large")]
    simple = torch.tensor(read_json(OUTPUT / "simple.json")["coefficients"], dtype=torch.float64)
    errors, valid_ids, phases = [], [], {"early": [], "middle": [], "late": []}
    for index in range(4608, 5120):
        record = load_record(index, manifest)
        if not record["valid_natural"]:
            continue
        data = torch.load(OUTPUT / "games" / f"game_{index:04d}.pt", weights_only=True)
        with torch.no_grad():
            predictions = np.column_stack((np.zeros(len(data["targets"])),
                heads[0](data["latent"]).squeeze(-1).numpy(), heads[1](data["latent"]).squeeze(-1).numpy(),
                (data["simple"].double() @ simple).clamp(-1, 1).numpy()))
        squared = (predictions - data["targets"].double().numpy()[:, None])**2
        mse = squared.mean(0)
        errors.append(mse)
        valid_ids.append(index)
        turns = data["turns"].numpy()
        for phase, mask in (("early", turns < 20), ("middle", (turns >= 20) & (turns < 40)), ("late", turns >= 40)):
            if mask.any():
                phases[phase].append(squared[mask].mean(0))
        tensor_save(OUTPUT / "test_predictions" / f"game_{index:04d}.pt",
                    {"index": index, "selection_sha256": file_hash(OUTPUT / "selection_lock.json"),
                     "predictions": torch.from_numpy(predictions), "mse": torch.from_numpy(mse)})
    if not errors:
        raise RuntimeError("No naturally completed test games")
    statistics = summarize_game_errors(errors)
    result = {"manifest": digest(manifest), "selection_sha256": file_hash(OUTPUT / "selection_lock.json"),
              "test_summary_sha256": file_hash(OUTPUT / "test_summary.json"),
              "test_games": 512, "valid_test_games": len(valid_ids), "valid_test_indices": valid_ids,
              "natural_test_gate": len(valid_ids) == 512,
              "status": "valid" if len(valid_ids) == 512 else "invalid_test_completion",
              "decision": classify_result(statistics, len(valid_ids)),
              "primary": "large_minus_neutral", "game_mse_rows": [v.tolist() for v in errors],
              "phase_mse_diagnostic": {name: dict(zip(MODEL_NAMES, np.mean(values, axis=0).tolist())) for name, values in phases.items()},
              **statistics}
    if not result["natural_test_gate"]:
        result["interpretation"] = "Descriptive naturally-completed subset only; no confirmatory claim"
    atomic_write(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("game_mse_rows", "valid_test_indices")}, indent=2), flush=True)
    assert check_selection_lock(manifest) == lock
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "train", "test", "all"))
    args = parser.parse_args(argv)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with process_lock(OUTPUT / "process.lock"):
        manifest = freeze()
        if args.stage == "freeze":
            return
        preflight = read_json(OUTPUT / "preflight.json")
        if preflight["status"] != "passed" or preflight["manifest"] != digest(manifest):
            raise ValueError("Successful matching preflight required")
        started = time.perf_counter()
        try:
            if args.stage in ("development", "all"):
                collect_range("development", list(range(4608)), manifest)
            if args.stage in ("train", "all"):
                validation = load_data(list(range(4096, 4608)), manifest)
                small = load_data(small_indices(), manifest)
                train_head("small", small, validation, manifest)
                del small
                large = load_data(list(range(4096)), manifest)
                train_head("large", large, validation, manifest)
                fit_simple(large, validation, manifest)
                del large, validation
                seal_selection(manifest)
            if args.stage in ("test", "all"):
                collect_range("test", list(range(4608, 5120)), manifest)
                evaluate_test(manifest)
            atomic_write(OUTPUT / f"{args.stage}_completion.json", {"manifest": digest(manifest), "wall_seconds": time.perf_counter()-started})
        except Exception as error:
            atomic_write(OUTPUT / "failure.json", {"manifest": digest(manifest), "stage": args.stage,
                                                  "error": repr(error), "wall_seconds": time.perf_counter()-started})
            raise


if __name__ == "__main__":
    main()
