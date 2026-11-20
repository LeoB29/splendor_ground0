"""Experiment 146: fixed 035 policy, public baseline plus learned value residual."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from splendor_ai.alphazero import run_value_generalization_scale as corpus
from splendor_ai.alphazero.network import NetworkPrediction
from splendor_ai.alphazero.run_public_value_search import PublicValueInference, SIMPLE, SIMPLE_SHA
from splendor_ai.alphazero.strong_policy import StrongPolicyInference
from splendor_ai.alphazero.run_strong_policy_pilot import CHECKPOINT, CHECKPOINT_SHA, policy_hash
from splendor_ai.encoding import encode_public_observation_tensor, fit_feature_vector
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.eval.run_independent_world_root_validation import atomic_write, digest, file_hash, process_lock, read_json
from splendor_ai.training.replay import serialize_action

OUTPUT = Path("outputs/value_representation_146")
PROBE_OUTPUT = OUTPUT / "preflight_v2"
PROTOCOL = Path("docs/engineering/value_representation_146.md")
PRIOR = Path("outputs/public_value_ablation_145/manifest.json")
CONFIG = {
    "training_ids": [0, 4096], "validation_ids": [4096, 4608],
    "test_games": 512, "test_seed_start": 1_460_000_000, "probe_seed": 1_469_000_000,
    "behavior": "exact_143_collection_recipe_including_rng_offset_143003",
    "workers": 8, "torch_threads": 1, "training_seed": 146010,
    "batch_size": 128, "max_updates": 4096, "validation_interval": 128,
    "patience": 4, "min_improvement": .001, "learning_rate": .0003,
    "weight_decay": .0001, "gradient_norm": 1.,
    "selection": "game_balanced_MSE_on_turns_below_40",
    "loss": "game_then_position_uniform_MSE_all_training_turns",
    "architecture": "copied_035_encoder_plus_copied_256_128_1_GELU_head_no_tanh_zero_final_linear",
    "value": "clip(float64_clipped_public_baseline_plus_float64_residual,-1,1)",
    "bootstrap_samples": 50000, "bootstrap_seed": 146999,
    "primary_public_relative_reduction": .03, "primary_frozen_relative_reduction": .01,
    "nonregression_absolute_margin": .01,
    "cost_repeats": 3, "cost_median_ratio_max": 1.5, "cost_p95_ratio_max": 2.,
}
NAMES = ("neutral", "public", "frozen", "trainable")
PHASES = ("early_middle", "overall", "late", "early", "middle")


def identity():
    # Validate the old manifest's explicit list: adding new source must not
    # retroactively change the inventory used to freeze an earlier experiment.
    for name, sha in read_json(PRIOR)["dependencies"].items():
        if file_hash(Path(name)) != sha:
            raise ValueError(f"Prior frozen dependency changed: {name}")
    for path, sha in ((CHECKPOINT, CHECKPOINT_SHA), (SIMPLE, SIMPLE_SHA)):
        if file_hash(path) != sha:
            raise ValueError(f"Frozen model changed: {path}")
    inputs = [CHECKPOINT, SIMPLE, PRIOR, corpus.OUTPUT / "manifest.json",
              corpus.OUTPUT / "development_summary.json", corpus.OUTPUT / "verification.json",
              Path("outputs/public_value_ablation_145/result.json"),
              Path("outputs/public_value_ablation_145/verification.json")]
    files = sorted(Path("src/splendor_ai").rglob("*.py")) + [PROTOCOL,
        Path("run_value_representation_146.py"), Path("tests/test_value_representation_146.py"),
        Path("scripts/analysis/verify_value_representation_146.py"),
        Path("scripts/analysis/verify_value_representation_arena_146.py"),
        Path("scripts/analysis/verify_value_generalization_scale_143.py")]
    from splendor_ai.alphazero import run_value_representation_arena as arena
    return {"format": "value_representation_146_v1", "config": CONFIG, "arena_config": arena.CONFIG,
            "inputs": {p.as_posix(): file_hash(p) for p in inputs},
            "dependencies": {p.as_posix(): file_hash(p) for p in files},
            "python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__}


class ResidualValue(nn.Module):
    def __init__(self, train_encoder):
        super().__init__()
        source = StrongPolicyInference(CHECKPOINT).model
        self.encoder = copy.deepcopy(source.state_encoder)
        self.head = nn.Sequential(*list(copy.deepcopy(source.value_head).children())[:-1])
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        for p in self.encoder.parameters():
            p.requires_grad_(train_encoder)
        self.register_buffer("coefficients", torch.tensor(read_json(SIMPLE)["coefficients"], dtype=torch.float64))

    def forward(self, observations, simple):
        baseline = (simple.double() @ self.coefficients).clamp(-1, 1)
        residual = self.head(self.encoder(observations)).squeeze(-1).double()
        return (baseline + residual).clamp(-1, 1)


def load_value(name):
    if name not in ("frozen", "trainable"):
        raise ValueError(name)
    model = ResidualValue(name == "trainable")
    payload = torch.load(OUTPUT / f"{name}.pt", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


class ResidualInference:
    """035 policy remains in a separate, unchanged model; only value is replaced."""
    def __init__(self, value_model, base=None):
        self.base = PublicValueInference() if base is None else base
        self.value_model = value_model.eval()

    @torch.inference_mode()
    def predict_one(self, state, actions):
        prediction = self.base.predict_one(state, actions)
        obs = torch.tensor(fit_feature_vector(
            encode_public_observation_tensor(state, state.current_player).vector,
            self.base.model.config.observation_size), dtype=torch.float32)
        residual = float(self.value_model.head(self.value_model.encoder(obs[None])).squeeze())
        # Reuse the exact deployed scalar. A one-row matrix-vector product can
        # round differently from the public evaluator's vector dot product.
        value = max(-1., min(1., prediction.value + residual))
        return NetworkPrediction(prediction.logits, value)


@dataclass
class Data:
    ids: list[int]
    observations: torch.Tensor
    simple: torch.Tensor
    y: torch.Tensor
    turns: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor


def load_development(indices):
    if not indices or not set(indices) <= set(range(4608)):
        raise ValueError("Only sealed 143 development IDs may enter fitting")
    summary = read_json(corpus.OUTPUT / "development_summary.json")
    binding = digest(read_json(corpus.OUTPUT / "manifest.json"))
    rows = []
    for i in indices:
        path = corpus.OUTPUT / "games" / f"game_{i:04d}.json"
        record = read_json(path)
        if summary["files"][str(i)] != file_hash(path) or record["tensor_sha256"] != file_hash(path.with_suffix(".pt")):
            raise ValueError("143 development data changed")
        if (record["index"], record["seed"], record["binding"]) != (i, 1_430_000_000+i, binding) or not record["valid_natural"]:
            raise ValueError("Invalid development game")
        row = torch.load(path.with_suffix(".pt"), weights_only=True)
        if row["binding"] != binding or row["index"] != i or row["targets"] is None:
            raise ValueError("Invalid development tensor")
        rows.append(row)
    lengths = torch.tensor([len(r["targets"]) for r in rows])
    return Data(list(indices), *(torch.cat([r[key] for r in rows]) for key in ("observations", "simple", "targets", "turns")),
                lengths, torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0))))


def phase_mask(turns, phase):
    if phase == "early_middle":
        return turns < 40
    if phase == "overall":
        return torch.ones_like(turns, dtype=torch.bool)
    if phase == "late":
        return turns >= 40
    if phase == "early":
        return turns < 20
    if phase == "middle":
        return (turns >= 20) & (turns < 40)
    raise ValueError(phase)


@torch.no_grad()
def predictions(model, data):
    return torch.cat([model(data.observations[i:i+4096], data.simple[i:i+4096]) for i in range(0, len(data.y), 4096)])


def validation_loss(model, data):
    errors = (predictions(model, data)-data.y.double()).square()
    mask = phase_mask(data.turns, "early_middle")
    return float(torch.stack([errors[a:b][mask[a:b]].mean() for a, b in zip(data.offsets[:-1], data.offsets[1:])]).mean())


def fit(name, training, validation, callback=None):
    if name not in ("frozen", "trainable") or set(training.ids) & set(validation.ids):
        raise ValueError("Invalid fit")
    torch.manual_seed(CONFIG["training_seed"])
    model = ResidualValue(name == "trainable")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    generator = torch.Generator().manual_seed(CONFIG["training_seed"])
    best = validation_loss(model, validation)
    best_step, stale = 0, 0
    selected = copy.deepcopy(model.state_dict())
    history = [{"step": 0, "validation_game_mse": best, "selected": True}]
    for step in range(1, CONFIG["max_updates"]+1):
        ids = corpus.sample_batch_indices(training, generator, CONFIG["batch_size"])
        optimizer.zero_grad(set_to_none=True)
        loss = (model(training.observations[ids], training.simple[ids])-training.y[ids].double()).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, CONFIG["gradient_norm"])
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite gradient")
        optimizer.step()
        if step % CONFIG["validation_interval"] == 0:
            current = validation_loss(model, validation)
            improved = current < best - CONFIG["min_improvement"]
            if improved:
                best, best_step, stale = current, step, 0
                selected = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            history.append({"step": step, "validation_game_mse": current, "selected": improved})
            if callback:
                callback(history)
            if stale >= CONFIG["patience"]:
                break
    model.load_state_dict(selected)
    report = {"name": name, "history": history, "selected_step": best_step, "completed_updates": step,
              "selected_validation_game_mse": validation_loss(model, validation),
              "training_ids": training.ids, "validation_ids": validation.ids,
              "training_positions": len(training.y), "validation_positions": len(validation.y),
              "trainable_parameters": sum(p.numel() for p in parameters)}
    return model.eval(), report


def train(manifest):
    training = load_development(list(range(4096)))
    validation = load_development(list(range(4096, 4608)))
    for name in ("frozen", "trainable"):
        started = time.perf_counter()
        def save_progress(history):
            atomic_write(OUTPUT / f"{name}_progress.json", {"manifest": digest(manifest), "history": history})
            print(json.dumps({"stage": name+"_training", "step": history[-1]["step"]}), flush=True)
        model, report = fit(name, training, validation, save_progress)
        report.update(manifest=digest(manifest), wall_seconds=time.perf_counter()-started, status="complete")
        corpus.tensor_save(OUTPUT / f"{name}.pt", {"state_dict": model.state_dict(), "manifest": digest(manifest), "name": name})
        report["checkpoint_sha256"] = file_hash(OUTPUT / f"{name}.pt")
        atomic_write(OUTPUT / f"{name}_training.json", report)


def collect_one(index, binding, root, probe=False):
    seed = CONFIG["probe_seed"]+index if probe else CONFIG["test_seed_start"]+index
    corpus.collect_one(index, binding, root=root, seed_override=seed)
    path = Path(root) / "games" / f"game_{index:04d}.json"
    record = read_json(path)
    record.update(split="probe" if probe else "protected_test", experiment=146)
    atomic_write(path, record)
    return index


def collect_test(manifest):
    check_lock(manifest)
    root = OUTPUT / "test"
    if (root / "games").exists():
        raise FileExistsError("Protected prediction collection already exists")
    with ProcessPoolExecutor(max_workers=CONFIG["workers"], initializer=corpus.initialize_worker) as pool:
        jobs = [pool.submit(collect_one, i, digest(manifest), root) for i in range(CONFIG["test_games"])]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 64 == 0:
                print(json.dumps({"stage": "prediction_collection", "completed": n}), flush=True)
    paths = [root / "games" / f"game_{i:04d}.json" for i in range(CONFIG["test_games"])]
    records = [read_json(p) for p in paths]
    atomic_write(OUTPUT / "test_summary.json", {"manifest": digest(manifest), "games": len(paths),
        "valid_games": sum(r["valid_natural"] for r in records), "positions": sum(r["positions"] for r in records),
        "files": {p.name: file_hash(p) for p in paths}})


def probe_states():
    for i in range(2):
        record = read_json(PROBE_OUTPUT / "games" / f"game_{i:04d}.json")
        env = SplendorEnv(seed=record["seed"])
        state = env.initial_state()
        for row in record["trace"]:
            legal = env.legal_actions(state)
            yield state, legal
            action = next(a for a in legal if serialize_action(a) == row["action"])
            state = env.step(state, action)


def cost_check():
    base = PublicValueInference()
    models = {"public": base, **{name: ResidualInference(load_value(name), base) for name in ("frozen", "trainable")}}
    times = {name: [] for name in models}
    states = list(probe_states())
    for model in models.values():
        for state, legal in states[:2]:
            model.predict_one(state, legal)
    for repeat in range(CONFIG["cost_repeats"]):
        for index, (state, legal) in enumerate(states):
            order = list(models) if (repeat+index) % 2 == 0 else list(reversed(models))
            logits = []
            for name in order:
                start = time.perf_counter()
                p = models[name].predict_one(state, legal)
                times[name].append(time.perf_counter()-start)
                logits.append(p.logits)
            if any(v != logits[0] for v in logits[1:]):
                raise ValueError("035 policy logits changed")
    ratios = {q: float(np.quantile(times["trainable"], p)/np.quantile(times["public"], p)) for q, p in (("median", .5), ("p95", .95))}
    return {"states": len(states), "repeats": CONFIG["cost_repeats"], "seconds": times, "ratio": ratios,
            "policy_logits_exact": True, "source_policy_sha256": policy_hash(base.model),
            "passed": bool(ratios["median"] <= CONFIG["cost_median_ratio_max"] and ratios["p95"] <= CONFIG["cost_p95_ratio_max"])}


def seal_selection(manifest):
    if identity() != manifest or (OUTPUT / "test" / "games").exists():
        raise ValueError("Source changed or protected data precede lock")
    atomic_write(OUTPUT / "cost.json", cost_check())
    files = ["frozen.pt", "trainable.pt", "frozen_training.json", "trainable_training.json", "cost.json"]
    atomic_write(OUTPUT / "selection_lock.json", {"manifest": digest(manifest), "primary_candidate": "trainable",
        "files": {name: file_hash(OUTPUT / name) for name in files},
        "test_seed_range": [CONFIG["test_seed_start"], CONFIG["test_seed_start"]+CONFIG["test_games"]]})


def check_lock(manifest):
    lock = read_json(OUTPUT / "selection_lock.json")
    if lock["manifest"] != digest(manifest) or any(file_hash(OUTPUT / name) != sha for name, sha in lock["files"].items()):
        raise ValueError("Selected model changed")
    return lock


def statistics(rows):
    result = {}
    for phase, values in rows.items():
        data = np.asarray(values, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 4 or not len(data) or not np.isfinite(data).all():
            raise ValueError("Invalid per-game error table")
        means = data.mean(0)
        rng = np.random.default_rng(CONFIG["bootstrap_seed"])
        contrasts = data[:, 3:4]-data[:, 1:3]
        boot = np.concatenate([contrasts[rng.integers(len(data), size=(500, len(data)))].mean(1)
                               for _ in range(CONFIG["bootstrap_samples"]//500)])
        result[phase] = {"games": len(data), "mse": dict(zip(NAMES, means.tolist())),
            "contrasts": {name: {"difference": float(means[3]-means[i]),
                "relative_reduction": float(1-means[3]/means[i]) if means[i] else 0.,
                "ci95": np.quantile(boot[:, i-1], [.025, .975]).tolist()} for i, name in ((1, "public"), (2, "frozen"))}}
    return result


def prediction_gate(stats, valid_games, cost):
    primary = stats["early_middle"]["contrasts"]
    checks = {
        "all_test_games_natural": valid_games == CONFIG["test_games"],
        "phase_coverage": all(stats[p]["games"] == CONFIG["test_games"] for p in ("early_middle", "overall", "late")),
        "early_middle_public": primary["public"]["relative_reduction"] >= CONFIG["primary_public_relative_reduction"] and primary["public"]["ci95"][1] < 0,
        "early_middle_frozen": primary["frozen"]["relative_reduction"] >= CONFIG["primary_frozen_relative_reduction"] and primary["frozen"]["ci95"][1] < 0,
        "nonregression": all(stats[p]["contrasts"][c]["ci95"][1] < CONFIG["nonregression_absolute_margin"] for p in ("overall", "late") for c in ("public", "frozen")),
        "inference_cost": cost["passed"],
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return {"checks": checks, "passed": all(checks.values())}


@torch.no_grad()
def evaluate(manifest):
    check_lock(manifest)
    models = [load_value(name) for name in ("frozen", "trainable")]
    rows = {p: [] for p in PHASES}
    valid = 0
    for i in range(CONFIG["test_games"]):
        path = OUTPUT / "test" / "games" / f"game_{i:04d}.pt"
        record = read_json(path.with_suffix(".json"))
        data = torch.load(path, weights_only=True)
        if not record["valid_natural"]:
            continue
        valid += 1
        values = torch.stack((torch.zeros(len(data["targets"]), dtype=torch.float64),
            (data["simple"].double() @ models[0].coefficients).clamp(-1, 1),
            *(model(data["observations"], data["simple"]) for model in models)), 1)
        errors = (values-data["targets"].double()[:, None]).square()
        for phase in PHASES:
            mask = phase_mask(data["turns"], phase)
            if mask.any():
                rows[phase].append(errors[mask].mean(0).tolist())
        corpus.tensor_save(OUTPUT / "predictions" / path.name, {"index": i, "values": values,
            "selection_sha256": file_hash(OUTPUT / "selection_lock.json")})
    stats = statistics(rows)
    gate = prediction_gate(stats, valid, read_json(OUTPUT / "cost.json"))
    result = {"manifest": digest(manifest), "selection_sha256": file_hash(OUTPUT / "selection_lock.json"),
        "test_summary_sha256": file_hash(OUTPUT / "test_summary.json"), "statistics": stats,
        "valid_games": valid, "game_error_rows": rows, "gate": gate,
        "decision": "prediction_gate_passed_pending_gameplay" if gate["passed"] else "prediction_gate_failed_stop"}
    atomic_write(OUTPUT / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "game_error_rows"}), flush=True)
    return result


def preflight():
    if (OUTPUT / "manifest.json").exists() or PROBE_OUTPUT.exists():
        raise FileExistsError("146 preparation already exists")
    torch.set_num_threads(1)
    binding = digest(identity())
    corpus.initialize_worker()
    for i in range(2):
        collect_one(i, binding, PROBE_OUTPUT, probe=True)
    audit = runpy.run_path("scripts/analysis/verify_value_generalization_scale_143.py")
    audit["initialize"]()
    checks = [audit["audit_one"](i, PROBE_OUTPUT, protected=False) for i in range(2)]
    base = PublicValueInference()
    models = [ResidualInference(ResidualValue(flag), base) for flag in (False, True)]
    count = 0
    for state, legal in probe_states():
        reference = base.predict_one(state, legal)
        assert all(model.predict_one(state, legal) == reference for model in models)
        count += 1
    from splendor_ai.alphazero import run_value_representation_arena as arena
    arena.preflight(binding)
    atomic_write(OUTPUT / "preflight.json", {"manifest": binding, "passed": all(c["valid"] for c in checks),
        "games": checks, "exact_step_zero_predictions": count, "arena_probe_passed": True})
    print(json.dumps({"preflight_passed": True, "exact_step_zero_predictions": count}), flush=True)


def run():
    torch.set_num_threads(1)
    with process_lock(OUTPUT / "process.lock"):
        manifest = identity()
        pre = read_json(OUTPUT / "preflight.json")
        if pre["manifest"] != digest(manifest) or not pre["passed"]:
            raise ValueError("Matching successful preflight required")
        if (OUTPUT / "manifest.json").exists():
            raise FileExistsError("146 already launched; no automatic restart")
        atomic_write(OUTPUT / "manifest.json", manifest)
        try:
            train(manifest)
            seal_selection(manifest)
            collect_test(manifest)
            result = evaluate(manifest)
            # Independently reconstruct predictions and the gate before gameplay.
            subprocess.run([sys.executable, "scripts/analysis/verify_value_representation_146.py", "--prediction"], check=True)
            if result["gate"]["passed"]:
                from splendor_ai.alphazero import run_value_representation_arena as arena
                arena.run(manifest)
            runpy.run_path("scripts/analysis/verify_value_representation_146.py")["finalize"]()
        except BaseException:
            atomic_write(OUTPUT / "failure.json", {"manifest": digest(manifest), "traceback": traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run"))
    args = parser.parse_args()
    (preflight if args.stage == "preflight" else run)()


if __name__ == "__main__":
    main()
