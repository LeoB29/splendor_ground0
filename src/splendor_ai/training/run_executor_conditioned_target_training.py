"""Train and gate experiment 083's executor-conditioned target selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.executor_conditioned_target import (
    ExecutorConditionedTargetConfig,
    ExecutorConditionedTargetModel,
    ExecutorTargetCandidate,
    collate_executor_target_batch,
    load_executor_conditioned_model,
    reconstruct_executor_candidates,
    save_executor_conditioned_checkpoint,
)
from splendor_ai.training.replay import deserialize_state_snapshot
from splendor_ai.training.run_action_value_training import _spearman


HISTORICAL_ROWS_SHA256 = (
    "7a4c080f6a50aa179ea7afa2fe08d7b4f936c139d703959e9fc65752888ab3b6"
)


@dataclass(frozen=True, slots=True)
class ExecutorConditionedTrainingConfig:
    historical_rows_path: Path
    output_dir: Path
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    listwise_temperature: float = 0.10
    advantage_mse_weight: float = 1.0
    max_grad_norm: float = 1.0
    fit_states: int = 800
    screen_states: int = 200
    seed: int = 783_083
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class ExecutorConditionedSample:
    state: Any
    candidates: tuple[ExecutorTargetCandidate, ...]
    advantages: torch.Tensor
    returns: torch.Tensor
    action_indices: torch.Tensor


class ExecutorConditionedDataset(Dataset[ExecutorConditionedSample]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.env = SplendorEnv()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ExecutorConditionedSample:
        row = self.rows[index]
        state = deserialize_state_snapshot(row["state_snapshot"])
        candidates = reconstruct_executor_candidates(
            state, row["targets"], self.env.legal_actions(state)
        )
        returns = torch.tensor(
            [float(target["full_mean"]) for target in row["targets"]],
            dtype=torch.float32,
        )
        advantages = returns - returns[0]
        actions = torch.tensor(
            [candidate.action_index for candidate in candidates], dtype=torch.long
        )
        return ExecutorConditionedSample(
            state,
            tuple(candidates),
            advantages,
            returns,
            actions,
        )


def collate_executor_conditioned_samples(
    samples: list[ExecutorConditionedSample],
) -> dict[str, Any]:
    batch = collate_executor_target_batch(
        [(sample.state, list(sample.candidates)) for sample in samples]
    )
    width = int(batch["target_mask"].shape[1])
    advantages = torch.zeros((len(samples), width), dtype=torch.float32)
    returns = torch.zeros((len(samples), width), dtype=torch.float32)
    action_indices = torch.zeros((len(samples), width), dtype=torch.long)
    objects = []
    for row_index, sample in enumerate(samples):
        count = len(sample.candidates)
        advantages[row_index, :count] = sample.advantages
        returns[row_index, :count] = sample.returns
        action_indices[row_index, :count] = sample.action_indices
        objects.append(sample.candidates)
    batch.update(
        {
            "label_advantages": advantages,
            "target_returns": returns,
            "executor_action_indices": action_indices,
            "candidate_objects": objects,
        }
    )
    return batch


def train_executor_conditioned_screen(
    config: ExecutorConditionedTrainingConfig,
) -> dict[str, Any]:
    _validate_training_config(config)
    if _sha(config.historical_rows_path) != HISTORICAL_ROWS_SHA256:
        raise ValueError("083 historical target-row hash does not match its frozen contract.")
    payload = json.loads(config.historical_rows_path.read_text(encoding="utf-8"))
    rows = payload.get("states", [])
    if len(rows) != 1_000:
        raise ValueError("083 requires exactly 1,000 historical target states.")
    indices = list(range(len(rows)))
    random.Random(config.seed).shuffle(indices)
    fit_rows = [rows[index] for index in indices[: config.fit_states]]
    screen_rows = [rows[index] for index in indices[config.fit_states :]]
    datasets = {
        "fit": ExecutorConditionedDataset(fit_rows),
        "screen": ExecutorConditionedDataset(screen_rows),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "fit",
            collate_fn=collate_executor_conditioned_samples,
            generator=torch.Generator().manual_seed(config.seed),
        )
        for split, dataset in datasets.items()
    }
    torch.manual_seed(config.seed)
    model = ExecutorConditionedTargetModel(
        ExecutorConditionedTargetConfig()
    ).to(config.device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_epoch = None
    best_regret = math.inf
    best_state = None
    history = []
    for epoch in range(1, config.epochs + 1):
        training = _train_epoch(model, loaders["fit"], optimizer, config)
        screen = evaluate_executor_conditioned(
            model, loaders["screen"], config.device, bootstrap_samples=0, seed=0
        )
        history.append({"epoch": epoch, "training": training, "screen": screen})
        if screen["model_mean_regret"] < best_regret:
            best_regret = screen["model_mean_regret"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"[083:fit] epoch={epoch}/{config.epochs} "
            f"loss={training['total_loss']:.5f} "
            f"screen_uplift={screen['selected_minus_baseline_mean']:.4f} "
            f"screen_rank={screen['mean_rank_spearman']:.4f}",
            flush=True,
        )
    if best_state is None or best_epoch is None:
        raise RuntimeError("No best 083 historical-screen epoch was selected.")
    model.load_state_dict(best_state)
    fit = evaluate_executor_conditioned(
        model, loaders["fit"], config.device, bootstrap_samples=0, seed=0
    )
    screen = evaluate_executor_conditioned(
        model, loaders["screen"], config.device, bootstrap_samples=0, seed=0
    )
    gates = {
        "fit_rank_spearman_at_least_050": fit["mean_rank_spearman"] >= 0.50,
        "screen_rank_spearman_at_least_035": screen["mean_rank_spearman"] >= 0.35,
        "fit_uplift_at_least_010": fit["selected_minus_baseline_mean"] >= 0.10,
        "screen_uplift_at_least_005": screen["selected_minus_baseline_mean"] >= 0.05,
        "screen_positive_recall_at_least_020": screen["positive_recall"] >= 0.20,
        "screen_positive_precision_above_prevalence": (
            screen["positive_precision"] > screen["beneficial_pair_prevalence"]
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.output_dir / "executor_conditioned_target_083.pt"
    save_executor_conditioned_checkpoint(
        checkpoint,
        model,
        {
            "training_seed": config.seed,
            "best_epoch": best_epoch,
            "historical_rows_sha256": HISTORICAL_ROWS_SHA256,
            "human_data_used": False,
        },
    )
    result = {
        "training": "executor_conditioned_target_screen_083_v1",
        "configuration": _config_payload(config),
        "model_configuration": asdict(model.config),
        "split_counts": {key: len(value) for key, value in datasets.items()},
        "best_epoch": best_epoch,
        "history": history,
        "fit": fit,
        "screen": screen,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "checkpoint_path": str(checkpoint),
        "human_data_used": False,
    }
    result_path = config.output_dir / "fit_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def evaluate_fresh_executor_conditioned(
    checkpoint_path: Path,
    corpus_path: Path,
    output_path: Path,
    device: str = "cpu",
    batch_size: int = 16,
    bootstrap_samples: int = 20_000,
    seed: int = 783_083,
) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not corpus.get("all_gates_passed", False):
        raise ValueError("083 fresh corpus did not pass every frozen gate.")
    rows_by_split = {
        split: [row for row in corpus["states"] if row["split"] == split]
        for split in ("validation", "confirmation")
    }
    if {key: len(value) for key, value in rows_by_split.items()} != {
        "validation": 200,
        "confirmation": 200,
    }:
        raise ValueError("083 requires exact fresh 200/200 splits.")
    model = load_executor_conditioned_model(checkpoint_path, device)
    evaluations = {}
    for offset, split in enumerate(("validation", "confirmation"), start=1):
        loader = DataLoader(
            ExecutorConditionedDataset(rows_by_split[split]),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_executor_conditioned_samples,
        )
        evaluations[split] = evaluate_executor_conditioned(
            model,
            loader,
            device,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10_000 * offset,
        )
    validation = evaluations["validation"]
    confirmation = evaluations["confirmation"]
    gates = {
        "validation_uplift_positive": validation["selected_minus_baseline_mean"] > 0.0,
        "confirmation_uplift_at_least_005": (
            confirmation["selected_minus_baseline_mean"] >= 0.05
        ),
        "confirmation_uplift_lower95_positive": (
            confirmation["selected_minus_baseline_ci95"][0] > 0.0
        ),
        "confirmation_regret_ratio_at_most_080": confirmation["regret_ratio"] <= 0.80,
        "confirmation_action_movement_at_least_015": (
            confirmation["action_movement"] >= 0.15
        ),
        "confirmation_nonclear_target_at_least_020": (
            confirmation["nonclear_target_rate"] >= 0.20
        ),
    }
    result = {
        "evaluation": "executor_conditioned_target_fresh_083_v1",
        "checkpoint_path": str(checkpoint_path),
        "corpus_path": str(corpus_path),
        "validation": validation,
        "confirmation": confirmation,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "human_data_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _train_epoch(
    model: ExecutorConditionedTargetModel,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    config: ExecutorConditionedTrainingConfig,
) -> dict[str, float]:
    model.train()
    totals = {"listwise_loss": 0.0, "advantage_mse_loss": 0.0, "total_loss": 0.0}
    states = 0
    for batch in loader:
        batch.pop("candidate_objects")
        batch = {key: value.to(config.device) for key, value in batch.items()}
        predictions = model(**batch)
        listwise_terms = []
        mse_terms = []
        for row_index in range(predictions.shape[0]):
            valid = batch["target_mask"][row_index]
            labels = batch["label_advantages"][row_index, valid]
            scores = predictions[row_index, valid]
            distribution = torch.softmax(labels / config.listwise_temperature, dim=0)
            listwise_terms.append(
                -(distribution * torch.log_softmax(scores, dim=0)).sum()
            )
            mse_terms.append(torch.mean((scores - labels) ** 2))
        listwise = torch.stack(listwise_terms).mean()
        mse = torch.stack(mse_terms).mean()
        total = listwise + config.advantage_mse_weight * mse
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        count = int(predictions.shape[0])
        totals["listwise_loss"] += float(listwise.detach().cpu()) * count
        totals["advantage_mse_loss"] += float(mse.detach().cpu()) * count
        totals["total_loss"] += float(total.detach().cpu()) * count
        states += count
    return {key: value / states for key, value in totals.items()}


def evaluate_executor_conditioned(
    model: ExecutorConditionedTargetModel,
    loader: DataLoader[Any],
    device: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    deltas = []
    model_regrets = []
    baseline_regrets = []
    rank_spearmans = []
    movement = []
    nonclear = []
    true_positive = predicted_positive = joint_positive = 0
    pair_count = 0
    selected_sources: dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            objects = batch.pop("candidate_objects")
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch)
            for row_index, candidates in enumerate(objects):
                count = len(candidates)
                scores = predictions[row_index, :count].cpu()
                returns = batch["target_returns"][row_index, :count].cpu()
                labels = batch["label_advantages"][row_index, :count].cpu()
                selected = int(torch.argmax(scores).item())
                best = float(returns.max().item())
                selected_return = float(returns[selected].item())
                baseline_return = float(returns[0].item())
                deltas.append(selected_return - baseline_return)
                model_regrets.append(best - selected_return)
                baseline_regrets.append(best - baseline_return)
                rank_spearmans.append(
                    _spearman(scores.tolist(), returns.tolist())
                )
                movement.append(
                    float(
                        int(batch["executor_action_indices"][row_index, selected].item())
                        != int(batch["executor_action_indices"][row_index, 0].item())
                    )
                )
                nonclear.append(float(selected != 0))
                source = candidates[selected].target.source
                selected_sources[source] = selected_sources.get(source, 0) + 1
                for prediction, label in zip(scores[1:], labels[1:], strict=True):
                    actual = float(label.item()) > 0.0
                    predicted = float(prediction.item()) > 0.0
                    true_positive += int(actual)
                    predicted_positive += int(predicted)
                    joint_positive += int(actual and predicted)
                    pair_count += 1
    baseline_regret = statistics.fmean(baseline_regrets)
    prevalence = true_positive / max(pair_count, 1)
    return {
        "states": len(deltas),
        "selected_minus_baseline_mean": statistics.fmean(deltas),
        "selected_minus_baseline_ci95": (
            _bootstrap_ci(deltas, bootstrap_samples, seed)
            if bootstrap_samples
            else [statistics.fmean(deltas), statistics.fmean(deltas)]
        ),
        "model_mean_regret": statistics.fmean(model_regrets),
        "baseline_mean_regret": baseline_regret,
        "regret_ratio": (
            statistics.fmean(model_regrets) / baseline_regret if baseline_regret else 0.0
        ),
        "mean_rank_spearman": statistics.fmean(rank_spearmans),
        "action_movement": statistics.fmean(movement),
        "nonclear_target_rate": statistics.fmean(nonclear),
        "beneficial_pair_prevalence": prevalence,
        "predicted_positive_rate": predicted_positive / max(pair_count, 1),
        "positive_recall": joint_positive / max(true_positive, 1),
        "positive_precision": joint_positive / max(predicted_positive, 1),
        "selected_target_sources": selected_sources,
    }


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    draws = sorted(
        statistics.fmean(values[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    return [draws[int(0.025 * (samples - 1))], draws[int(0.975 * (samples - 1))]]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_payload(config: ExecutorConditionedTrainingConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }


def _validate_training_config(config: ExecutorConditionedTrainingConfig) -> None:
    if (
        config.epochs,
        config.batch_size,
        config.fit_states,
        config.screen_states,
        config.seed,
    ) != (50, 16, 800, 200, 783_083):
        raise ValueError("083 epoch, batch, split, and seed contract is frozen.")
    if (
        config.learning_rate,
        config.weight_decay,
        config.listwise_temperature,
        config.advantage_mse_weight,
        config.max_grad_norm,
    ) != (3e-4, 1e-4, 0.10, 1.0, 1.0):
        raise ValueError("083 optimizer and direct-advantage loss contract is frozen.")
