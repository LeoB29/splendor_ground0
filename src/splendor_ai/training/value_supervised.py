"""Supervised training loop for state-value models."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import ValueReplayDataset, ValueReplaySample, collate_value_replay_samples
from .model import ValueMLP
from .supervised import BatchProgress, SupervisedTrainConfig, split_replay_dataset


@dataclass(frozen=True, slots=True)
class ValueTrainMetrics:
    mse_loss: float
    mae: float
    sign_accuracy: float
    outcome_sign_accuracy: float
    pearson_correlation: float
    samples: int
    batches: int


@dataclass(frozen=True, slots=True)
class ValueEpochMetrics:
    train: ValueTrainMetrics
    validation: ValueTrainMetrics | None = None


@dataclass(frozen=True, slots=True)
class FitValueArtifacts:
    model: ValueMLP
    metrics: tuple[ValueEpochMetrics, ...]
    best_epoch_index: int | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None


def split_value_replay_dataset(
    dataset: ValueReplayDataset,
    validation_fraction: float = 0.0,
    seed: int = 0,
    validation_split_unit: str = "row",
) -> tuple[Dataset[ValueReplaySample], Dataset[ValueReplaySample] | None]:
    """Split value data by row, whole game, or whole source corpus."""

    if validation_split_unit == "row":
        return split_replay_dataset(
            dataset,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    if validation_split_unit not in {"game", "source"}:
        raise ValueError("validation_split_unit must be 'row', 'game', or 'source'.")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0).")

    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Cannot split an empty value replay dataset.")
    if validation_fraction == 0.0 or dataset_size == 1:
        return dataset, None

    group_to_indices: dict[tuple[object, ...], list[int]] = {}
    for sample_index in range(dataset_size):
        group_key = dataset.validation_group_key_for_index(sample_index, validation_split_unit)
        group_to_indices.setdefault(group_key, []).append(sample_index)

    group_keys = list(group_to_indices)
    if len(group_keys) == 1:
        return dataset, None

    validation_group_count = int(len(group_keys) * validation_fraction)
    if validation_group_count == 0:
        validation_group_count = 1
    if validation_group_count >= len(group_keys):
        validation_group_count = len(group_keys) - 1

    generator = torch.Generator().manual_seed(seed)
    shuffled_group_indices = torch.randperm(len(group_keys), generator=generator).tolist()
    validation_group_indices = set(shuffled_group_indices[:validation_group_count])
    validation_groups = {group_keys[index] for index in validation_group_indices}

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for group_key, sample_indices in group_to_indices.items():
        if group_key in validation_groups:
            validation_indices.extend(sample_indices)
        else:
            train_indices.extend(sample_indices)

    if not train_indices or not validation_indices:
        raise ValueError("Value replay split must produce at least one train and one validation sample.")

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def create_value_replay_dataloader(
    path: str | Path | tuple[str | Path, ...] | list[str | Path] | Dataset[ValueReplaySample],
    batch_size: int,
    shuffle: bool = True,
    include_stalled_games: bool = True,
    include_timed_out_games: bool = True,
    max_model_loop_fallback_triggers: int | None = None,
    margin_scale: float = 15.0,
) -> DataLoader:
    if isinstance(path, Dataset):
        dataset = path
    else:
        dataset = ValueReplayDataset(
            path,
            include_stalled_games=include_stalled_games,
            include_timed_out_games=include_timed_out_games,
            max_model_loop_fallback_triggers=max_model_loop_fallback_triggers,
            margin_scale=margin_scale,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_value_replay_samples,
    )


def compute_value_loss(
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
) -> torch.Tensor:
    return nn.functional.mse_loss(value_pred, value_target)


def train_value_epoch(
    model: ValueMLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    epoch_index: int = 1,
    total_epochs: int = 1,
    log_every_batches: int = 0,
    progress_callback: Callable[[BatchProgress], None] | None = None,
) -> ValueTrainMetrics:
    model.train()
    total_batches = len(dataloader)
    start_time = time.perf_counter()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    batch_count = 0

    for batch_index, batch in enumerate(dataloader, start=1):
        observation = batch["observation"].to(device)
        value_target = batch["value_target"].to(device)
        outcome_target = batch["outcome_target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        value_pred = model(observation)
        loss = compute_value_loss(value_pred, value_target)
        loss.backward()
        optimizer.step()

        predictions.append(value_pred.detach().cpu())
        targets.append(value_target.detach().cpu())
        outcomes.append(outcome_target.detach().cpu())
        batch_count += 1

        if progress_callback is not None and (
            batch_index == 1
            or batch_index == total_batches
            or (log_every_batches > 0 and batch_index % log_every_batches == 0)
        ):
            metrics = _metrics_from_tensors(
                torch.cat(predictions),
                torch.cat(targets),
                torch.cat(outcomes),
                batch_count,
            )
            progress_callback(
                BatchProgress(
                    phase="train",
                    epoch_index=epoch_index,
                    total_epochs=total_epochs,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    elapsed_seconds=time.perf_counter() - start_time,
                    policy_loss=0.0,
                    value_loss=metrics.mse_loss,
                    total_loss=metrics.mse_loss,
                    policy_accuracy=metrics.sign_accuracy,
                    samples=metrics.samples,
                )
            )

    if batch_count == 0:
        raise ValueError("Training dataloader produced zero batches.")
    return _metrics_from_tensors(torch.cat(predictions), torch.cat(targets), torch.cat(outcomes), batch_count)


def evaluate_value_epoch(
    model: ValueMLP,
    dataloader: DataLoader,
    device: str = "cpu",
) -> ValueTrainMetrics:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    batch_count = 0
    with torch.no_grad():
        for batch in dataloader:
            observation = batch["observation"].to(device)
            value_target = batch["value_target"].to(device)
            outcome_target = batch["outcome_target"].to(device)
            value_pred = model(observation)
            predictions.append(value_pred.detach().cpu())
            targets.append(value_target.detach().cpu())
            outcomes.append(outcome_target.detach().cpu())
            batch_count += 1
    if batch_count == 0:
        raise ValueError("Validation dataloader produced zero batches.")
    return _metrics_from_tensors(torch.cat(predictions), torch.cat(targets), torch.cat(outcomes), batch_count)


def fit_value_dataloaders_with_artifacts(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None = None,
    config: SupervisedTrainConfig | None = None,
    model: ValueMLP | None = None,
    log_every_batches: int = 0,
    batch_progress_callback: Callable[[BatchProgress], None] | None = None,
) -> FitValueArtifacts:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or ValueMLP()
    model_instance.to(cfg.device)
    optimizer = AdamW(model_instance.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    metrics: list[ValueEpochMetrics] = []
    best_epoch_index: int | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_loss: float | None = None
    for epoch_index in range(1, cfg.epochs + 1):
        train_metrics = train_value_epoch(
            model=model_instance,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=cfg.device,
            epoch_index=epoch_index,
            total_epochs=cfg.epochs,
            log_every_batches=log_every_batches,
            progress_callback=batch_progress_callback,
        )
        validation_metrics = None
        if validation_dataloader is not None:
            validation_metrics = evaluate_value_epoch(
                model=model_instance,
                dataloader=validation_dataloader,
                device=cfg.device,
            )
            if best_validation_loss is None or validation_metrics.mse_loss < best_validation_loss:
                best_validation_loss = validation_metrics.mse_loss
                best_epoch_index = epoch_index
                best_model_state_dict = _clone_model_state_dict(model_instance)
        metrics.append(ValueEpochMetrics(train=train_metrics, validation=validation_metrics))

    return FitValueArtifacts(
        model=model_instance,
        metrics=tuple(metrics),
        best_epoch_index=best_epoch_index,
        best_model_state_dict=best_model_state_dict,
    )


def fit_value(
    replay_path: str | Path | tuple[str | Path, ...] | list[str | Path],
    config: SupervisedTrainConfig | None = None,
    model: ValueMLP | None = None,
    margin_scale: float = 15.0,
    validation_split_unit: str = "row",
) -> tuple[ValueMLP, tuple[ValueEpochMetrics, ...]]:
    cfg = config or SupervisedTrainConfig()
    dataset = ValueReplayDataset(
        replay_path,
        include_stalled_games=cfg.include_stalled_games,
        include_timed_out_games=cfg.include_timed_out_games,
        max_model_loop_fallback_triggers=cfg.max_model_loop_fallback_triggers,
        margin_scale=margin_scale,
    )
    train_dataset, validation_dataset = split_value_replay_dataset(
        dataset,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.validation_seed,
        validation_split_unit=validation_split_unit,
    )
    train_dataloader = create_value_replay_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_value_replay_dataloader(
            validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
        )
    artifacts = fit_value_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=cfg,
        model=model,
    )
    return artifacts.model, artifacts.metrics


def _metrics_from_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    outcomes: torch.Tensor,
    batches: int,
) -> ValueTrainMetrics:
    errors = predictions - targets
    mse = float(torch.mean(errors.square()).item())
    mae = float(torch.mean(errors.abs()).item())
    sign_accuracy = _sign_accuracy(predictions, targets)
    outcome_sign_accuracy = _sign_accuracy(predictions, outcomes)
    pearson = _pearson_correlation(predictions, targets)
    return ValueTrainMetrics(
        mse_loss=mse,
        mae=mae,
        sign_accuracy=sign_accuracy,
        outcome_sign_accuracy=outcome_sign_accuracy,
        pearson_correlation=pearson,
        samples=int(targets.numel()),
        batches=batches,
    )


def _sign_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    mask = targets != 0
    if not bool(mask.any().item()):
        return math.nan
    pred_sign = torch.sign(predictions[mask])
    target_sign = torch.sign(targets[mask])
    return float((pred_sign == target_sign).float().mean().item())


def _pearson_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    denom = torch.linalg.vector_norm(pred_centered) * torch.linalg.vector_norm(target_centered)
    if float(denom.item()) == 0.0:
        return math.nan
    return float((pred_centered * target_centered).sum().div(denom).item())


def _clone_model_state_dict(model: ValueMLP) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
