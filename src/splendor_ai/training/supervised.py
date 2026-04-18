"""Supervised warm-start training loop for replay data."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import SupervisedReplayDataset, collate_replay_samples
from .model import PolicyValueMLP, masked_policy_logits


@dataclass(frozen=True, slots=True)
class SupervisedTrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    epochs: int = 1
    device: str = "cpu"
    shuffle: bool = True
    include_stalled_games: bool = True
    include_timed_out_games: bool = True
    validation_fraction: float = 0.0
    validation_seed: int = 0


@dataclass(frozen=True, slots=True)
class TrainMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float
    policy_accuracy: float
    samples: int
    batches: int


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    train: TrainMetrics
    validation: TrainMetrics | None = None


@dataclass(frozen=True, slots=True)
class BatchProgress:
    phase: str
    epoch_index: int
    total_epochs: int
    batch_index: int
    total_batches: int
    elapsed_seconds: float
    policy_loss: float
    value_loss: float
    total_loss: float
    policy_accuracy: float
    samples: int


@dataclass(frozen=True, slots=True)
class FitSupervisedArtifacts:
    model: PolicyValueMLP
    metrics: tuple[EpochMetrics, ...]
    best_epoch_index: int | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None


def _create_dataloader(
    dataset: Dataset[ReplaySample],
    batch_size: int,
    action_space_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda samples: collate_replay_samples(
            samples,
            action_space_size=action_space_size,
        ),
    )


def create_replay_dataloader(
    path: str | Path | Dataset[ReplaySample],
    batch_size: int,
    action_space_size: int,
    shuffle: bool = True,
    include_stalled_games: bool = True,
    include_timed_out_games: bool = True,
) -> DataLoader:
    if isinstance(path, Dataset):
        dataset = path
    else:
        dataset = SupervisedReplayDataset(
            path,
            include_stalled_games=include_stalled_games,
            include_timed_out_games=include_timed_out_games,
        )
    return _create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        action_space_size=action_space_size,
        shuffle=shuffle,
    )


def split_replay_dataset(
    dataset: Dataset[ReplaySample],
    validation_fraction: float = 0.0,
    seed: int = 0,
) -> tuple[Dataset[ReplaySample], Dataset[ReplaySample] | None]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the range [0.0, 1.0).")
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Cannot split an empty replay dataset.")
    if validation_fraction == 0.0 or dataset_size == 1:
        return dataset, None

    validation_size = int(dataset_size * validation_fraction)
    if validation_size == 0:
        validation_size = 1
    if validation_size >= dataset_size:
        validation_size = dataset_size - 1

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]

    if not train_indices or not validation_indices:
        raise ValueError("Replay split must produce at least one train and one validation sample.")

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def compute_supervised_losses(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    action_index: torch.Tensor,
    legal_action_mask: torch.Tensor,
    value_target: torch.Tensor,
    value_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_logits = masked_policy_logits(policy_logits, legal_action_mask)
    policy_loss = nn.functional.cross_entropy(masked_logits, action_index)
    value_loss = nn.functional.mse_loss(value_pred, value_target)
    total_loss = policy_loss + value_loss_weight * value_loss
    return policy_loss, value_loss, total_loss


def train_supervised_epoch(
    model: PolicyValueMLP,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
    epoch_index: int = 1,
    total_epochs: int = 1,
    log_every_batches: int = 0,
    progress_callback: Callable[[BatchProgress], None] | None = None,
) -> TrainMetrics:
    model.train()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    total_batches = len(dataloader)
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(dataloader, start=1):
        observation = batch["observation"].to(device)
        action_index = batch["action_index"].to(device)
        legal_action_mask = batch["legal_action_mask"].to(device)
        value_target = batch["value_target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        policy_logits, value_pred = model(observation)
        policy_loss, value_loss, loss = compute_supervised_losses(
            policy_logits=policy_logits,
            value_pred=value_pred,
            action_index=action_index,
            legal_action_mask=legal_action_mask,
            value_target=value_target,
            value_loss_weight=value_loss_weight,
        )
        predictions = torch.argmax(masked_policy_logits(policy_logits, legal_action_mask), dim=1)
        loss.backward()
        optimizer.step()

        total_policy += float(policy_loss.detach().cpu())
        total_value += float(value_loss.detach().cpu())
        total_loss += float(loss.detach().cpu())
        total_correct += int((predictions == action_index).sum().item())
        total_samples += int(action_index.numel())
        batch_count += 1
        if progress_callback is not None and (
            batch_index == 1
            or batch_index == total_batches
            or (log_every_batches > 0 and batch_index % log_every_batches == 0)
        ):
            progress_callback(
                BatchProgress(
                    phase="train",
                    epoch_index=epoch_index,
                    total_epochs=total_epochs,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    elapsed_seconds=time.perf_counter() - start_time,
                    policy_loss=total_policy / batch_count,
                    value_loss=total_value / batch_count,
                    total_loss=total_loss / batch_count,
                    policy_accuracy=total_correct / total_samples,
                    samples=total_samples,
                )
            )

    if batch_count == 0:
        raise ValueError("Training dataloader produced zero batches.")

    return TrainMetrics(
        policy_loss=total_policy / batch_count,
        value_loss=total_value / batch_count,
        total_loss=total_loss / batch_count,
        policy_accuracy=total_correct / total_samples,
        samples=total_samples,
        batches=batch_count,
    )


def evaluate_supervised_epoch(
    model: PolicyValueMLP,
    dataloader: DataLoader,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
    epoch_index: int = 1,
    total_epochs: int = 1,
    log_every_batches: int = 0,
    progress_callback: Callable[[BatchProgress], None] | None = None,
) -> TrainMetrics:
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0
    total_batches = len(dataloader)
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            observation = batch["observation"].to(device)
            action_index = batch["action_index"].to(device)
            legal_action_mask = batch["legal_action_mask"].to(device)
            value_target = batch["value_target"].to(device)

            policy_logits, value_pred = model(observation)
            policy_loss, value_loss, loss = compute_supervised_losses(
                policy_logits=policy_logits,
                value_pred=value_pred,
                action_index=action_index,
                legal_action_mask=legal_action_mask,
                value_target=value_target,
                value_loss_weight=value_loss_weight,
            )
            predictions = torch.argmax(masked_policy_logits(policy_logits, legal_action_mask), dim=1)

            total_policy += float(policy_loss.detach().cpu())
            total_value += float(value_loss.detach().cpu())
            total_loss += float(loss.detach().cpu())
            total_correct += int((predictions == action_index).sum().item())
            total_samples += int(action_index.numel())
            batch_count += 1
            if progress_callback is not None and (
                batch_index == 1
                or batch_index == total_batches
                or (log_every_batches > 0 and batch_index % log_every_batches == 0)
            ):
                progress_callback(
                    BatchProgress(
                        phase="validation",
                        epoch_index=epoch_index,
                        total_epochs=total_epochs,
                        batch_index=batch_index,
                        total_batches=total_batches,
                        elapsed_seconds=time.perf_counter() - start_time,
                        policy_loss=total_policy / batch_count,
                        value_loss=total_value / batch_count,
                        total_loss=total_loss / batch_count,
                        policy_accuracy=total_correct / total_samples,
                        samples=total_samples,
                    )
                )

    if batch_count == 0:
        raise ValueError("Validation dataloader produced zero batches.")

    return TrainMetrics(
        policy_loss=total_policy / batch_count,
        value_loss=total_value / batch_count,
        total_loss=total_loss / batch_count,
        policy_accuracy=total_correct / total_samples,
        samples=total_samples,
        batches=batch_count,
    )


def _clone_model_state_dict(model: PolicyValueMLP) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def fit_supervised_dataloaders_with_artifacts(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None = None,
    config: SupervisedTrainConfig | None = None,
    model: PolicyValueMLP | None = None,
    log_every_batches: int = 0,
    batch_progress_callback: Callable[[BatchProgress], None] | None = None,
) -> FitSupervisedArtifacts:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or PolicyValueMLP()
    model_instance.to(cfg.device)
    optimizer = AdamW(
        model_instance.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    metrics: list[EpochMetrics] = []
    best_epoch_index: int | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_loss: float | None = None
    for epoch_index in range(1, cfg.epochs + 1):
        train_metrics = train_supervised_epoch(
            model=model_instance,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=cfg.device,
            value_loss_weight=cfg.value_loss_weight,
            epoch_index=epoch_index,
            total_epochs=cfg.epochs,
            log_every_batches=log_every_batches,
            progress_callback=batch_progress_callback,
        )
        validation_metrics = None
        if validation_dataloader is not None:
            validation_metrics = evaluate_supervised_epoch(
                model=model_instance,
                dataloader=validation_dataloader,
                device=cfg.device,
                value_loss_weight=cfg.value_loss_weight,
                epoch_index=epoch_index,
                total_epochs=cfg.epochs,
                log_every_batches=log_every_batches,
                progress_callback=batch_progress_callback,
            )
            if best_validation_loss is None or validation_metrics.total_loss < best_validation_loss:
                best_validation_loss = validation_metrics.total_loss
                best_epoch_index = epoch_index
                best_model_state_dict = _clone_model_state_dict(model_instance)
        metrics.append(EpochMetrics(train=train_metrics, validation=validation_metrics))

    return FitSupervisedArtifacts(
        model=model_instance,
        metrics=tuple(metrics),
        best_epoch_index=best_epoch_index,
        best_model_state_dict=best_model_state_dict,
    )


def fit_supervised_dataloaders(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None = None,
    config: SupervisedTrainConfig | None = None,
    model: PolicyValueMLP | None = None,
) -> tuple[PolicyValueMLP, tuple[EpochMetrics, ...]]:
    artifacts = fit_supervised_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=config,
        model=model,
    )
    return artifacts.model, artifacts.metrics


def fit_supervised(
    replay_path: str | Path,
    config: SupervisedTrainConfig | None = None,
    model: PolicyValueMLP | None = None,
) -> tuple[PolicyValueMLP, tuple[EpochMetrics, ...]]:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or PolicyValueMLP()
    dataset = SupervisedReplayDataset(
        replay_path,
        include_stalled_games=cfg.include_stalled_games,
        include_timed_out_games=cfg.include_timed_out_games,
    )
    train_dataset, validation_dataset = split_replay_dataset(
        dataset,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.validation_seed,
    )
    train_dataloader = create_replay_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        action_space_size=model_instance.config.action_space_size,
        shuffle=cfg.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_replay_dataloader(
            validation_dataset,
            batch_size=cfg.batch_size,
            action_space_size=model_instance.config.action_space_size,
            shuffle=False,
        )
    return fit_supervised_dataloaders(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=cfg,
        model=model_instance,
    )
