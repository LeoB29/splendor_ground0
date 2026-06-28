"""Supervised training loop for legal-action scorer models."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .dataset import (
    LegalActionReplayDataset,
    LegalActionReplaySample,
    collate_legal_action_replay_samples,
)
from .model import (
    LegalActionScorerModel,
    masked_legal_action_scores,
)
from .supervised import (
    BatchProgress,
    EpochMetrics,
    SupervisedTrainConfig,
    TrainMetrics,
    split_replay_dataset,
)


@dataclass(frozen=True, slots=True)
class FitLegalActionScorerArtifacts:
    model: LegalActionScorerModel
    metrics: tuple[EpochMetrics, ...]
    best_epoch_index: int | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None


def create_legal_action_replay_dataloader(
    path: str | Path | tuple[str | Path, ...] | list[str | Path] | Dataset[LegalActionReplaySample],
    batch_size: int,
    shuffle: bool = True,
    include_stalled_games: bool = True,
    include_timed_out_games: bool = True,
    max_model_loop_fallback_triggers: int | None = None,
) -> DataLoader:
    if isinstance(path, Dataset):
        dataset = path
    else:
        dataset = LegalActionReplayDataset(
            path,
            include_stalled_games=include_stalled_games,
            include_timed_out_games=include_timed_out_games,
            max_model_loop_fallback_triggers=max_model_loop_fallback_triggers,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_legal_action_replay_samples,
    )


def compute_legal_action_supervised_losses(
    action_scores: torch.Tensor,
    value_pred: torch.Tensor,
    target_action_row: torch.Tensor,
    legal_action_mask: torch.Tensor,
    value_target: torch.Tensor,
    value_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)
    policy_loss = nn.functional.cross_entropy(masked_scores, target_action_row)
    value_loss = nn.functional.mse_loss(value_pred, value_target)
    total_loss = policy_loss + value_loss_weight * value_loss
    return policy_loss, value_loss, total_loss


def train_legal_action_scorer_epoch(
    model: LegalActionScorerModel,
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
        legal_action_features = batch["legal_action_features"].to(device)
        target_action_row = batch["target_action_row"].to(device)
        legal_action_mask = batch["legal_action_mask"].to(device)
        value_target = batch["value_target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        action_scores, value_pred = model(observation, legal_action_features)
        policy_loss, value_loss, loss = compute_legal_action_supervised_losses(
            action_scores=action_scores,
            value_pred=value_pred,
            target_action_row=target_action_row,
            legal_action_mask=legal_action_mask,
            value_target=value_target,
            value_loss_weight=value_loss_weight,
        )
        predictions = torch.argmax(
            masked_legal_action_scores(action_scores, legal_action_mask),
            dim=1,
        )
        loss.backward()
        optimizer.step()

        total_policy += float(policy_loss.detach().cpu())
        total_value += float(value_loss.detach().cpu())
        total_loss += float(loss.detach().cpu())
        total_correct += int((predictions == target_action_row).sum().item())
        total_samples += int(target_action_row.numel())
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


def evaluate_legal_action_scorer_epoch(
    model: LegalActionScorerModel,
    dataloader: DataLoader,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
) -> TrainMetrics:
    model.eval()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            observation = batch["observation"].to(device)
            legal_action_features = batch["legal_action_features"].to(device)
            target_action_row = batch["target_action_row"].to(device)
            legal_action_mask = batch["legal_action_mask"].to(device)
            value_target = batch["value_target"].to(device)

            action_scores, value_pred = model(observation, legal_action_features)
            policy_loss, value_loss, loss = compute_legal_action_supervised_losses(
                action_scores=action_scores,
                value_pred=value_pred,
                target_action_row=target_action_row,
                legal_action_mask=legal_action_mask,
                value_target=value_target,
                value_loss_weight=value_loss_weight,
            )
            predictions = torch.argmax(
                masked_legal_action_scores(action_scores, legal_action_mask),
                dim=1,
            )

            total_policy += float(policy_loss.detach().cpu())
            total_value += float(value_loss.detach().cpu())
            total_loss += float(loss.detach().cpu())
            total_correct += int((predictions == target_action_row).sum().item())
            total_samples += int(target_action_row.numel())
            batch_count += 1

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


def fit_legal_action_scorer_dataloaders_with_artifacts(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None = None,
    config: SupervisedTrainConfig | None = None,
    model: LegalActionScorerModel | None = None,
    log_every_batches: int = 0,
    batch_progress_callback: Callable[[BatchProgress], None] | None = None,
) -> FitLegalActionScorerArtifacts:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or LegalActionScorerModel()
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
        train_metrics = train_legal_action_scorer_epoch(
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
            validation_metrics = evaluate_legal_action_scorer_epoch(
                model=model_instance,
                dataloader=validation_dataloader,
                device=cfg.device,
                value_loss_weight=cfg.value_loss_weight,
            )
            if best_validation_loss is None or validation_metrics.total_loss < best_validation_loss:
                best_validation_loss = validation_metrics.total_loss
                best_epoch_index = epoch_index
                best_model_state_dict = _clone_model_state_dict(model_instance)
        metrics.append(EpochMetrics(train=train_metrics, validation=validation_metrics))

    return FitLegalActionScorerArtifacts(
        model=model_instance,
        metrics=tuple(metrics),
        best_epoch_index=best_epoch_index,
        best_model_state_dict=best_model_state_dict,
    )


def fit_legal_action_scorer(
    replay_path: str | Path | tuple[str | Path, ...] | list[str | Path],
    config: SupervisedTrainConfig | None = None,
    model: LegalActionScorerModel | None = None,
) -> tuple[LegalActionScorerModel, tuple[EpochMetrics, ...]]:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or LegalActionScorerModel()
    dataset = LegalActionReplayDataset(
        replay_path,
        include_stalled_games=cfg.include_stalled_games,
        include_timed_out_games=cfg.include_timed_out_games,
        max_model_loop_fallback_triggers=cfg.max_model_loop_fallback_triggers,
    )
    train_dataset, validation_dataset = split_replay_dataset(
        dataset,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.validation_seed,
    )
    train_dataloader = create_legal_action_replay_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
    )
    validation_dataloader = None
    if validation_dataset is not None:
        validation_dataloader = create_legal_action_replay_dataloader(
            validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
        )
    artifacts = fit_legal_action_scorer_dataloaders_with_artifacts(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        config=cfg,
        model=model_instance,
    )
    return artifacts.model, artifacts.metrics


def _clone_model_state_dict(model: LegalActionScorerModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
