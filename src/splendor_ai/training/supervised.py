"""Supervised warm-start training loop for replay data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

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


@dataclass(frozen=True, slots=True)
class TrainMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float
    batches: int


def create_replay_dataloader(
    path: str | Path,
    batch_size: int,
    action_space_size: int,
    shuffle: bool = True,
    include_stalled_games: bool = True,
    include_timed_out_games: bool = True,
) -> DataLoader:
    dataset = SupervisedReplayDataset(
        path,
        include_stalled_games=include_stalled_games,
        include_timed_out_games=include_timed_out_games,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda samples: collate_replay_samples(
            samples,
            action_space_size=action_space_size,
        ),
    )


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
) -> TrainMetrics:
    model.train()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    batch_count = 0

    for batch in dataloader:
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
        loss.backward()
        optimizer.step()

        total_policy += float(policy_loss.detach().cpu())
        total_value += float(value_loss.detach().cpu())
        total_loss += float(loss.detach().cpu())
        batch_count += 1

    if batch_count == 0:
        raise ValueError("Training dataloader produced zero batches.")

    return TrainMetrics(
        policy_loss=total_policy / batch_count,
        value_loss=total_value / batch_count,
        total_loss=total_loss / batch_count,
        batches=batch_count,
    )


def fit_supervised(
    replay_path: str | Path,
    config: SupervisedTrainConfig | None = None,
    model: PolicyValueMLP | None = None,
) -> tuple[PolicyValueMLP, tuple[TrainMetrics, ...]]:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or PolicyValueMLP()
    model_instance.to(cfg.device)
    optimizer = AdamW(
        model_instance.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    dataloader = create_replay_dataloader(
        path=replay_path,
        batch_size=cfg.batch_size,
        action_space_size=model_instance.config.action_space_size,
        shuffle=cfg.shuffle,
        include_stalled_games=cfg.include_stalled_games,
        include_timed_out_games=cfg.include_timed_out_games,
    )

    metrics: list[TrainMetrics] = []
    for _ in range(cfg.epochs):
        metrics.append(
            train_supervised_epoch(
                model=model_instance,
                dataloader=dataloader,
                optimizer=optimizer,
                device=cfg.device,
                value_loss_weight=cfg.value_loss_weight,
            )
        )

    return model_instance, tuple(metrics)
