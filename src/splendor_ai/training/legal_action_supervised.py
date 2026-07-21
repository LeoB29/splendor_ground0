"""Supervised training loop for legal-action scorer models."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import (
    LegalActionReplayDataset,
    LegalActionReplaySample,
    collate_legal_action_replay_samples,
)
from .model import (
    LegalActionAttentionScorerModel,
    LegalActionScorerModel,
    LegalActionSetScorerModel,
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
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel
    metrics: tuple[EpochMetrics, ...]
    best_epoch_index: int | None = None
    best_model_state_dict: dict[str, torch.Tensor] | None = None


def split_legal_action_replay_dataset(
    dataset: LegalActionReplayDataset,
    validation_fraction: float = 0.0,
    seed: int = 0,
    validation_split_unit: str = "row",
) -> tuple[Dataset[LegalActionReplaySample], Dataset[LegalActionReplaySample] | None]:
    """Split legal-action data by row, whole game, or whole source corpus."""

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
        raise ValueError("Cannot split an empty legal-action replay dataset.")
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
    validation_groups = {
        group_keys[index]
        for index in shuffled_group_indices[:validation_group_count]
    }

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for group_key, sample_indices in group_to_indices.items():
        if group_key in validation_groups:
            validation_indices.extend(sample_indices)
        else:
            train_indices.extend(sample_indices)

    if not train_indices or not validation_indices:
        raise ValueError("Legal-action replay split must produce at least one train and one validation sample.")

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def create_legal_action_replay_dataloader(
    path: str | Path | tuple[str | Path, ...] | list[str | Path] | Dataset[LegalActionReplaySample],
    batch_size: int,
    shuffle: bool = True,
    include_stalled_games: bool = True,
    include_timed_out_games: bool = True,
    max_model_loop_fallback_triggers: int | None = None,
    observation_size: int | None = None,
    action_feature_size: int | None = None,
) -> DataLoader:
    if isinstance(path, Dataset):
        dataset = path
    else:
        dataset = LegalActionReplayDataset(
            path,
            include_stalled_games=include_stalled_games,
            include_timed_out_games=include_timed_out_games,
            max_model_loop_fallback_triggers=max_model_loop_fallback_triggers,
            observation_size=observation_size,
            action_feature_size=action_feature_size,
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
    search_value_target: torch.Tensor | None = None,
    search_value_mask: torch.Tensor | None = None,
    search_distill_loss_weight: float = 0.0,
    search_distill_temperature: float = 1_000.0,
    root_visit_target: torch.Tensor | None = None,
    root_visit_mask: torch.Tensor | None = None,
    root_visit_loss_weight: float = 0.0,
    sample_weight: torch.Tensor | None = None,
    policy_sample_weight: torch.Tensor | None = None,
    distill_sample_weight: torch.Tensor | None = None,
    preference_competitor_row: torch.Tensor | None = None,
    preference_mask: torch.Tensor | None = None,
    preference_sample_weight: torch.Tensor | None = None,
    preference_loss_weight: float = 0.0,
    preference_margin: float = 1.0,
    policy_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if policy_loss_weight < 0.0:
        raise ValueError("policy_loss_weight must be non-negative.")
    masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)
    policy_loss = _weighted_mean(
        nn.functional.cross_entropy(masked_scores, target_action_row, reduction="none"),
        policy_sample_weight if policy_sample_weight is not None else sample_weight,
        allow_zero_weight=True,
    )
    value_loss = _weighted_mean(
        nn.functional.mse_loss(value_pred, value_target, reduction="none"),
        sample_weight,
    )
    distill_loss = compute_search_score_distillation_loss(
        action_scores=action_scores,
        legal_action_mask=legal_action_mask,
        search_value_target=search_value_target,
        search_value_mask=search_value_mask,
        temperature=search_distill_temperature,
        sample_weight=distill_sample_weight if distill_sample_weight is not None else sample_weight,
    )
    root_visit_loss = compute_root_visit_distillation_loss(
        action_scores=action_scores,
        legal_action_mask=legal_action_mask,
        root_visit_target=root_visit_target,
        root_visit_mask=root_visit_mask,
        sample_weight=distill_sample_weight if distill_sample_weight is not None else sample_weight,
    )
    preference_loss = compute_pairwise_preference_loss(
        action_scores=action_scores,
        legal_action_mask=legal_action_mask,
        target_action_row=target_action_row,
        preference_competitor_row=preference_competitor_row,
        preference_mask=preference_mask,
        margin=preference_margin,
        sample_weight=preference_sample_weight,
    )
    total_loss = (
        policy_loss_weight * policy_loss
        + value_loss_weight * value_loss
        + search_distill_loss_weight * distill_loss
        + root_visit_loss_weight * root_visit_loss
        + preference_loss_weight * preference_loss
    )
    return policy_loss, value_loss, total_loss


def compute_search_score_distillation_loss(
    action_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    search_value_target: torch.Tensor | None,
    search_value_mask: torch.Tensor | None,
    temperature: float = 1_000.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if search_value_target is None or search_value_mask is None:
        return action_scores.sum() * 0.0
    if temperature <= 0.0:
        raise ValueError("search distillation temperature must be positive.")

    effective_mask = legal_action_mask & search_value_mask
    valid_rows = effective_mask.sum(dim=1) >= 2
    if not bool(valid_rows.any().item()):
        return action_scores.sum() * 0.0

    row_scores = action_scores[valid_rows]
    row_targets = search_value_target[valid_rows]
    row_mask = effective_mask[valid_rows]

    model_logits = row_scores.masked_fill(~row_mask, -torch.inf)
    target_logits = (row_targets / temperature).masked_fill(~row_mask, -torch.inf)
    model_log_probs = nn.functional.log_softmax(model_logits, dim=1)
    target_probs = nn.functional.softmax(target_logits, dim=1)
    model_log_probs = model_log_probs.masked_fill(~row_mask, 0.0)
    target_probs = target_probs.masked_fill(~row_mask, 0.0)
    per_sample_loss = -(target_probs.detach() * model_log_probs).sum(dim=1)
    valid_weights = None if sample_weight is None else sample_weight[valid_rows]
    return _weighted_mean(per_sample_loss, valid_weights, allow_zero_weight=True)


def compute_root_visit_distillation_loss(
    action_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    root_visit_target: torch.Tensor | None,
    root_visit_mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy against stored PUCT root-visit probabilities."""

    if root_visit_target is None or root_visit_mask is None:
        return action_scores.sum() * 0.0
    if root_visit_target.shape != action_scores.shape:
        raise ValueError("root_visit_target must match action_scores shape.")
    valid_rows = root_visit_mask.to(device=action_scores.device, dtype=torch.bool)
    valid_rows &= (root_visit_target.sum(dim=1) > 0.0).to(action_scores.device)
    if not bool(valid_rows.any().item()):
        return action_scores.sum() * 0.0
    row_mask = legal_action_mask[valid_rows]
    targets = root_visit_target[valid_rows].to(device=action_scores.device)
    targets = targets.masked_fill(~row_mask, 0.0)
    target_totals = targets.sum(dim=1, keepdim=True)
    targets = targets / target_totals.clamp_min(torch.finfo(targets.dtype).eps)
    logits = action_scores[valid_rows].masked_fill(~row_mask, -torch.inf)
    log_probs = nn.functional.log_softmax(logits, dim=1).masked_fill(~row_mask, 0.0)
    per_sample_loss = -(targets.detach() * log_probs).sum(dim=1)
    valid_weights = None if sample_weight is None else sample_weight[valid_rows]
    return _weighted_mean(per_sample_loss, valid_weights, allow_zero_weight=True)


def compute_pairwise_preference_loss(
    action_scores: torch.Tensor,
    legal_action_mask: torch.Tensor,
    target_action_row: torch.Tensor,
    preference_competitor_row: torch.Tensor | None,
    preference_mask: torch.Tensor | None,
    margin: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if preference_competitor_row is None or preference_mask is None:
        return action_scores.sum() * 0.0
    if margin < 0.0:
        raise ValueError("preference margin must be non-negative.")
    row_indices = torch.arange(action_scores.shape[0], device=action_scores.device)
    target_rows = target_action_row.to(device=action_scores.device)
    competitor_rows = preference_competitor_row.to(device=action_scores.device)
    row_mask = preference_mask.to(device=action_scores.device, dtype=torch.bool)
    valid_rows = (
        row_mask
        & legal_action_mask[row_indices, target_rows]
        & legal_action_mask[row_indices, competitor_rows]
        & (target_rows != competitor_rows)
    )
    if not bool(valid_rows.any().item()):
        return action_scores.sum() * 0.0
    teacher_scores = action_scores[valid_rows, target_rows[valid_rows]]
    competitor_scores = action_scores[valid_rows, competitor_rows[valid_rows]]
    per_sample_loss = nn.functional.softplus(competitor_scores - teacher_scores + margin)
    valid_weights = None if sample_weight is None else sample_weight.to(action_scores.device)[valid_rows]
    return _weighted_mean(per_sample_loss, valid_weights, allow_zero_weight=True)


def _weighted_mean(
    loss: torch.Tensor,
    sample_weight: torch.Tensor | None,
    allow_zero_weight: bool = False,
) -> torch.Tensor:
    if sample_weight is None:
        return loss.mean()
    weights = sample_weight.to(device=loss.device, dtype=loss.dtype)
    if weights.shape != loss.shape:
        raise ValueError(
            f"sample_weight shape must match per-sample loss shape: {weights.shape} != {loss.shape}"
        )
    weight_sum = weights.sum()
    if float(weight_sum.detach().cpu()) <= 0.0:
        if allow_zero_weight:
            return loss.sum() * 0.0
        raise ValueError("sample_weight sum must be positive.")
    return (loss * weights).sum() / weight_sum


def train_legal_action_scorer_epoch(
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
    epoch_index: int = 1,
    total_epochs: int = 1,
    log_every_batches: int = 0,
    progress_callback: Callable[[BatchProgress], None] | None = None,
    search_distill_loss_weight: float = 0.0,
    search_distill_temperature: float = 1_000.0,
    root_visit_loss_weight: float = 0.0,
    preference_loss_weight: float = 0.0,
    preference_margin: float = 1.0,
    policy_loss_weight: float = 1.0,
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
        search_value_target = batch["search_value_target"].to(device)
        search_value_mask = batch["search_value_mask"].to(device)
        root_visit_target = batch["root_visit_target"].to(device)
        root_visit_mask = batch["root_visit_mask"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        policy_sample_weight = batch["policy_sample_weight"].to(device)
        distill_sample_weight = batch["distill_sample_weight"].to(device)
        preference_competitor_row = batch["preference_competitor_row"].to(device)
        preference_mask = batch["preference_mask"].to(device)
        preference_sample_weight = batch["preference_sample_weight"].to(device)

        optimizer.zero_grad(set_to_none=True)
        action_scores, value_pred = model(
            observation,
            legal_action_features,
            legal_action_mask,
        )
        policy_loss, value_loss, loss = compute_legal_action_supervised_losses(
            action_scores=action_scores,
            value_pred=value_pred,
            target_action_row=target_action_row,
            legal_action_mask=legal_action_mask,
            value_target=value_target,
            value_loss_weight=value_loss_weight,
            search_value_target=search_value_target,
            search_value_mask=search_value_mask,
            search_distill_loss_weight=search_distill_loss_weight,
            search_distill_temperature=search_distill_temperature,
            root_visit_target=root_visit_target,
            root_visit_mask=root_visit_mask,
            root_visit_loss_weight=root_visit_loss_weight,
            sample_weight=sample_weight,
            policy_sample_weight=policy_sample_weight,
            distill_sample_weight=distill_sample_weight,
            preference_competitor_row=preference_competitor_row,
            preference_mask=preference_mask,
            preference_sample_weight=preference_sample_weight,
            preference_loss_weight=preference_loss_weight,
            preference_margin=preference_margin,
            policy_loss_weight=policy_loss_weight,
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
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel,
    dataloader: DataLoader,
    device: str = "cpu",
    value_loss_weight: float = 1.0,
    search_distill_loss_weight: float = 0.0,
    search_distill_temperature: float = 1_000.0,
    root_visit_loss_weight: float = 0.0,
    preference_loss_weight: float = 0.0,
    preference_margin: float = 1.0,
    policy_loss_weight: float = 1.0,
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
            search_value_target = batch["search_value_target"].to(device)
            search_value_mask = batch["search_value_mask"].to(device)
            root_visit_target = batch["root_visit_target"].to(device)
            root_visit_mask = batch["root_visit_mask"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            policy_sample_weight = batch["policy_sample_weight"].to(device)
            distill_sample_weight = batch["distill_sample_weight"].to(device)
            preference_competitor_row = batch["preference_competitor_row"].to(device)
            preference_mask = batch["preference_mask"].to(device)
            preference_sample_weight = batch["preference_sample_weight"].to(device)

            action_scores, value_pred = model(
                observation,
                legal_action_features,
                legal_action_mask,
            )
            policy_loss, value_loss, loss = compute_legal_action_supervised_losses(
                action_scores=action_scores,
                value_pred=value_pred,
                target_action_row=target_action_row,
                legal_action_mask=legal_action_mask,
                value_target=value_target,
                value_loss_weight=value_loss_weight,
                search_value_target=search_value_target,
                search_value_mask=search_value_mask,
                search_distill_loss_weight=search_distill_loss_weight,
                search_distill_temperature=search_distill_temperature,
                root_visit_target=root_visit_target,
                root_visit_mask=root_visit_mask,
                root_visit_loss_weight=root_visit_loss_weight,
                sample_weight=sample_weight,
                policy_sample_weight=policy_sample_weight,
                distill_sample_weight=distill_sample_weight,
                preference_competitor_row=preference_competitor_row,
                preference_mask=preference_mask,
                preference_sample_weight=preference_sample_weight,
                preference_loss_weight=preference_loss_weight,
                preference_margin=preference_margin,
                policy_loss_weight=policy_loss_weight,
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
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel | None = None,
    log_every_batches: int = 0,
    batch_progress_callback: Callable[[BatchProgress], None] | None = None,
    search_distill_loss_weight: float = 0.0,
    search_distill_temperature: float = 1_000.0,
    root_visit_loss_weight: float = 0.0,
    preference_loss_weight: float = 0.0,
    preference_margin: float = 1.0,
    policy_loss_weight: float = 1.0,
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
            search_distill_loss_weight=search_distill_loss_weight,
            search_distill_temperature=search_distill_temperature,
            root_visit_loss_weight=root_visit_loss_weight,
            preference_loss_weight=preference_loss_weight,
            preference_margin=preference_margin,
            policy_loss_weight=policy_loss_weight,
        )
        validation_metrics = None
        if validation_dataloader is not None:
            validation_metrics = evaluate_legal_action_scorer_epoch(
                model=model_instance,
                dataloader=validation_dataloader,
                device=cfg.device,
                value_loss_weight=cfg.value_loss_weight,
                search_distill_loss_weight=search_distill_loss_weight,
                search_distill_temperature=search_distill_temperature,
                root_visit_loss_weight=root_visit_loss_weight,
                preference_loss_weight=preference_loss_weight,
                preference_margin=preference_margin,
                policy_loss_weight=policy_loss_weight,
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
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel | None = None,
    validation_split_unit: str = "row",
) -> tuple[
    LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel,
    tuple[EpochMetrics, ...],
]:
    cfg = config or SupervisedTrainConfig()
    model_instance = model or LegalActionScorerModel()
    dataset = LegalActionReplayDataset(
        replay_path,
        include_stalled_games=cfg.include_stalled_games,
        include_timed_out_games=cfg.include_timed_out_games,
        max_model_loop_fallback_triggers=cfg.max_model_loop_fallback_triggers,
        observation_size=model_instance.config.observation_size,
        action_feature_size=model_instance.config.action_feature_size,
    )
    train_dataset, validation_dataset = split_legal_action_replay_dataset(
        dataset,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.validation_seed,
        validation_split_unit=validation_split_unit,
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


def _clone_model_state_dict(
    model: LegalActionScorerModel | LegalActionSetScorerModel | LegalActionAttentionScorerModel,
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
