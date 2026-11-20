"""Policy/value fitting from programme-090 search visits and outcomes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from splendor_ai.encoding.entity_encoding import ACTION_NUMERIC_SIZE, ENTITY_NUMERIC_SIZE

from .network import load_checkpoint, save_checkpoint
from .replay import AlphaZeroReplayRow, read_replay_jsonl


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 4
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 900090
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    bucket_batches: bool = False
    policy_divergence_weight: float = 0.0
    visit_argmax_aux_weight: float = 0.0

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("Invalid optimizer configuration.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0,1).")
        if self.policy_loss_weight <= 0.0 or self.value_loss_weight < 0.0:
            raise ValueError("Loss weights must be nonnegative and policy positive.")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when supplied.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be nonnegative.")
        if self.policy_divergence_weight < 0.0 or self.visit_argmax_aux_weight < 0.0:
            raise ValueError("Corrective policy weights must be nonnegative.")


class _ReplayDataset(Dataset[AlphaZeroReplayRow]):
    def __init__(self, rows: list[AlphaZeroReplayRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> AlphaZeroReplayRow:
        return self.rows[index]


def alpha_zero_loss(
    policy_logits: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    policy_row_weight: torch.Tensor | None = None,
    visit_argmax_target: torch.Tensor | None = None,
    decisive_mask: torch.Tensor | None = None,
    visit_argmax_aux_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    if policy_logits.shape != action_mask.shape or policy_logits.shape != policy_target.shape:
        raise ValueError("Policy logits, mask, and target must share shape.")
    masked_logits = policy_logits.masked_fill(~action_mask, torch.finfo(policy_logits.dtype).min)
    log_policy = torch.log_softmax(masked_logits, dim=1)
    row_policy_loss = -(policy_target * log_policy).sum(dim=1)
    policy_loss = row_policy_loss.mean()
    if policy_row_weight is None:
        policy_objective_loss = policy_loss
    else:
        if policy_row_weight.shape != row_policy_loss.shape:
            raise ValueError("Policy row weights must have one value per row.")
        policy_objective_loss = (row_policy_loss * policy_row_weight).sum() / policy_row_weight.sum().clamp_min(1e-12)
    decisive_aux_loss = policy_loss.new_zeros(())
    if visit_argmax_aux_weight > 0.0:
        if visit_argmax_target is None or decisive_mask is None:
            raise ValueError("Visit-argmax auxiliary loss requires targets and a decisive mask.")
        if decisive_mask.any():
            decisive_aux_loss = -log_policy[decisive_mask, visit_argmax_target[decisive_mask]].mean()
    value_loss = torch.mean((values - value_target) ** 2)
    return {
        "policy_loss": policy_loss,
        "policy_objective_loss": policy_objective_loss,
        "decisive_aux_loss": decisive_aux_loss,
        "value_loss": value_loss,
        "total_loss": policy_loss + value_loss,
        "objective_loss": policy_weight * (
            policy_objective_loss + visit_argmax_aux_weight * decisive_aux_loss
        ) + value_weight * value_loss,
    }


def train_successor(
    *,
    parent_checkpoint: str | Path,
    replay_paths: Iterable[str | Path],
    output_checkpoint: str | Path,
    metrics_path: str | Path,
    device: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    config.validate()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    paths = [Path(path) for path in replay_paths]
    rows = [row for path in paths for row in read_replay_jsonl(path)]
    if not rows:
        raise ValueError("No AlphaZero replay rows were supplied.")
    if any(row.value_target is None for row in rows):
        raise ValueError("Every training row must contain a terminal value target.")
    train_rows, validation_rows = split_rows_by_game(
        rows,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    model, parent = load_checkpoint(parent_checkpoint, device=device)
    child_generation = parent.generation + 1
    parent_state = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _make_loader(
        train_rows,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
        bucket_batches=config.bucket_batches,
        config=config,
    )
    validation_loader = (
        _make_loader(
            validation_rows,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            bucket_batches=config.bucket_batches,
            config=config,
        )
        if validation_rows
        else None
    )
    model.eval()
    parent_train_metrics = _run_epoch(model, train_loader, device, None, config)
    parent_validation_metrics = (
        _run_epoch(model, validation_loader, device, None, config)
        if validation_loader is not None
        else parent_train_metrics
    )
    model.train()
    history: list[dict[str, Any]] = []
    best_state = deepcopy(model.state_dict())
    best_validation = math.inf
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_metrics = _run_epoch(model, train_loader, device, optimizer, config)
        model.eval()
        validation_metrics = (
            _run_epoch(model, validation_loader, device, None, config)
            if validation_loader is not None
            else train_metrics
        )
        if (
            validation_metrics["objective_loss"]
            < best_validation - config.early_stopping_min_delta
        ):
            best_validation = validation_metrics["objective_loss"]
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            break
    model.load_state_dict(best_state)
    model.eval()
    child_train_metrics = _run_epoch(model, train_loader, device, None, config)
    child_validation_metrics = (
        _run_epoch(model, validation_loader, device, None, config)
        if validation_loader is not None
        else child_train_metrics
    )
    parent_model = deepcopy(model)
    parent_model.load_state_dict(parent_state)
    parent_model.eval()
    policy_comparison = _compare_policies(
        parent_model,
        model,
        validation_loader if validation_loader is not None else train_loader,
        device,
    )
    validation_slices = _validation_slice_metrics(
        parent_model=parent_model,
        child_model=model,
        rows=validation_rows,
        device=device,
        config=config,
    )
    replay_identity = _paths_identity(paths)
    save_checkpoint(
        output_checkpoint,
        model=model,
        generation=child_generation,
        metadata={
            "parent_checkpoint": str(parent_checkpoint),
            "parent_generation": parent.generation,
            "replay_paths": [str(path) for path in paths],
            "replay_identity": replay_identity,
            "training_config": asdict(config),
            "train_games": sorted(
                {f"{row.generation}:{row.game_seed}" for row in train_rows}
            ),
            "validation_games": sorted(
                {f"{row.generation}:{row.game_seed}" for row in validation_rows}
            ),
        },
    )
    result = {
        "parent_checkpoint": str(parent_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "parent_generation": parent.generation,
        "child_generation": child_generation,
        "replay_paths": [str(path) for path in paths],
        "replay_identity": replay_identity,
        "rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_games": len({(row.generation, row.game_seed) for row in train_rows}),
        "validation_games": len(
            {(row.generation, row.game_seed) for row in validation_rows}
        ),
        "best_validation_loss": best_validation,
        "selection_metric": "objective_loss",
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "parent_train_metrics": parent_train_metrics,
        "parent_validation_metrics": parent_validation_metrics,
        "child_train_metrics": child_train_metrics,
        "child_validation_metrics": child_validation_metrics,
        "validation_loss_delta_child_minus_parent": {
            key: child_validation_metrics[key] - parent_validation_metrics[key]
            for key in (
                "policy_loss",
                "value_loss",
                "total_loss",
                "objective_loss",
            )
        },
        "child_parent_policy_comparison": policy_comparison,
        "validation_slices": validation_slices,
        "history": history,
        "config": asdict(config),
    }
    output_metrics = Path(metrics_path)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    output_metrics.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def split_rows_by_game(
    rows: list[AlphaZeroReplayRow],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[AlphaZeroReplayRow], list[AlphaZeroReplayRow]]:
    games = sorted({(row.generation, row.game_seed) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(games)
    validation_count = (
        max(1, round(len(games) * validation_fraction))
        if validation_fraction > 0.0 and len(games) > 1
        else 0
    )
    validation_games = set(games[:validation_count])
    train = [
        row for row in rows if (row.generation, row.game_seed) not in validation_games
    ]
    validation = [
        row for row in rows if (row.generation, row.game_seed) in validation_games
    ]
    if not train:
        raise ValueError("Whole-game split produced an empty training set.")
    return train, validation


def collate_replay_rows(
    rows: list[AlphaZeroReplayRow],
    *,
    policy_divergence_weight: float = 0.0,
    include_corrective_fields: bool = False,
) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("Cannot collate an empty replay batch.")
    max_entities = max(len(row.entity_type_ids) for row in rows)
    max_actions = max(len(row.action_type_ids) for row in rows)
    batch_size = len(rows)
    batch = {
        "entity_type_ids": torch.zeros((batch_size, max_entities), dtype=torch.long),
        "entity_card_ids": torch.zeros((batch_size, max_entities), dtype=torch.long),
        "entity_noble_ids": torch.zeros((batch_size, max_entities), dtype=torch.long),
        "entity_numeric_features": torch.zeros(
            (batch_size, max_entities, ENTITY_NUMERIC_SIZE), dtype=torch.float32
        ),
        "entity_mask": torch.zeros((batch_size, max_entities), dtype=torch.bool),
        "action_type_ids": torch.zeros((batch_size, max_actions), dtype=torch.long),
        "action_card_ids": torch.zeros((batch_size, max_actions), dtype=torch.long),
        "action_noble_ids": torch.zeros((batch_size, max_actions), dtype=torch.long),
        "action_numeric_features": torch.zeros(
            (batch_size, max_actions, ACTION_NUMERIC_SIZE), dtype=torch.float32
        ),
        "action_mask": torch.zeros((batch_size, max_actions), dtype=torch.bool),
        "policy_target": torch.zeros((batch_size, max_actions), dtype=torch.float32),
        "value_target": torch.tensor(
            [float(row.value_target) for row in rows], dtype=torch.float32
        ),
    }
    if include_corrective_fields:
        batch.update(
            policy_row_weight=torch.ones(batch_size, dtype=torch.float32),
            visit_argmax_target=torch.tensor(
                [row.visit_argmax_row for row in rows], dtype=torch.long
            ),
            decisive_mask=torch.tensor(
                [row.visit_argmax_changed_raw_action for row in rows], dtype=torch.bool
            ),
        )
    for index, row in enumerate(rows):
        entity_count = len(row.entity_type_ids)
        action_count = len(row.action_type_ids)
        batch["entity_type_ids"][index, :entity_count] = torch.tensor(row.entity_type_ids)
        batch["entity_card_ids"][index, :entity_count] = torch.tensor(row.entity_card_ids)
        batch["entity_noble_ids"][index, :entity_count] = torch.tensor(row.entity_noble_ids)
        batch["entity_numeric_features"][index, :entity_count] = torch.tensor(
            row.entity_numeric_features,
            dtype=torch.float32,
        )
        batch["entity_mask"][index, :entity_count] = True
        batch["action_type_ids"][index, :action_count] = torch.tensor(row.action_type_ids)
        batch["action_card_ids"][index, :action_count] = torch.tensor(row.action_card_ids)
        batch["action_noble_ids"][index, :action_count] = torch.tensor(row.action_noble_ids)
        batch["action_numeric_features"][index, :action_count] = torch.tensor(
            row.action_numeric_features,
            dtype=torch.float32,
        )
        batch["action_mask"][index, :action_count] = True
        batch["policy_target"][index, :action_count] = torch.tensor(
            row.policy_target,
            dtype=torch.float32,
        )
        if include_corrective_fields and policy_divergence_weight > 0.0:
            target = torch.tensor(row.policy_target, dtype=torch.float32)
            prior = torch.tensor(row.root_network_priors, dtype=torch.float32)
            divergence = 0.5 * torch.abs(target - prior).sum()
            batch["policy_row_weight"][index] = 1.0 + policy_divergence_weight * divergence
    return batch


def _make_loader(
    rows: list[AlphaZeroReplayRow],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    bucket_batches: bool,
    config: TrainingConfig | None = None,
) -> DataLoader:
    dataset = _ReplayDataset(rows)
    actual_batch_size = min(batch_size, len(rows))
    collator = partial(
        collate_replay_rows,
        policy_divergence_weight=(config.policy_divergence_weight if config else 0.0),
        include_corrective_fields=True,
    )
    if not bucket_batches:
        return DataLoader(
            dataset,
            batch_size=actual_batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(seed),
            collate_fn=collator,
        )
    indices = list(range(len(rows)))
    indices.sort(
        key=lambda index: (
            len(rows[index].action_type_ids) // 4,
            len(rows[index].entity_type_ids) // 8,
            len(rows[index].action_type_ids),
            len(rows[index].entity_type_ids),
        )
    )
    batches = [
        indices[start : start + actual_batch_size]
        for start in range(0, len(indices), actual_batch_size)
    ]
    if shuffle:
        rng = random.Random(seed)
        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
    return DataLoader(
        dataset,
        batch_sampler=batches,
        collate_fn=collator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader | None,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingConfig,
) -> dict[str, float]:
    if loader is None:
        raise ValueError("Loader is required.")
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "total_loss": 0.0,
        "objective_loss": 0.0,
        "policy_objective_loss": 0.0,
        "decisive_aux_loss": 0.0,
        "policy_top_target_agreement": 0.0,
        "policy_target_top_probability": 0.0,
        "decisive_top_target_agreement": 0.0,
        "decisive_target_top_probability": 0.0,
        "decisive_rows": 0.0,
        "policy_entropy": 0.0,
        "value_mae": 0.0,
    }
    rows = 0
    for batch in loader:
        moved = {key: value.to(device) for key, value in batch.items()}
        logits, values = model(
            entity_type_ids=moved["entity_type_ids"],
            entity_card_ids=moved["entity_card_ids"],
            entity_noble_ids=moved["entity_noble_ids"],
            entity_numeric_features=moved["entity_numeric_features"],
            entity_mask=moved["entity_mask"],
            action_type_ids=moved["action_type_ids"],
            action_card_ids=moved["action_card_ids"],
            action_noble_ids=moved["action_noble_ids"],
            action_numeric_features=moved["action_numeric_features"],
            action_mask=moved["action_mask"],
        )
        losses = alpha_zero_loss(
            logits,
            values,
            moved["action_mask"],
            moved["policy_target"],
            moved["value_target"],
            policy_weight=config.policy_loss_weight,
            value_weight=config.value_loss_weight,
            policy_row_weight=moved["policy_row_weight"],
            visit_argmax_target=moved["visit_argmax_target"],
            decisive_mask=moved["decisive_mask"],
            visit_argmax_aux_weight=config.visit_argmax_aux_weight,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            losses["objective_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        count = int(moved["value_target"].shape[0])
        rows += count
        for key in ("policy_loss", "policy_objective_loss", "decisive_aux_loss", "value_loss", "total_loss", "objective_loss"):
            totals[key] += float(losses[key].item()) * count
        masked_logits = logits.masked_fill(
            ~moved["action_mask"],
            torch.finfo(logits.dtype).min,
        )
        probabilities = torch.softmax(masked_logits, dim=1)
        target_top = torch.argmax(moved["policy_target"], dim=1)
        predicted_top = torch.argmax(masked_logits, dim=1)
        totals["policy_top_target_agreement"] += float(
            (predicted_top == target_top).float().sum().item()
        )
        target_probability = probabilities.gather(1, target_top.unsqueeze(1)).squeeze(1)
        totals["policy_target_top_probability"] += float(target_probability.sum().item())
        decisive = moved["decisive_mask"]
        decisive_count = int(decisive.sum().item())
        totals["decisive_rows"] += decisive_count
        if decisive_count:
            visit_top = moved["visit_argmax_target"]
            totals["decisive_top_target_agreement"] += float(
                (predicted_top[decisive] == visit_top[decisive]).float().sum().item()
            )
            totals["decisive_target_top_probability"] += float(
                probabilities[decisive].gather(1, visit_top[decisive].unsqueeze(1)).sum().item()
            )
        totals["policy_entropy"] += float(
            (-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1))
            .sum()
            .item()
        )
        totals["value_mae"] += float(
            torch.abs(values - moved["value_target"]).sum().item()
        )
    decisive_rows = totals.pop("decisive_rows")
    result = {key: value / rows for key, value in totals.items()}
    result["decisive_rows"] = decisive_rows
    result["decisive_top_target_agreement"] = (
        result["decisive_top_target_agreement"] * rows / decisive_rows if decisive_rows else 0.0
    )
    result["decisive_target_top_probability"] = (
        result["decisive_target_top_probability"] * rows / decisive_rows if decisive_rows else 0.0
    )
    return result


def _compare_policies(
    parent: nn.Module,
    child: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict[str, float | int]:
    rows = 0
    top_changes = 0
    total_variation = 0.0
    child_to_parent_kl = 0.0
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            model_inputs = {key: moved[key] for key in (
                "entity_type_ids", "entity_card_ids", "entity_noble_ids",
                "entity_numeric_features", "entity_mask", "action_type_ids",
                "action_card_ids", "action_noble_ids", "action_numeric_features",
                "action_mask",
            )}
            parent_logits, _parent_values = parent(**model_inputs)
            child_logits, _child_values = child(**model_inputs)
            mask = moved["action_mask"]
            floor = torch.finfo(parent_logits.dtype).min
            parent_probability = torch.softmax(parent_logits.masked_fill(~mask, floor), dim=1)
            child_probability = torch.softmax(child_logits.masked_fill(~mask, floor), dim=1)
            count = int(mask.shape[0])
            rows += count
            top_changes += int(
                (
                    torch.argmax(parent_probability, dim=1)
                    != torch.argmax(child_probability, dim=1)
                )
                .sum()
                .item()
            )
            total_variation += float(
                (0.5 * torch.abs(child_probability - parent_probability).sum(dim=1))
                .sum()
                .item()
            )
            child_to_parent_kl += float(
                (
                    child_probability
                    * (
                        torch.log(child_probability.clamp_min(1e-12))
                        - torch.log(parent_probability.clamp_min(1e-12))
                    )
                )
                .sum(dim=1)
                .sum()
                .item()
            )
    return {
        "rows": rows,
        "top_action_changes": top_changes,
        "top_action_change_fraction": top_changes / rows,
        "mean_total_variation": total_variation / rows,
        "mean_child_to_parent_kl": child_to_parent_kl / rows,
    }


def _validation_slice_metrics(
    *,
    parent_model: nn.Module,
    child_model: nn.Module,
    rows: list[AlphaZeroReplayRow],
    device: str,
    config: TrainingConfig,
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {}
    slices: dict[str, list[AlphaZeroReplayRow]] = {
        "phase_early_lt20": [row for row in rows if row.turn_index < 20],
        "phase_middle_20_49": [row for row in rows if 20 <= row.turn_index < 50],
        "phase_late_ge50": [row for row in rows if row.turn_index >= 50],
        "decisive_search_change": [
            row for row in rows if row.visit_argmax_changed_raw_action
        ],
    }
    for generation in sorted({row.generation for row in rows}):
        slices[f"replay_generation_{generation}"] = [
            row for row in rows if row.generation == generation
        ]
    result: dict[str, dict[str, Any]] = {}
    for name, subset in slices.items():
        if not subset:
            continue
        loader = _make_loader(
            subset,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            bucket_batches=config.bucket_batches,
            config=config,
        )
        parent_metrics = _run_epoch(parent_model, loader, device, None, config)
        child_metrics = _run_epoch(child_model, loader, device, None, config)
        result[name] = {
            "rows": len(subset),
            "parent": parent_metrics,
            "child": child_metrics,
            "delta_child_minus_parent": {
                key: child_metrics[key] - parent_metrics[key]
                for key in (
                    "policy_loss",
                    "value_loss",
                    "total_loss",
                    "objective_loss",
                )
            },
        }
    return result


def _paths_identity(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
