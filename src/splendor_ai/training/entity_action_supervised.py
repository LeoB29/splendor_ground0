"""Training utilities for the canonical entity/action scorer."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from splendor_ai.encoding import (
    ActionCodec,
    actions_by_canonical_successor_key,
    canonical_successor_action_key,
    collate_entity_action_batch,
)
from splendor_ai.engine.actions import Action
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.engine.state import SplendorState
from splendor_ai.training.model import (
    CanonicalEntityActionScorerModel,
    masked_legal_action_scores,
)
from splendor_ai.training.replay import deserialize_state_snapshot


@dataclass(frozen=True, slots=True)
class EntityActionReplaySample:
    state: SplendorState
    legal_actions: tuple[Action, ...]
    target_action_row: int
    value_target: float
    game_seed: int


@dataclass(frozen=True, slots=True)
class EntityTrainMetrics:
    policy_loss: float
    value_loss: float
    total_loss: float
    policy_accuracy: float
    uniform_policy_loss: float
    samples: int


@dataclass(frozen=True, slots=True)
class EntityFitArtifacts:
    model: CanonicalEntityActionScorerModel
    train_metrics: tuple[EntityTrainMetrics, ...]
    validation_metrics: tuple[EntityTrainMetrics, ...]
    best_validation_state_dict: dict[str, torch.Tensor] | None
    best_epoch_index: int | None


class EntityActionReplayDataset(Dataset[EntityActionReplaySample]):
    """Snapshot replay dataset collapsed to canonical successor actions."""

    def __init__(self, replay_paths: str | Path | Sequence[str | Path]) -> None:
        paths = (Path(replay_paths),) if isinstance(replay_paths, (str, Path)) else tuple(Path(path) for path in replay_paths)
        if not paths:
            raise ValueError("At least one replay path is required.")
        self._rows: list[dict[str, object]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self._rows.append(json.loads(line))
        self._env = SplendorEnv()
        self._codec = ActionCodec()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> EntityActionReplaySample:
        row = self._rows[index]
        snapshot = row.get("state_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("EntityActionReplayDataset requires state_snapshot rows.")
        state = deserialize_state_snapshot(snapshot)
        raw_actions = self._env.legal_actions(state)
        target_index = int(row["action_index"])
        try:
            raw_target = next(
                action
                for action in raw_actions
                if self._codec.encode(state, action) == target_index
            )
        except StopIteration as exc:
            raise ValueError("Replay target is not legal in its reconstructed state.") from exc
        representatives = actions_by_canonical_successor_key(state, raw_actions)
        target_key = canonical_successor_action_key(state, raw_target)
        legal_actions = tuple(representatives.values())
        target_action_row = list(representatives).index(target_key)
        return EntityActionReplaySample(
            state=state,
            legal_actions=legal_actions,
            target_action_row=target_action_row,
            value_target=float(row.get("final_value", 0.0)),
            game_seed=int(row.get("seed", 0)),
        )

    def game_seed_for_index(self, index: int) -> int:
        return int(self._rows[index].get("seed", 0))


def collate_entity_action_replay_samples(
    samples: list[EntityActionReplaySample],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty entity replay batch.")
    batch = collate_entity_action_batch(
        [(sample.state, list(sample.legal_actions)) for sample in samples]
    )
    batch["target_action_row"] = torch.tensor(
        [sample.target_action_row for sample in samples], dtype=torch.long
    )
    batch["value_target"] = torch.tensor(
        [sample.value_target for sample in samples], dtype=torch.float32
    )
    batch["game_seed"] = torch.tensor(
        [sample.game_seed for sample in samples], dtype=torch.long
    )
    return batch


def split_entity_action_replay_dataset(
    dataset: EntityActionReplayDataset,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[Subset[EntityActionReplaySample], Subset[EntityActionReplaySample] | None]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if validation_fraction == 0.0 or len(dataset) < 2:
        return Subset(dataset, list(range(len(dataset)))), None
    games = sorted({dataset.game_seed_for_index(index) for index in range(len(dataset))})
    random.Random(seed).shuffle(games)
    validation_games = set(games[: max(1, round(len(games) * validation_fraction))])
    train_indices = [index for index in range(len(dataset)) if dataset.game_seed_for_index(index) not in validation_games]
    validation_indices = [index for index in range(len(dataset)) if dataset.game_seed_for_index(index) in validation_games]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def create_entity_action_replay_dataloader(
    dataset: Dataset[EntityActionReplaySample],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_entity_action_replay_samples)


def run_entity_action_epoch(
    model: CanonicalEntityActionScorerModel,
    dataloader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    value_loss_weight: float = 0.25,
) -> EntityTrainMetrics:
    training = optimizer is not None
    model.train(training)
    totals = [0.0, 0.0, 0.0, 0.0]
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in dataloader:
            batch = {name: value.to(device) for name, value in batch.items()}
            action_scores, value_pred = model(
                entity_type_ids=batch["entity_type_ids"],
                entity_card_ids=batch["entity_card_ids"],
                entity_noble_ids=batch["entity_noble_ids"],
                entity_numeric_features=batch["entity_numeric_features"],
                entity_mask=batch["entity_mask"],
                action_type_ids=batch["action_type_ids"],
                action_card_ids=batch["action_card_ids"],
                action_noble_ids=batch["action_noble_ids"],
                action_numeric_features=batch["action_numeric_features"],
                action_mask=batch["action_mask"],
            )
            policy_loss = torch.nn.functional.cross_entropy(
                masked_legal_action_scores(action_scores, batch["action_mask"]),
                batch["target_action_row"],
            )
            value_loss = torch.nn.functional.mse_loss(value_pred, batch["value_target"])
            loss = policy_loss + value_loss_weight * value_loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            predictions = action_scores.masked_fill(~batch["action_mask"], -torch.inf).argmax(dim=1)
            uniform_loss = torch.log(batch["action_mask"].sum(dim=1).to(torch.float32)).mean()
            batch_size = int(batch["target_action_row"].numel())
            totals[0] += float(policy_loss.detach().cpu()) * batch_size
            totals[1] += float(value_loss.detach().cpu()) * batch_size
            totals[2] += float(loss.detach().cpu()) * batch_size
            totals[3] += float(uniform_loss.detach().cpu()) * batch_size
            total_correct += int((predictions == batch["target_action_row"]).sum().item())
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("Entity replay dataloader produced zero samples.")
    return EntityTrainMetrics(
        policy_loss=totals[0] / total_samples,
        value_loss=totals[1] / total_samples,
        total_loss=totals[2] / total_samples,
        policy_accuracy=total_correct / total_samples,
        uniform_policy_loss=totals[3] / total_samples,
        samples=total_samples,
    )


def fit_entity_action_scorer(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader | None,
    model: CanonicalEntityActionScorerModel,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: str = "cpu",
    value_loss_weight: float = 0.25,
    selection_metric: str = "total_loss",
) -> EntityFitArtifacts:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if selection_metric not in {"policy_loss", "total_loss"}:
        raise ValueError("selection_metric must be 'policy_loss' or 'total_loss'.")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_history: list[EntityTrainMetrics] = []
    validation_history: list[EntityTrainMetrics] = []
    best_state = None
    best_epoch = None
    best_loss = None
    for epoch_index in range(1, epochs + 1):
        train_history.append(run_entity_action_epoch(model, train_dataloader, device, optimizer, value_loss_weight))
        if validation_dataloader is not None:
            validation = run_entity_action_epoch(model, validation_dataloader, device, None, value_loss_weight)
            validation_history.append(validation)
            selection_loss = float(getattr(validation, selection_metric))
            if best_loss is None or selection_loss < best_loss:
                best_loss = selection_loss
                best_epoch = epoch_index
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return EntityFitArtifacts(model, tuple(train_history), tuple(validation_history), best_state, best_epoch)
