"""Low-memory JSONL-indexed AlphaZero training with parent-safe selection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from .network import load_checkpoint, save_checkpoint
from .replay import AlphaZeroReplayRow, _row_from_payload
from .training import (
    TrainingConfig,
    _compare_policies,
    _run_epoch,
    collate_replay_rows,
)


@dataclass(frozen=True, slots=True)
class ReplayIndexEntry:
    path_index: int
    offset: int
    generation: int
    game_seed: int
    turn_index: int
    entity_count: int
    action_count: int


class IndexedReplayDataset(Dataset[AlphaZeroReplayRow]):
    def __init__(self, paths: list[Path], entries: list[ReplayIndexEntry]) -> None:
        self.paths = paths
        self.entries = entries
        self._handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> AlphaZeroReplayRow:
        entry = self.entries[index]
        handle = self._handles.get(entry.path_index)
        if handle is None or handle.closed:
            handle = self.paths[entry.path_index].open("rb")
            self._handles[entry.path_index] = handle
        handle.seek(entry.offset)
        line = handle.readline()
        if not line:
            raise ValueError("Indexed replay offset points beyond end of file.")
        row = _row_from_payload(json.loads(line))
        if (
            row.generation != entry.generation
            or row.game_seed != entry.game_seed
            or row.turn_index != entry.turn_index
            or len(row.entity_type_ids) != entry.entity_count
            or len(row.action_type_ids) != entry.action_count
        ):
            raise ValueError("Indexed replay row identity changed after indexing.")
        return row

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        if hasattr(self, "_handles"):
            self.close()


def index_replay_paths(paths: Iterable[str | Path]) -> tuple[list[Path], list[ReplayIndexEntry]]:
    replay_paths = [Path(path) for path in paths]
    entries: list[ReplayIndexEntry] = []
    for path_index, path in enumerate(replay_paths):
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                row = _row_from_payload(json.loads(line))
                if row.value_target is None:
                    raise ValueError("Every indexed training row needs a terminal target.")
                entries.append(
                    ReplayIndexEntry(
                        path_index=path_index,
                        offset=offset,
                        generation=row.generation,
                        game_seed=row.game_seed,
                        turn_index=row.turn_index,
                        entity_count=len(row.entity_type_ids),
                        action_count=len(row.action_type_ids),
                    )
                )
    if not entries:
        raise ValueError("No indexed replay rows were supplied.")
    return replay_paths, entries


def split_index_by_game(
    entries: list[ReplayIndexEntry],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[ReplayIndexEntry], list[ReplayIndexEntry]]:
    games = sorted({(entry.generation, entry.game_seed) for entry in entries})
    rng = random.Random(seed)
    rng.shuffle(games)
    validation_count = (
        max(1, round(len(games) * validation_fraction))
        if validation_fraction > 0.0 and len(games) > 1
        else 0
    )
    validation_games = set(games[:validation_count])
    train = [
        entry
        for entry in entries
        if (entry.generation, entry.game_seed) not in validation_games
    ]
    validation = [
        entry
        for entry in entries
        if (entry.generation, entry.game_seed) in validation_games
    ]
    if not train:
        raise ValueError("Indexed whole-game split produced an empty training set.")
    return train, validation


def make_indexed_loader(
    paths: list[Path],
    entries: list[ReplayIndexEntry],
    *,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    dataset = IndexedReplayDataset(paths, entries)
    actual_batch_size = min(config.batch_size, len(entries))
    indices = list(range(len(entries)))
    if config.bucket_batches:
        indices.sort(
            key=lambda index: (
                entries[index].action_count // 4,
                entries[index].entity_count // 8,
                entries[index].action_count,
                entries[index].entity_count,
            )
        )
    batches = [
        indices[start : start + actual_batch_size]
        for start in range(0, len(indices), actual_batch_size)
    ]
    if shuffle:
        rng = random.Random(config.seed)
        for batch in batches:
            rng.shuffle(batch)
        rng.shuffle(batches)
    collator = partial(
        collate_replay_rows,
        policy_divergence_weight=config.policy_divergence_weight,
        include_corrective_fields=True,
    )
    return DataLoader(dataset, batch_sampler=batches, collate_fn=collator)


def train_parent_safe_indexed_successor(
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
    paths, entries = index_replay_paths(replay_paths)
    train_entries, validation_entries = split_index_by_game(
        entries,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    model, parent = load_checkpoint(parent_checkpoint, device=device)
    parent_state = deepcopy(model.state_dict())
    child_generation = parent.generation + 1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_indexed_loader(paths, train_entries, config=config, shuffle=True)
    validation_loader = (
        make_indexed_loader(paths, validation_entries, config=config, shuffle=False)
        if validation_entries
        else None
    )
    model.eval()
    parent_train_metrics = _run_epoch(model, train_loader, device, None, config)
    parent_validation_metrics = (
        _run_epoch(model, validation_loader, device, None, config)
        if validation_loader is not None
        else parent_train_metrics
    )
    best_state = deepcopy(parent_state)
    best_validation = float(parent_validation_metrics["objective_loss"])
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
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
        if validation_metrics["objective_loss"] < best_validation - config.early_stopping_min_delta:
            best_validation = float(validation_metrics["objective_loss"])
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
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
    replay_identity = _streaming_paths_identity(paths)
    train_games = sorted({f"{entry.generation}:{entry.game_seed}" for entry in train_entries})
    validation_games = sorted(
        {f"{entry.generation}:{entry.game_seed}" for entry in validation_entries}
    )
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
            "parent_safe_selection": True,
            "selected_epoch": best_epoch,
            "train_games": train_games,
            "validation_games": validation_games,
        },
    )
    result = {
        "parent_checkpoint": str(parent_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "parent_generation": parent.generation,
        "child_generation": child_generation,
        "replay_paths": [str(path) for path in paths],
        "replay_identity": replay_identity,
        "rows": len(entries),
        "train_rows": len(train_entries),
        "validation_rows": len(validation_entries),
        "train_games": len(train_games),
        "validation_games": len(validation_games),
        "best_validation_loss": best_validation,
        "selection_metric": "objective_loss_with_epoch_zero_parent_eligible",
        "selected_epoch": best_epoch,
        "selected_parent_baseline": best_epoch == 0,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "parent_train_metrics": parent_train_metrics,
        "parent_validation_metrics": parent_validation_metrics,
        "child_train_metrics": child_train_metrics,
        "child_validation_metrics": child_validation_metrics,
        "validation_loss_delta_child_minus_parent": {
            key: child_validation_metrics[key] - parent_validation_metrics[key]
            for key in ("policy_loss", "value_loss", "total_loss", "objective_loss")
        },
        "child_parent_policy_comparison": policy_comparison,
        "validation_slices": {},
        "history": history,
        "config": asdict(config),
        "indexed_loading": True,
        "parent_safe_selection": True,
    }
    output_metrics = Path(metrics_path)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_metrics.with_suffix(output_metrics.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_metrics)
    for loader in (train_loader, validation_loader):
        if loader is not None and isinstance(loader.dataset, IndexedReplayDataset):
            loader.dataset.close()
    return result


def _streaming_paths_identity(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).replace("\\", "/").encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
