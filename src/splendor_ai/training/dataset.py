"""Replay dataset loading for supervised warm-start training."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from splendor_ai.encoding import ActionCodec


@dataclass(frozen=True, slots=True)
class ReplaySample:
    observation: torch.Tensor
    action_index: int
    legal_action_indices: tuple[int, ...]
    value_target: float


@dataclass(frozen=True, slots=True)
class ReplaySourceSummary:
    path: str
    samples: int


class SupervisedReplayDataset(Dataset[ReplaySample]):
    """Loads JSONL replay steps exported by `export_replay_games_jsonl`."""

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
    ) -> None:
        self._paths = _normalize_paths(path)
        self._include_stalled_games = include_stalled_games
        self._include_timed_out_games = include_timed_out_games
        self._samples = self._load_samples(
            self._paths,
            include_stalled_games=self._include_stalled_games,
            include_timed_out_games=self._include_timed_out_games,
        )

    @property
    def paths(self) -> tuple[Path, ...]:
        return self._paths

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ReplaySample:
        payload = self._samples[index]
        return ReplaySample(
            observation=torch.tensor(payload["observation_vector"], dtype=torch.float32),
            action_index=int(payload["action_index"]),
            legal_action_indices=tuple(int(idx) for idx in payload["legal_action_indices"]),
            value_target=float(payload["final_value"]),
        )

    def source_path_for_index(self, index: int) -> Path:
        return Path(str(self._samples[index]["_source_path"]))

    def source_counts(self, indices: Sequence[int] | None = None) -> tuple[ReplaySourceSummary, ...]:
        sample_indices = range(len(self._samples)) if indices is None else indices
        counts = Counter(str(self._samples[int(index)]["_source_path"]) for index in sample_indices)
        return tuple(
            ReplaySourceSummary(path=path, samples=counts[path])
            for path in sorted(counts)
        )

    @staticmethod
    def _load_samples(
        paths: tuple[Path, ...],
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
    ) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if not include_stalled_games and payload.get("game_stalled", False):
                        continue
                    if not include_timed_out_games and payload.get("game_timed_out", False):
                        continue
                    payload["_source_path"] = str(path)
                    samples.append(payload)
        return samples


def summarize_replay_dataset_sources(
    dataset: Dataset[ReplaySample],
) -> tuple[ReplaySourceSummary, ...]:
    if isinstance(dataset, SupervisedReplayDataset):
        return dataset.source_counts()
    if isinstance(dataset, Subset):
        if isinstance(dataset.dataset, SupervisedReplayDataset):
            return dataset.dataset.source_counts(dataset.indices)
        raise TypeError("Nested dataset subsets are not supported for source summarization.")
    raise TypeError(f"Unsupported dataset type for source summarization: {type(dataset)!r}")


def _normalize_paths(path: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
    if isinstance(path, (str, Path)):
        paths = (Path(path),)
    else:
        normalized = tuple(Path(item) for item in path)
        if not normalized:
            raise ValueError("At least one replay path is required.")
        paths = normalized
    return paths


def collate_replay_samples(
    samples: list[ReplaySample],
    action_space_size: int | None = None,
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    action_space = action_space_size or ActionCodec().action_space_size
    observations = torch.stack([sample.observation for sample in samples], dim=0)
    action_indices = torch.tensor([sample.action_index for sample in samples], dtype=torch.long)
    value_targets = torch.tensor([sample.value_target for sample in samples], dtype=torch.float32)

    legal_action_mask = torch.zeros((len(samples), action_space), dtype=torch.bool)
    for row_index, sample in enumerate(samples):
        legal_action_mask[row_index, list(sample.legal_action_indices)] = True

    return {
        "observation": observations,
        "action_index": action_indices,
        "legal_action_mask": legal_action_mask,
        "value_target": value_targets,
    }
