"""Replay dataset loading for supervised warm-start training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from splendor_ai.encoding import ActionCodec


@dataclass(frozen=True, slots=True)
class ReplaySample:
    observation: torch.Tensor
    action_index: int
    legal_action_indices: tuple[int, ...]
    value_target: float


class SupervisedReplayDataset(Dataset[ReplaySample]):
    """Loads JSONL replay steps exported by `export_replay_games_jsonl`."""

    def __init__(
        self,
        path: str | Path,
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
    ) -> None:
        self._path = Path(path)
        self._include_stalled_games = include_stalled_games
        self._include_timed_out_games = include_timed_out_games
        self._samples = self._load_samples(
            self._path,
            include_stalled_games=self._include_stalled_games,
            include_timed_out_games=self._include_timed_out_games,
        )

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

    @staticmethod
    def _load_samples(
        path: Path,
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
    ) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
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
                samples.append(payload)
        return samples


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
