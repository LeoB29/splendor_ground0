"""Replay dataset loading for supervised warm-start training."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from splendor_ai.encoding import (
    ActionCodec,
    encode_legal_action_features,
    encode_public_observation_tensor,
    fit_feature_vector,
)
from splendor_ai.engine.env import SplendorEnv
from splendor_ai.training.replay import deserialize_state_snapshot


@dataclass(frozen=True, slots=True)
class ReplaySample:
    observation: torch.Tensor
    action_index: int
    legal_action_indices: tuple[int, ...]
    value_target: float


@dataclass(frozen=True, slots=True)
class LegalActionReplaySample:
    observation: torch.Tensor
    legal_action_features: torch.Tensor
    target_action_row: int
    preference_competitor_row: int
    preference_mask: bool
    action_index: int
    legal_action_indices: tuple[int, ...]
    value_target: float
    search_value_target: torch.Tensor
    search_value_mask: torch.Tensor
    root_visit_target: torch.Tensor
    root_visit_mask: bool
    sample_weight: float
    policy_sample_weight: float
    distill_sample_weight: float
    preference_sample_weight: float
    trajectory_advantage: float
    behavior_log_probability: float
    trajectory_mask: bool


@dataclass(frozen=True, slots=True)
class ValueReplaySample:
    observation: torch.Tensor
    value_target: float
    outcome_target: float
    score_margin_target: float
    turn_index: int


@dataclass(frozen=True, slots=True)
class ReplaySourceSummary:
    path: str
    samples: int


@dataclass(frozen=True, slots=True)
class ReplayStepMetadata:
    source_path: Path
    game_seed: int
    turn_index: int


class SupervisedReplayDataset(Dataset[ReplaySample]):
    """Loads JSONL replay steps exported by `export_replay_games_jsonl`."""

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
        max_model_loop_fallback_triggers: int | None = None,
    ) -> None:
        self._paths = _normalize_paths(path)
        self._include_stalled_games = include_stalled_games
        self._include_timed_out_games = include_timed_out_games
        self._max_model_loop_fallback_triggers = max_model_loop_fallback_triggers
        self._samples = self._load_samples(
            self._paths,
            include_stalled_games=self._include_stalled_games,
            include_timed_out_games=self._include_timed_out_games,
            max_model_loop_fallback_triggers=self._max_model_loop_fallback_triggers,
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
        max_model_loop_fallback_triggers: int | None = None,
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
                    if (
                        max_model_loop_fallback_triggers is not None
                        and int(payload.get("game_model_loop_fallback_triggers", 0))
                        > max_model_loop_fallback_triggers
                    ):
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


class LegalActionReplayDataset(Dataset[LegalActionReplaySample]):
    """Replay dataset for variable-length legal-action scorer training.

    This dataset requires replay rows with `state_snapshot`, because it
    reconstructs the pre-action state, regenerates the legal actions, encodes
    each legal action as structured features, and maps the replay target to the
    row in that legal-action list.
    """

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
        max_model_loop_fallback_triggers: int | None = None,
        codec: ActionCodec | None = None,
        policy_target_filter: str = "all",
        winner_action_weight: float = 1.0,
        loser_action_weight: float = 1.0,
        draw_action_weight: float = 1.0,
        preference_rows_policy_weight: float = 1.0,
        preference_rows_distill_weight: float = 1.0,
        preference_rows_preference_weight: float = 1.0,
        observation_size: int | None = None,
        action_feature_size: int | None = None,
    ) -> None:
        if policy_target_filter not in {"all", "winner"}:
            raise ValueError("policy_target_filter must be 'all' or 'winner'.")
        for name, weight in (
            ("winner_action_weight", winner_action_weight),
            ("loser_action_weight", loser_action_weight),
            ("draw_action_weight", draw_action_weight),
        ):
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"{name} must be a positive finite value.")
        for name, weight in (
            ("preference_rows_policy_weight", preference_rows_policy_weight),
            ("preference_rows_distill_weight", preference_rows_distill_weight),
            ("preference_rows_preference_weight", preference_rows_preference_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} must be a finite non-negative value.")
        self._paths = _normalize_paths(path)
        self._include_stalled_games = include_stalled_games
        self._include_timed_out_games = include_timed_out_games
        self._max_model_loop_fallback_triggers = max_model_loop_fallback_triggers
        self._codec = codec or ActionCodec()
        self._env = SplendorEnv()
        self._policy_target_filter = policy_target_filter
        self._winner_action_weight = winner_action_weight
        self._loser_action_weight = loser_action_weight
        self._draw_action_weight = draw_action_weight
        self._preference_rows_policy_weight = preference_rows_policy_weight
        self._preference_rows_distill_weight = preference_rows_distill_weight
        self._preference_rows_preference_weight = preference_rows_preference_weight
        self._observation_size = observation_size
        self._action_feature_size = action_feature_size
        self._samples = SupervisedReplayDataset._load_samples(
            self._paths,
            include_stalled_games=self._include_stalled_games,
            include_timed_out_games=self._include_timed_out_games,
            max_model_loop_fallback_triggers=self._max_model_loop_fallback_triggers,
        )
        if self._policy_target_filter == "winner":
            self._samples = [
                payload
                for payload in self._samples
                if float(payload.get("final_value", 0.0)) > 0.0
            ]

    @property
    def paths(self) -> tuple[Path, ...]:
        return self._paths

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> LegalActionReplaySample:
        payload = self._samples[index]
        snapshot = payload.get("state_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError(
                "LegalActionReplayDataset requires replay rows with state_snapshot."
            )

        state = deserialize_state_snapshot(snapshot)
        legal_actions = self._env.legal_actions(state)
        legal_action_indices = tuple(self._codec.legal_action_indices(state, legal_actions))
        action_index = int(payload["action_index"])
        try:
            target_action_row = legal_action_indices.index(action_index)
        except ValueError as exc:
            raise ValueError(
                "Replay target action_index is not legal in the reconstructed state: "
                f"{action_index}"
            ) from exc
        preference_competitor_row = 0
        preference_mask = False
        model_top_action_index = payload.get("model_top_action_index")
        if model_top_action_index is not None:
            try:
                competitor_action_row = legal_action_indices.index(int(model_top_action_index))
            except (TypeError, ValueError):
                competitor_action_row = None
            if competitor_action_row is not None and competitor_action_row != target_action_row:
                preference_competitor_row = competitor_action_row
                preference_mask = True

        player_id = int(payload.get("player_id", state.current_player))
        observation = encode_public_observation_tensor(state, player_id)
        observation_vector = fit_feature_vector(
            observation.vector,
            self._observation_size or len(observation.vector),
        )
        action_feature_rows = [
            torch.tensor(
                fit_feature_vector(
                    features.vector,
                    self._action_feature_size or len(features.vector),
                ),
                dtype=torch.float32,
            )
            for features in encode_legal_action_features(
                state=state,
                legal_actions=legal_actions,
                player_id=player_id,
            )
        ]
        if not action_feature_rows:
            raise ValueError("Reconstructed state has no legal actions.")

        search_value_target = torch.zeros(len(legal_action_indices), dtype=torch.float32)
        search_value_mask = torch.zeros(len(legal_action_indices), dtype=torch.bool)
        for search_score in payload.get("search_action_scores", []):
            if not isinstance(search_score, dict):
                continue
            try:
                search_action_index = int(search_score["action_index"])
                search_action_row = legal_action_indices.index(search_action_index)
                search_value = float(search_score["value"])
            except (KeyError, TypeError, ValueError):
                continue
            search_value_target[search_action_row] = search_value
            search_value_mask[search_action_row] = True
        root_visit_target = torch.zeros(
            len(legal_action_indices),
            dtype=torch.float32,
        )
        for root_visit in payload.get("root_visit_distribution", []):
            if not isinstance(root_visit, dict):
                continue
            try:
                root_action_index = int(root_visit["action_index"])
                root_action_row = legal_action_indices.index(root_action_index)
                root_probability = float(
                    root_visit.get("probability", root_visit.get("visits", 0.0))
                )
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(root_probability) and root_probability > 0.0:
                root_visit_target[root_action_row] += root_probability
        root_visit_total = float(root_visit_target.sum().item())
        root_visit_mask = root_visit_total > 0.0
        if root_visit_mask:
            root_visit_target /= root_visit_total
        sample_weight = float(payload.get("training_weight", 1.0)) * self._outcome_sample_weight(payload)
        if not math.isfinite(sample_weight) or sample_weight <= 0.0:
            raise ValueError(f"Replay row has invalid training_weight: {sample_weight}")
        policy_sample_weight = sample_weight
        distill_sample_weight = sample_weight
        preference_sample_weight = 0.0
        if preference_mask:
            policy_sample_weight = sample_weight * self._preference_rows_policy_weight
            distill_sample_weight = sample_weight * self._preference_rows_distill_weight
            preference_sample_weight = sample_weight * self._preference_rows_preference_weight
        trajectory_advantage_raw = payload.get("trajectory_advantage")
        behavior_log_probability_raw = payload.get("behavior_log_probability")
        trajectory_mask = (
            trajectory_advantage_raw is not None
            and behavior_log_probability_raw is not None
        )
        trajectory_advantage = (
            float(trajectory_advantage_raw) if trajectory_mask else 0.0
        )
        behavior_log_probability = (
            float(behavior_log_probability_raw) if trajectory_mask else 0.0
        )
        if trajectory_mask and (
            not math.isfinite(trajectory_advantage)
            or not math.isfinite(behavior_log_probability)
        ):
            raise ValueError("Trajectory advantage and behavior log probability must be finite.")

        return LegalActionReplaySample(
            observation=torch.tensor(observation_vector, dtype=torch.float32),
            legal_action_features=torch.stack(action_feature_rows, dim=0),
            target_action_row=target_action_row,
            preference_competitor_row=preference_competitor_row,
            preference_mask=preference_mask,
            action_index=action_index,
            legal_action_indices=legal_action_indices,
            value_target=float(payload["final_value"]),
            search_value_target=search_value_target,
            search_value_mask=search_value_mask,
            root_visit_target=root_visit_target,
            root_visit_mask=root_visit_mask,
            sample_weight=sample_weight,
            policy_sample_weight=policy_sample_weight,
            distill_sample_weight=distill_sample_weight,
            preference_sample_weight=preference_sample_weight,
            trajectory_advantage=trajectory_advantage,
            behavior_log_probability=behavior_log_probability,
            trajectory_mask=trajectory_mask,
        )

    def source_path_for_index(self, index: int) -> Path:
        return Path(str(self._samples[index]["_source_path"]))

    def replay_step_metadata_for_index(self, index: int) -> ReplayStepMetadata:
        payload = self._samples[index]
        return ReplayStepMetadata(
            source_path=Path(str(payload["_source_path"])),
            game_seed=int(payload["game_seed"]),
            turn_index=int(payload["turn_index"]),
        )

    def state_for_index(self, index: int):
        snapshot = self._samples[index].get("state_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("LegalActionReplayDataset requires replay rows with state_snapshot.")
        return deserialize_state_snapshot(snapshot)

    def validation_group_key_for_index(self, index: int, split_unit: str) -> tuple[object, ...]:
        payload = self._samples[index]
        source_path = str(payload["_source_path"])
        if split_unit == "source":
            return (source_path,)
        if split_unit == "game":
            return (source_path, int(payload["game_seed"]))
        raise ValueError("split_unit must be 'game' or 'source'.")

    def _outcome_sample_weight(self, payload: dict[str, object]) -> float:
        final_value = float(payload.get("final_value", 0.0))
        if final_value > 0.0:
            return self._winner_action_weight
        if final_value < 0.0:
            return self._loser_action_weight
        return self._draw_action_weight


class ValueReplayDataset(Dataset[ValueReplaySample]):
    """Loads replay rows for state-value prediction."""

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        include_stalled_games: bool = True,
        include_timed_out_games: bool = True,
        max_model_loop_fallback_triggers: int | None = None,
        margin_scale: float = 15.0,
        min_turn_index: int | None = None,
        max_turn_index: int | None = None,
    ) -> None:
        if margin_scale <= 0.0:
            raise ValueError("margin_scale must be positive.")
        if min_turn_index is not None and min_turn_index < 0:
            raise ValueError("min_turn_index must be non-negative.")
        if max_turn_index is not None and max_turn_index < 0:
            raise ValueError("max_turn_index must be non-negative.")
        if (
            min_turn_index is not None
            and max_turn_index is not None
            and min_turn_index > max_turn_index
        ):
            raise ValueError("min_turn_index cannot exceed max_turn_index.")
        self._paths = _normalize_paths(path)
        self._margin_scale = margin_scale
        self._samples = SupervisedReplayDataset._load_samples(
            self._paths,
            include_stalled_games=include_stalled_games,
            include_timed_out_games=include_timed_out_games,
            max_model_loop_fallback_triggers=max_model_loop_fallback_triggers,
        )
        if min_turn_index is not None:
            self._samples = [
                payload
                for payload in self._samples
                if int(payload.get("turn_index", 0)) >= min_turn_index
            ]
        if max_turn_index is not None:
            self._samples = [
                payload
                for payload in self._samples
                if int(payload.get("turn_index", 0)) <= max_turn_index
            ]

    @property
    def paths(self) -> tuple[Path, ...]:
        return self._paths

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ValueReplaySample:
        payload = self._samples[index]
        player_id = int(payload["player_id"])
        scores = payload.get("game_final_scores")
        if not isinstance(scores, list | tuple) or len(scores) != 2:
            raise ValueError("ValueReplayDataset requires game_final_scores with two entries.")
        score_margin = float(scores[player_id]) - float(scores[1 - player_id])
        value_target = max(min(score_margin / self._margin_scale, 1.0), -1.0)
        return ValueReplaySample(
            observation=torch.tensor(payload["observation_vector"], dtype=torch.float32),
            value_target=value_target,
            outcome_target=float(payload.get("final_value", 0.0)),
            score_margin_target=score_margin,
            turn_index=int(payload.get("turn_index", 0)),
        )

    def source_path_for_index(self, index: int) -> Path:
        return Path(str(self._samples[index]["_source_path"]))

    def validation_group_key_for_index(self, index: int, split_unit: str) -> tuple[object, ...]:
        payload = self._samples[index]
        source_path = str(payload["_source_path"])
        if split_unit == "source":
            return (source_path,)
        if split_unit == "game":
            return (source_path, int(payload["game_seed"]))
        raise ValueError("split_unit must be 'game' or 'source'.")


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


def collate_value_replay_samples(
    samples: list[ValueReplaySample],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    return {
        "observation": torch.stack([sample.observation for sample in samples], dim=0),
        "value_target": torch.tensor([sample.value_target for sample in samples], dtype=torch.float32),
        "outcome_target": torch.tensor([sample.outcome_target for sample in samples], dtype=torch.float32),
        "score_margin_target": torch.tensor(
            [sample.score_margin_target for sample in samples],
            dtype=torch.float32,
        ),
        "turn_index": torch.tensor([sample.turn_index for sample in samples], dtype=torch.long),
    }


def collate_legal_action_replay_samples(
    samples: list[LegalActionReplaySample],
) -> dict[str, torch.Tensor]:
    """Pad variable-length legal-action feature rows into one batch."""

    if not samples:
        raise ValueError("Cannot collate an empty batch.")

    observations = torch.stack([sample.observation for sample in samples], dim=0)
    target_action_rows = torch.tensor(
        [sample.target_action_row for sample in samples],
        dtype=torch.long,
    )
    action_indices = torch.tensor(
        [sample.action_index for sample in samples],
        dtype=torch.long,
    )
    preference_competitor_rows = torch.tensor(
        [sample.preference_competitor_row for sample in samples],
        dtype=torch.long,
    )
    preference_masks = torch.tensor(
        [sample.preference_mask for sample in samples],
        dtype=torch.bool,
    )
    value_targets = torch.tensor(
        [sample.value_target for sample in samples],
        dtype=torch.float32,
    )
    sample_weights = torch.tensor(
        [sample.sample_weight for sample in samples],
        dtype=torch.float32,
    )
    policy_sample_weights = torch.tensor(
        [sample.policy_sample_weight for sample in samples],
        dtype=torch.float32,
    )
    distill_sample_weights = torch.tensor(
        [sample.distill_sample_weight for sample in samples],
        dtype=torch.float32,
    )
    preference_sample_weights = torch.tensor(
        [sample.preference_sample_weight for sample in samples],
        dtype=torch.float32,
    )
    trajectory_advantages = torch.tensor(
        [sample.trajectory_advantage for sample in samples],
        dtype=torch.float32,
    )
    behavior_log_probabilities = torch.tensor(
        [sample.behavior_log_probability for sample in samples],
        dtype=torch.float32,
    )
    trajectory_masks = torch.tensor(
        [sample.trajectory_mask for sample in samples],
        dtype=torch.bool,
    )

    max_legal_actions = max(
        sample.legal_action_features.shape[0]
        for sample in samples
    )
    action_feature_size = samples[0].legal_action_features.shape[1]
    action_features = torch.zeros(
        (len(samples), max_legal_actions, action_feature_size),
        dtype=torch.float32,
    )
    legal_action_mask = torch.zeros(
        (len(samples), max_legal_actions),
        dtype=torch.bool,
    )
    search_value_targets = torch.zeros(
        (len(samples), max_legal_actions),
        dtype=torch.float32,
    )
    search_value_mask = torch.zeros(
        (len(samples), max_legal_actions),
        dtype=torch.bool,
    )
    root_visit_targets = torch.zeros(
        (len(samples), max_legal_actions),
        dtype=torch.float32,
    )
    root_visit_masks = torch.tensor(
        [sample.root_visit_mask for sample in samples],
        dtype=torch.bool,
    )

    for row_index, sample in enumerate(samples):
        legal_count, sample_feature_size = sample.legal_action_features.shape
        if sample_feature_size != action_feature_size:
            raise ValueError("All action feature rows in a batch must have the same width.")
        action_features[row_index, :legal_count] = sample.legal_action_features
        legal_action_mask[row_index, :legal_count] = True
        search_value_targets[row_index, :legal_count] = sample.search_value_target
        search_value_mask[row_index, :legal_count] = sample.search_value_mask
        root_visit_targets[row_index, :legal_count] = sample.root_visit_target

    return {
        "observation": observations,
        "legal_action_features": action_features,
        "target_action_row": target_action_rows,
        "preference_competitor_row": preference_competitor_rows,
        "preference_mask": preference_masks,
        "action_index": action_indices,
        "legal_action_mask": legal_action_mask,
        "value_target": value_targets,
        "search_value_target": search_value_targets,
        "search_value_mask": search_value_mask,
        "root_visit_target": root_visit_targets,
        "root_visit_mask": root_visit_masks,
        "sample_weight": sample_weights,
        "policy_sample_weight": policy_sample_weights,
        "distill_sample_weight": distill_sample_weights,
        "preference_sample_weight": preference_sample_weights,
        "trajectory_advantage": trajectory_advantages,
        "behavior_log_probability": behavior_log_probabilities,
        "trajectory_mask": trajectory_masks,
    }
