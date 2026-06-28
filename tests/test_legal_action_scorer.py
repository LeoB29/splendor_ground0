import json
import math

import pytest
import torch

from splendor_ai.bots import RandomLegalBot
from splendor_ai.encoding import ACTION_FEATURE_SIZE
from splendor_ai.training import (
    LegalActionReplayDataset,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    collect_game_replay,
    collate_legal_action_replay_samples,
    export_replay_games_jsonl,
    masked_legal_action_scores,
)


def _write_snapshot_replay(tmp_path):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=4,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "snapshot_replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def test_legal_action_replay_dataset_maps_target_to_legal_action_row(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    dataset = LegalActionReplayDataset(replay_path)

    sample = dataset[0]

    assert sample.observation.shape == (256,)
    assert sample.legal_action_features.ndim == 2
    assert sample.legal_action_features.shape[1] == ACTION_FEATURE_SIZE
    assert 0 <= sample.target_action_row < sample.legal_action_features.shape[0]
    assert sample.legal_action_indices[sample.target_action_row] == sample.action_index
    assert sample.value_target in (-1.0, 0.0, 1.0)


def test_legal_action_replay_dataset_requires_state_snapshots(tmp_path) -> None:
    replay_path = tmp_path / "no_snapshot.jsonl"
    replay_path.write_text(
        json.dumps(
            {
                "observation_vector": [0.0] * 256,
                "action_index": 0,
                "legal_action_indices": [0],
                "final_value": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = LegalActionReplayDataset(replay_path)

    with pytest.raises(ValueError, match="state_snapshot"):
        _ = dataset[0]


def test_collate_legal_action_replay_samples_pads_variable_legal_action_counts(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    dataset = LegalActionReplayDataset(replay_path)
    samples = [dataset[0], dataset[1]]

    batch = collate_legal_action_replay_samples(samples)

    assert batch["observation"].shape == (2, 256)
    assert batch["legal_action_features"].shape[0] == 2
    assert batch["legal_action_features"].shape[2] == ACTION_FEATURE_SIZE
    assert batch["legal_action_mask"].dtype == torch.bool
    assert batch["target_action_row"].shape == (2,)
    assert batch["value_target"].shape == (2,)
    for row_index, sample in enumerate(samples):
        legal_count = sample.legal_action_features.shape[0]
        assert batch["legal_action_mask"][row_index, :legal_count].all()
        assert not batch["legal_action_mask"][row_index, legal_count:].any()


def test_legal_action_scorer_model_forward_and_masking() -> None:
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            observation_size=256,
            action_feature_size=ACTION_FEATURE_SIZE,
            state_hidden_size=32,
            action_hidden_size=16,
            scorer_hidden_size=24,
            state_depth=2,
            action_depth=2,
            dropout=0.0,
        )
    )
    observation = torch.zeros((2, 256), dtype=torch.float32)
    legal_action_features = torch.zeros((2, 5, ACTION_FEATURE_SIZE), dtype=torch.float32)
    legal_action_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    action_scores, value = model(observation, legal_action_features)
    masked_scores = masked_legal_action_scores(action_scores, legal_action_mask)

    assert action_scores.shape == (2, 5)
    assert value.shape == (2,)
    assert torch.isfinite(action_scores).all()
    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)
    assert math.isfinite(float(masked_scores[0, 0]))
    assert masked_scores[0, 3] < -1e30
