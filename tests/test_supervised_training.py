import json
import math

import torch

from splendor_ai.training import (
    PolicyValueMLP,
    PolicyValueModelConfig,
    SupervisedReplayDataset,
    SupervisedTrainConfig,
    collate_replay_samples,
    compute_supervised_losses,
    fit_supervised,
    masked_policy_logits,
)


def _write_synthetic_replay_jsonl(path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _synthetic_rows() -> list[dict[str, object]]:
    base_obs_a = [0.0] * 256
    base_obs_b = [0.0] * 256
    base_obs_a[0] = 1.0
    base_obs_b[1] = 1.0
    return [
        {
            "game_seed": 1,
            "game_turns": 2,
            "game_winner": 0,
            "game_final_scores": [15, 12],
            "game_stalled": False,
            "seed": 1,
            "turn_index": 0,
            "player_id": 0,
            "action_index": 2,
            "legal_action_indices": [1, 2, 3],
            "observation_vector": base_obs_a,
            "final_value": 1.0,
            "winner": 0,
        },
        {
            "game_seed": 1,
            "game_turns": 2,
            "game_winner": 0,
            "game_final_scores": [15, 12],
            "game_stalled": False,
            "seed": 1,
            "turn_index": 1,
            "player_id": 1,
            "action_index": 1,
            "legal_action_indices": [0, 1, 4],
            "observation_vector": base_obs_b,
            "final_value": -1.0,
            "winner": 0,
        },
    ]


def test_supervised_replay_dataset_loads_samples(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    _write_synthetic_replay_jsonl(replay_path, _synthetic_rows())

    dataset = SupervisedReplayDataset(replay_path)

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.observation.shape == (256,)
    assert sample.action_index == 2
    assert sample.legal_action_indices == (1, 2, 3)
    assert sample.value_target == 1.0


def test_supervised_replay_dataset_can_exclude_stalled_games(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    rows = _synthetic_rows()
    rows[1]["game_stalled"] = True
    _write_synthetic_replay_jsonl(replay_path, rows)

    dataset = SupervisedReplayDataset(replay_path, include_stalled_games=False)

    assert len(dataset) == 1
    assert dataset[0].action_index == 2


def test_supervised_replay_dataset_can_exclude_timed_out_games(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    rows = _synthetic_rows()
    rows[1]["game_timed_out"] = True
    _write_synthetic_replay_jsonl(replay_path, rows)

    dataset = SupervisedReplayDataset(replay_path, include_timed_out_games=False)

    assert len(dataset) == 1
    assert dataset[0].action_index == 2


def test_collate_and_masked_policy_logits_work_together(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    _write_synthetic_replay_jsonl(replay_path, _synthetic_rows())
    dataset = SupervisedReplayDataset(replay_path)
    batch = collate_replay_samples([dataset[0], dataset[1]], action_space_size=8)

    logits = torch.arange(16, dtype=torch.float32).view(2, 8)
    masked_logits = masked_policy_logits(logits, batch["legal_action_mask"])

    assert masked_logits.shape == (2, 8)
    assert math.isfinite(float(masked_logits[0, 2]))
    assert masked_logits[0, 7] < -1e30


def test_compute_supervised_losses_returns_finite_values() -> None:
    policy_logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 0.0, 2.0]],
        dtype=torch.float32,
    )
    value_pred = torch.tensor([0.5, -0.25], dtype=torch.float32)
    action_index = torch.tensor([2, 1], dtype=torch.long)
    legal_action_mask = torch.tensor(
        [[False, True, True, True], [True, True, False, False]],
        dtype=torch.bool,
    )
    value_target = torch.tensor([1.0, -1.0], dtype=torch.float32)

    policy_loss, value_loss, total_loss = compute_supervised_losses(
        policy_logits=policy_logits,
        value_pred=value_pred,
        action_index=action_index,
        legal_action_mask=legal_action_mask,
        value_target=value_target,
        value_loss_weight=2.0,
    )

    assert float(policy_loss) > 0.0
    assert float(value_loss) > 0.0
    assert float(total_loss) > float(policy_loss)


def test_fit_supervised_runs_end_to_end_on_synthetic_data(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    _write_synthetic_replay_jsonl(replay_path, _synthetic_rows() * 2)

    model = PolicyValueMLP(
        PolicyValueModelConfig(
            observation_size=256,
            action_space_size=8,
            trunk_hidden_size=32,
            trunk_depth=2,
            dropout=0.0,
        )
    )
    trained_model, metrics = fit_supervised(
        replay_path=replay_path,
        config=SupervisedTrainConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_loss_weight=1.0,
            epochs=2,
            device="cpu",
            shuffle=False,
        ),
        model=model,
    )

    assert isinstance(trained_model, PolicyValueMLP)
    assert len(metrics) == 2
    assert all(metric.batches == 2 for metric in metrics)
    assert all(math.isfinite(metric.total_loss) for metric in metrics)


def test_fit_supervised_can_exclude_stalled_games(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    rows = _synthetic_rows() * 2
    rows[1]["game_stalled"] = True
    rows[3]["game_stalled"] = True
    _write_synthetic_replay_jsonl(replay_path, rows)

    model = PolicyValueMLP(
        PolicyValueModelConfig(
            observation_size=256,
            action_space_size=8,
            trunk_hidden_size=32,
            trunk_depth=2,
            dropout=0.0,
        )
    )
    trained_model, metrics = fit_supervised(
        replay_path=replay_path,
        config=SupervisedTrainConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_loss_weight=1.0,
            epochs=1,
            device="cpu",
            shuffle=False,
            include_stalled_games=False,
        ),
        model=model,
    )

    assert isinstance(trained_model, PolicyValueMLP)
    assert len(metrics) == 1
    assert metrics[0].batches == 1


def test_fit_supervised_can_exclude_timed_out_games(tmp_path) -> None:
    replay_path = tmp_path / "synthetic.jsonl"
    rows = _synthetic_rows() * 2
    rows[1]["game_timed_out"] = True
    rows[3]["game_timed_out"] = True
    _write_synthetic_replay_jsonl(replay_path, rows)

    model = PolicyValueMLP(
        PolicyValueModelConfig(
            observation_size=256,
            action_space_size=8,
            trunk_hidden_size=32,
            trunk_depth=2,
            dropout=0.0,
        )
    )
    trained_model, metrics = fit_supervised(
        replay_path=replay_path,
        config=SupervisedTrainConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_loss_weight=1.0,
            epochs=1,
            device="cpu",
            shuffle=False,
            include_timed_out_games=False,
        ),
        model=model,
    )

    assert isinstance(trained_model, PolicyValueMLP)
    assert len(metrics) == 1
    assert metrics[0].batches == 1
