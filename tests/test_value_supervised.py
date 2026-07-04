import json
import subprocess
import sys

import torch

from splendor_ai.bots import RandomLegalBot
from splendor_ai.encoding import OBSERVATION_SIZE
from splendor_ai.training import (
    SupervisedTrainConfig,
    ValueMLP,
    ValueModelConfig,
    ValueReplayDataset,
    collect_game_replay,
    compute_value_loss,
    export_replay_games_jsonl,
    fit_value,
    split_value_replay_dataset,
)


def _write_replay(tmp_path, max_turns: int = 8):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=max_turns,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def _write_replays(tmp_path, max_turns: int = 8, count: int = 2):
    replays = [
        collect_game_replay(
            bot_seat_0=RandomLegalBot(seed=10 + index),
            bot_seat_1=RandomLegalBot(seed=20 + index),
            seed=30 + index,
            max_turns=max_turns,
            include_state_snapshots=True,
        )
        for index in range(count)
    ]
    replay_path = tmp_path / "replays.jsonl"
    export_replay_games_jsonl(replay_path, replays)
    return replay_path


def test_value_replay_dataset_uses_current_player_score_margin(tmp_path) -> None:
    replay_path = _write_replay(tmp_path)
    first_row = json.loads(replay_path.read_text(encoding="utf-8").splitlines()[0])
    dataset = ValueReplayDataset(replay_path, margin_scale=15.0)

    sample = dataset[0]
    player_id = int(first_row["player_id"])
    scores = first_row["game_final_scores"]
    expected_margin = float(scores[player_id]) - float(scores[1 - player_id])

    assert sample.observation.shape == (OBSERVATION_SIZE,)
    assert sample.score_margin_target == expected_margin
    assert sample.value_target == max(min(expected_margin / 15.0, 1.0), -1.0)
    assert sample.outcome_target in (-1.0, 0.0, 1.0)


def test_value_model_and_loss_return_finite_values() -> None:
    model = ValueMLP(ValueModelConfig(hidden_size=16, depth=1))
    observation = torch.zeros((3, OBSERVATION_SIZE), dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)

    prediction = model(observation)
    loss = compute_value_loss(prediction, target)

    assert prediction.shape == (3,)
    assert torch.all(prediction <= 1.0)
    assert torch.all(prediction >= -1.0)
    assert torch.isfinite(loss)


def test_fit_value_runs_on_replay(tmp_path) -> None:
    replay_path = _write_replay(tmp_path)
    model = ValueMLP(ValueModelConfig(hidden_size=16, depth=1))

    trained_model, metrics = fit_value(
        replay_path=replay_path,
        config=SupervisedTrainConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            epochs=1,
            device="cpu",
            shuffle=False,
        ),
        model=model,
    )

    assert isinstance(trained_model, ValueMLP)
    assert len(metrics) == 1
    assert metrics[0].train.samples == len(ValueReplayDataset(replay_path))
    assert metrics[0].train.batches > 0


def test_split_value_replay_dataset_by_game_keeps_games_disjoint(tmp_path) -> None:
    replay_path = _write_replays(tmp_path, count=4)
    dataset = ValueReplayDataset(replay_path)

    train_dataset, validation_dataset = split_value_replay_dataset(
        dataset,
        validation_fraction=0.5,
        seed=0,
        validation_split_unit="game",
    )

    assert validation_dataset is not None
    train_keys = {
        dataset.validation_group_key_for_index(int(index), "game")
        for index in train_dataset.indices
    }
    validation_keys = {
        dataset.validation_group_key_for_index(int(index), "game")
        for index in validation_dataset.indices
    }
    assert train_keys
    assert validation_keys
    assert train_keys.isdisjoint(validation_keys)


def test_run_value_supervised_cli_writes_checkpoint_and_metrics(tmp_path) -> None:
    replay_path = _write_replays(tmp_path)
    output_dir = tmp_path / "value_output"
    result = subprocess.run(
        [
            sys.executable,
            "run_value_supervised.py",
            "--replay-path",
            str(replay_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-size",
            "16",
            "--depth",
            "1",
            "--validation-fraction",
            "0.5",
            "--validation-split-unit",
            "game",
            "--log-every-batches",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Value training complete" in result.stdout
    assert (output_dir / "value_model.pt").exists()
    metrics = json.loads((output_dir / "value_model_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["total_samples"] == len(ValueReplayDataset(replay_path))
    assert metrics["dataset"]["validation_split_unit"] == "game"
    assert metrics["dataset"]["validation_samples"] > 0
    assert metrics["checkpoints"]["final_checkpoint_path"] == "value_model.pt"
