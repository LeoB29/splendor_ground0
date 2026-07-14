import json
import subprocess
import sys
from dataclasses import asdict

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
    compute_value_losses,
    export_replay_games_jsonl,
    fit_value,
    split_value_replay_dataset,
)
from splendor_ai.training.run_value_diagnostics import run_value_diagnostics


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


def test_value_replay_dataset_can_filter_by_turn_index(tmp_path) -> None:
    replay_path = _write_replay(tmp_path, max_turns=10)
    all_rows = [
        json.loads(line)
        for line in replay_path.read_text(encoding="utf-8").splitlines()
    ]
    expected = [
        row
        for row in all_rows
        if 2 <= int(row["turn_index"]) <= 5
    ]

    dataset = ValueReplayDataset(
        replay_path,
        min_turn_index=2,
        max_turn_index=5,
    )

    assert len(dataset) == len(expected)
    assert all(2 <= dataset[index].turn_index <= 5 for index in range(len(dataset)))


def test_value_model_and_loss_return_finite_values() -> None:
    model = ValueMLP(ValueModelConfig(hidden_size=16, depth=1))
    observation = torch.zeros((3, OBSERVATION_SIZE), dtype=torch.float32)
    target = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)
    outcome_target = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float32)

    prediction = model(observation)
    outcome_logits = model.outcome_logits(observation)
    loss = compute_value_loss(prediction, target)
    value_loss, outcome_loss, total_loss = compute_value_losses(
        value_pred=prediction,
        value_target=target,
        outcome_logits=outcome_logits,
        outcome_target=outcome_target,
        outcome_loss_weight=0.5,
    )

    assert prediction.shape == (3,)
    assert outcome_logits.shape == (3, 3)
    assert torch.all(prediction <= 1.0)
    assert torch.all(prediction >= -1.0)
    assert torch.isfinite(loss)
    assert torch.isfinite(value_loss)
    assert torch.isfinite(outcome_loss)
    assert torch.isfinite(total_loss)
    assert total_loss > value_loss


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
            "--min-turn-index",
            "1",
            "--outcome-loss-weight",
            "0.5",
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
    assert metrics["dataset"]["total_samples"] == len(
        ValueReplayDataset(replay_path, min_turn_index=1)
    )
    assert metrics["dataset"]["validation_split_unit"] == "game"
    assert metrics["dataset"]["min_turn_index"] == 1
    assert metrics["dataset"]["outcome_loss_weight"] == 0.5
    assert "outcome_loss" in metrics["epochs"][0]["train"]
    assert "outcome_class_accuracy" in metrics["epochs"][0]["train"]
    assert metrics["dataset"]["validation_samples"] > 0
    assert metrics["checkpoints"]["final_checkpoint_path"] == "value_model.pt"


def test_run_value_diagnostics_writes_bucketed_report(tmp_path) -> None:
    replay_path = _write_replays(tmp_path, max_turns=8, count=3)
    model = ValueMLP(ValueModelConfig(hidden_size=16, depth=1))
    checkpoint_path = tmp_path / "value_model.pt"
    torch.save(
        {
            "model_kind": "value_mlp",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    output_path = tmp_path / "diagnostics.json"

    payload = run_value_diagnostics(
        checkpoint_path=checkpoint_path,
        replay_paths=(replay_path,),
        output_path=output_path,
        split="validation",
        validation_fraction=0.5,
        validation_seed=0,
        validation_split_unit="game",
    )

    assert output_path.exists()
    assert payload["samples"] > 0
    assert "phase" in payload["buckets"]
    assert payload["overall"]["samples"] == payload["samples"]
    assert "outcome_class_accuracy" in payload["overall"]
    assert "mean_win_probability" in payload["overall"]
