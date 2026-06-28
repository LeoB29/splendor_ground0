import json
import subprocess
import sys
from dataclasses import asdict

import torch

from splendor_ai.bots import LegalActionScorerBot, RandomLegalBot
from splendor_ai.training import (
    LegalActionReplayDataset,
    LegalActionScorerConfig,
    LegalActionScorerModel,
    SupervisedTrainConfig,
    collect_game_replay,
    compute_legal_action_supervised_losses,
    export_replay_games_jsonl,
    fit_legal_action_scorer,
    masked_legal_action_scores,
)


def _write_snapshot_replay(tmp_path, max_turns: int = 8):
    replay = collect_game_replay(
        bot_seat_0=RandomLegalBot(seed=1),
        bot_seat_1=RandomLegalBot(seed=2),
        seed=3,
        max_turns=max_turns,
        include_state_snapshots=True,
    )
    replay_path = tmp_path / "snapshot_replay.jsonl"
    export_replay_games_jsonl(replay_path, [replay])
    return replay_path


def test_compute_legal_action_supervised_losses_returns_finite_values() -> None:
    action_scores = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, 1.0, -5.0]],
        dtype=torch.float32,
    )
    value_pred = torch.tensor([0.25, -0.5], dtype=torch.float32)
    target_action_row = torch.tensor([2, 0], dtype=torch.long)
    legal_action_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    value_target = torch.tensor([1.0, -1.0], dtype=torch.float32)

    policy_loss, value_loss, total_loss = compute_legal_action_supervised_losses(
        action_scores=action_scores,
        value_pred=value_pred,
        target_action_row=target_action_row,
        legal_action_mask=legal_action_mask,
        value_target=value_target,
        value_loss_weight=2.0,
    )

    assert float(policy_loss) > 0.0
    assert float(value_loss) > 0.0
    assert float(total_loss) > float(policy_loss)
    assert masked_legal_action_scores(action_scores, legal_action_mask)[1, 2] < -1e30


def test_fit_legal_action_scorer_runs_on_snapshot_replay(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=32,
            action_hidden_size=16,
            scorer_hidden_size=24,
            state_depth=2,
            action_depth=2,
        )
    )

    trained_model, metrics = fit_legal_action_scorer(
        replay_path=replay_path,
        config=SupervisedTrainConfig(
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_loss_weight=1.0,
            epochs=1,
            device="cpu",
            shuffle=False,
        ),
        model=model,
    )

    assert isinstance(trained_model, LegalActionScorerModel)
    assert len(metrics) == 1
    assert metrics[0].train.samples == len(LegalActionReplayDataset(replay_path))
    assert metrics[0].train.batches > 0


def test_legal_action_scorer_bot_loads_checkpoint_and_chooses_legal_action(tmp_path) -> None:
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=16,
            action_hidden_size=8,
            scorer_hidden_size=12,
            state_depth=1,
            action_depth=1,
        )
    )
    checkpoint_path = tmp_path / "legal_action_scorer.pt"
    torch.save(
        {
            "model_kind": "legal_action_scorer",
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )

    from splendor_ai.engine.env import SplendorEnv

    env = SplendorEnv(seed=0)
    state = env.initial_state()
    legal_actions = env.legal_actions(state)
    bot = LegalActionScorerBot(checkpoint_path, device="cpu")

    chosen_action = bot.choose_action(env, state, legal_actions)

    assert chosen_action in legal_actions


def test_run_legal_action_supervised_cli_writes_checkpoint_and_metrics(tmp_path) -> None:
    replay_path = _write_snapshot_replay(tmp_path)
    output_dir = tmp_path / "legal_output"
    result = subprocess.run(
        [
            sys.executable,
            "run_legal_action_supervised.py",
            "--replay-path",
            str(replay_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--state-hidden-size",
            "16",
            "--action-hidden-size",
            "8",
            "--scorer-hidden-size",
            "12",
            "--state-depth",
            "1",
            "--action-depth",
            "1",
            "--validation-fraction",
            "0",
            "--log-every-batches",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Legal-action scorer training complete" in result.stdout
    assert (output_dir / "legal_action_scorer.pt").exists()
    metrics = json.loads((output_dir / "legal_action_scorer_metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["total_samples"] == len(LegalActionReplayDataset(replay_path))
    assert metrics["checkpoints"]["final_checkpoint_path"] == "legal_action_scorer.pt"
