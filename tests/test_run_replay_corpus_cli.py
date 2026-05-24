from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots import CheckpointPolicyBot
from splendor_ai.training.model import PolicyValueMLP, PolicyValueModelConfig
from splendor_ai.training.run_replay_corpus import (
    _build_bot_factory,
    _summary_metadata,
    build_parser,
)


def _write_checkpoint(tmp_path):
    model = PolicyValueMLP(
        PolicyValueModelConfig(
            trunk_hidden_size=8,
            trunk_depth=1,
            dropout=0.0,
        )
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model.config),
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_checkpoint_bot_factory_loads_cached_checkpoint_bot(tmp_path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--seat0-bot",
            "checkpoint",
            "--checkpoint-path",
            str(checkpoint_path),
            "--checkpoint-device",
            "cpu",
        ]
    )

    factory = _build_bot_factory("checkpoint", seed=0, args=args)
    first_bot = factory()
    second_bot = factory()

    assert isinstance(first_bot, CheckpointPolicyBot)
    assert first_bot is second_bot
    assert first_bot.checkpoint_path == checkpoint_path
    assert first_bot.loop_fallback_config.enabled is True


def test_checkpoint_bot_factory_requires_checkpoint_path(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--seat0-bot",
            "checkpoint",
        ]
    )

    with pytest.raises(ValueError, match="requires --checkpoint-path"):
        _build_bot_factory("checkpoint", seed=0, args=args)


def test_summary_metadata_records_checkpoint_configuration(tmp_path) -> None:
    checkpoint_path = _write_checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--checkpoint-path",
            str(checkpoint_path),
            "--checkpoint-device",
            "cpu",
            "--disable-checkpoint-loop-fallback",
        ]
    )

    metadata = _summary_metadata(args, ["checkpoint:greedy"])

    assert metadata["pairings"] == ["checkpoint:greedy"]
    assert metadata["checkpoint"]["path"] == str(checkpoint_path)
    assert metadata["checkpoint"]["device"] == "cpu"
    assert metadata["checkpoint"]["loop_fallback"]["enabled"] is False
