from dataclasses import asdict

import pytest
import torch

from splendor_ai.bots import CheckpointPolicyBot, LegalActionScorerBot
from splendor_ai.training.model import (
    LegalActionScorerConfig,
    LegalActionScorerModel,
    PolicyValueMLP,
    PolicyValueModelConfig,
)
from splendor_ai.training.run_replay_corpus import (
    _build_bot_factory,
    _resolve_pairings,
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


def _write_legal_action_scorer_checkpoint(tmp_path):
    model = LegalActionScorerModel(
        LegalActionScorerConfig(
            state_hidden_size=8,
            action_hidden_size=8,
            scorer_hidden_size=8,
            state_depth=1,
            action_depth=1,
            dropout=0.0,
        )
    )
    checkpoint_path = tmp_path / "legal_action_scorer.pt"
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


def test_checkpoint_bot_factory_loads_cached_legal_action_scorer_bot(tmp_path) -> None:
    checkpoint_path = _write_legal_action_scorer_checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--seat0-bot",
            "checkpoint",
            "--checkpoint-path",
            str(checkpoint_path),
            "--checkpoint-model-kind",
            "legal_action_scorer",
            "--checkpoint-device",
            "cpu",
        ]
    )

    factory = _build_bot_factory("checkpoint", seed=0, args=args)
    first_bot = factory()
    second_bot = factory()

    assert isinstance(first_bot, LegalActionScorerBot)
    assert first_bot is second_bot
    assert first_bot.checkpoint_path == checkpoint_path


def test_checkpoint_bot_factory_uses_seat_specific_checkpoint_settings(tmp_path) -> None:
    flat_checkpoint_path = _write_checkpoint(tmp_path)
    scorer_checkpoint_path = _write_legal_action_scorer_checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--pairing",
            "checkpoint:checkpoint",
            "--seat0-checkpoint-path",
            str(flat_checkpoint_path),
            "--seat0-checkpoint-model-kind",
            "flat_policy",
            "--seat1-checkpoint-path",
            str(scorer_checkpoint_path),
            "--seat1-checkpoint-model-kind",
            "legal_action_scorer",
            "--checkpoint-device",
            "cpu",
        ]
    )

    ((seat0_factory, seat1_factory),) = _resolve_pairings(args)
    seat0_bot = seat0_factory()
    seat1_bot = seat1_factory()

    assert isinstance(seat0_bot, CheckpointPolicyBot)
    assert seat0_bot.checkpoint_path == flat_checkpoint_path
    assert isinstance(seat1_bot, LegalActionScorerBot)
    assert seat1_bot.checkpoint_path == scorer_checkpoint_path


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
    assert metadata["checkpoint"]["model_kind"] == "flat_policy"
    assert metadata["checkpoint"]["seat0"] == {
        "path": str(checkpoint_path),
        "model_kind": "flat_policy",
    }
    assert metadata["checkpoint"]["seat1"] == {
        "path": str(checkpoint_path),
        "model_kind": "flat_policy",
    }
    assert metadata["checkpoint"]["device"] == "cpu"
    assert metadata["checkpoint"]["loop_fallback"]["enabled"] is False


def test_summary_metadata_records_seat_specific_checkpoint_configuration(tmp_path) -> None:
    flat_checkpoint_path = _write_checkpoint(tmp_path)
    scorer_checkpoint_path = _write_legal_action_scorer_checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--seat0-checkpoint-path",
            str(flat_checkpoint_path),
            "--seat0-checkpoint-model-kind",
            "flat_policy",
            "--seat1-checkpoint-path",
            str(scorer_checkpoint_path),
            "--seat1-checkpoint-model-kind",
            "legal_action_scorer",
        ]
    )

    metadata = _summary_metadata(args, ["checkpoint:checkpoint"])

    assert metadata["checkpoint"]["path"] is None
    assert metadata["checkpoint"]["model_kind"] == "flat_policy"
    assert metadata["checkpoint"]["seat0"] == {
        "path": str(flat_checkpoint_path),
        "model_kind": "flat_policy",
    }
    assert metadata["checkpoint"]["seat1"] == {
        "path": str(scorer_checkpoint_path),
        "model_kind": "legal_action_scorer",
    }
