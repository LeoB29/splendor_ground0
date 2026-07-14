from dataclasses import asdict
import json
import subprocess
import sys

import pytest
import torch

from splendor_ai.bots import (
    CheckpointPolicyBot,
    InformationSetPuctBot,
    LegalActionScorerBot,
    PolicyGuidedSearchBot,
)
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


def test_checkpoint_bot_factory_loads_cached_policy_guided_search_bot(tmp_path) -> None:
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
            "policy_guided_search",
            "--checkpoint-device",
            "cpu",
            "--search-depth",
            "1",
            "--search-max-branching",
            "5",
            "--policy-prior-weight",
            "1500",
        ]
    )

    factory = _build_bot_factory("checkpoint", seed=0, args=args)
    first_bot = factory()
    second_bot = factory()

    assert isinstance(first_bot, PolicyGuidedSearchBot)
    assert first_bot is second_bot
    assert first_bot.checkpoint_path == checkpoint_path
    assert first_bot.policy_prior_weight == 1500


def test_puct_bot_factory_loads_cached_information_set_bot(tmp_path) -> None:
    checkpoint_path = _write_legal_action_scorer_checkpoint(tmp_path)
    args = build_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "corpus"),
            "--seat0-bot",
            "puct",
            "--checkpoint-path",
            str(checkpoint_path),
            "--puct-simulations",
            "8",
            "--puct-inference-batch-size",
            "4",
            "--puct-n-step-leaf-depth",
            "2",
            "--puct-n-step-leaf-candidate-limit",
            "6",
            "--puct-n-step-leaf-weight",
            "0.5",
            "--puct-n-step-leaf-delta-clip",
            "0.2",
            "--puct-n-step-leaf-min-turn-index",
            "40",
            "--puct-n-step-leaf-min-max-score",
            "10",
            "--puct-rollout-leaf-count",
            "2",
            "--puct-rollout-leaf-max-depth",
            "5",
            "--puct-rollout-leaf-policy",
            "random",
        ]
    )

    factory = _build_bot_factory("puct", seed=0, args=args)
    first_bot = factory()
    second_bot = factory()

    assert isinstance(first_bot, InformationSetPuctBot)
    assert first_bot is second_bot
    assert first_bot.puct_config.simulations == 8
    assert first_bot.puct_config.inference_batch_size == 4
    assert first_bot.puct_config.n_step_leaf_depth == 2
    assert first_bot.puct_config.n_step_leaf_candidate_limit == 6
    assert first_bot.puct_config.n_step_leaf_weight == 0.5
    assert first_bot.puct_config.n_step_leaf_delta_clip == 0.2
    assert first_bot.puct_config.n_step_leaf_min_turn_index == 40
    assert first_bot.puct_config.n_step_leaf_min_max_score == 10
    assert first_bot.puct_config.rollout_leaf_count == 2
    assert first_bot.puct_config.rollout_leaf_max_depth == 5
    assert first_bot.puct_config.rollout_leaf_policy == "random"


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
            "--source-type",
            "league",
            "--disable-checkpoint-loop-fallback",
            "--search-value-checkpoint",
            "value.pt",
            "--search-value-weight",
            "0.25",
            "--search-value-scale",
            "1234",
            "--search-value-prediction-mode",
            "outcome",
            "--search-value-min-turn-index",
            "40",
        ]
    )

    metadata = _summary_metadata(args, ["checkpoint:greedy"])

    assert metadata["pairings"] == ["checkpoint:greedy"]
    assert metadata["source_type"] == "league"
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
    assert metadata["checkpoint"]["policy_prior_weight"] == 2000.0
    assert metadata["checkpoint"]["loop_fallback"]["enabled"] is False
    assert metadata["search"]["value_checkpoint"] == "value.pt"
    assert metadata["search"]["value_weight"] == 0.25
    assert metadata["search"]["value_scale"] == 1234
    assert metadata["search"]["value_prediction_mode"] == "outcome"
    assert metadata["search"]["value_outcome_blend_weight"] == 0.5
    assert metadata["search"]["value_min_turn_index"] == 40
    assert metadata["search"]["value_max_turn_index"] is None
    assert metadata["puct"]["n_step_leaf_depth"] == 0
    assert metadata["puct"]["n_step_leaf_width"] == 1
    assert metadata["puct"]["n_step_leaf_candidate_limit"] == 8
    assert metadata["puct"]["n_step_leaf_weight"] == 1.0
    assert metadata["puct"]["n_step_leaf_delta_clip"] is None
    assert metadata["puct"]["n_step_leaf_min_turn_index"] is None
    assert metadata["puct"]["n_step_leaf_max_turn_index"] is None
    assert metadata["puct"]["n_step_leaf_min_max_score"] is None
    assert metadata["puct"]["rollout_leaf_count"] == 0
    assert metadata["puct"]["rollout_leaf_max_depth"] == 12
    assert metadata["puct"]["rollout_leaf_policy"] == "random"


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


def test_run_replay_corpus_cli_writes_manifest(tmp_path) -> None:
    output_dir = tmp_path / "corpus"
    result = subprocess.run(
        [
            sys.executable,
            "run_replay_corpus.py",
            "--output-dir",
            str(output_dir),
            "--games",
            "1",
            "--seed-start",
            "50",
            "--max-turns",
            "30",
            "--seat0-bot",
            "random",
            "--seat1-bot",
            "random",
            "--source-type",
            "league_smoke",
            "--log-every",
            "0",
        ],
        cwd=".",
        text=True,
        capture_output=True,
        check=True,
    )

    assert "wrote manifest" in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_type"] == "league_smoke"
    assert manifest["seed_block"] == {
        "seed_start": 50,
        "seed_end_inclusive": 50,
        "games": 1,
    }
    assert manifest["metadata"]["pairings"] == ["random:random"]
